"""
RS Screener Backtest v2 -- Audited, data-cleaned, multi-exit comparison

FIXES APPLIED IN THIS VERSION (see AUDIT NOTES at bottom of file):
    1. Data sanity cleaning -- caps/repairs implausible single-day price
       jumps (split/bonus/merger glitches) BEFORE any signal computation.
       Without this, a single corrupted data point can compound into an
       absurd total return over a multi-year daily-compounded backtest.
    2. Point-in-time liquidity filters (price + rolling volume), computed
       fresh at EACH historical date -- not today's live values, and not
       simply removed. Avoids both look-ahead bias AND untradeable-stock
       contamination.
    3. Configurable, comparable EXIT VARIANTS -- runs several exit rules
       against the SAME precomputed signals (no re-download needed) and
       writes a side-by-side comparison table, because a single fixed
       exit rule (e.g. RS < 3-EMA, state-based) was producing near-daily
       full-book turnover -- essentially as costly as no exit discipline
       at all.

ENTRY RULE (unchanged):
    Blue Dot (new 250-day RS high, using the PRIOR 250 days only)
    + Price Trend Template PASS (7/7)
    + RS Line Trend Template PASS (7/7)
    + sorted by raw RS Score, top N

EXIT: see EXIT_VARIANTS below -- several are run and compared.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

LOOKBACK_DAYS = 250                # RS new-high lookback (adjustable)
DOWNLOAD_YEARS_BEFORE_START = 3    # history pulled before BACKTEST_START

STOCKS_FILE = "stocks.csv"
TOP_N = 10                         # adjustable

# ---- Point-in-time liquidity filters (NOT look-ahead: computed per-date) ----
MIN_PRICE = 10                     # adjustable
MIN_AVG_VOLUME = 10000             # adjustable, 20-day rolling avg as of that date
VOLUME_LOOKBACK = 20               # adjustable

# ---- Data sanity cleaning ----
# NSE circuit limits are typically 2/5/10/20% for most stocks; a single-day
# move beyond this is virtually always a data error (unadjusted split/bonus/
# merger/ticker-reuse), not real price action. Points beyond this are
# repaired by holding the prior valid close flat, BEFORE any signal or
# return calculation touches them.
MAX_PLAUSIBLE_DAILY_MOVE = 0.30    # adjustable

BACKTEST_START = "2016-04-01"      # adjustable
BACKTEST_END = "2026-08-07"        # adjustable

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest"
COMPARISON_WORKSHEET = "Backtest_Exit_Comparison"

# ---- Exit variants to compare (all run against the same precomputed data) ----
# type options:
#   "rs_ema_state"     -- exit whenever RS line < RS EMA(span)      [current/original rule]
#   "rs_ema_cross"      -- exit only on the CROSSOVER day (was above, now below)
#   "rank_buffer"        -- no RS-based exit; hold until rank falls past `buffer`
#   "tt_fail_cross"      -- exit only when price Trend Template flips PASS->FAIL
EXIT_VARIANTS = [
    {"name": "RS<3EMA (state, original)",  "type": "rs_ema_state", "span": 3},
    {"name": "RS<3EMA (crossover)",         "type": "rs_ema_cross", "span": 3},
    {"name": "RS<10EMA (crossover)",        "type": "rs_ema_cross", "span": 10},
    {"name": "RS<20EMA (crossover)",        "type": "rs_ema_cross", "span": 20},
    {"name": "RS<20EMA (state)",            "type": "rs_ema_state", "span": 20},
    {"name": "Rank buffer only (buf=20)",   "type": "rank_buffer",  "buffer": 20},
    {"name": "Trend Template fail (cross)", "type": "tt_fail_cross"},
]
PRIMARY_VARIANT_INDEX = 3   # which variant's FULL trade log gets written (0-indexed)
# -----------------------------------------


def get_download_dates():
    backtest_start = pd.Timestamp(BACKTEST_START)
    backtest_end = pd.Timestamp(BACKTEST_END)
    download_start = backtest_start - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    download_end = backtest_end + pd.Timedelta(days=1)
    return download_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d")


def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")
    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


def clean_price_series(close):
    """
    AUDIT FIX #1 (critical): repairs implausible single-day price jumps
    (unadjusted splits/bonuses/mergers/ticker reuse) by holding the prior
    valid close flat wherever the day-over-day move exceeds
    MAX_PLAUSIBLE_DAILY_MOVE. Applied ONCE, before any signal/return math,
    so the fix propagates correctly through RS score, Blue Dot, Trend
    Template, AND the equity curve.
    """
    close = close.copy().sort_index()
    pct_change = close.pct_change()
    bad = pct_change.abs() > MAX_PLAUSIBLE_DAILY_MOVE
    n_bad = bad.sum()
    if n_bad > 0:
        # Repair iteratively: a repaired point can itself un-flag the NEXT
        # point's pct_change once the series is corrected.
        cleaned = close.copy()
        for idx in close.index[bad]:
            pos = cleaned.index.get_loc(idx)
            if pos > 0:
                cleaned.iloc[pos] = cleaned.iloc[pos - 1]
        return cleaned, int(n_bad)
    return close, 0


def download_benchmark():
    download_start, download_end = get_download_dates()
    print(f"\nBenchmark download: {download_start} to {download_end}")
    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(ticker, start=download_start, end=download_end,
                                interval="1d", auto_adjust=True, progress=False)
            if data.empty:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().sort_index()
            if close.empty:
                continue
            close, n_bad = clean_price_series(close)
            if n_bad:
                print(f"Benchmark {ticker}: repaired {n_bad} implausible data point(s)")
            print(f"Benchmark loaded: {ticker}")
            return close
        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")
    raise RuntimeError("Could not download any benchmark index data.")


def trend_template_series(s):
    sma50 = s.rolling(50).mean()
    sma150 = s.rolling(150).mean()
    sma200 = s.rolling(200).mean()
    sma200_1mo = sma200.shift(21)
    low52 = s.rolling(252).min()
    high52 = s.rolling(252).max()

    c1 = (s > sma150) & (s > sma200)
    c2 = sma150 > sma200
    c3 = sma200 > sma200_1mo
    c4 = (sma50 > sma150) & (sma50 > sma200)
    c5 = s > sma50
    c6 = s >= 1.25 * low52
    c7 = s >= 0.75 * high52

    met = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) \
        + c5.astype(int) + c6.astype(int) + c7.astype(int)
    return met == 7, met


def compute_signals_for_stock(close, volume, bench_close):
    """Computes all historical signals for one stock, including every exit
    variant's underlying data (multiple EMA spans, TT pass/fail transitions,
    point-in-time liquidity flags) so all variants can be backtested from a
    single precomputed pass -- no repeated downloads."""

    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) < 280:
        return None

    volume = volume.reindex(aligned.index)

    rs_ratio = aligned["s"] / aligned["b"]

    def pct_return(series, days):
        return series / series.shift(days) - 1

    rs_score = (0.40 * pct_return(aligned["s"], 63)
                + 0.20 * pct_return(aligned["s"], 126)
                + 0.20 * pct_return(aligned["s"], 189)
                + 0.20 * pct_return(aligned["s"], 252)) * 100

    # Blue Dot: uses the PRIOR LOOKBACK_DAYS only (today can't set its own comparison high)
    previous_rs_high = rs_ratio.shift(1).rolling(LOOKBACK_DAYS).max()
    blue_dot = rs_ratio > previous_rs_high

    tt_pass, tt_met = trend_template_series(aligned["s"])
    rs_tt_pass, rs_tt_met = trend_template_series(rs_ratio)

    # AUDIT FIX #2: point-in-time liquidity, computed fresh at each date
    rolling_avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (aligned["s"] >= MIN_PRICE) & (rolling_avg_volume >= MIN_AVG_VOLUME)

    out = pd.DataFrame({
        "price": aligned["s"],
        "rs_line": rs_ratio,
        "rs_score": rs_score,
        "blue_dot": blue_dot,
        "tt_pass": tt_pass,
        "rs_tt_pass": rs_tt_pass,
        "liquid": liquid,
    })

    # Precompute EMA + crossover state for every span used across EXIT_VARIANTS
    spans_needed = {v["span"] for v in EXIT_VARIANTS if "span" in v}
    for span in spans_needed:
        ema = rs_ratio.ewm(span=span, adjust=False).mean()
        below = rs_ratio < ema
        out[f"rs_below_ema{span}"] = below
        out[f"rs_cross_below_ema{span}"] = below & (~below.shift(1).fillna(False))

    # TT fail crossover (was PASS yesterday, FAIL today)
    out["tt_fail_cross"] = (~tt_pass) & tt_pass.shift(1).fillna(False)

    return out


def run_backtest_for_variant(all_signals, trading_days, variant):
    """Simulates the portfolio for ONE exit variant, reusing precomputed signals."""
    holdings = {}
    trade_log = []
    equity = 1.0
    equity_curve = []

    v_type = variant["type"]
    span = variant.get("span")
    buffer = variant.get("buffer")

    # For rank_buffer variant we need each day's rank of every eligible symbol
    for date in trading_days:
        pool = []
        for sym, df in all_signals.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["rs_score"]) or not bool(row["liquid"]):
                continue
            if bool(row["blue_dot"]) and bool(row["tt_pass"]) and bool(row["rs_tt_pass"]):
                pool.append((sym, float(row["rs_score"])))

        pool.sort(key=lambda x: x[1], reverse=True)
        rank_lookup = {sym: i + 1 for i, (sym, _) in enumerate(pool)}
        target_syms_topn = {sym for sym, _ in pool[:TOP_N]}
        target_prices = {}
        for sym in target_syms_topn:
            target_prices[sym] = float(all_signals[sym].loc[date, "price"])

        held_before_today = set(holdings.keys())

        # Daily portfolio return: only stocks held BEFORE today's decisions
        if held_before_today:
            rets = []
            for sym in held_before_today:
                df = all_signals[sym]
                if date not in df.index:
                    continue
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_price = df["price"].iloc[idx - 1]
                    curr_price = df["price"].iloc[idx]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        rets.append(curr_price / prev_price - 1)
            if rets:
                equity *= (1 + float(np.mean(rets)))

        # ---- Exit logic (variant-specific) ----
        for sym in list(holdings.keys()):
            df = all_signals[sym]
            if date not in df.index:
                continue
            row = df.loc[date]

            if v_type == "rank_buffer":
                rank = rank_lookup.get(sym, 9999)
                exit_trigger = rank > buffer
                reason = f"Rank > {buffer}"
            elif v_type == "rs_ema_state":
                exit_trigger = bool(row.get(f"rs_below_ema{span}", False))
                reason = f"RS Line < RS Line {span}-EMA"
            elif v_type == "rs_ema_cross":
                exit_trigger = bool(row.get(f"rs_cross_below_ema{span}", False))
                reason = f"RS Line crossed below {span}-EMA"
            elif v_type == "tt_fail_cross":
                exit_trigger = bool(row.get("tt_fail_cross", False))
                reason = "Trend Template PASS->FAIL"
            else:
                exit_trigger = False
                reason = "N/A"

            # Top-N membership exit applies to all variants EXCEPT rank_buffer,
            # whose whole purpose is to relax exactly this constraint (hold to
            # rank 20 instead of forcing an exit the moment rank crosses 10).
            if v_type == "rank_buffer":
                target_exit = False
            else:
                target_exit = sym not in target_syms_topn
            if exit_trigger or target_exit:
                entry = holdings.pop(sym)
                exit_price = float(row["price"])
                ret = (exit_price / entry["entry_price"] - 1) * 100
                trade_log.append({
                    "symbol": sym, "entry_date": entry["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(entry["entry_price"], 2), "exit_price": round(exit_price, 2),
                    "return_pct": round(ret, 2),
                    "days_held": (date - entry["entry_date"]).days,
                    "exit_reason": reason if exit_trigger else "No longer in Top-N pool",
                })

        # ---- Entries ----
        if v_type == "rank_buffer":
            # Only fill actually-open slots, in rank order -- prevents the
            # portfolio from growing past TOP_N when buffer-zone (rank 11-20)
            # positions are still being held.
            slots_open = TOP_N - len(holdings)
            if slots_open > 0:
                ranked_candidates = sorted(pool, key=lambda x: x[1], reverse=True)
                for sym, _ in ranked_candidates:
                    if slots_open <= 0:
                        break
                    if sym not in holdings:
                        holdings[sym] = {"entry_price": float(all_signals[sym].loc[date, "price"]),
                                          "entry_date": date}
                        slots_open -= 1
        else:
            for sym in target_syms_topn:
                if sym not in holdings:
                    holdings[sym] = {"entry_price": target_prices[sym], "entry_date": date}

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(equity, 6),
                              "n_holdings": len(holdings)})

    # Close any open positions at the end
    if trading_days.size:
        last_date = trading_days[-1]
        for sym, entry in holdings.items():
            df = all_signals[sym]
            exit_price = float(df.loc[last_date, "price"]) if last_date in df.index else entry["entry_price"]
            ret = (exit_price / entry["entry_price"] - 1) * 100
            trade_log.append({
                "symbol": sym, "entry_date": entry["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": last_date.strftime("%Y-%m-%d") + " (OPEN)",
                "entry_price": round(entry["entry_price"], 2), "exit_price": round(exit_price, 2),
                "return_pct": round(ret, 2),
                "days_held": (last_date - entry["entry_date"]).days,
                "exit_reason": "BACKTEST END",
            })

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)


def summarize(trade_df, equity_df):
    if equity_df.empty:
        return {}
    total_return_pct = round((equity_df["equity"].iloc[-1] - 1) * 100, 2)
    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] / running_max - 1) * 100
    max_dd = round(drawdown.min(), 2)

    closed = trade_df[~trade_df["exit_date"].astype(str).str.contains("OPEN", na=False)] \
        if not trade_df.empty else trade_df
    n = len(closed)
    if n:
        win_rate = round((closed["return_pct"] > 0).mean() * 100, 1)
        avg_return = round(closed["return_pct"].mean(), 2)
        avg_days = round(closed["days_held"].mean(), 1)
        best = closed["return_pct"].max()
        worst = closed["return_pct"].min()
    else:
        win_rate = avg_return = avg_days = best = worst = 0

    return {
        "total_return_pct": total_return_pct, "max_dd_pct": max_dd,
        "n_trades": n, "win_rate": win_rate, "avg_return_per_trade": avg_return,
        "avg_days_held": avg_days, "best_trade": best, "worst_trade": worst,
    }


def run_backtest():
    tickers = load_tickers()
    print(f"\nLoaded {len(tickers)} tickers.")

    download_start, download_end = get_download_dates()
    print("=" * 50)
    print(f"Download start : {download_start}")
    print(f"Backtest start : {BACKTEST_START}")
    print(f"Backtest end   : {BACKTEST_END}")
    print(f"Data cleaning threshold: +/-{MAX_PLAUSIBLE_DAILY_MOVE*100:.0f}% single-day move")
    print(f"Liquidity filter: price >= {MIN_PRICE}, {VOLUME_LOOKBACK}d avg volume >= {MIN_AVG_VOLUME}")
    print("=" * 50)

    bench_close = download_benchmark()

    all_signals = {}
    total_bad_points = 0
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\nDownloading batch {i}-{i+len(batch)}...")
        try:
            data = yf.download(batch, start=download_start, end=download_end,
                                interval="1d", auto_adjust=True, progress=False,
                                group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    sdata = data
                else:
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    sdata = data[symbol]
                if "Close" not in sdata.columns:
                    continue
                close = sdata["Close"].dropna().sort_index()
                volume = sdata["Volume"].reindex(close.index).fillna(0)
                if close.empty:
                    continue

                close, n_bad = clean_price_series(close)
                total_bad_points += n_bad

                sig = compute_signals_for_stock(close, volume, bench_close)
                if sig is not None:
                    all_signals[symbol.replace(".NS", "")] = sig
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue
        time.sleep(1)

    print(f"\nSignals computed for {len(all_signals)} stocks.")
    print(f"Total data points repaired across universe: {total_bad_points}")

    trading_days = bench_close.index[
        (bench_close.index >= pd.Timestamp(BACKTEST_START)) &
        (bench_close.index <= pd.Timestamp(BACKTEST_END))
    ]
    print(f"Trading days: {len(trading_days)}")

    # ---- Run every exit variant against the SAME precomputed signals ----
    comparison_rows = []
    primary_trades, primary_equity = pd.DataFrame(), pd.DataFrame()

    for idx, variant in enumerate(EXIT_VARIANTS):
        print(f"\nRunning variant: {variant['name']}...")
        trades, equity = run_backtest_for_variant(all_signals, trading_days, variant)
        summary = summarize(trades, equity)
        summary["variant"] = variant["name"]
        comparison_rows.append(summary)
        print(f"  Total Return: {summary.get('total_return_pct')}% | "
              f"Max DD: {summary.get('max_dd_pct')}% | "
              f"Trades: {summary.get('n_trades')} | Win Rate: {summary.get('win_rate')}%")

        if idx == PRIMARY_VARIANT_INDEX:
            primary_trades, primary_equity = trades, equity

    comparison_df = pd.DataFrame(comparison_rows)[
        ["variant", "total_return_pct", "max_dd_pct", "n_trades", "win_rate",
         "avg_return_per_trade", "avg_days_held", "best_trade", "worst_trade"]
    ]

    write_to_sheet(primary_trades, primary_equity, comparison_df)


def write_to_sheet(trade_df, equity_df, comparison_df):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- saving to CSV instead.")
        trade_df.to_csv("backtest_trades.csv", index=False)
        equity_df.to_csv("backtest_equity.csv", index=False)
        comparison_df.to_csv("backtest_exit_comparison.csv", index=False)
        return

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    # ---- Comparison tab ----
    n_rows_needed = len(comparison_df) + 10
    try:
        cws = sh.worksheet(COMPARISON_WORKSHEET)
        if cws.row_count < n_rows_needed:
            cws.resize(rows=n_rows_needed, cols=10)
    except gspread.WorksheetNotFound:
        cws = sh.add_worksheet(title=COMPARISON_WORKSHEET, rows=n_rows_needed, cols=10)
    cws.clear()
    cws.update([[f"Exit variant comparison run: {timestamp} | GROSS returns, data-cleaned"]], "A1")
    cws.update([list(comparison_df.columns)] + comparison_df.values.tolist(), "A3")
    print(f"\nComparison written to '{COMPARISON_WORKSHEET}' tab.")

    # ---- Primary variant full detail tab ----
    n_rows_needed2 = max(len(trade_df) + len(equity_df) + 50, 100)
    try:
        ws = sh.worksheet(BACKTEST_WORKSHEET)
        if ws.row_count < n_rows_needed2:
            ws.resize(rows=n_rows_needed2, cols=10)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=BACKTEST_WORKSHEET, rows=n_rows_needed2, cols=10)
    ws.clear()
    primary_name = EXIT_VARIANTS[PRIMARY_VARIANT_INDEX]["name"]
    ws.update([[f"Primary variant detail: {primary_name} | run {timestamp} | GROSS, data-cleaned"]], "A1")
    ws.update([["Trade Log"]], "A3")
    if not trade_df.empty:
        ws.update([list(trade_df.columns)] + trade_df.values.tolist(), "A4")
    equity_start = 4 + len(trade_df) + 3
    ws.update([["Daily Equity Curve"]], f"A{equity_start}")
    if not equity_df.empty:
        ws.update([list(equity_df.columns)] + equity_df.values.tolist(), f"A{equity_start+1}")
    print(f"Primary variant detail written to '{BACKTEST_WORKSHEET}' tab.")


if __name__ == "__main__":
    try:
        run_backtest()
        print("\nBACKTEST COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print("\nBACKTEST FAILED")
        print(f"{type(e).__name__}: {e}")
        raise


# ============================================================
# AUDIT NOTES (read before trusting any output)
# ============================================================
#
# 1. DATA CLEANING is a blunt instrument, not a perfect one. It repairs
#    single-day moves beyond MAX_PLAUSIBLE_DAILY_MOVE by holding price flat.
#    This can occasionally suppress a rare GENUINE large move (e.g. a real
#    de-merger re-rating), but the alternative -- leaving corrupted points
#    in -- was catastrophically worse (the 14-million-percent result).
#    Spot-check a sample of "repaired" points if you want full confidence.
#
# 2. SURVIVORSHIP BIAS is reduced but NOT eliminated. Point-in-time
#    volume/price filters stop today's liquidity from leaking into 2016
#    decisions, but the underlying UNIVERSE (stocks.csv) is still your
#    CURRENT list projected backward. Any stock that delisted, merged, or
#    was renamed between 2016-2026 is invisible to this test. Over a
#    10-year window this is a real, unresolved distortion -- there is no
#    free data source for India's point-in-time historical index
#    membership, so this caveat cannot be fully fixed without a paid
#    survivorship-bias-free dataset.
#
# 3. STILL NO COSTS MODELED. With multiple exit variants producing
#    thousands of trades over 10 years, brokerage/STT/slippage will matter
#    enormously and will affect variants differently (higher-turnover
#    variants get hurt disproportionately more).
#
# 4. The "Top-N membership" exit ALWAYS applies on top of whichever exit
#    variant is selected -- so even the "loosest" variants still see
#    turnover from stocks simply falling out of the top 10 by RS Score.
#    If turnover is still too high after comparing variants, the next
#    experiment worth running is combining "rank_buffer" (hold to rank 20)
#    WITH a slow RS-EMA crossover exit, instead of either alone.
