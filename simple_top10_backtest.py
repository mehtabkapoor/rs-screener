"""
============================================================
RS SCREENER — DAILY TOP 10 RS BACKTEST  (CORRECTED)
============================================================

CHANGES FROM PRIOR VERSION
---------------------------
1. EXIT RULE: unchanged from original — EXIT_RANK = 15 (rank buffer
   retained: a held stock exits only when its rank drops below 15,
   not merely when it falls out of the Top 10 entry list).

2. WEIGHT_PCT BUG FIXED:
       OLD: weight_pct was computed against a running partial sum
            of portfolio_value inside the same loop that built it,
            so weights did not sum to 100% and were iteration-order
            dependent.
       NEW: two-pass calculation. Total portfolio value (cash + all
            holdings marked at today's close) is computed FIRST,
            then each holding's weight_pct is computed against that
            fixed total.

3. EXECUTION TIMING CHANGED TO T+1:
       OLD: signal computed on day D executed at day D's own close
            (same-day fill) — unrealistic vs. live system, which
            requires manual Executed=Y confirmation the next session.
       NEW: signal (pool/ranks) computed on day D is staged and
            EXECUTED at day D+1's close. Equity is marked at each
            day's close using only positions that have actually been
            executed by that point (no look-ahead).

Everything else (RS Score formula, Trend Template, filters, costs,
STCG) is UNCHANGED from the prior version and still does NOT match
your locked rulebook (no 1M term, no skip-last-week, daily not
weekly rebalance, no regime/breadth/sector/circuit-breaker filters).
That is a separate, larger change — flag if you want it folded in too.
============================================================
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

STOCKS_FILE = "stocks.csv"

DOWNLOAD_YEARS_BEFORE_START = 3
BACKTEST_START = "2016-04-01"
BACKTEST_END = None  # None = latest available Yahoo Finance data

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250
MAX_PLAUSIBLE_DAILY_MOVE = 0.30

TOP_N = 10
EXIT_RANK = 15  # retain holding while rank <= 15 (rank buffer above Top 10 entry)

STARTING_CAPITAL = 1_000_000

# ============================================================
# TRANSACTION COSTS
# ============================================================

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20
STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "top 10 RS backtest"

WRITE_CHUNK_SIZE = 5000
MAX_WRITE_RETRIES = 6
INITIAL_RETRY_SECONDS = 5


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def normalize_series_index(series):
    s = series.copy()
    s.index = normalize_dates(s.index)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():
    backtest_start = pd.Timestamp(BACKTEST_START)
    download_start = backtest_start - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    if BACKTEST_END is None:
        return download_start.strftime("%Y-%m-%d"), None
    backtest_end = pd.Timestamp(BACKTEST_END)
    download_end = backtest_end + pd.Timedelta(days=1)
    return download_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d")


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")
    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


# ============================================================
# CLEAN PRICE SERIES
# ============================================================

def clean_price_series(close):
    close = normalize_series_index(close)
    pct_change = close.pct_change()
    bad = pct_change.abs() > MAX_PLAUSIBLE_DAILY_MOVE
    n_bad = int(bad.sum())
    if n_bad == 0:
        return close, 0
    cleaned = close.copy()
    for idx in close.index[bad]:
        pos = cleaned.index.get_loc(idx)
        if pos > 0:
            cleaned.iloc[pos] = cleaned.iloc[pos - 1]
    return cleaned, n_bad


# ============================================================
# COSTS / TAX
# ============================================================

def buy_side_cost(trade_value):
    stt = STT_RATE * trade_value
    stamp = STAMP_DUTY_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)
    return stt + stamp + exch + sebi + gst


def sell_side_cost(trade_value):
    stt = STT_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)
    return stt + exch + sebi + gst + DP_CHARGE_FLAT


def stcg_tax(net_gain):
    if net_gain <= 0:
        return 0.0
    return net_gain * STCG_EFFECTIVE_RATE


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():
    download_start, download_end = get_download_dates()
    print(f"\nBenchmark download: {download_start} to {download_end if download_end else 'LATEST AVAILABLE'}")
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
                print(f"Benchmark {ticker}: repaired {n_bad} implausible points")
            print(f"Benchmark loaded: {ticker}")
            print(f"Latest benchmark date: {close.index.max().strftime('%Y-%m-%d')}")
            return close
        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")
    raise RuntimeError("Could not download any benchmark index data.")


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template_series(s):
    s = normalize_series_index(s)
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

    met = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) + \
          c5.astype(int) + c6.astype(int) + c7.astype(int)
    return met == 7


# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_signals_for_stock(close, volume, bench_close):
    close = normalize_series_index(close)
    volume = normalize_series_index(volume)
    bench_close = normalize_series_index(bench_close)

    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) < 280:
        return None

    volume = volume.reindex(aligned.index).fillna(0)

    rs_ratio = aligned["s"] / aligned["b"]

    def pct_return(series, days):
        return series / series.shift(days) - 1

    rs_score = (
        0.40 * pct_return(aligned["s"], 63)
        + 0.20 * pct_return(aligned["s"], 126)
        + 0.20 * pct_return(aligned["s"], 189)
        + 0.20 * pct_return(aligned["s"], 252)
    ) * 100

    previous_rs_high = rs_ratio.shift(1).rolling(LOOKBACK_DAYS).max()
    blue_dot = rs_ratio > previous_rs_high

    previous_price_high = aligned["s"].shift(1).rolling(LOOKBACK_DAYS).max()
    price_at_new_high = aligned["s"] > previous_price_high

    green_dot = blue_dot & (~price_at_new_high)

    tt_pass = trend_template_series(aligned["s"])
    rs_tt_pass = trend_template_series(rs_ratio)

    rolling_avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (aligned["s"] > MIN_PRICE) & (rolling_avg_volume > MIN_AVG_VOLUME)

    out = pd.DataFrame({
        "price": aligned["s"],
        "rs_score": rs_score,
        "tt_pass": tt_pass,
        "rs_tt_pass": rs_tt_pass,
        "liquid": liquid,
        "blue_dot": blue_dot,
        "green_dot": green_dot,
    })
    out.index = normalize_dates(out.index)
    return out


# ============================================================
# SAFE DATE ROW ACCESS
# ============================================================

def get_row(df, date):
    date = pd.Timestamp(date).normalize()
    if date not in df.index:
        return None
    row = df.loc[date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


# ============================================================
# BUILD ELIGIBLE POOL FOR A GIVEN DATE
# ============================================================

def build_pool(all_signals, date):
    """Returns (pool sorted desc by rs_score, rank_lookup, diagnostics dict)."""
    pool = []
    diagnostics = {}
    for sym, df in all_signals.items():
        row = get_row(df, date)
        if row is None:
            continue
        if pd.isna(row["rs_score"]):
            continue
        if not bool(row["liquid"]):
            continue
        if not bool(row["tt_pass"]):
            continue
        if not bool(row["rs_tt_pass"]):
            continue
        pool.append((sym, float(row["rs_score"])))
        diagnostics[sym] = row

    pool.sort(key=lambda x: x[1], reverse=True)
    rank_lookup = {sym: rank + 1 for rank, (sym, _) in enumerate(pool)}
    return pool, rank_lookup, diagnostics


# ============================================================
# BACKTEST ENGINE  (T+1 EXECUTION)
# ============================================================
#
# On day D we OBSERVE the pool/ranks using day D's closing data.
# That observation determines what SHOULD be held after D.
# The actual BUY/SELL trades to reach that target are EXECUTED
# at day D+1's close (the next trading day in the loop).
# Equity is marked at each day's close using only positions that
# have actually been executed by that point — no look-ahead.
# ============================================================

def run_backtest(all_signals, trading_days):

    cash = float(STARTING_CAPITAL)
    holdings = {}          # currently executed positions
    pending_target = None  # target holding set decided on the PREVIOUS day, to execute today

    trade_log = []
    equity_curve = []
    daily_top10 = []
    daily_holdings = []

    n_days = len(trading_days)

    for day_number, date in enumerate(trading_days, start=1):
        date = pd.Timestamp(date).normalize()

        # ----------------------------------------------------
        # STEP 1: EXECUTE YESTERDAY'S DECISION AT TODAY'S CLOSE
        # ----------------------------------------------------
        if pending_target is not None:

            target_sell, target_buy_ranked = pending_target

            # --- SELLS ---
            for sym, (exit_rank, exit_reason) in target_sell.items():
                if sym not in holdings:
                    continue
                df = all_signals[sym]
                row = get_row(df, date)
                if row is None:
                    # no price today; defer — keep holding, retry next day
                    continue
                exit_price = float(row["price"])
                pos = holdings.pop(sym)

                gross_proceeds = pos["qty"] * exit_price
                s_cost = sell_side_cost(gross_proceeds)
                net_proceeds = gross_proceeds - s_cost
                cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
                net_gain = net_proceeds - cost_basis
                tax = stcg_tax(net_gain)
                cash += net_proceeds - tax

                gross_return_pct = (exit_price / pos["entry_price"] - 1) * 100
                net_return_pct = ((net_gain - tax) / cost_basis * 100) if cost_basis > 0 else 0

                trade_log.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "qty": pos["qty"],
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "gross_return_pct": round(gross_return_pct, 2),
                    "buy_cost_rs": round(pos["entry_cost"], 2),
                    "sell_cost_rs": round(s_cost, 2),
                    "stcg_tax_rs": round(tax, 2),
                    "net_pnl_rs": round(net_gain - tax, 2),
                    "net_return_pct": round(net_return_pct, 2),
                    "days_held": (date - pos["entry_date"]).days,
                    "exit_reason": exit_reason,
                    "exit_rank": exit_rank if exit_rank is not None else "",
                })

            # --- Portfolio value ahead of buys (post-sells) ---
            portfolio_value_before_buys = cash
            for sym, pos in holdings.items():
                row = get_row(all_signals[sym], date)
                mark_price = float(row["price"]) if row is not None else pos["entry_price"]
                portfolio_value_before_buys += pos["qty"] * mark_price

            # --- BUYS: fill vacant slots from yesterday's ranked target list ---
            slots_open = TOP_N - len(holdings)
            if slots_open > 0 and target_buy_ranked:
                slot_capital = portfolio_value_before_buys / TOP_N
                for sym in target_buy_ranked:
                    if slots_open <= 0:
                        break
                    if sym in holdings:
                        continue
                    row = get_row(all_signals[sym], date)
                    if row is None:
                        continue
                    price = float(row["price"])
                    if price <= 0:
                        continue
                    qty = int(slot_capital // price)
                    if qty < 1:
                        continue
                    trade_value = qty * price
                    b_cost = buy_side_cost(trade_value)
                    total_cost = trade_value + b_cost
                    if total_cost > cash:
                        continue
                    cash -= total_cost
                    holdings[sym] = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_date": date,
                        "entry_cost": b_cost,
                    }
                    slots_open -= 1

        # ----------------------------------------------------
        # STEP 2: OBSERVE TODAY'S POOL/RANKS (decision for TOMORROW's execution)
        # ----------------------------------------------------
        pool, rank_lookup, diagnostics = build_pool(all_signals, date)
        target_top10 = [sym for sym, _ in pool[:TOP_N]]

        # Determine sells for tomorrow: currently held but rank > EXIT_RANK, or dropped
        # out of the eligible universe entirely.
        sells_for_tomorrow = {}
        for sym in holdings.keys():
            r = rank_lookup.get(sym)
            if r is None:
                sells_for_tomorrow[sym] = (None, "Left eligible universe")
            elif r > EXIT_RANK:
                sells_for_tomorrow[sym] = (r, f"Rank > {EXIT_RANK}")
        pending_target = (sells_for_tomorrow, target_top10)

        # ----------------------------------------------------
        # DAILY TOP 10 RECORD (as observed today)
        # ----------------------------------------------------
        top10_row = {"date": date.strftime("%Y-%m-%d")}
        for i in range(TOP_N):
            if i < len(pool):
                sym, score = pool[i]
                top10_row[f"Rank_{i + 1}"] = sym
                top10_row[f"RS_{i + 1}"] = round(score, 4)
            else:
                top10_row[f"Rank_{i + 1}"] = ""
                top10_row[f"RS_{i + 1}"] = ""
        daily_top10.append(top10_row)

        # ----------------------------------------------------
        # STEP 3: MARK-TO-MARKET AT TODAY'S CLOSE (two-pass weight_pct fix)
        # ----------------------------------------------------
        total_value = cash
        marks = {}
        for sym, pos in holdings.items():
            row = get_row(all_signals[sym], date)
            mark_price = float(row["price"]) if row is not None else pos["entry_price"]
            marks[sym] = mark_price
            total_value += pos["qty"] * mark_price

        holdings_snapshot = []
        for sym, pos in holdings.items():
            mark_price = marks[sym]
            market_value = pos["qty"] * mark_price
            weight_pct = round(market_value / total_value * 100, 3) if total_value > 0 else 0
            rs_val = ""
            if sym in diagnostics:
                rs_val = round(float(diagnostics[sym]["rs_score"]), 4)
            holdings_snapshot.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": sym,
                "rank": rank_lookup.get(sym, ""),
                "rs_score": rs_val,
                "qty": pos["qty"],
                "close_price": round(mark_price, 4),
                "market_value_rs": round(market_value, 2),
                "weight_pct": weight_pct,
            })

        daily_holdings.extend(holdings_snapshot)

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(total_value, 2),
            "equity": round(total_value / STARTING_CAPITAL, 8),
            "cash_rs": round(cash, 2),
            "n_holdings": len(holdings),
        })

        if day_number % 100 == 0:
            print(f"Processed {day_number}/{n_days} days | {date.strftime('%Y-%m-%d')} | "
                  f"Holdings: {len(holdings)} | Equity: Rs.{total_value:,.0f}")

    # ========================================================
    # TERMINAL MARKED VALUE
    # ========================================================
    final_marked_value = equity_curve[-1]["portfolio_value_rs"] if equity_curve else STARTING_CAPITAL

    # ========================================================
    # TERMINAL LIQUIDATION
    # ========================================================
    liquidation_cash = cash
    open_positions_detail = []

    if len(trading_days) > 0 and holdings:
        last_date = pd.Timestamp(trading_days[-1]).normalize()
        for sym, pos in holdings.items():
            row = get_row(all_signals[sym], last_date)
            exit_price = float(row["price"]) if row is not None else pos["entry_price"]

            gross_proceeds = pos["qty"] * exit_price
            s_cost = sell_side_cost(gross_proceeds)
            net_proceeds = gross_proceeds - s_cost
            cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
            net_gain = net_proceeds - cost_basis
            tax = stcg_tax(net_gain)
            liquidation_cash += net_proceeds - tax

            open_positions_detail.append({
                "symbol": sym,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"], 4),
                "last_price": round(exit_price, 4),
                "unrealized_gross_return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
            })

    final_liquidation_value = liquidation_cash

    trade_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity_curve)
    top10_df = pd.DataFrame(daily_top10)
    holdings_df = pd.DataFrame(daily_holdings)
    open_df = pd.DataFrame(open_positions_detail)

    if not equity_df.empty:
        running_max = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = ((equity_df["equity"] / running_max - 1) * 100).round(3)

    return trade_df, equity_df, top10_df, holdings_df, open_df, final_marked_value, final_liquidation_value


# ============================================================
# SUMMARY
# ============================================================

def summarize(trade_df, equity_df, final_marked_value, final_liquidation_value):
    if equity_df.empty:
        return {}

    marked_return = (final_marked_value / STARTING_CAPITAL - 1) * 100
    liquidation_return = (final_liquidation_value / STARTING_CAPITAL - 1) * 100

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] / running_max - 1) * 100
    max_dd = round(drawdown.min(), 2)

    if not trade_df.empty:
        n = len(trade_df)
        win_rate_net = round((trade_df["net_return_pct"] > 0).mean() * 100, 1)
        win_rate_gross = round((trade_df["gross_return_pct"] > 0).mean() * 100, 1)
        avg_gross = round(trade_df["gross_return_pct"].mean(), 2)
        avg_net = round(trade_df["net_return_pct"].mean(), 2)
        median_net = round(trade_df["net_return_pct"].median(), 2)
        avg_days = round(trade_df["days_held"].mean(), 1)
        best_gross = trade_df["gross_return_pct"].max()
        worst_gross = trade_df["gross_return_pct"].min()
        total_costs_rs = round((trade_df["buy_cost_rs"] + trade_df["sell_cost_rs"]).sum(), 0)
        total_tax_rs = round(trade_df["stcg_tax_rs"].sum(), 0)
        winners = trade_df[trade_df["net_return_pct"] > 0]
        losers = trade_df[trade_df["net_return_pct"] < 0]
        avg_winner = round(winners["net_return_pct"].mean(), 2) if len(winners) else 0
        avg_loser = round(losers["net_return_pct"].mean(), 2) if len(losers) else 0
        gp = winners["net_pnl_rs"].sum() if len(winners) else 0
        gl = abs(losers["net_pnl_rs"].sum()) if len(losers) else 0
        profit_factor = round(gp / gl, 3) if gl > 0 else 0
    else:
        n = win_rate_net = win_rate_gross = avg_gross = avg_net = median_net = avg_days = 0
        best_gross = worst_gross = total_costs_rs = total_tax_rs = avg_winner = avg_loser = profit_factor = 0

    daily_returns = equity_df["equity"].pct_change().dropna()
    if len(daily_returns):
        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std()
        n_days = len(equity_df)
        annualized_return = equity_df["equity"].iloc[-1] ** (252 / max(n_days, 1)) - 1
        annualized_vol = daily_std * np.sqrt(252)
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) else 0
        sortino = (daily_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        annualized_return = annualized_vol = sharpe = sortino = 0

    calmar = annualized_return / abs(max_dd / 100) if abs(max_dd) > 0 else 0

    return {
        "Backtest Start": BACKTEST_START,
        "Backtest End": equity_df["date"].iloc[-1],
        "Starting Capital (Rs)": STARTING_CAPITAL,
        "Final Value (marked, Rs)": round(final_marked_value, 0),
        "Final Value (liquidation, Rs)": round(final_liquidation_value, 0),
        "Net Return - marked (%)": round(marked_return, 2),
        "Net Return - liquidation (%)": round(liquidation_return, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max Drawdown (%)": max_dd,
        "Number of Closed Trades": n,
        "Win Rate - Gross (%)": win_rate_gross,
        "Win Rate - Net (%)": win_rate_net,
        "Avg Gross Return/Trade (%)": avg_gross,
        "Avg Net Return/Trade (%)": avg_net,
        "Median Net Return/Trade (%)": median_net,
        "Avg Days Held": avg_days,
        "Avg Winner (%)": avg_winner,
        "Avg Loser (%)": avg_loser,
        "Profit Factor (net)": profit_factor,
        "Best Gross Trade (%)": best_gross,
        "Worst Gross Trade (%)": worst_gross,
        "Total Costs Paid (Rs)": total_costs_rs,
        "Total STCG Tax Paid (Rs)": total_tax_rs,
        "Exit Rule": f"Exit if not in Top {TOP_N} (no rank buffer)",
        "Execution Timing": "T+1 (signal on D, filled at D+1 close)",
    }


# ============================================================
# GOOGLE SHEETS HELPERS (unchanged from prior version)
# ============================================================

def safe_update(ws, values, cell_range, label=""):
    for attempt in range(MAX_WRITE_RETRIES):
        try:
            ws.update(values, cell_range)
            return
        except Exception as e:
            error_text = str(e)
            is_quota = "429" in error_text or "Quota exceeded" in error_text
            if not is_quota or attempt == MAX_WRITE_RETRIES - 1:
                raise
            wait = INITIAL_RETRY_SECONDS * (2 ** attempt)
            print(f"Google Sheets quota limit for {label}. Waiting {wait}s...")
            time.sleep(wait)


def write_dataframe(ws, df, start_row, start_col, label):
    if df is None or df.empty:
        return start_row

    values = [list(df.columns)] + [
        ["" if pd.isna(x) else x for x in row]
        for row in df.itertuples(index=False, name=None)
    ]
    total = len(values)

    def col_letter(col):
        result = ""
        while col:
            col, remainder = divmod(col - 1, 26)
            result = chr(65 + remainder) + result
        return result

    col1 = col_letter(start_col)
    col2 = col_letter(start_col + len(values[0]) - 1)

    for offset in range(0, total, WRITE_CHUNK_SIZE):
        chunk = values[offset:offset + WRITE_CHUNK_SIZE]
        row1 = start_row + offset
        row2 = row1 + len(chunk) - 1
        rng = f"{col1}{row1}:{col2}{row2}"
        safe_update(ws, chunk, rng, label)
        print(f"Wrote {label}: {min(offset + WRITE_CHUNK_SIZE, total)}/{total}")

    return start_row + total


def remove_existing_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()
        requests = []
        for sheet in meta.get("sheets", []):
            if sheet["properties"]["sheetId"] == sheet_id:
                for chart in sheet.get("charts", []):
                    requests.append({"deleteEmbeddedObject": {"objectId": chart["chartId"]}})
        if requests:
            sh.batch_update({"requests": requests})
            print(f"Removed {len(requests)} old chart(s).")
    except Exception as e:
        print(f"Chart removal skipped: {e}")


def add_charts(sh, sheet_id, equity_header_row_0idx, n_equity_rows):
    data_end_row = equity_header_row_0idx + 1 + n_equity_rows

    def make_chart(title, y_col_idx, y_axis_title, anchor_row):
        return {
            "addChart": {
                "chart": {
                    "spec": {
                        "title": title,
                        "basicChart": {
                            "chartType": "LINE",
                            "legendPosition": "NO_LEGEND",
                            "axis": [
                                {"position": "BOTTOM_AXIS", "title": "Date"},
                                {"position": "LEFT_AXIS", "title": y_axis_title},
                            ],
                            "domains": [{"domain": {"sourceRange": {"sources": [
                                {"sheetId": sheet_id, "startRowIndex": equity_header_row_0idx,
                                 "endRowIndex": data_end_row, "startColumnIndex": 0, "endColumnIndex": 1}
                            ]}}}],
                            "series": [{"series": {"sourceRange": {"sources": [
                                {"sheetId": sheet_id, "startRowIndex": equity_header_row_0idx,
                                 "endRowIndex": data_end_row, "startColumnIndex": y_col_idx,
                                 "endColumnIndex": y_col_idx + 1}
                            ]}}, "targetAxis": "LEFT_AXIS"}],
                        },
                    },
                    "position": {"overlayPosition": {
                        "anchorCell": {"sheetId": sheet_id, "rowIndex": anchor_row, "columnIndex": 9},
                        "widthPixels": 700, "heightPixels": 400,
                    }},
                }
            }
        }

    requests = [
        make_chart("Equity Curve", 1, "Portfolio Value (Rs)", equity_header_row_0idx),
        make_chart("Drawdown", 5, "Drawdown %", equity_header_row_0idx + 22),
    ]
    try:
        sh.batch_update({"requests": requests})
        print("Charts added.")
    except Exception as e:
        print(f"Charts could not be added: {e}")


def write_to_sheet(trade_df, equity_df, top10_df, holdings_df, open_df, summary, effective_end_str):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print("Missing Google credentials. Saving CSV files.")
        trade_df.to_csv("backtest_trades.csv", index=False)
        equity_df.to_csv("backtest_equity.csv", index=False)
        top10_df.to_csv("backtest_daily_top10.csv", index=False)
        holdings_df.to_csv("backtest_daily_holdings.csv", index=False)
        if not open_df.empty:
            open_df.to_csv("backtest_open_positions.csv", index=False)
        return

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(BACKTEST_WORKSHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=BACKTEST_WORKSHEET, rows=1000, cols=40)

    remove_existing_charts(sh, ws.id)

    required_rows = max(1000, len(summary) + 20, len(top10_df) + 20, len(holdings_df) + 20,
                         len(equity_df) + 20, len(trade_df) + 20, len(open_df) + 20)
    required_cols = max(40,
                         len(top10_df.columns) if not top10_df.empty else 1,
                         len(holdings_df.columns) if not holdings_df.empty else 1,
                         len(equity_df.columns) if not equity_df.empty else 1,
                         len(trade_df.columns) if not trade_df.empty else 1)

    if ws.row_count < required_rows or ws.col_count < required_cols:
        ws.resize(rows=max(ws.row_count, required_rows), cols=max(ws.col_count, required_cols))

    print("\nClearing old top 10 RS backtest...")
    safe_update(ws, [[""]], "A1", "initial sheet check")
    ws.clear()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    header = [[
        f"TOP 10 RS BACKTEST | Run: {timestamp} | NET of costs + STCG | "
        f"Capital: Rs.{STARTING_CAPITAL:,.0f} | EXECUTION: T+1 CLOSE | "
        f"Exit: not in Top {TOP_N} (no buffer) | Window: {BACKTEST_START} to {effective_end_str}"
    ]]
    safe_update(ws, header, "A1", "header")

    summary_rows = [["SUMMARY", ""]] + [[k, v] for k, v in summary.items()]
    safe_update(ws, summary_rows, f"A3:B{2 + len(summary_rows)}", "summary")

    top10_start = 5 + len(summary_rows)
    safe_update(ws, [["DAILY TOP 10 RS STOCKS"]], f"A{top10_start}", "top10 title")
    top10_header_row = top10_start + 1
    write_dataframe(ws, top10_df, top10_header_row, 1, "daily top 10")

    holdings_start = top10_header_row + len(top10_df) + 3
    safe_update(ws, [["DAILY HOLDINGS"]], f"A{holdings_start}", "holdings title")
    holdings_header_row = holdings_start + 1
    write_dataframe(ws, holdings_df, holdings_header_row, 1, "daily holdings")

    equity_start = holdings_header_row + len(holdings_df) + 3
    safe_update(ws, [["DAILY EQUITY CURVE"]], f"A{equity_start}", "equity title")
    equity_header_row = equity_start + 1
    write_dataframe(ws, equity_df, equity_header_row, 1, "equity curve")

    trade_start = equity_header_row + len(equity_df) + 3
    safe_update(ws, [["TRADE LOG"]], f"A{trade_start}", "trade title")
    trade_header_row = trade_start + 1
    write_dataframe(ws, trade_df, trade_header_row, 1, "trade log")

    open_start = trade_header_row + len(trade_df) + 3
    safe_update(ws, [["OPEN POSITIONS AT BACKTEST END"]], f"A{open_start}", "open positions title")
    open_header_row = open_start + 1
    write_dataframe(ws, open_df, open_header_row, 1, "open positions")

    if not equity_df.empty:
        add_charts(sh, ws.id, equity_header_row - 1, len(equity_df))

    print("\n================================================")
    print("RESULTS WRITTEN SUCCESSFULLY")
    print(f"Sheet: {BACKTEST_WORKSHEET}")
    print(f"Daily Top 10 rows: {len(top10_df)}")
    print(f"Daily holdings rows: {len(holdings_df)}")
    print(f"Equity rows: {len(equity_df)}")
    print(f"Trades: {len(trade_df)}")
    print("================================================")


# ============================================================
# MAIN
# ============================================================

def main():
    tickers = load_tickers()
    print(f"\nLoaded {len(tickers)} tickers.")

    download_start, download_end = get_download_dates()
    print("=" * 60)
    print(f"Download start   : {download_start}")
    print(f"Download end     : {download_end if download_end else 'LATEST AVAILABLE'}")
    print(f"Backtest start   : {BACKTEST_START}")
    print(f"Backtest end     : {BACKTEST_END if BACKTEST_END else 'LATEST AVAILABLE'}")
    print(f"Price filter     : > Rs.{MIN_PRICE}")
    print(f"Liquidity        : {VOLUME_LOOKBACK}d avg volume > {MIN_AVG_VOLUME:,}")
    print("Entry            : Top 10 RS")
    print("Execution        : T+1 CLOSE")
    print(f"Exit             : not in Top {TOP_N} (no rank buffer)")
    print(f"Capital          : Rs.{STARTING_CAPITAL:,.0f}")
    print("=" * 60)

    bench_close = download_benchmark()
    bench_close.index = normalize_dates(bench_close.index)

    all_signals = {}
    total_bad_points = 0
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\nDownloading batch {i}-{i + len(batch)}...")
        try:
            data = yf.download(batch, start=download_start, end=download_end, interval="1d",
                                auto_adjust=True, progress=False, group_by="ticker", threads=True)
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
                    clean_symbol = symbol.replace(".NS", "")
                    all_signals[clean_symbol] = sig

            except Exception as e:
                print(f"Skipping {symbol}: {e}")

        time.sleep(1)

    print(f"\nSignals computed for {len(all_signals)} stocks.")
    print(f"Total repaired data points: {total_bad_points}")

    if all_signals:
        latest_stock_date = max(df.index.max() for df in all_signals.values())
    else:
        latest_stock_date = None

    benchmark_latest_date = pd.Timestamp(bench_close.index.max()).normalize()

    if BACKTEST_END is None:
        effective_end = benchmark_latest_date
    else:
        effective_end = min(pd.Timestamp(BACKTEST_END).normalize(), benchmark_latest_date)

    print(f"\nLatest benchmark data: {benchmark_latest_date.strftime('%Y-%m-%d')}")
    if latest_stock_date is not None:
        print(f"Latest stock data: {pd.Timestamp(latest_stock_date).strftime('%Y-%m-%d')}")
    print(f"Effective backtest end: {effective_end.strftime('%Y-%m-%d')}")

    trading_days = bench_close.index[
        (bench_close.index >= pd.Timestamp(BACKTEST_START).normalize()) &
        (bench_close.index <= effective_end)
    ]
    trading_days = pd.DatetimeIndex(trading_days).drop_duplicates().sort_values()

    print(f"\nTrading days: {len(trading_days)}")
    if len(trading_days):
        print(f"First trading day: {trading_days[0].strftime('%Y-%m-%d')}")
        print(f"Last trading day: {trading_days[-1].strftime('%Y-%m-%d')}")

    if not len(trading_days):
        raise RuntimeError("No trading days found.")

    print("\nRunning T+1 execution, exit-if-not-Top10 backtest...")

    (trade_df, equity_df, top10_df, holdings_df, open_df,
     final_marked, final_liq) = run_backtest(all_signals, trading_days)

    summary = summarize(trade_df, equity_df, final_marked, final_liq)

    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    write_to_sheet(trade_df, equity_df, top10_df, holdings_df, open_df, summary,
                    effective_end.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    try:
        main()
        print("\nBACKTEST COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print("\nBACKTEST FAILED")
        print(f"{type(e).__name__}: {e}")
        raise
