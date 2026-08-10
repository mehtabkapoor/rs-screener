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

# ---- Position sizing ----
STARTING_CAPITAL = 1_000_000       # Rs 10 lakh default, adjustable

# ---- Regime / breadth filter (same thresholds as the live screener) ----
ENABLE_REGIME_FILTER = True
BREADTH_RISK_ON = 60               # % of universe above 50DMA -> full-size entries
BREADTH_RISK_CAUTION = 40          # % -> half-size entries
BREADTH_CIRCUIT_BREAKER = 25       # % -> full defensive exit, no new entries

# ---- Transaction costs (Zerodha delivery: zero brokerage, statutory only) ----
ENABLE_COSTS = True
STT_RATE = 0.001                   # 0.1% Securities Transaction Tax, EACH side (buy+sell)
STAMP_DUTY_RATE = 0.00015          # 0.015% stamp duty, BUY side only
EXCHANGE_CHARGE_RATE = 0.0000325   # ~NSE exchange transaction charge, each side
SEBI_CHARGE_RATE = 0.000001        # Rs 10/crore SEBI turnover fee, each side
GST_RATE = 0.18                    # GST on (exchange + SEBI) charges
DP_CHARGE_FLAT = 20                # Rs per sell, per symbol (CDSL/NSDL depository charge)

# ---- STCG tax (Section 111A: all trades here are short-term, <12 months) ----
ENABLE_STCG = True
STCG_RATE = 0.20                   # 20% flat, effective FY2025-26/2026-27
STCG_CESS = 0.04                   # 4% health & education cess ON the tax amount
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)   # ~20.8% all-in

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest"
COMPARISON_WORKSHEET = "Backtest_Exit_Comparison"

# ---- Rank hysteresis (merged in from v4) ----
# Entry threshold = Top 10. Exit threshold = rank > RANK_EXIT.
# This means a holding can drift #5 -> #11 -> #15 -> #19 WITHOUT being sold
# purely on rank -- it's only forced out once it falls past RANK_EXIT, or
# the variant's own secondary exit fires, or the circuit breaker forces a
# full defensive exit. This applies to EVERY variant below (not a separate
# mutually-exclusive option), because rank hysteresis is the structural
# baseline risk control, and the secondary rule is a tactical layer on top.
RANK_EXIT = 20

# ---- Secondary exit variants (compared side by side, ALL combined with
# the RANK_EXIT hysteresis above) ----
# type options:
#   "rs_ema_state"   -- exit whenever RS line < RS EMA(span)
#   "rs_ema_cross"   -- exit only on the CROSSOVER day (was above, now below)
#   "none"           -- no secondary exit; rank hysteresis alone governs exits
#   "tt_fail_cross"  -- exit only when price Trend Template flips PASS->FAIL
EXIT_VARIANTS = [
    {"name": "RS<3EMA (state) + Rank20",         "type": "rs_ema_state", "span": 3},
    {"name": "RS<3EMA (crossover) + Rank20",      "type": "rs_ema_cross", "span": 3},
    {"name": "RS<5EMA (crossover) + Rank20",      "type": "rs_ema_cross", "span": 5},
    {"name": "RS<10EMA (crossover) + Rank20",     "type": "rs_ema_cross", "span": 10},
    {"name": "RS<20EMA (crossover) + Rank20",     "type": "rs_ema_cross", "span": 20},
    {"name": "RS<20EMA (state) + Rank20",         "type": "rs_ema_state", "span": 20},
    {"name": "Rank20 only (no secondary exit)",   "type": "none"},
    {"name": "Trend Template fail (cross) + Rank20", "type": "tt_fail_cross"},
]
PRIMARY_VARIANT_INDEX = 2   # RS<5EMA (crossover) + Rank20 -- the recommended default
# -----------------------------------------


def buy_side_cost(trade_value):
    """Total statutory cost of a BUY leg (Zerodha delivery: zero brokerage)."""
    if not ENABLE_COSTS:
        return 0.0
    stt = STT_RATE * trade_value
    stamp = STAMP_DUTY_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)
    return stt + stamp + exch + sebi + gst


def sell_side_cost(trade_value):
    """Total statutory cost of a SELL leg, including flat DP charge."""
    if not ENABLE_COSTS:
        return 0.0
    stt = STT_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)
    return stt + exch + sebi + gst + DP_CHARGE_FLAT


def stcg_tax(net_gain):
    """20.8% effective tax on a positive realized short-term gain. No tax
    on losses (a simplification -- real-world losses can offset other
    gains, so this is a conservative/pessimistic approximation)."""
    if not ENABLE_STCG or net_gain <= 0:
        return 0.0
    return net_gain * STCG_EFFECTIVE_RATE


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

    # For the regime/breadth filter: is price above its own 50DMA today?
    sma50 = aligned["s"].rolling(50).mean()
    above_50dma = aligned["s"] > sma50

    out = pd.DataFrame({
        "price": aligned["s"],
        "rs_line": rs_ratio,
        "rs_score": rs_score,
        "blue_dot": blue_dot,
        "tt_pass": tt_pass,
        "rs_tt_pass": rs_tt_pass,
        "liquid": liquid,
        "above_50dma": above_50dma,
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


def compute_daily_breadth(all_signals, trading_days):
    """
    Computes % of the universe above its own 50DMA for each trading day
    (same regime logic as the live screener), ONCE, shared across every
    exit variant -- this doesn't depend on the exit rule, so no need to
    recompute it 7 times.
    Returns a DataFrame indexed by date: breadth_pct, regime, allow_new_entries,
    size_multiplier, circuit_breaker.
    """
    rows = []
    for date in trading_days:
        flags = []
        for sym, df in all_signals.items():
            if date in df.index:
                val = df.loc[date, "above_50dma"]
                if pd.notna(val):
                    flags.append(bool(val))
        breadth_pct = round(100 * np.mean(flags), 1) if flags else 0

        if not ENABLE_REGIME_FILTER:
            regime, allow_new, size_mult, circuit = "REGIME FILTER OFF", True, 1.0, False
        elif breadth_pct >= BREADTH_RISK_ON:
            regime, allow_new, size_mult, circuit = "RISK-ON", True, 1.0, False
        elif breadth_pct >= BREADTH_RISK_CAUTION:
            regime, allow_new, size_mult, circuit = "CAUTION", True, 0.5, False
        elif breadth_pct >= BREADTH_CIRCUIT_BREAKER:
            regime, allow_new, size_mult, circuit = "RISK-OFF", False, 0.0, False
        else:
            regime, allow_new, size_mult, circuit = "CIRCUIT BREAKER", False, 0.0, True

        rows.append({"date": date, "breadth_pct": breadth_pct, "regime": regime,
                      "allow_new_entries": allow_new, "size_multiplier": size_mult,
                      "circuit_breaker": circuit})
    return pd.DataFrame(rows).set_index("date")


def run_backtest_for_variant(all_signals, trading_days, variant, breadth_df):
    """
    Simulates the portfolio for ONE exit variant with REAL position tracking:
    integer share quantities, actual cash balance, buy/sell transaction
    costs, STCG tax on realized gains, and regime-scaled position sizing.
    Reuses precomputed signals -- no re-downloading between variants.
    """
    cash = STARTING_CAPITAL
    holdings = {}   # sym -> {qty, entry_price, entry_date, entry_cost}
    trade_log = []
    equity_curve = []

    v_type = variant["type"]
    span = variant.get("span")

    for date in trading_days:
        regime_row = breadth_df.loc[date] if date in breadth_df.index else None
        allow_new_entries = bool(regime_row["allow_new_entries"]) if regime_row is not None else True
        size_multiplier = float(regime_row["size_multiplier"]) if regime_row is not None else 1.0
        circuit_breaker = bool(regime_row["circuit_breaker"]) if regime_row is not None else False
        regime_label = regime_row["regime"] if regime_row is not None else "N/A"

        # ---- Build today's eligible pool ----
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

        # ---- Mark-to-market current portfolio value (for sizing new entries) ----
        portfolio_value = cash
        for sym, pos in holdings.items():
            df = all_signals[sym]
            price = float(df.loc[date, "price"]) if date in df.index else pos["entry_price"]
            portfolio_value += pos["qty"] * price

        # ---- Exit logic: rank-20 hysteresis + variant's secondary exit, both
        # combined for every variant (a stock exits if EITHER fires), plus a
        # circuit breaker override that forces every position out regardless. ----
        for sym in list(holdings.keys()):
            df = all_signals[sym]
            if date not in df.index:
                continue
            row = df.loc[date]
            current_rank = rank_lookup.get(sym, 9999)
            rank_exit_trigger = current_rank > RANK_EXIT

            if v_type == "rs_ema_state":
                secondary_trigger = bool(row.get(f"rs_below_ema{span}", False))
                secondary_reason = f"RS Line < RS Line {span}-EMA"
            elif v_type == "rs_ema_cross":
                secondary_trigger = bool(row.get(f"rs_cross_below_ema{span}", False))
                secondary_reason = f"RS Line crossed below {span}-EMA"
            elif v_type == "tt_fail_cross":
                secondary_trigger = bool(row.get("tt_fail_cross", False))
                secondary_reason = "Trend Template PASS->FAIL"
            else:  # "none" -- rank hysteresis is the only exit mechanism
                secondary_trigger, secondary_reason = False, "N/A"

            if circuit_breaker:
                exit_now, reason = True, "CIRCUIT BREAKER: forced exit"
            elif secondary_trigger:
                exit_now, reason = True, secondary_reason
            elif rank_exit_trigger:
                exit_now = True
                reason = ("No longer in eligible pool" if current_rank == 9999
                           else f"Rank {current_rank} > {RANK_EXIT}")
            else:
                exit_now, reason = False, None

            if exit_now:
                pos = holdings.pop(sym)
                exit_price = float(row["price"])
                gross_proceeds = pos["qty"] * exit_price
                s_cost = sell_side_cost(gross_proceeds)
                net_proceeds = gross_proceeds - s_cost

                cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
                net_gain = net_proceeds - cost_basis
                tax = stcg_tax(net_gain)
                net_proceeds_after_tax = net_proceeds - tax

                cash += net_proceeds_after_tax
                net_return_pct = round((net_gain - tax) / cost_basis * 100, 2) if cost_basis > 0 else 0
                gross_return_pct = round((exit_price / pos["entry_price"] - 1) * 100, 2)

                trade_log.append({
                    "symbol": sym, "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "qty": pos["qty"],
                    "entry_price": round(pos["entry_price"], 2), "exit_price": round(exit_price, 2),
                    "gross_return_pct": gross_return_pct,
                    "buy_cost_rs": round(pos["entry_cost"], 2), "sell_cost_rs": round(s_cost, 2),
                    "stcg_tax_rs": round(tax, 2),
                    "net_pnl_rs": round(net_gain - tax, 2),
                    "net_return_pct": net_return_pct,
                    "days_held": (date - pos["entry_date"]).days,
                    "exit_reason": reason,
                    "exit_rank": current_rank if current_rank != 9999 else "",
                })

        # ---- Entries (skipped entirely if circuit breaker just fired) ----
        if not circuit_breaker and allow_new_entries:
            slot_capital = (portfolio_value / TOP_N) * size_multiplier
            slots_open = TOP_N - len(holdings)
            candidates = [s for s, _ in pool]  # already sorted by RS Score desc

            for sym in candidates:
                if slots_open <= 0:
                    break
                if sym in holdings:
                    continue
                price = float(all_signals[sym].loc[date, "price"])
                qty = int(slot_capital // price) if price > 0 else 0
                if qty < 1:
                    continue  # slot capital too small to buy even 1 share
                trade_value = qty * price
                b_cost = buy_side_cost(trade_value)
                total_cost = trade_value + b_cost
                if total_cost > cash:
                    continue  # not enough cash (shouldn't normally happen, safety check)
                cash -= total_cost
                holdings[sym] = {"qty": qty, "entry_price": price, "entry_date": date,
                                  "entry_cost": b_cost}
                slots_open -= 1

        # ---- Recompute portfolio value after today's entries/exits ----
        portfolio_value = cash
        for sym, pos in holdings.items():
            df = all_signals[sym]
            price = float(df.loc[date, "price"]) if date in df.index else pos["entry_price"]
            portfolio_value += pos["qty"] * price

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(portfolio_value, 2),
            "equity": round(portfolio_value / STARTING_CAPITAL, 6),
            "cash_rs": round(cash, 2),
            "n_holdings": len(holdings),
            "breadth_pct": regime_row["breadth_pct"] if regime_row is not None else None,
            "regime": regime_label,
        })

    # ---- Close any open positions at the end ----
    if trading_days.size:
        last_date = trading_days[-1]
        for sym, pos in list(holdings.items()):
            df = all_signals[sym]
            exit_price = float(df.loc[last_date, "price"]) if last_date in df.index else pos["entry_price"]
            gross_proceeds = pos["qty"] * exit_price
            s_cost = sell_side_cost(gross_proceeds)
            net_proceeds = gross_proceeds - s_cost
            cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
            net_gain = net_proceeds - cost_basis
            tax = stcg_tax(net_gain)
            gross_return_pct = round((exit_price / pos["entry_price"] - 1) * 100, 2)
            net_return_pct = round((net_gain - tax) / cost_basis * 100, 2) if cost_basis > 0 else 0

            trade_log.append({
                "symbol": sym, "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": last_date.strftime("%Y-%m-%d") + " (OPEN)",
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"], 2), "exit_price": round(exit_price, 2),
                "gross_return_pct": gross_return_pct,
                "buy_cost_rs": round(pos["entry_cost"], 2), "sell_cost_rs": round(s_cost, 2),
                "stcg_tax_rs": round(tax, 2),
                "net_pnl_rs": round(net_gain - tax, 2),
                "net_return_pct": net_return_pct,
                "days_held": (last_date - pos["entry_date"]).days,
                "exit_reason": "BACKTEST END (mark-to-market, not actually sold)",
                "exit_rank": "",
            })

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)


def summarize(trade_df, equity_df):
    if equity_df.empty:
        return {}
    # Net total return: the REAL portfolio value change, since equity_curve
    # is built from actual cash + positions after costs and tax are deducted.
    final_value = equity_df["portfolio_value_rs"].iloc[-1]
    net_total_return_pct = round((final_value / STARTING_CAPITAL - 1) * 100, 2)

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] / running_max - 1) * 100
    max_dd = round(drawdown.min(), 2)

    closed = trade_df[~trade_df["exit_date"].astype(str).str.contains("OPEN", na=False)] \
        if not trade_df.empty else trade_df
    n = len(closed)
    if n:
        win_rate_gross = round((closed["gross_return_pct"] > 0).mean() * 100, 1)
        win_rate_net = round((closed["net_return_pct"] > 0).mean() * 100, 1)
        avg_gross = round(closed["gross_return_pct"].mean(), 2)
        avg_net = round(closed["net_return_pct"].mean(), 2)
        median_net = round(closed["net_return_pct"].median(), 2)
        avg_days = round(closed["days_held"].mean(), 1)
        median_days = round(closed["days_held"].median(), 1)
        best_gross = closed["gross_return_pct"].max()
        worst_gross = closed["gross_return_pct"].min()
        total_costs_rs = round((closed["buy_cost_rs"] + closed["sell_cost_rs"]).sum(), 0)
        total_tax_rs = round(closed["stcg_tax_rs"].sum(), 0)

        # Winner/loser split and profit factor -- computed NET (after cost+tax)
        winners = closed[closed["net_return_pct"] > 0]
        losers = closed[closed["net_return_pct"] < 0]
        avg_winner_net = round(winners["net_return_pct"].mean(), 2) if len(winners) else 0
        avg_loser_net = round(losers["net_return_pct"].mean(), 2) if len(losers) else 0
        gross_profit_net = winners["net_pnl_rs"].sum() if len(winners) else 0
        gross_loss_net = abs(losers["net_pnl_rs"].sum()) if len(losers) else 0
        profit_factor_net = round(gross_profit_net / gross_loss_net, 3) if gross_loss_net > 0 else 0
    else:
        win_rate_gross = win_rate_net = avg_gross = avg_net = median_net = 0
        avg_days = median_days = best_gross = worst_gross = 0
        total_costs_rs = total_tax_rs = avg_winner_net = avg_loser_net = profit_factor_net = 0

    # ---- Risk-adjusted metrics, computed on the REAL (net) daily equity curve ----
    daily_returns = equity_df["equity"].pct_change().dropna()
    if len(daily_returns):
        daily_mean, daily_std = daily_returns.mean(), daily_returns.std()
        n_days = len(equity_df)
        annualized_return = equity_df["equity"].iloc[-1] ** (252 / max(n_days, 1)) - 1
        annualized_vol = daily_std * np.sqrt(252)
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) else 0
        sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        annualized_return = annualized_vol = sharpe = sortino = 0

    calmar = round(annualized_return / abs(max_dd / 100), 3) if abs(max_dd) > 0 else 0

    return {
        "final_portfolio_value_rs": round(final_value, 0),
        "net_total_return_pct": net_total_return_pct,
        "annualized_return_pct": round(annualized_return * 100, 2),
        "annualized_volatility_pct": round(annualized_vol * 100, 2),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3), "calmar": calmar,
        "max_dd_pct": max_dd,
        "n_trades": n,
        "win_rate_gross": win_rate_gross, "win_rate_net": win_rate_net,
        "avg_gross_return_per_trade": avg_gross, "avg_net_return_per_trade": avg_net,
        "median_net_return_per_trade": median_net,
        "avg_days_held": avg_days, "median_days_held": median_days,
        "avg_winner_net": avg_winner_net, "avg_loser_net": avg_loser_net,
        "profit_factor_net": profit_factor_net,
        "best_gross_trade": best_gross, "worst_gross_trade": worst_gross,
        "total_costs_rs": total_costs_rs, "total_stcg_tax_rs": total_tax_rs,
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
    print(f"Maximum holdings: {TOP_N}  |  Rank exit: > {RANK_EXIT} (hysteresis: entry top {TOP_N}, exit past {RANK_EXIT})")
    print(f"Starting capital: Rs.{STARTING_CAPITAL:,.0f}  |  Regime filter: {'ON' if ENABLE_REGIME_FILTER else 'OFF'}")
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

    # ---- Compute regime/breadth ONCE, shared across every variant ----
    print("\nComputing daily regime/breadth (shared across all variants)...")
    breadth_df = compute_daily_breadth(all_signals, trading_days)
    regime_counts = breadth_df["regime"].value_counts()
    print(f"Regime day counts: {regime_counts.to_dict()}")

    # ---- Run every exit variant against the SAME precomputed signals ----
    comparison_rows = []
    primary_trades, primary_equity = pd.DataFrame(), pd.DataFrame()

    for idx, variant in enumerate(EXIT_VARIANTS):
        print(f"\nRunning variant: {variant['name']}...")
        trades, equity = run_backtest_for_variant(all_signals, trading_days, variant, breadth_df)
        summary = summarize(trades, equity)
        summary["variant"] = variant["name"]
        comparison_rows.append(summary)
        print(f"  Net Return: {summary.get('net_total_return_pct')}% | "
              f"CAGR: {summary.get('annualized_return_pct')}% | "
              f"Sharpe: {summary.get('sharpe')} | Sortino: {summary.get('sortino')} | "
              f"Calmar: {summary.get('calmar')} | "
              f"Final Value: Rs.{summary.get('final_portfolio_value_rs'):,.0f} | "
              f"Max DD: {summary.get('max_dd_pct')}% | "
              f"Trades: {summary.get('n_trades')} | Net Win Rate: {summary.get('win_rate_net')}% | "
              f"Costs+Tax: Rs.{summary.get('total_costs_rs', 0) + summary.get('total_stcg_tax_rs', 0):,.0f}")

        if idx == PRIMARY_VARIANT_INDEX:
            primary_trades, primary_equity = trades, equity

    comparison_df = pd.DataFrame(comparison_rows)[
        ["variant", "final_portfolio_value_rs", "net_total_return_pct",
         "annualized_return_pct", "annualized_volatility_pct", "sharpe", "sortino", "calmar",
         "max_dd_pct", "n_trades", "win_rate_gross", "win_rate_net",
         "avg_gross_return_per_trade", "avg_net_return_per_trade", "median_net_return_per_trade",
         "avg_days_held", "median_days_held", "avg_winner_net", "avg_loser_net",
         "profit_factor_net", "best_gross_trade", "worst_gross_trade",
         "total_costs_rs", "total_stcg_tax_rs"]
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
    n_cols_needed = len(comparison_df.columns) + 2
    try:
        cws = sh.worksheet(COMPARISON_WORKSHEET)
        if cws.row_count < n_rows_needed or cws.col_count < n_cols_needed:
            cws.resize(rows=max(cws.row_count, n_rows_needed), cols=max(cws.col_count, n_cols_needed))
    except gspread.WorksheetNotFound:
        cws = sh.add_worksheet(title=COMPARISON_WORKSHEET, rows=n_rows_needed, cols=n_cols_needed)
    cws.clear()
    cws.update([[f"Exit variant comparison run: {timestamp} | NET of costs+STCG tax | "
                 f"Starting capital: Rs.{STARTING_CAPITAL:,.0f} | Regime filter: "
                 f"{'ON' if ENABLE_REGIME_FILTER else 'OFF'} | data-cleaned"]], "A1")
    cws.update([list(comparison_df.columns)] + comparison_df.values.tolist(), "A3")
    print(f"\nComparison written to '{COMPARISON_WORKSHEET}' tab.")

    # ---- Primary variant full detail tab ----
    n_cols_needed2 = max(len(trade_df.columns) if not trade_df.empty else 0,
                          len(equity_df.columns) if not equity_df.empty else 0) + 2
    n_rows_needed2 = max(len(trade_df) + len(equity_df) + 50, 100)
    try:
        ws = sh.worksheet(BACKTEST_WORKSHEET)
        if ws.row_count < n_rows_needed2 or ws.col_count < n_cols_needed2:
            ws.resize(rows=max(ws.row_count, n_rows_needed2), cols=max(ws.col_count, n_cols_needed2))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=BACKTEST_WORKSHEET, rows=n_rows_needed2, cols=n_cols_needed2)
    ws.clear()
    primary_name = EXIT_VARIANTS[PRIMARY_VARIANT_INDEX]["name"]
    ws.update([[f"Primary variant detail: {primary_name} | run {timestamp} | "
                f"NET of costs+STCG tax | Starting capital: Rs.{STARTING_CAPITAL:,.0f} | data-cleaned"]], "A1")
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
# 3. COSTS, STCG TAX, AND REGIME FILTER ARE NOW MODELED (v3 update):
#    - Real cash + integer share quantities, not a % approximation
#    - Zerodha delivery costs: STT, stamp duty, exchange/SEBI charges, DP
#      charges -- zero brokerage assumed (correct for Zerodha delivery)
#    - STCG tax at 20.8% effective (20% + 4% cess) on every profitable
#      trade, since all holds here are well under 12 months
#    - Regime/breadth filter (same 60/40/25 thresholds as the live
#      screener): scales position size in CAUTION, blocks new entries in
#      RISK-OFF, forces a full defensive exit in CIRCUIT BREAKER
#    - Position sizing is now REAL: slot_capital = portfolio_value/TOP_N,
#      scaled by regime, floor-divided into whole shares. A slot too small
#      to buy even 1 share is skipped (visible if STARTING_CAPITAL is set
#      very low relative to high-priced stocks in the universe).
#    NOTE: losses are NOT used to offset gains for tax purposes (real STCG
#    law allows loss carry-forward/set-off) -- this makes the tax estimate
#    conservative (i.e. real net returns could be somewhat better than shown).
#
# 4. RANK-20 HYSTERESIS IS NOW UNIVERSAL (v4 merge). Every variant combines
#    two exit conditions: rank falls past RANK_EXIT (20), OR the variant's
#    own secondary rule fires (RS-EMA cross/state, Trend Template fail).
#    A holding can drift #5 -> #11 -> #19 without being force-sold purely
#    on rank; it only exits at rank 21+, on the secondary trigger, or on a
#    circuit breaker. "Rank20 only" (secondary exit disabled) isolates how
#    much the hysteresis alone contributes vs. each tactical exit layered
#    on top -- that comparison is the most direct answer to "is a faster
#    RS-based exit actually adding value over the buffer alone."
#
# 5. Entries fill open slots from the full ranked pool (not strictly
#    literal top 10) whenever slots are open -- so if hysteresis keeps 6
#    positions in the 11-20 buffer zone, only 4 new slots open up, filled
#    by the 4 highest-RS-Score eligible names not already held.
