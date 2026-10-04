#!/usr/bin/env python3
"""
RS BACKTESTS - COMBINED RUNNER
==============================

One file, one data download, all strategies run in a single execution.
Each strategy writes to its OWN Google Sheet tab; a "Backtest - Comparison"
tab puts the headline metrics side by side.

STRATEGIES
----------
 s1  "Backtest - RS Top10 + Price TT"
     Eligible = Price>20, 20D avg vol>100k, Price Trend Template 7/7.
     Rank by raw RS (40% 3M + 20% 6M + 20% 9M + 20% 12M). Hold Top 10,
     equal-weight at entry, exit when rank > 15. Daily EOD, T+0.
     (= the THIRD backtest)

 s2  "Backtest - RS Top10"
     Eligible = Price>20, 20D avg vol>100k. NO trend template.
     Hold exactly today's Top 10 RS. Sell on rank 11+. Freed cash funds
     all missing names; retained positions never resized. Daily EOD, T+0.
     (= the TOP-10 RS ROTATION backtest)

 s4  "Backtest - RAM Momentum"   (NEW - designed in this file)
     Risk-Adjusted Momentum + regime filter + weekly rebalance.
     See the RAM STRATEGY block below for the full rule set.

USAGE
-----
    python rs_backtests_combined.py                # run all
    python rs_backtests_combined.py --only s1 s4   # run a subset

Env: SHEET_ID, GOOGLE_CREDENTIALS  (missing -> CSVs in ./backtest_outputs)
Input: stocks.csv with a 'symbol' column.
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIG - SHARED
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"
STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = None            # None = latest available
DOWNLOAD_YEARS_BEFORE_START = 3

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20
MAX_PLAUSIBLE_DAILY_MOVE = 0.30

STARTING_CAPITAL = 1_000_000
OUTPUT_DIR = "backtest_outputs"

CHART_WINDOWS = (50, 100, 365)

# ---- s1 / s2 ----
TOP_N = 10
EXIT_RANK = 15                 # s1 only
MIN_HISTORY_S1 = 280           # aligned-with-benchmark history (as original)
MIN_HISTORY_S2 = 252 + 20      # as original

# ---- RAM STRATEGY (s4) ----
# Idea: rank by momentum PER UNIT OF RISK, not raw momentum, and only
# hold it when the market is healthy.
#   Score   = (50% 12-1M return + 50% 6M return) / max(126D vol, 15%)
#   Rank    = 70% pct-rank(Score) + 30% pct-rank(proximity to 52W high)
#   Filter  = price>20, liquid, close>200DMA, within 20% of 52W high,
#             blended momentum > 0 (absolute-momentum / dual-momentum gate)
#   Regime  = benchmark > its 200DMA, else everything goes to cash
#   Hold    = top 15, equal-weight at entry, NEVER resize winners
#   Buffer  = sell only if rank > 30 (hysteresis -> low turnover)
#   Stop    = 20% trailing stop from highest close since entry (checked daily)
#   Rebal   = weekly (first trading day of the ISO week)
#   Timing  = signals from PRIOR close, trades at TODAY'S close (no same-bar lookahead)
S4_N = 15
S4_EXIT_RANK = 30
S4_W_12_1 = 0.5
S4_W_RANK_MOM = 0.7
S4_MIN_PROX = 0.80
S4_VOL_FLOOR = 0.15
S4_TRAIL_STOP = 0.20
S4_REGIME_SMA = 200

# ---- transaction costs (NSE delivery) ----
STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20
STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
COMPARISON_SHEET = "Backtest - Comparison"


# ============================================================
# DATE / DATA HELPERS
# ============================================================

def normalize_dates(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def normalize_series_index(series):
    s = series.copy()
    s.index = normalize_dates(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def get_download_dates():
    start = pd.Timestamp(BACKTEST_START) - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    if BACKTEST_END is None:
        return start.strftime("%Y-%m-%d"), None
    end = pd.Timestamp(BACKTEST_END) + pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")
    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")
    symbols = [s for s in df["symbol"].dropna().astype(str).str.strip().tolist() if s]
    out = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    return list(dict.fromkeys(out))


def clean_price_series(close):
    """Forward-fill single-day moves larger than MAX_PLAUSIBLE_DAILY_MOVE."""
    close = normalize_series_index(close)
    bad = close.pct_change().abs() > MAX_PLAUSIBLE_DAILY_MOVE
    n_bad = int(bad.sum())
    if n_bad == 0:
        return close, 0
    cleaned = close.copy()
    for idx in close.index[bad]:
        pos = cleaned.index.get_loc(idx)
        if pos > 0:
            cleaned.iloc[pos] = cleaned.iloc[pos - 1]
    return cleaned, n_bad


def download_benchmark():
    import yfinance as yf
    start, end = get_download_dates()
    print(f"\nBenchmark download: {start} -> {end if end else 'LATEST'}")
    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(ticker, start=start, end=end, interval="1d",
                               auto_adjust=True, progress=False)
            if data.empty:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = normalize_series_index(close.dropna())
            if close.empty:
                continue
            close, n_bad = clean_price_series(close)
            if n_bad:
                print(f"Benchmark {ticker}: repaired {n_bad} points")
            print(f"Benchmark loaded: {ticker} (latest {close.index.max():%Y-%m-%d})")
            return close
        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")
    raise RuntimeError("Could not download benchmark data.")


# ============================================================
# COSTS / TAX
# ============================================================

def buy_side_cost(v):
    exch = EXCHANGE_CHARGE_RATE * v
    sebi = SEBI_CHARGE_RATE * v
    return STT_RATE * v + STAMP_DUTY_RATE * v + exch + sebi + GST_RATE * (exch + sebi)


def sell_side_cost(v):
    exch = EXCHANGE_CHARGE_RATE * v
    sebi = SEBI_CHARGE_RATE * v
    return STT_RATE * v + exch + sebi + GST_RATE * (exch + sebi) + DP_CHARGE_FLAT


def stcg_tax(gain):
    return gain * STCG_EFFECTIVE_RATE if gain > 0 else 0.0


def max_affordable_qty(price, budget):
    if price <= 0 or budget <= 0:
        return 0
    qty = int(budget / price)
    while qty > 0:
        v = qty * price
        if v + buy_side_cost(v) <= budget + 1e-9:
            return qty
        qty -= 1
    return 0


def min_cash_for_one_share(price):
    return price + buy_side_cost(price) if price > 0 else float("inf")


# ============================================================
# SIGNAL BUILDING (per stock) - computed ONCE, shared by all strategies
# ============================================================

def momentum_score(s):
    def r(d):
        return s / s.shift(d) - 1
    return (0.40 * r(63) + 0.20 * r(126) + 0.20 * r(189) + 0.20 * r(252)) * 100


def trend_template(s):
    sma50, sma150, sma200 = (s.rolling(n).mean() for n in (50, 150, 200))
    conds = [
        (s > sma150) & (s > sma200),
        sma150 > sma200,
        sma200 > sma200.shift(21),
        (sma50 > sma150) & (sma50 > sma200),
        s > sma50,
        s >= 1.25 * s.rolling(252).min(),
        s >= 0.75 * s.rolling(252).max(),
    ]
    return sum(c.astype(int) for c in conds) == 7


def ram_features(close, volume):
    """Risk-adjusted momentum score + 52W-high proximity, NaN where ineligible."""
    ret6 = close / close.shift(126) - 1
    ret12_1 = close.shift(21) / close.shift(252) - 1
    vol = close.pct_change().rolling(126).std() * np.sqrt(252)
    raw = S4_W_12_1 * ret12_1 + (1 - S4_W_12_1) * ret6
    score = raw / vol.clip(lower=S4_VOL_FLOOR)
    sma200 = close.rolling(200).mean()
    prox = close / close.rolling(252).max()
    liquid = (close > MIN_PRICE) & (volume.rolling(VOLUME_LOOKBACK).mean() > MIN_AVG_VOLUME)
    elig = (liquid & (close > sma200) & (prox >= S4_MIN_PROX)
            & (raw > 0) & score.notna())
    return score.where(elig), prox.where(elig)


def build_symbol_series(close, volume, bench_close):
    """Returns dict of Series keyed p1,s1 | p2,s2 | s4,x4 (subset if too little history)."""
    out = {}
    close = normalize_series_index(close)
    volume = normalize_series_index(volume)

    # ---- s1: benchmark-aligned history, price TT ----
    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) >= MIN_HISTORY_S1:
        p = aligned["s"]
        v = volume.reindex(aligned.index).fillna(0)
        liquid = (p > MIN_PRICE) & (v.rolling(VOLUME_LOOKBACK).mean() > MIN_AVG_VOLUME)
        rs = momentum_score(p)
        out["p1"] = p
        out["s1"] = rs.where(rs.notna() & liquid & trend_template(p))

    # ---- s2 / s4: own calendar ----
    if len(close) >= MIN_HISTORY_S2:
        liquid = (close > MIN_PRICE) & (volume.rolling(VOLUME_LOOKBACK).mean() > MIN_AVG_VOLUME)
        rs = momentum_score(close)
        out["p2"] = close
        out["s2"] = rs.where(rs.notna() & liquid & (close > 0))
        s4, x4 = ram_features(close, volume)
        out["s4"] = s4
        out["x4"] = x4
    return out


# ============================================================
# TRADE PRIMITIVES (shared by all engines)
# ============================================================

TRADE_COLS = ["symbol", "entry_date", "exit_date", "qty", "entry_price", "exit_price",
              "gross_return_pct", "buy_cost_rs", "sell_cost_rs", "stcg_tax_rs",
              "net_pnl_rs", "net_return_pct", "days_held", "action",
              "exit_reason", "exit_rank"]


def entry_row(sym, date, qty, price, buy_cost):
    r = {c: "" for c in TRADE_COLS}
    r.update(symbol=sym, entry_date=date.strftime("%Y-%m-%d"), qty=int(qty),
             entry_price=round(price, 4), buy_cost_rs=round(buy_cost, 2), action="ENTRY")
    return r


def do_buy(sym, price, qty, date, cash, holdings, trades):
    """Returns (cash, bought?)."""
    if qty < 1:
        return cash, False
    v = qty * price
    bc = buy_side_cost(v)
    if v + bc > cash + 1e-9:
        return cash, False
    holdings[sym] = {"qty": int(qty), "entry_price": float(price), "entry_date": date,
                     "entry_cost": float(bc), "last_price": float(price),
                     "peak": float(price)}
    trades.append(entry_row(sym, date, qty, price, bc))
    return cash - (v + bc), True


def do_sell(sym, pos, date, exit_price, reason, exit_rank=""):
    """Returns (cash_delta, trade_row). cash_delta is net of costs and STCG."""
    qty = int(pos["qty"])
    gross = qty * exit_price
    sc = sell_side_cost(gross)
    net = gross - sc
    basis = qty * pos["entry_price"] + pos["entry_cost"]
    gain = net - basis
    tax = stcg_tax(gain)
    pnl = gain - tax
    r = {c: "" for c in TRADE_COLS}
    r.update(
        symbol=sym, entry_date=pos["entry_date"].strftime("%Y-%m-%d"),
        exit_date=date.strftime("%Y-%m-%d"), qty=qty,
        entry_price=round(pos["entry_price"], 4), exit_price=round(exit_price, 4),
        gross_return_pct=round((exit_price / pos["entry_price"] - 1) * 100, 2),
        buy_cost_rs=round(pos["entry_cost"], 2), sell_cost_rs=round(sc, 2),
        stcg_tax_rs=round(tax, 2), net_pnl_rs=round(pnl, 2),
        net_return_pct=round(pnl / basis * 100, 2) if basis > 0 else 0,
        days_held=(date - pos["entry_date"]).days, action="EXIT",
        exit_reason=reason, exit_rank=exit_rank)
    return net - tax, r


def liquidate(holdings, final_prices, cash):
    """Terminal liquidation value (costs + STCG) and open-position table."""
    liq = float(cash)
    rows = []
    for sym, pos in holdings.items():
        px = float(final_prices.get(sym, pos["last_price"]))
        gross = pos["qty"] * px
        net = gross - sell_side_cost(gross)
        basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
        liq += net - stcg_tax(net - basis)
        rows.append({
            "symbol": sym, "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
            "qty": pos["qty"], "entry_price": round(pos["entry_price"], 4),
            "last_price": round(px, 4),
            "gross_return_pct": round((px / pos["entry_price"] - 1) * 100, 2),
            "entry_cost_rs": round(pos["entry_cost"], 2),
        })
    return pd.DataFrame(rows), liq


def eq_row(date, value, cash, n_hold, pool):
    return {"date": date.strftime("%Y-%m-%d"), "portfolio_value_rs": round(value, 2),
            "cash_rs": round(cash, 2), "invested_value_rs": round(value - cash, 2),
            "equity_multiple": round(value / STARTING_CAPITAL, 8),
            "n_holdings": n_hold, "eligible_pool_size": int(pool)}


def add_equity_analytics(df):
    if df.empty:
        return df
    run_max = df["equity_multiple"].cummax()
    df["drawdown_pct"] = ((df["equity_multiple"] / run_max - 1) * 100).round(3)
    first = float(df["portfolio_value_rs"].iloc[0])
    df["equity_curve_pct_norm"] = ((df["portfolio_value_rs"] / first - 1) * 100).round(3)
    for w in CHART_WINDOWS:
        start = max(len(df) - w, 0)
        base = float(df["portfolio_value_rs"].iloc[start])
        s = pd.Series(np.nan, index=df.index, dtype=float)
        s.iloc[start:] = ((df["portfolio_value_rs"].iloc[start:] / base - 1) * 100).round(3)
        df[f"equity_curve_pct_norm_last{w}"] = s
    return df


def ranked_from_scores(score_row, cols):
    """Eligible symbols, best score first (stable -> ties keep file order)."""
    idx = np.flatnonzero(~np.isnan(score_row))
    order = idx[np.argsort(-score_row[idx], kind="stable")]
    return list(cols[order])


# ============================================================
# ENGINE s1 - RS Top10 + Price TT, exit rank > 15
# ============================================================

def run_s1(P, days, ctx):
    score_df, price_df = P["s1"], P["p1"]
    if score_df.shape[1] == 0:
        raise RuntimeError("s1: no usable stocks.")
    cols = np.array(score_df.columns)
    cidx = {s: j for j, s in enumerate(cols)}
    score, price = score_df.values, price_df.values

    cash = float(STARTING_CAPITAL)
    holdings, trades, eq = {}, [], []

    for i, date in enumerate(days):
        ranked = ranked_from_scores(score[i], cols)
        rank_of = {s: r + 1 for r, s in enumerate(ranked)}
        pool = len(ranked)

        # exits: rank > EXIT_RANK or left eligible universe
        for sym in list(holdings):
            r = rank_of.get(sym)
            if r is not None and r <= EXIT_RANK:
                continue
            px = price[i, cidx[sym]]
            if np.isnan(px):
                continue
            pos = holdings.pop(sym)
            d, row = do_sell(sym, pos, date, float(px),
                             f"Rank > {EXIT_RANK}" if r is not None else "Left eligible universe",
                             r if r is not None else "")
            cash += d
            trades.append(row)

        # value before entries (original behaviour: missing price -> entry price)
        pv = cash
        for sym, pos in holdings.items():
            px = price[i, cidx[sym]]
            pv += pos["qty"] * (pos["entry_price"] if np.isnan(px) else float(px))

        slots = TOP_N - len(holdings)
        if slots > 0:
            slot_cap = pv / TOP_N
            for sym in ranked[:TOP_N]:
                if slots <= 0:
                    break
                if sym in holdings:
                    continue
                px = float(price[i, cidx[sym]])
                qty = int(slot_cap // px) if px > 0 else 0
                if qty < 1:
                    continue
                cash, ok = do_buy(sym, px, qty, date, cash, holdings, trades)
                if ok:
                    slots -= 1

        val = cash
        for sym, pos in holdings.items():
            px = price[i, cidx[sym]]
            val += pos["qty"] * (pos["entry_price"] if np.isnan(px) else float(px))
        eq.append(eq_row(date, val, cash, len(holdings), pool))

    final_px = {s: float(price[-1, cidx[s]]) for s in holdings
                if not np.isnan(price[-1, cidx[s]])}
    open_df, liq = liquidate(holdings, final_px, cash)
    eq_df = add_equity_analytics(pd.DataFrame(eq))
    return (pd.DataFrame(trades, columns=TRADE_COLS), eq_df, open_df,
            float(eq_df["portfolio_value_rs"].iloc[-1]), liq)


# ============================================================
# ENGINE s2 - exact Top 10 RS rotation
# ============================================================

def _buy_missing(missing, price_lookup, date, cash, holdings, trades):
    """Fund every missing Top-10 name from available cash; reserve one share
    for each later name so an early buy can never starve a later one."""
    if not missing:
        return cash
    for s in missing:
        p = price_lookup.get(s)
        if p is None or pd.isna(p) or p <= 0:
            raise RuntimeError(f"{date:%Y-%m-%d}: {s} has invalid entry price.")
    need = sum(min_cash_for_one_share(price_lookup[s]) for s in missing)
    if cash + 1e-9 < need:
        raise RuntimeError(f"{date:%Y-%m-%d}: cannot fill Top 10 - cash Rs.{cash:,.2f} "
                           f"< minimum Rs.{need:,.2f} for {len(missing)} names.")
    for i, s in enumerate(missing):
        p = float(price_lookup[s])
        reserve = sum(min_cash_for_one_share(price_lookup[x]) for x in missing[i + 1:])
        avail = cash - reserve
        budget = min(max(cash / (len(missing) - i), min_cash_for_one_share(p)), avail)
        qty = max_affordable_qty(p, budget)
        if qty < 1:
            raise RuntimeError(f"{date:%Y-%m-%d}: sizing failure for {s}.")
        cash, ok = do_buy(s, p, qty, date, cash, holdings, trades)
        if not ok:
            raise RuntimeError(f"{date:%Y-%m-%d}: buy failed for {s}.")
    return cash


def run_s2(P, days, ctx):
    score_df, price_df = P["s2"], P["p2"]
    if score_df.shape[1] == 0:
        raise RuntimeError("s2: no usable stocks.")
    cols = np.array(score_df.columns)
    cidx = {s: j for j, s in enumerate(cols)}
    score, price = score_df.values, price_df.reindex(columns=score_df.columns).values

    cash = float(STARTING_CAPITAL)
    holdings, trades, eq = {}, [], []
    initialized = False
    lookup = {}

    for i, date in enumerate(days):
        ranked = ranked_from_scores(score[i], cols)
        rank_of = {s: r + 1 for r, s in enumerate(ranked)}
        lookup = {s: float(price[i, cidx[s]]) for s in ranked}
        pool = len(ranked)
        target = min(TOP_N, pool)
        top = ranked[:TOP_N]
        top_set = set(top)

        if not initialized:
            if top:
                cash = _buy_missing(top, lookup, date, cash, holdings, trades)
            initialized = True
        else:
            for sym in [s for s in holdings if s not in top_set]:
                pos = holdings.pop(sym)
                if sym in lookup:
                    px, why = lookup[sym], f"RANK_{rank_of.get(sym)}_DROPPED_OUTSIDE_TOP10"
                else:
                    px, why = float(pos["last_price"]), "MISSING_FROM_RANKING_FORCE_EXIT"
                d, row = do_sell(sym, pos, date, px, why, rank_of.get(sym, ""))
                cash += d
                trades.append(row)
            for sym, pos in holdings.items():
                if sym in lookup:
                    pos["last_price"] = lookup[sym]
            missing = [s for s in top if s not in holdings]
            if missing:
                cash = _buy_missing(missing, lookup, date, cash, holdings, trades)

        # invariants: exactly today's Top 10 (or the whole pool if < 10)
        if set(holdings) != top_set:
            raise RuntimeError(f"{date:%Y-%m-%d}: holdings != Top-{TOP_N} "
                               f"(held {len(holdings)}, target {target}).")

        val = cash
        for sym, pos in holdings.items():
            if sym in lookup:
                pos["last_price"] = lookup[sym]
            val += pos["qty"] * pos["last_price"]
        eq.append(eq_row(date, val, cash, len(holdings), pool))

    open_df, liq = liquidate(holdings, lookup, cash)
    eq_df = add_equity_analytics(pd.DataFrame(eq))
    return (pd.DataFrame(trades, columns=TRADE_COLS), eq_df, open_df,
            float(eq_df["portfolio_value_rs"].iloc[-1]), liq)


# ============================================================
# ENGINE s4 - RAM: risk-adjusted momentum + regime + weekly
# ============================================================

def run_s4(P, days, ctx):
    score_df = P["s4"]
    if score_df.shape[1] == 0:
        raise RuntimeError("s4: no usable stocks.")
    cols = np.array(score_df.columns)
    cidx = {s: j for j, s in enumerate(cols)}
    score = score_df.values
    prox = P["x4"].reindex(columns=score_df.columns).values
    price = P["p2"].reindex(columns=score_df.columns).values
    regime = (ctx["bench_regime"].reindex(days, method="ffill")
              .fillna(False).astype(bool).values)
    iso = days.isocalendar()
    wk = list(zip(iso["year"].tolist(), iso["week"].tolist()))

    cash = float(STARTING_CAPITAL)
    holdings, trades, eq = {}, [], []
    started = False

    def exit_pos(sym, date, px, why, rank=""):
        nonlocal cash
        d, row = do_sell(sym, holdings.pop(sym), date, px, why, rank)
        cash += d
        trades.append(row)

    for i, date in enumerate(days):
        px_row = price[i]

        # refresh last price / trailing peak
        for sym, pos in holdings.items():
            p = px_row[cidx[sym]]
            if not np.isnan(p):
                pos["last_price"] = float(p)
                pos["peak"] = max(pos["peak"], float(p))

        # daily trailing stop
        for sym in list(holdings):
            pos = holdings[sym]
            p = px_row[cidx[sym]]
            if not np.isnan(p) and p <= pos["peak"] * (1 - S4_TRAIL_STOP):
                exit_pos(sym, date, float(p), f"TRAILING_STOP_{int(S4_TRAIL_STOP * 100)}PCT")

        # weekly rebalance using the PRIOR day's signals
        if i >= 1 and (not started or wk[i] != wk[i - 1]):
            started = True
            k = i - 1
            sc = score[k]
            idx = np.flatnonzero(~np.isnan(sc))
            ranked = []
            if len(idx):
                s_pct = pd.Series(sc[idx]).rank(pct=True).values
                x_pct = pd.Series(prox[k][idx]).rank(pct=True).values
                comp = S4_W_RANK_MOM * s_pct + (1 - S4_W_RANK_MOM) * x_pct
                ranked = list(cols[idx[np.argsort(-comp, kind="stable")]])
            rank_of = {s: r + 1 for r, s in enumerate(ranked)}
            risk_on = bool(regime[k])

            for sym in list(holdings):
                p = px_row[cidx[sym]]
                if np.isnan(p):
                    continue
                r = rank_of.get(sym)
                if not risk_on:
                    exit_pos(sym, date, float(p), "REGIME_OFF", r if r else "")
                elif r is None or r > S4_EXIT_RANK:
                    exit_pos(sym, date, float(p),
                             f"RANK_>{S4_EXIT_RANK}" if r else "LEFT_ELIGIBLE_UNIVERSE",
                             r if r else "")

            if risk_on:
                slots = S4_N - len(holdings)
                if slots > 0:
                    pv = cash + sum(p["qty"] * p["last_price"] for p in holdings.values())
                    slot_cap = pv / S4_N
                    for sym in ranked[:S4_N]:
                        if slots <= 0:
                            break
                        if sym in holdings:
                            continue
                        p = px_row[cidx[sym]]
                        if np.isnan(p) or p <= 0:
                            continue
                        qty = max_affordable_qty(float(p), min(slot_cap, cash))
                        if qty < 1:
                            continue
                        cash, ok = do_buy(sym, float(p), qty, date, cash, holdings, trades)
                        if ok:
                            slots -= 1

        val = cash + sum(p["qty"] * p["last_price"] for p in holdings.values())
        eq.append(eq_row(date, val, cash, len(holdings),
                         np.count_nonzero(~np.isnan(score[i]))))

    open_df, liq = liquidate(holdings, {}, cash)
    eq_df = add_equity_analytics(pd.DataFrame(eq))
    return (pd.DataFrame(trades, columns=TRADE_COLS), eq_df, open_df,
            float(eq_df["portfolio_value_rs"].iloc[-1]), liq)


# ============================================================
# SUMMARY
# ============================================================

def summarize(trade_df, equity_df, open_df, marked, liq, meta):
    if equity_df.empty:
        return {}
    eqm = equity_df["equity_multiple"]
    max_dd = float(equity_df["drawdown_pct"].min())
    daily = eqm.pct_change().dropna()
    if len(daily) > 1 and daily.std() > 0:
        n = len(equity_df)
        ann_ret = eqm.iloc[-1] ** (252 / max(n, 1)) - 1
        ann_vol = daily.std() * np.sqrt(252)
        sharpe = daily.mean() / daily.std() * np.sqrt(252)
        dn = daily[daily < 0]
        sortino = (daily.mean() / dn.std() * np.sqrt(252)) if len(dn) > 1 and dn.std() > 0 else 0
    else:
        ann_ret = ann_vol = sharpe = sortino = 0
    calmar = ann_ret / abs(max_dd / 100) if max_dd != 0 else 0

    exits = trade_df[trade_df["action"] == "EXIT"] if not trade_df.empty else pd.DataFrame()
    entries = trade_df[trade_df["action"] == "ENTRY"] if not trade_df.empty else pd.DataFrame()
    st = dict.fromkeys(["win_net", "win_gross", "avg_net", "med_net", "avg_gross", "avg_win",
                        "avg_loss", "pf", "days", "best", "worst", "costs", "tax"], 0)
    if not exits.empty:
        net = exits["net_return_pct"].astype(float)
        gross = exits["gross_return_pct"].astype(float)
        pnl = exits["net_pnl_rs"].astype(float)
        st.update(
            win_net=(net > 0).mean() * 100, win_gross=(gross > 0).mean() * 100,
            avg_net=net.mean(), med_net=net.median(), avg_gross=gross.mean(),
            avg_win=net[net > 0].mean() if (net > 0).any() else 0,
            avg_loss=net[net < 0].mean() if (net < 0).any() else 0,
            pf=(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 0,
            days=exits["days_held"].astype(float).mean(),
            best=gross.max(), worst=gross.min(),
            costs=(exits["buy_cost_rs"].astype(float) + exits["sell_cost_rs"].astype(float)).sum(),
            tax=exits["stcg_tax_rs"].astype(float).sum())
    if not open_df.empty:
        st["costs"] += open_df["entry_cost_rs"].sum()

    out = {
        "Strategy": meta["label"],
        "Backtest Start": BACKTEST_START,
        "Backtest End": equity_df["date"].iloc[-1],
        "Starting Capital (Rs)": STARTING_CAPITAL,
        "Final Value - Marked (Rs)": round(marked, 0),
        "Final Value - Liquidation (Rs)": round(liq, 0),
        "Net Return - Marked (%)": round((marked / STARTING_CAPITAL - 1) * 100, 2),
        "Net Return - Liquidation (%)": round((liq / STARTING_CAPITAL - 1) * 100, 2),
        "Annualized Return (%)": round(ann_ret * 100, 2),
        "Annualized Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3), "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3), "Max Drawdown (%)": round(max_dd, 2),
        "Closed Trades": len(exits), "Entries": len(entries),
        "Win Rate - Gross (%)": round(st["win_gross"], 1),
        "Win Rate - Net (%)": round(st["win_net"], 1),
        "Avg Gross Return/Trade (%)": round(st["avg_gross"], 2),
        "Avg Net Return/Trade (%)": round(st["avg_net"], 2),
        "Median Net Return/Trade (%)": round(st["med_net"], 2),
        "Avg Winner (%)": round(st["avg_win"], 2),
        "Avg Loser (%)": round(st["avg_loss"], 2),
        "Profit Factor (net)": round(st["pf"], 3),
        "Avg Days Held": round(st["days"], 1),
        "Best Gross Trade (%)": st["best"], "Worst Gross Trade (%)": st["worst"],
        "Total Costs Paid (Rs)": round(st["costs"], 0),
        "Total STCG Tax Paid (Rs)": round(st["tax"], 0),
    }
    out.update(meta["rules"])
    return out


# ============================================================
# STRATEGY REGISTRY
# ============================================================

STRATEGIES = {
    "s1": {
        "sheet": "Backtest - RS Top10 + Price TT",
        "label": "S1: RS Top10 + Price TT (exit rank>15)",
        "fn": run_s1,
        "rules": {
            "RS Score": "40% 3M + 20% 6M + 20% 9M + 20% 12M",
            "Eligibility": "Price>Rs.20 + 20D avg vol>100k + Price TT 7/7",
            "Portfolio": "Top 10, equal weight at entry",
            "Exit": f"Rank > {EXIT_RANK} or left eligible universe",
            "Rebalance": "Daily EOD, T+0",
        },
    },
    "s2": {
        "sheet": "Backtest - RS Top10",
        "label": "S2: Pure RS Top10 rotation",
        "fn": run_s2,
        "rules": {
            "RS Score": "40% 3M + 20% 6M + 20% 9M + 20% 12M",
            "Eligibility": "Price>Rs.20 + 20D avg vol>100k (no trend filter)",
            "Portfolio": "Exact daily Top 10; retained names never resized",
            "Exit": "Rank 11+ or missing from ranking",
            "Rebalance": "Daily EOD, T+0; freed cash funds all missing names",
        },
    },
    "s4": {
        "sheet": "Backtest - RAM Momentum",
        "label": "S4: RAM (risk-adj momentum + regime)",
        "fn": run_s4,
        "rules": {
            "Score": f"({S4_W_12_1:.0%} 12-1M + {1 - S4_W_12_1:.0%} 6M return) / max(126D vol, {S4_VOL_FLOOR:.0%})",
            "Rank": f"{S4_W_RANK_MOM:.0%} pct-rank(score) + {1 - S4_W_RANK_MOM:.0%} pct-rank(proximity to 52W high)",
            "Eligibility": f"Price>Rs.20, liquid, close>200DMA, within {1 - S4_MIN_PROX:.0%} of 52W high, momentum>0",
            "Regime": f"Benchmark > {S4_REGIME_SMA}DMA else 100% cash",
            "Portfolio": f"Top {S4_N}, equal weight at entry, winners never resized",
            "Exit": f"Rank > {S4_EXIT_RANK}, regime off, or {S4_TRAIL_STOP:.0%} trailing stop",
            "Rebalance": "Weekly; signals from prior close, trade at today's close",
        },
    },
}


# ============================================================
# GOOGLE SHEETS
# ============================================================

def sanitize_for_sheets(df):
    if df.empty:
        return df
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.where(pd.notnull(clean), "")


def sanitize_scalar(v):
    if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
        return ""
    if isinstance(v, np.generic):
        return v.item()
    return v


def with_retry(fn, *args, max_retries=6, wait0=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            quota = "429" in str(e) or "Quota exceeded" in str(e)
            if not quota or attempt == max_retries - 1:
                raise
            wait = wait0 * (2 ** attempt)
            print(f"  Google quota hit, waiting {wait}s...")
            time.sleep(wait)


def open_spreadsheet():
    sheet_id, creds_json = os.environ.get(SHEET_ID_ENV), os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        return None
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        json.loads(creds_json), scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(sheet_id)


def get_or_create_ws(sh, title, rows=1000, cols=16):
    import gspread
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        pass
    try:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)
    except gspread.exceptions.APIError as e:
        if "already exists" in str(e):
            return sh.worksheet(title)
        raise


def write_chunks(ws, rows, start_row, label, chunk=2000):
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        with_retry(ws.update, part, f"A{start_row + i}")
        print(f"  wrote {label}: {min(i + chunk, len(rows))}/{len(rows)}")


def remove_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()
        reqs = [{"deleteEmbeddedObject": {"objectId": c["chartId"]}}
                for s in meta.get("sheets", []) if s["properties"]["sheetId"] == sheet_id
                for c in s.get("charts", [])]
        if reqs:
            with_retry(sh.batch_update, {"requests": reqs})
    except Exception as e:
        print(f"  could not clear charts (non-fatal): {e}")


def add_charts(sh, sheet_id, header_row_0, n_rows, columns, prefix):
    col_idx = {c: i for i, c in enumerate(columns)}
    data_end = header_row_0 + 1 + n_rows
    anchor_col = len(columns) + 1          # to the right of the data, not on top of it

    def win_start(w):
        return header_row_0 + 1 + max(n_rows - w, 0)

    def chart(title, col, ytitle, anchor_row, start=None, points=False, width=650):
        y = col_idx[col]
        s0 = start if start is not None else header_row_0
        series = {"series": {"sourceRange": {"sources": [{
            "sheetId": sheet_id, "startRowIndex": s0, "endRowIndex": data_end,
            "startColumnIndex": y, "endColumnIndex": y + 1}]}}, "targetAxis": "LEFT_AXIS"}
        if points:
            series["pointStyle"] = {"size": 5, "shape": "CIRCLE"}
            series["dataLabel"] = {"type": "DATA", "placement": "BELOW",
                                   "textFormat": {"fontSize": 7}}
        return {"addChart": {"chart": {
            "spec": {"title": f"{prefix} - {title}", "basicChart": {
                "chartType": "LINE", "legendPosition": "NO_LEGEND",
                "axis": [{"position": "BOTTOM_AXIS", "title": "Date"},
                         {"position": "LEFT_AXIS", "title": ytitle}],
                "domains": [{"domain": {"sourceRange": {"sources": [{
                    "sheetId": sheet_id, "startRowIndex": s0, "endRowIndex": data_end,
                    "startColumnIndex": 0, "endColumnIndex": 1}]}}}],
                "series": [series]}},
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": sheet_id, "rowIndex": anchor_row,
                               "columnIndex": anchor_col},
                "widthPixels": width, "heightPixels": 380}}}}}

    r0 = header_row_0
    reqs = [
        chart("Equity Curve (Rs)", "portfolio_value_rs", "Portfolio Value (Rs)", r0),
        chart("Drawdown (%)", "drawdown_pct", "Drawdown %", r0 + 22),
        chart("Eligible Pool Size", "eligible_pool_size", "Stock Count", r0 + 44),
        chart("Equity Normalised (%, Since Inception)", "equity_curve_pct_norm",
              "Cumulative Change %", r0 + 66),
        chart("Equity Normalised (%, Last 50 Days)", "equity_curve_pct_norm_last50",
              "Cumulative Change %", r0 + 88, win_start(50), True, 1100),
        chart("Equity Normalised (%, Last 100 Days)", "equity_curve_pct_norm_last100",
              "Cumulative Change %", r0 + 110, win_start(100)),
        chart("Equity Normalised (%, Last 365 Days)", "equity_curve_pct_norm_last365",
              "Cumulative Change %", r0 + 132, win_start(365)),
    ]
    try:
        with_retry(sh.batch_update, {"requests": reqs})
    except Exception as e:
        print(f"  could not add charts (non-fatal): {e}")


def write_strategy_sheet(sh, cfg, trade_df, equity_df, open_df, summary, end_str):
    n_cols = max(len(trade_df.columns), len(equity_df.columns),
                 len(open_df.columns) if not open_df.empty else 0, 2)
    n_rows = len(trade_df) + len(equity_df) + len(open_df) + len(summary) + 60
    ws = get_or_create_ws(sh, cfg["sheet"], rows=n_rows, cols=n_cols)
    if ws.row_count < n_rows or ws.col_count < n_cols:
        with_retry(ws.resize, rows=max(ws.row_count, n_rows), cols=max(ws.col_count, n_cols))
    remove_charts(sh, ws.id)
    with_retry(ws.clear)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    with_retry(ws.update, [[f"{cfg['label']} | run {ts} | NET of costs+STCG | "
                            f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
                            f"Window: {BACKTEST_START} to {end_str}"]], "A1")

    srows = [["Summary", ""]] + [[k, sanitize_scalar(v)] for k, v in summary.items()]
    with_retry(ws.update, srows, "A3")

    t_title = 3 + len(srows) + 2
    with_retry(ws.update, [["Trade Log"]], f"A{t_title}")
    t_head = t_title + 1
    if not trade_df.empty:
        c = sanitize_for_sheets(trade_df)
        write_chunks(ws, [list(c.columns)] + c.values.tolist(), t_head, "trade log")

    o_title = t_head + len(trade_df) + 3
    with_retry(ws.update, [["Open Positions at Backtest End (mark-to-market)"]], f"A{o_title}")
    o_head = o_title + 1
    if not open_df.empty:
        c = sanitize_for_sheets(open_df)
        with_retry(ws.update, [list(c.columns)] + c.values.tolist(), f"A{o_head}")

    e_title = o_head + max(len(open_df), 1) + 3
    with_retry(ws.update, [["Daily Equity Curve"]], f"A{e_title}")
    e_head = e_title + 1
    if not equity_df.empty:
        c = sanitize_for_sheets(equity_df)
        write_chunks(ws, [list(c.columns)] + c.values.tolist(), e_head, "equity curve")
        add_charts(sh, ws.id, e_head - 1, len(equity_df), list(c.columns), cfg["label"][:2].upper())
    print(f"  -> '{cfg['sheet']}': {len(trade_df)} trade rows, {len(equity_df)} days.")


COMPARE_KEYS = [
    "Final Value - Liquidation (Rs)", "Net Return - Liquidation (%)",
    "Annualized Return (%)", "Annualized Volatility (%)", "Sharpe", "Sortino", "Calmar",
    "Max Drawdown (%)", "Closed Trades", "Win Rate - Net (%)", "Avg Net Return/Trade (%)",
    "Avg Winner (%)", "Avg Loser (%)", "Profit Factor (net)", "Avg Days Held",
    "Total Costs Paid (Rs)", "Total STCG Tax Paid (Rs)",
]


def build_comparison(summaries):
    rows = [["Metric"] + [s.get("Strategy", k) for k, s in summaries.items()]]
    for key in COMPARE_KEYS:
        rows.append([key] + [sanitize_scalar(s.get(key, "")) for s in summaries.values()])
    rows.append(["Rules"] + ["" for _ in summaries])
    for key in ("Eligibility", "Portfolio", "Exit", "Rebalance"):
        rows.append([key] + [str(s.get(key, "")) for s in summaries.values()])
    return rows


# ============================================================
# MAIN
# ============================================================

def main(only):
    import yfinance as yf

    keys = [k for k in STRATEGIES if (not only or k in only)]
    print("=" * 70)
    print("RS BACKTESTS - COMBINED RUN:", ", ".join(keys))
    print("=" * 70)

    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers.")
    d_start, d_end = get_download_dates()
    bench = download_benchmark()

    store = {k: {} for k in ("p1", "s1", "p2", "s2", "s4", "x4")}
    latest_stock = None
    bad_total = 0

    for b0 in range(0, len(tickers), 50):
        batch = tickers[b0:b0 + 50]
        print(f"\nDownloading {b0 + 1}-{b0 + len(batch)} of {len(tickers)}")
        try:
            data = yf.download(batch, start=d_start, end=d_end, interval="1d",
                               auto_adjust=True, progress=False,
                               group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch failed: {e}")
            continue
        for sym in batch:
            try:
                if len(batch) == 1:
                    sdata = data
                else:
                    if not isinstance(data.columns, pd.MultiIndex):
                        continue
                    if sym not in data.columns.get_level_values(0):
                        continue
                    sdata = data[sym]
                if "Close" not in sdata.columns:
                    continue
                close = sdata["Close"].dropna().sort_index()
                if close.empty:
                    continue
                volume = sdata["Volume"].reindex(close.index).fillna(0)
                close, n_bad = clean_price_series(close)
                bad_total += n_bad
                name = sym.replace(".NS", "")
                for k, v in build_symbol_series(close, volume, bench).items():
                    store[k][name] = v
                last = close.index.max()
                latest_stock = last if latest_stock is None else max(latest_stock, last)
            except Exception as e:
                print(f"Skipping {sym}: {e}")
        time.sleep(1)

    print(f"\nUsable stocks - s1: {len(store['s1'])} | s2: {len(store['s2'])} | "
          f"s4: {len(store['s4'])} | repaired points: {bad_total}")
    if not store["p2"]:
        raise RuntimeError("No usable stock data.")

    bench_latest = bench.index.max().normalize()
    end = min(bench_latest, pd.Timestamp(latest_stock).normalize())
    if BACKTEST_END is not None:
        end = min(end, pd.Timestamp(BACKTEST_END).normalize())
    days = bench.index[(bench.index >= pd.Timestamp(BACKTEST_START).normalize())
                       & (bench.index <= end)]
    days = pd.DatetimeIndex(days).drop_duplicates().sort_values()
    if len(days) == 0:
        raise RuntimeError("No trading days found.")
    print(f"Trading days: {len(days)} ({days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d})")

    # wide panels (dates x symbols), built once and shared
    P = {k: pd.DataFrame({s: v.reindex(days) for s, v in d.items()}) for k, d in store.items()}
    del store
    ctx = {"bench_regime": normalize_series_index(
        bench > bench.rolling(S4_REGIME_SMA).mean())}

    sh = open_spreadsheet()
    if sh is None:
        print("SHEET_ID/GOOGLE_CREDENTIALS missing -> saving CSVs to ./" + OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summaries, failures = {}, {}
    for k in keys:
        cfg = STRATEGIES[k]
        print("\n" + "-" * 70 + f"\nRunning {cfg['label']}\n" + "-" * 70)
        t0 = time.time()
        try:
            trade_df, eq_df, open_df, marked, liq = cfg["fn"](P, days, ctx)
            summary = summarize(trade_df, eq_df, open_df, marked, liq, cfg)
            summaries[k] = summary
            print(f"  done in {time.time() - t0:.1f}s | "
                  f"liquidation Rs.{liq:,.0f} | CAGR {summary['Annualized Return (%)']}% | "
                  f"MaxDD {summary['Max Drawdown (%)']}% | trades {summary['Closed Trades']}")
            trade_df.to_csv(f"{OUTPUT_DIR}/{k}_trades.csv", index=False)
            eq_df.to_csv(f"{OUTPUT_DIR}/{k}_equity.csv", index=False)
            if not open_df.empty:
                open_df.to_csv(f"{OUTPUT_DIR}/{k}_open_positions.csv", index=False)
            if sh is not None:
                write_strategy_sheet(sh, cfg, trade_df, eq_df, open_df, summary,
                                     days[-1].strftime("%Y-%m-%d"))
        except Exception as e:           # one failing strategy must not kill the others
            failures[k] = f"{type(e).__name__}: {e}"
            print(f"  {k} FAILED: {failures[k]}")
            traceback.print_exc()

    if summaries:
        comp = build_comparison(summaries)
        pd.DataFrame(comp[1:], columns=comp[0]).to_csv(f"{OUTPUT_DIR}/comparison.csv", index=False)
        if sh is not None:
            ws = get_or_create_ws(sh, COMPARISON_SHEET, rows=60, cols=len(comp[0]) + 1)
            with_retry(ws.clear)
            with_retry(ws.update, [[f"STRATEGY COMPARISON | run "
                                    f"{datetime.now():%Y-%m-%d %H:%M IST} | "
                                    f"{BACKTEST_START} to {days[-1]:%Y-%m-%d}"]], "A1")
            with_retry(ws.update, comp, "A3")
        print("\n" + "=" * 70 + "\nCOMPARISON\n" + "=" * 70)
        for row in comp[:len(COMPARE_KEYS) + 1]:
            print(" | ".join(str(x) for x in row))

    if failures:
        print("\nFAILED STRATEGIES:", failures)
        sys.exit(1)
    print("\nALL BACKTESTS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(STRATEGIES), help="subset to run")
    args = ap.parse_args()
    main(args.only)