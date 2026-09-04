"""
TOP 10 RS ROTATION BACKTEST -- BLUE DOT FILTERED (GREEN DOT TRACKED, NOT USED FOR ENTRY)

This is a MODIFICATION of the exact-Top-10 rotation engine
(top10_rs_backtest.py). The eligibility gate used to build the
daily ranking is:

    OLD (backtest1):  liquid
    NEW (this file):  liquid AND blue_dot

Green Dot is still COMPUTED for every stock (identical definition
to backtest2 / the Pine-style diagnostic) and stored in each
stock's signal data, but it is NOT part of the eligibility gate --
it's tracked for reference/analysis only, it does not affect which
stocks are bought, ranked, or held.

Where:

    rs_ratio          = stock price / benchmark price
    blue_dot          = rs_ratio makes a new N-day (LOOKBACK_DAYS) high
                         i.e. rs_ratio > rolling max of rs_ratio over the
                         prior LOOKBACK_DAYS (shifted by 1 to exclude today)
    price_at_new_high = stock price makes a new N-day high
    green_dot         = blue_dot AND (price NOT simultaneously at a new
                         N-day high) -- i.e. RS strength emerging while
                         price hasn't broken out yet. COMPUTED ONLY --
                         not used to gate entries in this backtest.

PORTFOLIO RULE (every trading day) -- UNCHANGED FROM BACKTEST1
  1. Rank all eligible stocks (liquid + blue_dot) by RS score.
  2. Target portfolio = today's Top 10 of that filtered/ranked pool.
  3. Existing holdings that remain in target are NEVER resized.
  4. Sell holdings that leave the target (drop out of filtered Top 10,
     or lose blue_dot/liquidity, or vanish from the ranking).
  5. Buy ALL missing names from today's target.
  6. Replacement purchases use AVAILABLE CASH after exits.
  7. Cash is divided across missing Top-10 positions.
  8. Transaction costs are included in affordability calculations.
  9. If >=10 eligible (filtered) stocks exist and all required stocks
     are affordable, the portfolio MUST finish with exactly 10 holdings.

There is NO daily sell/rebuy, NO continuous equal weighting,
NO rank-11 substitution, NO price/RS trend template, NO
sector/regime/breadth filter. Note this is a much narrower/stricter
filter than the trend template used in backtest2 -- expect fewer
eligible names on many days, and possibly days with <10 eligible.
"""

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = None  # None = latest available market data

DOWNLOAD_YEARS_BEFORE_START = 3

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

RS_3M, RS_6M, RS_9M, RS_12M = 63, 126, 189, 252
RS_WEIGHTS = (0.40, 0.20, 0.20, 0.20)  # 3M / 6M / 9M / 12M

# Blue Dot / Green Dot lookback window (trading days) for the
# RS-ratio new-high and price new-high checks. Same as backtest2.
LOOKBACK_DAYS = 250

TOP_N = 10
STARTING_CAPITAL = 1_000_000
MAX_PLAUSIBLE_DAILY_MOVE = 0.30

# Chart windows (trading days), each rendered as its own
# equity-curve chart rebased to 0% at the window's start.
CHART_WINDOWS = (50, 100, 365)

# ============================================================
# TRANSACTION COSTS (India: NSE equity delivery)
# ============================================================

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20  # per-scrip, sell side only

STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)  # 20.8%

# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest - RS Top10 BlueGreenDot"


# ============================================================
# DATE / SERIES HELPERS
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
    start = pd.Timestamp(BACKTEST_START)
    download_start = start - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)

    if BACKTEST_END is None:
        return download_start.strftime("%Y-%m-%d"), None

    download_end = pd.Timestamp(BACKTEST_END) + pd.Timedelta(days=1)
    return download_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d")


def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")

    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")

    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]

    output = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    return list(dict.fromkeys(output))  # de-dup, preserve order


def clean_price_series(close):
    """Forward-fills single-day price spikes larger than
    MAX_PLAUSIBLE_DAILY_MOVE. Known limitation: two consecutive
    bad days are each checked against the ORIGINAL series, so a
    genuine two-day move can be partially miscorrected -- flagged
    here rather than silently trusted."""

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


# ============================================================
# TRANSACTION COST FUNCTIONS
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
    return net_gain * STCG_EFFECTIVE_RATE if net_gain > 0 else 0.0


def max_affordable_qty(price, cash_budget):
    """Largest integer quantity whose trade value + buy-side
    costs fits inside cash_budget. Never assumes
    portfolio_value / N -- always solves from actual cash."""

    if price <= 0 or cash_budget <= 0:
        return 0

    qty = int(cash_budget / price)
    while qty > 0:
        trade_value = qty * price
        if trade_value + buy_side_cost(trade_value) <= cash_budget + 1e-9:
            return qty
        qty -= 1
    return 0


def minimum_cash_for_one_share(price):
    if price <= 0:
        return float("inf")
    return price + buy_side_cost(price)


# ============================================================
# BENCHMARK
#
# Here the benchmark is used for TWO things (unlike backtest1,
# where it was calendar-only):
#   1. Establishing the common trading-date calendar (as before).
#   2. Computing rs_ratio = stock price / benchmark price, which
#      feeds Blue Dot / Green Dot for every stock.
# It still plays NO role in the RS SCORE used for ranking.
# ============================================================

def download_benchmark():
    download_start, download_end = get_download_dates()
    print(f"\nBenchmark download: {download_start} -> "
          f"{download_end if download_end else 'LATEST'}")

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
                print(f"Benchmark {ticker}: repaired {n_bad} points")
            print(f"Benchmark loaded: {ticker}")
            return close

        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")

    raise RuntimeError("Could not download benchmark data.")


# ============================================================
# STOCK SIGNAL CALCULATION
#
# RS SCORE: pure price momentum, benchmark-independent (unchanged
# from backtest1).
#
# BLUE DOT: benchmark-relative diagnostic (same definition as
# backtest2), used as the ELIGIBILITY GATE.
#
# GREEN DOT: also computed and stored (same definition as backtest2),
# but NOT used as part of the eligibility gate here -- kept purely
# for reference/analysis (e.g. inspecting the sheet/trade log to see
# which Blue-Dot entries also happened to be Green-Dot).
# ============================================================

def compute_stock_data(close, volume, bench_close):
    close = normalize_series_index(close)
    volume = normalize_series_index(volume)
    bench_close = normalize_series_index(bench_close)

    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) < 300:
        return None

    volume = volume.reindex(aligned.index).fillna(0)
    avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (aligned["s"] > MIN_PRICE) & (avg_volume > MIN_AVG_VOLUME)

    w3, w6, w9, w12 = RS_WEIGHTS
    rs_score = (
        w3 * (aligned["s"] / aligned["s"].shift(RS_3M) - 1)
        + w6 * (aligned["s"] / aligned["s"].shift(RS_6M) - 1)
        + w9 * (aligned["s"] / aligned["s"].shift(RS_9M) - 1)
        + w12 * (aligned["s"] / aligned["s"].shift(RS_12M) - 1)
    ) * 100

    # -- RS ratio vs benchmark (diagnostic input only) --
    rs_ratio = aligned["s"] / aligned["b"]

    # -- Blue Dot: RS ratio makes a new LOOKBACK_DAYS high --
    previous_rs_high = rs_ratio.shift(1).rolling(LOOKBACK_DAYS).max()
    blue_dot = rs_ratio > previous_rs_high

    # -- Price at a new LOOKBACK_DAYS high --
    previous_price_high = aligned["s"].shift(1).rolling(LOOKBACK_DAYS).max()
    price_at_new_high = aligned["s"] > previous_price_high

    # -- Green Dot: Blue Dot firing while price has NOT also broken out --
    green_dot = blue_dot & (~price_at_new_high)

    result = pd.DataFrame({
        "price": aligned["s"],
        "avg_volume": avg_volume,
        "liquid": liquid,
        "rs_score": rs_score,
        "rs_ratio": rs_ratio,
        "blue_dot": blue_dot,
        "green_dot": green_dot,
    })
    result.index = normalize_dates(result.index)
    return result


def get_row(df, date):
    date = pd.Timestamp(date).normalize()
    if date not in df.index:
        return None
    row = df.loc[date]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def build_wide_frames(all_stocks):
    """Precompute wide (date x symbol) frames ONCE before the backtest
    loop, instead of doing a per-symbol df.loc[date] Python-level
    lookup for every stock on every single trading day.

    build_daily_ranking (below) previously iterated `for symbol, df in
    all_stocks.items(): row = get_row(df, date); ...` inside the daily
    loop -- that's stocks x days individual pandas .loc calls (e.g.
    750 stocks x 2,500 days = ~1.9M calls), each carrying real
    per-call overhead (index hashing, Series construction, dict
    building). At full NSE-universe scale that alone can run into
    tens of minutes.

    Building 4 wide DataFrames up front (one vectorized pd.concat
    each) and then doing ONE row-slice per frame per day turns that
    into O(days) vectorized numpy operations instead of O(days x
    stocks) Python-level ones. Missing dates for a given stock
    naturally become NaN/False in the wide frame, which matches the
    original "date not in df.index -> treat as ineligible" behavior.
    """

    if not all_stocks:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    price_wide = pd.concat({s: df["price"] for s, df in all_stocks.items()}, axis=1)
    rs_wide = pd.concat({s: df["rs_score"] for s, df in all_stocks.items()}, axis=1)
    liquid_wide = pd.concat({s: df["liquid"] for s, df in all_stocks.items()},
                             axis=1).fillna(False).astype(bool)
    blue_wide = pd.concat({s: df["blue_dot"] for s, df in all_stocks.items()},
                           axis=1).fillna(False).astype(bool)

    return price_wide, rs_wide, liquid_wide, blue_wide


def build_daily_ranking(date, price_wide, rs_wide, liquid_wide, blue_wide):
    """Rank ELIGIBLE stocks by RS, descending -- vectorized version.

    Eligibility (this backtest only):
        liquid AND blue_dot

    Green Dot is still COMPUTED and stored on every stock's signal
    row (see compute_stock_data), but it is NOT part of the buying
    criteria here -- it's tracked/available for reference only.

    Same semantics as the original per-symbol-loop version, just
    computed via one row-slice per wide frame instead of iterating
    every stock in a Python for-loop.
    """

    date = pd.Timestamp(date).normalize()
    if rs_wide.empty or date not in rs_wide.index:
        return []

    rs_row = rs_wide.loc[date]
    price_row = price_wide.loc[date]
    liquid_row = liquid_wide.loc[date]
    blue_row = blue_wide.loc[date]

    eligible = liquid_row & blue_row & rs_row.notna() & price_row.notna() & (price_row > 0)
    if not eligible.any():
        return []

    sub_rs = rs_row[eligible].astype(float)
    sub_price = price_row[eligible].astype(float)
    ordered = sub_rs.sort_values(ascending=False)

    return [(sym, float(ordered[sym]), float(sub_price[sym])) for sym in ordered.index]


# ============================================================
# TRADE EXECUTION -- UNCHANGED FROM BACKTEST1
# ============================================================

def execute_buy(symbol, price, qty, date, cash, holdings, trade_log):
    if qty < 1:
        return cash, False

    trade_value = qty * price
    buy_cost = buy_side_cost(trade_value)
    total_required = trade_value + buy_cost
    if total_required > cash + 1e-9:
        return cash, False

    cash -= total_required
    holdings[symbol] = {
        "qty": int(qty),
        "entry_price": float(price),
        "entry_date": date,
        "entry_cost": float(buy_cost),
        "last_price": float(price),
        "last_price_date": date,
    }

    trade_log.append({
        "symbol": symbol, "entry_date": date.strftime("%Y-%m-%d"), "exit_date": "",
        "qty": int(qty), "entry_price": round(price, 4), "exit_price": "",
        "gross_return_pct": "", "buy_cost_rs": round(buy_cost, 2), "sell_cost_rs": "",
        "stcg_tax_rs": "", "net_pnl_rs": "", "net_return_pct": "", "days_held": "",
        "action": "ENTRY", "exit_reason": "",
    })
    return cash, True


def buy_missing_top10(missing_symbols, price_lookup, date, cash, holdings, trade_log,
                       unfilled_log=None):
    """Buy every missing Top-10 name that cash actually allows.

    UNLIKE backtest1, this does NOT raise when the exact target can't
    be fully funded. Blue Dot is a transient single-day signal
    (unlike a slow-moving RS score), so the daily target set can
    rotate almost completely, fragmenting cash into remainders that
    occasionally can't cover every missing name. Rather than treat
    that as fatal, we fill what cash currently allows, evenly split
    across remaining slots (no reservation for later names -- see
    the in-loop comment for why), and SKIP any name we can't fund --
    it simply stays unheld until cash frees up or it drops out of
    the target on a later day.

    missing_symbols is already in RS-rank order (best first), so
    skips fall on the lowest-conviction names when cash is tight.
    """

    if not missing_symbols:
        return cash

    for symbol in missing_symbols:
        price = price_lookup.get(symbol)
        if price is None or pd.isna(price) or float(price) <= 0:
            raise RuntimeError(f"{date:%Y-%m-%d}: Top-10 stock {symbol} "
                                f"has invalid entry price.")

    for i, symbol in enumerate(missing_symbols):
        price = float(price_lookup[symbol])
        slots_remaining = len(missing_symbols) - i

        # NOTE: no "reserve cash for later names" here (unlike
        # backtest1). That reservation logic assumed every name was
        # guaranteed affordable overall (backtest1 checked this
        # upfront and raised if not). Once full-fill is no longer
        # guaranteed, reserving for a LATER name that turns out to be
        # unaffordable anyway would wrongly zero out the budget for
        # THIS affordable name too. Instead: divide currently
        # available cash evenly across remaining slots, try to buy
        # this one, and let any cash a skip leaves behind roll
        # forward into the next iteration's (recomputed) equal split.
        equal_budget = cash / slots_remaining
        minimum_current = minimum_cash_for_one_share(price)
        target_budget = min(max(equal_budget, minimum_current), cash)

        qty = max_affordable_qty(price, target_budget)
        if qty < 1:
            # Can't afford even one share within the budget carved out
            # for this name -- skip it, don't crash the backtest.
            if unfilled_log is not None:
                unfilled_log.append({
                    "date": date.strftime("%Y-%m-%d"), "symbol": symbol,
                    "price": round(price, 4), "cash_at_skip": round(cash, 2),
                    "reason": "INSUFFICIENT_CASH_FOR_ONE_SHARE",
                })
            continue

        cash, bought = execute_buy(symbol, price, qty, date, cash, holdings, trade_log)
        if not bought and unfilled_log is not None:
            unfilled_log.append({
                "date": date.strftime("%Y-%m-%d"), "symbol": symbol,
                "price": round(price, 4), "cash_at_skip": round(cash, 2),
                "reason": "BUY_EXECUTION_FAILED",
            })

    return cash


# ============================================================
# BACKTEST ENGINE -- UNCHANGED FROM BACKTEST1
#
# (build_daily_ranking already encodes the Blue Dot filter, so
# everything downstream -- entries, exits, the hard portfolio
# invariants, mark-to-market -- is identical.)
# ============================================================

def run_backtest(all_stocks, trading_days):
    cash = float(STARTING_CAPITAL)
    holdings = {}
    trade_log = []
    equity_curve = []
    unfilled_log = []
    initialized = False
    n_days = len(trading_days)

    print("Building wide (date x symbol) frames for fast daily ranking...")
    build_start = time.time()
    price_wide, rs_wide, liquid_wide, blue_wide = build_wide_frames(all_stocks)
    print(f"Wide frames built in {time.time() - build_start:.1f}s "
          f"({len(all_stocks)} symbols x {len(rs_wide.index)} dates).")

    for day_number, date in enumerate(trading_days, start=1):
        date = pd.Timestamp(date).normalize()

        ranking = build_daily_ranking(date, price_wide, rs_wide, liquid_wide, blue_wide)
        rank_lookup = {sym: rank for rank, (sym, _, _) in enumerate(ranking, start=1)}
        price_lookup = {sym: float(price) for sym, _, price in ranking}

        eligible_pool_size = len(ranking)
        target_size = min(TOP_N, eligible_pool_size)
        today_top10 = [sym for sym, _, _ in ranking[:TOP_N]]
        today_top10_set = set(today_top10)

        if not initialized:
            if today_top10:
                cash = buy_missing_top10(today_top10, price_lookup, date, cash,
                                          holdings, trade_log, unfilled_log)
            initialized = True

        else:
            # -- exits: sell every holding that dropped out of the target --
            exit_symbols = [s for s in holdings if s not in today_top10_set]

            for symbol in exit_symbols:
                position = holdings.pop(symbol)

                if symbol in price_lookup:
                    exit_price = float(price_lookup[symbol])
                    exit_reason = f"RANK_{rank_lookup.get(symbol)}_DROPPED_OUTSIDE_TOP10"
                else:
                    exit_price = float(position.get("last_price", position["entry_price"]))
                    exit_reason = "MISSING_FROM_FILTERED_RANKING_FORCE_EXIT"

                qty = int(position["qty"])
                gross_proceeds = qty * exit_price
                sell_cost = sell_side_cost(gross_proceeds)
                net_proceeds = gross_proceeds - sell_cost
                cost_basis = qty * position["entry_price"] + position["entry_cost"]
                net_gain = net_proceeds - cost_basis
                tax = stcg_tax(net_gain)
                cash += net_proceeds - tax

                gross_return_pct = (exit_price / position["entry_price"] - 1) * 100
                net_pnl = net_gain - tax
                net_return_pct = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0
                days_held = (date - position["entry_date"]).days

                trade_log.append({
                    "symbol": symbol,
                    "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"), "qty": qty,
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "gross_return_pct": round(gross_return_pct, 2),
                    "buy_cost_rs": round(position["entry_cost"], 2),
                    "sell_cost_rs": round(sell_cost, 2),
                    "stcg_tax_rs": round(tax, 2),
                    "net_pnl_rs": round(net_pnl, 2),
                    "net_return_pct": round(net_return_pct, 2),
                    "days_held": days_held, "action": "EXIT",
                    "exit_reason": exit_reason,
                })

            # -- mark retained holdings, then buy every missing name --
            for symbol, position in holdings.items():
                if symbol in price_lookup:
                    position["last_price"] = float(price_lookup[symbol])
                    position["last_price_date"] = date

            missing_top10 = [s for s in today_top10 if s not in holdings]
            if missing_top10:
                cash = buy_missing_top10(missing_top10, price_lookup, date, cash,
                                          holdings, trade_log, unfilled_log)

        # -- portfolio checks (rule 9, RELAXED for this filter) --
        # backtest1 treated "exactly target_size holdings" as a hard
        # invariant and raised on any shortfall. Here, because Blue
        # Dot/Green Dot is a transient signal that can rotate the
        # target set almost entirely day to day, cash can legitimately
        # be too fragmented to fill every slot. We still HARD-fail on
        # holding a stock outside today's target (that would be a real
        # logic bug), but an under-filled count is expected behavior,
        # logged rather than fatal.
        held_symbols = set(holdings.keys())
        expected_symbols = set(today_top10)

        illegal_holdings = held_symbols - expected_symbols
        if illegal_holdings:
            raise RuntimeError(f"{date:%Y-%m-%d}: Portfolio contains stocks "
                                f"outside today's target: {sorted(illegal_holdings)}")

        missing_after_rebalance = expected_symbols - held_symbols
        if missing_after_rebalance and day_number % 100 == 0:
            print(f"{date:%Y-%m-%d}: under target -- missing "
                  f"{sorted(missing_after_rebalance)}. "
                  f"Holdings={len(holdings)}, Target={target_size}, Cash=Rs.{cash:,.2f}")

        # -- daily mark-to-market --
        total_value = float(cash)
        for symbol, position in holdings.items():
            if symbol in price_lookup:
                mark_price = float(price_lookup[symbol])
                position["last_price"] = mark_price
                position["last_price_date"] = date
            else:
                mark_price = float(position.get("last_price", position["entry_price"]))
            total_value += position["qty"] * mark_price

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(total_value, 2),
            "cash_rs": round(cash, 2),
            "invested_value_rs": round(total_value - cash, 2),
            "equity_multiple": round(total_value / STARTING_CAPITAL, 8),
            "n_holdings": len(holdings),
            "eligible_pool_size": eligible_pool_size,
            "top10_target_size": target_size,
        })

        if day_number % 100 == 0:
            print(f"Processed {day_number}/{n_days} | {date:%Y-%m-%d} | "
                  f"EligiblePool(BlueDot)={eligible_pool_size} | "
                  f"Holdings={len(holdings)} | Cash=Rs.{cash:,.0f} | "
                  f"Equity=Rs.{total_value:,.0f}")

    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df = _add_equity_analytics_columns(equity_df)

    trade_df = pd.DataFrame(trade_log)

    final_marked_value = (float(equity_df["portfolio_value_rs"].iloc[-1])
                           if not equity_df.empty else STARTING_CAPITAL)

    open_df, final_liquidation_value = _liquidate_open_positions(
        price_wide, rs_wide, liquid_wide, blue_wide, trading_days, holdings, cash
    )

    unfilled_df = pd.DataFrame(unfilled_log)

    return (equity_df, trade_df, open_df, final_marked_value,
            final_liquidation_value, unfilled_df)


def _add_equity_analytics_columns(equity_df):
    """Adds drawdown and normalised-to-zero equity-curve columns.
    Unchanged from backtest1."""

    running_max = equity_df["equity_multiple"].cummax()
    equity_df["drawdown_pct"] = ((equity_df["equity_multiple"] / running_max - 1)
                                  * 100).round(3)

    first_value = float(equity_df["portfolio_value_rs"].iloc[0])
    equity_df["equity_curve_pct_norm"] = (
        (equity_df["portfolio_value_rs"] / first_value - 1) * 100
    ).round(3)

    for window_days in CHART_WINDOWS:
        window_start = max(len(equity_df) - window_days, 0)
        base_value = float(equity_df["portfolio_value_rs"].iloc[window_start])

        series = pd.Series(np.nan, index=equity_df.index, dtype=float)
        series.iloc[window_start:] = (
            (equity_df["portfolio_value_rs"].iloc[window_start:] / base_value - 1) * 100
        ).round(3)
        equity_df[f"equity_curve_pct_norm_last{window_days}"] = series

    return equity_df


def _liquidate_open_positions(price_wide, rs_wide, liquid_wide, blue_wide,
                               trading_days, holdings, cash):
    liquidation_cash = float(cash)
    open_positions = []

    if len(trading_days) == 0:
        return pd.DataFrame(open_positions), liquidation_cash

    last_date = pd.Timestamp(trading_days[-1]).normalize()
    final_price_lookup = {sym: float(price) for sym, _, price
                           in build_daily_ranking(last_date, price_wide, rs_wide,
                                                   liquid_wide, blue_wide)}

    for symbol, position in holdings.items():
        exit_price = final_price_lookup.get(
            symbol, float(position.get("last_price", position["entry_price"]))
        )
        gross_proceeds = position["qty"] * exit_price
        sell_cost = sell_side_cost(gross_proceeds)
        net_proceeds = gross_proceeds - sell_cost
        cost_basis = position["qty"] * position["entry_price"] + position["entry_cost"]
        net_gain = net_proceeds - cost_basis
        tax = stcg_tax(net_gain)
        liquidation_cash += net_proceeds - tax

        open_positions.append({
            "symbol": symbol,
            "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
            "qty": position["qty"],
            "entry_price": round(position["entry_price"], 4),
            "last_price": round(exit_price, 4),
            "gross_return_pct": round((exit_price / position["entry_price"] - 1) * 100, 2),
        })

    return pd.DataFrame(open_positions), liquidation_cash


# ============================================================
# SUMMARY
# ============================================================

def summarize(equity_df, trade_df, final_marked_value, final_liquidation_value):
    if equity_df.empty:
        return {}

    marked_return = (final_marked_value / STARTING_CAPITAL - 1) * 100
    liquidation_return = (final_liquidation_value / STARTING_CAPITAL - 1) * 100
    max_dd = float(equity_df["drawdown_pct"].min())

    daily_returns = equity_df["equity_multiple"].pct_change().dropna()
    if len(daily_returns) > 1:
        daily_mean, daily_std = daily_returns.mean(), daily_returns.std()
        n_days = len(equity_df)
        annualized_return = equity_df["equity_multiple"].iloc[-1] ** (252 / max(n_days, 1)) - 1
        annualized_vol = daily_std * np.sqrt(252)
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 0
        sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        annualized_return = annualized_vol = sharpe = sortino = 0

    calmar = (annualized_return / abs(max_dd / 100)) if max_dd != 0 else 0

    if not trade_df.empty:
        entries = trade_df[trade_df["action"] == "ENTRY"]
        exits = trade_df[trade_df["action"] == "EXIT"]

        total_costs = exits["sell_cost_rs"].fillna(0).sum() if not exits.empty else 0
        total_costs += entries["buy_cost_rs"].fillna(0).sum() if not entries.empty else 0
        total_tax = exits["stcg_tax_rs"].fillna(0).sum() if not exits.empty else 0

        if not exits.empty:
            net_returns = exits["net_return_pct"].astype(float)
            win_rate = (net_returns > 0).mean() * 100
            avg_trade = net_returns.mean()
            median_trade = net_returns.median()

            winners = exits[net_returns > 0]
            losers = exits[net_returns < 0]
            avg_winner = winners["net_return_pct"].astype(float).mean() if not winners.empty else 0
            avg_loser = losers["net_return_pct"].astype(float).mean() if not losers.empty else 0

            gross_profit = winners["net_pnl_rs"].astype(float).sum() if not winners.empty else 0
            gross_loss = abs(losers["net_pnl_rs"].astype(float).sum()) if not losers.empty else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            avg_days = exits["days_held"].astype(float).mean()
        else:
            win_rate = avg_trade = median_trade = 0
            avg_winner = avg_loser = profit_factor = avg_days = 0
    else:
        entries, exits = pd.DataFrame(), pd.DataFrame()
        total_costs = total_tax = 0
        win_rate = avg_trade = median_trade = 0
        avg_winner = avg_loser = profit_factor = avg_days = 0

    return {
        "Backtest Start": BACKTEST_START,
        "Backtest End": equity_df["date"].iloc[-1],
        "Starting Capital (Rs)": STARTING_CAPITAL,
        "Final Marked Value (Rs)": round(final_marked_value, 0),
        "Final Liquidation Value (Rs)": round(final_liquidation_value, 0),
        "Net Return - Marked (%)": round(marked_return, 2),
        "Net Return - Liquidation (%)": round(liquidation_return, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Maximum Drawdown (%)": round(max_dd, 2),
        "Closed Trades": len(exits),
        "Entries": len(entries),
        "Win Rate Net (%)": round(win_rate, 2),
        "Average Net Trade (%)": round(avg_trade, 2),
        "Median Net Trade (%)": round(median_trade, 2),
        "Average Winner (%)": round(avg_winner, 2),
        "Average Loser (%)": round(avg_loser, 2),
        "Profit Factor": round(profit_factor, 3),
        "Average Days Held": round(avg_days, 1),
        "Total Transaction Costs (Rs)": round(total_costs, 0),
        "Total STCG Tax (Rs)": round(total_tax, 0),
        "RS Formula": "40% 3M + 20% 6M + 20% 9M + 20% 12M",
        "Entry Filter": f"Blue Dot only (RS-ratio {LOOKBACK_DAYS}D new high). "
                         "Green Dot computed/logged but not part of entry criteria.",
        "Portfolio": "Exact daily Top 10 of Blue-Dot-filtered RS ranking",
        "Weight": "Available replacement cash divided across missing Top-10 "
                  "names; retained positions untouched",
        "Entry": "Initial filtered Top 10; subsequently every missing name",
        "Exit": "Drops out of filtered Top 10 (rank/dot/liquidity loss) or "
                "missing from ranking",
        "Execution": "Same-day close (T+0)",
        "Rebalance Frequency": "Daily membership check; no resizing of "
                                "retained positions",
        "Price Filter": f"> Rs.{MIN_PRICE}",
        "Liquidity Filter": f"{VOLUME_LOOKBACK}D average volume > {MIN_AVG_VOLUME:,}",
        "Other Filters": f"Blue Dot only (lookback {LOOKBACK_DAYS}D). Green Dot "
                          "computed/logged but not gating. No price/RS trend "
                          "template, no sector/regime/breadth filter.",
    }


# ============================================================
# GOOGLE SHEETS -- UNCHANGED FROM BACKTEST1
# ============================================================

def sanitize_for_sheets(df):
    if df.empty:
        return df
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.where(pd.notnull(clean), "")


def sanitize_scalar(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    return v


def get_or_create_worksheet(sh, title, rows=1000, cols=16):
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


def write_in_chunks(ws, all_rows, start_row, chunk_size, label,
                     max_retries=6, initial_retry_seconds=5):
    total = len(all_rows)
    if total == 0:
        return

    for i in range(0, total, chunk_size):
        chunk = all_rows[i:i + chunk_size]
        row_start = start_row + i

        for attempt in range(max_retries):
            try:
                ws.update(chunk, f"A{row_start}")
                break
            except Exception as e:
                is_quota = "429" in str(e) or "Quota exceeded" in str(e)
                if not is_quota or attempt == max_retries - 1:
                    print(f"Write failed for {label} rows {i}-{i + len(chunk)}: {e}")
                    raise
                wait = initial_retry_seconds * (2 ** attempt)
                print(f"Google quota for {label}. Waiting {wait}s...")
                time.sleep(wait)

        print(f"Wrote {label}: {min(i + chunk_size, total)}/{total} rows")


def remove_existing_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()
        requests = [
            {"deleteEmbeddedObject": {"objectId": chart["chartId"]}}
            for sheet in meta.get("sheets", [])
            if sheet["properties"]["sheetId"] == sheet_id
            for chart in sheet.get("charts", [])
        ]
        if requests:
            sh.batch_update({"requests": requests})
            print(f"Removed {len(requests)} existing chart(s).")
    except Exception as e:
        print(f"Could not check/remove existing charts (non-fatal): {e}")


def add_charts(sh, sheet_id, equity_header_row_0idx, n_equity_rows, equity_columns):
    col_idx = {name: i for i, name in enumerate(equity_columns)}
    data_end_row = equity_header_row_0idx + 1 + n_equity_rows

    def window_start_row(window_days):
        return equity_header_row_0idx + 1 + max(n_equity_rows - window_days, 0)

    def make_chart(title, y_col_name, y_axis_title, anchor_row, chart_type="LINE",
                    start_row_override=None, show_points=False, width_pixels=650):
        y_col = col_idx[y_col_name]
        series_start = (start_row_override if start_row_override is not None
                         else equity_header_row_0idx)

        series_entry = {
            "series": {"sourceRange": {"sources": [{
                "sheetId": sheet_id, "startRowIndex": series_start,
                "endRowIndex": data_end_row, "startColumnIndex": y_col,
                "endColumnIndex": y_col + 1,
            }]}},
            "targetAxis": "LEFT_AXIS",
        }

        if show_points:
            series_entry["pointStyle"] = {"size": 5, "shape": "CIRCLE"}
            series_entry["dataLabel"] = {
                "type": "DATA", "placement": "BELOW",
                "textFormat": {"fontSize": 7},
            }

        return {"addChart": {"chart": {
            "spec": {
                "title": title,
                "basicChart": {
                    "chartType": chart_type,
                    "legendPosition": "NO_LEGEND",
                    "axis": [
                        {"position": "BOTTOM_AXIS", "title": "Date"},
                        {"position": "LEFT_AXIS", "title": y_axis_title},
                    ],
                    "domains": [{"domain": {"sourceRange": {"sources": [{
                        "sheetId": sheet_id, "startRowIndex": series_start,
                        "endRowIndex": data_end_row, "startColumnIndex": 0,
                        "endColumnIndex": 1,
                    }]}}}],
                    "series": [series_entry],
                },
            },
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": sheet_id, "rowIndex": anchor_row, "columnIndex": 8},
                "widthPixels": width_pixels, "heightPixels": 380,
            }},
        }}}

    requests = [
        make_chart("Equity Curve (Rs)", "portfolio_value_rs",
                    "Portfolio Value (Rs)", equity_header_row_0idx),

        make_chart("Drawdown (%)", "drawdown_pct", "Drawdown %",
                    equity_header_row_0idx + 22),

        make_chart("Eligible Pool Size (Blue Dot)", "eligible_pool_size",
                    "Stock Count", equity_header_row_0idx + 44),

        make_chart("Equity Curve - Normalised to Zero (%, Since Inception)",
                    "equity_curve_pct_norm", "Cumulative Change %",
                    equity_header_row_0idx + 66),

        make_chart("Equity Curve - Normalised to Zero (%, Last 50 Days)",
                    "equity_curve_pct_norm_last50", "Cumulative Change %",
                    equity_header_row_0idx + 88,
                    start_row_override=window_start_row(50),
                    show_points=True, width_pixels=1100),

        make_chart("Equity Curve - Normalised to Zero (%, Last 100 Days)",
                    "equity_curve_pct_norm_last100", "Cumulative Change %",
                    equity_header_row_0idx + 110,
                    start_row_override=window_start_row(100)),

        make_chart("Equity Curve - Normalised to Zero (%, Last 365 Days)",
                    "equity_curve_pct_norm_last365", "Cumulative Change %",
                    equity_header_row_0idx + 132,
                    start_row_override=window_start_row(365)),
    ]

    try:
        sh.batch_update({"requests": requests})
        print("Equity, drawdown, pool-size, and normalised-to-zero equity "
              "curve charts (since-inception + last-50/100/365-day) added.")
    except Exception as e:
        print(f"Could not add charts (non-fatal): {e}")


def write_to_sheet(trade_df, equity_df, open_df, summary, unfilled_df, effective_end_str):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- saving to CSV instead.")
        trade_df.to_csv("RS_BlueGreenDot_Trade_Log.csv", index=False)
        equity_df.to_csv("RS_BlueGreenDot_Equity_Curve.csv", index=False)
        if not open_df.empty:
            open_df.to_csv("RS_BlueGreenDot_Open_Positions.csv", index=False)
        if not unfilled_df.empty:
            unfilled_df.to_csv("RS_BlueGreenDot_Unfilled_Slots.csv", index=False)
        return

    creds = Credentials.from_service_account_info(
        json.loads(creds_json), scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    n_cols_needed = max(
        len(trade_df.columns) if not trade_df.empty else 0,
        len(equity_df.columns) if not equity_df.empty else 0,
        len(open_df.columns) if not open_df.empty else 0,
        len(unfilled_df.columns) if not unfilled_df.empty else 0,
        2,
    )
    n_rows_needed = (len(trade_df) + len(equity_df) + len(open_df)
                      + len(unfilled_df) + len(summary) + 60)

    ws = get_or_create_worksheet(sh, BACKTEST_WORKSHEET,
                                  rows=n_rows_needed, cols=n_cols_needed)
    if ws.row_count < n_rows_needed or ws.col_count < n_cols_needed:
        ws.resize(rows=max(ws.row_count, n_rows_needed),
                   cols=max(ws.col_count, n_cols_needed))

    remove_existing_charts(sh, ws.id)
    ws.clear()

    ws.update([[
        "TOP 10 RS ROTATION BACKTEST -- BLUE DOT FILTERED (Green Dot tracked, "
        "not used for entry) | "
        f"run {timestamp} | NET of costs+STCG | "
        f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
        "Entry filter: Blue Dot only | "
        "Target: exact Top 10 of filtered RS ranking | "
        "Retained holdings NOT resized | Sell on drop from filtered Top 10 | "
        "Available exit cash funds all missing names | Same EOD bar | "
        f"Window: {BACKTEST_START} to {effective_end_str}"
    ]], "A1")

    summary_rows = [["Summary", ""]] + [[k, sanitize_scalar(v)] for k, v in summary.items()]
    ws.update(summary_rows, "A3")

    trade_start_row = 3 + len(summary_rows) + 2
    ws.update([["Trade Log"]], f"A{trade_start_row}")
    trade_header_row = trade_start_row + 1

    if not trade_df.empty:
        trade_df_clean = sanitize_for_sheets(trade_df)
        write_in_chunks(ws, [list(trade_df_clean.columns)] + trade_df_clean.values.tolist(),
                         start_row=trade_header_row, chunk_size=2000, label="trade log")

    open_start_row = trade_header_row + len(trade_df) + 3
    ws.update([["Open Positions at Backtest End (mark-to-market)"]], f"A{open_start_row}")
    open_header_row = open_start_row + 1

    if not open_df.empty:
        open_df_clean = sanitize_for_sheets(open_df)
        ws.update([list(open_df_clean.columns)] + open_df_clean.values.tolist(),
                  f"A{open_header_row}")

    equity_start_row = open_header_row + max(len(open_df), 1) + 3
    ws.update([["Daily Equity Curve"]], f"A{equity_start_row}")
    equity_header_row = equity_start_row + 1

    if not equity_df.empty:
        equity_df_clean = sanitize_for_sheets(equity_df)
        write_in_chunks(ws, [list(equity_df_clean.columns)] + equity_df_clean.values.tolist(),
                         start_row=equity_header_row, chunk_size=2000, label="equity curve")

        add_charts(sh, ws.id, equity_header_row - 1, len(equity_df),
                   equity_columns=list(equity_df_clean.columns))

    unfilled_start_row = equity_header_row + len(equity_df) + 3
    ws.update([["Unfilled Slots (buy attempts skipped -- insufficient cash)"]],
              f"A{unfilled_start_row}")
    unfilled_header_row = unfilled_start_row + 1

    if not unfilled_df.empty:
        unfilled_df_clean = sanitize_for_sheets(unfilled_df)
        ws.update([list(unfilled_df_clean.columns)] + unfilled_df_clean.values.tolist(),
                  f"A{unfilled_header_row}")

    print(f"\nBacktest results written to '{BACKTEST_WORKSHEET}' tab: "
          f"{len(trade_df)} trades, {len(equity_df)} trading days, "
          f"{len(open_df)} open positions, {len(unfilled_df)} unfilled buy attempts.")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("TOP 10 RS ROTATION BACKTEST -- BLUE DOT FILTERED (Green Dot tracked, not used for entry)")
    print("=" * 70)
    print(f"Backtest start : {BACKTEST_START}")
    print(f"Backtest end   : {BACKTEST_END if BACKTEST_END else 'LATEST'}")
    print("Ranking        : DAILY RS SCORE, FILTERED TO BLUE DOT")
    print("Portfolio      : EXACT TOP 10 OF FILTERED RANKING")
    print("Initial        : BUY ALL FILTERED TOP 10")
    print("Rotation       : SELL ONLY STOCKS LEAVING FILTERED TOP 10")
    print("New entries    : BUY EVERY MISSING FILTERED-TOP-10 STOCK")
    print("Existing names : HOLD / NO RESIZING")
    print("Sizing         : AVAILABLE CASH / MISSING SLOTS")
    print("Execution      : SAME EOD BAR (T+0)")
    print(f"Price filter   : > Rs.{MIN_PRICE}")
    print(f"Liquidity      : {VOLUME_LOOKBACK}D average volume > {MIN_AVG_VOLUME:,}")
    print(f"Entry filter   : Blue Dot only ({LOOKBACK_DAYS}D lookback). "
          f"Green Dot computed/logged but not gating.")
    print("Other filters  : NONE")
    print("=" * 70)

    tickers = load_tickers()
    print(f"\nLoaded {len(tickers)} tickers.")

    download_start, download_end = get_download_dates()
    print(f"Download start: {download_start}")
    print(f"Download end: {download_end if download_end else 'LATEST'}")

    run_start_time = time.time()

    bench_close = download_benchmark()
    bench_close.index = normalize_dates(bench_close.index)
    print(f"[elapsed {time.time() - run_start_time:.0f}s] Benchmark ready.")

    all_stocks = {}
    total_bad_points = 0
    failed_batches = []
    # Smaller batches than backtest1/2's 50 reduce how many concurrent
    # connections hit Yahoo at once -- large concurrent batches are
    # the most common trigger for Yahoo's per-IP rate limiting on
    # shared CI runners, which otherwise shows up as yfinance quietly
    # retrying for a very long time with NO log output at all.
    batch_size = 25
    # Hard cap per batch attempt. yfinance/requests can otherwise sit
    # retrying a rate-limited or unresponsive batch far longer than
    # this with nothing printed -- better to log the failure, skip
    # those tickers, and move on than to look "stuck" with no signal.
    DOWNLOAD_TIMEOUT_SECONDS = 30
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(tickers), batch_size), start=1):
        batch = tickers[start:start + batch_size]
        batch_start_time = time.time()
        print(f"\n[elapsed {time.time() - run_start_time:.0f}s] "
              f"Downloading batch {batch_num}/{n_batches} "
              f"({start + 1}-{start + len(batch)} of {len(tickers)})")

        data = None
        for attempt in range(1, 3):
            try:
                data = yf.download(batch, start=download_start, end=download_end,
                                    interval="1d", auto_adjust=True, progress=False,
                                    group_by="ticker", threads=True,
                                    timeout=DOWNLOAD_TIMEOUT_SECONDS)
                break
            except Exception as e:
                print(f"  Attempt {attempt}/2 failed after "
                      f"{time.time() - batch_start_time:.0f}s: {e}")
                if attempt == 2:
                    failed_batches.append((batch_num, batch))
                    print(f"  Giving up on batch {batch_num} after 2 attempts -- "
                          f"skipping its {len(batch)} tickers.")
                else:
                    time.sleep(3)

        print(f"  Batch {batch_num} download call finished in "
              f"{time.time() - batch_start_time:.0f}s.")

        if data is None:
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    sdata = data
                else:
                    if not isinstance(data.columns, pd.MultiIndex):
                        continue
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    sdata = data[symbol]

                if "Close" not in sdata.columns:
                    continue

                close = sdata["Close"].dropna().sort_index()
                if close.empty:
                    continue

                volume = sdata["Volume"].reindex(close.index).fillna(0)
                close, n_bad = clean_price_series(close)
                total_bad_points += n_bad

                stock_data = compute_stock_data(close, volume, bench_close)
                if stock_data is None:
                    continue

                all_stocks[symbol.replace(".NS", "")] = stock_data

            except Exception as e:
                print(f"Skipping {symbol}: {e}")

        time.sleep(1)

    print(f"\n[elapsed {time.time() - run_start_time:.0f}s] "
          f"Stocks with usable data: {len(all_stocks)}")
    print(f"Repaired data points: {total_bad_points}")
    if failed_batches:
        skipped_tickers = [t for _, batch in failed_batches for t in batch]
        print(f"WARNING: {len(failed_batches)} batch(es) failed after 2 attempts "
              f"each (likely Yahoo rate-limiting) -- {len(skipped_tickers)} "
              f"tickers skipped entirely this run: {skipped_tickers}")

    if not all_stocks:
        raise RuntimeError("No usable stock data.")

    latest_stock_date = max(df.index.max() for df in all_stocks.values())
    latest_benchmark_date = bench_close.index.max()

    if BACKTEST_END is None:
        effective_end = min(pd.Timestamp(latest_stock_date).normalize(),
                             pd.Timestamp(latest_benchmark_date).normalize())
    else:
        effective_end = min(pd.Timestamp(BACKTEST_END).normalize(),
                             pd.Timestamp(latest_stock_date).normalize(),
                             pd.Timestamp(latest_benchmark_date).normalize())

    print(f"\nLatest stock date: {pd.Timestamp(latest_stock_date):%Y-%m-%d}")
    print(f"Latest benchmark date: {pd.Timestamp(latest_benchmark_date):%Y-%m-%d}")
    print(f"Effective end: {effective_end:%Y-%m-%d}")

    trading_days = bench_close.index[
        (bench_close.index >= pd.Timestamp(BACKTEST_START).normalize())
        & (bench_close.index <= effective_end)
    ]
    trading_days = pd.DatetimeIndex(trading_days).drop_duplicates().sort_values()

    print(f"\nTrading days: {len(trading_days)}")
    if len(trading_days) == 0:
        raise RuntimeError("No trading days found.")
    print(f"First day: {trading_days[0]:%Y-%m-%d}")
    print(f"Last day: {trading_days[-1]:%Y-%m-%d}")

    print(f"\n[elapsed {time.time() - run_start_time:.0f}s] Running backtest...")
    backtest_start_time = time.time()
    equity_df, trade_df, open_df, final_marked, final_liq, unfilled_df = run_backtest(
        all_stocks, trading_days
    )
    print(f"[elapsed {time.time() - run_start_time:.0f}s] Backtest loop finished "
          f"in {time.time() - backtest_start_time:.0f}s.")

    if not equity_df.empty:
        # Under-fill is EXPECTED with a transient filter like Blue Dot
        # Dot -- reported for visibility, not treated as a failure.
        broken = equity_df[equity_df["n_holdings"] != equity_df["top10_target_size"]]
        if not broken.empty:
            print(f"\nNOTE: {len(broken)}/{len(equity_df)} trading days held fewer "
                  f"positions than that day's target ({TOP_N} or eligible pool size, "
                  f"whichever is smaller) -- cash was insufficient to fill every "
                  f"slot. This is expected with a transient entry filter; see the "
                  f"'Unfilled Slots' log for details.")

        low_pool_days = int((equity_df["eligible_pool_size"] < TOP_N).sum())
        if low_pool_days:
            print(f"NOTE: {low_pool_days}/{len(equity_df)} trading days had fewer "
                  f"than {TOP_N} names passing Blue Dot -- portfolio "
                  f"target itself was <10 names on those days by design.")

        if not unfilled_df.empty:
            print(f"NOTE: {len(unfilled_df)} individual buy attempts were skipped "
                  f"for insufficient cash across the whole backtest.")

    summary = summarize(equity_df, trade_df, final_marked, final_liq)

    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 70)

    write_to_sheet(trade_df, equity_df, open_df, summary, unfilled_df,
                   effective_end.strftime("%Y-%m-%d"))
    print(f"[elapsed {time.time() - run_start_time:.0f}s] Sheet write finished.")

    equity_df.to_csv("RS_BlueGreenDot_Equity_Curve.csv", index=False)
    trade_df.to_csv("RS_BlueGreenDot_Trade_Log.csv", index=False)
    if not open_df.empty:
        open_df.to_csv("RS_BlueGreenDot_Open_Positions.csv", index=False)
    if not unfilled_df.empty:
        unfilled_df.to_csv("RS_BlueGreenDot_Unfilled_Slots.csv", index=False)

    print("\nCSV files also saved.")
    print(f"\n[TOTAL elapsed {time.time() - run_start_time:.0f}s] "
          f"BACKTEST COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        print("=" * 70)
        print("BACKTEST FAILED")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        raise
