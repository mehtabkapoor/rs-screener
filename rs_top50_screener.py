"""
RS TOP 50 SCREENER + COMPOSITE TOP 50 SCREENER
-- INDIVIDUAL SCORE LINE + PRICE CHARTS

WHAT THIS DOES (single snapshot, not a backtest)
  1. Downloads price/volume history for the stock universe + benchmark
     ONCE (shared by both sheets).
  2. SHEET 1 -- "Screener - RS Top50" (UNCHANGED): weighted
     multi-timeframe RS Score (40% 3M + 20% 6M + 20% 9M + 20% 12M
     price momentum), ranked descending (Rank 1 = strongest).
  3. SHEET 2 -- "Screener - Composite Top50" (NEW): a MarketSmith-style
     COMPOSITE SCORE (0-100) that replaces the RS Score as the ranking
     key. Everything else (filters, layout, charts, equity curves,
     green/blue Top-10 split, rank-percentile line, audit columns) is
     identical to Sheet 1.
  4. For EACH of the Top N stocks (both sheets) individually: its own
     data table (last RS_LINE_WINDOW trading days) and its own chart.
     PRICE is rebased to 0% at day 1; GREEN on any day the stock was
     in the Top 10 by that sheet's score, BLUE otherwise; boundary
     days duplicated so the line stays continuous. The actual daily
     rank is written as a plain (uncharted) audit column.
  5. THREE Top 10 equal-weight, daily-rebalanced equity curves per
     sheet (no costs/slippage): last 50 days (raw, no SMAs), last 252
     days (with 20/50/200 SMAs), and FULL history (RAW index level,
     base=100, log-scale-safe, with SMAs). Google Sheets' Log scale is
     a UI-only toggle (not settable via API, and charts are rebuilt on
     every run), so toggle it manually if wanted. Each is overlaid with
     the Top 10 basket's average score (teal, right axis).
  6. All charts are stacked at the top of each sheet; each chart's data
     table stays below. Chart creation is batched with quota-aware
     retries and per-chart fallback; failures are named in the log.

COMPOSITE SCORE (price/volume proxy of MarketSmith's Composite Rating)
  MarketSmith blends EPS Rating, RS Rating, SMR Rating, Acc/Dis Rating
  and Industry Group RS. Weights are proprietary. EPS/SMR are NOT
  point-in-time in yfinance (backtesting them = lookahead bias), so the
  honest, backtestable proxy uses only price/volume:

    With an `industry` (or `sector`) column in stocks.csv:
        Composite = 50% RS percentile + 30% Acc/Dis percentile
                  + 20% Industry Group RS percentile
    Without it (or too little coverage):
        Composite = 60% RS percentile + 40% Acc/Dis percentile

  - RS percentile  : the existing 3-12M blended RS Score, percentile-
                     ranked across that day's eligible universe.
  - Acc/Dis        : (up-day volume - down-day volume) / total volume
                     over ACCDIS_WINDOW (65) days, percentile-ranked.
  - Group RS       : mean RS percentile of the stock's industry (groups
                     with < MIN_GROUP_SIZE eligible names -> neutral 50),
                     percentile-ranked across groups.
  Every input is computed per day from data available that day, so the
  daily rank history that drives equity curves / colour split /
  percentile line has no lookahead.

RS ACCELERATION OVERLAY (informational only -- not used for ranking):
  rs_score_20d / rs_accel_20d audit columns are kept on both sheets.
  Short-window momentum sits in the short-term reversal window
  (Jegadeesh 1990), so it is never used to rank/select.

This is a screener, not a trading system.
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

DOWNLOAD_YEARS = 3  # RS_12M lookback (252d) + 200-day SMA warmup +
                    # 50-day display window needs ~500+ trading days

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

RS_3M, RS_6M, RS_9M, RS_12M = 63, 126, 189, 252
RS_WEIGHTS = (0.40, 0.20, 0.20, 0.20)

# Short-window RS score + acceleration -- audit only, never ranks.
RS_ACCEL_WINDOW = 20

TOP_N = 100
RS_LINE_WINDOW = 50

# Rank threshold used to color the price line (green inside, blue outside)
TOP10_N = 10

# ---------------- Composite score config ----------------
ENABLE_COMPOSITE = True
ACCDIS_WINDOW = 65                       # ~13 weeks, as in MarketSmith
COMPOSITE_WEIGHTS_WITH_GROUP = (0.50, 0.30, 0.20)   # RS, Acc/Dis, Group
COMPOSITE_WEIGHTS_NO_GROUP = (0.60, 0.40, 0.00)
MIN_GROUP_SIZE = 3                       # eligible names needed for a group score
MIN_INDUSTRY_COVERAGE = 0.50             # share of universe needing an industry tag
INDUSTRY_COLUMN_CANDIDATES = ("industry", "sector")
NEUTRAL_GROUP_PERCENTILE = 50.0

# Chart colors (Google Sheets Color proto: 0-1 floats)
GREEN_COLOR = {"red": 0.20, "green": 0.65, "blue": 0.33}   # in Top 10
BLUE_COLOR = {"red": 0.26, "green": 0.52, "blue": 0.96}    # outside Top 10
SERIES_COLORS = [GREEN_COLOR, BLUE_COLOR]

# Rank trend line (percentile of daily rank vs that day's eligible
# universe): Rank 1 = BEST, weakest eligible = WORST. Rising = stronger.
RANK_LINE_COLOR = {"red": 0.55, "green": 0.15, "blue": 0.75}  # purple
RS_PERCENTILE_BEST = 99
RS_PERCENTILE_WORST = 1

# Top 10 equal-weight equity curve (regime/timing overlay)
EQUITY_SMA_PERIODS = (20, 50, 200)
EQUITY_1Y_WINDOW = 252

EQUITY_COLOR = {"red": 0.05, "green": 0.05, "blue": 0.05}        # bold black
EQUITY_SMA20_COLOR = {"red": 0.20, "green": 0.60, "blue": 0.86}  # blue
EQUITY_SMA50_COLOR = {"red": 0.95, "green": 0.60, "blue": 0.10}  # orange
EQUITY_SMA200_COLOR = {"red": 0.80, "green": 0.10, "blue": 0.10}  # red
EQUITY_SERIES_COLORS = [
    EQUITY_COLOR, EQUITY_SMA20_COLOR, EQUITY_SMA50_COLOR, EQUITY_SMA200_COLOR
]

EQUITY_LINE_WIDTH = 5
SMA_LINE_WIDTH = 2
EQUITY_SERIES_WIDTHS = [EQUITY_LINE_WIDTH, SMA_LINE_WIDTH, SMA_LINE_WIDTH, SMA_LINE_WIDTH]
EQUITY_SERIES_STYLES = ["SOLID", "MEDIUM_DASHED", "MEDIUM_DASHED", "MEDIUM_DASHED"]

# Avg score of the Top 10 basket, overlaid on a secondary (right) axis,
# RAW (not rebased). Thin + dashed so it never competes with equity.
AVG_RS_COLOR = {"red": 0.0, "green": 0.55, "blue": 0.55}  # teal
AVG_RS_LINE_WIDTH = 2
AVG_RS_LINE_STYLE = "MEDIUM_DASHED"

# Charts stacked at top of sheet; vertical gap between anchors (rows).
CHART_ROW_SPACING = 20
CHART_ZONE_BUFFER_ROWS = 2

CHART_BATCH_SIZE = 5
CHART_BATCH_PAUSE_SECONDS = 2

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
SCREENER_WORKSHEET = "Screener - RS Top50"
COMPOSITE_WORKSHEET = "Screener - Composite Top50"

# Pause between finishing sheet 1 and starting sheet 2 (Sheets API quota)
BETWEEN_SHEETS_PAUSE_SECONDS = 10

# If True, freeze ranking to yesterday's settled close when the latest
# bar is today's live intraday tick (run-to-run reproducibility).
STRICT_SETTLED_CLOSE = False


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


def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")

    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")

    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]

    output = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    return list(dict.fromkeys(output))


def load_industry_map():
    """
    {symbol_without_.NS: industry} from an optional `industry` (or
    `sector`) column in stocks.csv. Returns {} if absent -- the
    composite then falls back to RS + Acc/Dis only.
    """
    if not os.path.exists(STOCKS_FILE):
        return {}

    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        return {}

    cols = {str(c).strip().lower(): c for c in df.columns}
    ind_col = next(
        (cols[k] for k in INDUSTRY_COLUMN_CANDIDATES if k in cols),
        None
    )
    if ind_col is None:
        return {}

    out = {}
    for sym, ind in zip(df["symbol"], df[ind_col]):
        if pd.isna(sym) or pd.isna(ind):
            continue
        sym = str(sym).strip().replace(".NS", "")
        ind = str(ind).strip()
        if sym and ind:
            out[sym] = ind
    return out


def clean_price_series(close, max_move=0.30):
    close = normalize_series_index(close)
    bad = close.pct_change().abs() > max_move
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
# BENCHMARK
# ============================================================

def download_benchmark():
    download_start = (
        pd.Timestamp.today() -
        pd.DateOffset(years=DOWNLOAD_YEARS)
    ).strftime("%Y-%m-%d")

    print(f"\nBenchmark download: {download_start} -> LATEST")

    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(
                ticker,
                start=download_start,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

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
# ============================================================

def compute_stock_data(close, volume):
    close = normalize_series_index(close)
    volume = normalize_series_index(volume)

    if len(close) < RS_12M + 20:
        return None

    avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (
        (close > MIN_PRICE) &
        (avg_volume > MIN_AVG_VOLUME)
    )

    w3, w6, w9, w12 = RS_WEIGHTS

    rs_score = (
        w3 * (close / close.shift(RS_3M) - 1)
        + w6 * (close / close.shift(RS_6M) - 1)
        + w9 * (close / close.shift(RS_9M) - 1)
        + w12 * (close / close.shift(RS_12M) - 1)
    ) * 100

    # Audit-only short-window RS + acceleration (never ranks)
    rs_score_20d = (
        close / close.shift(RS_ACCEL_WINDOW) - 1
    ) * 100
    rs_accel_20d = rs_score_20d - rs_score_20d.shift(RS_ACCEL_WINDOW)

    # Accumulation/Distribution proxy: volume on up days minus volume on
    # down days, over ACCDIS_WINDOW days, as a share of total volume.
    # Range -1 (all distribution) to +1 (all accumulation). Used ONLY by
    # the composite sheet.
    signed_vol = np.sign(close.diff()) * volume
    total_vol = volume.rolling(ACCDIS_WINDOW).sum()
    accdis = (
        signed_vol.rolling(ACCDIS_WINDOW).sum()
        / total_vol.where(total_vol > 0)
    )

    result = pd.DataFrame({
        "price": close,
        "avg_volume": avg_volume,
        "liquid": liquid,
        "rs_score": rs_score,
        "rs_score_20d": rs_score_20d,
        "rs_accel_20d": rs_accel_20d,
        "accdis": accdis,
    })

    result.index = normalize_dates(result.index)

    return result


def build_composite_stocks(all_stocks, industry_map):
    """
    Returns (composite_stocks, info). composite_stocks has the same
    structure as all_stocks, but each stock's `rs_score` column is
    REPLACED by the 0-100 composite score (NaN on any day the stock is
    not eligible), so every downstream function (ranking, daily rank
    maps, equity curves, charts) works unchanged. The original RS score
    and the three component percentiles ride along as audit columns.

    Eligibility each day = liquid AND RS score available AND Acc/Dis
    available AND price > 0. All percentiles are cross-sectional within
    that day's eligible set -- no future data is used.
    """
    syms = list(all_stocks.keys())

    def panel(col):
        return pd.DataFrame(
            {s: all_stocks[s][col] for s in syms}
        ).sort_index()

    rs = panel("rs_score")
    acc = panel("accdis")
    price = panel("price")
    liquid = panel("liquid").astype(float).fillna(0.0).astype(bool)

    elig = liquid & rs.notna() & acc.notna() & (price > 0)

    rs_pct = rs.where(elig).rank(axis=1, pct=True) * 100
    acc_pct = acc.where(elig).rank(axis=1, pct=True) * 100

    # ---- Industry group RS ----
    ind = pd.Series({s: industry_map.get(s) for s in syms}, dtype=object)
    ind_valid = ind.dropna()
    n_groups = ind_valid.nunique()
    coverage = len(ind_valid) / max(len(syms), 1)
    has_group = (
        coverage >= MIN_INDUSTRY_COVERAGE and n_groups >= 3
    )

    grp_pct = pd.DataFrame(
        NEUTRAL_GROUP_PERCENTILE,
        index=rs.index,
        columns=syms,
        dtype=float
    )

    if has_group:
        v_syms = list(ind_valid.index)
        grp_mean = rs_pct[v_syms].T.groupby(ind_valid).mean().T
        grp_cnt = elig[v_syms].T.groupby(ind_valid).sum().T
        grp_mean = grp_mean.where(grp_cnt >= MIN_GROUP_SIZE)
        grp_rank = grp_mean.rank(axis=1, pct=True) * 100

        mapped = grp_rank[ind_valid.values]
        mapped.columns = v_syms
        grp_pct[v_syms] = mapped.fillna(NEUTRAL_GROUP_PERCENTILE)

        w_rs, w_acc, w_grp = COMPOSITE_WEIGHTS_WITH_GROUP
    else:
        w_rs, w_acc, w_grp = COMPOSITE_WEIGHTS_NO_GROUP

    composite = (
        w_rs * rs_pct + w_acc * acc_pct + w_grp * grp_pct
    ).where(elig)

    comp_stocks = {}
    for s in syms:
        d = all_stocks[s].copy()
        idx = d.index
        d["rs_score_raw"] = d["rs_score"]
        d["rs_score"] = composite[s].reindex(idx)
        d["rs_pctile"] = rs_pct[s].reindex(idx)
        d["accdis_pctile"] = acc_pct[s].reindex(idx)
        d["group_pctile"] = (
            grp_pct[s].reindex(idx) if has_group
            else pd.Series(np.nan, index=idx)
        )
        comp_stocks[s] = d

    if has_group:
        formula = (
            f"{w_rs:.0%} RS percentile + {w_acc:.0%} Acc/Dis "
            f"({ACCDIS_WINDOW}d) percentile + {w_grp:.0%} Industry "
            f"Group RS percentile"
        )
    else:
        formula = (
            f"{w_rs:.0%} RS percentile + {w_acc:.0%} Acc/Dis "
            f"({ACCDIS_WINDOW}d) percentile (no usable industry column "
            f"in stocks.csv -> Group RS factor dropped)"
        )

    info = {
        "has_group": has_group,
        "formula": formula,
        "n_groups": int(n_groups),
        "coverage": coverage,
    }
    return comp_stocks, info


def get_row(df, date):
    date = pd.Timestamp(date).normalize()

    if date not in df.index:
        return None

    row = df.loc[date]

    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def build_ranking(all_stocks, date):
    ranking = []

    for symbol, df in all_stocks.items():
        row = get_row(df, date)

        if row is None:
            continue

        rs = row["rs_score"]

        if pd.isna(rs) or not bool(row["liquid"]):
            continue

        price = row["price"]

        if pd.isna(price) or float(price) <= 0:
            continue

        avg_vol = row["avg_volume"]

        rs_20d = row.get("rs_score_20d")
        rs_accel = row.get("rs_accel_20d")

        ranking.append((
            symbol,
            float(rs),
            float(price),
            float(avg_vol) if not pd.isna(avg_vol) else 0.0,
            float(rs_20d) if rs_20d is not None and not pd.isna(rs_20d) else np.nan,
            float(rs_accel) if rs_accel is not None and not pd.isna(rs_accel) else np.nan,
        ))

    # Sort key is the score ONLY (index 1): RS blend on the RS sheet,
    # composite on the composite sheet.
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def compute_daily_rank_maps(all_stocks, trading_days_window, top_n_for_avg_rs=TOP10_N):
    """
    For every date in the window, rank the full eligible universe and
    return {date: {symbol: rank}} (rank 1 = strongest that day), plus
    {date: avg_score} of that day's Top `top_n_for_avg_rs` basket (NaN
    if fewer were eligible).
    """
    rank_maps = {}
    avg_rs_by_day = {}

    for d in trading_days_window:
        ranking = build_ranking(all_stocks, d)

        rank_maps[d] = {
            sym: i + 1
            for i, row in enumerate(ranking)
            for sym in [row[0]]
        }

        top_slice = ranking[:top_n_for_avg_rs]
        if len(top_slice) == top_n_for_avg_rs:
            avg_rs_by_day[d] = float(
                np.mean([row[1] for row in top_slice])
            )
        else:
            avg_rs_by_day[d] = np.nan

    avg_rs_series = pd.Series(avg_rs_by_day, dtype=float).sort_index()

    return rank_maps, avg_rs_series


def compute_top10_equity_curve(all_stocks, rank_maps, full_calendar, top_n=10):
    """
    Equal-weight, daily-rebalanced cumulative return index for the
    Top N stocks, using each day's rank as of the PRIOR trading day
    (no lookahead). Base = 100 on the first day a full Top N exists.
    Regime/timing overlay, not a real backtest: no costs or slippage.
    """
    daily_returns = {}

    for i in range(1, len(full_calendar)):
        prev_day = full_calendar[i - 1]
        day = full_calendar[i]

        prev_ranks = rank_maps.get(prev_day)
        if not prev_ranks:
            continue

        top_syms = [sym for sym, r in prev_ranks.items() if r <= top_n]
        if len(top_syms) < top_n:
            continue

        day_returns = []
        for sym in top_syms:
            df = all_stocks.get(sym)
            if df is None or prev_day not in df.index or day not in df.index:
                continue

            p_prev = df.at[prev_day, "price"]
            p_curr = df.at[day, "price"]

            if pd.isna(p_prev) or pd.isna(p_curr) or p_prev <= 0:
                continue

            day_returns.append(p_curr / p_prev - 1)

        if not day_returns:
            continue

        daily_returns[day] = float(np.mean(day_returns))

    if not daily_returns:
        return pd.Series(dtype=float)

    ret_series = pd.Series(daily_returns).sort_index()
    return 100.0 * (1.0 + ret_series).cumprod()


def build_top10_equity_table(
    equity_index,
    trading_days_window,
    avg_rs_series=None,
    include_sma=True,
    rebase_pct=True,
    avg_col_name="avg_rs_score_top10"
):
    """
    Slice the Top N equity curve (and, if `include_sma`, its 20/50/200
    SMA computed on the FULL curve so they're warmed up) to the display
    window. `rebase_pct=True` rebases to 0% at the window's first day
    (display-only transform, preserves equity/SMA crossovers);
    `rebase_pct=False` returns the RAW index level (strictly positive,
    log-scale-safe). The avg-score column is always RAW (never rebased).
    """
    if equity_index.empty:
        return None

    window_equity = equity_index.reindex(trading_days_window)

    if window_equity.dropna().empty:
        return None

    base = window_equity.dropna().iloc[0]
    if pd.isna(base) or base <= 0:
        return None

    def transform(s):
        return (s / base - 1) * 100 if rebase_pct else s

    equity_col_name = "top10_equity_pct" if rebase_pct else "top10_equity_index"

    sma_cols = {}
    if include_sma:
        for period in EQUITY_SMA_PERIODS:
            sma = equity_index.rolling(period).mean().reindex(trading_days_window)
            col_name = f"sma{period}_pct" if rebase_pct else f"sma{period}_index"
            sma_cols[col_name] = transform(sma).round(3).values

    table = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in trading_days_window],
        equity_col_name: transform(window_equity).round(3).values,
        **sma_cols,
    })

    if avg_rs_series is not None:
        table[avg_col_name] = (
            avg_rs_series.reindex(trading_days_window).round(2).values
        )

    return table


def split_by_rank(values, flags):
    """
    Split one series into two parallel series by a boolean flag per
    point (`top` where flag, `other` elsewhere). At every flip, the
    boundary point is duplicated into both arrays so the colour
    segments meet without a visual gap.
    """
    n = len(values)
    top = [np.nan] * n
    other = [np.nan] * n

    for i in range(n):
        if flags[i]:
            top[i] = values[i]
        else:
            other[i] = values[i]

    for i in range(1, n):
        if flags[i] != flags[i - 1]:
            if flags[i]:
                top[i - 1] = values[i - 1]
            else:
                other[i - 1] = values[i - 1]

    return top, other


# ============================================================
# PRICE
# ============================================================

def build_stock_series(
    all_stocks,
    symbol,
    trading_days_window,
    rank_maps,
    extra_audit_cols=()
):
    """
    Returns (dataframe, note). note is None on a clean run, or a short
    string saying how many no-trade/circuit days were carried forward.
    dataframe is None only if the stock has NO valid positive price in
    the window.

    PRICE % is rebased to 0% at day 1 of the window. Columns:
    `rank_percentile` (CHARTED, that day's rank rescaled against that
    day's eligible-universe size: Rank 1 -> BEST, weakest -> WORST) and
    `daily_rank` (raw positive rank, uncharted, for audit).
    Latest-value audit columns (uncharted): rs_score_20d, rs_accel_20d,
    plus any `extra_audit_cols` (composite sheet: raw RS score and the
    three component percentiles).
    """
    df = all_stocks[symbol]

    prices_raw = df["price"].reindex(trading_days_window)
    prices_raw = prices_raw.where(prices_raw > 0)

    n_filled = int(prices_raw.isna().sum())
    prices = prices_raw.ffill().bfill()

    if prices.isna().all():
        return None, "no valid (positive) price data anywhere in the window"

    price_base = prices.iloc[0]
    price_pct = (prices / price_base - 1) * 100

    daily_ranks = [
        rank_maps.get(d, {}).get(symbol)
        for d in trading_days_window
    ]

    flags = [
        (r is not None and r <= TOP10_N)
        for r in daily_ranks
    ]

    top_vals, other_vals = split_by_rank(price_pct.values, flags)

    rank_col = [
        float(r) if r is not None else np.nan
        for r in daily_ranks
    ]

    universe_sizes = [
        len(rank_maps.get(d, {}))
        for d in trading_days_window
    ]

    def rank_to_percentile(r, n):
        if r is None or n <= 0:
            return np.nan
        if n <= 1:
            return float(RS_PERCENTILE_BEST)
        return RS_PERCENTILE_WORST + (
            RS_PERCENTILE_BEST - RS_PERCENTILE_WORST
        ) * (n - r) / (n - 1)

    pct_col = [
        rank_to_percentile(r, n)
        for r, n in zip(daily_ranks, universe_sizes)
    ]

    result = pd.DataFrame({
        "date": [
            d.strftime("%Y-%m-%d")
            for d in trading_days_window
        ],
        "price_pct_top10": np.round(np.array(top_vals, dtype=float), 3),
        "price_pct_other": np.round(np.array(other_vals, dtype=float), 3),
        "rank_percentile": np.round(np.array(pct_col, dtype=float), 2),
        "daily_rank": rank_col,
    })

    latest_row = df.reindex(trading_days_window).iloc[-1]
    result["rs_score_20d"] = round(
        float(latest_row["rs_score_20d"]), 2
    ) if not pd.isna(latest_row.get("rs_score_20d", np.nan)) else np.nan
    result["rs_accel_20d"] = round(
        float(latest_row["rs_accel_20d"]), 2
    ) if not pd.isna(latest_row.get("rs_accel_20d", np.nan)) else np.nan

    for col in extra_audit_cols:
        v = latest_row.get(col, np.nan)
        result[col] = round(float(v), 2) if not pd.isna(v) else np.nan

    fill_note = f"{n_filled} no-trade/bad-tick day(s) carried forward" if n_filled else None

    return result, fill_note


# ============================================================
# GOOGLE SHEETS
# ============================================================

def sanitize_for_sheets(df):
    if df.empty:
        return df

    clean = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return clean.where(pd.notnull(clean), "")


def get_or_create_worksheet(
    sh,
    title,
    rows=1000,
    cols=16
):
    try:
        return sh.worksheet(title)

    except gspread.WorksheetNotFound:
        pass

    try:
        return sh.add_worksheet(
            title=title,
            rows=rows,
            cols=cols
        )

    except gspread.exceptions.APIError as e:
        if "already exists" in str(e):
            return sh.worksheet(title)
        raise


def remove_existing_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()

        requests = [
            {
                "deleteEmbeddedObject": {
                    "objectId": chart["chartId"]
                }
            }
            for sheet in meta.get("sheets", [])
            if sheet["properties"]["sheetId"] == sheet_id
            for chart in sheet.get("charts", [])
        ]

        if requests:
            sh.batch_update({"requests": requests})
            print(f"Removed {len(requests)} existing chart(s).")

    except Exception as e:
        print(
            "Could not check/remove existing charts "
            f"(non-fatal): {e}"
        )


def make_stock_chart(
    sheet_id,
    title,
    header_row_0idx,
    n_rows,
    anchor_row,
    data_col_start,
    n_series,
    colors,
    series_axes=None,
    series_widths=None,
    series_line_styles=None,
    left_axis_title="% Change from Day 1 (Base = 0)",
    right_axis_title=None
):
    """
    `header_row_0idx`/`n_rows` locate the chart's SOURCE DATA;
    `anchor_row` is purely the floating chart's on-screen position, so
    all charts can be stacked at the top while data tables stay put.
    `series_axes` / `series_widths` / `series_line_styles` are optional
    per-series lists (same length as `colors`); defaults are all
    LEFT_AXIS / width 2 / SOLID.
    """
    data_end_row = header_row_0idx + 1 + n_rows

    if series_axes is None:
        series_axes = ["LEFT_AXIS"] * n_series

    if series_widths is None:
        series_widths = [2] * n_series

    if series_line_styles is None:
        series_line_styles = ["SOLID"] * n_series

    uses_right_axis = "RIGHT_AXIS" in series_axes

    def series(col_index, color, target_axis, width, line_style):
        return {
            "series": {
                "sourceRange": {
                    "sources": [{
                        "sheetId": sheet_id,
                        "startRowIndex": header_row_0idx,
                        "endRowIndex": data_end_row,
                        "startColumnIndex": col_index,
                        "endColumnIndex": col_index + 1,
                    }]
                }
            },
            "targetAxis": target_axis,
            "color": color,
            "colorStyle": {"rgbColor": color},
            "lineStyle": {
                "width": width,
                "type": line_style
            },
            "pointStyle": {
                "size": 3 if line_style == "SOLID" else 0,
                "shape": "CIRCLE"
            },
        }

    axis_defs = [
        {
            "position": "BOTTOM_AXIS",
            "title": "Date"
        },
        {
            "position": "LEFT_AXIS",
            "title": left_axis_title
        },
    ]

    if uses_right_axis:
        axis_defs.append({
            "position": "RIGHT_AXIS",
            "title": right_axis_title or "Secondary axis"
        })

    return {
        "addChart": {
            "chart": {
                "spec": {
                    "title": title,
                    "basicChart": {
                        "chartType": "LINE",
                        "legendPosition": "BOTTOM_LEGEND",
                        "axis": axis_defs,
                        "domains": [{
                            "domain": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet_id,
                                        "startRowIndex":
                                            header_row_0idx,
                                        "endRowIndex":
                                            data_end_row,
                                        "startColumnIndex":
                                            data_col_start,
                                        "endColumnIndex":
                                            data_col_start + 1,
                                    }]
                                }
                            }
                        }],
                        "series": [
                            series(
                                data_col_start + 1 + i,
                                colors[i],
                                series_axes[i],
                                series_widths[i],
                                series_line_styles[i]
                            )
                            for i in range(n_series)
                        ],
                    },
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": sheet_id,
                            "rowIndex": anchor_row,
                            "columnIndex": 0,
                        },
                        "widthPixels": 850,
                        "heightPixels": 400,
                    }
                },
            }
        }
    }


def call_with_quota_retry(
    fn,
    label,
    max_retries=6,
    initial_wait_seconds=15
):
    for attempt in range(max_retries):
        try:
            return fn()

        except Exception as e:
            is_quota = (
                "429" in str(e)
                or "Quota exceeded" in str(e)
            )

            if not is_quota or attempt == max_retries - 1:
                print(f"{label} failed: {e}")
                raise

            wait = initial_wait_seconds * (2 ** attempt)

            print(
                f"Google quota hit on {label}. "
                f"Waiting {wait}s before retry..."
            )

            time.sleep(wait)


def add_charts_robust(sh, chart_requests, chart_labels, chunk=CHART_BATCH_SIZE):
    """
    Sends addChart requests in small batches with quota-aware retries.
    A batch that still fails is retried chart-by-chart; anything that
    still can't be added is returned by name for the run log.
    """
    total = len(chart_requests)
    if total == 0:
        return []

    added = 0
    failed = []

    for i in range(0, total, chunk):
        batch = chart_requests[i:i + chunk]
        labels = chart_labels[i:i + chunk]

        try:
            call_with_quota_retry(
                lambda b=batch: sh.batch_update({"requests": b}),
                label=f"chart batch {i + 1}-{i + len(batch)}"
            )
            added += len(batch)
            print(f"Added charts {i + 1}-{i + len(batch)} of {total}")

        except Exception as e:
            print(
                f"Chart batch {i + 1}-{i + len(batch)} failed after "
                f"retries ({e}); retrying charts individually..."
            )

            for req, lbl in zip(batch, labels):
                try:
                    call_with_quota_retry(
                        lambda r=req: sh.batch_update({"requests": [r]}),
                        label=f"chart retry: {lbl}",
                        max_retries=4,
                        initial_wait_seconds=10
                    )
                    added += 1
                    print(f"  Recovered chart: {lbl}")

                except Exception as e2:
                    failed.append(lbl)
                    detail = getattr(e2, "response", None)
                    if detail is not None:
                        try:
                            print(f"  FAILED chart: {lbl} -> {detail.text}")
                        except Exception:
                            print(f"  FAILED chart: {lbl} -> {e2!r}")
                    else:
                        print(f"  FAILED chart: {lbl} -> {e2}")

        time.sleep(CHART_BATCH_PAUSE_SECONDS)

    print(f"\nChart summary: {added}/{total} charts added.")
    if failed:
        print(f"Charts NOT added ({len(failed)}): {failed}")

    return failed


def col_letter(idx0):
    letters = ""
    idx = idx0

    while True:
        letters = (
            chr(ord("A") + idx % 26)
            + letters
        )

        idx = idx // 26 - 1

        if idx < 0:
            break

    return letters


def write_rows_in_chunks(
    ws,
    all_rows,
    chunk_size=1500,
    label="sheet write",
    start_col="A"
):
    total = len(all_rows)

    if total == 0:
        return

    for i in range(0, total, chunk_size):
        chunk = all_rows[i:i + chunk_size]
        row_start = i + 1

        call_with_quota_retry(
            lambda c=chunk, r=row_start:
                ws.update(
                    c,
                    f"{start_col}{r}"
                ),
            label=(
                f"{label} rows "
                f"{i}-{i + len(chunk)}"
            ),
        )

        print(
            f"Wrote {label}: "
            f"{min(i + chunk_size, total)}/{total} rows"
        )


def write_to_sheet(
    cfg,
    ranking_df,
    stock_series_list,
    skipped_symbols,
    as_of_date,
    equity_table_50d=None,
    equity_table_1y=None,
    equity_table_full=None,
    skip_reasons=None
):
    nm = cfg["name"]            # "RS" or "Composite"
    sc = cfg["score_label"]     # "RS Score" or "Composite Score"
    worksheet_name = cfg["worksheet"]

    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print(
            "Missing SHEET_ID/GOOGLE_CREDENTIALS "
            "-- saving to CSV instead."
        )

        ranking_df.to_csv(
            f"{cfg['csv_prefix']}_Ranking.csv",
            index=False
        )

        for tbl, suffix in (
            (equity_table_50d, "50d"),
            (equity_table_1y, "1y"),
            (equity_table_full, "full"),
        ):
            if tbl is not None:
                tbl.to_csv(
                    f"{cfg['eq_prefix']}_{suffix}.csv",
                    index=False
                )

        for rank, symbol, df in stock_series_list:
            df.to_csv(
                f"{cfg['csv_prefix']}_Stock_{rank:02d}_{symbol}.csv",
                index=False
            )

        return

    creds = Credentials.from_service_account_info(
        json.loads(creds_json),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets"
        ]
    )

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M IST"
    )

    DATA_COL_START = 9

    block_height = RS_LINE_WINDOW + 4
    block_height_1y = EQUITY_1Y_WINDOW + 4
    block_height_full = (
        len(equity_table_full) + 4
        if equity_table_full is not None else 0
    )

    n_equity_chart_50d = 1 if equity_table_50d is not None else 0
    n_equity_chart_1y = 1 if equity_table_1y is not None else 0
    n_equity_chart_full = 1 if equity_table_full is not None else 0
    total_charts = (
        n_equity_chart_50d
        + n_equity_chart_1y
        + n_equity_chart_full
        + len(stock_series_list)
    )
    charts_zone_rows = (
        total_charts * CHART_ROW_SPACING
        + CHART_ZONE_BUFFER_ROWS
    )

    n_rows_needed = (
        charts_zone_rows
        + 3
        + len(ranking_df)
        + 3
        + block_height
        + block_height_1y
        + block_height_full
        + len(stock_series_list) * block_height
        + 20
    )

    max_stock_cols = (
        len(stock_series_list[0][2].columns)
        if stock_series_list else 8
    )
    n_cols_needed = max(
        DATA_COL_START + max_stock_cols,
        len(ranking_df.columns)
    ) + 2

    ws = call_with_quota_retry(
        lambda: get_or_create_worksheet(
            sh,
            worksheet_name,
            rows=n_rows_needed,
            cols=n_cols_needed
        ),
        label="get_or_create_worksheet"
    )

    if (
        ws.row_count < n_rows_needed
        or ws.col_count < n_cols_needed
    ):
        call_with_quota_retry(
            lambda: ws.resize(
                rows=max(
                    ws.row_count,
                    n_rows_needed
                ),
                cols=max(
                    ws.col_count,
                    n_cols_needed
                )
            ),
            label="resize"
        )

    call_with_quota_retry(
        lambda: remove_existing_charts(
            sh,
            ws.id
        ),
        label="remove_existing_charts"
    )

    call_with_quota_retry(
        lambda: ws.clear(),
        label="clear"
    )

    rows_left = []
    rows_right = []

    def add_row(left=None, right=None):
        rows_left.append(
            left if left is not None else []
        )

        rows_right.append(
            right if right is not None else []
        )

        return len(rows_left)

    # Blank zone at the very top for the stacked charts.
    for _ in range(charts_zone_rows):
        add_row()

    next_chart_anchor_row = 0

    add_row(left=[
        f"{cfg['title']} | "
        f"run {timestamp} | "
        f"As of: {as_of_date} | "
        f"{cfg['formula_line']} | "
        f"Charts: Price % (GREEN = in Top {TOP10_N} by {sc} that "
        f"day, BLUE = outside Top {TOP10_N}, rebased to 0% at day 1 "
        f"of the {RS_LINE_WINDOW}-day window), plus a PURPLE {nm} "
        f"Rank Percentile trend line on the right axis (Rank 1 -> "
        f"{RS_PERCENTILE_BEST}th %ile, weakest eligible stock that "
        f"day -> {RS_PERCENTILE_WORST}th %ile, scaled against that "
        f"day's actual eligible-universe size) so RISING = "
        f"strengthening and FALLING = weakening | "
        f"All charts are grouped together at the top of this sheet; "
        f"each chart's own data table is still labeled below | "
        f"'daily_rank' column shows the actual positive rank each day "
        f"(not charted) to audit the percentile line and the "
        f"green/blue price split | "
        f"'rs_score_20d'/'rs_accel_20d' columns are a "
        f"{RS_ACCEL_WINDOW}-day short-term RS reading + its own "
        f"acceleration, audit/early-warning only -- NOT charted, NOT "
        f"used anywhere in ranking or Top N selection | "
        f"{cfg['extra_note']}"
        f"Top {TOP10_N} {nm} Equal-Weight Equity Curve included three "
        f"times (daily rebalance, no costs): last {RS_LINE_WINDOW} "
        f"days (rebased %, no SMA overlays), last "
        f"{EQUITY_1Y_WINDOW} days / 1 year (rebased %, with 20/50/200 "
        f"SMA overlays), and the full downloaded history / "
        f"~{DOWNLOAD_YEARS} years (RAW equity index level, NOT "
        f"rebased -- log-scale-safe; enable Log scale manually in "
        f"Customize > Vertical axis if wanted, since the API can't "
        f"set it and charts are rebuilt every run -- also with "
        f"20/50/200 SMA overlays), plus a TEAL Avg {sc} line (right "
        f"axis, raw/not rebased) on all three. The equity curve is "
        f"drawn BOLD/SOLID (width {EQUITY_LINE_WIDTH}); SMAs and the "
        f"teal line are thin/DASHED | "
        f"{len(stock_series_list)}/{TOP_N} stocks charted "
        f"({len(skipped_symbols)} truly unplottable: no valid price "
        f"data at all in window; circuit/no-trade days are carried "
        f"forward from last price, not skipped) | "
        f"Price filter > Rs.{MIN_PRICE} | "
        f"Liquidity > {MIN_AVG_VOLUME:,} "
        f"({VOLUME_LOOKBACK}D avg vol)"
    ])

    add_row()
    add_row(
        left=[f"Ranked Stocks (Rank 1 = Strongest {sc})"]
    )

    ranking_clean = sanitize_for_sheets(
        ranking_df
    )

    add_row(
        left=list(ranking_clean.columns)
    )

    for r in ranking_clean.values.tolist():
        add_row(left=r)

    if skipped_symbols:
        skip_reasons = skip_reasons or {}
        detail = "; ".join(
            f"{sym} ({skip_reasons.get(sym, 'unknown reason')})"
            for sym in skipped_symbols
        )
        add_row(
            left=[
                "Skipped (no valid price data at all in window -- "
                "not a completeness/circuit issue, genuinely no data): "
                + detail
            ]
        )

    add_row()
    add_row()

    chart_requests = []
    chart_labels = []

    def add_equity_block(equity_table, window_label, window_days, log_scale_hint=False):
        nonlocal next_chart_anchor_row

        if equity_table is None:
            add_row(left=[
                f"Top {TOP10_N} {nm} Equity Curve ({window_label}): "
                f"skipped -- not enough history yet for a full "
                f"Top {TOP10_N} portfolio plus "
                f"{max(EQUITY_SMA_PERIODS)}-day SMA warmup."
            ])
            add_row()
            return

        has_sma = any(
            f"sma{period}_pct" in equity_table.columns
            or f"sma{period}_index" in equity_table.columns
            for period in EQUITY_SMA_PERIODS
        )

        sma_legend = (
            "Black (BOLD) = equity curve, Blue (dashed) = 20 SMA, "
            "Orange (dashed) = 50 SMA, Red (dashed) = 200 SMA -- "
            "equity below the red 200 SMA is the classic cue to "
            "consider de-risking to cash | "
            if has_sma
            else "Black (BOLD) = equity curve (no SMA overlays on this window) | "
        )

        log_note = (
            "Plotted as RAW equity index (base=100, always positive) "
            "instead of rebased % change, so it's safe to view on a "
            "log axis -- Sheets' Log scale toggle is UI-only (can't "
            "be set via the API) and this chart is rebuilt every run, "
            "so manually check Customize > Vertical axis > Log scale "
            "after each run if you want it. | "
            if log_scale_hint else ""
        )

        add_row(left=[
            f"Top {TOP10_N} {nm} Equal-Weight Equity Curve "
            f"({window_label}, Daily Rebalance, No Costs) | "
            f"{log_note}"
            f"{sma_legend}"
            f"Teal (dashed, right axis) = Avg {sc} of that day's "
            f"Top {TOP10_N} basket, RAW (not rebased) -- rising teal "
            "= the leading basket's strength is increasing, "
            "falling teal = it's weakening, independent of whether "
            "the equity curve itself is up or down that day"
        ])

        eq_header_row = add_row(
            right=list(equity_table.columns)
        )

        eq_header_row_0idx = eq_header_row - 1

        eq_table = equity_table.copy()

        last_idx = eq_table.index[-1]
        time_part = timestamp.split(" ", 1)[1] if " " in timestamp else timestamp
        eq_table.loc[last_idx, "date"] = (
            f"{eq_table.loc[last_idx, 'date']} (as of {time_part})"
        )

        eq_clean = sanitize_for_sheets(eq_table)

        for r in eq_clean.values.tolist():
            add_row(right=r)

        add_row()
        add_row()

        has_avg_rs = cfg["avg_col"] in eq_table.columns

        if has_sma:
            base_colors = EQUITY_SERIES_COLORS
            base_widths = EQUITY_SERIES_WIDTHS
            base_styles = EQUITY_SERIES_STYLES
            n_base_series = 4
        else:
            base_colors = [EQUITY_COLOR]
            base_widths = [EQUITY_LINE_WIDTH]
            base_styles = ["SOLID"]
            n_base_series = 1

        chart_title = (
            f"Top {TOP10_N} {nm} Equity Curve vs "
            f"20/50/200 SMA + Avg {sc} ({window_label})"
            if has_sma
            else f"Top {TOP10_N} {nm} Equity Curve + Avg {sc} ({window_label})"
        )

        left_axis_title = (
            "Equity Index Level (Base = 100) -- enable Log scale manually"
            if log_scale_hint else "% Change from Day 1 (Base = 0)"
        )

        chart_requests.append(
            make_stock_chart(
                ws.id,
                chart_title,
                eq_header_row_0idx,
                len(eq_table),
                anchor_row=next_chart_anchor_row,
                data_col_start=DATA_COL_START,
                n_series=(
                    n_base_series + 1 if has_avg_rs else n_base_series
                ),
                colors=(
                    base_colors + [AVG_RS_COLOR]
                    if has_avg_rs else base_colors
                ),
                series_axes=(
                    ["LEFT_AXIS"] * n_base_series + ["RIGHT_AXIS"]
                    if has_avg_rs else None
                ),
                series_widths=(
                    base_widths + [AVG_RS_LINE_WIDTH]
                    if has_avg_rs else base_widths
                ),
                series_line_styles=(
                    base_styles + [AVG_RS_LINE_STYLE]
                    if has_avg_rs else base_styles
                ),
                left_axis_title=left_axis_title,
                right_axis_title=(
                    f"Avg {sc} of Top {TOP10_N} Basket (raw)"
                    if has_avg_rs else None
                ),
            )
        )
        chart_labels.append(f"Equity curve ({window_label})")
        next_chart_anchor_row += CHART_ROW_SPACING

    add_equity_block(
        equity_table_50d,
        f"Last {RS_LINE_WINDOW} Days",
        RS_LINE_WINDOW
    )
    add_equity_block(
        equity_table_1y,
        f"Last {EQUITY_1Y_WINDOW} Days / 1 Year",
        EQUITY_1Y_WINDOW
    )
    add_equity_block(
        equity_table_full,
        f"Full History / ~{DOWNLOAD_YEARS} Years",
        len(equity_table_full) if equity_table_full is not None else 0,
        log_scale_hint=True
    )

    for rank, symbol, df in stock_series_list:
        add_row(
            left=[f"Rank {rank} - {symbol}"]
        )

        header_row = add_row(
            right=list(df.columns)
        )

        header_row_0idx = header_row - 1

        df_clean = sanitize_for_sheets(df)

        for r in df_clean.values.tolist():
            add_row(right=r)

        add_row()
        add_row()

        chart_requests.append(
            make_stock_chart(
                ws.id,
                (
                    f"Rank {rank} - {symbol}: "
                    f"Price % (green=Top{TOP10_N}/blue=outside) + "
                    f"{nm} Rank Percentile trend (purple, right axis) "
                    f"(Last {RS_LINE_WINDOW} Days)"
                ),
                header_row_0idx,
                len(df),
                anchor_row=next_chart_anchor_row,
                data_col_start=DATA_COL_START,
                n_series=3,
                colors=SERIES_COLORS + [RANK_LINE_COLOR],
                series_axes=["LEFT_AXIS", "LEFT_AXIS", "RIGHT_AXIS"],
                left_axis_title="Price % Change from Day 1 (Base = 0)",
                right_axis_title=(
                    f"{nm} Rank Percentile ({RS_PERCENTILE_BEST} = "
                    f"strongest / Rank 1, {RS_PERCENTILE_WORST} = "
                    "weakest eligible that day; see daily_rank col "
                    "for actual rank)"
                ),
            )
        )
        chart_labels.append(f"Rank {rank} - {symbol}")
        next_chart_anchor_row += CHART_ROW_SPACING

    write_rows_in_chunks(
        ws,
        rows_left,
        chunk_size=1500,
        label=f"{nm} sheet (left)",
        start_col="A"
    )

    write_rows_in_chunks(
        ws,
        rows_right,
        chunk_size=1500,
        label=f"{nm} sheet (data, right)",
        start_col=col_letter(DATA_COL_START)
    )

    failed_charts = add_charts_robust(sh, chart_requests, chart_labels)

    print(
        f"\n{nm} screener results written to "
        f"'{worksheet_name}' tab: "
        f"{len(ranking_df)} ranked stocks, "
        f"{len(chart_requests) - len(failed_charts)}/"
        f"{len(chart_requests)} charts added."
    )

    if failed_charts:
        print(
            "NOTE: the charts above could not be added after retries "
            "-- their data tables are still on the sheet, only the "
            "chart objects are missing. Re-running the script will "
            "attempt them again."
        )


# ============================================================
# PIPELINE (runs once per sheet: RS, then Composite)
# ============================================================

RS_CFG = {
    "name": "RS",
    "score_label": "RS Score",
    "score_col": "rs_score_pct",
    "avg_col": "avg_rs_score_top10",
    "worksheet": SCREENER_WORKSHEET,
    "csv_prefix": "RS_Top50",
    "eq_prefix": "RS_Top10_Equity_Curve",
    "title": f"RS TOP {TOP_N} SCREENER",
    "formula_line": (
        "RS Formula: 40% 3M + 20% 6M + 20% 9M + 20% 12M "
        "Price Rate-of-Change"
    ),
    "extra_cols": (),
    "extra_note": "",
}


def make_composite_cfg(info):
    if info["has_group"]:
        extra = (
            "COMPOSITE audit columns (latest day, uncharted): "
            "rs_score_raw = original 3-12M RS score, rs_pctile / "
            "accdis_pctile / group_pctile = the three component "
            f"percentiles (Industry Group RS from {info['n_groups']} "
            f"groups, {info['coverage']:.0%} of universe tagged) | "
        )
    else:
        extra = (
            "COMPOSITE audit columns (latest day, uncharted): "
            "rs_score_raw = original 3-12M RS score, rs_pctile / "
            "accdis_pctile = component percentiles (group_pctile is "
            "blank -- no usable industry column in stocks.csv) | "
        )

    return {
        "name": "Composite",
        "score_label": "Composite Score",
        "score_col": "composite_score",
        "avg_col": "avg_composite_score_top10",
        "worksheet": COMPOSITE_WORKSHEET,
        "csv_prefix": "Composite_Top50",
        "eq_prefix": "Composite_Top10_Equity_Curve",
        "title": f"COMPOSITE TOP {TOP_N} SCREENER (MarketSmith-style proxy)",
        "formula_line": (
            f"Composite Score (0-100): {info['formula']} -- price/volume "
            "proxy; EPS/SMR omitted (no point-in-time fundamentals, "
            "would be lookahead bias)"
        ),
        "extra_cols": (
            "rs_score_raw", "rs_pctile", "accdis_pctile", "group_pctile"
        ),
        "extra_note": extra,
    }


def determine_as_of_date(all_stocks, bench_close):
    latest_stock_date = max(
        df.index.max()
        for df in all_stocks.values()
    )

    as_of_date = min(
        pd.Timestamp(
            latest_stock_date
        ).normalize(),
        pd.Timestamp(
            bench_close.index.max()
        ).normalize()
    )

    if STRICT_SETTLED_CLOSE:
        today = pd.Timestamp.today().normalize()

        if as_of_date == today:
            prior_days = bench_close.index[bench_close.index < today]

            if len(prior_days) > 0:
                settled_as_of_date = prior_days.max()
                print(
                    f"\nNOTE: today ({today:%Y-%m-%d}) is still live/"
                    "intraday -- its Close is a moving price, not a "
                    "settled one, so ranking off it would reshuffle "
                    "between re-runs. Ranking off the last SETTLED "
                    f"close instead: {settled_as_of_date:%Y-%m-%d}. "
                    "Set STRICT_SETTLED_CLOSE = False to rank off "
                    "today's live price anyway."
                )
                as_of_date = settled_as_of_date
            else:
                print(
                    "\nWARNING: today is the only available trading "
                    "day and STRICT_SETTLED_CLOSE is on, but there's "
                    "no prior settled day to fall back to -- ranking "
                    "off today's live price anyway."
                )

    return as_of_date


def run_screener(all_stocks, bench_close, as_of_date, cfg):
    nm = cfg["name"]
    score_col = cfg["score_col"]

    print()
    print("=" * 70)
    print(f"{cfg['title']} -- INDIVIDUAL SCORE LINE + PRICE CHARTS")
    print("=" * 70)
    print(f"Score          : {cfg['formula_line']}")
    print("Ranking        : Full eligible universe, descending score")
    print(f"Output         : Top {TOP_N}, Rank 1 downward -> '{cfg['worksheet']}'")
    print(
        "Per-stock chart: Price % (green=Top"
        f"{TOP10_N}/blue=outside) + {nm} Rank Percentile trend "
        f"(purple, right axis, {RS_PERCENTILE_WORST}-"
        f"{RS_PERCENTILE_BEST} scale), rebased to 0% at day 1, "
        f"last {RS_LINE_WINDOW} trading days"
    )
    print(f"Price filter   : > Rs.{MIN_PRICE}")
    print(
        f"Liquidity      : {VOLUME_LOOKBACK}D average volume "
        f"> {MIN_AVG_VOLUME:,}"
    )
    print("=" * 70)

    print(f"\nAs-of date: {as_of_date:%Y-%m-%d}")

    ranking = build_ranking(all_stocks, as_of_date)

    print(f"Eligible universe: {len(ranking)} stocks")

    if not ranking:
        raise RuntimeError(
            "No eligible stocks on the as-of date."
        )

    top_ranking = ranking[:TOP_N]

    ranking_df = pd.DataFrame([
        {
            "rank": i + 1,
            "symbol": sym,
            score_col: round(rs, 2),
            "price": round(price, 2),
            "avg_volume_20d": round(avg_vol, 0),
            "rs_score_20d": round(rs_20d, 2) if not pd.isna(rs_20d) else "",
            "rs_accel_20d": round(rs_accel, 2) if not pd.isna(rs_accel) else "",
        }
        for i, (
            sym,
            rs,
            price,
            avg_vol,
            rs_20d,
            rs_accel,
        ) in enumerate(top_ranking)
    ])

    # Composite sheet: component audit columns on the ranking table
    for col in cfg["extra_cols"]:
        vals = []
        for sym in ranking_df["symbol"]:
            r = get_row(all_stocks[sym], as_of_date)
            v = r.get(col, np.nan) if r is not None else np.nan
            vals.append(round(float(v), 2) if not pd.isna(v) else "")
        ranking_df[col] = vals

    print(
        f"\nTop {len(ranking_df)} {nm} Stocks "
        "(Rank 1 = Strongest):"
    )

    print(ranking_df.to_string(index=False))

    trading_days = bench_close.index[
        bench_close.index <= as_of_date
    ]

    trading_days_window = trading_days[-RS_LINE_WINDOW:]
    trading_days_window_1y = trading_days[-EQUITY_1Y_WINDOW:]

    if len(trading_days_window) < RS_LINE_WINDOW:
        print(
            f"\nWARNING: only "
            f"{len(trading_days_window)} "
            "trading days of benchmark history "
            "available; chart window shortened."
        )

    if len(trading_days_window_1y) < EQUITY_1Y_WINDOW:
        print(
            f"\nWARNING: only "
            f"{len(trading_days_window_1y)} "
            "trading days of benchmark history "
            "available; 1-year equity window shortened."
        )

    print(
        f"\nComputing daily rank across "
        f"{len(trading_days)} days "
        "(drives price line coloring, rank percentile, audit column, "
        "and all Top10 equity curves + avg score overlay)..."
    )

    rank_maps, avg_rs_series = compute_daily_rank_maps(
        all_stocks,
        trading_days,
        TOP10_N
    )

    equity_index = compute_top10_equity_curve(
        all_stocks,
        rank_maps,
        trading_days,
        TOP10_N
    )

    avg_col = cfg["avg_col"]

    # 50-day: raw equity curve, rebased %, no SMA overlays.
    equity_table_50d = build_top10_equity_table(
        equity_index,
        trading_days_window,
        avg_rs_series,
        include_sma=False,
        rebase_pct=True,
        avg_col_name=avg_col
    )

    # 1-year: rebased %, full 20/50/200 SMA overlays.
    equity_table_1y = build_top10_equity_table(
        equity_index,
        trading_days_window_1y,
        avg_rs_series,
        include_sma=True,
        rebase_pct=True,
        avg_col_name=avg_col
    )

    # Full history: RAW equity index (log-scale-safe), SMAs retained.
    equity_table_full = build_top10_equity_table(
        equity_index,
        trading_days,
        avg_rs_series,
        include_sma=True,
        rebase_pct=False,
        avg_col_name=avg_col
    )

    for tbl, label in (
        (equity_table_50d, "50-day"),
        (equity_table_1y, "1-year"),
        (equity_table_full, "full history"),
    ):
        if tbl is None:
            print(
                f"\nTop10 equity curve ({label}): skipped -- not enough "
                f"history yet for a full Top {TOP10_N} portfolio "
                f"(plus {max(EQUITY_SMA_PERIODS)}-day SMA warmup where "
                "applicable). Resolves itself as data accumulates."
            )
        else:
            print(
                f"Top10 equity curve ({label}) built: "
                f"{len(tbl)} days displayed."
            )

    stock_series_list = []
    skipped_symbols = []
    skip_reasons = {}
    fill_notes = {}

    for i, (
        sym,
        rs,
        price,
        avg_vol,
        rs_20d,
        rs_accel,
    ) in enumerate(top_ranking):

        rank = i + 1

        series_df, note = build_stock_series(
            all_stocks,
            sym,
            trading_days_window,
            rank_maps,
            extra_audit_cols=cfg["extra_cols"]
        )

        if series_df is None:
            skipped_symbols.append(sym)
            skip_reasons[sym] = note
            continue

        stock_series_list.append(
            (rank, sym, series_df)
        )

        if note:
            fill_notes[sym] = note

    print(
        f"\nCharted "
        f"{len(stock_series_list)}/"
        f"{len(top_ranking)} stocks "
        f"({len(skipped_symbols)} truly unplottable -- "
        f"no valid price data at all in the window):"
    )
    for sym in skipped_symbols:
        print(f"  {sym}: {skip_reasons.get(sym, 'unknown reason')}")

    if fill_notes:
        print(
            f"\n{len(fill_notes)} stock(s) had no-trade/circuit days "
            "carried forward from the last available price:"
        )
        for sym, note in fill_notes.items():
            print(f"  {sym}: {note}")

    write_to_sheet(
        cfg,
        ranking_df,
        stock_series_list,
        skipped_symbols,
        as_of_date.strftime("%Y-%m-%d"),
        equity_table_50d,
        equity_table_1y,
        equity_table_full,
        skip_reasons
    )

    ranking_df.to_csv(
        f"{cfg['csv_prefix']}_Ranking.csv",
        index=False
    )

    for rank, symbol, df in stock_series_list:
        df.to_csv(
            f"{cfg['csv_prefix']}_Stock_{rank:02d}_{symbol}.csv",
            index=False
        )

    print(f"\n{nm} CSV files also saved.")
    print(f"\n{nm} SCREENER COMPLETED SUCCESSFULLY.")


# ============================================================
# MAIN
# ============================================================

def main():
    tickers = load_tickers()
    industry_map = load_industry_map()

    print(f"\nLoaded {len(tickers)} tickers.")
    print(
        f"Industry tags loaded: {len(industry_map)} "
        "(used by Composite sheet only)"
    )

    bench_close = download_benchmark()
    bench_close.index = normalize_dates(
        bench_close.index
    )

    all_stocks = {}
    total_bad_points = 0

    batch_size = 50

    download_start = (
        pd.Timestamp.today() -
        pd.DateOffset(years=DOWNLOAD_YEARS)
    ).strftime("%Y-%m-%d")

    for start in range(
        0,
        len(tickers),
        batch_size
    ):
        batch = tickers[
            start:start + batch_size
        ]

        print(
            f"\nDownloading "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(tickers)}"
        )

        try:
            data = yf.download(
                batch,
                start=download_start,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True
            )

        except Exception as e:
            print(f"Batch failed: {e}")
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    sdata = data

                else:
                    if not isinstance(
                        data.columns,
                        pd.MultiIndex
                    ):
                        continue

                    if (
                        symbol
                        not in
                        data.columns.get_level_values(0)
                    ):
                        continue

                    sdata = data[symbol]

                if "Close" not in sdata.columns:
                    continue

                close = (
                    sdata["Close"]
                    .dropna()
                    .sort_index()
                )

                if close.empty:
                    continue

                volume = (
                    sdata["Volume"]
                    .reindex(close.index)
                    .fillna(0)
                )

                close, n_bad = clean_price_series(
                    close
                )

                total_bad_points += n_bad

                stock_data = compute_stock_data(
                    close,
                    volume
                )

                if stock_data is None:
                    continue

                all_stocks[
                    symbol.replace(".NS", "")
                ] = stock_data

            except Exception as e:
                print(
                    f"Skipping {symbol}: {e}"
                )

        time.sleep(1)

    print(
        f"\nStocks with usable data: "
        f"{len(all_stocks)}"
    )

    print(
        f"Repaired data points: "
        f"{total_bad_points}"
    )

    if not all_stocks:
        raise RuntimeError(
            "No usable stock data."
        )

    as_of_date = determine_as_of_date(all_stocks, bench_close)

    # ---- Sheet 1: RS Top 50 (unchanged logic) ----
    run_screener(all_stocks, bench_close, as_of_date, RS_CFG)

    # ---- Sheet 2: Composite Top 50 ----
    # Isolated so a composite failure can never undo/skip the RS sheet.
    if ENABLE_COMPOSITE:
        composite_error = None
        try:
            time.sleep(BETWEEN_SHEETS_PAUSE_SECONDS)

            print("\nBuilding composite scores (RS + Acc/Dis + Group)...")
            comp_stocks, info = build_composite_stocks(
                all_stocks,
                industry_map
            )
            print(f"Composite formula: {info['formula']}")

            run_screener(
                comp_stocks,
                bench_close,
                as_of_date,
                make_composite_cfg(info)
            )

        except Exception as e:
            composite_error = e
            print()
            print("=" * 70)
            print("COMPOSITE SCREENER FAILED (RS sheet already written)")
            print("=" * 70)
            print(f"{type(e).__name__}: {e}")

        if composite_error is not None:
            raise composite_error


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print()
        print("=" * 70)
        print("SCREENER FAILED")
        print("=" * 70)
        print(
            f"{type(e).__name__}: {e}"
        )
        raise
