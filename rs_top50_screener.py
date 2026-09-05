"""
RS TOP 50 SCREENER -- INDIVIDUAL RS LINE + PRICE CHARTS

WHAT THIS DOES (single snapshot, not a backtest)
  1. Downloads price/volume history for the stock universe + benchmark.
  2. Computes a weighted multi-timeframe RS Score (40% 3M + 20% 6M +
     20% 9M + 20% 12M price momentum) for every eligible stock as of
     the latest available trading day.
  3. Ranks all eligible stocks by RS Score, descending (Rank 1 = strongest).
  4. Writes the full ranked table, Rank 1 downward.
  5. For EACH of the Top 50 stocks individually, builds its own
     small data table (last RS_LINE_WINDOW trading days: date, price
     % change, RS Line % change) and its own two-line chart -- both
     series rebased to 0% at day 1 of the window.
  6. The PRICE line is split into two color-coded series so it renders
     GREEN on any day the stock was ranked in the Top 10 (by RS Score)
     and BLUE on any day it was outside the Top 10. The RS Line is a
     single, constant RED series. Boundary days are duplicated across
     both price color segments so the line stays visually continuous.
     The actual daily numeric rank is also written out as a plain
     (uncharted) column so the green/blue split can be audited against
     real numbers.
  7. A Top 10 RS equal-weight, daily-rebalanced cumulative-return
     equity curve (no costs/slippage modeled) is also built, with
     20/50/200-day SMA overlays, rebased to 0% at day 1 of the same
     display window -- a regime/timing overlay, not a real backtest.
     The most recent day is labeled with the time of the run since it
     reflects the latest fetched price, not a settled close.

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

DOWNLOAD_YEARS = 3  # bumped from 2: RS_12M lookback (252d) + 200-day SMA
                     # warmup + 50-day display window needs ~500+ trading
                     # days of buffer before the equity curve is stable

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

RS_3M, RS_6M, RS_9M, RS_12M = 63, 126, 189, 252
RS_WEIGHTS = (0.40, 0.20, 0.20, 0.20)

TOP_N = 100
RS_LINE_WINDOW = 50

# Rank threshold used to color the price line (green inside, blue outside)
TOP10_N = 10

# Chart colors (Google Sheets Color proto: 0-1 floats)
GREEN_COLOR = {"red": 0.20, "green": 0.65, "blue": 0.33}   # in Top 10
BLUE_COLOR = {"red": 0.26, "green": 0.52, "blue": 0.96}    # outside Top 10
RS_LINE_COLOR = {"red": 0.80, "green": 0.08, "blue": 0.08}  # RS line (solid red)
SERIES_COLORS = [GREEN_COLOR, BLUE_COLOR, RS_LINE_COLOR]

# Top 10 RS equal-weight equity curve (regime/timing overlay)
EQUITY_SMA_PERIODS = (20, 50, 200)
EQUITY_COLOR = {"red": 0.15, "green": 0.15, "blue": 0.15}        # equity curve (near-black)
EQUITY_SMA20_COLOR = {"red": 0.20, "green": 0.60, "blue": 0.86}  # 20 SMA (blue)
EQUITY_SMA50_COLOR = {"red": 0.95, "green": 0.60, "blue": 0.10}  # 50 SMA (orange)
EQUITY_SMA200_COLOR = {"red": 0.80, "green": 0.10, "blue": 0.10}  # 200 SMA (red - de-risk trigger)
EQUITY_SERIES_COLORS = [
    EQUITY_COLOR, EQUITY_SMA20_COLOR, EQUITY_SMA50_COLOR, EQUITY_SMA200_COLOR
]

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
SCREENER_WORKSHEET = "Screener - RS Top50"


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

    result = pd.DataFrame({
        "price": close,
        "avg_volume": avg_volume,
        "liquid": liquid,
        "rs_score": rs_score,
    })

    result.index = normalize_dates(result.index)

    return result


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

        ranking.append((
            symbol,
            float(rs),
            float(price),
            float(avg_vol) if not pd.isna(avg_vol) else 0.0
        ))

    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def compute_daily_rank_maps(all_stocks, trading_days_window):
    """
    For every date in the window, rank the full eligible universe and
    return {date: {symbol: rank}} (rank 1 = strongest that day). A
    symbol absent from a given day's map was not eligible that day
    (illiquid / below price floor / insufficient history).

    This is the single source of truth used both to color-split each
    stock's price line (green inside Top N, blue outside) and to
    print the raw daily rank next to the data, so the coloring can be
    visually audited against the actual numbers.
    """
    rank_maps = {}

    for d in trading_days_window:
        ranking = build_ranking(all_stocks, d)
        rank_maps[d] = {
            sym: i + 1
            for i, (sym, rs, price, vol) in enumerate(ranking)
        }

    return rank_maps


def compute_top10_equity_curve(all_stocks, rank_maps, full_calendar, top_n=10):
    """
    Equal-weight, daily-rebalanced cumulative return index for the
    Top N RS-ranked stocks, using each day's rank as of the PRIOR
    trading day (no lookahead). Base = 100 on the first day a full
    Top N portfolio can be formed.

    This is a regime/timing overlay, not a real backtest: no
    transaction costs, slippage, or entry/exit buffers are modeled,
    and a day only counts if a full Top N can be formed from the
    previous day's ranking.
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


def build_top10_equity_table(equity_index, trading_days_window):
    """
    Slice the Top N equity curve (and its 20/50/200 SMA, computed on
    the FULL curve so the SMAs are properly warmed up) down to the
    display window, then rebase everything to 0% at the window's
    first day. Rebasing is a simple division by a positive constant,
    so it preserves exactly where the equity curve crosses each SMA
    -- it's purely a display transform.
    """
    if equity_index.empty:
        return None

    window_equity = equity_index.reindex(trading_days_window)

    if window_equity.dropna().empty:
        return None

    base = window_equity.dropna().iloc[0]
    if pd.isna(base) or base <= 0:
        return None

    def rebase(s):
        return (s / base - 1) * 100

    sma_cols = {}
    for period in EQUITY_SMA_PERIODS:
        sma = equity_index.rolling(period).mean().reindex(trading_days_window)
        sma_cols[f"sma{period}_pct"] = rebase(sma).round(3).values

    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in trading_days_window],
        "top10_equity_pct": rebase(window_equity).round(3).values,
        **sma_cols,
    })


def split_by_rank(values, flags):
    """
    Split a single series into two parallel series based on a boolean
    flag per point: `in_flag` values go to `top`, everything else goes
    to `other`. At every day the flag flips, the boundary point is
    duplicated into both arrays so the two color segments meet at the
    same x-position instead of leaving a visual gap in the line.
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
                # entering Top N at i -> extend the green segment back to i-1
                top[i - 1] = values[i - 1]
            else:
                # exiting Top N at i -> extend the blue segment back to i-1
                other[i - 1] = values[i - 1]

    return top, other


# ============================================================
# PRICE + RS LINE
# ============================================================

def build_stock_series(
    all_stocks,
    symbol,
    bench_close,
    trading_days_window,
    rank_maps
):
    df = all_stocks[symbol]

    prices = df["price"].reindex(trading_days_window)
    bench = bench_close.reindex(trading_days_window)

    if (
        prices.isna().any()
        or bench.isna().any()
        or (bench <= 0).any()
    ):
        return None

    price_base = prices.iloc[0]

    if price_base <= 0 or pd.isna(price_base):
        return None

    price_pct = (prices / price_base - 1) * 100

    ratio = prices / bench
    rs_base = ratio.iloc[0]

    if rs_base <= 0 or pd.isna(rs_base):
        return None

    rs_line_pct = (ratio / rs_base - 1) * 100

    # Daily rank (None = not eligible that day) and the Top-N flag
    # derived directly from it -- this is what drives the green/blue
    # split, and the raw rank is also written out as its own column
    # so the split can be visually checked against real numbers.
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

    return pd.DataFrame({
        "date": [
            d.strftime("%Y-%m-%d")
            for d in trading_days_window
        ],
        "price_pct_top10": np.round(np.array(top_vals, dtype=float), 3),
        "price_pct_other": np.round(np.array(other_vals, dtype=float), 3),
        "rs_line_pct": rs_line_pct.round(3).values,
        "daily_rank": rank_col,
    })


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
    colors
):
    data_end_row = header_row_0idx + 1 + n_rows

    def series(col_index, color):
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
            "targetAxis": "LEFT_AXIS",
            "color": color,
            "colorStyle": {"rgbColor": color},
            "lineStyle": {
                "width": 2,
                "type": "SOLID"
            },
            "pointStyle": {
                "size": 3,
                "shape": "CIRCLE"
            },
        }

    return {
        "addChart": {
            "chart": {
                "spec": {
                    "title": title,
                    "basicChart": {
                        "chartType": "LINE",
                        "legendPosition": "BOTTOM_LEGEND",
                        "axis": [
                            {
                                "position": "BOTTOM_AXIS",
                                "title": "Date"
                            },
                            {
                                "position": "LEFT_AXIS",
                                "title":
                                    "% Change from Day 1 "
                                    "(Base = 0)"
                            },
                        ],
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
                                colors[i]
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
    ranking_df,
    stock_series_list,
    skipped_symbols,
    as_of_date,
    equity_table=None
):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print(
            "Missing SHEET_ID/GOOGLE_CREDENTIALS "
            "-- saving to CSV instead."
        )

        ranking_df.to_csv(
            "RS_Top50_Ranking.csv",
            index=False
        )

        if equity_table is not None:
            equity_table.to_csv(
                "RS_Top10_Equity_Curve.csv",
                index=False
            )

        for rank, symbol, df in stock_series_list:
            df.to_csv(
                f"RS_Top50_Stock_{rank:02d}_{symbol}.csv",
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

    n_rows_needed = (
        3
        + len(ranking_df)
        + 3
        + block_height  # Top10 equity curve block
        + len(stock_series_list) * block_height
        + 20
    )

    # date column + 3 charted series (price_pct_top10, price_pct_other,
    # rs_line_pct) + 1 audit column (daily_rank, not charted) + buffer
    n_cols_needed = DATA_COL_START + 5 + 2

    ws = call_with_quota_retry(
        lambda: get_or_create_worksheet(
            sh,
            SCREENER_WORKSHEET,
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

    add_row(left=[
        f"RS TOP {TOP_N} SCREENER | "
        f"run {timestamp} | "
        f"As of: {as_of_date} | "
        f"RS Formula: 40% 3M + 20% 6M + "
        f"20% 9M + 20% 12M Price Rate-of-Change | "
        f"Charts: Price % (GREEN = in Top {TOP10_N} by RS Score that "
        f"day, BLUE = outside Top {TOP10_N}) and RS Line % "
        f"(price/benchmark, RED), all rebased to 0% "
        f"at day 1 of the last "
        f"{RS_LINE_WINDOW}-day window | "
        f"'daily_rank' column shows the actual rank each day "
        f"(not charted) to audit the green/blue split | "
        f"Top {TOP10_N} RS Equal-Weight Equity Curve included "
        f"(daily rebalance, no costs) with 20/50/200 SMA overlays "
        f"for regime timing | "
        f"{len(stock_series_list)}/{TOP_N} stocks charted "
        f"({len(skipped_symbols)} skipped: incomplete "
        f"{RS_LINE_WINDOW}-day history) | "
        f"Price filter > Rs.{MIN_PRICE} | "
        f"Liquidity > {MIN_AVG_VOLUME:,} "
        f"({VOLUME_LOOKBACK}D avg vol)"
    ])

    add_row()
    add_row(
        left=["Ranked Stocks (Rank 1 = Strongest RS)"]
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
        add_row(
            left=[
                "Skipped (incomplete "
                f"{RS_LINE_WINDOW}-day history): "
                + ", ".join(skipped_symbols)
            ]
        )

    add_row()
    add_row()

    chart_requests = []

    if equity_table is not None:
        add_row(left=[
            f"Top {TOP10_N} RS Equal-Weight Equity Curve "
            f"(Last {RS_LINE_WINDOW} Days, Daily Rebalance, No Costs) | "
            "Black = equity curve, Blue = 20 SMA, Orange = 50 SMA, "
            "Red = 200 SMA -- equity below the red 200 SMA is the "
            "classic cue to consider de-risking to cash"
        ])

        eq_header_row = add_row(
            right=list(equity_table.columns)
        )

        eq_header_row_0idx = eq_header_row - 1

        eq_table = equity_table.copy()

        # Label the most recent day as intraday/live so it's clear
        # that value uses the latest fetched price, not a settled close.
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

        chart_requests.append(
            make_stock_chart(
                ws.id,
                (
                    f"Top {TOP10_N} RS Equity Curve vs "
                    f"20/50/200 SMA (Last {RS_LINE_WINDOW} Days)"
                ),
                eq_header_row_0idx,
                len(eq_table),
                anchor_row=eq_header_row_0idx,
                data_col_start=DATA_COL_START,
                n_series=4,
                colors=EQUITY_SERIES_COLORS,
            )
        )
    else:
        add_row(left=[
            f"Top {TOP10_N} RS Equity Curve: skipped -- not enough "
            f"history yet for a full Top {TOP10_N} portfolio plus "
            f"{max(EQUITY_SMA_PERIODS)}-day SMA warmup."
        ])
        add_row()

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
                    f"Price % (green=Top{TOP10_N}/blue=outside) "
                    f"vs RS Line % "
                    f"(Last {RS_LINE_WINDOW} Days, Base=0)"
                ),
                header_row_0idx,
                len(df),
                anchor_row=header_row_0idx,
                data_col_start=DATA_COL_START,
                n_series=3,
                colors=SERIES_COLORS,
            )
        )

    write_rows_in_chunks(
        ws,
        rows_left,
        chunk_size=1500,
        label="screener sheet (left)",
        start_col="A"
    )

    write_rows_in_chunks(
        ws,
        rows_right,
        chunk_size=1500,
        label="screener sheet (data, right)",
        start_col=col_letter(DATA_COL_START)
    )

    if chart_requests:
        chunk = 10

        for i in range(
            0,
            len(chart_requests),
            chunk
        ):
            batch = chart_requests[
                i:i + chunk
            ]

            try:
                call_with_quota_retry(
                    lambda b=batch:
                        sh.batch_update(
                            {"requests": b}
                        ),
                    label=(
                        f"chart batch "
                        f"{i + 1}-"
                        f"{i + len(batch)}"
                    )
                )

                print(
                    f"Added charts "
                    f"{i + 1}-"
                    f"{min(i + chunk, len(chart_requests))} "
                    f"of {len(chart_requests)}"
                )

            except Exception as e:
                print(
                    f"Could not add chart batch "
                    f"{i + 1}-"
                    f"{i + len(batch)} "
                    f"(non-fatal, continuing)"
                )
                # Print the full API error body (str(e) alone is often
                # just a status code) so the real cause is visible.
                detail = getattr(e, "response", None)
                if detail is not None:
                    try:
                        print("API error body:", detail.text)
                    except Exception:
                        print("API error (raw):", repr(e))
                else:
                    print("API error (raw):", repr(e))

    print(
        f"\nScreener results written to "
        f"'{SCREENER_WORKSHEET}' tab: "
        f"{len(ranking_df)} ranked stocks, "
        f"{len(stock_series_list)} charts."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print(
        f"RS TOP {TOP_N} SCREENER -- "
        "INDIVIDUAL RS LINE + PRICE CHARTS"
    )
    print("=" * 70)

    print(
        "RS Formula     : "
        "40% 3M + 20% 6M + 20% 9M + 20% 12M "
        "Price Rate-of-Change"
    )

    print(
        "Ranking        : "
        "Full eligible universe, descending RS Score"
    )

    print(
        f"Output         : Top {TOP_N}, Rank 1 downward"
    )

    print(
        "Per-stock chart: Price % (green=Top"
        f"{TOP10_N}/blue=outside) + RS Line % (orange), "
        "both rebased to 0% at day 1, "
        f"last {RS_LINE_WINDOW} trading days"
    )

    print(
        f"Price filter   : > Rs.{MIN_PRICE}"
    )

    print(
        f"Liquidity      : "
        f"{VOLUME_LOOKBACK}D average volume "
        f"> {MIN_AVG_VOLUME:,}"
    )

    print("=" * 70)

    tickers = load_tickers()

    print(f"\nLoaded {len(tickers)} tickers.")

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

    print(
        f"\nAs-of date: "
        f"{as_of_date:%Y-%m-%d}"
    )

    ranking = build_ranking(
        all_stocks,
        as_of_date
    )

    print(
        f"Eligible universe: "
        f"{len(ranking)} stocks"
    )

    if not ranking:
        raise RuntimeError(
            "No eligible stocks on the as-of date."
        )

    top_ranking = ranking[:TOP_N]

    ranking_df = pd.DataFrame([
        {
            "rank": i + 1,
            "symbol": sym,
            "rs_score_pct": round(rs, 2),
            "price": round(price, 2),
            "avg_volume_20d": round(avg_vol, 0),
        }
        for i, (
            sym,
            rs,
            price,
            avg_vol
        ) in enumerate(top_ranking)
    ])

    print(
        f"\nTop {len(ranking_df)} RS Stocks "
        "(Rank 1 = Strongest):"
    )

    print(
        ranking_df.to_string(index=False)
    )

    trading_days = bench_close.index[
        bench_close.index <= as_of_date
    ]

    trading_days_window = trading_days[
        -RS_LINE_WINDOW:
    ]

    if len(trading_days_window) < RS_LINE_WINDOW:
        print(
            f"\nWARNING: only "
            f"{len(trading_days_window)} "
            "trading days of benchmark history "
            "available; chart window shortened."
        )

    print(
        f"\nComputing daily rank across "
        f"{len(trading_days)} days "
        "(drives price line coloring, audit column, "
        "and the Top10 equity curve)..."
    )

    rank_maps = compute_daily_rank_maps(
        all_stocks,
        trading_days
    )

    equity_index = compute_top10_equity_curve(
        all_stocks,
        rank_maps,
        trading_days,
        TOP10_N
    )

    equity_table = build_top10_equity_table(
        equity_index,
        trading_days_window
    )

    if equity_table is None:
        print(
            "\nTop10 equity curve: skipped -- not enough history yet "
            f"for a full Top {TOP10_N} portfolio plus "
            f"{max(EQUITY_SMA_PERIODS)}-day SMA warmup. "
            "This resolves itself as more days of data accumulate."
        )
    else:
        print(
            f"\nTop10 equity curve built: "
            f"{len(equity_index)} days of history, "
            f"{len(equity_table)} days displayed."
        )

    stock_series_list = []
    skipped_symbols = []

    for i, (
        sym,
        rs,
        price,
        avg_vol
    ) in enumerate(top_ranking):

        rank = i + 1

        series_df = build_stock_series(
            all_stocks,
            sym,
            bench_close,
            trading_days_window,
            rank_maps
        )

        if series_df is None:
            skipped_symbols.append(sym)
            continue

        stock_series_list.append(
            (rank, sym, series_df)
        )

    print(
        f"\nCharted "
        f"{len(stock_series_list)}/"
        f"{len(top_ranking)} stocks "
        f"({len(skipped_symbols)} skipped "
        f"for incomplete "
        f"{RS_LINE_WINDOW}-day history): "
        f"{skipped_symbols}"
    )

    write_to_sheet(
        ranking_df,
        stock_series_list,
        skipped_symbols,
        as_of_date.strftime("%Y-%m-%d"),
        equity_table
    )

    ranking_df.to_csv(
        "RS_Top50_Ranking.csv",
        index=False
    )

    for rank, symbol, df in stock_series_list:
        df.to_csv(
            f"RS_Top50_Stock_{rank:02d}_{symbol}.csv",
            index=False
        )

    print("\nCSV files also saved.")
    print("\nSCREENER COMPLETED SUCCESSFULLY.")


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
