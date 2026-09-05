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
     and BLUE on any day it was outside the Top 10. The RS Line stays
     a single (orange) series. Boundary days are duplicated across
     both color segments so the line stays visually continuous.

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

DOWNLOAD_YEARS = 2

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
RS_LINE_COLOR = {"red": 0.95, "green": 0.55, "blue": 0.10}  # RS line (orange)
SERIES_COLORS = [GREEN_COLOR, BLUE_COLOR, RS_LINE_COLOR]

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


def compute_daily_top_sets(all_stocks, trading_days_window, top_n):
    """
    For every date in the window, rank the full eligible universe and
    return the set of symbols that were in the Top `top_n` on that date.
    Used to color-split each stock's price line (green inside, blue
    outside the Top N) over the charted window.
    """
    sets_by_date = {}

    for d in trading_days_window:
        ranking = build_ranking(all_stocks, d)
        sets_by_date[d] = {sym for sym, rs, price, vol in ranking[:top_n]}

    return sets_by_date


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
    top_sets_by_date
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

    flags = [
        symbol in top_sets_by_date.get(d, set())
        for d in trading_days_window
    ]

    top_vals, other_vals = split_by_rank(price_pct.values, flags)

    return pd.DataFrame({
        "date": [
            d.strftime("%Y-%m-%d")
            for d in trading_days_window
        ],
        "price_pct_top10": np.round(np.array(top_vals, dtype=float), 3),
        "price_pct_other": np.round(np.array(other_vals, dtype=float), 3),
        "rs_line_pct": rs_line_pct.round(3).values,
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
    as_of_date
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
        + len(stock_series_list) * block_height
        + 20
    )

    # date column + 3 data series (price_pct_top10, price_pct_other,
    # rs_line_pct) + buffer columns
    n_cols_needed = DATA_COL_START + 4 + 2

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
        f"(price/benchmark, orange), all rebased to 0% "
        f"at day 1 of the last "
        f"{RS_LINE_WINDOW}-day window | "
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
        f"\nComputing daily Top {TOP10_N} membership "
        f"across {len(trading_days_window)} days "
        "(for price line coloring)..."
    )

    top_sets_by_date = compute_daily_top_sets(
        all_stocks,
        trading_days_window,
        TOP10_N
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
            top_sets_by_date
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
        as_of_date.strftime("%Y-%m-%d")
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
