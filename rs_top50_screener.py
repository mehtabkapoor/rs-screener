"""
RS TOP 50 SCREENER -- INDIVIDUAL RS LINE + PRICE CHARTS

WHAT THIS DOES (single snapshot, not a backtest)
  1. Downloads price/volume history for the stock universe + benchmark.
  2. Computes 12-Month RS Score for every eligible stock as of the
     latest available trading day.
  3. Ranks all eligible stocks by RS Score, descending (Rank 1 = strongest).
  4. Writes the full ranked table, Rank 1 downward.
  5. For EACH of the Top 50 stocks individually, builds its own
     small data table (last RS_LINE_WINDOW trading days: date, price,
     RS Line) and its own two-line chart -- Price on the left axis,
     RS Line (that stock's price / benchmark price, rebased to 100 at
     the start of the window) on the right axis. The 50 charts are
     stacked vertically down the sheet so you can scroll through them
     one stock at a time, same feel as flipping through price charts.

This is a screener, not a trading system: no positions, no capital,
no transaction costs, no rebalancing rules. Run it any day for a
fresh ranking + per-stock RS Line reading.
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

DOWNLOAD_YEARS = 2  # history needed: RS_PERIOD + buffer + RS_LINE_WINDOW

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

RS_PERIOD = 252       # 12-month RS score used for ranking
TOP_N = 50
RS_LINE_WINDOW = 50   # trading days shown in each per-stock chart

# ============================================================
# GOOGLE SHEETS
# ============================================================

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
    return list(dict.fromkeys(output))  # de-dup, preserve order


def clean_price_series(close, max_move=0.30):
    """Forward-fills single-day price spikes larger than max_move."""
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
    download_start = (pd.Timestamp.today() - pd.DateOffset(years=DOWNLOAD_YEARS)).strftime("%Y-%m-%d")
    print(f"\nBenchmark download: {download_start} -> LATEST")

    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(ticker, start=download_start, interval="1d",
                                auto_adjust=True, progress=False)
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
    if len(close) < RS_PERIOD + 20:
        return None

    avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (close > MIN_PRICE) & (avg_volume > MIN_AVG_VOLUME)

    rs_score = (close / close.shift(RS_PERIOD) - 1) * 100

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
    """Rank all eligible stocks by RS Score, descending. Rank 1 = strongest."""
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
        ranking.append((symbol, float(rs), float(price),
                         float(avg_vol) if not pd.isna(avg_vol) else 0.0))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


# ============================================================
# PER-STOCK PRICE + RS LINE SERIES
# ============================================================

def build_stock_series(all_stocks, symbol, bench_close, trading_days_window):
    """Returns a DataFrame [date, price, rs_line] for one stock over
    the window, or None if the stock is missing any day's data in
    the window. rs_line = (price / benchmark price), rebased to 100
    at the first day of the window."""

    df = all_stocks[symbol]
    prices = df["price"].reindex(trading_days_window)
    bench = bench_close.reindex(trading_days_window)

    if prices.isna().any() or bench.isna().any() or (bench <= 0).any():
        return None

    ratio = prices / bench
    base = ratio.iloc[0]
    if base <= 0 or pd.isna(base):
        return None

    rs_line = (ratio / base) * 100

    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in trading_days_window],
        "price": prices.round(2).values,
        "rs_line": rs_line.round(4).values,
    })


# ============================================================
# GOOGLE SHEETS
# ============================================================

def sanitize_for_sheets(df):
    if df.empty:
        return df
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.where(pd.notnull(clean), "")


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


def make_stock_chart(sheet_id, title, header_row_0idx, n_rows, anchor_row):
    """Two-series chart for one stock: Price (col B, index 1) on the
    left axis, RS Line (col C, index 2) on the right axis. Date (col
    A, index 0) is the domain. Column layout is fixed because every
    per-stock block is written as its own [date, price, rs_line]
    table."""

    data_end_row = header_row_0idx + 1 + n_rows

    def series(col_index, axis):
        return {
            "series": {"sourceRange": {"sources": [{
                "sheetId": sheet_id, "startRowIndex": header_row_0idx,
                "endRowIndex": data_end_row, "startColumnIndex": col_index,
                "endColumnIndex": col_index + 1,
            }]}},
            "targetAxis": axis,
        }

    return {"addChart": {"chart": {
        "spec": {
            "title": title,
            "basicChart": {
                "chartType": "LINE",
                "legendPosition": "BOTTOM_LEGEND",
                "axis": [
                    {"position": "BOTTOM_AXIS", "title": "Date"},
                    {"position": "LEFT_AXIS", "title": "Price (Rs)"},
                    {"position": "RIGHT_AXIS", "title": "RS Line (Base=100)"},
                ],
                "domains": [{"domain": {"sourceRange": {"sources": [{
                    "sheetId": sheet_id, "startRowIndex": header_row_0idx,
                    "endRowIndex": data_end_row, "startColumnIndex": 0,
                    "endColumnIndex": 1,
                }]}}}],
                "series": [
                    series(1, "LEFT_AXIS"),   # price
                    series(2, "RIGHT_AXIS"),  # rs_line
                ],
            },
        },
        "position": {"overlayPosition": {
            "anchorCell": {"sheetId": sheet_id, "rowIndex": anchor_row, "columnIndex": 5},
            "widthPixels": 750, "heightPixels": 380,
        }},
    }}}


def call_with_quota_retry(fn, label, max_retries=6, initial_wait_seconds=15):
    """Runs a Sheets API call, retrying with exponential backoff on a
    429/quota-exceeded error. Any other exception is raised immediately."""

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_quota = "429" in str(e) or "Quota exceeded" in str(e)
            if not is_quota or attempt == max_retries - 1:
                print(f"{label} failed: {e}")
                raise
            wait = initial_wait_seconds * (2 ** attempt)
            print(f"Google quota hit on {label}. Waiting {wait}s before retry...")
            time.sleep(wait)


def write_rows_in_chunks(ws, all_rows, chunk_size=1500, label="sheet write"):
    """Writes a full [row1, row2, ...] grid starting at A1 in a small
    number of large batched calls (with quota retry) instead of one
    API call per logical section. This is what keeps a 50-chart
    screener run under the Sheets 'write requests per minute' quota
    -- a handful of big calls instead of ~150+ tiny ones."""

    total = len(all_rows)
    if total == 0:
        return

    for i in range(0, total, chunk_size):
        chunk = all_rows[i:i + chunk_size]
        row_start = i + 1
        call_with_quota_retry(
            lambda c=chunk, r=row_start: ws.update(c, f"A{r}"),
            label=f"{label} rows {i}-{i + len(chunk)}",
        )
        print(f"Wrote {label}: {min(i + chunk_size, total)}/{total} rows")


def write_to_sheet(ranking_df, stock_series_list, skipped_symbols, as_of_date):
    """stock_series_list: list of (rank, symbol, df[date,price,rs_line])
    in rank order, one block per stock, each with its own chart
    stacked below the previous one for easy scrolling.

    All cell content is assembled into a single in-memory grid first
    and written in a couple of large chunked calls (write_rows_in_chunks)
    rather than one ws.update() per stock -- 50 stocks x 2 small calls
    each blew straight through the Sheets 'write requests per minute'
    quota (429 error) when this was written incrementally."""

    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- saving to CSV instead.")
        ranking_df.to_csv("RS_Top50_Ranking.csv", index=False)
        for rank, symbol, df in stock_series_list:
            df.to_csv(f"RS_Top50_Stock_{rank:02d}_{symbol}.csv", index=False)
        return

    creds = Credentials.from_service_account_info(
        json.loads(creds_json), scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    # Each per-stock block: 1 label row + 1 header row + RS_LINE_WINDOW
    # data rows + 2 blank rows gap.
    block_height = RS_LINE_WINDOW + 4
    n_rows_needed = 3 + len(ranking_df) + 3 + len(stock_series_list) * block_height + 20
    n_cols_needed = 6  # A-C used by tables; chart floats over D-onward

    ws = call_with_quota_retry(
        lambda: get_or_create_worksheet(sh, SCREENER_WORKSHEET,
                                         rows=n_rows_needed, cols=n_cols_needed),
        label="get_or_create_worksheet",
    )
    if ws.row_count < n_rows_needed or ws.col_count < n_cols_needed:
        call_with_quota_retry(
            lambda: ws.resize(rows=max(ws.row_count, n_rows_needed),
                               cols=max(ws.col_count, n_cols_needed)),
            label="resize",
        )

    call_with_quota_retry(lambda: remove_existing_charts(sh, ws.id), label="remove_existing_charts")
    call_with_quota_retry(lambda: ws.clear(), label="clear")

    # -- assemble the entire sheet as one in-memory grid --
    rows = []

    def add_row(vals=None):
        rows.append(vals if vals is not None else [])
        return len(rows)  # 1-indexed sheet row just written

    add_row([
        f"RS TOP {TOP_N} SCREENER | run {timestamp} | As of: {as_of_date} | "
        f"RS Formula: {RS_PERIOD}D Price Rate-of-Change | "
        f"Per-stock RS Line: price/benchmark rebased to 100 at day 1 of the "
        f"last {RS_LINE_WINDOW}-day window | "
        f"{len(stock_series_list)}/{TOP_N} stocks charted "
        f"({len(skipped_symbols)} skipped: incomplete {RS_LINE_WINDOW}-day history) | "
        f"Price filter > Rs.{MIN_PRICE} | Liquidity > {MIN_AVG_VOLUME:,} ({VOLUME_LOOKBACK}D avg vol)"
    ])
    add_row([])
    add_row(["Ranked Stocks (Rank 1 = Strongest RS)"])

    ranking_clean = sanitize_for_sheets(ranking_df)
    add_row(list(ranking_clean.columns))
    for r in ranking_clean.values.tolist():
        add_row(r)

    if skipped_symbols:
        add_row([f"Skipped (incomplete {RS_LINE_WINDOW}-day history): "
                 + ", ".join(skipped_symbols)])

    add_row([])
    add_row([])

    # -- per-stock blocks, stacked vertically for scrolling --
    chart_requests = []
    for rank, symbol, df in stock_series_list:
        label_row = add_row([f"Rank {rank} - {symbol}"])
        header_row = add_row(list(df.columns))
        header_row_0idx = header_row - 1  # 0-indexed for chart API

        df_clean = sanitize_for_sheets(df)
        for r in df_clean.values.tolist():
            add_row(r)

        add_row([])
        add_row([])

        chart_requests.append(make_stock_chart(
            ws.id, f"Rank {rank} - {symbol}: Price & RS Line "
                   f"(Last {RS_LINE_WINDOW} Days)",
            header_row_0idx, len(df), anchor_row=header_row_0idx,
        ))

    write_rows_in_chunks(ws, rows, chunk_size=1500, label="screener sheet")

    if chart_requests:
        # Sheets API caps batch_update request size; send in chunks
        # (with quota retry) so a large Top-50 run can't fail as one
        # giant call or die on a single transient 429.
        chunk = 10
        for i in range(0, len(chart_requests), chunk):
            batch = chart_requests[i:i + chunk]
            try:
                call_with_quota_retry(
                    lambda b=batch: sh.batch_update({"requests": b}),
                    label=f"chart batch {i + 1}-{i + len(batch)}",
                )
                print(f"Added charts {i + 1}-{min(i + chunk, len(chart_requests))} "
                      f"of {len(chart_requests)}")
            except Exception as e:
                print(f"Could not add chart batch {i + 1}-{i + len(batch)} "
                      f"(non-fatal, continuing): {e}")

    print(f"\nScreener results written to '{SCREENER_WORKSHEET}' tab: "
          f"{len(ranking_df)} ranked stocks, {len(stock_series_list)} charts.")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print(f"RS TOP {TOP_N} SCREENER -- INDIVIDUAL RS LINE + PRICE CHARTS")
    print("=" * 70)
    print(f"RS Formula     : {RS_PERIOD}-day (12M) Price Rate-of-Change")
    print(f"Ranking        : Full eligible universe, descending RS Score")
    print(f"Output         : Top {TOP_N}, Rank 1 downward")
    print(f"Per-stock chart: Price + RS Line (price/benchmark, rebased 100), "
          f"last {RS_LINE_WINDOW} trading days, one chart per stock")
    print(f"Price filter   : > Rs.{MIN_PRICE}")
    print(f"Liquidity      : {VOLUME_LOOKBACK}D average volume > {MIN_AVG_VOLUME:,}")
    print("=" * 70)

    tickers = load_tickers()
    print(f"\nLoaded {len(tickers)} tickers.")

    bench_close = download_benchmark()
    bench_close.index = normalize_dates(bench_close.index)

    all_stocks = {}
    total_bad_points = 0
    batch_size = 50
    download_start = (pd.Timestamp.today() - pd.DateOffset(years=DOWNLOAD_YEARS)).strftime("%Y-%m-%d")

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        print(f"\nDownloading {start + 1}-{start + len(batch)} of {len(tickers)}")

        try:
            data = yf.download(batch, start=download_start, interval="1d",
                                auto_adjust=True, progress=False,
                                group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch failed: {e}")
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

                stock_data = compute_stock_data(close, volume)
                if stock_data is None:
                    continue

                all_stocks[symbol.replace(".NS", "")] = stock_data

            except Exception as e:
                print(f"Skipping {symbol}: {e}")

        time.sleep(1)

    print(f"\nStocks with usable data: {len(all_stocks)}")
    print(f"Repaired data points: {total_bad_points}")

    if not all_stocks:
        raise RuntimeError("No usable stock data.")

    latest_stock_date = max(df.index.max() for df in all_stocks.values())
    as_of_date = min(pd.Timestamp(latest_stock_date).normalize(),
                      pd.Timestamp(bench_close.index.max()).normalize())
    print(f"\nAs-of date: {as_of_date:%Y-%m-%d}")

    ranking = build_ranking(all_stocks, as_of_date)
    print(f"Eligible universe: {len(ranking)} stocks")

    if not ranking:
        raise RuntimeError("No eligible stocks on the as-of date.")

    top_ranking = ranking[:TOP_N]
    ranking_df = pd.DataFrame([{
        "rank": i + 1,
        "symbol": sym,
        "rs_score_pct": round(rs, 2),
        "price": round(price, 2),
        "avg_volume_20d": round(avg_vol, 0),
    } for i, (sym, rs, price, avg_vol) in enumerate(top_ranking)])

    print(f"\nTop {len(ranking_df)} RS Stocks (Rank 1 = Strongest):")
    print(ranking_df.to_string(index=False))

    trading_days = bench_close.index[bench_close.index <= as_of_date]
    trading_days_window = trading_days[-RS_LINE_WINDOW:]
    if len(trading_days_window) < RS_LINE_WINDOW:
        print(f"\nWARNING: only {len(trading_days_window)} trading days of "
              f"benchmark history available; chart window shortened.")

    stock_series_list = []
    skipped_symbols = []
    for i, (sym, rs, price, avg_vol) in enumerate(top_ranking):
        rank = i + 1
        series_df = build_stock_series(all_stocks, sym, bench_close, trading_days_window)
        if series_df is None:
            skipped_symbols.append(sym)
            continue
        stock_series_list.append((rank, sym, series_df))

    print(f"\nCharted {len(stock_series_list)}/{len(top_ranking)} stocks "
          f"({len(skipped_symbols)} skipped for incomplete "
          f"{RS_LINE_WINDOW}-day history): {skipped_symbols}")

    write_to_sheet(ranking_df, stock_series_list, skipped_symbols,
                   as_of_date.strftime("%Y-%m-%d"))

    ranking_df.to_csv("RS_Top50_Ranking.csv", index=False)
    for rank, symbol, df in stock_series_list:
        df.to_csv(f"RS_Top50_Stock_{rank:02d}_{symbol}.csv", index=False)

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
        print(f"{type(e).__name__}: {e}")
        raise
