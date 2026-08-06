"""
RS Screener - Clone of iArpanK's AmiBroker RS-Screener logic
Computes IBD-style Relative Strength Score, flags:
  - Blue Dot  : RS line made a new N-day high
  - Green Dot : RS made a new high AND price has NOT yet made a new high
                (this is the "about to breakout" precursor signal)

Runs fully free via GitHub Actions. No local machine needed after setup.
"""

import time
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import sys

# ---------------- CONFIG ----------------
BENCHMARK = "^CRSLDX"        # Yahoo Finance ticker for NIFTY 500
BENCHMARK_FALLBACK = "^NSEI"  # NIFTY 50, used if the above fails
LOOKBACK_DAYS = 250            # ~52 weeks, for RS new-high detection
HISTORY_PERIOD = "15mo"        # enough daily bars to cover 12mo return + buffer
STOCKS_FILE = "stocks.csv"     # your ticker list, one column named 'symbol'
SHEET_ID_ENV = "SHEET_ID"      # GitHub secret name holding target Sheet ID
CREDS_ENV = "GOOGLE_CREDENTIALS"  # GitHub secret name holding service account JSON
WORKSHEET_NAME = "RS_Screener"
MIN_PRICE = 10                 # filter: ignore penny stocks
MIN_AVG_VOLUME = 10000         # filter: ignore illiquid stocks
# -----------------------------------------


def load_tickers():
    """Reads your stock universe from stocks.csv (one 'symbol' column, NSE symbols without .NS)."""
    df = pd.read_csv(STOCKS_FILE)
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


def download_benchmark():
    for tkr in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(tkr, period=HISTORY_PERIOD, interval="1d",
                                auto_adjust=True, progress=False)
            if not data.empty:
                print(f"Benchmark loaded: {tkr}")
                return data["Close"]
        except Exception as e:
            print(f"Benchmark {tkr} failed: {e}")
    raise RuntimeError("Could not download any benchmark index data.")


def compute_rs_score(close_series):
    """IBD-style RS Score = 40%*P3 + 20%*P6 + 20%*P9 + 20%*P12 (trading-day approximations)."""
    n = len(close_series)
    periods = {"P3": 63, "P6": 126, "P9": 189, "P12": 252}
    returns = {}
    for label, days in periods.items():
        if n <= days:
            return None  # not enough history yet
        past = close_series.iloc[-days - 1]
        latest = close_series.iloc[-1]
        if past == 0 or pd.isna(past):
            return None
        returns[label] = (latest / past) - 1
    score = 0.40 * returns["P3"] + 0.20 * returns["P6"] + 0.20 * returns["P9"] + 0.20 * returns["P12"]
    return round(score * 100, 2)


def compute_flags(stock_close, bench_close):
    """Aligns stock and benchmark data, computes RS ratio, Blue Dot / Green Dot flags."""
    df = pd.concat([stock_close, bench_close], axis=1, join="inner")
    df.columns = ["stock", "bench"]
    df = df.dropna()
    if len(df) < LOOKBACK_DAYS + 2:
        return None

    df["rs_ratio"] = df["stock"] / df["bench"]
    df["rs_rolling_high"] = df["rs_ratio"].rolling(LOOKBACK_DAYS).max()
    df["price_rolling_high"] = df["stock"].rolling(LOOKBACK_DAYS).max()

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    rs_new_high_today = today["rs_ratio"] >= today["rs_rolling_high"]
    rs_was_not_high_yesterday = yesterday["rs_ratio"] < yesterday["rs_rolling_high"]
    blue_dot = bool(rs_new_high_today and rs_was_not_high_yesterday)

    price_at_new_high = today["stock"] >= today["price_rolling_high"]
    green_dot = bool(blue_dot and not price_at_new_high)

    return blue_dot, green_dot


def main():
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers.")

    bench_close = download_benchmark()

    results = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i}-{i+len(batch)}...")
        try:
            data = yf.download(batch, period=HISTORY_PERIOD, interval="1d",
                                auto_adjust=True, progress=False, group_by="ticker",
                                threads=True)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    sdata = data
                else:
                    sdata = data[symbol]
                close = sdata["Close"].dropna()
                volume = sdata["Volume"].dropna()
                if close.empty or len(close) < LOOKBACK_DAYS + 2:
                    continue
                if close.iloc[-1] < MIN_PRICE:
                    continue
                if volume.tail(20).mean() < MIN_AVG_VOLUME:
                    continue

                rs_score = compute_rs_score(close)
                flags = compute_flags(close, bench_close)
                if rs_score is None or flags is None:
                    continue
                blue_dot, green_dot = flags

                if blue_dot:  # only keep stocks that made an RS new high today
                    results.append({
                        "symbol": symbol.replace(".NS", ""),
                        "rs_score": rs_score,
                        "blue_dot": "YES",
                        "green_dot": "YES" if green_dot else "",
                        "last_close": round(float(close.iloc[-1]), 2),
                    })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue

        time.sleep(1)  # be polite to Yahoo's servers

    if not results:
        print("No RS new-high stocks found today.")
        results_df = pd.DataFrame(columns=["symbol", "rs_score", "blue_dot", "green_dot", "last_close"])
    else:
        results_df = pd.DataFrame(results).sort_values("rs_score", ascending=False)

    print(f"Found {len(results_df)} Blue Dot stocks ({(results_df['green_dot'] == 'YES').sum()} Green Dot).")

    write_to_sheet(results_df)


def write_to_sheet(df):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing SHEET_ID or GOOGLE_CREDENTIALS env vars — skipping Sheets write. "
              "Saving to CSV instead.")
        df.to_csv("rs_screener_output.csv", index=False)
        return

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=10)

    ws.clear()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    ws.update([[f"Last updated: {timestamp}"]], "A1")
    header = ["Symbol", "RS Score", "Blue Dot", "Green Dot (Breakout Watch)", "Last Close"]
    ws.update([header] + df.values.tolist(), "A3")
    print("Google Sheet updated successfully.")


if __name__ == "__main__":
    main()
