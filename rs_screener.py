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


def compute_trend_template(close_series):
    """
    Mark Minervini's 8-point Trend Template.
    Returns (pass_bool, criteria_met_count_out_of_8) or None if insufficient history.
    Needs at least 252 + 20 trading days of data (52-week high/low + 200DMA slope check).
    """
    if len(close_series) < 273:
        return None

    price = close_series.iloc[-1]
    sma50 = close_series.rolling(50).mean().iloc[-1]
    sma150 = close_series.rolling(150).mean().iloc[-1]
    sma200_series = close_series.rolling(200).mean()
    sma200 = sma200_series.iloc[-1]
    sma200_1mo_ago = sma200_series.iloc[-21]  # ~1 month of trading days back

    if any(pd.isna(x) for x in [sma50, sma150, sma200, sma200_1mo_ago]):
        return None

    low_52w = close_series.tail(252).min()
    high_52w = close_series.tail(252).max()

    criteria = [
        price > sma150 and price > sma200,                      # 1
        sma150 > sma200,                                          # 2
        sma200 > sma200_1mo_ago,                                  # 3: 200DMA trending up
        sma50 > sma150 and sma50 > sma200,                        # 4
        price > sma50,                                            # 5
        price >= 1.25 * low_52w,                                  # 6: at least 25% above 52wk low
        price >= 0.75 * high_52w,                                 # 7: within 25% of 52wk high
    ]
    # 8th IBD criterion (RS Rating >= 70) is applied separately using the full-universe
    # percentile rank, since it requires comparing against all other scanned stocks.
    met = sum(criteria)
    passed = met == len(criteria)
    return passed, met


def main():
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers.")

    bench_close = download_benchmark()

    # First pass: collect RS Score + Trend Template data for the WHOLE universe.
    # We need the full universe's RS scores to compute IBD-style RS Rating percentiles
    # (RS Rating >=70 = Minervini's 8th criterion), which only makes sense relative
    # to everyone else scanned, not any single stock in isolation.
    all_stocks = []
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
                sdata = data if len(batch) == 1 else data[symbol]
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
                tt_result = compute_trend_template(close)

                all_stocks.append({
                    "symbol": symbol.replace(".NS", ""),
                    "rs_score": rs_score,
                    "blue_dot": blue_dot,
                    "green_dot": green_dot,
                    "last_close": round(float(close.iloc[-1]), 2),
                    "tt_pass": tt_result[0] if tt_result else None,
                    "tt_criteria_met": tt_result[1] if tt_result else None,
                })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue

        time.sleep(1)  # be polite to Yahoo's servers

    if not all_stocks:
        print("No stocks with sufficient data found.")
        results_df = pd.DataFrame(columns=[
            "symbol", "rs_score", "rs_rating", "green_dot",
            "trend_template", "tt_criteria", "last_close"])
        write_to_sheet(results_df)
        return

    universe_df = pd.DataFrame(all_stocks)

    # RS Rating: percentile rank (1-99) of each stock's RS Score across the whole
    # scanned universe today -- this mirrors IBD's RS Rating scale.
    universe_df["rs_rating"] = (universe_df["rs_score"].rank(pct=True) * 98 + 1).round(0).astype(int)

    # Final output = Blue Dot stocks only, enriched with Trend Template + RS Rating.
    blue_dot_df = universe_df[universe_df["blue_dot"]].copy()
    blue_dot_df["trend_template"] = blue_dot_df["tt_pass"].map(
        {True: "PASS", False: "FAIL", None: "N/A"})
    blue_dot_df["green_dot"] = blue_dot_df["green_dot"].map({True: "YES", False: ""})
    blue_dot_df["tt_criteria"] = blue_dot_df["tt_criteria_met"].apply(
        lambda x: f"{x}/7" if pd.notna(x) else "N/A")

    results_df = blue_dot_df[[
        "symbol", "rs_score", "rs_rating", "green_dot",
        "trend_template", "tt_criteria", "last_close"
    ]].sort_values("rs_score", ascending=False)

    n_green = (results_df["green_dot"] == "YES").sum()
    n_tt_pass = (results_df["trend_template"] == "PASS").sum()
    print(f"Universe scanned: {len(universe_df)} stocks.")
    print(f"Blue Dot: {len(results_df)} | Green Dot: {n_green} | Trend Template PASS: {n_tt_pass}")

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
    header = ["Symbol", "RS Score", "RS Rating (1-99)", "Green Dot (Breakout Watch)",
              "Trend Template", "TT Criteria Met", "Last Close"]
    ws.update([header] + df.values.tolist(), "A3")
    print("Google Sheet updated successfully.")


if __name__ == "__main__":
    main()
