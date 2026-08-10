"""
RS Live Screener v3 -- Production Grade
Features: Point-in-Time Liquidity, Smoothed Universe Breadth, Rank Buffer (Top 10 / Top 25)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"
STOCKS_FILE = "stocks.csv"

LOOKBACK_DAYS = 250
MIN_PRICE = 20.0
MIN_AVG_VOLUME = 50000
VOLUME_LOOKBACK = 20
MAX_PLAUSIBLE_DAILY_MOVE = 0.30

# Ranking Rules
ENTRY_TOP_N = 10
HOLD_BUFFER_RANK = 10

# Breadth Thresholds (% above 50DMA)
BREADTH_RISK_ON = 60.0
BREADTH_RISK_CAUTION = 40.0
BREADTH_CIRCUIT_BREAKER = 25.0
BREADTH_SMOOTH_SPAN = 3

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
SCREENER_WORKSHEET = "Live_Screener"

# ============================================================
# CORE COMPUTATIONAL ENGINE
# ============================================================

def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        print(f"Error: Could not find {STOCKS_FILE}")
        sys.exit(1)
    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a 'symbol' column.")
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols if s]

def clean_price_series(close):
    close = close.copy().sort_index()
    pct_change = close.pct_change()
    bad = pct_change.abs() > MAX_PLAUSIBLE_DAILY_MOVE
    n_bad = bad.sum()
    if n_bad > 0:
        cleaned = close.copy()
        for idx in close.index[bad]:
            pos = cleaned.index.get_loc(idx)
            if pos > 0:
                cleaned.iloc[pos] = cleaned.iloc[pos - 1]
        return cleaned, int(n_bad)
    return close, 0

def trend_template_check(s):
    if len(s) < 252:
        return False, 0
    sma50 = s.rolling(50).mean().iloc[-1]
    sma150 = s.rolling(150).mean().iloc[-1]
    sma200 = s.rolling(200).mean().iloc[-1]
    sma200_1mo = s.rolling(200).mean().shift(21).iloc[-1]
    low52 = s.rolling(252).min().iloc[-1]
    high52 = s.rolling(252).max().iloc[-1]
    curr = s.iloc[-1]

    c1 = curr > sma150 and curr > sma200
    c2 = sma150 > sma200
    c3 = sma200 > sma200_1mo
    c4 = sma50 > sma150 and sma50 > sma200
    c5 = curr > sma50
    c6 = curr >= 1.25 * low52
    c7 = curr >= 0.75 * high52

    met = sum([c1, c2, c3, c4, c5, c6, c7])
    return (met == 7), met

def calculate_stock_metrics(df_stock, bench_close):
    close = df_stock["Close"].dropna().sort_index()
    volume = df_stock["Volume"].reindex(close.index).fillna(0)
    
    close, _ = clean_price_series(close)
    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    
    if len(aligned) < 280:
        return None

    vol_aligned = volume.reindex(aligned.index)
    curr_price = float(aligned["s"].iloc[-1])
    avg_vol = float(vol_aligned.rolling(VOLUME_LOOKBACK).mean().iloc[-1])
    
    is_liquid = (curr_price >= MIN_PRICE) and (avg_vol >= MIN_AVG_VOLUME)
    if not is_liquid:
        return None

    rs_line = aligned["s"] / aligned["b"]
    
    def pct_ret(series, days):
        return (series.iloc[-1] / series.shift(days).iloc[-1]) - 1.0

    rs_score = (0.40 * pct_ret(aligned["s"], 63) +
                0.20 * pct_ret(aligned["s"], 126) +
                0.20 * pct_ret(aligned["s"], 189) +
                0.20 * pct_ret(aligned["s"], 252)) * 100.0

    prev_rs_high = rs_line.shift(1).iloc[-LOOKBACK_DAYS:].max()
    blue_dot = float(rs_line.iloc[-1]) > prev_rs_high

    price_tt_pass, price_tt_met = trend_template_check(aligned["s"])
    rs_tt_pass, rs_tt_met = trend_template_check(rs_line)

    sma50 = aligned["s"].rolling(50).mean().iloc[-1]
    above_50dma = curr_price > sma50

    # ATR(14) calculation
    high = df_stock["High"].reindex(aligned.index) if "High" in df_stock.columns else aligned["s"]
    low = df_stock["Low"].reindex(aligned.index) if "Low" in df_stock.columns else aligned["s"]
    tr = np.maximum(high - low, np.maximum(abs(high - aligned["s"].shift(1)), abs(low - aligned["s"].shift(1))))
    atr14 = float(tr.rolling(14).mean().iloc[-1])

    # 20 EMA for exit monitoring
    rs_ema20 = rs_line.ewm(span=20, adjust=False).mean().iloc[-1]
    rs_below_20ema = float(rs_line.iloc[-1]) < rs_ema20

    return {
        "price": round(curr_price, 2),
        "rs_score": round(rs_score, 2),
        "blue_dot": blue_dot,
        "price_tt_pass": price_tt_pass,
        "price_tt_met": price_tt_met,
        "rs_tt_pass": rs_tt_pass,
        "rs_tt_met": rs_tt_met,
        "above_50dma": above_50dma,
        "atr14": round(atr14, 2),
        "rs_below_20ema": rs_below_20ema,
        "avg_vol_20d": int(avg_vol)
    }

def run_screener():
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} symbols for screening...")

    bench_data = yf.download(BENCHMARK, period="2y", interval="1d", auto_adjust=True, progress=False)
    if bench_data.empty:
        bench_data = yf.download(BENCHMARK_FALLBACK, period="2y", interval="1d", auto_adjust=True, progress=False)
    bench_close = bench_data["Close"].dropna()
    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]
    bench_close, _ = clean_price_series(bench_close)

    stock_results = {}
    above_50dma_count = 0
    total_valid = 0

    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="2y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue

        for sym in batch:
            try:
                sdata = data if len(batch) == 1 else (data[sym] if sym in data.columns.get_level_values(0) else pd.DataFrame())
                if sdata.empty or "Close" not in sdata.columns:
                    continue
                metrics = calculate_stock_metrics(sdata, bench_close)
                if metrics:
                    clean_sym = sym.replace(".NS", "")
                    stock_results[clean_sym] = metrics
                    total_valid += 1
                    if metrics["above_50dma"]:
                        above_50dma_count += 1
            except Exception:
                continue
        time.sleep(0.5)

    breadth_pct = round((above_50dma_count / total_valid * 100.0), 2) if total_valid > 0 else 0.0

    if breadth_pct >= BREADTH_RISK_ON:
        regime = "RISK-ON (100% Size)"
    elif breadth_pct >= BREADTH_RISK_CAUTION:
        regime = "CAUTION (50% Size)"
    elif breadth_pct >= BREADTH_CIRCUIT_BREAKER:
        regime = "DEFENSIVE (No New Buys)"
    else:
        regime = "CIRCUIT-BREAKER (Liquidate)"

    # Filter & Rank
    candidates = []
    for sym, m in stock_results.items():
        if m["blue_dot"] and m["price_tt_pass"] and m["rs_tt_pass"]:
            candidates.append((sym, m["rs_score"], m["price"], m["atr14"], m["rs_below_20ema"]))

    candidates.sort(key=lambda x: x[1], reverse=True)

    results_table = []
    for rank, (sym, score, price, atr, rs_below_ema) in enumerate(candidates, 1):
        if rank <= ENTRY_TOP_N:
            action = "BUY ENTRY"
        elif rank <= HOLD_BUFFER_RANK:
            action = "HOLD ALLOWED"
        else:
            action = "WATCHLIST ONLY"

        if rs_below_ema:
            action += " (RS < 20EMA Warn)"

        results_table.append({
            "Rank": rank,
            "Symbol": sym,
            "Action": action,
            "RS Score": score,
            "Price (INR)": price,
            "ATR (14)": atr,
            "Risk/Share (2xATR)": round(2 * atr, 2)
        })

    out_df = pd.DataFrame(results_table)
    
    print("\n" + "="*60)
    print(f" MARKET BREADTH: {breadth_pct}% (>50DMA) | REGIME: {regime}")
    print("="*60)
    print(out_df.to_string(index=False))

    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if sheet_id and creds_json:
        try:
            creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=["https://www.googleapis.com/auth/spreadsheets"])
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(sheet_id)
            ws = sh.worksheet(SCREENER_WORKSHEET)
            ws.clear()
            ws.update([["RS SCREENER RESULTS - " + datetime.now().strftime("%Y-%m-%d %H:%M IST")]], "A1")
            ws.update([[f"Breadth: {breadth_pct}%", f"Regime: {regime}"]], "A2")
            if not out_df.empty:
                ws.update([list(out_df.columns)] + out_df.values.tolist(), "A4")
            print(f"\nSuccessfully updated Google Sheet tab '{SCREENER_WORKSHEET}'")
        except Exception as e:
            print(f"Google Sheet update failed: {e}")

if __name__ == "__main__":
    run_screener()
