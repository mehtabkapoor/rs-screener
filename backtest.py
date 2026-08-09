"""
RS Screener Backtest
Simulates the EXACT live selection rule over history:
  Blue Dot (new 250-day RS high) + price Trend Template PASS (7/7) +
  RS Line Trend Template PASS (7/7), sorted by raw RS Score, top 10,
  rebalanced daily to exact membership.

Outputs a full trade log + daily equity curve + summary stats to a
'Backtest' tab in your Google Sheet (reuses your existing SHEET_ID /
GOOGLE_CREDENTIALS secrets -- no new setup needed).

IMPORTANT CAVEATS (read before trusting the numbers):
  - No brokerage, STT, or slippage costs are modeled. Your own audit found
    charges eat a meaningful chunk of gross returns at daily rebalance
    frequency -- treat this backtest's return as a GROSS, not NET, number.
  - Equal-weight daily-reset assumption: this approximates strict daily
    rebalancing to equal weight across current holdings, not literal share
    quantities/rounding.
  - Survivorship: uses your CURRENT stocks.csv list projected backward --
    any stock that was delisted/renamed within the window is simply missing,
    which can modestly overstate returns (a known, common backtest bias).
  - Entry-day accounting: a stock bought today only starts contributing to
    portfolio return from TOMORROW onward -- its overnight move on the day
    you actually enter is correctly excluded, since you couldn't have
    captured a move that already happened before your buy decision.
  - Look-ahead: none introduced -- every day's decision uses only data up to
    and including that day.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os

# ---------------- CONFIG (mirrors rs_screener.py) ----------------
BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"
LOOKBACK_DAYS = 250
DOWNLOAD_PERIOD = "24mo"        # enough history before BACKTEST_START for 250d+21d lookbacks
STOCKS_FILE = "stocks.csv"
TOP_N = 10
BACKTEST_START = "2026-04-01"   # edit this to change the backtest window start
MIN_PRICE = 10
MIN_AVG_VOLUME = 10000
SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest"
# -------------------------------------------------------------------


def load_tickers():
    df = pd.read_csv(STOCKS_FILE)
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


def download_benchmark():
    for tkr in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(tkr, period=DOWNLOAD_PERIOD, interval="1d",
                                auto_adjust=True, progress=False)
            if not data.empty:
                print(f"Benchmark loaded: {tkr}")
                return data["Close"]
        except Exception as e:
            print(f"Benchmark {tkr} failed: {e}")
    raise RuntimeError("Could not download any benchmark index data.")


def trend_template_series(s):
    """Vectorized Minervini 7-criteria Trend Template over an entire series at once."""
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
    passed = met == 7
    return passed


def compute_signals_for_stock(close, bench_close):
    """Returns a per-date DataFrame: price, rs_score, blue_dot, tt_pass, rs_tt_pass."""
    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) < 280:
        return None

    rs_ratio = aligned["s"] / aligned["b"]

    def pct_return(series, days):
        return series / series.shift(days) - 1

    rs_score = (0.40 * pct_return(aligned["s"], 63)
                + 0.20 * pct_return(aligned["s"], 126)
                + 0.20 * pct_return(aligned["s"], 189)
                + 0.20 * pct_return(aligned["s"], 252)) * 100

    rs_roll_high = rs_ratio.rolling(LOOKBACK_DAYS).max()
    blue_dot = (rs_ratio >= rs_roll_high) & (rs_ratio.shift(1) < rs_roll_high.shift(1))

    tt_pass = trend_template_series(aligned["s"])
    rs_tt_pass = trend_template_series(rs_ratio)

    return pd.DataFrame({
        "price": aligned["s"],
        "rs_score": rs_score,
        "blue_dot": blue_dot,
        "tt_pass": tt_pass,
        "rs_tt_pass": rs_tt_pass,
    })


def run_backtest():
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers. Backtesting from {BACKTEST_START} to today.")

    bench_close = download_benchmark()
    all_signals = {}

    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i}-{i+len(batch)}...")
        try:
            data = yf.download(batch, period=DOWNLOAD_PERIOD, interval="1d",
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
                if close.empty or len(close) < 280:
                    continue
                if close.iloc[-1] < MIN_PRICE:
                    continue
                if volume.tail(20).mean() < MIN_AVG_VOLUME:
                    continue
                sig = compute_signals_for_stock(close, bench_close)
                if sig is not None:
                    all_signals[symbol.replace(".NS", "")] = sig
            except Exception:
                continue

        time.sleep(1)

    print(f"Signals computed for {len(all_signals)} stocks with sufficient history.")

    trading_days = bench_close.index[bench_close.index >= pd.Timestamp(BACKTEST_START)]
    if len(trading_days) == 0:
        print("No trading days found in the backtest window.")
        return pd.DataFrame(), pd.DataFrame()

    holdings = {}       # symbol -> {"entry_price", "entry_date"}
    trade_log = []
    equity = 1.0
    equity_curve = []

    for date in trading_days:
        pool = []
        for sym, df in all_signals.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["rs_score"]):
                continue
            if bool(row["blue_dot"]) and bool(row["tt_pass"]) and bool(row["rs_tt_pass"]):
                pool.append((sym, float(row["rs_score"]), float(row["price"])))

        pool.sort(key=lambda x: x[1], reverse=True)
        target = pool[:TOP_N]
        target_syms = {s for s, _, _ in target}
        target_prices = {s: p for s, _, p in target}

        # Snapshot what was actually held BEFORE today's decisions -- only these
        # stocks' overnight move counts toward today's return. A stock bought
        # today (at today's close) can only start contributing from tomorrow;
        # crediting it with today's already-happened move would be look-ahead.
        held_before_today = set(holdings.keys())

        # Daily portfolio return: equal-weight average of yesterday's holdings' return
        if held_before_today:
            rets = []
            for sym in held_before_today:
                df = all_signals[sym]
                if date in df.index:
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev_price = df["price"].iloc[idx - 1]
                        curr_price = df["price"].iloc[idx]
                        if prev_price > 0:
                            rets.append(curr_price / prev_price - 1)
            if rets:
                equity *= (1 + float(np.mean(rets)))

        # Exit anything no longer in target (at today's close)
        for sym in list(holdings.keys()):
            if sym not in target_syms:
                entry = holdings.pop(sym)
                df = all_signals[sym]
                exit_price = float(df.loc[date, "price"]) if date in df.index else entry["entry_price"]
                ret = (exit_price / entry["entry_price"] - 1) * 100
                trade_log.append({
                    "symbol": sym, "entry_date": entry["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(entry["entry_price"], 2), "exit_price": round(exit_price, 2),
                    "return_pct": round(ret, 2),
                    "days_held": (date - entry["entry_date"]).days,
                })

        # Enter new target names (at today's close -- starts contributing tomorrow)
        for sym in target_syms:
            if sym not in holdings:
                holdings[sym] = {"entry_price": target_prices[sym], "entry_date": date}

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(equity, 4),
                              "n_holdings": len(holdings)})

    # Close any still-open positions at the final backtest date
    last_date = trading_days[-1]
    for sym, entry in holdings.items():
        df = all_signals[sym]
        exit_price = float(df.loc[last_date, "price"]) if last_date in df.index else entry["entry_price"]
        ret = (exit_price / entry["entry_price"] - 1) * 100
        trade_log.append({
            "symbol": sym, "entry_date": entry["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": last_date.strftime("%Y-%m-%d") + " (OPEN)",
            "entry_price": round(entry["entry_price"], 2), "exit_price": round(exit_price, 2),
            "return_pct": round(ret, 2),
            "days_held": (last_date - entry["entry_date"]).days,
        })

    trade_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity_curve)
    return trade_df, equity_df


def compute_summary(trade_df, equity_df):
    if equity_df.empty:
        return {}
    total_return_pct = round((equity_df["equity"].iloc[-1] - 1) * 100, 2)
    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] / running_max - 1) * 100
    max_dd = round(drawdown.min(), 2)

    closed_trades = trade_df[~trade_df["exit_date"].astype(str).str.contains("OPEN", na=False)] \
        if not trade_df.empty else trade_df
    n_trades = len(closed_trades)
    win_rate = round((closed_trades["return_pct"] > 0).mean() * 100, 1) if n_trades else 0
    avg_return = round(closed_trades["return_pct"].mean(), 2) if n_trades else 0
    avg_days_held = round(closed_trades["days_held"].mean(), 1) if n_trades else 0
    best_trade = closed_trades["return_pct"].max() if n_trades else 0
    worst_trade = closed_trades["return_pct"].min() if n_trades else 0

    return {
        "Total Return (gross, no costs)": f"{total_return_pct}%",
        "Max Drawdown": f"{max_dd}%",
        "Number of Closed Trades": n_trades,
        "Win Rate": f"{win_rate}%",
        "Avg Return per Trade": f"{avg_return}%",
        "Avg Days Held": avg_days_held,
        "Best Trade": f"{best_trade}%",
        "Worst Trade": f"{worst_trade}%",
        "Backtest Window": f"{BACKTEST_START} to {equity_df['date'].iloc[-1]}",
    }


def write_to_sheet(trade_df, equity_df, summary):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- saving to CSV instead.")
        trade_df.to_csv("backtest_trades.csv", index=False)
        equity_df.to_csv("backtest_equity.csv", index=False)
        return

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    n_rows_needed = len(trade_df) + len(equity_df) + 30
    try:
        ws = sh.worksheet(BACKTEST_WORKSHEET)
        if ws.row_count < n_rows_needed:
            ws.resize(rows=n_rows_needed, cols=10)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=BACKTEST_WORKSHEET, rows=n_rows_needed, cols=10)

    ws.clear()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    ws.update([[f"Backtest run: {timestamp}  |  GROSS returns, no brokerage/STT/slippage modeled"]], "A1")

    summary_rows = [["Summary", ""]] + [[k, v] for k, v in summary.items()]
    ws.update(summary_rows, "A3")

    trade_start_row = 3 + len(summary_rows) + 2
    ws.update([["Trade Log"]], f"A{trade_start_row}")
    trade_header_row = trade_start_row + 1
    if not trade_df.empty:
        ws.update([list(trade_df.columns)] + trade_df.values.tolist(), f"A{trade_header_row}")

    print(f"Backtest results written to '{BACKTEST_WORKSHEET}' tab: "
          f"{len(trade_df)} trades, {len(equity_df)} trading days.")


if __name__ == "__main__":
    trades, equity = run_backtest()
    summary = compute_summary(trades, equity)
    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    write_to_sheet(trades, equity, summary)
