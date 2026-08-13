"""
RS Screener - FINAL SIMPLIFIED MODEL (live, daily)

Implements exactly the same rule set as backtest_final.py:

  Universe        : stocks.csv
  Price filter    : Price > Rs.20
  Liquidity       : 20-day avg volume > 100,000
  Price TT        : 7/7 required
  RS Line TT      : 7/7 required
  Ranking         : raw RS Score, descending (Rank 1 = highest RS Score)
  Portfolio       : Top 10
  Weight          : Equal weight at entry
  Rebalance       : every trading day EOD
  Blue Dot        : diagnostic only, NOT a filter
  1Y RS cross     : same signal as Blue Dot, diagnostic only
  Green Dot       : diagnostic only, NOT a filter
  RS 5-EMA exit   : REMOVED
  Price stop      : NONE
  Rank-buffer exit: NONE
  Exit rule       : leaves the current Top 10 -- nothing else
  Buy/Sell costs  : reported per position (STT, stamp duty, exchange, SEBI, GST)
  DP charge       : Rs.20 per sell transaction
  STCG            : 20.8% effective, shown as an estimate on unrealized/realized gain

No regime filter, no circuit breaker, no Blue Dot/RS-EMA/rank-buffer used as
entry or exit logic -- those are shown as diagnostic columns only.

Two-step confirm model (unchanged): a signal firing does NOT touch Holdings.
Only marking a BUY/SELL row Executed=Y after you actually trade in Kite
updates Holdings, on the next run.
"""

import time
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from datetime import datetime

# ---------------- CONFIG ----------------
BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"
LOOKBACK_DAYS = 250              # 1-year lookback for diagnostic Blue Dot / 1Y RS cross
HISTORY_PERIOD = "15mo"
STOCKS_FILE = "stocks.csv"
SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
WORKSHEET_NAME = "RS_Screener"

MIN_PRICE = 20                   # Price > Rs.20
MIN_AVG_VOLUME = 100_000         # 20-day avg volume > 100,000
VOLUME_LOOKBACK = 20

TOP_N = 10                       # target portfolio size, equal weight
INTRADAY_INTERVAL = "5m"

# ---- Transaction cost estimates (Zerodha delivery: zero brokerage) ----
STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20
STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)   # 20.8%

# Must exactly match the ONE cron expression in the workflow YAML that you
# want treated as the real end-of-day run (the only one that updates
# Portfolio/Holdings). Any other scheduled time runs as PREVIEW instead.
EOD_CRON = "15 11 * * 1-5"       # 4:45 PM IST
HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"

PORTFOLIO_HEADER = [
    "Action", "Executed", "Execution Price", "Symbol", "Rank",
    "Entry Price", "Entry Date", "Current Price", "Qty", "Position Value (Rs)",
    "P&L %", "Buy Cost (Rs)", "Sell Cost (Rs)", "Est. STCG Tax (Rs)",
    "Blue Dot", "1Y RS Cross", "Green Dot",
]
# -----------------------------------------


def get_run_mode():
    event = os.environ.get("GITHUB_EVENT_NAME", "manual")
    force_eod = os.environ.get("FORCE_EOD", "false").strip().lower() == "true"
    if event == "schedule":
        triggering_cron = os.environ.get("SCHEDULE_CRON", "").strip()
        return "EOD" if triggering_cron == EOD_CRON else "PREVIEW"
    return "EOD" if force_eod else "PREVIEW"


def load_tickers():
    df = pd.read_csv(STOCKS_FILE)
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


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


def estimate_stcg(gross_pnl):
    """Estimate only -- actual tax depends on your full-year realized gains/losses."""
    if gross_pnl <= 0:
        return 0.0
    return gross_pnl * STCG_EFFECTIVE_RATE


def download_benchmark(run_mode):
    for tkr in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(tkr, period=HISTORY_PERIOD, interval="1d",
                                auto_adjust=True, progress=False)
            if not data.empty:
                print(f"Benchmark loaded: {tkr}")
                close = data["Close"]
                if run_mode == "PREVIEW":
                    live = fetch_intraday_last_price([tkr])
                    close = append_preview_price(close, live.get(tkr))
                return close
        except Exception as e:
            print(f"Benchmark {tkr} failed: {e}")
    raise RuntimeError("Could not download any benchmark index data.")


def fetch_intraday_last_price(tickers):
    prices = {}
    try:
        data = yf.download(tickers, period="1d", interval=INTRADAY_INTERVAL,
                            progress=False, group_by="ticker", threads=True)
        for tkr in tickers:
            try:
                sdata = data if len(tickers) == 1 else data[tkr]
                last_valid = sdata["Close"].dropna()
                if not last_valid.empty:
                    prices[tkr] = float(last_valid.iloc[-1])
            except Exception:
                continue
    except Exception as e:
        print(f"Intraday batch fetch failed: {e}")
    return prices


def append_preview_price(close_series, live_price):
    if live_price is None:
        return close_series
    today = pd.Timestamp.now(tz=close_series.index.tz).normalize() if close_series.index.tz \
        else pd.Timestamp.now().normalize()
    last_date = close_series.index[-1].normalize()
    if last_date == today:
        updated = close_series.copy()
        updated.iloc[-1] = live_price
        return updated
    new_point = pd.Series([live_price], index=[today])
    return pd.concat([close_series, new_point])


def compute_rs_score(close_series):
    """40%*3mo + 20%*6mo + 20%*9mo + 20%*12mo returns."""
    n = len(close_series)
    periods = {"P3": 63, "P6": 126, "P9": 189, "P12": 252}
    returns = {}
    for label, days in periods.items():
        if n <= days:
            return None
        past = close_series.iloc[-days - 1]
        latest = close_series.iloc[-1]
        if past == 0 or pd.isna(past):
            return None
        returns[label] = (latest / past) - 1
    score = 0.40 * returns["P3"] + 0.20 * returns["P6"] + 0.20 * returns["P9"] + 0.20 * returns["P12"]
    return round(score * 100, 2)


def compute_diagnostics(stock_close, bench_close):
    """Blue Dot / 1Y RS Cross / Green Dot -- DIAGNOSTIC ONLY, not used for
    entry or exit decisions in this model."""
    df = pd.concat([stock_close, bench_close], axis=1, join="inner")
    df.columns = ["stock", "bench"]
    df = df.dropna()
    if len(df) < LOOKBACK_DAYS + 2:
        return None

    df["rs_ratio"] = df["stock"] / df["bench"]
    # Previous-only high (excludes today) -- avoids look-ahead in the signal itself
    df["rs_prev_high"] = df["rs_ratio"].shift(1).rolling(LOOKBACK_DAYS).max()
    df["price_prev_high"] = df["stock"].shift(1).rolling(LOOKBACK_DAYS).max()

    today = df.iloc[-1]
    blue_dot = bool(today["rs_ratio"] > today["rs_prev_high"]) if pd.notna(today["rs_prev_high"]) else False
    price_at_new_high = bool(today["stock"] > today["price_prev_high"]) if pd.notna(today["price_prev_high"]) else False
    green_dot = bool(blue_dot and not price_at_new_high)

    return blue_dot, green_dot   # blue_dot doubles as "1Y RS Cross"


def compute_trend_template(series):
    """Minervini's 7-point Trend Template. Returns True only if all 7 pass."""
    if len(series) < 273:
        return None
    price = series.iloc[-1]
    sma50 = series.rolling(50).mean().iloc[-1]
    sma150 = series.rolling(150).mean().iloc[-1]
    sma200_series = series.rolling(200).mean()
    sma200 = sma200_series.iloc[-1]
    sma200_1mo_ago = sma200_series.iloc[-21]

    if any(pd.isna(x) for x in [sma50, sma150, sma200, sma200_1mo_ago]):
        return None

    low_52w = series.tail(252).min()
    high_52w = series.tail(252).max()

    criteria = [
        price > sma150 and price > sma200,
        sma150 > sma200,
        sma200 > sma200_1mo_ago,
        sma50 > sma150 and sma50 > sma200,
        price > sma50,
        price >= 1.25 * low_52w,
        price >= 0.75 * high_52w,
    ]
    return sum(criteria) == 7


def main():
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers.")

    run_mode = get_run_mode()
    print(f"Run mode: {run_mode}"
          + (" (manual preview -- Holdings/Portfolio will NOT be updated)" if run_mode == "PREVIEW" else ""))

    bench_close = download_benchmark(run_mode)

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

        intraday_batch_prices = fetch_intraday_last_price(batch) if run_mode == "PREVIEW" else {}

        for symbol in batch:
            try:
                sdata = data if len(batch) == 1 else data[symbol]
                close = sdata["Close"].dropna()
                volume = sdata["Volume"].dropna()
                if close.empty or len(close) < LOOKBACK_DAYS + 2:
                    continue
                if run_mode == "PREVIEW":
                    close = append_preview_price(close, intraday_batch_prices.get(symbol))
                if close.iloc[-1] <= MIN_PRICE:
                    continue
                if volume.tail(VOLUME_LOOKBACK).mean() <= MIN_AVG_VOLUME:
                    continue

                rs_score = compute_rs_score(close)
                if rs_score is None:
                    continue

                aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
                aligned.columns = ["s", "b"]
                rs_ratio_full = aligned["s"] / aligned["b"]

                tt_pass = compute_trend_template(close)
                rs_tt_pass = compute_trend_template(rs_ratio_full)
                diagnostics = compute_diagnostics(close, bench_close)
                if diagnostics is None:
                    continue
                blue_dot, green_dot = diagnostics

                all_stocks.append({
                    "symbol": symbol.replace(".NS", ""),
                    "rs_score": rs_score,
                    "tt_pass": tt_pass,
                    "rs_tt_pass": rs_tt_pass,
                    "blue_dot": blue_dot,
                    "one_year_rs_cross": blue_dot,   # same signal, shown under both names
                    "green_dot": green_dot,
                    "last_close": round(float(close.iloc[-1]), 2),
                })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue
        time.sleep(1)

    if not all_stocks:
        print("No stocks with sufficient data found.")
        results_df = pd.DataFrame(columns=[
            "rank", "symbol", "rs_score", "tt_pass", "rs_tt_pass",
            "blue_dot", "one_year_rs_cross", "green_dot", "last_close"])
        write_to_sheet(results_df, run_mode)
        return

    universe_df = pd.DataFrame(all_stocks)
    universe_df["tt_pass_label"] = universe_df["tt_pass"].map({True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["rs_tt_pass_label"] = universe_df["rs_tt_pass"].map({True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["blue_dot_label"] = universe_df["blue_dot"].map({True: "YES", False: ""})
    universe_df["one_year_rs_cross_label"] = universe_df["one_year_rs_cross"].map({True: "YES", False: ""})
    universe_df["green_dot_label"] = universe_df["green_dot"].map({True: "YES", False: ""})

    results_df = universe_df[[
        "symbol", "rs_score", "tt_pass_label", "rs_tt_pass_label",
        "blue_dot_label", "one_year_rs_cross_label", "green_dot_label", "last_close"
    ]].rename(columns={
        "tt_pass_label": "price_trend_template", "rs_tt_pass_label": "rs_trend_template",
        "blue_dot_label": "blue_dot", "one_year_rs_cross_label": "one_year_rs_cross",
        "green_dot_label": "green_dot",
    })
    results_df = results_df.sort_values("rs_score", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank"] + [c for c in results_df.columns if c != "rank"]]

    n_tt_pass = (results_df["price_trend_template"] == "PASS").sum()
    n_rs_tt_pass = (results_df["rs_trend_template"] == "PASS").sum()
    n_both = universe_df[(universe_df["tt_pass"] == True) & (universe_df["rs_tt_pass"] == True)].shape[0]
    print(f"Universe scanned: {len(universe_df)} stocks.")
    print(f"Price TT PASS: {n_tt_pass} | RS TT PASS: {n_rs_tt_pass} | Both PASS (entry-eligible): {n_both}")

    write_to_sheet(results_df, run_mode)

    if run_mode == "PREVIEW":
        print("Preview mode: RS_Screener tab updated with live intraday prices. "
              "Holdings/Portfolio left untouched -- nothing locked in on an unsettled price.")
        return

    build_portfolio(universe_df)


def read_config(sh):
    try:
        cfg_ws = sh.worksheet(CONFIG_WORKSHEET)
        records = cfg_ws.get_all_records()
        settings = {row["Setting"]: row["Value"] for row in records if row.get("Setting")}
    except gspread.WorksheetNotFound:
        cfg_ws = sh.add_worksheet(title=CONFIG_WORKSHEET, rows=10, cols=3)
        cfg_ws.update([
            ["Setting", "Value", "Notes"],
            ["Total Capital (INR)", 0, "EDIT ME: your available investment fund for this strategy"],
        ], "A1")
        settings = {"Total Capital (INR)": 0}
    try:
        capital = float(settings.get("Total Capital (INR)", 0) or 0)
    except (ValueError, TypeError):
        capital = 0
    return capital


def apply_confirmed_executions(sh):
    """The ONLY place Holdings ever changes -- reads yesterday's Portfolio
    tab for rows marked Executed=Y and applies them now."""
    try:
        port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
    except gspread.WorksheetNotFound:
        return

    try:
        prior_rows = port_ws.get_all_records(head=3)
    except Exception as e:
        print(f"Could not read prior Portfolio tab for confirmations: {e}")
        return

    try:
        holdings_ws = sh.worksheet(HOLDINGS_WORKSHEET)
        existing = holdings_ws.get_all_records()
        holdings = {
            row["symbol"]: {"entry_price": float(row.get("entry_price") or 0),
                             "entry_date": row.get("entry_date", "")}
            for row in existing if row.get("symbol")
        }
    except gspread.WorksheetNotFound:
        holdings_ws = sh.add_worksheet(title=HOLDINGS_WORKSHEET, rows=100, cols=5)
        holdings_ws.update([["symbol", "entry_price", "entry_date"]], "A1")
        holdings = {}

    today_str = datetime.now().strftime("%Y-%m-%d")
    changed = False

    for row in prior_rows:
        executed = str(row.get("Executed", "")).strip().upper()
        if executed not in ("Y", "YES"):
            continue
        action = str(row.get("Action", "")).strip().upper()
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue

        if action == "BUY":
            exec_price_raw = row.get("Execution Price") or row.get("Entry Price")
            try:
                exec_price = float(exec_price_raw)
            except (ValueError, TypeError):
                print(f"Skipping confirmed BUY for {symbol}: no valid price found.")
                continue
            holdings[symbol] = {"entry_price": exec_price, "entry_date": today_str}
            changed = True
            print(f"Confirmed BUY applied: {symbol} @ {exec_price}")
        elif action == "SELL":
            if symbol in holdings:
                del holdings[symbol]
                changed = True
                print(f"Confirmed SELL applied: {symbol}")

    if changed:
        holdings_ws.clear()
        rows_out = [["symbol", "entry_price", "entry_date"]] + [
            [s, v["entry_price"], v["entry_date"]] for s, v in holdings.items()
        ]
        holdings_ws.update(rows_out, "A1")
        print("Holdings updated based on your confirmed executions.")
    else:
        print("No confirmed executions found since last run (nothing marked Executed=Y).")


def build_portfolio(universe_df):
    """
    ENTRY : liquid + Price TT PASS + RS Line TT PASS, ranked by RS Score,
            top 10, equal weight.
    EXIT  : leaves the current Top 10. Nothing else.
    """
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- skipping portfolio construction.")
        return

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    capital = read_config(sh)
    apply_confirmed_executions(sh)

    try:
        holdings_ws = sh.worksheet(HOLDINGS_WORKSHEET)
        existing = holdings_ws.get_all_records()
        current_holdings = {
            row["symbol"]: {"entry_price": float(row.get("entry_price") or 0),
                             "entry_date": row.get("entry_date", "")}
            for row in existing if row.get("symbol")
        }
    except gspread.WorksheetNotFound:
        holdings_ws = sh.add_worksheet(title=HOLDINGS_WORKSHEET, rows=100, cols=5)
        holdings_ws.update([["symbol", "entry_price", "entry_date"]], "A1")
        current_holdings = {}

    # ---- Today's eligible pool: Price TT PASS + RS TT PASS, ranked by RS Score ----
    pool = universe_df[(universe_df["tt_pass"] == True) & (universe_df["rs_tt_pass"] == True)].copy()
    pool = pool.sort_values("rs_score", ascending=False).reset_index(drop=True)
    pool["rank"] = range(1, len(pool) + 1)
    pool_rank_lookup = dict(zip(pool["symbol"], pool["rank"]))
    price_lookup = dict(zip(universe_df["symbol"], universe_df["last_close"]))
    diag_lookup = {row["symbol"]: row for _, row in universe_df.iterrows()}

    target_top10 = set(pool.head(TOP_N)["symbol"].tolist())

    kept = [s for s in current_holdings if s in target_top10]
    pending_sell = [s for s in current_holdings if s not in target_top10]
    slots_open = TOP_N - len(kept)
    pending_buy = [s for s in pool.head(TOP_N)["symbol"].tolist() if s not in current_holdings][:slots_open] \
        if slots_open > 0 else []

    slot_capital = (capital / TOP_N) if capital > 0 else 0

    rows = []

    for s in kept:
        entry_price = current_holdings[s]["entry_price"]
        entry_date = current_holdings[s]["entry_date"]
        current_price = price_lookup.get(s, entry_price)
        position_value = round((capital / TOP_N), 0) if capital > 0 else 0
        qty = int(position_value / entry_price) if entry_price > 0 else 0
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
        gross_pnl_rs = qty * (current_price - entry_price)
        s_cost_est = sell_side_cost(qty * current_price) if qty > 0 else 0
        tax_est = estimate_stcg(gross_pnl_rs - s_cost_est)
        diag = diag_lookup.get(s, {})

        rows.append({
            "Action": "HOLD", "Symbol": s, "Rank": pool_rank_lookup.get(s, ""),
            "Entry Price": entry_price, "Entry Date": entry_date, "Current Price": current_price,
            "Qty": qty, "Position Value (Rs)": position_value, "P&L %": f"{pnl_pct}%",
            "Buy Cost (Rs)": round(buy_side_cost(qty * entry_price), 2) if qty > 0 else 0,
            "Sell Cost (Rs)": round(s_cost_est, 2),
            "Est. STCG Tax (Rs)": round(tax_est, 2),
            "Blue Dot": diag.get("blue_dot_label", ""),
            "1Y RS Cross": diag.get("one_year_rs_cross_label", ""),
            "Green Dot": diag.get("green_dot_label", ""),
        })

    for s in pending_buy:
        current_price = price_lookup.get(s, 0)
        position_value = round(slot_capital, 0)
        qty = int(position_value / current_price) if current_price > 0 else 0
        diag = diag_lookup.get(s, {})

        rows.append({
            "Action": "BUY", "Symbol": s, "Rank": pool_rank_lookup.get(s, ""),
            "Entry Price": current_price, "Entry Date": "PENDING", "Current Price": current_price,
            "Qty": qty, "Position Value (Rs)": position_value, "P&L %": "",
            "Buy Cost (Rs)": round(buy_side_cost(qty * current_price), 2) if qty > 0 else 0,
            "Sell Cost (Rs)": "", "Est. STCG Tax (Rs)": "",
            "Blue Dot": diag.get("blue_dot_label", ""),
            "1Y RS Cross": diag.get("one_year_rs_cross_label", ""),
            "Green Dot": diag.get("green_dot_label", ""),
        })

    for s in pending_sell:
        entry_price = current_holdings[s]["entry_price"]
        current_price = price_lookup.get(s, entry_price)
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
        rank_val = pool_rank_lookup.get(s, "")

        rows.append({
            "Action": "SELL", "Symbol": s, "Rank": rank_val,
            "Entry Price": entry_price, "Entry Date": current_holdings[s]["entry_date"],
            "Current Price": current_price, "Qty": "", "Position Value (Rs)": "",
            "P&L %": f"{pnl_pct}%", "Buy Cost (Rs)": "", "Sell Cost (Rs)": "",
            "Est. STCG Tax (Rs)": "",
            "Blue Dot": "", "1Y RS Cross": "", "Green Dot": "",
        })

    n_port_rows_needed = len(rows) + 10
    n_port_cols_needed = len(PORTFOLIO_HEADER) + 2
    try:
        port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
        if port_ws.row_count < n_port_rows_needed or port_ws.col_count < n_port_cols_needed:
            port_ws.resize(rows=max(port_ws.row_count, n_port_rows_needed),
                            cols=max(port_ws.col_count, n_port_cols_needed))
    except gspread.WorksheetNotFound:
        port_ws = sh.add_worksheet(title=PORTFOLIO_WORKSHEET, rows=n_port_rows_needed, cols=n_port_cols_needed)

    port_ws.clear()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    invested = sum(r.get("Position Value (Rs)", 0) for r in rows
                    if isinstance(r.get("Position Value (Rs)"), (int, float)))
    summary = (f"Last updated: {timestamp}  |  Capital: Rs.{capital:,.0f}  |  "
               f"Deployed (approx): Rs.{invested:,.0f}  |  "
               f"Rule: Price TT + RS TT, Top {TOP_N} by RS Score, exit = leaves Top {TOP_N}  |  "
               f"Mark Executed=Y on BUY/SELL rows once you actually trade in Kite")
    if capital == 0:
        summary += "  |  SET YOUR CAPITAL in the Config tab"
    port_ws.update([[summary]], "A1")

    row_lists = [[r.get(col, "") for col in PORTFOLIO_HEADER] for r in rows]
    port_ws.update([PORTFOLIO_HEADER] + row_lists, "A3")

    print(f"Portfolio updated: {len(kept)} held, {len(pending_buy)} pending BUY, "
          f"{len(pending_sell)} pending SELL awaiting your confirmation. Capital: Rs.{capital:,.0f}")


def write_to_sheet(df, run_mode="EOD"):
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

    n_rows_needed = len(df) + 10
    n_cols_needed = len(df.columns) + 2

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
        if ws.row_count < n_rows_needed or ws.col_count < n_cols_needed:
            ws.resize(rows=max(ws.row_count, n_rows_needed), cols=max(ws.col_count, n_cols_needed))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=n_rows_needed, cols=n_cols_needed)

    ws.clear()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    label = "PREVIEW (intraday, not final)" if run_mode == "PREVIEW" else "EOD FINAL"
    ws.update([[f"Last updated: {timestamp}  |  {label}"]], "A1")
    header = ["Rank", "Symbol", "RS Score", "Price Trend Template", "RS Line Trend Template",
              "Blue Dot", "1Y RS Cross", "Green Dot", "Last Close"]
    ws.update([header] + df.values.tolist(), "A3")
    print("Google Sheet updated successfully.")


if __name__ == "__main__":
    main()

