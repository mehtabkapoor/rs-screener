"""
RS Screener - Simplified Trend Template & RS Rank Portfolio System
Computes IBD-style Relative Strength Score and applies structural filters:
  - Stock Trend Template PASS (Minervini 7-point criteria)
  - RS Line Trend Template PASS (Relative strength structural durability)
  - Portfolio Selection: Top 10 stocks sorted by RS Score from high to low
  - Exit Condition: Position is liquidated when it falls outside the Top 10

Flags calculated & displayed (for reference only, NOT used for selection):
  - Blue Dot  : RS line made a new N-day high
  - Green Dot : RS made a new high AND price has NOT yet made a new high

Runs fully free via GitHub Actions.
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
LOOKBACK_DAYS = 250            # ~52 weeks, for trend/RS evaluation
HISTORY_PERIOD = "15mo"        # daily bars to cover 12mo return + buffer
STOCKS_FILE = "stocks.csv"     # ticker list, one column named 'symbol'
SHEET_ID_ENV = "SHEET_ID"      # GitHub secret name holding target Sheet ID
CREDS_ENV = "GOOGLE_CREDENTIALS"  # GitHub secret name holding service account JSON
WORKSHEET_NAME = "RS_Screener"
MIN_PRICE = 10                 # filter: ignore penny stocks
MIN_AVG_VOLUME = 10000         # filter: ignore illiquid stocks

# ---- Portfolio construction (quant layer) ----
TOP_N = 10                     # target portfolio size

# Selection rule: Trend Template PASS + RS Line Trend Template PASS,
# sorted high-to-low by RS Score, top 10 membership with daily rebalance exit.
STRICT_DAILY_REBALANCE = True
REQUIRE_TREND_TEMPLATE_PASS = True
REQUIRE_RS_TREND_TEMPLATE_PASS = True   
SELECTION_SORT_METRIC = "rs_score"   

BREADTH_RISK_ON = 60           # % of universe above 50DMA -> full-size new entries allowed
BREADTH_RISK_CAUTION = 40      # % above 50DMA -> half-size / caution zone
                                # below this -> no new entries, existing holdings still managed
BREADTH_CIRCUIT_BREAKER = 25   # % above 50DMA -> full defensive exit, sell everything
RS_HISTORY_DAYS = 90            # days of RS-line history to store for sparkline chart
INTRADAY_INTERVAL = "5m"        # granularity for live preview price

EOD_CRON = "15 11 * * 1-5"      # 4:45 PM IST
HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"
RS_HISTORY_WORKSHEET = "RS_History"
STOP_LOSS_PCT = 0.08           # 8% hard floor stop-loss

# Portfolio tab column layout
PORTFOLIO_HEADER = [
    "Action", "Executed", "Execution Price", "Symbol", "Rank", "Composite Score",
    "Entry Price", "Entry Date", "Current Price", "Qty", "Position Value (Rs)",
    "P&L %", "Stop-Loss", "Blue Dot", "Green Dot", "Trend Template",
    "RS Line Trend Template", "RS Line",
]
# -----------------------------------------


def get_run_mode():
    event = os.environ.get("GITHUB_EVENT_NAME", "manual")
    force_eod = os.environ.get("FORCE_EOD", "false").strip().lower() == "true"

    if event == "schedule":
        triggering_cron = os.environ.get("SCHEDULE_CRON", "").strip()
        return "EOD" if triggering_cron == EOD_CRON else "PREVIEW"
    return "EOD" if force_eod else "PREVIEW"


def col_letter(n):
    letters = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def load_tickers():
    df = pd.read_csv(STOCKS_FILE)
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


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
    """IBD-style RS Score = 40%*P3 + 20%*P6 + 20%*P9 + 20%*P12."""
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


def compute_trend_template(series):
    """
    Mark Minervini's 7-point Trend Template applied to Price or RS Line.
    Returns (pass_bool, criteria_met_count_out_of_7).
    """
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
        price > sma150 and price > sma200,                      # 1
        sma150 > sma200,                                          # 2
        sma200 > sma200_1mo_ago,                                  # 3: 200-MA trending up
        sma50 > sma150 and sma50 > sma200,                        # 4
        price > sma50,                                            # 5
        price >= 1.25 * low_52w,                                  # 6: >= 25% above 52wk low
        price >= 0.75 * high_52w,                                 # 7: within 25% of 52wk high
    ]
    met = sum(criteria)
    passed = met == len(criteria)
    return passed, met


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

                sma50_latest = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
                above_50dma = bool(close.iloc[-1] > sma50_latest) if pd.notna(sma50_latest) else None
                sma50_value = round(float(sma50_latest), 2) if pd.notna(sma50_latest) else None

                aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
                aligned.columns = ["s", "b"]
                rs_ratio_full = aligned["s"] / aligned["b"]
                rs_tt_result = compute_trend_template(rs_ratio_full)
                rs_history = [round(float(v), 4) for v in rs_ratio_full.tail(RS_HISTORY_DAYS).tolist()]

                all_stocks.append({
                    "symbol": symbol.replace(".NS", ""),
                    "rs_score": rs_score,
                    "blue_dot": blue_dot,
                    "green_dot": green_dot,
                    "last_close": round(float(close.iloc[-1]), 2),
                    "tt_pass": tt_result[0] if tt_result else None,
                    "tt_criteria_met": tt_result[1] if tt_result else None,
                    "rs_tt_pass": rs_tt_result[0] if rs_tt_result else None,
                    "rs_tt_criteria_met": rs_tt_result[1] if rs_tt_result else None,
                    "above_50dma": above_50dma,
                    "sma50": sma50_value,
                    "rs_history": rs_history,
                })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue

        time.sleep(1)

    if not all_stocks:
        print("No stocks with sufficient data found.")
        results_df = pd.DataFrame(columns=[
            "rank", "symbol", "composite_score", "rs_score", "rs_rating", "blue_dot", "green_dot",
            "trend_template", "tt_criteria", "rs_trend_template", "rs_tt_criteria", "last_close"])
        write_to_sheet(results_df, run_mode)
        return

    universe_df = pd.DataFrame(all_stocks)

    universe_df["rs_rating"] = (universe_df["rs_score"].rank(pct=True) * 98 + 1).round(0).astype(int)

    universe_df["trend_template"] = universe_df["tt_pass"].map(
        {True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["blue_dot_label"] = universe_df["blue_dot"].map({True: "YES", False: ""})
    universe_df["green_dot_label"] = universe_df["green_dot"].map({True: "YES", False: ""})
    universe_df["tt_criteria"] = universe_df["tt_criteria_met"].apply(
        lambda x: f"{x}/7" if pd.notna(x) else "N/A")

    universe_df["rs_trend_template"] = universe_df["rs_tt_pass"].map(
        {True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["rs_tt_criteria"] = universe_df["rs_tt_criteria_met"].apply(
        lambda x: f"{x}/7" if pd.notna(x) else "N/A")

    # Composite Score calculation
    tt_component = (universe_df["tt_criteria_met"].fillna(0) / 7 * 100)
    green_component = universe_df["green_dot_label"].map({"YES": 100, "": 0})
    universe_df["composite_score"] = (
        0.50 * universe_df["rs_rating"] +
        0.30 * tt_component +
        0.20 * green_component
    ).round(1)

    results_df = universe_df[[
        "symbol", "composite_score", "rs_score", "rs_rating",
        "blue_dot_label", "green_dot_label",
        "trend_template", "tt_criteria", "rs_trend_template", "rs_tt_criteria", "last_close"
    ]].rename(columns={"blue_dot_label": "blue_dot", "green_dot_label": "green_dot"})
    
    # Primary sort by RS Score from high to low
    results_df = results_df.sort_values(SELECTION_SORT_METRIC, ascending=False)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank"] + [c for c in results_df.columns if c != "rank"]]

    n_blue = (results_df["blue_dot"] == "YES").sum()
    n_green = (results_df["green_dot"] == "YES").sum()
    n_tt_pass = (results_df["trend_template"] == "PASS").sum()
    n_rs_tt_pass = (results_df["rs_trend_template"] == "PASS").sum()
    print(f"Universe scanned: {len(universe_df)} stocks.")
    print(f"Blue Dot: {n_blue} | Green Dot: {n_green} | Stock Trend Template PASS: {n_tt_pass} | RS Line Trend Template PASS: {n_rs_tt_pass}")

    write_to_sheet(results_df, run_mode)

    if run_mode == "PREVIEW":
        print("Preview mode executed. Holdings/Portfolio unmodified.")
        return

    # ---- Market Breadth Filter ----
    valid_breadth = universe_df["above_50dma"].dropna()
    breadth_pct = round(100 * valid_breadth.mean(), 1) if len(valid_breadth) else 0

    if breadth_pct >= BREADTH_RISK_ON:
        regime = "RISK-ON (full size)"
        allow_new_entries = True
    elif breadth_pct >= BREADTH_RISK_CAUTION:
        regime = "CAUTION (half size)"
        allow_new_entries = True
    elif breadth_pct >= BREADTH_CIRCUIT_BREAKER:
        regime = "RISK-OFF (no new entries)"
        allow_new_entries = False
    else:
        regime = "CIRCUIT BREAKER (reduce to cash)"
        allow_new_entries = False

    circuit_breaker = breadth_pct < BREADTH_CIRCUIT_BREAKER
    print(f"Breadth (% above 50DMA): {breadth_pct}% -> Regime: {regime}")

    # ---- Portfolio Construction ----
    sma50_lookup = dict(zip(universe_df["symbol"], universe_df["sma50"]))
    rs_history_lookup = dict(zip(universe_df["symbol"], universe_df["rs_history"]))
    build_portfolio(results_df, breadth_pct, regime, allow_new_entries, sma50_lookup,
                     rs_history_lookup, circuit_breaker)


def read_config(sh):
    try:
        cfg_ws = sh.worksheet(CONFIG_WORKSHEET)
        records = cfg_ws.get_all_records()
        settings = {row["Setting"]: row["Value"] for row in records if row.get("Setting")}
    except gspread.WorksheetNotFound:
        cfg_ws = sh.add_worksheet(title=CONFIG_WORKSHEET, rows=10, cols=3)
        cfg_ws.update([
            ["Setting", "Value", "Notes"],
            ["Total Capital (INR)", 0, "EDIT ME: total investment capital"],
        ], "A1")
        settings = {"Total Capital (INR)": 0}

    try:
        capital = float(settings.get("Total Capital (INR)", 0) or 0)
    except (ValueError, TypeError):
        capital = 0
    return capital


def apply_confirmed_executions(sh):
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

    from datetime import datetime
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
                print(f"Skipping confirmed BUY for {symbol}: invalid price.")
                continue
            holdings[symbol] = {"entry_price": exec_price, "entry_date": today_str}
            changed = True
            print(f"Confirmed BUY: {symbol} @ {exec_price}")

        elif action == "SELL":
            if symbol in holdings:
                del holdings[symbol]
                changed = True
                print(f"Confirmed SELL: {symbol}")

    if changed:
        holdings_ws.clear()
        rows_out = [["symbol", "entry_price", "entry_date"]] + [
            [s, v["entry_price"], v["entry_date"]] for s, v in holdings.items()
        ]
        holdings_ws.update(rows_out, "A1")
        print("Holdings updated.")


def compute_stop_loss(sma50_val, entry_price):
    if sma50_val:
        return round(max(sma50_val, entry_price * (1 - STOP_LOSS_PCT)), 2)
    elif entry_price:
        return round(entry_price * (1 - STOP_LOSS_PCT), 2)
    return None


def build_portfolio(ranked_df, breadth_pct, regime, allow_new_entries, sma50_lookup,
                     rs_history_lookup, circuit_breaker):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing credentials — skipping portfolio construction.")
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

    price_lookup = dict(zip(ranked_df["symbol"], ranked_df["last_close"]))

    # Target Selection: Stock TT = PASS and RS TT = PASS (NO blue_dot required), ordered high-to-low by RS Score
    pool = ranked_df
    if REQUIRE_TREND_TEMPLATE_PASS:
        pool = pool[pool["trend_template"] == "PASS"]
    if REQUIRE_RS_TREND_TEMPLATE_PASS:
        pool = pool[pool["rs_trend_template"] == "PASS"]
    
    pool = pool.sort_values(SELECTION_SORT_METRIC, ascending=False)
    target_symbols = pool.head(TOP_N)["symbol"].tolist()

    if circuit_breaker:
        target_symbols = []

    # Position Management Rules
    kept = [s for s in current_holdings if s in target_symbols]
    pending_sell = [s for s in current_holdings if s not in target_symbols]  # Exit when leaving Top 10
    pending_buy = [] if not allow_new_entries else \
        [s for s in target_symbols if s not in current_holdings]

    if regime.startswith("RISK-ON"):
        size_multiplier = 1.0
    elif regime.startswith("CAUTION"):
        size_multiplier = 0.5
    else:
        size_multiplier = 0.0
    slot_capital = (capital / TOP_N) * size_multiplier if capital > 0 else 0

    rows = []

    # HOLD Positions
    for s in kept:
        r = ranked_df[ranked_df["symbol"] == s].iloc[0]
        entry_price = current_holdings[s]["entry_price"]
        entry_date = current_holdings[s]["entry_date"]
        current_price = price_lookup.get(s, 0)
        position_value = round((capital / TOP_N), 0) if capital > 0 else 0
        qty = int(position_value / entry_price) if entry_price > 0 else 0
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0

        rows.append({
            "Action": "HOLD", "Symbol": s, "Rank": int(r["rank"]),
            "Composite Score": r["composite_score"],
            "Entry Price": entry_price, "Entry Date": entry_date, "Current Price": current_price,
            "Qty": qty, "Position Value (Rs)": position_value, "P&L %": f"{pnl_pct}%",
            "Stop-Loss": compute_stop_loss(sma50_lookup.get(s), entry_price),
            "Blue Dot": r["blue_dot"], "Green Dot": r["green_dot"],
            "Trend Template": r["trend_template"],
            "RS Line Trend Template": r["rs_trend_template"],
        })

    # BUY Suggestions
    for s in pending_buy:
        r = ranked_df[ranked_df["symbol"] == s].iloc[0]
        current_price = price_lookup.get(s, 0)
        position_value = round(slot_capital, 0)
        qty = int(position_value / current_price) if current_price > 0 else 0

        rows.append({
            "Action": "BUY", "Symbol": s, "Rank": int(r["rank"]),
            "Composite Score": r["composite_score"],
            "Entry Price": current_price, "Entry Date": "PENDING", "Current Price": current_price,
            "Qty": qty, "Position Value (Rs)": position_value, "P&L %": "",
            "Stop-Loss": compute_stop_loss(sma50_lookup.get(s), current_price),
            "Blue Dot": r["blue_dot"], "Green Dot": r["green_dot"],
            "Trend Template": r["trend_template"],
            "RS Line Trend Template": r["rs_trend_template"],
        })

    # SELL Suggestions (Dropped from Top 10 or triggered circuit breaker)
    for s in pending_sell:
        entry_price = current_holdings[s]["entry_price"]
        current_price = price_lookup.get(s, 0)
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
        r_match = ranked_df[ranked_df["symbol"] == s]
        rank_val = int(r_match.iloc[0]["rank"]) if not r_match.empty else "N/A"

        rows.append({
            "Action": "SELL", "Symbol": s, "Rank": rank_val, "Composite Score": "",
            "Entry Price": entry_price, "Entry Date": current_holdings[s]["entry_date"],
            "Current Price": current_price, "Qty": "", "Position Value (Rs)": "",
            "P&L %": f"{pnl_pct}%", "Stop-Loss": "", "Blue Dot": "", "Green Dot": "",
            "Trend Template": "", "RS Line Trend Template": "",
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
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    invested = sum(r.get("Position Value (Rs)", 0) for r in rows
                    if isinstance(r.get("Position Value (Rs)"), (int, float)))
    summary = (f"Last updated: {timestamp}  |  Breadth: {breadth_pct}%  |  Regime: {regime}  |  "
               f"Capital: Rs.{capital:,.0f}  |  Deployed (approx): Rs.{invested:,.0f}  |  "
               f"Mark Executed=Y on BUY/SELL rows once confirmed")
    if circuit_breaker:
        summary = "*** CIRCUIT BREAKER TRIGGERED: REDUCE TO CASH ***  |  " + summary
    if capital == 0:
        summary += "  |  SET YOUR CAPITAL in the Config tab"
    port_ws.update([[summary]], "A1")

    row_lists = [[r.get(col, "") for col in PORTFOLIO_HEADER] for r in rows]
    port_ws.update([PORTFOLIO_HEADER] + row_lists, "A3")

    write_rs_history_and_sparklines(sh, rows, rs_history_lookup)

    print(f"Portfolio updated: {len(kept)} held, {len(pending_buy)} pending BUY, "
          f"{len(pending_sell)} pending SELL.")


def write_rs_history_and_sparklines(sh, rows, rs_history_lookup):
    if not rows:
        return

    n_cols_needed = RS_HISTORY_DAYS + 2
    n_rows_needed = len(rows) + 5
    try:
        hist_ws = sh.worksheet(RS_HISTORY_WORKSHEET)
        if hist_ws.row_count < n_rows_needed or hist_ws.col_count < n_cols_needed:
            hist_ws.resize(rows=max(hist_ws.row_count, n_rows_needed),
                            cols=max(hist_ws.col_count, n_cols_needed))
    except gspread.WorksheetNotFound:
        hist_ws = sh.add_worksheet(title=RS_HISTORY_WORKSHEET, rows=n_rows_needed, cols=n_cols_needed)

    hist_header = ["symbol"] + [f"d-{RS_HISTORY_DAYS - 1 - i}" for i in range(RS_HISTORY_DAYS)]
    hist_rows = []
    for r in rows:
        symbol = r["Symbol"]
        history = rs_history_lookup.get(symbol, [])
        padded = [""] * (RS_HISTORY_DAYS - len(history)) + history
        hist_rows.append([symbol] + padded)

    hist_ws.clear()
    hist_ws.update([hist_header] + hist_rows, "A1")

    last_col_letter = col_letter(1 + RS_HISTORY_DAYS)
    port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
    formulas = []
    for i in range(len(rows)):
        hist_row = i + 2
        formulas.append([f"=SPARKLINE('RS_History'!B{hist_row}:{last_col_letter}{hist_row})"])

    rs_line_col = col_letter(len(PORTFOLIO_HEADER))
    first_port_row = 4
    last_port_row = first_port_row + len(rows) - 1
    port_ws.update(formulas, f"{rs_line_col}{first_port_row}:{rs_line_col}{last_port_row}",
                   value_input_option="USER_ENTERED")


def write_to_sheet(df, run_mode="EOD"):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing env vars — saving to CSV instead.")
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
            ws.resize(rows=max(ws.row_count, n_rows_needed),
                      cols=max(ws.col_count, n_cols_needed))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=n_rows_needed, cols=n_cols_needed)

    ws.clear()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    label = "PREVIEW (intraday)" if run_mode == "PREVIEW" else "EOD FINAL"
    ws.update([[f"Last updated: {timestamp}  |  {label}"]], "A1")
    header = ["Rank", "Symbol", "Composite Score", "RS Score", "RS Rating (1-99)", "Blue Dot",
              "Green Dot (Breakout Watch)", "Trend Template", "TT Criteria Met",
              "RS Line Trend Template", "RS Line TT Criteria", "Last Close"]
    ws.update([header] + df.values.tolist(), "A3")
    print("Google Sheet updated successfully.")


if __name__ == "__main__":
    main()
