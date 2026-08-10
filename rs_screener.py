"""
RS Screener - Trend Template & RS Rank Portfolio System
Calculates IBD-style Cross-Sectional Relative Strength Percentile Ranks (1-99).

Portfolio Entry Criteria:
  - Top 10 Ranked Stocks by RS Score
  - Stock Trend Template PASS
  - RS Line Trend Template PASS

Portfolio Exit Criteria (Trigger SELL if ANY met):
  1. Hard Stop-Loss: Loss from Entry Price >= 5%
  2. Max Drawdown: Drawdown from Peak Price > 10%
  3. Rank Drop: Rank drops below Top 20
  4. RS Breakdown: RS Line falls below its 5-day EMA
  5. Structural Fail: Stock or RS Line Trend Template FAIL

Runs fully automated via GitHub Actions.
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
BENCHMARK = "^CRSLDX"          # Primary Benchmark: NIFTY 500
BENCHMARK_FALLBACK = "^NSEI"    # Secondary Benchmark: NIFTY 50
LOOKBACK_DAYS = 250              # ~52 weeks lookback
HISTORY_PERIOD = "15mo"          # Data depth for 1-year returns + calculations
STOCKS_FILE = "stocks.csv"       # Stock universe file
SHEET_ID_ENV = "SHEET_ID"        # Secret holding Sheet ID
CREDS_ENV = "GOOGLE_CREDENTIALS"# Secret holding Service Account JSON
WORKSHEET_NAME = "RS_Screener"
MIN_PRICE = 10                   # Exclude penny stocks
MIN_AVG_VOLUME = 10000           # Exclude illiquid stocks

# ---- Portfolio Risk & Exit Limits ----
ENTRY_RANK_THRESHOLD = 10        # Entry: Top 10 stocks
EXIT_RANK_THRESHOLD = 20         # Exit: Rank drops below 20
STOP_LOSS_PCT = 0.05             # Hard Floor Stop-Loss: 5% drop from entry
MAX_DRAWDOWN_PCT = 0.10          # Trailing Drawdown Exit: 10% drop from peak price

BREADTH_RISK_ON = 60             # % above 50DMA -> Risk-On
BREADTH_RISK_CAUTION = 40        # % above 50DMA -> Caution (Half size)
BREADTH_CIRCUIT_BREAKER = 25     # % above 50DMA -> Liquidate to Cash
RS_HISTORY_DAYS = 90             # RS History length for Google Sheets sparklines
INTRADAY_INTERVAL = "5m"

EOD_CRON = "15 11 * * 1-5"        # EOD Trigger schedule
HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"
RS_HISTORY_WORKSHEET = "RS_History"

PORTFOLIO_HEADER = [
    "Action", "Executed", "Execution Price", "Symbol", "Rank", "Composite Score",
    "Entry Price", "Peak Price", "Entry Date", "Current Price", "Qty", "Position Value (Rs)",
    "P&L %", "Drawdown %", "Stop-Loss", "Exit Reason", "Blue Dot", "Green Dot",
    "Trend Template", "RS Line Trend Template", "RS Line",
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
                print(f"Benchmark loaded successfully: {tkr}")
                close = data["Close"]
                if run_mode == "PREVIEW":
                    live = fetch_intraday_last_price([tkr])
                    close = append_preview_price(close, live.get(tkr))
                return close
        except Exception as e:
            print(f"Benchmark fetch failed for {tkr}: {e}")
    raise RuntimeError("Unable to download benchmark data.")


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
        print(f"Intraday price collection failed: {e}")
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


def compute_raw_rs_performance(close_series):
    """Calculates weighted 3, 6, 9, and 12-month return composite."""
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
    
    raw_score = (0.40 * returns["P3"] + 
                 0.20 * returns["P6"] + 
                 0.20 * returns["P9"] + 
                 0.20 * returns["P12"])
    return raw_score


def compute_flags_and_rs_ema(stock_close, bench_close):
    """Computes RS Ratio, Blue/Green Dots, and RS Line 5 EMA condition."""
    df = pd.concat([stock_close, bench_close], axis=1, join="inner")
    df.columns = ["stock", "bench"]
    df = df.dropna()
    if len(df) < LOOKBACK_DAYS + 2:
        return None

    df["rs_ratio"] = df["stock"] / df["bench"]
    df["rs_ema5"] = df["rs_ratio"].ewm(span=5, adjust=False).mean()
    df["rs_rolling_high"] = df["rs_ratio"].rolling(LOOKBACK_DAYS).max()
    df["price_rolling_high"] = df["stock"].rolling(LOOKBACK_DAYS).max()

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    rs_new_high_today = today["rs_ratio"] >= today["rs_rolling_high"]
    rs_was_not_high_yesterday = yesterday["rs_ratio"] < yesterday["rs_rolling_high"]
    blue_dot = bool(rs_new_high_today and rs_was_not_high_yesterday)

    price_at_new_high = today["stock"] >= today["price_rolling_high"]
    green_dot = bool(blue_dot and not price_at_new_high)

    rs_below_ema5 = bool(today["rs_ratio"] < today["rs_ema5"])
    daily_change_pct = (today["stock"] / yesterday["stock"]) - 1.0

    return blue_dot, green_dot, rs_below_ema5, daily_change_pct, df["rs_ratio"]


def compute_trend_template(series):
    """Mark Minervini 7-Point Trend Template Criteria."""
    if len(series) < 273:
        return None

    price = series.iloc[-1]
    sma50 = series.rolling(50).mean().iloc[-1]
    sma150 = series.rolling(150).mean().iloc[-1]
    sma200_series = series.rolling(200).mean()
    sma200 = sma200_series.iloc[-1]
    
    sma200_1mo_ago = sma200_series.shift(21).iloc[-1]

    if any(pd.isna(x) for x in [sma50, sma150, sma200, sma200_1mo_ago]):
        return None

    low_52w = series.tail(252).min()
    high_52w = series.tail(252).max()

    criteria = [
        price > sma150 and price > sma200,                      # 1
        sma150 > sma200,                                          # 2
        sma200 > sma200_1mo_ago,                                  # 3
        sma50 > sma150 and sma50 > sma200,                        # 4
        price > sma50,                                            # 5
        price >= 1.25 * low_52w,                                  # 6
        price >= 0.75 * high_52w,                                 # 7
    ]
    met = sum(criteria)
    passed = met == len(criteria)
    return passed, met


def main():
    tickers = load_tickers()
    print(f"Scanning universe: {len(tickers)} tickers loaded.")

    run_mode = get_run_mode()
    print(f"Run Mode: {run_mode}")

    bench_close = download_benchmark(run_mode)

    all_stocks = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period=HISTORY_PERIOD, interval="1d",
                                auto_adjust=True, progress=False, group_by="ticker",
                                threads=True)
        except Exception as e:
            print(f"Batch execution failed: {e}")
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

                raw_rs_perf = compute_raw_rs_performance(close)
                flag_data = compute_flags_and_rs_ema(close, bench_close)
                if raw_rs_perf is None or flag_data is None:
                    continue
                
                blue_dot, green_dot, rs_below_ema5, daily_change_pct, rs_ratio_full = flag_data
                tt_result = compute_trend_template(close)

                sma50_latest = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
                above_50dma = bool(close.iloc[-1] > sma50_latest) if pd.notna(sma50_latest) else None
                sma50_value = round(float(sma50_latest), 2) if pd.notna(sma50_latest) else None

                rs_tt_result = compute_trend_template(rs_ratio_full)
                rs_history = [round(float(v), 4) for v in rs_ratio_full.tail(RS_HISTORY_DAYS).tolist()]

                all_stocks.append({
                    "symbol": symbol.replace(".NS", ""),
                    "raw_rs_perf": raw_rs_perf,
                    "blue_dot": blue_dot,
                    "green_dot": green_dot,
                    "rs_below_ema5": rs_below_ema5,
                    "daily_change_pct": daily_change_pct,
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
                continue

        time.sleep(0.5)

    if not all_stocks:
        print("No eligible stock records generated.")
        return

    universe_df = pd.DataFrame(all_stocks)

    # Standard Cross-Sectional Percentile Ranking (1-99)
    universe_df["rs_score"] = (universe_df["raw_rs_perf"].rank(pct=True) * 98 + 1).round(0).astype(int)
    universe_df["rs_rating"] = universe_df["rs_score"]

    universe_df["trend_template"] = universe_df["tt_pass"].map({True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["blue_dot_label"] = universe_df["blue_dot"].map({True: "YES", False: ""})
    universe_df["green_dot_label"] = universe_df["green_dot"].map({True: "YES", False: ""})
    universe_df["tt_criteria"] = universe_df["tt_criteria_met"].apply(lambda x: f"{x}/7" if pd.notna(x) else "N/A")

    universe_df["rs_trend_template"] = universe_df["rs_tt_pass"].map({True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["rs_tt_criteria"] = universe_df["rs_tt_criteria_met"].apply(lambda x: f"{x}/7" if pd.notna(x) else "N/A")

    tt_comp = (universe_df["tt_criteria_met"].fillna(0) / 7 * 100)
    green_comp = universe_df["green_dot_label"].map({"YES": 100, "": 0})
    universe_df["composite_score"] = (0.50 * universe_df["rs_score"] + 0.30 * tt_comp + 0.20 * green_comp).round(1)

    results_df = universe_df[[
        "symbol", "composite_score", "rs_score", "rs_rating", "blue_dot_label", "green_dot_label",
        "rs_below_ema5", "daily_change_pct", "trend_template", "tt_criteria", 
        "rs_trend_template", "rs_tt_criteria", "last_close"
    ]].rename(columns={"blue_dot_label": "blue_dot", "green_dot_label": "green_dot"})
    
    results_df = results_df.sort_values("rs_score", ascending=False)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank"] + [c for c in results_df.columns if c != "rank"]]

    write_to_sheet(results_df, run_mode)

    if run_mode == "PREVIEW":
        print("Preview Run Complete. Portfolio tables unchanged.")
        return

    valid_breadth = universe_df["above_50dma"].dropna()
    breadth_pct = round(100 * valid_breadth.mean(), 1) if len(valid_breadth) else 0

    if breadth_pct >= BREADTH_RISK_ON:
        regime = "RISK-ON (Full position size)"
        allow_new_entries = True
    elif breadth_pct >= BREADTH_RISK_CAUTION:
        regime = "CAUTION (Half position size)"
        allow_new_entries = True
    elif breadth_pct >= BREADTH_CIRCUIT_BREAKER:
        regime = "RISK-OFF (No new positions)"
        allow_new_entries = False
    else:
        regime = "CIRCUIT BREAKER (Liquidate to cash)"
        allow_new_entries = False

    circuit_breaker = breadth_pct < BREADTH_CIRCUIT_BREAKER

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
        cfg_ws.update([["Setting", "Value", "Notes"], ["Total Capital (INR)", 0, "Specify total capital"]], "A1")
        settings = {"Total Capital (INR)": 0}

    try:
        capital = float(settings.get("Total Capital (INR)", 0) or 0)
    except (ValueError, TypeError):
        capital = 0
    return capital


def apply_confirmed_executions(sh):
    try:
        port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
        prior_rows = port_ws.get_all_records(head=3)
    except Exception:
        return

    try:
        holdings_ws = sh.worksheet(HOLDINGS_WORKSHEET)
        existing = holdings_ws.get_all_records()
        holdings = {
            row["symbol"]: {
                "entry_price": float(row.get("entry_price") or 0),
                "peak_price": float(row.get("peak_price") or row.get("entry_price") or 0),
                "entry_date": row.get("entry_date", ""),
                "qty": int(row.get("qty") or 0)
            }
            for row in existing if row.get("symbol")
        }
    except gspread.WorksheetNotFound:
        holdings_ws = sh.add_worksheet(title=HOLDINGS_WORKSHEET, rows=100, cols=6)
        holdings_ws.update([["symbol", "entry_price", "peak_price", "entry_date", "qty"]], "A1")
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
            exec_price = float(row.get("Execution Price") or row.get("Entry Price") or 0)
            exec_qty = int(row.get("Qty") or 0)
            if exec_price > 0 and exec_qty > 0:
                holdings[symbol] = {
                    "entry_price": exec_price, 
                    "peak_price": exec_price, 
                    "entry_date": today_str, 
                    "qty": exec_qty
                }
                changed = True

        elif action == "SELL":
            if symbol in holdings:
                del holdings[symbol]
                changed = True

    if changed:
        holdings_ws.clear()
        rows_out = [["symbol", "entry_price", "peak_price", "entry_date", "qty"]] + [
            [s, v["entry_price"], v["peak_price"], v["entry_date"], v["qty"]] for s, v in holdings.items()
        ]
        holdings_ws.update(rows_out, "A1")


def build_portfolio(ranked_df, breadth_pct, regime, allow_new_entries, sma50_lookup,
                     rs_history_lookup, circuit_breaker):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
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
            row["symbol"]: {
                "entry_price": float(row.get("entry_price") or 0),
                "peak_price": float(row.get("peak_price") or row.get("entry_price") or 0),
                "entry_date": str(row.get("entry_date", "")),
                "qty": int(row.get("qty") or 0)
            }
            for row in existing if row.get("symbol")
        }
    except gspread.WorksheetNotFound:
        current_holdings = {}

    price_lookup = dict(zip(ranked_df["symbol"], ranked_df["last_close"]))

    # Update Peak Prices for held assets
    updated_holdings_file = False
    for s, hdata in current_holdings.items():
        curr_p = price_lookup.get(s, 0)
        if curr_p > hdata["peak_price"]:
            hdata["peak_price"] = curr_p
            updated_holdings_file = True

    if updated_holdings_file and holdings_ws:
        holdings_ws.clear()
        rows_out = [["symbol", "entry_price", "peak_price", "entry_date", "qty"]] + [
            [s, v["entry_price"], v["peak_price"], v["entry_date"], v["qty"]] for s, v in current_holdings.items()
        ]
        holdings_ws.update(rows_out, "A1")

    eligible_candidates = ranked_df[
        (ranked_df["trend_template"] == "PASS") & 
        (ranked_df["rs_trend_template"] == "PASS")
    ].sort_values("rs_score", ascending=False)

    target_pool_top10 = eligible_candidates.head(ENTRY_RANK_THRESHOLD)["symbol"].tolist()

    if circuit_breaker:
        target_pool_top10 = []

    kept = []
    pending_sell = []
    exit_reasons = {}

    # ---------------- EXIT EVALUATION ENGINE ----------------
    for s, hdata in current_holdings.items():
        if circuit_breaker:
            pending_sell.append(s)
            exit_reasons[s] = "CIRCUIT BREAKER"
            continue

        r_match = ranked_df[ranked_df["symbol"] == s]
        if r_match.empty:
            pending_sell.append(s)
            exit_reasons[s] = "DATA MISSING"
            continue

        r = r_match.iloc[0]
        rank_val = int(r["rank"])
        entry_p = hdata["entry_price"]
        peak_p = hdata["peak_price"]
        curr_p = price_lookup.get(s, 0)

        # Drawdown calculation relative to highest reached price
        drawdown_pct = ((peak_p - curr_p) / peak_p) if peak_p > 0 else 0.0
        # Hard Stop Loss loss from initial entry price
        stop_loss_trigger = curr_p <= (entry_p * (1.0 - STOP_LOSS_PCT))

        # 1. Hard Floor Stop-Loss Exit (> 5% loss from entry)
        if stop_loss_trigger:
            pending_sell.append(s)
            loss_from_entry = round(((curr_p / entry_p) - 1.0) * 100, 2)
            exit_reasons[s] = f"STOP LOSS (>5% loss: {loss_from_entry}%)"

        # 2. Maximum Drawdown Exit (> 10% decline from peak)
        elif drawdown_pct > MAX_DRAWDOWN_PCT:
            pending_sell.append(s)
            exit_reasons[s] = f"DRAWDOWN EXCEEDED ({round(drawdown_pct * 100, 2)}% > 10%)"

        # 3. Rank Drop Exit (Rank > 20)
        elif rank_val > EXIT_RANK_THRESHOLD:
            pending_sell.append(s)
            exit_reasons[s] = f"RANK DROP (Rank {rank_val} > {EXIT_RANK_THRESHOLD})"
        
        # 4. Stock Price below RS Line 5 EMA
        elif r["rs_below_ema5"]:
            pending_sell.append(s)
            exit_reasons[s] = "BELOW RS 5 EMA"

        # 5. Structural Trend Template Failure
        elif r["trend_template"] != "PASS" or r["rs_trend_template"] != "PASS":
            pending_sell.append(s)
            exit_reasons[s] = "TREND TEMPLATE FAIL"

        else:
            kept.append(s)

    # Fill open positions up to top 10 limit
    open_slots = ENTRY_RANK_THRESHOLD - len(kept)
    pending_buy = []
    if allow_new_entries and open_slots > 0:
        for cand in target_pool_top10:
            if cand not in current_holdings and cand not in pending_buy:
                pending_buy.append(cand)
                if len(pending_buy) == open_slots:
                    break

    size_multiplier = 1.0 if regime.startswith("RISK-ON") else (0.5 if regime.startswith("CAUTION") else 0.0)
    slot_capital = (capital / ENTRY_RANK_THRESHOLD) * size_multiplier if capital > 0 else 0

    rows = []

    # 1. HOLD POSITIONS
    for s in kept:
        r = ranked_df[ranked_df["symbol"] == s].iloc[0]
        entry_price = current_holdings[s]["entry_price"]
        peak_price = current_holdings[s]["peak_price"]
        entry_date = current_holdings[s]["entry_date"]
        qty = current_holdings[s]["qty"]
        current_price = price_lookup.get(s, 0)
        position_value = round(qty * current_price, 0)
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
        dd_pct = round(((peak_price - current_price) / peak_price) * 100, 2) if peak_price > 0 else 0

        rows.append({
            "Action": "HOLD", "Symbol": s, "Rank": int(r["rank"]),
            "Composite Score": r["composite_score"],
            "Entry Price": entry_price, "Peak Price": peak_price, "Entry Date": entry_date, 
            "Current Price": current_price, "Qty": qty, "Position Value (Rs)": position_value, 
            "P&L %": f"{pnl_pct}%", "Drawdown %": f"{dd_pct}%",
            "Stop-Loss": round(entry_price * (1 - STOP_LOSS_PCT), 2),
            "Exit Reason": "", "Blue Dot": r["blue_dot"], "Green Dot": r["green_dot"],
            "Trend Template": r["trend_template"],
            "RS Line Trend Template": r["rs_trend_template"],
        })

    # 2. BUY POSITIONS
    for s in pending_buy:
        r = ranked_df[ranked_df["symbol"] == s].iloc[0]
        current_price = price_lookup.get(s, 0)
        position_value = round(slot_capital, 0)
        qty = int(position_value / current_price) if current_price > 0 else 0

        rows.append({
            "Action": "BUY", "Symbol": s, "Rank": int(r["rank"]),
            "Composite Score": r["composite_score"],
            "Entry Price": current_price, "Peak Price": current_price, "Entry Date": "PENDING", 
            "Current Price": current_price, "Qty": qty, "Position Value (Rs)": position_value, 
            "P&L %": "", "Drawdown %": "",
            "Stop-Loss": round(current_price * (1 - STOP_LOSS_PCT), 2),
            "Exit Reason": "", "Blue Dot": r["blue_dot"], "Green Dot": r["green_dot"],
            "Trend Template": r["trend_template"],
            "RS Line Trend Template": r["rs_trend_template"],
        })

    # 3. SELL POSITIONS
    for s in pending_sell:
        entry_price = current_holdings[s]["entry_price"]
        peak_price = current_holdings[s]["peak_price"]
        current_price = price_lookup.get(s, 0)
        qty = current_holdings[s]["qty"]
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
        dd_pct = round(((peak_price - current_price) / peak_price) * 100, 2) if peak_price > 0 else 0
        r_match = ranked_df[ranked_df["symbol"] == s]
        rank_val = int(r_match.iloc[0]["rank"]) if not r_match.empty else "N/A"

        rows.append({
            "Action": "SELL", "Symbol": s, "Rank": rank_val, "Composite Score": "",
            "Entry Price": entry_price, "Peak Price": peak_price, 
            "Entry Date": current_holdings[s]["entry_date"], "Current Price": current_price, 
            "Qty": qty, "Position Value (Rs)": round(qty * current_price, 0),
            "P&L %": f"{pnl_pct}%", "Drawdown %": f"{dd_pct}%", "Stop-Loss": "", 
            "Exit Reason": exit_reasons.get(s, "EXIT TRIGGERED"),
            "Blue Dot": "", "Green Dot": "", "Trend Template": "", "RS Line Trend Template": "",
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
    invested = sum(r.get("Position Value (Rs)", 0) for r in rows if isinstance(r.get("Position Value (Rs)"), (int, float)))
    
    summary = (f"Last updated: {timestamp}  |  Breadth: {breadth_pct}%  |  Regime: {regime}  |  "
               f"Capital: Rs.{capital:,.0f}  |  Deployed: Rs.{invested:,.0f}")
    
    port_ws.update([[summary]], "A1")
    row_lists = [[r.get(col, "") for col in PORTFOLIO_HEADER] for r in rows]
    port_ws.update([PORTFOLIO_HEADER] + row_lists, "A3")

    write_rs_history_and_sparklines(sh, rows, rs_history_lookup)


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
    label = "PREVIEW (Intraday)" if run_mode == "PREVIEW" else "EOD FINAL"
    ws.update([[f"Last updated: {timestamp}  |  {label}"]], "A1")
    header = ["Rank", "Symbol", "Composite Score", "RS Score", "RS Rating (1-99)", "Blue Dot",
              "Green Dot (Breakout Watch)", "RS Below 5 EMA", "Daily Change %", "Trend Template", 
              "TT Criteria Met", "RS Line Trend Template", "RS Line TT Criteria", "Last Close"]
    ws.update([header] + df.values.tolist(), "A3")


if __name__ == "__main__":
    main()
