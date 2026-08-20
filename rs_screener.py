"""
RS SCREENER - MINERVINI TREND TEMPLATE + RS   (CORRECTED)

============================================================
CORE SCREEN  (RULES UNCHANGED FROM ORIGINAL)
============================================================

PRICE
------
Price > Rs.20

LIQUIDITY
---------
20-day average volume > 100,000 shares

PRICE TREND TEMPLATE
--------------------
7/7 Minervini Trend Template

RS
--
RS Score:
    40% 3-month return
    20% 6-month return
    20% 9-month return
    20% 12-month return

RS Line Trend Template
----------------------
Same 7/7 Trend Template logic applied to
relative-strength line = Stock / Benchmark

RANKING
-------
Eligible stocks ranked by raw RS Score descending.

PORTFOLIO
---------
Top 10
Equal weight
Daily EOD rebalance

EXIT
----
Sell when RS rank drops greater than 15
(i.e. rank > 15 among the current eligible universe).

NO:
----
VCP
Volume dry-up
Pivot / breakout volume
Price stop
Blue Dot entry
Green Dot entry
Regime filter

Blue Dot / 1Y RS Cross / Green Dot remain diagnostic.

============================================================
ERROR FIXES IN THIS VERSION (no rule/logic changes)
============================================================

1. QTY PERSISTENCE BUG FIXED:
   OLD: apply_confirmed_executions() stored only entry_price and
        entry_date on a confirmed BUY — never the executed qty.
        build_portfolio() then RECOMPUTED qty every run from
        today's (capital / TOP_N) / entry_price, so "Position
        Value (Rs)" for HOLD rows was a target allocation, not
        true mark-to-market, and silently drifted from what was
        actually bought whenever capital changed or your fill
        differed from the suggested qty.
   NEW: Holdings sheet gains a "qty" column. Confirmed BUY rows
        store the executed qty (from the Portfolio tab's Qty
        column). build_portfolio() uses that stored qty directly:
            Position Value (Rs) = qty * current_price
        No more silent re-derivation.

2. DATE/TIMEZONE NORMALIZATION ADDED:
   OLD: No normalization of yfinance's returned index. If a
        stock's index and the benchmark's index differ in
        tz-awareness, pd.concat(..., join="inner") can silently
        return an empty/reduced intersection. compute_diagnostics()
        would then return None, and the stock would be silently
        DROPPED from the entire screen with no error logged.
   NEW: All price/volume series are normalized to tz-naive,
        midnight-normalized timestamps before any join/concat,
        matching the safeguard already used in the backtest files.
"""

import time
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

LOOKBACK_DAYS = 250
HISTORY_PERIOD = "15mo"

STOCKS_FILE = "stocks.csv"

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

WORKSHEET_NAME = "RS_Screener"

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

TOP_N = 10
EXIT_RANK = 15          # sell when rank > EXIT_RANK
INTRADAY_INTERVAL = "5m"

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20
STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

EOD_CRON = "15 11 * * 1-5"
HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"

PORTFOLIO_HEADER = [
    "Action", "Executed", "Execution Price", "Symbol", "Rank",
    "Entry Price", "Entry Date", "Current Price", "Qty", "Position Value (Rs)",
    "P&L %", "Buy Cost (Rs)", "Sell Cost (Rs)", "Est. STCG Tax (Rs)",
    "Blue Dot", "1Y RS Cross", "Green Dot",
]

HOLDINGS_HEADER = ["symbol", "entry_price", "entry_date", "qty"]


# ============================================================
# DATE NORMALIZATION  (NEW)
# ============================================================

def normalize_dates(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def normalize_series_index(series):
    s = series.copy()
    s.index = normalize_dates(s.index)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


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
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = normalize_series_index(close.dropna())
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
    # close_series is already tz-naive normalized midnight timestamps
    today = pd.Timestamp.now().normalize()
    last_date = close_series.index[-1].normalize()
    if last_date == today:
        updated = close_series.copy()
        updated.iloc[-1] = live_price
        return updated
    new_point = pd.Series([live_price], index=[today])
    return pd.concat([close_series, new_point])


def compute_rs_score(close_series):
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


def compute_trend_template(series):
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


def compute_diagnostics(stock_close, bench_close):
    df = pd.concat([stock_close, bench_close], axis=1, join="inner")
    df.columns = ["stock", "bench"]
    df = df.dropna()
    if len(df) < LOOKBACK_DAYS + 2:
        return None
    df["rs_ratio"] = df["stock"] / df["bench"]
    df["rs_prev_high"] = df["rs_ratio"].shift(1).rolling(LOOKBACK_DAYS).max()
    df["price_prev_high"] = df["stock"].shift(1).rolling(LOOKBACK_DAYS).max()
    today = df.iloc[-1]
    blue_dot = bool(today["rs_ratio"] > today["rs_prev_high"]) if pd.notna(today["rs_prev_high"]) else False
    price_at_new_high = bool(today["stock"] > today["price_prev_high"]) if pd.notna(today["price_prev_high"]) else False
    green_dot = blue_dot and not price_at_new_high
    return (blue_dot, green_dot)


def main():
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers.")
    run_mode = get_run_mode()
    print(f"Run mode: {run_mode}")
    bench_close = download_benchmark(run_mode)

    all_stocks = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i}-{i + len(batch)}...")
        try:
            data = yf.download(batch, period=HISTORY_PERIOD, interval="1d",
                                auto_adjust=True, progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue

        intraday_batch_prices = fetch_intraday_last_price(batch) if run_mode == "PREVIEW" else {}

        for symbol in batch:
            try:
                sdata = data if len(batch) == 1 else data[symbol]
                close = normalize_series_index(sdata["Close"].dropna())
                volume = normalize_series_index(sdata["Volume"].dropna())
                if close.empty or len(close) < LOOKBACK_DAYS + 2:
                    continue
                if run_mode == "PREVIEW":
                    close = append_preview_price(close, intraday_batch_prices.get(symbol))

                last_price = float(close.iloc[-1])
                if last_price <= MIN_PRICE:
                    continue

                avg20_volume = volume.tail(VOLUME_LOOKBACK).mean()
                if pd.isna(avg20_volume) or avg20_volume <= MIN_AVG_VOLUME:
                    continue

                rs_score = compute_rs_score(close)
                if rs_score is None:
                    continue

                tt_pass = compute_trend_template(close)

                aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
                aligned.columns = ["s", "b"]
                rs_ratio_full = aligned["s"] / aligned["b"]
                rs_tt_pass = compute_trend_template(rs_ratio_full)

                diagnostics = compute_diagnostics(close, bench_close)
                if diagnostics is None:
                    continue
                blue_dot, green_dot = diagnostics

                screen_pass = (tt_pass is True and rs_tt_pass is True)

                all_stocks.append({
                    "symbol": symbol.replace(".NS", ""),
                    "rs_score": rs_score,
                    "last_close": round(last_price, 2),
                    "avg20_volume": round(float(avg20_volume), 0),
                    "tt_pass": tt_pass,
                    "rs_tt_pass": rs_tt_pass,
                    "screen_pass": screen_pass,
                    "blue_dot": blue_dot,
                    "one_year_rs_cross": blue_dot,
                    "green_dot": green_dot,
                })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue
        time.sleep(1)

    if not all_stocks:
        print("No stocks with sufficient data found.")
        results_df = pd.DataFrame()
        write_to_sheet(results_df, run_mode)
        return

    universe_df = pd.DataFrame(all_stocks)

    universe_df["tt_pass_label"] = universe_df["tt_pass"].map({True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["rs_tt_pass_label"] = universe_df["rs_tt_pass"].map({True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["screen_label"] = universe_df["screen_pass"].map({True: "PASS", False: "FAIL"})
    universe_df["blue_dot_label"] = universe_df["blue_dot"].map({True: "YES", False: ""})
    universe_df["one_year_rs_cross_label"] = universe_df["one_year_rs_cross"].map({True: "YES", False: ""})
    universe_df["green_dot_label"] = universe_df["green_dot"].map({True: "YES", False: ""})

    results_df = universe_df[[
        "symbol", "rs_score", "last_close", "avg20_volume",
        "tt_pass_label", "rs_tt_pass_label",
        "screen_label",
        "blue_dot_label", "one_year_rs_cross_label", "green_dot_label",
    ]].rename(columns={
        "tt_pass_label": "price_trend_template",
        "rs_tt_pass_label": "rs_trend_template",
        "screen_label": "screen",
        "blue_dot_label": "blue_dot",
        "one_year_rs_cross_label": "one_year_rs_cross",
        "green_dot_label": "green_dot",
    })

    results_df = results_df.sort_values("rs_score", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank"] + [c for c in results_df.columns if c != "rank"]]

    n_tt_pass = (universe_df["tt_pass"] == True).sum()
    n_rs_tt_pass = (universe_df["rs_tt_pass"] == True).sum()
    n_screen = (universe_df["screen_pass"] == True).sum()
    n_both_tt = universe_df[(universe_df["tt_pass"] == True) & (universe_df["rs_tt_pass"] == True)].shape[0]

    print("========================================")
    print(f"Universe scanned: {len(universe_df)}")
    print(f"Price TT PASS: {n_tt_pass}")
    print(f"RS TT PASS: {n_rs_tt_pass}")
    print(f"Both TT PASS: {n_both_tt}")
    print(f"FINAL SCREEN PASS: {n_screen}")
    print("========================================")

    write_to_sheet(results_df, run_mode)

    if run_mode == "PREVIEW":
        print("Preview mode: RS_Screener updated. Holdings/Portfolio untouched.")
        return

    build_portfolio(universe_df)


def read_config(sh):
    try:
        cfg_ws = sh.worksheet(CONFIG_WORKSHEET)
        records = cfg_ws.get_all_records()
        settings = {row["Setting"]: row["Value"] for row in records if row.get("Setting")}
    except gspread.WorksheetNotFound:
        cfg_ws = sh.add_worksheet(title=CONFIG_WORKSHEET, rows=10, cols=3)
        cfg_ws.update([["Setting", "Value", "Notes"], ["Total Capital (INR)", 0, "EDIT ME"]], "A1")
        settings = {"Total Capital (INR)": 0}
    try:
        capital = float(settings.get("Total Capital (INR)", 0) or 0)
    except (ValueError, TypeError):
        capital = 0
    return capital


def apply_confirmed_executions(sh):
    """
    FIXED: confirmed BUY rows now persist the executed qty
    (read from the Portfolio tab's Qty column) into the Holdings
    sheet, instead of storing only entry_price/entry_date and
    letting build_portfolio() silently re-derive qty later.
    """
    try:
        port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
    except gspread.WorksheetNotFound:
        return
    try:
        prior_rows = port_ws.get_all_records(head=3)
    except Exception as e:
        print(f"Could not read prior Portfolio tab: {e}")
        return

    try:
        holdings_ws = sh.worksheet(HOLDINGS_WORKSHEET)
        existing = holdings_ws.get_all_records()
        holdings = {
            row["symbol"]: {
                "entry_price": float(row.get("entry_price") or 0),
                "entry_date": row.get("entry_date", ""),
                "qty": int(row.get("qty") or 0),
            }
            for row in existing if row.get("symbol")
        }
    except gspread.WorksheetNotFound:
        holdings_ws = sh.add_worksheet(title=HOLDINGS_WORKSHEET, rows=100, cols=5)
        holdings_ws.update([HOLDINGS_HEADER], "A1")
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
            exec_qty_raw = row.get("Qty")
            try:
                exec_price = float(exec_price_raw)
                exec_qty = int(exec_qty_raw)
            except (ValueError, TypeError):
                print(f"Skipping confirmed BUY for {symbol}: invalid price/qty in row "
                      f"(Execution Price={exec_price_raw}, Qty={exec_qty_raw})")
                continue
            if exec_qty <= 0:
                print(f"Skipping confirmed BUY for {symbol}: qty must be > 0, got {exec_qty}")
                continue
            holdings[symbol] = {"entry_price": exec_price, "entry_date": today_str, "qty": exec_qty}
            changed = True
            print(f"Confirmed BUY applied: {symbol} @ {exec_price} x {exec_qty}")

        elif action == "SELL":
            if symbol in holdings:
                del holdings[symbol]
                changed = True
                print(f"Confirmed SELL applied: {symbol}")

    if changed:
        holdings_ws.clear()
        rows_out = [HOLDINGS_HEADER] + [
            [s, v["entry_price"], v["entry_date"], v["qty"]] for s, v in holdings.items()
        ]
        holdings_ws.update(rows_out, "A1")
        print("Holdings updated.")
    else:
        print("No confirmed executions.")


def build_portfolio(universe_df):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS.")
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
                "entry_date": row.get("entry_date", ""),
                "qty": int(row.get("qty") or 0),
            }
            for row in existing if row.get("symbol")
        }
    except gspread.WorksheetNotFound:
        holdings_ws = sh.add_worksheet(title=HOLDINGS_WORKSHEET, rows=100, cols=5)
        holdings_ws.update([HOLDINGS_HEADER], "A1")
        current_holdings = {}

    # Eligible universe = Price TT + RS TT
    pool = universe_df[
        (universe_df["tt_pass"] == True) & (universe_df["rs_tt_pass"] == True)
    ].copy()

    pool = pool.sort_values("rs_score", ascending=False).reset_index(drop=True)
    pool["rank"] = range(1, len(pool) + 1)
    pool_rank_lookup = dict(zip(pool["symbol"], pool["rank"]))
    price_lookup = dict(zip(universe_df["symbol"], universe_df["last_close"]))
    diag_lookup = {row["symbol"]: row for _, row in universe_df.iterrows()}

    # Target buys = Top N
    target_topN = set(pool.head(TOP_N)["symbol"].tolist())

    # Keep existing holdings that still rank <= EXIT_RANK
    kept = []
    pending_sell = []
    for s in current_holdings:
        rank = pool_rank_lookup.get(s)
        if rank is not None and rank <= EXIT_RANK:
            kept.append(s)
        else:
            pending_sell.append(s)

    slots_open = TOP_N - len(kept)
    pending_buy = ([s for s in pool.head(TOP_N)["symbol"].tolist() if s not in current_holdings][:slots_open]
                    if slots_open > 0 else [])
    slot_capital = capital / TOP_N if capital > 0 else 0

    rows = []

    for s in kept:
        entry_price = current_holdings[s]["entry_price"]
        entry_date = current_holdings[s]["entry_date"]
        qty = current_holdings[s]["qty"]  # FIXED: use actual executed qty, not recomputed
        current_price = price_lookup.get(s, entry_price)
        position_value = round(qty * current_price, 0)  # FIXED: true mark-to-market
        pnl_pct = round((current_price / entry_price - 1) * 100, 2) if entry_price > 0 else 0
        gross_pnl_rs = qty * (current_price - entry_price)
        s_cost_est = sell_side_cost(qty * current_price) if qty > 0 else 0
        tax_est = estimate_stcg(gross_pnl_rs - s_cost_est)
        diag = diag_lookup.get(s, {})
        rows.append({
            "Action": "HOLD", "Symbol": s, "Rank": pool_rank_lookup.get(s, ""),
            "Entry Price": entry_price, "Entry Date": entry_date, "Current Price": current_price,
            "Qty": qty, "Position Value (Rs)": position_value, "P&L %": f"{pnl_pct}%",
            "Buy Cost (Rs)": round(buy_side_cost(qty * entry_price), 2) if qty > 0 else 0,
            "Sell Cost (Rs)": round(s_cost_est, 2), "Est. STCG Tax (Rs)": round(tax_est, 2),
            "Blue Dot": ("YES" if diag.get("blue_dot", False) else ""),
            "1Y RS Cross": ("YES" if diag.get("one_year_rs_cross", False) else ""),
            "Green Dot": ("YES" if diag.get("green_dot", False) else ""),
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
            "Blue Dot": ("YES" if diag.get("blue_dot", False) else ""),
            "1Y RS Cross": ("YES" if diag.get("one_year_rs_cross", False) else ""),
            "Green Dot": ("YES" if diag.get("green_dot", False) else ""),
        })

    for s in pending_sell:
        entry_price = current_holdings[s]["entry_price"]
        qty = current_holdings[s]["qty"]  # FIXED: use actual executed qty
        current_price = price_lookup.get(s, entry_price)
        pnl_pct = round((current_price / entry_price - 1) * 100, 2) if entry_price > 0 else 0
        rank_val = pool_rank_lookup.get(s, "")
        gross_pnl_rs = qty * (current_price - entry_price)
        s_cost_est = sell_side_cost(qty * current_price) if qty > 0 else 0
        tax_est = estimate_stcg(gross_pnl_rs - s_cost_est)
        rows.append({
            "Action": "SELL", "Symbol": s, "Rank": rank_val,
            "Entry Price": entry_price, "Entry Date": current_holdings[s]["entry_date"],
            "Current Price": current_price, "Qty": qty,
            "Position Value (Rs)": round(qty * current_price, 0),
            "P&L %": f"{pnl_pct}%", "Buy Cost (Rs)": "",
            "Sell Cost (Rs)": round(s_cost_est, 2),
            "Est. STCG Tax (Rs)": round(tax_est, 2),
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
    summary = (f"Last updated: {timestamp} | Capital: Rs.{capital:,.0f} | Deployed: Rs.{invested:,.0f} | "
               f"Entry = Price TT + RS TT | Top {TOP_N} by RS Score | "
               f"Exit = rank > {EXIT_RANK}")
    if capital == 0:
        summary += " | SET CAPITAL IN CONFIG"
    port_ws.update([[summary]], "A1")

    row_lists = [[r.get(col, "") for col in PORTFOLIO_HEADER] for r in rows]
    port_ws.update([PORTFOLIO_HEADER] + row_lists, "A3")

    print(f"Portfolio updated: {len(kept)} held, {len(pending_buy)} BUY, "
          f"{len(pending_sell)} SELL. Capital: Rs.{capital:,.0f}")


def write_to_sheet(df, run_mode="EOD"):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)
    if not sheet_id or not creds_json:
        print("Missing SHEET_ID or GOOGLE_CREDENTIALS. Saving CSV instead.")
        df.to_csv("rs_screener_output.csv", index=False)
        return

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    n_rows_needed = len(df) + 10
    n_cols_needed = len(df.columns) + 2 if len(df.columns) else 5

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
    ws.update([[f"Last updated: {timestamp} | {label} | Price TT + RS TT | Exit rank > {EXIT_RANK}"]], "A1")

    header = list(df.columns)
    if header:
        ws.update([header] + df.fillna("").values.tolist(), "A3")

    print("Google Sheet updated successfully.")


if __name__ == "__main__":
    main()
