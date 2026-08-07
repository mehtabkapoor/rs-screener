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

# ---- Portfolio construction (quant layer) ----
TOP_N = 10                     # target portfolio size
RANK_BUFFER = 20               # hold existing positions until they fall past this rank
                                # (reduces whipsaw/charge drag vs re-ranking to exactly top 10 daily)
BREADTH_RISK_ON = 60           # % of universe above 50DMA -> full-size new entries allowed
BREADTH_RISK_CAUTION = 40      # % above 50DMA -> half-size / caution zone
                                # below this -> no new entries, existing holdings still managed
BREADTH_CIRCUIT_BREAKER = 25   # % above 50DMA -> full defensive exit, sell everything regardless of rank
RS_HISTORY_DAYS = 90            # days of RS-line history to store for the sparkline chart
INTRADAY_INTERVAL = "5m"        # granularity for live preview price
HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"
RS_HISTORY_WORKSHEET = "RS_History"
STOP_LOSS_PCT = 0.08           # 8% below entry as a hard floor (classic O'Neil stop)
# -----------------------------------------


def get_run_mode():
    """
    GitHub Actions automatically sets GITHUB_EVENT_NAME for every run:
      'schedule'         -> the automated 4:45 PM IST run, using final EOD prices
      'workflow_dispatch' -> a manual 'Run workflow' click, treated as an intraday
                             preview (doesn't touch Holdings/Portfolio, so nothing
                             gets locked in on an unsettled price)
    """
    event = os.environ.get("GITHUB_EVENT_NAME", "manual")
    return "EOD" if event == "schedule" else "PREVIEW"


def col_letter(n):
    """Converts a 1-indexed column number to a spreadsheet column letter (1=A, 27=AA, etc.)."""
    letters = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def load_tickers():
    """Reads your stock universe from stocks.csv (one 'symbol' column, NSE symbols without .NS)."""
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
    """Batch-fetches today's latest available intraday price for a list of tickers.
    Returns {ticker: last_price}. Used only in PREVIEW mode (manual runs before close)."""
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
    """Appends today's live intraday price as the latest point in a daily close
    series, ONLY if today's date isn't already the last bar (avoids duplicating
    a bar that yfinance may have already started forming)."""
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

    run_mode = get_run_mode()
    print(f"Run mode: {run_mode}"
          + (" (manual preview -- Holdings/Portfolio will NOT be updated)" if run_mode == "PREVIEW" else ""))

    bench_close = download_benchmark(run_mode)

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

                # RS ratio history for the sparkline chart (stock/benchmark, last N days)
                aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
                aligned.columns = ["s", "b"]
                rs_ratio_series = (aligned["s"] / aligned["b"]).tail(RS_HISTORY_DAYS)
                rs_history = [round(float(v), 4) for v in rs_ratio_series.tolist()]

                all_stocks.append({
                    "symbol": symbol.replace(".NS", ""),
                    "rs_score": rs_score,
                    "blue_dot": blue_dot,
                    "green_dot": green_dot,
                    "last_close": round(float(close.iloc[-1]), 2),
                    "tt_pass": tt_result[0] if tt_result else None,
                    "tt_criteria_met": tt_result[1] if tt_result else None,
                    "above_50dma": above_50dma,
                    "sma50": sma50_value,
                    "rs_history": rs_history,
                })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue

        time.sleep(1)  # be polite to Yahoo's servers

    if not all_stocks:
        print("No stocks with sufficient data found.")
        results_df = pd.DataFrame(columns=[
            "symbol", "composite_score", "rs_score", "rs_rating", "blue_dot", "green_dot",
            "trend_template", "tt_criteria", "last_close"])
        write_to_sheet(results_df, run_mode)
        return

    universe_df = pd.DataFrame(all_stocks)

    # RS Rating: percentile rank (1-99) of each stock's RS Score across the whole
    # scanned universe today -- this mirrors IBD's RS Rating scale.
    universe_df["rs_rating"] = (universe_df["rs_score"].rank(pct=True) * 98 + 1).round(0).astype(int)

    # Composite Score (0-100) for the FULL universe: blends RS strength, trend
    # structure, and breakout timing into one sortable number -- same weighted-
    # composite spirit as your Nifty 750 system.
    #   50% RS Rating          -- raw relative strength, the core signal
    #   30% Trend Template     -- structural health (Minervini's MA/52wk criteria, /7)
    #   20% Green Dot bonus    -- rewards RS leading price (only applies if Blue Dot fired)
    universe_df["trend_template"] = universe_df["tt_pass"].map(
        {True: "PASS", False: "FAIL", None: "N/A"})
    universe_df["blue_dot_label"] = universe_df["blue_dot"].map({True: "YES", False: ""})
    universe_df["green_dot_label"] = universe_df["green_dot"].map({True: "YES", False: ""})
    universe_df["tt_criteria"] = universe_df["tt_criteria_met"].apply(
        lambda x: f"{x}/7" if pd.notna(x) else "N/A")

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
        "trend_template", "tt_criteria", "last_close"
    ]].rename(columns={"blue_dot_label": "blue_dot", "green_dot_label": "green_dot"})
    results_df = results_df.sort_values("composite_score", ascending=False)
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank"] + [c for c in results_df.columns if c != "rank"]]

    n_blue = (results_df["blue_dot"] == "YES").sum()
    n_green = (results_df["green_dot"] == "YES").sum()
    n_tt_pass = (results_df["trend_template"] == "PASS").sum()
    print(f"Universe scanned: {len(universe_df)} stocks (all shown in output).")
    print(f"Blue Dot: {n_blue} | Green Dot: {n_green} | Trend Template PASS: {n_tt_pass}")

    write_to_sheet(results_df, run_mode)

    if run_mode == "PREVIEW":
        print("Preview mode: RS_Screener tab updated with live intraday prices. "
              "Holdings/Portfolio left untouched -- nothing locked in on an unsettled price.")
        return

    # ---- Regime / breadth filter (EOD run only) ----
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

    # ---- Portfolio construction: rank buffer + persistent holdings ----
    sma50_lookup = dict(zip(universe_df["symbol"], universe_df["sma50"]))
    rs_history_lookup = dict(zip(universe_df["symbol"], universe_df["rs_history"]))
    build_portfolio(results_df, breadth_pct, regime, allow_new_entries, sma50_lookup,
                     rs_history_lookup, circuit_breaker)


def read_config(sh):
    """Reads user-editable settings (available capital, etc.) from the Config tab.
    Creates the tab with a default row if it doesn't exist yet -- edit the value
    cell directly in Google Sheets to update your available investment fund."""
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


def build_portfolio(ranked_df, breadth_pct, regime, allow_new_entries, sma50_lookup,
                     rs_history_lookup, circuit_breaker):
    """
    Reads current holdings + available capital, applies rank-buffer logic
    (hold until rank falls past RANK_BUFFER, only add new names from top TOP_N
    when a slot opens), sizes positions equal-weight (scaled by regime), tracks
    entry price for real P&L, and computes a stop-loss level per position.
    Writes everything to the 'Portfolio' tab.
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

    # Read current holdings: symbol, entry_price, entry_date
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

    rank_lookup = dict(zip(ranked_df["symbol"], ranked_df["rank"]))
    price_lookup = dict(zip(ranked_df["symbol"], ranked_df["last_close"]))

    # Keep existing holdings whose rank hasn't fallen past the buffer.
    # Circuit breaker overrides everything -- full defensive exit regardless of rank.
    if circuit_breaker:
        kept = []
        sold = list(current_holdings.keys())
    else:
        kept = [s for s in current_holdings if rank_lookup.get(s, 9999) <= RANK_BUFFER]
        sold = [s for s in current_holdings if s not in kept]

    slots_open = TOP_N - len(kept)
    new_adds = []
    if allow_new_entries and slots_open > 0:
        candidates = ranked_df[~ranked_df["symbol"].isin(kept)].sort_values("rank")
        new_adds = candidates.head(slots_open)["symbol"].tolist()

    final_holdings = kept + new_adds

    # Position sizing: equal-weight across TOP_N, scaled down in caution regime
    if regime.startswith("RISK-ON"):
        size_multiplier = 1.0
    elif regime.startswith("CAUTION"):
        size_multiplier = 0.5
    else:
        size_multiplier = 0.0  # no new entries; existing holdings keep prior sizing
    slot_capital = (capital / TOP_N) * size_multiplier if capital > 0 else 0

    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")

    rows = []
    updated_holdings = {}
    for s in final_holdings:
        r = ranked_df[ranked_df["symbol"] == s].iloc[0]
        current_price = price_lookup.get(s, 0)

        if s in new_adds:
            action = "BUY"
            entry_price = current_price
            entry_date = today_str
            position_value = round(slot_capital, 0)
        else:
            action = "HOLD"
            entry_price = current_holdings[s]["entry_price"] or current_price
            entry_date = current_holdings[s]["entry_date"] or today_str
            # Recompute an indicative position value at original entry sizing basis
            position_value = round((capital / TOP_N), 0) if capital > 0 else 0

        updated_holdings[s] = {"entry_price": entry_price, "entry_date": entry_date}

        qty = int(position_value / entry_price) if entry_price > 0 else 0
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0

        sma50_val = sma50_lookup.get(s)
        stop_loss = None
        if sma50_val:
            stop_loss = round(max(sma50_val, entry_price * (1 - STOP_LOSS_PCT)), 2)
        elif entry_price:
            stop_loss = round(entry_price * (1 - STOP_LOSS_PCT), 2)

        rows.append([
            action, s, int(r["rank"]), r["composite_score"],
            entry_price, entry_date, current_price, qty, position_value,
            f"{pnl_pct}%", stop_loss,
            r["blue_dot"], r["green_dot"], r["trend_template"]
        ])

    for s in sold:
        entry_price = current_holdings[s]["entry_price"]
        current_price = price_lookup.get(s, 0)
        pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
        rows.append([
            "SELL", s, rank_lookup.get(s, "N/A"), "",
            entry_price, current_holdings[s]["entry_date"], current_price, "", "",
            f"{pnl_pct}%", "", "", "", ""
        ])

    try:
        port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
    except gspread.WorksheetNotFound:
        port_ws = sh.add_worksheet(title=PORTFOLIO_WORKSHEET, rows=50, cols=15)

    port_ws.clear()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    invested = sum(r[8] for r in rows if isinstance(r[8], (int, float)))
    summary = (f"Last updated: {timestamp}  |  Breadth: {breadth_pct}%  |  Regime: {regime}  |  "
               f"Capital: Rs.{capital:,.0f}  |  Deployed (approx): Rs.{invested:,.0f}")
    if circuit_breaker:
        summary = "*** CIRCUIT BREAKER TRIGGERED: REDUCE TO CASH ***  |  " + summary
    if capital == 0:
        summary += "  |  SET YOUR CAPITAL in the Config tab"
    port_ws.update([[summary]], "A1")
    header = ["Action", "Symbol", "Rank", "Composite Score", "Entry Price", "Entry Date",
              "Current Price", "Qty", "Position Value (Rs)", "P&L %", "Stop-Loss",
              "Blue Dot", "Green Dot", "Trend Template", "RS Line"]
    # Add a blank placeholder for the RS Line column -- filled in separately below via
    # a formula-mode update, since sparkline formulas need value_input_option='USER_ENTERED'.
    rows_with_placeholder = [r + [""] for r in rows]
    port_ws.update([header] + rows_with_placeholder, "A3")

    # ---- RS Line sparkline: store history data + insert SPARKLINE() formulas ----
    write_rs_history_and_sparklines(sh, rows, rs_history_lookup)

    # Persist updated holdings state (with entry price/date) for next run
    holdings_ws.clear()
    holdings_rows = [["symbol", "entry_price", "entry_date"]] + [
        [s, updated_holdings[s]["entry_price"], updated_holdings[s]["entry_date"]]
        for s in final_holdings
    ]
    holdings_ws.update(holdings_rows, "A1")

    print(f"Portfolio updated: {len(kept)} held, {len(new_adds)} bought, {len(sold)} sold. "
          f"Capital: Rs.{capital:,.0f}")


def write_rs_history_and_sparklines(sh, rows, rs_history_lookup):
    """
    Writes RS-ratio history (last RS_HISTORY_DAYS values) for every symbol in the
    Portfolio tab into a dedicated RS_History tab, then inserts a SPARKLINE()
    formula into the Portfolio tab's 'RS Line' column referencing that data --
    giving you a mini RS-line chart inline in each row.
    """
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
        symbol = r[1]
        history = rs_history_lookup.get(symbol, [])
        # pad on the left if a symbol has less than RS_HISTORY_DAYS of data
        padded = [""] * (RS_HISTORY_DAYS - len(history)) + history
        hist_rows.append([symbol] + padded)

    hist_ws.clear()
    hist_ws.update([hist_header] + hist_rows, "A1")

    # Insert SPARKLINE formulas into the Portfolio tab's RS Line column (column O)
    last_col_letter = col_letter(1 + RS_HISTORY_DAYS)  # data spans B..this column
    port_ws = sh.worksheet(PORTFOLIO_WORKSHEET)
    formulas = []
    for i in range(len(rows)):
        hist_row = i + 2  # RS_History data starts at row 2 (row 1 is header)
        formulas.append([f"=SPARKLINE('RS_History'!B{hist_row}:{last_col_letter}{hist_row})"])

    first_port_row = 4  # Portfolio data starts at row 4 (row 3 is header, row 1 is summary)
    last_port_row = first_port_row + len(rows) - 1
    port_ws.update(f"O{first_port_row}:O{last_port_row}", formulas, value_input_option="USER_ENTERED")


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

    n_rows_needed = len(df) + 10   # + buffer for header/timestamp rows
    n_cols_needed = len(df.columns) + 2

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
        # Resize up if your stock list has grown since the sheet was first created
        if ws.row_count < n_rows_needed or ws.col_count < n_cols_needed:
            ws.resize(rows=max(ws.row_count, n_rows_needed),
                      cols=max(ws.col_count, n_cols_needed))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=n_rows_needed, cols=n_cols_needed)

    ws.clear()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    label = "PREVIEW (intraday, not final)" if run_mode == "PREVIEW" else "EOD FINAL"
    ws.update([[f"Last updated: {timestamp}  |  {label}"]], "A1")
    header = ["Symbol", "Composite Score", "RS Score", "RS Rating (1-99)", "Blue Dot",
              "Green Dot (Breakout Watch)", "Trend Template", "TT Criteria Met", "Last Close"]
    ws.update([header] + df.values.tolist(), "A3")
    print("Google Sheet updated successfully.")


if __name__ == "__main__":
    main()
