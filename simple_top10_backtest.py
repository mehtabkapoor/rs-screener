import time
import numpy as np
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

STOCKS_FILE = "stocks.csv"

DOWNLOAD_YEARS_BEFORE_START = 3

BACKTEST_START = "2016-04-01"

# None = automatically use latest available market data
BACKTEST_END = None

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250

MAX_PLAUSIBLE_DAILY_MOVE = 0.30

TOP_N = 10

# EXIT when RS rank becomes worse than TOP_N
EXIT_RANK = 10

STARTING_CAPITAL = 1_000_000

# ============================================================
# TRANSACTION COSTS
# ============================================================

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18

DP_CHARGE_FLAT = 20

STCG_RATE = 0.20
STCG_CESS = 0.04

STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "rs top 10 backtest"

# ============================================================
# DATE NORMALIZATION
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

# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():
    backtest_start = pd.Timestamp(BACKTEST_START)
    download_start = backtest_start - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)

    if BACKTEST_END is None:
        return (download_start.strftime("%Y-%m-%d"), None)

    backtest_end = pd.Timestamp(BACKTEST_END)
    download_end = backtest_end + pd.Timedelta(days=1)
    return (download_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d"))

# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")

    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")

    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]

    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]

# ============================================================
# CLEAN PRICE DATA
# ============================================================

def clean_price_series(close):
    close = normalize_series_index(close)
    pct_change = close.pct_change()
    bad = pct_change.abs() > MAX_PLAUSIBLE_DAILY_MOVE
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
# COST FUNCTIONS
# ============================================================

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

def stcg_tax(net_gain):
    if net_gain <= 0:
        return 0.0
    return net_gain * STCG_EFFECTIVE_RATE

# ============================================================
# BENCHMARK DOWNLOAD
# ============================================================

def download_benchmark():
    download_start, download_end = get_download_dates()
    print(f"\nBenchmark download: {download_start} to {download_end if download_end else 'LATEST AVAILABLE'}")

    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(ticker, start=download_start, end=download_end, interval="1d", auto_adjust=True, progress=False)
            if data.empty:
                continue

            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = normalize_series_index(close.dropna())
            if close.empty:
                continue

            close, n_bad = clean_price_series(close)
            if n_bad:
                print(f"Benchmark {ticker}: repaired {n_bad} implausible data point(s)")

            print(f"Benchmark loaded: {ticker} | Latest date: {close.index.max().strftime('%Y-%m-%d')}")
            return close

        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")

    raise RuntimeError("Could not download any benchmark index data.")

# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_signals_for_stock(close, volume, bench_close):
    close = normalize_series_index(close)
    volume = normalize_series_index(volume)
    bench_close = normalize_series_index(bench_close)

    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]

    if len(aligned) < 280:
        return None

    volume = volume.reindex(aligned.index).fillna(0)

    def pct_return(series, days):
        return series / series.shift(days) - 1

    # Raw RS Score (Does NOT use benchmark 'b' in calc, just raw price returns)
    rs_score = (
        0.40 * pct_return(aligned["s"], 63) +
        0.20 * pct_return(aligned["s"], 126) +
        0.20 * pct_return(aligned["s"], 189) +
        0.20 * pct_return(aligned["s"], 252)
    ) * 100

    rolling_avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (aligned["s"] > MIN_PRICE) & (rolling_avg_volume > MIN_AVG_VOLUME)

    out = pd.DataFrame({
        "price": aligned["s"],
        "rs_score": rs_score,
        "liquid": liquid,
    })
    out.index = normalize_dates(out.index)
    return out

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(all_signals, trading_days):
    cash = STARTING_CAPITAL
    holdings = {}
    trade_log = []
    equity_curve = []

    for date in trading_days:
        date = pd.Timestamp(date).normalize()

        # 1. Eligible Pool
        pool = []
        for sym, df in all_signals.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["rs_score"]) or not bool(row["liquid"]):
                continue
            pool.append((sym, float(row["rs_score"])))

        # 2. Rank by RS Score
        pool.sort(key=lambda x: x[1], reverse=True)
        rank_lookup = {sym: rank + 1 for rank, (sym, _) in enumerate(pool)}
        target_topN = {sym for sym, _ in pool[:TOP_N]}

        # 3. Exit Logic (Rank > 10 or left universe)
        for sym in list(holdings.keys()):
            rank = rank_lookup.get(sym)
            if rank is not None and rank <= EXIT_RANK:
                continue

            df = all_signals[sym]
            if date not in df.index:
                continue

            pos = holdings.pop(sym)
            exit_price = float(df.loc[date, "price"])
            gross_proceeds = pos["qty"] * exit_price
            s_cost = sell_side_cost(gross_proceeds)
            net_proceeds = gross_proceeds - s_cost
            cost_basis = (pos["qty"] * pos["entry_price"]) + pos["entry_cost"]
            net_gain = net_proceeds - cost_basis
            tax = stcg_tax(net_gain)
            cash += net_proceeds - tax

            gross_return_pct = round((exit_price / pos["entry_price"] - 1) * 100, 2)
            net_return_pct = ((net_gain - tax) / cost_basis * 100) if cost_basis > 0 else 0
            exit_reason = f"RS rank > {EXIT_RANK}" if rank is not None else "Left eligible universe"

            trade_log.append({
                "symbol": sym,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": date.strftime("%Y-%m-%d"),
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(exit_price, 2),
                "gross_return_pct": gross_return_pct,
                "buy_cost_rs": round(pos["entry_cost"], 2),
                "sell_cost_rs": round(s_cost, 2),
                "stcg_tax_rs": round(tax, 2),
                "net_pnl_rs": round(net_gain - tax, 2),
                "net_return_pct": round(net_return_pct, 2),
                "days_held": (date - pos["entry_date"]).days,
                "exit_reason": exit_reason,
                "exit_rank": rank if rank is not None else "",
            })

        # 4. Portfolio Valuation before Entries
        portfolio_value = cash
        for sym, pos in holdings.items():
            df = all_signals[sym]
            price = float(df.loc[date, "price"]) if date in df.index else pos["entry_price"]
            portfolio_value += pos["qty"] * price

        # 5. Entry Logic (Fill empty slots)
        slots_open = TOP_N - len(holdings)
        if slots_open > 0:
            slot_capital = portfolio_value / TOP_N
            for sym in [s for s, _ in pool[:TOP_N]]:
                if slots_open <= 0:
                    break
                if sym in holdings:
                    continue

                price = float(all_signals[sym].loc[date, "price"])
                qty = int(slot_capital // price) if price > 0 else 0
                if qty < 1:
                    continue

                trade_value = qty * price
                b_cost = buy_side_cost(trade_value)
                total_cost = trade_value + b_cost

                if total_cost > cash:
                    continue

                cash -= total_cost
                holdings[sym] = {
                    "qty": qty,
                    "entry_price": price,
                    "entry_date": date,
                    "entry_cost": b_cost,
                }
                slots_open -= 1

        # 6. Daily Mark-to-Market
        portfolio_value = cash
        for sym, pos in holdings.items():
            df = all_signals[sym]
            price = float(df.loc[date, "price"]) if date in df.index else pos["entry_price"]
            portfolio_value += pos["qty"] * price

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(portfolio_value, 2),
            "equity": round(portfolio_value / STARTING_CAPITAL, 6),
            "cash_rs": round(cash, 2),
            "n_holdings": len(holdings),
        })

    # Terminal Liquidation
    final_marked_value = equity_curve[-1]["portfolio_value_rs"] if equity_curve else STARTING_CAPITAL
    liquidation_cash = cash
    open_positions_detail = []

    if len(trading_days) and holdings:
        last_date = pd.Timestamp(trading_days[-1]).normalize()
        for sym, pos in holdings.items():
            df = all_signals[sym]
            exit_price = float(df.loc[last_date, "price"]) if last_date in df.index else pos["entry_price"]
            gross_proceeds = pos["qty"] * exit_price
            s_cost = sell_side_cost(gross_proceeds)
            net_proceeds = gross_proceeds - s_cost
            cost_basis = (pos["qty"] * pos["entry_price"]) + pos["entry_cost"]
            net_gain = net_proceeds - cost_basis
            tax = stcg_tax(net_gain)
            liquidation_cash += net_proceeds - tax

            open_positions_detail.append({
                "symbol": sym,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"], 2),
                "last_price": round(exit_price, 2),
                "unrealized_gross_return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
            })

    final_liquidation_value = liquidation_cash
    trade_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity_curve)

    if not equity_df.empty:
        running_max = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = ((equity_df["equity"] / running_max - 1) * 100).round(3)

    open_df = pd.DataFrame(open_positions_detail)

    return trade_df, equity_df, open_df, final_marked_value, final_liquidation_value

# ============================================================
# SUMMARY & SHEETS OUTPUT (Abbreviated for brevity)
# ============================================================

def summarize(trade_df, equity_df, final_marked_value, final_liquidation_value):
    if equity_df.empty:
        return {}
    
    net_total_return_marked_pct = round((final_marked_value / STARTING_CAPITAL - 1) * 100, 2)
    net_total_return_liquidation_pct = round((final_liquidation_value / STARTING_CAPITAL - 1) * 100, 2)
    
    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] / running_max - 1) * 100
    max_dd = round(drawdown.min(), 2)

    if not trade_df.empty:
        closed = trade_df
        n = len(closed)
        win_rate_net = round((closed["net_return_pct"] > 0).mean() * 100, 1)
        avg_net = round(closed["net_return_pct"].mean(), 2)
        avg_days = round(closed["days_held"].mean(), 1)
        total_costs_rs = round((closed["buy_cost_rs"] + closed["sell_cost_rs"]).sum(), 0)
        total_tax_rs = round(closed["stcg_tax_rs"].sum(), 0)
    else:
        n, win_rate_net, avg_net, avg_days, total_costs_rs, total_tax_rs = 0, 0, 0, 0, 0, 0

    daily_returns = equity_df["equity"].pct_change().dropna()
    annualized_return = (equity_df["equity"].iloc[-1] ** (252 / max(len(equity_df), 1))) - 1
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    calmar = round(annualized_return / abs(max_dd / 100), 3) if abs(max_dd) > 0 else 0

    return {
        "Backtest Start": BACKTEST_START,
        "Backtest End": equity_df["date"].iloc[-1] if not equity_df.empty else "",
        "Starting Capital (Rs)": STARTING_CAPITAL,
        "Final Value (marked, Rs)": round(final_marked_value, 0),
        "Final Value (liquidation, Rs)": round(final_liquidation_value, 0),
        "Net Return - marked (%)": net_total_return_marked_pct,
        "Net Return - liquidation (%)": net_total_return_liquidation_pct,
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Calmar": calmar,
        "Max Drawdown (%)": max_dd,
        "Number of Closed Trades": n,
        "Win Rate - Net (%)": win_rate_net,
        "Avg Net Return/Trade (%)": avg_net,
        "Avg Days Held": avg_days,
        "Total Costs Paid (Rs)": total_costs_rs,
        "Total STCG Tax Paid (Rs)": total_tax_rs,
    }

def write_to_sheet(trade_df, equity_df, open_df, summary, effective_end_str):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- saving to CSV instead.")
        trade_df.to_csv("backtest_trades.csv", index=False)
        equity_df.to_csv("backtest_equity.csv", index=False)
        if not open_df.empty:
            open_df.to_csv("backtest_open_positions.csv", index=False)
        return

    print("Google Sheets credentials found. Uploading data...")
    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    
    ws = sh.worksheet(BACKTEST_WORKSHEET) if BACKTEST_WORKSHEET in [s.title for s in sh.worksheets()] else sh.add_worksheet(BACKTEST_WORKSHEET, 1000, 16)
    ws.clear()

    ws.update([["RS TOP 10 BACKTEST", f"Window: {BACKTEST_START} to {effective_end_str}"]], "A1")
    ws.update([["Summary", ""]], "A3")
    ws.update([[k, v] for k, v in summary.items()], "A4")
    
    ws.update([["Trade Log"]], f"A{3 + len(summary) + 2}")
    if not trade_df.empty:
        ws.update([list(trade_df.columns)] + trade_df.values.tolist(), f"A{3 + len(summary) + 3}")

    print(f"Backtest results written to '{BACKTEST_WORKSHEET}' tab.")

# ============================================================
# MAIN
# ============================================================

def main():
    tickers = load_tickers()
    download_start, download_end = get_download_dates()

    print("=" * 60)
    print(f"Loaded {len(tickers)} tickers.")
    print(f"Backtest Window: {BACKTEST_START} to {download_end if download_end else 'LATEST AVAILABLE'}")
    print(f"Filters: Price > {MIN_PRICE} | Avg Vol > {MIN_AVG_VOLUME}")
    print(f"Portfolio: Top {TOP_N} | Exit Rank: > {EXIT_RANK}")
    print(f"Starting Capital: Rs.{STARTING_CAPITAL:,.0f}")
    print("=" * 60)

    bench_close = download_benchmark()
    all_signals = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i}-{i + len(batch)}...")
        try:
            data = yf.download(batch, start=download_start, end=download_end, interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch failed: {e}")
            continue

        for symbol in batch:
            try:
                sdata = data if len(batch) == 1 else data[symbol]
                if "Close" not in sdata.columns:
                    continue

                close = sdata["Close"].dropna().sort_index()
                volume = sdata["Volume"].reindex(close.index).fillna(0)
                if close.empty:
                    continue

                close, _ = clean_price_series(close)
                sig = compute_signals_for_stock(close, volume, bench_close)
                if sig is not None:
                    all_signals[symbol.replace(".NS", "")] = sig
            except Exception as e:
                print(f"Skipping {symbol}: {e}")

        time.sleep(1)

    print(f"\nSignals computed for {len(all_signals)} stocks.")
    
    if BACKTEST_END is None:
        effective_end = pd.Timestamp(bench_close.index.max()).normalize()
    else:
        effective_end = min(pd.Timestamp(BACKTEST_END).normalize(), pd.Timestamp(bench_close.index.max()).normalize())

    trading_days = bench_close.index[(bench_close.index >= pd.Timestamp(BACKTEST_START).normalize()) & (bench_close.index <= effective_end)]
    trading_days = pd.DatetimeIndex(trading_days).drop_duplicates().sort_values()
    
    print(f"Trading days to process: {len(trading_days)}")

    trade_df, equity_df, open_df, final_marked, final_liq = run_backtest(all_signals, trading_days)
    summary = summarize(trade_df, equity_df, final_marked, final_liq)

    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    write_to_sheet(trade_df, equity_df, open_df, summary, effective_end.strftime("%Y-%m-%d"))

if __name__ == "__main__":
    try:
        main()
        print("\nBACKTEST COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print("\nBACKTEST FAILED")
        print(f"{type(e).__name__}: {e}")
        raise