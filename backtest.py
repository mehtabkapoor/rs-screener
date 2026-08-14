"""
RS Screener Backtest - SIMPLIFIED MODEL (matches live screener)

Implements exactly this rule set:

  Universe        : stocks.csv
  Price filter    : Price > Rs.20
  Liquidity       : 20-day avg volume > 100,000
  Price TT        : 7/7 required
  RS Line TT      : 7/7 required
  Ranking         : raw RS Score, descending (Rank 1 = highest RS Score)
  Portfolio       : Top 10
  Weight          : Equal weight at entry
  Rebalance       : every trading day EOD
  Exit rule       : sell when rank among eligible universe > 15
  Blue Dot / 1Y RS cross / Green Dot : diagnostic only

  Buy costs       : STT, stamp duty, exchange, SEBI, GST
  Sell costs      : STT, exchange, SEBI, GST + Rs.20 DP
  STCG            : 20.8% effective on positive realized gains
  Equity          : daily mark-to-market
  Terminal value  : marked AND liquidation (net of sell costs + tax)
  Charts          : Equity Curve + Drawdown (native Google Sheets)
"""

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
BACKTEST_END = None          # None = latest available trading day

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250

MAX_PLAUSIBLE_DAILY_MOVE = 0.30

TOP_N = 10
EXIT_RANK = 15               # sell when rank > EXIT_RANK
STARTING_CAPITAL = 1_000_000  # Rs 10 lakh

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20

STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest"


def get_download_dates():
    backtest_start = pd.Timestamp(BACKTEST_START)
    download_start = backtest_start - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    if BACKTEST_END is None:
        return download_start.strftime("%Y-%m-%d"), None
    backtest_end = pd.Timestamp(BACKTEST_END)
    download_end = backtest_end + pd.Timedelta(days=1)
    return download_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d")


def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")
    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")
    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]
    return [s if s.endswith(".NS") else s + ".NS" for s in symbols]


def clean_price_series(close):
    close = close.copy().sort_index()
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


def download_benchmark():
    download_start, download_end = get_download_dates()
    print(f"\nBenchmark download: {download_start} to {download_end}")
    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(ticker, start=download_start, end=download_end,
                                interval="1d", auto_adjust=True, progress=False)
            if data.empty:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().sort_index()
            if close.empty:
                continue
            close, n_bad = clean_price_series(close)
            if n_bad:
                print(f"Benchmark {ticker}: repaired {n_bad} implausible data point(s)")
            print(f"Benchmark loaded: {ticker}")
            return close
        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")
    raise RuntimeError("Could not download any benchmark index data.")


def trend_template_series(s):
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

    met = (c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
           + c5.astype(int) + c6.astype(int) + c7.astype(int))
    return met == 7


def compute_signals_for_stock(close, volume, bench_close):
    aligned = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) < 280:
        return None

    volume = volume.reindex(aligned.index)
    rs_ratio = aligned["s"] / aligned["b"]

    def pct_return(series, days):
        return series / series.shift(days) - 1

    rs_score = (0.40 * pct_return(aligned["s"], 63)
                + 0.20 * pct_return(aligned["s"], 126)
                + 0.20 * pct_return(aligned["s"], 189)
                + 0.20 * pct_return(aligned["s"], 252)) * 100

    previous_rs_high = rs_ratio.shift(1).rolling(LOOKBACK_DAYS).max()
    blue_dot = rs_ratio > previous_rs_high

    previous_price_high = aligned["s"].shift(1).rolling(LOOKBACK_DAYS).max()
    price_at_new_high = aligned["s"] > previous_price_high
    green_dot = blue_dot & (~price_at_new_high)

    tt_pass = trend_template_series(aligned["s"])
    rs_tt_pass = trend_template_series(rs_ratio)

    rolling_avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (aligned["s"] > MIN_PRICE) & (rolling_avg_volume > MIN_AVG_VOLUME)

    out = pd.DataFrame({
        "price": aligned["s"],
        "rs_score": rs_score,
        "tt_pass": tt_pass,
        "rs_tt_pass": rs_tt_pass,
        "liquid": liquid,
        "blue_dot": blue_dot,
        "green_dot": green_dot,
    })
    return out


def run_backtest(all_signals, trading_days):
    """
    ENTRY : liquid + Price TT + RS Line TT, ranked by RS Score, Top 10 equal weight.
    EXIT  : rank among eligible universe > EXIT_RANK (15).
    Rebalanced every trading day EOD.
    """
    cash = STARTING_CAPITAL
    holdings = {}   # sym -> {qty, entry_price, entry_date, entry_cost}
    trade_log = []
    equity_curve = []

    for date in trading_days:
        # Eligible pool
        pool = []
        for sym, df in all_signals.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["rs_score"]) or not bool(row["liquid"]):
                continue
            if bool(row["tt_pass"]) and bool(row["rs_tt_pass"]):
                pool.append((sym, float(row["rs_score"])))

        pool.sort(key=lambda x: x[1], reverse=True)
        rank_lookup = {sym: rank + 1 for rank, (sym, _) in enumerate(pool)}
        target_topN = {sym for sym, _ in pool[:TOP_N]}

        # ---- Exit: rank missing or rank > EXIT_RANK ----
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

            cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
            net_gain = net_proceeds - cost_basis
            tax = stcg_tax(net_gain)
            cash += net_proceeds - tax

            gross_return_pct = round((exit_price / pos["entry_price"] - 1) * 100, 2)
            net_return_pct = round((net_gain - tax) / cost_basis * 100, 2) if cost_basis > 0 else 0

            exit_reason = f"Rank > {EXIT_RANK}" if rank is not None else "Left eligible universe"
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
                "net_return_pct": net_return_pct,
                "days_held": (date - pos["entry_date"]).days,
                "exit_reason": exit_reason,
                "exit_rank": rank if rank is not None else "",
            })

        # ---- Entry: fill open slots from Top N ----
        portfolio_value = cash
        for sym, pos in holdings.items():
            df = all_signals[sym]
            price = float(df.loc[date, "price"]) if date in df.index else pos["entry_price"]
            portfolio_value += pos["qty"] * price

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

        # ---- Daily mark-to-market ----
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

    # Terminal values
    final_marked_value = equity_curve[-1]["portfolio_value_rs"] if equity_curve else STARTING_CAPITAL

    liquidation_cash = cash
    open_positions_detail = []
    if len(trading_days) and holdings:
        last_date = trading_days[-1]
        for sym, pos in holdings.items():
            df = all_signals[sym]
            exit_price = float(df.loc[last_date, "price"]) if last_date in df.index else pos["entry_price"]
            gross_proceeds = pos["qty"] * exit_price
            s_cost = sell_side_cost(gross_proceeds)
            net_proceeds = gross_proceeds - s_cost
            cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
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


def summarize(trade_df, equity_df, final_marked_value, final_liquidation_value):
    if equity_df.empty:
        return {}
    net_total_return_marked_pct = round((final_marked_value / STARTING_CAPITAL - 1) * 100, 2)
    net_total_return_liquidation_pct = round((final_liquidation_value / STARTING_CAPITAL - 1) * 100, 2)

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] / running_max - 1) * 100
    max_dd = round(drawdown.min(), 2)

    closed = trade_df if not trade_df.empty else trade_df
    n = len(closed)
    if n:
        win_rate_net = round((closed["net_return_pct"] > 0).mean() * 100, 1)
        win_rate_gross = round((closed["gross_return_pct"] > 0).mean() * 100, 1)
        avg_gross = round(closed["gross_return_pct"].mean(), 2)
        avg_net = round(closed["net_return_pct"].mean(), 2)
        median_net = round(closed["net_return_pct"].median(), 2)
        avg_days = round(closed["days_held"].mean(), 1)
        best_gross = closed["gross_return_pct"].max()
        worst_gross = closed["gross_return_pct"].min()
        total_costs_rs = round((closed["buy_cost_rs"] + closed["sell_cost_rs"]).sum(), 0)
        total_tax_rs = round(closed["stcg_tax_rs"].sum(), 0)
        winners = closed[closed["net_return_pct"] > 0]
        losers = closed[closed["net_return_pct"] < 0]
        avg_winner = round(winners["net_return_pct"].mean(), 2) if len(winners) else 0
        avg_loser = round(losers["net_return_pct"].mean(), 2) if len(losers) else 0
        gp = winners["net_pnl_rs"].sum() if len(winners) else 0
        gl = abs(losers["net_pnl_rs"].sum()) if len(losers) else 0
        profit_factor = round(gp / gl, 3) if gl > 0 else 0
    else:
        win_rate_net = win_rate_gross = avg_gross = avg_net = median_net = 0
        avg_days = best_gross = worst_gross = total_costs_rs = total_tax_rs = 0
        avg_winner = avg_loser = profit_factor = 0

    daily_returns = equity_df["equity"].pct_change().dropna()
    if len(daily_returns):
        daily_mean, daily_std = daily_returns.mean(), daily_returns.std()
        n_days = len(equity_df)
        annualized_return = equity_df["equity"].iloc[-1] ** (252 / max(n_days, 1)) - 1
        annualized_vol = daily_std * np.sqrt(252)
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) else 0
        sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        annualized_return = annualized_vol = sharpe = sortino = 0
    calmar = round(annualized_return / abs(max_dd / 100), 3) if abs(max_dd) > 0 else 0

    return {
        "Final Value (marked, Rs)": round(final_marked_value, 0),
        "Final Value (liquidation, Rs)": round(final_liquidation_value, 0),
        "Net Return - marked (%)": net_total_return_marked_pct,
        "Net Return - liquidation (%)": net_total_return_liquidation_pct,
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": calmar,
        "Max Drawdown (%)": max_dd,
        "Number of Closed Trades": n,
        "Win Rate - Gross (%)": win_rate_gross,
        "Win Rate - Net (%)": win_rate_net,
        "Avg Gross Return/Trade (%)": avg_gross,
        "Avg Net Return/Trade (%)": avg_net,
        "Median Net Return/Trade (%)": median_net,
        "Avg Days Held": avg_days,
        "Avg Winner (%)": avg_winner,
        "Avg Loser (%)": avg_loser,
        "Profit Factor (net)": profit_factor,
        "Best Gross Trade (%)": best_gross,
        "Worst Gross Trade (%)": worst_gross,
        "Total Costs Paid (Rs)": total_costs_rs,
        "Total STCG Tax Paid (Rs)": total_tax_rs,
    }


def write_in_chunks(ws, all_rows, start_row, chunk_size, label):
    total = len(all_rows)
    if total == 0:
        return
    for i in range(0, total, chunk_size):
        chunk = all_rows[i:i + chunk_size]
        row_start = start_row + i
        try:
            ws.update(chunk, f"A{row_start}")
        except Exception as e:
            print(f"  Write failed for {label} rows {i}-{i+len(chunk)}, retrying once: {e}")
            time.sleep(5)
            try:
                ws.update(chunk, f"A{row_start}")
            except Exception as e2:
                print(f"  RETRY FAILED for {label} rows {i}-{i+len(chunk)}: {e2}")
                raise
        print(f"  Wrote {label}: {min(i+chunk_size, total)}/{total} rows")


def remove_existing_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()
        requests = []
        for sheet in meta.get("sheets", []):
            if sheet["properties"]["sheetId"] == sheet_id:
                for chart in sheet.get("charts", []):
                    requests.append({"deleteEmbeddedObject": {"objectId": chart["chartId"]}})
        if requests:
            sh.batch_update({"requests": requests})
            print(f"Removed {len(requests)} existing chart(s).")
    except Exception as e:
        print(f"Could not check/remove existing charts (non-fatal): {e}")


def add_charts(sh, sheet_id, equity_header_row_0idx, n_equity_rows):
    data_end_row = equity_header_row_0idx + 1 + n_equity_rows

    def make_chart(title, y_col_idx, y_axis_title, anchor_row):
        return {
            "addChart": {
                "chart": {
                    "spec": {
                        "title": title,
                        "basicChart": {
                            "chartType": "LINE",
                            "legendPosition": "NO_LEGEND",
                            "axis": [
                                {"position": "BOTTOM_AXIS", "title": "Date"},
                                {"position": "LEFT_AXIS", "title": y_axis_title},
                            ],
                            "domains": [{
                                "domain": {"sourceRange": {"sources": [{
                                    "sheetId": sheet_id,
                                    "startRowIndex": equity_header_row_0idx,
                                    "endRowIndex": data_end_row,
                                    "startColumnIndex": 0,
                                    "endColumnIndex": 1,
                                }]}}
                            }],
                            "series": [{
                                "series": {"sourceRange": {"sources": [{
                                    "sheetId": sheet_id,
                                    "startRowIndex": equity_header_row_0idx,
                                    "endRowIndex": data_end_row,
                                    "startColumnIndex": y_col_idx,
                                    "endColumnIndex": y_col_idx + 1,
                                }]}},
                                "targetAxis": "LEFT_AXIS",
                            }],
                        },
                    },
                    "position": {
                        "overlayPosition": {
                            "anchorCell": {
                                "sheetId": sheet_id,
                                "rowIndex": anchor_row,
                                "columnIndex": 8,
                            },
                            "widthPixels": 650,
                            "heightPixels": 380,
                        }
                    },
                }
            }
        }

    requests = [
        make_chart("Equity Curve (Rs)", 1, "Portfolio Value (Rs)", equity_header_row_0idx),
        make_chart("Drawdown (%)", 5, "Drawdown %", equity_header_row_0idx + 22),
    ]
    try:
        sh.batch_update({"requests": requests})
        print("Equity and drawdown charts added.")
    except Exception as e:
        print(f"Could not add charts (non-fatal): {e}")


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

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    n_rows_needed = len(trade_df) + len(equity_df) + len(open_df) + len(summary) + 60
    n_cols_needed = 16
    try:
        ws = sh.worksheet(BACKTEST_WORKSHEET)
        if ws.row_count < n_rows_needed or ws.col_count < n_cols_needed:
            ws.resize(rows=max(ws.row_count, n_rows_needed),
                      cols=max(ws.col_count, n_cols_needed))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=BACKTEST_WORKSHEET, rows=n_rows_needed, cols=n_cols_needed)

    remove_existing_charts(sh, ws.id)
    ws.clear()

    ws.update([[
        f"SIMPLIFIED MODEL BACKTEST | run {timestamp} | NET of costs+STCG | "
        f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
        f"Entry: Price TT + RS TT | Exit: rank > {EXIT_RANK} | "
        f"Window: {BACKTEST_START} to {effective_end_str}"
    ]], "A1")

    summary_rows = [["Summary", ""]] + [[k, v] for k, v in summary.items()]
    ws.update(summary_rows, "A3")

    trade_start_row = 3 + len(summary_rows) + 2
    ws.update([["Trade Log"]], f"A{trade_start_row}")
    trade_header_row = trade_start_row + 1
    if not trade_df.empty:
        write_in_chunks(ws, [list(trade_df.columns)] + trade_df.values.tolist(),
                        start_row=trade_header_row, chunk_size=2000, label="trade log")

    open_start_row = trade_header_row + len(trade_df) + 3
    ws.update([["Open Positions at Backtest End (mark-to-market)"]], f"A{open_start_row}")
    open_header_row = open_start_row + 1
    if not open_df.empty:
        ws.update([list(open_df.columns)] + open_df.values.tolist(), f"A{open_header_row}")

    equity_start_row = open_header_row + max(len(open_df), 1) + 3
    ws.update([["Daily Equity Curve"]], f"A{equity_start_row}")
    equity_header_row = equity_start_row + 1
    if not equity_df.empty:
        write_in_chunks(ws, [list(equity_df.columns)] + equity_df.values.tolist(),
                        start_row=equity_header_row, chunk_size=2000, label="equity curve")
        add_charts(sh, ws.id, equity_header_row - 1, len(equity_df))

    print(f"\nBacktest results written to '{BACKTEST_WORKSHEET}' tab: "
          f"{len(trade_df)} trades, {len(equity_df)} trading days, {len(open_df)} open positions.")


def main():
    tickers = load_tickers()
    print(f"\nLoaded {len(tickers)} tickers.")

    download_start, download_end = get_download_dates()
    print("=" * 55)
    print(f"Download start   : {download_start}")
    print(f"Backtest start   : {BACKTEST_START}")
    print(f"Backtest end     : {BACKTEST_END if BACKTEST_END else 'latest available'}")
    print(f"Price filter     : > Rs.{MIN_PRICE}")
    print(f"Liquidity filter : {VOLUME_LOOKBACK}d avg volume > {MIN_AVG_VOLUME:,}")
    print(f"Entry filter     : Price TT + RS TT")
    print(f"Portfolio        : Top {TOP_N}, equal weight")
    print(f"Exit             : rank > {EXIT_RANK}")
    print(f"Starting capital : Rs.{STARTING_CAPITAL:,.0f}")
    print("=" * 55)

    bench_close = download_benchmark()

    all_signals = {}
    total_bad_points = 0
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\nDownloading batch {i}-{i+len(batch)}...")
        try:
            data = yf.download(batch, start=download_start, end=download_end,
                                interval="1d", auto_adjust=True, progress=False,
                                group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    sdata = data
                else:
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    sdata = data[symbol]
                if "Close" not in sdata.columns:
                    continue
                close = sdata["Close"].dropna().sort_index()
                volume = sdata["Volume"].reindex(close.index).fillna(0)
                if close.empty:
                    continue

                close, n_bad = clean_price_series(close)
                total_bad_points += n_bad

                sig = compute_signals_for_stock(close, volume, bench_close)
                if sig is not None:
                    all_signals[symbol.replace(".NS", "")] = sig
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue
        time.sleep(1)

    print(f"\nSignals computed for {len(all_signals)} stocks.")
    print(f"Total data points repaired: {total_bad_points}")

    # Filter funnel on latest date
    if all_signals:
        latest_date = max(df.index.max() for df in all_signals.values())
        counts = {"liquid": 0, "+tt_pass": 0, "+rs_tt_pass": 0}
        for sym, df in all_signals.items():
            if latest_date not in df.index:
                continue
            row = df.loc[latest_date]
            if pd.isna(row["rs_score"]) or not bool(row["liquid"]):
                continue
            counts["liquid"] += 1
            if not bool(row["tt_pass"]):
                continue
            counts["+tt_pass"] += 1
            if not bool(row["rs_tt_pass"]):
                continue
            counts["+rs_tt_pass"] += 1
        print(f"\nFilter funnel on {latest_date.strftime('%Y-%m-%d')}:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    effective_end = pd.Timestamp(BACKTEST_END) if BACKTEST_END else bench_close.index.max()
    print(f"Effective backtest end date: {effective_end.strftime('%Y-%m-%d')}"
          + ("" if BACKTEST_END else "  (auto-detected)"))

    trading_days = bench_close.index[
        (bench_close.index >= pd.Timestamp(BACKTEST_START)) &
        (bench_close.index <= effective_end)
    ]
    print(f"Trading days: {len(trading_days)}")

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