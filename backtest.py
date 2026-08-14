# RS SCREENER BACKTEST — SYNCED WITH LIVE SCREENER

import os, json, time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from datetime import datetime
from zoneinfo import ZoneInfo
from google.oauth2.service_account import Credentials

# =========================
# PARAMETERS
# =========================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"
STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

STARTING_CAPITAL = 1_000_000

TOP_N = 10
EXIT_RANK = 15

MIN_PRICE = 10
MIN_AVG_VOLUME = 50_000
VOLUME_LOOKBACK = 50

DOWNLOAD_YEARS_BEFORE_START = 3

ENABLE_COSTS = True
STT_BUY_RATE = 0.001
STT_SELL_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20

ENABLE_STCG = True
STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

SUMMARY_SHEET = "Backtest_Summary"
BACKTEST_SHEET = "Backtest"

# =========================
# COSTS
# =========================

def buy_cost(v):
    if not ENABLE_COSTS: return 0
    return (
        STT_BUY_RATE*v +
        STAMP_DUTY_RATE*v +
        EXCHANGE_CHARGE_RATE*v +
        SEBI_CHARGE_RATE*v +
        GST_RATE*(EXCHANGE_CHARGE_RATE*v + SEBI_CHARGE_RATE*v)
    )

def sell_cost(v):
    if not ENABLE_COSTS: return 0
    return (
        STT_SELL_RATE*v +
        EXCHANGE_CHARGE_RATE*v +
        SEBI_CHARGE_RATE*v +
        GST_RATE*(EXCHANGE_CHARGE_RATE*v + SEBI_CHARGE_RATE*v) +
        DP_CHARGE_FLAT
    )

def stcg(gain):
    return (
        max(0, gain) * STCG_EFFECTIVE_RATE
        if ENABLE_STCG else 0
    )

# =========================
# LOAD UNIVERSE
# =========================

def load_tickers():
    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv needs a 'symbol' column")
    return [
        x if x.endswith(".NS") else x+".NS"
        for x in df["symbol"].dropna().astype(str).str.strip()
        if x
    ]

# =========================
# TREND TEMPLATE
# =========================

def tt(s):
    sma50 = s.rolling(50).mean()
    sma150 = s.rolling(150).mean()
    sma200 = s.rolling(200).mean()
    low52 = s.rolling(252).min()
    high52 = s.rolling(252).max()

    c = [
        (s > sma150) & (s > sma200),
        sma150 > sma200,
        sma200 > sma200.shift(21),
        (sma50 > sma150) & (sma50 > sma200),
        s > sma50,
        s >= 1.25 * low52,
        s >= 0.75 * high52
    ]

    met = sum(x.astype(int) for x in c)
    return met == 7, met

# =========================
# STOCK SIGNALS
# =========================

def stock_signals(close, volume, bench):

    x = pd.concat(
        [close, bench],
        axis=1,
        join="inner"
    ).dropna()

    x.columns = ["s", "b"]

    if len(x) < 280:
        return None

    volume = volume.reindex(x.index).fillna(0)

    avg_vol = volume.rolling(VOLUME_LOOKBACK).mean()

    price_tt, price_met = tt(x.s)

    rs_line = x.s / x.b

    rs_tt, rs_met = tt(rs_line)

    def ret(days):
        return x.s / x.s.shift(days) - 1

    rs_score = (
        0.40 * ret(63) +
        0.20 * ret(126) +
        0.20 * ret(189) +
        0.20 * ret(252)
    ) * 100

    liquid = (
        (x.s >= MIN_PRICE) &
        (avg_vol >= MIN_AVG_VOLUME)
    )

    return pd.DataFrame({
        "price": x.s,
        "rs_score": rs_score,
        "price_tt": price_tt,
        "price_tt_met": price_met,
        "rs_tt": rs_tt,
        "rs_tt_met": rs_met,
        "liquid": liquid,
        "avg_volume": avg_vol
    })

# =========================
# DOWNLOAD BENCHMARK
# =========================

def get_benchmark(start, end):

    for ticker in [BENCHMARK, BENCHMARK_FALLBACK]:

        try:
            d = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if not d.empty:
                c = d["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:,0]
                return c.dropna().sort_index()

        except Exception as e:
            print("Benchmark error:", ticker, e)

    raise RuntimeError("Benchmark download failed")

# =========================
# DOWNLOAD STOCKS
# =========================

def build_signals(tickers, start, end, benchmark):

    signals = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):

        batch = tickers[i:i+batch_size]

        print(
            f"Downloading {i+1}-"
            f"{i+len(batch)} / {len(tickers)}"
        )

        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )
        except Exception as e:
            print("Batch failed:", e)
            continue

        for sym in batch:

            try:

                d = data if len(batch) == 1 else (
                    data[sym]
                    if sym in data.columns.get_level_values(0)
                    else pd.DataFrame()
                )

                if d.empty or "Close" not in d.columns:
                    continue

                close = d["Close"].dropna().sort_index()
                volume = d["Volume"].reindex(close.index).fillna(0)

                sig = stock_signals(
                    close,
                    volume,
                    benchmark
                )

                if sig is not None:
                    signals[sym.replace(".NS","")] = sig

            except Exception:
                continue

        time.sleep(0.5)

    return signals

# =========================
# BACKTEST
# =========================

def run_backtest(signals, days):

    cash = STARTING_CAPITAL
    holdings = {}
    trades = []
    equity = []
    ranking_log = []

    for date in days:

        # -------------------------
        # ENTRY-ELIGIBLE UNIVERSE
        # SAME AS LIVE SCREENER
        # -------------------------

        candidates = []

        for sym, df in signals.items():

            if date not in df.index:
                continue

            r = df.loc[date]

            if pd.isna(r.rs_score):
                continue

            if not bool(r.price_tt):
                continue

            if not bool(r.rs_tt):
                continue

            if not bool(r.liquid):
                continue

            candidates.append(
                (sym, float(r.rs_score))
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # IMPORTANT:
        # Rank is the rank among eligible stocks,
        # exactly like the live screener.

        rank = {
            sym: i
            for i, (sym, _) in enumerate(
                candidates, 1
            )
        }

        top10 = [
            sym for sym, _ in candidates[:TOP_N]
        ]

        # -------------------------
        # DAILY RANKING AUDIT
        # -------------------------

        for i, (sym, score) in enumerate(
            candidates,
            1
        ):

            r = signals[sym].loc[date]

            ranking_log.append({
                "date": date.strftime("%Y-%m-%d"),
                "rank": i,
                "symbol": sym,
                "rs_score": round(score,4),
                "price": round(float(r.price),2),
                "price_tt_met": int(r.price_tt_met),
                "rs_tt_met": int(r.rs_tt_met),
                "avg_volume_50d": round(float(r.avg_volume),0),
                "action": (
                    "BUY ENTRY" if i <= TOP_N
                    else "HOLD ALLOWED" if i <= EXIT_RANK
                    else "WATCHLIST ONLY"
                )
            })

        # -------------------------
        # EXIT
        # -------------------------

        for sym in list(holdings):

            if date not in signals[sym].index:
                continue

            r = signals[sym].loc[date]
            current_rank = rank.get(sym, 999999)

            if current_rank <= EXIT_RANK:
                continue

            pos = holdings.pop(sym)

            price = float(r.price)
            gross = pos["qty"] * price
            scost = sell_cost(gross)
            proceeds = gross - scost

            basis = (
                pos["qty"] * pos["entry_price"]
                + pos["buy_cost"]
            )

            gain = proceeds - basis
            tax = stcg(gain)
            pnl = gain - tax

            cash += proceeds - tax

            trades.append({
                "symbol": sym,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": date.strftime("%Y-%m-%d"),
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"],2),
                "exit_price": round(price,2),
                "entry_rank": pos["entry_rank"],
                "exit_rank": current_rank,
                "entry_rs_score": pos["entry_score"],
                "exit_rs_score": round(float(r.rs_score),4),
                "gross_return_pct": round(
                    (price/pos["entry_price"]-1)*100,2
                ),
                "buy_cost_rs": round(pos["buy_cost"],2),
                "sell_cost_rs": round(scost,2),
                "stcg_tax_rs": round(tax,2),
                "net_pnl_rs": round(pnl,2),
                "net_return_pct": round(
                    pnl/basis*100,2
                ),
                "days_held": (date-pos["entry_date"]).days,
                "exit_reason": "RS RANK > 15"
            })

        # -------------------------
        # PORTFOLIO VALUE
        # AFTER EXITS
        # -------------------------

        value = cash

        for sym, pos in holdings.items():

            if date in signals[sym].index:
                price = float(signals[sym].loc[date].price)
            else:
                price = pos["entry_price"]

            value += pos["qty"] * price

        # -------------------------
        # NEW ENTRIES
        # -------------------------

        slots = TOP_N - len(holdings)

        if slots > 0:

            target = value / TOP_N

            for sym in top10:

                if slots <= 0:
                    break

                if sym in holdings:
                    continue

                r = signals[sym].loc[date]
                price = float(r.price)

                qty = int(target // price)

                if qty < 1:
                    continue

                trade_value = qty * price
                bcost = buy_cost(trade_value)

                if trade_value + bcost > cash:
                    continue

                cash -= trade_value + bcost

                holdings[sym] = {
                    "qty": qty,
                    "entry_price": price,
                    "entry_date": date,
                    "buy_cost": bcost,
                    "entry_rank": rank[sym],
                    "entry_score": round(
                        float(r.rs_score),4
                    )
                }

                slots -= 1

        # -------------------------
        # FINAL EQUITY
        # -------------------------

        value = cash

        for sym, pos in holdings.items():

            if date in signals[sym].index:
                price = float(signals[sym].loc[date].price)
            else:
                price = pos["entry_price"]

            value += pos["qty"] * price

        equity.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(value,2),
            "equity": round(value/STARTING_CAPITAL,8),
            "return_pct": round(
                (value/STARTING_CAPITAL-1)*100,4
            ),
            "cash_rs": round(cash,2),
            "n_holdings": len(holdings),
            "top10": ",".join(top10),
            "holdings": ",".join(sorted(holdings))
        })

    # -------------------------
    # MARK OPEN POSITIONS
    # -------------------------

    if days:

        last = days[-1]

        for sym, pos in holdings.items():

            r = signals[sym].loc[last]
            price = float(r.price)

            gross = pos["qty"] * price
            scost = sell_cost(gross)
            proceeds = gross - scost

            basis = (
                pos["qty"] * pos["entry_price"]
                + pos["buy_cost"]
            )

            gain = proceeds - basis
            tax = stcg(gain)
            pnl = gain - tax

            trades.append({
                "symbol": sym,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": last.strftime("%Y-%m-%d") + " OPEN",
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"],2),
                "exit_price": round(price,2),
                "entry_rank": pos["entry_rank"],
                "exit_rank": rank.get(sym,""),
                "entry_rs_score": pos["entry_score"],
                "exit_rs_score": round(float(r.rs_score),4),
                "gross_return_pct": round(
                    (price/pos["entry_price"]-1)*100,2
                ),
                "buy_cost_rs": round(pos["buy_cost"],2),
                "sell_cost_rs": round(scost,2),
                "stcg_tax_rs": round(tax,2),
                "net_pnl_rs": round(pnl,2),
                "net_return_pct": round(
                    pnl/basis*100,2
                ),
                "days_held": (last-pos["entry_date"]).days,
                "exit_reason": "BACKTEST END"
            })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity),
        pd.DataFrame(ranking_log)
    )

# =========================
# SUMMARY
# =========================

def summary(trades, equity):

    if equity.empty:
        return {}

    final = float(
        equity.portfolio_value_rs.iloc[-1]
    )

    total_return = (
        final/STARTING_CAPITAL-1
    ) * 100

    peak = equity.equity.cummax()

    dd = (
        equity.equity/peak-1
    ) * 100

    max_dd = dd.min()

    closed = (
        trades[
            ~trades.exit_date.astype(str).str.contains("OPEN")
        ]
        if not trades.empty else trades
    )

    n = len(closed)

    if n:
        win = (closed.net_return_pct > 0).mean()*100
        avg = closed.net_return_pct.mean()
        median = closed.net_return_pct.median()
        avg_days = closed.days_held.mean()
        best = closed.net_return_pct.max()
        worst = closed.net_return_pct.min()
        costs = (
            closed.buy_cost_rs.sum()
            + closed.sell_cost_rs.sum()
        )
        tax = closed.stcg_tax_rs.sum()

        winners = closed[closed.net_pnl_rs > 0]
        losers = closed[closed.net_pnl_rs < 0]

        pf = (
            winners.net_pnl_rs.sum()
            / abs(losers.net_pnl_rs.sum())
            if len(losers) else 0
        )
    else:
        win=avg=median=avg_days=best=worst=costs=tax=pf=0

    daily = equity.equity.pct_change().dropna()

    ann_return = (
        equity.equity.iloc[-1]
        ** (252/len(equity))
        - 1
        if len(equity) else 0
    )

    vol = (
        daily.std()*np.sqrt(252)
        if len(daily) else 0
    )

    sharpe = (
        daily.mean()/daily.std()*np.sqrt(252)
        if len(daily) and daily.std() > 0 else 0
    )

    return {
        "run_timestamp_ist":
            datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
                "%Y-%m-%d %H:%M:%S IST"
            ),
        "backtest_start": BACKTEST_START,
        "backtest_end": BACKTEST_END,
        "starting_capital_rs": STARTING_CAPITAL,
        "final_portfolio_value_rs": round(final,2),
        "net_total_return_pct": round(total_return,2),
        "annualized_return_pct": round(ann_return*100,2),
        "annualized_volatility_pct": round(vol*100,2),
        "max_drawdown_pct": round(max_dd,2),
        "sharpe": round(sharpe,3),
        "n_closed_trades": n,
        "win_rate_net_pct": round(win,1),
        "avg_net_return_trade_pct": round(avg,2),
        "median_net_return_trade_pct": round(median,2),
        "avg_days_held": round(avg_days,1),
        "best_net_trade_pct": round(best,2),
        "worst_net_trade_pct": round(worst,2),
        "profit_factor": round(pf,3),
        "transaction_costs_rs": round(costs,2),
        "stcg_tax_rs": round(tax,2)
    }

# =========================
# GOOGLE SHEETS
# =========================

def write_sheets(trades, equity, ranking, result):

    sid = os.environ.get(SHEET_ID_ENV)
    cred = os.environ.get(CREDS_ENV)

    if not sid or not cred:
        trades.to_csv("backtest_trades.csv",index=False)
        equity.to_csv("backtest_equity.csv",index=False)
        ranking.to_csv("backtest_daily_ranking.csv",index=False)
        pd.DataFrame([result]).to_csv(
            "backtest_summary.csv",index=False
        )
        return

    credentials = Credentials.from_service_account_info(
        json.loads(cred),
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

    sh = gspread.authorize(credentials).open_by_key(sid)

    # -------------------------
    # SUMMARY + PARAMETERS
    # -------------------------

    try:
        ws = sh.worksheet(SUMMARY_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=SUMMARY_SHEET,
            rows=100,
            cols=10
        )

    ws.clear()

    params = {
        "BACKTEST_START": BACKTEST_START,
        "BACKTEST_END": BACKTEST_END,
        "RUN_TIMESTAMP": result["run_timestamp_ist"],
        "BENCHMARK": BENCHMARK,
        "STARTING_CAPITAL": STARTING_CAPITAL,
        "TOP_N": TOP_N,
        "EXIT_RANK": EXIT_RANK,
        "MIN_PRICE": MIN_PRICE,
        "MIN_AVG_VOLUME": MIN_AVG_VOLUME,
        "VOLUME_LOOKBACK": VOLUME_LOOKBACK,
        "PRICE_TT": "7/7",
        "RS_LINE_TT": "7/7",
        "RS_SCORE": "40% 63D + 20% 126D + 20% 189D + 20% 252D",
        "ENTRY": "Top 10 eligible stocks",
        "HOLD": "Rank 1-15",
        "EXIT": "Rank >15",
        "BLUE_DOT": "NO",
        "GREEN_DOT": "NO",
        "VCP": "NO",
        "STOP_LOSS": "NO",
        "TRAILING_STOP": "NO",
        "RS_20EMA_EXIT": "NO",
        "EXECUTION": "Daily EOD close",
        "CALCULATION": "Python",
        "SHEET_ROLE": "Output only"
    }

    rows = [["PARAMETER","VALUE"]]
    rows += [[k,v] for k,v in params.items()]
    rows += [["",""]]
    rows += [["PERFORMANCE","VALUE"]]
    rows += [[k,v] for k,v in result.items()]

    ws.update(rows,"A1")

    # -------------------------
    # BACKTEST DATA
    # -------------------------

    try:
        wb = sh.worksheet(BACKTEST_SHEET)
    except gspread.WorksheetNotFound:
        wb = sh.add_worksheet(
            title=BACKTEST_SHEET,
            rows=1000,
            cols=20
        )

    wb.clear()

    row = 1

    def put(title, df):

        nonlocal row

        wb.update([[title]],f"A{row}")
        row += 1

        if not df.empty:

            values = [
                list(df.columns)
            ] + df.fillna("").values.tolist()

            wb.update(values,f"A{row}")
            row += len(values) + 2
        else:
            row += 2

    put("TRADE LOG", trades)
    put("DAILY EQUITY CURVE", equity)
    put("DAILY ELIGIBLE RANKING", ranking)

# =========================
# MAIN
# =========================

def main():

    started = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    print(
        f"\nBACKTEST START: "
        f"{started.strftime('%Y-%m-%d %H:%M:%S IST')}"
    )

    tickers = load_tickers()

    download_start = (
        pd.Timestamp(BACKTEST_START)
        - pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    ).strftime("%Y-%m-%d")

    download_end = (
        pd.Timestamp(BACKTEST_END)
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    benchmark = get_benchmark(
        download_start,
        download_end
    )

    signals = build_signals(
        tickers,
        download_start,
        download_end,
        benchmark
    )

    days = benchmark.index[
        (benchmark.index >= pd.Timestamp(BACKTEST_START)) &
        (benchmark.index <= pd.Timestamp(BACKTEST_END))
    ]

    print(
        f"Stocks: {len(signals)} | "
        f"Days: {len(days)} | "
        f"End: {BACKTEST_END}"
    )

    trades, equity, ranking = run_backtest(
        signals,
        days
    )

    result = summary(
        trades,
        equity
    )

    print("\nFINAL RESULTS")
    for k,v in result.items():
        print(f"{k}: {v}")

    write_sheets(
        trades,
        equity,
        ranking,
        result
    )

    finished = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    print(
        f"\nBACKTEST COMPLETE: "
        f"{finished.strftime('%Y-%m-%d %H:%M:%S IST')}"
    )


if __name__ == "__main__":
    main()