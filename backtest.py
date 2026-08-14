import os, json, time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials


# ================= CONFIG =================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = pd.Timestamp.today().normalize()

STARTING_CAPITAL = 1_000_000

TOP_N = 10
RS_EXIT_RANK = 15

MIN_PRICE = 10
MIN_AVG_VOLUME = 50_000
VOLUME_LOOKBACK = 50

DOWNLOAD_YEARS_BEFORE_START = 3

# Costs
ENABLE_COSTS = True
STT_BUY = 0.001
STT_SELL = 0.001
STAMP = 0.00015
EXCHANGE = 0.0000325
SEBI = 0.000001
GST = 0.18
DP_CHARGE = 20

# STCG
ENABLE_STCG = True
STCG_RATE = 0.20
CESS = 0.04
STCG_EFFECTIVE = STCG_RATE * (1 + CESS)

# Google Sheets
SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_SHEET = "Backtest"
SUMMARY_SHEET = "Backtest_Summary"


# ================= COSTS =================

def buy_cost(v):
    if not ENABLE_COSTS:
        return 0
    exchange = EXCHANGE * v
    sebi = SEBI * v
    return (
        STT_BUY * v +
        STAMP * v +
        exchange +
        sebi +
        GST * (exchange + sebi)
    )


def sell_cost(v):
    if not ENABLE_COSTS:
        return 0
    exchange = EXCHANGE * v
    sebi = SEBI * v
    return (
        STT_SELL * v +
        exchange +
        sebi +
        GST * (exchange + sebi) +
        DP_CHARGE
    )


def stcg(gain):
    if not ENABLE_STCG or gain <= 0:
        return 0
    return gain * STCG_EFFECTIVE


# ================= DATA =================

def load_tickers():
    df = pd.read_csv(STOCKS_FILE)
    symbols = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return [
        s if s.endswith(".NS") else s + ".NS"
        for s in symbols if s
    ]


def download_dates():
    start = (
        pd.Timestamp(BACKTEST_START)
        - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    )
    end = BACKTEST_END + pd.Timedelta(days=1)
    return (
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d")
    )


def download_benchmark():
    start, end = download_dates()

    for ticker in [BENCHMARK, BENCHMARK_FALLBACK]:
        try:
            d = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if d.empty:
                continue

            c = d["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]

            c = c.dropna().sort_index()

            if len(c):
                print("Benchmark:", ticker)
                return c

        except Exception as e:
            print("Benchmark error:", e)

    raise RuntimeError("Benchmark download failed")


# ================= TREND TEMPLATE =================

def trend_template(s):
    sma50 = s.rolling(50).mean()
    sma150 = s.rolling(150).mean()
    sma200 = s.rolling(200).mean()

    low52 = s.rolling(252).min()
    high52 = s.rolling(252).max()

    conditions = [
        (s > sma150) & (s > sma200),
        sma150 > sma200,
        sma200 > sma200.shift(21),
        (sma50 > sma150) & (sma50 > sma200),
        s > sma50,
        s >= 1.25 * low52,
        s >= 0.75 * high52
    ]

    met = sum(x.astype(int) for x in conditions)

    return met == 7, met


# ================= SIGNALS =================

def stock_signals(close, volume, benchmark):
    x = pd.concat(
        [close, benchmark],
        axis=1,
        join="inner"
    ).dropna()

    x.columns = ["price", "benchmark"]

    if len(x) < 280:
        return None

    rs_line = x.price / x.benchmark

    def ret(n):
        return x.price / x.price.shift(n) - 1

    rs_score = (
        0.40 * ret(63) +
        0.20 * ret(126) +
        0.20 * ret(189) +
        0.20 * ret(252)
    ) * 100

    tt_pass, tt_met = trend_template(x.price)
    rs_pass, rs_met = trend_template(rs_line)

    avg_volume = (
        volume.reindex(x.index)
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    liquid = (
        (x.price >= MIN_PRICE) &
        (avg_volume > MIN_AVG_VOLUME)
    )

    return pd.DataFrame({
        "price": x.price,
        "rs_score": rs_score,
        "tt_pass": tt_pass,
        "tt_met": tt_met,
        "rs_tt_pass": rs_pass,
        "rs_tt_met": rs_met,
        "avg_volume": avg_volume,
        "liquid": liquid
    })


# ================= BACKTEST =================

def run_backtest(signals, days):
    cash = STARTING_CAPITAL
    holdings = {}
    trades = []
    equity = []
    selections = []

    for date in days:

        # ---- Raw RS ranking ----
        pool = []

        for sym, df in signals.items():
            if date not in df.index:
                continue

            r = df.loc[date]

            if pd.notna(r.rs_score):
                pool.append(
                    (sym, float(r.rs_score))
                )

        pool.sort(key=lambda x: x[1], reverse=True)

        ranks = {
            sym: i
            for i, (sym, _) in enumerate(pool, 1)
        }

        # ---- Entry candidates ----
        candidates = []

        for sym, score in pool:
            r = signals[sym].loc[date]

            if (
                bool(r.tt_pass) and
                bool(r.rs_tt_pass) and
                bool(r.liquid)
            ):
                candidates.append((sym, score))

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top10 = [x[0] for x in candidates[:TOP_N]]

        for rank, (sym, score) in enumerate(
            candidates[:TOP_N], 1
        ):
            r = signals[sym].loc[date]

            selections.append({
                "date": date.strftime("%Y-%m-%d"),
                "entry_rank": rank,
                "symbol": sym,
                "rs_score": round(score, 4),
                "overall_rs_rank": ranks.get(sym, 999999),
                "price": round(float(r.price), 2),
                "tt_met": int(r.tt_met),
                "rs_tt_met": int(r.rs_tt_met),
                "avg_volume": round(float(r.avg_volume), 0)
            })

        # ---- Exits ----
        for sym in list(holdings):
            r = signals[sym].loc[date]
            rank = ranks.get(sym, 999999)

            if rank <= RS_EXIT_RANK:
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

            trades.append({
                "symbol": sym,
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": date.strftime("%Y-%m-%d"),
                "qty": pos["qty"],
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(price, 2),
                "entry_rs_rank": pos["entry_rank"],
                "entry_rs_score": pos["entry_score"],
                "exit_rs_rank": rank,
                "exit_rs_score": round(float(r.rs_score), 4),
                "gross_return_pct": round(
                    (price / pos["entry_price"] - 1) * 100, 2
                ),
                "buy_cost_rs": round(pos["buy_cost"], 2),
                "sell_cost_rs": round(scost, 2),
                "stcg_tax_rs": round(tax, 2),
                "net_pnl_rs": round(pnl, 2),
                "net_return_pct": round(
                    pnl / basis * 100, 2
                ),
                "days_held": (date - pos["entry_date"]).days,
                "exit_reason": "RS RANK > 15"
            })

            cash += proceeds - tax

        # ---- Portfolio value before entries ----
        portfolio = cash

        for sym, pos in holdings.items():
            portfolio += (
                pos["qty"] *
                float(signals[sym].loc[date, "price"])
            )

        # ---- New entries ----
        slots = TOP_N - len(holdings)

        if slots > 0:
            target = portfolio / TOP_N

            for sym in top10:
                if slots <= 0:
                    break

                if sym in holdings:
                    continue

                price = float(
                    signals[sym].loc[date, "price"]
                )

                qty = int(target // price)

                if qty < 1:
                    continue

                value = qty * price
                cost = buy_cost(value)

                if value + cost > cash:
                    continue

                cash -= value + cost

                holdings[sym] = {
                    "qty": qty,
                    "entry_price": price,
                    "entry_date": date,
                    "buy_cost": cost,
                    "entry_rank": ranks.get(sym, 999999),
                    "entry_score": round(
                        float(signals[sym].loc[date, "rs_score"]),
                        4
                    )
                }

                slots -= 1

        # ---- Final mark ----
        portfolio = cash

        for sym, pos in holdings.items():
            portfolio += (
                pos["qty"] *
                float(signals[sym].loc[date, "price"])
            )

        equity.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(portfolio, 2),
            "equity": round(
                portfolio / STARTING_CAPITAL, 8
            ),
            "return_pct": round(
                (portfolio / STARTING_CAPITAL - 1) * 100,
                4
            ),
            "cash_rs": round(cash, 2),
            "n_holdings": len(holdings),
            "top10_buy_candidates": ",".join(top10),
            "holdings": ",".join(sorted(holdings))
        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity),
        pd.DataFrame(selections)
    )


# ================= SUMMARY =================

def summarize(trades, equity):
    if equity.empty:
        return {}

    final_value = equity.portfolio_value_rs.iloc[-1]

    returns = equity.equity.pct_change().dropna()

    running_max = equity.equity.cummax()

    dd = equity.equity / running_max - 1

    max_dd = dd.min()

    annual_return = (
        equity.equity.iloc[-1] **
        (252 / len(equity)) - 1
    )

    annual_vol = (
        returns.std() * np.sqrt(252)
        if len(returns) else 0
    )

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if len(returns) and returns.std() > 0
        else 0
    )

    downside = returns[returns < 0]

    sortino = (
        returns.mean() / downside.std() * np.sqrt(252)
        if len(downside) and downside.std() > 0
        else 0
    )

    calmar = (
        annual_return / abs(max_dd)
        if max_dd < 0 else 0
    )

    closed = trades[
        ~trades.exit_date.astype(str).str.contains(
            "OPEN", na=False
        )
    ] if not trades.empty else trades

    if len(closed):
        winners = closed[closed.net_return_pct > 0]
        losers = closed[closed.net_return_pct < 0]

        gross_win_rate = (
            (closed.gross_return_pct > 0).mean() * 100
        )

        net_win_rate = (
            (closed.net_return_pct > 0).mean() * 100
        )

        avg_gross = closed.gross_return_pct.mean()
        avg_net = closed.net_return_pct.mean()
        median_net = closed.net_return_pct.median()

        avg_days = closed.days_held.mean()
        median_days = closed.days_held.median()

        best = closed.gross_return_pct.max()
        worst = closed.gross_return_pct.min()

        buy_costs = closed.buy_cost_rs.sum()
        sell_costs = closed.sell_cost_rs.sum()
        tax = closed.stcg_tax_rs.sum()

        winning_pnl = winners.net_pnl_rs.sum()
        losing_pnl = abs(losers.net_pnl_rs.sum())

        profit_factor = (
            winning_pnl / losing_pnl
            if losing_pnl > 0 else 0
        )

    else:
        gross_win_rate = net_win_rate = 0
        avg_gross = avg_net = median_net = 0
        avg_days = median_days = 0
        best = worst = 0
        buy_costs = sell_costs = tax = 0
        profit_factor = 0

    return {
        "run_timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "backtest_start": BACKTEST_START,
        "backtest_end": str(BACKTEST_END.date()),
        "starting_capital_rs": STARTING_CAPITAL,
        "final_portfolio_value_rs": round(final_value, 2),
        "net_total_return_pct": round(
            (final_value / STARTING_CAPITAL - 1) * 100, 2
        ),
        "annualized_return_pct": round(
            annual_return * 100, 2
        ),
        "annualized_volatility_pct": round(
            annual_vol * 100, 2
        ),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "closed_trades": len(closed),
        "gross_win_rate_pct": round(gross_win_rate, 1),
        "net_win_rate_pct": round(net_win_rate, 1),
        "avg_gross_trade_pct": round(avg_gross, 2),
        "avg_net_trade_pct": round(avg_net, 2),
        "median_net_trade_pct": round(median_net, 2),
        "avg_days_held": round(avg_days, 1),
        "median_days_held": round(median_days, 1),
        "best_gross_trade_pct": round(best, 2),
        "worst_gross_trade_pct": round(worst, 2),
        "profit_factor_net": round(profit_factor, 3),
        "total_buy_costs_rs": round(buy_costs, 2),
        "total_sell_costs_rs": round(sell_costs, 2),
        "total_transaction_costs_rs": round(
            buy_costs + sell_costs, 2
        ),
        "total_stcg_tax_rs": round(tax, 2),
        "top_n": TOP_N,
        "exit_rank": RS_EXIT_RANK,
        "min_price": MIN_PRICE,
        "min_avg_volume": MIN_AVG_VOLUME,
        "volume_lookback": VOLUME_LOOKBACK,
        "entry_rule": (
            "Price TT 7/7 + RS Line TT 7/7 + "
            "50D Avg Volume > 50,000; Top 10 Raw RS"
        ),
        "exit_rule": "RS Rank > 15",
        "execution": "Daily EOD close",
        "position_sizing": "Equal target weighting",
        "stcg_effective_rate_pct": STCG_EFFECTIVE * 100
    }


# ================= GOOGLE SHEETS =================

def write_sheet(trades, equity, selections, summary):

    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        raise RuntimeError(
            "SHEET_ID or GOOGLE_CREDENTIALS missing"
        )

    creds = Credentials.from_service_account_info(
        json.loads(creds_json),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets"
        ]
    )

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # ---- Summary ----
    try:
        ws = sh.worksheet(SUMMARY_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=SUMMARY_SHEET,
            rows=100,
            cols=30
        )

    ws.clear()

    summary_df = pd.DataFrame([summary])

    ws.update(
        [list(summary_df.columns)] +
        summary_df.astype(object).values.tolist(),
        "A1"
    )

    # ---- Main sheet ----
    try:
        ws = sh.worksheet(BACKTEST_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=BACKTEST_SHEET,
            rows=1000,
            cols=30
        )

    ws.clear()

    row = 1

    def write_df(title, df):
        nonlocal row

        ws.update([[title]], f"A{row}")
        row += 1

        if not df.empty:
            values = (
                [list(df.columns)] +
                df.astype(object).values.tolist()
            )

            ws.update(values, f"A{row}")
            row += len(values) + 2
        else:
            row += 2

    write_df("TRADE LOG", trades)
    write_df("DAILY EQUITY CURVE", equity)
    write_df("DAILY TOP-10 AUDIT", selections)

    print("Google Sheets updated.")


# ================= MAIN =================

def main():

    print("=" * 60)
    print("RS BACKTEST")
    print("=" * 60)
    print("Run:", datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ))
    print("Start:", BACKTEST_START)
    print("End:", BACKTEST_END.date())

    tickers = load_tickers()
    start, end = download_dates()

    benchmark = download_benchmark()

    signals = {}

    for i in range(0, len(tickers), 50):

        batch = tickers[i:i + 50]

        print(
            f"Downloading {i + 1}-{i + len(batch)} "
            f"/ {len(tickers)}"
        )

        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True
            )
        except Exception as e:
            print("Batch failed:", e)
            continue

        for symbol in batch:
            try:
                sdata = (
                    data if len(batch) == 1
                    else data[symbol]
                )

                if "Close" not in sdata:
                    continue

                close = sdata["Close"].dropna()
                volume = (
                    sdata["Volume"]
                    .reindex(close.index)
                    .fillna(0)
                )

                sig = stock_signals(
                    close,
                    volume,
                    benchmark
                )

                if sig is not None:
                    signals[
                        symbol.replace(".NS", "")
                    ] = sig

            except Exception:
                continue

        time.sleep(1)

    days = benchmark.index[
        (benchmark.index >= pd.Timestamp(BACKTEST_START)) &
        (benchmark.index <= BACKTEST_END)
    ]

    print("Stocks:", len(signals))
    print("Trading days:", len(days))

    trades, equity, selections = run_backtest(
        signals,
        days
    )

    summary = summarize(
        trades,
        equity
    )

    print("\nFINAL RESULTS")
    for k, v in summary.items():
        print(f"{k}: {v}")

    write_sheet(
        trades,
        equity,
        selections,
        summary
    )

    print("\nDONE")


if __name__ == "__main__":
    main()