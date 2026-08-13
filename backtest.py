"""
RS Screener Backtest - FINAL SIMPLIFIED MODEL

Implements exactly this rule set, nothing more:

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
  1Y RS cross     : same underlying signal as Blue Dot, diagnostic only
  Green Dot       : diagnostic only, NOT a filter
  RS 5-EMA exit   : REMOVED
  Price stop      : NONE
  Rank-buffer exit: NONE
  Exit rule       : leaves the current Top 10 -- nothing else
  Buy costs       : included (STT, stamp duty, exchange, SEBI, GST)
  Sell costs      : included (STT, exchange, SEBI, GST)
  DP charge       : Rs.20 per sell transaction
  STCG            : 20.8% effective on positive realized gains
  Tax accounting  : FIFO lots (trivially satisfied -- this model never holds
                    more than one open lot per symbol at a time, since a
                    symbol is fully exited before it can be re-entered)
  Equity          : daily mark-to-market
  Terminal value  : reported BOTH as marked (last-close valuation) AND as
                    liquidation value (net of hypothetical sell costs + tax
                    on any still-open positions)
  Equity chart    : generated (native Google Sheets line chart)
  Drawdown chart  : generated (native Google Sheets line chart)

No regime filter, no circuit breaker, no position-sizing multiplier beyond
equal weight, no Blue Dot / RS-EMA / rank-buffer as ENTRY or EXIT logic --
those signals are computed and shown for information only.

IMPORTANT DATE LOGIC:
  BACKTEST_START : fixed
  BACKTEST_END   : automatically derived from the last benchmark date
                   actually returned by Yahoo Finance.

  All Yahoo Finance downloads are open-ended. No hardcoded end date.
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

# ------------------------------------------------------------
# BACKTEST_END IS DERIVED DYNAMICALLY FROM BENCHMARK DATA.
# DO NOT HARD-CODE IT.
# ------------------------------------------------------------
BACKTEST_END = None


# ---- Filters ----

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250


# ---- Data sanity cleaning ----
#
# Repairs implausible single-day price jumps caused by:
#   - unadjusted splits
#   - bonuses
#   - mergers
#   - ticker reuse
#
# BEFORE any calculation touches the data.
#

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ---- Portfolio ----

TOP_N = 10
STARTING_CAPITAL = 1_000_000


# ---- Costs ----
#
# Zerodha delivery:
# zero brokerage, statutory charges only.
#

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18

DP_CHARGE_FLAT = 20


# ---- STCG tax ----

STCG_RATE = 0.20
STCG_CESS = 0.04

STCG_EFFECTIVE_RATE = (
    STCG_RATE * (1 + STCG_CESS)
)


# ---- Google Sheets ----

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest"


# ============================================================
# DOWNLOAD DATE LOGIC
# ============================================================

def get_download_dates():
    """
    Returns only the download START date.

    There is deliberately NO download end date.

    Yahoo Finance is therefore allowed to return the latest
    available market data.
    """

    backtest_start = pd.Timestamp(BACKTEST_START)

    download_start = (
        backtest_start
        - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    )

    return download_start.strftime("%Y-%m-%d")


# ============================================================
# LOAD TICKERS
# ============================================================

def load_tickers():

    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(
            f"Could not find {STOCKS_FILE}"
        )

    df = pd.read_csv(STOCKS_FILE)

    if "symbol" not in df.columns:
        raise ValueError(
            "stocks.csv must contain a column named 'symbol'."
        )

    symbols = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    symbols = [s for s in symbols if s]

    return [
        s if s.endswith(".NS") else s + ".NS"
        for s in symbols
    ]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_price_series(close):
    """
    Repairs implausible single-day price jumps.

    If the day-over-day movement exceeds
    MAX_PLAUSIBLE_DAILY_MOVE, the suspicious value is replaced
    by the previous valid close.

    Applied once before calculations.
    """

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


# ============================================================
# BUY COST
# ============================================================

def buy_side_cost(trade_value):

    stt = STT_RATE * trade_value

    stamp = STAMP_DUTY_RATE * trade_value

    exch = EXCHANGE_CHARGE_RATE * trade_value

    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return (
        stt
        + stamp
        + exch
        + sebi
        + gst
    )


# ============================================================
# SELL COST
# ============================================================

def sell_side_cost(trade_value):

    stt = STT_RATE * trade_value

    exch = EXCHANGE_CHARGE_RATE * trade_value

    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return (
        stt
        + exch
        + sebi
        + gst
        + DP_CHARGE_FLAT
    )


# ============================================================
# STCG TAX
# ============================================================

def stcg_tax(net_gain):
    """
    20.8% effective tax on positive realized gain only.

    No tax on losses.
    """

    if net_gain <= 0:
        return 0.0

    return (
        net_gain
        * STCG_EFFECTIVE_RATE
    )


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():
    """
    Downloads benchmark data from the warm-up period through
    the latest data actually available from Yahoo Finance.

    There is NO end date.

    BACKTEST_END is then derived from the final benchmark
    observation actually returned.
    """

    download_start = get_download_dates()

    print(
        f"\nBenchmark download: "
        f"{download_start} to latest available"
    )

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            data = yf.download(
                ticker,
                start=download_start,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                continue

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = (
                close
                .dropna()
                .sort_index()
            )

            if close.empty:
                continue

            close, n_bad = clean_price_series(
                close
            )

            if n_bad:

                print(
                    f"Benchmark {ticker}: "
                    f"repaired {n_bad} "
                    f"implausible data point(s)"
                )

            # ------------------------------------------------
            # IMPORTANT:
            # Derive BACKTEST_END from actual benchmark data.
            # ------------------------------------------------

            global BACKTEST_END

            BACKTEST_END = (
                close.index[-1]
                .strftime("%Y-%m-%d")
            )

            print(
                f"Benchmark loaded: {ticker}"
            )

            print(
                f"Derived BACKTEST_END: "
                f"{BACKTEST_END}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: {e}"
            )

    raise RuntimeError(
        "Could not download any benchmark index data."
    )


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template_series(s):
    """
    Minervini 7-point Trend Template.

    Works for:
      - price
      - RS ratio
    """

    sma50 = s.rolling(50).mean()

    sma150 = s.rolling(150).mean()

    sma200 = s.rolling(200).mean()

    sma200_1mo = sma200.shift(21)

    low52 = s.rolling(252).min()

    high52 = s.rolling(252).max()


    c1 = (
        (s > sma150)
        & (s > sma200)
    )

    c2 = sma150 > sma200

    c3 = sma200 > sma200_1mo

    c4 = (
        (sma50 > sma150)
        & (sma50 > sma200)
    )

    c5 = s > sma50

    c6 = s >= 1.25 * low52

    c7 = s >= 0.75 * high52


    met = (
        c1.astype(int)
        + c2.astype(int)
        + c3.astype(int)
        + c4.astype(int)
        + c5.astype(int)
        + c6.astype(int)
        + c7.astype(int)
    )

    return met == 7


# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
    bench_close
):
    """
    Computes:

      RS Score
      Price Trend Template
      RS Line Trend Template
      Liquidity
      Blue Dot
      Green Dot

    Blue/Green diagnostics do NOT affect trading.
    """

    aligned = pd.concat(
        [close, bench_close],
        axis=1,
        join="inner"
    ).dropna()

    aligned.columns = [
        "s",
        "b"
    ]

    if len(aligned) < 280:
        return None

    volume = volume.reindex(
        aligned.index
    )

    rs_ratio = (
        aligned["s"]
        / aligned["b"]
    )


    # --------------------------------------------------------
    # Percentage return
    # --------------------------------------------------------

    def pct_return(series, days):

        return (
            series
            / series.shift(days)
            - 1
        )


    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

    rs_score = (
        0.40
        * pct_return(
            aligned["s"],
            63
        )

        + 0.20
        * pct_return(
            aligned["s"],
            126
        )

        + 0.20
        * pct_return(
            aligned["s"],
            189
        )

        + 0.20
        * pct_return(
            aligned["s"],
            252
        )
    ) * 100


    # --------------------------------------------------------
    # BLUE DOT / 1Y RS CROSS
    # Diagnostic only.
    # --------------------------------------------------------

    previous_rs_high = (
        rs_ratio
        .shift(1)
        .rolling(LOOKBACK_DAYS)
        .max()
    )

    blue_dot = (
        rs_ratio
        > previous_rs_high
    )


    # --------------------------------------------------------
    # GREEN DOT
    # Diagnostic only.
    # --------------------------------------------------------

    previous_price_high = (
        aligned["s"]
        .shift(1)
        .rolling(LOOKBACK_DAYS)
        .max()
    )

    price_at_new_high = (
        aligned["s"]
        > previous_price_high
    )

    green_dot = (
        blue_dot
        & (~price_at_new_high)
    )


    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
    # --------------------------------------------------------

    tt_pass = trend_template_series(
        aligned["s"]
    )


    # --------------------------------------------------------
    # RS LINE TREND TEMPLATE
    # --------------------------------------------------------

    rs_tt_pass = trend_template_series(
        rs_ratio
    )


    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    rolling_avg_volume = (
        volume
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    liquid = (
        (aligned["s"] > MIN_PRICE)
        &
        (
            rolling_avg_volume
            > MIN_AVG_VOLUME
        )
    )


    return pd.DataFrame({

        "price": aligned["s"],

        "rs_score": rs_score,

        "tt_pass": tt_pass,

        "rs_tt_pass": rs_tt_pass,

        "liquid": liquid,

        "blue_dot": blue_dot,

        "green_dot": green_dot,

    })


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    """
    ENTRY:
        liquid
        Price TT PASS
        RS Line TT PASS
        Top 10 by raw RS Score

    EXIT:
        Leaves current Top 10.

    Nothing else.
    """

    cash = STARTING_CAPITAL

    holdings = {}

    trade_log = []

    equity_curve = []


    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in trading_days:


        # ----------------------------------------------------
        # Build today's eligible pool
        # ----------------------------------------------------

        pool = []


        for sym, df in all_signals.items():

            if date not in df.index:
                continue

            row = df.loc[date]


            if pd.isna(
                row["rs_score"]
            ):
                continue


            if not bool(
                row["liquid"]
            ):
                continue


            if not bool(
                row["tt_pass"]
            ):
                continue


            if not bool(
                row["rs_tt_pass"]
            ):
                continue


            pool.append(
                (
                    sym,
                    float(
                        row["rs_score"]
                    )
                )
            )


        # Highest RS first

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )


        target_top10 = {
            sym
            for sym, _ in pool[:TOP_N]
        }


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        for sym in list(
            holdings.keys()
        ):

            if sym in target_top10:
                continue


            df = all_signals[sym]


            if date not in df.index:
                continue


            pos = holdings.pop(sym)


            exit_price = float(
                df.loc[
                    date,
                    "price"
                ]
            )


            gross_proceeds = (
                pos["qty"]
                * exit_price
            )


            s_cost = sell_side_cost(
                gross_proceeds
            )


            net_proceeds = (
                gross_proceeds
                - s_cost
            )


            cost_basis = (
                pos["qty"]
                * pos["entry_price"]
                + pos["entry_cost"]
            )


            net_gain = (
                net_proceeds
                - cost_basis
            )


            tax = stcg_tax(
                net_gain
            )


            cash += (
                net_proceeds
                - tax
            )


            gross_return_pct = round(
                (
                    exit_price
                    / pos["entry_price"]
                    - 1
                ) * 100,
                2
            )


            net_return_pct = (
                round(
                    (
                        net_gain
                        - tax
                    )
                    / cost_basis
                    * 100,
                    2
                )
                if cost_basis > 0
                else 0
            )


            trade_log.append({

                "symbol": sym,

                "entry_date":
                    pos["entry_date"]
                    .strftime("%Y-%m-%d"),

                "exit_date":
                    date.strftime("%Y-%m-%d"),

                "qty":
                    pos["qty"],

                "entry_price":
                    round(
                        pos["entry_price"],
                        2
                    ),

                "exit_price":
                    round(
                        exit_price,
                        2
                    ),

                "gross_return_pct":
                    gross_return_pct,

                "buy_cost_rs":
                    round(
                        pos["entry_cost"],
                        2
                    ),

                "sell_cost_rs":
                    round(
                        s_cost,
                        2
                    ),

                "stcg_tax_rs":
                    round(
                        tax,
                        2
                    ),

                "net_pnl_rs":
                    round(
                        net_gain - tax,
                        2
                    ),

                "net_return_pct":
                    net_return_pct,

                "days_held":
                    (
                        date
                        - pos["entry_date"]
                    ).days,

                "exit_reason":
                    "Left Top 10",

            })


        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        portfolio_value = cash


        for sym, pos in holdings.items():

            df = all_signals[sym]


            price = (
                float(
                    df.loc[
                        date,
                        "price"
                    ]
                )
                if date in df.index
                else pos["entry_price"]
            )


            portfolio_value += (
                pos["qty"]
                * price
            )


        slots_open = (
            TOP_N
            - len(holdings)
        )


        if slots_open > 0:

            slot_capital = (
                portfolio_value
                / TOP_N
            )


            for sym in [
                s for s, _
                in pool[:TOP_N]
            ]:


                if slots_open <= 0:
                    break


                if sym in holdings:
                    continue


                price = float(
                    all_signals[sym]
                    .loc[
                        date,
                        "price"
                    ]
                )


                qty = (
                    int(
                        slot_capital
                        // price
                    )
                    if price > 0
                    else 0
                )


                if qty < 1:
                    continue


                trade_value = (
                    qty
                    * price
                )


                b_cost = buy_side_cost(
                    trade_value
                )


                total_cost = (
                    trade_value
                    + b_cost
                )


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


        # ----------------------------------------------------
        # DAILY MARK-TO-MARKET
        # ----------------------------------------------------

        portfolio_value = cash


        for sym, pos in holdings.items():

            df = all_signals[sym]


            price = (
                float(
                    df.loc[
                        date,
                        "price"
                    ]
                )
                if date in df.index
                else pos["entry_price"]
            )


            portfolio_value += (
                pos["qty"]
                * price
            )


        equity_curve.append({

            "date":
                date.strftime("%Y-%m-%d"),

            "portfolio_value_rs":
                round(
                    portfolio_value,
                    2
                ),

            "equity":
                round(
                    portfolio_value
                    / STARTING_CAPITAL,
                    6
                ),

            "cash_rs":
                round(
                    cash,
                    2
                ),

            "n_holdings":
                len(holdings),

        })


    # ========================================================
    # TERMINAL VALUE
    # ========================================================

    final_marked_value = (
        equity_curve[-1][
            "portfolio_value_rs"
        ]
        if equity_curve
        else STARTING_CAPITAL
    )


    liquidation_cash = cash

    open_positions_detail = []


    if trading_days.size and holdings:

        last_date = trading_days[-1]


        for sym, pos in holdings.items():

            df = all_signals[sym]


            exit_price = (
                float(
                    df.loc[
                        last_date,
                        "price"
                    ]
                )
                if last_date in df.index
                else pos["entry_price"]
            )


            gross_proceeds = (
                pos["qty"]
                * exit_price
            )


            s_cost = sell_side_cost(
                gross_proceeds
            )


            net_proceeds = (
                gross_proceeds
                - s_cost
            )


            cost_basis = (
                pos["qty"]
                * pos["entry_price"]
                + pos["entry_cost"]
            )


            net_gain = (
                net_proceeds
                - cost_basis
            )


            tax = stcg_tax(
                net_gain
            )


            liquidation_cash += (
                net_proceeds
                - tax
            )


            open_positions_detail.append({

                "symbol":
                    sym,

                "entry_date":
                    pos["entry_date"]
                    .strftime("%Y-%m-%d"),

                "qty":
                    pos["qty"],

                "entry_price":
                    round(
                        pos["entry_price"],
                        2
                    ),

                "last_price":
                    round(
                        exit_price,
                        2
                    ),

                "unrealized_gross_return_pct":
                    round(
                        (
                            exit_price
                            / pos["entry_price"]
                            - 1
                        ) * 100,
                        2
                    ),

            })


    final_liquidation_value = (
        liquidation_cash
    )


    trade_df = pd.DataFrame(
        trade_log
    )

    equity_df = pd.DataFrame(
        equity_curve
    )


    if not equity_df.empty:

        running_max = (
            equity_df["equity"]
            .cummax()
        )

        equity_df["drawdown_pct"] = (
            (
                equity_df["equity"]
                / running_max
                - 1
            )
            * 100
        ).round(3)


    open_df = pd.DataFrame(
        open_positions_detail
    )


    return (
        trade_df,
        equity_df,
        open_df,
        final_marked_value,
        final_liquidation_value
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    trade_df,
    equity_df,
    final_marked_value,
    final_liquidation_value
):

    if equity_df.empty:
        return {}


    net_total_return_marked_pct = round(
        (
            final_marked_value
            / STARTING_CAPITAL
            - 1
        ) * 100,
        2
    )


    net_total_return_liquidation_pct = round(
        (
            final_liquidation_value
            / STARTING_CAPITAL
            - 1
        ) * 100,
        2
    )


    running_max = (
        equity_df["equity"]
        .cummax()
    )


    drawdown = (
        equity_df["equity"]
        / running_max
        - 1
    ) * 100


    max_dd = round(
        drawdown.min(),
        2
    )


    closed = (
        trade_df[
            trade_df["exit_reason"]
            == "Left Top 10"
        ]
        if not trade_df.empty
        else trade_df
    )


    n = len(closed)


    if n:

        win_rate_net = round(
            (
                closed["net_return_pct"]
                > 0
            ).mean()
            * 100,
            1
        )


        win_rate_gross = round(
            (
                closed["gross_return_pct"]
                > 0
            ).mean()
            * 100,
            1
        )


        avg_gross = round(
            closed[
                "gross_return_pct"
            ].mean(),
            2
        )


        avg_net = round(
            closed[
                "net_return_pct"
            ].mean(),
            2
        )


        median_net = round(
            closed[
                "net_return_pct"
            ].median(),
            2
        )


        avg_days = round(
            closed[
                "days_held"
            ].mean(),
            1
        )


        best_gross = closed[
            "gross_return_pct"
        ].max()


        worst_gross = closed[
            "gross_return_pct"
        ].min()


        total_costs_rs = round(
            (
                closed["buy_cost_rs"]
                + closed["sell_cost_rs"]
            ).sum(),
            0
        )


        total_tax_rs = round(
            closed[
                "stcg_tax_rs"
            ].sum(),
            0
        )


        winners = closed[
            closed["net_return_pct"] > 0
        ]


        losers = closed[
            closed["net_return_pct"] < 0
        ]


        avg_winner = (
            round(
                winners[
                    "net_return_pct"
                ].mean(),
                2
            )
            if len(winners)
            else 0
        )


        avg_loser = (
            round(
                losers[
                    "net_return_pct"
                ].mean(),
                2
            )
            if len(losers)
            else 0
        )


        gp = (
            winners[
                "net_pnl_rs"
            ].sum()
            if len(winners)
            else 0
        )


        gl = abs(
            losers[
                "net_pnl_rs"
            ].sum()
        ) if len(losers) else 0


        profit_factor = (
            round(
                gp / gl,
                3
            )
            if gl > 0
            else 0
        )


    else:

        win_rate_net = 0

        win_rate_gross = 0

        avg_gross = 0

        avg_net = 0

        median_net = 0

        avg_days = 0

        best_gross = 0

        worst_gross = 0

        total_costs_rs = 0

        total_tax_rs = 0

        avg_winner = 0

        avg_loser = 0

        profit_factor = 0


    # ========================================================
    # DAILY RETURNS
    # ========================================================

    daily_returns = (
        equity_df["equity"]
        .pct_change()
        .dropna()
    )


    if len(daily_returns):

        daily_mean = (
            daily_returns.mean()
        )

        daily_std = (
            daily_returns.std()
        )

        n_days = len(
            equity_df
        )


        annualized_return = (
            equity_df["equity"].iloc[-1]
            ** (
                252
                / max(n_days, 1)
            )
            - 1
        )


        annualized_vol = (
            daily_std
            * np.sqrt(252)
        )


        sharpe = (
            daily_mean
            / daily_std
            * np.sqrt(252)
            if daily_std > 0
            else 0
        )


        downside = daily_returns[
            daily_returns < 0
        ]


        downside_std = (
            downside.std()
            if len(downside)
            else 0
        )


        sortino = (
            daily_mean
            / downside_std
            * np.sqrt(252)
            if downside_std > 0
            else 0
        )


    else:

        annualized_return = 0

        annualized_vol = 0

        sharpe = 0

        sortino = 0


    calmar = (
        round(
            annualized_return
            / abs(max_dd / 100),
            3
        )
        if abs(max_dd) > 0
        else 0
    )


    return {

        "Final Value (marked, Rs)":
            round(
                final_marked_value,
                0
            ),

        "Final Value (liquidation, Rs)":
            round(
                final_liquidation_value,
                0
            ),

        "Net Return - marked (%)":
            net_total_return_marked_pct,

        "Net Return - liquidation (%)":
            net_total_return_liquidation_pct,

        "Annualized Return (%)":
            round(
                annualized_return * 100,
                2
            ),

        "Annualized Volatility (%)":
            round(
                annualized_vol * 100,
                2
            ),

        "Sharpe":
            round(
                sharpe,
                3
            ),

        "Sortino":
            round(
                sortino,
                3
            ),

        "Calmar":
            calmar,

        "Max Drawdown (%)":
            max_dd,

        "Number of Closed Trades":
            n,

        "Win Rate - Gross (%)":
            win_rate_gross,

        "Win Rate - Net (%)":
            win_rate_net,

        "Avg Gross Return/Trade (%)":
            avg_gross,

        "Avg Net Return/Trade (%)":
            avg_net,

        "Median Net Return/Trade (%)":
            median_net,

        "Avg Days Held":
            avg_days,

        "Avg Winner (%)":
            avg_winner,

        "Avg Loser (%)":
            avg_loser,

        "Profit Factor (net)":
            profit_factor,

        "Best Gross Trade (%)":
            best_gross,

        "Worst Gross Trade (%)":
            worst_gross,

        "Total Costs Paid (Rs)":
            total_costs_rs,

        "Total STCG Tax Paid (Rs)":
            total_tax_rs,

    }


# ============================================================
# GOOGLE SHEETS CHUNK WRITER
# ============================================================

def write_in_chunks(
    ws,
    all_rows,
    start_row,
    chunk_size,
    label
):

    total = len(all_rows)

    if total == 0:
        return


    for i in range(
        0,
        total,
        chunk_size
    ):

        chunk = all_rows[
            i:i + chunk_size
        ]

        row_start = (
            start_row + i
        )


        try:

            ws.update(
                chunk,
                f"A{row_start}"
            )


        except Exception as e:

            print(
                f"Write failed for "
                f"{label} rows "
                f"{i}-{i+len(chunk)}, "
                f"retrying once: {e}"
            )

            time.sleep(5)


            try:

                ws.update(
                    chunk,
                    f"A{row_start}"
                )


            except Exception as e2:

                print(
                    f"RETRY FAILED for "
                    f"{label} rows "
                    f"{i}-{i+len(chunk)}: "
                    f"{e2}"
                )

                print(
                    f"!! {label} INCOMPLETE "
                    f"past row {row_start} "
                    f"-- {total-i} rows "
                    f"not written !!"
                )

                raise


        print(
            f"Wrote {label}: "
            f"{min(i + chunk_size, total)}"
            f"/{total} rows"
        )


# ============================================================
# REMOVE EXISTING CHARTS
# ============================================================

def remove_existing_charts(
    sh,
    sheet_id
):

    try:

        meta = (
            sh.fetch_sheet_metadata()
        )

        requests = []


        for sheet in meta.get(
            "sheets",
            []
        ):

            if (
                sheet["properties"]
                ["sheetId"]
                == sheet_id
            ):

                for chart in sheet.get(
                    "charts",
                    []
                ):

                    requests.append({

                        "deleteEmbeddedObject": {
                            "objectId":
                                chart["chartId"]
                        }

                    })


        if requests:

            sh.batch_update({
                "requests":
                    requests
            })

            print(
                f"Removed "
                f"{len(requests)} "
                f"existing chart(s) "
                f"before adding new ones."
            )


    except Exception as e:

        print(
            "Could not check/remove "
            f"existing charts "
            f"(non-fatal): {e}"
        )


# ============================================================
# ADD CHARTS
# ============================================================

def add_charts(
    sh,
    sheet_id,
    equity_header_row_0idx,
    n_equity_rows
):

    data_end_row = (
        equity_header_row_0idx
        + 1
        + n_equity_rows
    )


    def make_chart(
        title,
        y_col_idx,
        y_axis_title,
        anchor_row
    ):

        return {

            "addChart": {

                "chart": {

                    "spec": {

                        "title":
                            title,

                        "basicChart": {

                            "chartType":
                                "LINE",

                            "legendPosition":
                                "NO_LEGEND",

                            "axis": [

                                {
                                    "position":
                                        "BOTTOM_AXIS",

                                    "title":
                                        "Date",
                                },

                                {
                                    "position":
                                        "LEFT_AXIS",

                                    "title":
                                        y_axis_title,
                                },

                            ],

                            "domains": [

                                {

                                    "domain": {

                                        "sourceRange": {

                                            "sources": [

                                                {

                                                    "sheetId":
                                                        sheet_id,

                                                    "startRowIndex":
                                                        equity_header_row_0idx,

                                                    "endRowIndex":
                                                        data_end_row,

                                                    "startColumnIndex":
                                                        0,

                                                    "endColumnIndex":
                                                        1,

                                                }

                                            ]

                                        }

                                    }

                                }

                            ],

                            "series": [

                                {

                                    "series": {

                                        "sourceRange": {

                                            "sources": [

                                                {

                                                    "sheetId":
                                                        sheet_id,

                                                    "startRowIndex":
                                                        equity_header_row_0idx,

                                                    "endRowIndex":
                                                        data_end_row,

                                                    "startColumnIndex":
                                                        y_col_idx,

                                                    "endColumnIndex":
                                                        y_col_idx +