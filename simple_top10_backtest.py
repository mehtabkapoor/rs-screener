# ============================================================
# SIMPLE RS TOP-10 DAILY REBALANCED BACKTEST
# ============================================================
#
# Strategy
# --------
# Universe        : stocks.csv
# Ranking         : Raw RS Score, descending
# Portfolio       : Top 10
# Weight          : Equal weight
# Rebalance       : Every trading day EOD
# Entry filter    : NONE other than having a valid RS Score
# Exit            : Stock leaves Top 10
# Costs            : Buy + sell transaction costs
# STCG            : 20.8% effective on positive realized gains
# Equity           : Daily mark-to-market
# Terminal value   : Marked AND liquidation
#
# IMPORTANT
# ---------
# This is intentionally a SIMPLE RS ranking model.
#
# It does NOT use:
#   - Price Trend Template
#   - RS Line Trend Template
#   - Price > Rs.20 filter
#   - Liquidity filter
#   - Blue Dot
#   - Green Dot
#   - Rank > 15 exit buffer
#
# Every trading day:
#   1. Calculate raw RS score for all available stocks.
#   2. Rank descending.
#   3. Hold exactly the Top 10 where possible.
#   4. Equal-weight the portfolio.
#   5. Rebalance daily.
#
# Results are written to a separate Google Sheets tab:
#
#       Simple Backtest Top 10
#
# ============================================================


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

# None = latest available Yahoo Finance data
BACKTEST_END = None

LOOKBACK_DAYS = 250

TOP_N = 10

STARTING_CAPITAL = 1_000_000

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


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

STCG_EFFECTIVE_RATE = (
    STCG_RATE *
    (1 + STCG_CESS)
)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Simple Backtest Top 10"


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():

    backtest_start = pd.Timestamp(
        BACKTEST_START
    )

    download_start = (
        backtest_start -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime("%Y-%m-%d"),
            None
        )

    backtest_end = pd.Timestamp(
        BACKTEST_END
    )

    download_end = (
        backtest_end +
        pd.Timedelta(days=1)
    )

    return (
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d")
    )


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

def load_tickers():

    if not os.path.exists(
        STOCKS_FILE
    ):

        raise FileNotFoundError(
            f"Could not find {STOCKS_FILE}"
        )

    df = pd.read_csv(
        STOCKS_FILE
    )

    if "symbol" not in df.columns:

        raise ValueError(
            "stocks.csv must contain "
            "a column named 'symbol'."
        )

    symbols = (

        df["symbol"]

        .dropna()

        .astype(str)

        .str.strip()

        .tolist()

    )

    symbols = [
        s for s in symbols
        if s
    ]

    return [

        s if s.endswith(".NS")
        else s + ".NS"

        for s in symbols

    ]


# ============================================================
# CLEAN PRICE SERIES
# ============================================================

def clean_price_series(close):

    close = (
        close
        .copy()
        .sort_index()
    )

    pct_change = (
        close.pct_change()
    )

    bad = (
        pct_change.abs()
        > MAX_PLAUSIBLE_DAILY_MOVE
    )

    n_bad = int(
        bad.sum()
    )

    if n_bad == 0:

        return close, 0

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = (
            cleaned.index.get_loc(
                idx
            )
        )

        if pos > 0:

            cleaned.iloc[pos] = (
                cleaned.iloc[pos - 1]
            )

    return cleaned, n_bad


# ============================================================
# BUY COST
# ============================================================

def buy_side_cost(
    trade_value
):

    stt = (
        STT_RATE *
        trade_value
    )

    stamp = (
        STAMP_DUTY_RATE *
        trade_value
    )

    exch = (
        EXCHANGE_CHARGE_RATE *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE *
        trade_value
    )

    gst = (
        GST_RATE *
        (exch + sebi)
    )

    return (
        stt +
        stamp +
        exch +
        sebi +
        gst
    )


# ============================================================
# SELL COST
# ============================================================

def sell_side_cost(
    trade_value
):

    stt = (
        STT_RATE *
        trade_value
    )

    exch = (
        EXCHANGE_CHARGE_RATE *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE *
        trade_value
    )

    gst = (
        GST_RATE *
        (exch + sebi)
    )

    return (
        stt +
        exch +
        sebi +
        gst +
        DP_CHARGE_FLAT
    )


# ============================================================
# STCG
# ============================================================

def stcg_tax(
    net_gain
):

    if net_gain <= 0:

        return 0.0

    return (
        net_gain *
        STCG_EFFECTIVE_RATE
    )


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        f"\nBenchmark download: "
        f"{download_start} to "
        f"{download_end if download_end else 'LATEST AVAILABLE'}"
    )

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            data = yf.download(

                ticker,

                start=download_start,

                end=download_end,

                interval="1d",

                auto_adjust=True,

                progress=False

            )

            if data.empty:

                continue

            close = data["Close"]

            if isinstance(
                close,
                pd.DataFrame
            ):

                close = (
                    close.iloc[:, 0]
                )

            close = (
                close
                .dropna()
                .sort_index()
            )

            if close.empty:

                continue

            close, n_bad = (
                clean_price_series(
                    close
                )
            )

            if n_bad:

                print(
                    f"Benchmark {ticker}: "
                    f"repaired {n_bad} "
                    f"implausible data point(s)"
                )

            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            print(
                f"Latest benchmark date: "
                f"{close.index.max().strftime('%Y-%m-%d')}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: "
                f"{e}"
            )

    raise RuntimeError(
        "Could not download any "
        "benchmark index data."
    )


# ============================================================
# RS SCORE
# ============================================================
#
# EXACT SAME RAW RS SCORE FORMULA AS YOUR ORIGINAL CODE
#
# 40% = 3-month return
# 20% = 6-month return
# 20% = 9-month return
# 20% = 12-month return
#
# No TT.
# No liquidity filter.
# No price filter.
# No RS-line filter.
#
# ============================================================

def compute_rs_score(
    close,
    bench_close
):

    aligned = pd.concat(

        [
            close,
            bench_close
        ],

        axis=1,

        join="inner"

    ).dropna()

    aligned.columns = [
        "s",
        "b"
    ]

    if len(aligned) < 280:

        return None


    def pct_return(
        series,
        days
    ):

        return (
            series /
            series.shift(days) -
            1
        )


    rs_score = (

        0.40 *
        pct_return(
            aligned["s"],
            63
        )

        +

        0.20 *
        pct_return(
            aligned["s"],
            126
        )

        +

        0.20 *
        pct_return(
            aligned["s"],
            189
        )

        +

        0.20 *
        pct_return(
            aligned["s"],
            252
        )

    ) * 100


    out = pd.DataFrame({

        "price":
        aligned["s"],

        "rs_score":
        rs_score

    })


    return out


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    cash = STARTING_CAPITAL

    holdings = {}

    trade_log = []

    equity_curve = []


    # ========================================================
    # EACH TRADING DAY
    # ========================================================

    for date in trading_days:

        # ----------------------------------------------------
        # BUILD RS RANKING
        # ----------------------------------------------------

        ranking = []


        for sym, df in (
            all_signals.items()
        ):

            if date not in df.index:

                continue


            row = df.loc[date]


            if pd.isna(
                row["rs_score"]
            ):

                continue


            price = float(
                row["price"]
            )


            if price <= 0:

                continue


            ranking.append(

                (
                    sym,
                    float(
                        row["rs_score"]
                    )
                )

            )


        # ----------------------------------------------------
        # SORT HIGHEST RS FIRST
        # ----------------------------------------------------

        ranking.sort(

            key=lambda x: x[1],

            reverse=True

        )


        # ----------------------------------------------------
        # TOP 10 TARGET
        # ----------------------------------------------------

        target = {

            sym

            for sym, _

            in ranking[:TOP_N]

        }


        rank_lookup = {

            sym: rank + 1

            for rank, (sym, _) in
            enumerate(ranking)

        }


        # ====================================================
        # SELL STOCKS NO LONGER IN TOP 10
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            if sym in target:

                continue


            df = all_signals[sym]


            if date not in df.index:

                continue


            pos = holdings.pop(
                sym
            )


            exit_price = float(

                df.loc[
                    date,
                    "price"
                ]

            )


            gross_proceeds = (

                pos["qty"] *
                exit_price

            )


            s_cost = sell_side_cost(

                gross_proceeds

            )


            net_proceeds = (

                gross_proceeds -
                s_cost

            )


            cost_basis = (

                pos["qty"] *
                pos["entry_price"]

                +

                pos["entry_cost"]

            )


            net_gain = (

                net_proceeds -
                cost_basis

            )


            tax = stcg_tax(
                net_gain
            )


            cash += (

                net_proceeds -
                tax

            )


            gross_return_pct = (

                (
                    exit_price /
                    pos["entry_price"] -
                    1

                ) * 100

            )


            net_return_pct = (

                (
                    net_gain -
                    tax

                ) /
                cost_basis *
                100

                if cost_basis > 0

                else 0

            )


            trade_log.append({

                "symbol":
                sym,

                "entry_date":
                pos["entry_date"].strftime(
                    "%Y-%m-%d"
                ),

                "exit_date":
                date.strftime(
                    "%Y-%m-%d"
                ),

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

                "entry_rs_score":
                round(
                    pos["entry_rs_score"],
                    4
                ),

                "exit_rs_score":
                round(
                    float(
                        df.loc[
                            date,
                            "rs_score"
                        ]
                    ),
                    4
                ),

                "exit_rank":
                rank_lookup.get(
                    sym,
                    ""
                ),

                "gross_return_pct":
                round(
                    gross_return_pct,
                    2
                ),

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
                round(
                    net_return_pct,
                    2
                ),

                "days_held":
                (
                    date -
                    pos["entry_date"]
                ).days,

                "exit_reason":
                "Left Top 10"

            })


        # ====================================================
        # CURRENT PORTFOLIO VALUE
        # ====================================================

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            df = all_signals[sym]


            if date in df.index:

                price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    pos["entry_price"]
                )


            portfolio_value += (

                pos["qty"] *
                price

            )


        # ====================================================
        # DAILY REBALANCE
        # ====================================================
        #
        # IMPORTANT:
        #
        # This version does NOT simply buy new positions
        # with a fixed historical slot size.
        #
        # It calculates the CURRENT portfolio value and
        # attempts to move the portfolio toward equal
        # weights across the current Top 10.
        #
        # This means the portfolio is genuinely
        # DAILY REBALANCED.
        #
        # ====================================================

        if len(target) > 0:

            target_value = (

                portfolio_value /
                len(target)

            )

        else:

            target_value = 0


        # ----------------------------------------------------
        # SELL EXISTING POSITIONS THAT ARE OVERWEIGHT
        # ----------------------------------------------------
        #
        # We first reduce overweight positions.
        #
        # This generates cash which can then be used for
        # purchases.
        #
        # ----------------------------------------------------

        for sym in list(
            holdings.keys()
        ):

            if sym not in target:

                continue


            pos = holdings[sym]

            df = all_signals[sym]


            if date not in df.index:

                continue


            price = float(
                df.loc[
                    date,
                    "price"
                ]
            )


            current_value = (

                pos["qty"] *
                price

            )


            desired_value = (
                target_value
            )


            excess_value = (

                current_value -
                desired_value

            )


            if excess_value <= price:

                continue


            sell_qty = int(

                excess_value //
                price

            )


            if sell_qty <= 0:

                continue


            # Never sell the whole position here.
            # Whole-position exits are handled above.

            sell_qty = min(

                sell_qty,

                pos["qty"] - 1

            )


            if sell_qty <= 0:

                continue


            trade_value = (

                sell_qty *
                price

            )


            s_cost = sell_side_cost(

                trade_value

            )


            # Approximate proportional cost basis
            # for tax calculation.

            proportional_entry_cost = (

                pos["entry_cost"] *
                (
                    sell_qty /
                    pos["qty"]
                )

            )


            cost_basis = (

                sell_qty *
                pos["entry_price"]

                +

                proportional_entry_cost

            )


            net_proceeds = (

                trade_value -
                s_cost

            )


            net_gain = (

                net_proceeds -
                cost_basis

            )


            tax = stcg_tax(
                net_gain
            )


            cash += (

                net_proceeds -
                tax

            )


            pos["qty"] -= sell_qty

            pos["entry_cost"] -= (
                proportional_entry_cost
            )


            # This partial rebalance is logged.

            trade_log.append({

                "symbol":
                sym,

                "entry_date":
                pos["entry_date"].strftime(
                    "%Y-%m-%d"
                ),

                "exit_date":
                date.strftime(
                    "%Y-%m-%d"
                ),

                "qty":
                sell_qty,

                "entry_price":
                round(
                    pos["entry_price"],
                    2
                ),

                "exit_price":
                round(
                    price,
                    2
                ),

                "entry_rs_score":
                round(
                    pos["entry_rs_score"],
                    4
                ),

                "exit_rs_score":
                round(
                    float(
                        df.loc[
                            date,
                            "rs_score"
                        ]
                    ),
                    4
                ),

                "exit_rank":
                rank_lookup.get(
                    sym,
                    ""
                ),

                "gross_return_pct":
                round(
                    (
                        price /
                        pos["entry_price"] -
                        1
                    ) * 100,
                    2
                ),

                "buy_cost_rs":
                round(
                    proportional_entry_cost,
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
                round(
                    (
                        (
                            net_gain -
                            tax
                        ) /
                        cost_basis *
                        100
                    )
                    if cost_basis > 0
                    else 0,
                    2
                ),

                "days_held":
                (
                    date -
                    pos["entry_date"]
                ).days,

                "exit_reason":
                "Daily rebalance"

            })


        # ====================================================
        # RECALCULATE CASH + PORTFOLIO VALUE
        # ====================================================

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            df = all_signals[sym]


            if date in df.index:

                price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    pos["entry_price"]
                )


            portfolio_value += (

                pos["qty"] *
                price

            )


        # ====================================================
        # BUY / ADD UNDERWEIGHT TOP-10 STOCKS
        # ====================================================

        for sym in [
            s for s, _
            in ranking[:TOP_N]
        ]:

            df = all_signals[sym]


            if date not in df.index:

                continue


            price = float(
                df.loc[
                    date,
                    "price"
                ]
            )


            if price <= 0:

                continue


            if sym in holdings:

                current_qty = (
                    holdings[sym]["qty"]
                )

                current_value = (

                    current_qty *
                    price

                )

            else:

                current_qty = 0

                current_value = 0


            desired_value = (
                target_value
            )


            required_value = (

                desired_value -
                current_value

            )


            if required_value <= 0:

                continue


            qty = int(

                required_value //
                price

            )


            if qty <= 0:

                continue


            # Account for transaction cost.

            while qty > 0:

                trade_value = (

                    qty *
                    price

                )


                b_cost = (
                    buy_side_cost(
                        trade_value
                    )
                )


                total_cost = (

                    trade_value +
                    b_cost

                )


                if total_cost <= cash:

                    break


                qty -= 1


            if qty <= 0:

                continue


            trade_value = (

                qty *
                price

            )


            b_cost = buy_side_cost(
                trade_value
            )


            total_cost = (

                trade_value +
                b_cost

            )


            if total_cost > cash:

                continue


            cash -= total_cost


            if sym in holdings:

                # Weighted-average entry price.

                old_qty = (
                    holdings[sym]["qty"]
                )

                old_entry_value = (

                    old_qty *
                    holdings[sym]["entry_price"]

                )

                new_entry_value = (

                    qty *
                    price

                )

                new_qty = (
                    old_qty +
                    qty
                )

                holdings[sym][
                    "entry_price"
                ] = (

                    old_entry_value +
                    new_entry_value

                ) / new_qty


                holdings[sym][
                    "qty"
                ] = new_qty


                holdings[sym][
                    "entry_cost"
                ] += b_cost


                holdings[sym][
                    "entry_rs_score"
                ] = float(

                    df.loc[
                        date,
                        "rs_score"
                    ]

                )

            else:

                holdings[sym] = {

                    "qty":
                    qty,

                    "entry_price":
                    price,

                    "entry_date":
                    date,

                    "entry_cost":
                    b_cost,

                    "entry_rs_score":
                    float(
                        df.loc[
                            date,
                            "rs_score"
                        ]
                    )

                }


        # ====================================================
        # DAILY MARK-TO-MARKET
        # ====================================================

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            df = all_signals[sym]


            if date in df.index:

                price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    pos["entry_price"]
                )


            portfolio_value += (

                pos["qty"] *
                price

            )


        # ----------------------------------------------------
        # RECORD TOP 10
        # ----------------------------------------------------

        top10_symbols = [
            sym
            for sym, _
            in ranking[:TOP_N]
        ]


        equity_curve.append({

            "date":
            date.strftime(
                "%Y-%m-%d"
            ),

            "portfolio_value_rs":
            round(
                portfolio_value,
                2
            ),

            "equity":
            round(
                portfolio_value /
                STARTING_CAPITAL,
                6
            ),

            "cash_rs":
            round(
                cash,
                2
            ),

            "n_holdings":
            len(holdings),

            "top_10":
            ", ".join(
                top10_symbols
            )

        })


    # ========================================================
    # TERMINAL MARKED VALUE
    # ========================================================

    if equity_curve:

        final_marked_value = (

            equity_curve[-1]
            ["portfolio_value_rs"]

        )

    else:

        final_marked_value = (
            STARTING_CAPITAL
        )


    # ========================================================
    # TERMINAL LIQUIDATION
    # ========================================================

    liquidation_cash = cash

    open_positions_detail = []


    if len(trading_days) and holdings:

        last_date = trading_days[-1]


        for sym, pos in (
            holdings.items()
        ):

            df = all_signals[sym]


            if last_date in df.index:

                exit_price = float(

                    df.loc[
                        last_date,
                        "price"
                    ]

                )

            else:

                exit_price = (
                    pos["entry_price"]
                )


            gross_proceeds = (

                pos["qty"] *
                exit_price

            )


            s_cost = sell_side_cost(

                gross_proceeds

            )


            net_proceeds = (

                gross_proceeds -
                s_cost

            )


            cost_basis = (

                pos["qty"] *
                pos["entry_price"]

                +

                pos["entry_cost"]

            )


            net_gain = (

                net_proceeds -
                cost_basis

            )


            tax = stcg_tax(
                net_gain
            )


            liquidation_cash += (

                net_proceeds -
                tax

            )


            open_positions_detail.append({

                "symbol":
                sym,

                "entry_date":
                pos["entry_date"].strftime(
                    "%Y-%m-%d"
                ),

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

                "entry_rs_score":
                round(
                    pos["entry_rs_score"],
                    4
                ),

                "unrealized_gross_return_pct":
                round(
                    (
                        exit_price /
                        pos["entry_price"] -
                        1
                    ) * 100,
                    2
                )

            })


    final_liquidation_value = (
        liquidation_cash
    )


    # ========================================================
    # DATAFRAMES
    # ========================================================

    trade_df = pd.DataFrame(
        trade_log
    )

    equity_df = pd.DataFrame(
        equity_curve
    )

    open_df = pd.DataFrame(
        open_positions_detail
    )


    # --------------------------------------------------------
    # DRAWNDOWN
    # --------------------------------------------------------

    if not equity_df.empty:

        running_max = (

            equity_df[
                "equity"
            ].cummax()

        )


        equity_df[
            "drawdown_pct"
        ] = (

            (
                equity_df[
                    "equity"
                ] /
                running_max -
                1
            ) * 100

        ).round(3)


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


    marked_return = (

        (
            final_marked_value /
            STARTING_CAPITAL -
            1
        ) * 100

    )


    liquidation_return = (

        (
            final_liquidation_value /
            STARTING_CAPITAL -
            1
        ) * 100

    )


    running_max = (

        equity_df[
            "equity"
        ].cummax()

    )


    drawdown = (

        equity_df[
            "equity"
        ] /
        running_max -
        1

    ) * 100


    max_dd = drawdown.min()


    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    if not trade_df.empty:

        closed = trade_df


        n = len(
            closed
        )


        win_rate_net = (

            (
                closed[
                    "net_return_pct"
                ] > 0
            ).mean() * 100

        )


        win_rate_gross = (

            (
                closed[
                    "gross_return_pct"
                ] > 0
            ).mean() * 100

        )


        avg_gross = (
            closed[
                "gross_return_pct"
            ].mean()
        )


        avg_net = (
            closed[
                "net_return_pct"
            ].mean()
        )


        median_net = (
            closed[
                "net_return_pct"
            ].median()
        )


        avg_days = (
            closed[
                "days_held"
            ].mean()
        )


        best_gross = (
            closed[
                "gross_return_pct"
            ].max()
        )


        worst_gross = (
            closed[
                "gross_return_pct"
            ].min()
        )


        total_costs = (

            closed[
                "buy_cost_rs"
            ].sum()

            +

            closed[
                "sell_cost_rs"
            ].sum()

        )


        total_tax = (
            closed[
                "stcg_tax_rs"
            ].sum()
        )


        winners = closed[
            closed[
                "net_return_pct"
            ] > 0
        ]


        losers = closed[
            closed[
                "net_return_pct"
            ] < 0
        ]


        avg_winner = (

            winners[
                "net_return_pct"
            ].mean()

            if len(winners)

            else 0

        )


        avg_loser = (

            losers[
                "net_return_pct"
            ].mean()

            if len(losers)

            else 0

        )


        gross_profit = (

            winners[
                "net_pnl_rs"
            ].sum()

            if len(winners)

            else 0

        )


        gross_loss = (

            abs(
                losers[
                    "net_pnl_rs"
                ].sum()
            )

            if len(losers)

            else 0

        )


        profit_factor = (

            gross_profit /
            gross_loss

            if gross_loss > 0

            else 0

        )

    else:

        n = 0

        win_rate_net = 0

        win_rate_gross = 0

        avg_gross = 0

        avg_net = 0

        median_net = 0

        avg_days = 0

        best_gross = 0

        worst_gross = 0

        total_costs = 0

        total_tax = 0

        avg_winner = 0

        avg_loser = 0

        profit_factor = 0


    # ========================================================
    # DAILY STATISTICS
    # ========================================================

    daily_returns = (

        equity_df[
            "equity"
        ]
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

            equity_df[
                "equity"
            ].iloc[-1]

            **

            (
                252 /
                max(
                    n_days,
                    1
                )
            )

        ) - 1


        annualized_vol = (

            daily_std *
            np.sqrt(252)

        )


        sharpe = (

            (
                daily_mean /
                daily_std
            ) *
            np.sqrt(252)

            if daily_std > 0

            else 0

        )


        downside = (

            daily_returns[
                daily_returns < 0
            ]

        )


        downside_std = (

            downside.std()

            if len(downside)

            else 0

        )


        sortino = (

            (
                daily_mean /
                downside_std
            ) *
            np.sqrt(252)

            if downside_std > 0

            else 0

        )

    else:

        annualized_return = 0

        annualized_vol = 0

        sharpe = 0

        sortino = 0


    calmar = (

        annualized_return /
        abs(
            max_dd / 100
        )

        if abs(max_dd) > 0

        else 0

    )


    return {

        "Strategy":
        "Simple RS Ranking - Top 10",

        "Backtest Start":
        BACKTEST_START,

        "Backtest End":
        equity_df[
            "date"
        ].iloc[-1],

        "Starting Capital (Rs)":
        STARTING_CAPITAL,

        "Portfolio":
        f"Top {TOP_N}",

        "Rebalance":
        "Daily",

        "Weight":
        "Equal Weight",

        "Ranking":
        "Raw RS Score Descending",

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
        round(
            marked_return,
            2
        ),

        "Net Return - liquidation (%)":
        round(
            liquidation_return,
            2
        ),

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
        round(
            calmar,
            3
        ),

        "Max Drawdown (%)":
        round(
            max_dd,
            2
        ),

        "Number of Trades":
        n,

        "Win Rate - Gross (%)":
        round(
            win_rate_gross,
            1
        ),

        "Win Rate - Net (%)":
        round(
            win_rate_net,
            1
        ),

        "Avg Gross Return/Trade (%)":
        round(
            avg_gross,
            2
        ),

        "Avg Net Return/Trade (%)":
        round(
            avg_net,
            2
        ),

        "Median Net Return/Trade (%)":
        round(
            median_net,
            2
        ),

        "Avg Days Held":
        round(
            avg_days,
            1
        ),

        "Avg Winner (%)":
        round(
            avg_winner,
            2
        ),

        "Avg Loser (%)":
        round(
            avg_loser,
            2
        ),

        "Profit Factor":
        round(
            profit_factor,
            3
        ),

        "Best Gross Trade (%)":
        round(
            best_gross,
            2
        ),

        "Worst Gross Trade (%)":
        round(
            worst_gross,
            2
        ),

        "Total Costs Paid (Rs)":
        round(
            total_costs,
            0
        ),

        "Total STCG Tax Paid (Rs)":
        round(
            total_tax,
            0
        )

    }


# ============================================================
# GOOGLE SHEETS CHARTS
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
                sheet[
                    "properties"
                ][
                    "sheetId"
                ]
                ==
                sheet_id
            ):

                for chart in sheet.get(
                    "charts",
                    []
                ):

                    requests.append({

                        "deleteEmbeddedObject": {

                            "objectId":
                            chart[
                                "chartId"
                            ]

                        }

                    })


        if requests:

            sh.batch_update({

                "requests":
                requests

            })


    except Exception as e:

        print(
            f"Chart removal failed "
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

        equity_header_row_0idx +
        1 +
        n_equity_rows

    )


    def chart_request(

        title,

        y_col_idx,

        y_title,

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
                                    "Date"

                                },

                                {

                                    "position":
                                    "LEFT_AXIS",

                                    "title":
                                    y_title

                                }

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
                                                    1

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
                                                    y_col_idx + 1

                                                }

                                            ]

                                        }

                                    },

                                    "targetAxis":
                                    "LEFT_AXIS"

                                }

                            ]

                        }

                    },

                    "position": {

                        "overlayPosition": {

                            "anchorCell": {

                                "sheetId":
                                sheet_id,

                                "rowIndex":
                                anchor_row,

                                "columnIndex":
                                8

                            },

                            "widthPixels":
                            650,

                            "heightPixels":
                            380

                        }

                    }

                }

            }

        }


    requests = [

        chart_request(

            "Simple RS Top 10 - Equity Curve",

            1,

            "Portfolio Value (Rs)",

            equity_header_row_0idx

        ),

        chart_request(

            "Simple RS Top 10 - Drawdown",

            6,

            "Drawdown (%)",

            equity_header_row_0idx + 22

        )

    ]


    try:

        sh.batch_update({

            "requests":
            requests

        })

        print(
            "Charts added."
        )

    except Exception as e:

        print(
            f"Could not add charts "
            f"(non-fatal): {e}"
        )


# ============================================================
# CHUNK WRITER
# ============================================================

def write_in_chunks(

    ws,

    all_rows,

    start_row,

    chunk_size,

    label

):

    total = len(
        all_rows
    )


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

                f"Write failed "
                f"for {label} "
                f"{i}-{i+len(chunk)}: "
                f"{e}"

            )


            time.sleep(5)


            ws.update(

                chunk,

                f"A{row_start}"

            )


        print(

            f"Wrote {label}: "
            f"{min(i + chunk_size, total)}"
            f"/{total}"

        )


# ============================================================
# WRITE RESULTS TO GOOGLE SHEETS
# ============================================================

def write_to_sheet(

    trade_df,

    equity_df,

    open_df,

    summary,

    effective_end_str

):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )


    # --------------------------------------------------------
    # CSV FALLBACK
    # --------------------------------------------------------

    if (
        not sheet_id
        or
        not creds_json
    ):

        print(
            "Missing Google credentials."
        )

        trade_df.to_csv(
            "simple_rs_top10_trades.csv",
            index=False
        )

        equity_df.to_csv(
            "simple_rs_top10_equity.csv",
            index=False
        )

        if not open_df.empty:

            open_df.to_csv(
                "simple_rs_top10_open_positions.csv",
                index=False
            )

        return


    # --------------------------------------------------------
    # GOOGLE AUTH
    # --------------------------------------------------------

    creds_dict = json.loads(
        creds_json
    )


    scopes = [

        "https://www.googleapis.com/auth/spreadsheets"

    ]


    creds = (
        Credentials
        .from_service_account_info(

            creds_dict,

            scopes=scopes

        )
    )


    gc = gspread.authorize(
        creds
    )


    sh = gc.open_by_key(
        sheet_id
    )


    timestamp = (

        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )

    )


    # --------------------------------------------------------
    # REQUIRED SHEET SIZE
    # --------------------------------------------------------

    n_rows_needed = (

        len(summary)

        +

        len(trade_df)

        +

        len(open_df)

        +

        len(equity_df)

        +

        80

    )


    n_cols_needed = 16


    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )


        if (
            ws.row_count <
            n_rows_needed
            or
            ws.col_count <
            n_cols_needed
        ):

            ws.resize(

                rows=max(
                    ws.row_count,
                    n_rows_needed
                ),

                cols=max(
                    ws.col_count,
                    n_cols_needed
                )

            )


    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(

            title=
            BACKTEST_WORKSHEET,

            rows=
            n_rows_needed,

            cols=
            n_cols_needed

        )


    # --------------------------------------------------------
    # CLEAR OLD CHARTS + DATA
    # --------------------------------------------------------

    remove_existing_charts(

        sh,

        ws.id

    )


    ws.clear()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ws.update(

        [[

            "SIMPLE RS TOP 10 DAILY REBALANCE | "
            f"Run: {timestamp} | "
            "NET of costs + STCG | "
            f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
            "Ranking: Raw RS Score | "
            f"Top {TOP_N} | "
            "Equal Weight | "
            "Daily Rebalance | "
            f"Window: {BACKTEST_START} "
            f"to {effective_end_str}"

        ]],

        "A1"

    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = (

        [
            [
                "Summary",
                ""
            ]
        ]

        +

        [
            [
                k,
                v
            ]

            for k, v in
            summary.items()

        ]

    )


    ws.update(

        summary_rows,

        "A3"

    )


    # --------------------------------------------------------
    # TRADE LOG
    # --------------------------------------------------------

    trade_start_row = (

        3 +

        len(summary_rows) +

        2

    )


    ws.update(

        [
            [
                "Trade Log"
            ]
        ],

        f"A{trade_start_row}"

    )


    trade_header_row = (

        trade_start_row + 1

    )


    if not trade_df.empty:

        write_in_chunks(

            ws,

            [

                list(
                    trade_df.columns
                )

            ]

            +

            trade_df.values.tolist(),

            trade_header_row,

            2000,

            "trade log"

        )


    # --------------------------------------------------------
    # OPEN POSITIONS
    # --------------------------------------------------------

    open_start_row = (

        trade_header_row +

        len(trade_df) +

        3

    )


    ws.update(

        [[

            "Open Positions at "
            "Backtest End "
            "(mark-to-market)"

        ]],

        f"A{open_start_row}"

    )


    open_header_row = (

        open_start_row + 1

    )


    if not open_df.empty:

        ws.update(

            [

                list(
                    open_df.columns
                )

            ]

            +

            open_df.values.tolist(),

            f"A{open_header_row}"

        )


    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity_start_row = (

        open_header_row +

        max(
            len(open_df),
            1
        ) +

        3

    )


    ws.update(

        [
            [
                "Daily Equity Curve"
            ]
        ],

        f"A{equity_start_row}"

    )


    equity_header_row = (

        equity_start_row + 1

    )


    if not equity_df.empty:

        write_in_chunks(

            ws,

            [

                list(
                    equity_df.columns
                )

            ]

            +

            equity_df.values.tolist(),

            equity_header_row,

            2000,

            "equity curve"

        )


        add_charts(

            sh,

            ws.id,

            equity_header_row - 1,

            len(equity_df)

        )


    print(

        f"\nSimple RS Top 10 "
        f"results written to "
        f"'{BACKTEST_WORKSHEET}' tab."

    )

    print(

        f"Trades: "
        f"{len(trade_df)}"

    )

    print(

        f"Trading days: "
        f"{len(equity_df)}"

    )

    print(

        f"Open positions: "
        f"{len(open_df)}"

    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD UNIVERSE
    # --------------------------------------------------------

    tickers = load_tickers()


    print(

        f"\nLoaded "
        f"{len(tickers)} tickers."

    )


    # --------------------------------------------------------
    # DOWNLOAD DATES
    # --------------------------------------------------------

    download_start, download_end = (
        get_download_dates()
    )


    print(
        "=" * 60
    )


    print(

        f"Download start : "
        f"{download_start}"

    )


    print(

        f"Download end   : "
        f"{download_end if download_end else 'LATEST AVAILABLE'}"

    )


    print(

        f"Backtest start : "
        f"{BACKTEST_START}"

    )


    print(

        f"Backtest end   : "
        f"{BACKTEST_END if BACKTEST_END else 'LATEST AVAILABLE'}"

    )


    print(

        f"Portfolio      : "
        f"Top {TOP_N}"

    )


    print(

        "Ranking        : "
        "Raw RS Score descending"

    )


    print(

        "Rebalance      : "
        "Every trading day"

    )


    print(

        "Weight         : "
        "Equal weight"

    )


    print(

        f"Capital        : "
        f"Rs.{STARTING_CAPITAL:,.0f}"

    )


    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    bench_close = (
        download_benchmark()
    )


    # --------------------------------------------------------
    # DOWNLOAD STOCKS
    # --------------------------------------------------------

    all_signals = {}

    total_bad_points = 0

    batch_size = 50


    for i in range(

        0,

        len(tickers),

        batch_size

    ):

        batch = tickers[
            i:i + batch_size
        ]


        print(

            f"\nDownloading batch "
            f"{i}-"
            f"{i + len(batch)}..."

        )


        try:

            data = yf.download(

                batch,

                start=download_start,

                end=download_end,

                interval="1d",

                auto_adjust=True,

                progress=False,

                group_by="ticker",

                threads=True

            )


        except Exception as e:

            print(

                f"Batch download failed: "
                f"{e}"

            )

            continue


        # ----------------------------------------------------
        # PROCESS STOCKS
        # ----------------------------------------------------

        for symbol in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    if (
                        symbol not in
                        data.columns
                        .get_level_values(0)
                    ):

                        continue


                    sdata = data[
                        symbol
                    ]


                if (
                    "Close"
                    not in
                    sdata.columns
                ):

                    continue


                close = (

                    sdata[
                        "Close"
                    ]

                    .dropna()

                    .sort_index()

                )


                if close.empty:

                    continue


                close, n_bad = (
                    clean_price_series(
                        close
                    )
                )


                total_bad_points += (
                    n_bad
                )


                sig = (
                    compute_rs_score(

                        close,

                        bench_close

                    )
                )


                if sig is not None:

                    clean_symbol = (
                        symbol
                        .replace(
                            ".NS",
                            ""
                        )
                    )


                    all_signals[
                        clean_symbol
                    ] = sig


            except Exception as e:

                print(

                    f"Skipping "
                    f"{symbol}: "
                    f"{e}"

                )

                continue


        time.sleep(1)


    # --------------------------------------------------------
    # SIGNAL SUMMARY
    # --------------------------------------------------------

    print(

        f"\nRS scores computed "
        f"for {len(all_signals)} stocks."

    )


    print(

        f"Total data points repaired: "
        f"{total_bad_points}"

    )


    # --------------------------------------------------------
    # LATEST AVAILABLE DATES
    # --------------------------------------------------------

    if not all_signals:

        raise RuntimeError(
            "No stock signals were calculated."
        )


    latest_stock_date = max(

        df.index.max()

        for df in
        all_signals.values()

    )


    benchmark_latest_date = (
        pd.Timestamp(
            bench_close.index.max()
        )
    )


    # --------------------------------------------------------
    # EFFECTIVE END
    # --------------------------------------------------------

    if BACKTEST_END is None:

        effective_end = (
            benchmark_latest_date
        )

    else:

        effective_end = min(

            pd.Timestamp(
                BACKTEST_END
            ),

            benchmark_latest_date

        )


    print(

        "\nLatest benchmark data: "
        f"{benchmark_latest_date.strftime('%Y-%m-%d')}"

    )


    print(

        "Latest stock data: "
        f"{pd.Timestamp(latest_stock_date).strftime('%Y-%m-%d')}"

    )


    print(

        "Effective backtest end: "
        f"{effective_end.strftime('%Y-%m-%d')}"

    )


    # --------------------------------------------------------
    # TRADING DAYS
    # --------------------------------------------------------

    trading_days = (

        bench_close.index[

            (
                bench_close.index >=
                pd.Timestamp(
                    BACKTEST_START
                )
            )

            &

            (
                bench_close.index <=
                effective_end
            )

        ]

    )


    print(

        f"\nTrading days: "
        f"{len(trading_days)}"

    )


    if len(trading_days):

        print(

            "First trading day: "
            f"{trading_days[0].strftime('%Y-%m-%d')}"

        )

        print(

            "Last trading day: "
            f"{trading_days[-1].strftime('%Y-%m-%d')}"

        )


    # --------------------------------------------------------
    # RUN BACKTEST
    # --------------------------------------------------------

    (

        trade_df,

        equity_df,

        open_df,

        final_marked,

        final_liq

    ) = run_backtest(

        all_signals,

        trading_days

    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize(

        trade_df,

        equity_df,

        final_marked,

        final_liq

    )


    print(
        "\n--- SIMPLE RS TOP 10 SUMMARY ---"
    )


    for k, v in summary.items():

        print(
            f"{k}: {v}"
        )


    # --------------------------------------------------------
    # WRITE TO SEPARATE SHEET
    # --------------------------------------------------------

    write_to_sheet(

        trade_df,

        equity_df,

        open_df,

        summary,

        effective_end.strftime(
            "%Y-%m-%d"
        )

    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

        print(

            "\nSIMPLE RS TOP 10 "
            "BACKTEST COMPLETED "
            "SUCCESSFULLY."

        )

    except Exception as e:

        print(
            "\nSIMPLE RS TOP 10 "
            "BACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise