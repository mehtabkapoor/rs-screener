"""
RS Screener Backtest - SIMPLE TOP 10 RS MODEL

RULES
-----
Universe        : stocks.csv
Price filter    : Price > Rs.20
Liquidity       : 20-day avg volume > 100,000
RS Score        : 40% 3M + 20% 6M + 20% 9M + 20% 12M
Ranking         : Raw RS Score, descending
Portfolio       : Top 10
Weight          : Equal weight = 10% each
Rebalance       : Every trading day EOD
Entry/Exit      : Determined entirely by daily Top-10 ranking
Costs           : Buy + sell transaction costs
STCG            : 20.8% effective on positive realized gains
Equity          : Daily mark-to-market
Terminal value  : Marked AND liquidated

IMPORTANT
---------
BACKTEST_END = None

The backtest automatically runs through the latest
benchmark trading date actually available from Yahoo Finance.

No Trend Template.
No RS Trend Template.
No Blue Dot.
No Green Dot.
No rank > 15 exit.
No fractal logic.

This is deliberately a simple Top-10 RS portfolio.
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

# None = latest available market data
BACKTEST_END = None

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20

# Maximum number of stocks
TOP_N = 10

STARTING_CAPITAL = 1_000_000

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ============================================================
# RS PARAMETERS
# ============================================================

RS_3M_WEIGHT = 0.40
RS_6M_WEIGHT = 0.20
RS_9M_WEIGHT = 0.20
RS_12M_WEIGHT = 0.20

RS_3M_DAYS = 63
RS_6M_DAYS = 126
RS_9M_DAYS = 189
RS_12M_DAYS = 252


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
    STCG_RATE * (1 + STCG_CESS)
)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Simple Top 10"


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
# LOAD UNIVERSE
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
# CLEAN PRICE DATA
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
        >
        MAX_PLAUSIBLE_DAILY_MOVE
    )

    n_bad = int(
        bad.sum()
    )

    if n_bad == 0:

        return close, 0

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = (
            cleaned.index
            .get_loc(idx)
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
# BENCHMARK
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\nBenchmark download: "
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
                    "implausible point(s)"
                )

            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            print(
                "Latest benchmark date: "
                f"{close.index.max().strftime('%Y-%m-%d')}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} "
                f"failed: {e}"
            )

    raise RuntimeError(
        "Could not download benchmark."
    )


# ============================================================
# STOCK SIGNALS
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
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
        "stock",
        "benchmark"
    ]

    # Need enough data for 252-day RS
    if len(aligned) < 280:

        return None

    volume = (
        volume
        .reindex(
            aligned.index
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # RAW RS SCORE
    # --------------------------------------------------------

    stock = aligned[
        "stock"
    ]

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

        RS_3M_WEIGHT *
        pct_return(
            stock,
            RS_3M_DAYS
        )

        +

        RS_6M_WEIGHT *
        pct_return(
            stock,
            RS_6M_DAYS
        )

        +

        RS_9M_WEIGHT *
        pct_return(
            stock,
            RS_9M_DAYS
        )

        +

        RS_12M_WEIGHT *
        pct_return(
            stock,
            RS_12M_DAYS
        )

    ) * 100

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    avg_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (

        stock >
        MIN_PRICE

    ) & (

        avg_volume >
        MIN_AVG_VOLUME

    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    return pd.DataFrame({

        "price":
        stock,

        "rs_score":
        rs_score,

        "liquid":
        liquid

    })


# ============================================================
# GET DAILY TOP 10
# ============================================================

def get_top10(
    all_signals,
    date
):

    candidates = []

    for symbol, df in (
        all_signals.items()
    ):

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

        candidates.append({

            "symbol":
            symbol,

            "rs_score":
            float(
                row["rs_score"]
            ),

            "price":
            float(
                row["price"]
            )

        })

    if not candidates:

        return []

    candidates.sort(

        key=lambda x:
        x["rs_score"],

        reverse=True

    )

    return candidates[
        :TOP_N
    ]


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    cash = (
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    rebalance_log = []


    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in trading_days:

        # ----------------------------------------------------
        # TOP 10 FOR TODAY
        # ----------------------------------------------------

        top10 = get_top10(
            all_signals,
            date
        )

        target_symbols = {
            x["symbol"]
            for x in top10
        }

        target_prices = {
            x["symbol"]:
            x["price"]
            for x in top10
        }


        # ----------------------------------------------------
        # CURRENT PORTFOLIO VALUE
        # ----------------------------------------------------

        current_value = cash

        current_prices = {}

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

            current_prices[sym] = (
                price
            )

            current_value += (
                pos["qty"] *
                price
            )


        # ----------------------------------------------------
        # TARGET CAPITAL
        # ----------------------------------------------------

        target_weight = (
            1.0 / TOP_N
        )

        target_value = (
            current_value *
            target_weight
        )


        # ----------------------------------------------------
        # FIRST: SELL STOCKS NOT IN TOP 10
        # ----------------------------------------------------

        symbols_to_sell = [

            sym

            for sym in
            list(holdings.keys())

            if sym not in
            target_symbols

        ]


        for sym in symbols_to_sell:

            pos = holdings.pop(
                sym
            )

            exit_price = (
                current_prices.get(
                    sym,
                    pos["entry_price"]
                )
            )

            gross_proceeds = (
                pos["qty"] *
                exit_price
            )

            sell_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds -
                sell_cost
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

            net_pnl = (
                net_gain -
                tax
            )

            trade_log.append({

                "symbol":
                sym,

                "entry_date":
                pos["entry_date"]
                .strftime(
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

                "gross_return_pct":
                round(
                    (
                        exit_price /
                        pos["entry_price"] -
                        1
                    ) * 100,
                    2
                ),

                "buy_cost_rs":
                round(
                    pos["entry_cost"],
                    2
                ),

                "sell_cost_rs":
                round(
                    sell_cost,
                    2
                ),

                "stcg_tax_rs":
                round(
                    tax,
                    2
                ),

                "net_pnl_rs":
                round(
                    net_pnl,
                    2
                ),

                "net_return_pct":
                round(
                    (
                        net_pnl /
                        cost_basis
                    ) * 100,
                    2
                ) if cost_basis > 0
                else 0,

                "days_held":
                (
                    date -
                    pos["entry_date"]
                ).days,

                "exit_reason":
                "Dropped out of Top 10",

                "rs_score_exit":
                "",

            })


        # ----------------------------------------------------
        # RECALCULATE PORTFOLIO VALUE
        # ----------------------------------------------------

        portfolio_value = cash

        for sym, pos in (
            holdings.items()
        ):

            price = float(
                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]
            )

            portfolio_value += (
                pos["qty"] *
                price
            )


        # ----------------------------------------------------
        # BUY / REBALANCE TOP 10
        # ----------------------------------------------------

        if top10:

            target_value = (
                portfolio_value /
                TOP_N
            )


            # ------------------------------------------------
            # SELL EXCESS FROM EXISTING TOP-10 POSITIONS
            # ------------------------------------------------

            for item in top10:

                sym = item["symbol"]

                price = item["price"]

                if sym not in holdings:

                    continue

                pos = holdings[sym]

                current_position_value = (
                    pos["qty"] *
                    price
                )

                excess_value = (
                    current_position_value -
                    target_value
                )

                if excess_value <= price:

                    continue

                sell_qty = int(
                    excess_value //
                    price
                )

                if sell_qty <= 0:

                    continue

                gross_proceeds = (
                    sell_qty *
                    price
                )

                sell_cost = (
                    sell_side_cost(
                        gross_proceeds
                    )
                )

                net_proceeds = (
                    gross_proceeds -
                    sell_cost
                )

                # Pro-rata approximation of cost basis
                avg_entry_price = (
                    pos["entry_price"]
                )

                allocated_entry_cost = (
                    pos["entry_cost"] *
                    (
                        sell_qty /
                        pos["qty"]
                    )
                )

                cost_basis = (
                    sell_qty *
                    avg_entry_price
                    +
                    allocated_entry_cost
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

                pos["qty"] -= (
                    sell_qty
                )

                pos["entry_cost"] -= (
                    allocated_entry_cost
                )

                trade_log.append({

                    "symbol":
                    sym,

                    "entry_date":
                    pos["entry_date"]
                    .strftime(
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
                        avg_entry_price,
                        2
                    ),

                    "exit_price":
                    round(
                        price,
                        2
                    ),

                    "gross_return_pct":
                    round(
                        (
                            price /
                            avg_entry_price -
                            1
                        ) * 100,
                        2
                    ),

                    "buy_cost_rs":
                    round(
                        allocated_entry_cost,
                        2
                    ),

                    "sell_cost_rs":
                    round(
                        sell_cost,
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
                            cost_basis
                        ) * 100,
                        2
                    ) if cost_basis > 0
                    else 0,

                    "days_held":
                    (
                        date -
                        pos["entry_date"]
                    ).days,

                    "exit_reason":
                    "Daily rebalance",

                    "rs_score_exit":
                    round(
                        item["rs_score"],
                        3
                    ),

                })


            # ------------------------------------------------
            # BUY / ADD TO UNDERWEIGHT POSITIONS
            # ------------------------------------------------

            for item in top10:

                sym = item["symbol"]

                price = item["price"]

                if sym in holdings:

                    pos = holdings[sym]

                    current_position_value = (
                        pos["qty"] *
                        price
                    )

                    required_value = (
                        target_value -
                        current_position_value
                    )

                    if required_value <= price:

                        continue

                    buy_qty = int(

                        required_value //
                        price

                    )

                    if buy_qty <= 0:

                        continue

                    trade_value = (
                        buy_qty *
                        price
                    )

                    buy_cost = (
                        buy_side_cost(
                            trade_value
                        )
                    )

                    total_required = (
                        trade_value +
                        buy_cost
                    )

                    if total_required > cash:

                        # Buy as much as affordable
                        buy_qty = int(

                            cash /
                            (
                                price *
                                (
                                    1 +
                                    STT_RATE +
                                    STAMP_DUTY_RATE +
                                    EXCHANGE_CHARGE_RATE +
                                    SEBI_CHARGE_RATE +
                                    GST_RATE *
                                    (
                                        EXCHANGE_CHARGE_RATE +
                                        SEBI_CHARGE_RATE
                                    )
                                )
                            )

                        )

                        if buy_qty <= 0:

                            continue

                        trade_value = (
                            buy_qty *
                            price
                        )

                        buy_cost = (
                            buy_side_cost(
                                trade_value
                            )
                        )

                        total_required = (
                            trade_value +
                            buy_cost
                        )

                    if total_required > cash:

                        continue

                    cash -= (
                        total_required
                    )

                    old_qty = (
                        pos["qty"]
                    )

                    old_cost = (
                        old_qty *
                        pos["entry_price"]
                    )

                    new_cost = (
                        buy_qty *
                        price
                    )

                    new_qty = (
                        old_qty +
                        buy_qty
                    )

                    pos["entry_price"] = (
                        (
                            old_cost +
                            new_cost
                        ) /
                        new_qty
                    )

                    pos["qty"] = (
                        new_qty
                    )

                    pos["entry_cost"] += (
                        buy_cost
                    )

                    continue


                # ------------------------------------------------
                # NEW POSITION
                # ------------------------------------------------

                buy_qty = int(
                    target_value //
                    price
                )

                if buy_qty <= 0:

                    continue

                trade_value = (
                    buy_qty *
                    price
                )

                buy_cost = (
                    buy_side_cost(
                        trade_value
                    )
                )

                total_required = (
                    trade_value +
                    buy_cost
                )

                if total_required > cash:

                    buy_qty = int(

                        cash /
                        (
                            price *
                            (
                                1 +
                                STT_RATE +
                                STAMP_DUTY_RATE +
                                EXCHANGE_CHARGE_RATE +
                                SEBI_CHARGE_RATE +
                                GST_RATE *
                                (
                                    EXCHANGE_CHARGE_RATE +
                                    SEBI_CHARGE_RATE
                                )
                            )
                        )

                    )

                    if buy_qty <= 0:

                        continue

                    trade_value = (
                        buy_qty *
                        price
                    )

                    buy_cost = (
                        buy_side_cost(
                            trade_value
                        )
                    )

                    total_required = (
                        trade_value +
                        buy_cost
                    )

                if total_required > cash:

                    continue

                cash -= (
                    total_required
                )

                holdings[sym] = {

                    "qty":
                    buy_qty,

                    "entry_price":
                    price,

                    "entry_date":
                    date,

                    "entry_cost":
                    buy_cost,

                }


        # ----------------------------------------------------
        # FINAL DAILY MARK-TO-MARKET
        # ----------------------------------------------------

        portfolio_value = cash

        for sym, pos in (
            holdings.items()
        ):

            if date in all_signals[sym].index:

                price = float(

                    all_signals[sym]
                    .loc[
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
        # DAILY LOG
        # ----------------------------------------------------

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

            "top10":
            ", ".join(
                x["symbol"]
                for x in top10
            ),

        })


        rebalance_log.append({

            "date":
            date.strftime(
                "%Y-%m-%d"
            ),

            "n_top10":
            len(top10),

            "top10":
            ", ".join(
                x["symbol"]
                for x in top10
            ),

            "portfolio_value_rs":
            round(
                portfolio_value,
                2
            ),

        })


    # ========================================================
    # TERMINAL VALUE
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


    if holdings and len(trading_days):

        last_date = (
            trading_days[-1]
        )

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

            sell_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds -
                sell_cost
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
                pos["entry_date"]
                .strftime(
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

                "gross_return_pct":
                round(
                    (
                        exit_price /
                        pos["entry_price"] -
                        1
                    ) * 100,
                    2
                ),

                "terminal_sell_cost_rs":
                round(
                    sell_cost,
                    2
                ),

                "terminal_stcg_rs":
                round(
                    tax,
                    2
                ),

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

    rebalance_df = pd.DataFrame(
        rebalance_log
    )

    open_df = pd.DataFrame(
        open_positions_detail
    )


    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    if not equity_df.empty:

        running_max = (
            equity_df["equity"]
            .cummax()
        )

        equity_df[
            "drawdown_pct"
        ] = (

            (
                equity_df["equity"] /
                running_max -
                1
            ) * 100

        ).round(3)


    return (

        trade_df,

        equity_df,

        rebalance_df,

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


    drawdown = (
        equity_df["drawdown_pct"]
    )

    max_dd = (
        drawdown.min()
    )


    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    if not trade_df.empty:

        n = len(
            trade_df
        )

        win_rate = (

            (
                trade_df[
                    "net_return_pct"
                ] > 0
            ).mean() * 100

        )

        avg_net = (
            trade_df[
                "net_return_pct"
            ].mean()
        )

        median_net = (
            trade_df[
                "net_return_pct"
            ].median()
        )

        avg_days = (
            trade_df[
                "days_held"
            ].mean()
        )

        winners = trade_df[
            trade_df[
                "net_return_pct"
            ] > 0
        ]

        losers = trade_df[
            trade_df[
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

        total_costs = (

            trade_df[
                "buy_cost_rs"
            ].sum()

            +

            trade_df[
                "sell_cost_rs"
            ].sum()

        )

        total_tax = (
            trade_df[
                "stcg_tax_rs"
            ].sum()
        )

    else:

        n = 0
        win_rate = 0
        avg_net = 0
        median_net = 0
        avg_days = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0
        total_costs = 0
        total_tax = 0


    # ========================================================
    # DAILY RETURNS
    # ========================================================

    daily_returns = (
        equity_df["equity"]
        .pct_change()
        .dropna()
    )


    if len(daily_returns) > 1:

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
                max(n_days, 1)
            )

        ) - 1


        annualized_vol = (
            daily_std *
            np.sqrt(252)
        )


        sharpe = (

            daily_mean /
            daily_std *
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

            if len(downside) > 1

            else 0

        )


        sortino = (

            daily_mean /
            downside_std *
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
        abs(max_dd / 100)

        if max_dd < 0

        else 0

    )


    return {

        "Model":
        "Simple Daily Top 10 RS",

        "Backtest Start":
        BACKTEST_START,

        "Backtest End":
        equity_df["date"].iloc[-1],

        "Starting Capital (Rs)":
        STARTING_CAPITAL,

        "Portfolio":
        f"Top {TOP_N}",

        "Weight":
        "Equal 10%",

        "Rebalance":
        "Daily",

        "RS Formula":
        "40% 3M + 20% 6M + 20% 9M + 20% 12M",

        "Final Value - Marked (Rs)":
        round(
            final_marked_value,
            0
        ),

        "Final Value - Liquidated (Rs)":
        round(
            final_liquidation_value,
            0
        ),

        "Net Return - Marked (%)":
        round(
            marked_return,
            2
        ),

        "Net Return - Liquidated (%)":
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

        "Closed Trade Events":
        n,

        "Win Rate (%)":
        round(
            win_rate,
            1
        ),

        "Avg Net Trade Return (%)":
        round(
            avg_net,
            2
        ),

        "Median Net Trade Return (%)":
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

        "Total Transaction Costs (Rs)":
        round(
            total_costs,
            0
        ),

        "Total STCG Tax (Rs)":
        round(
            total_tax,
            0
        ),

    }


# ============================================================
# GOOGLE SHEETS
# ============================================================

def write_to_sheet(

    trade_df,

    equity_df,

    rebalance_df,

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

    if not sheet_id or not creds_json:

        print(
            "Google credentials missing. "
            "Writing CSV files."
        )

        trade_df.to_csv(
            "simple_top10_trades.csv",
            index=False
        )

        equity_df.to_csv(
            "simple_top10_equity.csv",
            index=False
        )

        rebalance_df.to_csv(
            "simple_top10_rebalance.csv",
            index=False
        )

        if not open_df.empty:

            open_df.to_csv(
                "simple_top10_open.csv",
                index=False
            )

        return


    # --------------------------------------------------------
    # AUTH
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


    # --------------------------------------------------------
    # WORKSHEET
    # --------------------------------------------------------

    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(

            title=
            BACKTEST_WORKSHEET,

            rows=1000,

            cols=20

        )


    ws.clear()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )


    ws.update(

        [[

            "SIMPLE TOP 10 RS BACKTEST | "
            f"run {timestamp} | "
            f"Daily rebalance | "
            "Equal weight | "
            f"{BACKTEST_START} "
            f"to {effective_end_str}"

        ]],

        "A1"

    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = (

        [["SUMMARY", ""]]

        +

        [
            [k, v]
            for k, v in
            summary.items()
        ]

    )


    ws.update(
        summary_rows,
        "A3"
    )


    current_row = (
        3 +
        len(summary_rows) +
        2
    )


    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    ws.update(

        [["TRADE LOG"]],

        f"A{current_row}"

    )

    current_row += 1


    if not trade_df.empty:

        values = (

            [
                list(
                    trade_df.columns
                )
            ]

            +

            trade_df.values.tolist()

        )

        ws.update(

            values,

            f"A{current_row}"

        )

        current_row += (
            len(values) + 2
        )


    # --------------------------------------------------------
    # DAILY TOP 10
    # --------------------------------------------------------

    ws.update(

        [["DAILY TOP 10"]],

        f"A{current_row}"

    )

    current_row += 1


    if not rebalance_df.empty:

        values = (

            [
                list(
                    rebalance_df.columns
                )
            ]

            +

            rebalance_df.values.tolist()

        )

        ws.update(

            values,

            f"A{current_row}"

        )

        current_row += (
            len(values) + 2
        )


    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    ws.update(

        [["DAILY EQUITY CURVE"]],

        f"A{current_row}"

    )

    current_row += 1


    if not equity_df.empty:

        values = (

            [
                list(
                    equity_df.columns
                )
            ]

            +

            equity_df.values.tolist()

        )

        ws.update(

            values,

            f"A{current_row}"

        )

        current_row += (
            len(values) + 2
        )


    # --------------------------------------------------------
    # OPEN POSITIONS
    # --------------------------------------------------------

    ws.update(

        [["OPEN POSITIONS AT END"]],

        f"A{current_row}"

    )

    current_row += 1


    if not open_df.empty:

        values = (

            [
                list(
                    open_df.columns
                )
            ]

            +

            open_df.values.tolist()

        )

        ws.update(

            values,

            f"A{current_row}"

        )


    print(
        f"Results written to "
        f"'{BACKTEST_WORKSHEET}'."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    tickers = load_tickers()

    print(
        f"\nLoaded "
        f"{len(tickers)} tickers."
    )


    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    download_start, download_end = (
        get_download_dates()
    )


    print("=" * 60)

    print(
        f"Download start   : "
        f"{download_start}"
    )

    print(
        f"Download end     : "
        f"{download_end if download_end else 'LATEST AVAILABLE'}"
    )

    print(
        f"Backtest start   : "
        f"{BACKTEST_START}"
    )

    print(
        f"Backtest end     : "
        f"{BACKTEST_END if BACKTEST_END else 'LATEST AVAILABLE'}"
    )

    print(
        f"Price filter     : "
        f"> Rs.{MIN_PRICE}"
    )

    print(
        f"Liquidity        : "
        f"{VOLUME_LOOKBACK}d avg volume "
        f"> {MIN_AVG_VOLUME:,}"
    )

    print(
        "RS score         : "
        "40% 3M + 20% 6M + 20% 9M + 20% 12M"
    )

    print(
        f"Portfolio        : "
        f"Top {TOP_N}"
    )

    print(
        "Weight           : "
        "Equal"
    )

    print(
        "Rebalance        : "
        "Every trading day"
    )

    print(
        f"Starting capital : "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    bench_close = (
        download_benchmark()
    )


    # --------------------------------------------------------
    # STOCK DOWNLOADS
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

                start=
                download_start,

                end=
                download_end,

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
                    sdata["Close"]
                    .dropna()
                    .sort_index()
                )

                volume = (
                    sdata["Volume"]
                    .reindex(
                        close.index
                    )
                    .fillna(0)
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


                signals = (
                    compute_signals_for_stock(

                        close,

                        volume,

                        bench_close

                    )
                )


                if signals is not None:

                    all_signals[
                        symbol.replace(
                            ".NS",
                            ""
                        )
                    ] = signals


            except Exception as e:

                print(
                    f"Skipping {symbol}: "
                    f"{e}"
                )

                continue


        time.sleep(1)


    print(
        f"\nSignals computed for "
        f"{len(all_signals)} stocks."
    )

    print(
        f"Data points repaired: "
        f"{total_bad_points}"
    )


    # --------------------------------------------------------
    # EFFECTIVE END DATE
    # --------------------------------------------------------

    benchmark_latest_date = (
        pd.Timestamp(
            bench_close.index.max()
        )
    )


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
    # RUN
    # --------------------------------------------------------

    (
        trade_df,

        equity_df,

        rebalance_df,

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
        "\n"
        + "=" * 60
    )

    print(
        "SIMPLE TOP 10 RS BACKTEST"
    )

    print(
        "=" * 60
    )


    for k, v in (
        summary.items()
    ):

        print(
            f"{k}: {v}"
        )


    # --------------------------------------------------------
    # WRITE RESULTS
    # --------------------------------------------------------

    write_to_sheet(

        trade_df,

        equity_df,

        rebalance_df,

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
            "\nBACKTEST COMPLETED "
            "SUCCESSFULLY."
        )

    except Exception as e:

        print(
            "\nBACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise