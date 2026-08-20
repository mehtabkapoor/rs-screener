"""
RS SCREENER — DAILY TOP 10 RS BACKTEST
=======================================

MODEL
-----
Universe        : stocks.csv
Price filter    : Price > Rs.20
Liquidity       : 20-day average volume > 100,000
Price TT        : 7/7 required
RS Line TT      : 7/7 required
Ranking         : Raw RS Score, descending
Portfolio       : Top 10 eligible stocks
Weight          : Equal weight
Rebalance       : EVERY TRADING DAY
Execution       : SAME-DAY CLOSE
Exit            : Stock leaves Top 15 / becomes ineligible
Costs           : Buy + sell transaction costs
STCG            : 20.8% effective on positive realized gains
Equity          : Daily mark-to-market
Terminal        : Marked AND liquidated

IMPORTANT
---------
BACKTEST_END = None

The backtest automatically runs until the latest trading date
actually available from Yahoo Finance.

It does NOT assume today's calendar date is available.

GOOGLE SHEETS
-------------
Backtest
    -> Summary
    -> Trade Log
    -> Open Positions
    -> Daily Equity Curve

Top 10 RS Backtest
    -> Daily ranking of ALL eligible stocks
    -> Rank
    -> RS Score
    -> Price
    -> Price TT
    -> RS Line TT
    -> Liquidity
    -> Eligible
    -> Top 10
    -> Held EOD
    -> EOD Weight

DATE HANDLING
-------------
All dates are normalized to midnight before any lookup.

This prevents:
KeyError: Timestamp('2026-07-20 00:00:00')

EXECUTION
---------
Signals use the CLOSE of the current trading day.

The portfolio is rebalanced at THAT SAME CLOSE.

There is NO next-day execution.

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

# None = latest available Yahoo Finance data
BACKTEST_END = None

MIN_PRICE = 20

MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20

LOOKBACK_DAYS = 250

MAX_PLAUSIBLE_DAILY_MOVE = 0.30

TOP_N = 10

EXIT_RANK = 15

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

STCG_EFFECTIVE_RATE = (
    STCG_RATE * (1 + STCG_CESS)
)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"

TOP10_WORKSHEET = "Top 10 RS Backtest"


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_index(index):

    """
    Convert every timestamp to timezone-naive midnight.

    This is critical because Yahoo can return timestamps with
    different representations.
    """

    idx = pd.DatetimeIndex(index)

    if idx.tz is not None:

        idx = idx.tz_localize(None)

    return idx.normalize()


def normalize_series_index(series):

    s = series.copy()

    s.index = normalize_index(s.index)

    s = s[~s.index.duplicated(keep="last")]

    return s.sort_index()


def normalize_dataframe_index(df):

    x = df.copy()

    x.index = normalize_index(x.index)

    x = x[
        ~x.index.duplicated(
            keep="last"
        )
    ]

    return x.sort_index()


# ============================================================
# DOWNLOAD DATES
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
# LOAD TICKERS
# ============================================================

def load_tickers():

    if not os.path.exists(STOCKS_FILE):

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

    close = normalize_series_index(
        close
    )

    pct_change = close.pct_change()

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

def buy_side_cost(trade_value):

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

def sell_side_cost(trade_value):

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

def stcg_tax(net_gain):

    if net_gain <= 0:

        return 0.0

    return (
        net_gain *
        STCG_EFFECTIVE_RATE
    )


# ============================================================
# BENCHMARK DOWNLOAD
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

                close = close.iloc[:, 0]

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
                f"Benchmark loaded: {ticker}"
            )

            print(
                "Latest benchmark date: "
                f"{close.index.max().strftime('%Y-%m-%d')}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: {e}"
            )

    raise RuntimeError(
        "Could not download benchmark."
    )


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template_series(s):

    sma50 = (
        s.rolling(50)
        .mean()
    )

    sma150 = (
        s.rolling(150)
        .mean()
    )

    sma200 = (
        s.rolling(200)
        .mean()
    )

    sma200_1mo = (
        sma200.shift(21)
    )

    low52 = (
        s.rolling(252)
        .min()
    )

    high52 = (
        s.rolling(252)
        .max()
    )

    c1 = (
        (s > sma150) &
        (s > sma200)
    )

    c2 = (
        sma150 > sma200
    )

    c3 = (
        sma200 > sma200_1mo
    )

    c4 = (
        (sma50 > sma150) &
        (sma50 > sma200)
    )

    c5 = (
        s > sma50
    )

    c6 = (
        s >= 1.25 * low52
    )

    c7 = (
        s >= 0.75 * high52
    )

    met = (

        c1.astype(int) +

        c2.astype(int) +

        c3.astype(int) +

        c4.astype(int) +

        c5.astype(int) +

        c6.astype(int) +

        c7.astype(int)

    )

    return met == 7


# ============================================================
# STOCK SIGNALS
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
    bench_close
):

    close = normalize_series_index(
        close
    )

    volume = normalize_series_index(
        volume
    )

    bench_close = normalize_series_index(
        bench_close
    )

    aligned = pd.concat(

        [
            close.rename("s"),
            bench_close.rename("b")
        ],

        axis=1,

        join="inner"

    ).dropna()

    if len(aligned) < 280:

        return None

    volume = (
        volume
        .reindex(aligned.index)
        .fillna(0)
    )

    # --------------------------------------------------------
    # RS RATIO
    # --------------------------------------------------------

    rs_ratio = (
        aligned["s"] /
        aligned["b"]
    )

    # --------------------------------------------------------
    # RETURN FUNCTION
    # --------------------------------------------------------

    def pct_return(
        series,
        days
    ):

        return (
            series /
            series.shift(days)
            - 1
        )

    # --------------------------------------------------------
    # RAW RS SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BLUE DOT
    # --------------------------------------------------------

    previous_rs_high = (

        rs_ratio
        .shift(1)
        .rolling(LOOKBACK_DAYS)
        .max()

    )

    blue_dot = (
        rs_ratio >
        previous_rs_high
    )

    # --------------------------------------------------------
    # PRICE NEW HIGH
    # --------------------------------------------------------

    previous_price_high = (

        aligned["s"]
        .shift(1)
        .rolling(LOOKBACK_DAYS)
        .max()

    )

    price_at_new_high = (
        aligned["s"] >
        previous_price_high
    )

    # --------------------------------------------------------
    # GREEN DOT
    # --------------------------------------------------------

    green_dot = (

        blue_dot &
        (~price_at_new_high)

    )

    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
    # --------------------------------------------------------

    tt_pass = (
        trend_template_series(
            aligned["s"]
        )
    )

    # --------------------------------------------------------
    # RS LINE TREND TEMPLATE
    # --------------------------------------------------------

    rs_tt_pass = (
        trend_template_series(
            rs_ratio
        )
    )

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

        aligned["s"] >
        MIN_PRICE

    ) & (

        avg_volume >
        MIN_AVG_VOLUME

    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    out = pd.DataFrame({

        "price":
        aligned["s"],

        "rs_score":
        rs_score,

        "tt_pass":
        tt_pass,

        "rs_tt_pass":
        rs_tt_pass,

        "liquid":
        liquid,

        "blue_dot":
        blue_dot,

        "green_dot":
        green_dot

    })

    out.index = normalize_index(
        out.index
    )

    return out


# ============================================================
# GET PRICE FOR DATE
# ============================================================

def get_price(
    all_signals,
    symbol,
    date
):

    date = pd.Timestamp(
        date
    ).normalize()

    df = all_signals[
        symbol
    ]

    if date not in df.index:

        return None

    value = df.loc[
        date,
        "price"
    ]

    if pd.isna(value):

        return None

    return float(value)


# ============================================================
# BUILD DAILY RANKINGS
# ============================================================

def build_daily_rankings(
    all_signals,
    trading_days
):

    daily_rows = []

    ranking_by_date = {}

    for date in trading_days:

        date = pd.Timestamp(
            date
        ).normalize()

        pool = []

        # ----------------------------------------------------
        # FIND ELIGIBLE STOCKS
        # ----------------------------------------------------

        for symbol, df in (
            all_signals.items()
        ):

            if date not in df.index:

                continue

            row = df.loc[
                date
            ]

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

            pool.append({

                "symbol":
                symbol,

                "rs_score":
                float(
                    row["rs_score"]
                ),

                "price":
                float(
                    row["price"]
                ),

                "liquid":
                True,

                "tt_pass":
                True,

                "rs_tt_pass":
                True,

                "blue_dot":
                bool(
                    row["blue_dot"]
                ),

                "green_dot":
                bool(
                    row["green_dot"]
                )

            })

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        pool.sort(

            key=lambda x:
            x["rs_score"],

            reverse=True

        )

        ranking_by_date[
            date
        ] = pool

        # ----------------------------------------------------
        # WRITE DAILY TOP-10 SHEET DATA
        # ----------------------------------------------------

        for rank, item in enumerate(
            pool,
            start=1
        ):

            is_top10 = (
                rank <= TOP_N
            )

            daily_rows.append({

                "Date":
                date.strftime(
                    "%Y-%m-%d"
                ),

                "Rank":
                rank,

                "Symbol":
                item["symbol"],

                "RS Score":
                round(
                    item["rs_score"],
                    4
                ),

                "Price":
                round(
                    item["price"],
                    2
                ),

                "Liquid":
                "YES",

                "Price TT":
                "PASS",

                "RS Line TT":
                "PASS",

                "Eligible":
                "YES",

                "Top 10":
                "YES"
                if is_top10
                else "NO",

                "Blue Dot":
                "YES"
                if item["blue_dot"]
                else "NO",

                "Green Dot":
                "YES"
                if item["green_dot"]
                else "NO",

                "Held EOD":
                "",

                "EOD Weight %":
                ""

            })

    daily_df = pd.DataFrame(
        daily_rows
    )

    return (
        ranking_by_date,
        daily_df
    )


# ============================================================
# SAME-DAY DAILY REBALANCE
# ============================================================

def run_backtest(
    all_signals,
    ranking_by_date,
    trading_days
):

    cash = float(
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    daily_holdings = []

    for date in trading_days:

        date = pd.Timestamp(
            date
        ).normalize()

        pool = ranking_by_date.get(
            date,
            []
        )

        rank_lookup = {

            item["symbol"]:
            rank

            for rank, item
            in enumerate(
                pool,
                start=1
            )

        }

        target_symbols = [

            item["symbol"]

            for item in pool[:TOP_N]

        ]

        # ====================================================
        # STEP 1
        # VALUE EXISTING PORTFOLIO AT TODAY'S CLOSE
        # ====================================================

        pre_rebalance_value = cash

        for symbol, pos in (
            holdings.items()
        ):

            price = get_price(
                all_signals,
                symbol,
                date
            )

            if price is None:

                price = (
                    pos["last_price"]
                )

            pre_rebalance_value += (
                pos["qty"] *
                price
            )

        # ====================================================
        # STEP 2
        # DETERMINE TARGET CAPITAL
        # ====================================================

        target_value = (
            pre_rebalance_value /
            TOP_N
        )

        # ====================================================
        # STEP 3
        # SELL STOCKS THAT ARE NO LONGER TOP 15
        # OR NO LONGER ELIGIBLE
        #
        # ALSO SELL EXCESS SHARES FROM CURRENT
        # TOP-10 POSITIONS TO REBALANCE.
        # ====================================================

        for symbol in list(
            holdings.keys()
        ):

            pos = holdings[
                symbol
            ]

            price = get_price(
                all_signals,
                symbol,
                date
            )

            if price is None:

                price = (
                    pos["last_price"]
                )

            current_value = (
                pos["qty"] *
                price
            )

            rank = rank_lookup.get(
                symbol
            )

            # ------------------------------------------------
            # EXIT CONDITION
            # ------------------------------------------------

            must_exit = (

                rank is None

                or

                rank > EXIT_RANK

            )

            # ------------------------------------------------
            # TARGET VALUE
            # ------------------------------------------------

            if (
                symbol in target_symbols
                and not must_exit
            ):

                desired_value = (
                    target_value
                )

                desired_qty = int(
                    desired_value //
                    price
                )

                excess_qty = max(

                    0,

                    pos["qty"] -
                    desired_qty

                )

            else:

                excess_qty = (
                    pos["qty"]
                )

            # ------------------------------------------------
            # SELL
            # ------------------------------------------------

            if excess_qty <= 0:

                continue

            sell_qty = (
                excess_qty
            )

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

            cost_basis_per_share = (

                (
                    pos["qty"] *
                    pos["entry_price"]
                )
                +
                pos["entry_cost"]

            ) / pos["qty"]

            cost_basis = (
                sell_qty *
                cost_basis_per_share
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

            exit_reason = (

                "Rank > 15 / "
                "left eligible universe"

                if must_exit

                else

                "Daily equal-weight rebalance"

            )

            trade_log.append({

                "symbol":
                symbol,

                "entry_date":
                pos["entry_date"]
                .strftime("%Y-%m-%d"),

                "exit_date":
                date.strftime(
                    "%Y-%m-%d"
                ),

                "side":
                "SELL",

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
                    pos["entry_cost"] *
                    (
                        sell_qty /
                        (
                            pos["qty"] +
                            sell_qty
                        )
                    ),
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
                )
                if cost_basis > 0
                else 0,

                "days_held":
                (
                    date -
                    pos["entry_date"]
                ).days,

                "reason":
                exit_reason,

                "rank":
                rank
                if rank is not None
                else ""

            })

            pos["last_price"] = (
                price
            )

            if pos["qty"] <= 0:

                del holdings[
                    symbol
                ]

        # ====================================================
        # STEP 4
        # BUY / TOP-UP TODAY'S TOP 10
        #
        # TARGET = EQUAL WEIGHT AT TODAY'S CLOSE
        # ====================================================

        # Recalculate portfolio value after sells.
        current_value = cash

        for symbol, pos in (
            holdings.items()
        ):

            price = get_price(
                all_signals,
                symbol,
                date
            )

            if price is None:

                price = (
                    pos["last_price"]
                )

            current_value += (
                pos["qty"] *
                price
            )

        target_value = (
            current_value /
            TOP_N
        )

        for symbol in target_symbols:

            price = get_price(
                all_signals,
                symbol,
                date
            )

            if price is None:

                continue

            if symbol in holdings:

                current_qty = (
                    holdings[symbol]["qty"]
                )

            else:

                current_qty = 0

            current_value_stock = (
                current_qty *
                price
            )

            required_value = (
                target_value -
                current_value_stock
            )

            if required_value <= 0:

                continue

            # ------------------------------------------------
            # COST-AWARE QUANTITY
            # ------------------------------------------------

            qty = int(
                required_value //
                price
            )

            while qty > 0:

                trade_value = (
                    qty *
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

                if (
                    total_required <=
                    cash
                ):

                    break

                qty -= 1

            if qty <= 0:

                continue

            trade_value = (
                qty *
                price
            )

            buy_cost = (
                buy_side_cost(
                    trade_value
                )

            )

            cash -= (
                trade_value +
                buy_cost
            )

            # ------------------------------------------------
            # NEW POSITION
            # ------------------------------------------------

            if symbol not in holdings:

                holdings[symbol] = {

                    "qty":
                    qty,

                    "entry_price":
                    price,

                    "entry_date":
                    date,

                    "entry_cost":
                    buy_cost,

                    "last_price":
                    price

                }

            # ------------------------------------------------
            # EXISTING POSITION TOP-UP
            # ------------------------------------------------

            else:

                pos = holdings[
                    symbol
                ]

                old_qty = (
                    pos["qty"]
                )

                old_cost = (
                    old_qty *
                    pos["entry_price"]
                )

                new_cost = (
                    qty *
                    price
                )

                total_qty = (
                    old_qty +
                    qty
                )

                pos["entry_price"] = (

                    (
                        old_cost +
                        new_cost
                    ) /
                    total_qty

                )

                pos["entry_cost"] += (
                    buy_cost
                )

                pos["qty"] = (
                    total_qty
                )

                pos["last_price"] = (
                    price
                )

            trade_log.append({

                "symbol":
                symbol,

                "entry_date":
                date.strftime(
                    "%Y-%m-%d"
                ),

                "exit_date":
                "",

                "side":
                "BUY",

                "qty":
                qty,

                "entry_price":
                round(
                    price,
                    2
                ),

                "exit_price":
                "",

                "gross_return_pct":
                "",

                "buy_cost_rs":
                round(
                    buy_cost,
                    2
                ),

                "sell_cost_rs":
                0,

                "stcg_tax_rs":
                0,

                "net_pnl_rs":
                "",

                "net_return_pct":
                "",

                "days_held":
                "",

                "reason":
                "Same-day EOD Top-10 rebalance",

                "rank":
                rank_lookup.get(
                    symbol,
                    ""
                )

            })

        # ====================================================
        # STEP 5
        # FINAL SAME-DAY EOD MARK
        # ====================================================

        eod_value = cash

        for symbol, pos in (
            holdings.items()
        ):

            price = get_price(
                all_signals,
                symbol,
                date
            )

            if price is None:

                price = (
                    pos["last_price"]
                )

            pos["last_price"] = (
                price
            )

            eod_value += (
                pos["qty"] *
                price
            )

        # ====================================================
        # EOD HOLDING DETAILS
        # ====================================================

        for symbol in target_symbols:

            if symbol not in holdings:

                continue

            pos = holdings[
                symbol
            ]

            price = get_price(
                all_signals,
                symbol,
                date
            )

            if price is None:

                price = (
                    pos["last_price"]
                )

            stock_value = (
                pos["qty"] *
                price
            )

            weight = (

                stock_value /
                eod_value *
                100

            ) if eod_value > 0 else 0

            daily_holdings.append({

                "Date":
                date.strftime(
                    "%Y-%m-%d"
                ),

                "Symbol":
                symbol,

                "Rank":
                rank_lookup.get(
                    symbol,
                    ""
                ),

                "Qty":
                pos["qty"],

                "Close":
                round(
                    price,
                    2
                ),

                "Market Value Rs":
                round(
                    stock_value,
                    2
                ),

                "EOD Weight %":
                round(
                    weight,
                    3
                )

            })

        # ====================================================
        # EQUITY CURVE
        # ====================================================

        equity_curve.append({

            "date":
            date.strftime(
                "%Y-%m-%d"
            ),

            "portfolio_value_rs":
            round(
                eod_value,
                2
            ),

            "equity":
            round(
                eod_value /
                STARTING_CAPITAL,
                6
            ),

            "cash_rs":
            round(
                cash,
                2
            ),

            "n_holdings":
            len(holdings)

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

    liquidation_cash = (
        cash
    )

    open_positions_detail = []

    if len(trading_days) and holdings:

        last_date = (
            pd.Timestamp(
                trading_days[-1]
            ).normalize()
        )

        for symbol, pos in (
            holdings.items()
        ):

            exit_price = get_price(
                all_signals,
                symbol,
                last_date
            )

            if exit_price is None:

                exit_price = (
                    pos["last_price"]
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
                symbol,

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

    trade_df = pd.DataFrame(
        trade_log
    )

    equity_df = pd.DataFrame(
        equity_curve
    )

    open_df = pd.DataFrame(
        open_positions_detail
    )

    daily_holdings_df = pd.DataFrame(
        daily_holdings
    )

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

        open_df,

        daily_holdings_df,

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
        equity_df["equity"]
        .cummax()
    )

    drawdown = (

        equity_df["equity"] /
        running_max -
        1

    ) * 100

    max_dd = round(
        drawdown.min(),
        2
    )

    if not trade_df.empty:

        sells = trade_df[
            trade_df["side"] == "SELL"
        ].copy()

        n = len(sells)

        if n:

            win_rate = round(

                (
                    sells[
                        "net_return_pct"
                    ].astype(float)
                    > 0
                ).mean() * 100,

                1

            )

            avg_net = round(

                sells[
                    "net_return_pct"
                ].astype(float)
                .mean(),

                2

            )

            median_net = round(

                sells[
                    "net_return_pct"
                ].astype(float)
                .median(),

                2

            )

            avg_days = round(

                sells[
                    "days_held"
                ].astype(float)
                .mean(),

                1

            )

            best = sells[
                "gross_return_pct"
            ].astype(float).max()

            worst = sells[
                "gross_return_pct"
            ].astype(float).min()

            total_costs = (

                sells[
                    "sell_cost_rs"
                ].astype(float).sum()

                +

                trade_df[
                    trade_df["side"] == "BUY"
                ][
                    "buy_cost_rs"
                ].astype(float).sum()

            )

            total_tax = sells[
                "stcg_tax_rs"
            ].astype(float).sum()

            winners = sells[
                sells[
                    "net_return_pct"
                ].astype(float) > 0
            ]

            losers = sells[
                sells[
                    "net_return_pct"
                ].astype(float) < 0
            ]

            avg_winner = (

                winners[
                    "net_return_pct"
                ].astype(float).mean()

                if len(winners)

                else 0

            )

            avg_loser = (

                losers[
                    "net_return_pct"
                ].astype(float).mean()

                if len(losers)

                else 0

            )

            gp = (

                winners[
                    "net_pnl_rs"
                ].astype(float).sum()

                if len(winners)

                else 0

            )

            gl = abs(

                losers[
                    "net_pnl_rs"
                ].astype(float).sum()

            ) if len(losers) else 0

            profit_factor = (

                gp / gl

                if gl > 0

                else 0

            )

        else:

            n = 0
            win_rate = 0
            avg_net = 0
            median_net = 0
            avg_days = 0
            best = 0
            worst = 0
            total_costs = 0
            total_tax = 0
            avg_winner = 0
            avg_loser = 0
            profit_factor = 0

    else:

        n = 0
        win_rate = 0
        avg_net = 0
        median_net = 0
        avg_days = 0
        best = 0
        worst = 0
        total_costs = 0
        total_tax = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0

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
        abs(max_dd / 100)

        if abs(max_dd) > 0

        else 0

    )

    return {

        "Backtest Start":
        BACKTEST_START,

        "Backtest End":
        equity_df["date"].iloc[-1],

        "Starting Capital (Rs)":
        STARTING_CAPITAL,

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
        max_dd,

        "Number of Closed Trades":
        n,

        "Win Rate - Net (%)":
        win_rate,

        "Average Net Return / Exit (%)":
        round(
            avg_net,
            2
        ),

        "Median Net Return / Exit (%)":
        round(
            median_net,
            2
        ),

        "Average Days Held":
        avg_days,

        "Average Winner (%)":
        round(
            avg_winner,
            2
        ),

        "Average Loser (%)":
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
            best,
            2
        ),

        "Worst Gross Trade (%)":
        round(
            worst,
            2
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
        )

    }


# ============================================================
# CHUNK WRITER
# ============================================================

def write_in_chunks(
    ws,
    rows,
    start_row,
    chunk_size=2000,
    label="data"
):

    total = len(rows)

    if total == 0:

        return

    for i in range(
        0,
        total,
        chunk_size
    ):

        chunk = rows[
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
                f"{label} {i}: {e}"
            )

            time.sleep(5)

            ws.update(
                chunk,
                f"A{row_start}"
            )

        print(
            f"Wrote {label}: "
            f"{min(i + chunk_size, total)}/"
            f"{total}"
        )


# ============================================================
# DELETE EXISTING CHARTS
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
                ==
                sheet_id
            ):

                for chart in sheet.get(
                    "charts",
                    []
                ):

                    requests.append({

                        "deleteEmbeddedObject":
                        {

                            "objectId":
                            chart["chartId"]

                        }

                    })

        if requests:

            sh.batch_update({

                "requests":
                requests

            })

    except Exception as e:

        print(
            "Chart removal failed "
            f"(non-fatal): {e}"
        )


# ============================================================
# ADD CHARTS
# ============================================================

def add_charts(
    sh,
    sheet_id,
    header_row_0idx,
    n_rows
):

    end_row = (
        header_row_0idx +
        1 +
        n_rows
    )

    def make_chart(
        title,
        y_col,
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
                                                    header_row_0idx,

                                                    "endRowIndex":
                                                    end_row,

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
                                                    header_row_0idx,

                                                    "endRowIndex":
                                                    end_row,

                                                    "startColumnIndex":
                                                    y_col,

                                                    "endColumnIndex":
                                                    y_col + 1

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

        make_chart(

            "Equity Curve",

            1,

            "Portfolio Value Rs",

            header_row_0idx

        ),

        make_chart(

            "Drawdown",

            5,

            "Drawdown %",

            header_row_0idx + 22

        )

    ]

    try:

        sh.batch_update({

            "requests":
            requests

        })

    except Exception as e:

        print(
            "Chart creation failed "
            f"(non-fatal): {e}"
        )


# ============================================================
# GET / CREATE WORKSHEET
# ============================================================

def get_or_create_worksheet(
    sh,
    title,
    rows,
    cols
):

    try:

        ws = sh.worksheet(
            title
        )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(

            title=title,

            rows=max(
                rows,
                100
            ),

            cols=max(
                cols,
                10
            )

        )

    if ws.row_count < rows:

        ws.resize(
            rows=rows
        )

    if ws.col_count < cols:

        ws.resize(
            cols=cols
        )

    return ws


# ============================================================
# WRITE BACKTEST SHEET
# ============================================================

def write_backtest_sheet(
    sh,
    trade_df,
    equity_df,
    open_df,
    summary,
    effective_end_str
):

    required_rows = (

        len(trade_df)
        +
        len(equity_df)
        +
        len(open_df)
        +
        len(summary)
        +
        100

    )

    ws = get_or_create_worksheet(

        sh,

        BACKTEST_WORKSHEET,

        required_rows,

        16

    )

    remove_existing_charts(
        sh,
        ws.id
    )

    ws.clear()

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    ws.update(

        [[

            "DAILY TOP 10 RS BACKTEST | "
            f"Run {timestamp} | "
            "SAME-DAY CLOSE EXECUTION | "
            f"{BACKTEST_START} to "
            f"{effective_end_str}"

        ]],

        "A1"

    )

    summary_rows = (

        [["Summary", ""]]

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

    # --------------------------------------------------------
    # TRADE LOG
    # --------------------------------------------------------

    trade_start = (
        3 +
        len(summary_rows) +
        2
    )

    ws.update(
        [["Trade Log"]],
        f"A{trade_start}"
    )

    trade_header = (
        trade_start + 1
    )

    if not trade_df.empty:

        rows = (

            [
                list(
                    trade_df.columns
                )
            ]

            +

            trade_df.fillna("")
            .values
            .tolist()

        )

        write_in_chunks(

            ws,

            rows,

            trade_header,

            label="trade log"

        )

    # --------------------------------------------------------
    # OPEN POSITIONS
    # --------------------------------------------------------

    open_start = (

        trade_header
        +
        max(
            len(trade_df),
            1
        )
        +
        3

    )

    ws.update(

        [[
            "Open Positions at "
            "Backtest End"
        ]],

        f"A{open_start}"

    )

    open_header = (
        open_start + 1
    )

    if not open_df.empty:

        ws.update(

            [
                list(
                    open_df.columns
                )
            ]

            +

            open_df.fillna("")
            .values
            .tolist(),

            f"A{open_header}"

        )

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    equity_start = (

        open_header
        +
        max(
            len(open_df),
            1
        )
        +
        3

    )

    ws.update(

        [["Daily Equity Curve"]],

        f"A{equity_start}"

    )

    equity_header = (
        equity_start + 1
    )

    if not equity_df.empty:

        rows = (

            [
                list(
                    equity_df.columns
                )
            ]

            +

            equity_df.fillna("")
            .values
            .tolist()

        )

        write_in_chunks(

            ws,

            rows,

            equity_header,

            label="equity"

        )

        add_charts(

            sh,

            ws.id,

            equity_header - 1,

            len(equity_df)

        )

    print(
        f"Written: {BACKTEST_WORKSHEET}"
    )


# ============================================================
# WRITE TOP 10 SHEET
# ============================================================

def write_top10_sheet(
    sh,
    daily_rank_df,
    daily_holdings_df
):

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    required_rows = (

        len(daily_rank_df)
        +
        len(daily_holdings_df)
        +
        100

    )

    ws = get_or_create_worksheet(

        sh,

        TOP10_WORKSHEET,

        required_rows,

        16

    )

    ws.clear()

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ws.update(

        [[

            "TOP 10 RS BACKTEST | "
            "DAILY RANKING | "
            "SAME-DAY CLOSE REBALANCE"

        ]],

        "A1"

    )

    ws.update(

        [[

            "Every row below is the "
            "actual eligible-universe ranking "
            "for that trading day's close."

        ]],

        "A2"

    )

    # --------------------------------------------------------
    # DAILY RANKING
    # --------------------------------------------------------

    ws.update(

        [["Daily RS Ranking"]],

        "A4"

    )

    rank_header_row = 5

    if not daily_rank_df.empty:

        rank_df = (
            daily_rank_df
            .copy()
        )

        # Add Held EOD information.
        if not daily_holdings_df.empty:

            held_lookup = set(

                zip(

                    daily_holdings_df[
                        "Date"
                    ],

                    daily_holdings_df[
                        "Symbol"
                    ]

                )

            )

            weight_lookup = {

                (
                    row["Date"],
                    row["Symbol"]
                ):
                row["EOD Weight %"]

                for _, row
                in daily_holdings_df.iterrows()

            }

            rank_df["Held EOD"] = [

                "YES"

                if (
                    row["Date"],
                    row["Symbol"]
                ) in held_lookup

                else "NO"

                for _, row
                in rank_df.iterrows()

            ]

            rank_df["EOD Weight %"] = [

                weight_lookup.get(

                    (
                        row["Date"],
                        row["Symbol"]
                    ),

                    ""

                )

                for _, row
                in rank_df.iterrows()

            ]

        rows = (

            [
                list(
                    rank_df.columns
                )
            ]

            +

            rank_df.fillna("")
            .values
            .tolist()

        )

        write_in_chunks(

            ws,

            rows,

            rank_header_row,

            chunk_size=2000,

            label="daily RS ranking"

        )

    # --------------------------------------------------------
    # DAILY HELD PORTFOLIO
    # --------------------------------------------------------

    if not daily_holdings_df.empty:

        holdings_start = (

            rank_header_row
            +
            len(daily_rank_df)
            +
            4

        )

        ws.update(

            [["Daily EOD Holdings"]],

            f"A{holdings_start}"

        )

        holdings_header = (
            holdings_start + 1
        )

        rows = (

            [
                list(
                    daily_holdings_df.columns
                )
            ]

            +

            daily_holdings_df
            .fillna("")
            .values
            .tolist()

        )

        write_in_chunks(

            ws,

            rows,

            holdings_header,

            chunk_size=2000,

            label="daily holdings"

        )

    print(
        f"Written: {TOP10_WORKSHEET}"
    )


# ============================================================
# GOOGLE SHEETS AUTH
# ============================================================

def get_google_sheet():

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if not sheet_id or not creds_json:

        return None

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

    return gc.open_by_key(
        sheet_id
    )


# ============================================================
# CSV FALLBACK
# ============================================================

def write_csv_fallback(
    trade_df,
    equity_df,
    open_df,
    daily_rank_df,
    daily_holdings_df
):

    print(
        "\nGoogle Sheets credentials "
        "not found."
    )

    print(
        "Saving results to CSV files."
    )

    trade_df.to_csv(
        "backtest_trades.csv",
        index=False
    )

    equity_df.to_csv(
        "backtest_equity.csv",
        index=False
    )

    open_df.to_csv(
        "backtest_open_positions.csv",
        index=False
    )

    daily_rank_df.to_csv(
        "top_10_rs_daily.csv",
        index=False
    )

    daily_holdings_df.to_csv(
        "daily_eod_holdings.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD UNIVERSE
    # ========================================================

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    # ========================================================
    # DOWNLOAD RANGE
    # ========================================================

    download_start, download_end = (
        get_download_dates()
    )

    print("=" * 70)

    print(
        f"Download start   : {download_start}"
    )

    print(
        "Download end     : "
        f"{download_end if download_end else 'LATEST AVAILABLE'}"
    )

    print(
        f"Backtest start   : {BACKTEST_START}"
    )

    print(
        "Backtest end     : "
        f"{BACKTEST_END if BACKTEST_END else 'LATEST AVAILABLE'}"
    )

    print(
        f"Price filter     : > Rs.{MIN_PRICE}"
    )

    print(
        "Liquidity filter : "
        f"{VOLUME_LOOKBACK}d avg volume "
        f"> {MIN_AVG_VOLUME:,}"
    )

    print(
        "Entry            : "
        "Price TT + RS Line TT"
    )

    print(
        f"Portfolio        : Top {TOP_N}"
    )

    print(
        "Weight           : Equal weight"
    )

    print(
        "Rebalance        : EVERY DAY"
    )

    print(
        "Execution        : SAME-DAY CLOSE"
    )

    print(
        f"Exit rank        : > {EXIT_RANK}"
    )

    print(
        f"Capital          : Rs.{STARTING_CAPITAL:,.0f}"
    )

    print("=" * 70)

    # ========================================================
    # BENCHMARK
    # ========================================================

    bench_close = (
        download_benchmark()
    )

    bench_close = (
        normalize_series_index(
            bench_close
        )
    )

    # ========================================================
    # DOWNLOAD STOCKS
    # ========================================================

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
            f"{i} - "
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
                f"Batch failed: {e}"
            )

            continue

        # ====================================================
        # PROCESS STOCKS
        # ====================================================

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

                sig = (
                    compute_signals_for_stock(

                        close,

                        volume,

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
                    f"Skipping {symbol}: {e}"
                )

        time.sleep(1)

    # ========================================================
    # SIGNAL SUMMARY
    # ========================================================

    print(
        f"\nSignals computed for "
        f"{len(all_signals)} stocks."
    )

    print(
        f"Data points repaired: "
        f"{total_bad_points}"
    )

    if not all_signals:

        raise RuntimeError(
            "No stock signals were generated."
        )

    # ========================================================
    # LATEST AVAILABLE DATE
    # ========================================================

    latest_stock_date = max(

        df.index.max()

        for df in
        all_signals.values()

    )

    benchmark_latest_date = (
        bench_close.index.max()
    )

    # ========================================================
    # EFFECTIVE END
    # ========================================================

    if BACKTEST_END is None:

        effective_end = (
            benchmark_latest_date
        )

    else:

        effective_end = min(

            pd.Timestamp(
                BACKTEST_END
            ).normalize(),

            benchmark_latest_date

        )

    # ========================================================
    # IMPORTANT:
    # USE COMMON LAST AVAILABLE DATE
    # ========================================================

    effective_end = pd.Timestamp(
        effective_end
    ).normalize()

    print(
        "\nLatest benchmark date: "
        f"{benchmark_latest_date.strftime('%Y-%m-%d')}"
    )

    print(
        "Latest stock data date: "
        f"{pd.Timestamp(latest_stock_date).strftime('%Y-%m-%d')}"
    )

    print(
        "Effective backtest end: "
        f"{effective_end.strftime('%Y-%m-%d')}"
    )

    # ========================================================
    # TRADING DAYS
    # ========================================================

    start_date = pd.Timestamp(
        BACKTEST_START
    ).normalize()

    trading_days = (
        bench_close.index[

            (
                bench_close.index
                >= start_date
            )

            &

            (
                bench_close.index
                <= effective_end
            )

        ]

    )

    trading_days = pd.DatetimeIndex(
        trading_days
    ).normalize()

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

    # ========================================================
    # DAILY RANKINGS
    # ========================================================

    print(
        "\nBuilding daily RS rankings..."
    )

    (
        ranking_by_date,
        daily_rank_df

    ) = build_daily_rankings(

        all_signals,

        trading_days

    )

    print(
        f"Daily ranking rows: "
        f"{len(daily_rank_df):,}"
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    print(
        "\nRunning same-day EOD "
        "daily rebalance..."
    )

    (

        trade_df,

        equity_df,

        open_df,

        daily_holdings_df,

        final_marked,

        final_liquidation

    ) = run_backtest(

        all_signals,

        ranking_by_date,

        trading_days

    )

    # ========================================================
    # MARK DAILY TOP-10 HOLDING STATUS
    # ========================================================

    if not daily_rank_df.empty:

        held_lookup = set(

            zip(

                daily_holdings_df[
                    "Date"
                ],

                daily_holdings_df[
                    "Symbol"
                ]

            )

        ) if not daily_holdings_df.empty else set()

        weight_lookup = (

            {

                (
                    row["Date"],
                    row["Symbol"]
                ):
                row["EOD Weight %"]

                for _, row
                in daily_holdings_df.iterrows()

            }

            if not daily_holdings_df.empty

            else {}

        )

        daily_rank_df[
            "Held EOD"
        ] = [

            "YES"

            if (
                row["Date"],
                row["Symbol"]
            ) in held_lookup

            else "NO"

            for _, row
            in daily_rank_df.iterrows()

        ]

        daily_rank_df[
            "EOD Weight %"
        ] = [

            weight_lookup.get(

                (
                    row["Date"],
                    row["Symbol"]
                ),

                ""

            )

            for _, row
            in daily_rank_df.iterrows()

        ]

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = summarize(

        trade_df,

        equity_df,

        final_marked,

        final_liquidation

    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKTEST SUMMARY"
    )

    print(
        "=" * 70
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # GOOGLE SHEETS
    # ========================================================

    sh = get_google_sheet()

    if sh is None:

        write_csv_fallback(

            trade_df,

            equity_df,

            open_df,

            daily_rank_df,

            daily_holdings_df

        )

    else:

        # ----------------------------------------------------
        # BACKTEST SHEET
        # ----------------------------------------------------

        write_backtest_sheet(

            sh,

            trade_df,

            equity_df,

            open_df,

            summary,

            effective_end.strftime(
                "%Y-%m-%d"
            )

        )

        # ----------------------------------------------------
        # SEPARATE TOP-10 SHEET
        # ----------------------------------------------------

        write_top10_sheet(

            sh,

            daily_rank_df,

            daily_holdings_df

        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKTEST COMPLETED SUCCESSFULLY."
    )

    print(
        f"Backtest end: "
        f"{effective_end.strftime('%Y-%m-%d')}"
    )

    print(
        f"Final marked value: "
        f"Rs.{final_marked:,.2f}"
    )

    print(
        f"Final liquidation value: "
        f"Rs.{final_liquidation:,.2f}"
    )

    print(
        f"Daily RS rows: "
        f"{len(daily_rank_df):,}"
    )

    print(
        f"Trade rows: "
        f"{len(trade_df):,}"
    )

    print(
        "=" * 70
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            "\nBACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise