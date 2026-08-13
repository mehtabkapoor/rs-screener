"""
RS SCREENER — SYNCHRONIZED BACKTEST
====================================

IMPORTANT
---------
This backtest uses the ACTUAL screening functions from rs_screener.py.

Do NOT duplicate screening logic here.

Source of truth:
    rs_screener.py

Therefore the backtest uses the same:

    Price filter
    Liquidity filter
    RS Score
    Price Trend Template
    RS Line Trend Template
    VCP
    Volume Dry-up
    Pivot proximity
    Breakout-volume requirement
    TOP_N
    Transaction-cost configuration

CURRENT LIVE ENTRY LOGIC
------------------------
Price > Rs.20
20D average volume > 100,000
Price Trend Template = PASS
RS Line Trend Template = PASS
VCP = PASS
Volume Dry-up = PASS
Pivot proximity = PASS

If REQUIRE_BREAKOUT_VOLUME = True:
    Breakout volume must also PASS.

Ranking:
    Raw RS Score descending

Portfolio:
    Top 10
    Equal weight
    Daily EOD rebalance

Exit:
    Leaves current Top 10

No:
    Price stop
    RS 5EMA exit
    Rank buffer
    Blue Dot entry
    Green Dot entry
    Regime filter

IMPORTANT HISTORICAL EXECUTION ASSUMPTION
-----------------------------------------
The live screener is an EOD system.

The backtest therefore uses the historical daily adjusted close
as the EOD execution price.

It does NOT use today's intraday preview price.

IMPORTANT DATA ASSUMPTION
-------------------------
The live screener does not repair historical prices before calculating
signals.

Therefore this synchronized backtest does NOT use the old backtest's
implausible-price-jump cleaning.

Both systems use Yahoo Finance auto-adjusted daily data.

OUTPUT
------
If Google credentials are configured:

    Backtest worksheet

Otherwise:

    backtest_trades.csv
    backtest_equity.csv
    backtest_open_positions.csv

"""

# ============================================================
# STANDARD LIBRARY
# ============================================================

import json
import os
import time
from datetime import datetime


# ============================================================
# THIRD-PARTY
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials


# ============================================================
# IMPORT THE LIVE SCREENER ENGINE
# ============================================================
#
# THIS IS THE KEY CHANGE.
#
# rs_screener.py is the single source of truth.
#
# ============================================================

import rs_screener as screener


# ============================================================
# BACKTEST CONFIGURATION
# ============================================================

BACKTEST_START = "2016-04-01"

# None = latest available Yahoo trading day.
#
# Example:
# BACKTEST_END = "2026-08-13"
#
BACKTEST_END = None

# Need enough history BEFORE the requested backtest start
# for:
#
#   252-day RS
#   273-day Trend Template
#   250-day diagnostics
#   60-day VCP
#
DOWNLOAD_YEARS_BEFORE_START = 3

STARTING_CAPITAL = 1_000_000

BACKTEST_WORKSHEET = "Backtest"

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BATCH_SIZE = 50


# ============================================================
# LOAD UNIVERSE
# ============================================================

def load_tickers():

    if not os.path.exists(
        screener.STOCKS_FILE
    ):
        raise FileNotFoundError(
            f"Could not find "
            f"{screener.STOCKS_FILE}"
        )

    df = pd.read_csv(
        screener.STOCKS_FILE
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
        s
        for s in symbols
        if s
    ]

    return [
        s
        if s.endswith(".NS")
        else s + ".NS"
        for s in symbols
    ]


# ============================================================
# DATE RANGE
# ============================================================

def get_download_dates():

    start = pd.Timestamp(
        BACKTEST_START
    )

    download_start = (
        start
        -
        pd.DateOffset(
            years=
            DOWNLOAD_YEARS_BEFORE_START
        )
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime(
                "%Y-%m-%d"
            ),
            None
        )

    end = pd.Timestamp(
        BACKTEST_END
    )

    download_end = (
        end
        +
        pd.Timedelta(
            days=1
        )
    )

    return (
        download_start.strftime(
            "%Y-%m-%d"
        ),
        download_end.strftime(
            "%Y-%m-%d"
        )
    )


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\nBenchmark download:"
    )

    print(
        f"  Start: {download_start}"
    )

    print(
        f"  End  : "
        f"{download_end}"
    )

    for ticker in (
        screener.BENCHMARK,
        screener.BENCHMARK_FALLBACK
    ):

        try:

            kwargs = {

                "ticker":
                    ticker,

                "start":
                    download_start,

                "interval":
                    "1d",

                "auto_adjust":
                    True,

                "progress":
                    False,
            }

            if download_end is not None:

                kwargs["end"] = (
                    download_end
                )

            data = yf.download(
                **kwargs
            )

            if data.empty:
                continue

            close = data["Close"]

            if isinstance(
                close,
                pd.DataFrame
            ):
                close = (
                    close
                    .iloc[:, 0]
                )

            close = (
                close
                .dropna()
                .sort_index()
            )

            if close.empty:
                continue

            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark "
                f"{ticker} failed: "
                f"{e}"
            )

    raise RuntimeError(
        "Could not download "
        "benchmark data."
    )


# ============================================================
# EXTRACT STOCK DATA FROM YFINANCE BATCH
# ============================================================

def extract_symbol_data(
    data,
    symbol,
    batch_size
):

    try:

        if batch_size == 1:

            sdata = data

        else:

            if not isinstance(
                data.columns,
                pd.MultiIndex
            ):
                return None

            level0 = (
                data.columns
                .get_level_values(0)
            )

            if symbol not in level0:
                return None

            sdata = data[symbol]

        if (
            "Close"
            not in
            sdata.columns
        ):
            return None

        close = (
            sdata["Close"]
            .dropna()
            .sort_index()
        )

        if close.empty:
            return None

        if (
            "Volume"
            in
            sdata.columns
        ):

            volume = (
                sdata["Volume"]
                .reindex(
                    close.index
                )
                .fillna(0)
            )

        else:

            volume = pd.Series(
                0.0,
                index=close.index
            )

        return (
            close,
            volume
        )

    except Exception:

        return None


# ============================================================
# VECTORISED PRE-FILTER
# ============================================================
#
# These calculations are NOT the final signal engine.
#
# They only identify dates where it is worth calling the actual
# functions from rs_screener.py.
#
# Final eligibility is always calculated using:
#
#     screener.compute_rs_score()
#     screener.compute_trend_template()
#     screener.compute_rs_line_template()
#     screener.compute_vcp()
#     screener.compute_volume_dryup()
#     screener.compute_pivot()
#
# ============================================================

def preliminary_candidate_mask(
    close,
    volume,
    benchmark
):

    aligned = pd.concat(
        [
            close,
            benchmark
        ],
        axis=1,
        join="inner"
    ).dropna()

    aligned.columns = [
        "stock",
        "benchmark"
    ]

    if len(aligned) < 273:

        return None

    stock = aligned[
        "stock"
    ]

    bench = aligned[
        "benchmark"
    ]

    volume = (
        volume
        .reindex(
            aligned.index
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price_pass = (
        stock
        >
        screener.MIN_PRICE
    )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    avg20 = (
        volume
        .rolling(
            screener.VOLUME_LOOKBACK
        )
        .mean()
    )

    liquidity_pass = (
        avg20
        >
        screener.MIN_AVG_VOLUME
    )

    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
    #
    # Same mathematical conditions as the live function.
    # Used only as a cheap pre-filter.
    # --------------------------------------------------------

    sma50 = (
        stock
        .rolling(50)
        .mean()
    )

    sma150 = (
        stock
        .rolling(150)
        .mean()
    )

    sma200 = (
        stock
        .rolling(200)
        .mean()
    )

    sma200_1m = (
        sma200
        .shift(21)
    )

    low52 = (
        stock
        .rolling(252)
        .min()
    )

    high52 = (
        stock
        .rolling(252)
        .max()
    )

    tt_pass = (

        (stock > sma150)

        &

        (stock > sma200)

        &

        (sma150 > sma200)

        &

        (sma200 > sma200_1m)

        &

        (sma50 > sma150)

        &

        (sma50 > sma200)

        &

        (stock > sma50)

        &

        (stock >= 1.25 * low52)

        &

        (stock >= 0.75 * high52)
    )

    # --------------------------------------------------------
    # RS LINE
    # --------------------------------------------------------

    rs_line = (
        stock
        /
        bench
    )

    rs_sma50 = (
        rs_line
        .rolling(50)
        .mean()
    )

    rs_sma150 = (
        rs_line
        .rolling(150)
        .mean()
    )

    rs_sma200 = (
        rs_line
        .rolling(200)
        .mean()
    )

    rs_sma200_1m = (
        rs_sma200
        .shift(21)
    )

    rs_low52 = (
        rs_line
        .rolling(252)
        .min()
    )

    rs_high52 = (
        rs_line
        .rolling(252)
        .max()
    )

    rs_tt_pass = (

        (rs_line > rs_sma150)

        &

        (rs_line > rs_sma200)

        &

        (rs_sma150 > rs_sma200)

        &

        (rs_sma200 > rs_sma200_1m)

        &

        (rs_sma50 > rs_sma150)

        &

        (rs_sma50 > rs_sma200)

        &

        (rs_line > rs_sma50)

        &

        (rs_line >= 1.25 * rs_low52)

        &

        (rs_line >= 0.75 * rs_high52)
    )

    # --------------------------------------------------------
    # RS SCORE PRE-FILTER
    #
    # IMPORTANT:
    #
    # The actual screener uses:
    #
    #     close.iloc[-days - 1]
    #
    # Therefore:
    #
    #     shift(days + 1)
    #
    # is used here.
    #
    # Final score is still calculated by the actual
    # screener.compute_rs_score() function.
    # --------------------------------------------------------

    p3 = (
        stock
        /
        stock.shift(64)
        - 1
    )

    p6 = (
        stock
        /
        stock.shift(127)
        - 1
    )

    p9 = (
        stock
        /
        stock.shift(190)
        - 1
    )

    p12 = (
        stock
        /
        stock.shift(253)
        - 1
    )

    rs_score = (
        (
            0.40 * p3
            +
            0.20 * p6
            +
            0.20 * p9
            +
            0.20 * p12
        )
        * 100
    )

    candidate = (

        price_pass

        &

        liquidity_pass

        &

        tt_pass

        &

        rs_tt_pass

        &

        rs_score.notna()
    )

    return {
        "aligned":
            aligned,

        "volume":
            volume,

        "candidate":
            candidate,

        "rs_score":
            rs_score,

        "tt_pass":
            tt_pass,

        "rs_tt_pass":
            rs_tt_pass,
    }


# ============================================================
# DIAGNOSTICS SERIES
# ============================================================

def compute_diagnostics_series(
    close,
    benchmark
):

    aligned = pd.concat(
        [
            close,
            benchmark
        ],
        axis=1,
        join="inner"
    ).dropna()

    aligned.columns = [
        "stock",
        "benchmark"
    ]

    rs_ratio = (
        aligned["stock"]
        /
        aligned["benchmark"]
    )

    lookback = (
        screener.LOOKBACK_DAYS
    )

    previous_rs_high = (
        rs_ratio
        .shift(1)
        .rolling(
            lookback
        )
        .max()
    )

    previous_price_high = (
        aligned["stock"]
        .shift(1)
        .rolling(
            lookback
        )
        .max()
    )

    blue_dot = (
        rs_ratio
        >
        previous_rs_high
    )

    price_new_high = (
        aligned["stock"]
        >
        previous_price_high
    )

    green_dot = (
        blue_dot
        &
        (~price_new_high)
    )

    return pd.DataFrame({

        "blue_dot":
            blue_dot
            .fillna(False),

        "one_year_rs_cross":
            blue_dot
            .fillna(False),

        "green_dot":
            green_dot
            .fillna(False),
    })


# ============================================================
# EXACT HISTORICAL SIGNAL ENGINE
# ============================================================
#
# This is the important function.
#
# The expensive VCP / dry-up / pivot calculations are performed
# only on preliminary candidates.
#
# But the actual functions are imported from rs_screener.py.
#
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
    benchmark
):

    preliminary = (
        preliminary_candidate_mask(
            close,
            volume,
            benchmark
        )
    )

    if preliminary is None:

        return None

    aligned = (
        preliminary[
            "aligned"
        ]
    )

    volume_aligned = (
        preliminary[
            "volume"
        ]
    )

    candidate_mask = (
        preliminary[
            "candidate"
        ]
    )

    # --------------------------------------------------------
    # RESULT DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        index=aligned.index
    )

    result["price"] = (
        aligned["stock"]
    )

    # --------------------------------------------------------
    # ACTUAL RS SCORE
    #
    # Use the actual live function at every date where the
    # cheap pre-filter says a calculation is possible.
    # --------------------------------------------------------

    result["rs_score"] = np.nan

    # --------------------------------------------------------
    # ACTUAL TREND TEMPLATE FLAGS
    # --------------------------------------------------------

    result["tt_pass"] = False

    result["rs_tt_pass"] = False

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    avg20 = (
        volume_aligned
        .rolling(
            screener.VOLUME_LOOKBACK
        )
        .mean()
    )

    result["liquid"] = (

        (
            aligned["stock"]
            >
            screener.MIN_PRICE
        )

        &

        (
            avg20
            >
            screener.MIN_AVG_VOLUME
        )
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    #
    # Vectorized equivalent of the actual diagnostic function.
    # These are informational only.
    # --------------------------------------------------------

    diagnostics = (
        compute_diagnostics_series(
            aligned["stock"],
            aligned["benchmark"]
        )
    )

    result[
        "blue_dot"
    ] = diagnostics[
        "blue_dot"
    ]

    result[
        "one_year_rs_cross"
    ] = diagnostics[
        "one_year_rs_cross"
    ]

    result[
        "green_dot"
    ] = diagnostics[
        "green_dot"
    ]

    # --------------------------------------------------------
    # FINAL SCREEN FIELDS
    # --------------------------------------------------------

    result[
        "vcp_pass"
    ] = False

    result[
        "volume_dryup_pass"
    ] = False

    result[
        "pivot_proximity_pass"
    ] = False

    result[
        "breakout_volume_pass"
    ] = False

    result[
        "screen_pass"
    ] = False

    # --------------------------------------------------------
    # EXACT CANDIDATE DATES
    # --------------------------------------------------------

    candidate_dates = (
        result.index[
            candidate_mask
        ]
    )

    print(
        f"    Preliminary candidates: "
        f"{len(candidate_dates)}"
    )

    # --------------------------------------------------------
    # EVALUATE EXACT LIVE FUNCTIONS
    # --------------------------------------------------------

    for date in candidate_dates:

        # Historical data up to THIS DATE ONLY.
        #
        # This prevents look-ahead.
        #
        close_to_date = (
            aligned["stock"]
            .loc[:date]
        )

        bench_to_date = (
            aligned["benchmark"]
            .loc[:date]
        )

        volume_to_date = (
            volume_aligned
            .loc[:date]
        )

        # ----------------------------------------------------
        # ACTUAL RS SCORE
        # ----------------------------------------------------

        rs_score = (
            screener.compute_rs_score(
                close_to_date
            )
        )

        if rs_score is None:
            continue

        # ----------------------------------------------------
        # ACTUAL PRICE TT
        # ----------------------------------------------------

        tt_pass = (
            screener.compute_trend_template(
                close_to_date
            )
        )

        if tt_pass is None:
            continue

        # ----------------------------------------------------
        # ACTUAL RS LINE TT
        # ----------------------------------------------------

        rs_tt_pass = (
            screener.compute_rs_line_template(
                close_to_date,
                bench_to_date
            )
        )

        if rs_tt_pass is None:
            continue

        # ----------------------------------------------------
        # ACTUAL VCP
        # ----------------------------------------------------

        vcp = (
            screener.compute_vcp(
                close_to_date,
                volume_to_date
            )
        )

        if vcp is None:
            continue

        # ----------------------------------------------------
        # ACTUAL VOLUME DRY-UP
        # ----------------------------------------------------

        dryup = (
            screener.compute_volume_dryup(
                volume_to_date
            )
        )

        if dryup is None:
            continue

        # ----------------------------------------------------
        # ACTUAL PIVOT
        # ----------------------------------------------------

        pivot = (
            screener.compute_pivot(
                close_to_date,
                volume_to_date
            )
        )

        if pivot is None:
            continue

        # ----------------------------------------------------
        # STORE EXACT SIGNALS
        # ----------------------------------------------------

        result.loc[
            date,
            "rs_score"
        ] = rs_score

        result.loc[
            date,
            "tt_pass"
        ] = bool(
            tt_pass
        )

        result.loc[
            date,
            "rs_tt_pass"
        ] = bool(
            rs_tt_pass
        )

        result.loc[
            date,
            "vcp_pass"
        ] = bool(
            vcp["pass"]
        )

        result.loc[
            date,
            "volume_dryup_pass"
        ] = bool(
            dryup["pass"]
        )

        result.loc[
            date,
            "pivot_proximity_pass"
        ] = bool(
            pivot[
                "proximity_pass"
            ]
        )

        result.loc[
            date,
            "breakout_volume_pass"
        ] = bool(
            pivot[
                "breakout_pass"
            ]
        )

        # ----------------------------------------------------
        # EXACT LIVE SCREEN LOGIC
        #
        # Matches rs_screener.py.
        # ----------------------------------------------------

        core_screen_pass = (

            tt_pass is True

            and

            rs_tt_pass is True

            and

            vcp[
                "pass"
            ] is True

            and

            dryup[
                "pass"
            ] is True

            and

            pivot[
                "proximity_pass"
            ] is True
        )

        if (
            screener.REQUIRE_BREAKOUT_VOLUME
        ):

            screen_pass = (

                core_screen_pass

                and

                pivot[
                    "breakout_pass"
                ]
            )

        else:

            screen_pass = (
                core_screen_pass
            )

        result.loc[
            date,
            "screen_pass"
        ] = bool(
            screen_pass
        )

    return result


# ============================================================
# TRANSACTION COSTS
# ============================================================
#
# Importing these from the live screener avoids a second
# configuration.
#
# ============================================================

def buy_side_cost(
    trade_value
):

    return (
        screener.buy_side_cost(
            trade_value
        )
    )


def sell_side_cost(
    trade_value
):

    return (
        screener.sell_side_cost(
            trade_value
        )
    )


def stcg_tax(
    net_gain
):

    return (
        screener.estimate_stcg(
            net_gain
        )
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    """
    Portfolio implementation.

    ENTRY
    -----
    Today's screen_pass universe.

    Rank:
        RS Score descending.

    Select:
        Top 10.

    Weight:
        Equal capital allocation per slot.

    EXIT
    ----
    Anything not in today's Top 10.

    No other exit.
    """

    cash = (
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    # --------------------------------------------------------
    # DAILY LOOP
    # --------------------------------------------------------

    for date in trading_days:

        # ====================================================
        # BUILD TODAY'S EXACT ELIGIBLE POOL
        # ====================================================

        pool = []

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
                row["screen_pass"]
            ):
                continue

            pool.append(
                (
                    symbol,
                    float(
                        row["rs_score"]
                    )
                )
            )

        # ====================================================
        # RANK BY RAW RS SCORE
        # ====================================================

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        target_top10 = {
            symbol
            for symbol, score
            in pool[:screener.TOP_N]
        }

        # ====================================================
        # EXITS
        # ====================================================

        for symbol in list(
            holdings.keys()
        ):

            if symbol in target_top10:
                continue

            df = (
                all_signals[
                    symbol
                ]
            )

            if date not in df.index:
                continue

            position = (
                holdings.pop(
                    symbol
                )
            )

            exit_price = float(
                df.loc[
                    date,
                    "price"
                ]
            )

            gross_proceeds = (
                position["qty"]
                *
                exit_price
            )

            sell_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds
                -
                sell_cost
            )

            cost_basis = (
                position["qty"]
                *
                position[
                    "entry_price"
                ]
                +
                position[
                    "entry_cost"
                ]
            )

            net_gain = (
                net_proceeds
                -
                cost_basis
            )

            tax = (
                stcg_tax(
                    net_gain
                )
            )

            cash += (
                net_proceeds
                -
                tax
            )

            gross_return_pct = (
                (
                    exit_price
                    /
                    position[
                        "entry_price"
                    ]
                    -
                    1
                )
                *
                100
            )

            net_return_pct = (

                (
                    net_gain
                    -
                    tax
                )
                /
                cost_basis
                *
                100

                if cost_basis > 0
                else 0
            )

            days_held = (
                date
                -
                position[
                    "entry_date"
                ]
            ).days

            trade_log.append({

                "symbol":
                    symbol,

                "entry_date":
                    position[
                        "entry_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "exit_date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "qty":
                    position[
                        "qty"
                    ],

                "entry_price":
                    round(
                        position[
                            "entry_price"
                        ],
                        2
                    ),

                "exit_price":
                    round(
                        exit_price,
                        2
                    ),

                "gross_return_pct":
                    round(
                        gross_return_pct,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        position[
                            "entry_cost"
                        ],
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
                        net_return_pct,
                        2
                    ),

                "days_held":
                    days_held,

                "exit_reason":
                    "Left Top 10",
            })

        # ====================================================
        # PORTFOLIO VALUE AFTER EXITS
        # ====================================================

        portfolio_value = cash

        for symbol, position in (
            holdings.items()
        ):

            df = (
                all_signals[
                    symbol
                ]
            )

            if date in df.index:

                price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    position[
                        "entry_price"
                    ]
                )

            portfolio_value += (
                position["qty"]
                *
                price
            )

        # ====================================================
        # NEW ENTRIES
        # ====================================================

        slots_open = (
            screener.TOP_N
            -
            len(holdings)
        )

        if slots_open > 0:

            # Equal-weight target allocation.
            #
            # Each new position receives approximately:
            #
            # portfolio_value / TOP_N
            #
            slot_capital = (
                portfolio_value
                /
                screener.TOP_N
            )

            for symbol, score in (
                pool[
                    :screener.TOP_N
                ]
            ):

                if slots_open <= 0:
                    break

                if symbol in holdings:
                    continue

                df = (
                    all_signals[
                        symbol
                    ]
                )

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

                qty = int(
                    slot_capital
                    //
                    price
                )

                if qty < 1:
                    continue

                trade_value = (
                    qty
                    *
                    price
                )

                buy_cost = (
                    buy_side_cost(
                        trade_value
                    )
                )

                total_required = (
                    trade_value
                    +
                    buy_cost
                )

                if (
                    total_required
                    >
                    cash
                ):
                    continue

                cash -= (
                    total_required
                )

                holdings[
                    symbol
                ] = {

                    "qty":
                        qty,

                    "entry_price":
                        price,

                    "entry_date":
                        date,

                    "entry_cost":
                        buy_cost,
                }

                slots_open -= 1

        # ====================================================
        # DAILY MARK TO MARKET
        # ====================================================

        portfolio_value = cash

        for symbol, position in (
            holdings.items()
        ):

            df = (
                all_signals[
                    symbol
                ]
            )

            if date in df.index:

                price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    position[
                        "entry_price"
                    ]
                )

            portfolio_value += (
                position["qty"]
                *
                price
            )

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
                    portfolio_value
                    /
                    STARTING_CAPITAL,
                    8
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

    if equity_curve:

        final_marked_value = (
            equity_curve[-1][
                "portfolio_value_rs"
            ]
        )

    else:

        final_marked_value = (
            STARTING_CAPITAL
        )

    liquidation_cash = cash

    open_positions = []

    if (
        len(trading_days) > 0
        and holdings
    ):

        last_date = (
            trading_days[-1]
        )

        for symbol, position in (
            holdings.items()
        ):

            df = (
                all_signals[
                    symbol
                ]
            )

            if last_date in df.index:

                exit_price = float(
                    df.loc[
                        last_date,
                        "price"
                    ]
                )

            else:

                exit_price = (
                    position[
                        "entry_price"
                    ]
                )

            gross_proceeds = (
                position["qty"]
                *
                exit_price
            )

            sell_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds
                -
                sell_cost
            )

            cost_basis = (
                position["qty"]
                *
                position[
                    "entry_price"
                ]
                +
                position[
                    "entry_cost"
                ]
            )

            net_gain = (
                net_proceeds
                -
                cost_basis
            )

            tax = (
                stcg_tax(
                    net_gain
                )
            )

            liquidation_cash += (
                net_proceeds
                -
                tax
            )

            open_positions.append({

                "symbol":
                    symbol,

                "entry_date":
                    position[
                        "entry_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "qty":
                    position[
                        "qty"
                    ],

                "entry_price":
                    round(
                        position[
                            "entry_price"
                        ],
                        2
                    ),

                "last_price":
                    round(
                        exit_price,
                        2
                    ),

                "marked_value_rs":
                    round(
                        gross_proceeds,
                        2
                    ),

                "hypothetical_sell_cost_rs":
                    round(
                        sell_cost,
                        2
                    ),

                "hypothetical_stcg_rs":
                    round(
                        tax,
                        2
                    ),

                "liquidation_value_rs":
                    round(
                        net_proceeds
                        -
                        tax,
                        2
                    ),

                "unrealized_gross_return_pct":
                    round(
                        (
                            exit_price
                            /
                            position[
                                "entry_price"
                            ]
                            -
                            1
                        )
                        * 100,
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

    open_df = pd.DataFrame(
        open_positions
    )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    if not equity_df.empty:

        running_max = (
            equity_df[
                "equity"
            ]
            .cummax()
        )

        equity_df[
            "drawdown_pct"
        ] = (
            (
                equity_df[
                    "equity"
                ]
                /
                running_max
                -
                1
            )
            *
            100
        ).round(3)

    else:

        equity_df[
            "drawdown_pct"
        ] = pd.Series(
            dtype=float
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

    marked_return = (
        (
            final_marked_value
            /
            STARTING_CAPITAL
            -
            1
        )
        *
        100
    )

    liquidation_return = (
        (
            final_liquidation_value
            /
            STARTING_CAPITAL
            -
            1
        )
        *
        100
    )

    running_max = (
        equity_df[
            "equity"
        ]
        .cummax()
    )

    drawdown = (
        equity_df[
            "equity"
        ]
        /
        running_max
        -
        1
    ) * 100

    max_dd = (
        drawdown.min()
    )

    if not trade_df.empty:

        closed = trade_df[
            trade_df[
                "exit_reason"
            ]
            ==
            "Left Top 10"
        ]

    else:

        closed = trade_df

    n = len(
        closed
    )

    if n:

        win_rate_net = (
            (
                closed[
                    "net_return_pct"
                ]
                > 0
            )
            .mean()
            *
            100
        )

        win_rate_gross = (
            (
                closed[
                    "gross_return_pct"
                ]
                > 0
            )
            .mean()
            *
            100
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
            ]
            > 0
        ]

        losers = closed[
            closed[
                "net_return_pct"
            ]
            < 0
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

        gross_loss = abs(
            losers[
                "net_pnl_rs"
            ].sum()
        ) if len(losers) else 0

        profit_factor = (
            gross_profit
            /
            gross_loss
            if gross_loss > 0
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
        total_costs = 0
        total_tax = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0

    # ========================================================
    # RISK METRICS
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

        n_days = (
            len(equity_df)
        )

        annualized_return = (
            equity_df[
                "equity"
            ].iloc[-1]
            **
            (
                252
                /
                max(
                    n_days,
                    1
                )
            )
            -
            1
        )

        annualized_vol = (
            daily_std
            *
            np.sqrt(252)
        )

        sharpe = (
            daily_mean
            /
            daily_std
            *
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
            daily_mean
            /
            downside_std
            *
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
        annualized_return
        /
        abs(
            max_dd / 100
        )
        if abs(max_dd) > 0
        else 0
    )

    return {

        "Strategy":
            "LIVE SCREENER ENGINE",

        "Entry":
            (
                "Price + Liquidity + "
                "Price TT + RS TT + "
                "VCP + Dry-up + Pivot"
            ),

        "Breakout Volume Required":
            screener.REQUIRE_BREAKOUT_VOLUME,

        "Ranking":
            "Raw RS Score descending",

        "Portfolio":
            f"Top {screener.TOP_N}, "
            "equal weight",

        "Exit":
            f"Leaves Top {screener.TOP_N}",

        "Starting Capital (Rs)":
            STARTING_CAPITAL,

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
                annualized_return
                * 100,
                2
            ),

        "Annualized Volatility (%)":
            round(
                annualized_vol
                * 100,
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

        "Number of Closed Trades":
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

        "Profit Factor (net)":
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
            ),
    }


# ============================================================
# GOOGLE SHEETS — CHUNKED WRITE
# ============================================================

def write_in_chunks(
    ws,
    rows,
    start_row,
    chunk_size=2000,
    label="data"
):

    total = len(
        rows
    )

    if total == 0:
        return

    for i in range(
        0,
        total,
        chunk_size
    ):

        chunk = rows[
            i:
            i + chunk_size
        ]

        row_start = (
            start_row
            +
            i
        )

        try:

            ws.update(
                chunk,
                f"A{row_start}"
            )

        except Exception as e:

            print(
                f"Write failed "
                f"for {label}: "
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
# REMOVE OLD CHARTS
# ============================================================

def remove_existing_charts(
    sh,
    sheet_id
):

    try:

        metadata = (
            sh.fetch_sheet_metadata()
        )

        requests = []

        for sheet in metadata.get(
            "sheets",
            []
        ):

            properties = (
                sheet.get(
                    "properties",
                    {}
                )
            )

            if (
                properties.get(
                    "sheetId"
                )
                !=
                sheet_id
            ):
                continue

            for chart in sheet.get(
                "charts",
                []
            ):

                requests.append({

                    "deleteEmbeddedObject":
                        {
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

            print(
                f"Removed "
                f"{len(requests)} "
                f"old chart(s)."
            )

    except Exception as e:

        print(
            "Could not remove "
            f"old charts: {e}"
        )


# ============================================================
# ADD GOOGLE SHEETS CHARTS
# ============================================================

def add_charts(
    sh,
    sheet_id,
    equity_header_row_0idx,
    n_equity_rows
):

    data_end_row = (
        equity_header_row_0idx
        +
        1
        +
        n_equity_rows
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
                                        "Date"
                                },

                                {
                                    "position":
                                        "LEFT_AXIS",

                                    "title":
                                        y_axis_title
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

        make_chart(
            "Equity Curve (Rs)",
            1,
            "Portfolio Value (Rs)",
            equity_header_row_0idx
        ),

        make_chart(
            "Drawdown (%)",
            5,
            "Drawdown %",
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
            f"Could not add "
            f"charts: {e}"
        )


# ============================================================
# WRITE RESULTS
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
            "\nMissing "
            "SHEET_ID/"
            "GOOGLE_CREDENTIALS."
        )

        print(
            "Saving CSV files."
        )

        trade_df.to_csv(
            "backtest_trades.csv",
            index=False
        )

        equity_df.to_csv(
            "backtest_equity.csv",
            index=False
        )

        if not open_df.empty:

            open_df.to_csv(
                "backtest_open_positions.csv",
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

    # --------------------------------------------------------
    # SHEET SIZE
    # --------------------------------------------------------

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    n_rows_needed = (
        len(trade_df)
        +
        len(equity_df)
        +
        len(open_df)
        +
        len(summary)
        +
        80
    )

    n_cols_needed = 20

    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )

        if (
            ws.row_count
            <
            n_rows_needed
            or
            ws.col_count
            <
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
    # CLEAR
    # --------------------------------------------------------

    remove_existing_charts(
        sh,
        ws.id
    )

    ws.clear()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    ws.update(

        [[

            "SYNCHRONIZED LIVE-SCREENER BACKTEST"
            f" | Run {timestamp}"
            " | NET of costs + STCG"
            f" | Starting capital "
            f"Rs.{STARTING_CAPITAL:,.0f}"
            f" | {BACKTEST_START}"
            f" to {effective_end_str}"
            " | VCP + Dry-up + Pivot"

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
                key,
                value
            ]
            for key, value
            in summary.items()
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
        3
        +
        len(summary_rows)
        +
        2
    )

    ws.update(
        [["Trade Log"]],
        f"A{trade_start_row}"
    )

    trade_header_row = (
        trade_start_row
        +
        1
    )

    if not trade_df.empty:

        trade_rows = (

            [
                list(
                    trade_df.columns
                )
            ]

            +

            trade_df
            .fillna("")
            .values
            .tolist()
        )

        write_in_chunks(

            ws,

            trade_rows,

            start_row=
                trade_header_row,

            chunk_size=
                2000,

            label=
                "trade log"
        )

    # --------------------------------------------------------
    # OPEN POSITIONS
    # --------------------------------------------------------

    open_start_row = (
        trade_header_row
        +
        len(trade_df)
        +
        3
    )

    ws.update(

        [[
            "Open Positions at "
            "Backtest End "
            "(mark-to-market, not sold)"
        ]],

        f"A{open_start_row}"
    )

    open_header_row = (
        open_start_row
        +
        1
    )

    if not open_df.empty:

        open_rows = (

            [
                list(
                    open_df.columns
                )
            ]

            +

            open_df
            .fillna("")
            .values
            .tolist()
        )

        write_in_chunks(

            ws,

            open_rows,

            start_row=
                open_header_row,

            chunk_size=
                2000,

            label=
                "open positions"
        )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity_start_row = (
        open_header_row
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
        f"A{equity_start_row}"
    )

    equity_header_row = (
        equity_start_row
        +
        1
    )

    if not equity_df.empty:

        equity_rows = (

            [
                list(
                    equity_df.columns
                )
            ]

            +

            equity_df
            .fillna("")
            .values
            .tolist()
        )

        write_in_chunks(

            ws,

            equity_rows,

            start_row=
                equity_header_row,

            chunk_size=
                2000,

            label=
                "equity curve"
        )

        add_charts(

            sh,

            ws.id,

            equity_header_row - 1,

            len(equity_df)
        )

    print(
        "\nResults written to "
        f"'{BACKTEST_WORKSHEET}' "
        "worksheet."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RS SCREENER — "
        "SYNCHRONIZED BACKTEST"
    )

    print(
        "=" * 75
    )

    print(
        "\nSOURCE OF TRUTH:"
    )

    print(
        "  rs_screener.py"
    )

    print(
        "\nENTRY RULES:"
    )

    print(
        f"  Price > Rs.{screener.MIN_PRICE}"
    )

    print(
        f"  "
        f"{screener.VOLUME_LOOKBACK}D "
        f"avg volume > "
        f"{screener.MIN_AVG_VOLUME:,}"
    )

    print(
        "  Price Trend Template = 7/7"
    )

    print(
        "  RS Line Trend Template = 7/7"
    )

    print(
        "  VCP = PASS"
    )

    print(
        "  Volume Dry-up = PASS"
    )

    print(
        "  Pivot proximity = PASS"
    )

    print(
        "  Breakout volume required = "
        f"{screener.REQUIRE_BREAKOUT_VOLUME}"
    )

    print(
        "\nPORTFOLIO:"
    )

    print(
        f"  Top {screener.TOP_N}"
    )

    print(
        "  Equal weight"
    )

    print(
        "  Daily EOD rebalance"
    )

    print(
        f"  Exit = leaves Top "
        f"{screener.TOP_N}"
    )

    print(
        "\nSTARTING CAPITAL:"
    )

    print(
        f"  Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        "\nBACKTEST WINDOW:"
    )

    print(
        f"  {BACKTEST_START}"
        f" -> "
        f"{BACKTEST_END or 'latest available'}"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # LOAD STOCK LIST
    # ========================================================

    tickers = load_tickers()

    print(
        f"\nLoaded "
        f"{len(tickers)} "
        f"stocks."
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    benchmark = (
        download_benchmark()
    )

    download_start, download_end = (
        get_download_dates()
    )

    # ========================================================
    # STOCK SIGNALS
    # ========================================================

    all_signals = {}

    total_candidates = 0

    total_screen_pass = 0

    processed = 0

    # --------------------------------------------------------
    # BATCH DOWNLOAD
    # --------------------------------------------------------

    for batch_start in range(
        0,
        len(tickers),
        BATCH_SIZE
    ):

        batch = tickers[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        print(
            "\n"
            + "=" * 65
        )

        print(
            f"Downloading "
            f"batch "
            f"{batch_start + 1}"
            f"-"
            f"{batch_start + len(batch)}"
            f" / "
            f"{len(tickers)}"
        )

        print(
            "=" * 65
        )

        try:

            kwargs = {

                "tickers":
                    batch,

                "start":
                    download_start,

                "interval":
                    "1d",

                "auto_adjust":
                    True,

                "progress":
                    False,

                "group_by":
                    "ticker",

                "threads":
                    True,
            }

            if download_end is not None:

                kwargs["end"] = (
                    download_end
                )

            data = yf.download(
                **kwargs
            )

        except Exception as e:

            print(
                f"Batch failed: "
                f"{e}"
            )

            continue

        # ----------------------------------------------------
        # EACH STOCK
        # ----------------------------------------------------

        for symbol in batch:

            processed += 1

            try:

                extracted = (
                    extract_symbol_data(
                        data,
                        symbol,
                        len(batch)
                    )
                )

                if extracted is None:

                    print(
                        f"{symbol}: "
                        "no usable data"
                    )

                    continue

                close, volume = (
                    extracted
                )

                if len(close) < 273:

                    print(
                        f"{symbol}: "
                        "insufficient history"
                    )

                    continue

                print(
                    f"\n[{processed}/"
                    f"{len(tickers)}] "
                    f"{symbol}"
                )

                signals = (
                    compute_signals_for_stock(

                        close,

                        volume,

                        benchmark
                    )
                )

                if signals is None:
                    continue

                clean_symbol = (
                    symbol
                    .replace(
                        ".NS",
                        ""
                    )
                )

                all_signals[
                    clean_symbol
                ] = signals

                n_candidate = int(
                    (
                        signals[
                            "liquid"
                        ]

                        &

                        (
                            signals[
                                "tt_pass"
                            ]
                            ==
                            True
                        )

                        &

                        (
                            signals[
                                "rs_tt_pass"
                            ]
                            ==
                            True
                        )

                        &

                        signals[
                            "rs_score"
                        ].notna()
                    )
                    .sum()
                )

                n_screen = int(
                    signals[
                        "screen_pass"
                    ].sum()
                )

                total_candidates += (
                    n_candidate
                )

                total_screen_pass += (
                    n_screen
                )

                print(
                    f"    Final "
                    f"screen-pass days: "
                    f"{n_screen}"
                )

            except Exception as e:

                print(
                    f"{symbol}: "
                    f"FAILED — "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                continue

        # Small pause to reduce API pressure.
        time.sleep(1)

    # ========================================================
    # CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SIGNAL ENGINE COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Stocks with usable signals: "
        f"{len(all_signals)}"
    )

    print(
        f"Preliminary candidate days: "
        f"{total_candidates:,}"
    )

    print(
        f"Final screen-pass days: "
        f"{total_screen_pass:,}"
    )

    if not all_signals:

        raise RuntimeError(
            "No usable stock signals "
            "were generated."
        )

    # ========================================================
    # EFFECTIVE END
    # ========================================================

    if BACKTEST_END is None:

        effective_end = (
            benchmark.index.max()
        )

    else:

        effective_end = (
            pd.Timestamp(
                BACKTEST_END
            )
        )

    print(
        f"\nEffective end date: "
        f"{effective_end.strftime('%Y-%m-%d')}"
    )

    # ========================================================
    # TRADING DAYS
    # ========================================================

    trading_days = (
        benchmark.index[
            (
                benchmark.index
                >=
                pd.Timestamp(
                    BACKTEST_START
                )
            )
            &
            (
                benchmark.index
                <=
                effective_end
            )
        ]
    )

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    if len(trading_days) == 0:

        raise RuntimeError(
            "No trading days in "
            "requested backtest window."
        )

    # ========================================================
    # RUN PORTFOLIO BACKTEST
    # ========================================================

    (
        trade_df,
        equity_df,
        open_df,
        final_marked_value,
        final_liquidation_value
    ) = run_backtest(

        all_signals,

        trading_days
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = summarize(

        trade_df,

        equity_df,

        final_marked_value,

        final_liquidation_value
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL BACKTEST"
    )

    print(
        "=" * 75
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 75
    )

    # ========================================================
    # WRITE RESULTS
    # ========================================================

    write_to_sheet(

        trade_df,

        equity_df,

        open_df,

        summary,

        effective_end.strftime(
            "%Y-%m-%d"
        )
    )

    print(
        "\nBACKTEST COMPLETED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "BACKTEST FAILED"
        )

        print(
            "=" * 75
        )

        print(
            f"{type(e).__name__}: "
            f"{e}"
        )

        raise