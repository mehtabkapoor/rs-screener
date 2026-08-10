"""
======================================================================
DAILY EOD TOP-10 RS BACKTEST
======================================================================

STRATEGY
--------

Every trading day:

1. Price > Rs.20
2. 20-day average volume > 100,000 shares
3. Price Trend Template = 7/7
4. RS Line Trend Template = 7/7
5. Sort eligible stocks by RAW RS SCORE, highest first
6. Select TOP 10
7. Rebalance portfolio DAILY toward equal weights

NO:
- Blue Dot requirement
- 1-year RS crossover requirement
- RS < 5EMA exit
- 8% trailing stop
- 5% hard stop
- RS rank >20 exit

A stock is sold when:
- it leaves the current Top 10, OR
- it is overweight relative to the daily target weight.

The portfolio is therefore reconstituted DAILY.

======================================================================
EXECUTION
======================================================================

The model assumes:

Today's EOD signal
        ↓
Today's EOD close execution

This is a theoretical EOD model.

It is NOT an intraday executable backtest.

======================================================================
COSTS
======================================================================

BUY:
- STT
- Stamp duty
- NSE exchange transaction charge
- SEBI charge
- GST on exchange + SEBI

SELL:
- STT
- NSE exchange transaction charge
- SEBI charge
- GST on exchange + SEBI
- DP charge

Brokerage = zero.

======================================================================
TAX
======================================================================

Historical Section 111A STCG rates:

Before 23-Jul-2024:
    15%

From 23-Jul-2024:
    20%

Health & Education cess:
    4%

This produces:

Before 23-Jul-2024:
    15.6%

From 23-Jul-2024:
    20.8%

Tax is charged only on positive realized gains.

FIFO is used.

Losses are NOT set off against gains.

Surcharge/rebate and whole-taxpayer income interaction are NOT
modelled.

======================================================================
DATA CLEANING
======================================================================

Single-day price move > +/-30%:

    treated as potential split/bonus/data corruption

Affected close is replaced by previous valid close.

Cleaning occurs before signal calculation.

======================================================================
SURVIVORSHIP
======================================================================

stocks.csv defines the universe.

If stocks.csv is today's surviving universe projected backwards,
survivorship bias remains.

======================================================================
OUTPUT
======================================================================

backtest_transactions.csv
backtest_equity.csv
backtest_daily_top10.csv
backtest_summary.csv

RS_Top10_Equity_Curve.png
RS_Top10_Drawdown.png
"""


# ======================================================================
# IMPORTS
# ======================================================================

import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = "2026-08-07"

STARTING_CAPITAL = 1_000_000.0

TOP_N = 10


# ======================================================================
# DATA HISTORY
# ======================================================================

DOWNLOAD_YEARS_BEFORE_START = 3


# ======================================================================
# RS
# ======================================================================

RS_ONE_YEAR_LOOKBACK = 250


# ======================================================================
# LIQUIDITY
# ======================================================================

MIN_PRICE = 20.0

MIN_AVG_VOLUME = 100_000.0

VOLUME_LOOKBACK = 20


# ======================================================================
# DATA CLEANING
# ======================================================================

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ======================================================================
# TRANSACTION COSTS
# ======================================================================

ENABLE_COSTS = True

STT_BUY_RATE = 0.001

STT_SELL_RATE = 0.001

STAMP_DUTY_RATE = 0.00015

EXCHANGE_CHARGE_RATE = 0.0000325

SEBI_CHARGE_RATE = 0.000001

GST_RATE = 0.18

DP_CHARGE_FLAT = 20.0


# ======================================================================
# STCG
# ======================================================================

ENABLE_STCG = True

STCG_RATE_BEFORE_2024_07_23 = 0.15

STCG_RATE_FROM_2024_07_23 = 0.20

CESS_RATE = 0.04

STCG_CHANGE_DATE = pd.Timestamp(
    "2024-07-23"
)


# ======================================================================
# COST FUNCTIONS
# ======================================================================

def buy_side_cost(trade_value):
    """
    BUY-side transaction costs.
    """

    if not ENABLE_COSTS:
        return 0.0

    stt = (
        STT_BUY_RATE
        *
        trade_value
    )

    stamp = (
        STAMP_DUTY_RATE
        *
        trade_value
    )

    exchange = (
        EXCHANGE_CHARGE_RATE
        *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        *
        trade_value
    )

    gst = (
        GST_RATE
        *
        (
            exchange
            +
            sebi
        )
    )

    return (
        stt
        +
        stamp
        +
        exchange
        +
        sebi
        +
        gst
    )


def sell_side_cost(trade_value):
    """
    SELL-side transaction costs.
    """

    if not ENABLE_COSTS:
        return 0.0

    stt = (
        STT_SELL_RATE
        *
        trade_value
    )

    exchange = (
        EXCHANGE_CHARGE_RATE
        *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        *
        trade_value
    )

    gst = (
        GST_RATE
        *
        (
            exchange
            +
            sebi
        )
    )

    return (
        stt
        +
        exchange
        +
        sebi
        +
        gst
        +
        DP_CHARGE_FLAT
    )


def get_stcg_rate(sale_date):
    """
    Historical Section 111A rate.
    """

    sale_date = pd.Timestamp(
        sale_date
    )

    if (
        sale_date
        <
        STCG_CHANGE_DATE
    ):

        return (
            STCG_RATE_BEFORE_2024_07_23
        )

    return (
        STCG_RATE_FROM_2024_07_23
    )


def stcg_tax(
    taxable_gain,
    sale_date
):
    """
    Tax only on positive realized gains.

    Cess = 4%.
    """

    if not ENABLE_STCG:
        return 0.0

    if taxable_gain <= 0:
        return 0.0

    rate = get_stcg_rate(
        sale_date
    )

    return (
        taxable_gain
        *
        rate
        *
        (
            1
            +
            CESS_RATE
        )
    )


# ======================================================================
# DATE RANGE
# ======================================================================

def get_download_dates():

    start = pd.Timestamp(
        BACKTEST_START
    )

    end = pd.Timestamp(
        BACKTEST_END
    )

    download_start = (
        start
        -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
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


# ======================================================================
# LOAD STOCK UNIVERSE
# ======================================================================

def load_tickers():

    if not os.path.exists(
        STOCKS_FILE
    ):

        raise FileNotFoundError(
            f"Could not find "
            f"{STOCKS_FILE}"
        )

    df = pd.read_csv(
        STOCKS_FILE
    )

    if "symbol" not in df.columns:

        raise ValueError(
            "stocks.csv must contain "
            "a column named 'symbol'."
        )

    symbols = []

    for symbol in (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    ):

        if not symbol:
            continue

        if not symbol.endswith(
            ".NS"
        ):

            symbol += ".NS"

        if symbol not in symbols:

            symbols.append(
                symbol
            )

    if not symbols:

        raise ValueError(
            "No valid symbols found."
        )

    return symbols


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_price_series(close):

    close = (
        pd.to_numeric(
            close,
            errors="coerce"
        )
        .dropna()
        .sort_index()
        .copy()
    )

    daily_change = (
        close.pct_change()
    )

    bad = (
        daily_change.abs()
        >
        MAX_PLAUSIBLE_DAILY_MOVE
    )

    bad_indices = (
        close.index[bad]
    )

    repaired = 0

    for idx in bad_indices:

        position = (
            close.index.get_loc(
                idx
            )
        )

        if position > 0:

            close.iloc[position] = (
                close.iloc[
                    position - 1
                ]
            )

            repaired += 1

    return (
        close,
        repaired
    )


# ======================================================================
# BENCHMARK
# ======================================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
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

            (
                close,
                repaired
            ) = clean_price_series(
                close
            )

            if close.empty:
                continue

            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            print(
                f"Benchmark repairs: "
                f"{repaired}"
            )

            return close

        except Exception as exc:

            print(
                f"Benchmark {ticker} failed: "
                f"{exc}"
            )

    raise RuntimeError(
        "Unable to download benchmark."
    )


# ======================================================================
# TREND TEMPLATE
# ======================================================================

def trend_template_series(
    series
):

    sma50 = (
        series
        .rolling(50)
        .mean()
    )

    sma150 = (
        series
        .rolling(150)
        .mean()
    )

    sma200 = (
        series
        .rolling(200)
        .mean()
    )

    sma200_1month = (
        sma200.shift(21)
    )

    low52 = (
        series
        .rolling(252)
        .min()
    )

    high52 = (
        series
        .rolling(252)
        .max()
    )

    c1 = (
        (series > sma150)
        &
        (series > sma200)
    )

    c2 = (
        sma150 > sma200
    )

    c3 = (
        sma200 > sma200_1month
    )

    c4 = (
        (sma50 > sma150)
        &
        (sma50 > sma200)
    )

    c5 = (
        series > sma50
    )

    c6 = (
        series >=
        1.25 * low52
    )

    c7 = (
        series >=
        0.75 * high52
    )

    met = (
        c1.astype(int)
        +
        c2.astype(int)
        +
        c3.astype(int)
        +
        c4.astype(int)
        +
        c5.astype(int)
        +
        c6.astype(int)
        +
        c7.astype(int)
    )

    return (
        met == 7,
        met
    )


# ======================================================================
# STOCK SIGNALS
# ======================================================================

def compute_signals(
    close,
    volume,
    benchmark
):

    aligned = pd.concat(
        [
            close.rename("stock"),
            benchmark.rename("benchmark")
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(aligned) < 280:

        return None

    volume = (
        pd.to_numeric(
            volume,
            errors="coerce"
        )
        .reindex(
            aligned.index
        )
        .fillna(0)
    )

    price = (
        aligned["stock"]
    )

    benchmark_price = (
        aligned["benchmark"]
    )

    # ==================================================================
    # RS LINE
    # ==================================================================

    rs_line = (
        price
        /
        benchmark_price
    )

    # ==================================================================
    # RAW RS SCORE
    # ==================================================================

    def return_n_days(
        days
    ):

        return (
            price
            /
            price.shift(days)
            -
            1
        )

    rs_score = (
        0.40
        *
        return_n_days(63)
        +
        0.20
        *
        return_n_days(126)
        +
        0.20
        *
        return_n_days(189)
        +
        0.20
        *
        return_n_days(252)
    ) * 100

    # ==================================================================
    # PRICE TREND TEMPLATE
    # ==================================================================

    (
        price_tt_pass,
        price_tt_met
    ) = trend_template_series(
        price
    )

    # ==================================================================
    # RS TREND TEMPLATE
    # ==================================================================

    (
        rs_tt_pass,
        rs_tt_met
    ) = trend_template_series(
        rs_line
    )

    # ==================================================================
    # POINT-IN-TIME LIQUIDITY
    # ==================================================================

    avg_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (
        (price > MIN_PRICE)
        &
        (
            avg_volume
            >
            MIN_AVG_VOLUME
        )
    )

    # ==================================================================
    # DIAGNOSTICS ONLY
    # ==================================================================

    rs_1y_high = (
        rs_line
        .shift(1)
        .rolling(
            RS_ONE_YEAR_LOOKBACK
        )
        .max()
    )

    blue_dot = (
        rs_line
        >
        rs_1y_high
    )

    previous_rs_1y_high = (
        rs_1y_high.shift(1)
    )

    rs_cross_1y = (
        blue_dot
        &
        (
            rs_line.shift(1)
            <=
            previous_rs_1y_high
        )
    )

    previous_rs_score_high = (
        rs_score
        .shift(1)
        .rolling(
            RS_ONE_YEAR_LOOKBACK
        )
        .max()
    )

    green_dot = (
        rs_score
        >
        previous_rs_score_high
    )

    # ==================================================================
    # OUTPUT
    # ==================================================================

    return pd.DataFrame({

        "price":
            price,

        "volume":
            volume,

        "avg_volume":
            avg_volume,

        "liquid":
            liquid,

        "rs_line":
            rs_line,

        "rs_score":
            rs_score,

        "price_tt_pass":
            price_tt_pass,

        "price_tt_met":
            price_tt_met,

        "rs_tt_pass":
            rs_tt_pass,

        "rs_tt_met":
            rs_tt_met,

        # Diagnostic only
        "rs_1y_high":
            rs_1y_high,

        "blue_dot":
            blue_dot,

        "rs_cross_1y":
            rs_cross_1y,

        "green_dot":
            green_dot,
    })


# ======================================================================
# PORTFOLIO VALUE
# ======================================================================

def get_position_value(
    symbol,
    position,
    date,
    signals
):

    if symbol not in signals:
        return 0.0

    df = signals[
        symbol
    ]

    if date not in df.index:
        return 0.0

    price = float(
        df.at[
            date,
            "price"
        ]
    )

    return (
        position["qty"]
        *
        price
    )


def get_portfolio_value(
    cash,
    holdings,
    date,
    signals
):

    value = float(
        cash
    )

    for symbol, position in (
        holdings.items()
    ):

        value += (
            get_position_value(
                symbol,
                position,
                date,
                signals
            )
        )

    return value


# ======================================================================
# FIFO POSITION
# ======================================================================

def create_position(
    qty,
    price,
    buy_cost,
    date
):

    return {

        "qty":
            int(qty),

        "lots": [

            {
                "qty":
                    int(qty),

                "price":
                    float(price),

                "buy_cost":
                    float(buy_cost),

                "entry_date":
                    pd.Timestamp(
                        date
                    )
            }

        ]
    }


def add_lot(
    position,
    qty,
    price,
    buy_cost,
    date
):

    if position is None:

        return create_position(
            qty,
            price,
            buy_cost,
            date
        )

    position["qty"] += int(
        qty
    )

    position["lots"].append({

        "qty":
            int(qty),

        "price":
            float(price),

        "buy_cost":
            float(buy_cost),

        "entry_date":
            pd.Timestamp(
                date
            )
    })

    return position


# ======================================================================
# FIFO SELL
# ======================================================================

def sell_fifo(
    position,
    sell_qty,
    sell_price,
    sale_date
):

    sell_qty = int(
        sell_qty
    )

    if sell_qty <= 0:

        raise ValueError(
            "Sell quantity must be > 0."
        )

    if sell_qty > position["qty"]:

        raise ValueError(
            "Cannot sell more shares "
            "than position contains."
        )

    gross_proceeds = (
        sell_qty
        *
        sell_price
    )

    sell_cost = (
        sell_side_cost(
            gross_proceeds
        )
    )

    remaining = (
        sell_qty
    )

    cost_basis = 0.0

    new_lots = []

    for lot in position["lots"]:

        if remaining <= 0:

            new_lots.append(
                lot
            )

            continue

        take = min(
            remaining,
            lot["qty"]
        )

        lot_total_cost = (
            lot["qty"]
            *
            lot["price"]
            +
            lot["buy_cost"]
        )

        cost_per_share = (
            lot_total_cost
            /
            lot["qty"]
        )

        cost_basis += (
            take
            *
            cost_per_share
        )

        remaining_lot_qty = (
            lot["qty"]
            -
            take
        )

        if remaining_lot_qty > 0:

            remaining_total_cost = (
                remaining_lot_qty
                *
                cost_per_share
            )

            remaining_buy_cost = max(
                0.0,
                remaining_total_cost
                -
                (
                    remaining_lot_qty
                    *
                    lot["price"]
                )
            )

            new_lots.append({

                "qty":
                    remaining_lot_qty,

                "price":
                    lot["price"],

                "buy_cost":
                    remaining_buy_cost,

                "entry_date":
                    lot["entry_date"]
            })

        remaining -= take

    if remaining != 0:

        raise RuntimeError(
            "FIFO accounting failure."
        )

    # --------------------------------------------------------------
    # Simplified taxable gain:
    #
    # sale proceeds
    # - sell transaction expenses
    # - acquisition cost
    #
    # This is a backtest tax approximation, not an ITR engine.
    # --------------------------------------------------------------

    taxable_gain = (
        gross_proceeds
        -
        sell_cost
        -
        cost_basis
    )

    tax = stcg_tax(
        taxable_gain,
        sale_date
    )

    net_cash = (
        gross_proceeds
        -
        sell_cost
        -
        tax
    )

    position["lots"] = (
        new_lots
    )

    position["qty"] -= (
        sell_qty
    )

    return {

        "qty":
            sell_qty,

        "gross_proceeds":
            gross_proceeds,

        "sell_cost":
            sell_cost,

        "cost_basis":
            cost_basis,

        "taxable_gain":
            taxable_gain,

        "tax":
            tax,

        "net_cash":
            net_cash,

        "remaining_qty":
            position["qty"]
    }


# ======================================================================
# AFFORDABLE BUY
# ======================================================================

def maximum_affordable_quantity(
    requested_qty,
    price,
    cash
):

    qty = int(
        requested_qty
    )

    while qty > 0:

        trade_value = (
            qty
            *
            price
        )

        cost = (
            buy_side_cost(
                trade_value
            )
        )

        total = (
            trade_value
            +
            cost
        )

        if total <= (
            cash
            +
            1e-9
        ):

            return (
                qty,
                cost
            )

        qty -= 1

    return (
        0,
        0.0
    )


# ======================================================================
# MAIN BACKTEST
# ======================================================================

def run_backtest(
    signals,
    trading_days
):

    cash = float(
        STARTING_CAPITAL
    )

    holdings = {}

    transactions = []

    equity_rows = []

    selection_rows = []

    # ==================================================================
    # DAILY LOOP
    # ==================================================================

    for date in trading_days:

        # ==============================================================
        # BUILD RAW RS RANKING
        # ==============================================================

        rank_pool = []

        eligible = []

        for symbol, df in (
            signals.items()
        ):

            if date not in df.index:
                continue

            row = df.loc[date]

            if pd.isna(
                row["rs_score"]
            ):
                continue

            score = float(
                row["rs_score"]
            )

            # Complete RS ranking
            rank_pool.append(
                (
                    symbol,
                    score
                )
            )

            # ----------------------------------------------------------
            # ELIGIBILITY
            # ----------------------------------------------------------

            if not bool(
                row["liquid"]
            ):
                continue

            if not bool(
                row["price_tt_pass"]
            ):
                continue

            if not bool(
                row["rs_tt_pass"]
            ):
                continue

            eligible.append(
                (
                    symbol,
                    score
                )
            )

        # ==============================================================
        # SORT
        # ==============================================================

        rank_pool.sort(
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        overall_rank = {

            symbol:
                rank

            for rank, (
                symbol,
                _
            ) in enumerate(
                rank_pool,
                start=1
            )
        }

        eligible.sort(
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        # ==============================================================
        # TOP 10
        # ==============================================================

        top10 = [
            symbol
            for symbol, _
            in eligible[:TOP_N]
        ]

        top10_set = set(
            top10
        )

        # ==============================================================
        # DAILY SELECTION AUDIT
        # ==============================================================

        for rank, (
            symbol,
            score
        ) in enumerate(
            eligible[:TOP_N],
            start=1
        ):

            row = signals[
                symbol
            ].loc[date]

            selection_rows.append({

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "rank":
                    rank,

                "symbol":
                    symbol,

                "rs_score":
                    round(
                        score,
                        6
                    ),

                "overall_rs_rank":
                    overall_rank[
                        symbol
                    ],

                "price":
                    round(
                        float(
                            row["price"]
                        ),
                        2
                    ),

                "avg_volume":
                    round(
                        float(
                            row[
                                "avg_volume"
                            ]
                        ),
                        0
                    ),

                "price_tt_met":
                    int(
                        row[
                            "price_tt_met"
                        ]
                    ),

                "rs_tt_met":
                    int(
                        row[
                            "rs_tt_met"
                        ]
                    ),

                "blue_dot":
                    bool(
                        row[
                            "blue_dot"
                        ]
                    ),

                "rs_cross_1y":
                    bool(
                        row[
                            "rs_cross_1y"
                        ]
                    ),

                "green_dot":
                    bool(
                        row[
                            "green_dot"
                        ]
                    )
            })

        # ==============================================================
        # PORTFOLIO VALUE BEFORE REBALANCE
        # ==============================================================

        portfolio_value = (
            get_portfolio_value(
                cash,
                holdings,
                date,
                signals
            )
        )

        n_targets = len(
            top10
        )

        if n_targets > 0:

            target_value = (
                portfolio_value
                /
                n_targets
            )

        else:

            target_value = 0.0

        # ==============================================================
        # TARGET SHARE COUNTS
        # ==============================================================

        target_qty = {}

        for symbol in top10:

            price = float(
                signals[
                    symbol
                ].at[
                    date,
                    "price"
                ]
            )

            target_qty[
                symbol
            ] = int(
                target_value
                //
                price
            )

        # ==============================================================
        # SELL:
        #
        # 1. Stocks leaving Top 10
        # 2. Existing stocks above target quantity
        # ==============================================================

        sell_orders = []

        for symbol, position in list(
            holdings.items()
        ):

            if symbol not in top10_set:

                sell_quantity = (
                    position["qty"]
                )

                reason = (
                    "LEFT TOP 10"
                )

            else:

                sell_quantity = max(
                    0,
                    position["qty"]
                    -
                    target_qty[
                        symbol
                    ]
                )

                reason = (
                    "DAILY REBALANCE"
                )

            if sell_quantity > 0:

                sell_orders.append({

                    "symbol":
                        symbol,

                    "qty":
                        sell_quantity,

                    "reason":
                        reason
                })

        # ==============================================================
        # EXECUTE SELLS
        # ==============================================================

        for order in sell_orders:

            symbol = (
                order["symbol"]
            )

            quantity = (
                order["qty"]
            )

            reason = (
                order["reason"]
            )

            if (
                symbol
                not in
                holdings
            ):

                continue

            if date not in signals[
                symbol
            ].index:

                continue

            price = float(
                signals[
                    symbol
                ].at[
                    date,
                    "price"
                ]
            )

            position = holdings[
                symbol
            ]

            result = sell_fifo(
                position,
                quantity,
                price,
                date
            )

            cash += (
                result["net_cash"]
            )

            row = signals[
                symbol
            ].loc[date]

            transactions.append({

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "symbol":
                    symbol,

                "side":
                    "SELL",

                "qty":
                    quantity,

                "price":
                    price,

                "gross_value_rs":
                    result[
                        "gross_proceeds"
                    ],

                "buy_cost_rs":
                    0.0,

                "sell_cost_rs":
                    result[
                        "sell_cost"
                    ],

                "stcg_tax_rs":
                    result[
                        "tax"
                    ],

                "cost_basis_rs":
                    result[
                        "cost_basis"
                    ],

                "realized_gain_after_tax_rs":
                    (
                        result[
                            "taxable_gain"
                        ]
                        -
                        result[
                            "tax"
                        ]
                    ),

                "net_cash_flow_rs":
                    result[
                        "net_cash"
                    ],

                "reason":
                    reason,

                "rs_score":
                    float(
                        row[
                            "rs_score"
                        ]
                    ),

                "overall_rs_rank":
                    overall_rank.get(
                        symbol,
                        ""
                    ),

                "price_tt_met":
                    int(
                        row[
                            "price_tt_met"
                        ]
                    ),

                "rs_tt_met":
                    int(
                        row[
                            "rs_tt_met"
                        ]
                    ),

                "holding_qty_after":
                    position[
                        "qty"
                    ]
            })

            if position[
                "qty"
            ] == 0:

                del holdings[
                    symbol
                ]

        # ==============================================================
        # REVALUE AFTER SELLS
        # ==============================================================

        portfolio_value = (
            get_portfolio_value(
                cash,
                holdings,
                date,
                signals
            )
        )

        if n_targets > 0:

            target_value = (
                portfolio_value
                /
                n_targets
            )

        else:

            target_value = 0.0

        # ==============================================================
        # BUY UNDERWEIGHT POSITIONS
        # ==============================================================

        for rank, symbol in enumerate(
            top10,
            start=1
        ):

            price = float(
                signals[
                    symbol
                ].at[
                    date,
                    "price"
                ]
            )

            current_qty = (

                holdings[
                    symbol
                ]["qty"]

                if symbol in holdings

                else 0
            )

            target_qty_now = int(
                target_value
                //
                price
            )

            buy_quantity = max(
                0,
                target_qty_now
                -
                current_qty
            )

            if buy_quantity <= 0:
                continue

            (
                buy_quantity,
                buy_cost
            ) = maximum_affordable_quantity(
                buy_quantity,
                price,
                cash
            )

            if buy_quantity <= 0:
                continue

            trade_value = (
                buy_quantity
                *
                price
            )

            total_cash_required = (
                trade_value
                +
                buy_cost
            )

            cash -= (
                total_cash_required
            )

            if symbol not in holdings:

                holdings[
                    symbol
                ] = create_position(
                    buy_quantity,
                    price,
                    buy_cost,
                    date
                )

            else:

                holdings[
                    symbol
                ] = add_lot(
                    holdings[
                        symbol
                    ],
                    buy_quantity,
                    price,
                    buy_cost,
                    date
                )

            row = signals[
                symbol
            ].loc[date]

            transactions.append({

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "symbol":
                    symbol,

                "side":
                    "BUY",

                "qty":
                    buy_quantity,

                "price":
                    price,

                "gross_value_rs":
                    trade_value,

                "buy_cost_rs":
                    buy_cost,

                "sell_cost_rs":
                    0.0,

                "stcg_tax_rs":
                    0.0,

                "cost_basis_rs":
                    trade_value
                    +
                    buy_cost,

                "realized_gain_after_tax_rs":
                    0.0,

                "net_cash_flow_rs":
                    -(
                        trade_value
                        +
                        buy_cost
                    ),

                "reason":
                    "TOP 10 / REBALANCE",

                "rs_score":
                    float(
                        row[
                            "rs_score"
                        ]
                    ),

                "overall_rs_rank":
                    overall_rank[
                        symbol
                    ],

                "price_tt_met":
                    int(
                        row[
                            "price_tt_met"
                        ]
                    ),

                "rs_tt_met":
                    int(
                        row[
                            "rs_tt_met"
                        ]
                    ),

                "holding_qty_after":
                    holdings[
                        symbol
                    ]["qty"]
            })

        # ==============================================================
        # FINAL DAILY MARK-TO-MARKET
        # ==============================================================

        portfolio_value = (
            get_portfolio_value(
                cash,
                holdings,
                date,
                signals
            )
        )

        invested_value = (
            portfolio_value
            -
            cash
        )

        equity_rows.append({

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "portfolio_value_rs":
                portfolio_value,

            "cash_rs":
                cash,

            "invested_value_rs":
                invested_value,

            "n_holdings":
                len(holdings),

            "top10":
                ",".join(
                    top10
                ),

            "holdings":
                ",".join(
                    sorted(
                        holdings.keys()
                    )
                )
        })

    # ==================================================================
    # EQUITY CURVE
    # ==================================================================

    equity = pd.DataFrame(
        equity_rows
    )

    if equity.empty:

        raise RuntimeError(
            "No equity data generated."
        )

    equity[
        "portfolio_value_rs"
    ] = pd.to_numeric(
        equity[
            "portfolio_value_rs"
        ]
    )

    equity[
        "daily_pnl_rs"
    ] = (
        equity[
            "portfolio_value_rs"
        ]
        .diff()
        .fillna(
            equity[
                "portfolio_value_rs"
            ].iloc[0]
            -
            STARTING_CAPITAL
        )
    )

    equity[
        "daily_return_pct"
    ] = (
        equity[
            "portfolio_value_rs"
        ]
        .pct_change()
        .fillna(
            equity[
                "portfolio_value_rs"
            ].iloc[0]
            /
            STARTING_CAPITAL
            -
            1
        )
        *
        100
    )

    equity[
        "equity_multiple"
    ] = (
        equity[
            "portfolio_value_rs"
        ]
        /
        STARTING_CAPITAL
    )

    equity[
        "running_peak_rs"
    ] = (
        equity[
            "portfolio_value_rs"
        ]
        .cummax()
    )

    equity[
        "drawdown_rs"
    ] = (
        equity[
            "portfolio_value_rs"
        ]
        -
        equity[
            "running_peak_rs"
        ]
    )

    equity[
        "drawdown_pct"
    ] = (
        equity[
            "portfolio_value_rs"
        ]
        /
        equity[
            "running_peak_rs"
        ]
        -
        1
    ) * 100

    # ==================================================================
    # BACKTEST-END VALUATION
    # ==================================================================

    last_date = trading_days[-1]

    marked_equity = (
        get_portfolio_value(
            cash,
            holdings,
            last_date,
            signals
        )
    )

    liquidation_equity = (
        marked_equity
    )

    open_position_rows = []

    for symbol, position in (
        holdings.items()
    ):

        if (
            symbol not in signals
            or
            last_date
            not in
            signals[
                symbol
            ].index
        ):

            continue

        price = float(
            signals[
                symbol
            ].at[
                last_date,
                "price"
            ]
        )

        market_value = (
            position["qty"]
            *
            price
        )

        sell_cost = (
            sell_side_cost(
                market_value
            )
        )

        cost_basis = sum(

            lot["qty"]
            *
            lot["price"]
            +
            lot["buy_cost"]

            for lot
            in position["lots"]
        )

        hypothetical_gain = (
            market_value
            -
            sell_cost
            -
            cost_basis
        )

        hypothetical_tax = (
            stcg_tax(
                hypothetical_gain,
                last_date
            )
        )

        liquidation_equity -= (
            sell_cost
            +
            hypothetical_tax
        )

        open_position_rows.append({

            "date":
                last_date.strftime(
                    "%Y-%m-%d"
                ),

            "symbol":
                symbol,

            "side":
                "OPEN",

            "qty":
                position[
                    "qty"
                ],

            "price":
                price,

            "gross_value_rs":
                market_value,

            "buy_cost_rs":
                sum(
                    lot[
                        "buy_cost"
                    ]
                    for lot
                    in position[
                        "lots"
                    ]
                ),

            "sell_cost_rs":
                sell_cost,

            "stcg_tax_rs":
                hypothetical_tax,

            "cost_basis_rs":
                cost_basis,

            "realized_gain_after_tax_rs":
                np.nan,

            "net_cash_flow_rs":
                np.nan,

            "reason":
                "OPEN AT BACKTEST END",

            "rs_score":
                float(
                    signals[
                        symbol
                    ].at[
                        last_date,
                        "rs_score"
                    ]
                ),

            "overall_rs_rank":
                "",

            "price_tt_met":
                int(
                    signals[
                        symbol
                    ].at[
                        last_date,
                        "price_tt_met"
                    ]
                ),

            "rs_tt_met":
                int(
                    signals[
                        symbol
                    ].at[
                        last_date,
                        "rs_tt_met"
                    ]
                ),

            "holding_qty_after":
                position[
                    "qty"
                ]
        })

    transactions.extend(
        open_position_rows
    )

    return (
        pd.DataFrame(
            transactions
        ),
        equity,
        pd.DataFrame(
            selection_rows
        ),
        marked_equity,
        liquidation_equity
    )


# ======================================================================
# SUMMARY
# ======================================================================

def summarize(
    transactions,
    equity,
    marked_equity,
    liquidation_equity
):

    final_equity = float(
        equity[
            "portfolio_value_rs"
        ].iloc[-1]
    )

    total_return = (
        final_equity
        /
        STARTING_CAPITAL
        -
        1
    )

    max_drawdown = float(
        equity[
            "drawdown_pct"
        ].min()
    )

    daily_returns = (
        equity[
            "portfolio_value_rs"
        ]
        .pct_change()
        .dropna()
    )

    n_days = len(
        equity
    )

    annualized_return = (
        (
            final_equity
            /
            STARTING_CAPITAL
        )
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

    if len(
        daily_returns
    ) > 1:

        annualized_volatility = (
            daily_returns.std()
            *
            np.sqrt(252)
        )

        if (
            daily_returns.std()
            >
            0
        ):

            sharpe = (
                daily_returns.mean()
                /
                daily_returns.std()
                *
                np.sqrt(252)
            )

        else:

            sharpe = 0.0

        downside = (
            daily_returns[
                daily_returns < 0
            ]
        )

        if len(
            downside
        ) > 1 and (
            downside.std()
            >
            0
        ):

            sortino = (
                daily_returns.mean()
                /
                downside.std()
                *
                np.sqrt(252)
            )

        else:

            sortino = 0.0

    else:

        annualized_volatility = 0.0

        sharpe = 0.0

        sortino = 0.0

    if max_drawdown < 0:

        calmar = (
            annualized_return
            /
            abs(
                max_drawdown
                /
                100
            )
        )

    else:

        calmar = np.inf

    # ==================================================================
    # TRANSACTION TOTALS
    # ==================================================================

    if transactions.empty:

        buys = pd.DataFrame()

        sells = pd.DataFrame()

    else:

        buys = transactions[
            transactions[
                "side"
            ]
            ==
            "BUY"
        ]

        sells = transactions[
            transactions[
                "side"
            ]
            ==
            "SELL"
        ]

    def sum_col(
        df,
        column
    ):

        if (
            df.empty
            or
            column not in df.columns
        ):

            return 0.0

        return (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    buy_costs = sum_col(
        transactions,
        "buy_cost_rs"
    )

    sell_costs = sum_col(
        transactions,
        "sell_cost_rs"
    )

    taxes = sum_col(
        transactions,
        "stcg_tax_rs"
    )

    # ==================================================================
    # REALIZED SELL PERFORMANCE
    # ==================================================================

    if not sells.empty:

        realized_pnl = (
            pd.to_numeric(
                sells[
                    "realized_gain_after_tax_rs"
                ],
                errors="coerce"
            )
        )

        winning_sells = (
            realized_pnl
            >
            0
        )

        losing_sells = (
            realized_pnl
            <
            0
        )

        win_count = int(
            winning_sells.sum()
        )

        loss_count = int(
            losing_sells.sum()
        )

        total_winner_pnl = (
            realized_pnl[
                winning_sells
            ]
            .sum()
        )

        total_loser_pnl = abs(
            realized_pnl[
                losing_sells
            ]
            .sum()
        )

        if total_loser_pnl > 0:

            profit_factor = (
                total_winner_pnl
                /
                total_loser_pnl
            )

        else:

            profit_factor = np.inf

    else:

        win_count = 0

        loss_count = 0

        profit_factor = 0.0

    return {

        "strategy":
            (
                "Daily EOD Top-10 RS "
                "Equal-Weight Rebalance"
            ),

        "eligibility":
            (
                "Price > Rs.20 + "
                "20D Avg Volume > 100,000 + "
                "Price TT 7/7 + "
                "RS TT 7/7"
            ),

        "ranking":
            "Raw RS Score descending",

        "portfolio_size":
            TOP_N,

        "rebalance":
            "Daily EOD",

        "exit_rule":
            "Leaves Top 10 / daily equal-weight rebalance",

        "rs_5ema_exit":
            "REMOVED",

        "blue_dot_entry":
            "REMOVED",

        "one_year_rs_cross_entry":
            "REMOVED",

        "price_stops":
            "REMOVED",

        "rank_exit":
            "REMOVED",

        "starting_capital_rs":
            round(
                STARTING_CAPITAL,
                2
            ),

        "final_marked_equity_rs":
            round(
                marked_equity,
                2
            ),

        "final_liquidation_equity_rs":
            round(
                liquidation_equity,
                2
            ),

        "net_return_pct":
            round(
                total_return
                *
                100,
                2
            ),

        "annualized_return_pct":
            round(
                annualized_return
                *
                100,
                2
            ),

        "annualized_volatility_pct":
            round(
                annualized_volatility
                *
                100,
                2
            ),

        "sharpe":
            round(
                sharpe,
                3
            ),

        "sortino":
            round(
                sortino,
                3
            ),

        "calmar":
            (
                round(
                    calmar,
                    3
                )
                if np.isfinite(
                    calmar
                )
                else "INF"
            ),

        "max_drawdown_pct":
            round(
                max_drawdown,
                2
            ),

        "buy_transaction_costs_rs":
            round(
                buy_costs,
                2
            ),

        "sell_transaction_costs_rs":
            round(
                sell_costs,
                2
            ),

        "total_transaction_costs_rs":
            round(
                buy_costs
                +
                sell_costs,
                2
            ),

        "stcg_tax_rs":
            round(
                taxes,
                2
            ),

        "total_friction_rs":
            round(
                buy_costs
                +
                sell_costs
                +
                taxes,
                2
            ),

        "buy_transactions":
            int(
                len(buys)
            ),

        "sell_transactions":
            int(
                len(sells)
            ),

        "winning_sell_transactions":
            win_count,

        "losing_sell_transactions":
            loss_count,

        "win_rate_pct":
            round(
                (
                    win_count
                    /
                    len(sells)
                    *
                    100
                )
                if len(sells)
                else 0,
                2
            ),

        "profit_factor":
            (
                round(
                    profit_factor,
                    3
                )
                if np.isfinite(
                    profit_factor
                )
                else "INF"
            ),

        "stcg_rate_before_2024_07_23":
            "15% + 4% cess = 15.6%",

        "stcg_rate_from_2024_07_23":
            "20% + 4% cess = 20.8%",

        "data_cleaning":
            "Single-day moves > +/-30% repaired",

        "execution":
            "Same-day EOD theoretical execution"
    }


# ======================================================================
# PLOT EQUITY
# ======================================================================

def plot_equity_curve(
    equity
):

    dates = pd.to_datetime(
        equity["date"]
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        dates,
        equity[
            "portfolio_value_rs"
        ]
    )

    plt.title(
        "Daily EOD Top-10 RS Strategy - Equity Curve"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Portfolio Value (Rs.)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "RS_Top10_Equity_Curve.png",
        dpi=200
    )

    plt.close()


# ======================================================================
# PLOT DRAWDOWN
# ======================================================================

def plot_drawdown(
    equity
):

    dates = pd.to_datetime(
        equity["date"]
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        dates,
        equity[
            "drawdown_pct"
        ]
    )

    plt.title(
        "Daily EOD Top-10 RS Strategy - Drawdown"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Drawdown (%)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "RS_Top10_Drawdown.png",
        dpi=200
    )

    plt.close()


# ======================================================================
# DOWNLOAD + SIGNAL BUILD
# ======================================================================

def build_all_signals():

    tickers = load_tickers()

    benchmark = (
        download_benchmark()
    )

    download_start, download_end = (
        get_download_dates()
    )

    all_signals = {}

    total_repairs = 0

    batch_size = 50

    for start in range(
        0,
        len(tickers),
        batch_size
    ):

        batch = tickers[
            start:
            start + batch_size
        ]

        print(
            f"Downloading "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"/ {len(tickers)}"
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

        except Exception as exc:

            print(
                f"Batch failed: {exc}"
            )

            continue

        for ticker in batch:

            try:

                if len(batch) == 1:

                    stock_data = data

                else:

                    if ticker not in (
                        data.columns
                        .get_level_values(
                            0
                        )
                    ):

                        continue

                    stock_data = data[
                        ticker
                    ]

                if "Close" not in (
                    stock_data.columns
                ):

                    continue

                close = (
                    stock_data[
                        "Close"
                    ]
                    .dropna()
                    .sort_index()
                )

                volume = (
                    stock_data[
                        "Volume"
                    ]
                    .reindex(
                        close.index
                    )
                    .fillna(0)
                )

                (
                    close,
                    repairs
                ) = clean_price_series(
                    close
                )

                total_repairs += (
                    repairs
                )

                signals = (
                    compute_signals(
                        close,
                        volume,
                        benchmark
                    )
                )

                if signals is None:
                    continue

                symbol = (
                    ticker
                    .replace(
                        ".NS",
                        ""
                    )
                )

                all_signals[
                    symbol
                ] = signals

            except Exception as exc:

                print(
                    f"Skipping "
                    f"{ticker}: "
                    f"{exc}"
                )

        time.sleep(
            0.5
        )

    print(
        f"\nSignals calculated: "
        f"{len(all_signals)}"
    )

    print(
        f"Price points repaired: "
        f"{total_repairs}"
    )

    return (
        all_signals,
        benchmark
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "DAILY EOD TOP-10 RS BACKTEST"
    )

    print(
        "=" * 80
    )

    print(
        f"Period: "
        f"{BACKTEST_START} -> {BACKTEST_END}"
    )

    print(
        f"Starting capital: "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        "\nENTRY / ELIGIBILITY:"
    )

    print(
        "Price > Rs.20"
    )

    print(
        "20-day average volume > 100,000"
    )

    print(
        "Price TT = 7/7"
    )

    print(
        "RS Line TT = 7/7"
    )

    print(
        "\nRANKING:"
    )

    print(
        "Raw RS Score descending"
    )

    print(
        "Top 10"
    )

    print(
        "\nREBALANCING:"
    )

    print(
        "Daily EOD"
    )

    print(
        "Equal weight"
    )

    print(
        "\nEXITS:"
    )

    print(
        "Leave Top 10 / rebalance"
    )

    print(
        "\nREMOVED:"
    )

    print(
        "Blue Dot"
    )

    print(
        "1-year RS cross"
    )

    print(
        "RS < 5EMA"
    )

    print(
        "8% trailing stop"
    )

    print(
        "5% hard stop"
    )

    print(
        "RS rank >20 exit"
    )

    print(
        "=" * 80
    )

    # ==================================================================
    # DATA
    # ==================================================================

    (
        signals,
        benchmark
    ) = build_all_signals()

    # ==================================================================
    # TRADING DAYS
    # ==================================================================

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
                pd.Timestamp(
                    BACKTEST_END
                )
            )
        ]
    )

    if len(
        trading_days
    ) == 0:

        raise RuntimeError(
            "No trading days found."
        )

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    # ==================================================================
    # BACKTEST
    # ==================================================================

    print(
        "\nRunning backtest..."
    )

    (
        transactions,
        equity,
        daily_top10,
        marked_equity,
        liquidation_equity
    ) = run_backtest(
        signals,
        trading_days
    )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    summary = summarize(
        transactions,
        equity,
        marked_equity,
        liquidation_equity
    )

    # ==================================================================
    # SAVE
    # ==================================================================

    transactions.to_csv(
        "backtest_transactions.csv",
        index=False
    )

    equity.to_csv(
        "backtest_equity.csv",
        index=False
    )

    daily_top10.to_csv(
        "backtest_daily_top10.csv",
        index=False
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        "backtest_summary.csv",
        index=False
    )

    # ==================================================================
    # CHARTS
    # ==================================================================

    plot_equity_curve(
        equity
    )

    plot_drawdown(
        equity
    )

    # ==================================================================
    # RESULTS
    # ==================================================================

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "FINAL BACKTEST RESULTS"
    )

    print(
        "=" * 80
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 80
    )

    print(
        "\nOUTPUT FILES:"
    )

    print(
        "backtest_transactions.csv"
    )

    print(
        "backtest_equity.csv"
    )

    print(
        "backtest_daily_top10.csv"
    )

    print(
        "backtest_summary.csv"
    )

    print(
        "RS_Top10_Equity_Curve.png"
    )

    print(
        "RS_Top10_Drawdown.png"
    )


# ======================================================================
# EXECUTION
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n"
            +
            "=" * 80
        )

        print(
            "BACKTEST FAILED"
        )

        print(
            "=" * 80
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise