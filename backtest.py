"""
RS SCREENER BACKTEST
====================

EOD TOP-10 RS STRATEGY
WITH EXPLICIT 1-YEAR RS-LINE BREAKOUT ENTRY

============================================================
TRADING LOGIC
============================================================

ENTRY
-----

A stock is eligible for NEW PURCHASE only when ALL of the
following conditions are true:

1. BLUE DOT = TRUE

2. CURRENT RS LINE > PREVIOUS 1-YEAR RS-LINE HIGH

3. PREVIOUS-DAY RS LINE <= PREVIOUS-DAY 1-YEAR RS-LINE HIGH

4. PRICE TREND TEMPLATE = 7/7 PASS

5. RS LINE TREND TEMPLATE = 7/7 PASS

6. POINT-IN-TIME LIQUIDITY FILTER PASSES

Eligible stocks are then sorted by RAW RS SCORE:

    Highest RS Score
          ↓
       Rank 1
          ↓
       Rank 2
          ↓
        ...
          ↓
       Rank 10

The TOP 10 eligible stocks are the target portfolio.

============================================================
BLUE DOT
============================================================

Blue Dot remains an actual BUY requirement.

Blue Dot is calculated as:

    Current RS Line
        >
    Previous 250-trading-day maximum RS Line

The current day's value is excluded from the rolling high.

Therefore Blue Dot means that today's RS Line is above the
previous approximately one-year RS-line high.

============================================================
EXPLICIT 1-YEAR RS-LINE CROSSING
============================================================

A separate explicit crossing condition is also required.

Previous 1-year RS high:

    rolling maximum of previous 250 trading days

Entry requires:

    Today's RS Line
        >
    Previous day's 1-year RS high

AND

    Yesterday's RS Line
        <=
    Yesterday's 1-year RS high

Therefore the stock must CROSS the 1-year RS-line high
on the current EOD.

This prevents a stock that has already been above the
1-year RS high for several days from generating a new
crossing signal.

============================================================
GREEN DOT
============================================================

Green Dot is calculated and stored for analysis only.

Green Dot has ZERO effect on buying or selling.

============================================================
EXIT
============================================================

THERE IS ONLY ONE EXIT CONDITION:

    Current EOD RS Line
        <
    Current EOD 5-day EMA of RS Line

Therefore:

    EXIT = RS Line < RS Line 5EMA

Removed completely:

    - 8% trailing stop
    - 5% hard stop
    - RS rank >20 exit

A position remains open regardless of price drawdown,
provided the RS-line exit condition has not occurred.

============================================================
PORTFOLIO ENTRY RANKING
============================================================

For new purchases:

    Blue Dot
    +
    1-year RS-line CROSS
    +
    Price TT 7/7
    +
    RS Line TT 7/7
    +
    Liquidity
            ↓
    sort by RAW RS SCORE
            ↓
    TOP 10
            ↓
    BUY

============================================================
POSITION SIZING
============================================================

Equal-weight target:

    Portfolio Value / 10

Integer shares are purchased.

Actual cash is tracked.

============================================================
REBALANCING
============================================================

Daily EOD.

This is a DAILY EOD theoretical execution model.

It assumes today's EOD signal can be executed at today's
EOD closing price.

It is NOT a genuine 3:00-3:30 PM intraday backtest.

============================================================
RS-LINE EXIT
============================================================

The RS-line exit uses:

    RS Line < 5-day EMA of RS Line

Both are calculated from EOD prices.

No intraday information is used.

============================================================
COSTS
============================================================

Modeled:

- STT
- Stamp duty
- NSE exchange transaction charge
- SEBI charge
- GST on exchange + SEBI charges
- DP charge on selling

Brokerage assumed zero for delivery.

============================================================
STCG
============================================================

20% STCG + 4% cess = 20.8%.

Tax is applied to positive realized gains only.

Loss set-off/carry-forward is NOT modeled.

This is a simplified conservative tax treatment.

============================================================
DATA CLEANING
============================================================

Single-day price changes exceeding +/-30% are treated as
potential split/bonus/merger/data corruption and repaired
by carrying forward the previous valid close.

This is deliberately conservative.

============================================================
SURVIVORSHIP BIAS
============================================================

The stock universe still comes from stocks.csv.

Therefore this remains exposed to survivorship bias if stocks.csv
contains today's universe projected backwards.

============================================================
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
# CONFIGURATION
# ============================================================

BENCHMARK = "^CRSLDX"

BENCHMARK_FALLBACK = "^NSEI"


# ============================================================
# RS CONFIGURATION
# ============================================================

# Approximately one trading year.
RS_ONE_YEAR_LOOKBACK = 250


# ============================================================
# DATA DOWNLOAD
# ============================================================

DOWNLOAD_YEARS_BEFORE_START = 3


# ============================================================
# STOCK UNIVERSE
# ============================================================

STOCKS_FILE = "stocks.csv"


# ============================================================
# PORTFOLIO
# ============================================================

TOP_N = 10


# ============================================================
# POINT-IN-TIME LIQUIDITY
# ============================================================

MIN_PRICE = 10

MIN_AVG_VOLUME = 10000

VOLUME_LOOKBACK = 20


# ============================================================
# DATA SANITY CLEANING
# ============================================================

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ============================================================
# BACKTEST DATES
# ============================================================

BACKTEST_START = "2016-04-01"

BACKTEST_END = "2026-08-07"


# ============================================================
# CAPITAL
# ============================================================

STARTING_CAPITAL = 1_000_000


# ============================================================
# EXIT
# ============================================================

# ONLY EXIT CONDITION:
#
# RS Line < 5-day EMA of RS Line

RS_EMA_SPAN = 5


# ============================================================
# TRANSACTION COSTS
# ============================================================

ENABLE_COSTS = True


# Delivery STT
STT_BUY_RATE = 0.001

STT_SELL_RATE = 0.001


# Stamp duty on BUY
STAMP_DUTY_RATE = 0.00015


# NSE exchange transaction charge
EXCHANGE_CHARGE_RATE = 0.0000325


# SEBI charge
SEBI_CHARGE_RATE = 0.000001


# GST on exchange + SEBI
GST_RATE = 0.18


# DP charge per SELL
DP_CHARGE_FLAT = 20


# ============================================================
# STCG
# ============================================================

ENABLE_STCG = True

STCG_RATE = 0.20

STCG_CESS = 0.04

STCG_EFFECTIVE_RATE = (
    STCG_RATE
    *
    (1 + STCG_CESS)
)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"

SUMMARY_WORKSHEET = "Backtest_Summary"


# ============================================================
# COST CALCULATIONS
# ============================================================

def buy_side_cost(trade_value):
    """
    Total modeled BUY transaction cost.
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
    Total modeled SELL transaction cost.
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


def stcg_tax(net_gain):
    """
    Simplified STCG.

    Tax is applied only to positive realized gains.
    """

    if not ENABLE_STCG:
        return 0.0

    if net_gain <= 0:
        return 0.0

    return (
        net_gain
        *
        STCG_EFFECTIVE_RATE
    )


# ============================================================
# DATE RANGE
# ============================================================

def get_download_dates():

    backtest_start = pd.Timestamp(
        BACKTEST_START
    )

    backtest_end = pd.Timestamp(
        BACKTEST_END
    )

    download_start = (
        backtest_start
        -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    download_end = (
        backtest_end
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
# LOAD TICKERS
# ============================================================

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
# DATA CLEANING
# ============================================================

def clean_price_series(close):
    """
    Repair implausible single-day moves.

    The repair is performed BEFORE:

        RS calculation
        Trend Template
        Blue Dot
        Green Dot
        RS EMA
        1-year RS high
        backtest
    """

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

        return (
            close,
            0
        )

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = (
            cleaned.index
            .get_loc(idx)
        )

        if pos > 0:

            cleaned.iloc[pos] = (
                cleaned.iloc[
                    pos - 1
                ]
            )

    return (
        cleaned,
        n_bad
    )


# ============================================================
# BENCHMARK
# ============================================================

def download_benchmark():

    (
        download_start,
        download_end
    ) = get_download_dates()

    print(
        f"\nBenchmark download: "
        f"{download_start} "
        f"to "
        f"{download_end}"
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

            (
                close,
                n_bad
            ) = clean_price_series(
                close
            )

            if n_bad:

                print(
                    f"Benchmark {ticker}: "
                    f"repaired "
                    f"{n_bad} "
                    f"point(s)"
                )

            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} "
                f"failed: {e}"
            )

    raise RuntimeError(
        "Could not download "
        "benchmark data."
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

    # Condition 1
    c1 = (
        (s > sma150)
        &
        (s > sma200)
    )

    # Condition 2
    c2 = (
        sma150 > sma200
    )

    # Condition 3
    c3 = (
        sma200 > sma200_1mo
    )

    # Condition 4
    c4 = (
        (sma50 > sma150)
        &
        (sma50 > sma200)
    )

    # Condition 5
    c5 = (
        s > sma50
    )

    # Condition 6
    c6 = (
        s >= 1.25 * low52
    )

    # Condition 7
    c7 = (
        s >= 0.75 * high52
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


# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
    bench_close
):
    """
    Calculate all variables needed by the strategy.

    ENTRY:

        Blue Dot
        +
        explicit 1-year RS-line crossing
        +
        Price TT 7/7
        +
        RS TT 7/7
        +
        liquidity

    EXIT:

        RS Line < 5-day EMA

    Green Dot is diagnostic only.
    """

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

    # Need sufficient history for:
    #
    # 252-day Price TT
    # 250-day RS breakout
    # 252-day RS score
    # plus previous-day comparison.

    if len(aligned) < 280:

        return None

    volume = (
        volume
        .reindex(
            aligned.index
        )
    )

    # ========================================================
    # RS LINE
    # ========================================================

    rs_line = (
        aligned["s"]
        /
        aligned["b"]
    )

    # ========================================================
    # RAW RS SCORE
    # ========================================================

    def pct_return(
        series,
        days
    ):

        return (
            series
            /
            series.shift(days)
            -
            1
        )

    rs_score = (
        0.40
        *
        pct_return(
            aligned["s"],
            63
        )
        +
        0.20
        *
        pct_return(
            aligned["s"],
            126
        )
        +
        0.20
        *
        pct_return(
            aligned["s"],
            189
        )
        +
        0.20
        *
        pct_return(
            aligned["s"],
            252
        )
    ) * 100

    # ========================================================
    # PREVIOUS 1-YEAR RS-LINE HIGH
    # ========================================================
    #
    # CRITICAL:
    #
    # shift(1) happens BEFORE rolling().
    #
    # Therefore today's RS value can NEVER be part of
    # today's breakout threshold.
    #
    # At date T:
    #
    # rs_1y_high[T]
    #
    # = maximum RS Line from T-250 through T-1.
    #
    # This prevents look-ahead bias.
    # ========================================================

    rs_1y_high = (
        rs_line
        .shift(1)
        .rolling(
            RS_ONE_YEAR_LOOKBACK
        )
        .max()
    )

    # ========================================================
    # PREVIOUS DAY'S 1-YEAR HIGH
    # ========================================================
    #
    # This is the threshold that existed yesterday.
    #
    # At date T:
    #
    # yesterday_rs_1y_high =
    #     rs_1y_high[T-1]
    #
    # ========================================================

    previous_rs_1y_high = (
        rs_1y_high.shift(1)
    )

    # ========================================================
    # EXPLICIT 1-YEAR RS CROSSING
    # ========================================================
    #
    # TODAY:
    #
    # RS[T] > RS_1Y_HIGH[T]
    #
    # YESTERDAY:
    #
    # RS[T-1] <= RS_1Y_HIGH[T-1]
    #
    # This is a genuine crossing event.
    # ========================================================

    rs_cross_1y = (
        (
            rs_line
            >
            rs_1y_high
        )
        &
        (
            rs_line.shift(1)
            <=
            previous_rs_1y_high
        )
    )

    # ========================================================
    # BLUE DOT
    # ========================================================
    #
    # Blue Dot remains unchanged.
    #
    # It is also based on the previous 250 trading days.
    #
    # Because the current day's value is excluded, this is
    # equivalent to:
    #
    #     today's RS > previous 250-day RS maximum
    #
    # ========================================================

    blue_dot = (
        rs_line
        >
        rs_1y_high
    )

    # ========================================================
    # GREEN DOT
    # ========================================================
    #
    # Diagnostic only.
    #
    # Does NOT affect trading.
    # ========================================================

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

    # ========================================================
    # PRICE TREND TEMPLATE
    # ========================================================

    (
        tt_pass,
        tt_met
    ) = trend_template_series(
        aligned["s"]
    )

    # ========================================================
    # RS LINE TREND TEMPLATE
    # ========================================================

    (
        rs_tt_pass,
        rs_tt_met
    ) = trend_template_series(
        rs_line
    )

    # ========================================================
    # RS LINE 5-DAY EMA
    # ========================================================

    rs_ema5 = (
        rs_line
        .ewm(
            span=RS_EMA_SPAN,
            adjust=False
        )
        .mean()
    )

    # ========================================================
    # ONLY EXIT CONDITION
    # ========================================================

    rs_below_ema5 = (
        rs_line
        <
        rs_ema5
    )

    # ========================================================
    # POINT-IN-TIME LIQUIDITY
    # ========================================================

    rolling_avg_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (
        (aligned["s"] >= MIN_PRICE)
        &
        (
            rolling_avg_volume
            >=
            MIN_AVG_VOLUME
        )
    )

    # ========================================================
    # PRICE 50 DMA DIAGNOSTIC
    # ========================================================

    sma50 = (
        aligned["s"]
        .rolling(50)
        .mean()
    )

    above_50dma = (
        aligned["s"]
        >
        sma50
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    return pd.DataFrame({

        "price":
            aligned["s"],

        "rs_line":
            rs_line,

        "rs_score":
            rs_score,

        # 1-year RS breakout fields
        "rs_1y_high":
            rs_1y_high,

        "previous_rs_1y_high":
            previous_rs_1y_high,

        "rs_cross_1y":
            rs_cross_1y,

        # Entry
        "blue_dot":
            blue_dot,

        # Diagnostic
        "green_dot":
            green_dot,

        # Price TT
        "tt_pass":
            tt_pass,

        "tt_met":
            tt_met,

        # RS TT
        "rs_tt_pass":
            rs_tt_pass,

        "rs_tt_met":
            rs_tt_met,

        # ONLY EXIT
        "rs_ema5":
            rs_ema5,

        "rs_below_ema5":
            rs_below_ema5,

        # Liquidity
        "liquid":
            liquid,

        "avg_volume":
            rolling_avg_volume,

        # Diagnostic
        "above_50dma":
            above_50dma,
    })


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    """
    DAILY EOD PORTFOLIO SIMULATION.

    ENTRY:

        Blue Dot
        +
        1-year RS-line CROSS
        +
        Price TT 7/7
        +
        RS Line TT 7/7
        +
        Liquidity
        +
        Top 10 by RS Score

    EXIT:

        ONLY:

            RS Line < 5-day EMA

    NO PRICE STOP.

    NO 8% TRAILING STOP.

    NO RS RANK EXIT.
    """

    cash = (
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    daily_selection_log = []

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in trading_days:

        # ====================================================
        # 1. BUILD COMPLETE RS RANKING
        #
        # This ranking is STILL calculated for analysis and
        # entry ranking.
        #
        # It is NO LONGER an exit condition.
        # ====================================================

        rs_rank_pool = []

        for sym, df in all_signals.items():

            if date not in df.index:
                continue

            row = df.loc[date]

            if pd.isna(
                row["rs_score"]
            ):
                continue

            rs_rank_pool.append(
                (
                    sym,
                    float(
                        row["rs_score"]
                    )
                )
            )

        rs_rank_pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        rs_rank_lookup = {
            sym: i + 1
            for i, (
                sym,
                _
            ) in enumerate(
                rs_rank_pool
            )
        }

        # ====================================================
        # 2. BUILD BUY-ELIGIBLE UNIVERSE
        # ====================================================

        buy_pool = []

        for sym, df in all_signals.items():

            if date not in df.index:
                continue

            row = df.loc[date]

            if pd.isna(
                row["rs_score"]
            ):
                continue

            # ------------------------------------------------
            # LIQUIDITY
            # ------------------------------------------------

            if not bool(
                row["liquid"]
            ):
                continue

            # ------------------------------------------------
            # PRICE TREND TEMPLATE
            # ------------------------------------------------

            if not bool(
                row["tt_pass"]
            ):
                continue

            # ------------------------------------------------
            # RS LINE TREND TEMPLATE
            # ------------------------------------------------

            if not bool(
                row["rs_tt_pass"]
            ):
                continue

            # ------------------------------------------------
            # BLUE DOT
            # ------------------------------------------------

            if not bool(
                row["blue_dot"]
            ):
                continue

            # ------------------------------------------------
            # EXPLICIT 1-YEAR RS CROSSING
            # ------------------------------------------------

            if not bool(
                row["rs_cross_1y"]
            ):
                continue

            # ------------------------------------------------
            # ADD TO BUY POOL
            # ------------------------------------------------

            buy_pool.append(
                (
                    sym,
                    float(
                        row["rs_score"]
                    )
                )
            )

        # ====================================================
        # 3. SORT BUY CANDIDATES BY RAW RS SCORE
        # ====================================================

        buy_pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ====================================================
        # 4. TOP 10 BUY TARGETS
        # ====================================================

        top10 = [
            sym
            for sym, _
            in buy_pool[:TOP_N]
        ]

        # ====================================================
        # 5. DAILY TOP-10 AUDIT LOG
        # ====================================================

        for rank, (
            sym,
            score
        ) in enumerate(
            buy_pool[:TOP_N],
            start=1
        ):

            row = all_signals[
                sym
            ].loc[date]

            daily_selection_log.append({

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "entry_rank":
                    rank,

                "symbol":
                    sym,

                "rs_score":
                    round(
                        score,
                        4
                    ),

                "overall_rs_rank":
                    rs_rank_lookup.get(
                        sym,
                        999999
                    ),

                "price":
                    round(
                        float(
                            row["price"]
                        ),
                        2
                    ),

                "blue_dot":
                    bool(
                        row["blue_dot"]
                    ),

                "rs_cross_1y":
                    bool(
                        row["rs_cross_1y"]
                    ),

                "rs_line":
                    round(
                        float(
                            row["rs_line"]
                        ),
                        8
                    ),

                "rs_1y_high":
                    round(
                        float(
                            row["rs_1y_high"]
                        ),
                        8
                    ),

                "previous_rs_1y_high":
                    round(
                        float(
                            row[
                                "previous_rs_1y_high"
                            ]
                        ),
                        8
                    ),

                "rs_ema5":
                    round(
                        float(
                            row["rs_ema5"]
                        ),
                        8
                    ),

                "green_dot":
                    bool(
                        row["green_dot"]
                    ),

                "tt_met":
                    int(
                        row["tt_met"]
                    ),

                "rs_tt_met":
                    int(
                        row["rs_tt_met"]
                    ),

                "avg_volume":
                    round(
                        float(
                            row["avg_volume"]
                        ),
                        0
                    ),
            })

        # ====================================================
        # 6. CHECK ONLY EXIT CONDITION
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            df = all_signals[sym]

            if date not in df.index:
                continue

            row = df.loc[date]

            current_price = float(
                row["price"]
            )

            # ------------------------------------------------
            # ONLY EXIT:
            #
            # RS LINE CLOSE < 5 EMA
            # ------------------------------------------------

            exit_rs_ema = bool(
                row[
                    "rs_below_ema5"
                ]
            )

            if not exit_rs_ema:

                continue

            # ------------------------------------------------
            # EXIT EXECUTION
            # ------------------------------------------------

            pos = holdings.pop(
                sym
            )

            exit_price = (
                current_price
            )

            gross_proceeds = (
                pos["qty"]
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
                pos["qty"]
                *
                pos["entry_price"]
                +
                pos["entry_cost"]
            )

            net_gain = (
                net_proceeds
                -
                cost_basis
            )

            tax = stcg_tax(
                net_gain
            )

            net_proceeds_after_tax = (
                net_proceeds
                -
                tax
            )

            cash += (
                net_proceeds_after_tax
            )

            net_pnl = (
                net_gain
                -
                tax
            )

            gross_return_pct = (
                exit_price
                /
                pos["entry_price"]
                -
                1
            ) * 100

            net_return_pct = (
                net_pnl
                /
                cost_basis
                *
                100
                if cost_basis > 0
                else 0
            )

            current_rs_rank = (
                rs_rank_lookup.get(
                    sym,
                    999999
                )
            )

            trade_log.append({

                "symbol":
                    sym,

                "entry_date":
                    pos[
                        "entry_date"
                    ].strftime(
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
                        pos[
                            "entry_price"
                        ],
                        2
                    ),

                "exit_price":
                    round(
                        exit_price,
                        2
                    ),

                "rs_score":
                    round(
                        float(
                            row[
                                "rs_score"
                            ]
                        ),
                        4
                    ),

                "rs_line":
                    round(
                        float(
                            row[
                                "rs_line"
                            ]
                        ),
                        8
                    ),

                "rs_ema5":
                    round(
                        float(
                            row[
                                "rs_ema5"
                            ]
                        ),
                        8
                    ),

                "current_rs_rank":
                    current_rs_rank,

                "gross_return_pct":
                    round(
                        gross_return_pct,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        pos[
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
                        net_pnl,
                        2
                    ),

                "net_return_pct":
                    round(
                        net_return_pct,
                        2
                    ),

                "days_held":
                    (
                        date
                        -
                        pos[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "RS < 5EMA",

                "entry_blue_dot":
                    pos.get(
                        "entry_blue_dot",
                        False
                    ),

                "entry_rs_cross_1y":
                    pos.get(
                        "entry_rs_cross_1y",
                        False
                    ),

                "entry_green_dot":
                    pos.get(
                        "entry_green_dot",
                        False
                    ),

                "exit_blue_dot":
                    bool(
                        row[
                            "blue_dot"
                        ]
                    ),

                "exit_rs_cross_1y":
                    bool(
                        row[
                            "rs_cross_1y"
                        ]
                    ),

                "exit_green_dot":
                    bool(
                        row[
                            "green_dot"
                        ]
                    ),
            })

        # ====================================================
        # 7. PORTFOLIO VALUE AFTER EXITS
        # ====================================================

        portfolio_value = cash

        for sym, pos in holdings.items():

            df = all_signals[sym]

            if date in df.index:

                mark_price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                mark_price = float(
                    pos[
                        "entry_price"
                    ]
                )

            portfolio_value += (
                pos["qty"]
                *
                mark_price
            )

        # ====================================================
        # 8. BUY NEW TOP-10 STOCKS
        # ====================================================

        slots_open = (
            TOP_N
            -
            len(holdings)
        )

        if slots_open > 0:

            target_per_position = (
                portfolio_value
                /
                TOP_N
            )

            for sym in top10:

                if slots_open <= 0:
                    break

                if sym in holdings:
                    continue

                df = all_signals[sym]

                if date not in df.index:
                    continue

                row = df.loc[date]

                price = float(
                    row["price"]
                )

                if price <= 0:
                    continue

                qty = int(
                    target_per_position
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

                if total_required > cash:

                    continue

                cash -= (
                    total_required
                )

                holdings[sym] = {

                    "qty":
                        qty,

                    "entry_price":
                        price,

                    "entry_date":
                        date,

                    "entry_cost":
                        buy_cost,

                    # Entry diagnostics
                    "entry_blue_dot":
                        bool(
                            row[
                                "blue_dot"
                            ]
                        ),

                    "entry_rs_cross_1y":
                        bool(
                            row[
                                "rs_cross_1y"
                            ]
                        ),

                    "entry_green_dot":
                        bool(
                            row[
                                "green_dot"
                            ]
                        ),
                }

                slots_open -= 1

        # ====================================================
        # 9. FINAL DAILY MARK-TO-MARKET
        # ====================================================

        portfolio_value = cash

        invested_value = 0.0

        for sym, pos in holdings.items():

            df = all_signals[sym]

            if date in df.index:

                mark_price = float(
                    df.loc[
                        date,
                        "price"
                    ]
                )

            else:

                mark_price = float(
                    pos[
                        "entry_price"
                    ]
                )

            position_value = (
                pos["qty"]
                *
                mark_price
            )

            invested_value += (
                position_value
            )

            portfolio_value += (
                position_value
            )

        # ====================================================
        # 10. DAILY EQUITY CURVE
        # ====================================================

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

            "daily_pnl_rs":
                "",
            
            "daily_return_pct":
                "",

            "running_peak_rs":
                "",

            "drawdown_rs":
                "",

            "drawdown_pct":
                "",

            "cash_rs":
                round(
                    cash,
                    2
                ),

            "invested_value_rs":
                round(
                    invested_value,
                    2
                ),

            "n_holdings":
                len(holdings),

            "top10_buy_candidates":
                ",".join(
                    top10
                ),

            "holdings":
                ",".join(
                    sorted(
                        holdings.keys()
                    )
                ),
        })

    # ========================================================
    # 11. COMPLETE DAILY EQUITY CALCULATION
    # ========================================================
    #
    # Calculated AFTER the entire daily portfolio series has
    # been generated.
    #
    # This avoids mixing current-day and previous-day values.
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    if not equity_df.empty:

        equity_df[
            "portfolio_value_rs"
        ] = pd.to_numeric(
            equity_df[
                "portfolio_value_rs"
            ]
        )

        equity_df[
            "daily_pnl_rs"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ]
            .diff()
            .fillna(0)
        )

        equity_df[
            "daily_return_pct"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ]
            .pct_change()
            .fillna(0)
            *
            100
        )

        equity_df[
            "running_peak_rs"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ]
            .cummax()
        )

        equity_df[
            "drawdown_rs"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ]
            -
            equity_df[
                "running_peak_rs"
            ]
        )

        equity_df[
            "drawdown_pct"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ]
            /
            equity_df[
                "running_peak_rs"
            ]
            -
            1
        ) * 100

        equity_df[
            "portfolio_value_rs"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ].round(2)
        )

        equity_df[
            "daily_pnl_rs"
        ] = (
            equity_df[
                "daily_pnl_rs"
            ].round(2)
        )

        equity_df[
            "daily_return_pct"
        ] = (
            equity_df[
                "daily_return_pct"
            ].round(4)
        )

        equity_df[
            "running_peak_rs"
        ] = (
            equity_df[
                "running_peak_rs"
            ].round(2)
        )

        equity_df[
            "drawdown_rs"
        ] = (
            equity_df[
                "drawdown_rs"
            ].round(2)
        )

        equity_df[
            "drawdown_pct"
        ] = (
            equity_df[
                "drawdown_pct"
            ].round(4)
        )

    # ========================================================
    # 12. MARK OPEN POSITIONS AT BACKTEST END
    # ========================================================

    if len(trading_days):

        last_date = (
            trading_days[-1]
        )

        for sym, pos in list(
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

                row = df.loc[
                    last_date
                ]

            else:

                exit_price = float(
                    pos[
                        "entry_price"
                    ]
                )

                row = None

            gross_proceeds = (
                pos["qty"]
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
                pos["qty"]
                *
                pos["entry_price"]
                +
                pos["entry_cost"]
            )

            net_gain = (
                net_proceeds
                -
                cost_basis
            )

            tax = stcg_tax(
                net_gain
            )

            net_pnl = (
                net_gain
                -
                tax
            )

            gross_return_pct = (
                exit_price
                /
                pos["entry_price"]
                -
                1
            ) * 100

            net_return_pct = (
                net_pnl
                /
                cost_basis
                *
                100
                if cost_basis > 0
                else 0
            )

            if row is not None:

                final_rs_rank = ""

                final_rs_score = (
                    round(
                        float(
                            row[
                                "rs_score"
                            ]
                        ),
                        4
                    )
                    if pd.notna(
                        row[
                            "rs_score"
                        ]
                    )
                    else ""
                )

                final_rs_line = (
                    round(
                        float(
                            row[
                                "rs_line"
                            ]
                        ),
                        8
                    )
                    if pd.notna(
                        row[
                            "rs_line"
                        ]
                    )
                    else ""
                )

                final_rs_ema5 = (
                    round(
                        float(
                            row[
                                "rs_ema5"
                            ]
                        ),
                        8
                    )
                    if pd.notna(
                        row[
                            "rs_ema5"
                        ]
                    )
                    else ""
                )

                final_blue_dot = bool(
                    row[
                        "blue_dot"
                    ]
                )

                final_rs_cross_1y = bool(
                    row[
                        "rs_cross_1y"
                    ]
                )

                final_green_dot = bool(
                    row[
                        "green_dot"
                    ]
                )

            else:

                final_rs_rank = ""

                final_rs_score = ""

                final_rs_line = ""

                final_rs_ema5 = ""

                final_blue_dot = False

                final_rs_cross_1y = False

                final_green_dot = False

            trade_log.append({

                "symbol":
                    sym,

                "entry_date":
                    pos[
                        "entry_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "exit_date":
                    last_date.strftime(
                        "%Y-%m-%d"
                    )
                    +
                    " (OPEN)",

                "qty":
                    pos["qty"],

                "entry_price":
                    round(
                        pos[
                            "entry_price"
                        ],
                        2
                    ),

                "exit_price":
                    round(
                        exit_price,
                        2
                    ),

                "rs_score":
                    final_rs_score,

                "rs_line":
                    final_rs_line,

                "rs_ema5":
                    final_rs_ema5,

                "current_rs_rank":
                    final_rs_rank,

                "gross_return_pct":
                    round(
                        gross_return_pct,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        pos[
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
                        net_pnl,
                        2
                    ),

                "net_return_pct":
                    round(
                        net_return_pct,
                        2
                    ),

                "days_held":
                    (
                        last_date
                        -
                        pos[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "BACKTEST END",

                "entry_blue_dot":
                    pos.get(
                        "entry_blue_dot",
                        False
                    ),

                "entry_rs_cross_1y":
                    pos.get(
                        "entry_rs_cross_1y",
                        False
                    ),

                "entry_green_dot":
                    pos.get(
                        "entry_green_dot",
                        False
                    ),

                "exit_blue_dot":
                    final_blue_dot,

                "exit_rs_cross_1y":
                    final_rs_cross_1y,

                "exit_green_dot":
                    final_green_dot,
            })

    return (
        pd.DataFrame(
            trade_log
        ),
        equity_df,
        pd.DataFrame(
            daily_selection_log
        )
    )


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def summarize(
    trade_df,
    equity_df
):

    if equity_df.empty:

        return {}

    # ========================================================
    # FINAL PORTFOLIO VALUE
    # ========================================================

    final_value = float(
        equity_df[
            "portfolio_value_rs"
        ].iloc[-1]
    )

    net_total_return_pct = (
        final_value
        /
        STARTING_CAPITAL
        -
        1
    ) * 100

    # ========================================================
    # DRAWDOWN
    # ========================================================

    running_max = (
        equity_df[
            "equity"
        ].cummax()
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

    # ========================================================
    # CLOSED TRADES
    # ========================================================

    if not trade_df.empty:

        closed = trade_df[
            ~trade_df[
                "exit_date"
            ]
            .astype(str)
            .str.contains(
                "OPEN",
                na=False
            )
        ]

    else:

        closed = trade_df

    n = len(
        closed
    )

    if n:

        win_rate_gross = (
            (
                closed[
                    "gross_return_pct"
                ]
                > 0
            ).mean()
            *
            100
        )

        win_rate_net = (
            (
                closed[
                    "net_return_pct"
                ]
                > 0
            ).mean()
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

        median_days = (
            closed[
                "days_held"
            ].median()
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

        total_buy_costs = (
            closed[
                "buy_cost_rs"
            ].sum()
        )

        total_sell_costs = (
            closed[
                "sell_cost_rs"
            ].sum()
        )

        total_costs = (
            total_buy_costs
            +
            total_sell_costs
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

        total_winning_pnl = (
            winners[
                "net_pnl_rs"
            ].sum()
            if len(winners)
            else 0
        )

        total_losing_pnl = abs(
            losers[
                "net_pnl_rs"
            ].sum()
        ) if len(losers) else 0

        if total_losing_pnl > 0:

            profit_factor = (
                total_winning_pnl
                /
                total_losing_pnl
            )

        else:

            profit_factor = 0

        exit_reason_counts = (
            closed[
                "exit_reason"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        win_rate_gross = 0

        win_rate_net = 0

        avg_gross = 0

        avg_net = 0

        median_net = 0

        avg_days = 0

        median_days = 0

        best_gross = 0

        worst_gross = 0

        total_buy_costs = 0

        total_sell_costs = 0

        total_costs = 0

        total_tax = 0

        avg_winner = 0

        avg_loser = 0

        profit_factor = 0

        exit_reason_counts = {}

    # ========================================================
    # DAILY RETURN METRICS
    # ========================================================

    daily_returns = (
        equity_df[
            "equity"
        ]
        .pct_change()
        .dropna()
    )

    if len(
        daily_returns
    ):

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

        if daily_std > 0:

            sharpe = (
                daily_mean
                /
                daily_std
                *
                np.sqrt(252)
            )

        else:

            sharpe = 0

        downside = (
            daily_returns[
                daily_returns < 0
            ]
        )

        if len(
            downside
        ):

            downside_std = (
                downside.std()
            )

        else:

            downside_std = 0

        if downside_std > 0:

            sortino = (
                daily_mean
                /
                downside_std
                *
                np.sqrt(252)
            )

        else:

            sortino = 0

    else:

        annualized_return = 0

        annualized_vol = 0

        sharpe = 0

        sortino = 0

    # ========================================================
    # CALMAR
    # ========================================================

    if abs(
        max_dd
    ) > 0:

        calmar = (
            annualized_return
            /
            abs(
                max_dd / 100
            )
        )

    else:

        calmar = 0

    # ========================================================
    # SUMMARY
    # ========================================================

    return {

        "strategy":
            (
                "Blue Dot + "
                "1-Year RS-Line CROSS + "
                "Price TT 7/7 + "
                "RS Line TT 7/7 -> "
                "Top 10 RS Score"
            ),

        "entry_rule":
            (
                "RS today > previous 250-day RS high "
                "AND "
                "RS yesterday <= yesterday's previous "
                "250-day RS high"
            ),

        "exit_rule":
            "ONLY: RS Line < 5-day EMA",

        "removed_exits":
            (
                "8% trailing stop; "
                "5% hard stop; "
                "RS rank >20"
            ),

        "starting_capital_rs":
            round(
                STARTING_CAPITAL,
                0
            ),

        "final_portfolio_value_rs":
            round(
                final_value,
                0
            ),

        "net_total_return_pct":
            round(
                net_total_return_pct,
                2
            ),

        "annualized_return_pct":
            round(
                annualized_return * 100,
                2
            ),

        "annualized_volatility_pct":
            round(
                annualized_vol * 100,
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
            round(
                calmar,
                3
            ),

        "max_dd_pct":
            round(
                max_dd,
                2
            ),

        "n_closed_trades":
            int(n),

        "win_rate_gross_pct":
            round(
                win_rate_gross,
                1
            ),

        "win_rate_net_pct":
            round(
                win_rate_net,
                1
            ),

        "avg_gross_return_per_trade_pct":
            round(
                avg_gross,
                2
            ),

        "avg_net_return_per_trade_pct":
            round(
                avg_net,
                2
            ),

        "median_net_return_per_trade_pct":
            round(
                median_net,
                2
            ),

        "avg_days_held":
            round(
                avg_days,
                1
            ),

        "median_days_held":
            round(
                median_days,
                1
            ),

        "avg_winner_net_pct":
            round(
                avg_winner,
                2
            ),

        "avg_loser_net_pct":
            round(
                avg_loser,
                2
            ),

        "profit_factor_net":
            round(
                profit_factor,
                3
            ),

        "best_gross_trade_pct":
            round(
                best_gross,
                2
            ),

        "worst_gross_trade_pct":
            round(
                worst_gross,
                2
            ),

        "total_buy_costs_rs":
            round(
                total_buy_costs,
                0
            ),

        "total_sell_costs_rs":
            round(
                total_sell_costs,
                0
            ),

        "total_transaction_costs_rs":
            round(
                total_costs,
                0
            ),

        "total_stcg_tax_rs":
            round(
                total_tax,
                0
            ),

        "exit_reason_counts":
            str(
                exit_reason_counts
            ),
    }


# ============================================================
# GOOGLE SHEETS / CSV OUTPUT
# ============================================================

def write_to_sheet(
    trade_df,
    equity_df,
    selection_df,
    summary
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    # ========================================================
    # CSV FALLBACK
    # ========================================================

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

        selection_df.to_csv(
            "backtest_daily_top10.csv",
            index=False
        )

        pd.DataFrame(
            [summary]
        ).to_csv(
            "backtest_summary.csv",
            index=False
        )

        return

    # ========================================================
    # GOOGLE AUTHENTICATION
    # ========================================================

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

    # ========================================================
    # SUMMARY WORKSHEET
    # ========================================================

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_rows = (
        len(summary_df)
        +
        10
    )

    summary_cols = (
        len(summary_df.columns)
        +
        2
    )

    try:

        sws = sh.worksheet(
            SUMMARY_WORKSHEET
        )

        if (
            sws.row_count
            <
            summary_rows
            or
            sws.col_count
            <
            summary_cols
        ):

            sws.resize(
                rows=max(
                    sws.row_count,
                    summary_rows
                ),
                cols=max(
                    sws.col_count,
                    summary_cols
                )
            )

    except gspread.WorksheetNotFound:

        sws = sh.add_worksheet(
            title=SUMMARY_WORKSHEET,
            rows=summary_rows,
            cols=summary_cols
        )

    sws.clear()

    sws.update(
        [[
            "EOD TOP-10 RS STRATEGY | "
            f"Run: {timestamp} | "
            f"Starting Capital: "
            f"Rs.{STARTING_CAPITAL:,.0f} | "
            "Net of modeled costs + STCG"
        ]],
        "A1"
    )

    sws.update(
        [
            list(
                summary_df.columns
            )
        ]
        +
        summary_df.values.tolist(),
        "A3"
    )

    print(
        f"\nSummary written to "
        f"'{SUMMARY_WORKSHEET}'"
    )

    # ========================================================
    # MAIN BACKTEST WORKSHEET
    # ========================================================

    max_cols = max(

        len(trade_df.columns)
        if not trade_df.empty
        else 0,

        len(equity_df.columns)
        if not equity_df.empty
        else 0,

        len(selection_df.columns)
        if not selection_df.empty
        else 0

    ) + 2

    max_rows = (
        len(trade_df)
        +
        len(equity_df)
        +
        len(selection_df)
        +
        100
    )

    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )

        if (
            ws.row_count
            <
            max_rows
            or
            ws.col_count
            <
            max_cols
        ):

            ws.resize(
                rows=max(
                    ws.row_count,
                    max_rows
                ),
                cols=max(
                    ws.col_count,
                    max_cols
                )
            )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(
            title=BACKTEST_WORKSHEET,
            rows=max_rows,
            cols=max_cols
        )

    ws.clear()

    # ========================================================
    # HEADER
    # ========================================================

    ws.update(
        [[
            "EOD TOP-10 RS STRATEGY | "
            "ENTRY: Blue Dot + 1-Year RS CROSS + "
            "Price TT 7/7 + RS TT 7/7 | "
            "EXIT ONLY: RS < 5EMA | "
            "Green Dot diagnostic only"
        ]],
        "A1"
    )

    # ========================================================
    # TRADE LOG
    # ========================================================

    ws.update(
        [["TRADE LOG"]],
        "A3"
    )

    trade_start = 4

    if not trade_df.empty:

        ws.update(
            [
                list(
                    trade_df.columns
                )
            ]
            +
            trade_df.values.tolist(),
            f"A{trade_start}"
        )

    # ========================================================
    # DAILY EQUITY CURVE
    # ========================================================

    equity_start = (
        trade_start
        +
        len(trade_df)
        +
        3
    )

    ws.update(
        [["DAILY EQUITY CURVE"]],
        f"A{equity_start}"
    )

    if not equity_df.empty:

        ws.update(
            [
                list(
                    equity_df.columns
                )
            ]
            +
            equity_df.values.tolist(),
            f"A{equity_start + 1}"
        )

    # ========================================================
    # DAILY TOP-10 SELECTION AUDIT
    # ========================================================

    selection_start = (
        equity_start
        +
        len(equity_df)
        +
        3
    )

    ws.update(
        [["DAILY TOP-10 BUY CANDIDATE AUDIT"]],
        f"A{selection_start}"
    )

    if not selection_df.empty:

        ws.update(
            [
                list(
                    selection_df.columns
                )
            ]
            +
            selection_df.values.tolist(),
            f"A{selection_start + 1}"
        )

    print(
        f"Backtest written to "
        f"'{BACKTEST_WORKSHEET}'"
    )


# ============================================================
# MAIN
# ============================================================

def run_backtest_main():

    # ========================================================
    # LOAD UNIVERSE
    # ========================================================

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    (
        download_start,
        download_end
    ) = get_download_dates()

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "RS SCREENER EOD BACKTEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Download start       : "
        f"{download_start}"
    )

    print(
        f"Backtest start       : "
        f"{BACKTEST_START}"
    )

    print(
        f"Backtest end         : "
        f"{BACKTEST_END}"
    )

    print(
        f"Starting capital     : "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        f"Portfolio             : "
        f"TOP {TOP_N}"
    )

    print(
        "ENTRY:"
    )

    print(
        "  Blue Dot"
    )

    print(
        "  Current RS > previous 250-day RS high"
    )

    print(
        "  Previous RS <= previous day's 250-day RS high"
    )

    print(
        "  Price TT 7/7"
    )

    print(
        "  RS Line TT 7/7"
    )

    print(
        "  Liquidity"
    )

    print(
        "  Top 10 by raw RS Score"
    )

    print(
        "EXIT:"
    )

    print(
        "  ONLY RS Line < 5-day EMA"
    )

    print(
        "REMOVED:"
    )

    print(
        "  8% trailing stop"
    )

    print(
        "  5% hard stop"
    )

    print(
        "  RS rank >20 exit"
    )

    print(
        "Green Dot             : "
        "Diagnostic only"
    )

    print(
        "Execution             : "
        "EOD theoretical close"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    bench_close = (
        download_benchmark()
    )

    # ========================================================
    # DOWNLOAD STOCK DATA
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
            i:
            i + batch_size
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

        for symbol in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    if (
                        symbol
                        not in
                        data.columns.get_level_values(
                            0
                        )
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

                volume = (
                    sdata[
                        "Volume"
                    ]
                    .reindex(
                        close.index
                    )
                    .fillna(0)
                )

                if close.empty:
                    continue

                # ------------------------------------------------
                # CLEAN DATA FIRST
                # ------------------------------------------------

                (
                    close,
                    n_bad
                ) = clean_price_series(
                    close
                )

                total_bad_points += (
                    n_bad
                )

                # ------------------------------------------------
                # COMPUTE SIGNALS
                # ------------------------------------------------

                sig = (
                    compute_signals_for_stock(
                        close,
                        volume,
                        bench_close
                    )
                )

                if sig is None:
                    continue

                clean_symbol = (
                    symbol.replace(
                        ".NS",
                        ""
                    )
                )

                all_signals[
                    clean_symbol
                ] = sig

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
        f"Total data points repaired: "
        f"{total_bad_points}"
    )

    # ========================================================
    # TRADING DAYS
    # ========================================================

    trading_days = (
        bench_close.index[
            (
                bench_close.index
                >=
                pd.Timestamp(
                    BACKTEST_START
                )
            )
            &
            (
                bench_close.index
                <=
                pd.Timestamp(
                    BACKTEST_END
                )
            )
        ]
    )

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    print(
        "\nRunning backtest..."
    )

    (
        trades,
        equity,
        daily_top10
    ) = run_backtest(
        all_signals,
        trading_days
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = summarize(
        trades,
        equity
    )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "FINAL BACKTEST RESULTS"
    )

    print(
        "=" * 70
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 70
    )

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    write_to_sheet(
        trades,
        equity,
        daily_top10,
        summary
    )

    print(
        "\nBACKTEST COMPLETED SUCCESSFULLY."
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        run_backtest_main()

    except Exception as e:

        print(
            "\nBACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise