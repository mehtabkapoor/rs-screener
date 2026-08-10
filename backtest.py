"""
======================================================================
RS SCREENER BACKTEST
======================================================================

DAILY EOD TOP-10 RS REBALANCE
WITH PRICE TT + RS-LINE TT + LIQUIDITY

======================================================================
TRADING LOGIC
======================================================================

EVERY TRADING DAY:

1. PRICE > Rs.20

2. 20-DAY AVERAGE VOLUME > 100,000 SHARES

3. PRICE TREND TEMPLATE = 7/7

4. RS LINE TREND TEMPLATE = 7/7

5. Eligible stocks are sorted by RAW RS SCORE:

       Rank 1 = highest RS Score
       Rank 2
       ...
       Rank 10

6. TOP 10 become the target portfolio.

7. Portfolio is rebalanced DAILY at EOD.

8. Equal-weight target:

       Portfolio Value / Number of Target Stocks

9. Stocks leaving TOP 10 are sold.

10. New TOP-10 stocks are bought.

11. Existing positions are resized toward equal weight.

======================================================================
REMOVED FROM TRADING LOGIC
======================================================================

- Blue Dot entry requirement
- 1-year RS-line crossover requirement
- RS < 5-day EMA exit
- 8% trailing stop
- 5% hard stop
- RS rank >20 exit
- Green Dot requirement

Blue Dot / 1-year RS crossover / Green Dot may still be calculated
as DIAGNOSTIC fields only.

======================================================================
RS SCORE
======================================================================

Raw RS Score:

40% 63-day price return
20% 126-day price return
20% 189-day price return
20% 252-day price return

======================================================================
LIQUIDITY
======================================================================

Price > Rs.20

AND

20-day average volume > 100,000 shares

Both are point-in-time filters.

======================================================================
TRANSACTION COSTS
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
- DP charge Rs.20 per SELL TRANSACTION

Brokerage = zero.

======================================================================
STCG
======================================================================

20% STCG
+ 4% cess
= 20.8%

Tax is applied only to POSITIVE REALIZED gains.

No loss set-off/carry-forward.

FIFO lots are used for realized cost basis.

======================================================================
DATA CLEANING
======================================================================

Single-day price changes > +/-30% are treated as potential
split/bonus/merger/data corruption.

The affected price is replaced by the previous valid close.

Cleaning occurs BEFORE:

- RS calculation
- Trend Template
- RS Score
- RS diagnostics
- Backtest

======================================================================
EXECUTION
======================================================================

The model is a THEORETICAL SAME-DAY EOD execution model.

Today's EOD signals are executed at today's EOD close.

This is NOT an executable intraday backtest.

======================================================================
SURVIVORSHIP BIAS
======================================================================

Universe comes from stocks.csv.

If stocks.csv contains today's surviving universe projected backward,
the backtest remains exposed to survivorship bias.

======================================================================
OUTPUT
======================================================================

CSV fallback:

- backtest_trades.csv
- backtest_equity.csv
- backtest_daily_top10.csv
- backtest_summary.csv

Charts:

- RS_Top10_Equity_Curve.png
- RS_Top10_Drawdown.png

Google Sheets:

- Backtest
- Backtest_Summary
"""


# ======================================================================
# IMPORTS
# ======================================================================

import os
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import gspread

from google.oauth2.service_account import Credentials


# ======================================================================
# CONFIGURATION
# ======================================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

TOP_N = 10

BACKTEST_START = "2016-04-01"
BACKTEST_END = "2026-08-07"

STARTING_CAPITAL = 1_000_000

DOWNLOAD_YEARS_BEFORE_START = 3


# ======================================================================
# RS CONFIGURATION
# ======================================================================

RS_ONE_YEAR_LOOKBACK = 250


# ======================================================================
# LIQUIDITY
# ======================================================================

MIN_PRICE = 20

MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20


# ======================================================================
# DATA CLEANING
# ======================================================================

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ======================================================================
# TRANSACTION COSTS
# ======================================================================

ENABLE_COSTS = True

# Delivery STT
STT_BUY_RATE = 0.001
STT_SELL_RATE = 0.001

# Stamp duty on BUY
STAMP_DUTY_RATE = 0.00015

# NSE transaction charge
EXCHANGE_CHARGE_RATE = 0.0000325

# SEBI charge
SEBI_CHARGE_RATE = 0.000001

# GST on exchange + SEBI
GST_RATE = 0.18

# DP charge per SELL transaction
DP_CHARGE_FLAT = 20


# ======================================================================
# STCG
# ======================================================================

ENABLE_STCG = True

STCG_RATE = 0.20

STCG_CESS = 0.04

STCG_EFFECTIVE_RATE = (
    STCG_RATE
    *
    (1 + STCG_CESS)
)


# ======================================================================
# GOOGLE SHEETS
# ======================================================================

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"

SUMMARY_WORKSHEET = "Backtest_Summary"


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
        * trade_value
    )

    stamp = (
        STAMP_DUTY_RATE
        * trade_value
    )

    exchange = (
        EXCHANGE_CHARGE_RATE
        * trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        * trade_value
    )

    gst = (
        GST_RATE
        * (
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
        * trade_value
    )

    exchange = (
        EXCHANGE_CHARGE_RATE
        * trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        * trade_value
    )

    gst = (
        GST_RATE
        * (
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


def stcg_tax(realized_gain):
    """
    Simplified STCG.

    Tax only on positive realized gain.
    """

    if not ENABLE_STCG:
        return 0.0

    if realized_gain <= 0:
        return 0.0

    return (
        realized_gain
        *
        STCG_EFFECTIVE_RATE
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
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d")
    )


# ======================================================================
# LOAD TICKERS
# ======================================================================

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


# ======================================================================
# DATA CLEANING
# ======================================================================

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
        return (
            close,
            0
        )

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = (
            cleaned.index.get_loc(idx)
        )

        if pos > 0:

            cleaned.iloc[pos] = (
                cleaned.iloc[pos - 1]
            )

    return (
        cleaned,
        n_bad
    )


# ======================================================================
# BENCHMARK DOWNLOAD
# ======================================================================

def download_benchmark():

    (
        download_start,
        download_end
    ) = get_download_dates()

    print(
        f"\nBenchmark download: "
        f"{download_start} -> {download_end}"
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
                    f"repaired {n_bad} points"
                )

            print(
                f"Benchmark loaded: {ticker}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: {e}"
            )

    raise RuntimeError(
        "Could not download benchmark data."
    )


# ======================================================================
# TREND TEMPLATE
# ======================================================================

def trend_template_series(s):

    sma50 = (
        s
        .rolling(50)
        .mean()
    )

    sma150 = (
        s
        .rolling(150)
        .mean()
    )

    sma200 = (
        s
        .rolling(200)
        .mean()
    )

    sma200_1mo = (
        sma200.shift(21)
    )

    low52 = (
        s
        .rolling(252)
        .min()
    )

    high52 = (
        s
        .rolling(252)
        .max()
    )

    # --------------------------------------------------------------
    # TT CONDITIONS
    # --------------------------------------------------------------

    c1 = (
        (s > sma150)
        &
        (s > sma200)
    )

    c2 = (
        sma150 > sma200
    )

    c3 = (
        sma200 > sma200_1mo
    )

    c4 = (
        (sma50 > sma150)
        &
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
# SIGNAL CALCULATION
# ======================================================================

def compute_signals_for_stock(
    close,
    volume,
    bench_close
):
    """
    Calculate all stock-level signals.

    TRADING FILTERS:

        Price > Rs.20
        20-day average volume > 100,000
        Price TT 7/7
        RS TT 7/7

    RANKING:

        Raw RS Score

    DIAGNOSTIC ONLY:

        Blue Dot
        1-year RS crossover
        Green Dot

    NO RS EMA.
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

    if len(aligned) < 280:
        return None

    volume = (
        volume
        .reindex(
            aligned.index
        )
    )

    # ==================================================================
    # RS LINE
    # ==================================================================

    rs_line = (
        aligned["s"]
        /
        aligned["b"]
    )

    # ==================================================================
    # RAW RS SCORE
    # ==================================================================

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

    # ==================================================================
    # 1-YEAR RS HIGH
    # ==================================================================
    #
    # DIAGNOSTIC ONLY.
    #
    # Today's RS is excluded.
    # ==================================================================

    rs_1y_high = (
        rs_line
        .shift(1)
        .rolling(
            RS_ONE_YEAR_LOOKBACK
        )
        .max()
    )

    previous_rs_1y_high = (
        rs_1y_high.shift(1)
    )

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

    blue_dot = (
        rs_line
        >
        rs_1y_high
    )

    # ==================================================================
    # GREEN DOT
    # ==================================================================
    #
    # DIAGNOSTIC ONLY.
    # ==================================================================

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
    # PRICE TREND TEMPLATE
    # ==================================================================

    (
        tt_pass,
        tt_met
    ) = trend_template_series(
        aligned["s"]
    )

    # ==================================================================
    # RS LINE TREND TEMPLATE
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
        (aligned["s"] > MIN_PRICE)
        &
        (
            avg_volume
            >
            MIN_AVG_VOLUME
        )
    )

    # ==================================================================
    # 50 DMA DIAGNOSTIC
    # ==================================================================

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

    # ==================================================================
    # OUTPUT
    # ==================================================================

    return pd.DataFrame({

        "price":
            aligned["s"],

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

        "rs_1y_high":
            rs_1y_high,

        "previous_rs_1y_high":
            previous_rs_1y_high,

        "rs_cross_1y":
            rs_cross_1y,

        "blue_dot":
            blue_dot,

        "green_dot":
            green_dot,

        "tt_pass":
            tt_pass,

        "tt_met":
            tt_met,

        "rs_tt_pass":
            rs_tt_pass,

        "rs_tt_met":
            rs_tt_met,

        "above_50dma":
            above_50dma,
    })


# ======================================================================
# FIFO SELL
# ======================================================================

def execute_sell_fifo(
    position,
    sell_qty,
    sell_price
):
    """
    Sell shares using FIFO lots.

    Returns:

        gross_proceeds
        sell_cost
        realized_gain
        tax
        net_cash
        remaining_qty
        realized_cost_basis
    """

    sell_qty = int(
        sell_qty
    )

    if sell_qty <= 0:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            position["qty"],
            0.0
        )

    if sell_qty > position["qty"]:

        raise ValueError(
            "Attempting to sell more shares "
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

    remaining_to_sell = (
        sell_qty
    )

    realized_cost_basis = 0.0

    new_lots = []

    for lot in position["lots"]:

        if remaining_to_sell <= 0:

            new_lots.append(
                lot
            )

            continue

        lot_qty = int(
            lot["qty"]
        )

        qty_from_lot = min(
            lot_qty,
            remaining_to_sell
        )

        # Original purchase cost of lot
        lot_total_cost = (
            lot_qty
            *
            lot["price"]
            +
            lot["buy_cost"]
        )

        per_share_cost = (
            lot_total_cost
            /
            lot_qty
        )

        realized_cost_basis += (
            qty_from_lot
            *
            per_share_cost
        )

        remaining_lot_qty = (
            lot_qty
            -
            qty_from_lot
        )

        if remaining_lot_qty > 0:

            remaining_lot_total_cost = (
                remaining_lot_qty
                *
                per_share_cost
            )

            remaining_lot_buy_cost = max(
                remaining_lot_total_cost
                -
                (
                    remaining_lot_qty
                    *
                    lot["price"]
                ),
                0.0
            )

            new_lots.append({

                "qty":
                    remaining_lot_qty,

                "price":
                    lot["price"],

                "buy_cost":
                    remaining_lot_buy_cost,

                "entry_date":
                    lot["entry_date"],
            })

        remaining_to_sell -= (
            qty_from_lot
        )

    if remaining_to_sell != 0:

        raise RuntimeError(
            "FIFO lot accounting failure."
        )

    realized_gain = (
        gross_proceeds
        -
        sell_cost
        -
        realized_cost_basis
    )

    tax = stcg_tax(
        realized_gain
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

    position["qty"] = sum(
        int(
            lot["qty"]
        )
        for lot in new_lots
    )

    return (
        gross_proceeds,
        sell_cost,
        realized_gain,
        tax,
        net_cash,
        position["qty"],
        realized_cost_basis
    )


# ======================================================================
# POSITION MARKET VALUE
# ======================================================================

def get_position_value(
    sym,
    position,
    date,
    all_signals
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

        price = float(
            position["lots"][-1]["price"]
        )

    return (
        position["qty"]
        *
        price
    )


# ======================================================================
# DAILY TOP-10 BACKTEST
# ======================================================================

def run_backtest(
    all_signals,
    trading_days
):

    cash = float(
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    daily_selection_log = []

    # ==================================================================
    # DAILY LOOP
    # ==================================================================

    for date in trading_days:

        # ==============================================================
        # 1. COMPLETE RS RANKING
        # ==============================================================

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
            sym: rank
            for rank, (
                sym,
                _
            ) in enumerate(
                rs_rank_pool,
                start=1
            )
        }

        # ==============================================================
        # 2. BUILD ELIGIBLE UNIVERSE
        # ==============================================================

        eligible = []

        for sym, df in all_signals.items():

            if date not in df.index:
                continue

            row = df.loc[date]

            if pd.isna(
                row["rs_score"]
            ):
                continue

            # ----------------------------------------------------------
            # PRICE
            # ----------------------------------------------------------

            if float(
                row["price"]
            ) <= MIN_PRICE:
                continue

            # ----------------------------------------------------------
            # 20-DAY AVG VOLUME
            # ----------------------------------------------------------

            if pd.isna(
                row["avg_volume"]
            ):
                continue

            if float(
                row["avg_volume"]
            ) <= MIN_AVG_VOLUME:
                continue

            # ----------------------------------------------------------
            # PRICE TREND TEMPLATE
            # ----------------------------------------------------------

            if not bool(
                row["tt_pass"]
            ):
                continue

            # ----------------------------------------------------------
            # RS-LINE TREND TEMPLATE
            # ----------------------------------------------------------

            if not bool(
                row["rs_tt_pass"]
            ):
                continue

            eligible.append(
                (
                    sym,
                    float(
                        row["rs_score"]
                    )
                )
            )

        # ==============================================================
        # 3. SORT ELIGIBLE STOCKS
        # ==============================================================

        eligible.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ==============================================================
        # 4. TARGET TOP 10
        # ==============================================================

        top10_records = (
            eligible[:TOP_N]
        )

        target_symbols = [
            sym
            for sym, _
            in top10_records
        ]

        target_set = set(
            target_symbols
        )

        # ==============================================================
        # 5. DAILY TOP-10 AUDIT
        # ==============================================================

        for rank, (
            sym,
            score
        ) in enumerate(
            top10_records,
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

                "rank":
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
                        ""
                    ),

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
                            row["avg_volume"]
                        ),
                        0
                    ),

                "tt_met":
                    int(
                        row["tt_met"]
                    ),

                "rs_tt_met":
                    int(
                        row["rs_tt_met"]
                    ),

                # ------------------------------------------------------
                # DIAGNOSTIC ONLY
                # ------------------------------------------------------

                "blue_dot":
                    bool(
                        row["blue_dot"]
                    ),

                "rs_cross_1y":
                    bool(
                        row["rs_cross_1y"]
                    ),

                "green_dot":
                    bool(
                        row["green_dot"]
                    ),

                "liquid":
                    bool(
                        row["liquid"]
                    ),
            })

        # ==============================================================
        # 6. SELL STOCKS THAT LEFT TOP 10
        # ==============================================================

        symbols_to_remove = [
            sym
            for sym in list(
                holdings.keys()
            )
            if sym not in target_set
        ]

        for sym in symbols_to_remove:

            position = holdings[sym]

            df = all_signals[sym]

            if date not in df.index:
                continue

            row = df.loc[date]

            exit_price = float(
                row["price"]
            )

            sell_qty = int(
                position["qty"]
            )

            original_qty = sell_qty

            (
                gross_proceeds,
                sell_cost,
                realized_gain,
                tax,
                net_cash,
                remaining_qty,
                realized_cost_basis
            ) = execute_sell_fifo(
                position,
                sell_qty,
                exit_price
            )

            cash += net_cash

            entry_total_cost = (
                sum(
                    lot["qty"]
                    *
                    lot["price"]
                    +
                    lot["buy_cost"]
                    for lot
                    in position.get(
                        "lots",
                        []
                    )
                )
            )

            net_return_pct = (
                realized_gain
                /
                realized_cost_basis
                *
                100
                if realized_cost_basis > 0
                else 0
            )

            trade_log.append({

                "symbol":
                    sym,

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
                    original_qty,

                "entry_price":
                    "",

                "exit_price":
                    round(
                        exit_price,
                        2
                    ),

                "entry_rs_score":
                    position.get(
                        "entry_rs_score",
                        ""
                    ),

                "exit_rs_score":
                    round(
                        float(
                            row["rs_score"]
                        ),
                        4
                    ),

                "entry_rank":
                    position.get(
                        "entry_rank",
                        ""
                    ),

                "exit_rank":
                    rs_rank_lookup.get(
                        sym,
                        ""
                    ),

                "realized_cost_basis_rs":
                    round(
                        realized_cost_basis,
                        2
                    ),

                "gross_proceeds_rs":
                    round(
                        gross_proceeds,
                        2
                    ),

                "gross_return_pct":
                    round(
                        (
                            gross_proceeds
                            /
                            realized_cost_basis
                            -
                            1
                        )
                        * 100,
                        2
                    )
                    if realized_cost_basis > 0
                    else 0,

                "buy_cost_rs":
                    round(
                        realized_cost_basis
                        -
                        (
                            sum(
                                lot["qty"]
                                *
                                lot["price"]
                                for lot
                                in position.get(
                                    "lots",
                                    []
                                )
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
                        realized_gain
                        -
                        tax,
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
                        position[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "LEFT TOP 10",

                "entry_blue_dot":
                    position.get(
                        "entry_blue_dot",
                        False
                    ),

                "entry_rs_cross_1y":
                    position.get(
                        "entry_rs_cross_1y",
                        False
                    ),

                "entry_green_dot":
                    position.get(
                        "entry_green_dot",
                        False
                    ),

                "exit_blue_dot":
                    bool(
                        row["blue_dot"]
                    ),

                "exit_rs_cross_1y":
                    bool(
                        row["rs_cross_1y"]
                    ),

                "exit_green_dot":
                    bool(
                        row["green_dot"]
                    ),
            })

            del holdings[sym]

        # ==============================================================
        # 7. PORTFOLIO VALUE AFTER MANDATORY EXITS
        # ==============================================================

        portfolio_value = cash

        for sym, position in holdings.items():

            portfolio_value += (
                get_position_value(
                    sym,
                    position,
                    date,
                    all_signals
                )
            )

        # ==============================================================
        # 8. TARGET WEIGHT
        # ==============================================================

        n_targets = len(
            target_symbols
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
        # 9. SELL OVERWEIGHT EXISTING POSITIONS
        # ==============================================================

        for sym in target_symbols:

            if sym not in holdings:
                continue

            df = all_signals[sym]

            if date not in df.index:
                continue

            price = float(
                df.loc[
                    date,
                    "price"
                ]
            )

            position = holdings[sym]

            current_qty = int(
                position["qty"]
            )

            target_qty = int(
                target_value
                //
                price
            )

            excess_qty = (
                current_qty
                -
                target_qty
            )

            if excess_qty <= 0:
                continue

            (
                gross_proceeds,
                sell_cost,
                realized_gain,
                tax,
                net_cash,
                remaining_qty,
                realized_cost_basis
            ) = execute_sell_fifo(
                position,
                excess_qty,
                price
            )

            cash += net_cash

            net_return_pct = (
                realized_gain
                /
                realized_cost_basis
                *
                100
                if realized_cost_basis > 0
                else 0
            )

            trade_log.append({

                "symbol":
                    sym,

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
                    excess_qty,

                "entry_price":
                    "",

                "exit_price":
                    round(
                        price,
                        2
                    ),

                "entry_rs_score":
                    position.get(
                        "entry_rs_score",
                        ""
                    ),

                "exit_rs_score":
                    round(
                        float(
                            all_signals[
                                sym
                            ].loc[
                                date,
                                "rs_score"
                            ]
                        ),
                        4
                    ),

                "entry_rank":
                    position.get(
                        "entry_rank",
                        ""
                    ),

                "exit_rank":
                    rs_rank_lookup.get(
                        sym,
                        ""
                    ),

                "realized_cost_basis_rs":
                    round(
                        realized_cost_basis,
                        2
                    ),

                "gross_proceeds_rs":
                    round(
                        gross_proceeds,
                        2
                    ),

                "gross_return_pct":
                    round(
                        (
                            gross_proceeds
                            /
                            realized_cost_basis
                            -
                            1
                        )
                        * 100,
                        2
                    )
                    if realized_cost_basis > 0
                    else 0,

                "buy_cost_rs":
                    "",

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
                        realized_gain
                        -
                        tax,
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
                        position[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "DAILY EQUAL-WEIGHT REBALANCE",

                "entry_blue_dot":
                    position.get(
                        "entry_blue_dot",
                        False
                    ),

                "entry_rs_cross_1y":
                    position.get(
                        "entry_rs_cross_1y",
                        False
                    ),

                "entry_green_dot":
                    position.get(
                        "entry_green_dot",
                        False
                    ),

                "exit_blue_dot":
                    bool(
                        all_signals[
                            sym
                        ].loc[
                            date,
                            "blue_dot"
                        ]
                    ),

                "exit_rs_cross_1y":
                    bool(
                        all_signals[
                            sym
                        ].loc[
                            date,
                            "rs_cross_1y"
                        ]
                    ),

                "exit_green_dot":
                    bool(
                        all_signals[
                            sym
                        ].loc[
                            date,
                            "green_dot"
                        ]
                    ),
            })

        # ==============================================================
        # 10. RECALCULATE TARGET VALUE
        #
        # After overweight sales, cash changed.
        # Recalculate portfolio value before buying.
        # ==============================================================

        portfolio_value = cash

        for sym, position in holdings.items():

            portfolio_value += (
                get_position_value(
                    sym,
                    position,
                    date,
                    all_signals
                )
            )

        n_targets = len(
            target_symbols
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
        # 11. BUY UNDERWEIGHT POSITIONS
        # ==============================================================

        for rank, sym in enumerate(
            target_symbols,
            start=1
        ):

            df = all_signals[sym]

            if date not in df.index:
                continue

            row = df.loc[date]

            price = float(
                row["price"]
            )

            if price <= 0:
                continue

            if sym in holdings:

                current_qty = int(
                    holdings[sym]["qty"]
                )

            else:

                current_qty = 0

            current_value = (
                current_qty
                *
                price
            )

            required_value = (
                target_value
                -
                current_value
            )

            if required_value <= 0:
                continue

            additional_qty = int(
                required_value
                //
                price
            )

            if additional_qty < 1:
                continue

            # ----------------------------------------------------------
            # Ensure transaction costs fit available cash.
            # ----------------------------------------------------------

            while additional_qty > 0:

                trade_value = (
                    additional_qty
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

                if total_required <= cash:
                    break

                additional_qty -= 1

            if additional_qty < 1:
                continue

            trade_value = (
                additional_qty
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

            # ----------------------------------------------------------
            # NEW POSITION
            # ----------------------------------------------------------

            if sym not in holdings:

                holdings[sym] = {

                    "qty":
                        additional_qty,

                    "lots": [
                        {
                            "qty":
                                additional_qty,

                            "price":
                                price,

                            "buy_cost":
                                buy_cost,

                            "entry_date":
                                date,
                        }
                    ],

                    "entry_date":
                        date,

                    "entry_rank":
                        rank,

                    "entry_rs_score":
                        float(
                            row["rs_score"]
                        ),

                    "entry_blue_dot":
                        bool(
                            row["blue_dot"]
                        ),

                    "entry_rs_cross_1y":
                        bool(
                            row["rs_cross_1y"]
                        ),

                    "entry_green_dot":
                        bool(
                            row["green_dot"]
                        ),
                }

                trade_log.append({

                    "symbol":
                        sym,

                    "entry_date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "exit_date":
                        "",

                    "qty":
                        additional_qty,

                    "entry_price":
                        round(
                            price,
                            2
                        ),

                    "exit_price":
                        "",

                    "entry_rs_score":
                        round(
                            float(
                                row[
                                    "rs_score"
                                ]
                            ),
                            4
                        ),

                    "exit_rs_score":
                        "",

                    "entry_rank":
                        rank,

                    "exit_rank":
                        "",

                    "realized_cost_basis_rs":
                        "",

                    "gross_proceeds_rs":
                        "",

                    "gross_return_pct":
                        "",

                    "buy_cost_rs":
                        round(
                            buy_cost,
                            2
                        ),

                    "sell_cost_rs":
                        "",

                    "stcg_tax_rs":
                        "",

                    "net_pnl_rs":
                        "",

                    "net_return_pct":
                        "",

                    "days_held":
                        "",

                    "exit_reason":
                        "",

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

                    "exit_blue_dot":
                        "",

                    "exit_rs_cross_1y":
                        "",

                    "exit_green_dot":
                        "",
                })

            # ----------------------------------------------------------
            # EXISTING POSITION
            # ----------------------------------------------------------

            else:

                holdings[sym]["qty"] += (
                    additional_qty
                )

                holdings[sym][
                    "lots"
                ].append({

                    "qty":
                        additional_qty,

                    "price":
                        price,

                    "buy_cost":
                        buy_cost,

                    "entry_date":
                        date,
                })

        # ==============================================================
        # 12. FINAL MARK-TO-MARKET
        # ==============================================================

        portfolio_value = cash

        invested_value = 0.0

        for sym, position in holdings.items():

            position_value = (
                get_position_value(
                    sym,
                    position,
                    date,
                    all_signals
                )
            )

            invested_value += (
                position_value
            )

            portfolio_value += (
                position_value
            )

        # ==============================================================
        # 13. DAILY EQUITY
        # ==============================================================

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

            "top10":
                ",".join(
                    target_symbols
                ),

            "holdings":
                ",".join(
                    sorted(
                        holdings.keys()
                    )
                ),
        })

    # ==================================================================
    # EQUITY METRICS
    # ==================================================================

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
            "equity_multiple"
        ] = (
            equity_df[
                "portfolio_value_rs"
            ]
            /
            STARTING_CAPITAL
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

    # ==================================================================
    # MARK OPEN POSITIONS AT BACKTEST END
    # ==================================================================
    #
    # These are NOT added to realized trade P&L.
    #
    # They remain open.
    #
    # We calculate:
    #
    # 1. Marked equity
    # 2. Hypothetical liquidation value
    #
    # separately.
    # ==================================================================

    final_marked_equity = 0.0
    final_liquidation_equity = 0.0

    if len(trading_days):

        last_date = (
            trading_days[-1]
        )

        final_marked_equity = cash

        for sym, position in holdings.items():

            df = all_signals[sym]

            if last_date in df.index:

                row = df.loc[
                    last_date
                ]

                final_price = float(
                    row["price"]
                )

            else:

                row = None

                final_price = float(
                    position[
                        "lots"
                    ][-1]["price"]
                )

            market_value = (
                position["qty"]
                *
                final_price
            )

            final_marked_equity += (
                market_value
            )

            # ----------------------------------------------------------
            # Hypothetical liquidation
            # ----------------------------------------------------------

            hypothetical_sell_cost = (
                sell_side_cost(
                    market_value
                )
            )

            total_cost_basis = sum(
                lot["qty"]
                *
                lot["price"]
                +
                lot["buy_cost"]
                for lot in position["lots"]
            )

            hypothetical_gain = (
                market_value
                -
                hypothetical_sell_cost
                -
                total_cost_basis
            )

            hypothetical_tax = (
                stcg_tax(
                    hypothetical_gain
                )
            )

            final_liquidation_equity += (
                market_value
                -
                hypothetical_sell_cost
                -
                hypothetical_tax
            )

        final_liquidation_equity += (
            cash
        )

        # --------------------------------------------------------------
        # Open-position audit records
        # --------------------------------------------------------------

        for sym, position in holdings.items():

            df = all_signals[sym]

            if last_date in df.index:

                row = df.loc[
                    last_date
                ]

                final_price = float(
                    row["price"]
                )

                final_rs_score = (
                    float(
                        row["rs_score"]
                    )
                    if pd.notna(
                        row["rs_score"]
                    )
                    else np.nan
                )

                final_rank = (
                    rs_rank_lookup.get(
                        sym,
                        ""
                    )
                    if "rs_rank_lookup" in locals()
                    else ""
                )

            else:

                row = None

                final_price = float(
                    position[
                        "lots"
                    ][-1]["price"]
                )

                final_rs_score = np.nan

                final_rank = ""

            market_value = (
                position["qty"]
                *
                final_price
            )

            total_cost_basis = sum(
                lot["qty"]
                *
                lot["price"]
                +
                lot["buy_cost"]
                for lot in position["lots"]
            )

            hypothetical_sell_cost = (
                sell_side_cost(
                    market_value
                )
            )

            hypothetical_gain = (
                market_value
                -
                hypothetical_sell_cost
                -
                total_cost_basis
            )

            hypothetical_tax = (
                stcg_tax(
                    hypothetical_gain
                )
            )

            hypothetical_net_pnl = (
                hypothetical_gain
                -
                hypothetical_tax
            )

            trade_log.append({

                "symbol":
                    sym,

                "entry_date":
                    position[
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
                    position["qty"],

                "entry_price":
                    "",

                "exit_price":
                    round(
                        final_price,
                        2
                    ),

                "entry_rs_score":
                    position.get(
                        "entry_rs_score",
                        ""
                    ),

                "exit_rs_score":
                    (
                        round(
                            final_rs_score,
                            4
                        )
                        if pd.notna(
                            final_rs_score
                        )
                        else ""
                    ),

                "entry_rank":
                    position.get(
                        "entry_rank",
                        ""
                    ),

                "exit_rank":
                    final_rank,

                "realized_cost_basis_rs":
                    "",

                "gross_proceeds_rs":
                    round(
                        market_value,
                        2
                    ),

                "gross_return_pct":
                    round(
                        (
                            market_value
                            /
                            total_cost_basis
                            -
                            1
                        )
                        * 100,
                        2
                    )
                    if total_cost_basis > 0
                    else 0,

                "buy_cost_rs":
                    round(
                        sum(
                            lot[
                                "buy_cost"
                            ]
                            for lot
                            in position[
                                "lots"
                            ]
                        ),
                        2
                    ),

                "sell_cost_rs":
                    round(
                        hypothetical_sell_cost,
                        2
                    ),

                "stcg_tax_rs":
                    round(
                        hypothetical_tax,
                        2
                    ),

                "net_pnl_rs":
                    round(
                        hypothetical_net_pnl,
                        2
                    ),

                "net_return_pct":
                    round(
                        (
                            hypothetical_net_pnl
                            /
                            total_cost_basis
                        )
                        * 100,
                        2
                    )
                    if total_cost_basis > 0
                    else 0,

                "days_held":
                    (
                        last_date
                        -
                        position[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "BACKTEST END - OPEN",

                "entry_blue_dot":
                    position.get(
                        "entry_blue_dot",
                        False
                    ),

                "entry_rs_cross_1y":
                    position.get(
                        "entry_rs_cross_1y",
                        False
                    ),

                "entry_green_dot":
                    position.get(
                        "entry_green_dot",
                        False
                    ),

                "exit_blue_dot":
                    (
                        bool(
                            row[
                                "blue_dot"
                            ]
                        )
                        if row is not None
                        else False
                    ),

                "exit_rs_cross_1y":
                    (
                        bool(
                            row[
                                "rs_cross_1y"
                            ]
                        )
                        if row is not None
                        else False
                    ),

                "exit_green_dot":
                    (
                        bool(
                            row[
                                "green_dot"
                            ]
                        )
                        if row is not None
                        else False
                    ),
            })

    return (
        pd.DataFrame(
            trade_log
        ),
        equity_df,
        pd.DataFrame(
            daily_selection_log
        ),
        final_marked_equity,
        final_liquidation_equity
    )


# ======================================================================
# PERFORMANCE SUMMARY
# ======================================================================

def summarize(
    trade_df,
    equity_df,
    final_marked_equity,
    final_liquidation_equity
):

    if equity_df.empty:
        return {}

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

    # ==================================================================
    # DRAWDOWN
    # ==================================================================

    running_peak = (
        equity_df[
            "portfolio_value_rs"
        ]
        .cummax()
    )

    drawdown_pct = (
        equity_df[
            "portfolio_value_rs"
        ]
        /
        running_peak
        -
        1
    ) * 100

    max_dd = (
        drawdown_pct.min()
    )

    # ==================================================================
    # CLOSED TRADES
    # ==================================================================

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
        ].copy()

    else:

        closed = pd.DataFrame()

    # ==================================================================
    # TRADE STATISTICS
    # ==================================================================

    if not closed.empty:

        closed["net_return_pct"] = (
            pd.to_numeric(
                closed[
                    "net_return_pct"
                ],
                errors="coerce"
            )
        )

        closed["net_pnl_rs"] = (
            pd.to_numeric(
                closed[
                    "net_pnl_rs"
                ],
                errors="coerce"
            )
            .fillna(0)
        )

        closed["gross_return_pct"] = (
            pd.to_numeric(
                closed[
                    "gross_return_pct"
                ],
                errors="coerce"
            )
        )

        closed["days_held"] = (
            pd.to_numeric(
                closed[
                    "days_held"
                ],
                errors="coerce"
            )
        )

        win_rate_net = (
            (
                closed[
                    "net_return_pct"
                ]
                > 0
            )
            .mean()
            * 100
        )

        avg_net = (
            closed[
                "net_return_pct"
            ]
            .mean()
        )

        median_net = (
            closed[
                "net_return_pct"
            ]
            .median()
        )

        avg_days = (
            closed[
                "days_held"
            ]
            .mean()
        )

        median_days = (
            closed[
                "days_held"
            ]
            .median()
        )

        best_net = (
            closed[
                "net_return_pct"
            ]
            .max()
        )

        worst_net = (
            closed[
                "net_return_pct"
            ]
            .min()
        )

        winners = closed[
            closed[
                "net_pnl_rs"
            ] > 0
        ]

        losers = closed[
            closed[
                "net_pnl_rs"
            ] < 0
        ]

        total_winning_pnl = (
            winners[
                "net_pnl_rs"
            ]
            .sum()
        )

        total_losing_pnl = abs(
            losers[
                "net_pnl_rs"
            ]
            .sum()
        )

        if total_losing_pnl > 0:

            profit_factor = (
                total_winning_pnl
                /
                total_losing_pnl
            )

        else:

            profit_factor = np.inf

        n_closed = len(
            closed
        )

    else:

        win_rate_net = 0
        avg_net = 0
        median_net = 0
        avg_days = 0
        median_days = 0
        best_net = 0
        worst_net = 0
        profit_factor = 0
        n_closed = 0

    # ==================================================================
    # COST / TAX TOTALS
    # ==================================================================

    def numeric_sum(column):

        if (
            trade_df.empty
            or
            column not in trade_df.columns
        ):
            return 0.0

        return (
            pd.to_numeric(
                trade_df[column],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    total_buy_costs = (
        numeric_sum(
            "buy_cost_rs"
        )
    )

    total_sell_costs = (
        numeric_sum(
            "sell_cost_rs"
        )
    )

    total_stcg = (
        numeric_sum(
            "stcg_tax_rs"
        )
    )

    total_transaction_costs = (
        total_buy_costs
        +
        total_sell_costs
    )

    # ==================================================================
    # DAILY METRICS
    # ==================================================================

    daily_returns = (
        equity_df[
            "portfolio_value_rs"
        ]
        .pct_change()
        .dropna()
    )

    n_days = len(
        equity_df
    )

    if len(
        daily_returns
    ) > 1:

        daily_mean = (
            daily_returns.mean()
        )

        daily_std = (
            daily_returns.std()
        )

        annualized_return = (
            (
                final_value
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

        annualized_volatility = (
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
        ) > 1:

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
        annualized_volatility = 0
        sharpe = 0
        sortino = 0

    # ==================================================================
    # CALMAR
    # ==================================================================

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

    # ==================================================================
    # SUMMARY
    # ==================================================================

    return {

        "strategy":
            (
                "DAILY EOD TOP-10 RS REBALANCE | "
                "Price TT 7/7 + RS TT 7/7 + "
                "Price > Rs.20 + "
                "20D Avg Volume > 100,000 | "
                "Rank by Raw RS Score"
            ),

        "entry_rule":
            (
                "Eligible = Price > Rs.20 AND "
                "20-day average volume > 100,000 AND "
                "Price TT 7/7 AND RS Line TT 7/7; "
                "rank by raw RS score; select top 10"
            ),

        "rebalance_rule":
            "Daily EOD equal-weight rebalance to current Top 10",

        "exit_rule":
            "Stock leaves daily Top 10",

        "removed_rules":
            (
                "RS < 5EMA; Blue Dot; "
                "1-Year RS Cross; "
                "8% trailing stop; "
                "5% hard stop; "
                "Rank >20 exit"
            ),

        "starting_capital_rs":
            round(
                STARTING_CAPITAL,
                2
            ),

        "final_marked_equity_rs":
            round(
                final_marked_equity,
                2
            ),

        "final_liquidation_equity_rs":
            round(
                final_liquidation_equity,
                2
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
                annualized_volatility * 100,
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

        "max_drawdown_pct":
            round(
                max_dd,
                2
            ),

        "max_drawdown_rs":
            round(
                equity_df[
                    "drawdown_rs"
                ].min(),
                2
            ),

        "n_closed_transactions":
            int(
                n_closed
            ),

        "win_rate_net_pct":
            round(
                win_rate_net,
                2
            ),

        "avg_net_return_per_sell_pct":
            round(
                avg_net,
                2
            ),

        "median_net_return_per_sell_pct":
            round(
                median_net,
                2
            ),

        "avg_days_held":
            round(
                avg_days,
                2
            ),

        "median_days_held":
            round(
                median_days,
                2
            ),

        "best_net_transaction_pct":
            round(
                best_net,
                2
            ),

        "worst_net_transaction_pct":
            round(
                worst_net,
                2
            ),

        "profit_factor_net":
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

        "total_buy_costs_rs":
            round(
                total_buy_costs,
                2
            ),

        "total_sell_costs_rs":
            round(
                total_sell_costs,
                2
            ),

        "total_transaction_costs_rs":
            round(
                total_transaction_costs,
                2
            ),

        "total_stcg_tax_rs":
            round(
                total_stcg,
                2
            ),

        "total_friction_rs":
            round(
                total_transaction_costs
                +
                total_stcg,
                2
            ),

        "data_cleaning_threshold":
            "+/-30% single-day price move",

        "price_filter":
            "> Rs.20",

        "volume_filter":
            "> 100,000 20-day average shares",

        "position_count":
            TOP_N,

        "execution":
            "Same-day EOD theoretical close",
    }


# ======================================================================
# EQUITY CURVE
# ======================================================================

def plot_equity_curve(
    equity_df
):

    if equity_df.empty:
        return

    dates = pd.to_datetime(
        equity_df["date"]
    )

    equity = (
        equity_df[
            "portfolio_value_rs"
        ]
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        dates,
        equity
    )

    plt.title(
        "Daily EOD TOP-10 RS Strategy - Equity Curve"
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
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ======================================================================
# DRAWDOWN CHART
# ======================================================================

def plot_drawdown(
    equity_df
):

    if equity_df.empty:
        return

    dates = pd.to_datetime(
        equity_df["date"]
    )

    drawdown = (
        equity_df[
            "drawdown_pct"
        ]
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        dates,
        drawdown
    )

    plt.title(
        "Daily EOD TOP-10 RS Strategy - Drawdown"
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
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ======================================================================
# GOOGLE SHEETS / CSV
# ======================================================================

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

    # ==================================================================
    # CSV FALLBACK
    # ==================================================================

    if (
        not sheet_id
        or
        not creds_json
    ):

        print(
            "\nGoogle credentials not found."
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

    # ==================================================================
    # AUTHENTICATION
    # ==================================================================

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

    # ==================================================================
    # SUMMARY
    # ==================================================================

    summary_df = pd.DataFrame(
        [summary]
    )

    try:

        sws = sh.worksheet(
            SUMMARY_WORKSHEET
        )

    except gspread.WorksheetNotFound:

        sws = sh.add_worksheet(
            title=SUMMARY_WORKSHEET,
            rows=100,
            cols=max(
                20,
                len(
                    summary_df.columns
                )
                +
                5
            )
        )

    sws.clear()

    sws.update(
        [[
            "DAILY TOP-10 RS BACKTEST | "
            "Net of modeled transaction costs + STCG"
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

    # ==================================================================
    # MAIN SHEET
    # ==================================================================

    max_cols = max(

        len(
            trade_df.columns
        )
        if not trade_df.empty
        else 0,

        len(
            equity_df.columns
        )
        if not equity_df.empty
        else 0,

        len(
            selection_df.columns
        )
        if not selection_df.empty
        else 0

    ) + 5

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

    # ==================================================================
    # TRADE LOG
    # ==================================================================

    ws.update(
        [["TRADE / TRANSACTION LOG"]],
        "A1"
    )

    trade_start = 3

    if not trade_df.empty:

        ws.update(
            [
                list(
                    trade_df.columns
                )
            ]
            +
            trade_df.fillna("").values.tolist(),
            f"A{trade_start}"
        )

    # ==================================================================
    # EQUITY
    # ==================================================================

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
            equity_df.fillna("").values.tolist(),
            f"A{equity_start + 1}"
        )

    # ==================================================================
    # DAILY TOP 10
    # ==================================================================

    selection_start = (
        equity_start
        +
        len(equity_df)
        +
        3
    )

    ws.update(
        [["DAILY TOP-10 SELECTION AUDIT"]],
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
            selection_df.fillna("").values.tolist(),
            f"A{selection_start + 1}"
        )

    print(
        f"\nGoogle Sheets updated:"
        f" {BACKTEST_WORKSHEET}"
    )

    print(
        f"Summary updated:"
        f" {SUMMARY_WORKSHEET}"
    )


# ======================================================================
# MAIN
# ======================================================================

def run_backtest_main():

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
        f"Backtest: "
        f"{BACKTEST_START} -> {BACKTEST_END}"
    )

    print(
        f"Starting capital: "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        f"Portfolio size: "
        f"{TOP_N}"
    )

    print(
        "\nELIGIBILITY:"
    )

    print(
        "  Price > Rs.20"
    )

    print(
        "  20-day average volume > 100,000"
    )

    print(
        "  Price Trend Template = 7/7"
    )

    print(
        "  RS Line Trend Template = 7/7"
    )

    print(
        "\nRANKING:"
    )

    print(
        "  Raw RS Score descending"
    )

    print(
        "  Rank 1 = highest RS Score"
    )

    print(
        "\nREBALANCING:"
    )

    print(
        "  Daily EOD"
    )

    print(
        "  Equal-weight Top 10"
    )

    print(
        "\nREMOVED:"
    )

    print(
        "  Blue Dot"
    )

    print(
        "  1-year RS crossover"
    )

    print(
        "  RS < 5EMA"
    )

    print(
        "  8% trailing stop"
    )

    print(
        "  5% hard stop"
    )

    print(
        "  Rank >20 exit"
    )

    print(
        "\nCOSTS:"
    )

    print(
        "  STT + stamp + exchange + SEBI + GST + DP"
    )

    print(
        "\nTAX:"
    )

    print(
        "  20.8% STCG on positive realized gains"
    )

    print(
        "=" * 80
    )

    # ==================================================================
    # LOAD UNIVERSE
    # ==================================================================

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    (
        download_start,
        download_end
    ) = get_download_dates()

    # ==================================================================
    # BENCHMARK
    # ==================================================================

    bench_close = (
        download_benchmark()
    )

    # ==================================================================
    # STOCK DATA
    # ==================================================================

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
            f"{i + 1}-"
            f"{i + len(batch)} "
            f"of {len(tickers)}..."
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

                # ------------------------------------------------------
                # CLEAN PRICE
                # ------------------------------------------------------

                (
                    close,
                    n_bad
                ) = clean_price_series(
                    close
                )

                total_bad_points += (
                    n_bad
                )

                # ------------------------------------------------------
                # SIGNALS
                # ------------------------------------------------------

                signals = (
                    compute_signals_for_stock(
                        close,
                        volume,
                        bench_close
                    )
                )

                if signals is None:
                    continue

                clean_symbol = (
                    symbol.replace(
                        ".NS",
                        ""
                    )
                )

                all_signals[
                    clean_symbol
                ] = signals

            except Exception as e:

                print(
                    f"Skipping {symbol}: {e}"
                )

        time.sleep(1)

    print(
        f"\nSignals computed: "
        f"{len(all_signals)} stocks."
    )

    print(
        f"Data points repaired: "
        f"{total_bad_points}"
    )

    # ==================================================================
    # TRADING DAYS
    # ==================================================================

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

    # ==================================================================
    # BACKTEST
    # ==================================================================

    print(
        "\nRunning backtest..."
    )

    (
        trades,
        equity,
        daily_top10,
        final_marked_equity,
        final_liquidation_equity
    ) = run_backtest(
        all_signals,
        trading_days
    )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    summary = summarize(
        trades,
        equity,
        final_marked_equity,
        final_liquidation_equity
    )

    # ==================================================================
    # PRINT RESULTS
    # ==================================================================

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 80
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 80
    )

    # ==================================================================
    # CHARTS
    # ==================================================================

    print(
        "\nGenerating equity curve..."
    )

    plot_equity_curve(
        equity
    )

    print(
        "Generating drawdown chart..."
    )

    plot_drawdown(
        equity
    )

    # ==================================================================
    # OUTPUT
    # ==================================================================

    write_to_sheet(
        trades,
        equity,
        daily_top10,
        summary
    )

    # ==================================================================
    # ALWAYS SAVE LOCAL CSV COPIES
    # ==================================================================

    trades.to_csv(
        "backtest_trades.csv",
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

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "BACKTEST COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "Files:"
    )

    print(
        "  backtest_trades.csv"
    )

    print(
        "  backtest_equity.csv"
    )

    print(
        "  backtest_daily_top10.csv"
    )

    print(
        "  backtest_summary.csv"
    )

    print(
        "  RS_Top10_Equity_Curve.png"
    )

    print(
        "  RS_Top10_Drawdown.png"
    )


# ======================================================================
# EXECUTION
# ======================================================================

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