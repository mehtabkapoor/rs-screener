"""
RS Screener Backtest v4
AUDITED + DATA CLEANED + TOP-20 RANK BUFFER
PRIMARY EXIT = RS LINE CROSSES BELOW 5-EMA OR RANK > 20

ENTRY
-----
Blue Dot
+
Price Trend Template PASS (7/7)
+
RS Line Trend Template PASS (7/7)
+
Point-in-time liquidity filter
+
Sort by raw RS Score
+
Top 10

PORTFOLIO
---------
Maximum holdings = 10

RANK HYSTERESIS
---------------
Entry threshold = Top 10
Exit threshold  = Rank > 20

Therefore a holding can deteriorate:

    #5 -> #11 -> #15 -> #19

without being sold solely because of rank.

It is sold when:

    #21+

OR

    it disappears from the qualifying pool

OR

    RS Line crosses below its 5-EMA.

PRIMARY STRATEGY
----------------
TOP 10 ENTRY
TOP 20 RANK EXIT
RS LINE 5-EMA CROSSOVER EXIT

GROSS BACKTEST
--------------
No brokerage/STT/slippage/tax included.
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

LOOKBACK_DAYS = 250
DOWNLOAD_YEARS_BEFORE_START = 3

STOCKS_FILE = "stocks.csv"


# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

# Maximum number of actual holdings.
TOP_N = 10

# Existing holdings may remain while ranked 1-20.
# They are exited when rank becomes >20.
RANK_EXIT = 20


# ============================================================
# PRIMARY EXIT
# ============================================================

# PRIMARY:
#
# RS Line crosses below 5 EMA
#
# OR
#
# Rank > 20
#
PRIMARY_RS_EMA_SPAN = 5


# ============================================================
# LIQUIDITY FILTER
# ============================================================

MIN_PRICE = 10

MIN_AVG_VOLUME = 10000

VOLUME_LOOKBACK = 20


# ============================================================
# DATA SANITY CLEANING
# ============================================================

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ============================================================
# BACKTEST PERIOD
# ============================================================

BACKTEST_START = "2016-04-01"

BACKTEST_END = "2026-08-07"


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"

COMPARISON_WORKSHEET = "Backtest_Exit_Comparison"


# ============================================================
# EXIT VARIANTS
# ============================================================
#
# All variants use:
#
#     ENTRY = Top 10
#
# For normal variants:
#
#     RANK EXIT = >20
#
# The secondary exit varies.
#
# This allows comparison of different EMA exits while keeping
# the Top-20 rank buffer constant.
#
# ============================================================

EXIT_VARIANTS = [

    {
        "name":
            "RS<3EMA (state) + Rank20",

        "type":
            "rs_ema_state",

        "span":
            3
    },

    {
        "name":
            "RS<3EMA (crossover) + Rank20",

        "type":
            "rs_ema_cross",

        "span":
            3
    },

    {
        "name":
            "RS<5EMA (crossover) + Rank20",

        "type":
            "rs_ema_cross",

        "span":
            5
    },

    {
        "name":
            "RS<10EMA (crossover) + Rank20",

        "type":
            "rs_ema_cross",

        "span":
            10
    },

    {
        "name":
            "RS<20EMA (crossover) + Rank20",

        "type":
            "rs_ema_cross",

        "span":
            20
    },

    {
        "name":
            "RS<20EMA (state) + Rank20",

        "type":
            "rs_ema_state",

        "span":
            20
    },

    {
        "name":
            "Rank20 only",

        "type":
            "rank_buffer",

        "buffer":
            RANK_EXIT
    },

    {
        "name":
            "Trend Template fail + Rank20",

        "type":
            "tt_fail_cross"
    },

]


# ============================================================
# PRIMARY VARIANT
# ============================================================
#
# Index:
#
# 0 = RS 3 EMA state
# 1 = RS 3 EMA crossover
# 2 = RS 5 EMA crossover  <-- PRIMARY
# 3 = RS 10 EMA crossover
# 4 = RS 20 EMA crossover
# 5 = RS 20 EMA state
# 6 = Rank20 only
# 7 = TT fail
#
# ============================================================

PRIMARY_VARIANT_INDEX = 2


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():

    backtest_start = pd.Timestamp(
        BACKTEST_START
    )

    backtest_end = pd.Timestamp(
        BACKTEST_END
    )

    download_start = (
        backtest_start -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    download_end = (
        backtest_end +
        pd.Timedelta(days=1)
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
# CLEAN PRICE SERIES
# ============================================================

def clean_price_series(
    close
):

    """
    Repairs implausible single-day
    price jumps before ANY signal
    or return calculation.

    If the absolute daily move exceeds
    MAX_PLAUSIBLE_DAILY_MOVE, the point
    is replaced by the previous valid
    close.
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

        return close, 0

    cleaned = close.copy()

    bad_indices = list(
        close.index[bad]
    )

    for idx in bad_indices:

        pos = (
            cleaned.index
            .get_loc(idx)
        )

        if pos > 0:

            cleaned.iloc[pos] = (
                cleaned.iloc[pos - 1]
            )

    return (
        cleaned,
        n_bad
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

            close = data[
                "Close"
            ]

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

                    f"Benchmark "
                    f"{ticker}: repaired "
                    f"{n_bad} "
                    f"implausible point(s)"

                )

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
# TREND TEMPLATE
# ============================================================

def trend_template_series(
    s
):

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


# ============================================================
# COMPUTE SIGNALS
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

    # ========================================================
    # RS LINE
    # ========================================================

    rs_ratio = (
        aligned["s"]
        /
        aligned["b"]
    )

    # ========================================================
    # RETURN FUNCTION
    # ========================================================

    def pct_return(
        series,
        days
    ):

        return (

            series
            /
            series.shift(days)
            - 1

        )

    # ========================================================
    # RAW RS SCORE
    # ========================================================

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

    # ========================================================
    # BLUE DOT
    # ========================================================

    previous_rs_high = (

        rs_ratio
        .shift(1)
        .rolling(
            LOOKBACK_DAYS
        )
        .max()

    )

    blue_dot = (

        rs_ratio
        >
        previous_rs_high

    )

    # ========================================================
    # PRICE TREND TEMPLATE
    # ========================================================

    tt_pass, tt_met = (
        trend_template_series(
            aligned["s"]
        )
    )

    # ========================================================
    # RS TREND TEMPLATE
    # ========================================================

    rs_tt_pass, rs_tt_met = (
        trend_template_series(
            rs_ratio
        )
    )

    # ========================================================
    # LIQUIDITY
    # ========================================================

    rolling_avg_volume = (

        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()

    )

    liquid = (

        aligned["s"]
        >=
        MIN_PRICE

    ) & (

        rolling_avg_volume
        >=
        MIN_AVG_VOLUME

    )

    # ========================================================
    # SIGNAL DATAFRAME
    # ========================================================

    out = pd.DataFrame({

        "price":
            aligned["s"],

        "rs_line":
            rs_ratio,

        "rs_score":
            rs_score,

        "blue_dot":
            blue_dot,

        "tt_pass":
            tt_pass,

        "rs_tt_pass":
            rs_tt_pass,

        "liquid":
            liquid,

    })

    # ========================================================
    # EMA DATA
    # ========================================================

    spans_needed = {

        v["span"]

        for v in EXIT_VARIANTS

        if "span" in v

    }

    for span in spans_needed:

        ema = (

            rs_ratio
            .ewm(
                span=span,
                adjust=False
            )
            .mean()

        )

        below = (
            rs_ratio <
            ema
        )

        previous_below = (
            below
            .shift(1)
            .fillna(False)
        )

        cross_below = (

            below
            &
            (~previous_below)

        )

        out[
            f"rs_below_ema{span}"
        ] = below

        out[
            f"rs_cross_below_ema{span}"
        ] = cross_below

    # ========================================================
    # TREND TEMPLATE FAIL CROSS
    # ========================================================

    previous_tt_pass = (

        tt_pass
        .shift(1)
        .fillna(False)

    )

    out[
        "tt_fail_cross"
    ] = (

        (~tt_pass)
        &
        previous_tt_pass

    )

    return out


# ============================================================
# RUN ONE VARIANT
# ============================================================

def run_backtest_for_variant(

    all_signals,

    trading_days,

    variant

):

    holdings = {}

    trade_log = []

    equity = 1.0

    equity_curve = []

    v_type = variant[
        "type"
    ]

    span = variant.get(
        "span"
    )

    buffer = variant.get(
        "buffer",
        RANK_EXIT
    )

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in trading_days:

        # ====================================================
        # BUILD ELIGIBLE POOL
        # ====================================================

        pool = []

        for sym, df in (
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
                row["blue_dot"]
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
                        row[
                            "rs_score"
                        ]
                    )
                )

            )

        # ====================================================
        # SORT BY RS SCORE
        # ====================================================

        pool.sort(

            key=lambda x: x[1],

            reverse=True

        )

        rank_lookup = {

            sym:
                i + 1

            for i, (
                sym,
                _
            )
            in enumerate(
                pool
            )

        }

        # ====================================================
        # TOP-10 ENTRY TARGET
        # ====================================================

        target_syms_topn = {

            sym

            for sym, _

            in pool[
                :TOP_N
            ]

        }

        target_prices = {}

        for sym in (
            target_syms_topn
        ):

            target_prices[sym] = (

                float(

                    all_signals[
                        sym
                    ]
                    .loc[
                        date,
                        "price"
                    ]

                )

            )

        # ====================================================
        # HOLDINGS BEFORE DECISIONS
        # ====================================================

        held_before_today = set(
            holdings.keys()
        )

        # ====================================================
        # DAILY RETURN
        # ====================================================

        if held_before_today:

            rets = []

            for sym in (
                held_before_today
            ):

                df = (
                    all_signals[
                        sym
                    ]
                )

                if date not in df.index:

                    continue

                idx = (
                    df.index
                    .get_loc(
                        date
                    )
                )

                if idx > 0:

                    prev_price = (

                        df[
                            "price"
                        ]
                        .iloc[
                            idx - 1
                        ]

                    )

                    curr_price = (

                        df[
                            "price"
                        ]
                        .iloc[
                            idx
                        ]

                    )

                    if (

                        pd.notna(
                            prev_price
                        )

                        and

                        pd.notna(
                            curr_price
                        )

                        and

                        prev_price > 0

                    ):

                        rets.append(

                            curr_price /
                            prev_price
                            - 1

                        )

            if rets:

                equity *= (

                    1 +
                    float(
                        np.mean(
                            rets
                        )
                    )

                )

        # ====================================================
        # EXIT LOGIC
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            df = (
                all_signals[
                    sym
                ]
            )

            if date not in df.index:

                continue

            row = df.loc[
                date
            ]

            # ------------------------------------------------
            # SECONDARY EXIT
            # ------------------------------------------------

            if (
                v_type ==
                "rank_buffer"
            ):

                exit_trigger = False

                reason = (
                    f"Rank > "
                    f"{buffer}"
                )

            elif (
                v_type ==
                "rs_ema_state"
            ):

                exit_trigger = bool(

                    row.get(

                        f"rs_below_ema{span}",

                        False

                    )

                )

                reason = (

                    f"RS Line < "
                    f"RS Line {span}-EMA"

                )

            elif (
                v_type ==
                "rs_ema_cross"
            ):

                exit_trigger = bool(

                    row.get(

                        f"rs_cross_below_ema{span}",

                        False

                    )

                )

                reason = (

                    f"RS Line crossed "
                    f"below {span}-EMA"

                )

            elif (
                v_type ==
                "tt_fail_cross"
            ):

                exit_trigger = bool(

                    row.get(

                        "tt_fail_cross",

                        False

                    )

                )

                reason = (
                    "Trend Template "
                    "PASS->FAIL"
                )

            else:

                exit_trigger = False

                reason = "N/A"

            # ------------------------------------------------
            # RANK EXIT
            # ------------------------------------------------

            if (
                v_type ==
                "rank_buffer"
            ):

                rank_exit_trigger = False

                current_rank = (
                    rank_lookup.get(
                        sym,
                        9999
                    )
                )

            else:

                current_rank = (
                    rank_lookup.get(
                        sym,
                        9999
                    )
                )

                rank_exit_trigger = (

                    current_rank
                    >
                    RANK_EXIT

                )

            # ------------------------------------------------
            # FINAL EXIT
            # ------------------------------------------------

            if (

                exit_trigger

                or

                rank_exit_trigger

            ):

                entry = (
                    holdings.pop(
                        sym
                    )
                )

                exit_price = float(
                    row["price"]
                )

                ret = (

                    exit_price /
                    entry[
                        "entry_price"
                    ]
                    - 1

                ) * 100

                if exit_trigger:

                    final_reason = reason

                else:

                    if (
                        current_rank
                        == 9999
                    ):

                        final_reason = (
                            "No longer in "
                            "eligible pool"
                        )

                    else:

                        final_reason = (

                            f"Rank "
                            f"{current_rank} "
                            f"> "
                            f"{RANK_EXIT}"

                        )

                trade_log.append({

                    "symbol":
                        sym,

                    "entry_date":
                        entry[
                            "entry_date"
                        ].strftime(
                            "%Y-%m-%d"
                        ),

                    "exit_date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "entry_price":
                        round(
                            entry[
                                "entry_price"
                            ],
                            2
                        ),

                    "exit_price":
                        round(
                            exit_price,
                            2
                        ),

                    "return_pct":
                        round(
                            ret,
                            2
                        ),

                    "days_held":
                        (
                            date -
                            entry[
                                "entry_date"
                            ]
                        ).days,

                    "exit_reason":
                        final_reason,

                    "exit_rank":
                        (
                            current_rank
                            if
                            current_rank
                            != 9999
                            else ""
                        ),

                })

        # ====================================================
        # ENTRIES
        # ====================================================

        slots_open = (

            TOP_N -
            len(holdings)

        )

        if slots_open > 0:

            for sym, _ in pool:

                if slots_open <= 0:

                    break

                if sym in holdings:

                    continue

                holdings[sym] = {

                    "entry_price":

                        float(

                            all_signals[
                                sym
                            ]
                            .loc[
                                date,
                                "price"
                            ]

                        ),

                    "entry_date":
                        date,

                }

                slots_open -= 1

        # ====================================================
        # EQUITY CURVE
        # ====================================================

        equity_curve.append({

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "equity":
                round(
                    equity,
                    6
                ),

            "n_holdings":
                len(holdings),

        })

    # ========================================================
    # CLOSE OPEN POSITIONS
    # ========================================================

    if trading_days.size:

        last_date = (
            trading_days[-1]
        )

        for sym, entry in (
            holdings.items()
        ):

            df = (
                all_signals[
                    sym
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

                    entry[
                        "entry_price"
                    ]

                )

            ret = (

                exit_price /
                entry[
                    "entry_price"
                ]
                - 1

            ) * 100

            trade_log.append({

                "symbol":
                    sym,

                "entry_date":
                    entry[
                        "entry_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "exit_date":
                    last_date.strftime(
                        "%Y-%m-%d"
                    ) + " (OPEN)",

                "entry_price":
                    round(
                        entry[
                            "entry_price"
                        ],
                        2
                    ),

                "exit_price":
                    round(
                        exit_price,
                        2
                    ),

                "return_pct":
                    round(
                        ret,
                        2
                    ),

                "days_held":
                    (
                        last_date -
                        entry[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "BACKTEST END",

                "exit_rank":
                    "",

            })

    return (

        pd.DataFrame(
            trade_log
        ),

        pd.DataFrame(
            equity_curve
        )

    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    trade_df,
    equity_df
):

    if equity_df.empty:

        return {}

    # ========================================================
    # TOTAL RETURN
    # ========================================================

    total_return_pct = round(

        (

            equity_df[
                "equity"
            ].iloc[-1]

            - 1

        ) * 100,

        2

    )

    # ========================================================
    # DRAW DOWN
    # ========================================================

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
        - 1

    ) * 100

    max_dd = round(
        drawdown.min(),
        2
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

        win_rate = round(

            (

                closed[
                    "return_pct"
                ] > 0

            ).mean() * 100,

            1

        )

        avg_return = round(

            closed[
                "return_pct"
            ].mean(),

            2

        )

        median_return = round(

            closed[
                "return_pct"
            ].median(),

            2

        )

        avg_days = round(

            closed[
                "days_held"
            ].mean(),

            1

        )

        median_days = round(

            closed[
                "days_held"
            ].median(),

            1

        )

        best = (
            closed[
                "return_pct"
            ].max()
        )

        worst = (
            closed[
                "return_pct"
            ].min()
        )

        winners = closed[
            closed[
                "return_pct"
            ] > 0
        ]

        losers = closed[
            closed[
                "return_pct"
            ] < 0
        ]

        if len(winners):

            avg_winner = round(

                winners[
                    "return_pct"
                ].mean(),

                2

            )

        else:

            avg_winner = 0

        if len(losers):

            avg_loser = round(

                losers[
                    "return_pct"
                ].mean(),

                2

            )

        else:

            avg_loser = 0

        gross_profit = (

            winners[
                "return_pct"
            ].sum()

            if len(winners)

            else 0

        )

        gross_loss = abs(

            losers[
                "return_pct"
            ].sum()

        ) if len(losers) else 0

        if gross_loss > 0:

            profit_factor = round(

                gross_profit /
                gross_loss,

                3

            )

        else:

            profit_factor = 0

    else:

        win_rate = 0

        avg_return = 0

        median_return = 0

        avg_days = 0

        median_days = 0

        best = 0

        worst = 0

        avg_winner = 0

        avg_loser = 0

        profit_factor = 0

    # ========================================================
    # DAILY EQUITY METRICS
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
                252 /
                max(
                    n_days,
                    1
                )
            )

            - 1

        )

        annualized_vol = (

            daily_std *
            np.sqrt(252)

        )

        if daily_std > 0:

            sharpe = (

                daily_mean /
                daily_std *
                np.sqrt(252)

            )

        else:

            sharpe = 0

        downside = (
            daily_returns[
                daily_returns < 0
            ]
        )

        if len(downside):

            downside_std = (
                downside.std()
            )

            if downside_std > 0:

                sortino = (

                    daily_mean /
                    downside_std *
                    np.sqrt(252)

                )

            else:

                sortino = 0

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

    if abs(max_dd) > 0:

        calmar = (

            annualized_return /
            abs(
                max_dd / 100
            )

        )

    else:

        calmar = 0

    return {

        "total_return_pct":
            total_return_pct,

        "annualized_return_pct":
            round(
                annualized_return *
                100,
                2
            ),

        "annualized_volatility_pct":
            round(
                annualized_vol *
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
            round(
                calmar,
                3
            ),

        "max_dd_pct":
            max_dd,

        "n_trades":
            n,

        "win_rate":
            win_rate,

        "avg_return_per_trade":
            avg_return,

        "median_return_per_trade":
            median_return,

        "avg_days_held":
            avg_days,

        "median_days_held":
            median_days,

        "avg_winner":
            avg_winner,

        "avg_loser":
            avg_loser,

        "profit_factor":
            profit_factor,

        "best_trade":
            best,

        "worst_trade":
            worst,

    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest():

    tickers = load_tickers()

    print(
        f"\nLoaded "
        f"{len(tickers)} tickers."
    )

    download_start, download_end = (
        get_download_dates()
    )

    print("=" * 65)

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
        f"Data cleaning        : "
        f"+/-"
        f"{MAX_PLAUSIBLE_DAILY_MOVE * 100:.0f}%"
    )

    print(
        f"Liquidity             : "
        f"price >= {MIN_PRICE}, "
        f"{VOLUME_LOOKBACK}d "
        f"avg volume >= "
        f"{MIN_AVG_VOLUME}"
    )

    print(
        f"Maximum holdings     : "
        f"{TOP_N}"
    )

    print(
        f"Rank exit            : "
        f"> {RANK_EXIT}"
    )

    print(
        f"PRIMARY RS EXIT      : "
        f"RS Line crosses below "
        f"{PRIMARY_RS_EMA_SPAN}-EMA"
    )

    print("=" * 65)

    # ========================================================
    # BENCHMARK
    # ========================================================

    bench_close = (
        download_benchmark()
    )

    # ========================================================
    # STOCK DOWNLOAD
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

                    all_signals[
                        symbol.replace(
                            ".NS",
                            ""
                        )
                    ] = sig

            except Exception as e:

                print(
                    f"Skipping "
                    f"{symbol}: "
                    f"{e}"
                )

                continue

        time.sleep(1)

    # ========================================================
    # SIGNAL SUMMARY
    # ========================================================

    print(
        f"\nSignals computed for "
        f"{len(all_signals)} stocks."
    )

    print(
        f"Total repaired data points: "
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
    # RUN VARIANTS
    # ========================================================

    comparison_rows = []

    primary_trades = (
        pd.DataFrame()
    )

    primary_equity = (
        pd.DataFrame()
    )

    for idx, variant in enumerate(
        EXIT_VARIANTS
    ):

        print(
            f"\nRunning variant: "
            f"{variant['name']}..."
        )

        trades, equity = (
            run_backtest_for_variant(

                all_signals,

                trading_days,

                variant

            )
        )

        summary = summarize(

            trades,

            equity

        )

        summary[
            "variant"
        ] = variant[
            "name"
        ]

        comparison_rows.append(
            summary
        )

        print(
            f"  Total Return: "
            f"{summary.get('total_return_pct')}%"
        )

        print(
            f"  CAGR: "
            f"{summary.get('annualized_return_pct')}%"
        )

        print(
            f"  Max DD: "
            f"{summary.get('max_dd_pct')}%"
        )

        print(
            f"  Trades: "
            f"{summary.get('n_trades')}"
        )

        print(
            f"  Win Rate: "
            f"{summary.get('win_rate')}%"
        )

        print(
            f"  Avg Hold: "
            f"{summary.get('avg_days_held')} days"
        )

        if (
            idx ==
            PRIMARY_VARIANT_INDEX
        ):

            primary_trades = (
                trades
            )

            primary_equity = (
                equity
            )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison_df = (

        pd.DataFrame(
            comparison_rows
        )

        [

            [

                "variant",

                "total_return_pct",

                "annualized_return_pct",

                "annualized_volatility_pct",

                "sharpe",

                "sortino",

                "calmar",

                "max_dd_pct",

                "n_trades",

                "win_rate",

                "avg_return_per_trade",

                "median_return_per_trade",

                "avg_days_held",

                "median_days_held",

                "avg_winner",

                "avg_loser",

                "profit_factor",

                "best_trade",

                "worst_trade",

            ]

        ]

    )

    # ========================================================
    # WRITE
    # ========================================================

    write_to_sheet(

        primary_trades,

        primary_equity,

        comparison_df

    )


# ============================================================
# WRITE RESULTS
# ============================================================

def write_to_sheet(

    trade_df,

    equity_df,

    comparison_df

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
            "Missing "
            "SHEET_ID/"
            "GOOGLE_CREDENTIALS "
            "-- saving CSV."
        )

        trade_df.to_csv(
            "backtest_trades.csv",
            index=False
        )

        equity_df.to_csv(
            "backtest_equity.csv",
            index=False
        )

        comparison_df.to_csv(
            "backtest_exit_comparison.csv",
            index=False
        )

        return

    # ========================================================
    # GOOGLE AUTH
    # ========================================================

    creds_dict = json.loads(
        creds_json
    )

    scopes = [

        "https://www.googleapis.com/"
        "auth/spreadsheets"

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
    # COMPARISON SHEET
    # ========================================================

    n_rows_needed = (
        len(comparison_df)
        + 10
    )

    try:

        cws = sh.worksheet(
            COMPARISON_WORKSHEET
        )

        if (
            cws.row_count
            <
            n_rows_needed
        ):

            cws.resize(

                rows=n_rows_needed,

                cols=25

            )

    except gspread.WorksheetNotFound:

        cws = sh.add_worksheet(

            title=
            COMPARISON_WORKSHEET,

            rows=n_rows_needed,

            cols=25

        )

    cws.clear()

    cws.update(

        [[

            f"Exit comparison | "
            f"{timestamp} | "
            f"GROSS | "
            f"Entry Top {TOP_N} | "
            f"Rank Exit > {RANK_EXIT} | "
            f"Primary RS {PRIMARY_RS_EMA_SPAN}-EMA"

        ]],

        "A1"

    )

    cws.update(

        [

            list(
                comparison_df.columns
            )

        ]

        +

        comparison_df.values.tolist(),

        "A3"

    )

    print(

        f"\nComparison written "
        f"to '{COMPARISON_WORKSHEET}'"

    )

    # ========================================================
    # PRIMARY BACKTEST SHEET
    # ========================================================

    n_rows_needed2 = max(

        len(trade_df)
        +
        len(equity_df)
        +
        50,

        100

    )

    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )

        if (
            ws.row_count
            <
            n_rows_needed2
        ):

            ws.resize(

                rows=n_rows_needed2,

                cols=15

            )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(

            title=
            BACKTEST_WORKSHEET,

            rows=n_rows_needed2,

            cols=15

        )

    ws.clear()

    primary_name = (
        EXIT_VARIANTS[
            PRIMARY_VARIANT_INDEX
        ][
            "name"
        ]
    )

    ws.update(

        [[

            f"PRIMARY: "
            f"{primary_name} | "
            f"Entry Top {TOP_N} | "
            f"Rank Exit > {RANK_EXIT} | "
            f"GROSS | "
            f"{timestamp}"

        ]],

        "A1"

    )

    # ========================================================
    # TRADE LOG
    # ========================================================

    ws.update(

        [["Trade Log"]],

        "A3"

    )

    if not trade_df.empty:

        ws.update(

            [

                list(
                    trade_df.columns
                )

            ]

            +

            trade_df.values.tolist(),

            "A4"

        )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity_start = (

        4
        +
        len(trade_df)
        +
        3

    )

    ws.update(

        [["Daily Equity Curve"]],

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

    print(

        f"Primary variant detail "
        f"written to "
        f"'{BACKTEST_WORKSHEET}'"

    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        run_backtest()

        print(
            "\nBACKTEST COMPLETED "
            "SUCCESSFULLY."
        )

    except Exception as e:

        print(
            "\nBACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: "
            f"{e}"
        )

        raise


# ============================================================
# AUDIT NOTES
# ============================================================
#
# PRIMARY SYSTEM:
#
#     Blue Dot
#     + Price TT 7/7
#     + RS TT 7/7
#     + Liquidity
#     + Top 10 RS Score
#
#     HOLD:
#         Rank 1-20
#
#     EXIT:
#         RS Line crosses below 5 EMA
#         OR rank >20
#
#
# EXAMPLE:
#
#     Stock enters at rank 4.
#
#     Later:
#
#         rank 8   -> HOLD
#         rank 11  -> HOLD
#         rank 14  -> HOLD
#         rank 18  -> HOLD
#         rank 20  -> HOLD
#         rank 21  -> EXIT
#
#     However, if RS Line crosses below 5 EMA
#     at rank 14:
#
#         EXIT immediately.
#
#
# IMPORTANT IMPLEMENTATION DETAIL:
#
# "Rank >20" means rank among stocks that currently satisfy:
#
#     - valid RS score
#     - liquidity
#     - Blue Dot
#     - Price TT 7/7
#     - RS TT 7/7
#
# If the stock fails one of these conditions and therefore disappears
# from rank_lookup, it receives an internal rank of 9999 and exits.
#
#
# MAXIMUM HOLDINGS:
#
#     10
#
# This does NOT become a 20-stock portfolio.
#
#
# PRIMARY VARIANT:
#
#     EXIT_VARIANTS[2]
#
#     RS<5EMA (crossover) + Rank20
#
#
# CROSSOVER DEFINITION:
#
# Today:
#
#     RS Line < 5 EMA
#
# Yesterday:
#
#     RS Line >= 5 EMA
#
# Therefore the exit occurs only on the actual
# above-to-below crossover day.
#
# It does NOT exit every subsequent day merely because
# RS remains below the 5 EMA.
#
#
# COSTS:
#
#     NOT INCLUDED.
#
# The equity curve is GROSS.
#
# Add later:
#
#     STT
#     stamp duty
#     exchange charges
#     SEBI charges
#     GST
#     DP charges
#     slippage
#     STCG
#
#
# SURVIVORSHIP:
#
#     NOT ELIMINATED.
#
# stocks.csv remains the historical universe.
#
#
# DATA CLEANING:
#
#     Single-day moves >30% are repaired by holding the previous
#     close flat.
#
# This is intentionally conservative but should be independently
# audited against corporate actions for every repaired observation.
#
# ============================================================