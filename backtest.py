"""
RS SCREENER BACKTEST
====================

IMPORTANT ARCHITECTURE
----------------------
This backtest DOES NOT re-implement the screening rules.

It imports the actual signal functions and configuration from:

    rs_screener.py

Therefore:

    LIVE SCREENER
          |
          v
    rs_screener.py
          ^
          |
    backtest.py

Both use the same:
    - RS Score
    - Price Trend Template
    - RS Line Trend Template
    - VCP
    - Volume Dry-up
    - Pivot
    - Breakout-volume requirement
    - Price filter
    - Liquidity filter
    - TOP_N
    - portfolio ranking logic

If a screening rule is changed in rs_screener.py,
the backtest automatically uses that changed rule.

CURRENT STRATEGY
----------------
Entry pool:

    Price > Rs.20
    20-day average volume > 100,000
    Price Trend Template = PASS
    RS Line Trend Template = PASS
    VCP = PASS
    Volume Dry-up = PASS
    Pivot proximity = PASS

    If REQUIRE_BREAKOUT_VOLUME=True:
        Breakout volume = PASS

Ranking:

    Raw RS Score descending

Portfolio:

    Top 10
    Equal weight
    Daily EOD rebalance

Exit:

    Stock leaves current Top 10

No:
    - Price stop
    - RS 5EMA exit
    - Rank buffer
    - Blue Dot entry
    - Green Dot entry
    - Regime filter

Costs:

    Same cost constants imported from rs_screener.py.

Tax:

    Same STCG calculation imported from rs_screener.py.

DATA
----
Yahoo Finance
Adjusted daily OHLCV.

The historical backtest intentionally does NOT apply the old
"implausible price jump cleaning" because the live screener does
not apply that transformation. Using it would make the historical
signal engine different from the live engine.

OUTPUT
------
If Google credentials are available:

    Backtest worksheet

Otherwise:

    backtest_trades.csv
    backtest_equity.csv
    backtest_open_positions.csv

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
# IMPORT THE ACTUAL LIVE SCREENER ENGINE
# ============================================================

from rs_screener import (
    BENCHMARK,
    BENCHMARK_FALLBACK,

    STOCKS_FILE,

    MIN_PRICE,
    MIN_AVG_VOLUME,
    VOLUME_LOOKBACK,

    TOP_N,

    REQUIRE_BREAKOUT_VOLUME,

    STT_RATE,
    STAMP_DUTY_RATE,
    EXCHANGE_CHARGE_RATE,
    SEBI_CHARGE_RATE,
    GST_RATE,
    DP_CHARGE_FLAT,

    STCG_RATE,
    STCG_CESS,
    STCG_EFFECTIVE_RATE,

    VCP_LOOKBACK,
    VCP_MIN_CONTRACTIONS,
    VCP_MAX_FINAL_CONTRACTION,
    VCP_MAX_BASE_DEPTH,
    VCP_CONTRACTION_IMPROVEMENT,
    VCP_VOLUME_DRYUP_RATIO,
    VCP_DRYUP_DAYS,
    PIVOT_LOOKBACK,
    PIVOT_PROXIMITY_PCT,
    BREAKOUT_VOLUME_MULTIPLIER,

    compute_rs_score,
    compute_trend_template,
    compute_volume_dryup,
    compute_vcp,
    compute_pivot,
    compute_rs_line_template,
    compute_diagnostics,
)


# ============================================================
# BACKTEST CONFIG
# ============================================================

BACKTEST_START = "2016-04-01"

# None = latest trading day available from Yahoo.
#
# Example:
# BACKTEST_END = "2026-08-13"
#
BACKTEST_END = None

# Need enough history BEFORE the backtest start for:
# 252-day RS score
# 273-day Trend Template
# 250-day diagnostics
# 60-day VCP
#
# Three years is deliberately conservative.
DOWNLOAD_YEARS_BEFORE_START = 3

STARTING_CAPITAL = 1_000_000

BACKTEST_WORKSHEET = "Backtest"

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BATCH_SIZE = 50


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
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():

    start = pd.Timestamp(BACKTEST_START)

    download_start = (
        start
        - pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime("%Y-%m-%d"),
            None
        )

    end = pd.Timestamp(BACKTEST_END)

    download_end = (
        end
        + pd.Timedelta(days=1)
    )

    return (
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d")
    )


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        f"\nBenchmark download:"
        f"\n  Start: {download_start}"
        f"\n  End  : {download_end}"
    )

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            kwargs = {
                "ticker": ticker,
                "start": download_start,
                "interval": "1d",
                "auto_adjust": True,
                "progress": False,
            }

            if download_end is not None:
                kwargs["end"] = download_end

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
                close = close.iloc[:, 0]

            close = (
                close
                .dropna()
                .sort_index()
            )

            if close.empty:
                continue

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


# ============================================================
# SAFE DATA EXTRACTION
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

            if (
                not isinstance(
                    data.columns,
                    pd.MultiIndex
                )
            ):
                return None

            level0 = (
                data.columns
                .get_level_values(0)
            )

            if symbol not in level0:
                return None

            sdata = data[symbol]

        if "Close" not in sdata.columns:
            return None

        close = (
            sdata["Close"]
            .dropna()
            .sort_index()
        )

        if close.empty:
            return None

        if "Volume" in sdata.columns:

            volume = (
                sdata["Volume"]
                .reindex(close.index)
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
# VECTORISED RS SCORE
# ============================================================

def compute_rs_score_series(close):

    """
    This reproduces the actual screener's
    compute_rs_score() semantics.

    IMPORTANT:

    rs_screener.py does:

        past = close.iloc[-days - 1]

    Therefore the historical series equivalent is:

        shift(days + 1)

    NOT shift(days).

    This fixes the old backtest's hidden RS mismatch.
    """

    p3 = (
        close
        / close.shift(64)
        - 1
    )

    p6 = (
        close
        / close.shift(127)
        - 1
    )

    p9 = (
        close
        / close.shift(190)
        - 1
    )

    p12 = (
        close
        / close.shift(253)
        - 1
    )

    score = (
        0.40 * p3
        + 0.20 * p6
        + 0.20 * p9
        + 0.20 * p12
    ) * 100

    return score


# ============================================================
# VECTORISED TREND TEMPLATE
# ============================================================

def compute_trend_template_series(series):

    """
    Vectorised equivalent of the ACTUAL
    compute_trend_template() function in rs_screener.py.

    It uses the same seven conditions:

        Price > SMA150 and SMA200
        SMA150 > SMA200
        SMA200 rising vs 1 month ago
        SMA50 > SMA150 and SMA200
        Price > SMA50
        Price >= 125% of 52-week low
        Price >= 75% of 52-week high

    The live function requires at least 273 observations.
    """

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

    sma200_series = (
        series
        .rolling(200)
        .mean()
    )

    sma200 = (
        sma200_series
    )

    sma200_1mo = (
        sma200_series
        .shift(21)
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
        sma200 > sma200_1mo
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
        series >= 1.25 * low52
    )

    c7 = (
        series >= 0.75 * high52
    )

    result = (
        c1
        & c2
        & c3
        & c4
        & c5
        & c6
        & c7
    )

    # Match live screener behaviour:
    # fewer than 273 observations = None,
    # which means "not usable".
    result = result.astype("object")

    result.iloc[:273] = None

    return result


# ============================================================
# LIQUIDITY
# ============================================================

def compute_liquidity_series(
    close,
    volume
):

    avg20 = (
        volume
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    return (
        (close > MIN_PRICE)
        &
        (avg20 > MIN_AVG_VOLUME)
    )


# ============================================================
# RS LINE
# ============================================================

def compute_rs_line_series(
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
        "stock",
        "bench"
    ]

    rs_line = (
        aligned["stock"]
        /
        aligned["bench"]
    )

    rs_tt = (
        compute_trend_template_series(
            rs_line
        )
    )

    return (
        rs_line,
        rs_tt
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def compute_diagnostic_series(
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
        "stock",
        "bench"
    ]

    rs_ratio = (
        aligned["stock"]
        /
        aligned["bench"]
    )

    previous_rs_high = (
        rs_ratio
        .shift(1)
        .rolling(250)
        .max()
    )

    blue_dot = (
        rs_ratio
        >
        previous_rs_high
    )

    previous_price_high = (
        aligned["stock"]
        .shift(1)
        .rolling(250)
        .max()
    )

    price_at_new_high = (
        aligned["stock"]
        >
        previous_price_high
    )

    green_dot = (
        blue_dot
        &
        (~price_at_new_high)
    )

    return (
        blue_dot,
        blue_dot.copy(),
        green_dot
    )


# ============================================================
# EXACT DAILY SETUP CHECK
# ============================================================

def evaluate_day(
    close,
    volume,
    bench_close,
    date
):

    """
    Evaluate ONE historical date.

    All non-vectorised setup logic is taken directly from
    rs_screener.py.

    This is the critical synchronization point.
    """

    close_to_date = (
        close.loc[
            :date
        ]
    )

    volume_to_date = (
        volume.loc[
            :date
        ]
    )

    bench_to_date = (
        bench_close.loc[
            :date
        ]
    )

    if len(close_to_date) < 273:
        return None

    if len(volume_to_date) < 60:
        return None

    if len(bench_to_date) < 273:
        return None

    price = float(
        close_to_date.iloc[-1]
    )

    # --------------------------------------------------------
    # PRICE FILTER
    # --------------------------------------------------------

    if price <= MIN_PRICE:
        return None

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    avg20_volume = (
        volume_to_date
        .tail(VOLUME_LOOKBACK)
        .mean()
    )

    if (
        pd.isna(avg20_volume)
        or
        avg20_volume <= MIN_AVG_VOLUME
    ):
        return None

    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

    rs_score = (
        compute_rs_score(
            close_to_date
        )
    )

    if rs_score is None:
        return None

    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
    # --------------------------------------------------------

    tt_pass = (
        compute_trend_template(
            close_to_date
        )
    )

    if tt_pass is None:
        return None

    # --------------------------------------------------------
    # RS LINE TREND TEMPLATE
    # --------------------------------------------------------

    rs_tt_pass = (
        compute_rs_line_template(
            close_to_date,
            bench_to_date
        )
    )

    if rs_tt_pass is None:
        return None

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    diagnostics = (
        compute_diagnostics(
            close_to_date,
            bench_to_date
        )
    )

    if diagnostics is None:
        return None

    blue_dot, green_dot = diagnostics

    # --------------------------------------------------------
    # VCP
    # --------------------------------------------------------

    vcp = (
        compute_vcp(
            close_to_date,
            volume_to_date
        )
    )

    if vcp is None:
        return None

    # --------------------------------------------------------
    # VOLUME DRY-UP
    # --------------------------------------------------------

    dryup = (
        compute_volume_dryup(
            volume_to_date
        )
    )

    if dryup is None:
        return None

    # --------------------------------------------------------
    # PIVOT
    # --------------------------------------------------------

    pivot_data = (
        compute_pivot(
            close_to_date,
            volume_to_date
        )
    )

    if pivot_data is None:
        return None

    # --------------------------------------------------------
    # FINAL SCREEN
    #
    # This is copied conceptually from the live screener:
    #
    # Price TT
    # + RS TT
    # + VCP
    # + Dry-up
    # + Pivot proximity
    # + optional breakout volume
    # --------------------------------------------------------

    core_screen_pass = (
        tt_pass is True
        and
        rs_tt_pass is True
        and
        vcp["pass"] is True
        and
        dryup["pass"] is True
        and
        pivot_data[
            "proximity_pass"
        ] is True
    )

    if REQUIRE_BREAKOUT_VOLUME:

        screen_pass = (
            core_screen_pass
            and
            pivot_data[
                "breakout_pass"
            ]
        )

    else:

        screen_pass = (
            core_screen_pass
        )

    return {
        "price": price,

        "avg20_volume":
            float(avg20_volume),

        "rs_score":
            float(rs_score),

        "tt_pass":
            bool(tt_pass),

        "rs_tt_pass":
            bool(rs_tt_pass),

        "vcp_pass":
            bool(vcp["pass"]),

        "volume_dryup_pass":
            bool(dryup["pass"]),

        "pivot_proximity_pass":
            bool(
                pivot_data[
                    "proximity_pass"
                ]
            ),

        "breakout_volume_pass":
            bool(
                pivot_data[
                    "breakout_pass"
                ]
            ),

        "screen_pass":
            bool(screen_pass),

        "blue_dot":
            bool(blue_dot),

        "one_year_rs_cross":
            bool(blue_dot),

        "green_dot":
            bool(green_dot),
    }


# ============================================================
# PRECOMPUTE HISTORICAL SIGNALS
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
    bench_close,
    trading_days
):

    """
    Compute the complete historical signal dataframe.

    Simple calculations are vectorised.

    VCP / dry-up / pivot use the exact live screener
    functions and are evaluated only on candidate days.
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
        "stock",
        "bench"
    ]

    close = aligned["stock"]

    bench_close = aligned["bench"]

    volume = (
        volume
        .reindex(close.index)
        .fillna(0)
    )

    if len(close) < 273:
        return None

    # --------------------------------------------------------
    # VECTORISED CORE SIGNALS
    # --------------------------------------------------------

    rs_score = (
        compute_rs_score_series(
            close
        )
    )

    tt_pass = (
        compute_trend_template_series(
            close
        )
    )

    rs_ratio = (
        close
        /
        bench_close
    )

    rs_tt_pass = (
        compute_trend_template_series(
            rs_ratio
        )
    )

    avg20_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (
        (close > MIN_PRICE)
        &
        (avg20_volume > MIN_AVG_VOLUME)
    )

    # Diagnostics
    previous_rs_high = (
        rs_ratio
        .shift(1)
        .rolling(250)
        .max()
    )

    blue_dot = (
        rs_ratio
        >
        previous_rs_high
    )

    previous_price_high = (
        close
        .shift(1)
        .rolling(250)
        .max()
    )

    price_at_new_high = (
        close
        >
        previous_price_high
    )

    green_dot = (
        blue_dot
        &
        (~price_at_new_high)
    )

    # --------------------------------------------------------
    # RESULT FRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        index=close.index
    )

    result["price"] = close

    result["rs_score"] = rs_score

    result["tt_pass"] = tt_pass

    result["rs_tt_pass"] = rs_tt_pass

    result["liquid"] = liquid

    result["blue_dot"] = blue_dot

    result["one_year_rs_cross"] = blue_dot

    result["green_dot"] = green_dot

    # Default hard-screen fields
    result["vcp_pass"] = False
    result["volume_dryup_pass"] = False
    result["pivot_proximity_pass"] = False
    result["breakout_volume_pass"] = False
    result["screen_pass"] = False

    # --------------------------------------------------------
    # CANDIDATE DAYS
    #
    # Do expensive VCP calculations only when:
    #
    # price filter
    # liquidity
    # Price TT
    # RS TT
    #
    # already pass.
    # --------------------------------------------------------

    candidate_mask = (
        liquid
        &
        (tt_pass == True)
        &
        (rs_tt_pass == True)
        &
        rs_score.notna()
    )

    candidate_dates = (
        result.index[
            candidate_mask
        ]
    )

    print(
        f"    Candidate setup days: "
        f"{len(candidate_dates)}"
    )

    # --------------------------------------------------------
    # EXACT LIVE FUNCTIONS
    # --------------------------------------------------------

    for date in candidate_dates:

        setup = evaluate_day(
            close,
            volume,
            bench_close,
            date
        )

        if setup is None:
            continue

        result.loc[
            date,
            "vcp_pass"
        ] = setup[
            "vcp_pass"
        ]

        result.loc[
            date,
            "volume_dryup_pass"
        ] = setup[
            "volume_dryup_pass"
        ]

        result.loc[
            date,
            "pivot_proximity_pass"
        ] = setup[
            "pivot_proximity_pass"
        ]

        result.loc[
            date,
            "breakout_volume_pass"
        ] = setup[
            "breakout_volume_pass"
        ]

        result.loc[
            date,
            "screen_pass"
        ] = setup[
            "screen_pass"
        ]

    return result


# ============================================================
# TRANSACTION COSTS
# ============================================================

def buy_side_cost(
    trade_value
):

    stt = (
        STT_RATE
        *
        trade_value
    )

    stamp = (
        STAMP_DUTY_RATE
        *
        trade_value
    )

    exch = (
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
        (exch + sebi)
    )

    return (
        stt
        + stamp
        + exch
        + sebi
        + gst
    )


def sell_side_cost(
    trade_value
):

    stt = (
        STT_RATE
        *
        trade_value
    )

    exch = (
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
        (exch + sebi)
    )

    return (
        stt
        + exch
        + sebi
        + gst
        + DP_CHARGE_FLAT
    )


def stcg_tax(
    net_gain
):

    if net_gain <= 0:
        return 0.0

    return (
        net_gain
        *
        STCG_EFFECTIVE_RATE
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    """
    PORTFOLIO RULES MATCH LIVE SCREENER

    Entry pool:

        screen_pass == True

    Ranking:

        raw RS Score descending

    Portfolio:

        Top 10

    Weight:

        Equal weight

    Exit:

        Leaves current Top 10

    Rebalance:

        Daily EOD

    No other exit.
    """

    cash = (
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    for date in trading_days:

        # ====================================================
        # BUILD TODAY'S EXACT LIVE SCREEN
        # ====================================================

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
                row["screen_pass"]
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

        # ====================================================
        # RANK
        # ====================================================

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        target_top10 = {
            sym
            for sym, _ in pool[:TOP_N]
        }

        # ====================================================
        # EXITS
        # ====================================================

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

            cash += (
                net_proceeds
                -
                tax
            )

            gross_return_pct = (
                (
                    exit_price
                    /
                    pos["entry_price"]
                    - 1
                )
                * 100
            )

            net_return_pct = (
                (
                    net_gain
                    -
                    tax
                )
                /
                cost_basis
                * 100
                if cost_basis > 0
                else 0
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
                        date
                        -
                        pos[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "Left Top 10",
            })

        # ====================================================
        # PORTFOLIO VALUE AFTER EXITS
        # ====================================================

        portfolio_value = cash

        for sym, pos in holdings.items():

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
                    pos[
                        "entry_price"
                    ]
                )

            portfolio_value += (
                pos["qty"]
                *
                price
            )

        # ====================================================
        # NEW ENTRIES
        # ====================================================

        slots_open = (
            TOP_N
            -
            len(holdings)
        )

        if slots_open > 0:

            slot_capital = (
                portfolio_value
                /
                TOP_N
            )

            for sym, _ in pool[:TOP_N]:

                if slots_open <= 0:
                    break

                if sym in holdings:
                    continue

                df = all_signals[
                    sym
                ]

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

                holdings[sym] = {

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
        # DAILY MARK-TO-MARKET
        # ====================================================

        portfolio_value = cash

        for sym, pos in holdings.items():

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
                    pos[
                        "entry_price"
                    ]
                )

            portfolio_value += (
                pos["qty"]
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
            equity_curve[-1]
            [
                "portfolio_value_rs"
            ]
        )

    else:

        final_marked_value = (
            STARTING_CAPITAL
        )

    liquidation_cash = cash

    open_positions_detail = []

    if (
        len(trading_days)
        and holdings
    ):

        last_date = (
            trading_days[-1]
        )

        for sym, pos in holdings.items():

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
                    pos[
                        "entry_price"
                    ]
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

            liquidation_cash += (
                net_proceeds
                -
                tax
            )

            open_positions_detail.append({

                "symbol":
                    sym,

                "entry_date":
                    pos[
                        "entry_date"
                    ].strftime(
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

    # ========================================================
    # DRAWDOWN
    # ========================================================

    if not equity_df.empty:

        running_max = (
            equity_df[
                "portfolio_value_rs"
            ]
            .cummax()
        )

        equity_df[
            "drawdown_pct"
        ] = (
            (
                equity_df[
                    "portfolio_value_rs"
                ]
                /
                running_max
                - 1
            )
            *
            100
        )

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

    marked_return = (
        (
            final_marked_value
            /
            STARTING_CAPITAL
            - 1
        )
        *
        100
    )

    liquidation_return = (
        (
            final_liquidation_value
            /
            STARTING_CAPITAL
            - 1
        )
        *
        100
    )

    if not equity_df.empty:

        max_dd = (
            equity_df[
                "drawdown_pct"
            ]
            .min()
        )

        daily_returns = (
            equity_df[
                "portfolio_value_rs"
            ]
            .pct_change()
            .dropna()
        )

    else:

        max_dd = 0

        daily_returns = pd.Series(
            dtype=float
        )

    closed = (
        trade_df
        if not trade_df.empty
        else pd.DataFrame()
    )

    if not closed.empty:

        win_rate_gross = (
            (
                closed[
                    "gross_return_pct"
                ]
                > 0
            )
            .mean()
            * 100
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

        winners = (
            closed[
                closed[
                    "net_return_pct"
                ]
                > 0
            ]
        )

        losers = (
            closed[
                closed[
                    "net_return_pct"
                ]
                < 0
            ]
        )

        avg_winner = (
            winners[
                "net_return_pct"
            ].mean()
            if not winners.empty
            else 0
        )

        avg_loser = (
            losers[
                "net_return_pct"
            ].mean()
            if not losers.empty
            else 0
        )

        gross_profit = (
            winners[
                "net_pnl_rs"
            ].sum()
            if not winners.empty
            else 0
        )

        gross_loss = (
            abs(
                losers[
                    "net_pnl_rs"
                ].sum()
            )
            if not losers.empty
            else 0
        )

        profit_factor = (
            gross_profit
            /
            gross_loss
            if gross_loss > 0
            else 0
        )

    else:

        win_rate_gross = 0
        win_rate_net = 0
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

        if (
            n_days > 0
            and final_marked_value > 0
        ):

            annualized_return = (
                (
                    final_marked_value
                    /
                    STARTING_CAPITAL
                )
                **
                (
                    252
                    /
                    n_days
                )
                - 1
            )

        else:

            annualized_return = 0

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
                "Price TT + RS TT + "
                "VCP + Volume Dry-up + "
                "Pivot Proximity"
            ),

        "Breakout Volume Required":
            REQUIRE_BREAKOUT_VOLUME,

        "Ranking":
            "Raw RS Score descending",

        "Portfolio":
            f"Top {TOP_N}, equal weight",

        "Exit":
            f"Leaves Top {TOP_N}",

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
            len(closed),

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
# GOOGLE SHEETS HELPERS
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
                f"{label} rows "
                f"{i}-{i + len(chunk)}: "
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

            if (
                sheet[
                    "properties"
                ][
                    "sheetId"
                ]
                != sheet_id
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
                f"existing chart(s)."
            )

    except Exception as e:

        print(
            "Could not remove charts: "
            f"{e}"
        )


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
            "Equity and drawdown charts added."
        )

    except Exception as e:

        print(
            f"Could not add charts: {e}"
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

    sheet_id = (
        os.environ.get(
            SHEET_ID_ENV
        )
    )

    creds_json = (
        os.environ.get(
            CREDS_ENV
        )
    )

    if (
        not sheet_id
        or not creds_json
    ):

        print(
            "Missing "
            "SHEET_ID/GOOGLE_CREDENTIALS."
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

    gc = (
        gspread
        .authorize(creds)
    )

    sh = (
        gc
        .open_by_key(sheet_id)
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    summary_rows = (
        [["Summary", ""]]
        +
        [
            [k, v]
            for k, v
            in summary.items()
        ]
    )

    n_rows_needed = (
        len(trade_df)
        +
        len(equity_df)
        +
        len(open_df)
        +
        len(summary_rows)
        +
        80
    )

    n_cols_needed = 15

    try:

        ws = (
            sh
            .worksheet(
                BACKTEST_WORKSHEET
            )
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

        ws = (
            sh
            .add_worksheet(

                title=
                    BACKTEST_WORKSHEET,

                rows=
                    n_rows_needed,

                cols=
                    n_cols_needed
            )
        )

    remove_existing_charts(
        sh,
        ws.id
    )

    ws.clear()

    # ========================================================
    # HEADER
    # ========================================================

    ws.update(

        [[

            "LIVE SCREENER ENGINE BACKTEST"
            f" | Run {timestamp}"
            " | NET of costs + STCG"
            f" | Starting capital: "
            f"Rs.{STARTING_CAPITAL:,.0f}"
            f" | Window: "
            f"{BACKTEST_START}"
            f" to "
            f"{effective_end_str}"
            " | VCP + Dry-up + Pivot"

        ]],

        "A1"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    ws.update(
        summary_rows,
        "A3"
    )

    # ========================================================
    # TRADE LOG
    # ========================================================

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

        rows = (
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

            rows,

            start_row=
                trade_header_row,

            chunk_size=
                2000,

            label=
                "trade log"
        )

    # ========================================================
    # OPEN POSITIONS
    # ========================================================

    open_start_row = (
        trade_header_row
        +
        len(trade_df)
        +
        3
    )

    ws.update(

        [[
            "Open Positions at Backtest End "
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

        rows = (
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

            rows,

            start_row=
                open_header_row,

            chunk_size=
                2000,

            label=
                "open positions"
        )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

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

        rows = (
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

            rows,

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
        "\nBacktest results written to "
        f"'{BACKTEST_WORKSHEET}' tab."
    )

    print(
        f"Trades: {len(trade_df)}"
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

    print(
        "\n"
        "=" * 70
    )

    print(
        "RS SCREENER — SYNCHRONIZED BACKTEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nSignal engine:"
        "\n  rs_screener.py"
    )

    print(
        "\nEntry:"
        "\n  Price > Rs.20"
        f"\n  {VOLUME_LOOKBACK}D avg volume > "
        f"{MIN_AVG_VOLUME:,}"
        "\n  Price Trend Template"
        "\n  RS Line Trend Template"
        "\n  VCP"
        "\n  Volume Dry-up"
        "\n  Pivot Proximity"
        f"\n  Breakout volume required: "
        f"{REQUIRE_BREAKOUT_VOLUME}"
    )

    print(
        "\nPortfolio:"
        f"\n  Top {TOP_N}"
        "\n  Equal weight"
        "\n  Daily EOD rebalance"
        f"\n  Exit = leaves Top {TOP_N}"
    )

    print(
        f"\nStarting capital:"
        f" Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        f"\nBacktest start:"
        f" {BACKTEST_START}"
    )

    print(
        f"Backtest end:"
        f" {BACKTEST_END if BACKTEST_END else 'latest available'}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # LOAD UNIVERSE
    # ========================================================

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    # ========================================================
    # DOWNLOAD BENCHMARK
    # ========================================================

    bench_close = (
        download_benchmark()
    )

    download_start, download_end = (
        get_download_dates()
    )

    # ========================================================
    # HISTORICAL SIGNALS
    # ========================================================

    all_signals = {}

    batch_size = BATCH_SIZE

    total_candidates = 0

    total_final_screen_days = 0

    for i in range(
        0,
        len(tickers),
        batch_size
    ):

        batch = tickers[
            i:i + batch_size
        ]

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Downloading batch "
            f"{i}"
            f"-"
            f"{i + len(batch)}"
        )

        print(
            "=" * 60
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

                kwargs[
                    "end"
                ] = download_end

            data = yf.download(
                **kwargs
            )

        except Exception as e:

            print(
                f"Batch download failed: "
                f"{e}"
            )

            continue

        for symbol in batch:

            try:

                extracted = (
                    extract_symbol_data(
                        data,
                        symbol,
                        len(batch)
                    )
                )

                if extracted is None:
                    continue

                close, volume = (
                    extracted
                )

                if len(close) < 273:
                    continue

                print(
                    f"\nProcessing "
                    f"{symbol}"
                )

                sig = (
                    compute_signals_for_stock(

                        close,

                        volume,

                        bench_close,

                        bench_close.index
                    )
                )

                if sig is None:
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
                ] = sig

                candidates = (
                    (
                        sig[
                            "liquid"
                        ]
                        &
                        (
                            sig[
                                "tt_pass"
                            ]
                            == True
                        )
                        &
                        (
                            sig[
                                "rs_tt_pass"
                            ]
                            == True
                        )
                    )
                    .sum()
                )

                final_days = (
                    sig[
                        "screen_pass"
                    ]
                    .sum()
                )

                total_candidates += (
                    int(candidates)
                )

                total_final_screen_days += (
                    int(final_days)
                )

                print(
                    f"  Candidate days: "
                    f"{candidates}"
                )

                print(
                    f"  Final screen days: "
                    f"{final_days}"
                )

            except Exception as e:

                print(
                    f"Skipping "
                    f"{symbol}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                continue

        time.sleep(1)

    # ========================================================
    # SIGNAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SIGNAL CALCULATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Stocks with signals: "
        f"{len(all_signals)}"
    )

    print(
        f"Candidate setup days: "
        f"{total_candidates:,}"
    )

    print(
        f"Final screen-pass days: "
        f"{total_final_screen_days:,}"
    )

    # ========================================================
    # EFFECTIVE END DATE
    # ========================================================

    if BACKTEST_END is not None:

        effective_end = (
            pd.Timestamp(
                BACKTEST_END
            )
        )

    else:

        effective_end = (
            bench_close.index.max()
        )

    print(
        f"\nEffective end date: "
        f"{effective_end.strftime('%Y-%m-%d')}"
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
                effective_end
            )
        ]
    )

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    # ========================================================
    # RUN
    # ========================================================

    (
        trade_df,
        equity_df,
        open_df,
        final_marked,
        final_liquidation
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

        final_marked,

        final_liquidation
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL BACKTEST SUMMARY"
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

    print(
        "=" * 70
    )

    # ========================================================
    # WRITE
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
        "\nBACKTEST COMPLETED SUCCESSFULLY."
    )


# ============================================================
# RUN
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