"""
RS SCREENER - MINERVINI TREND TEMPLATE + RS + VCP

============================================================
CORE SCREEN
============================================================

PRICE
------
Price > Rs.20

LIQUIDITY
---------
20-day average volume > 100,000 shares

PRICE TREND TEMPLATE
--------------------
7/7 Minervini Trend Template

RS
--
RS Score:
    40% 3-month return
    20% 6-month return
    20% 9-month return
    20% 12-month return

RS Line Trend Template
----------------------
Same 7/7 Trend Template logic applied to
relative-strength line = Stock / Benchmark

VCP
---
Approximate mechanical VCP detection:

1. Recent base is examined.
2. At least 3 contraction legs are required.
3. Each later contraction must be smaller than the prior contraction.
4. Price must remain above the 200 DMA.
5. Recent base must remain reasonably tight.
6. Volume must contract toward the pivot.

VOLUME DRY-UP
-------------
Recent volume must be materially below the
50-day average volume.

PIVOT
-----
Pivot = highest high in the recent base/pivot window.

Pivot proximity:
    price must be within PIVOT_PROXIMITY_PCT of pivot.

BREAKOUT VOLUME
---------------
Diagnostic by default.

Breakout volume PASS when:
    current volume >= BREAKOUT_VOLUME_MULTIPLIER
    * 50-day average volume

IMPORTANT:
-----------
Breakout volume is NOT required for entry by default.

Set:
    REQUIRE_BREAKOUT_VOLUME = True

if you want only stocks that have already broken out
with volume.

RANKING
-------
Eligible stocks ranked by raw RS Score descending.

PORTFOLIO
---------
Top 10
Equal weight
Daily EOD rebalance

EXIT
----
Leaves current Top 10.

NO:
----
RS < 5EMA exit
Price stop
Rank buffer
Blue Dot entry
Green Dot entry
Regime filter

Blue Dot / 1Y RS Cross / Green Dot remain diagnostic.
"""

import time
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
HISTORY_PERIOD = "15mo"

STOCKS_FILE = "stocks.csv"

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

WORKSHEET_NAME = "RS_Screener"

# ---------------- PRICE / LIQUIDITY ----------------

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20

# ---------------- PORTFOLIO ----------------

TOP_N = 10

INTRADAY_INTERVAL = "5m"

# ============================================================
# VCP CONFIGURATION
# ============================================================

# Number of trading days examined for the VCP base.
VCP_LOOKBACK = 60

# Minimum number of contraction legs required.
VCP_MIN_CONTRACTIONS = 3

# Maximum acceptable final contraction depth.
# Example: 0.12 = 12%
VCP_MAX_FINAL_CONTRACTION = 0.12

# Maximum total width of the recent base.
# Example: 0.25 = 25%
VCP_MAX_BASE_DEPTH = 0.25

# Minimum improvement required between contractions.
#
# Example:
# contraction 1 = 20%
# contraction 2 = 13%
# contraction 3 = 8%
#
# Every later contraction must be <= previous contraction * 0.85
#
VCP_CONTRACTION_IMPROVEMENT = 0.85

# Volume dry-up:
# recent average volume must be <= this multiple of
# 50-day average volume.
#
# 0.75 means recent volume <= 75% of 50D average.
VCP_VOLUME_DRYUP_RATIO = 0.75

# Number of recent sessions used for dry-up calculation.
VCP_DRYUP_DAYS = 10

# Pivot window.
PIVOT_LOOKBACK = 20

# Price must be within this distance of pivot.
#
# 0.05 = 5%
#
# Allows price up to 5% below pivot and 5% above pivot.
PIVOT_PROXIMITY_PCT = 0.05

# Breakout volume confirmation.
BREAKOUT_VOLUME_MULTIPLIER = 1.50

# IMPORTANT:
#
# False = VCP setup + dry-up + pivot proximity are enough.
#
# True = additionally require current volume >= 1.5x
# 50-day average volume.
#
# Keep FALSE for a setup screener.
REQUIRE_BREAKOUT_VOLUME = False


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

STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)


# ============================================================
# WORKSHEETS
# ============================================================

EOD_CRON = "15 11 * * 1-5"

HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"


PORTFOLIO_HEADER = [
    "Action",
    "Executed",
    "Execution Price",
    "Symbol",
    "Rank",
    "Entry Price",
    "Entry Date",
    "Current Price",
    "Qty",
    "Position Value (Rs)",
    "P&L %",
    "Buy Cost (Rs)",
    "Sell Cost (Rs)",
    "Est. STCG Tax (Rs)",
    "Blue Dot",
    "1Y RS Cross",
    "Green Dot",
]


# ============================================================
# RUN MODE
# ============================================================

def get_run_mode():

    event = os.environ.get("GITHUB_EVENT_NAME", "manual")

    force_eod = (
        os.environ.get("FORCE_EOD", "false")
        .strip()
        .lower()
        == "true"
    )

    if event == "schedule":

        triggering_cron = (
            os.environ.get("SCHEDULE_CRON", "")
            .strip()
        )

        return (
            "EOD"
            if triggering_cron == EOD_CRON
            else "PREVIEW"
        )

    return "EOD" if force_eod else "PREVIEW"


# ============================================================
# LOAD TICKERS
# ============================================================

def load_tickers():

    df = pd.read_csv(STOCKS_FILE)

    symbols = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return [
        s if s.endswith(".NS") else s + ".NS"
        for s in symbols
    ]


# ============================================================
# COSTS
# ============================================================

def buy_side_cost(trade_value):

    stt = STT_RATE * trade_value
    stamp = STAMP_DUTY_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return stt + stamp + exch + sebi + gst


def sell_side_cost(trade_value):

    stt = STT_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return stt + exch + sebi + gst + DP_CHARGE_FLAT


def estimate_stcg(gross_pnl):

    if gross_pnl <= 0:
        return 0.0

    return gross_pnl * STCG_EFFECTIVE_RATE


# ============================================================
# BENCHMARK
# ============================================================

def download_benchmark(run_mode):

    for tkr in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            data = yf.download(
                tkr,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if not data.empty:

                print(
                    f"Benchmark loaded: {tkr}"
                )

                close = data["Close"]

                if run_mode == "PREVIEW":

                    live = fetch_intraday_last_price(
                        [tkr]
                    )

                    close = append_preview_price(
                        close,
                        live.get(tkr)
                    )

                return close

        except Exception as e:

            print(
                f"Benchmark {tkr} failed: {e}"
            )

    raise RuntimeError(
        "Could not download any benchmark index data."
    )


# ============================================================
# INTRADAY PRICE
# ============================================================

def fetch_intraday_last_price(tickers):

    prices = {}

    try:

        data = yf.download(
            tickers,
            period="1d",
            interval=INTRADAY_INTERVAL,
            progress=False,
            group_by="ticker",
            threads=True
        )

        for tkr in tickers:

            try:

                sdata = (
                    data
                    if len(tickers) == 1
                    else data[tkr]
                )

                last_valid = (
                    sdata["Close"]
                    .dropna()
                )

                if not last_valid.empty:

                    prices[tkr] = float(
                        last_valid.iloc[-1]
                    )

            except Exception:

                continue

    except Exception as e:

        print(
            f"Intraday batch fetch failed: {e}"
        )

    return prices


def append_preview_price(
    close_series,
    live_price
):

    if live_price is None:
        return close_series

    today = (
        pd.Timestamp.now(
            tz=close_series.index.tz
        ).normalize()
        if close_series.index.tz
        else pd.Timestamp.now().normalize()
    )

    last_date = (
        close_series.index[-1].normalize()
    )

    if last_date == today:

        updated = close_series.copy()

        updated.iloc[-1] = live_price

        return updated

    new_point = pd.Series(
        [live_price],
        index=[today]
    )

    return pd.concat(
        [close_series, new_point]
    )


# ============================================================
# RS SCORE
# ============================================================

def compute_rs_score(close_series):

    """
    40% 3M
    20% 6M
    20% 9M
    20% 12M
    """

    n = len(close_series)

    periods = {
        "P3": 63,
        "P6": 126,
        "P9": 189,
        "P12": 252,
    }

    returns = {}

    for label, days in periods.items():

        if n <= days:
            return None

        past = close_series.iloc[-days - 1]
        latest = close_series.iloc[-1]

        if past == 0 or pd.isna(past):
            return None

        returns[label] = (
            latest / past
        ) - 1

    score = (
        0.40 * returns["P3"]
        + 0.20 * returns["P6"]
        + 0.20 * returns["P9"]
        + 0.20 * returns["P12"]
    )

    return round(score * 100, 2)


# ============================================================
# MINERVINI TREND TEMPLATE
# ============================================================

def compute_trend_template(series):

    """
    7/7 Trend Template
    """

    if len(series) < 273:
        return None

    price = series.iloc[-1]

    sma50 = (
        series
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    sma150 = (
        series
        .rolling(150)
        .mean()
        .iloc[-1]
    )

    sma200_series = (
        series
        .rolling(200)
        .mean()
    )

    sma200 = sma200_series.iloc[-1]

    sma200_1mo_ago = (
        sma200_series.iloc[-21]
    )

    if any(
        pd.isna(x)
        for x in [
            sma50,
            sma150,
            sma200,
            sma200_1mo_ago
        ]
    ):
        return None

    low_52w = (
        series.tail(252).min()
    )

    high_52w = (
        series.tail(252).max()
    )

    criteria = [

        # 1
        price > sma150
        and price > sma200,

        # 2
        sma150 > sma200,

        # 3
        sma200 > sma200_1mo_ago,

        # 4
        sma50 > sma150
        and sma50 > sma200,

        # 5
        price > sma50,

        # 6
        price >= 1.25 * low_52w,

        # 7
        price >= 0.75 * high_52w,
    ]

    return sum(criteria) == 7


# ============================================================
# VOLUME DRY-UP
# ============================================================

def compute_volume_dryup(volume):

    """
    Recent volume must contract relative to
    the 50-day average.

    Returns:
        PASS/FAIL
        recent_avg_volume
        avg50_volume
        ratio
    """

    if len(volume) < 60:
        return None

    avg50 = (
        volume
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    recent_avg = (
        volume
        .tail(VCP_DRYUP_DAYS)
        .mean()
    )

    if (
        pd.isna(avg50)
        or pd.isna(recent_avg)
        or avg50 <= 0
    ):
        return None

    ratio = (
        recent_avg / avg50
    )

    passed = (
        ratio <= VCP_VOLUME_DRYUP_RATIO
    )

    return {
        "pass": bool(passed),
        "recent_avg": float(recent_avg),
        "avg50": float(avg50),
        "ratio": float(ratio),
    }


# ============================================================
# VCP DETECTION
# ============================================================

def compute_vcp(close, volume):

    """
    Mechanical approximation of a VCP.

    This is deliberately not a black-box pattern claim.

    We identify contraction ranges using progressively
    shorter windows and require:

        contraction 1 > contraction 2 > contraction 3

    plus:

        final contraction <= 12%
        total base depth <= 25%
        recent volume contraction
    """

    required = max(
        VCP_LOOKBACK,
        60
    )

    if len(close) < required:
        return None

    price = float(close.iloc[-1])

    base_close = (
        close.tail(VCP_LOOKBACK)
    )

    if len(base_close) < VCP_LOOKBACK:
        return None

    # --------------------------------------------------------
    # Base depth
    # --------------------------------------------------------

    base_high = float(
        base_close.max()
    )

    base_low = float(
        base_close.min()
    )

    if base_high <= 0:
        return None

    base_depth = (
        base_high - base_low
    ) / base_high

    base_depth_pass = (
        base_depth <= VCP_MAX_BASE_DEPTH
    )

    # --------------------------------------------------------
    # Contraction measurements
    #
    # Use sequential windows ending at today.
    # Longer windows represent earlier contractions.
    # --------------------------------------------------------

    window_sizes = [
        50,
        35,
        20
    ]

    contractions = []

    for window in window_sizes:

        if len(close) < window:
            continue

        segment = (
            close.tail(window)
        )

        high = float(segment.max())
        low = float(segment.min())

        if high <= 0:
            continue

        depth = (
            high - low
        ) / high

        contractions.append(depth)

    if len(contractions) < VCP_MIN_CONTRACTIONS:
        return {
            "pass": False,
            "contractions": contractions,
            "base_depth": base_depth,
            "final_contraction": None,
            "reason": "INSUFFICIENT_CONTRACTIONS",
        }

    # --------------------------------------------------------
    # Progressive contraction
    # --------------------------------------------------------

    progressive = True

    for i in range(1, len(contractions)):

        if (
            contractions[i]
            > contractions[i - 1]
            * VCP_CONTRACTION_IMPROVEMENT
        ):
            progressive = False
            break

    final_contraction = contractions[-1]

    final_depth_pass = (
        final_contraction
        <= VCP_MAX_FINAL_CONTRACTION
    )

    # --------------------------------------------------------
    # Volume contraction
    # --------------------------------------------------------

    dryup = compute_volume_dryup(volume)

    volume_pass = (
        dryup is not None
        and dryup["pass"]
    )

    # --------------------------------------------------------
    # Final VCP
    #
    # Note: volume is separately displayed, but VCP itself
    # requires progressive price contraction.
    # --------------------------------------------------------

    vcp_pass = (
        progressive
        and final_depth_pass
        and base_depth_pass
        and volume_pass
    )

    return {
        "pass": bool(vcp_pass),
        "contractions": contractions,
        "base_depth": float(base_depth),
        "final_contraction": float(
            final_contraction
        ),
        "progressive": bool(progressive),
        "base_depth_pass": bool(
            base_depth_pass
        ),
        "final_depth_pass": bool(
            final_depth_pass
        ),
        "volume_pass": bool(
            volume_pass
        ),
    }


# ============================================================
# PIVOT
# ============================================================

def compute_pivot(close, volume):

    """
    Pivot = highest high/close in recent pivot window.

    Using adjusted close because the screener itself is
    based on adjusted yfinance price data.

    Pivot proximity is measured relative to pivot.
    """

    if len(close) < PIVOT_LOOKBACK:
        return None

    recent = (
        close.tail(PIVOT_LOOKBACK)
    )

    pivot = float(
        recent.max()
    )

    price = float(
        close.iloc[-1]
    )

    if pivot <= 0:
        return None

    distance = (
        price / pivot
    ) - 1

    proximity_pass = (
        abs(distance)
        <= PIVOT_PROXIMITY_PCT
    )

    avg50_volume = (
        volume
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    current_volume = float(
        volume.iloc[-1]
    )

    if (
        pd.isna(avg50_volume)
        or avg50_volume <= 0
    ):
        breakout_ratio = None
        breakout_pass = False
    else:

        breakout_ratio = (
            current_volume
            / avg50_volume
        )

        breakout_pass = (
            breakout_ratio
            >= BREAKOUT_VOLUME_MULTIPLIER
        )

    return {
        "pivot": pivot,
        "distance": distance,
        "proximity_pass": bool(
            proximity_pass
        ),
        "current_volume": current_volume,
        "avg50_volume": float(
            avg50_volume
        ),
        "breakout_ratio": (
            float(breakout_ratio)
            if breakout_ratio is not None
            else None
        ),
        "breakout_pass": bool(
            breakout_pass
        ),
    }


# ============================================================
# RS LINE TREND TEMPLATE
# ============================================================

def compute_rs_line_template(
    stock_close,
    bench_close
):

    aligned = pd.concat(
        [
            stock_close,
            bench_close
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(aligned) < 273:
        return None

    aligned.columns = [
        "stock",
        "bench"
    ]

    rs_line = (
        aligned["stock"]
        / aligned["bench"]
    )

    return compute_trend_template(
        rs_line
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def compute_diagnostics(
    stock_close,
    bench_close
):

    df = pd.concat(
        [
            stock_close,
            bench_close
        ],
        axis=1,
        join="inner"
    )

    df.columns = [
        "stock",
        "bench"
    ]

    df = df.dropna()

    if len(df) < LOOKBACK_DAYS + 2:
        return None

    df["rs_ratio"] = (
        df["stock"]
        / df["bench"]
    )

    df["rs_prev_high"] = (
        df["rs_ratio"]
        .shift(1)
        .rolling(LOOKBACK_DAYS)
        .max()
    )

    df["price_prev_high"] = (
        df["stock"]
        .shift(1)
        .rolling(LOOKBACK_DAYS)
        .max()
    )

    today = df.iloc[-1]

    blue_dot = (
        bool(
            today["rs_ratio"]
            > today["rs_prev_high"]
        )
        if pd.notna(
            today["rs_prev_high"]
        )
        else False
    )

    price_at_new_high = (
        bool(
            today["stock"]
            > today["price_prev_high"]
        )
        if pd.notna(
            today["price_prev_high"]
        )
        else False
    )

    green_dot = (
        blue_dot
        and not price_at_new_high
    )

    return (
        blue_dot,
        green_dot
    )


# ============================================================
# MAIN
# ============================================================

def main():

    tickers = load_tickers()

    print(
        f"Loaded {len(tickers)} tickers."
    )

    run_mode = get_run_mode()

    print(
        f"Run mode: {run_mode}"
    )

    bench_close = download_benchmark(
        run_mode
    )

    all_stocks = []

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
            f"Downloading batch "
            f"{i}-{i + len(batch)}..."
        )

        try:

            data = yf.download(
                batch,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True
            )

        except Exception as e:

            print(
                f"Batch download failed: {e}"
            )

            continue

        intraday_batch_prices = (
            fetch_intraday_last_price(batch)
            if run_mode == "PREVIEW"
            else {}
        )

        for symbol in batch:

            try:

                sdata = (
                    data
                    if len(batch) == 1
                    else data[symbol]
                )

                close = (
                    sdata["Close"]
                    .dropna()
                )

                volume = (
                    sdata["Volume"]
                    .dropna()
                )

                if (
                    close.empty
                    or len(close)
                    < LOOKBACK_DAYS + 2
                ):
                    continue

                if run_mode == "PREVIEW":

                    close = append_preview_price(
                        close,
                        intraday_batch_prices.get(
                            symbol
                        )
                    )

                # ------------------------------------------------
                # PRICE FILTER
                # ------------------------------------------------

                last_price = float(
                    close.iloc[-1]
                )

                if last_price <= MIN_PRICE:
                    continue

                # ------------------------------------------------
                # LIQUIDITY FILTER
                # ------------------------------------------------

                avg20_volume = (
                    volume
                    .tail(VOLUME_LOOKBACK)
                    .mean()
                )

                if (
                    pd.isna(avg20_volume)
                    or avg20_volume
                    <= MIN_AVG_VOLUME
                ):
                    continue

                # ------------------------------------------------
                # RS SCORE
                # ------------------------------------------------

                rs_score = compute_rs_score(
                    close
                )

                if rs_score is None:
                    continue

                # ------------------------------------------------
                # PRICE TREND TEMPLATE
                # ------------------------------------------------

                tt_pass = (
                    compute_trend_template(
                        close
                    )
                )

                # ------------------------------------------------
                # RS LINE
                # ------------------------------------------------

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

                rs_ratio_full = (
                    aligned["s"]
                    / aligned["b"]
                )

                rs_tt_pass = (
                    compute_trend_template(
                        rs_ratio_full
                    )
                )

                # ------------------------------------------------
                # DIAGNOSTICS
                # ------------------------------------------------

                diagnostics = compute_diagnostics(
                    close,
                    bench_close
                )

                if diagnostics is None:
                    continue

                blue_dot, green_dot = diagnostics

                # ------------------------------------------------
                # VCP
                # ------------------------------------------------

                vcp = compute_vcp(
                    close,
                    volume
                )

                if vcp is None:
                    continue

                # ------------------------------------------------
                # VOLUME DRY-UP
                # ------------------------------------------------

                dryup = compute_volume_dryup(
                    volume
                )

                if dryup is None:
                    continue

                # ------------------------------------------------
                # PIVOT
                # ------------------------------------------------

                pivot_data = compute_pivot(
                    close,
                    volume
                )

                if pivot_data is None:
                    continue

                # ------------------------------------------------
                # FINAL SCREEN
                # ------------------------------------------------

                core_screen_pass = (
                    tt_pass is True
                    and rs_tt_pass is True
                    and vcp["pass"] is True
                    and dryup["pass"] is True
                    and pivot_data[
                        "proximity_pass"
                    ] is True
                )

                if REQUIRE_BREAKOUT_VOLUME:

                    screen_pass = (
                        core_screen_pass
                        and pivot_data[
                            "breakout_pass"
                        ]
                    )

                else:

                    screen_pass = (
                        core_screen_pass
                    )

                # ------------------------------------------------
                # STORE
                # ------------------------------------------------

                all_stocks.append({

                    "symbol": symbol.replace(
                        ".NS",
                        ""
                    ),

                    "rs_score": rs_score,

                    "last_close": round(
                        last_price,
                        2
                    ),

                    "avg20_volume": round(
                        float(avg20_volume),
                        0
                    ),

                    "tt_pass": tt_pass,

                    "rs_tt_pass": rs_tt_pass,

                    # VCP
                    "vcp_pass":
                        vcp["pass"],

                    "vcp_contraction_1":
                        (
                            vcp["contractions"][0]
                            if len(
                                vcp["contractions"]
                            ) > 0
                            else None
                        ),

                    "vcp_contraction_2":
                        (
                            vcp["contractions"][1]
                            if len(
                                vcp["contractions"]
                            ) > 1
                            else None
                        ),

                    "vcp_contraction_3":
                        (
                            vcp["contractions"][2]
                            if len(
                                vcp["contractions"]
                            ) > 2
                            else None
                        ),

                    "vcp_final_depth":
                        vcp[
                            "final_contraction"
                        ],

                    "vcp_base_depth":
                        vcp[
                            "base_depth"
                        ],

                    # Dry-up
                    "volume_dryup_pass":
                        dryup["pass"],

                    "recent_avg_volume":
                        round(
                            dryup[
                                "recent_avg"
                            ],
                            0
                        ),

                    "volume_50d_avg":
                        round(
                            dryup[
                                "avg50"
                            ],
                            0
                        ),

                    "volume_dryup_ratio":
                        round(
                            dryup[
                                "ratio"
                            ],
                            3
                        ),

                    # Pivot
                    "pivot":
                        round(
                            pivot_data[
                                "pivot"
                            ],
                            2
                        ),

                    "pivot_distance":
                        pivot_data[
                            "distance"
                        ],

                    "pivot_proximity_pass":
                        pivot_data[
                            "proximity_pass"
                        ],

                    # Breakout volume
                    "current_volume":
                        round(
                            pivot_data[
                                "current_volume"
                            ],
                            0
                        ),

                    "breakout_volume_ratio":
                        (
                            round(
                                pivot_data[
                                    "breakout_ratio"
                                ],
                                2
                            )
                            if pivot_data[
                                "breakout_ratio"
                            ] is not None
                            else None
                        ),

                    "breakout_volume_pass":
                        pivot_data[
                            "breakout_pass"
                        ],

                    # Final
                    "screen_pass":
                        screen_pass,

                    # Diagnostics
                    "blue_dot":
                        blue_dot,

                    "one_year_rs_cross":
                        blue_dot,

                    "green_dot":
                        green_dot,
                })

            except Exception as e:

                print(
                    f"Skipping {symbol}: {e}"
                )

                continue

        time.sleep(1)

    # ========================================================
    # NO DATA
    # ========================================================

    if not all_stocks:

        print(
            "No stocks with sufficient data found."
        )

        results_df = pd.DataFrame()

        write_to_sheet(
            results_df,
            run_mode
        )

        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    universe_df = pd.DataFrame(
        all_stocks
    )

    # ========================================================
    # LABELS
    # ========================================================

    universe_df[
        "tt_pass_label"
    ] = universe_df[
        "tt_pass"
    ].map({
        True: "PASS",
        False: "FAIL",
        None: "N/A"
    })

    universe_df[
        "rs_tt_pass_label"
    ] = universe_df[
        "rs_tt_pass"
    ].map({
        True: "PASS",
        False: "FAIL",
        None: "N/A"
    })

    universe_df[
        "vcp_label"
    ] = universe_df[
        "vcp_pass"
    ].map({
        True: "PASS",
        False: "FAIL"
    })

    universe_df[
        "dryup_label"
    ] = universe_df[
        "volume_dryup_pass"
    ].map({
        True: "PASS",
        False: "FAIL"
    })

    universe_df[
        "pivot_label"
    ] = universe_df[
        "pivot_proximity_pass"
    ].map({
        True: "PASS",
        False: "FAIL"
    })

    universe_df[
        "breakout_volume_label"
    ] = universe_df[
        "breakout_volume_pass"
    ].map({
        True: "PASS",
        False: "FAIL"
    })

    universe_df[
        "screen_label"
    ] = universe_df[
        "screen_pass"
    ].map({
        True: "PASS",
        False: "FAIL"
    })

    universe_df[
        "blue_dot_label"
    ] = universe_df[
        "blue_dot"
    ].map({
        True: "YES",
        False: ""
    })

    universe_df[
        "one_year_rs_cross_label"
    ] = universe_df[
        "one_year_rs_cross"
    ].map({
        True: "YES",
        False: ""
    })

    universe_df[
        "green_dot_label"
    ] = universe_df[
        "green_dot"
    ].map({
        True: "YES",
        False: ""
    })

    # ========================================================
    # SCREENING OUTPUT
    # ========================================================

    results_df = universe_df[[
        "symbol",
        "rs_score",
        "last_close",
        "avg20_volume",

        "tt_pass_label",
        "rs_tt_pass_label",

        "vcp_label",
        "vcp_contraction_1",
        "vcp_contraction_2",
        "vcp_contraction_3",
        "vcp_final_depth",
        "vcp_base_depth",

        "dryup_label",
        "recent_avg_volume",
        "volume_50d_avg",
        "volume_dryup_ratio",

        "pivot",
        "pivot_distance",
        "pivot_label",

        "current_volume",
        "breakout_volume_ratio",
        "breakout_volume_label",

        "screen_label",

        "blue_dot_label",
        "one_year_rs_cross_label",
        "green_dot_label",
    ]].rename(columns={

        "tt_pass_label":
            "price_trend_template",

        "rs_tt_pass_label":
            "rs_trend_template",

        "vcp_label":
            "vcp",

        "dryup_label":
            "volume_dryup",

        "pivot_label":
            "pivot_proximity",

        "breakout_volume_label":
            "breakout_volume",

        "screen_label":
            "screen",

        "blue_dot_label":
            "blue_dot",

        "one_year_rs_cross_label":
            "one_year_rs_cross",

        "green_dot_label":
            "green_dot",
    })

    # ========================================================
    # RANK ALL STOCKS BY RAW RS SCORE
    # ========================================================

    results_df = (
        results_df
        .sort_values(
            "rs_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    results_df["rank"] = (
        range(
            1,
            len(results_df) + 1
        )
    )

    results_df = results_df[
        ["rank"]
        + [
            c
            for c in results_df.columns
            if c != "rank"
        ]
    ]

    # ========================================================
    # STATISTICS
    # ========================================================

    n_tt_pass = (
        universe_df["tt_pass"]
        == True
    ).sum()

    n_rs_tt_pass = (
        universe_df["rs_tt_pass"]
        == True
    ).sum()

    n_vcp = (
        universe_df["vcp_pass"]
        == True
    ).sum()

    n_dryup = (
        universe_df[
            "volume_dryup_pass"
        ]
        == True
    ).sum()

    n_pivot = (
        universe_df[
            "pivot_proximity_pass"
        ]
        == True
    ).sum()

    n_breakout = (
        universe_df[
            "breakout_volume_pass"
        ]
        == True
    ).sum()

    n_screen = (
        universe_df[
            "screen_pass"
        ]
        == True
    ).sum()

    n_both_tt = universe_df[
        (
            universe_df["tt_pass"]
            == True
        )
        &
        (
            universe_df["rs_tt_pass"]
            == True
        )
    ].shape[0]

    print(
        "========================================"
    )

    print(
        f"Universe scanned: "
        f"{len(universe_df)}"
    )

    print(
        f"Price TT PASS: "
        f"{n_tt_pass}"
    )

    print(
        f"RS TT PASS: "
        f"{n_rs_tt_pass}"
    )

    print(
        f"Both TT PASS: "
        f"{n_both_tt}"
    )

    print(
        f"VCP PASS: "
        f"{n_vcp}"
    )

    print(
        f"Volume dry-up PASS: "
        f"{n_dryup}"
    )

    print(
        f"Pivot proximity PASS: "
        f"{n_pivot}"
    )

    print(
        f"Breakout volume PASS: "
        f"{n_breakout}"
    )

    print(
        f"FINAL SCREEN PASS: "
        f"{n_screen}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # WRITE SCREENER
    # ========================================================

    write_to_sheet(
        results_df,
        run_mode
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    if run_mode == "PREVIEW":

        print(
            "Preview mode: RS_Screener updated. "
            "Holdings/Portfolio untouched."
        )

        return

    # ========================================================
    # PORTFOLIO
    # ========================================================

    build_portfolio(
        universe_df
    )


# ============================================================
# READ CONFIG
# ============================================================

def read_config(sh):

    try:

        cfg_ws = sh.worksheet(
            CONFIG_WORKSHEET
        )

        records = cfg_ws.get_all_records()

        settings = {
            row["Setting"]:
                row["Value"]
            for row in records
            if row.get("Setting")
        }

    except gspread.WorksheetNotFound:

        cfg_ws = sh.add_worksheet(
            title=CONFIG_WORKSHEET,
            rows=10,
            cols=3
        )

        cfg_ws.update(
            [
                [
                    "Setting",
                    "Value",
                    "Notes"
                ],
                [
                    "Total Capital (INR)",
                    0,
                    "EDIT ME"
                ],
            ],
            "A1"
        )

        settings = {
            "Total Capital (INR)": 0
        }

    try:

        capital = float(
            settings.get(
                "Total Capital (INR)",
                0
            )
            or 0
        )

    except (
        ValueError,
        TypeError
    ):

        capital = 0

    return capital


# ============================================================
# CONFIRMED EXECUTIONS
# ============================================================

def apply_confirmed_executions(sh):

    try:

        port_ws = sh.worksheet(
            PORTFOLIO_WORKSHEET
        )

    except gspread.WorksheetNotFound:

        return

    try:

        prior_rows = (
            port_ws.get_all_records(
                head=3
            )
        )

    except Exception as e:

        print(
            "Could not read prior Portfolio "
            f"tab: {e}"
        )

        return

    try:

        holdings_ws = sh.worksheet(
            HOLDINGS_WORKSHEET
        )

        existing = (
            holdings_ws.get_all_records()
        )

        holdings = {

            row["symbol"]: {

                "entry_price":
                    float(
                        row.get(
                            "entry_price"
                        )
                        or 0
                    ),

                "entry_date":
                    row.get(
                        "entry_date",
                        ""
                    ),
            }

            for row in existing
            if row.get("symbol")
        }

    except gspread.WorksheetNotFound:

        holdings_ws = sh.add_worksheet(
            title=HOLDINGS_WORKSHEET,
            rows=100,
            cols=5
        )

        holdings_ws.update(
            [
                [
                    "symbol",
                    "entry_price",
                    "entry_date"
                ]
            ],
            "A1"
        )

        holdings = {}

    today_str = datetime.now().strftime(
        "%Y-%m-%d"
    )

    changed = False

    for row in prior_rows:

        executed = (
            str(
                row.get(
                    "Executed",
                    ""
                )
            )
            .strip()
            .upper()
        )

        if executed not in (
            "Y",
            "YES"
        ):
            continue

        action = (
            str(
                row.get(
                    "Action",
                    ""
                )
            )
            .strip()
            .upper()
        )

        symbol = (
            str(
                row.get(
                    "Symbol",
                    ""
                )
            )
            .strip()
        )

        if not symbol:
            continue

        if action == "BUY":

            exec_price_raw = (
                row.get(
                    "Execution Price"
                )
                or
                row.get(
                    "Entry Price"
                )
            )

            try:

                exec_price = float(
                    exec_price_raw
                )

            except (
                ValueError,
                TypeError
            ):

                print(
                    f"Skipping confirmed BUY "
                    f"for {symbol}"
                )

                continue

            holdings[symbol] = {
                "entry_price":
                    exec_price,
                "entry_date":
                    today_str
            }

            changed = True

            print(
                f"Confirmed BUY applied: "
                f"{symbol} @ {exec_price}"
            )

        elif action == "SELL":

            if symbol in holdings:

                del holdings[symbol]

                changed = True

                print(
                    f"Confirmed SELL applied: "
                    f"{symbol}"
                )

    if changed:

        holdings_ws.clear()

        rows_out = [
            [
                "symbol",
                "entry_price",
                "entry_date"
            ]
        ] + [

            [
                s,
                v["entry_price"],
                v["entry_date"]
            ]

            for s, v
            in holdings.items()
        ]

        holdings_ws.update(
            rows_out,
            "A1"
        )

        print(
            "Holdings updated."
        )

    else:

        print(
            "No confirmed executions."
        )


# ============================================================
# PORTFOLIO
# ============================================================

def build_portfolio(universe_df):

    """
    ENTRY:

        Price > Rs20
        20D average volume >100k
        Price TT PASS
        RS TT PASS
        VCP PASS
        Volume dry-up PASS
        Pivot proximity PASS

        ranked by RS Score

    EXIT:

        leaves current Top 10

    Breakout volume:

        diagnostic unless REQUIRE_BREAKOUT_VOLUME=True
    """

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if not sheet_id or not creds_json:

        print(
            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS."
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

    gc = gspread.authorize(
        creds
    )

    sh = gc.open_by_key(
        sheet_id
    )

    capital = read_config(sh)

    apply_confirmed_executions(
        sh
    )

    try:

        holdings_ws = sh.worksheet(
            HOLDINGS_WORKSHEET
        )

        existing = (
            holdings_ws.get_all_records()
        )

        current_holdings = {

            row["symbol"]: {

                "entry_price":
                    float(
                        row.get(
                            "entry_price"
                        )
                        or 0
                    ),

                "entry_date":
                    row.get(
                        "entry_date",
                        ""
                    ),
            }

            for row in existing
            if row.get("symbol")
        }

    except gspread.WorksheetNotFound:

        holdings_ws = sh.add_worksheet(
            title=HOLDINGS_WORKSHEET,
            rows=100,
            cols=5
        )

        holdings_ws.update(
            [
                [
                    "symbol",
                    "entry_price",
                    "entry_date"
                ]
            ],
            "A1"
        )

        current_holdings = {}

    # ========================================================
    # HARD SCREEN
    # ========================================================

    pool = universe_df[
        (universe_df["tt_pass"] == True)
        &
        (universe_df["rs_tt_pass"] == True)
        &
        (universe_df["vcp_pass"] == True)
        &
        (
            universe_df[
                "volume_dryup_pass"
            ]
            == True
        )
        &
        (
            universe_df[
                "pivot_proximity_pass"
            ]
            == True
        )
    ].copy()

    if REQUIRE_BREAKOUT_VOLUME:

        pool = pool[
            pool[
                "breakout_volume_pass"
            ]
            == True
        ].copy()

    # ========================================================
    # RANK
    # ========================================================

    pool = (
        pool
        .sort_values(
            "rs_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    pool["rank"] = range(
        1,
        len(pool) + 1
    )

    pool_rank_lookup = dict(
        zip(
            pool["symbol"],
            pool["rank"]
        )
    )

    price_lookup = dict(
        zip(
            universe_df["symbol"],
            universe_df["last_close"]
        )
    )

    diag_lookup = {
        row["symbol"]:
            row
        for _, row
        in universe_df.iterrows()
    }

    # ========================================================
    # TOP 10
    # ========================================================

    target_top10 = set(
        pool
        .head(TOP_N)
        ["symbol"]
        .tolist()
    )

    kept = [
        s
        for s in current_holdings
        if s in target_top10
    ]

    pending_sell = [
        s
        for s in current_holdings
        if s not in target_top10
    ]

    slots_open = (
        TOP_N - len(kept)
    )

    pending_buy = (

        [
            s
            for s in
            pool
            .head(TOP_N)
            ["symbol"]
            .tolist()
            if s not in current_holdings
        ][:slots_open]

        if slots_open > 0
        else []
    )

    slot_capital = (
        capital / TOP_N
        if capital > 0
        else 0
    )

    rows = []

    # ========================================================
    # HOLDINGS
    # ========================================================

    for s in kept:

        entry_price = (
            current_holdings[s]
            ["entry_price"]
        )

        entry_date = (
            current_holdings[s]
            ["entry_date"]
        )

        current_price = (
            price_lookup.get(
                s,
                entry_price
            )
        )

        position_value = (
            round(
                capital / TOP_N,
                0
            )
            if capital > 0
            else 0
        )

        qty = (
            int(
                position_value
                / entry_price
            )
            if entry_price > 0
            else 0
        )

        pnl_pct = (
            round(
                (
                    current_price
                    / entry_price
                    - 1
                )
                * 100,
                2
            )
            if entry_price > 0
            else 0
        )

        gross_pnl_rs = (
            qty
            * (
                current_price
                - entry_price
            )
        )

        s_cost_est = (
            sell_side_cost(
                qty
                * current_price
            )
            if qty > 0
            else 0
        )

        tax_est = estimate_stcg(
            gross_pnl_rs
            - s_cost_est
        )

        diag = diag_lookup.get(
            s,
            {}
        )

        rows.append({

            "Action":
                "HOLD",

            "Symbol":
                s,

            "Rank":
                pool_rank_lookup.get(
                    s,
                    ""
                ),

            "Entry Price":
                entry_price,

            "Entry Date":
                entry_date,

            "Current Price":
                current_price,

            "Qty":
                qty,

            "Position Value (Rs)":
                position_value,

            "P&L %":
                f"{pnl_pct}%",

            "Buy Cost (Rs)":
                round(
                    buy_side_cost(
                        qty
                        * entry_price
                    ),
                    2
                )
                if qty > 0
                else 0,

            "Sell Cost (Rs)":
                round(
                    s_cost_est,
                    2
                ),

            "Est. STCG Tax (Rs)":
                round(
                    tax_est,
                    2
                ),

            "Blue Dot":
                (
                    "YES"
                    if diag.get(
                        "blue_dot",
                        False
                    )
                    else ""
                ),

            "1Y RS Cross":
                (
                    "YES"
                    if diag.get(
                        "one_year_rs_cross",
                        False
                    )
                    else ""
                ),

            "Green Dot":
                (
                    "YES"
                    if diag.get(
                        "green_dot",
                        False
                    )
                    else ""
                ),
        })

    # ========================================================
    # BUYS
    # ========================================================

    for s in pending_buy:

        current_price = (
            price_lookup.get(
                s,
                0
            )
        )

        position_value = round(
            slot_capital,
            0
        )

        qty = (
            int(
                position_value
                / current_price
            )
            if current_price > 0
            else 0
        )

        diag = diag_lookup.get(
            s,
            {}
        )

        rows.append({

            "Action":
                "BUY",

            "Symbol":
                s,

            "Rank":
                pool_rank_lookup.get(
                    s,
                    ""
                ),

            "Entry Price":
                current_price,

            "Entry Date":
                "PENDING",

            "Current Price":
                current_price,

            "Qty":
                qty,

            "Position Value (Rs)":
                position_value,

            "P&L %":
                "",

            "Buy Cost (Rs)":
                round(
                    buy_side_cost(
                        qty
                        * current_price
                    ),
                    2
                )
                if qty > 0
                else 0,

            "Sell Cost (Rs)":
                "",

            "Est. STCG Tax (Rs)":
                "",

            "Blue Dot":
                (
                    "YES"
                    if diag.get(
                        "blue_dot",
                        False
                    )
                    else ""
                ),

            "1Y RS Cross":
                (
                    "YES"
                    if diag.get(
                        "one_year_rs_cross",
                        False
                    )
                    else ""
                ),

            "Green Dot":
                (
                    "YES"
                    if diag.get(
                        "green_dot",
                        False
                    )
                    else ""
                ),
        })

    # ========================================================
    # SELLS
    # ========================================================

    for s in pending_sell:

        entry_price = (
            current_holdings[s]
            ["entry_price"]
        )

        current_price = (
            price_lookup.get(
                s,
                entry_price
            )
        )

        pnl_pct = (
            round(
                (
                    current_price
                    / entry_price
                    - 1
                )
                * 100,
                2
            )
            if entry_price > 0
            else 0
        )

        rank_val = (
            pool_rank_lookup.get(
                s,
                ""
            )
        )

        rows.append({

            "Action":
                "SELL",

            "Symbol":
                s,

            "Rank":
                rank_val,

            "Entry Price":
                entry_price,

            "Entry Date":
                current_holdings[s]
                ["entry_date"],

            "Current Price":
                current_price,

            "Qty":
                "",

            "Position Value (Rs)":
                "",

            "P&L %":
                f"{pnl_pct}%",

            "Buy Cost (Rs)":
                "",

            "Sell Cost (Rs)":
                "",

            "Est. STCG Tax (Rs)":
                "",

            "Blue Dot":
                "",

            "1Y RS Cross":
                "",

            "Green Dot":
                "",
        })

    # ========================================================
    # SHEET
    # ========================================================

    n_port_rows_needed = (
        len(rows) + 10
    )

    n_port_cols_needed = (
        len(PORTFOLIO_HEADER) + 2
    )

    try:

        port_ws = sh.worksheet(
            PORTFOLIO_WORKSHEET
        )

        if (
            port_ws.row_count
            < n_port_rows_needed
            or
            port_ws.col_count
            < n_port_cols_needed
        ):

            port_ws.resize(
                rows=max(
                    port_ws.row_count,
                    n_port_rows_needed
                ),

                cols=max(
                    port_ws.col_count,
                    n_port_cols_needed
                )
            )

    except gspread.WorksheetNotFound:

        port_ws = sh.add_worksheet(
            title=PORTFOLIO_WORKSHEET,
            rows=n_port_rows_needed,
            cols=n_port_cols_needed
        )

    port_ws.clear()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M IST"
    )

    invested = sum(
        r.get(
            "Position Value (Rs)",
            0
        )

        for r in rows

        if isinstance(
            r.get(
                "Position Value (Rs)"
            ),
            (
                int,
                float
            )
        )
    )

    summary = (

        f"Last updated: {timestamp}"
        f" | Capital: Rs.{capital:,.0f}"
        f" | Deployed: Rs.{invested:,.0f}"
        f" | "
        f"Entry = Price TT + RS TT + VCP + "
        f"Volume Dry-up + Pivot Proximity"
        f" | "
        f"Top {TOP_N} by RS Score"
        f" | "
        f"Breakout volume required: "
        f"{REQUIRE_BREAKOUT_VOLUME}"
        f" | "
        f"Exit = leaves Top {TOP_N}"
    )

    if capital == 0:

        summary += (
            " | SET CAPITAL IN CONFIG"
        )

    port_ws.update(
        [[summary]],
        "A1"
    )

    row_lists = [

        [
            r.get(
                col,
                ""
            )

            for col in PORTFOLIO_HEADER
        ]

        for r in rows
    ]

    port_ws.update(
        [
            PORTFOLIO_HEADER
        ]
        + row_lists,
        "A3"
    )

    print(
        f"Portfolio updated: "
        f"{len(kept)} held, "
        f"{len(pending_buy)} BUY, "
        f"{len(pending_sell)} SELL. "
        f"Capital: Rs.{capital:,.0f}"
    )


# ============================================================
# WRITE SCREENER TO GOOGLE SHEETS
# ============================================================

def write_to_sheet(
    df,
    run_mode="EOD"
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if not sheet_id or not creds_json:

        print(
            "Missing SHEET_ID or "
            "GOOGLE_CREDENTIALS. "
            "Saving CSV instead."
        )

        df.to_csv(
            "rs_screener_output.csv",
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

    gc = gspread.authorize(
        creds
    )

    n_rows_needed = (
        len(df) + 10
    )

    n_cols_needed = (
        len(df.columns) + 2
    )

    sh = gc.open_by_key(
        sheet_id
    )

    try:

        ws = sh.worksheet(
            WORKSHEET_NAME
        )

        if (
            ws.row_count
            < n_rows_needed
            or
            ws.col_count
            < n_cols_needed
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
            title=WORKSHEET_NAME,
            rows=n_rows_needed,
            cols=n_cols_needed
        )

    ws.clear()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M IST"
    )

    label = (
        "PREVIEW (intraday, not final)"
        if run_mode == "PREVIEW"
        else
        "EOD FINAL"
    )

    ws.update(
        [[
            f"Last updated: {timestamp}"
            f" | {label}"
            f" | VCP + Dry-up + Pivot"
        ]],
        "A1"
    )

    header = list(
        df.columns
    )

    ws.update(
        [header]
        + df.fillna("").values.tolist(),
        "A3"
    )

    print(
        "Google Sheet updated successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()