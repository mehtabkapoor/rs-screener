"""
RS Screener Backtest -- EOD TOP-10 RS Ranking Strategy
========================================================

REVISED TRADING LOGIC
---------------------

ENTRY / HOLDING ELIGIBILITY
    1. Price Trend Template = 7/7 PASS
    2. RS Line Trend Template = 7/7 PASS
    3. Point-in-time liquidity filter must pass
    4. Rank all eligible stocks by raw RS Score
    5. Highest 10 stocks are the portfolio

EXIT
    A holding is EXITED when it is no longer in the current TOP 10.

REBALANCING
    Daily using EOD prices.

GREEN DOT / BLUE DOT
    Both are retained as diagnostic/information fields.
    NEITHER is used to make any trading decision.

REMOVED FROM TRADING LOGIC
    - Rank-20 hysteresis
    - RS < EMA state exit
    - RS < EMA crossover exit
    - Trend Template fail exit
    - Regime/breadth filter
    - Risk-on / caution / risk-off sizing
    - Circuit breaker
    - Multiple exit variants

POSITION SIZING
    Equal-weight target across 10 positions.
    Integer shares are purchased.
    Actual cash balance is tracked.

COSTS
    Zerodha delivery-style statutory costs:
    - STT
    - Stamp duty
    - Exchange transaction charges
    - SEBI charges
    - GST on exchange + SEBI charges
    - DP charge on sale

TAX
    20.8% effective STCG tax on positive realized gains.
    Losses are not used to offset gains in this simplified model.

IMPORTANT EXECUTION ASSUMPTION
    This is an EOD theoretical backtest.
    Today's EOD signals are calculated using today's EOD close
    and the rebalance is assumed to occur at that same EOD price.

    This is NOT a true 3:00-3:30 PM intraday backtest.

SURVIVORSHIP BIAS
    The stock universe still comes from the current stocks.csv.
    Therefore historical delisted/merged stocks not present in stocks.csv
    are not represented.
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

# RS new-high diagnostic lookback.
# IMPORTANT: Blue Dot is diagnostic ONLY.
LOOKBACK_DAYS = 250

DOWNLOAD_YEARS_BEFORE_START = 3

STOCKS_FILE = "stocks.csv"

# Number of portfolio holdings.
TOP_N = 10


# ============================================================
# POINT-IN-TIME LIQUIDITY FILTER
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
# STARTING CAPITAL
# ============================================================

STARTING_CAPITAL = 1_000_000


# ============================================================
# TRANSACTION COSTS
# ============================================================

ENABLE_COSTS = True

# 0.1% STT on delivery buy
STT_BUY_RATE = 0.001

# 0.1% STT on delivery sell
STT_SELL_RATE = 0.001

# 0.015% stamp duty on buy
STAMP_DUTY_RATE = 0.00015

# Approx NSE exchange transaction charge
EXCHANGE_CHARGE_RATE = 0.0000325

# SEBI charge
SEBI_CHARGE_RATE = 0.000001

# GST on exchange + SEBI charges
GST_RATE = 0.18

# Approx DP charge per sell per symbol
DP_CHARGE_FLAT = 20


# ============================================================
# STCG TAX
# ============================================================

ENABLE_STCG = True

STCG_RATE = 0.20

STCG_CESS = 0.04

STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"

SUMMARY_WORKSHEET = "Backtest_Summary"


# ============================================================
# COST FUNCTIONS
# ============================================================

def buy_side_cost(trade_value):
    """
    Total statutory BUY cost for delivery.

    Brokerage assumed zero.
    """

    if not ENABLE_COSTS:
        return 0.0

    stt = STT_BUY_RATE * trade_value

    stamp = STAMP_DUTY_RATE * trade_value

    exch = EXCHANGE_CHARGE_RATE * trade_value

    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return stt + stamp + exch + sebi + gst


def sell_side_cost(trade_value):
    """
    Total statutory SELL cost for delivery.
    Includes DP charge.
    """

    if not ENABLE_COSTS:
        return 0.0

    stt = STT_SELL_RATE * trade_value

    exch = EXCHANGE_CHARGE_RATE * trade_value

    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return (
        stt
        + exch
        + sebi
        + gst
        + DP_CHARGE_FLAT
    )


def stcg_tax(net_gain):
    """
    Simplified STCG treatment.

    Positive realized gains:
        20% + 4% cess = 20.8%

    Losses:
        No tax.

    NOTE:
    This does not model set-off/carry-forward of losses.
    Therefore it is conservative versus a full tax computation.
    """

    if not ENABLE_STCG:
        return 0.0

    if net_gain <= 0:
        return 0.0

    return net_gain * STCG_EFFECTIVE_RATE


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():

    backtest_start = pd.Timestamp(BACKTEST_START)

    backtest_end = pd.Timestamp(BACKTEST_END)

    download_start = (
        backtest_start
        - pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    download_end = (
        backtest_end
        + pd.Timedelta(days=1)
    )

    return (
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d")
    )


# ============================================================
# LOAD STOCK UNIVERSE
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
# DATA CLEANING
# ============================================================

def clean_price_series(close):

    """
    Repairs implausible single-day price jumps.

    If an individual daily move exceeds the threshold,
    the price is held at the previous valid close.

    This is intentionally conservative.

    It is performed BEFORE:
        - RS calculation
        - Trend Template calculation
        - Blue Dot calculation
        - Green Dot calculation
        - portfolio returns
    """

    close = (
        close
        .copy()
        .sort_index()
    )

    pct_change = close.pct_change()

    bad = (
        pct_change.abs()
        > MAX_PLAUSIBLE_DAILY_MOVE
    )

    n_bad = int(bad.sum())

    if n_bad == 0:
        return close, 0

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = cleaned.index.get_loc(idx)

        if pos > 0:

            cleaned.iloc[pos] = (
                cleaned.iloc[pos - 1]
            )

    return cleaned, n_bad


# ============================================================
# BENCHMARK DOWNLOAD
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        f"\nBenchmark download: "
        f"{download_start} to {download_end}"
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
                clean_price_series(close)
            )

            if n_bad:

                print(
                    f"Benchmark {ticker}: "
                    f"repaired {n_bad} "
                    f"implausible data point(s)"
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
        "Could not download any benchmark index data."
    )


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template_series(s):

    sma50 = s.rolling(50).mean()

    sma150 = s.rolling(150).mean()

    sma200 = s.rolling(200).mean()

    sma200_1mo = sma200.shift(21)

    low52 = s.rolling(252).min()

    high52 = s.rolling(252).max()

    # 1
    c1 = (
        (s > sma150)
        &
        (s > sma200)
    )

    # 2
    c2 = (
        sma150 > sma200
    )

    # 3
    c3 = (
        sma200 > sma200_1mo
    )

    # 4
    c4 = (
        (sma50 > sma150)
        &
        (sma50 > sma200)
    )

    # 5
    c5 = (
        s > sma50
    )

    # 6
    c6 = (
        s >= 1.25 * low52
    )

    # 7
    c7 = (
        s >= 0.75 * high52
    )

    met = (
        c1.astype(int)
        + c2.astype(int)
        + c3.astype(int)
        + c4.astype(int)
        + c5.astype(int)
        + c6.astype(int)
        + c7.astype(int)
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
    Calculates all historical variables.

    ACTUAL TRADING DECISIONS:
        - Price Trend Template PASS
        - RS Line Trend Template PASS
        - Rank by RS Score
        - Top 10

    DIAGNOSTIC ONLY:
        - Green Dot
        - Blue Dot
        - 50DMA status

    Green/Blue dots do NOT influence trading.
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

    volume = volume.reindex(
        aligned.index
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
            - 1
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
    # BLUE DOT -- DIAGNOSTIC ONLY
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
    # GREEN DOT -- DIAGNOSTIC ONLY
    # ========================================================
    #
    # IMPORTANT:
    # The supplied script did not contain the original
    # Green Dot definition.
    #
    # Therefore this implementation uses a simple diagnostic:
    # RS Score making a new 250-day high using PRIOR days only.
    #
    # It has ZERO effect on trading.
    #
    # If your live screener uses a different Green Dot formula,
    # replace ONLY this calculation.
    # ========================================================

    previous_rs_score_high = (
        rs_score
        .shift(1)
        .rolling(
            LOOKBACK_DAYS
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

    tt_pass, tt_met = (
        trend_template_series(
            aligned["s"]
        )
    )

    # ========================================================
    # RS LINE TREND TEMPLATE
    # ========================================================

    rs_tt_pass, rs_tt_met = (
        trend_template_series(
            rs_ratio
        )
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
            >= MIN_AVG_VOLUME
        )
    )

    # ========================================================
    # DIAGNOSTIC 50DMA
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

    out = pd.DataFrame({

        "price":
            aligned["s"],

        "rs_line":
            rs_ratio,

        "rs_score":
            rs_score,

        # DIAGNOSTIC ONLY
        "green_dot":
            green_dot,

        "blue_dot":
            blue_dot,

        # ACTUAL TRADING CONDITIONS
        "tt_pass":
            tt_pass,

        "tt_met":
            tt_met,

        "rs_tt_pass":
            rs_tt_pass,

        "rs_tt_met":
            rs_tt_met,

        # LIQUIDITY
        "liquid":
            liquid,

        "avg_volume":
            rolling_avg_volume,

        # DIAGNOSTIC
        "above_50dma":
            above_50dma,
    })

    return out


# ============================================================
# TOP-10 EOD BACKTEST
# ============================================================

def run_backtest(
    all_signals,
    trading_days
):

    """
    Core strategy.

    Every trading day:

        1. Build eligible universe
        2. Price TT must PASS
        3. RS-line TT must PASS
        4. Liquidity must PASS
        5. Sort by raw RS Score
        6. Select top 10
        7. Sell existing positions outside top 10
        8. Buy missing top-10 positions
        9. Mark portfolio to market

    Green Dot and Blue Dot:
        recorded but ignored.

    There is NO hysteresis.
    There is NO EMA exit.
    There is NO regime filter.
    There is NO circuit breaker.
    """

    cash = STARTING_CAPITAL

    holdings = {}

    trade_log = []

    equity_curve = []

    daily_selection_log = []

    for date in trading_days:

        # ====================================================
        # 1. BUILD ELIGIBLE UNIVERSE
        # ====================================================

        pool = []

        for sym, df in all_signals.items():

            if date not in df.index:
                continue

            row = df.loc[date]

            # Need valid RS score
            if pd.isna(
                row["rs_score"]
            ):
                continue

            # Point-in-time liquidity
            if not bool(
                row["liquid"]
            ):
                continue

            # PRICE TREND TEMPLATE
            if not bool(
                row["tt_pass"]
            ):
                continue

            # RS LINE TREND TEMPLATE
            if not bool(
                row["rs_tt_pass"]
            ):
                continue

            # =================================================
            # NO GREEN DOT FILTER
            # NO BLUE DOT FILTER
            #
            # Both remain in the dataset but are ignored.
            # =================================================

            pool.append(
                (
                    sym,
                    float(
                        row["rs_score"]
                    )
                )
            )

        # ====================================================
        # 2. SORT BY RS SCORE
        # ====================================================

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ====================================================
        # 3. ASSIGN RANK
        # ====================================================

        rank_lookup = {
            sym: i + 1
            for i, (sym, _) in enumerate(pool)
        }

        # ====================================================
        # 4. TOP 10
        # ====================================================

        top10 = [
            sym
            for sym, _ in pool[:TOP_N]
        ]

        # ====================================================
        # 5. DAILY DIAGNOSTIC LOG
        # ====================================================

        selected_rows = []

        for rank, (sym, score) in enumerate(
            pool[:TOP_N],
            start=1
        ):

            row = all_signals[
                sym
            ].loc[date]

            selected_rows.append({

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

                "price":
                    round(
                        float(
                            row["price"]
                        ),
                        2
                    ),

                "green_dot":
                    bool(
                        row["green_dot"]
                    ),

                "blue_dot":
                    bool(
                        row["blue_dot"]
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

        daily_selection_log.extend(
            selected_rows
        )

        # ====================================================
        # 6. EXIT:
        #    SELL ANY HOLDING OUTSIDE TOP 10
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            if sym not in all_signals:
                continue

            df = all_signals[sym]

            if date not in df.index:
                continue

            # ------------------------------------------------
            # ONLY EXIT RULE:
            # stock is no longer in TOP 10
            # ------------------------------------------------

            if sym not in top10:

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
                        "LEFT TOP 10",

                    "exit_rank":
                        rank_lookup.get(
                            sym,
                            9999
                        ),

                    "entry_green_dot":
                        pos.get(
                            "entry_green_dot",
                            False
                        ),

                    "entry_blue_dot":
                        pos.get(
                            "entry_blue_dot",
                            False
                        ),

                    "exit_green_dot":
                        bool(
                            df.loc[
                                date,
                                "green_dot"
                            ]
                        ),

                    "exit_blue_dot":
                        bool(
                            df.loc[
                                date,
                                "blue_dot"
                            ]
                        ),
                })

        # ====================================================
        # 7. CALCULATE PORTFOLIO VALUE BEFORE BUYING
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

                price = pos[
                    "entry_price"
                ]

            portfolio_value += (
                pos["qty"]
                *
                price
            )

        # ====================================================
        # 8. BUY MISSING TOP-10 POSITIONS
        # ====================================================

        slots_open = (
            TOP_N
            -
            len(holdings)
        )

        if slots_open > 0:

            # Equal-weight target.
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

                # Integer number of shares
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

                    # Diagnostics only
                    "entry_green_dot":
                        bool(
                            row[
                                "green_dot"
                            ]
                        ),

                    "entry_blue_dot":
                        bool(
                            row[
                                "blue_dot"
                            ]
                        ),
                }

                slots_open -= 1

        # ====================================================
        # 9. FINAL DAILY MARK-TO-MARKET
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

                price = pos[
                    "entry_price"
                ]

            portfolio_value += (
                pos["qty"]
                *
                price
            )

        # ====================================================
        # 10. HOLDING SNAPSHOT
        # ====================================================

        holding_details = []

        for sym, pos in holdings.items():

            df = all_signals[sym]

            if date in df.index:

                row = df.loc[date]

                current_price = float(
                    row["price"]
                )

                current_rs = float(
                    row["rs_score"]
                ) if pd.notna(
                    row["rs_score"]
                ) else np.nan

            else:

                current_price = (
                    pos["entry_price"]
                )

                current_rs = np.nan

            holding_details.append({

                "symbol":
                    sym,

                "qty":
                    pos["qty"],

                "price":
                    round(
                        current_price,
                        2
                    ),

                "rs_score":
                    round(
                        current_rs,
                        4
                    )
                    if pd.notna(
                        current_rs
                    )
                    else None,
            })

        # ====================================================
        # 11. EQUITY CURVE
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

            "cash_rs":
                round(
                    cash,
                    2
                ),

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
                ),
        })

    # ========================================================
    # 12. CLOSE REMAINING POSITIONS AT BACKTEST END
    # ========================================================

    if len(trading_days):

        last_date = trading_days[-1]

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

            else:

                exit_price = pos[
                    "entry_price"
                ]

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
                    last_date.strftime(
                        "%Y-%m-%d"
                    )
                    +
                    " (OPEN)",

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

                "exit_rank":
                    "",

                "entry_green_dot":
                    pos.get(
                        "entry_green_dot",
                        False
                    ),

                "entry_blue_dot":
                    pos.get(
                        "entry_blue_dot",
                        False
                    ),

                "exit_green_dot":
                    bool(
                        df.loc[
                            last_date,
                            "green_dot"
                        ]
                    )
                    if last_date in df.index
                    else False,

                "exit_blue_dot":
                    bool(
                        df.loc[
                            last_date,
                            "blue_dot"
                        ]
                    )
                    if last_date in df.index
                    else False,
            })

    return (
        pd.DataFrame(trade_log),
        pd.DataFrame(equity_curve),
        pd.DataFrame(daily_selection_log)
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
    # FINAL VALUE
    # ========================================================

    final_value = (
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
    # DRAWdown
    # ========================================================

    running_max = (
        equity_df[
            "equity"
        ].cummax()
    )

    drawdown = (
        equity_df["equity"]
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

    n = len(closed)

    if n:

        win_rate_gross = (
            closed[
                "gross_return_pct"
            ]
            > 0
        ).mean() * 100

        win_rate_net = (
            closed[
                "net_return_pct"
            ]
            > 0
        ).mean() * 100

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

        total_costs_rs = (
            total_buy_costs
            +
            total_sell_costs
        )

        total_tax_rs = (
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

        avg_winner_net = (
            winners[
                "net_return_pct"
            ].mean()
            if len(winners)
            else 0
        )

        avg_loser_net = (
            losers[
                "net_return_pct"
            ].mean()
            if len(losers)
            else 0
        )

        net_profit = (
            winners[
                "net_pnl_rs"
            ].sum()
            if len(winners)
            else 0
        )

        net_loss = abs(
            losers[
                "net_pnl_rs"
            ].sum()
        ) if len(losers) else 0

        profit_factor_net = (
            net_profit
            /
            net_loss
            if net_loss > 0
            else 0
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

        total_costs_rs = 0

        total_tax_rs = 0

        avg_winner_net = 0

        avg_loser_net = 0

        profit_factor_net = 0

    # ========================================================
    # DAILY RETURNS
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

        # CAGR
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

        # Annualized volatility
        annualized_vol = (
            daily_std
            *
            np.sqrt(252)
        )

        # Sharpe
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

        # Sortino
        downside = (
            daily_returns[
                daily_returns < 0
            ]
        )

        if len(downside):

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

    if abs(max_dd) > 0:

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
    # ADDITIONAL TURNOVER METRICS
    # ========================================================

    if n:

        total_trades = n

        average_trade_days = (
            closed[
                "days_held"
            ].mean()
        )

        median_trade_days = (
            closed[
                "days_held"
            ].median()
        )

        total_net_pnl = (
            closed[
                "net_pnl_rs"
            ].sum()
        )

    else:

        total_trades = 0

        average_trade_days = 0

        median_trade_days = 0

        total_net_pnl = 0

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "strategy":
            "Top 10 RS Score | Price TT 7/7 + RS Line TT 7/7",

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

        "n_trades":
            int(n),

        "win_rate_gross":
            round(
                win_rate_gross,
                1
            ),

        "win_rate_net":
            round(
                win_rate_net,
                1
            ),

        "avg_gross_return_per_trade":
            round(
                avg_gross,
                2
            ),

        "avg_net_return_per_trade":
            round(
                avg_net,
                2
            ),

        "median_net_return_per_trade":
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

        "avg_winner_net":
            round(
                avg_winner_net,
                2
            ),

        "avg_loser_net":
            round(
                avg_loser_net,
                2
            ),

        "profit_factor_net":
            round(
                profit_factor_net,
                3
            ),

        "best_gross_trade":
            round(
                best_gross,
                2
            ),

        "worst_gross_trade":
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

        "total_costs_rs":
            round(
                total_costs_rs,
                0
            ),

        "total_stcg_tax_rs":
            round(
                total_tax_rs,
                0
            ),

        "total_net_realized_pnl_rs":
            round(
                total_net_pnl,
                0
            ),

        "average_trade_days":
            round(
                average_trade_days,
                1
            ),

        "median_trade_days":
            round(
                median_trade_days,
                1
            ),
    }


# ============================================================
# GOOGLE SHEETS OUTPUT
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
    # FALLBACK TO CSV
    # ========================================================

    if not sheet_id or not creds_json:

        print(
            "\nMissing SHEET_ID/"
            "GOOGLE_CREDENTIALS."
        )

        print(
            "Saving results to CSV."
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
    # SUMMARY SHEET
    # ========================================================

    summary_df = pd.DataFrame(
        [summary]
    )

    n_rows = (
        len(summary_df)
        +
        10
    )

    n_cols = (
        len(summary_df.columns)
        +
        2
    )

    try:

        sws = sh.worksheet(
            SUMMARY_WORKSHEET
        )

        if (
            sws.row_count < n_rows
            or
            sws.col_count < n_cols
        ):

            sws.resize(
                rows=max(
                    sws.row_count,
                    n_rows
                ),
                cols=max(
                    sws.col_count,
                    n_cols
                )
            )

    except gspread.WorksheetNotFound:

        sws = sh.add_worksheet(
            title=SUMMARY_WORKSHEET,
            rows=n_rows,
            cols=n_cols
        )

    sws.clear()

    sws.update(
        [[
            "EOD TOP-10 RS STRATEGY | "
            f"Run: {timestamp} | "
            f"Starting capital: "
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
    # MAIN BACKTEST SHEET
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
        else 0,

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
            ws.row_count < max_rows
            or
            ws.col_count < max_cols
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
            "PRIMARY STRATEGY: "
            "PRICE TT 7/7 + RS LINE TT 7/7 "
            "-> TOP 10 BY RS SCORE | "
            f"Run: {timestamp} | "
            "EOD execution | "
            "Green/Blue dots diagnostic only"
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
    # EQUITY CURVE
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
    # DAILY TOP-10 SELECTION
    # ========================================================

    selection_start = (
        equity_start
        +
        len(equity_df)
        +
        3
    )

    ws.update(
        [["DAILY TOP-10 RANKING / SIGNAL AUDIT"]],
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
        f"Primary backtest written to "
        f"'{BACKTEST_WORKSHEET}'"
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest_main():

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "RS SCREENER EOD TOP-10 BACKTEST"
    )

    print(
        "=" * 60
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
        f"Data cleaning        : "
        f"+/-{MAX_PLAUSIBLE_DAILY_MOVE * 100:.0f}%"
    )

    print(
        f"Liquidity             : "
        f"Price >= Rs.{MIN_PRICE}, "
        f"{VOLUME_LOOKBACK}D avg volume "
        f">= {MIN_AVG_VOLUME}"
    )

    print(
        f"Portfolio size        : "
        f"TOP {TOP_N}"
    )

    print(
        "Entry conditions      : "
        "Price TT 7/7 + RS Line TT 7/7"
    )

    print(
        "Ranking               : "
        "Raw RS Score descending"
    )

    print(
        "Exit                  : "
        "Stock leaves TOP 10"
    )

    print(
        "Green Dot             : "
        "Diagnostic only"
    )

    print(
        "Blue Dot              : "
        "Diagnostic only"
    )

    print(
        "Regime filter         : OFF"
    )

    print(
        "Rank hysteresis       : OFF"
    )

    print(
        "Secondary exits       : NONE"
    )

    print(
        "Execution             : "
        "EOD theoretical close"
    )

    print(
        f"Starting capital      : "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        "=" * 60
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
            i:i + batch_size
        ]

        print(
            f"\nDownloading batch "
            f"{i}-{i + len(batch)}..."
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

                if "Close" not in sdata.columns:
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

                # ------------------------------------------------
                # DATA CLEANING BEFORE SIGNALS
                # ------------------------------------------------

                close, n_bad = (
                    clean_price_series(
                        close
                    )
                )

                total_bad_points += (
                    n_bad
                )

                # ------------------------------------------------
                # SIGNAL CALCULATION
                # ------------------------------------------------

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
        "\nRunning EOD TOP-10 RS strategy..."
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
        "=" * 60
    )

    print(
        "FINAL BACKTEST RESULTS"
    )

    print(
        "=" * 60
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 60
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
        "\nBACKTEST COMPLETED."
    )


# ============================================================
# ENTRY POINT
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