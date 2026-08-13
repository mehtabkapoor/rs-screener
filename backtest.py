"""
RS SCREENER BACKTEST - FINAL SIMPLIFIED MODEL
==============================================

RULE SET
--------

Universe        : stocks.csv
Price filter    : Price > Rs.20
Liquidity       : 20-day avg volume > 100,000
Price TT        : 7/7 required
RS Line TT      : 7/7 required
Ranking         : raw RS Score descending
Portfolio       : Top 10
Weight          : Equal weight at entry
Rebalance       : Every trading day EOD

Blue Dot        : Diagnostic only
1Y RS cross     : Diagnostic only
Green Dot       : Diagnostic only

RS 5-EMA exit   : REMOVED
Price stop      : NONE
Rank buffer     : NONE

EXIT:
    Leaves current Top 10.
    Nothing else.

Costs:
    Buy:
        STT
        Stamp duty
        Exchange
        SEBI
        GST

    Sell:
        STT
        Exchange
        SEBI
        GST
        Rs.20 DP charge

STCG:
    20.8% effective on positive realized gain.

Equity:
    Daily mark-to-market.

Terminal value:
    1. Marked value
    2. Liquidation value

DATE LOGIC:
    BACKTEST_START = fixed
    BACKTEST_END   = automatically derived from the last
                      benchmark trading date actually returned.

DOWNLOAD LOGIC:
    No hardcoded download END date.
    Yahoo Finance is queried through latest available data.
"""


# ============================================================
# IMPORTS
# ============================================================

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

# Derived dynamically from benchmark.
BACKTEST_END = None


# ============================================================
# FILTERS
# ============================================================

MIN_PRICE = 20

MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20

LOOKBACK_DAYS = 250


# ============================================================
# DATA CLEANING
# ============================================================

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ============================================================
# PORTFOLIO
# ============================================================

TOP_N = 10

STARTING_CAPITAL = 1_000_000


# ============================================================
# COSTS
# ============================================================

STT_RATE = 0.001

STAMP_DUTY_RATE = 0.00015

EXCHANGE_CHARGE_RATE = 0.0000325

SEBI_CHARGE_RATE = 0.000001

GST_RATE = 0.18

DP_CHARGE_FLAT = 20


# ============================================================
# STCG
# ============================================================

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


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_index(index):
    """
    Converts Yahoo Finance dates into timezone-naive normalized
    daily timestamps.

    This prevents comparisons such as:

        timezone-aware >= timezone-naive

    from failing.
    """

    idx = pd.DatetimeIndex(index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    idx = idx.normalize()

    return idx


# ============================================================
# DOWNLOAD START
# ============================================================

def get_download_start():

    backtest_start = pd.Timestamp(
        BACKTEST_START
    )

    download_start = (
        backtest_start
        - pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    return download_start.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# LOAD TICKERS
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
# CLEAN PRICE SERIES
# ============================================================

def clean_price_series(close):

    close = close.copy()

    close.index = normalize_index(
        close.index
    )

    close = (
        close
        .sort_index()
        .loc[~close.index.duplicated(
            keep="last"
        )]
    )

    pct_change = (
        close.pct_change()
    )

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

    bad_indices = close.index[
        bad
    ]

    for idx in bad_indices:

        pos = (
            cleaned.index.get_loc(idx)
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
        STT_RATE
        * trade_value
    )

    stamp = (
        STAMP_DUTY_RATE
        * trade_value
    )

    exch = (
        EXCHANGE_CHARGE_RATE
        * trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        * trade_value
    )

    gst = (
        GST_RATE
        * (exch + sebi)
    )

    return (
        stt
        + stamp
        + exch
        + sebi
        + gst
    )


# ============================================================
# SELL COST
# ============================================================

def sell_side_cost(
    trade_value
):

    stt = (
        STT_RATE
        * trade_value
    )

    exch = (
        EXCHANGE_CHARGE_RATE
        * trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        * trade_value
    )

    gst = (
        GST_RATE
        * (exch + sebi)
    )

    return (
        stt
        + exch
        + sebi
        + gst
        + DP_CHARGE_FLAT
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
        net_gain
        * STCG_EFFECTIVE_RATE
    )


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():

    download_start = (
        get_download_start()
    )

    print()
    print(
        "Benchmark download:"
    )
    print(
        f"  Start   : {download_start}"
    )
    print(
        "  End     : latest available"
    )

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            print(
                f"Trying benchmark "
                f"{ticker}..."
            )

            data = yf.download(

                ticker,

                start=download_start,

                interval="1d",

                auto_adjust=True,

                progress=False,

                threads=False

            )

            if data is None:
                continue

            if data.empty:
                print(
                    f"{ticker}: empty data"
                )
                continue


            # ------------------------------------------------
            # Extract Close
            # ------------------------------------------------

            close = data["Close"]


            if isinstance(
                close,
                pd.DataFrame
            ):

                close = (
                    close.iloc[:, 0]
                )


            close = close.dropna()


            if close.empty:
                continue


            # ------------------------------------------------
            # Normalize dates
            # ------------------------------------------------

            close.index = (
                normalize_index(
                    close.index
                )
            )


            close = (
                close
                .sort_index()
                .loc[
                    ~close.index.duplicated(
                        keep="last"
                    )
                ]
            )


            # ------------------------------------------------
            # Clean prices
            # ------------------------------------------------

            close, n_bad = (
                clean_price_series(
                    close
                )
            )


            if n_bad:

                print(
                    f"{ticker}: repaired "
                    f"{n_bad} "
                    f"implausible "
                    f"data point(s)"
                )


            # ------------------------------------------------
            # DYNAMIC BACKTEST END
            # ------------------------------------------------

            global BACKTEST_END

            BACKTEST_END = (
                close.index[-1]
                .strftime("%Y-%m-%d")
            )


            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            print(
                f"Benchmark first date: "
                f"{close.index[0].date()}"
            )

            print(
                f"Benchmark last date: "
                f"{close.index[-1].date()}"
            )

            print(
                f"Derived BACKTEST_END: "
                f"{BACKTEST_END}"
            )

            return close


        except Exception as e:

            print(
                f"Benchmark "
                f"{ticker} failed:"
            )

            print(
                f"  {type(e).__name__}: "
                f"{e}"
            )


    raise RuntimeError(
        "Could not download any "
        "benchmark index data."
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
        sma150
        > sma200
    )

    c3 = (
        sma200
        > sma200_1mo
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

        + c2.astype(int)

        + c3.astype(int)

        + c4.astype(int)

        + c5.astype(int)

        + c6.astype(int)

        + c7.astype(int)

    )


    return (
        met == 7
    )


# ============================================================
# COMPUTE STOCK SIGNALS
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


    rs_ratio = (
        aligned["s"]
        / aligned["b"]
    )


    # --------------------------------------------------------
    # Return helper
    # --------------------------------------------------------

    def pct_return(
        series,
        days
    ):

        return (
            series
            / series.shift(days)
            - 1
        )


    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

    rs_score = (

        0.40
        * pct_return(
            aligned["s"],
            63
        )

        +

        0.20
        * pct_return(
            aligned["s"],
            126
        )

        +

        0.20
        * pct_return(
            aligned["s"],
            189
        )

        +

        0.20
        * pct_return(
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
        .rolling(
            LOOKBACK_DAYS
        )
        .max()
    )


    blue_dot = (
        rs_ratio
        > previous_rs_high
    )


    # --------------------------------------------------------
    # GREEN DOT
    # --------------------------------------------------------

    previous_price_high = (
        aligned["s"]
        .shift(1)
        .rolling(
            LOOKBACK_DAYS
        )
        .max()
    )


    price_at_new_high = (
        aligned["s"]
        > previous_price_high
    )


    green_dot = (
        blue_dot
        &
        (~price_at_new_high)
    )


    # --------------------------------------------------------
    # TREND TEMPLATES
    # --------------------------------------------------------

    tt_pass = (
        trend_template_series(
            aligned["s"]
        )
    )


    rs_tt_pass = (
        trend_template_series(
            rs_ratio
        )
    )


    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    rolling_avg_volume = (
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
            rolling_avg_volume
            > MIN_AVG_VOLUME
        )

    )


    return pd.DataFrame({

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
            green_dot,

    })


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


    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in trading_days:


        # ----------------------------------------------------
        # ELIGIBLE POOL
        # ----------------------------------------------------

        pool = []


        for sym, df in (
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
                        row["rs_score"]
                    )
                )
            )


        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )


        target_top10 = {
            sym
            for sym, _ in
            pool[:TOP_N]
        }


        # ----------------------------------------------------
        # EXITS
        # ----------------------------------------------------

        for sym in list(
            holdings.keys()
        ):


            if sym in target_top10:
                continue


            df = (
                all_signals[sym]
            )


            if date not in df.index:
                continue


            pos = (
                holdings.pop(sym)
            )


            exit_price = float(
                df.loc[
                    date,
                    "price"
                ]
            )


            gross_proceeds = (
                pos["qty"]
                * exit_price
            )


            s_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )


            net_proceeds = (
                gross_proceeds
                - s_cost
            )


            cost_basis = (

                pos["qty"]
                * pos["entry_price"]

                +

                pos["entry_cost"]

            )


            net_gain = (
                net_proceeds
                - cost_basis
            )


            tax = (
                stcg_tax(
                    net_gain
                )
            )


            cash += (
                net_proceeds
                - tax
            )


            gross_return_pct = round(

                (
                    exit_price
                    / pos["entry_price"]
                    - 1
                )
                * 100,

                2

            )


            net_return_pct = (

                round(

                    (
                        net_gain
                        - tax
                    )
                    / cost_basis
                    * 100,

                    2

                )

                if cost_basis > 0
                else 0

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
                    gross_return_pct,

                "buy_cost_rs":
                    round(
                        pos["entry_cost"],
                        2
                    ),

                "sell_cost_rs":
                    round(
                        s_cost,
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
                    net_return_pct,

                "days_held":
                    (
                        date
                        - pos["entry_date"]
                    ).days,

                "exit_reason":
                    "Left Top 10",

            })


        # ----------------------------------------------------
        # ENTRIES
        # ----------------------------------------------------

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            df = (
                all_signals[sym]
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
                    pos["entry_price"]
                )


            portfolio_value += (
                pos["qty"]
                * price
            )


        slots_open = (
            TOP_N
            - len(holdings)
        )


        if slots_open > 0:

            slot_capital = (
                portfolio_value
                / TOP_N
            )


            for sym in [
                s for s, _
                in pool[:TOP_N]
            ]:


                if slots_open <= 0:
                    break


                if sym in holdings:
                    continue


                price = float(
                    all_signals[sym]
                    .loc[
                        date,
                        "price"
                    ]
                )


                if price <= 0:
                    continue


                qty = int(
                    slot_capital
                    // price
                )


                if qty < 1:
                    continue


                trade_value = (
                    qty
                    * price
                )


                b_cost = (
                    buy_side_cost(
                        trade_value
                    )
                )


                total_cost = (
                    trade_value
                    + b_cost
                )


                if total_cost > cash:
                    continue


                cash -= (
                    total_cost
                )


                holdings[sym] = {

                    "qty":
                        qty,

                    "entry_price":
                        price,

                    "entry_date":
                        date,

                    "entry_cost":
                        b_cost,

                }


                slots_open -= 1


        # ----------------------------------------------------
        # MARK TO MARKET
        # ----------------------------------------------------

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            df = (
                all_signals[sym]
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
                    pos["entry_price"]
                )


            portfolio_value += (
                pos["qty"]
                * price
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
                    / STARTING_CAPITAL,
                    6
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
    # TERMINAL VALUES
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


    liquidation_cash = cash

    open_positions_detail = []


    if (
        len(trading_days) > 0
        and holdings
    ):

        last_date = (
            trading_days[-1]
        )


        for sym, pos in (
            holdings.items()
        ):

            df = (
                all_signals[sym]
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
                    pos["entry_price"]
                )


            gross_proceeds = (
                pos["qty"]
                * exit_price
            )


            s_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )


            net_proceeds = (
                gross_proceeds
                - s_cost
            )


            cost_basis = (

                pos["qty"]
                * pos["entry_price"]

                +

                pos["entry_cost"]

            )


            net_gain = (
                net_proceeds
                - cost_basis
            )


            tax = (
                stcg_tax(
                    net_gain
                )
            )


            liquidation_cash += (
                net_proceeds
                - tax
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

                "unrealized_gross_return_pct":
                    round(

                        (
                            exit_price
                            / pos["entry_price"]
                            - 1
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


    if not equity_df.empty:

        running_max = (
            equity_df["equity"]
            .cummax()
        )


        equity_df[
            "drawdown_pct"
        ] = (

            (
                equity_df["equity"]
                / running_max
                - 1
            )
            * 100

        ).round(3)


    open_df = pd.DataFrame(
        open_positions_detail
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


    net_total_return_marked_pct = round(

        (
            final_marked_value
            / STARTING_CAPITAL
            - 1
        )
        * 100,

        2

    )


    net_total_return_liquidation_pct = round(

        (
            final_liquidation_value
            / STARTING_CAPITAL
            - 1
        )
        * 100,

        2

    )


    running_max = (
        equity_df["equity"]
        .cummax()
    )


    drawdown = (

        equity_df["equity"]
        / running_max
        - 1

    ) * 100


    max_dd = round(
        drawdown.min(),
        2
    )


    if not trade_df.empty:

        closed = trade_df[
            trade_df["exit_reason"]
            == "Left Top 10"
        ]

    else:

        closed = trade_df


    n = len(closed)


    if n:

        win_rate_net = round(

            (
                closed[
                    "net_return_pct"
                ]
                > 0
            ).mean()
            * 100,

            1

        )


        win_rate_gross = round(

            (
                closed[
                    "gross_return_pct"
                ]
                > 0
            ).mean()
            * 100,

            1

        )


        avg_gross = round(
            closed[
                "gross_return_pct"
            ].mean(),
            2
        )


        avg_net = round(
            closed[
                "net_return_pct"
            ].mean(),
            2
        )


        median_net = round(
            closed[
                "net_return_pct"
            ].median(),
            2
        )


        avg_days = round(
            closed[
                "days_held"
            ].mean(),
            1
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


        total_costs_rs = round(

            (
                closed[
                    "buy_cost_rs"
                ]

                +

                closed[
                    "sell_cost_rs"
                ]

            ).sum(),

            0

        )


        total_tax_rs = round(

            closed[
                "stcg_tax_rs"
            ].sum(),

            0

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


        if len(winners):

            avg_winner = round(

                winners[
                    "net_return_pct"
                ].mean(),

                2

            )

        else:

            avg_winner = 0


        if len(losers):

            avg_loser = round(

                losers[
                    "net_return_pct"
                ].mean(),

                2

            )

        else:

            avg_loser = 0


        if len(winners):

            gp = (
                winners[
                    "net_pnl_rs"
                ].sum()
            )

        else:

            gp = 0


        if len(losers):

            gl = abs(

                losers[
                    "net_pnl_rs"
                ].sum()

            )

        else:

            gl = 0


        if gl > 0:

            profit_factor = round(
                gp / gl,
                3
            )

        else:

            profit_factor = 0


    else:

        win_rate_net = 0

        win_rate_gross = 0

        avg_gross = 0

        avg_net = 0

        median_net = 0

        avg_days = 0

        best_gross = 0

        worst_gross = 0

        total_costs_rs = 0

        total_tax_rs = 0

        avg_winner = 0

        avg_loser = 0

        profit_factor = 0


    # ========================================================
    # RISK METRICS
    # ========================================================

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

        n_days = (
            len(equity_df)
        )


        annualized_return = (

            equity_df["equity"].iloc[-1]
            **
            (
                252
                / max(
                    n_days,
                    1
                )
            )

            - 1

        )


        annualized_volatility = (

            daily_std
            * np.sqrt(252)

        )


        if daily_std > 0:

            sharpe = (

                daily_mean
                / daily_std
                * np.sqrt(252)

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

        else:

            downside_std = 0


        if downside_std > 0:

            sortino = (

                daily_mean
                / downside_std
                * np.sqrt(252)

            )

        else:

            sortino = 0


    else:

        annualized_return = 0

        annualized_volatility = 0

        sharpe = 0

        sortino = 0


    if abs(max_dd) > 0:

        calmar = round(

            annualized_return
            / abs(max_dd / 100),

            3

        )

    else:

        calmar = 0


    return {

        "Backtest Start":
            BACKTEST_START,

        "Backtest End":
            BACKTEST_END,

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
            net_total_return_marked_pct,

        "Net Return - liquidation (%)":
            net_total_return_liquidation_pct,

        "Annualized Return (%)":
            round(
                annualized_return
                * 100,
                2
            ),

        "Annualized Volatility (%)":
            round(
                annualized_volatility
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
            calmar,

        "Max Drawdown (%)":
            max_dd,

        "Number of Closed Trades":
            n,

        "Win Rate - Gross (%)":
            win_rate_gross,

        "Win Rate - Net (%)":
            win_rate_net,

        "Avg Gross Return/Trade (%)":
            avg_gross,

        "Avg Net Return/Trade (%)":
            avg_net,

        "Median Net Return/Trade (%)":
            median_net,

        "Avg Days Held":
            avg_days,

        "Avg Winner (%)":
            avg_winner,

        "Avg Loser (%)":
            avg_loser,

        "Profit Factor (net)":
            profit_factor,

        "Best Gross Trade (%)":
            best_gross,

        "Worst Gross Trade (%)":
            worst_gross,

        "Total Costs Paid (Rs)":
            total_costs_rs,

        "Total STCG Tax Paid (Rs)":
            total_tax_rs,

    }


# ============================================================
# GOOGLE SHEETS CHUNK WRITER
# ============================================================

def write_in_chunks(
    ws,
    all_rows,
    start_row,
    chunk_size,
    label
):

    total = len(
        all_rows
    )


    if total == 0:
        return


    for i in range(
        0,
        total,
        chunk_size
    ):

        chunk = all_rows[
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
                f"{i}-{i+len(chunk)}, "
                f"retrying once: {e}"
            )


            time.sleep(5)


            try:

                ws.update(
                    chunk,
                    f"A{row_start}"
                )


            except Exception as e2:

                print(
                    f"RETRY FAILED for "
                    f"{label}: {e2}"
                )

                raise


        print(
            f"Wrote {label}: "
            f"{min(i + chunk_size, total)}"
            f"/{total} rows"
        )


# ============================================================
# REMOVE CHARTS
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
                != sheet_id
            ):
                continue


            for chart in sheet.get(
                "charts",
                []
            ):

                requests.append({

                    "deleteEmbeddedObject": {

                        "objectId":
                            chart["chartId"]

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
            "Could not remove "
            f"existing charts: {e}"
        )


# ============================================================
# ADD CHARTS
# ============================================================

def add_charts(
    sh,
    sheet_id,
    equity_header_row_0idx,
    n_equity_rows
):

    data_end_row = (
        equity_header_row_0idx
        + 1
        + n_equity_rows
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
            "Equity and drawdown "
            "charts added."
        )


    except Exception as e:

        print(
            "Could not add charts "
            f"(non-fatal): {e}"
        )


# ============================================================
# WRITE TO GOOGLE SHEETS
# ============================================================

def write_to_sheet(
    trade_df,
    equity_df,
    open_df,
    summary
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )


    if (
        not sheet_id
        or not creds_json
    ):

        print(
            "Missing "
            "SHEET_ID/"
            "GOOGLE_CREDENTIALS."
        )

        print(
            "Saving CSV files instead."
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


    n_rows_needed = (

        len(trade_df)

        +

        len(equity_df)

        +

        len(open_df)

        +

        len(summary)

        +

        60

    )


    n_cols_needed = 15


    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
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

            title=
                BACKTEST_WORKSHEET,

            rows=
                n_rows_needed,

            cols=
                n_cols_needed

        )


    remove_existing_charts(
        sh,
        ws.id
    )


    ws.clear()


    ws.update(

        [[

            f"FINAL MODEL BACKTEST | "
            f"run {timestamp} | "
            f"NET of costs+STCG tax | "
            f"Starting capital: "
            f"Rs.{STARTING_CAPITAL:,.0f} | "
            f"data-cleaned | "
            f"Window: "
            f"{BACKTEST_START} "
            f"to "
            f"{BACKTEST_END}"

        ]],

        "A1"

    )


    summary_rows = (

        [
            [
                "Summary",
                ""
            ]
        ]

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


    trade_start_row = (
        3
        + len(summary_rows)
        + 2
    )


    ws.update(
        [["Trade Log"]],
        f"A{trade_start_row}"
    )


    trade_header_row = (
        trade_start_row + 1
    )


    if not trade_df.empty:

        write_in_chunks(

            ws,

            [
                list(
                    trade_df.columns
                )
            ]
            +
            trade_df.values.tolist(),

            start_row=
                trade_header_row,

            chunk_size=
                2000,

            label=
                "trade log"

        )


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
        open_start_row + 1
    )


    if not open_df.empty:

        ws.update(

            [
                list(
                    open_df.columns
                )
            ]
            +
            open_df.values.tolist(),

            f"A{open_header_row}"

        )


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
        equity_start_row + 1
    )


    if not equity_df.empty:

        write_in_chunks(

            ws,

            [
                list(
                    equity_df.columns
                )
            ]
            +
            equity_df.values.tolist(),

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


    print()
    print(
        f"Backtest results written "
        f"to '{BACKTEST_WORKSHEET}' tab:"
    )

    print(
        f"  Trades          : "
        f"{len(trade_df)}"
    )

    print(
        f"  Trading days    : "
        f"{len(equity_df)}"
    )

    print(
        f"  Open positions  : "
        f"{len(open_df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD UNIVERSE
    # --------------------------------------------------------

    tickers = load_tickers()


    print()
    print(
        f"Loaded "
        f"{len(tickers)} tickers."
    )


    # --------------------------------------------------------
    # DOWNLOAD START
    # --------------------------------------------------------

    download_start = (
        get_download_start()
    )


    print()
    print(
        "=" * 60
    )

    print(
        f"Download start   : "
        f"{download_start}"
    )

    print(
        f"Backtest start   : "
        f"{BACKTEST_START}"
    )

    print(
        "Backtest end     : "
        "DYNAMIC"
    )

    print(
        f"Price filter     : "
        f"> Rs.{MIN_PRICE}"
    )

    print(
        f"Liquidity filter : "
        f"{VOLUME_LOOKBACK}d avg "
        f"volume > "
        f"{MIN_AVG_VOLUME:,}"
    )

    print(
        f"Portfolio        : "
        f"Top {TOP_N}"
    )

    print(
        "Rebalance        : "
        "Every trading day"
    )

    print(
        "Exit             : "
        "Leaves Top 10"
    )

    print(
        f"Starting capital : "
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
    # SAFETY CHECK
    # ========================================================

    if BACKTEST_END is None:

        raise RuntimeError(
            "BACKTEST_END was not "
            "derived from benchmark."
        )


    backtest_start_ts = (
        pd.Timestamp(
            BACKTEST_START
        )
    )


    backtest_end_ts = (
        pd.Timestamp(
            BACKTEST_END
        )
    )


    if (
        backtest_end_ts
        < backtest_start_ts
    ):

        raise RuntimeError(

            "Invalid backtest dates: "
            f"{BACKTEST_START} -> "
            f"{BACKTEST_END}"

        )


    print()
    print(
        "FINAL BACKTEST WINDOW"
    )

    print(
        f"  Start: "
        f"{BACKTEST_START}"
    )

    print(
        f"  End  : "
        f"{BACKTEST_END}"
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


        print()
        print(
            f"Downloading batch "
            f"{i}-{i + len(batch)} "
            f"through latest available..."
        )


        try:

            # IMPORTANT:
            # NO END PARAMETER.

            data = yf.download(

                batch,

                start=
                    download_start,

                interval=
                    "1d",

                auto_adjust=
                    True,

                progress=
                    False,

                group_by=
                    "ticker",

                threads=
                    True

            )


        except Exception as e:

            print(
                f"Batch download failed:"
            )

            print(
                f"  {type(e).__name__}: "
                f"{e}"
            )

            continue


        if data is None:
            continue


        if data.empty:

            print(
                "Batch returned no data."
            )

            continue


        # ====================================================
        # PROCESS EACH STOCK
        # ====================================================

        for symbol in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    if (
                        not isinstance(
                            data.columns,
                            pd.MultiIndex
                        )
                    ):

                        print(
                            f"{symbol}: "
                            "Unexpected Yahoo "
                            "column structure."
                        )

                        continue


                    level0 = (
                        data.columns
                        .get_level_values(0)
                    )


                    if symbol not in level0:

                        continue


                    sdata = data[
                        symbol
                    ]


                if (
                    "Close"
                    not in sdata.columns
                ):

                    continue


                close = (
                    sdata["Close"]
                    .dropna()
                    .copy()
                )


                if close.empty:
                    continue


                close.index = (
                    normalize_index(
                        close.index
                    )
                )


                close = (
                    close
                    .sort_index()
                    .loc[
                        ~close.index
                        .duplicated(
                            keep="last"
                        )
                    ]
                )


                # ------------------------------------------------
                # Volume
                # ------------------------------------------------

                if "Volume" in (
                    sdata.columns
                ):

                    volume = (
                        sdata["Volume"]
                        .copy()
                    )

                    volume.index = (
                        normalize_index(
                            volume.index
                        )
                    )

                    volume = (
                        volume
                        .sort_index()
                        .loc[
                            ~volume.index
                            .duplicated(
                                keep="last"
                            )
                        ]
                    )

                    volume = (
                        volume
                        .reindex(
                            close.index
                        )
                        .fillna(0)
                    )

                else:

                    volume = pd.Series(
                        0,
                        index=close.index
                    )


                # ------------------------------------------------
                # Clean price
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
                # Signals
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

    print()
    print(
        "=" * 60
    )

    print(
        f"Signals computed: "
        f"{len(all_signals)} stocks"
    )

    print(
        f"Total repaired data points: "
        f"{total_bad_points}"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # TRADING DAYS
    # ========================================================

    trading_days = (
        bench_close.index[
            (
                bench_close.index
                >=
                backtest_start_ts
            )
            &
            (
                bench_close.index
                <=
                backtest_end_ts
            )
        ]
    )


    if len(trading_days) == 0:

        raise RuntimeError(

            "No trading days found "
            f"between "
            f"{BACKTEST_START} "
            f"and "
            f"{BACKTEST_END}."

        )


    print()
    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    print(
        f"First trading day: "
        f"{trading_days[0].date()}"
    )

    print(
        f"Last trading day: "
        f"{trading_days[-1].date()}"
    )


    # ========================================================
    # RUN BACKTEST
    # ========================================================

    (
        trade_df,
        equity_df,
        open_df,
        final_marked,
        final_liq

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

        final_liq

    )


    print()
    print(
        "=" * 60
    )

    print(
        "--- SUMMARY ---"
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


    # ========================================================
    # GOOGLE SHEETS
    # ========================================================

    write_to_sheet(

        trade_df,

        equity_df,

        open_df,

        summary

    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

        print()
        print(
            "BACKTEST COMPLETED "
            "SUCCESSFULLY."
        )


    except Exception as e:

        print()
        print(
            "BACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: "
            f"{e}"
        )

        raise