"""
RS Screener Backtest - SIMPLIFIED MODEL - CORRECTED

RULES
-----
Universe        : stocks.csv
Price filter    : Price > Rs.20
Liquidity       : 20-day avg volume > 100,000
Price TT        : 7/7 required
RS Line TT      : 7/7 required
Ranking         : raw RS Score, descending
Portfolio       : Top 10
Weight          : Equal weight at entry
Rebalance       : Every trading day EOD
Exit rule       : Sell when rank among eligible universe > 15

IMPORTANT PORTFOLIO LOGIC
--------------------------
After exits, the portfolio is immediately refilled from the
current Top 10 eligible stocks.

A stock is NOT bought merely because a slot is empty.

The replacement must be:
    1. eligible
    2. inside today's Top 10
    3. not already held

Available cash is allocated across vacant target positions.

If fewer than 10 eligible stocks exist, fewer than 10 holdings
are allowed.

If sufficient cash exists but integer-share rounding prevents
a purchase, the engine attempts to use the remaining available
cash rather than abandoning the slot.

Costs
-----
Buy + sell transaction costs
STCG = 20.8% effective on positive realized gains

Equity
------
Daily mark-to-market.

Terminal value
--------------
Marked AND liquidation.

Charts
------
Equity Curve + Drawdown
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

# None = latest available market data
BACKTEST_END = None

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250

MAX_PLAUSIBLE_DAILY_MOVE = 0.30

TOP_N = 10

# Exit only when rank becomes worse than this.
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

STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():

    backtest_start = pd.Timestamp(BACKTEST_START)

    download_start = (
        backtest_start -
        pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime("%Y-%m-%d"),
            None
        )

    backtest_end = pd.Timestamp(BACKTEST_END)

    download_end = (
        backtest_end +
        pd.Timedelta(days=1)
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
# CLEAN PRICE DATA
# ============================================================

def clean_price_series(close):

    close = (
        close
        .copy()
        .sort_index()
    )

    pct_change = close.pct_change()

    bad = (
        pct_change.abs() >
        MAX_PLAUSIBLE_DAILY_MOVE
    )

    n_bad = int(
        bad.sum()
    )

    if n_bad == 0:

        return close, 0

    cleaned = close.copy()

    for idx in close.index[bad]:

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
# DOWNLOAD BENCHMARK
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        f"\nBenchmark download: "
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
                    f"Benchmark {ticker}: "
                    f"repaired {n_bad} "
                    f"implausible data point(s)"
                )

            print(
                f"Benchmark loaded: {ticker}"
            )

            print(
                f"Latest benchmark date: "
                f"{close.index.max().strftime('%Y-%m-%d')}"
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
# STOCK SIGNAL CALCULATION
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
        .reindex(aligned.index)
    )

    # --------------------------------------------------------
    # RS RATIO
    # --------------------------------------------------------

    rs_ratio = (
        aligned["s"] /
        aligned["b"]
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    def pct_return(
        series,
        days
    ):

        return (
            series /
            series.shift(days) -
            1
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
    # TREND TEMPLATE
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

    rolling_avg_volume = (
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

        rolling_avg_volume >
        MIN_AVG_VOLUME
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

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
# MARK PORTFOLIO
# ============================================================

def mark_portfolio(
    cash,
    holdings,
    all_signals,
    date
):

    value = cash

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

            price = float(
                pos["entry_price"]
            )

        value += (
            pos["qty"] *
            price
        )

    return value


# ============================================================
# BACKTEST ENGINE
# ============================================================

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

    portfolio_diagnostics = []


    # ========================================================
    # EACH TRADING DAY
    # ========================================================

    for date in trading_days:

        # ----------------------------------------------------
        # BUILD ELIGIBLE UNIVERSE
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
        # RANK ELIGIBLE UNIVERSE
        # ----------------------------------------------------

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        rank_lookup = {

            sym:
            rank + 1

            for rank, (sym, _) in
            enumerate(pool)
        }

        # ----------------------------------------------------
        # TARGET TOP 10
        # ----------------------------------------------------

        target_topN = [
            sym
            for sym, _ in
            pool[:TOP_N]
        ]

        target_set = set(
            target_topN
        )

        # ----------------------------------------------------
        # EXIT
        #
        # IMPORTANT:
        # Exit only if:
        #   rank > 15
        # OR
        #   stock disappeared from eligible universe.
        # ----------------------------------------------------

        exited_today = []

        for sym in list(
            holdings.keys()
        ):

            rank = (
                rank_lookup.get(sym)
            )

            if (
                rank is not None
                and rank <= EXIT_RANK
            ):

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
                pos["qty"] *
                exit_price
            )

            s_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds -
                s_cost
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

            tax = (
                stcg_tax(
                    net_gain
                )
            )

            cash += (
                net_proceeds -
                tax
            )

            gross_return_pct = (

                (
                    exit_price /
                    pos["entry_price"] -
                    1
                ) * 100
            )

            net_return_pct = (

                (
                    net_gain -
                    tax
                ) /
                cost_basis *
                100

                if cost_basis > 0

                else 0
            )

            if rank is None:

                exit_reason = (
                    "Left eligible universe"
                )

            else:

                exit_reason = (
                    f"Rank > {EXIT_RANK}"
                )

            trade_log.append({

                "symbol":
                sym,

                "entry_date":
                pos["entry_date"].strftime(
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
                round(
                    net_return_pct,
                    2
                ),

                "days_held":
                (
                    date -
                    pos["entry_date"]
                ).days,

                "exit_reason":
                exit_reason,

                "exit_rank":
                rank
                if rank is not None
                else "",

            })

            exited_today.append(
                sym
            )

        # ----------------------------------------------------
        # CURRENT PORTFOLIO VALUE AFTER EXITS
        # ----------------------------------------------------

        portfolio_value = (
            mark_portfolio(
                cash,
                holdings,
                all_signals,
                date
            )
        )

        # ----------------------------------------------------
        # VACANT TARGET SLOTS
        #
        # Only current Top 10 stocks can fill them.
        # ----------------------------------------------------

        candidates = [

            sym

            for sym in target_topN

            if sym not in holdings
        ]

        slots_open = (
            TOP_N -
            len(holdings)
        )

        slots_open = max(
            0,
            slots_open
        )

        # ----------------------------------------------------
        # REFILL
        #
        # Allocate the available capital equally among
        # the vacant target slots.
        #
        # This is the critical correction.
        # ----------------------------------------------------

        if (
            slots_open > 0
            and len(candidates) > 0
            and cash > 0
        ):

            # Number of actual candidates that can fill slots
            n_to_buy = min(
                slots_open,
                len(candidates)
            )

            # Divide currently available cash among
            # remaining target slots.
            capital_per_slot = (
                cash /
                n_to_buy
            )

            for sym in candidates:

                if (
                    len(holdings) >=
                    TOP_N
                ):

                    break

                df = all_signals[sym]

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

                # --------------------------------------------
                # Initial equal-weight target
                # --------------------------------------------

                qty = int(
                    capital_per_slot //
                    price
                )

                # --------------------------------------------
                # If that doesn't work because of rounding,
                # use whatever cash can actually buy.
                # --------------------------------------------

                if qty < 1:

                    # Maximum affordable quantity
                    # accounting approximately for buy costs.
                    qty = int(
                        cash //
                        (
                            price *
                            (
                                1 +
                                STT_RATE +
                                STAMP_DUTY_RATE +
                                EXCHANGE_CHARGE_RATE +
                                SEBI_CHARGE_RATE +
                                GST_RATE *
                                (
                                    EXCHANGE_CHARGE_RATE +
                                    SEBI_CHARGE_RATE
                                )
                            )
                        )
                    )

                if qty < 1:

                    continue

                trade_value = (
                    qty *
                    price
                )

                b_cost = (
                    buy_side_cost(
                        trade_value
                    )
                )

                total_cost = (
                    trade_value +
                    b_cost
                )

                # --------------------------------------------
                # Never spend more than available cash.
                # --------------------------------------------

                if total_cost > cash:

                    # Reduce quantity until affordable.
                    max_qty = int(
                        qty
                    )

                    while (
                        max_qty > 0
                    ):

                        test_value = (
                            max_qty *
                            price
                        )

                        test_cost = (
                            test_value +
                            buy_side_cost(
                                test_value
                            )
                        )

                        if (
                            test_cost <=
                            cash
                        ):

                            break

                        max_qty -= 1

                    qty = max_qty

                    if qty < 1:

                        continue

                    trade_value = (
                        qty *
                        price
                    )

                    b_cost = (
                        buy_side_cost(
                            trade_value
                        )
                    )

                    total_cost = (
                        trade_value +
                        b_cost
                    )

                # --------------------------------------------
                # EXECUTE BUY
                # --------------------------------------------

                cash -= total_cost

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

                # --------------------------------------------
                # Recalculate remaining slots and capital.
                # --------------------------------------------

                remaining_slots = (
                    TOP_N -
                    len(holdings)
                )

                if remaining_slots > 0:

                    capital_per_slot = (
                        cash /
                        remaining_slots
                    )

        # ----------------------------------------------------
        # FINAL DAILY MARK-TO-MARKET
        # ----------------------------------------------------

        portfolio_value = (
            mark_portfolio(
                cash,
                holdings,
                all_signals,
                date
            )
        )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        held_target_count = sum(
            1
            for sym in holdings
            if sym in target_set
        )

        eligible_count = len(
            pool
        )

        portfolio_diagnostics.append({

            "date":
            date.strftime(
                "%Y-%m-%d"
            ),

            "eligible_stocks":
            eligible_count,

            "top10_count":
            len(target_topN),

            "n_holdings":
            len(holdings),

            "cash_rs":
            round(
                cash,
                2
            ),

            "portfolio_value_rs":
            round(
                portfolio_value,
                2
            ),

            "exits_today":
            len(exited_today),

            "entries_today":
            max(
                0,
                len(holdings)
                -
                (
                    len(holdings)
                    if not exited_today
                    else 0
                )
            ),

            "target_holdings_held":
            held_target_count,

        })

        # ----------------------------------------------------
        # EQUITY CURVE
        # ----------------------------------------------------

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
                portfolio_value /
                STARTING_CAPITAL,
                6
            ),

            "cash_rs":
            round(
                cash,
                2
            ),

            "n_holdings":
            len(holdings),

            "eligible_stocks":
            eligible_count,

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

    liquidation_cash = cash

    open_positions_detail = []

    if (
        len(trading_days)
        and holdings
    ):

        last_date = (
            trading_days[-1]
        )

        for sym, pos in (
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

                exit_price = (
                    pos["entry_price"]
                )

            gross_proceeds = (
                pos["qty"] *
                exit_price
            )

            s_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds -
                s_cost
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

            tax = (
                stcg_tax(
                    net_gain
                )
            )

            liquidation_cash += (
                net_proceeds -
                tax
            )

            open_positions_detail.append({

                "symbol":
                sym,

                "entry_date":
                pos["entry_date"].strftime(
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
                        exit_price /
                        pos["entry_price"] -
                        1
                    ) * 100,
                    2
                ),

            })

    final_liquidation_value = (
        liquidation_cash
    )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    trade_df = pd.DataFrame(
        trade_log
    )

    equity_df = pd.DataFrame(
        equity_curve
    )

    diagnostics_df = pd.DataFrame(
        portfolio_diagnostics
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

    open_df = pd.DataFrame(
        open_positions_detail
    )

    return (
        trade_df,
        equity_df,
        open_df,
        diagnostics_df,
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
            final_marked_value /
            STARTING_CAPITAL -
            1
        ) * 100,

        2
    )

    net_total_return_liquidation_pct = round(

        (
            final_liquidation_value /
            STARTING_CAPITAL -
            1
        ) * 100,

        2
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

    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    if not trade_df.empty:

        closed = trade_df

        n = len(closed)

        win_rate_net = round(

            (
                closed[
                    "net_return_pct"
                ] > 0
            ).mean() * 100,

            1
        )

        win_rate_gross = round(

            (
                closed[
                    "gross_return_pct"
                ] > 0
            ).mean() * 100,

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

        avg_winner = (

            round(
                winners[
                    "net_return_pct"
                ].mean(),
                2
            )

            if len(winners)

            else 0
        )

        avg_loser = (

            round(
                losers[
                    "net_return_pct"
                ].mean(),
                2
            )

            if len(losers)

            else 0
        )

        gp = (

            winners[
                "net_pnl_rs"
            ].sum()

            if len(winners)

            else 0
        )

        gl = (

            abs(
                losers[
                    "net_pnl_rs"
                ].sum()
            )

            if len(losers)

            else 0
        )

        profit_factor = (

            round(
                gp / gl,
                3
            )

            if gl > 0

            else 0
        )

    else:

        n = 0
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
    # DAILY STATISTICS
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

        n_days = len(
            equity_df
        )

        annualized_return = (

            equity_df["equity"].iloc[-1]
            **
            (
                252 /
                max(
                    n_days,
                    1
                )
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
            )
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

            (
                daily_mean /
                downside_std
            )
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

        round(
            annualized_return /
            abs(
                max_dd / 100
            ),
            3
        )

        if abs(max_dd) > 0

        else 0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    return {

        "Backtest Start":
        BACKTEST_START,

        "Backtest End":
        equity_df["date"].iloc[-1]
        if not equity_df.empty
        else "",

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
        net_total_return_marked_pct,

        "Net Return - liquidation (%)":
        net_total_return_liquidation_pct,

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

    total = len(all_rows)

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
                f"{i}-{i + len(chunk)}, "
                f"retrying once: {e}"
            )

            time.sleep(5)

            ws.update(
                chunk,
                f"A{row_start}"
            )

        print(
            f"Wrote {label}: "
            f"{min(i + chunk_size, total)}"
            f"/{total} rows"
        )


# ============================================================
# REMOVE EXISTING CHARTS
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
            "Could not check/remove "
            f"existing charts: {e}"
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
        equity_header_row_0idx +
        1 +
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
                                                    1,

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
                                                    y_col_idx + 1,

                                                }

                                            ]

                                        }

                                    },

                                    "targetAxis":
                                    "LEFT_AXIS",

                                }

                            ],

                        },

                    },

                    "position": {

                        "overlayPosition": {

                            "anchorCell": {

                                "sheetId":
                                sheet_id,

                                "rowIndex":
                                anchor_row,

                                "columnIndex":
                                8,

                            },

                            "widthPixels":
                            650,

                            "heightPixels":
                            380,

                        }

                    },

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
        ),

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
            "Could not add charts: "
            f"{e}"
        )


# ============================================================
# WRITE RESULTS
# ============================================================

def write_to_sheet(
    trade_df,
    equity_df,
    open_df,
    diagnostics_df,
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
        or not creds_json
    ):

        print(
            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS -- "
            "saving to CSV."
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

        diagnostics_df.to_csv(
            "backtest_diagnostics.csv",
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

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    # --------------------------------------------------------
    # SHEET
    # --------------------------------------------------------

    n_rows_needed = (

        len(trade_df)
        +
        len(equity_df)
        +
        len(open_df)
        +
        len(diagnostics_df)
        +
        len(summary)
        +
        100

    )

    n_cols_needed = 16

    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )

        if (
            ws.row_count <
            n_rows_needed
            or
            ws.col_count <
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

            "CORRECTED TOP-10 BACKTEST | "
            f"run {timestamp} | "
            "NET of costs+STCG | "
            f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
            "Entry: Price TT + RS TT | "
            f"Exit: rank > {EXIT_RANK} | "
            "Vacancies refilled from current Top 10 | "
            f"Window: {BACKTEST_START} "
            f"to {effective_end_str}"

        ]],

        "A1"

    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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

    trade_start_row = (
        3 +
        len(summary_rows) +
        2
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
            "Backtest End"
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

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    diag_start_row = (
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
        [["Portfolio Diagnostics"]],
        f"A{diag_start_row}"
    )

    diag_header_row = (
        diag_start_row + 1
    )

    if not diagnostics_df.empty:

        write_in_chunks(

            ws,

            [
                list(
                    diagnostics_df.columns
                )
            ]
            +
            diagnostics_df.values.tolist(),

            start_row=
            diag_header_row,

            chunk_size=
            2000,

            label=
            "diagnostics"

        )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity_start_row = (
        diag_header_row
        +
        max(
            len(diagnostics_df),
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

    print(

        f"\nBacktest results written "
        f"to '{BACKTEST_WORKSHEET}' tab: "

        f"{len(trade_df)} trades, "

        f"{len(equity_df)} trading days, "

        f"{len(open_df)} open positions."

    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD UNIVERSE
    # --------------------------------------------------------

    tickers = load_tickers()

    print(
        f"\nLoaded "
        f"{len(tickers)} tickers."
    )

    # --------------------------------------------------------
    # DOWNLOAD DATES
    # --------------------------------------------------------

    download_start, download_end = (
        get_download_dates()
    )

    print("=" * 60)

    print(
        f"Download start   : "
        f"{download_start}"
    )

    print(
        f"Download end     : "
        f"{download_end if download_end else 'LATEST AVAILABLE'}"
    )

    print(
        f"Backtest start   : "
        f"{BACKTEST_START}"
    )

    print(
        f"Backtest end     : "
        f"{BACKTEST_END if BACKTEST_END else 'LATEST AVAILABLE'}"
    )

    print(
        f"Price filter     : "
        f"> Rs.{MIN_PRICE}"
    )

    print(
        f"Liquidity filter : "
        f"{VOLUME_LOOKBACK}d avg volume "
        f"> {MIN_AVG_VOLUME:,}"
    )

    print(
        "Entry filter     : "
        "Price TT + RS TT"
    )

    print(
        f"Portfolio        : "
        f"Top {TOP_N}, equal weight at entry"
    )

    print(
        f"Exit             : "
        f"rank > {EXIT_RANK}"
    )

    print(
        "Vacancy refill   : "
        "Current Top 10"
    )

    print(
        f"Starting capital : "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    bench_close = (
        download_benchmark()
    )

    # --------------------------------------------------------
    # STOCK DATA
    # --------------------------------------------------------

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

                start=
                download_start,

                end=
                download_end,

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
                f"Batch download failed: "
                f"{e}"
            )

            continue

        # ----------------------------------------------------
        # PROCESS STOCKS
        # ----------------------------------------------------

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

                    all_signals[
                        symbol.replace(
                            ".NS",
                            ""
                        )
                    ] = sig

            except Exception as e:

                print(
                    f"Skipping {symbol}: "
                    f"{e}"
                )

                continue

        time.sleep(1)

    # --------------------------------------------------------
    # SIGNAL SUMMARY
    # --------------------------------------------------------

    print(
        f"\nSignals computed for "
        f"{len(all_signals)} stocks."
    )

    print(
        f"Total data points repaired: "
        f"{total_bad_points}"
    )

    # --------------------------------------------------------
    # LATEST DATA
    # --------------------------------------------------------

    if all_signals:

        latest_stock_date = max(

            df.index.max()

            for df in
            all_signals.values()

        )

    else:

        latest_stock_date = None

    benchmark_latest_date = (
        pd.Timestamp(
            bench_close.index.max()
        )
    )

    # --------------------------------------------------------
    # EFFECTIVE END
    # --------------------------------------------------------

    if BACKTEST_END is None:

        effective_end = (
            benchmark_latest_date
        )

    else:

        effective_end = min(

            pd.Timestamp(
                BACKTEST_END
            ),

            benchmark_latest_date

        )

    print(
        "\nLatest available "
        "benchmark data: "
        f"{benchmark_latest_date.strftime('%Y-%m-%d')}"
    )

    if latest_stock_date is not None:

        print(
            "Latest available "
            "stock data: "
            f"{pd.Timestamp(latest_stock_date).strftime('%Y-%m-%d')}"
        )

    print(
        "Effective backtest "
        "end date: "
        f"{effective_end.strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # LATEST FILTER FUNNEL
    # --------------------------------------------------------

    if all_signals:

        latest_date = max(

            df.index.max()

            for df in
            all_signals.values()

        )

        counts = {

            "liquid":
            0,

            "+tt_pass":
            0,

            "+rs_tt_pass":
            0

        }

        for sym, df in (
            all_signals.items()
        ):

            if (
                latest_date
                not in
                df.index
            ):

                continue

            row = df.loc[
                latest_date
            ]

            if pd.isna(
                row["rs_score"]
            ):

                continue

            if not bool(
                row["liquid"]
            ):

                continue

            counts["liquid"] += 1

            if not bool(
                row["tt_pass"]
            ):

                continue

            counts["+tt_pass"] += 1

            if not bool(
                row["rs_tt_pass"]
            ):

                continue

            counts["+rs_tt_pass"] += 1

        print(
            f"\nFilter funnel on "
            f"{latest_date.strftime('%Y-%m-%d')}:"
        )

        for k, v in counts.items():

            print(
                f"  {k}: {v}"
            )

    # --------------------------------------------------------
    # TRADING DAYS
    # --------------------------------------------------------

    trading_days = (
        bench_close.index[

            (
                bench_close.index >=
                pd.Timestamp(
                    BACKTEST_START
                )
            )

            &

            (
                bench_close.index <=
                effective_end
            )

        ]
    )

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

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    (
        trade_df,
        equity_df,
        open_df,
        diagnostics_df,
        final_marked,
        final_liq

    ) = run_backtest(

        all_signals,

        trading_days

    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize(

        trade_df,

        equity_df,

        final_marked,

        final_liq

    )

    print(
        "\n--- SUMMARY ---"
    )

    for k, v in (
        summary.items()
    ):

        print(
            f"{k}: {v}"
        )

    # --------------------------------------------------------
    # CURRENT HOLDINGS
    # --------------------------------------------------------

    print(
        "\n--- OPEN POSITIONS ---"
    )

    if not open_df.empty:

        print(
            open_df.to_string(
                index=False
            )
        )

    else:

        print(
            "No open positions."
        )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    write_to_sheet(

        trade_df,

        equity_df,

        open_df,

        diagnostics_df,

        summary,

        effective_end.strftime(
            "%Y-%m-%d"
        )

    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

        print(
            "\nBACKTEST COMPLETED "
            "SUCCESSFULLY."
        )

    except Exception as e:

        print(
            "\nBACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise