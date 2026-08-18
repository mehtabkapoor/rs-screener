"""
RS SCREENER - SIMPLE TOP 10 BACKTEST

============================================================
RULES
============================================================

Universe:
    stocks.csv

Price filter:
    Price > Rs.20

Liquidity:
    20-day average volume > 100,000

Price Trend Template:
    7/7 conditions required

RS Line Trend Template:
    7/7 conditions required

Ranking:
    Raw RS Score descending

Portfolio:
    Top 10 stocks

Weight:
    Equal weight

Rebalancing:
    Every trading day

Exit:
    Existing holding is sold when:
        Rank > 15
    OR
        stock leaves eligible universe

Costs:
    Buy + sell transaction costs

Tax:
    20.8% effective STCG on positive realized gains

Equity:
    Daily mark-to-market

Terminal:
    Both marked value and liquidation value calculated

Google Sheet:
    Results written to "Backtest" tab

============================================================
IMPORTANT
============================================================

BACKTEST_END = None

This automatically uses the latest benchmark
trading date actually available from Yahoo Finance.

============================================================
"""

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"

# None = latest available Yahoo Finance data
BACKTEST_END = None

DOWNLOAD_YEARS_BEFORE_START = 3

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250

TOP_N = 10
EXIT_RANK = 15

STARTING_CAPITAL = 1_000_000

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ============================================================
# GOOGLE SHEETS
# ============================================================

# YOUR GOOGLE SHEET
SHEET_ID = "12Ln5LY-g_GqeZWAu9JJOD2pxJf5_m31psadQDqpmJYo"

BACKTEST_WORKSHEET = "Backtest"

# Google credentials are still read from the environment.
CREDS_ENV = "GOOGLE_CREDENTIALS"


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

STCG_EFFECTIVE_RATE = (
    STCG_RATE *
    (1 + STCG_CESS)
)


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================

def get_download_dates():

    start = pd.Timestamp(
        BACKTEST_START
    )

    download_start = (
        start -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime("%Y-%m-%d"),
            None
        )

    end = pd.Timestamp(
        BACKTEST_END
    )

    download_end = (
        end +
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

    df = pd.read_csv(
        STOCKS_FILE
    )

    if "symbol" not in df.columns:

        raise ValueError(
            "stocks.csv must contain a "
            "'symbol' column."
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

    tickers = []

    for symbol in symbols:

        if symbol.endswith(".NS"):

            tickers.append(symbol)

        else:

            tickers.append(
                symbol + ".NS"
            )

    return tickers


# ============================================================
# CLEAN PRICE DATA
# ============================================================

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

def buy_side_cost(
    trade_value
):

    stt = (
        STT_RATE *
        trade_value
    )

    stamp = (
        STAMP_DUTY_RATE *
        trade_value
    )

    exchange = (
        EXCHANGE_CHARGE_RATE *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE *
        trade_value
    )

    gst = (
        GST_RATE *
        (exchange + sebi)
    )

    return (
        stt +
        stamp +
        exchange +
        sebi +
        gst
    )


# ============================================================
# SELL COST
# ============================================================

def sell_side_cost(
    trade_value
):

    stt = (
        STT_RATE *
        trade_value
    )

    exchange = (
        EXCHANGE_CHARGE_RATE *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE *
        trade_value
    )

    gst = (
        GST_RATE *
        (exchange + sebi)
    )

    return (
        stt +
        exchange +
        sebi +
        gst +
        DP_CHARGE_FLAT
    )


# ============================================================
# STCG TAX
# ============================================================

def stcg_tax(
    net_gain
):

    if net_gain <= 0:

        return 0.0

    return (
        net_gain *
        STCG_EFFECTIVE_RATE
    )


# ============================================================
# BENCHMARK DOWNLOAD
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\nBenchmark download:"
    )

    print(
        f"Start: {download_start}"
    )

    print(
        f"End: "
        f"{download_end or 'LATEST AVAILABLE'}"
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
                    f"Repaired "
                    f"{n_bad} benchmark "
                    f"data point(s)."
                )

            print(
                f"Benchmark loaded: "
                f"{ticker}"
            )

            print(
                "Latest benchmark date: "
                f"{close.index.max().strftime('%Y-%m-%d')}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark "
                f"{ticker} failed: {e}"
            )

    raise RuntimeError(
        "Could not download benchmark."
    )


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template_series(
    price
):

    sma50 = (
        price
        .rolling(50)
        .mean()
    )

    sma150 = (
        price
        .rolling(150)
        .mean()
    )

    sma200 = (
        price
        .rolling(200)
        .mean()
    )

    sma200_1mo = (
        sma200.shift(21)
    )

    low52 = (
        price
        .rolling(252)
        .min()
    )

    high52 = (
        price
        .rolling(252)
        .max()
    )

    condition1 = (
        (price > sma150) &
        (price > sma200)
    )

    condition2 = (
        sma150 > sma200
    )

    condition3 = (
        sma200 > sma200_1mo
    )

    condition4 = (
        (sma50 > sma150) &
        (sma50 > sma200)
    )

    condition5 = (
        price > sma50
    )

    condition6 = (
        price >=
        1.25 * low52
    )

    condition7 = (
        price >=
        0.75 * high52
    )

    total = (

        condition1.astype(int)

        +

        condition2.astype(int)

        +

        condition3.astype(int)

        +

        condition4.astype(int)

        +

        condition5.astype(int)

        +

        condition6.astype(int)

        +

        condition7.astype(int)

    )

    return (
        total == 7
    )


# ============================================================
# STOCK SIGNALS
# ============================================================

def compute_signals_for_stock(
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
        "price",
        "benchmark"
    ]

    if len(aligned) < 280:

        return None

    volume = (
        volume
        .reindex(
            aligned.index
        )
    )

    # --------------------------------------------------------
    # RS LINE
    # --------------------------------------------------------

    rs_ratio = (
        aligned["price"] /
        aligned["benchmark"]
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
            aligned["price"],
            63
        )

        +

        0.20 *
        pct_return(
            aligned["price"],
            126
        )

        +

        0.20 *
        pct_return(
            aligned["price"],
            189
        )

        +

        0.20 *
        pct_return(
            aligned["price"],
            252
        )

    ) * 100

    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
    # --------------------------------------------------------

    price_tt = (
        trend_template_series(
            aligned["price"]
        )
    )

    # --------------------------------------------------------
    # RS TREND TEMPLATE
    # --------------------------------------------------------

    rs_tt = (
        trend_template_series(
            rs_ratio
        )
    )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    avg_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (

        aligned["price"] >
        MIN_PRICE

    ) & (

        avg_volume >
        MIN_AVG_VOLUME
    )

    # --------------------------------------------------------
    # SIGNAL DATA
    # --------------------------------------------------------

    result = pd.DataFrame({

        "price":
        aligned["price"],

        "rs_score":
        rs_score,

        "price_tt":
        price_tt,

        "rs_tt":
        rs_tt,

        "liquid":
        liquid,

    })

    return result


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
        # BUILD ELIGIBLE UNIVERSE
        # ----------------------------------------------------

        eligible = []

        for symbol, df in (
            all_signals.items()
        ):

            if date not in df.index:

                continue

            row = (
                df.loc[date]
            )

            if pd.isna(
                row["rs_score"]
            ):

                continue

            if not bool(
                row["liquid"]
            ):

                continue

            if not bool(
                row["price_tt"]
            ):

                continue

            if not bool(
                row["rs_tt"]
            ):

                continue

            eligible.append(

                (
                    symbol,
                    float(
                        row["rs_score"]
                    )
                )
            )

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        eligible.sort(
            key=lambda x: x[1],
            reverse=True
        )

        rank_lookup = {

            symbol:
            rank + 1

            for rank,
            (symbol, _) in
            enumerate(eligible)
        }

        top10 = {
            symbol
            for symbol, _
            in eligible[:TOP_N]
        }

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        for symbol in list(
            holdings.keys()
        ):

            rank = (
                rank_lookup.get(
                    symbol
                )
            )

            # Keep stock if rank <= 15
            if (
                rank is not None
                and
                rank <= EXIT_RANK
            ):

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
                position["qty"] *
                exit_price
            )

            sell_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds -
                sell_cost
            )

            cost_basis = (

                position["qty"] *
                position["entry_price"]

                +

                position["entry_cost"]

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

            realized_cash = (
                net_proceeds -
                tax
            )

            cash += (
                realized_cash
            )

            gross_return = (

                (
                    exit_price /
                    position[
                        "entry_price"
                    ]
                    -
                    1
                )

                * 100

            )

            net_pnl = (
                net_gain -
                tax
            )

            net_return = (

                net_pnl /
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
                position["qty"],

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
                    gross_return,
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
                    net_pnl,
                    2
                ),

                "net_return_pct":
                round(
                    net_return,
                    2
                ),

                "days_held":
                (
                    date -
                    position[
                        "entry_date"
                    ]
                ).days,

                "exit_reason":
                exit_reason,

                "exit_rank":
                rank
                if rank is not None
                else "",

            })

        # ----------------------------------------------------
        # CURRENT PORTFOLIO VALUE
        # ----------------------------------------------------

        portfolio_value = (
            cash
        )

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
                position["qty"] *
                price
            )

        # ----------------------------------------------------
        # NEW ENTRIES
        # ----------------------------------------------------

        slots_open = (
            TOP_N -
            len(holdings)
        )

        if slots_open > 0:

            slot_capital = (
                portfolio_value /
                TOP_N
            )

            for symbol in [
                s for s, _
                in eligible[:TOP_N]
            ]:

                if slots_open <= 0:

                    break

                if symbol in holdings:

                    continue

                price = float(

                    all_signals[
                        symbol
                    ].loc[
                        date,
                        "price"
                    ]

                )

                if price <= 0:

                    continue

                qty = int(
                    slot_capital //
                    price
                )

                if qty < 1:

                    continue

                trade_value = (
                    qty *
                    price
                )

                buy_cost = (
                    buy_side_cost(
                        trade_value
                    )
                )

                total_required = (
                    trade_value +
                    buy_cost
                )

                if (
                    total_required >
                    cash
                ):

                    continue

                cash -= (
                    total_required
                )

                holdings[symbol] = {

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

        # ----------------------------------------------------
        # DAILY MARK-TO-MARKET
        # ----------------------------------------------------

        portfolio_value = (
            cash
        )

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
                position["qty"] *
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
                portfolio_value /
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

    liquidation_cash = (
        cash
    )

    open_positions = []

    if (
        len(trading_days) > 0
        and
        holdings
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
                position["qty"] *
                exit_price
            )

            sell_cost = (
                sell_side_cost(
                    gross_proceeds
                )
            )

            net_proceeds = (
                gross_proceeds -
                sell_cost
            )

            cost_basis = (

                position["qty"] *
                position["entry_price"]

                +

                position["entry_cost"]

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
                position["qty"],

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

                "unrealized_gross_return_pct":
                round(

                    (
                        exit_price /
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

    # ========================================================
    # DATAFRAMES
    # ========================================================

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
            ].cummax()
        )

        equity_df[
            "drawdown_pct"
        ] = (

            equity_df["equity"] /
            running_max -
            1

        ) * 100

        equity_df[
            "drawdown_pct"
        ] = (
            equity_df[
                "drawdown_pct"
            ].round(3)
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

        final_marked_value /
        STARTING_CAPITAL -
        1

    ) * 100

    liquidation_return = (

        final_liquidation_value /
        STARTING_CAPITAL -
        1

    ) * 100

    running_max = (
        equity_df[
            "equity"
        ].cummax()
    )

    drawdown = (

        equity_df["equity"] /
        running_max -
        1

    ) * 100

    max_drawdown = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------------

    if not trade_df.empty:

        winners = (
            trade_df[
                trade_df[
                    "net_return_pct"
                ] > 0
            ]
        )

        losers = (
            trade_df[
                trade_df[
                    "net_return_pct"
                ] < 0
            ]
        )

        n_trades = (
            len(trade_df)
        )

        gross_win_rate = (

            (
                trade_df[
                    "gross_return_pct"
                ] > 0
            ).mean()
            * 100

        )

        net_win_rate = (

            (
                trade_df[
                    "net_return_pct"
                ] > 0
            ).mean()
            * 100

        )

        avg_gross = (
            trade_df[
                "gross_return_pct"
            ].mean()
        )

        avg_net = (
            trade_df[
                "net_return_pct"
            ].mean()
        )

        median_net = (
            trade_df[
                "net_return_pct"
            ].median()
        )

        avg_days = (
            trade_df[
                "days_held"
            ].mean()
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

            gross_profit /
            gross_loss

            if gross_loss > 0
            else 0

        )

        best_trade = (
            trade_df[
                "gross_return_pct"
            ].max()
        )

        worst_trade = (
            trade_df[
                "gross_return_pct"
            ].min()
        )

        total_costs = (

            trade_df[
                "buy_cost_rs"
            ].sum()

            +

            trade_df[
                "sell_cost_rs"
            ].sum()

        )

        total_tax = (
            trade_df[
                "stcg_tax_rs"
            ].sum()
        )

    else:

        n_trades = 0
        gross_win_rate = 0
        net_win_rate = 0
        avg_gross = 0
        avg_net = 0
        median_net = 0
        avg_days = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0
        best_trade = 0
        worst_trade = 0
        total_costs = 0
        total_tax = 0

    # --------------------------------------------------------
    # DAILY STATISTICS
    # --------------------------------------------------------

    daily_returns = (
        equity_df[
            "equity"
        ]
        .pct_change()
        .dropna()
    )

    if len(daily_returns) > 1:

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
                252 /
                max(n_days, 1)
            )

        ) - 1

        annualized_vol = (

            daily_std *
            np.sqrt(252)

        )

        sharpe = (

            daily_mean /
            daily_std *
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

            if len(downside) > 1
            else 0

        )

        sortino = (

            daily_mean /
            downside_std *
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

        annualized_return /
        abs(max_drawdown / 100)

        if max_drawdown < 0
        else 0

    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    return {

        "Backtest Start":
        BACKTEST_START,

        "Backtest End":
        equity_df[
            "date"
        ].iloc[-1],

        "Starting Capital (Rs)":
        STARTING_CAPITAL,

        "Final Value - Marked (Rs)":
        round(
            final_marked_value,
            0
        ),

        "Final Value - Liquidated (Rs)":
        round(
            final_liquidation_value,
            0
        ),

        "Net Return - Marked (%)":
        round(
            marked_return,
            2
        ),

        "Net Return - Liquidated (%)":
        round(
            liquidation_return,
            2
        ),

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
        round(
            calmar,
            3
        ),

        "Max Drawdown (%)":
        round(
            max_drawdown,
            2
        ),

        "Number of Closed Trades":
        n_trades,

        "Win Rate - Gross (%)":
        round(
            gross_win_rate,
            1
        ),

        "Win Rate - Net (%)":
        round(
            net_win_rate,
            1
        ),

        "Average Gross Trade (%)":
        round(
            avg_gross,
            2
        ),

        "Average Net Trade (%)":
        round(
            avg_net,
            2
        ),

        "Median Net Trade (%)":
        round(
            median_net,
            2
        ),

        "Average Days Held":
        round(
            avg_days,
            1
        ),

        "Average Winner (%)":
        round(
            avg_winner,
            2
        ),

        "Average Loser (%)":
        round(
            avg_loser,
            2
        ),

        "Profit Factor":
        round(
            profit_factor,
            3
        ),

        "Best Gross Trade (%)":
        round(
            best_trade,
            2
        ),

        "Worst Gross Trade (%)":
        round(
            worst_trade,
            2
        ),

        "Total Transaction Costs (Rs)":
        round(
            total_costs,
            0
        ),

        "Total STCG Tax (Rs)":
        round(
            total_tax,
            0
        ),

    }


# ============================================================
# GOOGLE SHEET CHUNK WRITER
# ============================================================

def write_in_chunks(

    worksheet,

    rows,

    start_row,

    chunk_size=2000,

    label="data"

):

    if not rows:

        return

    total = len(rows)

    for i in range(
        0,
        total,
        chunk_size
    ):

        chunk = rows[
            i:i + chunk_size
        ]

        row = (
            start_row + i
        )

        try:

            worksheet.update(
                chunk,
                f"A{row}"
            )

        except Exception as e:

            print(
                f"Write failed "
                f"for {label}: {e}"
            )

            time.sleep(5)

            worksheet.update(
                chunk,
                f"A{row}"
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
    spreadsheet,
    sheet_id
):

    try:

        metadata = (
            spreadsheet
            .fetch_sheet_metadata()
        )

        requests = []

        for sheet in metadata.get(
            "sheets",
            []
        ):

            properties = (
                sheet["properties"]
            )

            if (
                properties["sheetId"]
                !=
                sheet_id
            ):

                continue

            for chart in (
                sheet.get(
                    "charts",
                    []
                )
            ):

                requests.append({

                    "deleteEmbeddedObject": {

                        "objectId":
                        chart["chartId"]

                    }

                })

        if requests:

            spreadsheet.batch_update({

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
            "Chart cleanup failed "
            f"(non-fatal): {e}"
        )


# ============================================================
# ADD CHARTS
# ============================================================

def add_charts(

    spreadsheet,

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

    def chart_request(

        title,

        column_index,

        y_title,

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
                                    y_title

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
                                                    column_index,

                                                    "endColumnIndex":
                                                    column_index + 1

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

        chart_request(

            "RS Top 10 Equity Curve",

            1,

            "Portfolio Value (Rs)",

            equity_header_row_0idx

        ),

        chart_request(

            "RS Top 10 Drawdown",

            5,

            "Drawdown (%)",

            equity_header_row_0idx + 22

        )

    ]

    try:

        spreadsheet.batch_update({

            "requests":
            requests

        })

        print(
            "Charts added."
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

    summary,

    effective_end

):

    credentials_json = (
        os.environ.get(
            CREDS_ENV
        )
    )

    if not credentials_json:

        print(
            "\nGOOGLE_CREDENTIALS "
            "not found."
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

        open_df.to_csv(
            "backtest_open_positions.csv",
            index=False
        )

        return

    # --------------------------------------------------------
    # GOOGLE AUTH
    # --------------------------------------------------------

    credentials_dict = json.loads(
        credentials_json
    )

    scopes = [

        "https://www.googleapis.com/auth/spreadsheets"

    ]

    credentials = (
        Credentials
        .from_service_account_info(

            credentials_dict,

            scopes=scopes

        )
    )

    client = (
        gspread.authorize(
            credentials
        )
    )

    spreadsheet = (
        client.open_by_key(
            SHEET_ID
        )
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    # --------------------------------------------------------
    # GET / CREATE BACKTEST TAB
    # --------------------------------------------------------

    try:

        worksheet = (
            spreadsheet
            .worksheet(
                BACKTEST_WORKSHEET
            )
        )

    except gspread.WorksheetNotFound:

        worksheet = (
            spreadsheet.add_worksheet(

                title=
                BACKTEST_WORKSHEET,

                rows=
                1000,

                cols=
                16

            )
        )

    # --------------------------------------------------------
    # REMOVE OLD CHARTS
    # --------------------------------------------------------

    remove_existing_charts(

        spreadsheet,

        worksheet.id

    )

    # --------------------------------------------------------
    # CLEAR OLD DATA
    # --------------------------------------------------------

    worksheet.clear()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    worksheet.update(

        [[

            "RS TOP 10 BACKTEST | "
            f"Run: {timestamp} | "
            f"Start: {BACKTEST_START} | "
            f"End: {effective_end} | "
            f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
            "Entry: Price TT 7/7 + RS TT 7/7 | "
            f"Exit: Rank > {EXIT_RANK} | "
            "Net of Costs + STCG"

        ]],

        "A1"

    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = [

        ["SUMMARY", ""]

    ]

    for key, value in (
        summary.items()
    ):

        summary_rows.append(

            [
                key,
                value
            ]

        )

    worksheet.update(

        summary_rows,

        "A3"

    )

    # --------------------------------------------------------
    # TRADE LOG
    # --------------------------------------------------------

    trade_start = (
        3 +
        len(summary_rows) +
        2
    )

    worksheet.update(

        [["TRADE LOG"]],

        f"A{trade_start}"

    )

    trade_header = (
        trade_start + 1
    )

    if not trade_df.empty:

        rows = [

            list(
                trade_df.columns
            )

        ] + (

            trade_df
            .replace(
                {np.nan: ""}
            )
            .values
            .tolist()

        )

        write_in_chunks(

            worksheet,

            rows,

            trade_header,

            2000,

            "trade log"

        )

    # --------------------------------------------------------
    # OPEN POSITIONS
    # --------------------------------------------------------

    open_start = (

        trade_header +

        max(
            len(trade_df),
            1
        ) +

        3

    )

    worksheet.update(

        [[
            "OPEN POSITIONS AT "
            "BACKTEST END"
        ]],

        f"A{open_start}"

    )

    open_header = (
        open_start + 1
    )

    if not open_df.empty:

        rows = [

            list(
                open_df.columns
            )

        ] + (

            open_df
            .replace(
                {np.nan: ""}
            )
            .values
            .tolist()

        )

        write_in_chunks(

            worksheet,

            rows,

            open_header,

            2000,

            "open positions"

        )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity_start = (

        open_header +

        max(
            len(open_df),
            1
        ) +

        3

    )

    worksheet.update(

        [["DAILY EQUITY CURVE"]],

        f"A{equity_start}"

    )

    equity_header = (
        equity_start + 1
    )

    if not equity_df.empty:

        rows = [

            list(
                equity_df.columns
            )

        ] + (

            equity_df
            .replace(
                {np.nan: ""}
            )
            .values
            .tolist()

        )

        write_in_chunks(

            worksheet,

            rows,

            equity_header,

            2000,

            "equity curve"

        )

        # ----------------------------------------------------
        # ADD CHARTS
        # ----------------------------------------------------

        add_charts(

            spreadsheet,

            worksheet.id,

            equity_header - 1,

            len(equity_df)

        )

    print(
        "\n=========================================="
    )

    print(
        "RESULTS WRITTEN TO GOOGLE SHEETS"
    )

    print(
        "Sheet tab: Backtest"
    )

    print(
        f"Trades: {len(trade_df)}"
    )

    print(
        f"Trading days: {len(equity_df)}"
    )

    print(
        f"Open positions: {len(open_df)}"
    )

    print(
        "=========================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD STOCKS
    # --------------------------------------------------------

    tickers = (
        load_tickers()
    )

    print(
        f"\nLoaded "
        f"{len(tickers)} tickers."
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\n=================================================="
    )

    print(
        f"Download start: {download_start}"
    )

    print(
        "Download end: "
        f"{download_end or 'LATEST AVAILABLE'}"
    )

    print(
        f"Backtest start: {BACKTEST_START}"
    )

    print(
        "Backtest end: "
        f"{BACKTEST_END or 'LATEST AVAILABLE'}"
    )

    print(
        f"Price > Rs.{MIN_PRICE}"
    )

    print(
        f"{VOLUME_LOOKBACK}-day "
        f"average volume > "
        f"{MIN_AVG_VOLUME:,}"
    )

    print(
        "Price TT: 7/7"
    )

    print(
        "RS TT: 7/7"
    )

    print(
        f"Portfolio: Top {TOP_N}"
    )

    print(
        f"Exit: Rank > {EXIT_RANK}"
    )

    print(
        f"Starting capital: "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        "=================================================="
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    benchmark = (
        download_benchmark()
    )

    # --------------------------------------------------------
    # STOCK SIGNALS
    # --------------------------------------------------------

    all_signals = {}

    total_repaired = 0

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
            f"\nDownloading "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(tickers)}..."
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

                    stock_data = data

                else:

                    if (
                        symbol not in
                        data.columns
                        .get_level_values(0)
                    ):

                        continue

                    stock_data = (
                        data[symbol]
                    )

                if (
                    "Close"
                    not in
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

                if close.empty:

                    continue

                close, n_bad = (
                    clean_price_series(
                        close
                    )
                )

                total_repaired += (
                    n_bad
                )

                signals = (
                    compute_signals_for_stock(

                        close,

                        volume,

                        benchmark

                    )
                )

                if signals is not None:

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

            except Exception as e:

                print(
                    f"Skipping "
                    f"{symbol}: {e}"
                )

        time.sleep(1)

    # --------------------------------------------------------
    # SIGNAL SUMMARY
    # --------------------------------------------------------

    print(
        "\nSignals calculated for "
        f"{len(all_signals)} stocks."
    )

    print(
        "Repaired price points: "
        f"{total_repaired}"
    )

    if not all_signals:

        raise RuntimeError(
            "No stock signals were calculated."
        )

    # --------------------------------------------------------
    # EFFECTIVE END DATE
    # --------------------------------------------------------

    latest_stock_date = max(

        df.index.max()

        for df in
        all_signals.values()

    )

    latest_benchmark_date = (
        pd.Timestamp(
            benchmark.index.max()
        )
    )

    if BACKTEST_END is None:

        effective_end = (
            latest_benchmark_date
        )

    else:

        effective_end = min(

            pd.Timestamp(
                BACKTEST_END
            ),

            latest_benchmark_date

        )

    print(
        "\nLatest stock data: "
        f"{pd.Timestamp(latest_stock_date).strftime('%Y-%m-%d')}"
    )

    print(
        "Latest benchmark data: "
        f"{latest_benchmark_date.strftime('%Y-%m-%d')}"
    )

    print(
        "Effective backtest end: "
        f"{effective_end.strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # TRADING DAYS
    # --------------------------------------------------------

    trading_days = (
        benchmark.index[

            (
                benchmark.index >=
                pd.Timestamp(
                    BACKTEST_START
                )
            )

            &

            (
                benchmark.index <=
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
    # RUN BACKTEST
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize(

        trade_df,

        equity_df,

        final_marked,

        final_liquidation

    )

    print(
        "\n================ SUMMARY ================"
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # GOOGLE SHEET
    # --------------------------------------------------------

    write_to_sheet(

        trade_df,

        equity_df,

        open_df,

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
            "\nBACKTEST COMPLETED SUCCESSFULLY."
        )

    except Exception as e:

        print(
            "\nBACKTEST FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise