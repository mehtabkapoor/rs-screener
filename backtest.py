"""
RS SCREENER BACKTEST
====================

SINGLE SOURCE OF TRUTH:
    rs_screener.py

The backtest imports the actual screening functions and configuration
from rs_screener.py.

Therefore these cannot silently diverge:

    Price filter
    Liquidity filter
    RS Score
    Price Trend Template
    RS Line Trend Template
    VCP
    Volume Dry-up
    Pivot
    Breakout-volume requirement
    Transaction costs
    TOP_N

PORTFOLIO MODEL
---------------

Universe:
    stocks.csv

Entry:
    Same FINAL screen as rs_screener.py

Ranking:
    Raw RS Score descending

Portfolio:
    Top 10

Weight:
    Equal weight

Rebalance:
    Daily EOD

Exit:
    Leaves current Top 10

No:
    Stop loss
    RS EMA exit
    Rank buffer
    Blue Dot filter
    Green Dot filter
    Regime filter

IMPORTANT
---------
The backtest is historical/EOD.

It does NOT use the screener's intraday PREVIEW price.

It uses Yahoo Finance auto-adjusted daily OHLCV.

Historical signal on date D is calculated using data available
through date D only.

This avoids look-ahead.

"""

# ============================================================
# IMPORTS
# ============================================================

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials

# ============================================================
# SINGLE SOURCE OF TRUTH
# ============================================================

import rs_screener as screener


# ============================================================
# BACKTEST CONFIG
# ============================================================

BACKTEST_START = "2016-04-01"

# None = latest available Yahoo trading day
BACKTEST_END = None

DOWNLOAD_YEARS_BEFORE_START = 3

STARTING_CAPITAL = 1_000_000

STOCKS_FILE = screener.STOCKS_FILE

SHEET_ID_ENV = screener.SHEET_ID_ENV
CREDS_ENV = screener.CREDS_ENV

BACKTEST_WORKSHEET = "Backtest"

BATCH_SIZE = 50


# ============================================================
# LOAD TICKERS
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
            "stocks.csv must contain "
            "a 'symbol' column."
        )

    symbols = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return [
        s
        if s.endswith(".NS")
        else s + ".NS"
        for s in symbols
    ]


# ============================================================
# DATE RANGE
# ============================================================

def get_download_dates():

    start = pd.Timestamp(
        BACKTEST_START
    )

    download_start = (
        start
        -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime(
                "%Y-%m-%d"
            ),
            None
        )

    end = pd.Timestamp(
        BACKTEST_END
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


# ============================================================
# BENCHMARK
# ============================================================

def download_benchmark():

    start, end = (
        get_download_dates()
    )

    for ticker in (
        screener.BENCHMARK,
        screener.BENCHMARK_FALLBACK
    ):

        try:

            kwargs = {
                "tickers": ticker,
                "start": start,
                "interval": "1d",
                "auto_adjust": True,
                "progress": False,
            }

            if end is not None:
                kwargs["end"] = end

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

            if not close.empty:

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
        "Could not download benchmark."
    )


# ============================================================
# EXTRACT STOCK DATA
# ============================================================

def extract_stock_data(
    data,
    symbol,
    batch_size
):

    try:

        if batch_size == 1:

            sdata = data

        else:

            if not isinstance(
                data.columns,
                pd.MultiIndex
            ):

                return None

            if symbol not in (
                data.columns
                .get_level_values(0)
            ):

                return None

            sdata = data[symbol]

        if "Close" not in sdata.columns:
            return None

        if "Volume" not in sdata.columns:
            return None

        close = (
            sdata["Close"]
            .dropna()
            .sort_index()
        )

        volume = (
            sdata["Volume"]
            .reindex(close.index)
            .fillna(0)
        )

        if close.empty:
            return None

        return close, volume

    except Exception:

        return None


# ============================================================
# EXACT HISTORICAL SCREEN
# ============================================================

def evaluate_date(
    close,
    volume,
    benchmark,
    date
):

    """
    Calculate the exact same screen used by rs_screener.py
    as of one historical date.

    CRITICAL:
        Every calculation receives data only through `date`.

    This prevents future information entering today's signal.
    """

    close_d = (
        close.loc[:date]
        .dropna()
    )

    volume_d = (
        volume.loc[:date]
        .dropna()
    )

    benchmark_d = (
        benchmark.loc[:date]
        .dropna()
    )

    if len(close_d) < (
        screener.LOOKBACK_DAYS + 2
    ):

        return None

    # --------------------------------------------------------
    # ALIGN STOCK / BENCHMARK
    # --------------------------------------------------------

    aligned = pd.concat(
        [
            close_d,
            benchmark_d
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(aligned) < (
        screener.LOOKBACK_DAYS + 2
    ):

        return None

    stock_close = (
        aligned.iloc[:, 0]
    )

    bench_close = (
        aligned.iloc[:, 1]
    )

    volume_d = (
        volume_d
        .reindex(stock_close.index)
        .fillna(0)
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    last_price = float(
        stock_close.iloc[-1]
    )

    if last_price <= (
        screener.MIN_PRICE
    ):

        return None

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    avg20_volume = (
        volume_d
        .tail(
            screener.VOLUME_LOOKBACK
        )
        .mean()
    )

    if (
        pd.isna(avg20_volume)
        or
        avg20_volume
        <= screener.MIN_AVG_VOLUME
    ):

        return None

    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

    rs_score = (
        screener.compute_rs_score(
            stock_close
        )
    )

    if rs_score is None:
        return None

    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
    # --------------------------------------------------------

    tt_pass = (
        screener.compute_trend_template(
            stock_close
        )
    )

    if tt_pass is None:
        return None

    # --------------------------------------------------------
    # RS LINE TREND TEMPLATE
    # --------------------------------------------------------

    rs_tt_pass = (
        screener.compute_rs_line_template(
            stock_close,
            bench_close
        )
    )

    if rs_tt_pass is None:
        return None

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    diagnostics = (
        screener.compute_diagnostics(
            stock_close,
            bench_close
        )
    )

    if diagnostics is None:
        return None

    blue_dot, green_dot = (
        diagnostics
    )

    # --------------------------------------------------------
    # VCP
    # --------------------------------------------------------

    vcp = (
        screener.compute_vcp(
            stock_close,
            volume_d
        )
    )

    if vcp is None:
        return None

    # --------------------------------------------------------
    # VOLUME DRY-UP
    # --------------------------------------------------------

    dryup = (
        screener.compute_volume_dryup(
            volume_d
        )
    )

    if dryup is None:
        return None

    # --------------------------------------------------------
    # PIVOT
    # --------------------------------------------------------

    pivot = (
        screener.compute_pivot(
            stock_close,
            volume_d
        )
    )

    if pivot is None:
        return None

    # --------------------------------------------------------
    # EXACT FINAL SCREEN
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

        pivot[
            "proximity_pass"
        ] is True
    )

    if (
        screener.REQUIRE_BREAKOUT_VOLUME
    ):

        screen_pass = (

            core_screen_pass

            and

            pivot[
                "breakout_pass"
            ]
        )

    else:

        screen_pass = (
            core_screen_pass
        )

    return {

        "price":
            last_price,

        "rs_score":
            float(rs_score),

        "avg20_volume":
            float(avg20_volume),

        "tt_pass":
            bool(tt_pass),

        "rs_tt_pass":
            bool(rs_tt_pass),

        "vcp_pass":
            bool(
                vcp["pass"]
            ),

        "volume_dryup_pass":
            bool(
                dryup["pass"]
            ),

        "pivot_proximity_pass":
            bool(
                pivot[
                    "proximity_pass"
                ]
            ),

        "breakout_volume_pass":
            bool(
                pivot[
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

        "pivot":
            float(
                pivot["pivot"]
            ),

        "pivot_distance":
            float(
                pivot["distance"]
            ),

        "breakout_volume_ratio":
            (
                float(
                    pivot[
                        "breakout_ratio"
                    ]
                )
                if pivot[
                    "breakout_ratio"
                ] is not None
                else np.nan
            ),
    }


# ============================================================
# PRECOMPUTE HISTORICAL SIGNALS
# ============================================================

def compute_signals_for_stock(
    close,
    volume,
    benchmark,
    trading_days
):

    """
    Exact signal engine.

    The screener functions are called for each historical date.

    This is computationally expensive.

    That is intentional.

    Correctness > speed.

    """

    rows = []

    for i, date in enumerate(
        trading_days
    ):

        try:

            result = evaluate_date(
                close,
                volume,
                benchmark,
                date
            )

            if result is None:
                continue

            result["date"] = date

            rows.append(
                result
            )

        except Exception as e:

            print(
                f"Signal error on "
                f"{date} : {e}"
            )

    if not rows:
        return None

    df = pd.DataFrame(
        rows
    )

    df = (
        df
        .set_index("date")
        .sort_index()
    )

    return df


# ============================================================
# COSTS
# ============================================================

def buy_side_cost(
    trade_value
):

    return (
        screener.buy_side_cost(
            trade_value
        )
    )


def sell_side_cost(
    trade_value
):

    return (
        screener.sell_side_cost(
            trade_value
        )
    )


def stcg_tax(
    gross_pnl
):

    return (
        screener.estimate_stcg(
            gross_pnl
        )
    )


# ============================================================
# PORTFOLIO BACKTEST
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

    for date in trading_days:

        # ====================================================
        # TODAY'S ELIGIBLE UNIVERSE
        # ====================================================

        pool = []

        for symbol, df in (
            all_signals.items()
        ):

            if date not in df.index:
                continue

            row = df.loc[date]

            if not bool(
                row["screen_pass"]
            ):

                continue

            pool.append(
                (
                    symbol,
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

        target_symbols = {
            symbol
            for symbol, _score
            in pool[
                :screener.TOP_N
            ]
        }

        # ====================================================
        # EXIT POSITIONS
        # ====================================================

        for symbol in list(
            holdings.keys()
        ):

            if symbol in target_symbols:
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
                position["qty"]
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
                position["qty"]
                *
                position[
                    "entry_price"
                ]
                +
                position[
                    "entry_cost"
                ]
            )

            gross_pnl = (
                net_proceeds
                -
                cost_basis
            )

            tax = (
                stcg_tax(
                    gross_pnl
                )
            )

            cash += (
                net_proceeds
                -
                tax
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
                    position[
                        "qty"
                    ],

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
                        (
                            exit_price
                            /
                            position[
                                "entry_price"
                            ]
                            -
                            1
                        )
                        * 100,
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
                        gross_pnl - tax,
                        2
                    ),

                "net_return_pct":
                    round(
                        (
                            gross_pnl
                            -
                            tax
                        )
                        /
                        cost_basis
                        *
                        100,
                        2
                    )
                    if cost_basis > 0
                    else 0,

                "days_held":
                    (
                        date
                        -
                        position[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "Left Top 10",
            })

        # ====================================================
        # CURRENT PORTFOLIO VALUE
        # ====================================================

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
                position["qty"]
                *
                price
            )

        # ====================================================
        # BUY NEW TARGET POSITIONS
        # ====================================================

        new_symbols = [
            symbol
            for symbol, _score
            in pool[
                :screener.TOP_N
            ]
            if symbol not in holdings
        ]

        # Target portfolio is equal weighted.
        #
        # Recalculate after exits.
        #
        target_value = (
            portfolio_value
            /
            screener.TOP_N
        )

        for symbol in new_symbols:

            if len(holdings) >= (
                screener.TOP_N
            ):

                break

            df = (
                all_signals[
                    symbol
                ]
            )

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

            qty = int(
                target_value
                //
                price
            )

            if qty <= 0:
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

            holdings[
                symbol
            ] = {

                "qty":
                    qty,

                "entry_price":
                    price,

                "entry_date":
                    date,

                "entry_cost":
                    buy_cost,
            }

        # ====================================================
        # END-OF-DAY MARK
        # ====================================================

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
                position["qty"]
                *
                price
            )

        equity_curve.append({

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "portfolio_value_rs":
                portfolio_value,

            "equity":
                (
                    portfolio_value
                    /
                    STARTING_CAPITAL
                ),

            "cash_rs":
                cash,

            "n_holdings":
                len(holdings),
        })

    # ========================================================
    # EQUITY DATAFRAME
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    if not equity_df.empty:

        running_max = (
            equity_df[
                "equity"
            ]
            .cummax()
        )

        equity_df[
            "drawdown_pct"
        ] = (

            equity_df[
                "equity"
            ]
            /
            running_max
            -
            1

        ) * 100

    # ========================================================
    # OPEN POSITIONS
    # ========================================================

    open_positions = []

    if trading_days:

        final_date = (
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

            if final_date in df.index:

                price = float(
                    df.loc[
                        final_date,
                        "price"
                    ]
                )

            else:

                price = (
                    position[
                        "entry_price"
                    ]
                )

            marked_value = (
                position["qty"]
                *
                price
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
                    position[
                        "qty"
                    ],

                "entry_price":
                    position[
                        "entry_price"
                    ],

                "last_price":
                    price,

                "marked_value_rs":
                    marked_value,

                "gross_return_pct":
                    (
                        price
                        /
                        position[
                            "entry_price"
                        ]
                        -
                        1
                    )
                    * 100,
            })

    trade_df = pd.DataFrame(
        trade_log
    )

    open_df = pd.DataFrame(
        open_positions
    )

    final_marked_value = (
        equity_df[
            "portfolio_value_rs"
        ].iloc[-1]
        if not equity_df.empty
        else STARTING_CAPITAL
    )

    return (
        trade_df,
        equity_df,
        open_df,
        final_marked_value
    )


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    trade_df,
    equity_df,
    final_value
):

    if equity_df.empty:

        return {}

    total_return = (
        final_value
        /
        STARTING_CAPITAL
        -
        1
    ) * 100

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

    daily_returns = (
        equity_df[
            "equity"
        ]
        .pct_change()
        .dropna()
    )

    n_days = len(
        equity_df
    )

    if n_days > 0:

        annualized_return = (
            equity_df[
                "equity"
            ].iloc[-1]
            **
            (
                252
                /
                n_days
            )
            -
            1
        )

    else:

        annualized_return = 0

    if len(daily_returns) > 1:

        volatility = (
            daily_returns.std()
            *
            np.sqrt(252)
        )

        sharpe = (
            daily_returns.mean()
            /
            daily_returns.std()
            *
            np.sqrt(252)
        )

    else:

        volatility = 0
        sharpe = 0

    if trade_df.empty:

        n_trades = 0
        win_rate = 0
        avg_trade = 0
        median_trade = 0
        best_trade = 0
        worst_trade = 0
        total_costs = 0
        total_tax = 0

    else:

        n_trades = len(
            trade_df
        )

        win_rate = (
            (
                trade_df[
                    "net_return_pct"
                ]
                > 0
            )
            .mean()
            *
            100
        )

        avg_trade = (
            trade_df[
                "net_return_pct"
            ].mean()
        )

        median_trade = (
            trade_df[
                "net_return_pct"
            ].median()
        )

        best_trade = (
            trade_df[
                "net_return_pct"
            ].max()
        )

        worst_trade = (
            trade_df[
                "net_return_pct"
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

    return {

        "Strategy":
            "Synchronized rs_screener.py",

        "Backtest start":
            BACKTEST_START,

        "Backtest end":
            (
                equity_df[
                    "date"
                ].iloc[-1]
                if not equity_df.empty
                else ""
            ),

        "Starting capital Rs":
            STARTING_CAPITAL,

        "Final marked value Rs":
            round(
                final_value,
                2
            ),

        "Total return %":
            round(
                total_return,
                2
            ),

        "Annualized return %":
            round(
                annualized_return
                * 100,
                2
            ),

        "Annualized volatility %":
            round(
                volatility
                * 100,
                2
            ),

        "Sharpe":
            round(
                sharpe,
                3
            ),

        "Max drawdown %":
            round(
                max_dd,
                2
            ),

        "Closed trades":
            n_trades,

        "Win rate %":
            round(
                win_rate,
                2
            ),

        "Average net trade %":
            round(
                avg_trade,
                2
            ),

        "Median net trade %":
            round(
                median_trade,
                2
            ),

        "Best net trade %":
            round(
                best_trade,
                2
            ),

        "Worst net trade %":
            round(
                worst_trade,
                2
            ),

        "Total transaction costs Rs":
            round(
                total_costs,
                2
            ),

        "Total STCG tax Rs":
            round(
                total_tax,
                2
            ),

        "TOP_N":
            screener.TOP_N,

        "VCP lookback":
            screener.VCP_LOOKBACK,

        "VCP min contractions":
            screener.VCP_MIN_CONTRACTIONS,

        "VCP max final contraction":
            screener.VCP_MAX_FINAL_CONTRACTION,

        "VCP max base depth":
            screener.VCP_MAX_BASE_DEPTH,

        "VCP contraction improvement":
            screener.VCP_CONTRACTION_IMPROVEMENT,

        "Dry-up ratio":
            screener.VCP_VOLUME_DRYUP_RATIO,

        "Dry-up days":
            screener.VCP_DRYUP_DAYS,

        "Pivot lookback":
            screener.PIVOT_LOOKBACK,

        "Pivot proximity":
            screener.PIVOT_PROXIMITY_PCT,

        "Breakout volume required":
            screener.REQUIRE_BREAKOUT_VOLUME,
    }


# ============================================================
# SAVE CSV FALLBACK
# ============================================================

def save_csvs(
    trade_df,
    equity_df,
    open_df,
    summary
):

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

    pd.DataFrame(
        list(
            summary.items()
        ),
        columns=[
            "Metric",
            "Value"
        ]
    ).to_csv(
        "backtest_summary.csv",
        index=False
    )


# ============================================================
# GOOGLE SHEETS
# ============================================================

def write_to_google_sheet(
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

    if not sheet_id or not creds_json:

        print(
            "Google Sheets credentials "
            "not configured."
        )

        save_csvs(
            trade_df,
            equity_df,
            open_df,
            summary
        )

        return

    try:

        creds_dict = json.loads(
            creds_json
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets"
        ]

        credentials = (
            Credentials
            .from_service_account_info(
                creds_dict,
                scopes=scopes
            )
        )

        gc = gspread.authorize(
            credentials
        )

        sh = gc.open_by_key(
            sheet_id
        )

        try:

            ws = sh.worksheet(
                BACKTEST_WORKSHEET
            )

            ws.clear()

        except gspread.WorksheetNotFound:

            ws = sh.add_worksheet(

                title=
                    BACKTEST_WORKSHEET,

                rows=
                    max(
                        1000,
                        len(equity_df)
                        +
                        len(trade_df)
                        +
                        100
                    ),

                cols=20
            )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        ws.update(
            [[
                "SYNCHRONIZED RS SCREENER BACKTEST"
            ]],
            "A1"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        row = 3

        ws.update(
            [["SUMMARY"]],
            f"A{row}"
        )

        row += 1

        summary_rows = [
            [
                key,
                value
            ]
            for key, value
            in summary.items()
        ]

        if summary_rows:

            ws.update(
                summary_rows,
                f"A{row}"
            )

            row += len(
                summary_rows
            ) + 2

        # ----------------------------------------------------
        # TRADES
        # ----------------------------------------------------

        ws.update(
            [["TRADE LOG"]],
            f"A{row}"
        )

        row += 1

        if not trade_df.empty:

            rows = [

                list(
                    trade_df.columns
                )

            ] + (

                trade_df
                .replace(
                    {
                        np.nan: ""
                    }
                )
                .values
                .tolist()
            )

            ws.update(
                rows,
                f"A{row}"
            )

            row += len(rows) + 2

        # ----------------------------------------------------
        # OPEN POSITIONS
        # ----------------------------------------------------

        ws.update(
            [["OPEN POSITIONS"]],
            f"A{row}"
        )

        row += 1

        if not open_df.empty:

            rows = [

                list(
                    open_df.columns
                )

            ] + (

                open_df
                .replace(
                    {
                        np.nan: ""
                    }
                )
                .values
                .tolist()
            )

            ws.update(
                rows,
                f"A{row}"
            )

            row += len(rows) + 2

        # ----------------------------------------------------
        # EQUITY
        # ----------------------------------------------------

        equity_header_row = row

        ws.update(
            [["DAILY EQUITY CURVE"]],
            f"A{row}"
        )

        row += 1

        if not equity_df.empty:

            rows = [

                list(
                    equity_df.columns
                )

            ] + (

                equity_df
                .replace(
                    {
                        np.nan: ""
                    }
                )
                .values
                .tolist()
            )

            ws.update(
                rows,
                f"A{row}"
            )

        print(
            f"Results written to "
            f"{BACKTEST_WORKSHEET}"
        )

    except Exception as e:

        print(
            f"Google Sheets write failed: "
            f"{e}"
        )

        save_csvs(
            trade_df,
            equity_df,
            open_df,
            summary
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RS SCREENER — SYNCHRONIZED BACKTEST"
    )

    print(
        "=" * 80
    )

    print(
        "SOURCE OF TRUTH: rs_screener.py"
    )

    print(
        f"Start: {BACKTEST_START}"
    )

    print(
        f"End: "
        f"{BACKTEST_END or 'latest available'}"
    )

    print(
        f"Starting capital: "
        f"Rs.{STARTING_CAPITAL:,.0f}"
    )

    print(
        f"Top N: {screener.TOP_N}"
    )

    print(
        "\nENTRY:"
    )

    print(
        f"  Price > Rs.{screener.MIN_PRICE}"
    )

    print(
        f"  {screener.VOLUME_LOOKBACK}D "
        f"average volume > "
        f"{screener.MIN_AVG_VOLUME:,}"
    )

    print(
        "  Price Trend Template"
    )

    print(
        "  RS Line Trend Template"
    )

    print(
        "  VCP"
    )

    print(
        "  Volume Dry-up"
    )

    print(
        "  Pivot Proximity"
    )

    print(
        "  Breakout volume required: "
        f"{screener.REQUIRE_BREAKOUT_VOLUME}"
    )

    print(
        "\nEXIT:"
    )

    print(
        "  Leaves current Top 10"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # DATA
    # ========================================================

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    benchmark = (
        download_benchmark()
    )

    start_download, end_download = (
        get_download_dates()
    )

    # --------------------------------------------------------
    # BACKTEST TRADING DAYS
    # --------------------------------------------------------

    benchmark = (
        benchmark
        .sort_index()
        .dropna()
    )

    effective_end = (
        benchmark.index.max()
        if BACKTEST_END is None
        else pd.Timestamp(
            BACKTEST_END
        )
    )

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
                effective_end
            )
        ]
    )

    if len(trading_days) == 0:

        raise RuntimeError(
            "No trading days found."
        )

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    # ========================================================
    # STOCK DATA + SIGNALS
    # ========================================================

    all_signals = {}

    for batch_start in range(
        0,
        len(tickers),
        BATCH_SIZE
    ):

        batch = tickers[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Downloading "
            f"{batch_start + 1}-"
            f"{batch_start + len(batch)} "
            f"/ {len(tickers)}"
        )

        print(
            "-" * 70
        )

        try:

            kwargs = {

                "tickers":
                    batch,

                "start":
                    start_download,

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

            if end_download is not None:

                kwargs["end"] = (
                    end_download
                )

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
                    extract_stock_data(
                        data,
                        symbol,
                        len(batch)
                    )
                )

                if extracted is None:

                    print(
                        f"{symbol}: "
                        "no usable data"
                    )

                    continue

                close, volume = (
                    extracted
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Do NOT require only 280 rows.
                #
                # The actual screening functions require
                # 273-ish rows, while RS requires 252+1.
                # ------------------------------------------------

                if len(close) < (
                    screener.LOOKBACK_DAYS + 2
                ):

                    print(
                        f"{symbol}: "
                        "insufficient history"
                    )

                    continue

                print(
                    f"{symbol}: "
                    "calculating historical "
                    "screen..."
                )

                signals = (
                    compute_signals_for_stock(

                        close,

                        volume,

                        benchmark,

                        trading_days
                    )
                )

                if signals is None:

                    print(
                        f"{symbol}: "
                        "no historical signals"
                    )

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
                ] = signals

                screen_days = int(
                    signals[
                        "screen_pass"
                    ].sum()
                )

                print(
                    f"{symbol}: "
                    f"{screen_days} "
                    "screen-pass days"
                )

            except Exception as e:

                print(
                    f"{symbol}: "
                    f"FAILED — "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                continue

        time.sleep(1)

    # ========================================================
    # VALIDATION
    # ========================================================

    if not all_signals:

        raise RuntimeError(
            "No historical signals "
            "were generated."
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SIGNAL CALCULATION COMPLETE"
    )

    print(
        f"Stocks with usable data: "
        f"{len(all_signals)}"
    )

    total_screen_days = sum(
        int(
            df[
                "screen_pass"
            ].sum()
        )
        for df in all_signals.values()
    )

    print(
        f"Total screen-pass "
        f"stock-days: "
        f"{total_screen_days:,}"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    (
        trade_df,
        equity_df,
        open_df,
        final_value
    ) = run_backtest(

        all_signals,

        trading_days
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = make_summary(

        trade_df,

        equity_df,

        final_value
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL RESULT"
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

    # ========================================================
    # OUTPUT
    # ========================================================

    write_to_google_sheet(

        trade_df,

        equity_df,

        open_df,

        summary
    )

    print(
        "\nBACKTEST COMPLETE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
