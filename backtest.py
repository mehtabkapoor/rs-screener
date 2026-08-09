"""
RS Screener Backtest

EXACT SELECTION RULE:
    Blue Dot (new 250-day RS high)
    +
    Price Trend Template PASS (7/7)
    +
    RS Line Trend Template PASS (7/7)
    +
    Sort by raw RS Score
    +
    Top 10

EXIT:
    RS LINE < RS LINE 3-EMA

IMPORTANT:
    RS exit is STATE-BASED, NOT CROSSOVER-BASED.

    Exit whenever:
        RS Line < RS Line 3-EMA

    It does NOT require:
        Yesterday RS Line > EMA
        AND
        Today RS Line < EMA

DATA:
    Historical data is downloaded automatically from Yahoo Finance.

    Download starts exactly 3 years before BACKTEST_START.

EXAMPLE:
    BACKTEST_START = "2016-04-01"
    DOWNLOAD_START = "2013-04-01"

EXECUTION:
    Existing model is retained:
    today's close is used for the signal and exit.

CAVEATS:
    - No brokerage/STT/slippage in this version.
    - Equal-weight approximation.
    - Current stocks.csv universe creates survivorship bias.
    - Historical current-price/current-volume filtering has been REMOVED
      because using today's liquidity to select stocks for a 2016 trade
      would introduce look-ahead bias.
    - Entry-day price movement is not credited.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os


# ============================================================
# CONFIG
# ============================================================

BENCHMARK = "^CRSLDX"

BENCHMARK_FALLBACK = "^NSEI"


# ------------------------------------------------------------
# RS NEW HIGH LOOKBACK
# ------------------------------------------------------------

LOOKBACK_DAYS = 250


# ------------------------------------------------------------
# DOWNLOAD HISTORY
# ------------------------------------------------------------
#
# Data begins exactly this many years before BACKTEST_START.
#
# Example:
#
# BACKTEST_START = 2016-04-01
# DOWNLOAD_START = 2013-04-01
#
# ------------------------------------------------------------

DOWNLOAD_YEARS_BEFORE_START = 3


# ------------------------------------------------------------
# STOCK UNIVERSE
# ------------------------------------------------------------

STOCKS_FILE = "stocks.csv"


# ------------------------------------------------------------
# PORTFOLIO
# ------------------------------------------------------------

TOP_N = 10


# ------------------------------------------------------------
# RS EXIT
# ------------------------------------------------------------

RS_EXIT_EMA = 3


# ------------------------------------------------------------
# BACKTEST PERIOD
# ------------------------------------------------------------

BACKTEST_START = "2016-04-01"

BACKTEST_END = "2026-08-07"


# ------------------------------------------------------------
# GOOGLE SHEETS
# ------------------------------------------------------------

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"


# ============================================================
# CALCULATE DOWNLOAD DATES
# ============================================================

def get_download_dates():

    """
    Calculates the historical Yahoo Finance download window.

    Example:

        BACKTEST_START = 2016-04-01

        DOWNLOAD_START = 2013-04-01

        BACKTEST_END = 2026-08-07

    The download end is one day after BACKTEST_END because
    yfinance's 'end' parameter is exclusive.
    """

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
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d")
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
# DOWNLOAD BENCHMARK
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
        f"End:   {download_end}"
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

                print(
                    f"No data returned for {ticker}"
                )

                continue

            print(
                f"Benchmark loaded: {ticker}"
            )

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

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: {e}"
            )

    raise RuntimeError(
        "Could not download any benchmark index data."
    )


# ============================================================
# MINERVINI TREND TEMPLATE
# ============================================================

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
        sma200
        .shift(21)
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

    # --------------------------------------------------------
    # 7 TREND TEMPLATE CRITERIA
    # --------------------------------------------------------

    c1 = (
        s > sma150
    ) & (
        s > sma200
    )

    c2 = (
        sma150 > sma200
    )

    c3 = (
        sma200 > sma200_1mo
    )

    c4 = (
        sma50 > sma150
    ) & (
        sma50 > sma200
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
        met == 7
    )


# ============================================================
# COMPUTE SIGNALS FOR ONE STOCK
# ============================================================

def compute_signals_for_stock(
    close,
    bench_close
):

    """
    Calculates all historical signals for one stock.
    """

    # --------------------------------------------------------
    # ALIGN STOCK + BENCHMARK
    # --------------------------------------------------------

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

    # Need enough data for:
    # 252-day RS score
    # 252-day TT
    # + 21-day SMA slope

    if len(aligned) < 280:

        return None

    # --------------------------------------------------------
    # RS LINE
    # --------------------------------------------------------

    rs_ratio = (
        aligned["s"]
        /
        aligned["b"]
    )

    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BLUE DOT
    # --------------------------------------------------------
    #
    # New 250-day RS high.
    #
    # IMPORTANT:
    # The previous 250 observations are used.
    # Today's RS is not allowed to establish its own
    # comparison high.
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
        >
        previous_rs_high
    )

    # --------------------------------------------------------
    # PRICE TREND TEMPLATE
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
    # RS LINE 3-EMA
    # --------------------------------------------------------

    rs_exit_ema = (
        rs_ratio
        .ewm(
            span=RS_EXIT_EMA,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------
    #
    # EXACT RULE:
    #
    #     RS LINE < RS LINE 3-EMA
    #
    # STATE BASED.
    #
    # No crossover requirement.
    # --------------------------------------------------------

    rs_exit = (
        rs_ratio
        <
        rs_exit_ema
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return pd.DataFrame({

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

        "rs_exit_ema":
            rs_exit_ema,

        "rs_exit":
            rs_exit,

    })


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest():

    # --------------------------------------------------------
    # LOAD STOCK UNIVERSE
    # --------------------------------------------------------

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} tickers."
    )

    # --------------------------------------------------------
    # CALCULATE DOWNLOAD WINDOW
    # --------------------------------------------------------

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\n=============================================="
    )

    print(
        "BACKTEST CONFIGURATION"
    )

    print(
        "=============================================="
    )

    print(
        f"Download start : {download_start}"
    )

    print(
        f"Backtest start : {BACKTEST_START}"
    )

    print(
        f"Backtest end   : {BACKTEST_END}"
    )

    print(
        f"Pre-backtest history: "
        f"{DOWNLOAD_YEARS_BEFORE_START} years"
    )

    print(
        f"RS high lookback: "
        f"{LOOKBACK_DAYS} trading days"
    )

    print(
        f"RS exit: "
        f"RS Line < {RS_EXIT_EMA}-EMA"
    )

    print(
        f"Top N: {TOP_N}"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    bench_close = (
        download_benchmark()
    )

    # --------------------------------------------------------
    # STOCK SIGNALS
    # --------------------------------------------------------

    all_signals = {}

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
            f"{i} - "
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
                f"Batch download failed: {e}"
            )

            continue

        # ----------------------------------------------------
        # PROCESS EACH STOCK
        # ----------------------------------------------------

        for symbol in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    if symbol not in data.columns:

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

                if close.empty:

                    continue

                # ------------------------------------------------
                # IMPORTANT:
                #
                # NO CURRENT PRICE FILTER.
                #
                # NO CURRENT VOLUME FILTER.
                #
                # Those would create look-ahead bias when testing
                # historical dates.
                # ------------------------------------------------

                sig = (
                    compute_signals_for_stock(
                        close,
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
                    f"Skipping {symbol}: {e}"
                )

                continue

        time.sleep(1)

    print(
        f"\nSignals computed for "
        f"{len(all_signals)} stocks."
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

    if len(trading_days) == 0:

        print(
            "No trading days found "
            "in the backtest window."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    print(
        f"\nTrading days: "
        f"{len(trading_days)}"
    )

    # ========================================================
    # PORTFOLIO STATE
    # ========================================================

    holdings = {}

    trade_log = []

    equity = 1.0

    equity_curve = []

    # ========================================================
    # DAILY BACKTEST
    # ========================================================

    for date in trading_days:

        # ----------------------------------------------------
        # BUILD ELIGIBLE POOL
        # ----------------------------------------------------

        pool = []

        for sym, df in all_signals.items():

            if date not in df.index:

                continue

            row = df.loc[date]

            if pd.isna(
                row["rs_score"]
            ):

                continue

            # EXACT ENTRY RULE

            if (

                bool(
                    row["blue_dot"]
                )

                and

                bool(
                    row["tt_pass"]
                )

                and

                bool(
                    row["rs_tt_pass"]
                )

            ):

                pool.append(

                    (

                        sym,

                        float(
                            row["rs_score"]
                        ),

                        float(
                            row["price"]
                        )

                    )

                )

        # ----------------------------------------------------
        # SORT BY RAW RS SCORE
        # ----------------------------------------------------

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ----------------------------------------------------
        # TOP 10
        # ----------------------------------------------------

        target = pool[
            :TOP_N
        ]

        target_syms = {

            s

            for s, _, _

            in target

        }

        target_prices = {

            s: p

            for s, _, p

            in target

        }

        # ====================================================
        # HOLDINGS BEFORE TODAY'S DECISION
        # ====================================================

        held_before_today = set(
            holdings.keys()
        )

        # ====================================================
        # DAILY PORTFOLIO RETURN
        # ====================================================
        #
        # Only stocks already held before today's decisions
        # contribute today's price move.
        #
        # New positions bought at today's close begin
        # contributing from the next trading day.
        # ====================================================

        if held_before_today:

            rets = []

            for sym in held_before_today:

                df = all_signals[
                    sym
                ]

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
                        df["price"]
                        .iloc[
                            idx - 1
                        ]
                    )

                    curr_price = (
                        df["price"]
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

                            curr_price
                            /
                            prev_price
                            -
                            1

                        )

            if rets:

                equity *= (

                    1

                    +

                    float(
                        np.mean(rets)
                    )

                )

        # ====================================================
        # EXIT LOGIC
        # ====================================================
        #
        # Exit if:
        #
        # 1. RS Line < RS Line 3-EMA
        #
        # OR
        #
        # 2. Stock no longer belongs to Top 10.
        #
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            df = all_signals[
                sym
            ]

            if date not in df.index:

                continue

            row = df.loc[date]

            # ------------------------------------------------
            # RS EXIT
            # ------------------------------------------------

            rs_exit = bool(
                row["rs_exit"]
            )

            # ------------------------------------------------
            # TOP-N EXIT
            # ------------------------------------------------

            target_exit = (
                sym
                not in
                target_syms
            )

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if (
                rs_exit
                or
                target_exit
            ):

                entry = holdings.pop(
                    sym
                )

                exit_price = float(
                    row["price"]
                )

                ret = (

                    exit_price
                    /
                    entry[
                        "entry_price"
                    ]
                    -
                    1

                ) * 100

                # ------------------------------------------------
                # REASON
                # ------------------------------------------------

                if rs_exit:

                    exit_reason = (

                        f"RS Line < "
                        f"RS Line {RS_EXIT_EMA}-EMA"

                    )

                else:

                    exit_reason = (
                        "No longer in Top-N target"
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
                            date
                            -
                            entry[
                                "entry_date"
                            ]
                        ).days,

                    "exit_reason":
                        exit_reason,

                })

        # ====================================================
        # ENTER NEW TARGET NAMES
        # ====================================================
        #
        # Entry is at today's close.
        # Therefore today's already completed movement is not
        # credited to the new position.
        # ====================================================

        for sym in target_syms:

            if sym not in holdings:

                holdings[sym] = {

                    "entry_price":
                        target_prices[
                            sym
                        ],

                    "entry_date":
                        date

                }

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
                len(
                    holdings
                )

        })

    # ========================================================
    # CLOSE OPEN POSITIONS AT BACKTEST END
    # ========================================================

    last_date = (
        trading_days[-1]
    )

    for sym, entry in holdings.items():

        df = all_signals[
            sym
        ]

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

            exit_price
            /
            entry[
                "entry_price"
            ]
            -
            1

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
                (
                    last_date.strftime(
                        "%Y-%m-%d"
                    )
                    +
                    " (OPEN)"
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
                    last_date
                    -
                    entry[
                        "entry_date"
                    ]
                ).days,

            "exit_reason":
                "BACKTEST END",

        })

    # ========================================================
    # DATAFRAMES
    # ========================================================

    trade_df = pd.DataFrame(
        trade_log
    )

    equity_df = pd.DataFrame(
        equity_curve
    )

    return (
        trade_df,
        equity_df
    )


# ============================================================
# SUMMARY
# ============================================================

def compute_summary(
    trade_df,
    equity_df
):

    if equity_df.empty:

        return {}

    # FIX:
    # Calculate download_start INSIDE this function.
    # This prevents the previous NameError.
    download_start, download_end = (
        get_download_dates()
    )

    # --------------------------------------------------------
    # TOTAL RETURN
    # --------------------------------------------------------

    total_return_pct = round(

        (

            equity_df[
                "equity"
            ].iloc[-1]

            -

            1

        )

        * 100,

        2

    )

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

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

        -

        1

    ) * 100

    max_dd = round(
        drawdown.min(),
        2
    )

    # --------------------------------------------------------
    # CLOSED TRADES
    # --------------------------------------------------------

    if not trade_df.empty:

        closed_trades = (

            trade_df[

                ~

                trade_df[
                    "exit_date"
                ]
                .astype(str)
                .str.contains(
                    "OPEN",
                    na=False
                )

            ]

        )

    else:

        closed_trades = (
            trade_df
        )

    n_trades = len(
        closed_trades
    )

    # --------------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------------

    if n_trades:

        win_rate = round(

            (

                closed_trades[
                    "return_pct"
                ]

                >

                0

            )

            .mean()

            *

            100,

            1

        )

        avg_return = round(

            closed_trades[
                "return_pct"
            ]

            .mean(),

            2

        )

        avg_days_held = round(

            closed_trades[
                "days_held"
            ]

            .mean(),

            1

        )

        best_trade = (

            closed_trades[
                "return_pct"
            ].max()

        )

        worst_trade = (

            closed_trades[
                "return_pct"
            ].min()

        )

    else:

        win_rate = 0

        avg_return = 0

        avg_days_held = 0

        best_trade = 0

        worst_trade = 0

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    return {

        "Total Return (gross, no costs)":
            f"{total_return_pct}%",

        "Max Drawdown":
            f"{max_dd}%",

        "Number of Closed Trades":
            n_trades,

        "Win Rate":
            f"{win_rate}%",

        "Avg Return per Trade":
            f"{avg_return}%",

        "Avg Days Held":
            avg_days_held,

        "Best Trade":
            f"{best_trade}%",

        "Worst Trade":
            f"{worst_trade}%",

        "RS Exit Rule":
            (
                f"RS Line < "
                f"RS Line {RS_EXIT_EMA}-EMA"
            ),

        "Exit Type":
            "STATE-BASED, NOT CROSSOVER",

        "Download Start":
            download_start,

        "Download End":
            download_end,

        "Backtest Start":
            BACKTEST_START,

        "Backtest End":
            BACKTEST_END,

    }


# ============================================================
# WRITE TO GOOGLE SHEET
# ============================================================

def write_to_sheet(
    trade_df,
    equity_df,
    summary
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    # --------------------------------------------------------
    # FALLBACK TO CSV
    # --------------------------------------------------------

    if not sheet_id or not creds_json:

        print(
            "Missing "
            "SHEET_ID/GOOGLE_CREDENTIALS "
            "-- saving to CSV instead."
        )

        trade_df.to_csv(
            "backtest_trades.csv",
            index=False
        )

        equity_df.to_csv(
            "backtest_equity.csv",
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

    # --------------------------------------------------------
    # WORKSHEET
    # --------------------------------------------------------

    n_rows_needed = max(

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
            n_rows_needed
        ):

            ws.resize(
                rows=n_rows_needed,
                cols=10
            )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(

            title=BACKTEST_WORKSHEET,

            rows=n_rows_needed,

            cols=10

        )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    ws.clear()

    from datetime import datetime

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    ws.update(

        [

            [

                (

                    f"Backtest run: "
                    f"{timestamp} | "
                    f"GROSS returns | "
                    f"RS exit = "
                    f"RS Line < "
                    f"RS Line {RS_EXIT_EMA}-EMA"

                )

            ]

        ],

        "A1"

    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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
            for k, v
            in summary.items()
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

        3
        +
        len(summary_rows)
        +
        2

    )

    ws.update(

        [
            [
                "Trade Log"
            ]
        ],

        f"A{trade_start_row}"

    )

    trade_header_row = (

        trade_start_row
        +
        1

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

            f"A{trade_header_row}"

        )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity_start_row = (

        trade_header_row
        +
        len(trade_df)
        +
        3

    )

    ws.update(

        [
            [
                "Daily Equity Curve"
            ]
        ],

        f"A{equity_start_row}"

    )

    equity_header_row = (

        equity_start_row
        +
        1

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

            f"A{equity_header_row}"

        )

    print(

        f"\nBacktest results written "
        f"to '{BACKTEST_WORKSHEET}' tab: "

        f"{len(trade_df)} trades, "

        f"{len(equity_df)} trading days."

    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        trades, equity = (
            run_backtest()
        )

        summary = (
            compute_summary(
                trades,
                equity
            )
        )

        print(
            "\n=============================================="
        )

        print(
            "SUMMARY"
        )

        print(
            "=============================================="
        )

        for k, v in summary.items():

            print(
                f"{k}: {v}"
            )

        print(
            "=============================================="
        )

        write_to_sheet(

            trades,

            equity,

            summary

        )

        print(
            "\nBACKTEST COMPLETED SUCCESSFULLY."
        )

    except Exception as e:

        print(
            "\n=============================================="
        )

        print(
            "BACKTEST FAILED"
        )

        print(
            "=============================================="
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise