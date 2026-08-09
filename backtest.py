"""
RS Screener Backtest
Simulates the EXACT live selection rule over history:
  Blue Dot (new 250-day RS high) + price Trend Template PASS (7/7) +
  RS Line Trend Template PASS (7/7), sorted by raw RS Score, top 10,
  rebalanced daily to exact membership.

BACKTEST:
  Start: 2015-01-01
  Data download starts: 2013-01-01
  The earlier data provides sufficient warm-up for:
    - 252-day RS score
    - 250-day RS high
    - 200-day moving averages
    - 21-day SMA200 slope
    - 252-day 52-week high/low

Outputs:
  - Full trade log
  - Daily equity curve
  - Summary statistics

Writes results to the 'Backtest' tab in your Google Sheet.

IMPORTANT CAVEATS:
  - No brokerage, STT, or slippage costs are modeled.
  - Equal-weight daily-reset assumption.
  - Uses CURRENT stocks.csv projected backward, so survivorship bias exists.
  - Current price and volume filters are applied exactly as in the original
    script. This can remove historically valid stocks if they currently fail
    the filters.
  - Entry-day accounting: a stock bought at today's close starts contributing
    from tomorrow.
  - No look-ahead is introduced by the signal calculations.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import json
import os


# ================================================================
# CONFIG
# ================================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

LOOKBACK_DAYS = 250

# Historical data starts before the actual backtest so that all
# indicators have sufficient warm-up history.
DOWNLOAD_START = "2013-01-01"
DOWNLOAD_END = None

STOCKS_FILE = "stocks.csv"

TOP_N = 10

# ACTUAL BACKTEST START
BACKTEST_START = "2015-01-01"

MIN_PRICE = 10
MIN_AVG_VOLUME = 10000

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"


# ================================================================
# LOAD TICKERS
# ================================================================

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


# ================================================================
# DOWNLOAD BENCHMARK
# ================================================================

def download_benchmark():

    for tkr in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            print(
                f"Downloading benchmark {tkr} "
                f"from {DOWNLOAD_START}..."
            )

            data = yf.download(
                tkr,
                start=DOWNLOAD_START,
                end=DOWNLOAD_END,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if not data.empty:

                print(
                    f"Benchmark loaded: {tkr}"
                )

                close = data["Close"]

                # Handle possible MultiIndex returned by yfinance
                if isinstance(close, pd.DataFrame):

                    close = close.iloc[:, 0]

                return close

        except Exception as e:

            print(
                f"Benchmark {tkr} failed: {e}"
            )

    raise RuntimeError(
        "Could not download any benchmark index data."
    )


# ================================================================
# MINERVINI TREND TEMPLATE
# ================================================================

def trend_template_series(s):

    """
    Vectorized Minervini 7-criteria Trend Template.

    Criteria:

    1. Price > 150 SMA and Price > 200 SMA
    2. 150 SMA > 200 SMA
    3. 200 SMA rising over approximately 1 month
    4. 50 SMA > 150 SMA and 50 SMA > 200 SMA
    5. Price > 50 SMA
    6. Price >= 125% of 52-week low
    7. Price >= 75% of 52-week high
    """

    sma50 = s.rolling(50).mean()

    sma150 = s.rolling(150).mean()

    sma200 = s.rolling(200).mean()

    sma200_1mo = sma200.shift(21)

    low52 = s.rolling(252).min()

    high52 = s.rolling(252).max()

    c1 = (
        (s > sma150)
        & (s > sma200)
    )

    c2 = (
        sma150 > sma200
    )

    c3 = (
        sma200 > sma200_1mo
    )

    c4 = (
        (sma50 > sma150)
        & (sma50 > sma200)
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

    passed = (
        met == 7
    )

    return passed


# ================================================================
# COMPUTE SIGNALS FOR ONE STOCK
# ================================================================

def compute_signals_for_stock(
    close,
    bench_close
):

    """
    Returns per-date:

      price
      rs_score
      blue_dot
      tt_pass
      rs_tt_pass
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

    # Need enough history for all calculations.
    if len(aligned) < 280:

        return None

    # ------------------------------------------------------------
    # RS LINE
    # ------------------------------------------------------------

    rs_ratio = (
        aligned["s"]
        / aligned["b"]
    )

    # ------------------------------------------------------------
    # RS SCORE
    # ------------------------------------------------------------

    def pct_return(
        series,
        days
    ):

        return (
            series
            / series.shift(days)
            - 1
        )

    rs_score = (

        0.40
        * pct_return(
            aligned["s"],
            63
        )

        + 0.20
        * pct_return(
            aligned["s"],
            126
        )

        + 0.20
        * pct_return(
            aligned["s"],
            189
        )

        + 0.20
        * pct_return(
            aligned["s"],
            252
        )

    ) * 100

    # ------------------------------------------------------------
    # BLUE DOT
    #
    # RS line makes a NEW 250-day high.
    # ------------------------------------------------------------

    rs_roll_high = (
        rs_ratio
        .rolling(LOOKBACK_DAYS)
        .max()
    )

    blue_dot = (

        (rs_ratio >= rs_roll_high)

        & (

            rs_ratio.shift(1)
            < rs_roll_high.shift(1)

        )

    )

    # ------------------------------------------------------------
    # PRICE TREND TEMPLATE
    # ------------------------------------------------------------

    tt_pass = (
        trend_template_series(
            aligned["s"]
        )
    )

    # ------------------------------------------------------------
    # RS LINE TREND TEMPLATE
    # ------------------------------------------------------------

    rs_tt_pass = (
        trend_template_series(
            rs_ratio
        )
    )

    # ------------------------------------------------------------
    # RETURN
    # ------------------------------------------------------------

    return pd.DataFrame({

        "price":
            aligned["s"],

        "rs_score":
            rs_score,

        "blue_dot":
            blue_dot,

        "tt_pass":
            tt_pass,

        "rs_tt_pass":
            rs_tt_pass

    })


# ================================================================
# RUN BACKTEST
# ================================================================

def run_backtest():

    tickers = load_tickers()

    print(
        f"Loaded {len(tickers)} tickers."
    )

    print(
        f"Downloading historical data "
        f"from {DOWNLOAD_START}..."
    )

    print(
        f"Actual backtest starts "
        f"{BACKTEST_START}."
    )

    # ------------------------------------------------------------
    # BENCHMARK
    # ------------------------------------------------------------

    bench_close = (
        download_benchmark()
    )

    # ------------------------------------------------------------
    # SIGNAL STORAGE
    # ------------------------------------------------------------

    all_signals = {}

    batch_size = 50

    # ------------------------------------------------------------
    # DOWNLOAD STOCK DATA
    # ------------------------------------------------------------

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
            f"{i}-{i + len(batch)} "
            f"of {len(tickers)}..."
        )

        try:

            data = yf.download(

                batch,

                start=DOWNLOAD_START,

                end=DOWNLOAD_END,

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

        # --------------------------------------------------------
        # PROCESS EACH STOCK
        # --------------------------------------------------------

        for symbol in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    sdata = data[symbol]

                close = (
                    sdata["Close"]
                    .dropna()
                )

                volume = (
                    sdata["Volume"]
                    .dropna()
                )

                # ------------------------------------------------
                # BASIC HISTORY CHECK
                # ------------------------------------------------

                if (
                    close.empty
                    or len(close) < 280
                ):

                    continue

                # ------------------------------------------------
                # CURRENT PRICE FILTER
                #
                # Kept exactly as original script.
                # ------------------------------------------------

                if (
                    close.iloc[-1]
                    < MIN_PRICE
                ):

                    continue

                # ------------------------------------------------
                # CURRENT VOLUME FILTER
                #
                # Kept exactly as original script.
                # ------------------------------------------------

                if (
                    volume.tail(20).mean()
                    < MIN_AVG_VOLUME
                ):

                    continue

                # ------------------------------------------------
                # COMPUTE SIGNALS
                # ------------------------------------------------

                sig = (
                    compute_signals_for_stock(
                        close,
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

                # Keep original behavior:
                # skip stocks that fail processing.
                continue

        time.sleep(1)

    print(
        f"Signals computed for "
        f"{len(all_signals)} stocks "
        f"with sufficient history."
    )

    # ============================================================
    # BACKTEST TRADING DAYS
    # ============================================================

    trading_days = (
        bench_close.index[
            bench_close.index
            >= pd.Timestamp(
                BACKTEST_START
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

    # ============================================================
    # PORTFOLIO STATE
    # ============================================================

    holdings = {}

    trade_log = []

    equity = 1.0

    equity_curve = []

    # ============================================================
    # DAY-BY-DAY BACKTEST
    # ============================================================

    for date in trading_days:

        # --------------------------------------------------------
        # BUILD CURRENT SIGNAL POOL
        # --------------------------------------------------------

        pool = []

        for sym, df in all_signals.items():

            if date not in df.index:

                continue

            row = df.loc[date]

            # No RS score = cannot rank.
            if pd.isna(
                row["rs_score"]
            ):

                continue

            # ----------------------------------------------------
            # EXACT ENTRY FILTER
            #
            # Blue Dot
            # +
            # Price Trend Template
            # +
            # RS Line Trend Template
            # ----------------------------------------------------

            if (

                bool(
                    row["blue_dot"]
                )

                and bool(
                    row["tt_pass"]
                )

                and bool(
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

        # --------------------------------------------------------
        # SORT BY RAW RS SCORE
        # --------------------------------------------------------

        pool.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # --------------------------------------------------------
        # TOP 10
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # SNAPSHOT HOLDINGS BEFORE TODAY'S DECISION
        # --------------------------------------------------------

        held_before_today = set(
            holdings.keys()
        )

        # ========================================================
        # DAILY PORTFOLIO RETURN
        #
        # ONLY EXISTING POSITIONS CONTRIBUTE.
        #
        # New positions bought at today's close do NOT get
        # today's return.
        # ========================================================

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
                    .get_loc(date)
                )

                if idx > 0:

                    prev_price = (
                        df["price"]
                        .iloc[idx - 1]
                    )

                    curr_price = (
                        df["price"]
                        .iloc[idx]
                    )

                    if prev_price > 0:

                        rets.append(

                            curr_price
                            / prev_price
                            - 1

                        )

            if rets:

                # Equal-weight average.
                equity *= (

                    1
                    + float(
                        np.mean(rets)
                    )

                )

        # ========================================================
        # EXIT ANYTHING NO LONGER IN TARGET
        # ========================================================

        for sym in list(
            holdings.keys()
        ):

            if sym not in target_syms:

                entry = holdings.pop(
                    sym
                )

                df = all_signals[
                    sym
                ]

                if date in df.index:

                    exit_price = float(
                        df.loc[
                            date,
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
                    / entry[
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
                            - entry[
                                "entry_date"
                            ]
                        ).days

                })

        # ========================================================
        # ENTER NEW TARGET STOCKS
        #
        # Entry price = today's close.
        #
        # Contribution begins tomorrow.
        # ========================================================

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

        # ========================================================
        # SAVE EQUITY CURVE
        # ========================================================

        equity_curve.append({

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "equity":
                round(
                    equity,
                    4
                ),

            "n_holdings":
                len(holdings)

        })

    # ============================================================
    # CLOSE OPEN POSITIONS AT FINAL DATE
    # ============================================================

    last_date = trading_days[
        -1
    ]

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
            / entry[
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
                    last_date
                    - entry[
                        "entry_date"
                    ]
                ).days

        })

    # ============================================================
    # DATAFRAMES
    # ============================================================

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


# ================================================================
# SUMMARY STATISTICS
# ================================================================

def compute_summary(
    trade_df,
    equity_df
):

    if equity_df.empty:

        return {}

    # ------------------------------------------------------------
    # TOTAL RETURN
    # ------------------------------------------------------------

    total_return_pct = round(

        (
            equity_df[
                "equity"
            ].iloc[-1]

            - 1

        ) * 100,

        2

    )

    # ------------------------------------------------------------
    # RUNNING HIGH
    # ------------------------------------------------------------

    running_max = (
        equity_df[
            "equity"
        ].cummax()
    )

    # ------------------------------------------------------------
    # DRAWDOWN
    # ------------------------------------------------------------

    drawdown = (

        equity_df[
            "equity"
        ]

        / running_max

        - 1

    ) * 100

    max_dd = round(
        drawdown.min(),
        2
    )

    # ------------------------------------------------------------
    # CLOSED TRADES ONLY
    # ------------------------------------------------------------

    if not trade_df.empty:

        closed_trades = (
            trade_df[
                ~trade_df[
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

    # ------------------------------------------------------------
    # TRADE COUNT
    # ------------------------------------------------------------

    n_trades = len(
        closed_trades
    )

    # ------------------------------------------------------------
    # WIN RATE
    # ------------------------------------------------------------

    if n_trades:

        win_rate = round(

            (
                closed_trades[
                    "return_pct"
                ] > 0
            ).mean()

            * 100,

            1

        )

    else:

        win_rate = 0

    # ------------------------------------------------------------
    # AVERAGE TRADE RETURN
    # ------------------------------------------------------------

    if n_trades:

        avg_return = round(

            closed_trades[
                "return_pct"
            ].mean(),

            2

        )

    else:

        avg_return = 0

    # ------------------------------------------------------------
    # AVERAGE HOLDING PERIOD
    # ------------------------------------------------------------

    if n_trades:

        avg_days_held = round(

            closed_trades[
                "days_held"
            ].mean(),

            1

        )

    else:

        avg_days_held = 0

    # ------------------------------------------------------------
    # BEST TRADE
    # ------------------------------------------------------------

    if n_trades:

        best_trade = (
            closed_trades[
                "return_pct"
            ].max()
        )

    else:

        best_trade = 0

    # ------------------------------------------------------------
    # WORST TRADE
    # ------------------------------------------------------------

    if n_trades:

        worst_trade = (
            closed_trades[
                "return_pct"
            ].min()
        )

    else:

        worst_trade = 0

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

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

        "Backtest Window":
            (
                f"{BACKTEST_START} to "
                f"{equity_df['date'].iloc[-1]}"
            )

    }


# ================================================================
# WRITE RESULTS TO GOOGLE SHEETS
# ================================================================

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

    # ------------------------------------------------------------
    # FALLBACK TO CSV
    # ------------------------------------------------------------

    if (
        not sheet_id
        or not creds_json
    ):

        print(
            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS "
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

    # ------------------------------------------------------------
    # GOOGLE AUTH
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # SHEET SIZE
    # ------------------------------------------------------------

    n_rows_needed = (

        len(trade_df)

        + len(equity_df)

        + 30

    )

    try:

        ws = sh.worksheet(
            BACKTEST_WORKSHEET
        )

        if (
            ws.row_count
            < n_rows_needed
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

    # ------------------------------------------------------------
    # CLEAR OLD RESULTS
    # ------------------------------------------------------------

    ws.clear()

    # ------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------

    from datetime import datetime

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    ws.update(

        [[
            f"Backtest run: {timestamp} | "
            f"BACKTEST START: {BACKTEST_START} | "
            f"DATA START: {DOWNLOAD_START} | "
            f"GROSS returns, no brokerage/STT/slippage modeled"
        ]],

        "A1"

    )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    summary_rows = (

        [
            [
                "Summary",
                ""
            ]
        ]

        +

        [
            [
                k,
                v
            ]

            for k, v
            in summary.items()
        ]

    )

    ws.update(
        summary_rows,
        "A3"
    )

    # ------------------------------------------------------------
    # TRADE LOG
    # ------------------------------------------------------------

    trade_start_row = (

        3

        + len(summary_rows)

        + 2

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
        trade_start_row + 1
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

    print(

        f"Backtest results written to "
        f"'{BACKTEST_WORKSHEET}' tab: "

        f"{len(trade_df)} trades, "

        f"{len(equity_df)} trading days."

    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

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
        "\n--- SUMMARY ---"
    )

    for k, v in summary.items():

        print(
            f"{k}: {v}"
        )

    write_to_sheet(
        trades,
        equity,
        summary
    )