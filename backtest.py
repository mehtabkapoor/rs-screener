"""
RS Screener Backtest
====================

ENTRY:
  Blue Dot (new 250-day RS high)
  +
  Price Trend Template PASS (7/7)
  +
  RS Line Trend Template PASS (7/7)

  Candidates are sorted by RAW RS SCORE.
  Maximum 10 holdings.

HOLD:
  Once a stock is bought, it remains held even if:
    - it loses its Blue Dot
    - its RS Score falls
    - it leaves the top 10
    - Price Trend Template subsequently fails
    - RS Line Trend Template subsequently fails

EXIT:
  ONLY when the RS Line crosses BELOW its 5-day EMA.

  Cross definition:
      Yesterday RS Line >= yesterday RS 5-EMA
      AND
      Today RS Line < today RS 5-EMA

EXECUTION:
  - Signal is evaluated using today's closing data.
  - Exit occurs at today's close.
  - New entries occur at today's close.
  - A new entry starts contributing to portfolio return TOMORROW.
  - Equal-weight portfolio.
  - Maximum 10 holdings.
  - When a position exits, the highest-ranked qualifying candidate
    fills the vacant slot.

IMPORTANT CAVEATS:
  - No brokerage, STT, slippage or market impact.
  - Returns are GROSS.
  - Current stocks.csv universe is projected backward.
    Survivorship bias therefore remains.
  - Equal-weight daily return approximation.
  - No look-ahead in entry returns:
      today's new purchases do not receive today's price movement.
  - Benchmark: ^CRSLDX with ^NSEI fallback.
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
# CONFIGURATION
# ================================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

LOOKBACK_DAYS = 250

# Need enough history for:
# 252-day RS score
# 200-day SMA
# 21-day slope comparison
# 250-day RS high
DOWNLOAD_PERIOD = "24mo"

STOCKS_FILE = "stocks.csv"

TOP_N = 10

BACKTEST_START = "2025-04-01"

MIN_PRICE = 20

MIN_AVG_VOLUME = 100000

SHEET_ID_ENV = "SHEET_ID"

CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest"


# ================================================================
# LOAD STOCK UNIVERSE
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

            data = yf.download(
                tkr,
                period=DOWNLOAD_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if not data.empty:

                print(
                    f"Benchmark loaded: {tkr}"
                )

                close = data["Close"]

                # Handle possible MultiIndex
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                close = close.dropna()

                return close

        except Exception as e:

            print(
                f"Benchmark {tkr} failed: {e}"
            )

    raise RuntimeError(
        "Could not download any benchmark index data."
    )


# ================================================================
# MINERVINI 7/7 TREND TEMPLATE
# ================================================================

def trend_template_series(s):

    sma50 = s.rolling(50).mean()

    sma150 = s.rolling(150).mean()

    sma200 = s.rolling(200).mean()

    sma200_1mo = sma200.shift(21)

    low52 = s.rolling(252).min()

    high52 = s.rolling(252).max()


    # ------------------------------------------------------------
    # Criterion 1
    # Price above 150 SMA and 200 SMA
    # ------------------------------------------------------------

    c1 = (
        (s > sma150)
        &
        (s > sma200)
    )


    # ------------------------------------------------------------
    # Criterion 2
    # 150 SMA above 200 SMA
    # ------------------------------------------------------------

    c2 = (
        sma150 > sma200
    )


    # ------------------------------------------------------------
    # Criterion 3
    # 200 SMA rising
    # ------------------------------------------------------------

    c3 = (
        sma200 > sma200_1mo
    )


    # ------------------------------------------------------------
    # Criterion 4
    # 50 SMA above 150 and 200 SMA
    # ------------------------------------------------------------

    c4 = (
        (sma50 > sma150)
        &
        (sma50 > sma200)
    )


    # ------------------------------------------------------------
    # Criterion 5
    # Price above 50 SMA
    # ------------------------------------------------------------

    c5 = (
        s > sma50
    )


    # ------------------------------------------------------------
    # Criterion 6
    # Price at least 25% above 52-week low
    # ------------------------------------------------------------

    c6 = (
        s >= 1.25 * low52
    )


    # ------------------------------------------------------------
    # Criterion 7
    # Price within 25% of 52-week high
    # ------------------------------------------------------------

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


    passed = (
        met == 7
    )


    return passed


# ================================================================
# COMPUTE ALL SIGNALS FOR ONE STOCK
# ================================================================

def compute_signals_for_stock(
    close,
    bench_close
):

    aligned = pd.concat(
        [close, bench_close],
        axis=1,
        join="inner"
    ).dropna()

    aligned.columns = [
        "s",
        "b"
    ]


    if len(aligned) < 280:

        return None


    # ============================================================
    # RS LINE
    # ============================================================

    # Relative strength line:
    #
    # Stock price / Benchmark price
    #
    rs_line = (
        aligned["s"]
        /
        aligned["b"]
    )


    # ============================================================
    # RAW RS SCORE
    # ============================================================

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


    # ============================================================
    # RS LINE 5-DAY EMA
    # ============================================================

    rs_5ema = (
        rs_line
        .ewm(
            span=5,
            adjust=False,
            min_periods=5
        )
        .mean()
    )


    # ============================================================
    # RS LINE BEARISH CROSS
    # ============================================================

    # Exit ONLY on an actual downward crossover.
    #
    # Yesterday:
    #     RS >= EMA
    #
    # Today:
    #     RS < EMA

    rs_cross_below_5ema = (

        (
            rs_line.shift(1)
            >=
            rs_5ema.shift(1)
        )

        &

        (
            rs_line
            <
            rs_5ema
        )

    )


    # ============================================================
    # BLUE DOT
    # ============================================================

    rs_roll_high = (
        rs_line
        .rolling(
            LOOKBACK_DAYS
        )
        .max()
    )


    blue_dot = (

        (
            rs_line
            >=
            rs_roll_high
        )

        &

        (
            rs_line.shift(1)
            <
            rs_roll_high.shift(1)
        )

    )


    # ============================================================
    # PRICE TREND TEMPLATE
    # ============================================================

    tt_pass = trend_template_series(
        aligned["s"]
    )


    # ============================================================
    # RS LINE TREND TEMPLATE
    # ============================================================

    rs_tt_pass = trend_template_series(
        rs_line
    )


    # ============================================================
    # RETURN DATAFRAME
    # ============================================================

    return pd.DataFrame({

        "price":
            aligned["s"],

        "rs_score":
            rs_score,

        "rs_line":
            rs_line,

        "rs_5ema":
            rs_5ema,

        "rs_cross_below_5ema":
            rs_cross_below_5ema,

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
        f"Backtesting from "
        f"{BACKTEST_START} to today."
    )


    # ============================================================
    # BENCHMARK
    # ============================================================

    bench_close = download_benchmark()


    # ============================================================
    # STOCK SIGNAL DATABASE
    # ============================================================

    all_signals = {}


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

                period=DOWNLOAD_PERIOD,

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


        for symbol in batch:

            try:

                # ------------------------------------------------
                # Extract stock data
                # ------------------------------------------------

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
                # Basic data filters
                # ------------------------------------------------

                if close.empty:

                    continue


                if len(close) < 280:

                    continue


                if close.iloc[-1] < MIN_PRICE:

                    continue


                if (
                    volume.tail(20).mean()
                    <
                    MIN_AVG_VOLUME
                ):

                    continue


                # ------------------------------------------------
                # Compute signals
                # ------------------------------------------------

                sig = compute_signals_for_stock(

                    close,

                    bench_close

                )


                if sig is not None:

                    clean_symbol = (
                        symbol
                        .replace(".NS", "")
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
        f"Signals computed for "
        f"{len(all_signals)} stocks "
        f"with sufficient history."
    )


    # ============================================================
    # TRADING DAYS
    # ============================================================

    trading_days = (

        bench_close.index[
            bench_close.index
            >=
            pd.Timestamp(
                BACKTEST_START
            )
        ]

    )


    if len(trading_days) == 0:

        print(
            "No trading days found."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )


    # ============================================================
    # PORTFOLIO STATE
    # ============================================================

    # symbol ->
    #
    # {
    #     entry_price,
    #     entry_date
    # }

    holdings = {}


    trade_log = []


    # Portfolio starts at 1.0.
    #
    # This is later scalable to any capital amount.
    #
    # Example:
    # 1.0 = ₹10 lakh
    # 1.5 = ₹15 lakh

    equity = 1.0


    equity_curve = []


    # ============================================================
    # DAILY BACKTEST LOOP
    # ============================================================

    for date in trading_days:


        # ========================================================
        # 1. HOLDINGS BEFORE TODAY'S DECISION
        # ========================================================

        # These are the stocks that actually participated in
        # today's price movement.

        held_before_today = set(
            holdings.keys()
        )


        # ========================================================
        # 2. CALCULATE TODAY'S PORTFOLIO RETURN
        # ========================================================

        if held_before_today:

            rets = []


            for sym in held_before_today:

                df = all_signals[sym]


                if date not in df.index:

                    continue


                idx = df.index.get_loc(
                    date
                )


                if idx <= 0:

                    continue


                prev_price = (
                    df["price"]
                    .iloc[idx - 1]
                )


                curr_price = (
                    df["price"]
                    .iloc[idx]
                )


                if prev_price <= 0:

                    continue


                daily_return = (
                    curr_price
                    /
                    prev_price
                    -
                    1
                )


                rets.append(
                    daily_return
                )


            if rets:

                portfolio_return = (
                    float(
                        np.mean(rets)
                    )
                )


                equity *= (
                    1
                    +
                    portfolio_return
                )


        # ========================================================
        # 3. EXIT EXISTING POSITIONS
        # ========================================================

        # IMPORTANT:
        #
        # There is NO longer a daily top-10 exit.
        #
        # Existing holdings are ignored by the entry ranking.
        #
        # They remain held until:
        #
        # RS LINE CROSSES BELOW 5 EMA
        #
        # This is the ONLY normal exit.

        for sym in list(
            holdings.keys()
        ):

            df = all_signals[sym]


            if date not in df.index:

                continue


            row = df.loc[date]


            exit_signal = bool(
                row[
                    "rs_cross_below_5ema"
                ]
            )


            if not exit_signal:

                continue


            # ----------------------------------------------------
            # EXIT
            # ----------------------------------------------------

            entry = holdings.pop(sym)


            exit_price = float(
                row["price"]
            )


            ret = (
                exit_price
                /
                entry["entry_price"]
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
                    "RS Line crossed below 5 EMA"

            })


        # ========================================================
        # 4. BUILD TODAY'S ENTRY CANDIDATE POOL
        # ========================================================

        pool = []


        for sym, df in all_signals.items():

            if date not in df.index:

                continue


            row = df.loc[date]


            # ----------------------------------------------------
            # RS score must exist
            # ----------------------------------------------------

            if pd.isna(
                row["rs_score"]
            ):

                continue


            # ----------------------------------------------------
            # ENTRY CONDITIONS
            # ----------------------------------------------------

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


            # ----------------------------------------------------
            # Candidate
            # ----------------------------------------------------

            pool.append((

                sym,

                float(
                    row["rs_score"]
                ),

                float(
                    row["price"]
                )

            ))


        # ========================================================
        # 5. RANK ENTRY CANDIDATES
        # ========================================================

        pool.sort(

            key=lambda x: x[1],

            reverse=True

        )


        # ========================================================
        # 6. FILL EMPTY SLOTS
        # ========================================================

        available_slots = (

            TOP_N
            -
            len(holdings)

        )


        if available_slots > 0:


            for (

                sym,

                rs_score,

                price

            ) in pool:


                if available_slots <= 0:

                    break


                # ------------------------------------------------
                # Already held
                # ------------------------------------------------

                if sym in holdings:

                    continue


                # ------------------------------------------------
                # ENTER
                # ------------------------------------------------

                holdings[sym] = {

                    "entry_price":
                        price,

                    "entry_date":
                        date

                }


                available_slots -= 1


        # ========================================================
        # 7. RECORD DAILY EQUITY
        # ========================================================

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
                len(holdings)

        })


    # ============================================================
    # 8. CLOSE OPEN POSITIONS AT BACKTEST END
    # ============================================================

    last_date = trading_days[-1]


    for sym, entry in holdings.items():

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
                entry["entry_price"]
            )


        ret = (
            exit_price
            /
            entry["entry_price"]
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
                last_date.strftime(
                    "%Y-%m-%d"
                )
                +
                " (OPEN)",

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
                "Backtest end"

        })


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


    # ============================================================
    # TOTAL RETURN
    # ============================================================

    total_return_pct = round(

        (
            equity_df[
                "equity"
            ].iloc[-1]
            -
            1
        )
        *
        100,

        2

    )


    # ============================================================
    # MAX DRAWDOWN
    # ============================================================

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


    max_dd = round(
        drawdown.min(),
        2
    )


    # ============================================================
    # CLOSED TRADES
    # ============================================================

    if trade_df.empty:

        closed_trades = (
            trade_df.copy()
        )

    else:

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


    n_trades = len(
        closed_trades
    )


    # ============================================================
    # WIN RATE
    # ============================================================

    if n_trades:

        win_rate = round(

            (
                closed_trades[
                    "return_pct"
                ]
                >
                0
            ).mean()
            *
            100,

            1

        )

    else:

        win_rate = 0


    # ============================================================
    # AVERAGE TRADE
    # ============================================================

    if n_trades:

        avg_return = round(

            closed_trades[
                "return_pct"
            ].mean(),

            2

        )

    else:

        avg_return = 0


    # ============================================================
    # MEDIAN TRADE
    # ============================================================

    if n_trades:

        median_return = round(

            closed_trades[
                "return_pct"
            ].median(),

            2

        )

    else:

        median_return = 0


    # ============================================================
    # AVERAGE HOLDING PERIOD
    # ============================================================

    if n_trades:

        avg_days_held = round(

            closed_trades[
                "days_held"
            ].mean(),

            1

        )

    else:

        avg_days_held = 0


    # ============================================================
    # MEDIAN HOLDING PERIOD
    # ============================================================

    if n_trades:

        median_days_held = round(

            closed_trades[
                "days_held"
            ].median(),

            1

        )

    else:

        median_days_held = 0


    # ============================================================
    # BEST / WORST TRADE
    # ============================================================

    if n_trades:

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

        best_trade = 0

        worst_trade = 0


    # ============================================================
    # PROFIT FACTOR
    # ============================================================

    if n_trades:

        gross_profit = (
            closed_trades.loc[
                closed_trades[
                    "return_pct"
                ] > 0,
                "return_pct"
            ].sum()
        )


        gross_loss = abs(

            closed_trades.loc[
                closed_trades[
                    "return_pct"
                ] < 0,
                "return_pct"
            ].sum()

        )


        if gross_loss > 0:

            profit_factor = round(

                gross_profit
                /
                gross_loss,

                2

            )

        else:

            profit_factor = np.inf

    else:

        profit_factor = 0


    # ============================================================
    # AVERAGE WINNER
    # ============================================================

    winners = closed_trades[
        closed_trades[
            "return_pct"
        ] > 0
    ]


    losers = closed_trades[
        closed_trades[
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


    # ============================================================
    # EXIT REASONS
    # ============================================================

    if n_trades:

        exit_reason_counts = (

            closed_trades[
                "exit_reason"
            ]
            .value_counts()
            .to_dict()

        )

    else:

        exit_reason_counts = {}


    # ============================================================
    # AVERAGE DAILY RETURN
    # ============================================================

    daily_returns = (
        equity_df[
            "equity"
        ].pct_change()
        .dropna()
    )


    if len(
        daily_returns
    ):

        avg_daily_return = round(

            daily_returns.mean()
            *
            100,

            4

        )

    else:

        avg_daily_return = 0


    # ============================================================
    # DAILY VOLATILITY
    # ============================================================

    if len(
        daily_returns
    ) > 1:

        daily_volatility = round(

            daily_returns.std()
            *
            100,

            4

        )

    else:

        daily_volatility = 0


    # ============================================================
    # DAILY SHARPE
    # ============================================================

    if (

        len(
            daily_returns
        ) > 1

        and

        daily_returns.std() > 0

    ):

        daily_sharpe = round(

            (
                daily_returns.mean()
                /
                daily_returns.std()
            )
            *
            np.sqrt(252),

            2

        )

    else:

        daily_sharpe = 0


    # ============================================================
    # SUMMARY
    # ============================================================

    return {

        "Total Return (gross, no costs)":
            f"{total_return_pct}%",

        "Max Drawdown":
            f"{max_dd}%",

        "Number of Closed Trades":
            n_trades,

        "Winning Trades":
            len(winners),

        "Losing Trades":
            len(losers),

        "Win Rate":
            f"{win_rate}%",

        "Average Trade":
            f"{avg_return}%",

        "Median Trade":
            f"{median_return}%",

        "Average Winner":
            f"{avg_winner}%",

        "Average Loser":
            f"{avg_loser}%",

        "Profit Factor":
            profit_factor,

        "Best Trade":
            f"{best_trade}%",

        "Worst Trade":
            f"{worst_trade}%",

        "Average Days Held":
            avg_days_held,

        "Median Days Held":
            median_days_held,

        "Average Daily Return":
            f"{avg_daily_return}%",

        "Daily Volatility":
            f"{daily_volatility}%",

        "Daily Sharpe":
            daily_sharpe,

        "Exit Reasons":
            str(exit_reason_counts),

        "Backtest Window":
            (
                f"{BACKTEST_START}"
                f" to "
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


    # ============================================================
    # NO GOOGLE CREDENTIALS
    # ============================================================

    if (
        not sheet_id
        or
        not creds_json
    ):

        print(
            "Missing SHEET_ID/"
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


        return


    # ============================================================
    # GOOGLE AUTH
    # ============================================================

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


    # ============================================================
    # WORKSHEET SIZE
    # ============================================================

    n_rows_needed = (

        len(trade_df)

        +

        len(equity_df)

        +

        50

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
                cols=12
            )


    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(

            title=
                BACKTEST_WORKSHEET,

            rows=
                n_rows_needed,

            cols=
                12

        )


    # ============================================================
    # CLEAR OLD RESULTS
    # ============================================================

    ws.clear()


    # ============================================================
    # TIMESTAMP
    # ============================================================

    from datetime import datetime


    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )


    ws.update(

        [[
            "Backtest run: "
            + timestamp
            +
            " | RS 5EMA EXIT | "
            "GROSS returns | "
            "No brokerage/STT/slippage"
        ]],

        "A1"

    )


    # ============================================================
    # SUMMARY
    # ============================================================

    summary_rows = [

        ["Summary", ""]

    ]


    summary_rows += [

        [k, v]

        for k, v in summary.items()

    ]


    ws.update(

        summary_rows,

        "A3"

    )


    # ============================================================
    # TRADE LOG
    # ============================================================

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


    # ============================================================
    # EQUITY CURVE
    # ============================================================

    equity_start_row = (

        trade_header_row

        +

        len(trade_df)

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
        f"Backtest results written "
        f"to '{BACKTEST_WORKSHEET}' tab."
    )


    print(
        f"Trades: {len(trade_df)}"
    )


    print(
        f"Trading days: "
        f"{len(equity_df)}"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    trades, equity = run_backtest()


    summary = compute_summary(

        trades,

        equity

    )


    print(
        "\n"
        "================================================"
    )

    print(
        "                 BACKTEST SUMMARY"
    )

    print(
        "================================================"
    )


    for k, v in summary.items():

        print(
            f"{k}: {v}"
        )


    print(
        "================================================"
    )


    write_to_sheet(

        trades,

        equity,

        summary

    )