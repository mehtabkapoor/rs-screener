"""
============================================================
TOP 10 DAILY RS BACKTEST — BARE BONES
============================================================

RULES
-----
1. Universe:
   stocks.csv
   Column required: symbol

2. RS RANKING:
   Rank stocks ONLY by daily RS Score.

   RS Score =
       40% × 3-month return
     + 20% × 6-month return
     + 20% × 9-month return
     + 20% × 12-month return

   No:
     - RS Line
     - Trend Template
     - Blue dots
     - Green dots
     - 52-week high filter
     - Sector filter
     - Market regime
     - Breadth
     - Any other signal

3. ENTRY:
   Hold the daily Top 10 RS stocks.

4. EXIT:
   Exit any holding that is NOT in today's Top 10.

5. EXECUTION:
   Signal on day D.
   Execute changes at day D+1 CLOSE.
   This prevents same-day look-ahead.

6. PORTFOLIO:
   - Maximum 10 stocks
   - Equal weighted
   - 10% target allocation per stock
   - Rebalance when stocks enter/exit Top 10
   - No discretionary weighting

7. FILTERS RETAINED:
   - Price > Rs.20
   - 20-day average volume > 100,000
   These are liquidity/universe filters, NOT ranking signals.

8. COSTS RETAINED:
   Buy:
     STT
     Stamp duty
     Exchange charges
     SEBI charges
     GST

   Sell:
     STT
     Exchange charges
     SEBI charges
     GST
     DP charge

   STCG:
     20% + 4% cess = 20.8%
     Applied to profitable closed trades.

9. OUTPUT:
   Google Sheet containing ONLY:

   A) DAILY TOP 10
      Date
      Rank 1–10
      RS Score 1–10
      New Entry
      Exit

   B) EQUITY CURVE
      Date
      Portfolio Value
      Equity Multiple
      Cash
      Number of Holdings
      Drawdown %

   C) TRADE LOG
      Entry/exit information
      Gross return
      Costs
      STCG
      Net P&L

   D) SUMMARY

============================================================
"""

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS_FILE = "stocks.csv"

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

BACKTEST_START = "2016-04-01"
BACKTEST_END = None       # None = latest available

DOWNLOAD_YEARS_BEFORE_START = 3

# Universe filters retained
MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

# RS calculation
TOP_N = 10

RS_3M_DAYS = 63
RS_6M_DAYS = 126
RS_9M_DAYS = 189
RS_12M_DAYS = 252

RS_WEIGHT_3M = 0.40
RS_WEIGHT_6M = 0.20
RS_WEIGHT_9M = 0.20
RS_WEIGHT_12M = 0.20

STARTING_CAPITAL = 1_000_000

# Reject obviously corrupted daily prices
MAX_PLAUSIBLE_DAILY_MOVE = 0.30


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

BACKTEST_WORKSHEET = "top 10 RS backtest"

WRITE_CHUNK_SIZE = 5000
MAX_WRITE_RETRIES = 6
INITIAL_RETRY_SECONDS = 5


# ============================================================
# DATE HELPERS
# ============================================================

def normalize_dates(index):

    idx = pd.DatetimeIndex(index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    return idx.normalize()


def normalize_series_index(series):

    s = series.copy()

    s.index = normalize_dates(s.index)

    s = s[~s.index.duplicated(keep="last")]

    return s.sort_index()


def get_download_dates():

    start = pd.Timestamp(BACKTEST_START)

    download_start = (
        start -
        pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime("%Y-%m-%d"),
            None
        )

    end = pd.Timestamp(BACKTEST_END)

    return (
        download_start.strftime("%Y-%m-%d"),
        (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )


# ============================================================
# LOAD UNIVERSE
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

    symbols = [s for s in symbols if s]

    return [
        s if s.endswith(".NS") else s + ".NS"
        for s in symbols
    ]


# ============================================================
# CLEAN PRICE DATA
# ============================================================

def clean_price_series(close):

    close = normalize_series_index(close)

    pct_change = close.pct_change()

    bad = pct_change.abs() > MAX_PLAUSIBLE_DAILY_MOVE

    n_bad = int(bad.sum())

    if n_bad == 0:
        return close, 0

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = cleaned.index.get_loc(idx)

        if pos > 0:
            cleaned.iloc[pos] = cleaned.iloc[pos - 1]

    return cleaned, n_bad


# ============================================================
# TRANSACTION COST FUNCTIONS
# ============================================================

def buy_side_cost(trade_value):

    stt = STT_RATE * trade_value

    stamp = STAMP_DUTY_RATE * trade_value

    exch = EXCHANGE_CHARGE_RATE * trade_value

    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return (
        stt +
        stamp +
        exch +
        sebi +
        gst
    )


def sell_side_cost(trade_value):

    stt = STT_RATE * trade_value

    exch = EXCHANGE_CHARGE_RATE * trade_value

    sebi = SEBI_CHARGE_RATE * trade_value

    gst = GST_RATE * (exch + sebi)

    return (
        stt +
        exch +
        sebi +
        gst +
        DP_CHARGE_FLAT
    )


def stcg_tax(net_gain):

    if net_gain <= 0:
        return 0.0

    return net_gain * STCG_EFFECTIVE_RATE


# ============================================================
# BENCHMARK
# ============================================================

def download_benchmark():

    download_start, download_end = get_download_dates()

    print(
        f"\nBenchmark download: "
        f"{download_start} → "
        f"{download_end if download_end else 'LATEST'}"
    )

    for ticker in [
        BENCHMARK,
        BENCHMARK_FALLBACK
    ]:

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

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = (
                close
                .dropna()
                .sort_index()
            )

            if close.empty:
                continue

            close, n_bad = clean_price_series(close)

            if n_bad:
                print(
                    f"{ticker}: repaired "
                    f"{n_bad} price points"
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
        "Could not download benchmark."
    )


# ============================================================
# DAILY RS SCORE
# ============================================================

def calculate_rs_score(close):

    """
    Pure price momentum.

    NO:
      RS line
      Trend template
      Blue dot
      Green dot
      52-week high
      sector
      market regime

    Only the weighted multi-period return.
    """

    r3 = (
        close /
        close.shift(RS_3M_DAYS) -
        1
    )

    r6 = (
        close /
        close.shift(RS_6M_DAYS) -
        1
    )

    r9 = (
        close /
        close.shift(RS_9M_DAYS) -
        1
    )

    r12 = (
        close /
        close.shift(RS_12M_DAYS) -
        1
    )

    rs_score = (

        RS_WEIGHT_3M * r3 +

        RS_WEIGHT_6M * r6 +

        RS_WEIGHT_9M * r9 +

        RS_WEIGHT_12M * r12

    ) * 100

    return rs_score


# ============================================================
# BUILD SIGNAL DATA FOR ONE STOCK
# ============================================================

def compute_stock_data(close, volume):

    close = normalize_series_index(close)

    volume = normalize_series_index(volume)

    rs_score = calculate_rs_score(close)

    avg_volume = (
        volume
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    liquid = (

        (close > MIN_PRICE)

        &

        (avg_volume > MIN_AVG_VOLUME)

    )

    result = pd.DataFrame({

        "price": close,

        "rs_score": rs_score,

        "liquid": liquid

    })

    result.index = normalize_dates(
        result.index
    )

    return result


# ============================================================
# GET ROW
# ============================================================

def get_row(df, date):

    date = pd.Timestamp(date).normalize()

    if date not in df.index:
        return None

    row = df.loc[date]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    return row


# ============================================================
# BUILD DAILY TOP 10
# ============================================================

def build_top10(all_stocks, date):

    candidates = []

    for symbol, df in all_stocks.items():

        row = get_row(df, date)

        if row is None:
            continue

        if pd.isna(row["rs_score"]):
            continue

        if not bool(row["liquid"]):
            continue

        candidates.append(
            (
                symbol,
                float(row["rs_score"])
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top10 = candidates[:TOP_N]

    return top10


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(all_stocks, trading_days):

    cash = float(STARTING_CAPITAL)

    holdings = {}

    pending_target = None

    trade_log = []

    equity_curve = []

    daily_top10 = []

    previous_top10 = set()

    n_days = len(trading_days)

    for day_number, date in enumerate(
        trading_days,
        start=1
    ):

        date = pd.Timestamp(date).normalize()

        # ====================================================
        # 1. EXECUTE PREVIOUS DAY'S SIGNAL
        # ====================================================

        if pending_target is not None:

            target_top10 = pending_target

            target_set = set(target_top10)

            # ------------------------------------------------
            # SELL STOCKS NO LONGER IN TOP 10
            # ------------------------------------------------

            stocks_to_sell = [
                sym
                for sym in holdings
                if sym not in target_set
            ]

            for sym in stocks_to_sell:

                pos = holdings[sym]

                row = get_row(
                    all_stocks[sym],
                    date
                )

                if row is None:
                    continue

                exit_price = float(
                    row["price"]
                )

                qty = pos["qty"]

                gross_proceeds = (
                    qty *
                    exit_price
                )

                sell_cost = sell_side_cost(
                    gross_proceeds
                )

                net_proceeds = (
                    gross_proceeds -
                    sell_cost
                )

                cost_basis = (
                    qty *
                    pos["entry_price"]
                    +
                    pos["entry_cost"]
                )

                net_gain = (
                    net_proceeds -
                    cost_basis
                )

                tax = stcg_tax(
                    net_gain
                )

                net_cash_received = (
                    net_proceeds -
                    tax
                )

                cash += net_cash_received

                gross_return_pct = (

                    exit_price /
                    pos["entry_price"] -
                    1

                ) * 100

                net_pnl = (
                    net_gain -
                    tax
                )

                net_return_pct = (

                    net_pnl /
                    cost_basis *

                    100

                    if cost_basis > 0
                    else 0
                )

                trade_log.append({

                    "symbol":
                        sym,

                    "entry_date":
                        pos["entry_date"]
                        .strftime("%Y-%m-%d"),

                    "exit_date":
                        date
                        .strftime("%Y-%m-%d"),

                    "qty":
                        qty,

                    "entry_price":
                        round(
                            pos["entry_price"],
                            4
                        ),

                    "exit_price":
                        round(
                            exit_price,
                            4
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
                            date -
                            pos["entry_date"]
                        ).days,

                    "exit_reason":
                        "Left Top 10"

                })

                del holdings[sym]

            # ------------------------------------------------
            # PORTFOLIO VALUE AFTER SELLS
            # ------------------------------------------------

            portfolio_value = cash

            for sym, pos in holdings.items():

                row = get_row(
                    all_stocks[sym],
                    date
                )

                if row is not None:

                    portfolio_value += (
                        pos["qty"] *
                        float(row["price"])
                    )

            # ------------------------------------------------
            # BUY NEW TOP 10 STOCKS
            # ------------------------------------------------

            new_stocks = [
                sym
                for sym in target_top10
                if sym not in holdings
            ]

            if new_stocks:

                # Equal 10% target allocation
                target_value = (
                    portfolio_value /
                    TOP_N
                )

                for sym in new_stocks:

                    if sym in holdings:
                        continue

                    row = get_row(
                        all_stocks[sym],
                        date
                    )

                    if row is None:
                        continue

                    price = float(
                        row["price"]
                    )

                    if price <= 0:
                        continue

                    qty = int(
                        target_value //
                        price
                    )

                    if qty < 1:
                        continue

                    trade_value = (
                        qty *
                        price
                    )

                    buy_cost = buy_side_cost(
                        trade_value
                    )

                    total_required = (
                        trade_value +
                        buy_cost
                    )

                    if total_required > cash:
                        continue

                    cash -= total_required

                    holdings[sym] = {

                        "qty":
                            qty,

                        "entry_price":
                            price,

                        "entry_date":
                            date,

                        "entry_cost":
                            buy_cost

                    }

        # ====================================================
        # 2. TODAY'S TOP 10 SIGNAL
        # ====================================================

        top10 = build_top10(
            all_stocks,
            date
        )

        today_top10 = [
            sym
            for sym, score in top10
        ]

        today_set = set(
            today_top10
        )

        # ----------------------------------------------------
        # NEW ENTRIES
        # ----------------------------------------------------

        new_entries = (
            today_set -
            previous_top10
        )

        # ----------------------------------------------------
        # EXITS
        # ----------------------------------------------------

        exits = (
            previous_top10 -
            today_set
        )

        # ----------------------------------------------------
        # DAILY TOP 10 RECORD
        # ----------------------------------------------------

        row = {

            "date":
                date.strftime("%Y-%m-%d"),

            "Entry":
                ", ".join(
                    sorted(new_entries)
                ),

            "Exit":
                ", ".join(
                    sorted(exits)
                )

        }

        for rank in range(TOP_N):

            if rank < len(top10):

                sym, score = top10[rank]

                row[
                    f"Rank_{rank + 1}"
                ] = sym

                row[
                    f"RS_{rank + 1}"
                ] = round(
                    score,
                    4
                )

            else:

                row[
                    f"Rank_{rank + 1}"
                ] = ""

                row[
                    f"RS_{rank + 1}"
                ] = ""

        daily_top10.append(row)

        previous_top10 = today_set

        # ====================================================
        # 3. MARK PORTFOLIO TO MARKET
        # ====================================================

        total_value = cash

        for sym, pos in holdings.items():

            row = get_row(
                all_stocks[sym],
                date
            )

            if row is None:
                mark_price = (
                    pos["entry_price"]
                )
            else:
                mark_price = float(
                    row["price"]
                )

            total_value += (
                pos["qty"] *
                mark_price
            )

        equity_curve.append({

            "date":
                date.strftime("%Y-%m-%d"),

            "portfolio_value_rs":
                round(
                    total_value,
                    2
                ),

            "equity":
                round(
                    total_value /
                    STARTING_CAPITAL,
                    8
                ),

            "cash_rs":
                round(
                    cash,
                    2
                ),

            "n_holdings":
                len(holdings)

        })

        # ====================================================
        # 4. STAGE TODAY'S TOP 10 FOR TOMORROW
        # ====================================================

        pending_target = today_top10

        if day_number % 100 == 0:

            print(
                f"Processed "
                f"{day_number}/{n_days} | "
                f"{date.strftime('%Y-%m-%d')} | "
                f"Holdings: {len(holdings)} | "
                f"Equity: Rs."
                f"{total_value:,.0f}"
            )

    # ========================================================
    # EQUITY DRAWDOWN
    # ========================================================

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

            equity_df["equity"] /
            running_max -
            1

        ) * 100

        equity_df[
            "drawdown_pct"
        ] = equity_df[
            "drawdown_pct"
        ].round(3)

    trade_df = pd.DataFrame(
        trade_log
    )

    top10_df = pd.DataFrame(
        daily_top10
    )

    # ========================================================
    # TERMINAL LIQUIDATION
    # ========================================================

    final_marked_value = (

        equity_df[
            "portfolio_value_rs"
        ].iloc[-1]

        if not equity_df.empty

        else STARTING_CAPITAL
    )

    liquidation_cash = cash

    if len(trading_days) > 0:

        last_date = (
            pd.Timestamp(
                trading_days[-1]
            ).normalize()
        )

        for sym, pos in holdings.items():

            row = get_row(
                all_stocks[sym],
                last_date
            )

            if row is None:
                exit_price = (
                    pos["entry_price"]
                )
            else:
                exit_price = float(
                    row["price"]
                )

            gross_proceeds = (
                pos["qty"] *
                exit_price
            )

            sell_cost = sell_side_cost(
                gross_proceeds
            )

            net_proceeds = (
                gross_proceeds -
                sell_cost
            )

            cost_basis = (
                pos["qty"] *
                pos["entry_price"]
                +
                pos["entry_cost"]
            )

            gain = (
                net_proceeds -
                cost_basis
            )

            tax = stcg_tax(gain)

            liquidation_cash += (
                net_proceeds -
                tax
            )

    final_liquidation_value = (
        liquidation_cash
    )

    return (
        trade_df,
        equity_df,
        top10_df,
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
        equity_df["equity"]
        .cummax()
    )

    drawdown = (

        equity_df["equity"] /
        running_max -
        1

    ) * 100

    max_dd = drawdown.min()

    # --------------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------------

    if not trade_df.empty:

        n = len(trade_df)

        win_rate_net = (
            trade_df["net_return_pct"]
            .gt(0)
            .mean() *
            100
        )

        avg_net = (
            trade_df["net_return_pct"]
            .mean()
        )

        median_net = (
            trade_df["net_return_pct"]
            .median()
        )

        avg_days = (
            trade_df["days_held"]
            .mean()
        )

        best_trade = (
            trade_df["net_return_pct"]
            .max()
        )

        worst_trade = (
            trade_df["net_return_pct"]
            .min()
        )

        total_costs = (

            trade_df["buy_cost_rs"] +
            trade_df["sell_cost_rs"]

        ).sum()

        total_tax = (
            trade_df["stcg_tax_rs"]
            .sum()
        )

        winners = trade_df[
            trade_df[
                "net_return_pct"
            ] > 0
        ]

        losers = trade_df[
            trade_df[
                "net_return_pct"
            ] < 0
        ]

        gross_profit = (
            winners["net_pnl_rs"]
            .sum()
            if not winners.empty
            else 0
        )

        gross_loss = abs(
            losers["net_pnl_rs"]
            .sum()
            if not losers.empty
            else 0
        )

        profit_factor = (

            gross_profit /
            gross_loss

            if gross_loss > 0
            else 0
        )

    else:

        n = 0
        win_rate_net = 0
        avg_net = 0
        median_net = 0
        avg_days = 0
        best_trade = 0
        worst_trade = 0
        total_costs = 0
        total_tax = 0
        profit_factor = 0

    # --------------------------------------------------------
    # DAILY STATISTICS
    # --------------------------------------------------------

    daily_returns = (
        equity_df["equity"]
        .pct_change()
        .dropna()
    )

    if len(daily_returns) > 1:

        mean_daily = (
            daily_returns.mean()
        )

        std_daily = (
            daily_returns.std()
        )

        annualized_return = (

            equity_df["equity"].iloc[-1]
            ** (
                252 /
                max(
                    len(equity_df),
                    1
                )
            )

            - 1
        )

        annualized_vol = (
            std_daily *
            np.sqrt(252)
        )

        sharpe = (

            mean_daily /
            std_daily *
            np.sqrt(252)

            if std_daily > 0
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

            mean_daily /
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
        abs(max_dd / 100)

        if max_dd != 0
        else 0
    )

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
                max_dd,
                2
            ),

        "Closed Trades":
            n,

        "Win Rate Net (%)":
            round(
                win_rate_net,
                1
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

        "Best Net Trade (%)":
            round(
                best_trade,
                2
            ),

        "Worst Net Trade (%)":
            round(
                worst_trade,
                2
            ),

        "Profit Factor":
            round(
                profit_factor,
                3
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

        "Position Weight":
            "10% target per position",

        "Ranking":
            "Daily RS Score only",

        "Entry":
            "Top 10",

        "Exit":
            "Leaves Top 10",

        "Execution":
            "T+1 close",

        "Other Filters":
            "Price > Rs.20; "
            "20-day average volume > 100,000"

    }


# ============================================================
# GOOGLE SHEETS
# ============================================================

def get_or_create_worksheet(
    sh,
    title,
    rows=1000,
    cols=40
):

    try:
        return sh.worksheet(title)

    except gspread.WorksheetNotFound:
        pass

    try:

        return sh.add_worksheet(
            title=title,
            rows=rows,
            cols=cols
        )

    except gspread.exceptions.APIError as e:

        if "already exists" in str(e):

            return sh.worksheet(title)

        raise


def safe_update(
    ws,
    values,
    cell_range,
    label=""
):

    for attempt in range(
        MAX_WRITE_RETRIES
    ):

        try:

            ws.update(
                values,
                cell_range
            )

            return

        except Exception as e:

            error_text = str(e)

            is_quota = (
                "429" in error_text
                or
                "Quota exceeded"
                in error_text
            )

            if (
                not is_quota
                or
                attempt ==
                MAX_WRITE_RETRIES - 1
            ):

                raise

            wait = (
                INITIAL_RETRY_SECONDS *
                (2 ** attempt)
            )

            print(
                f"Google quota limit "
                f"for {label}. "
                f"Waiting {wait}s..."
            )

            time.sleep(wait)


# ============================================================
# COLUMN LETTER
# ============================================================

def col_letter(col):

    result = ""

    while col:

        col, remainder = divmod(
            col - 1,
            26
        )

        result = (
            chr(65 + remainder)
            +
            result
        )

    return result


# ============================================================
# SAFE DATAFRAME WRITE
# ============================================================

def write_dataframe(
    ws,
    df,
    start_row,
    start_col,
    label
):

    if df is None or df.empty:
        return start_row

    values = [

        list(df.columns)

    ] + [

        [
            "" if pd.isna(x) else x
            for x in row
        ]

        for row in df.itertuples(
            index=False,
            name=None
        )
    ]

    total = len(values)

    col1 = col_letter(
        start_col
    )

    col2 = col_letter(
        start_col +
        len(values[0]) -
        1
    )

    for offset in range(
        0,
        total,
        WRITE_CHUNK_SIZE
    ):

        chunk = values[
            offset:
            offset +
            WRITE_CHUNK_SIZE
        ]

        row1 = (
            start_row +
            offset
        )

        row2 = (
            row1 +
            len(chunk) -
            1
        )

        rng = (
            f"{col1}{row1}:"
            f"{col2}{row2}"
        )

        safe_update(
            ws,
            chunk,
            rng,
            label
        )

        print(
            f"Wrote {label}: "
            f"{min("
            f"offset + WRITE_CHUNK_SIZE,"
            f"total"
            f")}/{total}"
        )

    return start_row + total


# ============================================================
# REMOVE OLD CHARTS
# ============================================================

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

                    "deleteEmbeddedObject": {

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

    except Exception as e:

        print(
            f"Chart removal skipped: {e}"
        )


# ============================================================
# ADD EQUITY CURVE CHART
# ============================================================

def add_equity_chart(
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

    request = {

        "addChart": {

            "chart": {

                "spec": {

                    "title":
                        "Top 10 RS Equity Curve",

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
                                    "Portfolio Value (Rs)"
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
                                                    1,

                                                "endColumnIndex":
                                                    2

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
                                equity_header_row_0idx,

                            "columnIndex":
                                9

                        },

                        "widthPixels":
                            800,

                        "heightPixels":
                            450

                    }

                }

            }

        }

    }

    try:

        sh.batch_update({
            "requests":
                [request]
        })

        print(
            "Equity curve chart added."
        )

    except Exception as e:

        print(
            f"Chart could not be added: {e}"
        )


# ============================================================
# WRITE EVERYTHING
# ============================================================

def write_to_sheet(
    trade_df,
    equity_df,
    top10_df,
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
    # FALLBACK TO CSV
    # --------------------------------------------------------

    if not sheet_id or not creds_json:

        print(
            "\nGoogle credentials not found."
        )

        top10_df.to_csv(
            "daily_top10.csv",
            index=False
        )

        equity_df.to_csv(
            "equity_curve.csv",
            index=False
        )

        trade_df.to_csv(
            "trade_log.csv",
            index=False
        )

        print(
            "CSV files saved."
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

    ws = get_or_create_worksheet(
        sh,
        BACKTEST_WORKSHEET,
        rows=1000,
        cols=40
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # Calculate total required rows FIRST.
    #
    # Google Sheets default row limit is not enough for:
    #
    # daily top10
    # + equity
    # + trades
    #
    # The previous code calculated each section's size but
    # then placed them sequentially, causing:
    #
    # A22596:H24130 exceeds grid limits.
    #
    # This version resizes BEFORE writing.
    # --------------------------------------------------------

    summary_rows_count = (
        len(summary) + 1
    )

    top10_rows = (
        len(top10_df) + 1
    )

    equity_rows = (
        len(equity_df) + 1
    )

    trade_rows = (
        len(trade_df) + 1
    )

    total_required_rows = (

        1 +       # header
        2 +       # summary spacing
        summary_rows_count +
        3 +       # top10 title/headers/spacing
        top10_rows +
        3 +       # equity section spacing
        equity_rows +
        3 +       # trade section spacing
        trade_rows +
        10

    )

    required_cols = max(

        40,

        len(top10_df.columns),

        len(equity_df.columns),

        len(trade_df.columns)

    )

    # --------------------------------------------------------
    # RESIZE BEFORE CLEAR/WRITE
    # --------------------------------------------------------

    if (
        ws.row_count <
        total_required_rows
        or
        ws.col_count <
        required_cols
    ):

        ws.resize(

            rows=max(
                ws.row_count,
                total_required_rows
            ),

            cols=max(
                ws.col_count,
                required_cols
            )

        )

        print(
            f"Sheet resized to "
            f"{max(ws.row_count, total_required_rows)} "
            f"rows × "
            f"{max(ws.col_count, required_cols)} "
            f"columns."
        )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    print(
        "\nClearing old backtest..."
    )

    ws.clear()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    header = [[

        "TOP 10 DAILY RS BACKTEST | "
        f"Run: {timestamp} | "
        f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
        "RS ONLY | "
        "TOP 10 ENTRY / EXIT | "
        "EQUAL WEIGHT | "
        "T+1 CLOSE | "
        f"Window: {BACKTEST_START} "
        f"to {effective_end_str}"

    ]]

    safe_update(
        ws,
        header,
        "A1",
        "header"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = (

        [["SUMMARY", ""]]

        +

        [
            [k, v]
            for k, v in summary.items()
        ]

    )

    summary_start = 3

    summary_end = (
        summary_start +
        len(summary_rows) -
        1
    )

    safe_update(

        ws,

        summary_rows,

        f"A{summary_start}:"
        f"B{summary_end}",

        "summary"

    )

    # --------------------------------------------------------
    # DAILY TOP 10
    # --------------------------------------------------------

    top10_start = (
        summary_end +
        3
    )

    safe_update(

        ws,

        [["DAILY TOP 10 RS"]],

        f"A{top10_start}",

        "top10 title"

    )

    top10_header_row = (
        top10_start +
        1
    )

    write_dataframe(

        ws,

        top10_df,

        top10_header_row,

        1,

        "daily top 10"

    )

    top10_end = (
        top10_header_row +
        len(top10_df)
    )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity_start = (
        top10_end +
        3
    )

    safe_update(

        ws,

        [["EQUITY CURVE"]],

        f"A{equity_start}",

        "equity title"

    )

    equity_header_row = (
        equity_start +
        1
    )

    write_dataframe(

        ws,

        equity_df,

        equity_header_row,

        1,

        "equity curve"

    )

    equity_end = (
        equity_header_row +
        len(equity_df)
    )

    # --------------------------------------------------------
    # TRADE LOG
    # --------------------------------------------------------

    trade_start = (
        equity_end +
        3
    )

    safe_update(

        ws,

        [["TRADE LOG"]],

        f"A{trade_start}",

        "trade title"

    )

    trade_header_row = (
        trade_start +
        1
    )

    write_dataframe(

        ws,

        trade_df,

        trade_header_row,

        1,

        "trade log"

    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    if not equity_df.empty:

        remove_existing_charts(
            sh,
            ws.id
        )

        add_equity_chart(

            sh,

            ws.id,

            equity_header_row - 1,

            len(equity_df)

        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        "\n================================================"
    )

    print(
        "RESULTS WRITTEN SUCCESSFULLY"
    )

    print(
        f"Daily Top 10 rows: "
        f"{len(top10_df)}"
    )

    print(
        f"Equity rows: "
        f"{len(equity_df)}"
    )

    print(
        f"Closed trades: "
        f"{len(trade_df)}"
    )

    print(
        "================================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD STOCKS
    # --------------------------------------------------------

    tickers = load_tickers()

    print(
        f"\nLoaded {len(tickers)} stocks."
    )

    download_start, download_end = (
        get_download_dates()
    )

    print(
        "\n================================================"
    )

    print(
        "TOP 10 DAILY RS BACKTEST"
    )

    print(
        "================================================"
    )

    print(
        f"Download start : {download_start}"
    )

    print(
        f"Backtest start : {BACKTEST_START}"
    )

    print(
        f"Backtest end   : "
        f"{BACKTEST_END or 'LATEST'}"
    )

    print(
        f"Ranking        : DAILY RS SCORE ONLY"
    )

    print(
        f"Entry          : TOP {TOP_N}"
    )

    print(
        f"Exit           : LEAVES TOP {TOP_N}"
    )

    print(
        "Weight         : EQUAL 10% TARGET"
    )

    print(
        "Execution      : T+1 CLOSE"
    )

    print(
        f"Price filter   : > Rs.{MIN_PRICE}"
    )

    print(
        f"Liquidity      : "
        f"{VOLUME_LOOKBACK}D AVG VOL > "
        f"{MIN_AVG_VOLUME:,}"
    )

    print(
        "RS formula     : "
        "40% 3M + 20% 6M + "
        "20% 9M + 20% 12M"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # BENCHMARK
    #
    # Benchmark is downloaded only to establish the market
    # trading calendar.
    #
    # IT IS NOT USED IN THE RS RANKING.
    # --------------------------------------------------------

    benchmark = download_benchmark()

    benchmark.index = normalize_dates(
        benchmark.index
    )

    # --------------------------------------------------------
    # DOWNLOAD STOCK DATA
    # --------------------------------------------------------

    all_stocks = {}

    total_bad_points = 0

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
            f"\nDownloading "
            f"{i + 1}-"
            f"{i + len(batch)} "
            f"of {len(tickers)}..."
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
                f"Batch failed: {e}"
            )

            continue

        for symbol in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    if symbol not in (
                        data
                        .columns
                        .get_level_values(0)
                    ):

                        continue

                    sdata = data[symbol]

                if (
                    "Close"
                    not in sdata.columns
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

                stock_data = (
                    compute_stock_data(
                        close,
                        volume
                    )
                )

                clean_symbol = (
                    symbol
                    .replace(".NS", "")
                )

                all_stocks[
                    clean_symbol
                ] = stock_data

            except Exception as e:

                print(
                    f"Skipping {symbol}: "
                    f"{e}"
                )

        time.sleep(1)

    print(
        f"\nStocks with usable data: "
        f"{len(all_stocks)}"
    )

    print(
        f"Repaired price points: "
        f"{total_bad_points}"
    )

    if not all_stocks:

        raise RuntimeError(
            "No usable stock data."
        )

    # --------------------------------------------------------
    # BACKTEST END
    # --------------------------------------------------------

    latest_stock_date = max(

        df.index.max()

        for df
        in all_stocks.values()

    )

    benchmark_latest_date = (
        benchmark.index.max()
    )

    if BACKTEST_END is None:

        effective_end = min(
            latest_stock_date,
            benchmark_latest_date
        )

    else:

        effective_end = min(

            pd.Timestamp(
                BACKTEST_END
            ).normalize(),

            latest_stock_date,

            benchmark_latest_date

        )

    # --------------------------------------------------------
    # MARKET TRADING DAYS
    # --------------------------------------------------------

    trading_days = (

        benchmark.index[

            (benchmark.index >=
             pd.Timestamp(
                 BACKTEST_START
             ).normalize())

            &

            (benchmark.index <=
             effective_end)

        ]

    )

    trading_days = (

        pd.DatetimeIndex(
            trading_days
        )
        .drop_duplicates()
        .sort_values()

    )

    print(
        f"\nTrading days: "
        f"{len(trading_days)}"
    )

    print(
        f"First day: "
        f"{trading_days[0].strftime('%Y-%m-%d')}"
    )

    print(
        f"Last day: "
        f"{trading_days[-1].strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    print(
        "\nRunning bare-bones "
        "Top 10 RS backtest..."
    )

    (
        trade_df,
        equity_df,
        top10_df,
        final_marked,
        final_liq

    ) = run_backtest(

        all_stocks,

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
        "\n================================================"
    )

    print(
        "FINAL BACKTEST RESULTS"
    )

    print(
        "================================================"
    )

    for k, v in summary.items():

        print(
            f"{k}: {v}"
        )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    write_to_sheet(

        trade_df,

        equity_df,

        top10_df,

        summary,

        effective_end.strftime(
            "%Y-%m-%d"
        )

    )

    print(
        "\nBACKTEST COMPLETED."
    )


# ============================================================
# EXECUTE
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