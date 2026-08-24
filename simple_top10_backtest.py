"""
============================================================
TOP 10 RS ROTATION BACKTEST
============================================================

RULES
-----
1. Rank stocks ONLY by daily RS score.

2. RS score:
      40% = 3-month return
      20% = 6-month return
      20% = 9-month return
      20% = 12-month return

3. Eligibility:
      Price > Rs.20
      20-day average volume > 100,000

4. Portfolio:
      Hold Top 10 RS stocks.

      INITIAL ENTRY:
          Buy the Top 10 stocks.

      AFTER INITIAL ENTRY:
          Hold a stock as long as it remains rank 1-10.

          If a held stock falls to rank 11 or worse,
          sell it.

          If a held stock disappears from today's eligible
          ranking, sell it immediately.

          Buy only stocks that are NEW entrants into today's
          Top 10.

      EXISTING POSITIONS ARE NOT RESIZED.

      There is NO daily sell/rebuy.

      There is NO continuous equal-weight rebalancing.

      A stock that remains Top 10 for 100 trading days can
      remain in the portfolio for all 100 days without being
      sold/rebought.

5. Position sizing:
      Initial/replacement target:

          slot_value = current portfolio value / 10

      The new position is sized approximately to one slot.

      Existing positions are NOT resized when their market
      value changes.

6. Entry:
      Initial portfolio:
          Buy Top 10 RS.

      Replacement:
          Buy stocks newly entering Top 10.

      Do NOT buy rank 11+ merely because a Top-10 candidate
      is unaffordable.

7. Exit:
      Held stock is sold when:

          rank > 10

      OR

          stock is missing from today's eligible ranking.

      Missing-data / failed-filter exits use the last known
      traded price.

8. Execution:
      Same EOD bar as the ranking signal.
      No T+1 lag.

9. Transaction costs, STCG and starting capital:
      UNCHANGED.

10. No:
      RS Line
      Trend Template
      Blue/green dots
      Sector filter
      Regime filter
      Breadth filter

GOOGLE SHEETS OUTPUT
--------------------
Single "Backtest - RS Top10" worksheet.

Layout:

    Header
    Summary
    Trade Log
    Open Positions
    Daily Equity Curve

Charts:
    Equity Curve
    Drawdown
    Eligible Pool Size
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

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = None

# Extra history required to calculate 12-month RS.
DOWNLOAD_YEARS_BEFORE_START = 3

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

# RS lookbacks
RS_3M = 63
RS_6M = 126
RS_9M = 189
RS_12M = 252

TOP_N = 10

STARTING_CAPITAL = 1_000_000

# Maximum plausible daily price movement.
# Used only for bad-data repair.
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

STCG_EFFECTIVE_RATE = (
    STCG_RATE * (1 + STCG_CESS)
)


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

BACKTEST_WORKSHEET = "Backtest - RS Top10"


# ============================================================
# DATE FUNCTIONS
# ============================================================

def normalize_dates(index):

    idx = pd.DatetimeIndex(index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    return idx.normalize()


def normalize_series_index(series):

    s = series.copy()

    s.index = normalize_dates(
        s.index
    )

    s = s[
        ~s.index.duplicated(
            keep="last"
        )
    ]

    return s.sort_index()


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
# LOAD STOCK UNIVERSE
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
        s
        for s in symbols
        if s
    ]

    output = []

    for s in symbols:

        if s.endswith(".NS"):

            output.append(s)

        else:

            output.append(
                s + ".NS"
            )

    output = list(
        dict.fromkeys(
            output
        )
    )

    return output


# ============================================================
# CLEAN PRICE DATA
# ============================================================

def clean_price_series(close):

    close = normalize_series_index(
        close
    )

    pct_change = close.pct_change()

    bad = (
        pct_change.abs()
        >
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
            cleaned.index.get_loc(
                idx
            )
        )

        if pos > 0:

            cleaned.iloc[pos] = (
                cleaned.iloc[pos - 1]
            )

    return cleaned, n_bad


# ============================================================
# TRANSACTION COST FUNCTIONS
# ============================================================

def buy_side_cost(trade_value):

    stt = (
        STT_RATE
        *
        trade_value
    )

    stamp = (
        STAMP_DUTY_RATE
        *
        trade_value
    )

    exch = (
        EXCHANGE_CHARGE_RATE
        *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        *
        trade_value
    )

    gst = (
        GST_RATE
        *
        (
            exch
            +
            sebi
        )
    )

    return (
        stt
        +
        stamp
        +
        exch
        +
        sebi
        +
        gst
    )


def sell_side_cost(trade_value):

    stt = (
        STT_RATE
        *
        trade_value
    )

    exch = (
        EXCHANGE_CHARGE_RATE
        *
        trade_value
    )

    sebi = (
        SEBI_CHARGE_RATE
        *
        trade_value
    )

    gst = (
        GST_RATE
        *
        (
            exch
            +
            sebi
        )
    )

    return (
        stt
        +
        exch
        +
        sebi
        +
        gst
        +
        DP_CHARGE_FLAT
    )


def stcg_tax(net_gain):

    if net_gain <= 0:

        return 0.0

    return (
        net_gain
        *
        STCG_EFFECTIVE_RATE
    )


# ============================================================
# BENCHMARK
#
# Used ONLY to establish the common trading-date calendar.
# It is NOT used in the RS score.
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print()

    print(
        f"Benchmark download: "
        f"{download_start} -> "
        f"{download_end if download_end else 'LATEST'}"
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
                    f"repaired {n_bad} points"
                )

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
        "Could not download benchmark data."
    )


# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_stock_data(
    close,
    volume
):

    close = normalize_series_index(
        close
    )

    volume = normalize_series_index(
        volume
    )

    if len(close) < 300:

        return None

    avg_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (
        (close > MIN_PRICE)
        &
        (
            avg_volume
            >
            MIN_AVG_VOLUME
        )
    )

    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

    ret_3m = (
        close
        /
        close.shift(RS_3M)
        - 1
    )

    ret_6m = (
        close
        /
        close.shift(RS_6M)
        - 1
    )

    ret_9m = (
        close
        /
        close.shift(RS_9M)
        - 1
    )

    ret_12m = (
        close
        /
        close.shift(RS_12M)
        - 1
    )

    rs_score = (
        0.40 * ret_3m
        +
        0.20 * ret_6m
        +
        0.20 * ret_9m
        +
        0.20 * ret_12m
    ) * 100

    result = pd.DataFrame({

        "price":
            close,

        "avg_volume":
            avg_volume,

        "liquid":
            liquid,

        "rs_score":
            rs_score

    })

    result.index = normalize_dates(
        result.index
    )

    return result


# ============================================================
# SAFE ROW ACCESS
# ============================================================

def get_row(
    df,
    date
):

    date = (
        pd.Timestamp(
            date
        )
        .normalize()
    )

    if date not in df.index:

        return None

    row = df.loc[date]

    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[-1]

    return row


# ============================================================
# BUILD DAILY RS RANKING
#
# This is the single source of truth for:
#
#     - Top 10 membership
#     - exits
#     - new entries
#
# Only stocks with valid price, liquidity and RS score
# are included.
# ============================================================

def build_daily_ranking(
    all_stocks,
    date
):

    ranking = []

    for symbol, df in (
        all_stocks.items()
    ):

        row = get_row(
            df,
            date
        )

        if row is None:

            continue

        rs = row["rs_score"]

        if pd.isna(rs):

            continue

        if not bool(
            row["liquid"]
        ):

            continue

        price = row["price"]

        if pd.isna(price):

            continue

        ranking.append(
            (
                symbol,
                float(rs),
                float(price)
            )
        )

    # Highest RS first.
    ranking.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return ranking


# ============================================================
# BACKTEST ENGINE
#
# TOP-10 ROTATION
#
# INITIAL DAY:
#     Buy Top 10.
#
# EVERY SUBSEQUENT DAY:
#
#     1. Rank universe.
#
#     2. Sell ONLY existing holdings that are no longer
#        in Top 10.
#
#     3. Hold existing Top-10 positions unchanged.
#
#     4. Buy ONLY the new Top-10 entrants.
#
#     5. Existing positions are never resized.
#
#     6. No daily sell/rebuy.
# ============================================================

def run_backtest(
    all_stocks,
    trading_days
):

    cash = float(
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    n_days = len(
        trading_days
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This flag distinguishes the initial portfolio from
    # subsequent daily rotation.
    # --------------------------------------------------------

    initialized = False

    for day_number, date in enumerate(
        trading_days,
        start=1
    ):

        date = (
            pd.Timestamp(
                date
            )
            .normalize()
        )

        # ====================================================
        # STEP 1
        # BUILD TODAY'S RANKING
        # ====================================================

        ranking = build_daily_ranking(
            all_stocks,
            date
        )

        rank_lookup = {
            symbol: rank
            for rank, (
                symbol,
                rs,
                price
            ) in enumerate(
                ranking,
                start=1
            )
        }

        price_lookup = {
            symbol: float(price)
            for symbol, rs, price
            in ranking
        }

        eligible_pool_size = len(
            ranking
        )

        today_top10 = [
            symbol
            for symbol, rs, price
            in ranking[:TOP_N]
        ]

        today_top10_set = set(
            today_top10
        )

        # ====================================================
        # STEP 2
        # INITIAL PORTFOLIO
        #
        # On the first trading day:
        #
        #     Buy Top 10.
        #
        # No existing positions exist.
        # ====================================================

        if not initialized:

            if len(today_top10) > 0:

                # Entire starting capital is the portfolio
                # capital before initial purchases.
                #
                # Each slot targets 1/10 of portfolio value.

                slot_value = (
                    STARTING_CAPITAL
                    /
                    TOP_N
                )

                for symbol in today_top10:

                    price = price_lookup[
                        symbol
                    ]

                    if price <= 0:

                        continue

                    qty = int(
                        slot_value
                        /
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

                    if (
                        total_required
                        >
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

                        "last_price":
                            price,

                        "last_price_date":
                            date

                    }

                    trade_log.append({

                        "symbol":
                            symbol,

                        "entry_date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "exit_date":
                            "",

                        "qty":
                            qty,

                        "entry_price":
                            round(
                                price,
                                4
                            ),

                        "exit_price":
                            "",

                        "gross_return_pct":
                            "",

                        "buy_cost_rs":
                            round(
                                buy_cost,
                                2
                            ),

                        "sell_cost_rs":
                            "",

                        "stcg_tax_rs":
                            "",

                        "net_pnl_rs":
                            "",

                        "net_return_pct":
                            "",

                        "days_held":
                            "",

                        "action":
                            "ENTRY",

                        "exit_reason":
                            ""

                    })

            initialized = True

        # ====================================================
        # STEP 3
        # NORMAL DAILY ROTATION
        #
        # Existing holdings that remain Top 10 are untouched.
        #
        # Only holdings outside Top 10 are sold.
        # ====================================================

        else:

            exit_symbols = []

            for symbol in holdings:

                rank_today = rank_lookup.get(
                    symbol,
                    TOP_N + 1
                )

                if rank_today > TOP_N:

                    exit_symbols.append(
                        symbol
                    )

            # ------------------------------------------------
            # SELL ONLY EXITING HOLDINGS
            # ------------------------------------------------

            for symbol in exit_symbols:

                position = holdings.pop(
                    symbol
                )

                if symbol in price_lookup:

                    exit_price = float(
                        price_lookup[
                            symbol
                        ]
                    )

                    rank_today = (
                        rank_lookup.get(
                            symbol
                        )
                    )

                    exit_reason = (
                        f"RANK_{rank_today}_"
                        f"DROPPED_OUTSIDE_TOP10"
                    )

                else:

                    exit_price = float(
                        position.get(
                            "last_price",
                            position[
                                "entry_price"
                            ]
                        )
                    )

                    exit_reason = (
                        "MISSING_FROM_RANKING_"
                        "FORCE_EXIT"
                    )

                qty = int(
                    position["qty"]
                )

                gross_proceeds = (
                    qty
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

                # ------------------------------------------------
                # Tax basis includes original buy cost.
                # ------------------------------------------------

                cost_basis = (
                    qty
                    *
                    position[
                        "entry_price"
                    ]
                    +
                    position[
                        "entry_cost"
                    ]
                )

                net_gain = (
                    net_proceeds
                    -
                    cost_basis
                )

                tax = stcg_tax(
                    net_gain
                )

                cash += (
                    net_proceeds
                    -
                    tax
                )

                gross_return_pct = (
                    (
                        exit_price
                        /
                        position[
                            "entry_price"
                        ]
                    )
                    - 1
                ) * 100

                net_pnl = (
                    net_gain
                    -
                    tax
                )

                net_return_pct = (
                    net_pnl
                    /
                    cost_basis
                    * 100
                ) if cost_basis > 0 else 0

                days_held = (
                    date
                    -
                    position[
                        "entry_date"
                    ]
                ).days

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
                        qty,

                    "entry_price":
                        round(
                            position[
                                "entry_price"
                            ],
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
                            net_return_pct,
                            2
                        ),

                    "days_held":
                        days_held,

                    "action":
                        "EXIT",

                    "exit_reason":
                        exit_reason

                })

            # =================================================
            # STEP 4
            # MARK EXISTING HOLDINGS
            #
            # This value is used to calculate the size of a
            # replacement slot.
            # =================================================

            portfolio_value_before_entries = (
                cash
            )

            for symbol, position in (
                holdings.items()
            ):

                if symbol in price_lookup:

                    mark_price = float(
                        price_lookup[
                            symbol
                        ]
                    )

                    position[
                        "last_price"
                    ] = mark_price

                    position[
                        "last_price_date"
                    ] = date

                else:

                    mark_price = float(
                        position.get(
                            "last_price",
                            position[
                                "entry_price"
                            ]
                        )
                    )

                portfolio_value_before_entries += (
                    position["qty"]
                    *
                    mark_price
                )

            # =================================================
            # STEP 5
            # NEW TOP-10 ENTRANTS
            #
            # A new entry is:
            #
            #     today's Top 10
            #     AND
            #     not already held.
            #
            # Existing Top-10 stocks are ignored.
            # =================================================

            new_entries = []

            for symbol in today_top10:

                if symbol not in holdings:

                    new_entries.append(
                        (
                            symbol,
                            rank_lookup[
                                symbol
                            ],
                            price_lookup[
                                symbol
                            ]
                        )
                    )

            # =================================================
            # STEP 6
            # BUY NEW ENTRANTS
            #
            # Target = one portfolio slot.
            #
            # Existing positions remain untouched.
            # =================================================

            if new_entries:

                slot_value = (
                    portfolio_value_before_entries
                    /
                    TOP_N
                )

                for (
                    symbol,
                    rank_today,
                    price
                ) in new_entries:

                    if price <= 0:

                        continue

                    qty = int(
                        slot_value
                        /
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

                    if (
                        total_required
                        >
                        cash
                    ):

                        # No borrowing.
                        #
                        # Do NOT fill using a rank-11+
                        # stock. If this entrant cannot be
                        # funded, the slot remains empty.
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

                        "last_price":
                            price,

                        "last_price_date":
                            date

                    }

                    trade_log.append({

                        "symbol":
                            symbol,

                        "entry_date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "exit_date":
                            "",

                        "qty":
                            qty,

                        "entry_price":
                            round(
                                price,
                                4
                            ),

                        "exit_price":
                            "",

                        "gross_return_pct":
                            "",

                        "buy_cost_rs":
                            round(
                                buy_cost,
                                2
                            ),

                        "sell_cost_rs":
                            "",

                        "stcg_tax_rs":
                            "",

                        "net_pnl_rs":
                            "",

                        "net_return_pct":
                            "",

                        "days_held":
                            "",

                        "action":
                            "ENTRY",

                        "exit_reason":
                            ""

                    })

        # ====================================================
        # STEP 7
        # FINAL DAILY MARK-TO-MARKET
        # ====================================================

        total_value = float(
            cash
        )

        for symbol, position in (
            holdings.items()
        ):

            if symbol in price_lookup:

                mark_price = float(
                    price_lookup[
                        symbol
                    ]
                )

                position[
                    "last_price"
                ] = mark_price

                position[
                    "last_price_date"
                ] = date

            else:

                mark_price = float(
                    position.get(
                        "last_price",
                        position[
                            "entry_price"
                        ]
                    )
                )

            total_value += (
                position["qty"]
                *
                mark_price
            )

        # ====================================================
        # STEP 8
        # DAILY EQUITY RECORD
        # ====================================================

        equity_curve.append({

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "portfolio_value_rs":
                round(
                    total_value,
                    2
                ),

            "cash_rs":
                round(
                    cash,
                    2
                ),

            "invested_value_rs":
                round(
                    total_value - cash,
                    2
                ),

            "equity_multiple":
                round(
                    total_value
                    /
                    STARTING_CAPITAL,
                    8
                ),

            "n_holdings":
                len(holdings),

            "eligible_pool_size":
                eligible_pool_size,

            "top10_target_size":
                min(
                    TOP_N,
                    eligible_pool_size
                )

        })

        # ====================================================
        # PROGRESS
        # ====================================================

        if day_number % 100 == 0:

            print(
                f"Processed "
                f"{day_number}/{n_days} | "
                f"{date.strftime('%Y-%m-%d')} | "
                f"EligiblePool="
                f"{eligible_pool_size} | "
                f"Holdings="
                f"{len(holdings)} | "
                f"Equity="
                f"Rs.{total_value:,.0f}"
            )

    # ========================================================
    # EQUITY DRAWDOWN
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    if not equity_df.empty:

        running_max = (
            equity_df[
                "equity_multiple"
            ].cummax()
        )

        equity_df[
            "drawdown_pct"
        ] = (
            equity_df[
                "equity_multiple"
            ]
            /
            running_max
            - 1
        ) * 100

        equity_df[
            "drawdown_pct"
        ] = equity_df[
            "drawdown_pct"
        ].round(3)

    # ========================================================
    # TRADE DATAFRAME
    # ========================================================

    trade_df = pd.DataFrame(
        trade_log
    )

    # ========================================================
    # TERMINAL MARKED VALUE
    # ========================================================

    if equity_df.empty:

        final_marked_value = (
            STARTING_CAPITAL
        )

    else:

        final_marked_value = float(
            equity_df[
                "portfolio_value_rs"
            ].iloc[-1]
        )

    # ========================================================
    # TERMINAL LIQUIDATION
    #
    # This does NOT modify the daily equity curve.
    #
    # It simply calculates what the final portfolio would
    # be worth if every open position were sold on the final
    # trading day, including sell costs and STCG.
    # ========================================================

    liquidation_cash = float(
        cash
    )

    open_positions = []

    if len(trading_days) > 0:

        last_date = (
            pd.Timestamp(
                trading_days[-1]
            )
            .normalize()
        )

        # Build the final-day price lookup.
        final_ranking = (
            build_daily_ranking(
                all_stocks,
                last_date
            )
        )

        final_price_lookup = {
            symbol: float(price)
            for symbol, rs, price
            in final_ranking
        }

        for symbol, position in (
            holdings.items()
        ):

            if symbol in final_price_lookup:

                exit_price = (
                    final_price_lookup[
                        symbol
                    ]
                )

            else:

                exit_price = float(
                    position.get(
                        "last_price",
                        position[
                            "entry_price"
                        ]
                    )
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

            net_gain = (
                net_proceeds
                -
                cost_basis
            )

            tax = stcg_tax(
                net_gain
            )

            liquidation_cash += (
                net_proceeds
                -
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
                        4
                    ),

                "last_price":
                    round(
                        exit_price,
                        4
                    ),

                "gross_return_pct":
                    round(
                        (
                            exit_price
                            /
                            position[
                                "entry_price"
                            ]
                            - 1
                        )
                        *
                        100,
                        2
                    )

            })

    final_liquidation_value = (
        liquidation_cash
    )

    open_df = pd.DataFrame(
        open_positions
    )

    return (
        equity_df,
        trade_df,
        open_df,
        final_marked_value,
        final_liquidation_value
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    equity_df,
    trade_df,
    final_marked_value,
    final_liquidation_value
):

    if equity_df.empty:

        return {}

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    marked_return = (
        final_marked_value
        /
        STARTING_CAPITAL
        - 1
    ) * 100

    liquidation_return = (
        final_liquidation_value
        /
        STARTING_CAPITAL
        - 1
    ) * 100

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    max_dd = float(
        equity_df[
            "drawdown_pct"
        ].min()
    )

    # --------------------------------------------------------
    # DAILY RETURN
    # --------------------------------------------------------

    daily_returns = (
        equity_df[
            "equity_multiple"
        ]
        .pct_change()
        .dropna()
    )

    if len(
        daily_returns
    ) > 1:

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
            equity_df[
                "equity_multiple"
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
            - 1
        )

        annualized_vol = (
            daily_std
            *
            np.sqrt(252)
        )

        sharpe = (
            daily_mean
            /
            daily_std
            *
            np.sqrt(252)
        ) if daily_std > 0 else 0

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
            daily_mean
            /
            downside_std
            *
            np.sqrt(252)
        ) if downside_std > 0 else 0

    else:

        annualized_return = 0

        annualized_vol = 0

        sharpe = 0

        sortino = 0

    calmar = (
        annualized_return
        /
        abs(
            max_dd / 100
        )
    ) if max_dd != 0 else 0

    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    if not trade_df.empty:

        entries = trade_df[
            trade_df[
                "action"
            ] == "ENTRY"
        ]

        exits = trade_df[
            trade_df[
                "action"
            ] == "EXIT"
        ]

        total_costs = 0

        total_tax = 0

        if not exits.empty:

            total_costs += (
                exits[
                    "sell_cost_rs"
                ]
                .fillna(0)
                .sum()
            )

            total_tax += (
                exits[
                    "stcg_tax_rs"
                ]
                .fillna(0)
                .sum()
            )

        if not entries.empty:

            total_costs += (
                entries[
                    "buy_cost_rs"
                ]
                .fillna(0)
                .sum()
            )

        if not exits.empty:

            net_returns = (
                exits[
                    "net_return_pct"
                ]
                .astype(float)
            )

            win_rate = (
                (
                    net_returns > 0
                ).mean()
                *
                100
            )

            avg_trade = (
                net_returns.mean()
            )

            median_trade = (
                net_returns.median()
            )

            winners = exits[
                net_returns > 0
            ]

            losers = exits[
                net_returns < 0
            ]

            avg_winner = (
                winners[
                    "net_return_pct"
                ]
                .astype(float)
                .mean()
                if not winners.empty
                else 0
            )

            avg_loser = (
                losers[
                    "net_return_pct"
                ]
                .astype(float)
                .mean()
                if not losers.empty
                else 0
            )

            gross_profit = (
                winners[
                    "net_pnl_rs"
                ]
                .astype(float)
                .sum()
                if not winners.empty
                else 0
            )

            gross_loss = abs(
                losers[
                    "net_pnl_rs"
                ]
                .astype(float)
                .sum()
            ) if not losers.empty else 0

            profit_factor = (
                gross_profit
                /
                gross_loss
            ) if gross_loss > 0 else 0

            avg_days = (
                exits[
                    "days_held"
                ]
                .astype(float)
                .mean()
            )

        else:

            win_rate = 0

            avg_trade = 0

            median_trade = 0

            avg_winner = 0

            avg_loser = 0

            profit_factor = 0

            avg_days = 0

    else:

        entries = pd.DataFrame()

        exits = pd.DataFrame()

        total_costs = 0

        total_tax = 0

        win_rate = 0

        avg_trade = 0

        median_trade = 0

        avg_winner = 0

        avg_loser = 0

        profit_factor = 0

        avg_days = 0

    return {

        "Backtest Start":
            BACKTEST_START,

        "Backtest End":
            equity_df[
                "date"
            ].iloc[-1],

        "Starting Capital (Rs)":
            STARTING_CAPITAL,

        "Final Marked Value (Rs)":
            round(
                final_marked_value,
                0
            ),

        "Final Liquidation Value (Rs)":
            round(
                final_liquidation_value,
                0
            ),

        "Net Return - Marked (%)":
            round(
                marked_return,
                2
            ),

        "Net Return - Liquidation (%)":
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

        "Maximum Drawdown (%)":
            round(
                max_dd,
                2
            ),

        "Closed Trades":
            len(exits),

        "Entries":
            len(entries),

        "Win Rate Net (%)":
            round(
                win_rate,
                2
            ),

        "Average Net Trade (%)":
            round(
                avg_trade,
                2
            ),

        "Median Net Trade (%)":
            round(
                median_trade,
                2
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

        "Average Days Held":
            round(
                avg_days,
                1
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

        "RS Formula":
            "40% 3M + 20% 6M + 20% 9M + 20% 12M",

        "Portfolio":
            "Top 10 RS rotation",

        "Weight":
            "Equal slot size at entry; existing holdings NOT rebalanced",

        "Entry":
            "Initial Top 10; subsequently only new Top-10 entrants",

        "Exit":
            "Only when rank becomes 11+ or stock disappears from ranking",

        "Execution":
            "Same-day close (T+0)",

        "Rebalance Frequency":
            "Daily rank check; turnover only when Top-10 membership changes",

        "Price Filter":
            "> Rs.20",

        "Liquidity Filter":
            "20D average volume > 100,000",

        "Other Filters":
            "NONE"

    }


# ============================================================
# RACE-SAFE GET-OR-CREATE WORKSHEET
# ============================================================

def get_or_create_worksheet(
    sh,
    title,
    rows=1000,
    cols=16
):

    try:

        return sh.worksheet(
            title
        )

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

            return sh.worksheet(
                title
            )

        raise


# ============================================================
# GOOGLE SHEETS CHUNK WRITER
# ============================================================

def write_in_chunks(
    ws,
    all_rows,
    start_row,
    chunk_size,
    label,
    max_retries=6,
    initial_retry_seconds=5
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
            start_row
            +
            i
        )

        for attempt in range(
            max_retries
        ):

            try:

                ws.update(
                    chunk,
                    f"A{row_start}"
                )

                break

            except Exception as e:

                error_text = str(e)

                is_quota = (
                    "429"
                    in
                    error_text
                    or
                    "Quota exceeded"
                    in
                    error_text
                )

                if (
                    not is_quota
                    or
                    attempt
                    ==
                    max_retries - 1
                ):

                    print(
                        f"Write failed for "
                        f"{label} rows "
                        f"{i}-"
                        f"{i + len(chunk)}: "
                        f"{e}"
                    )

                    raise

                wait = (
                    initial_retry_seconds
                    *
                    (
                        2 ** attempt
                    )
                )

                print(
                    f"Google quota for "
                    f"{label}. "
                    f"Waiting {wait}s..."
                )

                time.sleep(
                    wait
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

            print(
                f"Removed "
                f"{len(requests)} "
                f"existing chart(s)."
            )

    except Exception as e:

        print(
            "Could not check/remove "
            "existing charts "
            f"(non-fatal): {e}"
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
        equity_header_row_0idx
        +
        1
        +
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
            8,
            "Drawdown %",
            equity_header_row_0idx + 22
        ),

        make_chart(
            "Eligible Pool Size vs Top-10 Target",
            6,
            "Stock Count",
            equity_header_row_0idx + 44
        )

    ]

    try:

        sh.batch_update({
            "requests":
                requests
        })

        print(
            "Equity, drawdown, "
            "and pool-size charts added."
        )

    except Exception as e:

        print(
            "Could not add charts "
            f"(non-fatal): {e}"
        )


# ============================================================
# WRITE RESULTS TO GOOGLE SHEETS
# ============================================================

def write_to_sheet(
    trade_df,
    equity_df,
    open_df,
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

    if (
        not sheet_id
        or
        not creds_json
    ):

        print(
            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS -- "
            "saving to CSV instead."
        )

        trade_df.to_csv(
            "RS_Trade_Log.csv",
            index=False
        )

        equity_df.to_csv(
            "RS_Equity_Curve.csv",
            index=False
        )

        if not open_df.empty:

            open_df.to_csv(
                "RS_Open_Positions.csv",
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
    # SHEET SIZE
    # --------------------------------------------------------

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

    n_cols_needed = 16

    ws = get_or_create_worksheet(
        sh,
        BACKTEST_WORKSHEET,
        rows=n_rows_needed,
        cols=n_cols_needed
    )

    if (
        ws.row_count
        <
        n_rows_needed
        or
        ws.col_count
        <
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

    # --------------------------------------------------------
    # CLEAR OLD RESULTS
    # --------------------------------------------------------

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

            "TOP 10 RS ROTATION BACKTEST | "
            f"run {timestamp} | "
            "NET of costs+STCG | "
            f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
            "Initial: Top 10 RS | "
            "Rotation: sell only rank 11+ / missing | "
            "Buy: new Top-10 entrants | "
            "Existing positions NOT resized | "
            "Same EOD bar | "
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
        [["Trade Log"]],
        f"A{trade_start_row}"
    )

    trade_header_row = (
        trade_start_row
        +
        1
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
            "Backtest End "
            "(mark-to-market)"
        ]],

        f"A{open_start_row}"

    )

    open_header_row = (
        open_start_row
        +
        1
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
    # EQUITY CURVE
    # --------------------------------------------------------

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
        equity_start_row
        +
        1
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
        f"\nBacktest results "
        f"written to "
        f"'{BACKTEST_WORKSHEET}' tab: "
        f"{len(trade_df)} trades, "
        f"{len(equity_df)} "
        f"trading days, "
        f"{len(open_df)} "
        f"open positions."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TOP 10 RS ROTATION BACKTEST"
    )

    print(
        "Sell only stocks leaving Top 10; "
        "hold stocks remaining Top 10"
    )

    print("=" * 70)

    print(
        f"Backtest start : "
        f"{BACKTEST_START}"
    )

    print(
        f"Backtest end   : "
        f"{BACKTEST_END if BACKTEST_END else 'LATEST'}"
    )

    print(
        "Ranking        : "
        "DAILY RS SCORE ONLY"
    )

    print(
        "Portfolio      : "
        "TOP 10 RS"
    )

    print(
        "Initial        : "
        "BUY TOP 10"
    )

    print(
        "Rotation       : "
        "SELL ONLY RANK 11+ / MISSING"
    )

    print(
        "New entries    : "
        "BUY ONLY NEW TOP-10 ENTRANTS"
    )

    print(
        "Existing names : "
        "HOLD / NO RESIZING"
    )

    print(
        "Weight         : "
        "ONE SLOT AT ENTRY (~PORTFOLIO/10)"
    )

    print(
        "Execution      : "
        "SAME EOD BAR (T+0)"
    )

    print(
        f"Price filter   : "
        f"> Rs.{MIN_PRICE}"
    )

    print(
        f"Liquidity      : "
        f"{VOLUME_LOOKBACK}D average volume "
        f"> {MIN_AVG_VOLUME:,}"
    )

    print(
        "Other filters  : NONE"
    )

    print("=" * 70)

    # ========================================================
    # LOAD STOCKS
    # ========================================================

    tickers = load_tickers()

    print(
        f"\nLoaded "
        f"{len(tickers)} tickers."
    )

    download_start, download_end = (
        get_download_dates()
    )

    print(
        f"Download start: "
        f"{download_start}"
    )

    print(
        f"Download end: "
        f"{download_end if download_end else 'LATEST'}"
    )

    # ========================================================
    # DOWNLOAD BENCHMARK
    # ========================================================

    bench_close = (
        download_benchmark()
    )

    bench_close.index = (
        normalize_dates(
            bench_close.index
        )
    )

    # ========================================================
    # DOWNLOAD STOCK DATA
    # ========================================================

    all_stocks = {}

    total_bad_points = 0

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

        print()

        print(
            f"Downloading "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(tickers)}"
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

                    if not isinstance(
                        data.columns,
                        pd.MultiIndex
                    ):

                        continue

                    level0 = (
                        data
                        .columns
                        .get_level_values(0)
                    )

                    if symbol not in level0:

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

                if close.empty:

                    continue

                volume = (
                    sdata["Volume"]
                    .reindex(
                        close.index
                    )
                    .fillna(0)
                )

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

                if stock_data is None:

                    continue

                clean_symbol = (
                    symbol
                    .replace(
                        ".NS",
                        ""
                    )
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

    print()

    print(
        f"Stocks with usable data: "
        f"{len(all_stocks)}"
    )

    print(
        f"Repaired data points: "
        f"{total_bad_points}"
    )

    if not all_stocks:

        raise RuntimeError(
            "No usable stock data."
        )

    # ========================================================
    # EFFECTIVE BACKTEST END
    # ========================================================

    latest_stock_date = max(
        df.index.max()
        for df
        in all_stocks.values()
    )

    latest_benchmark_date = (
        bench_close.index.max()
    )

    if BACKTEST_END is None:

        effective_end = min(

            pd.Timestamp(
                latest_stock_date
            ).normalize(),

            pd.Timestamp(
                latest_benchmark_date
            ).normalize()

        )

    else:

        effective_end = min(

            pd.Timestamp(
                BACKTEST_END
            ).normalize(),

            pd.Timestamp(
                latest_stock_date
            ).normalize(),

            pd.Timestamp(
                latest_benchmark_date
            ).normalize()

        )

    print()

    print(
        f"Latest stock date: "
        f"{pd.Timestamp(latest_stock_date).strftime('%Y-%m-%d')}"
    )

    print(
        f"Latest benchmark date: "
        f"{pd.Timestamp(latest_benchmark_date).strftime('%Y-%m-%d')}"
    )

    print(
        f"Effective end: "
        f"{effective_end.strftime('%Y-%m-%d')}"
    )

    # ========================================================
    # TRADING CALENDAR
    # ========================================================

    trading_days = bench_close.index[

        (
            bench_close.index
            >=
            pd.Timestamp(
                BACKTEST_START
            ).normalize()
        )

        &

        (
            bench_close.index
            <=
            effective_end
        )

    ]

    trading_days = (
        pd.DatetimeIndex(
            trading_days
        )
        .drop_duplicates()
        .sort_values()
    )

    print()

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    if len(
        trading_days
    ) == 0:

        raise RuntimeError(
            "No trading days found."
        )

    print(
        f"First day: "
        f"{trading_days[0].strftime('%Y-%m-%d')}"
    )

    print(
        f"Last day: "
        f"{trading_days[-1].strftime('%Y-%m-%d')}"
    )

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    print()

    print(
        "Running backtest..."
    )

    (
        equity_df,
        trade_df,
        open_df,
        final_marked,
        final_liq

    ) = run_backtest(
        all_stocks,
        trading_days
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = summarize(
        equity_df,
        trade_df,
        final_marked,
        final_liq
    )

    print()

    print("=" * 70)

    print(
        "BACKTEST RESULTS"
    )

    print("=" * 70)

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print("=" * 70)

    # ========================================================
    # GOOGLE SHEETS
    # ========================================================

    write_to_sheet(
        trade_df,
        equity_df,
        open_df,
        summary,
        effective_end.strftime(
            "%Y-%m-%d"
        )
    )

    # ========================================================
    # LOCAL CSV BACKUP
    # ========================================================

    equity_df.to_csv(
        "RS_Equity_Curve.csv",
        index=False
    )

    trade_df.to_csv(
        "RS_Trade_Log.csv",
        index=False
    )

    if not open_df.empty:

        open_df.to_csv(
            "RS_Open_Positions.csv",
            index=False
        )

    print()

    print(
        "CSV files also saved."
    )

    print()

    print(
        "BACKTEST COMPLETED SUCCESSFULLY."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print()
        print("=" * 70)

        print(
            "BACKTEST FAILED"
        )

        print("=" * 70)

        print(
            f"{type(e).__name__}: {e}"
        )

        raise