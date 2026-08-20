"""
RS SCREENER BACKTEST
STRICT DAILY TOP-10 RS REBALANCE
SAME-DAY CLOSE EXECUTION

RULES
-----
Universe        : stocks.csv
Price filter    : Price > Rs.20
Liquidity       : 20-day avg volume > 100,000
Price TT        : 7/7 required
RS Line TT      : 7/7 required
Ranking         : raw RS Score, descending
Portfolio       : DAILY TOP 10
Weight          : Equal weight
Rebalance       : EVERY TRADING DAY AT THAT DAY'S CLOSE
Execution       : SAME DAY CLOSE
Exit            : Any stock not in today's Top 10 is sold
Costs           : Buy + sell transaction costs
STCG            : 20.8% effective on positive realized gains
Equity          : Daily mark-to-market
Terminal value  : Marked AND liquidation
Charts          : Equity Curve + Drawdown

IMPORTANT
---------
This is a SAME-CLOSE model.

The ranking is calculated using the closing price of the same
day on which the trade is executed.

Therefore this model intentionally contains LOOK-AHEAD BIAS
relative to a realistically executable strategy.

It answers:

"If I could know the completed day's close/ranking and execute
at exactly that same close, what would the portfolio have done?"

It does NOT represent an executable next-day strategy.

BACKTEST_END = None
means automatically use the latest available Yahoo Finance data.
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

# None = latest available Yahoo Finance data
BACKTEST_END = None

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000

VOLUME_LOOKBACK = 20
LOOKBACK_DAYS = 250

MAX_PLAUSIBLE_DAILY_MOVE = 0.30

# ============================================================
# PORTFOLIO
# ============================================================

TOP_N = 10

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

STCG_EFFECTIVE_RATE = (
    STCG_RATE *
    (1 + STCG_CESS)
)


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

    backtest_start = pd.Timestamp(
        BACKTEST_START
    )

    download_start = (
        backtest_start -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    )

    if BACKTEST_END is None:

        return (
            download_start.strftime("%Y-%m-%d"),
            None
        )

    backtest_end = pd.Timestamp(
        BACKTEST_END
    )

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
                "Latest benchmark date: "
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

    return (
        met == 7
    )


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
        volume.reindex(
            aligned.index
        )
    )


    # ========================================================
    # RS RATIO
    # ========================================================

    rs_ratio = (

        aligned["s"] /
        aligned["b"]

    )


    # ========================================================
    # RETURN FUNCTION
    # ========================================================

    def pct_return(
        series,
        days
    ):

        return (

            series /
            series.shift(days) -
            1

        )


    # ========================================================
    # RAW RS SCORE
    # ========================================================

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


    # ========================================================
    # BLUE DOT
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

        rs_ratio >
        previous_rs_high

    )


    # ========================================================
    # PRICE NEW HIGH
    # ========================================================

    previous_price_high = (

        aligned["s"]
        .shift(1)
        .rolling(
            LOOKBACK_DAYS
        )
        .max()

    )

    price_at_new_high = (

        aligned["s"] >
        previous_price_high

    )


    # ========================================================
    # GREEN DOT
    # ========================================================

    green_dot = (

        blue_dot &
        (~price_at_new_high)

    )


    # ========================================================
    # TREND TEMPLATE
    # ========================================================

    tt_pass = (
        trend_template_series(
            aligned["s"]
        )
    )


    # ========================================================
    # RS LINE TREND TEMPLATE
    # ========================================================

    rs_tt_pass = (
        trend_template_series(
            rs_ratio
        )
    )


    # ========================================================
    # LIQUIDITY
    # ========================================================

    rolling_avg_volume = (

        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()

    )

    liquid = (

        (aligned["s"] >
         MIN_PRICE)

        &

        (
            rolling_avg_volume >
            MIN_AVG_VOLUME
        )

    )


    # ========================================================
    # OUTPUT
    # ========================================================

    out = pd.DataFrame({

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


    return out


# ============================================================
# CALCULATE DAILY ELIGIBLE POOL
# ============================================================

def get_daily_pool(
    all_signals,
    date
):

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


        pool.append({

            "symbol":
            sym,

            "rs_score":
            float(
                row["rs_score"]
            ),

            "price":
            float(
                row["price"]
            ),

            "blue_dot":
            bool(
                row["blue_dot"]
            ),

            "green_dot":
            bool(
                row["green_dot"]
            )

        })


    # ========================================================
    # RANK BY RAW RS SCORE
    # ========================================================

    pool.sort(

        key=lambda x:
        x["rs_score"],

        reverse=True

    )


    for rank, item in enumerate(
        pool,
        start=1
    ):

        item["rank"] = rank


    return pool


# ============================================================
# SELL POSITION
# ============================================================

def sell_position(
    sym,
    pos,
    exit_price,
    date,
    cash,
    trade_log,
    reason
):

    qty = pos["qty"]


    gross_proceeds = (
        qty *
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

        qty *
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
        )

        * 100

    )


    net_return_pct = (

        (
            net_gain -
            tax
        )

        /

        cost_basis

        * 100

    ) if cost_basis > 0 else 0


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
        qty,

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
        reason,

        "exit_rank":
        pos.get(
            "last_rank",
            ""
        ),

    })


    return cash


# ============================================================
# BACKTEST ENGINE
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

    daily_top10_log = []


    # ========================================================
    # EACH TRADING DAY
    # ========================================================

    for date in trading_days:

        print(
            f"Processing "
            f"{date.strftime('%Y-%m-%d')}"
        )


        # ====================================================
        # GET TODAY'S TOP 10
        # ====================================================

        pool = get_daily_pool(
            all_signals,
            date
        )


        today_top10 = pool[
            :TOP_N
        ]


        today_top10_symbols = {

            x["symbol"]

            for x in today_top10

        }


        rank_lookup = {

            x["symbol"]:
            x["rank"]

            for x in pool

        }


        # ====================================================
        # SAVE DAILY TOP 10
        # ====================================================

        top10_record = {

            "date":
            date.strftime(
                "%Y-%m-%d"
            )

        }


        for i in range(
            TOP_N
        ):

            if i < len(
                today_top10
            ):

                item = (
                    today_top10[i]
                )

                top10_record[
                    f"Rank_{i+1}"
                ] = (
                    item["symbol"]
                )

                top10_record[
                    f"RS_{i+1}"
                ] = round(
                    item["rs_score"],
                    2
                )

            else:

                top10_record[
                    f"Rank_{i+1}"
                ] = ""

                top10_record[
                    f"RS_{i+1}"
                ] = ""


        daily_top10_log.append(
            top10_record
        )


        # ====================================================
        # UPDATE EXISTING HOLDING RANKS
        # ====================================================

        for sym in holdings:

            holdings[sym][
                "last_rank"
            ] = rank_lookup.get(
                sym,
                ""
            )


        # ====================================================
        # 1. SELL EVERYTHING NOT IN TODAY'S TOP 10
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            if sym in (
                today_top10_symbols
            ):

                continue


            pos = holdings.pop(
                sym
            )


            exit_price = float(
                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]
            )


            rank = rank_lookup.get(
                sym
            )


            if rank is None:

                reason = (
                    "Left eligible universe"
                )

            else:

                reason = (
                    f"Rank dropped to "
                    f"{rank}"
                )


            cash = sell_position(

                sym,

                pos,

                exit_price,

                date,

                cash,

                trade_log,

                reason

            )


        # ====================================================
        # 2. CALCULATE PORTFOLIO VALUE AT TODAY'S CLOSE
        #    BEFORE REBALANCING
        # ====================================================

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            price = float(

                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]

            )

            portfolio_value += (

                pos["qty"] *
                price

            )


        # ====================================================
        # 3. TARGET CAPITAL PER STOCK
        # ====================================================

        target_value = (

            portfolio_value /
            TOP_N

        )


        # ====================================================
        # 4. REDUCE EXISTING POSITIONS
        #    THAT ARE ABOVE TARGET WEIGHT
        # ====================================================

        for sym in list(
            holdings.keys()
        ):

            pos = holdings[sym]


            current_price = float(

                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]

            )


            current_value = (

                pos["qty"] *
                current_price

            )


            excess_value = (

                current_value -
                target_value

            )


            if excess_value <= 0:

                continue


            qty_to_sell = int(

                excess_value //
                current_price

            )


            if qty_to_sell <= 0:

                continue


            # Never sell more than held
            qty_to_sell = min(

                qty_to_sell,
                pos["qty"]

            )


            if qty_to_sell <= 0:

                continue


            sell_value = (

                qty_to_sell *
                current_price

            )


            s_cost = (
                sell_side_cost(
                    sell_value
                )
            )


            # ------------------------------------------------
            # Partial sale
            #
            # For tax accounting, allocate original entry
            # cost proportionally to shares sold.
            # ------------------------------------------------

            fraction_sold = (

                qty_to_sell /
                pos["qty"]

            )


            allocated_entry_cost = (

                pos["entry_cost"] *
                fraction_sold

            )


            cost_basis = (

                qty_to_sell *
                pos["entry_price"]

                +

                allocated_entry_cost

            )


            net_proceeds = (

                sell_value -
                s_cost

            )


            net_gain = (

                net_proceeds -
                cost_basis

            )


            tax = stcg_tax(
                net_gain
            )


            cash += (

                net_proceeds -
                tax

            )


            net_pnl = (
                net_gain -
                tax
            )


            net_return = (

                net_pnl /
                cost_basis *
                100

            ) if cost_basis > 0 else 0


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
                qty_to_sell,

                "entry_price":
                round(
                    pos["entry_price"],
                    2
                ),

                "exit_price":
                round(
                    current_price,
                    2
                ),

                "gross_return_pct":
                round(
                    (
                        current_price /
                        pos["entry_price"] -
                        1
                    ) * 100,
                    2
                ),

                "buy_cost_rs":
                round(
                    allocated_entry_cost,
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
                    pos["entry_date"]
                ).days,

                "exit_reason":
                "Daily rebalance - reduce to target",

                "exit_rank":
                rank_lookup.get(
                    sym,
                    ""
                ),

            })


            pos["qty"] -= (
                qty_to_sell
            )


            pos["entry_cost"] -= (
                allocated_entry_cost
            )


            if pos["qty"] <= 0:

                del holdings[sym]


        # ====================================================
        # 5. BUY NEW TOP-10 STOCKS
        # ====================================================

        # Recalculate available portfolio value after sells.
        portfolio_value = cash

        for sym, pos in (
            holdings.items()
        ):

            current_price = float(

                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]

            )

            portfolio_value += (

                pos["qty"] *
                current_price

            )


        target_value = (

            portfolio_value /
            TOP_N

        )


        # ====================================================
        # BUY / TOP-UP ALL TOP-10 POSITIONS
        # ====================================================

        for item in today_top10:

            sym = item["symbol"]

            price = item["price"]


            if sym in holdings:

                current_qty = (
                    holdings[sym]["qty"]
                )

                current_value = (

                    current_qty *
                    price

                )

            else:

                current_qty = 0

                current_value = 0


            required_value = (

                target_value -
                current_value

            )


            if required_value <= 0:

                continue


            qty_to_buy = int(

                required_value //
                price

            )


            if qty_to_buy <= 0:

                continue


            trade_value = (

                qty_to_buy *
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


            if total_cost > cash:

                # Reduce quantity until affordable
                while (

                    qty_to_buy > 0
                    and
                    total_cost > cash

                ):

                    qty_to_buy -= 1

                    trade_value = (

                        qty_to_buy *
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


            if qty_to_buy <= 0:

                continue


            cash -= total_cost


            # =================================================
            # NEW POSITION
            # =================================================

            if sym not in holdings:

                holdings[sym] = {

                    "qty":
                    qty_to_buy,

                    "entry_price":
                    price,

                    "entry_date":
                    date,

                    "entry_cost":
                    b_cost,

                    "last_rank":
                    item["rank"],

                }


            # =================================================
            # EXISTING POSITION
            # =================================================

            else:

                pos = holdings[sym]


                # Existing position is treated as one
                # tax lot for this simplified model.
                #
                # Entry price is updated as weighted average.
                old_qty = pos["qty"]

                old_value = (

                    old_qty *
                    pos["entry_price"]

                )

                new_value = (

                    qty_to_buy *
                    price

                )

                total_qty = (

                    old_qty +
                    qty_to_buy

                )


                if total_qty > 0:

                    pos["entry_price"] = (

                        old_value +
                        new_value

                    ) / total_qty


                pos["qty"] = (
                    total_qty
                )


                pos["entry_cost"] += (
                    b_cost
                )


                pos["last_rank"] = (
                    item["rank"]
                )


        # ====================================================
        # 6. FINAL EQUAL-WEIGHT CLEANUP
        # ====================================================
        #
        # Integer shares mean exact 10% weighting is impossible.
        # We perform one additional sell pass if a holding is
        # materially above target.
        # ====================================================

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            price = float(

                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]

            )

            portfolio_value += (

                pos["qty"] *
                price

            )


        target_value = (

            portfolio_value /
            TOP_N

        )


        # ====================================================
        # 7. DAILY MARK-TO-MARKET
        # ====================================================

        portfolio_value = cash


        for sym, pos in (
            holdings.items()
        ):

            price = float(

                all_signals[sym]
                .loc[
                    date,
                    "price"
                ]

            )

            portfolio_value += (

                pos["qty"] *
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
        and
        holdings
    ):

        last_date = (
            trading_days[-1]
        )


        for sym, pos in (
            holdings.items()
        ):

            exit_price = float(

                all_signals[sym]
                .loc[
                    last_date,
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

    daily_top10_df = pd.DataFrame(
        daily_top10_log
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

        daily_top10_df,

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


        annualized_return = (

            equity_df[
                "equity"
            ].iloc[-1]

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
    # SUMMARY DICTIONARY
    # ========================================================

    return {

        "Backtest Start":
        BACKTEST_START,

        "Backtest End":
        equity_df[
            "date"
        ].iloc[-1]

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
                    f"{label} rows "
                    f"{i}-{i+len(chunk)}: "
                    f"{e2}"

                )

                raise


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
            f"existing charts "
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
# WRITE RESULTS TO GOOGLE SHEETS
# ============================================================

def write_to_sheet(

    trade_df,

    equity_df,

    open_df,

    daily_top10_df,

    summary,

    effective_end_str

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

            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS -- "
            "saving to CSV instead."

        )


        trade_df.to_csv(

            "backtest_trades.csv",

            index=False

        )


        equity_df.to_csv(

            "backtest_equity.csv",

            index=False

        )


        daily_top10_df.to_csv(

            "backtest_daily_top10.csv",

            index=False

        )


        if not open_df.empty:

            open_df.to_csv(

                "backtest_open_positions.csv",

                index=False

            )


        return


    # ========================================================
    # GOOGLE AUTH
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
    # SHEET SIZE
    # ========================================================

    n_rows_needed = (

        len(trade_df) +

        len(equity_df) +

        len(open_df) +

        len(daily_top10_df) +

        len(summary) +

        100

    )


    n_cols_needed = 25


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


    # ========================================================
    # CLEAR OLD RESULTS
    # ========================================================

    remove_existing_charts(

        sh,

        ws.id

    )

    ws.clear()


    # ========================================================
    # HEADER
    # ========================================================

    ws.update(

        [[

            "STRICT DAILY TOP-10 RS BACKTEST | "
            f"run {timestamp} | "
            "NET of costs+STCG | "
            f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
            "Entry: Price TT + RS TT | "
            "Ranking: Daily raw RS Score | "
            "Portfolio: Daily Top 10 | "
            "Weight: Equal | "
            "Execution: SAME DAY CLOSE | "
            f"Window: {BACKTEST_START} "
            f"to {effective_end_str}"

        ]],

        "A1"

    )


    # ========================================================
    # SUMMARY
    # ========================================================

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


    # ========================================================
    # DAILY TOP 10
    # ========================================================

    top10_start_row = (

        3 +
        len(summary_rows) +
        2

    )


    ws.update(

        [["Daily Top 10 RS Stocks"]],

        f"A{top10_start_row}"

    )


    top10_header_row = (

        top10_start_row + 1

    )


    if not daily_top10_df.empty:

        write_in_chunks(

            ws,

            [

                list(
                    daily_top10_df.columns
                )

            ]

            +

            daily_top10_df.values.tolist(),

            start_row=
            top10_header_row,

            chunk_size=
            1000,

            label=
            "daily top 10"

        )


    # ========================================================
    # TRADE LOG
    # ========================================================

    trade_start_row = (

        top10_header_row +

        len(daily_top10_df) +

        3

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


    # ========================================================
    # OPEN POSITIONS
    # ========================================================

    open_start_row = (

        trade_header_row +

        len(trade_df) +

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


    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity_start_row = (

        open_header_row +

        max(
            len(open_df),
            1
        ) +

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

        f"\nBacktest results "
        f"written to "
        f"'{BACKTEST_WORKSHEET}' tab: "

        f"{len(trade_df)} trades, "

        f"{len(equity_df)} "
        f"trading days, "

        f"{len(daily_top10_df)} "
        f"daily top-10 records, "

        f"{len(open_df)} "
        f"open positions."

    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD UNIVERSE
    # ========================================================

    tickers = load_tickers()


    print(

        f"\nLoaded "
        f"{len(tickers)} tickers."

    )


    # ========================================================
    # DOWNLOAD DATES
    # ========================================================

    download_start, download_end = (
        get_download_dates()
    )


    print(
        "=" * 60
    )


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
        f"Daily Top {TOP_N}, equal weight"

    )


    print(

        "Rebalance        : "
        "EVERY TRADING DAY"

    )


    print(

        "Execution        : "
        "SAME DAY CLOSE"

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
    # STOCK DATA
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

            i:
            i + batch_size

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


        # ====================================================
        # PROCESS EACH STOCK
        # ====================================================

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


    # ========================================================
    # SIGNAL SUMMARY
    # ========================================================

    print(

        f"\nSignals computed for "
        f"{len(all_signals)} stocks."

    )


    print(

        f"Total data points repaired: "
        f"{total_bad_points}"

    )


    # ========================================================
    # LATEST AVAILABLE DATE
    # ========================================================

    if all_signals:

        latest_stock_date = max(

            df.index.max()

            for df in
            all_signals.values()

        )

    else:

        latest_stock_date = None


    benchmark_latest_date = pd.Timestamp(

        bench_close.index.max()

    )


    # ========================================================
    # EFFECTIVE BACKTEST END
    # ========================================================

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


    # ========================================================
    # FILTER FUNNEL
    # ========================================================

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

            if latest_date not in df.index:

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


            counts[
                "liquid"
            ] += 1


            if not bool(
                row["tt_pass"]
            ):

                continue


            counts[
                "+tt_pass"
            ] += 1


            if not bool(
                row["rs_tt_pass"]
            ):

                continue


            counts[
                "+rs_tt_pass"
            ] += 1


        print(

            f"\nFilter funnel on "
            f"{latest_date.strftime('%Y-%m-%d')}:"

        )


        for k, v in counts.items():

            print(

                f"  {k}: {v}"

            )


    # ========================================================
    # TRADING DAYS
    # ========================================================

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


    # ========================================================
    # RUN BACKTEST
    # ========================================================

    (

        trade_df,

        equity_df,

        open_df,

        daily_top10_df,

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


    print(
        "\n--- SUMMARY ---"
    )


    for k, v in summary.items():

        print(
            f"{k}: {v}"
        )


    # ========================================================
    # WRITE RESULTS
    # ========================================================

    write_to_sheet(

        trade_df,

        equity_df,

        open_df,

        daily_top10_df,

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