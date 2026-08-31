import os
import json
import time
from datetime import datetime
from time import perf_counter

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

DOWNLOAD_YEARS_BEFORE_START = 3

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

RS_3M = 63
RS_6M = 126
RS_9M = 189
RS_12M = 252

RS_WEIGHTS = (0.40, 0.20, 0.20, 0.20)

LOOKBACK_DAYS = 250

TOP_N = 10
STARTING_CAPITAL = 1_000_000

MAX_PLAUSIBLE_DAILY_MOVE = 0.30

CHART_WINDOWS = (50, 100, 365)


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

BACKTEST_WORKSHEET = "Backtest - RS Top10 BlueGreenDot"


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

    s = s[
        ~s.index.duplicated(
            keep="last"
        )
    ]

    return s.sort_index()


def get_download_dates():
    start = pd.Timestamp(BACKTEST_START)

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

    download_end = (
        pd.Timestamp(BACKTEST_END)
        + pd.Timedelta(days=1)
    )

    return (
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d")
    )


# ============================================================
# LOAD TICKERS
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

    output = [
        s if s.endswith(".NS") else s + ".NS"
        for s in symbols
    ]

    return list(
        dict.fromkeys(output)
    )


# ============================================================
# CLEAN PRICE DATA
# ============================================================

def clean_price_series(close):

    close = normalize_series_index(close)

    bad = (
        close.pct_change().abs()
        > MAX_PLAUSIBLE_DAILY_MOVE
    )

    n_bad = int(bad.sum())

    if n_bad == 0:
        return close, 0

    cleaned = close.copy()

    for idx in close.index[bad]:

        pos = cleaned.index.get_loc(idx)

        if pos > 0:
            cleaned.iloc[pos] = (
                cleaned.iloc[pos - 1]
            )

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

    if net_gain > 0:
        return (
            net_gain *
            STCG_EFFECTIVE_RATE
        )

    return 0.0


def max_affordable_qty(price, cash_budget):

    if price <= 0 or cash_budget <= 0:
        return 0

    qty = int(
        cash_budget / price
    )

    while qty > 0:

        trade_value = qty * price

        total_required = (
            trade_value +
            buy_side_cost(trade_value)
        )

        if total_required <= cash_budget + 1e-9:
            return qty

        qty -= 1

    return 0


def minimum_cash_for_one_share(price):

    if price <= 0:
        return float("inf")

    return (
        price +
        buy_side_cost(price)
    )


# ============================================================
# BENCHMARK DOWNLOAD
# ============================================================

def download_benchmark():

    download_start, download_end = (
        get_download_dates()
    )

    print(
        f"\nBenchmark download: "
        f"{download_start} -> "
        f"{download_end if download_end else 'LATEST'}"
    )

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            t0 = perf_counter()

            data = yf.download(
                ticker,
                start=download_start,
                end=download_end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            elapsed = (
                perf_counter() - t0
            )

            print(
                f"Benchmark {ticker} "
                f"download: {elapsed:.1f}s"
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
                f"Benchmark loaded: {ticker}"
            )

            return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: {e}"
            )

    raise RuntimeError(
        "Could not download benchmark data."
    )


# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_stock_data(
    close,
    volume,
    bench_close
):

    close = normalize_series_index(
        close
    )

    volume = normalize_series_index(
        volume
    )

    bench_close = normalize_series_index(
        bench_close
    )

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

    if len(aligned) < 300:
        return None

    volume = (
        volume
        .reindex(
            aligned.index
        )
        .fillna(0)
    )

    avg_volume = (
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
            avg_volume >
            MIN_AVG_VOLUME
        )
    )

    w3, w6, w9, w12 = RS_WEIGHTS

    rs_score = (
        w3 * (
            aligned["s"] /
            aligned["s"].shift(RS_3M)
            - 1
        )
        +
        w6 * (
            aligned["s"] /
            aligned["s"].shift(RS_6M)
            - 1
        )
        +
        w9 * (
            aligned["s"] /
            aligned["s"].shift(RS_9M)
            - 1
        )
        +
        w12 * (
            aligned["s"] /
            aligned["s"].shift(RS_12M)
            - 1
        )
    ) * 100

    # ========================================================
    # RS RATIO
    # ========================================================

    rs_ratio = (
        aligned["s"] /
        aligned["b"]
    )

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

    result = pd.DataFrame(
        {
            "price":
                aligned["s"],

            "avg_volume":
                avg_volume,

            "liquid":
                liquid,

            "rs_score":
                rs_score,

            "rs_ratio":
                rs_ratio,

            "blue_dot":
                blue_dot,

            "green_dot":
                green_dot,
        },
        index=aligned.index
    )

    result.index = normalize_dates(
        result.index
    )

    return result


# ============================================================
# BUILD WIDE MATRICES
#
# CRITICAL PERFORMANCE FIX
# ============================================================

def build_signal_matrices(all_stocks):

    print(
        "\nBuilding vectorised signal matrices..."
    )

    t0 = perf_counter()

    price = pd.concat(
        {
            symbol:
                df["price"]
            for symbol, df
            in all_stocks.items()
        },
        axis=1
    )

    rs_score = pd.concat(
        {
            symbol:
                df["rs_score"]
            for symbol, df
            in all_stocks.items()
        },
        axis=1
    )

    liquid = pd.concat(
        {
            symbol:
                df["liquid"]
            for symbol, df
            in all_stocks.items()
        },
        axis=1
    ).fillna(False)

    blue_dot = pd.concat(
        {
            symbol:
                df["blue_dot"]
            for symbol, df
            in all_stocks.items()
        },
        axis=1
    ).fillna(False)

    green_dot = pd.concat(
        {
            symbol:
                df["green_dot"]
            for symbol, df
            in all_stocks.items()
        },
        axis=1
    ).fillna(False)

    for df in (
        price,
        rs_score,
        liquid,
        blue_dot,
        green_dot
    ):

        df.index = normalize_dates(
            df.index
        )

    elapsed = (
        perf_counter() - t0
    )

    print(
        f"Signal matrices built in "
        f"{elapsed:.2f}s"
    )

    print(
        f"Matrix size: "
        f"{len(price.index):,} dates x "
        f"{len(price.columns):,} stocks"
    )

    return (
        price,
        rs_score,
        liquid,
        blue_dot,
        green_dot
    )


# ============================================================
# PRECOMPUTE DAILY RANKINGS
# ============================================================

def precompute_daily_rankings(
    trading_days,
    symbols,
    price,
    rs_score,
    liquid,
    blue_dot
):

    print(
        "\nPrecomputing daily rankings..."
    )

    t0 = perf_counter()

    price = price.reindex(
        columns=symbols
    )

    rs_score = rs_score.reindex(
        columns=symbols
    )

    liquid = liquid.reindex(
        columns=symbols,
        fill_value=False
    )

    blue_dot = blue_dot.reindex(
        columns=symbols,
        fill_value=False
    )

    price_values = price.to_numpy(
        dtype=np.float64
    )

    rs_values = rs_score.to_numpy(
        dtype=np.float64
    )

    liquid_values = liquid.to_numpy(
        dtype=bool
    )

    blue_values = blue_dot.to_numpy(
        dtype=bool
    )

    date_to_row = {
        pd.Timestamp(date).normalize():
            i
        for i, date
        in enumerate(
            price.index
        )
    }

    rankings = {}

    price_by_day = {}

    total_days = len(
        trading_days
    )

    for counter, raw_date in enumerate(
        trading_days,
        start=1
    ):

        date = pd.Timestamp(
            raw_date
        ).normalize()

        row_idx = date_to_row.get(
            date
        )

        if row_idx is None:

            rankings[date] = []
            price_by_day[date] = {}

            continue

        prices = price_values[
            row_idx
        ]

        scores = rs_values[
            row_idx
        ]

        liquid_row = liquid_values[
            row_idx
        ]

        blue_row = blue_values[
            row_idx
        ]

        eligible = (
            liquid_row
            &
            blue_row
            &
            np.isfinite(scores)
            &
            np.isfinite(prices)
            &
            (prices > 0)
        )

        eligible_indices = np.flatnonzero(
            eligible
        )

        if len(
            eligible_indices
        ) == 0:

            rankings[date] = []

        else:

            eligible_scores = scores[
                eligible_indices
            ]

            order = np.argsort(
                -eligible_scores,
                kind="stable"
            )

            sorted_indices = (
                eligible_indices[
                    order
                ]
            )

            rankings[date] = [
                (
                    symbols[i],
                    float(scores[i]),
                    float(prices[i])
                )
                for i in sorted_indices
            ]

        # Actual prices for ALL stocks.
        # This is deliberately separate from ranking.

        valid_price_indices = np.flatnonzero(
            np.isfinite(prices)
            &
            (prices > 0)
        )

        price_by_day[date] = {
            symbols[i]:
                float(prices[i])
            for i in valid_price_indices
        }

        if (
            counter % 250 == 0
            or counter == total_days
        ):

            elapsed = (
                perf_counter() - t0
            )

            print(
                f"Rankings: "
                f"{counter:,}/"
                f"{total_days:,} days | "
                f"{elapsed:.1f}s"
            )

    elapsed = (
        perf_counter() - t0
    )

    print(
        f"Daily rankings completed in "
        f"{elapsed:.2f}s"
    )

    return (
        rankings,
        price_by_day
    )


# ============================================================
# BUY
# ============================================================

def execute_buy(
    symbol,
    price,
    qty,
    date,
    cash,
    holdings,
    trade_log
):

    if qty < 1:
        return cash, False

    trade_value = (
        qty * price
    )

    buy_cost = buy_side_cost(
        trade_value
    )

    total_required = (
        trade_value +
        buy_cost
    )

    if (
        total_required >
        cash + 1e-9
    ):
        return cash, False

    cash -= total_required

    holdings[symbol] = {
        "qty":
            int(qty),

        "entry_price":
            float(price),

        "entry_date":
            date,

        "entry_cost":
            float(buy_cost),

        "last_price":
            float(price),

        "last_price_date":
            date,
    }

    trade_log.append(
        {
            "symbol":
                symbol,

            "entry_date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "exit_date":
                "",

            "qty":
                int(qty),

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
                "",
        }
    )

    return cash, True


# ============================================================
# BUY MISSING TOP 10
# ============================================================

def buy_missing_top10(
    missing_symbols,
    price_lookup,
    date,
    cash,
    holdings,
    trade_log,
    unfilled_log=None
):

    if not missing_symbols:
        return cash

    for symbol in missing_symbols:

        price = price_lookup.get(
            symbol
        )

        if (
            price is None
            or pd.isna(price)
            or float(price) <= 0
        ):

            raise RuntimeError(
                f"{date:%Y-%m-%d}: "
                f"Top-10 stock {symbol} "
                f"has invalid entry price."
            )

    for i, symbol in enumerate(
        missing_symbols
    ):

        price = float(
            price_lookup[symbol]
        )

        slots_remaining = (
            len(missing_symbols) -
            i
        )

        equal_budget = (
            cash /
            slots_remaining
        )

        minimum_current = (
            minimum_cash_for_one_share(
                price
            )
        )

        target_budget = min(
            max(
                equal_budget,
                minimum_current
            ),
            cash
        )

        qty = max_affordable_qty(
            price,
            target_budget
        )

        if qty < 1:

            if unfilled_log is not None:

                unfilled_log.append(
                    {
                        "date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "symbol":
                            symbol,

                        "price":
                            round(
                                price,
                                4
                            ),

                        "cash_at_skip":
                            round(
                                cash,
                                2
                            ),

                        "reason":
                            "INSUFFICIENT_CASH_FOR_ONE_SHARE",
                    }
                )

            continue

        cash, bought = execute_buy(
            symbol,
            price,
            qty,
            date,
            cash,
            holdings,
            trade_log
        )

        if (
            not bought
            and
            unfilled_log is not None
        ):

            unfilled_log.append(
                {
                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "symbol":
                        symbol,

                    "price":
                        round(
                            price,
                            4
                        ),

                    "cash_at_skip":
                        round(
                            cash,
                            2
                        ),

                    "reason":
                        "BUY_EXECUTION_FAILED",
                }
            )

    return cash


# ============================================================
# OPEN POSITION DATAFRAME
# ============================================================

def build_open_positions_df(
    final_price_lookup,
    holdings
):

    rows = []

    for symbol, position in (
        holdings.items()
    ):

        last_price = float(
            final_price_lookup.get(
                symbol,
                position.get(
                    "last_price",
                    position[
                        "entry_price"
                    ]
                )
            )
        )

        rows.append(
            {
                "symbol":
                    symbol,

                "entry_date":
                    position[
                        "entry_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "qty":
                    int(
                        position["qty"]
                    ),

                "entry_price":
                    round(
                        position[
                            "entry_price"
                        ],
                        4
                    ),

                "last_price":
                    round(
                        last_price,
                        4
                    ),

                "gross_return_pct":
                    round(
                        (
                            last_price /
                            position[
                                "entry_price"
                            ]
                            - 1
                        ) * 100,
                        2
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LIQUIDATE OPEN POSITIONS
# ============================================================

def liquidate_open_positions(
    final_price_lookup,
    holdings,
    cash
):

    liquidation_cash = float(
        cash
    )

    for symbol, position in (
        holdings.items()
    ):

        exit_price = float(
            final_price_lookup.get(
                symbol,
                position.get(
                    "last_price",
                    position[
                        "entry_price"
                    ]
                )
            )
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

        tax = stcg_tax(
            net_gain
        )

        liquidation_cash += (
            net_proceeds -
            tax
        )

    return liquidation_cash


# ============================================================
# FAST BACKTEST ENGINE
# ============================================================

def run_backtest(
    trading_days,
    rankings,
    price_by_day
):

    cash = float(
        STARTING_CAPITAL
    )

    holdings = {}

    trade_log = []

    equity_curve = []

    unfilled_log = []

    initialized = False

    n_days = len(
        trading_days
    )

    t0 = perf_counter()

    print(
        "\nRunning optimised backtest..."
    )

    for day_number, raw_date in enumerate(
        trading_days,
        start=1
    ):

        date = pd.Timestamp(
            raw_date
        ).normalize()

        ranking = rankings.get(
            date,
            []
        )

        today_prices = (
            price_by_day.get(
                date,
                {}
            )
        )

        # ====================================================
        # TODAY'S TARGET
        # ====================================================

        eligible_pool_size = len(
            ranking
        )

        target_size = min(
            TOP_N,
            eligible_pool_size
        )

        today_top10 = [
            symbol
            for symbol, _, _
            in ranking[:TOP_N]
        ]

        today_top10_set = set(
            today_top10
        )

        rank_lookup = {
            symbol:
                rank
            for rank, (
                symbol,
                _,
                _
            )
            in enumerate(
                ranking,
                start=1
            )
        }

        # ====================================================
        # INITIAL PURCHASE
        # ====================================================

        if not initialized:

            if today_top10:

                entry_prices = {
                    symbol:
                        today_prices[symbol]
                    for symbol in today_top10
                    if symbol in today_prices
                }

                cash = buy_missing_top10(
                    today_top10,
                    entry_prices,
                    date,
                    cash,
                    holdings,
                    trade_log,
                    unfilled_log
                )

            initialized = True

        # ====================================================
        # DAILY REBALANCE
        # ====================================================

        else:

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            exit_symbols = [
                symbol
                for symbol in list(
                    holdings.keys()
                )
                if symbol
                not in today_top10_set
            ]

            for symbol in exit_symbols:

                position = holdings.pop(
                    symbol
                )

                # USE ACTUAL STOCK PRICE.
                # Do NOT use filtered ranking price.

                exit_price = today_prices.get(
                    symbol,
                    position.get(
                        "last_price",
                        position[
                            "entry_price"
                        ]
                    )
                )

                exit_price = float(
                    exit_price
                )

                if symbol in rank_lookup:

                    exit_reason = (
                        f"RANK_"
                        f"{rank_lookup[symbol]}"
                        f"_DROPPED_OUTSIDE_TOP10"
                    )

                else:

                    exit_reason = (
                        "MISSING_FROM_FILTERED_"
                        "RANKING_FORCE_EXIT"
                    )

                qty = int(
                    position["qty"]
                )

                gross_proceeds = (
                    qty *
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
                    qty *
                    position[
                        "entry_price"
                    ]
                    +
                    position[
                        "entry_cost"
                    ]
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

                gross_return_pct = (
                    (
                        exit_price /
                        position[
                            "entry_price"
                        ]
                        - 1
                    )
                    * 100
                )

                net_pnl = (
                    net_gain -
                    tax
                )

                net_return_pct = (
                    (
                        net_pnl /
                        cost_basis
                    ) * 100
                    if cost_basis > 0
                    else 0
                )

                days_held = (
                    date -
                    position[
                        "entry_date"
                    ]
                ).days

                trade_log.append(
                    {
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
                            exit_reason,
                    }
                )

            # ------------------------------------------------
            # MARK RETAINED HOLDINGS
            # ------------------------------------------------

            for symbol, position in (
                holdings.items()
            ):

                if symbol in today_prices:

                    position[
                        "last_price"
                    ] = float(
                        today_prices[
                            symbol
                        ]
                    )

                    position[
                        "last_price_date"
                    ] = date

            # ------------------------------------------------
            # BUY MISSING TARGET NAMES
            # ------------------------------------------------

            missing_top10 = [
                symbol
                for symbol in today_top10
                if symbol not in holdings
            ]

            if missing_top10:

                entry_prices = {
                    symbol:
                        today_prices[symbol]
                    for symbol in missing_top10
                    if symbol in today_prices
                }

                cash = buy_missing_top10(
                    missing_top10,
                    entry_prices,
                    date,
                    cash,
                    holdings,
                    trade_log,
                    unfilled_log
                )

        # ====================================================
        # PORTFOLIO INVARIANT
        # ====================================================

        held_symbols = set(
            holdings.keys()
        )

        illegal_holdings = (
            held_symbols -
            today_top10_set
        )

        if illegal_holdings:

            raise RuntimeError(
                f"{date:%Y-%m-%d}: "
                f"Portfolio contains stocks "
                f"outside today's target: "
                f"{sorted(illegal_holdings)}"
            )

        # ====================================================
        # MARK TO MARKET
        # ====================================================

        total_value = float(
            cash
        )

        for symbol, position in (
            holdings.items()
        ):

            mark_price = today_prices.get(
                symbol,
                position.get(
                    "last_price",
                    position[
                        "entry_price"
                    ]
                )
            )

            mark_price = float(
                mark_price
            )

            position[
                "last_price"
            ] = mark_price

            position[
                "last_price_date"
            ] = date

            total_value += (
                position["qty"] *
                mark_price
            )

        equity_curve.append(
            {
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
                        total_value -
                        cash,
                        2
                    ),

                "equity_multiple":
                    round(
                        total_value /
                        STARTING_CAPITAL,
                        8
                    ),

                "n_holdings":
                    len(holdings),

                "eligible_pool_size":
                    eligible_pool_size,

                "top10_target_size":
                    target_size,
            }
        )

        if (
            day_number % 250 == 0
            or
            day_number == n_days
        ):

            elapsed = (
                perf_counter() -
                t0
            )

            print(
                f"Processed "
                f"{day_number:,}/"
                f"{n_days:,} | "
                f"{date:%Y-%m-%d} | "
                f"Eligible="
                f"{eligible_pool_size} | "
                f"Holdings="
                f"{len(holdings)} | "
                f"Equity="
                f"Rs.{total_value:,.0f} | "
                f"Elapsed="
                f"{elapsed:.1f}s"
            )

    # ========================================================
    # EQUITY DATAFRAME
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    if not equity_df.empty:

        equity_df = (
            _add_equity_analytics_columns(
                equity_df
            )
        )

    trade_df = pd.DataFrame(
        trade_log
    )

    final_marked_value = (
        float(
            equity_df[
                "portfolio_value_rs"
            ].iloc[-1]
        )
        if not equity_df.empty
        else STARTING_CAPITAL
    )

    # ========================================================
    # FINAL LIQUIDATION
    # ========================================================

    last_date = (
        pd.Timestamp(
            trading_days[-1]
        ).normalize()
    )

    final_price_lookup = (
        price_by_day.get(
            last_date,
            {}
        )
    )

    final_liquidation_value = (
        liquidate_open_positions(
            final_price_lookup,
            holdings,
            cash
        )
    )

    open_df = (
        build_open_positions_df(
            final_price_lookup,
            holdings
        )
    )

    unfilled_df = pd.DataFrame(
        unfilled_log
    )

    elapsed = (
        perf_counter() -
        t0
    )

    print(
        f"\nBacktest engine completed "
        f"in {elapsed:.2f}s"
    )

    return (
        equity_df,
        trade_df,
        open_df,
        final_marked_value,
        final_liquidation_value,
        unfilled_df
    )


# ============================================================
# EQUITY ANALYTICS
# ============================================================

def _add_equity_analytics_columns(
    equity_df
):

    running_max = (
        equity_df[
            "equity_multiple"
        ].cummax()
    )

    equity_df[
        "drawdown_pct"
    ] = (
        (
            equity_df[
                "equity_multiple"
            ]
            /
            running_max
            - 1
        )
        * 100
    ).round(3)

    first_value = float(
        equity_df[
            "portfolio_value_rs"
        ].iloc[0]
    )

    equity_df[
        "equity_curve_pct_norm"
    ] = (
        (
            equity_df[
                "portfolio_value_rs"
            ]
            /
            first_value
            - 1
        )
        * 100
    ).round(3)

    for window_days in (
        CHART_WINDOWS
    ):

        window_start = max(
            len(equity_df) -
            window_days,
            0
        )

        base_value = float(
            equity_df[
                "portfolio_value_rs"
            ].iloc[
                window_start
            ]
        )

        series = pd.Series(
            np.nan,
            index=equity_df.index,
            dtype=float
        )

        series.iloc[
            window_start:
        ] = (
            (
                equity_df[
                    "portfolio_value_rs"
                ].iloc[
                    window_start:
                ]
                /
                base_value
                - 1
            )
            * 100
        ).round(3)

        equity_df[
            f"equity_curve_pct_norm_last{window_days}"
        ] = series

    return equity_df


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

    marked_return = (
        final_marked_value /
        STARTING_CAPITAL
        - 1
    ) * 100

    liquidation_return = (
        final_liquidation_value /
        STARTING_CAPITAL
        - 1
    ) * 100

    max_dd = float(
        equity_df[
            "drawdown_pct"
        ].min()
    )

    daily_returns = (
        equity_df[
            "equity_multiple"
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

        n_days = len(
            equity_df
        )

        annualized_return = (
            equity_df[
                "equity_multiple"
            ].iloc[-1]
            **
            (
                252 /
                max(
                    n_days,
                    1
                )
            )
            - 1
        )

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
        abs(max_dd / 100)
        if max_dd != 0
        else 0
    )

    if not trade_df.empty:

        entries = trade_df[
            trade_df["action"] ==
            "ENTRY"
        ]

        exits = trade_df[
            trade_df["action"] ==
            "EXIT"
        ]

        total_costs = (
            exits[
                "sell_cost_rs"
            ]
            .fillna(0)
            .sum()
            if not exits.empty
            else 0
        )

        total_costs += (
            entries[
                "buy_cost_rs"
            ]
            .fillna(0)
            .sum()
            if not entries.empty
            else 0
        )

        total_tax = (
            exits[
                "stcg_tax_rs"
            ]
            .fillna(0)
            .sum()
            if not exits.empty
            else 0
        )

        if not exits.empty:

            net_returns = (
                exits[
                    "net_return_pct"
                ].astype(float)
            )

            win_rate = (
                net_returns > 0
            ).mean() * 100

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

            gross_loss = (
                abs(
                    losers[
                        "net_pnl_rs"
                    ]
                    .astype(float)
                    .sum()
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
            "40% 3M + 20% 6M + "
            "20% 9M + 20% 12M",

        "Entry Filter":
            f"Blue Dot only "
            f"(RS-ratio "
            f"{LOOKBACK_DAYS}D new high). "
            f"Green Dot computed/logged "
            f"but not used for entry.",

        "Portfolio":
            "Exact Top 10 of "
            "Blue-Dot-filtered "
            "RS ranking",

        "Weight":
            "Available replacement cash "
            "divided across missing "
            "Top-10 names; retained "
            "positions untouched",

        "Entry":
            "Initial filtered Top 10; "
            "subsequently every missing "
            "name",

        "Exit":
            "Drops out of filtered "
            "Top 10 or missing from "
            "ranking",

        "Execution":
            "Same-day close (T+0)",

        "Rebalance Frequency":
            "Daily membership check; "
            "no resizing of retained "
            "positions",

        "Price Filter":
            f"> Rs.{MIN_PRICE}",

        "Liquidity Filter":
            f"{VOLUME_LOOKBACK}D average "
            f"volume > "
            f"{MIN_AVG_VOLUME:,}",

        "Other Filters":
            f"Blue Dot only "
            f"(lookback "
            f"{LOOKBACK_DAYS}D). "
            f"Green Dot computed/logged "
            f"but not gating. "
            f"No trend template, "
            f"sector, regime, or "
            f"breadth filter.",
    }


# ============================================================
# GOOGLE SHEETS
# ============================================================

def sanitize_for_sheets(df):

    if df.empty:
        return df

    clean = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return clean.where(
        pd.notnull(clean),
        ""
    )


def sanitize_scalar(v):

    if (
        isinstance(v, float)
        and (
            np.isnan(v)
            or np.isinf(v)
        )
    ):
        return ""

    return v


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
            i:
            i + chunk_size
        ]

        row_start = (
            start_row + i
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

                is_quota = (
                    "429" in str(e)
                    or
                    "Quota exceeded"
                    in str(e)
                )

                if (
                    not is_quota
                    or
                    attempt ==
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
                    initial_retry_seconds *
                    (2 ** attempt)
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
            f"{min(i + chunk_size, total)}/"
            f"{total} rows"
        )


def remove_existing_charts(
    sh,
    sheet_id
):

    try:

        meta = (
            sh.fetch_sheet_metadata()
        )

        requests = [
            {
                "deleteEmbeddedObject":
                    {
                        "objectId":
                            chart["chartId"]
                    }
            }
            for sheet in meta.get(
                "sheets",
                []
            )
            if sheet[
                "properties"
            ][
                "sheetId"
            ] == sheet_id
            for chart in sheet.get(
                "charts",
                []
            )
        ]

        if requests:

            sh.batch_update(
                {
                    "requests":
                        requests
                }
            )

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


def add_charts(
    sh,
    sheet_id,
    equity_header_row_0idx,
    n_equity_rows,
    equity_columns
):

    col_idx = {
        name: i
        for i, name
        in enumerate(
            equity_columns
        )
    }

    data_end_row = (
        equity_header_row_0idx +
        1 +
        n_equity_rows
    )

    def window_start_row(
        window_days
    ):

        return (
            equity_header_row_0idx +
            1 +
            max(
                n_equity_rows -
                window_days,
                0
            )
        )

    def make_chart(
        title,
        y_col_name,
        y_axis_title,
        anchor_row,
        chart_type="LINE",
        start_row_override=None,
        show_points=False,
        width_pixels=650
    ):

        y_col = col_idx[
            y_col_name
        ]

        series_start = (
            start_row_override
            if start_row_override is not None
            else equity_header_row_0idx
        )

        series_entry = {
            "series": {
                "sourceRange": {
                    "sources": [
                        {
                            "sheetId":
                                sheet_id,

                            "startRowIndex":
                                series_start,

                            "endRowIndex":
                                data_end_row,

                            "startColumnIndex":
                                y_col,

                            "endColumnIndex":
                                y_col + 1,
                        }
                    ]
                }
            },

            "targetAxis":
                "LEFT_AXIS",
        }

        if show_points:

            series_entry[
                "pointStyle"
            ] = {
                "size": 5,
                "shape": "CIRCLE"
            }

            series_entry[
                "dataLabel"
            ] = {
                "type": "DATA",
                "placement": "BELOW",
                "textFormat": {
                    "fontSize": 7
                },
            }

        return {
            "addChart": {
                "chart": {

                    "spec": {

                        "title":
                            title,

                        "basicChart": {

                            "chartType":
                                chart_type,

                            "legendPosition":
                                "NO_LEGEND",

                            "axis": [

                                {
                                    "position":
                                        "BOTTOM_AXIS",

                                    "title":
                                        "Date",
                                },

                                {
                                    "position":
                                        "LEFT_AXIS",

                                    "title":
                                        y_axis_title,
                                },
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
                                                        series_start,

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
                                series_entry
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
                                width_pixels,

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
            "portfolio_value_rs",
            "Portfolio Value (Rs)",
            equity_header_row_0idx
        ),

        make_chart(
            "Drawdown (%)",
            "drawdown_pct",
            "Drawdown %",
            equity_header_row_0idx + 22
        ),

        make_chart(
            "Eligible Pool Size (Blue Dot)",
            "eligible_pool_size",
            "Stock Count",
            equity_header_row_0idx + 44
        ),

        make_chart(
            "Equity Curve - Normalised "
            "to Zero (%, Since Inception)",
            "equity_curve_pct_norm",
            "Cumulative Change %",
            equity_header_row_0idx + 66
        ),

        make_chart(
            "Equity Curve - Normalised "
            "to Zero (%, Last 50 Days)",
            "equity_curve_pct_norm_last50",
            "Cumulative Change %",
            equity_header_row_0idx + 88,
            start_row_override=
                window_start_row(50),
            show_points=True,
            width_pixels=1100
        ),

        make_chart(
            "Equity Curve - Normalised "
            "to Zero (%, Last 100 Days)",
            "equity_curve_pct_norm_last100",
            "Cumulative Change %",
            equity_header_row_0idx + 110,
            start_row_override=
                window_start_row(100)
        ),

        make_chart(
            "Equity Curve - Normalised "
            "to Zero (%, Last 365 Days)",
            "equity_curve_pct_norm_last365",
            "Cumulative Change %",
            equity_header_row_0idx + 132,
            start_row_override=
                window_start_row(365)
        ),
    ]

    try:

        sh.batch_update(
            {
                "requests":
                    requests
            }
        )

        print(
            "Charts added."
        )

    except Exception as e:

        print(
            f"Could not add charts "
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
    unfilled_df,
    effective_end_str
):

    t0 = perf_counter()

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if (
        not sheet_id
        or
        not creds_json
    ):

        print(
            "Missing "
            "SHEET_ID/"
            "GOOGLE_CREDENTIALS "
            "-- saving to CSV instead."
        )

        trade_df.to_csv(
            "RS_BlueGreenDot_Trade_Log.csv",
            index=False
        )

        equity_df.to_csv(
            "RS_BlueGreenDot_Equity_Curve.csv",
            index=False
        )

        if not open_df.empty:

            open_df.to_csv(
                "RS_BlueGreenDot_Open_Positions.csv",
                index=False
            )

        if not unfilled_df.empty:

            unfilled_df.to_csv(
                "RS_BlueGreenDot_Unfilled_Slots.csv",
                index=False
            )

        return

    creds = (
        Credentials
        .from_service_account_info(
            json.loads(
                creds_json
            ),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets"
            ]
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

    n_cols_needed = max(
        len(trade_df.columns)
        if not trade_df.empty
        else 0,

        len(equity_df.columns)
        if not equity_df.empty
        else 0,

        len(open_df.columns)
        if not open_df.empty
        else 0,

        len(unfilled_df.columns)
        if not unfilled_df.empty
        else 0,

        2
    )

    n_rows_needed = (
        len(trade_df)
        +
        len(equity_df)
        +
        len(open_df)
        +
        len(unfilled_df)
        +
        len(summary)
        +
        60
    )

    ws = get_or_create_worksheet(
        sh,
        BACKTEST_WORKSHEET,
        rows=n_rows_needed,
        cols=n_cols_needed
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

    remove_existing_charts(
        sh,
        ws.id
    )

    ws.clear()

    ws.update(
        [[
            "TOP 10 RS ROTATION BACKTEST "
            "-- BLUE DOT FILTERED "
            "(Green Dot tracked, "
            "not used for entry) | "
            f"run {timestamp} | "
            "NET of costs+STCG | "
            f"Capital: "
            f"Rs.{STARTING_CAPITAL:,.0f} | "
            "Entry filter: Blue Dot only | "
            "Target: exact Top 10 "
            "of filtered RS ranking | "
            "Retained holdings NOT resized | "
            "Sell on drop from filtered "
            "Top 10 | "
            "Same EOD bar | "
            f"Window: "
            f"{BACKTEST_START} to "
            f"{effective_end_str}"
        ]],
        "A1"
    )

    summary_rows = (
        [["Summary", ""]]
        +
        [
            [
                k,
                sanitize_scalar(v)
            ]
            for k, v
            in summary.items()
        ]
    )

    ws.update(
        summary_rows,
        "A3"
    )

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

        trade_df_clean = (
            sanitize_for_sheets(
                trade_df
            )
        )

        write_in_chunks(
            ws,
            [
                list(
                    trade_df_clean.columns
                )
            ]
            +
            trade_df_clean.values.tolist(),
            start_row=
                trade_header_row,
            chunk_size=2000,
            label="trade log"
        )

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

        open_df_clean = (
            sanitize_for_sheets(
                open_df
            )
        )

        ws.update(
            [
                list(
                    open_df_clean.columns
                )
            ]
            +
            open_df_clean.values.tolist(),
            f"A{open_header_row}"
        )

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

        equity_df_clean = (
            sanitize_for_sheets(
                equity_df
            )
        )

        write_in_chunks(
            ws,
            [
                list(
                    equity_df_clean.columns
                )
            ]
            +
            equity_df_clean.values.tolist(),
            start_row=
                equity_header_row,
            chunk_size=2000,
            label="equity curve"
        )

        add_charts(
            sh,
            ws.id,
            equity_header_row - 1,
            len(equity_df),
            equity_columns=
                list(
                    equity_df_clean.columns
                )
        )

    unfilled_start_row = (
        equity_header_row +
        len(equity_df) +
        3
    )

    ws.update(
        [[
            "Unfilled Slots "
            "(buy attempts skipped "
            "-- insufficient cash)"
        ]],
        f"A{unfilled_start_row}"
    )

    unfilled_header_row = (
        unfilled_start_row + 1
    )

    if not unfilled_df.empty:

        unfilled_df_clean = (
            sanitize_for_sheets(
                unfilled_df
            )
        )

        ws.update(
            [
                list(
                    unfilled_df_clean.columns
                )
            ]
            +
            unfilled_df_clean.values.tolist(),
            f"A{unfilled_header_row}"
        )

    elapsed = (
        perf_counter() -
        t0
    )

    print(
        f"\nGoogle Sheets/output completed "
        f"in {elapsed:.2f}s"
    )

    print(
        f"Results written to "
        f"'{BACKTEST_WORKSHEET}' tab."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "TOP 10 RS ROTATION BACKTEST "
        "-- BLUE DOT FILTERED"
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
        "DAILY RS SCORE, "
        "FILTERED TO BLUE DOT"
    )

    print(
        "Portfolio      : "
        "EXACT TOP 10 OF "
        "FILTERED RANKING"
    )

    print(
        "Initial        : "
        "BUY ALL FILTERED TOP 10"
    )

    print(
        "Rotation       : "
        "SELL ONLY STOCKS LEAVING "
        "FILTERED TOP 10"
    )

    print(
        "New entries    : "
        "BUY EVERY MISSING "
        "FILTERED-TOP-10 STOCK"
    )

    print(
        "Existing names : "
        "HOLD / NO RESIZING"
    )

    print(
        "Sizing         : "
        "AVAILABLE CASH / "
        "MISSING SLOTS"
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
        f"{VOLUME_LOOKBACK}D average "
        f"volume > "
        f"{MIN_AVG_VOLUME:,}"
    )

    print(
        f"Entry filter   : "
        f"Blue Dot only "
        f"({LOOKBACK_DAYS}D lookback)"
    )

    print(
        "Green Dot      : "
        "COMPUTED/LOGGED ONLY"
    )

    print(
        "Other filters  : NONE"
    )

    print("=" * 70)

    total_t0 = perf_counter()

    # ========================================================
    # LOAD TICKERS
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
    # BENCHMARK
    # ========================================================

    bench_close = (
        download_benchmark()
    )

    bench_close.index = normalize_dates(
        bench_close.index
    )

    # ========================================================
    # STOCK DOWNLOADS
    # ========================================================

    all_stocks = {}

    total_bad_points = 0

    batch_size = 50

    download_t0 = perf_counter()

    total_batches = (
        (
            len(tickers) +
            batch_size -
            1
        )
        //
        batch_size
    )

    for start in range(
        0,
        len(tickers),
        batch_size
    ):

        batch = tickers[
            start:
            start + batch_size
        ]

        batch_number = (
            start // batch_size
        ) + 1

        print(
            f"\nDownloading batch "
            f"{batch_number}/"
            f"{total_batches}: "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(tickers)}"
        )

        batch_t0 = perf_counter()

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

        # ----------------------------------------------------
        # PROCESS EACH STOCK
        # ----------------------------------------------------

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

                    if (
                        symbol
                        not in
                        data.columns.get_level_values(
                            0
                        )
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

                if close.empty:
                    continue

                if "Volume" in sdata.columns:

                    volume = (
                        sdata["Volume"]
                        .reindex(
                            close.index
                        )
                        .fillna(0)
                    )

                else:

                    volume = pd.Series(
                        0.0,
                        index=close.index
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
                        volume,
                        bench_close
                    )
                )

                if stock_data is None:
                    continue

                clean_symbol = (
                    symbol.replace(
                        ".NS",
                        ""
                    )
                )

                all_stocks[
                    clean_symbol
                ] = stock_data

            except Exception as e:

                print(
                    f"Skipping "
                    f"{symbol}: "
                    f"{e}"
                )

        batch_elapsed = (
            perf_counter() -
            batch_t0
        )

        print(
            f"Batch completed in "
            f"{batch_elapsed:.1f}s | "
            f"Usable stocks so far: "
            f"{len(all_stocks)}"
        )

        time.sleep(0.25)

    download_elapsed = (
        perf_counter() -
        download_t0
    )

    print(
        f"\nStock download/signal stage: "
        f"{download_elapsed:.1f}s"
    )

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
    # EFFECTIVE END
    # ========================================================

    latest_stock_date = max(
        df.index.max()
        for df in all_stocks.values()
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

    print(
        f"\nLatest stock date: "
        f"{pd.Timestamp(latest_stock_date):%Y-%m-%d}"
    )

    print(
        f"Latest benchmark date: "
        f"{pd.Timestamp(latest_benchmark_date):%Y-%m-%d}"
    )

    print(
        f"Effective end: "
        f"{effective_end:%Y-%m-%d}"
    )

    # ========================================================
    # TRADING CALENDAR
    # ========================================================

    trading_days = (
        bench_close.index[
            (
                bench_close.index >=
                pd.Timestamp(
                    BACKTEST_START
                ).normalize()
            )
            &
            (
                bench_close.index <=
                effective_end
            )
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
        f"{len(trading_days):,}"
    )

    if len(trading_days) == 0:

        raise RuntimeError(
            "No trading days found."
        )

    print(
        f"First day: "
        f"{trading_days[0]:%Y-%m-%d}"
    )

    print(
        f"Last day: "
        f"{trading_days[-1]:%Y-%m-%d}"
    )

    # ========================================================
    # VECTORISED MATRICES
    # ========================================================

    (
        price_matrix,
        rs_matrix,
        liquid_matrix,
        blue_matrix,
        green_matrix
    ) = build_signal_matrices(
        all_stocks
    )

    symbols = list(
        price_matrix.columns
    )

    # ========================================================
    # PRECOMPUTE RANKINGS
    # ========================================================

    (
        rankings,
        price_by_day
    ) = precompute_daily_rankings(
        trading_days,
        symbols,
        price_matrix,
        rs_matrix,
        liquid_matrix,
        blue_matrix
    )

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    (
        equity_df,
        trade_df,
        open_df,
        final_marked,
        final_liq,
        unfilled_df
    ) = run_backtest(
        trading_days,
        rankings,
        price_by_day
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if not equity_df.empty:

        broken = equity_df[
            equity_df[
                "n_holdings"
            ]
            !=
            equity_df[
                "top10_target_size"
            ]
        ]

        if not broken.empty:

            print(
                f"\nNOTE: "
                f"{len(broken):,}/"
                f"{len(equity_df):,} "
                f"days held fewer "
                f"positions than target."
            )

        low_pool_days = int(
            (
                equity_df[
                    "eligible_pool_size"
                ]
                <
                TOP_N
            ).sum()
        )

        if low_pool_days:

            print(
                f"NOTE: "
                f"{low_pool_days:,}/"
                f"{len(equity_df):,} "
                f"days had fewer than "
                f"{TOP_N} Blue-Dot "
                f"eligible stocks."
            )

        if not unfilled_df.empty:

            print(
                f"NOTE: "
                f"{len(unfilled_df):,} "
                f"buy attempts skipped "
                f"for insufficient cash."
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
    # GOOGLE SHEETS / CSV
    # ========================================================

    write_to_sheet(
        trade_df,
        equity_df,
        open_df,
        summary,
        unfilled_df,
        effective_end.strftime(
            "%Y-%m-%d"
        )
    )

    equity_df.to_csv(
        "RS_BlueGreenDot_Equity_Curve.csv",
        index=False
    )

    trade_df.to_csv(
        "RS_BlueGreenDot_Trade_Log.csv",
        index=False
    )

    if not open_df.empty:

        open_df.to_csv(
            "RS_BlueGreenDot_Open_Positions.csv",
            index=False
        )

    if not unfilled_df.empty:

        unfilled_df.to_csv(
            "RS_BlueGreenDot_Unfilled_Slots.csv",
            index=False
        )

    total_elapsed = (
        perf_counter() -
        total_t0
    )

    print(
        f"\nTotal runtime: "
        f"{total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min)"
    )

    print(
        "\nCSV files also saved."
    )

    print(
        "\nBACKTEST COMPLETED SUCCESSFULLY."
    )


# ============================================================
# ENTRY POINT
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