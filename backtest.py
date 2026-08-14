"""
RS SCREENER BACKTEST — FINAL SYNCHRONIZED MODEL

RULES
=====

UNIVERSE
--------
stocks.csv

PRICE
-----
Price >= Rs.10

LIQUIDITY
---------
50-day average volume >= 50,000 shares

PRICE TREND TEMPLATE
--------------------
7/7 Minervini Trend Template

RS LINE TREND TEMPLATE
----------------------
7/7 Minervini Trend Template
applied to Stock / Benchmark

RS SCORE
--------
40% 63-day return
20% 126-day return
20% 189-day return
20% 252-day return

RANKING
-------
Rank eligible stocks by raw RS Score.

ENTRY
-----
Top 10 eligible stocks.

HOLD
----
Rank 1-15.

EXIT
----
Rank >15.

PORTFOLIO
---------
Equal weight.
Daily EOD execution.

NO
--
VCP
Contraction
Volume dry-up
Pivot
Breakout volume
Blue Dot
Green Dot
RS 5EMA exit
Stop loss
Trailing stop
Regime filter
"""


import os
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from datetime import datetime
from zoneinfo import ZoneInfo

from google.oauth2.service_account import Credentials


# ============================================================
# PARAMETERS
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"

BACKTEST_END = (
    datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%Y-%m-%d")
)

STARTING_CAPITAL = 1_000_000

TOP_N = 10
EXIT_RANK = 15

MIN_PRICE = 10

MIN_AVG_VOLUME = 50_000
VOLUME_LOOKBACK = 50

DOWNLOAD_YEARS_BEFORE_START = 3

# ============================================================
# COSTS
# ============================================================

ENABLE_COSTS = True

STT_BUY_RATE = 0.001
STT_SELL_RATE = 0.001

STAMP_DUTY_RATE = 0.00015

EXCHANGE_CHARGE_RATE = 0.0000325

SEBI_CHARGE_RATE = 0.000001

GST_RATE = 0.18

DP_CHARGE_FLAT = 20

# ============================================================
# STCG
# ============================================================

ENABLE_STCG = True

STCG_RATE = 0.20
STCG_CESS = 0.04

STCG_EFFECTIVE_RATE = (
    STCG_RATE
    * (1 + STCG_CESS)
)

# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

SUMMARY_SHEET = "Backtest_Summary"
BACKTEST_SHEET = "Backtest"


# ============================================================
# COST FUNCTIONS
# ============================================================

def buy_cost(value):

    if not ENABLE_COSTS:
        return 0.0

    return (
        STT_BUY_RATE * value
        +
        STAMP_DUTY_RATE * value
        +
        EXCHANGE_CHARGE_RATE * value
        +
        SEBI_CHARGE_RATE * value
        +
        GST_RATE * (
            EXCHANGE_CHARGE_RATE * value
            +
            SEBI_CHARGE_RATE * value
        )
    )


def sell_cost(value):

    if not ENABLE_COSTS:
        return 0.0

    return (
        STT_SELL_RATE * value
        +
        EXCHANGE_CHARGE_RATE * value
        +
        SEBI_CHARGE_RATE * value
        +
        GST_RATE * (
            EXCHANGE_CHARGE_RATE * value
            +
            SEBI_CHARGE_RATE * value
        )
        +
        DP_CHARGE_FLAT
    )


def stcg(gain):

    if not ENABLE_STCG:
        return 0.0

    return (
        max(0, gain)
        * STCG_EFFECTIVE_RATE
    )


# ============================================================
# LOAD TICKERS
# ============================================================

def load_tickers():

    df = pd.read_csv(
        STOCKS_FILE
    )

    if "symbol" not in df.columns:

        raise ValueError(
            "stocks.csv needs a 'symbol' column"
        )

    tickers = []

    for symbol in (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    ):

        if not symbol:
            continue

        if symbol.endswith(".NS"):
            tickers.append(symbol)
        else:
            tickers.append(
                symbol + ".NS"
            )

    return tickers


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template(series):

    if len(series) < 273:
        return None, None

    sma50 = (
        series
        .rolling(50)
        .mean()
    )

    sma150 = (
        series
        .rolling(150)
        .mean()
    )

    sma200 = (
        series
        .rolling(200)
        .mean()
    )

    price = series.iloc[-1]

    sma50_now = sma50.iloc[-1]

    sma150_now = sma150.iloc[-1]

    sma200_now = sma200.iloc[-1]

    sma200_1mo = sma200.iloc[-21]

    low52 = (
        series
        .tail(252)
        .min()
    )

    high52 = (
        series
        .tail(252)
        .max()
    )

    values = [
        price,
        sma50_now,
        sma150_now,
        sma200_now,
        sma200_1mo,
        low52,
        high52
    ]

    if any(
        pd.isna(x)
        for x in values
    ):
        return None, None

    criteria = [

        # 1
        (
            price > sma150_now
            and price > sma200_now
        ),

        # 2
        sma150_now > sma200_now,

        # 3
        sma200_now > sma200_1mo,

        # 4
        (
            sma50_now > sma150_now
            and sma50_now > sma200_now
        ),

        # 5
        price > sma50_now,

        # 6
        price >= 1.25 * low52,

        # 7
        price >= 0.75 * high52,
    ]

    met = sum(
        bool(x)
        for x in criteria
    )

    return (
        met == 7,
        met
    )


# ============================================================
# STOCK SIGNALS
# ============================================================

def stock_signals(
    close,
    volume,
    benchmark
):

    x = pd.concat(
        [
            close,
            benchmark
        ],
        axis=1,
        join="inner"
    ).dropna()

    x.columns = [
        "s",
        "b"
    ]

    if len(x) < 280:
        return None

    volume = (
        volume
        .reindex(x.index)
        .fillna(0)
    )

    # ========================================================
    # LIQUIDITY
    # ========================================================

    avg_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    liquid = (
        (
            x["s"]
            >= MIN_PRICE
        )
        &
        (
            avg_volume
            >= MIN_AVG_VOLUME
        )
    )

    # ========================================================
    # PRICE TREND TEMPLATE
    # ========================================================

    price_tt, price_met = (
        trend_template(
            x["s"]
        )
    )

    # ========================================================
    # RS LINE
    # ========================================================

    rs_line = (
        x["s"]
        / x["b"]
    )

    rs_tt, rs_met = (
        trend_template(
            rs_line
        )
    )

    # ========================================================
    # RS SCORE
    # ========================================================

    def ret(days):

        return (
            x["s"]
            / x["s"].shift(days)
            - 1
        )

    rs_score = (
        0.40 * ret(63)
        +
        0.20 * ret(126)
        +
        0.20 * ret(189)
        +
        0.20 * ret(252)
    ) * 100

    # ========================================================
    # ELIGIBILITY
    # ========================================================

    eligible = (
        price_tt
        &
        rs_tt
        &
        liquid
    )

    return pd.DataFrame({

        "price":
            x["s"],

        "rs_score":
            rs_score,

        "price_tt":
            price_tt,

        "price_tt_met":
            price_met,

        "rs_tt":
            rs_tt,

        "rs_tt_met":
            rs_met,

        "liquid":
            liquid,

        "avg_volume":
            avg_volume,

        "eligible":
            eligible
    })


# ============================================================
# DOWNLOAD BENCHMARK
# ============================================================

def get_benchmark(
    start,
    end
):

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            data = yf.download(
                ticker,
                start=start,
                end=end,
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
                close = close.iloc[:, 0]

            return (
                close
                .dropna()
                .sort_index()
            )

        except Exception as e:

            print(
                "Benchmark error:",
                ticker,
                e
            )

    raise RuntimeError(
        "Benchmark download failed"
    )


# ============================================================
# DOWNLOAD STOCKS
# ============================================================

def build_signals(
    tickers,
    start,
    end,
    benchmark
):

    signals = {}

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
            f"Downloading "
            f"{i + 1}-"
            f"{i + len(batch)} / "
            f"{len(tickers)}"
        )

        try:

            data = yf.download(
                batch,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )

        except Exception as e:

            print(
                "Batch failed:",
                e
            )

            continue

        for sym in batch:

            try:

                if len(batch) == 1:

                    d = data

                else:

                    if (
                        not isinstance(
                            data.columns,
                            pd.MultiIndex
                        )
                    ):
                        continue

                    if sym not in (
                        data
                        .columns
                        .get_level_values(0)
                    ):
                        continue

                    d = data[sym]

                if (
                    d.empty
                    or "Close"
                    not in d.columns
                ):
                    continue

                close = (
                    d["Close"]
                    .dropna()
                    .sort_index()
                )

                if "Volume" in d.columns:

                    volume = (
                        d["Volume"]
                        .reindex(
                            close.index
                        )
                        .fillna(0)
                    )

                else:

                    volume = pd.Series(
                        0,
                        index=close.index
                    )

                sig = stock_signals(
                    close,
                    volume,
                    benchmark
                )

                if sig is not None:

                    signals[
                        sym.replace(
                            ".NS",
                            ""
                        )
                    ] = sig

            except Exception as e:

                print(
                    f"Skipping {sym}: {e}"
                )

                continue

        time.sleep(0.5)

    return signals


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    signals,
    days
):

    cash = STARTING_CAPITAL

    holdings = {}

    trades = []

    equity = []

    ranking_log = []

    for date in days:

        # ====================================================
        # BUILD ELIGIBLE UNIVERSE
        # ====================================================

        candidates = []

        for sym, df in signals.items():

            if date not in df.index:
                continue

            r = df.loc[date]

            if pd.isna(
                r["rs_score"]
            ):
                continue

            if not bool(
                r["price_tt"]
            ):
                continue

            if not bool(
                r["rs_tt"]
            ):
                continue

            if not bool(
                r["liquid"]
            ):
                continue

            candidates.append(
                (
                    sym,
                    float(
                        r["rs_score"]
                    )
                )
            )

        # ====================================================
        # RANK
        # ====================================================

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        rank = {
            sym: i
            for i, (
                sym,
                score
            )
            in enumerate(
                candidates,
                1
            )
        }

        top10 = [
            sym
            for sym, score
            in candidates[:TOP_N]
        ]

        # ====================================================
        # RANKING AUDIT
        # ====================================================

        for i, (
            sym,
            score
        ) in enumerate(
            candidates,
            1
        ):

            r = signals[
                sym
            ].loc[date]

            ranking_log.append({

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "rank":
                    i,

                "symbol":
                    sym,

                "rs_score":
                    round(
                        score,
                        4
                    ),

                "price":
                    round(
                        float(
                            r["price"]
                        ),
                        2
                    ),

                "price_tt_met":
                    int(
                        r["price_tt_met"]
                    ),

                "rs_tt_met":
                    int(
                        r["rs_tt_met"]
                    ),

                "avg_volume_50d":
                    round(
                        float(
                            r["avg_volume"]
                        ),
                        0
                    ),

                "action":
                    (
                        "BUY / HOLD"
                        if i <= TOP_N
                        else
                        "HOLD ALLOWED"
                        if i <= EXIT_RANK
                        else
                        "EXIT"
                    )
            })

        # ====================================================
        # EXIT
        # ====================================================

        for sym in list(
            holdings
        ):

            if date not in signals[
                sym
            ].index:

                continue

            r = signals[
                sym
            ].loc[date]

            current_rank = rank.get(
                sym,
                999999
            )

            # HOLD 1-15
            if current_rank <= EXIT_RANK:
                continue

            pos = holdings.pop(
                sym
            )

            price = float(
                r["price"]
            )

            gross = (
                pos["qty"]
                * price
            )

            scost = sell_cost(
                gross
            )

            proceeds = (
                gross
                - scost
            )

            basis = (
                pos["qty"]
                * pos["entry_price"]
                +
                pos["buy_cost"]
            )

            gain = (
                proceeds
                - basis
            )

            tax = stcg(
                gain
            )

            pnl = (
                gain
                - tax
            )

            cash += (
                proceeds
                - tax
            )

            trades.append({

                "symbol":
                    sym,

                "entry_date":
                    pos["entry_date"]
                    .strftime(
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
                        price,
                        2
                    ),

                "entry_rank":
                    pos["entry_rank"],

                "exit_rank":
                    current_rank,

                "entry_rs_score":
                    pos["entry_score"],

                "exit_rs_score":
                    round(
                        float(
                            r["rs_score"]
                        ),
                        4
                    ),

                "gross_return_pct":
                    round(
                        (
                            price
                            / pos[
                                "entry_price"
                            ]
                            - 1
                        ) * 100,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        pos["buy_cost"],
                        2
                    ),

                "sell_cost_rs":
                    round(
                        scost,
                        2
                    ),

                "stcg_tax_rs":
                    round(
                        tax,
                        2
                    ),

                "net_pnl_rs":
                    round(
                        pnl,
                        2
                    ),

                "net_return_pct":
                    round(
                        pnl / basis * 100,
                        2
                    ),

                "days_held":
                    (
                        date
                        - pos["entry_date"]
                    ).days,

                "exit_reason":
                    "RS RANK > 15"
            })

        # ====================================================
        # PORTFOLIO VALUE AFTER EXITS
        # ====================================================

        value = cash

        for sym, pos in (
            holdings.items()
        ):

            if date in signals[
                sym
            ].index:

                price = float(
                    signals[
                        sym
                    ].loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    pos["entry_price"]
                )

            value += (
                pos["qty"]
                * price
            )

        # ====================================================
        # NEW ENTRIES
        # ====================================================

        slots = (
            TOP_N
            - len(holdings)
        )

        if slots > 0:

            target = (
                value / TOP_N
            )

            for sym in top10:

                if slots <= 0:
                    break

                if sym in holdings:
                    continue

                r = signals[
                    sym
                ].loc[date]

                price = float(
                    r["price"]
                )

                qty = int(
                    target
                    // price
                )

                if qty < 1:
                    continue

                trade_value = (
                    qty
                    * price
                )

                bcost = buy_cost(
                    trade_value
                )

                if (
                    trade_value
                    + bcost
                    > cash
                ):
                    continue

                cash -= (
                    trade_value
                    + bcost
                )

                holdings[sym] = {

                    "qty":
                        qty,

                    "entry_price":
                        price,

                    "entry_date":
                        date,

                    "buy_cost":
                        bcost,

                    "entry_rank":
                        rank[sym],

                    "entry_score":
                        round(
                            float(
                                r[
                                    "rs_score"
                                ]
                            ),
                            4
                        )
                }

                slots -= 1

        # ====================================================
        # FINAL DAILY EQUITY
        # ====================================================

        value = cash

        for sym, pos in (
            holdings.items()
        ):

            if date in signals[
                sym
            ].index:

                price = float(
                    signals[
                        sym
                    ].loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = (
                    pos["entry_price"]
                )

            value += (
                pos["qty"]
                * price
            )

        equity.append({

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "portfolio_value_rs":
                round(
                    value,
                    2
                ),

            "equity":
                round(
                    value
                    / STARTING_CAPITAL,
                    8
                ),

            "return_pct":
                round(
                    (
                        value
                        / STARTING_CAPITAL
                        - 1
                    ) * 100,
                    4
                ),

            "cash_rs":
                round(
                    cash,
                    2
                ),

            "n_holdings":
                len(holdings),

            "top10":
                ",".join(
                    top10
                ),

            "holdings":
                ",".join(
                    sorted(
                        holdings
                    )
                )
        })

    # ========================================================
    # MARK OPEN POSITIONS
    # ========================================================

    if days:

        last = days[-1]

        for sym, pos in (
            holdings.items()
        ):

            if (
                last
                not in signals[sym].index
            ):
                continue

            r = signals[
                sym
            ].loc[last]

            price = float(
                r["price"]
            )

            gross = (
                pos["qty"]
                * price
            )

            scost = sell_cost(
                gross
            )

            proceeds = (
                gross
                - scost
            )

            basis = (
                pos["qty"]
                * pos["entry_price"]
                +
                pos["buy_cost"]
            )

            gain = (
                proceeds
                - basis
            )

            tax = stcg(
                gain
            )

            pnl = (
                gain
                - tax
            )

            trades.append({

                "symbol":
                    sym,

                "entry_date":
                    pos["entry_date"]
                    .strftime(
                        "%Y-%m-%d"
                    ),

                "exit_date":
                    last.strftime(
                        "%Y-%m-%d"
                    )
                    + " OPEN",

                "qty":
                    pos["qty"],

                "entry_price":
                    round(
                        pos["entry_price"],
                        2
                    ),

                "exit_price":
                    round(
                        price,
                        2
                    ),

                "entry_rank":
                    pos["entry_rank"],

                "exit_rank":
                    "",

                "entry_rs_score":
                    pos["entry_score"],

                "exit_rs_score":
                    round(
                        float(
                            r[
                                "rs_score"
                            ]
                        ),
                        4
                    ),

                "gross_return_pct":
                    round(
                        (
                            price
                            / pos[
                                "entry_price"
                            ]
                            - 1
                        ) * 100,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        pos["buy_cost"],
                        2
                    ),

                "sell_cost_rs":
                    round(
                        scost,
                        2
                    ),

                "stcg_tax_rs":
                    round(
                        tax,
                        2
                    ),

                "net_pnl_rs":
                    round(
                        pnl,
                        2
                    ),

                "net_return_pct":
                    round(
                        pnl / basis * 100,
                        2
                    ),

                "days_held":
                    (
                        last
                        - pos[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "BACKTEST END"
            })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity),
        pd.DataFrame(ranking_log)
    )


# ============================================================
# SUMMARY
# ============================================================

def summary(
    trades,
    equity
):

    if equity.empty:
        return {}

    final = float(
        equity[
            "portfolio_value_rs"
        ].iloc[-1]
    )

    total_return = (
        final
        / STARTING_CAPITAL
        - 1
    ) * 100

    peak = (
        equity["equity"]
        .cummax()
    )

    dd = (
        equity["equity"]
        / peak
        - 1
    ) * 100

    max_dd = dd.min()

    if not trades.empty:

        closed = trades[
            ~trades[
                "exit_date"
            ]
            .astype(str)
            .str.contains(
                "OPEN"
            )
        ].copy()

    else:

        closed = trades

    n = len(
        closed
    )

    if n:

        win = (
            closed[
                "net_return_pct"
            ] > 0
        ).mean() * 100

        avg = (
            closed[
                "net_return_pct"
            ].mean()
        )

        median = (
            closed[
                "net_return_pct"
            ].median()
        )

        avg_days = (
            closed[
                "days_held"
            ].mean()
        )

        best = (
            closed[
                "net_return_pct"
            ].max()
        )

        worst = (
            closed[
                "net_return_pct"
            ].min()
        )

        costs = (
            closed[
                "buy_cost_rs"
            ].sum()
            +
            closed[
                "sell_cost_rs"
            ].sum()
        )

        tax = (
            closed[
                "stcg_tax_rs"
            ].sum()
        )

        winners = closed[
            closed[
                "net_pnl_rs"
            ] > 0
        ]

        losers = closed[
            closed[
                "net_pnl_rs"
            ] < 0
        ]

        if len(losers):

            pf = (
                winners[
                    "net_pnl_rs"
                ].sum()
                /
                abs(
                    losers[
                        "net_pnl_rs"
                    ].sum()
                )
            )

        else:

            pf = 0

    else:

        (
            win,
            avg,
            median,
            avg_days,
            best,
            worst,
            costs,
            tax,
            pf
        ) = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        )

    daily = (
        equity[
            "equity"
        ]
        .pct_change()
        .dropna()
    )

    if len(equity):

        ann_return = (
            equity[
                "equity"
            ].iloc[-1]
            **
            (
                252
                / len(equity)
            )
            - 1
        )

    else:

        ann_return = 0

    if len(daily):

        volatility = (
            daily.std()
            * np.sqrt(252)
        )

    else:

        volatility = 0

    if (
        len(daily)
        and daily.std() > 0
    ):

        sharpe = (
            daily.mean()
            / daily.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    return {

        "run_timestamp_ist":
            datetime.now(
                ZoneInfo(
                    "Asia/Kolkata"
                )
            ).strftime(
                "%Y-%m-%d %H:%M:%S IST"
            ),

        "backtest_start":
            BACKTEST_START,

        "backtest_end":
            BACKTEST_END,

        "starting_capital_rs":
            STARTING_CAPITAL,

        "final_portfolio_value_rs":
            round(
                final,
                2
            ),

        "net_total_return_pct":
            round(
                total_return,
                2
            ),

        "annualized_return_pct":
            round(
                ann_return * 100,
                2
            ),

        "annualized_volatility_pct":
            round(
                volatility * 100,
                2
            ),

        "max_drawdown_pct":
            round(
                max_dd,
                2
            ),

        "sharpe":
            round(
                sharpe,
                3
            ),

        "n_closed_trades":
            n,

        "win_rate_net_pct":
            round(
                win,
                1
            ),

        "avg_net_return_trade_pct":
            round(
                avg,
                2
            ),

        "median_net_return_trade_pct":
            round(
                median,
                2
            ),

        "avg_days_held":
            round(
                avg_days,
                1
            ),

        "best_net_trade_pct":
            round(
                best,
                2
            ),

        "worst_net_trade_pct":
            round(
                worst,
                2
            ),

        "profit_factor":
            round(
                pf,
                3
            ),

        "transaction_costs_rs":
            round(
                costs,
                2
            ),

        "stcg_tax_rs":
            round(
                tax,
                2
            )
    }


# ============================================================
# GOOGLE SHEETS
# ============================================================

def write_sheets(
    trades,
    equity,
    ranking,
    result
):

    sid = os.environ.get(
        SHEET_ID_ENV
    )

    cred = os.environ.get(
        CREDS_ENV
    )

    # ========================================================
    # CSV FALLBACK
    # ========================================================

    if not sid or not cred:

        trades.to_csv(
            "backtest_trades.csv",
            index=False
        )

        equity.to_csv(
            "backtest_equity.csv",
            index=False
        )

        ranking.to_csv(
            "backtest_daily_ranking.csv",
            index=False
        )

        pd.DataFrame(
            [result]
        ).to_csv(
            "backtest_summary.csv",
            index=False
        )

        print(
            "Google credentials missing."
            " CSV files written."
        )

        return

    # ========================================================
    # CONNECT
    # ========================================================

    credentials = (
        Credentials
        .from_service_account_info(
            json.loads(cred),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets"
            ]
        )
    )

    sh = (
        gspread
        .authorize(credentials)
        .open_by_key(sid)
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    try:

        ws = sh.worksheet(
            SUMMARY_SHEET
        )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(
            title=SUMMARY_SHEET,
            rows=100,
            cols=10
        )

    ws.clear()

    params = {

        "BACKTEST_START":
            BACKTEST_START,

        "BACKTEST_END":
            BACKTEST_END,

        "RUN_TIMESTAMP":
            result[
                "run_timestamp_ist"
            ],

        "BENCHMARK":
            BENCHMARK,

        "STARTING_CAPITAL":
            STARTING_CAPITAL,

        "TOP_N":
            TOP_N,

        "EXIT_RANK":
            EXIT_RANK,

        "MIN_PRICE":
            MIN_PRICE,

        "MIN_AVG_VOLUME":
            MIN_AVG_VOLUME,

        "VOLUME_LOOKBACK":
            VOLUME_LOOKBACK,

        "PRICE_TT":
            "7/7",

        "RS_LINE_TT":
            "7/7",

        "RS_SCORE":
            "40% 63D + 20% 126D + "
            "20% 189D + 20% 252D",

        "ENTRY":
            "Top 10 eligible stocks",

        "HOLD":
            "Rank 1-15",

        "EXIT":
            "Rank >15",

        "VCP":
            "NO",

        "VOLUME_DRYUP":
            "NO",

        "PIVOT":
            "NO",

        "BREAKOUT_VOLUME":
            "NO",

        "STOP_LOSS":
            "NO",

        "TRAILING_STOP":
            "NO",

        "RS_5EMA_EXIT":
            "NO",

        "BLUE_DOT_ENTRY":
            "NO",

        "GREEN_DOT_ENTRY":
            "NO",

        "REGIME_FILTER":
            "NO",

        "EXECUTION":
            "Daily EOD close",

        "CALCULATION":
            "Python",

        "SHEET_ROLE":
            "Output only",
    }

    rows = [
        [
            "PARAMETER",
            "VALUE"
        ]
    ]

    rows += [
        [
            key,
            value
        ]
        for key, value
        in params.items()
    ]

    rows += [
        [
            "",
            ""
        ]
    ]

    rows += [
        [
            "PERFORMANCE",
            "VALUE"
        ]
    ]

    rows += [
        [
            key,
            value
        ]
        for key, value
        in result.items()
    ]

    ws.update(
        rows,
        "A1"
    )

    # ========================================================
    # BACKTEST SHEET
    # ========================================================

    try:

        wb = sh.worksheet(
            BACKTEST_SHEET
        )

    except gspread.WorksheetNotFound:

        wb = sh.add_worksheet(
            title=BACKTEST_SHEET,
            rows=1000,
            cols=25
        )

    wb.clear()

    row = 1

    def put(
        title,
        df
    ):

        nonlocal row

        wb.update(
            [[title]],
            f"A{row}"
        )

        row += 1

        if not df.empty:

            values = [
                list(
                    df.columns
                )
            ] + (
                df.fillna("")
                .values
                .tolist()
            )

            # Resize if required
            required_rows = (
                row
                + len(values)
                + 5
            )

            required_cols = max(
                len(
                    df.columns
                ) + 2,
                10
            )

            if (
                wb.row_count
                < required_rows
                or
                wb.col_count
                < required_cols
            ):

                wb.resize(
                    rows=max(
                        wb.row_count,
                        required_rows
                    ),
                    cols=max(
                        wb.col_count,
                        required_cols
                    )
                )

            wb.update(
                values,
                f"A{row}"
            )

            row += (
                len(values)
                + 2
            )

        else:

            row += 2

    put(
        "TRADE LOG",
        trades
    )

    put(
        "DAILY EQUITY CURVE",
        equity
    )

    put(
        "DAILY ELIGIBLE RANKING",
        ranking
    )

    print(
        "Backtest Google Sheet updated."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    started = datetime.now(
        ZoneInfo(
            "Asia/Kolkata"
        )
    )

    print(
        "\n========================================"
    )

    print(
        "RS BACKTEST — FINAL SYNCHRONIZED MODEL"
    )

    print(
        "========================================"
    )

    print(
        "Start:",
        started.strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
    )

    # ========================================================
    # LOAD
    # ========================================================

    tickers = load_tickers()

    print(
        f"Stocks in CSV: "
        f"{len(tickers)}"
    )

    # ========================================================
    # EXTRA HISTORY
    # ========================================================

    download_start = (
        pd.Timestamp(
            BACKTEST_START
        )
        - pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    ).strftime(
        "%Y-%m-%d"
    )

    download_end = (
        pd.Timestamp(
            BACKTEST_END
        )
        + pd.Timedelta(
            days=1
        )
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"Download start: "
        f"{download_start}"
    )

    print(
        f"Download end: "
        f"{download_end}"
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    benchmark = get_benchmark(
        download_start,
        download_end
    )

    # ========================================================
    # STOCK DATA
    # ========================================================

    signals = build_signals(
        tickers,
        download_start,
        download_end,
        benchmark
    )

    # ========================================================
    # BACKTEST DAYS
    # ========================================================

    days = benchmark.index[
        (
            benchmark.index
            >= pd.Timestamp(
                BACKTEST_START
            )
        )
        &
        (
            benchmark.index
            <= pd.Timestamp(
                BACKTEST_END
            )
        )
    ]

    print(
        f"Usable stocks: "
        f"{len(signals)}"
    )

    print(
        f"Trading days: "
        f"{len(days)}"
    )

    print(
        f"Backtest end: "
        f"{BACKTEST_END}"
    )

    # ========================================================
    # RUN
    # ========================================================

    trades, equity, ranking = (
        run_backtest(
            signals,
            days
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    result = summary(
        trades,
        equity
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "========================================"
    )

    for key, value in (
        result.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "========================================"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    write_sheets(
        trades,
        equity,
        ranking,
        result
    )

    finished = datetime.now(
        ZoneInfo(
            "Asia/Kolkata"
        )
    )

    print(
        "\nBACKTEST COMPLETE:"
    )

    print(
        finished.strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()