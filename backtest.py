# RS SCREENER BACKTEST
# PYTHON CALCULATION -> GOOGLE SHEETS OUTPUT
# SYNCED WITH LIVE SCREENER

import os
import json
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials


# ============================================================
# PARAMETERS
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

BACKTEST_START = "2016-04-01"
BACKTEST_END = datetime.now(
    ZoneInfo("Asia/Kolkata")
).strftime("%Y-%m-%d")

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
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)


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

    exchange = EXCHANGE_CHARGE_RATE * value
    sebi = SEBI_CHARGE_RATE * value

    return (
        STT_BUY_RATE * value
        + STAMP_DUTY_RATE * value
        + exchange
        + sebi
        + GST_RATE * (exchange + sebi)
    )


def sell_cost(value):

    if not ENABLE_COSTS:
        return 0.0

    exchange = EXCHANGE_CHARGE_RATE * value
    sebi = SEBI_CHARGE_RATE * value

    return (
        STT_SELL_RATE * value
        + exchange
        + sebi
        + GST_RATE * (exchange + sebi)
        + DP_CHARGE_FLAT
    )


def stcg(gain):

    if not ENABLE_STCG:
        return 0.0

    return max(0.0, gain) * STCG_EFFECTIVE_RATE


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

def load_tickers():

    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(
            f"{STOCKS_FILE} not found"
        )

    df = pd.read_csv(STOCKS_FILE)

    if "symbol" not in df.columns:
        raise ValueError(
            "stocks.csv needs a 'symbol' column"
        )

    symbols = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return [
        x if x.endswith(".NS") else x + ".NS"
        for x in symbols
        if x
    ]


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template(s):

    sma50 = s.rolling(50).mean()
    sma150 = s.rolling(150).mean()
    sma200 = s.rolling(200).mean()

    low52 = s.rolling(252).min()
    high52 = s.rolling(252).max()

    conditions = [
        (s > sma150) & (s > sma200),
        sma150 > sma200,
        sma200 > sma200.shift(21),
        (sma50 > sma150) & (sma50 > sma200),
        s > sma50,
        s >= 1.25 * low52,
        s >= 0.75 * high52
    ]

    met = sum(
        condition.astype(int)
        for condition in conditions
    )

    return met == 7, met


# ============================================================
# STOCK SIGNALS
# ============================================================

def stock_signals(close, volume, benchmark):

    x = pd.concat(
        [close, benchmark],
        axis=1,
        join="inner"
    ).dropna()

    x.columns = ["s", "b"]

    if len(x) < 280:
        return None

    volume = (
        volume
        .reindex(x.index)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 50D VOLUME
    # --------------------------------------------------------

    avg_volume = (
        volume
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    # --------------------------------------------------------
    # PRICE TT
    # --------------------------------------------------------

    price_tt, price_tt_met = trend_template(
        x["s"]
    )

    # --------------------------------------------------------
    # RS LINE
    # --------------------------------------------------------

    rs_line = x["s"] / x["b"]

    rs_tt, rs_tt_met = trend_template(
        rs_line
    )

    # --------------------------------------------------------
    # RS SCORE
    # --------------------------------------------------------

    def ret(days):

        return (
            x["s"]
            / x["s"].shift(days)
            - 1
        )

    rs_score = (
        0.40 * ret(63)
        + 0.20 * ret(126)
        + 0.20 * ret(189)
        + 0.20 * ret(252)
    ) * 100

    # --------------------------------------------------------
    # LIVE-SCREENER IDENTICAL ENTRY FILTER
    # --------------------------------------------------------

    liquid = (
        (x["s"] >= MIN_PRICE)
        &
        (avg_volume >= MIN_AVG_VOLUME)
    )

    eligible = (
        price_tt
        &
        rs_tt
        &
        liquid
    )

    return pd.DataFrame({

        "price": x["s"],

        "rs_score": rs_score,

        "price_tt": price_tt,

        "price_tt_met": price_tt_met,

        "rs_tt": rs_tt,

        "rs_tt_met": rs_tt_met,

        "avg_volume": avg_volume,

        "liquid": liquid,

        "eligible": eligible

    })


# ============================================================
# BENCHMARK DOWNLOAD
# ============================================================

def download_benchmark(start, end):

    # Yahoo end date is exclusive.
    yahoo_end = (
        pd.Timestamp(end)
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    for ticker in [
        BENCHMARK,
        BENCHMARK_FALLBACK
    ]:

        try:

            data = yf.download(
                ticker,
                start=start,
                end=yahoo_end,
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

            close = (
                close
                .dropna()
                .sort_index()
            )

            if not close.empty:
                print(
                    f"Benchmark: {ticker}"
                )
                return close

        except Exception as e:

            print(
                f"Benchmark {ticker} failed: {e}"
            )

    raise RuntimeError(
        "Benchmark download failed"
    )


# ============================================================
# STOCK DOWNLOAD + SIGNAL BUILD
# ============================================================

def build_signals(
    tickers,
    start,
    end,
    benchmark
):

    signals = {}

    batch_size = 50

    yahoo_end = (
        pd.Timestamp(end)
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

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
            f"{i + 1}-{i + len(batch)} "
            f"/ {len(tickers)}"
        )

        try:

            data = yf.download(
                batch,
                start=start,
                end=yahoo_end,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )

        except Exception as e:

            print(
                f"Batch failed: {e}"
            )

            continue

        for symbol in batch:

            try:

                if len(batch) == 1:

                    stock = data

                else:

                    if (
                        symbol
                        not in
                        data.columns.get_level_values(0)
                    ):
                        continue

                    stock = data[symbol]

                if (
                    stock.empty
                    or
                    "Close" not in stock.columns
                ):
                    continue

                close = (
                    stock["Close"]
                    .dropna()
                    .sort_index()
                )

                volume = (
                    stock["Volume"]
                    .reindex(close.index)
                    .fillna(0)
                )

                if close.empty:
                    continue

                result = stock_signals(
                    close,
                    volume,
                    benchmark
                )

                if result is not None:

                    name = symbol.replace(
                        ".NS",
                        ""
                    )

                    signals[name] = result

            except Exception as e:

                print(
                    f"Skipping {symbol}: {e}"
                )

        time.sleep(0.5)

    print(
        f"Signals built: "
        f"{len(signals)} stocks"
    )

    return signals


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    signals,
    trading_days
):

    cash = float(
        STARTING_CAPITAL
    )

    holdings = {}

    trades = []
    equity = []
    ranking_log = []

    last_rank = {}

    for date in trading_days:

        # ====================================================
        # ELIGIBLE UNIVERSE
        # EXACT LIVE SCREENER RULE
        # ====================================================

        candidates = []

        for symbol, df in signals.items():

            if date not in df.index:
                continue

            row = df.loc[date]

            if pd.isna(
                row["rs_score"]
            ):
                continue

            if not bool(
                row["eligible"]
            ):
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

        rank = {
            symbol: i
            for i, (symbol, _) in enumerate(
                candidates,
                start=1
            )
        }

        last_rank = rank.copy()

        top10 = [
            symbol
            for symbol, _
            in candidates[:TOP_N]
        ]

        # ====================================================
        # DAILY RANKING LOG
        # ====================================================

        for i, (
            symbol,
            score
        ) in enumerate(
            candidates,
            start=1
        ):

            row = signals[
                symbol
            ].loc[date]

            if i <= TOP_N:
                action = "BUY ENTRY"
            elif i <= EXIT_RANK:
                action = "HOLD ALLOWED"
            else:
                action = "WATCHLIST ONLY"

            ranking_log.append({

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "rank":
                    i,

                "symbol":
                    symbol,

                "action":
                    action,

                "rs_score":
                    round(
                        score,
                        4
                    ),

                "price":
                    round(
                        float(
                            row["price"]
                        ),
                        2
                    ),

                "price_tt_met":
                    int(
                        row["price_tt_met"]
                    ),

                "rs_tt_met":
                    int(
                        row["rs_tt_met"]
                    ),

                "avg_volume_50d":
                    round(
                        float(
                            row["avg_volume"]
                        ),
                        0
                    )
            })

        # ====================================================
        # EXITS
        #
        # IMPORTANT:
        # Exit rank is based on the SAME eligible ranking
        # used for live screener.
        # ====================================================

        for symbol in list(
            holdings.keys()
        ):

            if date not in signals[
                symbol
            ].index:
                continue

            row = signals[
                symbol
            ].loc[date]

            current_rank = rank.get(
                symbol,
                999999
            )

            if current_rank <= EXIT_RANK:
                continue

            position = holdings.pop(
                symbol
            )

            price = float(
                row["price"]
            )

            gross = (
                position["qty"]
                * price
            )

            sell_fee = sell_cost(
                gross
            )

            proceeds = (
                gross
                - sell_fee
            )

            basis = (
                position["qty"]
                * position["entry_price"]
                + position["buy_cost"]
            )

            gain = (
                proceeds
                - basis
            )

            tax = stcg(gain)

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
                        price,
                        2
                    ),

                "entry_rank":
                    position[
                        "entry_rank"
                    ],

                "exit_rank":
                    current_rank,

                "entry_rs_score":
                    position[
                        "entry_score"
                    ],

                "exit_rs_score":
                    round(
                        float(
                            row["rs_score"]
                        ),
                        4
                    ),

                "gross_return_pct":
                    round(
                        (
                            price
                            /
                            position[
                                "entry_price"
                            ]
                            - 1
                        ) * 100,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        position[
                            "buy_cost"
                        ],
                        2
                    ),

                "sell_cost_rs":
                    round(
                        sell_fee,
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
                        -
                        position[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "RS RANK > 15"
            })

        # ====================================================
        # VALUE AFTER EXITS
        # ====================================================

        portfolio_value = cash

        for symbol, position in (
            holdings.items()
        ):

            if date in signals[
                symbol
            ].index:

                price = float(
                    signals[
                        symbol
                    ].loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = position[
                    "entry_price"
                ]

            portfolio_value += (
                position["qty"]
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

            target_value = (
                portfolio_value
                / TOP_N
            )

            for symbol in top10:

                if slots <= 0:
                    break

                if symbol in holdings:
                    continue

                row = signals[
                    symbol
                ].loc[date]

                price = float(
                    row["price"]
                )

                if price <= 0:
                    continue

                qty = int(
                    target_value
                    // price
                )

                if qty < 1:
                    continue

                trade_value = (
                    qty * price
                )

                buy_fee = buy_cost(
                    trade_value
                )

                required = (
                    trade_value
                    + buy_fee
                )

                if required > cash:
                    continue

                cash -= required

                holdings[symbol] = {

                    "qty":
                        qty,

                    "entry_price":
                        price,

                    "entry_date":
                        date,

                    "buy_cost":
                        buy_fee,

                    "entry_rank":
                        rank[symbol],

                    "entry_score":
                        round(
                            float(
                                row["rs_score"]
                            ),
                            4
                        )
                }

                slots -= 1

        # ====================================================
        # FINAL DAILY EQUITY
        # ====================================================

        portfolio_value = cash

        for symbol, position in (
            holdings.items()
        ):

            if date in signals[
                symbol
            ].index:

                price = float(
                    signals[
                        symbol
                    ].loc[
                        date,
                        "price"
                    ]
                )

            else:

                price = position[
                    "entry_price"
                ]

            portfolio_value += (
                position["qty"]
                * price
            )

        equity.append({

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
                    portfolio_value
                    / STARTING_CAPITAL,
                    8
                ),

            "return_pct":
                round(
                    (
                        portfolio_value
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
                ",".join(top10),

            "holdings":
                ",".join(
                    sorted(
                        holdings.keys()
                    )
                )
        })

    # ========================================================
    # BACKTEST-END OPEN POSITIONS
    # ========================================================

    if trading_days:

        last_date = trading_days[-1]

        final_rank = last_rank

        for symbol, position in (
            list(holdings.items())
        ):

            if (
                symbol not in signals
                or
                last_date
                not in signals[
                    symbol
                ].index
            ):
                continue

            row = signals[
                symbol
            ].loc[last_date]

            price = float(
                row["price"]
            )

            gross = (
                position["qty"]
                * price
            )

            sell_fee = sell_cost(
                gross
            )

            proceeds = (
                gross
                - sell_fee
            )

            basis = (
                position["qty"]
                * position["entry_price"]
                + position["buy_cost"]
            )

            gain = (
                proceeds
                - basis
            )

            tax = stcg(gain)

            pnl = (
                gain
                - tax
            )

            trades.append({

                "symbol":
                    symbol,

                "entry_date":
                    position[
                        "entry_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "exit_date":
                    last_date.strftime(
                        "%Y-%m-%d"
                    ) + " OPEN",

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
                        price,
                        2
                    ),

                "entry_rank":
                    position[
                        "entry_rank"
                    ],

                "exit_rank":
                    final_rank.get(
                        symbol,
                        ""
                    ),

                "entry_rs_score":
                    position[
                        "entry_score"
                    ],

                "exit_rs_score":
                    round(
                        float(
                            row["rs_score"]
                        ),
                        4
                    ),

                "gross_return_pct":
                    round(
                        (
                            price
                            /
                            position[
                                "entry_price"
                            ]
                            - 1
                        ) * 100,
                        2
                    ),

                "buy_cost_rs":
                    round(
                        position[
                            "buy_cost"
                        ],
                        2
                    ),

                "sell_cost_rs":
                    round(
                        sell_fee,
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
                        last_date
                        -
                        position[
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

def make_summary(
    trades,
    equity,
    run_timestamp
):

    if equity.empty:
        return {}

    final_value = float(
        equity[
            "portfolio_value_rs"
        ].iloc[-1]
    )

    total_return = (
        final_value
        / STARTING_CAPITAL
        - 1
    ) * 100

    peak = (
        equity["equity"]
        .cummax()
    )

    drawdown = (
        equity["equity"]
        / peak
        - 1
    ) * 100

    max_dd = float(
        drawdown.min()
    )

    if trades.empty:

        closed = trades

    else:

        closed = trades[
            ~trades[
                "exit_date"
            ]
            .astype(str)
            .str.contains(
                "OPEN",
                na=False
            )
        ]

    n = len(closed)

    if n:

        win_rate = (
            (
                closed[
                    "net_return_pct"
                ] > 0
            ).mean()
            * 100
        )

        avg_return = (
            closed[
                "net_return_pct"
            ].mean()
        )

        median_return = (
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

        transaction_costs = (
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

            profit_factor = (
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

            profit_factor = 0

    else:

        win_rate = 0
        avg_return = 0
        median_return = 0
        avg_days = 0
        best = 0
        worst = 0
        transaction_costs = 0
        tax = 0
        profit_factor = 0

    daily_returns = (
        equity["equity"]
        .pct_change()
        .dropna()
    )

    if len(equity) > 1:

        annualized_return = (
            equity[
                "equity"
            ].iloc[-1]
            **
            (
                252
                /
                len(equity)
            )
            - 1
        )

    else:

        annualized_return = 0

    if len(daily_returns):

        volatility = (
            daily_returns.std()
            * np.sqrt(252)
        )

    else:

        volatility = 0

    if (
        len(daily_returns)
        and
        daily_returns.std() > 0
    ):

        sharpe = (
            daily_returns.mean()
            /
            daily_returns.std()
            *
            np.sqrt(252)
        )

    else:

        sharpe = 0

    return {

        "run_timestamp_ist":
            run_timestamp,

        "backtest_start":
            BACKTEST_START,

        "backtest_end":
            BACKTEST_END,

        "starting_capital_rs":
            STARTING_CAPITAL,

        "final_portfolio_value_rs":
            round(
                final_value,
                2
            ),

        "net_total_return_pct":
            round(
                total_return,
                2
            ),

        "annualized_return_pct":
            round(
                annualized_return * 100,
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
                win_rate,
                1
            ),

        "avg_net_return_trade_pct":
            round(
                avg_return,
                2
            ),

        "median_net_return_trade_pct":
            round(
                median_return,
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
                profit_factor,
                3
            ),

        "transaction_costs_rs":
            round(
                transaction_costs,
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

def get_worksheet(
    spreadsheet,
    name,
    rows=100,
    cols=20
):

    try:

        return spreadsheet.worksheet(name)

    except gspread.WorksheetNotFound:

        return spreadsheet.add_worksheet(
            title=name,
            rows=rows,
            cols=cols
        )


def resize_sheet(
    worksheet,
    rows,
    cols
):

    rows = max(
        int(rows),
        100
    )

    cols = max(
        int(cols),
        20
    )

    if (
        worksheet.row_count < rows
        or
        worksheet.col_count < cols
    ):

        worksheet.resize(
            rows=max(
                worksheet.row_count,
                rows
            ),
            cols=max(
                worksheet.col_count,
                cols
            )
        )


def write_dataframe(
    worksheet,
    start_row,
    dataframe,
    chunk_size=2000
):

    if dataframe.empty:
        return start_row

    values = [
        list(dataframe.columns)
    ] + dataframe.fillna("").values.tolist()

    total_rows = len(values)
    total_cols = len(
        values[0]
    )

    resize_sheet(
        worksheet,
        start_row + total_rows + 5,
        total_cols + 2
    )

    for i in range(
        0,
        total_rows,
        chunk_size
    ):

        chunk = values[
            i:i + chunk_size
        ]

        worksheet.update(
            chunk,
            f"A{start_row + i}"
        )

        time.sleep(0.1)

    return (
        start_row
        + total_rows
    )


def write_sheets(
    trades,
    equity,
    ranking,
    result
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    credentials_json = os.environ.get(
        CREDS_ENV
    )

    if (
        not sheet_id
        or
        not credentials_json
    ):

        print(
            "Google credentials missing. "
            "Saving CSV."
        )

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

        return

    credentials = (
        Credentials
        .from_service_account_info(
            json.loads(
                credentials_json
            ),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets"
            ]
        )
    )

    spreadsheet = (
        gspread
        .authorize(credentials)
        .open_by_key(sheet_id)
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_ws = get_worksheet(
        spreadsheet,
        SUMMARY_SHEET,
        rows=100,
        cols=10
    )

    summary_ws.clear()

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
            "40% 63D + 20% 126D + 20% 189D + 20% 252D",

        "ENTRY":
            "Top 10 eligible stocks",

        "HOLD":
            "Rank 1-15",

        "EXIT":
            "Rank >15",

        "BLUE_DOT":
            "NO",

        "GREEN_DOT":
            "NO",

        "VCP":
            "NO",

        "STOP_LOSS":
            "NO",

        "TRAILING_STOP":
            "NO",

        "RS_20EMA_EXIT":
            "NO",

        "EXECUTION":
            "Daily EOD close",

        "CALCULATION":
            "Python",

        "SHEET_ROLE":
            "Output only"
    }

    summary_rows = [
        ["PARAMETER", "VALUE"]
    ]

    summary_rows += [
        [key, value]
        for key, value in params.items()
    ]

    summary_rows += [
        ["", ""],
        ["PERFORMANCE", "VALUE"]
    ]

    summary_rows += [
        [key, value]
        for key, value in result.items()
    ]

    resize_sheet(
        summary_ws,
        len(summary_rows) + 5,
        4
    )

    summary_ws.update(
        summary_rows,
        "A1"
    )

    # ========================================================
    # BACKTEST DATA
    # ========================================================

    backtest_ws = get_worksheet(
        spreadsheet,
        BACKTEST_SHEET,
        rows=1000,
        cols=20
    )

    backtest_ws.clear()

    row = 1

    backtest_ws.update(
        [["TRADE LOG"]],
        f"A{row}"
    )

    row += 1

    row = write_dataframe(
        backtest_ws,
        row,
        trades
    )

    row += 2

    backtest_ws.update(
        [["DAILY EQUITY CURVE"]],
        f"A{row}"
    )

    row += 1

    row = write_dataframe(
        backtest_ws,
        row,
        equity
    )

    row += 2

    backtest_ws.update(
        [["DAILY ELIGIBLE RANKING"]],
        f"A{row}"
    )

    row += 1

    write_dataframe(
        backtest_ws,
        row,
        ranking
    )

    print(
        "Google Sheets updated."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    timestamp = start_time.strftime(
        "%Y-%m-%d %H:%M:%S IST"
    )

    print(
        f"\nBACKTEST START: {timestamp}"
    )

    print(
        f"BACKTEST END: {BACKTEST_END}"
    )

    tickers = load_tickers()

    download_start = (
        pd.Timestamp(
            BACKTEST_START
        )
        -
        pd.DateOffset(
            years=DOWNLOAD_YEARS_BEFORE_START
        )
    ).strftime("%Y-%m-%d")

    # Yahoo end is exclusive.
    download_end = (
        pd.Timestamp(
            BACKTEST_END
        )
        +
        pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    print(
        f"Universe: {len(tickers)}"
    )

    print(
        f"Download: "
        f"{download_start} -> "
        f"{download_end}"
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    benchmark = download_benchmark(
        download_start,
        BACKTEST_END
    )

    # ========================================================
    # STOCK SIGNALS
    # ========================================================

    signals = build_signals(
        tickers,
        download_start,
        BACKTEST_END,
        benchmark
    )

    # ========================================================
    # TRADING DAYS
    # ========================================================

    trading_days = benchmark.index[
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

    # If today's EOD data is not available,
    # Yahoo's latest available trading day is used.
    if len(trading_days):

        actual_last_day = (
            trading_days[-1]
            .strftime("%Y-%m-%d")
        )

    else:

        raise RuntimeError(
            "No trading days available."
        )

    print(
        f"Trading days: "
        f"{len(trading_days)}"
    )

    print(
        f"Actual data end: "
        f"{actual_last_day}"
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    trades, equity, ranking = run_backtest(
        signals,
        trading_days
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    result = make_summary(
        trades,
        equity,
        timestamp
    )

    print(
        "\nFINAL RESULTS"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # GOOGLE SHEETS
    # ========================================================

    write_sheets(
        trades,
        equity,
        ranking,
        result
    )

    finish_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    print(
        f"\nBACKTEST COMPLETE: "
        f"{finish_time.strftime('%Y-%m-%d %H:%M:%S IST')}"
    )


# ============================================================
# RUN
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

        traceback.print_exc()

        raise