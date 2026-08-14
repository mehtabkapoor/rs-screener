"""
RS SCREENER — FINAL SYNCHRONIZED VERSION

RULES
=====

PRICE
-----
Price >= Rs.10

LIQUIDITY
---------
50-day average volume >= 50,000 shares

PRICE TREND TEMPLATE
--------------------
7/7 Minervini Trend Template

RS SCORE
--------
40% 3-month return
20% 6-month return
20% 9-month return
20% 12-month return

RS LINE TREND TEMPLATE
----------------------
7/7 Minervini Trend Template
applied to Stock / Benchmark

RANKING
-------
Only stocks passing:

    Price TT 7/7
    RS Line TT 7/7
    Liquidity

are ranked.

Rank 1 = highest raw RS Score.

PORTFOLIO
---------
Top 10 eligible stocks
Equal weight

HOLD
----
Rank 1-15

EXIT
----
Rank >15

NO
--
VCP
Contractions
Volume dry-up
Pivot
Breakout-volume requirement
Blue Dot entry
Green Dot entry
RS 5EMA exit
Price stop
Trailing stop
Regime filter
"""

import os
import json
import time
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

HISTORY_PERIOD = "15mo"

STOCKS_FILE = "stocks.csv"

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

WORKSHEET_NAME = "RS_Screener"

# ----------------------------
# CORE SCREEN
# ----------------------------

MIN_PRICE = 10

MIN_AVG_VOLUME = 50_000
VOLUME_LOOKBACK = 50

TOP_N = 10
EXIT_RANK = 15

# ----------------------------
# PREVIEW
# ----------------------------

INTRADAY_INTERVAL = "5m"

EOD_CRON = "15 11 * * 1-5"

# ----------------------------
# COSTS
# ----------------------------

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20

STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

# ----------------------------
# PORTFOLIO SHEETS
# ----------------------------

HOLDINGS_WORKSHEET = "Holdings"
PORTFOLIO_WORKSHEET = "Portfolio"
CONFIG_WORKSHEET = "Config"

PORTFOLIO_HEADER = [
    "Action",
    "Executed",
    "Execution Price",
    "Symbol",
    "Rank",
    "Entry Price",
    "Entry Date",
    "Current Price",
    "Qty",
    "Position Value (Rs)",
    "P&L %",
    "Buy Cost (Rs)",
    "Sell Cost (Rs)",
    "Est. STCG Tax (Rs)",
]


# ============================================================
# RUN MODE
# ============================================================

def get_run_mode():

    event = os.environ.get(
        "GITHUB_EVENT_NAME",
        "manual"
    )

    force_eod = (
        os.environ.get(
            "FORCE_EOD",
            "false"
        ).strip().lower() == "true"
    )

    if event == "schedule":

        triggering_cron = os.environ.get(
            "SCHEDULE_CRON",
            ""
        ).strip()

        return (
            "EOD"
            if triggering_cron == EOD_CRON
            else "PREVIEW"
        )

    return "EOD" if force_eod else "PREVIEW"


# ============================================================
# LOAD UNIVERSE
# ============================================================

def load_tickers():

    df = pd.read_csv(STOCKS_FILE)

    if "symbol" not in df.columns:
        raise ValueError(
            "stocks.csv must contain a 'symbol' column"
        )

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
        if s
    ]


# ============================================================
# COSTS
# ============================================================

def buy_side_cost(trade_value):

    stt = STT_RATE * trade_value
    stamp = STAMP_DUTY_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)

    return stt + stamp + exch + sebi + gst


def sell_side_cost(trade_value):

    stt = STT_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)

    return (
        stt
        + exch
        + sebi
        + gst
        + DP_CHARGE_FLAT
    )


def estimate_stcg(gross_pnl):

    if gross_pnl <= 0:
        return 0.0

    return gross_pnl * STCG_EFFECTIVE_RATE


# ============================================================
# BENCHMARK
# ============================================================

def download_benchmark(run_mode):

    for ticker in (
        BENCHMARK,
        BENCHMARK_FALLBACK
    ):

        try:

            data = yf.download(
                ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                continue

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna().sort_index()

            print(
                f"Benchmark loaded: {ticker}"
            )

            if run_mode == "PREVIEW":

                live = fetch_intraday_last_price(
                    [ticker]
                )

                close = append_preview_price(
                    close,
                    live.get(ticker)
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
# INTRADAY
# ============================================================

def fetch_intraday_last_price(tickers):

    prices = {}

    try:

        data = yf.download(
            tickers,
            period="1d",
            interval=INTRADAY_INTERVAL,
            progress=False,
            group_by="ticker",
            threads=True
        )

        for ticker in tickers:

            try:

                if len(tickers) == 1:
                    sdata = data
                else:
                    sdata = data[ticker]

                last_valid = (
                    sdata["Close"]
                    .dropna()
                )

                if not last_valid.empty:
                    prices[ticker] = float(
                        last_valid.iloc[-1]
                    )

            except Exception:
                continue

    except Exception as e:

        print(
            f"Intraday fetch failed: {e}"
        )

    return prices


def append_preview_price(
    close_series,
    live_price
):

    if live_price is None:
        return close_series

    if close_series.empty:
        return close_series

    if close_series.index.tz:

        today = (
            pd.Timestamp.now(
                tz=close_series.index.tz
            ).normalize()
        )

    else:

        today = (
            pd.Timestamp.now()
            .normalize()
        )

    last_date = (
        close_series.index[-1]
        .normalize()
    )

    if last_date == today:

        updated = close_series.copy()

        updated.iloc[-1] = live_price

        return updated

    new_point = pd.Series(
        [live_price],
        index=[today]
    )

    return pd.concat(
        [close_series, new_point]
    )


# ============================================================
# RS SCORE
# ============================================================

def compute_rs_score(close_series):

    periods = {
        "P3": 63,
        "P6": 126,
        "P9": 189,
        "P12": 252,
    }

    if len(close_series) <= 252:
        return None

    returns = {}

    latest = close_series.iloc[-1]

    for label, days in periods.items():

        past = close_series.iloc[-days - 1]

        if (
            pd.isna(past)
            or past == 0
        ):
            return None

        returns[label] = (
            latest / past
        ) - 1

    score = (
        0.40 * returns["P3"]
        + 0.20 * returns["P6"]
        + 0.20 * returns["P9"]
        + 0.20 * returns["P12"]
    )

    return round(
        score * 100,
        4
    )


# ============================================================
# TREND TEMPLATE
# ============================================================

def compute_trend_template(series):

    if len(series) < 273:
        return None

    sma50 = series.rolling(50).mean()
    sma150 = series.rolling(150).mean()
    sma200 = series.rolling(200).mean()

    price = series.iloc[-1]

    sma50_now = sma50.iloc[-1]
    sma150_now = sma150.iloc[-1]
    sma200_now = sma200.iloc[-1]

    sma200_1mo = sma200.iloc[-21]

    low52 = (
        series.tail(252).min()
    )

    high52 = (
        series.tail(252).max()
    )

    values = [
        price,
        sma50_now,
        sma150_now,
        sma200_now,
        sma200_1mo,
        low52,
        high52,
    ]

    if any(pd.isna(x) for x in values):
        return None

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

    return bool(all(criteria))


# ============================================================
# RS LINE TREND TEMPLATE
# ============================================================

def compute_rs_line_tt(
    stock_close,
    benchmark_close
):

    aligned = pd.concat(
        [
            stock_close,
            benchmark_close
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(aligned) < 273:
        return None

    aligned.columns = [
        "stock",
        "benchmark"
    ]

    rs_line = (
        aligned["stock"]
        / aligned["benchmark"]
    )

    return compute_trend_template(
        rs_line
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def compute_diagnostics(
    stock_close,
    benchmark_close
):

    df = pd.concat(
        [
            stock_close,
            benchmark_close
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(df) < 252 + 2:
        return None

    df.columns = [
        "stock",
        "benchmark"
    ]

    df["rs_line"] = (
        df["stock"]
        / df["benchmark"]
    )

    rs_previous_high = (
        df["rs_line"]
        .shift(1)
        .rolling(252)
        .max()
    )

    price_previous_high = (
        df["stock"]
        .shift(1)
        .rolling(252)
        .max()
    )

    today = df.iloc[-1]

    blue_dot = bool(
        today["rs_line"]
        > rs_previous_high.iloc[-1]
    ) if pd.notna(
        rs_previous_high.iloc[-1]
    ) else False

    price_new_high = bool(
        today["stock"]
        > price_previous_high.iloc[-1]
    ) if pd.notna(
        price_previous_high.iloc[-1]
    ) else False

    green_dot = (
        blue_dot
        and not price_new_high
    )

    return (
        blue_dot,
        green_dot
    )


# ============================================================
# MAIN SCREEN
# ============================================================

def main():

    tickers = load_tickers()

    print(
        f"Loaded {len(tickers)} tickers."
    )

    run_mode = get_run_mode()

    print(
        f"Run mode: {run_mode}"
    )

    benchmark = download_benchmark(
        run_mode
    )

    all_stocks = []

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
            f"{i + 1}-{i + len(batch)}..."
        )

        try:

            data = yf.download(
                batch,
                period=HISTORY_PERIOD,
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

        intraday_prices = {}

        if run_mode == "PREVIEW":

            intraday_prices = (
                fetch_intraday_last_price(
                    batch
                )
            )

        for symbol in batch:

            try:

                if len(batch) == 1:
                    sdata = data
                else:
                    sdata = data[symbol]

                if (
                    "Close" not in sdata.columns
                    or "Volume" not in sdata.columns
                ):
                    continue

                close = (
                    sdata["Close"]
                    .dropna()
                    .sort_index()
                )

                volume = (
                    sdata["Volume"]
                    .dropna()
                    .sort_index()
                )

                if len(close) < 273:
                    continue

                if run_mode == "PREVIEW":

                    close = append_preview_price(
                        close,
                        intraday_prices.get(
                            symbol
                        )
                    )

                last_price = float(
                    close.iloc[-1]
                )

                # ------------------------
                # PRICE
                # ------------------------

                if last_price < MIN_PRICE:
                    continue

                # ------------------------
                # LIQUIDITY
                # ------------------------

                aligned_volume = (
                    volume
                    .reindex(close.index)
                    .fillna(0)
                )

                avg50_volume = (
                    aligned_volume
                    .rolling(VOLUME_LOOKBACK)
                    .mean()
                    .iloc[-1]
                )

                if pd.isna(avg50_volume):
                    continue

                liquid = (
                    avg50_volume
                    >= MIN_AVG_VOLUME
                )

                # ------------------------
                # RS SCORE
                # ------------------------

                rs_score = compute_rs_score(
                    close
                )

                if rs_score is None:
                    continue

                # ------------------------
                # PRICE TT
                # ------------------------

                price_tt = (
                    compute_trend_template(
                        close
                    )
                )

                # ------------------------
                # RS LINE TT
                # ------------------------

                rs_tt = compute_rs_line_tt(
                    close,
                    benchmark
                )

                if (
                    price_tt is None
                    or rs_tt is None
                ):
                    continue

                # ------------------------
                # DIAGNOSTICS
                # ------------------------

                diagnostics = compute_diagnostics(
                    close,
                    benchmark
                )

                if diagnostics is None:
                    continue

                blue_dot, green_dot = (
                    diagnostics
                )

                # ------------------------
                # ELIGIBILITY
                # ------------------------

                eligible = (
                    liquid
                    and price_tt
                    and rs_tt
                )

                all_stocks.append({

                    "symbol":
                        symbol.replace(".NS", ""),

                    "rs_score":
                        rs_score,

                    "last_close":
                        round(last_price, 2),

                    "avg_volume_50d":
                        round(
                            float(avg50_volume),
                            0
                        ),

                    "price_tt":
                        bool(price_tt),

                    "rs_line_tt":
                        bool(rs_tt),

                    "liquid":
                        bool(liquid),

                    "eligible":
                        bool(eligible),

                    "blue_dot":
                        bool(blue_dot),

                    "one_year_rs_cross":
                        bool(blue_dot),

                    "green_dot":
                        bool(green_dot),
                })

            except Exception as e:

                print(
                    f"Skipping {symbol}: {e}"
                )

                continue

        time.sleep(0.5)

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not all_stocks:

        print(
            "No stocks with sufficient data."
        )

        write_to_sheet(
            pd.DataFrame(),
            run_mode
        )

        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    universe_df = pd.DataFrame(
        all_stocks
    )

    # ========================================================
    # RANK ONLY ELIGIBLE STOCKS
    # ========================================================

    eligible_df = universe_df[
        universe_df["eligible"] == True
    ].copy()

    eligible_df = (
        eligible_df
        .sort_values(
            "rs_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    eligible_df["rank"] = (
        range(
            1,
            len(eligible_df) + 1
        )
    )

    rank_lookup = dict(
        zip(
            eligible_df["symbol"],
            eligible_df["rank"]
        )
    )

    universe_df["rank"] = (
        universe_df["symbol"]
        .map(rank_lookup)
    )

    # ========================================================
    # OUTPUT LABELS
    # ========================================================

    universe_df["price_trend_template"] = (
        universe_df["price_tt"]
        .map({
            True: "PASS",
            False: "FAIL"
        })
    )

    universe_df["rs_trend_template"] = (
        universe_df["rs_line_tt"]
        .map({
            True: "PASS",
            False: "FAIL"
        })
    )

    universe_df["liquidity"] = (
        universe_df["liquid"]
        .map({
            True: "PASS",
            False: "FAIL"
        })
    )

    universe_df["screen"] = (
        universe_df["eligible"]
        .map({
            True: "PASS",
            False: "FAIL"
        })
    )

    universe_df["action"] = ""

    universe_df.loc[
        universe_df["rank"].between(
            1,
            TOP_N
        ),
        "action"
    ] = "BUY / HOLD"

    universe_df.loc[
        universe_df["rank"].between(
            TOP_N + 1,
            EXIT_RANK
        ),
        "action"
    ] = "HOLD ALLOWED"

    universe_df.loc[
        universe_df["rank"] > EXIT_RANK,
        "action"
    ] = "EXIT"

    universe_df["blue_dot"] = (
        universe_df["blue_dot"]
        .map({
            True: "YES",
            False: ""
        })
    )

    universe_df["one_year_rs_cross"] = (
        universe_df["one_year_rs_cross"]
        .map({
            True: "YES",
            False: ""
        })
    )

    universe_df["green_dot"] = (
        universe_df["green_dot"]
        .map({
            True: "YES",
            False: ""
        })
    )

    # ========================================================
    # SORT
    # ========================================================

    results_df = (
        universe_df
        .sort_values(
            "rank",
            na_position="last"
        )
        .reset_index(drop=True)
    )

    results_df = results_df[
        [
            "rank",
            "symbol",
            "rs_score",
            "last_close",
            "avg_volume_50d",
            "price_trend_template",
            "rs_trend_template",
            "liquidity",
            "screen",
            "action",
            "blue_dot",
            "one_year_rs_cross",
            "green_dot",
        ]
    ]

    # ========================================================
    # AUDIT COUNTS
    # ========================================================

    n_universe = len(
        universe_df
    )

    n_price_tt = (
        universe_df["price_tt"]
        == True
    ).sum()

    n_rs_tt = (
        universe_df["rs_line_tt"]
        == True
    ).sum()

    n_liquid = (
        universe_df["liquid"]
        == True
    ).sum()

    n_eligible = (
        universe_df["eligible"]
        == True
    ).sum()

    print(
        "========================================"
    )

    print(
        f"Universe scanned: {n_universe}"
    )

    print(
        f"Price TT PASS: {n_price_tt}"
    )

    print(
        f"RS Line TT PASS: {n_rs_tt}"
    )

    print(
        f"Liquidity PASS: {n_liquid}"
    )

    print(
        f"FINAL ELIGIBLE: {n_eligible}"
    )

    print(
        f"TOP {TOP_N}: "
        f"{', '.join("
        + repr(
            eligible_df.head(TOP_N)["symbol"].tolist()
        )
        + ")}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # SHEET
    # ========================================================

    write_to_sheet(
        results_df,
        run_mode
    )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    if run_mode == "PREVIEW":

        print(
            "Preview mode: "
            "Portfolio untouched."
        )

        return

    build_portfolio(
        universe_df
    )


# ============================================================
# CONFIG
# ============================================================

def read_config(sh):

    try:

        cfg_ws = sh.worksheet(
            CONFIG_WORKSHEET
        )

        records = (
            cfg_ws.get_all_records()
        )

        settings = {
            row["Setting"]:
                row["Value"]
            for row in records
            if row.get("Setting")
        }

    except gspread.WorksheetNotFound:

        cfg_ws = sh.add_worksheet(
            title=CONFIG_WORKSHEET,
            rows=10,
            cols=3
        )

        cfg_ws.update(
            [
                [
                    "Setting",
                    "Value",
                    "Notes"
                ],
                [
                    "Total Capital (INR)",
                    0,
                    "EDIT ME"
                ]
            ],
            "A1"
        )

        settings = {
            "Total Capital (INR)": 0
        }

    try:

        capital = float(
            settings.get(
                "Total Capital (INR)",
                0
            )
            or 0
        )

    except (
        ValueError,
        TypeError
    ):

        capital = 0

    return capital


# ============================================================
# APPLY CONFIRMED EXECUTIONS
# ============================================================

def apply_confirmed_executions(sh):

    try:

        port_ws = sh.worksheet(
            PORTFOLIO_WORKSHEET
        )

    except gspread.WorksheetNotFound:

        return

    try:

        prior_rows = (
            port_ws.get_all_records(
                head=3
            )
        )

    except Exception as e:

        print(
            f"Could not read Portfolio: {e}"
        )

        return

    try:

        holdings_ws = sh.worksheet(
            HOLDINGS_WORKSHEET
        )

        existing = (
            holdings_ws.get_all_records()
        )

        holdings = {
            row["symbol"]: {
                "entry_price":
                    float(
                        row.get(
                            "entry_price"
                        ) or 0
                    ),
                "entry_date":
                    row.get(
                        "entry_date",
                        ""
                    )
            }
            for row in existing
            if row.get("symbol")
        }

    except gspread.WorksheetNotFound:

        holdings_ws = sh.add_worksheet(
            title=HOLDINGS_WORKSHEET,
            rows=100,
            cols=5
        )

        holdings_ws.update(
            [
                [
                    "symbol",
                    "entry_price",
                    "entry_date"
                ]
            ],
            "A1"
        )

        holdings = {}

    today_str = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    changed = False

    for row in prior_rows:

        executed = str(
            row.get(
                "Executed",
                ""
            )
        ).strip().upper()

        if executed not in (
            "Y",
            "YES"
        ):
            continue

        action = str(
            row.get(
                "Action",
                ""
            )
        ).strip().upper()

        symbol = str(
            row.get(
                "Symbol",
                ""
            )
        ).strip()

        if not symbol:
            continue

        if action == "BUY":

            exec_price_raw = (
                row.get(
                    "Execution Price"
                )
                or row.get(
                    "Entry Price"
                )
            )

            try:

                exec_price = float(
                    exec_price_raw
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            holdings[symbol] = {
                "entry_price":
                    exec_price,
                "entry_date":
                    today_str
            }

            changed = True

            print(
                f"Confirmed BUY: "
                f"{symbol} @ {exec_price}"
            )

        elif action == "SELL":

            if symbol in holdings:

                del holdings[
                    symbol
                ]

                changed = True

                print(
                    f"Confirmed SELL: "
                    f"{symbol}"
                )

    if changed:

        holdings_ws.clear()

        rows_out = [
            [
                "symbol",
                "entry_price",
                "entry_date"
            ]
        ]

        rows_out += [
            [
                symbol,
                value["entry_price"],
                value["entry_date"]
            ]
            for symbol, value
            in holdings.items()
        ]

        holdings_ws.update(
            rows_out,
            "A1"
        )

        print(
            "Holdings updated."
        )


# ============================================================
# BUILD PORTFOLIO
# ============================================================

def build_portfolio(universe_df):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if not sheet_id or not creds_json:

        print(
            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS."
        )

        return

    creds_dict = json.loads(
        creds_json
    )

    credentials = (
        Credentials
        .from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets"
            ]
        )
    )

    gc = gspread.authorize(
        credentials
    )

    sh = gc.open_by_key(
        sheet_id
    )

    capital = read_config(
        sh
    )

    apply_confirmed_executions(
        sh
    )

    try:

        holdings_ws = sh.worksheet(
            HOLDINGS_WORKSHEET
        )

        existing = (
            holdings_ws.get_all_records()
        )

        current_holdings = {
            row["symbol"]: {
                "entry_price":
                    float(
                        row.get(
                            "entry_price"
                        ) or 0
                    ),
                "entry_date":
                    row.get(
                        "entry_date",
                        ""
                    )
            }
            for row in existing
            if row.get("symbol")
        }

    except gspread.WorksheetNotFound:

        holdings_ws = sh.add_worksheet(
            title=HOLDINGS_WORKSHEET,
            rows=100,
            cols=5
        )

        holdings_ws.update(
            [
                [
                    "symbol",
                    "entry_price",
                    "entry_date"
                ]
            ],
            "A1"
        )

        current_holdings = {}

    # ========================================================
    # EXACT SAME ELIGIBILITY AS BACKTEST
    # ========================================================

    pool = universe_df[
        (
            universe_df["price_tt"]
            == True
        )
        &
        (
            universe_df["rs_line_tt"]
            == True
        )
        &
        (
            universe_df["liquid"]
            == True
        )
    ].copy()

    pool = (
        pool
        .sort_values(
            "rs_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    pool["rank"] = range(
        1,
        len(pool) + 1
    )

    rank_lookup = dict(
        zip(
            pool["symbol"],
            pool["rank"]
        )
    )

    price_lookup = dict(
        zip(
            universe_df["symbol"],
            universe_df["last_close"]
        )
    )

    target_top10 = set(
        pool
        .head(TOP_N)
        ["symbol"]
        .tolist()
    )

    kept = [
        symbol
        for symbol
        in current_holdings
        if rank_lookup.get(
            symbol,
            999999
        ) <= EXIT_RANK
    ]

    pending_sell = [
        symbol
        for symbol
        in current_holdings
        if rank_lookup.get(
            symbol,
            999999
        ) > EXIT_RANK
    ]

    slots_open = (
        TOP_N
        - len(
            [
                s
                for s in kept
                if s in target_top10
            ]
        )
    )

    pending_buy = []

    if slots_open > 0:

        for symbol in pool["symbol"]:

            if symbol in current_holdings:
                continue

            if len(pending_buy) >= slots_open:
                break

            pending_buy.append(
                symbol
            )

    slot_capital = (
        capital / TOP_N
        if capital > 0
        else 0
    )

    rows = []

    # ========================================================
    # HOLD
    # ========================================================

    for symbol in kept:

        entry_price = (
            current_holdings[
                symbol
            ]["entry_price"]
        )

        entry_date = (
            current_holdings[
                symbol
            ]["entry_date"]
        )

        current_price = (
            price_lookup.get(
                symbol,
                entry_price
            )
        )

        position_value = (
            round(
                slot_capital,
                0
            )
            if capital > 0
            else 0
        )

        qty = (
            int(
                position_value
                / entry_price
            )
            if entry_price > 0
            else 0
        )

        pnl_pct = (
            (
                current_price
                / entry_price
                - 1
            )
            * 100
            if entry_price > 0
            else 0
        )

        gross_pnl = (
            qty
            * (
                current_price
                - entry_price
            )
        )

        sell_cost = (
            sell_side_cost(
                qty * current_price
            )
            if qty > 0
            else 0
        )

        tax = estimate_stcg(
            gross_pnl
            - sell_cost
        )

        rows.append({

            "Action":
                "HOLD",

            "Executed":
                "",

            "Execution Price":
                "",

            "Symbol":
                symbol,

            "Rank":
                rank_lookup.get(
                    symbol,
                    ""
                ),

            "Entry Price":
                entry_price,

            "Entry Date":
                entry_date,

            "Current Price":
                current_price,

            "Qty":
                qty,

            "Position Value (Rs)":
                position_value,

            "P&L %":
                f"{pnl_pct:.2f}%",

            "Buy Cost (Rs)":
                round(
                    buy_side_cost(
                        qty
                        * entry_price
                    ),
                    2
                ),

            "Sell Cost (Rs)":
                round(
                    sell_cost,
                    2
                ),

            "Est. STCG Tax (Rs)":
                round(
                    tax,
                    2
                ),
        })

    # ========================================================
    # BUY
    # ========================================================

    for symbol in pending_buy:

        current_price = (
            price_lookup.get(
                symbol,
                0
            )
        )

        qty = (
            int(
                slot_capital
                / current_price
            )
            if current_price > 0
            else 0
        )

        rows.append({

            "Action":
                "BUY",

            "Executed":
                "",

            "Execution Price":
                "",

            "Symbol":
                symbol,

            "Rank":
                rank_lookup.get(
                    symbol,
                    ""
                ),

            "Entry Price":
                current_price,

            "Entry Date":
                "PENDING",

            "Current Price":
                current_price,

            "Qty":
                qty,

            "Position Value (Rs)":
                round(
                    slot_capital,
                    0
                ),

            "P&L %":
                "",

            "Buy Cost (Rs)":
                round(
                    buy_side_cost(
                        qty
                        * current_price
                    ),
                    2
                ),

            "Sell Cost (Rs)":
                "",

            "Est. STCG Tax (Rs)":
                "",
        })

    # ========================================================
    # SELL
    # ========================================================

    for symbol in pending_sell:

        entry_price = (
            current_holdings[
                symbol
            ]["entry_price"]
        )

        entry_date = (
            current_holdings[
                symbol
            ]["entry_date"]
        )

        current_price = (
            price_lookup.get(
                symbol,
                entry_price
            )
        )

        pnl_pct = (
            (
                current_price
                / entry_price
                - 1
            )
            * 100
            if entry_price > 0
            else 0
        )

        rows.append({

            "Action":
                "SELL",

            "Executed":
                "",

            "Execution Price":
                "",

            "Symbol":
                symbol,

            "Rank":
                rank_lookup.get(
                    symbol,
                    ""
                ),

            "Entry Price":
                entry_price,

            "Entry Date":
                entry_date,

            "Current Price":
                current_price,

            "Qty":
                "",

            "Position Value (Rs)":
                "",

            "P&L %":
                f"{pnl_pct:.2f}%",

            "Buy Cost (Rs)":
                "",

            "Sell Cost (Rs)":
                "",

            "Est. STCG Tax (Rs)":
                "",
        })

    # ========================================================
    # GOOGLE SHEET
    # ========================================================

    required_rows = (
        len(rows)
        + 10
    )

    required_cols = (
        len(PORTFOLIO_HEADER)
        + 2
    )

    try:

        port_ws = sh.worksheet(
            PORTFOLIO_WORKSHEET
        )

        port_ws.resize(
            rows=max(
                port_ws.row_count,
                required_rows
            ),
            cols=max(
                port_ws.col_count,
                required_cols
            )
        )

    except gspread.WorksheetNotFound:

        port_ws = sh.add_worksheet(
            title=PORTFOLIO_WORKSHEET,
            rows=required_rows,
            cols=required_cols
        )

    port_ws.clear()

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    invested = sum(
        r["Position Value (Rs)"]
        for r in rows
        if isinstance(
            r["Position Value (Rs)"],
            (int, float)
        )
    )

    summary_text = (
        f"Last updated: {timestamp} | "
        f"Capital: Rs.{capital:,.0f} | "
        f"Deployed: Rs.{invested:,.0f} | "
        f"Entry = Price TT + RS Line TT + "
        f"Liquidity | "
        f"Buy = Top {TOP_N} | "
        f"Hold = Rank 1-{EXIT_RANK} | "
        f"Exit = Rank >{EXIT_RANK}"
    )

    if capital == 0:

        summary_text += (
            " | SET CAPITAL IN CONFIG"
        )

    port_ws.update(
        [[summary_text]],
        "A1"
    )

    row_lists = [
        [
            r.get(
                col,
                ""
            )
            for col in PORTFOLIO_HEADER
        ]
        for r in rows
    ]

    port_ws.update(
        [PORTFOLIO_HEADER]
        + row_lists,
        "A3"
    )

    print(
        f"Portfolio updated: "
        f"{len(kept)} held, "
        f"{len(pending_buy)} BUY, "
        f"{len(pending_sell)} SELL."
    )


# ============================================================
# WRITE SCREENER
# ============================================================

def write_to_sheet(
    df,
    run_mode="EOD"
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if not sheet_id or not creds_json:

        print(
            "Missing SHEET_ID/"
            "GOOGLE_CREDENTIALS."
        )

        df.to_csv(
            "rs_screener_output.csv",
            index=False
        )

        return

    credentials = (
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
        credentials
    )

    sh = gc.open_by_key(
        sheet_id
    )

    n_rows = (
        len(df)
        + 10
    )

    n_cols = (
        len(df.columns)
        + 2
        if len(df.columns)
        else 5
    )

    try:

        ws = sh.worksheet(
            WORKSHEET_NAME
        )

        ws.resize(
            rows=max(
                ws.row_count,
                n_rows
            ),
            cols=max(
                ws.col_count,
                n_cols
            )
        )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(
            title=WORKSHEET_NAME,
            rows=n_rows,
            cols=n_cols
        )

    ws.clear()

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M IST"
        )
    )

    mode_label = (
        "PREVIEW (intraday)"
        if run_mode == "PREVIEW"
        else "EOD FINAL"
    )

    ws.update(
        [[
            f"Last updated: {timestamp} | "
            f"{mode_label} | "
            f"Price TT + RS Line TT + "
            f"50D Avg Volume"
        ]],
        "A1"
    )

    if len(df.columns):

        ws.update(
            [
                list(df.columns)
            ]
            +
            df.fillna("")
            .values
            .tolist(),
            "A3"
        )

    print(
        "Google Sheet updated successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()