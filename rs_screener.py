# RS LIVE SCREENER — SYNCED WITH BACKTEST

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

MIN_PRICE = 10
MIN_AVG_VOLUME = 50_000
VOLUME_LOOKBACK = 50

ENTRY_TOP_N = 10
EXIT_RANK = 15

DOWNLOAD_PERIOD = "2y"

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

SHEET_NAME = "Live_Screener"


# ============================================================
# LOAD TICKERS
# ============================================================

def load_tickers():

    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(
            f"{STOCKS_FILE} not found"
        )

    df = pd.read_csv(STOCKS_FILE)

    if "symbol" not in df.columns:
        raise ValueError(
            "stocks.csv must contain 'symbol'"
        )

    return [
        s if s.endswith(".NS") else s + ".NS"
        for s in
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        if s
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

    c1 = (
        (s > sma150) &
        (s > sma200)
    )

    c2 = sma150 > sma200

    c3 = (
        sma200 >
        sma200.shift(21)
    )

    c4 = (
        (sma50 > sma150) &
        (sma50 > sma200)
    )

    c5 = s > sma50

    c6 = s >= 1.25 * low52

    c7 = s >= 0.75 * high52

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
        met == 7,
        met
    )


# ============================================================
# STOCK METRICS
# ============================================================

def calculate_metrics(
    stock,
    benchmark
):

    if (
        stock.empty or
        "Close" not in stock.columns
    ):
        return None

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

    aligned = pd.concat(
        [
            close,
            benchmark
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(aligned) < 280:
        return None

    aligned.columns = [
        "price",
        "benchmark"
    ]

    # --------------------------------------------------------
    # PRICE / VOLUME
    # --------------------------------------------------------

    price = float(
        aligned["price"].iloc[-1]
    )

    avg_volume = float(
        volume
        .reindex(aligned.index)
        .rolling(VOLUME_LOOKBACK)
        .mean()
        .iloc[-1]
    )

    # Same eligibility rule as backtest
    liquid = (
        price >= MIN_PRICE and
        avg_volume >= MIN_AVG_VOLUME
    )

    if not liquid:
        return None

    # --------------------------------------------------------
    # RS LINE
    # --------------------------------------------------------

    rs_line = (
        aligned["price"] /
        aligned["benchmark"]
    )

    # --------------------------------------------------------
    # RAW RS SCORE
    # EXACT BACKTEST FORMULA
    # --------------------------------------------------------

    def ret(days):

        return (
            aligned["price"] /
            aligned["price"].shift(days)
            - 1
        )

    rs_score = (
        0.40 * ret(63) +
        0.20 * ret(126) +
        0.20 * ret(189) +
        0.20 * ret(252)
    ) * 100

    rs_score = float(
        rs_score.iloc[-1]
    )

    # --------------------------------------------------------
    # TREND TEMPLATES
    # --------------------------------------------------------

    price_tt, price_tt_met = (
        trend_template(aligned["price"])
    )

    rs_tt, rs_tt_met = (
        trend_template(rs_line)
    )

    price_tt_pass = bool(
        price_tt.iloc[-1]
    )

    rs_tt_pass = bool(
        rs_tt.iloc[-1]
    )

    price_tt_met = int(
        price_tt_met.iloc[-1]
    )

    rs_tt_met = int(
        rs_tt_met.iloc[-1]
    )

    # --------------------------------------------------------
    # 50 DMA
    # --------------------------------------------------------

    sma50 = (
        aligned["price"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    above_50dma = (
        price > sma50
    )

    # --------------------------------------------------------
    # RS 20 EMA — WARNING ONLY
    # --------------------------------------------------------

    rs_ema20 = (
        rs_line
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    rs_below_20ema = (
        float(rs_line.iloc[-1])
        < float(rs_ema20)
    )

    # --------------------------------------------------------
    # ATR 14
    # --------------------------------------------------------

    high = (
        stock["High"]
        .reindex(aligned.index)
        if "High" in stock.columns
        else aligned["price"]
    )

    low = (
        stock["Low"]
        .reindex(aligned.index)
        if "Low" in stock.columns
        else aligned["price"]
    )

    prev_close = aligned["price"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr14 = float(
        tr
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    return {
        "price": round(price, 2),
        "rs_score": round(rs_score, 2),
        "price_tt_pass": price_tt_pass,
        "price_tt_met": price_tt_met,
        "rs_tt_pass": rs_tt_pass,
        "rs_tt_met": rs_tt_met,
        "avg_volume_50d": int(avg_volume),
        "above_50dma": bool(above_50dma),
        "atr14": round(atr14, 2),
        "rs_below_20ema": bool(rs_below_20ema)
    }


# ============================================================
# BENCHMARK
# ============================================================

def get_benchmark():

    for ticker in [
        BENCHMARK,
        BENCHMARK_FALLBACK
    ]:

        try:

            data = yf.download(
                ticker,
                period=DOWNLOAD_PERIOD,
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
                return close

        except Exception as e:

            print(
                f"Benchmark error {ticker}: {e}"
            )

    raise RuntimeError(
        "Benchmark download failed"
    )


# ============================================================
# GOOGLE SHEETS
# ============================================================

def write_sheet(
    df,
    breadth,
    regime,
    timestamp
):

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )

    if not sheet_id or not creds_json:

        df.to_csv(
            "live_screener.csv",
            index=False
        )

        print(
            "Google credentials missing. "
            "Saved live_screener.csv"
        )

        return

    credentials = (
        Credentials
        .from_service_account_info(
            json.loads(creds_json),
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

    try:

        ws = sh.worksheet(
            SHEET_NAME
        )

    except gspread.WorksheetNotFound:

        ws = sh.add_worksheet(
            title=SHEET_NAME,
            rows=1000,
            cols=20
        )

    ws.clear()

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    params = [
        ["RS LIVE SCREENER"],
        ["Run timestamp", timestamp],
        ["Benchmark", BENCHMARK],
        ["Price minimum", MIN_PRICE],
        ["50D average volume minimum", MIN_AVG_VOLUME],
        ["Volume lookback", VOLUME_LOOKBACK],
        ["Price Trend Template", "7/7"],
        ["RS Line Trend Template", "7/7"],
        ["RS Score", "40% 63D + 20% 126D + 20% 189D + 20% 252D"],
        ["Entry", "Rank 1-10"],
        ["Hold", "Rank 1-15"],
        ["Exit", "Rank >15"],
        ["Blue Dot", "NO"],
        ["Green Dot", "NO"],
        ["VCP", "NO"],
        ["Stop Loss", "NO"],
        ["Trailing Stop", "NO"],
        ["RS <20 EMA exit", "NO"],
        ["Calculation", "Python"],
        ["Google Sheets", "Output only"],
        [],
        ["Market breadth", breadth],
        ["Market regime", regime],
        []
    ]

    ws.update(
        params,
        "A1"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    start_row = len(params) + 1

    if not df.empty:

        values = [
            list(df.columns)
        ] + df.fillna("").values.tolist()

        ws.update(
            values,
            f"A{start_row}"
        )

    print(
        f"Google Sheet updated: "
        f"{SHEET_NAME}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    timestamp = (
        datetime.now(
            ZoneInfo("Asia/Kolkata")
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
    )

    print(
        f"\nSCREENER RUN: {timestamp}"
    )

    tickers = load_tickers()

    benchmark = get_benchmark()

    results = []

    total_valid = 0
    above_50dma = 0

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
            f"{i+1}-{i+len(batch)} "
            f"/ {len(tickers)}"
        )

        try:

            data = yf.download(
                batch,
                period=DOWNLOAD_PERIOD,
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

        for sym in batch:

            try:

                if len(batch) == 1:

                    stock = data

                else:

                    if (
                        sym
                        not in
                        data.columns.get_level_values(0)
                    ):
                        continue

                    stock = data[sym]

                metrics = calculate_metrics(
                    stock,
                    benchmark
                )

                if metrics is None:
                    continue

                total_valid += 1

                if metrics["above_50dma"]:
                    above_50dma += 1

                results.append({
                    "Symbol":
                        sym.replace(".NS", ""),
                    **metrics
                })

            except Exception as e:

                print(
                    f"{sym}: {e}"
                )

        time.sleep(0.5)

    # ========================================================
    # BREADTH
    # ========================================================

    breadth = round(
        above_50dma /
        total_valid * 100,
        2
    ) if total_valid else 0

    # Informational only.
    # Does NOT alter ranking or sizing.

    if breadth >= 60:
        regime = "RISK-ON"
    elif breadth >= 40:
        regime = "CAUTION"
    elif breadth >= 25:
        regime = "DEFENSIVE"
    else:
        regime = "CIRCUIT-BREAKER"

    # ========================================================
    # ENTRY ELIGIBLE UNIVERSE
    # EXACT BACKTEST LOGIC
    # ========================================================

    df = pd.DataFrame(results)

    if df.empty:

        print(
            "No stocks passed."
        )

        write_sheet(
            df,
            breadth,
            regime,
            timestamp
        )

        return

    df = df[
        (df["price_tt_pass"] == True) &
        (df["rs_tt_pass"] == True)
    ].copy()

    # ========================================================
    # RANK
    # SAME UNIVERSE AS BACKTEST
    # ========================================================

    df = df.sort_values(
        "rs_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    df["Rank"] = (
        np.arange(len(df)) + 1
    )

    # ========================================================
    # ACTION
    # ========================================================

    df["Action"] = np.where(
        df["Rank"] <= ENTRY_TOP_N,
        "BUY ENTRY",
        np.where(
            df["Rank"] <= EXIT_RANK,
            "HOLD ALLOWED",
            "WATCHLIST ONLY"
        )
    )

    # RS EMA is warning only.

    df["RS 20EMA Warning"] = np.where(
        df["rs_below_20ema"],
        "RS < 20EMA",
        ""
    )

    # ========================================================
    # OUTPUT COLUMNS
    # ========================================================

    df = df[
        [
            "Rank",
            "Symbol",
            "Action",
            "rs_score",
            "price",
            "price_tt_met",
            "rs_tt_met",
            "avg_volume_50d",
            "atr14",
            "RS 20EMA Warning"
        ]
    ]

    df.columns = [
        "Rank",
        "Symbol",
        "Action",
        "RS Score",
        "Price (INR)",
        "Price TT",
        "RS Line TT",
        "50D Avg Volume",
        "ATR (14)",
        "RS 20EMA Warning"
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        f"RUN: {timestamp}"
    )

    print(
        f"Breadth: {breadth}%"
    )

    print(
        f"Regime: {regime}"
    )

    print(
        "=" * 70
    )

    print(
        df.to_string(index=False)
    )

    # ========================================================
    # GOOGLE SHEETS
    # ========================================================

    write_sheet(
        df,
        breadth,
        regime,
        timestamp
    )

    print(
        "\nDONE"
    )


if __name__ == "__main__":
    main()