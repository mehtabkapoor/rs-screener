"""
RS Live Screener v4
===================

RULES
-----

ENTRY FILTERS:
1. Price Trend Template = 7/7
2. RS Line Trend Template = 7/7
3. 50-day average volume >= 50,000 shares
4. Rank eligible stocks by raw RS Score
5. Top 10 = BUY ENTRY

HOLD / EXIT:
- Rank 1-15  : HOLD
- Rank >15   : EXIT
- Trend Template failure after entry does NOT cause exit.
- Volume deterioration after entry does NOT cause exit.

OTHER:
- Blue Dot is NOT required.
- No VCP/contraction filter.
- No stop-loss.
- No trailing stop.
- RS <20 EMA is displayed only as a warning.
- Market breadth remains informational and does not alter
  the stock ranking.

The Price Trend Template, RS Line Trend Template and RS Score
are preserved from the supplied production screener.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import gspread

from google.oauth2.service_account import Credentials
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

STOCKS_FILE = "stocks.csv"

LOOKBACK_DAYS = 250

MIN_PRICE = 10.0

# NEW RULE
MIN_AVG_VOLUME = 50000
VOLUME_LOOKBACK = 50

MAX_PLAUSIBLE_DAILY_MOVE = 0.30


# ============================================================
# RANKING RULES
# ============================================================

# Buy top 10
ENTRY_TOP_N = 10

# Hold until rank >15
HOLD_BUFFER_RANK = 15


# ============================================================
# BREADTH
# ============================================================

BREADTH_RISK_ON = 60.0
BREADTH_RISK_CAUTION = 40.0
BREADTH_CIRCUIT_BREAKER = 25.0
BREADTH_SMOOTH_SPAN = 3


# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"

SCREENER_WORKSHEET = "Live_Screener"


# ============================================================
# LOAD TICKERS
# ============================================================

def load_tickers():

    if not os.path.exists(STOCKS_FILE):

        print(
            f"Error: Could not find {STOCKS_FILE}"
        )

        sys.exit(1)

    df = pd.read_csv(STOCKS_FILE)

    if "symbol" not in df.columns:

        raise ValueError(
            "stocks.csv must contain a 'symbol' column."
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
# DATA CLEANING
# ============================================================

def clean_price_series(close):

    close = (
        close
        .copy()
        .sort_index()
    )

    pct_change = close.pct_change()

    bad = (
        pct_change.abs()
        > MAX_PLAUSIBLE_DAILY_MOVE
    )

    n_bad = bad.sum()

    if n_bad > 0:

        cleaned = close.copy()

        for idx in close.index[bad]:

            pos = (
                cleaned.index
                .get_loc(idx)
            )

            if pos > 0:

                cleaned.iloc[pos] = (
                    cleaned.iloc[pos - 1]
                )

        return cleaned, int(n_bad)

    return close, 0


# ============================================================
# TREND TEMPLATE
# ============================================================

def trend_template_check(s):

    if len(s) < 252:

        return False, 0

    sma50 = (
        s.rolling(50)
        .mean()
        .iloc[-1]
    )

    sma150 = (
        s.rolling(150)
        .mean()
        .iloc[-1]
    )

    sma200 = (
        s.rolling(200)
        .mean()
        .iloc[-1]
    )

    sma200_1mo = (
        s.rolling(200)
        .mean()
        .shift(21)
        .iloc[-1]
    )

    low52 = (
        s.rolling(252)
        .min()
        .iloc[-1]
    )

    high52 = (
        s.rolling(252)
        .max()
        .iloc[-1]
    )

    curr = s.iloc[-1]


    # ========================================================
    # EXACT EXISTING 7 CONDITIONS
    # ========================================================

    c1 = (
        curr > sma150
        and
        curr > sma200
    )

    c2 = (
        sma150 > sma200
    )

    c3 = (
        sma200 > sma200_1mo
    )

    c4 = (
        sma50 > sma150
        and
        sma50 > sma200
    )

    c5 = (
        curr > sma50
    )

    c6 = (
        curr >= 1.25 * low52
    )

    c7 = (
        curr >= 0.75 * high52
    )


    met = sum([
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7
    ])

    return (
        met == 7,
        met
    )


# ============================================================
# STOCK METRICS
# ============================================================

def calculate_stock_metrics(
    df_stock,
    bench_close
):

    close = (
        df_stock["Close"]
        .dropna()
        .sort_index()
    )

    volume = (
        df_stock["Volume"]
        .reindex(close.index)
        .fillna(0)
    )


    # --------------------------------------------------------
    # Clean price data
    # --------------------------------------------------------

    close, _ = clean_price_series(
        close
    )


    # --------------------------------------------------------
    # Align stock and benchmark
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    vol_aligned = (
        volume
        .reindex(aligned.index)
    )

    curr_price = float(
        aligned["s"].iloc[-1]
    )


    # ========================================================
    # NEW: 50-DAY AVERAGE VOLUME
    # ========================================================

    avg_vol = float(
        vol_aligned
        .rolling(VOLUME_LOOKBACK)
        .mean()
        .iloc[-1]
    )


    # --------------------------------------------------------
    # Liquidity filter
    # --------------------------------------------------------

    is_liquid = (
        curr_price >= MIN_PRICE
        and
        avg_vol >= MIN_AVG_VOLUME
    )

    if not is_liquid:

        return None


    # ========================================================
    # RS LINE
    # ========================================================

    rs_line = (
        aligned["s"]
        /
        aligned["b"]
    )


    # ========================================================
    # RS SCORE
    # EXACT EXISTING FORMULA
    # ========================================================

    def pct_ret(
        series,
        days
    ):

        return (
            series.iloc[-1]
            /
            series.shift(days).iloc[-1]
        ) - 1.0


    rs_score = (
        0.40 *
        pct_ret(
            aligned["s"],
            63
        )
        +
        0.20 *
        pct_ret(
            aligned["s"],
            126
        )
        +
        0.20 *
        pct_ret(
            aligned["s"],
            189
        )
        +
        0.20 *
        pct_ret(
            aligned["s"],
            252
        )
    ) * 100.0


    # ========================================================
    # PRICE TREND TEMPLATE
    # ========================================================

    price_tt_pass, price_tt_met = (
        trend_template_check(
            aligned["s"]
        )
    )


    # ========================================================
    # RS LINE TREND TEMPLATE
    # ========================================================

    rs_tt_pass, rs_tt_met = (
        trend_template_check(
            rs_line
        )
    )


    # ========================================================
    # 50 DMA
    # ========================================================

    sma50 = (
        aligned["s"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    above_50dma = (
        curr_price > sma50
    )


    # ========================================================
    # ATR(14)
    # ========================================================

    high = (
        df_stock["High"]
        .reindex(aligned.index)
        if "High" in df_stock.columns
        else aligned["s"]
    )

    low = (
        df_stock["Low"]
        .reindex(aligned.index)
        if "Low" in df_stock.columns
        else aligned["s"]
    )


    tr = np.maximum(
        high - low,
        np.maximum(
            abs(
                high -
                aligned["s"].shift(1)
            ),
            abs(
                low -
                aligned["s"].shift(1)
            )
        )
    )


    atr14 = float(
        tr
        .rolling(14)
        .mean()
        .iloc[-1]
    )


    # ========================================================
    # RS 20 EMA
    # ========================================================

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
        float(
            rs_line.iloc[-1]
        )
        <
        rs_ema20
    )


    # ========================================================
    # RETURN METRICS
    # ========================================================

    return {

        "price":
            round(
                curr_price,
                2
            ),

        "rs_score":
            round(
                rs_score,
                2
            ),

        "price_tt_pass":
            price_tt_pass,

        "price_tt_met":
            price_tt_met,

        "rs_tt_pass":
            rs_tt_pass,

        "rs_tt_met":
            rs_tt_met,

        "above_50dma":
            above_50dma,

        "atr14":
            round(
                atr14,
                2
            ),

        "rs_below_20ema":
            rs_below_20ema,

        "avg_vol_50d":
            int(avg_vol)
    }


# ============================================================
# RUN SCREENER
# ============================================================

def run_screener():

    tickers = load_tickers()

    print(
        f"Loaded {len(tickers)} symbols "
        "for screening..."
    )


    # ========================================================
    # BENCHMARK
    # ========================================================

    bench_data = yf.download(
        BENCHMARK,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )


    if bench_data.empty:

        print(
            f"{BENCHMARK} unavailable. "
            f"Using {BENCHMARK_FALLBACK}."
        )

        bench_data = yf.download(
            BENCHMARK_FALLBACK,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )


    if bench_data.empty:

        raise RuntimeError(
            "Unable to download benchmark data."
        )


    bench_close = (
        bench_data["Close"]
        .dropna()
    )


    if isinstance(
        bench_close,
        pd.DataFrame
    ):

        bench_close = (
            bench_close.iloc[:, 0]
        )


    bench_close, _ = (
        clean_price_series(
            bench_close
        )
    )


    # ========================================================
    # PROCESS STOCKS
    # ========================================================

    stock_results = {}

    above_50dma_count = 0

    total_valid = 0

    batch_size = 50


    for i in range(
        0,
        len(tickers),
        batch_size
    ):

        batch = tickers[
            i:i + batch_size
        ]


        try:

            data = yf.download(
                batch,
                period="2y",
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


        for sym in batch:

            try:

                if len(batch) == 1:

                    sdata = data

                else:

                    if (
                        sym in
                        data.columns
                        .get_level_values(0)
                    ):

                        sdata = data[sym]

                    else:

                        sdata = (
                            pd.DataFrame()
                        )


                if (
                    sdata.empty
                    or
                    "Close"
                    not in sdata.columns
                ):

                    continue


                metrics = (
                    calculate_stock_metrics(
                        sdata,
                        bench_close
                    )
                )


                if metrics:

                    clean_sym = (
                        sym.replace(
                            ".NS",
                            ""
                        )
                    )

                    stock_results[
                        clean_sym
                    ] = metrics

                    total_valid += 1


                    if metrics[
                        "above_50dma"
                    ]:

                        above_50dma_count += 1


            except Exception:

                continue


        time.sleep(0.5)


    # ========================================================
    # MARKET BREADTH
    # ========================================================

    breadth_pct = round(
        (
            above_50dma_count
            /
            total_valid
            *
            100.0
        )
        if total_valid > 0
        else 0.0,
        2
    )


    if (
        breadth_pct
        >=
        BREADTH_RISK_ON
    ):

        regime = (
            "RISK-ON (100% Size)"
        )

    elif (
        breadth_pct
        >=
        BREADTH_RISK_CAUTION
    ):

        regime = (
            "CAUTION (50% Size)"
        )

    elif (
        breadth_pct
        >=
        BREADTH_CIRCUIT_BREAKER
    ):

        regime = (
            "DEFENSIVE (No New Buys)"
        )

    else:

        regime = (
            "CIRCUIT-BREAKER (Liquidate)"
        )


    # ========================================================
    # FILTER
    #
    # IMPORTANT:
    # Blue Dot has been REMOVED.
    #
    # Only:
    #   Price TT 7/7
    #   RS Line TT 7/7
    #   50D volume >= 50,000
    #
    # are required for entry.
    # ========================================================

    candidates = []


    for sym, m in stock_results.items():

        if (
            m["price_tt_pass"]
            and
            m["rs_tt_pass"]
        ):

            candidates.append(
                (
                    sym,
                    m["rs_score"],
                    m["price"],
                    m["atr14"],
                    m["rs_below_20ema"],
                    m["price_tt_met"],
                    m["rs_tt_met"],
                    m["avg_vol_50d"]
                )
            )


    # ========================================================
    # RANK
    # ========================================================

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    results_table = []


    for rank, (
        sym,
        score,
        price,
        atr,
        rs_below_ema,
        price_tt_met,
        rs_tt_met,
        avg_vol
    ) in enumerate(
        candidates,
        1
    ):


        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        if rank <= ENTRY_TOP_N:

            action = "BUY ENTRY"

        elif rank <= HOLD_BUFFER_RANK:

            action = "HOLD ALLOWED"

        else:

            action = "WATCHLIST ONLY"


        # ----------------------------------------------------
        # RS 20 EMA WARNING
        # ----------------------------------------------------

        if rs_below_ema:

            action += (
                " (RS < 20EMA Warn)"
            )


        results_table.append({

            "Rank":
                rank,

            "Symbol":
                sym,

            "Action":
                action,

            "RS Score":
                score,

            "Price (INR)":
                price,

            "ATR (14)":
                atr,

            "Risk/Share (2xATR)":
                round(
                    2 * atr,
                    2
                ),

            "50D Avg Volume":
                avg_vol
        })


    out_df = pd.DataFrame(
        results_table
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        f" MARKET BREADTH: "
        f"{breadth_pct}% (>50DMA)"
        f" | REGIME: {regime}"
    )

    print(
        "=" * 70
    )


    if out_df.empty:

        print(
            "NO STOCKS PASSED "
            "THE ENTRY FILTERS."
        )

    else:

        print(
            out_df.to_string(
                index=False
            )
        )


    # ========================================================
    # GOOGLE SHEETS
    # ========================================================

    sheet_id = os.environ.get(
        SHEET_ID_ENV
    )

    creds_json = os.environ.get(
        CREDS_ENV
    )


    if (
        sheet_id
        and
        creds_json
    ):

        try:

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


            gc = (
                gspread
                .authorize(creds)
            )


            sh = (
                gc.open_by_key(
                    sheet_id
                )
            )


            ws = (
                sh.worksheet(
                    SCREENER_WORKSHEET
                )
            )


            ws.clear()


            ws.update(
                [[
                    "RS SCREENER RESULTS - "
                    +
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M IST"
                    )
                ]],
                "A1"
            )


            ws.update(
                [[
                    f"Breadth: "
                    f"{breadth_pct}%",
                    f"Regime: "
                    f"{regime}"
                ]],
                "A2"
            )


            if not out_df.empty:

                ws.update(
                    [
                        list(
                            out_df.columns
                        )
                    ]
                    +
                    out_df.values.tolist(),
                    "A4"
                )


            print(
                f"\nSuccessfully updated "
                f"Google Sheet tab "
                f"'{SCREENER_WORKSHEET}'"
            )


        except Exception as e:

            print(
                f"Google Sheet update failed: "
                f"{e}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_screener()