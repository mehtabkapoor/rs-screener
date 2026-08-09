"""
===============================================================
RS SCREENER BACKTEST — MODIFIABLE / PLUG-AND-PLAY
===============================================================

ENTRY:
    Blue Dot / RS New High
    +
    Price Trend Template
    +
    RS-Line Trend Template
    +
    Rank by RS Score
    +
    Hold TOP_N stocks

EXIT:
    RS LINE < RS LINE 3-EMA

IMPORTANT:
    EXIT IS STATE-BASED, NOT CROSSOVER-BASED.

    The rule is simply:

        RS Line < 3 EMA of RS Line

    It does NOT require:
        yesterday RS Line > EMA
        AND
        today RS Line < EMA

EXECUTION:
    Signals calculated using today's CLOSE.
    With NEXT_OPEN execution, orders execute at
    the following trading day's OPEN.

REQUIRED INPUT COLUMNS:

    date
    symbol
    open
    high
    low
    close
    rs_line
    rs_score

===============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================
# CONFIGURATION — CHANGE PARAMETERS HERE
# =============================================================

CONFIG = {

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    "INPUT_FILE": "nse_daily_data.xlsx",

    "INPUT_SHEET": 0,

    "OUTPUT_FILE": "RS_Backtest_Output.xlsx",


    # ---------------------------------------------------------
    # BACKTEST PERIOD
    # ---------------------------------------------------------

    "START_DATE": "2025-04-01",

    "END_DATE": "2026-08-07",


    # ---------------------------------------------------------
    # PORTFOLIO
    # ---------------------------------------------------------

    "TOP_N": 10,

    "EQUAL_WEIGHT": True,


    # ---------------------------------------------------------
    # ENTRY — RS NEW HIGH
    # ---------------------------------------------------------

    "RS_NEW_HIGH_LOOKBACK": 250,

    # None = no minimum RS score
    #
    # Example:
    # 70
    # 80
    # 90

    "MIN_RS_SCORE": None,


    # ---------------------------------------------------------
    # PRICE TREND TEMPLATE
    # ---------------------------------------------------------

    "PRICE_SMA_SHORT": 50,

    "PRICE_SMA_MEDIUM": 150,

    "PRICE_SMA_LONG": 200,

    "PRICE_SMA_LONG_RISING_DAYS": 20,

    "PRICE_52W_LOOKBACK": 252,

    # Price must be at least this multiple
    # of 52-week low.
    #
    # 1.30 = 30% above 52-week low

    "MIN_ABOVE_52W_LOW": 1.30,

    # Price must be within this percentage
    # of 52-week high.
    #
    # 25 = within 25% of 52-week high

    "MAX_BELOW_52W_HIGH_PCT": 25,


    # ---------------------------------------------------------
    # RS-LINE TREND TEMPLATE
    # ---------------------------------------------------------

    "RS_SMA_SHORT": 50,

    "RS_SMA_MEDIUM": 150,

    "RS_SMA_LONG": 200,

    "RS_SMA_LONG_RISING_DAYS": 20,


    # ---------------------------------------------------------
    # EXIT
    # ---------------------------------------------------------

    # YOUR REQUEST:
    #
    # RS LINE < RS LINE 3-EMA

    "RS_EXIT_EMA": 3,


    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------

    # RECOMMENDED:
    #
    # Signal at today's close
    # Execute at next trading day's open

    "EXECUTION": "NEXT_OPEN",

    # Alternative:
    #
    # "SAME_CLOSE"
    #
    # Do not use as the main result unless the signal
    # is genuinely executable before that close.


    # ---------------------------------------------------------
    # TRANSACTION COSTS
    # ---------------------------------------------------------

    # Total trading cost per side.
    #
    # Example:
    # 0.15 = 0.15%

    "COST_PER_SIDE_PCT": 0.15,


    # Additional slippage per side.
    #
    # Example:
    # 0.05 = 0.05%

    "SLIPPAGE_PER_SIDE_PCT": 0.05,


    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    "PLOT_EQUITY_CURVE": True,

    "MARK_HIGHER_HIGHS_LOWER_LOWS": True,

}


# =============================================================
# LOAD DATA
# =============================================================

def load_data():

    filename = CONFIG["INPUT_FILE"]

    if not os.path.exists(filename):

        raise FileNotFoundError(
            "\nINPUT FILE NOT FOUND:\n"
            f"{filename}\n\n"
            "Put the raw market-data file in the same "
            "folder as this Python script or change "
            "INPUT_FILE in CONFIG."
        )


    extension = os.path.splitext(
        filename
    )[1].lower()


    if extension == ".csv":

        df = pd.read_csv(filename)


    elif extension in [".xlsx", ".xls"]:

        df = pd.read_excel(
            filename,
            sheet_name=CONFIG["INPUT_SHEET"]
        )


    else:

        raise ValueError(
            "Only CSV, XLSX and XLS files are supported."
        )


    # Normalize column names

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]


    required = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "rs_line",
        "rs_score",
    ]


    missing = [
        c for c in required
        if c not in df.columns
    ]


    if missing:

        raise ValueError(
            "\nMISSING REQUIRED COLUMNS:\n"
            + "\n".join(missing)
            + "\n\nRequired columns:\n"
            + ", ".join(required)
        )


    # Dates

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    # Numbers

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "rs_line",
        "rs_score",
    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.dropna(
        subset=[
            "date",
            "symbol",
            "close",
            "rs_line",
            "rs_score",
        ]
    )


    # Sort

    df = df.sort_values(
        ["symbol", "date"]
    )


    # Remove duplicate stock/date rows

    df = df.drop_duplicates(
        subset=[
            "symbol",
            "date"
        ],
        keep="last"
    )


    return df.reset_index(drop=True)


# =============================================================
# CALCULATE INDICATORS
# =============================================================

def calculate_indicators(df):

    output = []


    for symbol, g in df.groupby(
        "symbol",
        sort=False
    ):

        g = g.copy()

        g = g.sort_values(
            "date"
        ).reset_index(drop=True)


        # -----------------------------------------------------
        # PRICE MOVING AVERAGES
        # -----------------------------------------------------

        g["sma50"] = (
            g["close"]
            .rolling(
                CONFIG["PRICE_SMA_SHORT"]
            )
            .mean()
        )


        g["sma150"] = (
            g["close"]
            .rolling(
                CONFIG["PRICE_SMA_MEDIUM"]
            )
            .mean()
        )


        g["sma200"] = (
            g["close"]
            .rolling(
                CONFIG["PRICE_SMA_LONG"]
            )
            .mean()
        )


        # -----------------------------------------------------
        # 200 SMA RISING
        # -----------------------------------------------------

        slope_days = CONFIG[
            "PRICE_SMA_LONG_RISING_DAYS"
        ]


        g["sma200_previous"] = (
            g["sma200"]
            .shift(slope_days)
        )


        g["price_sma200_rising"] = (
            g["sma200"]
            >
            g["sma200_previous"]
        )


        # -----------------------------------------------------
        # 52-WEEK HIGH / LOW
        # -----------------------------------------------------

        lookback = CONFIG[
            "PRICE_52W_LOOKBACK"
        ]


        g["high_52w"] = (
            g["high"]
            .rolling(lookback)
            .max()
        )


        g["low_52w"] = (
            g["low"]
            .rolling(lookback)
            .min()
        )


        # -----------------------------------------------------
        # RS MOVING AVERAGES
        # -----------------------------------------------------

        g["rs_sma50"] = (
            g["rs_line"]
            .rolling(
                CONFIG["RS_SMA_SHORT"]
            )
            .mean()
        )


        g["rs_sma150"] = (
            g["rs_line"]
            .rolling(
                CONFIG["RS_SMA_MEDIUM"]
            )
            .mean()
        )


        g["rs_sma200"] = (
            g["rs_line"]
            .rolling(
                CONFIG["RS_SMA_LONG"]
            )
            .mean()
        )


        # -----------------------------------------------------
        # RS 200 SMA RISING
        # -----------------------------------------------------

        rs_slope_days = CONFIG[
            "RS_SMA_LONG_RISING_DAYS"
        ]


        g["rs_sma200_previous"] = (
            g["rs_sma200"]
            .shift(rs_slope_days)
        )


        g["rs_sma200_rising"] = (
            g["rs_sma200"]
            >
            g["rs_sma200_previous"]
        )


        # -----------------------------------------------------
        # RS LINE 3-EMA
        # -----------------------------------------------------

        g["rs_exit_ema"] = (
            g["rs_line"]
            .ewm(
                span=CONFIG["RS_EXIT_EMA"],
                adjust=False
            )
            .mean()
        )


        # -----------------------------------------------------
        # RS NEW HIGH / BLUE DOT
        # -----------------------------------------------------
        #
        # Today's RS is compared against the
        # PREVIOUS 250 trading days.
        #
        # shift(1) prevents today's RS value
        # from being included in its own high.
        #

        rs_lookback = CONFIG[
            "RS_NEW_HIGH_LOOKBACK"
        ]


        g["previous_rs_high"] = (
            g["rs_line"]
            .shift(1)
            .rolling(
                rs_lookback
            )
            .max()
        )


        g["blue_dot"] = (
            g["rs_line"]
            >
            g["previous_rs_high"]
        )


        # -----------------------------------------------------
        # PRICE TREND TEMPLATE
        # -----------------------------------------------------

        g["price_tt"] = (

            # 1
            (
                g["close"]
                >
                g["sma50"]
            )

            &

            # 2
            (
                g["close"]
                >
                g["sma150"]
            )

            &

            # 3
            (
                g["close"]
                >
                g["sma200"]
            )

            &

            # 4
            (
                g["sma50"]
                >
                g["sma150"]
            )

            &

            # 5
            (
                g["sma150"]
                >
                g["sma200"]
            )

            &

            # 6
            g["price_sma200_rising"]

            &

            # 7
            (
                g["close"]
                >=
                (
                    g["low_52w"]
                    *
                    CONFIG[
                        "MIN_ABOVE_52W_LOW"
                    ]
                )
            )

            &

            # 8
            (
                g["close"]
                >=
                (
                    g["high_52w"]
                    *
                    (
                        1
                        -
                        CONFIG[
                            "MAX_BELOW_52W_HIGH_PCT"
                        ] / 100
                    )
                )
            )

        )


        # -----------------------------------------------------
        # RS TREND TEMPLATE
        # -----------------------------------------------------

        g["rs_tt"] = (

            (
                g["rs_line"]
                >
                g["rs_sma50"]
            )

            &

            (
                g["rs_line"]
                >
                g["rs_sma150"]
            )

            &

            (
                g["rs_line"]
                >
                g["rs_sma200"]
            )

            &

            (
                g["rs_sma50"]
                >
                g["rs_sma150"]
            )

            &

            (
                g["rs_sma150"]
                >
                g["rs_sma200"]
            )

            &

            g["rs_sma200_rising"]

        )


        # -----------------------------------------------------
        # ENTRY SIGNAL
        # -----------------------------------------------------

        g["entry_signal"] = (

            g["blue_dot"]

            &

            g["price_tt"]

            &

            g["rs_tt"]

        )


        # Optional minimum RS score

        if CONFIG["MIN_RS_SCORE"] is not None:

            g["entry_signal"] &= (

                g["rs_score"]
                >=
                CONFIG["MIN_RS_SCORE"]

            )


        # -----------------------------------------------------
        # EXIT SIGNAL
        # -----------------------------------------------------
        #
        # THIS IS THE IMPORTANT PART.
        #
        # EXIT whenever:
        #
        #       RS LINE < RS LINE 3-EMA
        #
        # This is NOT a crossover test.
        #
        # If RS was already below its 3-EMA yesterday
        # and remains below it today, exit_signal remains TRUE.
        #
        # -----------------------------------------------------

        g["exit_signal"] = (

            g["rs_line"]
            <
            g["rs_exit_ema"]

        )


        output.append(g)


    return pd.concat(
        output,
        ignore_index=True
    )


# =============================================================
# BACKTEST ENGINE
# =============================================================

def run_backtest(df):

    start_date = pd.Timestamp(
        CONFIG["START_DATE"]
    )

    end_date = pd.Timestamp(
        CONFIG["END_DATE"]
    )


    df = df[
        (df["date"] >= start_date)
        &
        (df["date"] <= end_date)
    ].copy()


    df = df.sort_values(
        ["date", "symbol"]
    )


    dates = sorted(
        df["date"].unique()
    )


    # ---------------------------------------------------------
    # PORTFOLIO STATE
    # ---------------------------------------------------------

    positions = {}

    pending_entries = {}

    pending_exits = set()

    trade_log = []

    equity_curve = []


    # Starting equity

    equity = 1.0


    # Total cost per side

    cost_per_side = (

        CONFIG["COST_PER_SIDE_PCT"]

        +

        CONFIG["SLIPPAGE_PER_SIDE_PCT"]

    ) / 100


    # =========================================================
    # DAILY LOOP
    # =========================================================

    for date in dates:

        date = pd.Timestamp(date)


        day = df[
            df["date"] == date
        ].copy()


        # =====================================================
        # 1. EXECUTE PREVIOUS DAY EXIT SIGNALS
        # =====================================================

        if (
            CONFIG["EXECUTION"]
            ==
            "NEXT_OPEN"
        ):

            for symbol in list(
                pending_exits
            ):

                if symbol not in positions:

                    pending_exits.discard(
                        symbol
                    )

                    continue


                row = day[
                    day["symbol"] == symbol
                ]


                if row.empty:

                    continue


                row = row.iloc[0]


                exit_price = float(
                    row["open"]
                )


                position = positions[
                    symbol
                ]


                entry_price = (
                    position[
                        "entry_price"
                    ]
                )


                gross_return = (
                    exit_price
                    /
                    entry_price
                    - 1
                )


                net_return = (
                    gross_return
                    -
                    (
                        cost_per_side
                        * 2
                    )
                )


                trade_log.append({

                    "symbol":
                        symbol,

                    "entry_date":
                        position[
                            "entry_date"
                        ],

                    "exit_date":
                        date,

                    "entry_price":
                        entry_price,

                    "exit_price":
                        exit_price,

                    "gross_return_pct":
                        gross_return * 100,

                    "net_return_pct":
                        net_return * 100,

                    "days_held":
                        (
                            date
                            -
                            position[
                                "entry_date"
                            ]
                        ).days,

                    "exit_reason":
                        (
                            "RS Line < "
                            f"{CONFIG['RS_EXIT_EMA']}-EMA"
                        ),

                })


                del positions[
                    symbol
                ]


                pending_exits.discard(
                    symbol
                )


        # =====================================================
        # 2. EXECUTE PREVIOUS DAY ENTRY SIGNALS
        # =====================================================

        if (
            CONFIG["EXECUTION"]
            ==
            "NEXT_OPEN"
        ):

            for symbol in list(
                pending_entries.keys()
            ):

                if symbol in positions:

                    del pending_entries[
                        symbol
                    ]

                    continue


                row = day[
                    day["symbol"] == symbol
                ]


                if row.empty:

                    continue


                row = row.iloc[0]


                entry_price = float(
                    row["open"]
                )


                positions[symbol] = {

                    "entry_date":
                        date,

                    "entry_price":
                        entry_price,

                    "last_close":
                        entry_price,

                }


                del pending_entries[
                    symbol
                ]


        # =====================================================
        # 3. DAILY PORTFOLIO RETURN
        # =====================================================
        #
        # Correct daily mark-to-market:
        #
        # previous close/open
        #          ↓
        # today's close
        #
        # We do NOT repeatedly apply the complete
        # entry-to-date return.
        #

        if positions:

            position_returns = []


            for symbol in list(
                positions.keys()
            ):

                row = day[
                    day["symbol"] == symbol
                ]


                if row.empty:

                    continue


                row = row.iloc[0]


                current_close = float(
                    row["close"]
                )


                previous_price = (
                    positions[symbol][
                        "last_close"
                    ]
                )


                daily_return = (

                    current_close
                    /
                    previous_price
                    - 1

                )


                position_returns.append(
                    daily_return
                )


                positions[symbol][
                    "last_close"
                ] = current_close


            if position_returns:

                portfolio_return = (
                    np.mean(
                        position_returns
                    )
                )

            else:

                portfolio_return = 0.0


        else:

            portfolio_return = 0.0


        # Update equity

        equity *= (
            1
            +
            portfolio_return
        )


        equity_curve.append({

            "date":
                date,

            "equity":
                equity,

            "n_holdings":
                len(positions),

        })


        # =====================================================
        # 4. GENERATE EXIT SIGNALS
        # =====================================================
        #
        # Signal uses TODAY'S CLOSE.
        #
        # Rule:
        #
        #     RS Line < RS Line 3-EMA
        #
        # NOT:
        #
        #     crossover
        #
        # =====================================================

        for symbol in list(
            positions.keys()
        ):

            row = day[
                day["symbol"] == symbol
            ]


            if row.empty:

                continue


            row = row.iloc[0]


            if bool(
                row["exit_signal"]
            ):

                if (
                    CONFIG["EXECUTION"]
                    ==
                    "NEXT_OPEN"
                ):

                    pending_exits.add(
                        symbol
                    )


                else:

                    # SAME_CLOSE EXIT

                    exit_price = float(
                        row["close"]
                    )


                    position = positions[
                        symbol
                    ]


                    entry_price = (
                        position[
                            "entry_price"
                        ]
                    )


                    gross_return = (
                        exit_price
                        /
                        entry_price
                        - 1
                    )


                    net_return = (
                        gross_return
                        -
                        (
                            cost_per_side
                            * 2
                        )
                    )


                    trade_log.append({

                        "symbol":
                            symbol,

                        "entry_date":
                            position[
                                "entry_date"
                            ],

                        "exit_date":
                            date,

                        "entry_price":
                            entry_price,

                        "exit_price":
                            exit_price,

                        "gross_return_pct":
                            gross_return * 100,

                        "net_return_pct":
                            net_return * 100,

                        "days_held":
                            (
                                date
                                -
                                position[
                                    "entry_date"
                                ]
                            ).days,

                        "exit_reason":
                            (
                                "RS Line < "
                                f"{CONFIG['RS_EXIT_EMA']}-EMA"
                            ),

                    })


                    del positions[
                        symbol
                    ]


        # =====================================================
        # 5. GENERATE ENTRY SIGNALS
        # =====================================================

        candidates = day[
            day["entry_signal"]
        ].copy()


        # Strongest RS first

        candidates = candidates.sort_values(
            "rs_score",
            ascending=False
        )


        # Current positions

        current_symbols = set(
            positions.keys()
        )


        # Pending entries

        current_symbols.update(
            pending_entries.keys()
        )


        # Available slots

        available_slots = (

            CONFIG["TOP_N"]
            -
            len(current_symbols)

        )


        if available_slots > 0:

            selected = candidates[
                ~candidates[
                    "symbol"
                ].isin(
                    current_symbols
                )
            ].head(
                available_slots
            )


            for _, row in selected.iterrows():

                symbol = row[
                    "symbol"
                ]


                # -------------------------------------------------
                # SAME CLOSE
                # -------------------------------------------------

                if (
                    CONFIG["EXECUTION"]
                    ==
                    "SAME_CLOSE"
                ):

                    entry_price = float(
                        row["close"]
                    )


                    positions[symbol] = {

                        "entry_date":
                            date,

                        "entry_price":
                            entry_price,

                        "last_close":
                            entry_price,

                    }


                # -------------------------------------------------
                # NEXT OPEN
                # -------------------------------------------------

                else:

                    pending_entries[
                        symbol
                    ] = {

                        "signal_date":
                            date,

                    }


    # =========================================================
    # CLOSE OPEN POSITIONS AT BACKTEST END
    # =========================================================

    if dates:

        last_date = pd.Timestamp(
            dates[-1]
        )


        last_day = df[
            df["date"] == last_date
        ]


        for symbol in list(
            positions.keys()
        ):

            row = last_day[
                last_day["symbol"] == symbol
            ]


            if row.empty:

                continue


            row = row.iloc[0]


            exit_price = float(
                row["close"]
            )


            position = positions[
                symbol
            ]


            entry_price = (
                position[
                    "entry_price"
                ]
            )


            gross_return = (
                exit_price
                /
                entry_price
                - 1
            )


            net_return = (
                gross_return
                -
                (
                    cost_per_side
                    * 2
                )
            )


            trade_log.append({

                "symbol":
                    symbol,

                "entry_date":
                    position[
                        "entry_date"
                    ],

                "exit_date":
                    last_date,

                "entry_price":
                    entry_price,

                "exit_price":
                    exit_price,

                "gross_return_pct":
                    gross_return * 100,

                "net_return_pct":
                    net_return * 100,

                "days_held":
                    (
                        last_date
                        -
                        position[
                            "entry_date"
                        ]
                    ).days,

                "exit_reason":
                    "BACKTEST END",

            })


    trades = pd.DataFrame(
        trade_log
    )


    equity_df = pd.DataFrame(
        equity_curve
    )


    return trades, equity_df


# =============================================================
# PERFORMANCE STATISTICS
# =============================================================

def calculate_statistics(
    trades,
    equity_curve
):

    if equity_curve.empty:

        return {}


    equity = (
        equity_curve[
            "equity"
        ]
        .astype(float)
    )


    # ---------------------------------------------------------
    # TOTAL RETURN
    # ---------------------------------------------------------

    total_return = (
        equity.iloc[-1]
        /
        equity.iloc[0]
        - 1
    )


    # ---------------------------------------------------------
    # DRAWDOWN
    # ---------------------------------------------------------

    running_high = (
        equity
        .cummax()
    )


    drawdown = (
        equity
        /
        running_high
        - 1
    )


    max_drawdown = (
        drawdown.min()
    )


    # ---------------------------------------------------------
    # CAGR
    # ---------------------------------------------------------

    start = pd.Timestamp(
        equity_curve[
            "date"
        ].iloc[0]
    )


    end = pd.Timestamp(
        equity_curve[
            "date"
        ].iloc[-1]
    )


    years = (
        end - start
    ).days / 365.25


    if years > 0:

        cagr = (

            equity.iloc[-1]
            /
            equity.iloc[0]

        ) ** (

            1 / years

        ) - 1

    else:

        cagr = np.nan


    # ---------------------------------------------------------
    # TRADE STATISTICS
    # ---------------------------------------------------------

    if not trades.empty:

        winners = trades[
            trades[
                "net_return_pct"
            ] > 0
        ]


        losers = trades[
            trades[
                "net_return_pct"
            ] <= 0
        ]


        win_rate = (
            len(winners)
            /
            len(trades)
        )


        average_trade = (
            trades[
                "net_return_pct"
            ].mean()
        )


        median_trade = (
            trades[
                "net_return_pct"
            ].median()
        )


        average_winner = (

            winners[
                "net_return_pct"
            ].mean()

            if not winners.empty

            else np.nan

        )


        average_loser = (

            losers[
                "net_return_pct"
            ].mean()

            if not losers.empty

            else np.nan

        )


        gross_profit = (
            winners[
                "net_return_pct"
            ].sum()
        )


        gross_loss = abs(
            losers[
                "net_return_pct"
            ].sum()
        )


        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = np.inf


        best_trade = (
            trades[
                "net_return_pct"
            ].max()
        )


        worst_trade = (
            trades[
                "net_return_pct"
            ].min()
        )


        average_days = (
            trades[
                "days_held"
            ].mean()
        )


        median_days = (
            trades[
                "days_held"
            ].median()
        )


        # -----------------------------------------------------
        # MAX CONSECUTIVE LOSSES
        # -----------------------------------------------------

        max_loss_streak = 0

        current_streak = 0


        for result in trades[
            "net_return_pct"
        ]:

            if result <= 0:

                current_streak += 1

                max_loss_streak = max(
                    max_loss_streak,
                    current_streak
                )

            else:

                current_streak = 0


    else:

        winners = pd.DataFrame()

        losers = pd.DataFrame()

        win_rate = np.nan

        average_trade = np.nan

        median_trade = np.nan

        average_winner = np.nan

        average_loser = np.nan

        profit_factor = np.nan

        best_trade = np.nan

        worst_trade = np.nan

        average_days = np.nan

        median_days = np.nan

        max_loss_streak = 0


    # ---------------------------------------------------------
    # DAILY SHARPE
    # ---------------------------------------------------------

    daily_returns = (
        equity
        .pct_change()
        .dropna()
    )


    if (

        len(daily_returns) > 1

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

        sharpe = np.nan


    # ---------------------------------------------------------
    # CALMAR
    # ---------------------------------------------------------

    if max_drawdown < 0:

        calmar = (
            cagr
            /
            abs(max_drawdown)
        )

    else:

        calmar = np.nan


    return {

        "Starting Equity":
            equity.iloc[0],

        "Ending Equity":
            equity.iloc[-1],

        "Total Return %":
            total_return * 100,

        "CAGR %":
            cagr * 100,

        "Max Drawdown %":
            max_drawdown * 100,

        "Calmar Ratio":
            calmar,

        "Daily Sharpe":
            sharpe,

        "Closed Trades":
            len(trades),

        "Winning Trades":
            len(winners),

        "Losing Trades":
            len(losers),

        "Win Rate %":
            win_rate * 100,

        "Average Trade %":
            average_trade,

        "Median Trade %":
            median_trade,

        "Average Winner %":
            average_winner,

        "Average Loser %":
            average_loser,

        "Winner / Loser Ratio":
            (
                abs(
                    average_winner
                    /
                    average_loser
                )
                if (
                    pd.notna(
                        average_winner
                    )
                    and
                    pd.notna(
                        average_loser
                    )
                    and
                    average_loser != 0
                )
                else np.nan
            ),

        "Profit Factor":
            profit_factor,

        "Best Trade %":
            best_trade,

        "Worst Trade %":
            worst_trade,

        "Average Days Held":
            average_days,

        "Median Days Held":
            median_days,

        "Max Consecutive Losses":
            max_loss_streak,

    }


# =============================================================
# EQUITY STRUCTURE
# =============================================================

def identify_equity_structure(
    equity_curve
):

    x = equity_curve.copy()


    x["running_high"] = (
        x["equity"]
        .cummax()
    )


    x["running_low"] = (
        x["equity"]
        .cummin()
    )


    x["higher_high"] = (

        x["equity"]
        >
        x["running_high"]
        .shift(1)

    )


    x["lower_low"] = (

        x["equity"]
        <
        x["running_low"]
        .shift(1)

    )


    if len(x) > 0:

        x.loc[
            x.index[0],
            "higher_high"
        ] = False


        x.loc[
            x.index[0],
            "lower_low"
        ] = False


    x["drawdown"] = (

        x["equity"]
        /
        x["running_high"]
        - 1

    )


    return x


# =============================================================
# PLOT EQUITY CURVE
# =============================================================

def plot_equity_curve(
    equity_curve
):

    x = identify_equity_structure(
        equity_curve
    )


    plt.figure(
        figsize=(14, 7)
    )


    plt.plot(
        x["date"],
        x["equity"],
        linewidth=1.8
    )


    if CONFIG[
        "MARK_HIGHER_HIGHS_LOWER_LOWS"
    ]:

        hh = x[
            x["higher_high"]
        ]


        ll = x[
            x["lower_low"]
        ]


        plt.scatter(
            hh["date"],
            hh["equity"],
            marker="^",
            s=55
        )


        plt.scatter(
            ll["date"],
            ll["equity"],
            marker="v",
            s=55
        )


        # -----------------------------------------------------
        # HH LABELS
        # -----------------------------------------------------

        for _, row in hh.iterrows():

            plt.annotate(

                (
                    f"HH\n"
                    f"{row['date']:%d %b %y}\n"
                    f"{row['equity']:.3f}"
                ),

                (
                    row["date"],
                    row["equity"]
                ),

                xytext=(
                    0,
                    9
                ),

                textcoords=(
                    "offset points"
                ),

                ha="center",

                va="bottom",

                fontsize=7

            )


        # -----------------------------------------------------
        # LL LABELS
        # -----------------------------------------------------

        for _, row in ll.iterrows():

            plt.annotate(

                (
                    f"LL\n"
                    f"{row['date']:%d %b %y}\n"
                    f"{row['equity']:.3f}"
                ),

                (
                    row["date"],
                    row["equity"]
                ),

                xytext=(
                    0,
                    -11
                ),

                textcoords=(
                    "offset points"
                ),

                ha="center",

                va="top",

                fontsize=7

            )


    # Starting equity

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=0.9
    )


    plt.title(
        (
            "RS Screener — "
            "RS Line < "
            f"{CONFIG['RS_EXIT_EMA']}-EMA Exit"
        )
    )


    plt.xlabel(
        "Date"
    )


    plt.ylabel(
        "Equity "
        "(starting equity = 1.00)"
    )


    plt.grid(
        True,
        alpha=0.25
    )


    plt.tight_layout()


    filename = (
        "EquityCurve_"
        f"{CONFIG['RS_EXIT_EMA']}"
        "EMA.png"
    )


    plt.savefig(
        filename,
        dpi=180,
        bbox_inches="tight"
    )


    plt.show()


    print(
        "\nEquity curve saved:"
    )

    print(
        filename
    )


# =============================================================
# EXPORT RESULTS
# =============================================================

def export_results(

    trades,
    equity_curve,
    statistics,
    data

):

    filename = CONFIG[
        "OUTPUT_FILE"
    ]


    structure = (
        identify_equity_structure(
            equity_curve
        )
    )


    summary = pd.DataFrame(

        list(
            statistics.items()
        ),

        columns=[
            "Metric",
            "Value"
        ]

    )


    # ---------------------------------------------------------
    # CONFIG EXPORT
    # ---------------------------------------------------------

    config_rows = []


    for key, value in CONFIG.items():

        config_rows.append({

            "Parameter":
                key,

            "Value":
                value,

        })


    config_df = pd.DataFrame(
        config_rows
    )


    # ---------------------------------------------------------
    # EXCEL
    # ---------------------------------------------------------

    with pd.ExcelWriter(
        filename,
        engine="openpyxl"
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )


        config_df.to_excel(
            writer,
            sheet_name="Config",
            index=False
        )


        trades.to_excel(
            writer,
            sheet_name="Trade Log",
            index=False
        )


        structure.to_excel(
            writer,
            sheet_name="Equity Curve",
            index=False
        )


        data.to_excel(
            writer,
            sheet_name="Signal Data",
            index=False
        )


    print(
        "\nExcel output saved:"
    )

    print(
        filename
    )


# =============================================================
# PRINT RESULTS
# =============================================================

def print_results(
    statistics
):

    print("\n")

    print("=" * 70)

    print(
        "BACKTEST RESULTS"
    )

    print("=" * 70)


    for key, value in statistics.items():

        if isinstance(
            value,
            float
        ):

            if np.isfinite(value):

                print(
                    f"{key:30s}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{key:30s}: "
                    f"{value}"
                )

        else:

            print(
                f"{key:30s}: "
                f"{value}"
            )


    print("=" * 70)


# =============================================================
# MAIN
# =============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "RS SCREENER BACKTEST"
    )

    print("=" * 70)


    # ---------------------------------------------------------
    # CONFIG DISPLAY
    # ---------------------------------------------------------

    print(
        "\nINPUT FILE:"
    )

    print(
        CONFIG["INPUT_FILE"]
    )


    print(
        "\nBACKTEST:"
    )

    print(
        CONFIG["START_DATE"],
        "→",
        CONFIG["END_DATE"]
    )


    print(
        "\nPORTFOLIO:"
    )

    print(
        f"Top N = {CONFIG['TOP_N']}"
    )


    print(
        "\nENTRY:"
    )

    print(
        "RS New High = "
        f"{CONFIG['RS_NEW_HIGH_LOOKBACK']} days"
    )


    print(
        "Minimum RS Score = "
        f"{CONFIG['MIN_RS_SCORE']}"
    )


    print(
        "\nEXIT:"
    )

    print(
        "RS Line < "
        f"RS Line {CONFIG['RS_EXIT_EMA']}-EMA"
    )


    print(
        "This is STATE-BASED, not crossover-based."
    )


    print(
        "\nEXECUTION:"
    )

    print(
        CONFIG["EXECUTION"]
    )


    print(
        "\nCOSTS:"
    )

    print(
        f"Cost per side = "
        f"{CONFIG['COST_PER_SIDE_PCT']}%"
    )


    print(
        f"Slippage per side = "
        f"{CONFIG['SLIPPAGE_PER_SIDE_PCT']}%"
    )


    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    print(
        "\nLoading data..."
    )


    raw_data = load_data()


    print(
        f"Loaded {len(raw_data):,} rows."
    )


    print(
        f"Stocks: "
        f"{raw_data['symbol'].nunique():,}"
    )


    print(
        f"Raw dates: "
        f"{raw_data['date'].min():%d %b %Y}"
        f" → "
        f"{raw_data['date'].max():%d %b %Y}"
    )


    # ---------------------------------------------------------
    # INDICATORS
    # ---------------------------------------------------------

    print(
        "\nCalculating indicators..."
    )


    data = calculate_indicators(
        raw_data
    )


    # ---------------------------------------------------------
    # BACKTEST
    # ---------------------------------------------------------

    print(
        "\nRunning backtest..."
    )


    trades, equity_curve = (
        run_backtest(
            data
        )
    )


    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    statistics = (
        calculate_statistics(
            trades,
            equity_curve
        )
    )


    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    print_results(
        statistics
    )


    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------

    export_results(

        trades,

        equity_curve,

        statistics,

        data

    )


    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------

    if CONFIG[
        "PLOT_EQUITY_CURVE"
    ]:

        print(
            "\nGenerating equity curve..."
        )


        plot_equity_curve(
            equity_curve
        )


    print("\n")

    print("=" * 70)

    print(
        "BACKTEST COMPLETE"
    )

    print("=" * 70)


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":

    main()