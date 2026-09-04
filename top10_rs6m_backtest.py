"""
TOP 10 RS6M ROTATION BACKTEST

PORTFOLIO RULE (every trading day)
  1. Rank all eligible stocks by 6-Month RS.
  2. Target portfolio = today's Top 10 RS stocks.
  3. Existing holdings that remain Top 10 are NEVER resized.
  4. Sell holdings that leave Top 10 or disappear from ranking.
  5. Buy ALL missing names from today's Top 10.
  6. Replacement purchases use AVAILABLE CASH after exits.
  7. Cash is divided across missing Top-10 positions.
  8. Transaction costs are included in affordability calculations.
  9. If >=10 eligible stocks exist and all required stocks are
     affordable, the portfolio MUST finish with exactly 10 holdings.

There is NO daily sell/rebuy, NO continuous equal weighting,
NO rank-11 substitution, NO trend template, NO RS line,
NO sector/regime/breadth filter.
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
BACKTEST_END = None  # None = latest available market data

DOWNLOAD_YEARS_BEFORE_START = 3

MIN_PRICE = 20
MIN_AVG_VOLUME = 100_000
VOLUME_LOOKBACK = 20

RS_PERIOD = 126  # ~6 trading months

TOP_N = 10
STARTING_CAPITAL = 1_000_000
MAX_PLAUSIBLE_DAILY_MOVE = 0.30

# Chart windows (trading days), each rendered as its own
# equity-curve chart rebased to 0% at the window's start.
CHART_WINDOWS = (50, 100, 365)

# ============================================================
# TRANSACTION COSTS (India: NSE equity delivery)
# ============================================================

STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_CHARGE_RATE = 0.0000325
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_FLAT = 20  # per-scrip, sell side only

STCG_RATE = 0.20
STCG_CESS = 0.04
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)  # 20.8%

# ============================================================
# GOOGLE SHEETS
# ============================================================

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest - RS6M Top10"


# ============================================================
# DATE / SERIES HELPERS
# ============================================================

def normalize_dates(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def normalize_series_index(series):
    s = series.copy()
    s.index = normalize_dates(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def get_download_dates():
    start = pd.Timestamp(BACKTEST_START)
    download_start = start - pd.DateOffset(years=DOWNLOAD_YEARS_BEFORE_START)

    if BACKTEST_END is None:
        return download_start.strftime("%Y-%m-%d"), None

    download_end = pd.Timestamp(BACKTEST_END) + pd.Timedelta(days=1)
    return download_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d")


def load_tickers():
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Could not find {STOCKS_FILE}")

    df = pd.read_csv(STOCKS_FILE)
    if "symbol" not in df.columns:
        raise ValueError("stocks.csv must contain a column named 'symbol'.")

    symbols = df["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s]

    output = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    return list(dict.fromkeys(output))  # de-dup, preserve order


def clean_price_series(close):
    """Forward-fills single-day price spikes larger than
    MAX_PLAUSIBLE_DAILY_MOVE. Known limitation: two consecutive
    bad days are each checked against the ORIGINAL series, so a
    genuine two-day move can be partially miscorrected -- flagged
    here rather than silently trusted."""

    close = normalize_series_index(close)
    bad = close.pct_change().abs() > MAX_PLAUSIBLE_DAILY_MOVE
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
    return stt + stamp + exch + sebi + gst


def sell_side_cost(trade_value):
    stt = STT_RATE * trade_value
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)
    return stt + exch + sebi + gst + DP_CHARGE_FLAT


def stcg_tax(net_gain):
    return net_gain * STCG_EFFECTIVE_RATE if net_gain > 0 else 0.0


def max_affordable_qty(price, cash_budget):
    """Largest integer quantity whose trade value + buy-side
    costs fits inside cash_budget. Never assumes
    portfolio_value / N -- always solves from actual cash."""

    if price <= 0 or cash_budget <= 0:
        return 0

    qty = int(cash_budget / price)
    while qty > 0:
        trade_value = qty * price
        if trade_value + buy_side_cost(trade_value) <= cash_budget + 1e-9:
            return qty
        qty -= 1
    return 0


def minimum_cash_for_one_share(price):
    if price <= 0:
        return float("inf")
    return price + buy_side_cost(price)


# ============================================================
# BENCHMARK (calendar only -- not part of the RS calculation)
# ============================================================

def download_benchmark():
    download_start, download_end = get_download_dates()
    print(f"\nBenchmark download: {download_start} -> "
          f"{download_end if download_end else 'LATEST'}")

    for ticker in (BENCHMARK, BENCHMARK_FALLBACK):
        try:
            data = yf.download(ticker, start=download_start, end=download_end,
                                interval="1d", auto_adjust=True, progress=False)
            if data.empty:
                continue

            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().sort_index()
            if close.empty:
                continue

            close, n_bad = clean_price_series(close)
            if n_bad:
                print(f"Benchmark {ticker}: repaired {n_bad} points")
            print(f"Benchmark loaded: {ticker}")
            return close

        except Exception as e:
            print(f"Benchmark {ticker} failed: {e}")

    raise RuntimeError("Could not download benchmark data.")


# ============================================================
# STOCK SIGNAL CALCULATION
# ============================================================

def compute_stock_data(close, volume):
    close = normalize_series_index(close)
    volume = normalize_series_index(volume)
    if len(close) < 300:
        return None

    avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (close > MIN_PRICE) & (avg_volume > MIN_AVG_VOLUME)

    rs_score = (close / close.shift(RS_PERIOD) - 1) * 100

    result = pd.DataFrame({
        "price": close,
        "avg_volume": avg_volume,
        "liquid": liquid,
        "rs_score": rs_score,
    })
    result.index = normalize_dates(result.index)
    return result


def get_row(df, date):
    date = pd.Timestamp(date).normalize()
    if date not in df.index:
        return None
    row = df.loc[date]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def build_daily_ranking(all_stocks, date):
    """Rank all eligible stocks by RS, descending. Rule 1."""

    ranking = []
    for symbol, df in all_stocks.items():
        row = get_row(df, date)
        if row is None:
            continue
        rs = row["rs_score"]
        if pd.isna(rs) or not bool(row["liquid"]):
            continue
        price = row["price"]
        if pd.isna(price) or float(price) <= 0:
            continue
        ranking.append((symbol, float(rs), float(price)))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


# ============================================================
# TRADE EXECUTION
# ============================================================

def execute_buy(symbol, price, qty, date, cash, holdings, trade_log):
    if qty < 1:
        return cash, False

    trade_value = qty * price
    buy_cost = buy_side_cost(trade_value)
    total_required = trade_value + buy_cost
    if total_required > cash + 1e-9:
        return cash, False

    cash -= total_required
    holdings[symbol] = {
        "qty": int(qty),
        "entry_price": float(price),
        "entry_date": date,
        "entry_cost": float(buy_cost),
        "last_price": float(price),
        "last_price_date": date,
    }

    trade_log.append({
        "symbol": symbol, "entry_date": date.strftime("%Y-%m-%d"), "exit_date": "",
        "qty": int(qty), "entry_price": round(price, 4), "exit_price": "",
        "gross_return_pct": "", "buy_cost_rs": round(buy_cost, 2), "sell_cost_rs": "",
        "stcg_tax_rs": "", "net_pnl_rs": "", "net_return_pct": "", "days_held": "",
        "action": "ENTRY", "exit_reason": "",
    })
    return cash, True


def buy_missing_top10(missing_symbols, price_lookup, date, cash, holdings, trade_log):
    """Buy every missing Top-10 name (rules 5-8): reserve enough
    cash for at least one share of every remaining entrant before
    sizing the current one, so an earlier purchase can never starve
    a later one. Cash is otherwise divided ~equally across slots."""

    if not missing_symbols:
        return cash

    for symbol in missing_symbols:
        price = price_lookup.get(symbol)
        if price is None or pd.isna(price) or float(price) <= 0:
            raise RuntimeError(f"{date:%Y-%m-%d}: Top-10 stock {symbol} "
                                f"has invalid entry price.")

    minimum_required = sum(minimum_cash_for_one_share(price_lookup[s])
                            for s in missing_symbols)
    if cash + 1e-9 < minimum_required:
        raise RuntimeError(
            f"{date:%Y-%m-%d}: Cannot fill Top 10. Cash Rs.{cash:,.2f}, but "
            f"minimum cash required for one share each of {len(missing_symbols)} "
            f"missing Top-10 stocks is Rs.{minimum_required:,.2f}. Exact "
            f"10-stock portfolio is impossible without borrowing/fractional shares."
        )

    for i, symbol in enumerate(missing_symbols):
        price = float(price_lookup[symbol])
        remaining = missing_symbols[i + 1:]
        reserve_for_later = sum(minimum_cash_for_one_share(price_lookup[s])
                                 for s in remaining)
        cash_available = cash - reserve_for_later
        slots_remaining = len(missing_symbols) - i

        equal_budget = cash / slots_remaining
        minimum_current = minimum_cash_for_one_share(price)
        target_budget = max(equal_budget, minimum_current)
        target_budget = min(target_budget, cash_available)

        qty = max_affordable_qty(price, target_budget)
        if qty < 1:
            raise RuntimeError(
                f"{date:%Y-%m-%d}: Internal sizing failure for {symbol}. "
                f"Price Rs.{price:,.2f}, budget Rs.{target_budget:,.2f}."
            )

        cash, bought = execute_buy(symbol, price, qty, date, cash, holdings, trade_log)
        if not bought:
            raise RuntimeError(f"{date:%Y-%m-%d}: Buy execution unexpectedly "
                                f"failed for {symbol}.")

    return cash


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(all_stocks, trading_days):
    cash = float(STARTING_CAPITAL)
    holdings = {}
    trade_log = []
    equity_curve = []
    initialized = False
    n_days = len(trading_days)

    for day_number, date in enumerate(trading_days, start=1):
        date = pd.Timestamp(date).normalize()

        ranking = build_daily_ranking(all_stocks, date)
        rank_lookup = {sym: rank for rank, (sym, _, _) in enumerate(ranking, start=1)}
        price_lookup = {sym: float(price) for sym, _, price in ranking}

        eligible_pool_size = len(ranking)
        target_size = min(TOP_N, eligible_pool_size)
        today_top10 = [sym for sym, _, _ in ranking[:TOP_N]]
        today_top10_set = set(today_top10)

        if not initialized:
            if today_top10:
                cash = buy_missing_top10(today_top10, price_lookup, date, cash,
                                          holdings, trade_log)
            initialized = True

        else:
            # -- exits: sell every holding that dropped out of Top 10 --
            exit_symbols = [s for s in holdings if s not in today_top10_set]

            for symbol in exit_symbols:
                position = holdings.pop(symbol)

                if symbol in price_lookup:
                    exit_price = float(price_lookup[symbol])
                    exit_reason = f"RANK_{rank_lookup.get(symbol)}_DROPPED_OUTSIDE_TOP10"
                else:
                    exit_price = float(position.get("last_price", position["entry_price"]))
                    exit_reason = "MISSING_FROM_RANKING_FORCE_EXIT"

                qty = int(position["qty"])
                gross_proceeds = qty * exit_price
                sell_cost = sell_side_cost(gross_proceeds)
                net_proceeds = gross_proceeds - sell_cost
                cost_basis = qty * position["entry_price"] + position["entry_cost"]
                net_gain = net_proceeds - cost_basis
                tax = stcg_tax(net_gain)
                cash += net_proceeds - tax

                gross_return_pct = (exit_price / position["entry_price"] - 1) * 100
                net_pnl = net_gain - tax
                net_return_pct = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0
                days_held = (date - position["entry_date"]).days

                trade_log.append({
                    "symbol": symbol,
                    "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"), "qty": qty,
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "gross_return_pct": round(gross_return_pct, 2),
                    "buy_cost_rs": round(position["entry_cost"], 2),
                    "sell_cost_rs": round(sell_cost, 2),
                    "stcg_tax_rs": round(tax, 2),
                    "net_pnl_rs": round(net_pnl, 2),
                    "net_return_pct": round(net_return_pct, 2),
                    "days_held": days_held, "action": "EXIT",
                    "exit_reason": exit_reason,
                })

            # -- mark retained holdings, then buy every missing Top-10 name --
            for symbol, position in holdings.items():
                if symbol in price_lookup:
                    position["last_price"] = float(price_lookup[symbol])
                    position["last_price_date"] = date

            missing_top10 = [s for s in today_top10 if s not in holdings]
            if missing_top10:
                cash = buy_missing_top10(missing_top10, price_lookup, date, cash,
                                          holdings, trade_log)

        # -- hard portfolio invariants (rule 9) --
        held_symbols = set(holdings.keys())
        expected_symbols = set(today_top10)

        illegal_holdings = held_symbols - expected_symbols
        if illegal_holdings:
            raise RuntimeError(f"{date:%Y-%m-%d}: Portfolio contains stocks "
                                f"outside today's Top 10: {sorted(illegal_holdings)}")

        missing_after_rebalance = expected_symbols - held_symbols
        if missing_after_rebalance:
            raise RuntimeError(
                f"{date:%Y-%m-%d}: REBALANCE FAILURE. Missing Top-10 stocks "
                f"after trading: {sorted(missing_after_rebalance)}. "
                f"Holdings={len(holdings)}, Target={target_size}, Cash=Rs.{cash:,.2f}"
            )

        if len(holdings) != target_size:
            raise RuntimeError(f"{date:%Y-%m-%d}: HOLDING COUNT FAILURE. "
                                f"Expected {target_size}, found {len(holdings)}.")

        if eligible_pool_size >= TOP_N and len(holdings) != TOP_N:
            raise RuntimeError(
                f"{date:%Y-%m-%d}: Expected exactly {TOP_N} holdings because "
                f"eligible pool contains {eligible_pool_size} stocks. "
                f"Actual holdings={len(holdings)}."
            )

        # -- daily mark-to-market --
        total_value = float(cash)
        for symbol, position in holdings.items():
            if symbol in price_lookup:
                mark_price = float(price_lookup[symbol])
                position["last_price"] = mark_price
                position["last_price_date"] = date
            else:
                mark_price = float(position.get("last_price", position["entry_price"]))
            total_value += position["qty"] * mark_price

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(total_value, 2),
            "cash_rs": round(cash, 2),
            "invested_value_rs": round(total_value - cash, 2),
            "equity_multiple": round(total_value / STARTING_CAPITAL, 8),
            "n_holdings": len(holdings),
            "eligible_pool_size": eligible_pool_size,
            "top10_target_size": target_size,
        })

        if day_number % 100 == 0:
            print(f"Processed {day_number}/{n_days} | {date:%Y-%m-%d} | "
                  f"EligiblePool={eligible_pool_size} | Holdings={len(holdings)} | "
                  f"Cash=Rs.{cash:,.0f} | Equity=Rs.{total_value:,.0f}")

    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df = _add_equity_analytics_columns(equity_df)

    trade_df = pd.DataFrame(trade_log)

    final_marked_value = (float(equity_df["portfolio_value_rs"].iloc[-1])
                           if not equity_df.empty else STARTING_CAPITAL)

    open_df, final_liquidation_value = _liquidate_open_positions(
        all_stocks, trading_days, holdings, cash
    )

    return equity_df, trade_df, open_df, final_marked_value, final_liquidation_value


def _add_equity_analytics_columns(equity_df):
    """Adds drawdown and normalised-to-zero equity-curve columns.

    equity_curve_pct_norm is the cumulative % move of the total
    portfolio value since day 1 of the whole backtest (same
    information as equity_multiple, expressed as % starting at 0
    instead of a multiple starting at 1). One additional column is
    added per entry in CHART_WINDOWS, each rebased to 0% at the
    start of its own rolling window (blank before that window).
    """

    running_max = equity_df["equity_multiple"].cummax()
    equity_df["drawdown_pct"] = ((equity_df["equity_multiple"] / running_max - 1)
                                  * 100).round(3)

    first_value = float(equity_df["portfolio_value_rs"].iloc[0])
    equity_df["equity_curve_pct_norm"] = (
        (equity_df["portfolio_value_rs"] / first_value - 1) * 100
    ).round(3)

    for window_days in CHART_WINDOWS:
        window_start = max(len(equity_df) - window_days, 0)
        base_value = float(equity_df["portfolio_value_rs"].iloc[window_start])

        # NaN (not "") keeps the column numeric end-to-end; newer
        # pandas can infer a strict string dtype for a column first
        # populated with "", which then rejects numeric values
        # written into it afterwards. sanitize_for_sheets() blanks
        # any remaining NaN to "" at write time.
        series = pd.Series(np.nan, index=equity_df.index, dtype=float)
        series.iloc[window_start:] = (
            (equity_df["portfolio_value_rs"].iloc[window_start:] / base_value - 1) * 100
        ).round(3)
        equity_df[f"equity_curve_pct_norm_last{window_days}"] = series

    return equity_df


def _liquidate_open_positions(all_stocks, trading_days, holdings, cash):
    liquidation_cash = float(cash)
    open_positions = []

    if len(trading_days) == 0:
        return pd.DataFrame(open_positions), liquidation_cash

    last_date = pd.Timestamp(trading_days[-1]).normalize()
    final_price_lookup = {sym: float(price) for sym, _, price
                           in build_daily_ranking(all_stocks, last_date)}

    for symbol, position in holdings.items():
        exit_price = final_price_lookup.get(
            symbol, float(position.get("last_price", position["entry_price"]))
        )
        gross_proceeds = position["qty"] * exit_price
        sell_cost = sell_side_cost(gross_proceeds)
        net_proceeds = gross_proceeds - sell_cost
        cost_basis = position["qty"] * position["entry_price"] + position["entry_cost"]
        net_gain = net_proceeds - cost_basis
        tax = stcg_tax(net_gain)
        liquidation_cash += net_proceeds - tax

        open_positions.append({
            "symbol": symbol,
            "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
            "qty": position["qty"],
            "entry_price": round(position["entry_price"], 4),
            "last_price": round(exit_price, 4),
            "gross_return_pct": round((exit_price / position["entry_price"] - 1) * 100, 2),
        })

    return pd.DataFrame(open_positions), liquidation_cash


# ============================================================
# SUMMARY
# ============================================================

def summarize(equity_df, trade_df, final_marked_value, final_liquidation_value):
    if equity_df.empty:
        return {}

    marked_return = (final_marked_value / STARTING_CAPITAL - 1) * 100
    liquidation_return = (final_liquidation_value / STARTING_CAPITAL - 1) * 100
    max_dd = float(equity_df["drawdown_pct"].min())

    daily_returns = equity_df["equity_multiple"].pct_change().dropna()
    if len(daily_returns) > 1:
        daily_mean, daily_std = daily_returns.mean(), daily_returns.std()
        n_days = len(equity_df)
        annualized_return = equity_df["equity_multiple"].iloc[-1] ** (252 / max(n_days, 1)) - 1
        annualized_vol = daily_std * np.sqrt(252)
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 0
        sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        annualized_return = annualized_vol = sharpe = sortino = 0

    calmar = (annualized_return / abs(max_dd / 100)) if max_dd != 0 else 0

    if not trade_df.empty:
        entries = trade_df[trade_df["action"] == "ENTRY"]
        exits = trade_df[trade_df["action"] == "EXIT"]

        total_costs = exits["sell_cost_rs"].fillna(0).sum() if not exits.empty else 0
        total_costs += entries["buy_cost_rs"].fillna(0).sum() if not entries.empty else 0
        total_tax = exits["stcg_tax_rs"].fillna(0).sum() if not exits.empty else 0

        if not exits.empty:
            net_returns = exits["net_return_pct"].astype(float)
            win_rate = (net_returns > 0).mean() * 100
            avg_trade = net_returns.mean()
            median_trade = net_returns.median()

            winners = exits[net_returns > 0]
            losers = exits[net_returns < 0]
            avg_winner = winners["net_return_pct"].astype(float).mean() if not winners.empty else 0
            avg_loser = losers["net_return_pct"].astype(float).mean() if not losers.empty else 0

            gross_profit = winners["net_pnl_rs"].astype(float).sum() if not winners.empty else 0
            gross_loss = abs(losers["net_pnl_rs"].astype(float).sum()) if not losers.empty else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            avg_days = exits["days_held"].astype(float).mean()
        else:
            win_rate = avg_trade = median_trade = 0
            avg_winner = avg_loser = profit_factor = avg_days = 0
    else:
        entries, exits = pd.DataFrame(), pd.DataFrame()
        total_costs = total_tax = 0
        win_rate = avg_trade = median_trade = 0
        avg_winner = avg_loser = profit_factor = avg_days = 0

    return {
        "Backtest Start": BACKTEST_START,
        "Backtest End": equity_df["date"].iloc[-1],
        "Starting Capital (Rs)": STARTING_CAPITAL,
        "Final Marked Value (Rs)": round(final_marked_value, 0),
        "Final Liquidation Value (Rs)": round(final_liquidation_value, 0),
        "Net Return - Marked (%)": round(marked_return, 2),
        "Net Return - Liquidation (%)": round(liquidation_return, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Maximum Drawdown (%)": round(max_dd, 2),
        "Closed Trades": len(exits),
        "Entries": len(entries),
        "Win Rate Net (%)": round(win_rate, 2),
        "Average Net Trade (%)": round(avg_trade, 2),
        "Median Net Trade (%)": round(median_trade, 2),
        "Average Winner (%)": round(avg_winner, 2),
        "Average Loser (%)": round(avg_loser, 2),
        "Profit Factor": round(profit_factor, 3),
        "Average Days Held": round(avg_days, 1),
        "Total Transaction Costs (Rs)": round(total_costs, 0),
        "Total STCG Tax (Rs)": round(total_tax, 0),
        "RS Formula": "6M Price Rate-of-Change (100% weight)",
        "Portfolio": "Exact daily Top 10 RS membership",
        "Weight": "Available replacement cash divided across missing Top-10 "
                  "names; retained positions untouched",
        "Entry": "Initial Top 10; subsequently every missing Top-10 stock",
        "Exit": "Rank 11+ or missing from eligible ranking",
        "Execution": "Same-day close (T+0)",
        "Rebalance Frequency": "Daily membership check; no resizing of "
                                "retained positions",
        "Price Filter": f"> Rs.{MIN_PRICE}",
        "Liquidity Filter": f"{VOLUME_LOOKBACK}D average volume > {MIN_AVG_VOLUME:,}",
        "Other Filters": "NONE",
    }


# ============================================================
# GOOGLE SHEETS
# ============================================================

def sanitize_for_sheets(df):
    """NaN/+-Infinity are not valid JSON, and gspread's ws.update()
    will raise InvalidJSONError if either reaches it. Blanks such
    cells to "" rather than letting one bad value fail the whole
    upload -- protects any numeric column, present now or added
    later, that could end up with NaN/inf before enough data exists."""

    if df.empty:
        return df
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.where(pd.notnull(clean), "")


def sanitize_scalar(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    return v


def get_or_create_worksheet(sh, title, rows=1000, cols=16):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        pass
    try:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)
    except gspread.exceptions.APIError as e:
        if "already exists" in str(e):
            return sh.worksheet(title)
        raise


def write_in_chunks(ws, all_rows, start_row, chunk_size, label,
                     max_retries=6, initial_retry_seconds=5):
    total = len(all_rows)
    if total == 0:
        return

    for i in range(0, total, chunk_size):
        chunk = all_rows[i:i + chunk_size]
        row_start = start_row + i

        for attempt in range(max_retries):
            try:
                ws.update(chunk, f"A{row_start}")
                break
            except Exception as e:
                is_quota = "429" in str(e) or "Quota exceeded" in str(e)
                if not is_quota or attempt == max_retries - 1:
                    print(f"Write failed for {label} rows {i}-{i + len(chunk)}: {e}")
                    raise
                wait = initial_retry_seconds * (2 ** attempt)
                print(f"Google quota for {label}. Waiting {wait}s...")
                time.sleep(wait)

        print(f"Wrote {label}: {min(i + chunk_size, total)}/{total} rows")


def remove_existing_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()
        requests = [
            {"deleteEmbeddedObject": {"objectId": chart["chartId"]}}
            for sheet in meta.get("sheets", [])
            if sheet["properties"]["sheetId"] == sheet_id
            for chart in sheet.get("charts", [])
        ]
        if requests:
            sh.batch_update({"requests": requests})
            print(f"Removed {len(requests)} existing chart(s).")
    except Exception as e:
        print(f"Could not check/remove existing charts (non-fatal): {e}")


def add_charts(sh, sheet_id, equity_header_row_0idx, n_equity_rows, equity_columns):
    """All series are addressed by COLUMN NAME (via equity_columns,
    the equity_df.columns actually written to the sheet) rather than
    hardcoded integer positions -- if a column is ever added, removed,
    or reordered upstream, a chart will raise KeyError instead of
    silently plotting the wrong series."""

    col_idx = {name: i for i, name in enumerate(equity_columns)}
    data_end_row = equity_header_row_0idx + 1 + n_equity_rows

    def window_start_row(window_days):
        return equity_header_row_0idx + 1 + max(n_equity_rows - window_days, 0)

    def make_chart(title, y_col_name, y_axis_title, anchor_row, chart_type="LINE",
                    start_row_override=None, show_points=False, width_pixels=650):
        y_col = col_idx[y_col_name]
        series_start = (start_row_override if start_row_override is not None
                         else equity_header_row_0idx)

        series_entry = {
            "series": {"sourceRange": {"sources": [{
                "sheetId": sheet_id, "startRowIndex": series_start,
                "endRowIndex": data_end_row, "startColumnIndex": y_col,
                "endColumnIndex": y_col + 1,
            }]}},
            "targetAxis": "LEFT_AXIS",
        }

        if show_points:
            # Dot on every end-of-day point plus that point's own
            # value printed below it. The Sheets Charts API has no
            # rotate/vertical-text property for data labels --
            # "placement" only accepts ABOVE/BELOW/LEFT/RIGHT/
            # CENTER/INSIDE_END -- so BELOW + a small font is the
            # closest available fix for reducing label overlap.
            series_entry["pointStyle"] = {"size": 5, "shape": "CIRCLE"}
            series_entry["dataLabel"] = {
                "type": "DATA", "placement": "BELOW",
                "textFormat": {"fontSize": 7},
            }

        return {"addChart": {"chart": {
            "spec": {
                "title": title,
                "basicChart": {
                    "chartType": chart_type,
                    "legendPosition": "NO_LEGEND",
                    "axis": [
                        {"position": "BOTTOM_AXIS", "title": "Date"},
                        {"position": "LEFT_AXIS", "title": y_axis_title},
                    ],
                    "domains": [{"domain": {"sourceRange": {"sources": [{
                        "sheetId": sheet_id, "startRowIndex": series_start,
                        "endRowIndex": data_end_row, "startColumnIndex": 0,
                        "endColumnIndex": 1,
                    }]}}}],
                    "series": [series_entry],
                },
            },
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": sheet_id, "rowIndex": anchor_row, "columnIndex": 8},
                "widthPixels": width_pixels, "heightPixels": 380,
            }},
        }}}

    requests = [
        make_chart("Equity Curve (Rs)", "portfolio_value_rs",
                    "Portfolio Value (Rs)", equity_header_row_0idx),

        make_chart("Drawdown (%)", "drawdown_pct", "Drawdown %",
                    equity_header_row_0idx + 22),

        make_chart("Eligible Pool Size", "eligible_pool_size", "Stock Count",
                    equity_header_row_0idx + 44),

        # Since inception -- same start row as the equity curve
        # itself, so it is never truncated.
        make_chart("Equity Curve - Normalised to Zero (%, Since Inception)",
                    "equity_curve_pct_norm", "Cumulative Change %",
                    equity_header_row_0idx + 66),

        # Last 50 days -- dot + label per point, widened for room.
        make_chart("Equity Curve - Normalised to Zero (%, Last 50 Days)",
                    "equity_curve_pct_norm_last50", "Cumulative Change %",
                    equity_header_row_0idx + 88,
                    start_row_override=window_start_row(50),
                    show_points=True, width_pixels=1100),

        make_chart("Equity Curve - Normalised to Zero (%, Last 100 Days)",
                    "equity_curve_pct_norm_last100", "Cumulative Change %",
                    equity_header_row_0idx + 110,
                    start_row_override=window_start_row(100)),

        make_chart("Equity Curve - Normalised to Zero (%, Last 365 Days)",
                    "equity_curve_pct_norm_last365", "Cumulative Change %",
                    equity_header_row_0idx + 132,
                    start_row_override=window_start_row(365)),
    ]

    try:
        sh.batch_update({"requests": requests})
        print("Equity, drawdown, pool-size, and normalised-to-zero equity "
              "curve charts (since-inception + last-50/100/365-day) added.")
    except Exception as e:
        print(f"Could not add charts (non-fatal): {e}")


def write_to_sheet(trade_df, equity_df, open_df, summary, effective_end_str):
    sheet_id = os.environ.get(SHEET_ID_ENV)
    creds_json = os.environ.get(CREDS_ENV)

    if not sheet_id or not creds_json:
        print("Missing SHEET_ID/GOOGLE_CREDENTIALS -- saving to CSV instead.")
        trade_df.to_csv("RS6M_Trade_Log.csv", index=False)
        equity_df.to_csv("RS6M_Equity_Curve.csv", index=False)
        if not open_df.empty:
            open_df.to_csv("RS6M_Open_Positions.csv", index=False)
        return

    creds = Credentials.from_service_account_info(
        json.loads(creds_json), scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    # Sheet must be wide enough for the widest section actually
    # being written -- not a fixed guess. Previously hardcoded to
    # 16, which happened to be enough today but would silently
    # truncate/error the moment any section grew past it.
    n_cols_needed = max(
        len(trade_df.columns) if not trade_df.empty else 0,
        len(equity_df.columns) if not equity_df.empty else 0,
        len(open_df.columns) if not open_df.empty else 0,
        2,  # summary is always 2 columns (key, value)
    )
    n_rows_needed = len(trade_df) + len(equity_df) + len(open_df) + len(summary) + 60

    ws = get_or_create_worksheet(sh, BACKTEST_WORKSHEET,
                                  rows=n_rows_needed, cols=n_cols_needed)
    if ws.row_count < n_rows_needed or ws.col_count < n_cols_needed:
        ws.resize(rows=max(ws.row_count, n_rows_needed),
                   cols=max(ws.col_count, n_cols_needed))

    remove_existing_charts(sh, ws.id)
    ws.clear()

    ws.update([[
        "TOP 10 RS6M ROTATION BACKTEST | "
        f"run {timestamp} | NET of costs+STCG | "
        f"Capital: Rs.{STARTING_CAPITAL:,.0f} | "
        "Target: exact Top 10 6M-RS membership | "
        "Retained Top-10 holdings NOT resized | Sell rank 11+ / missing | "
        "Available exit cash funds all missing Top-10 names | Same EOD bar | "
        f"Window: {BACKTEST_START} to {effective_end_str}"
    ]], "A1")

    summary_rows = [["Summary", ""]] + [[k, sanitize_scalar(v)] for k, v in summary.items()]
    ws.update(summary_rows, "A3")

    trade_start_row = 3 + len(summary_rows) + 2
    ws.update([["Trade Log"]], f"A{trade_start_row}")
    trade_header_row = trade_start_row + 1

    if not trade_df.empty:
        trade_df_clean = sanitize_for_sheets(trade_df)
        write_in_chunks(ws, [list(trade_df_clean.columns)] + trade_df_clean.values.tolist(),
                         start_row=trade_header_row, chunk_size=2000, label="trade log")

    open_start_row = trade_header_row + len(trade_df) + 3
    ws.update([["Open Positions at Backtest End (mark-to-market)"]], f"A{open_start_row}")
    open_header_row = open_start_row + 1

    if not open_df.empty:
        open_df_clean = sanitize_for_sheets(open_df)
        ws.update([list(open_df_clean.columns)] + open_df_clean.values.tolist(),
                  f"A{open_header_row}")

    equity_start_row = open_header_row + max(len(open_df), 1) + 3
    ws.update([["Daily Equity Curve"]], f"A{equity_start_row}")
    equity_header_row = equity_start_row + 1

    if not equity_df.empty:
        equity_df_clean = sanitize_for_sheets(equity_df)
        write_in_chunks(ws, [list(equity_df_clean.columns)] + equity_df_clean.values.tolist(),
                         start_row=equity_header_row, chunk_size=2000, label="equity curve")

        add_charts(sh, ws.id, equity_header_row - 1, len(equity_df),
                   equity_columns=list(equity_df_clean.columns))

    print(f"\nBacktest results written to '{BACKTEST_WORKSHEET}' tab: "
          f"{len(trade_df)} trades, {len(equity_df)} trading days, "
          f"{len(open_df)} open positions.")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("TOP 10 RS6M ROTATION BACKTEST")
    print("=" * 70)
    print(f"Backtest start : {BACKTEST_START}")
    print(f"Backtest end   : {BACKTEST_END if BACKTEST_END else 'LATEST'}")
    print("Ranking        : DAILY 6-MONTH RS SCORE ONLY")
    print("Portfolio      : EXACT TOP 10 RS MEMBERSHIP")
    print("Initial        : BUY ALL TOP 10")
    print("Rotation       : SELL ONLY STOCKS LEAVING TOP 10")
    print("New entries    : BUY EVERY MISSING TOP-10 STOCK")
    print("Existing names : HOLD / NO RESIZING")
    print("Sizing         : AVAILABLE CASH / MISSING SLOTS")
    print("Execution      : SAME EOD BAR (T+0)")
    print(f"Price filter   : > Rs.{MIN_PRICE}")
    print(f"Liquidity      : {VOLUME_LOOKBACK}D average volume > {MIN_AVG_VOLUME:,}")
    print("Other filters  : NONE")
    print("=" * 70)

    tickers = load_tickers()
    print(f"\nLoaded {len(tickers)} tickers.")

    download_start, download_end = get_download_dates()
    print(f"Download start: {download_start}")
    print(f"Download end: {download_end if download_end else 'LATEST'}")

    bench_close = download_benchmark()
    bench_close.index = normalize_dates(bench_close.index)

    all_stocks = {}
    total_bad_points = 0
    batch_size = 50

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        print(f"\nDownloading {start + 1}-{start + len(batch)} of {len(tickers)}")

        try:
            data = yf.download(batch, start=download_start, end=download_end,
                                interval="1d", auto_adjust=True, progress=False,
                                group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch failed: {e}")
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    sdata = data
                else:
                    if not isinstance(data.columns, pd.MultiIndex):
                        continue
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    sdata = data[symbol]

                if "Close" not in sdata.columns:
                    continue

                close = sdata["Close"].dropna().sort_index()
                if close.empty:
                    continue

                volume = sdata["Volume"].reindex(close.index).fillna(0)
                close, n_bad = clean_price_series(close)
                total_bad_points += n_bad

                stock_data = compute_stock_data(close, volume)
                if stock_data is None:
                    continue

                all_stocks[symbol.replace(".NS", "")] = stock_data

            except Exception as e:
                print(f"Skipping {symbol}: {e}")

        time.sleep(1)

    print(f"\nStocks with usable data: {len(all_stocks)}")
    print(f"Repaired data points: {total_bad_points}")

    if not all_stocks:
        raise RuntimeError("No usable stock data.")

    latest_stock_date = max(df.index.max() for df in all_stocks.values())
    latest_benchmark_date = bench_close.index.max()

    if BACKTEST_END is None:
        effective_end = min(pd.Timestamp(latest_stock_date).normalize(),
                             pd.Timestamp(latest_benchmark_date).normalize())
    else:
        effective_end = min(pd.Timestamp(BACKTEST_END).normalize(),
                             pd.Timestamp(latest_stock_date).normalize(),
                             pd.Timestamp(latest_benchmark_date).normalize())

    print(f"\nLatest stock date: {pd.Timestamp(latest_stock_date):%Y-%m-%d}")
    print(f"Latest benchmark date: {pd.Timestamp(latest_benchmark_date):%Y-%m-%d}")
    print(f"Effective end: {effective_end:%Y-%m-%d}")

    trading_days = bench_close.index[
        (bench_close.index >= pd.Timestamp(BACKTEST_START).normalize())
        & (bench_close.index <= effective_end)
    ]
    trading_days = pd.DatetimeIndex(trading_days).drop_duplicates().sort_values()

    print(f"\nTrading days: {len(trading_days)}")
    if len(trading_days) == 0:
        raise RuntimeError("No trading days found.")
    print(f"First day: {trading_days[0]:%Y-%m-%d}")
    print(f"Last day: {trading_days[-1]:%Y-%m-%d}")

    print("\nRunning backtest...")
    equity_df, trade_df, open_df, final_marked, final_liq = run_backtest(
        all_stocks, trading_days
    )

    if not equity_df.empty:
        broken = equity_df[equity_df["n_holdings"] != equity_df["top10_target_size"]]
        if not broken.empty:
            print()
            print(broken.to_string())
            raise RuntimeError(f"PORTFOLIO AUDIT FAILED: {len(broken)} trading days "
                                f"did not contain the required number of holdings.")

        full_pool = equity_df[equity_df["eligible_pool_size"] >= TOP_N]
        non_ten = full_pool[full_pool["n_holdings"] != TOP_N]
        if not non_ten.empty:
            print()
            print(non_ten.to_string())
            raise RuntimeError("TOP-10 AUDIT FAILED: at least one trading day had "
                                "10+ eligible stocks but did not hold exactly 10.")

        print("\nPORTFOLIO AUDIT PASSED.")
        print("Every day held exactly the required Top-10 target count.")

    summary = summarize(equity_df, trade_df, final_marked, final_liq)

    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 70)

    write_to_sheet(trade_df, equity_df, open_df, summary, effective_end.strftime("%Y-%m-%d"))

    equity_df.to_csv("RS6M_Equity_Curve.csv", index=False)
    trade_df.to_csv("RS6M_Trade_Log.csv", index=False)
    if not open_df.empty:
        open_df.to_csv("RS6M_Open_Positions.csv", index=False)

    print("\nCSV files also saved.")
    print("\nBACKTEST COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        print("=" * 70)
        print("BACKTEST FAILED")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        raise
