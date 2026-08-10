"""
RS Screener Backtest v3 -- Enhanced Breadth Integration & Execution Dynamics
Includes: Point-in-time Market Breadth Engine, Next-Day Open Execution,
ATR Volatility Parity Sizing, and Dynamic Friction Modeling.
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
# CONFIGURATION & PARAMETERS
# ============================================================

BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"

LOOKBACK_DAYS = 250                
DOWNLOAD_YEARS_BEFORE_START = 3    

STOCKS_FILE = "stocks.csv"
TOP_N = 10                         

# Liquidity Parameters
MIN_PRICE = 10                     
MIN_AVG_VOLUME = 10000             
VOLUME_LOOKBACK = 20               

MAX_PLAUSIBLE_DAILY_MOVE = 0.30    

BACKTEST_START = "2016-04-01"      
BACKTEST_END = "2026-08-07"        

STARTING_CAPITAL = 1_000_000       

# Market Breadth Regime Framework (50-DMA Universe Participation)
ENABLE_REGIME_FILTER = True
BREADTH_RISK_ON = 0.60              # >= 60% Universe > 50-DMA -> Risk-On (100% Allocation)
BREADTH_RISK_CAUTION = 0.40         # 40% - 60% -> Caution (50% Allocation)
BREADTH_CIRCUIT_BREAKER = 0.25      # < 25% -> Circuit Breaker (Liquidate to Cash)

# Execution & Cost Parameters
SLIPPAGE_BPS = 10                   # 10 bps estimated execution slippage
ENABLE_COSTS = True
STT_RATE = 0.001                   
STAMP_DUTY_RATE = 0.00015          
EXCHANGE_CHARGE_RATE = 0.0000325   
SEBI_CHARGE_RATE = 0.000001        
GST_RATE = 0.18                    
DP_CHARGE_FLAT = 20                

ENABLE_STCG = True
STCG_RATE = 0.20                   
STCG_CESS = 0.04                   
STCG_EFFECTIVE_RATE = STCG_RATE * (1 + STCG_CESS)

SHEET_ID_ENV = "SHEET_ID"
CREDS_ENV = "GOOGLE_CREDENTIALS"
BACKTEST_WORKSHEET = "Backtest"
COMPARISON_WORKSHEET = "Backtest_Exit_Comparison"

EXIT_VARIANTS = [
    {"name": "RS<3EMA (state, original)",  "type": "rs_ema_state", "span": 3},
    {"name": "RS<3EMA (crossover)",         "type": "rs_ema_cross", "span": 3},
    {"name": "RS<10EMA (crossover)",        "type": "rs_ema_cross", "span": 10},
    {"name": "RS<20EMA (crossover)",        "type": "rs_ema_cross", "span": 20},
    {"name": "RS<20EMA (state)",            "type": "rs_ema_state", "span": 20},
    {"name": "Rank buffer only (buf=20)",   "type": "rank_buffer",  "buffer": 20},
    {"name": "Trend Template fail (cross)", "type": "tt_fail_cross"},
]
PRIMARY_VARIANT_INDEX = 3   

# ============================================================
# HELPER FUNCTIONS & COMPUTATIONAL MODULES
# ============================================================

def calculate_costs(trade_value, side="BUY"):
    """Computes friction metrics including statutory taxes and market impact."""
    if not ENABLE_COSTS or trade_value <= 0:
        return 0.0
    
    slippage = trade_value * (SLIPPAGE_BPS / 10000.0)
    exch = EXCHANGE_CHARGE_RATE * trade_value
    sebi = SEBI_CHARGE_RATE * trade_value
    gst = GST_RATE * (exch + sebi)
    stt = STT_RATE * trade_value
    
    if side == "BUY":
        stamp = STAMP_DUTY_RATE * trade_value
        return stt + stamp + exch + sebi + gst + slippage
    else:
        return stt + exch + sebi + gst + DP_CHARGE_FLAT + slippage

def stcg_tax(net_gain):
    """Calculates Section 111A Short-Term Capital Gains Tax Liability."""
    if not ENABLE_STCG or net_gain <= 0:
        return 0.0
    return net_gain * STCG_EFFECTIVE_RATE

def clean_price_series(close):
    """Identifies and adjusts non-economic anomalous pricing jumps."""
    close = close.copy().sort_index()
    pct_change = close.pct_change()
    bad = pct_change.abs() > MAX_PLAUSIBLE_DAILY_MOVE
    n_bad = bad.sum()
    if n_bad > 0:
        cleaned = close.copy()
        for idx in close.index[bad]:
            pos = cleaned.index.get_loc(idx)
            if pos > 0:
                cleaned.iloc[pos] = cleaned.iloc[pos - 1]
        return cleaned, int(n_bad)
    return close, 0

def trend_template_series(s):
    """Evaluates Mark Minervini Trend Template conditions (7/7 criteria)."""
    sma50 = s.rolling(50).mean()
    sma150 = s.rolling(150).mean()
    sma200 = s.rolling(200).mean()
    sma200_1mo = sma200.shift(21)
    low52 = s.rolling(252).min()
    high52 = s.rolling(252).max()

    c1 = (s > sma150) & (s > sma200)
    c2 = sma150 > sma200
    c3 = sma200 > sma200_1mo
    c4 = (sma50 > sma150) & (sma50 > sma200)
    c5 = s > sma50
    c6 = s >= 1.25 * low52
    c7 = s >= 0.75 * high52

    met = (c1.astype(int) + c2.astype(int) + c3.astype(int) + 
           c4.astype(int) + c5.astype(int) + c6.astype(int) + c7.astype(int))
    return met == 7, met

def compute_signals_for_stock(df_stock, bench_close):
    """Computes trend metrics, Relative Strength parameters, and execution vectors."""
    close = df_stock["Close"]
    volume = df_stock["Volume"]
    open_p = df_stock["Open"] if "Open" in df_stock.columns else close

    aligned = pd.concat([close, open_p, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["s", "open", "b"]
    if len(aligned) < 280:
        return None

    volume = volume.reindex(aligned.index)
    rs_ratio = aligned["s"] / aligned["b"]

    def pct_return(series, days):
        return series / series.shift(days) - 1

    rs_score = (0.40 * pct_return(aligned["s"], 63) +
                0.20 * pct_return(aligned["s"], 126) +
                0.20 * pct_return(aligned["s"], 189) +
                0.20 * pct_return(aligned["s"], 252)) * 100

    previous_rs_high = rs_ratio.shift(1).rolling(LOOKBACK_DAYS).max()
    blue_dot = rs_ratio > previous_rs_high

    tt_pass, _ = trend_template_series(aligned["s"])
    rs_tt_pass, _ = trend_template_series(rs_ratio)

    rolling_avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    liquid = (aligned["s"] >= MIN_PRICE) & (rolling_avg_volume >= MIN_AVG_VOLUME)

    sma50 = aligned["s"].rolling(50).mean()
    above_50dma = aligned["s"] > sma50

    # True Range for ATR volatility sizing
    high = df_stock["High"].reindex(aligned.index) if "High" in df_stock.columns else aligned["s"]
    low = df_stock["Low"].reindex(aligned.index) if "Low" in df_stock.columns else aligned["s"]
    tr = np.maximum(high - low, np.maximum(abs(high - aligned["s"].shift(1)), abs(low - aligned["s"].shift(1))))
    atr14 = tr.rolling(14).mean()

    out = pd.DataFrame({
        "price": aligned["s"],
        "open_price": aligned["open"],
        "rs_line": rs_ratio,
        "rs_score": rs_score,
        "blue_dot": blue_dot,
        "tt_pass": tt_pass,
        "rs_tt_pass": rs_tt_pass,
        "liquid": liquid,
        "above_50dma": above_50dma,
        "atr14": atr14
    })

    spans_needed = {v["span"] for v in EXIT_VARIANTS if "span" in v}
    for span in spans_needed:
        ema = rs_ratio.ewm(span=span, adjust=False).mean()
        below = rs_ratio < ema
        out[f"rs_below_ema{span}"] = below
        out[f"rs_cross_below_ema{span}"] = below & (~below.shift(1).fillna(False))

    out["tt_fail_cross"] = (~tt_pass) & tt_pass.shift(1).fillna(False)
    return out

def compute_daily_breadth(all_signals, trading_days):
    """Point-in-time calculation of universe breadth metrics."""
    rows = []
    for date in trading_days:
        flags = []
        for sym, df in all_signals.items():
            if date in df.index:
                val = df.loc[date, "above_50dma"]
                if pd.notna(val):
                    flags.append(bool(val))
        
        breadth_ratio = np.mean(flags) if flags else 0.0

        if not ENABLE_REGIME_FILTER:
            regime, allow_new, size_mult, circuit = "DISABLED", True, 1.0, False
        elif breadth_ratio >= BREADTH_RISK_ON:
            regime, allow_new, size_mult, circuit = "RISK-ON", True, 1.0, False
        elif breadth_ratio >= BREADTH_RISK_CAUTION:
            regime, allow_new, size_mult, circuit = "CAUTION", True, 0.5, False
        elif breadth_ratio >= BREADTH_CIRCUIT_BREAKER:
            regime, allow_new, size_mult, circuit = "DEFENSIVE", False, 0.0, False
        else:
            regime, allow_new, size_mult, circuit = "CIRCUIT-BREAKER", False, 0.0, True

        rows.append({
            "date": date,
            "breadth_pct": round(breadth_ratio * 100, 2),
            "regime": regime,
            "allow_new_entries": allow_new,
            "size_multiplier": size_mult,
            "circuit_breaker": circuit
        })
    return pd.DataFrame(rows).set_index("date")

# ============================================================
# SYSTEM EXECUTION & PERFORMANCE EVALUATION
# ============================================================

def run_backtest_for_variant(all_signals, trading_days, variant, breadth_df):
    """Executes state-machine backtest iteration for a specified exit variant."""
    cash = STARTING_CAPITAL
    holdings = {}
    trade_log = []
    equity_curve = []

    v_type = variant["type"]
    span = variant.get("span")
    buffer = variant.get("buffer")

    for i in range(len(trading_days) - 1):
        date = trading_days[i]
        next_date = trading_days[i+1]

        regime_row = breadth_df.loc[date]
        allow_new_entries = bool(regime_row["allow_new_entries"])
        size_multiplier = float(regime_row["size_multiplier"])
        circuit_breaker = bool(regime_row["circuit_breaker"])

        pool = []
        for sym, df in all_signals.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["rs_score"]) or not bool(row["liquid"]):
                continue
            if bool(row["blue_dot"]) and bool(row["tt_pass"]) and bool(row["rs_tt_pass"]):
                pool.append((sym, float(row["rs_score"])))
        
        pool.sort(key=lambda x: x[1], reverse=True)
        rank_lookup = {sym: r + 1 for r, (sym, _) in enumerate(pool)}
        target_syms_topn = {sym for sym, _ in pool[:TOP_N]}

        # Valuation
        portfolio_value = cash
        for sym, pos in holdings.items():
            df = all_signals[sym]
            p = float(df.loc[date, "price"]) if date in df.index else pos["entry_price"]
            portfolio_value += pos["qty"] * p

        # Evaluate Exit Signals
        for sym in list(holdings.keys()):
            df = all_signals[sym]
            if date not in df.index or next_date not in df.index:
                continue
            row = df.loc[date]

            if circuit_breaker:
                exit_trigger, reason = True, "CIRCUIT BREAKER"
            elif v_type == "rank_buffer":
                rank = rank_lookup.get(sym, 9999)
                exit_trigger = rank > buffer
                reason = f"Rank > {buffer}"
            elif v_type == "rs_ema_state":
                exit_trigger = bool(row.get(f"rs_below_ema{span}", False))
                reason = f"RS < {span}EMA"
            elif v_type == "rs_ema_cross":
                exit_trigger = bool(row.get(f"rs_cross_below_ema{span}", False))
                reason = f"RS Cross < {span}EMA"
            elif v_type == "tt_fail_cross":
                exit_trigger = bool(row.get("tt_fail_cross", False))
                reason = "Trend Template Fail"
            else:
                exit_trigger, reason = False, "N/A"

            target_exit = (not circuit_breaker) and (sym not in target_syms_topn) if v_type != "rank_buffer" else False

            if exit_trigger or target_exit:
                pos = holdings.pop(sym)
                exec_price = float(df.loc[next_date, "open_price"])
                proceeds = pos["qty"] * exec_price
                s_cost = calculate_costs(proceeds, side="SELL")
                net_proceeds = proceeds - s_cost

                cost_basis = pos["qty"] * pos["entry_price"] + pos["entry_cost"]
                net_gain = net_proceeds - cost_basis
                tax = stcg_tax(net_gain)
                cash += (net_proceeds - tax)

                trade_log.append({
                    "symbol": sym, 
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": next_date.strftime("%Y-%m-%d"),
                    "qty": pos["qty"],
                    "entry_price": round(pos["entry_price"], 2), 
                    "exit_price": round(exec_price, 2),
                    "net_pnl_rs": round(net_gain - tax, 2),
                    "net_return_pct": round((net_gain - tax) / cost_basis * 100, 2) if cost_basis > 0 else 0,
                    "days_held": (next_date - pos["entry_date"]).days,
                    "exit_reason": reason if exit_trigger else "Rank Degradation"
                })

        # Process Entries at Next-Open
        if not circuit_breaker and allow_new_entries:
            target_allocation = (portfolio_value / TOP_N) * size_multiplier
            slots_open = TOP_N - len(holdings)
            candidates = [s for s in target_syms_topn if s not in holdings]

            for sym in candidates:
                if slots_open <= 0:
                    break
                df = all_signals[sym]
                if next_date not in df.index:
                    continue
                
                exec_price = float(df.loc[next_date, "open_price"])
                if exec_price <= 0:
                    continue
                
                qty = int(target_allocation // exec_price)
                if qty < 1:
                    continue
                
                trade_value = qty * exec_price
                b_cost = calculate_costs(trade_value, side="BUY")
                
                if (trade_value + b_cost) <= cash:
                    cash -= (trade_value + b_cost)
                    holdings[sym] = {
                        "qty": qty, 
                        "entry_price": exec_price, 
                        "entry_date": next_date, 
                        "entry_cost": b_cost
                    }
                    slots_open -= 1

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_value_rs": round(portfolio_value, 2),
            "cash_rs": round(cash, 2),
            "n_holdings": len(holdings),
            "breadth_pct": regime_row["breadth_pct"],
            "regime": regime_row["regime"]
        })

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)
