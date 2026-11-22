"""
Single source of truth for Google Sheet tab names.

Naming rule:  <Universe> - <What>
  Trend Template = universe/trend_template.csv  (~180 stocks)
  Total Market   = universe/total_market.csv    (~940 stocks)

migrate_tabs(sh) renames old tabs in place (data, formulas and history
are kept). It is safe to call on every run: it only renames a tab when
the old name exists and the new name does not.
"""

# --- Trend Template list -------------------------------------------------
TT_TOP10_SCREENER  = "Trend Template - Top10 Screener"
TT_TOP10_PORTFOLIO = "Trend Template - Top10 Portfolio"
TT_TOP10_HOLDINGS  = "Trend Template - Top10 Holdings"
TT_TOP10_CONFIG    = "Trend Template - Top10 Config"
TT_TOP50           = "Trend Template - Top50"
TT_BT_RS_TOP10_PRICE_TT = "Trend Template - Backtest RS Top10 + Price TT"
TT_BT_RS_TOP10          = "Trend Template - Backtest RS Top10"
TT_BT_RAM_MOMENTUM      = "Trend Template - Backtest RAM Momentum"
TT_BT_COMPARISON        = "Trend Template - Backtest Comparison"

# --- Total Market list ---------------------------------------------------
TM_TOP50 = "Total Market - Top50"

# old name -> new name
_RENAMES = {
    "RS_Screener": TT_TOP10_SCREENER,
    "Portfolio": TT_TOP10_PORTFOLIO,
    "Holdings": TT_TOP10_HOLDINGS,
    "Config": TT_TOP10_CONFIG,
    "Screener - RS Top50 - Code": TT_TOP50,
    "Screener - RS Top50 - Total Market": TM_TOP50,
    "Backtest - RS Top10 + Price TT": TT_BT_RS_TOP10_PRICE_TT,
    "Backtest - RS Top10": TT_BT_RS_TOP10,
    "Backtest - RAM Momentum": TT_BT_RAM_MOMENTUM,
    "Backtest - Comparison": TT_BT_COMPARISON,
}


def migrate_tabs(sh):
    try:
        titles = {ws.title: ws for ws in sh.worksheets()}
    except Exception as e:
        print(f"Tab migration skipped (could not list tabs): {e}")
        return
    print(f"Sheet tabs: {list(titles)}")
    for old, new in _RENAMES.items():
        if old in titles and new not in titles:
            try:
                titles[old].update_title(new)
                print(f"Renamed sheet tab: '{old}' -> '{new}'")
            except Exception as e:
                print(f"Could not rename tab '{old}': {e}")
