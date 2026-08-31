# ============================================================
# BLUE + GREEN DOT ENTRY LOGIC
# ============================================================

# Blue Dot:
# RS ratio makes a new LOOKBACK_DAYS high.
previous_rs_high = (
    rs_ratio
    .shift(1)
    .rolling(LOOKBACK_DAYS)
    .max()
)

blue_dot = (
    rs_ratio >
    previous_rs_high
)

# Price new high:
previous_price_high = (
    aligned["s"]
    .shift(1)
    .rolling(LOOKBACK_DAYS)
    .max()
)

price_at_new_high = (
    aligned["s"] >
    previous_price_high
)

# Green Dot:
# RS breaks out, but price has NOT simultaneously
# broken its own LOOKBACK_DAYS high.
green_dot = (
    blue_dot &
    (~price_at_new_high)
)

# ------------------------------------------------------------
# COMBINED ENTRY SIGNAL
# ------------------------------------------------------------
#
# Either Blue or Green qualifies.
#
# Because Green is a subset of Blue under the definitions above,
# this is mathematically equivalent to blue_dot.
#
# It is nevertheless retained explicitly so both signals are
# available for analysis and future modification.
#
entry_signal = (
    blue_dot |
    green_dot
)