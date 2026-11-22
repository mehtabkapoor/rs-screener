# RS Screener

RS (relative-strength) momentum screeners and backtests for NSE stocks. Results go to Google Sheets via GitHub Actions.

## Naming rule

Every file and Sheet tab starts with the stock list it uses:

| Prefix | Stock list | File |
|---|---|---|
| `tt_` / **Trend Template** | ~180 stocks | `universe/trend_template.csv` |
| `tm_` / **Total Market** | ~940 stocks | `universe/total_market.csv` |

## Files → Sheet tabs

| Code | Stock list | Sheet tabs |
|---|---|---|
| `tt_top10_screener.py` — live pure-RS Top 10 portfolio. 5 PM run = EOD (updates Portfolio/Holdings); other runs = preview | Trend Template | `Trend Template - Top10 Screener` · `- Top10 Portfolio` · `- Top10 Holdings` · `- Top10 Config` |
| `tt_tm_top50_screener.py` — Top 100 ranking + per-stock RS/price charts | Trend Template **and** Total Market | `Trend Template - Top50` · `Total Market - Top50` |
| `tt_backtests.py` — all backtests in one run (s1 RS Top10 + Price TT, s2 RS Top10, s4 RAM Momentum) | Trend Template | `Trend Template - Backtest RS Top10 + Price TT` · `- Backtest RS Top10` · `- Backtest RAM Momentum` · `- Backtest Comparison` |
| `sheet_tabs.py` — all tab names in one place; renames old tabs automatically | | |
| `archive/` — older standalone versions of s1/s2, superseded by `tt_backtests.py` | | |

## Workflow: **Run All** (`.github/workflows/run-all.yml`)

- **Auto:** Mon–Fri at 9:00 AM, 12:00 PM, 3:00 PM, 5:00 PM IST. Runs Top 10 → Top 50 → Backtests, one after another. The 5 PM run is the Top 10 EOD.
- **Manual:** Actions → Run All → Run workflow. Pick `all` / `top10` / `top50` / `backtests`; optional Force EOD and backtest subset.
- Fresh install every run (no cache). Timestamps in IST.

## Editing stock lists

Edit `universe/*.csv` (one `symbol` column, NSE symbols without `.NS`). Each run logs a `No data / <1yr history, skipped` line. Mostly recent IPOs (need ~1 year of prices for the 12M RS) — they join automatically. Replace a symbol only if it was delisted or renamed.

## Conventions

- Code: `<list>_<what>.py`, snake_case. Sheet tabs: `<List> - <What>`, defined only in `sheet_tabs.py`.
- Secrets: `SHEET_ID`, `GOOGLE_CREDENTIALS`.
- CSV outputs from each run are downloadable as artifacts on the run page.
