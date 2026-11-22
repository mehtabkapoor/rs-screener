# RS Screener

RS (relative-strength) momentum screeners and backtests for NSE stocks. Results go to Google Sheets via GitHub Actions.

## Layout

| File | What it does | Workflow (Actions tab) | Sheet tab(s) |
|---|---|---|---|
| `screener_top10.py` | Live pure-RS Top 10 portfolio. Auto EOD run 4:45 PM IST Mon–Fri, preview on manual run | **Screener - Top 10 (daily)** | `RS_Screener`, `Portfolio`, `Holdings`, `Config` |
| `screener_top50.py` | Top 100 ranking + per-stock RS/price charts for both universes | **Screener - Top 50 (Core + Total Market)** | `Screener - RS Top50 - Code`, `Screener - RS Top50 - Total Market` |
| `backtest_all.py` | All backtests in one run: s1 RS Top10 + Price TT, s2 RS Top10, s4 RAM Momentum | **Backtests (all strategies)** — optional `only` input, e.g. `s1 s4` | `Backtest - RS Top10 + Price TT`, `Backtest - RS Top10`, `Backtest - RAM Momentum`, `Backtest - Comparison` |
| `universe/core.csv` | Core stock list (~180), used by all three scripts | | |
| `universe/total_market.csv` | Full market list (~940), used by the Top 50 screener | | |
| `archive/` | Older standalone versions of s1 and s2 — superseded by `backtest_all.py`, kept for reference | | |

## Editing stock lists

Edit `universe/*.csv` (one `symbol` column, NSE symbols without `.NS`). Each run logs a `No usable data (...)` line naming symbols Yahoo returned nothing for — replace those (renamed or delisted).

## Conventions

- Scripts: `screener_*.py`, `backtest_*.py` (snake_case). Workflows: kebab-case matching the script.
- Secrets: `SHEET_ID`, `GOOGLE_CREDENTIALS`.
- CSV outputs from each run are downloadable as artifacts on the run page.
