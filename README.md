# RS Screener

RS (relative-strength) momentum screeners and backtests for NSE stocks. Results go to Google Sheets via GitHub Actions.

## Layout

| File | What it does | Sheet tab(s) |
|---|---|---|
| `screener_top10.py` | Live pure-RS Top 10 portfolio. 5 PM run = EOD (updates Portfolio/Holdings); other runs = preview | `RS_Screener`, `Portfolio`, `Holdings`, `Config` |
| `screener_top50.py` | Top 100 ranking + per-stock RS/price charts for both universes | `Screener - RS Top50 - Code`, `Screener - RS Top50 - Total Market` |
| `backtest_all.py` | All backtests in one run: s1 RS Top10 + Price TT, s2 RS Top10, s4 RAM Momentum | `Backtest - RS Top10 + Price TT`, `Backtest - RS Top10`, `Backtest - RAM Momentum`, `Backtest - Comparison` |
| `universe/core.csv` | Core stock list (~180), used by all three scripts | |
| `universe/total_market.csv` | Full market list (~940), used by the Top 50 screener | |
| `archive/` | Older standalone versions of s1 and s2 — superseded by `backtest_all.py`, kept for reference | |

## Workflow: **Run All** (`.github/workflows/run-all.yml`)

- **Auto:** Mon–Fri at 9:00 AM, 12:00 PM, 3:00 PM, 5:00 PM IST. Runs Top 10 → Top 50 → Backtests, one after another. The 5 PM run is the Top 10 EOD.
- **Manual:** Actions → Run All → Run workflow. Pick `all` / `top10` / `top50` / `backtests`; optional Force EOD and backtest subset.
- Fresh install every run (no cache). Timestamps in IST.

## Editing stock lists

Edit `universe/*.csv` (one `symbol` column, NSE symbols without `.NS`). Each run logs a `No data / <1yr history, skipped` line. Mostly recent IPOs (need ~1 year of prices for the 12M RS) — they join automatically. Replace a symbol only if it was delisted or renamed.

## Conventions

- Scripts: `screener_*.py`, `backtest_*.py` (snake_case). One workflow: `run-all.yml`.
- Secrets: `SHEET_ID`, `GOOGLE_CREDENTIALS`.
- CSV outputs from each run are downloadable as artifacts on the run page.
