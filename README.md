# ETF Momentum Lab – Quick Start (Windows)

## Run it in the browser (GitHub Codespaces)

On the repository page click **Code → Codespaces → Create codespace on main**. The libraries install
automatically and the app opens on port 8501. Nothing is installed on your own computer.

## New computer? Run setup once

Double-click **setup_install.bat**. It installs Python 3.12 (via winget, if missing), creates the app's
environment in %LOCALAPPDATA%\us_etf_momentum_venv and installs all libraries. A log is written to
install_log.txt. run_dashboard.bat also starts setup automatically if it hasn't been run yet.

## Dashboard app (recommended)

Double-click **run_dashboard.bat**. It opens the US ETF Momentum Lab in your browser
(http://localhost:8501). Set parameters in the left panel, click **Run backtest**, and use the
tabs: Overview, Returns, Trades, Today's signal, Compare runs. Keep the black window open
while you use it; close it to stop the app.

## Command line

1. Install Python 3.11+ from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
2. Create a folder, e.g. `C:\Trading\us_etf_momentum`, and put `us_etf_momentum.py` and `requirements.txt` in it.
3. Open **Command Prompt** in that folder and run:

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the base backtest (first run downloads prices into `data\prices.csv`):

```
python us_etf_momentum.py
```

5. Run the comparison set (11 variants):

```
python us_etf_momentum.py --sweep
```

6. See what the strategy would hold today:

```
python us_etf_momentum.py --signal --refresh
```

## Useful options

| Option | Example | Effect |
|---|---|---|
| `--rebalance` | `--rebalance monthly` | weekly, biweekly or monthly |
| `--stop` | `--stop 0.10` / `--stop 0` | stop loss; 0 = off |
| `--no-rs` | | turn off the relative-strength (vs SPY) filter |
| `--exclude` | `--exclude GLD,SLV` | drop tickers |
| `--rank` | `--rank return` | rank by raw return instead of Sharpe |
| `--start` / `--end` | `--start 2015-01-01` | backtest window |
| `--name` | `--name my_test` | results folder name |
| `--refresh` | | re-download prices |

## Outputs (in `results\<name>\`)

- `summary.txt` – headline stats vs SPY and 60/40, P&L by group, yearly returns
- `trades.csv` – every trade with entry/exit and reason
- `monthly_returns.csv` – month-by-year table
- `equity_curve.png` – equity curve (log) and drawdown
- `results\sweep\comparison.csv` – side-by-side variants (sweep mode)

Every time you open a new Command Prompt, run `.venv\Scripts\activate` first.

Analytical tool only – not investment advice.
