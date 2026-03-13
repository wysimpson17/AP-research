# AP Research AI Trading Strategy Comparison

This project is a PyCharm-friendly Python research project that compares two trading system architectures on public daily market data:

- Strategy A: a traditional rule-based quantitative system
- Strategy B: a machine learning prediction system

The default study uses `SPY`, `QQQ`, and `AAPL` from `2005-01-01` through `2025-12-31`, then trains on the earlier period and tests on the later period.

## What the project does

1. Downloads adjusted daily market data from Yahoo Finance.
2. Engineers research features such as 5-day return, 10-day return, rolling volatility, moving average relationships, RSI, and volume change.
3. Builds a rule-based trading signal using:
   - 20-day vs 50-day moving average trend
   - 10-day momentum threshold
   - RSI confirmation
4. Trains a machine learning classifier to predict next-day direction.
5. Converts both strategies into long/cash trading signals.
6. Compares cumulative return, Sharpe ratio, maximum drawdown, win rate, and number of trades.

## Project structure

```text
quant vs. trad/
+-- main.py
+-- requirements.txt
+-- README.md
+-- outputs/
\-- strategy_comparison/
    +-- __init__.py
    +-- config.py
    +-- data.py
    +-- features.py
    +-- metrics.py
    +-- pipeline.py
    \-- strategies.py
```

## How to run in PyCharm

1. Open the folder `quant vs. trad` in PyCharm.
2. Create or select a Python virtual environment.
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the project:

```powershell
python main.py
```

## Optional arguments

```powershell
python main.py --tickers SPY QQQ NVDA --model-type random_forest
```

Available options:

- `--tickers`: space-separated list of symbols
- `--start-date`: default `2005-01-01`
- `--end-date`: default `2025-12-31`
- `--train-end-date`: default `2018-12-31`
- `--model-type`: `logistic` or `random_forest`
- `--output-dir`: folder for results

## Outputs

After the run finishes, the `outputs` folder will contain:

- `asset_level_metrics.csv`: performance metrics for each asset and strategy
- `strategy_average_metrics.csv`: average metrics across all assets
- `ml_thresholds.csv`: selected ML probability threshold for each asset
- `portfolio_daily_returns.csv`: equal-weight daily returns for each strategy during the test period
- `strategy_equity_curves.png`: visual comparison chart
- `comparison_report.md`: ready-to-read summary for your paper
- per-ticker signal and feature-importance CSV files

## Notes for your paper

- The train/test split is time-based, not randomly shuffled.
- Win rate is measured per trade, not per day.
- Buy-and-hold is included as a baseline so you can discuss whether either active strategy actually adds value.
- Logistic regression is the default ML model because it is easier to explain in an AP Research paper, but a random forest option is included if you want a nonlinear comparison.
