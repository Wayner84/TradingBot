# Day Trading Simulator (Simulation Only)

> **Warning:** This project is a **simulation-only research environment**. It must **never** be used for live trading or financial advice.

## Overview

`day-trading-simulator` provides an end-to-end generational research workflow for systematic trading strategies. The platform pulls historical OHLCV data, validates quality, engineers features, labels targets, trains machine learning models with time-series aware cross validation, performs execution-aware backtests, and produces comprehensive reports with per-generation performance tracking.

## Key Features

- **Generational Pipeline** – Versioned artifacts for each generation (`GEN_001`, `GEN_002`, ...), including model cards, CV metrics, learning curves, drift reports, and backtest KPIs.
- **Data Quality & Drift Monitoring** – Automated data cards (`data_card.json`) and HTML quality summaries, drift metrics (`drift_metrics.json`), and PSI/KS visualisations.
- **Feature Engineering** – EMA/SMA/RSI/MACD/ATR indicators, returns, volatility measures, and leakage checks.
- **Labeling** – Fixed-horizon return classification/regression plus optional triple-barrier labels.
- **Training** – Blocked or purged cross-validation, hyperparameter search, calibrated metrics, and feature importance exports per generation.
- **Execution Simulator** – Market-like fills with slippage, spread, latency, fees, partial fills, and journal exports.
- **Risk Management** – Volatility targeting, leverage caps, ATR-based stops, trailing stops, and circuit breakers.
- **Backtesting** – Walk-forward engine, rich KPIs, equity/drawdown plots, and per-generation KPI tables.
- **Reporting** – HTML dashboard stitching together data quality, drift, model cards, learning curves, KPIs, and improvement timelines.
- **UI & CLI** – FastAPI endpoints, Streamlit dashboard, and Typer-based CLI for automation.

## Project Layout

```
app/
  core/          # Configuration, logging, registry, seeds
  data/          # yfinance ingestion, validation, data cards
  features/      # Indicator and feature builders
  labels/        # Return and triple-barrier labeling
  monitoring/    # Data quality & drift reports
  strategies/    # Rule-based and ML strategies
  training/      # Generational trainer with CV & artifacts
  execution/     # Execution simulator & journal outputs
  risk/          # Risk sizing & limits
  backtest/      # Walk-forward engine & KPI calculators
  tuning/        # Search utilities & generation summaries
  reporting/     # HTML report builder & timelines
  ui/            # FastAPI API + Streamlit dashboard
  cli.py         # Typer CLI entrypoint
```

## Quickstart

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.sample` to `.env` and adjust configuration if needed.
4. Run the demo sequence below to generate a full pipeline run.

## Demo Sequence

```bash
# 0) One-shot pipeline across multiple symbols
tradebot pipeline.run --symbols AAPL,MSFT,GOOG --timeframe 1h --start 2020-01-01 --generation GEN_001

# 1) Pull historical data (mocked when offline)
python -m app.cli data.pull --symbols AAPL --timeframe 1h --start 2020-01-01

# 2) Validate data and produce quality reports
python -m app.cli data.validate

# 3) Build features and labels
python -m app.cli features.build
python -m app.cli labels.build

# 4) Train generation GEN_001 with blocked CV
python -m app.cli train --cv blocked --n-iters 20 --generation GEN_001

# 5) Backtest the ML classifier strategy for GEN_001
python -m app.cli backtest --strategy ml_classifier --generation GEN_001

# 6) Generate aggregate generational timeline report
python -m app.cli report.generations

# 7) Launch the Streamlit dashboard
streamlit run app/ui/dashboard.py
```

## Testing & Quality

- Unit tests cover leakage checks, indicator alignment, execution math, and KPI calculations.
- Pre-commit hooks (black, isort, ruff) enforce consistent style.
- Deterministic seeding ensures reproducible experiments.

Run the test suite:

```bash
pytest
```

## Requirements

The project targets Python 3.11+. Core libraries include pandas, numpy, scikit-learn, fastapi, typer, streamlit, matplotlib, plotly, yfinance, and pydantic. See `requirements.txt` for the full list.

## Configuration

- `config.sample.yaml` – Example configuration file for storage paths and thresholds.
- `.env.sample` – Environment variable template (optional).

## License

For internal research use only. No warranty. Not financial advice.
