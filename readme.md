# Day Trading Simulator — With Generational Monitoring

**Purpose:** A safe, offline-first environment to **simulate** many small trades and use **machine learning** for signal improvement — with **deep monitoring of training data quality** and **per-generation performance tracking**.

> ⚠️ **Simulation Only.** No real trading. Educational/research use at your own risk.

---

## Core Capabilities

- **Versioned Data & Data Cards**
  - Dataset versioning (`dataset_id`), gap/duplicate checks, missingness, outlier stats.
  - `data_card.json` + `data_quality.html` for every dataset.

- **Leakage-Safe Features & Labels**
  - Indicators (EMA/SMA/RSI/MACD/ATR), lags, returns, regimes; fixed-horizon & triple-barrier labels.

- **Generational ML Training**
  - Blocked/purged CV; hyperparameter search.
  - **Artifacts per generation:** `model_card.json`, `cv_metrics.json`, learning curves, feature importance, confusion matrices.

- **Drift & Integrity Monitoring**
  - PSI/KS drift vs. baselines; label drift; integrity/leakage probes.
  - `drift_report.html` + `drift_metrics.json`.

- **Execution & Risk**
  - Market/limit/stop, slippage/spread/latency; volatility targeting, ATR SL/TP, daily loss limits, circuit breakers.

- **Backtests & Reports**
  - Walk-forward, KPIs (CAGR, Sharpe, Sortino, Max DD, Calmar, Win rate, Expectancy, Profit Factor).
  - **Per-generation KPI tables** + improvement charts.
  - `reports/{run_id}/index.html` with an end-to-end audit trail.

- **UI**
  - **CLI** for automation.
  - **FastAPI** endpoints for orchestration/registry/artifacts.
  - **Streamlit Dashboard** with:
    - Data Quality panel (missingness, gaps, outliers).
    - Drift panel (PSI/KS, label drift).
    - **Generations panel**: learning curves, CV metrics, KPIs per generation, equity/drawdown overlays, feature importance drift, confusion matrices.
    - Trades & PnL table with filters.

---

## Quickstart

```bash
# 1) Environment
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) Config
cp config.sample.yaml config.yaml
cp .env.sample .env

# 3) Pull data (versioned dataset)
python -m app.cli data.pull --symbols AAPL,MSFT --timeframe 1h --start 2019-01-01
python -m app.cli data.validate

# 4) Build features & labels
python -m app.cli features.build
python -m app.cli labels.build

# 5) Run a **generation** of training (e.g., GEN_001)
python -m app.cli train --cv blocked --n-iters 30 --generation GEN_001

# 6) Backtest the best model/strategy from GEN_001
python -m app.cli backtest --strategy ml_classifier --generation GEN_001

# 7) Repeat for GEN_002, GEN_003... (the tool will compare)
python -m app.cli train --cv blocked --n-iters 30 --generation GEN_002
python -m app.cli backtest --strategy ml_classifier --generation GEN_002

# 8) See **per-generation improvement** report
python -m app.cli report.generations

# 9) Optional UI
uvicorn app.ui.api:app --reload
streamlit run app/ui/dashboard.py
