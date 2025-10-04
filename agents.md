# Agents

This project uses a modular, agent-style architecture. Each agent has a focused responsibility with typed inputs/outputs, deterministic behavior under a set seed, and clear success metrics. Agents communicate via the Orchestrator and share artifacts through a versioned experiment store.

---

## 0) Orchestrator
**Goal:** Coordinate data → features → labels → training (generations) → backtests → reports → dashboards.

- **Responsibilities:** Dependency graph, seeds, caching, experiment lineage, retries, logging.
- **Interfaces:** CLI (`trade-sim ...`), FastAPI endpoints, Streamlit dashboard.
- **Artifacts:** `runs/{run_id}/manifest.json`, `datasets/{dataset_id}`, `models/{model_id}`, `reports/{run_id}`.

---

## 1) Data Ingestion Agent
**Goal:** Pull **historical** OHLCV (and pseudo-live for sim), adjust, validate, and **version** datasets.

- **Inputs:** `symbols[]`, `timeframe`, `start`, `end`, `adjust`.
- **Process:** yfinance adapter → OHLCV parquet; add metadata: source, time pulled, split/dividend flags, checksum.
- **Outputs:** Canonical DataFrame `[timestamp, open, high, low, close, volume]` + `dataset_id`.
- **Success:** Completeness, no duplicated timestamps/gaps (unless documented), checksums stable.

---

## 2) Data Quality Monitor Agent
**Goal:** **Quantify dataset quality** and produce **data cards**.

- **Inputs:** Dataset from Ingestion.
- **Process:** Missingness %, duplicate bars %, gap statistics, outlier z-scores, volatility sanity checks, stationarity hints, label balance preview.
- **Outputs:** `data_card.json` (metrics), `data_quality.html` (visuals), alerts if thresholds exceeded.
- **Success:** Transparent data quality with thresholds configurable in `config.yaml`.

---

## 3) Feature Engineering Agent
**Goal:** Create **leakage-safe** features.

- **Inputs:** OHLCV frames.
- **Process:** Indicators (EMA/SMA/RSI/MACD/ATR), rolling stats, lags, volatility, calendar features, regime tags.
- **Outputs:** Feature matrix `X` (aligned) + provenance (`features_card.json`).
- **Success:** No look-ahead; NaN-safe; columns described & versioned.

---

## 4) Labeling Agent
**Goal:** Build supervised targets.

- **Inputs:** Price series, horizon config.
- **Process:** Fixed-horizon returns; optional triple-barrier (PT/SL/time), sample weights, side labels.
- **Outputs:** `y` (+ weights optionally) + `labels_card.json`.
- **Success:** Targets match business intent; leakage tests pass.

---

## 5) Drift & Integrity Agent
**Goal:** Monitor **feature/label drift** and data integrity across **generations**.

- **Inputs:** `X`, `y`, prior baselines (previous generations).
- **Process:** PSI/KS statistics vs. baselines; label distribution drift; feature importance drift; train/test split integrity; data leakage probes.
- **Outputs:** `drift_report.html`, `drift_metrics.json`, drift alerts.
- **Success:** Drift surfaced early; corrective actions suggested.

---

## 6) Strategy Research Agent
**Goal:** Rule-based signal generation.

- **Inputs:** Features, strategy YAML, risk budget.
- **Process:** EMA cross, breakout, mean-revert, filters by RSI/ATR/volatility; return `signal ∈ {-1,0,1}` with confidence.
- **Outputs:** Signal timeline.
- **Success:** Deterministic and unit-tested rules.

---

## 7) Model Training Agent (Generational)
**Goal:** Train ML models and **track improvement per generation**.

- **Inputs:** `X, y`, CV scheme (blocked/purged), search space, `generation_id`.
- **Process:** Time-series-aware CV; fit pipelines (Scaler + RF/GB/XGB*); calibration; hyperparam search.
- **Outputs:** `model_id`, `model_card.json`, `cv_metrics.json`, **learning curves**, **feature importance**, **confusion matrices**, per-fold metrics.
- **Success:** Out-of-sample uplift vs. previous generation; reproducible under seed.
- \*XGBoost optional; gracefully fallback to sklearn-only.

---

## 8) Execution Simulator Agent
**Goal:** Turn signals into fills with microstructure realism.

- **Inputs:** signals, OHLCV/bid-ask (if available), fees/slippage, order types, sizing rules.
- **Process:** Market/limit/stop; spread + slippage (bps), latency; partial fills, queue approximations.
- **Outputs:** Trades journal (entries/exits, fees, PnL), per-trade diagnostics.
- **Success:** Transparent assumptions; sensitivity runs are stable.

---

## 9) Risk Manager Agent
**Goal:** Enforce portfolio/risk constraints.

- **Inputs:** trades journal, open positions, risk config.
- **Process:** Vol targeting, leverage caps, symbol caps, ATR SL/TP, trailing stops, daily loss limits, circuit breakers.
- **Outputs:** Adjusted orders/position states; risk log.
- **Success:** DD constraints respected; no breaches.

---

## 10) Backtest Engine Agent (Walk-Forward)
**Goal:** Honest historical evaluation + **per-generation KPIs**.

- **Inputs:** rolling train/test windows, strategy/model artifacts.
- **Process:** Walk-forward with blocked/purged CV; aggregate KPIs; uncertainty bands; bootstrap/MC resamples.
- **Outputs:** `backtest_metrics.json`, equity curves, drawdown curves, **per-generation KPI table**.
- **Success:** Stable KPIs; narrow generalization gap.

---

## 11) Tuning Agent (Generational Search)
**Goal:** Optimise params over **generations**.

- **Inputs:** search space; objective; constraints.
- **Process:** Random/Bayesian search; early stopping; archive Pareto front; **generation = set of trials**; survivors propagated.
- **Outputs:** `generations/GEN_XXX/summary.json`, best configs, improvements plot.
- **Success:** Diminishing returns detected; avoids overfit.

---

## 12) Reporting Agent
**Goal:** Produce comprehensive, **generation-aware** reports.

- **Inputs:** data/model/backtest metrics; drift; risk logs; trades; configs.
- **Process:** Generate HTML/Markdown/PDF: data card, drift report, model card, KPIs, equity/D.D., sensitivity, **per-generation improvement charts**.
- **Outputs:** `/reports/{run_id}/index.html`, assets.
- **Success:** One-click audit trail with **timeline of improvements**.

---

## 13) Tracking & Registry Agent
**Goal:** Maintain a **lightweight experiment registry**.

- **Inputs:** all manifests & artifacts.
- **Process:** Create `run_id`, `dataset_id`, `model_id`, `generation_id`; persist lineage graph; expose via API/UI.
- **Outputs:** `registry.json`, run listings, search/index endpoints.
- **Success:** Every artifact traceable; reproducible with manifest alone.
