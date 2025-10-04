"""Command line interface for the simulator."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
import typer

from app.backtest.engine import generation_kpi_table, run_walk_forward
from app.core.config import settings
from app.core.registry import registry
from app.data.yfinance_adapter import fetch_ohlcv
from app.features.engineer import build_feature_matrix
from app.labels.fixed_horizon import fixed_horizon_returns
from app.monitoring.quality import run_quality_checks
from app.monitoring.drift import run_drift_analysis
from app.reporting.builder import build_generation_timeline, build_report
from app.strategies import ema_crossover
from app.training.trainer import train_generation

app = typer.Typer(help="Simulation only - no live trading")


def _load_price_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _parse_symbols(symbols: str) -> List[str]:
    parsed = [sym.strip().upper() for sym in symbols.split(",") if sym.strip()]
    if not parsed:
        raise typer.Exit("At least one symbol must be provided")
    return parsed


def _cache_dir_for(dataset_id: str) -> Path:
    path = settings.storage.cache_dir / dataset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_returns(price: pd.DataFrame) -> pd.Series:
    frame = price.copy()
    if "symbol" not in frame.columns:
        frame["symbol"] = "__SINGLE__"
    frame = frame.sort_values(["symbol", "timestamp"])
    series = (
        frame.set_index(["symbol", "timestamp"])["close"].groupby(level="symbol").pct_change()
    )
    return series.dropna()


def _model_signals(model_path: Path, features: pd.DataFrame) -> pd.Series:
    if not model_path.exists():
        raise typer.Exit(f"Model artifact not found at {model_path}")
    model = joblib.load(model_path)
    preds = pd.Series(model.predict(features), index=features.index)
    return preds.apply(lambda x: 0 if x == 0 else (1 if x > 0 else -1)).rename("signal")


@app.command("data.pull")
def data_pull(
    symbols: str,
    timeframe: str = "1h",
    start: str = "2020-01-01",
    end: str = "2020-12-31",
    adjust: bool = True,
) -> None:
    symbol_list = _parse_symbols(symbols)
    frame, metadata = fetch_ohlcv(symbol_list, timeframe, start, end, adjust)
    registry.register_dataset(metadata.dataset_id, metadata.__dict__)
    typer.echo(f"Saved dataset {metadata.dataset_id} to {metadata.path}")


@app.command("data.validate")
def data_validate(dataset_id: Optional[str] = None) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No datasets available")
    path = settings.storage.data_dir / f"{dataset_id}.parquet"
    frame = pd.read_parquet(path)
    metrics = run_quality_checks(frame, settings.storage.data_dir)
    typer.echo(metrics)


@app.command("features.build")
def features_build(dataset_id: Optional[str] = None) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    features, path = build_feature_matrix(price, _cache_dir_for(dataset_id))
    typer.echo(f"Features saved to {path}")


@app.command("labels.build")
def labels_build(dataset_id: Optional[str] = None, horizon: int = 24) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    cache_dir = _cache_dir_for(dataset_id)
    cls, reg = fixed_horizon_returns(price, horizon, output_dir=cache_dir)
    typer.echo(f"Labels generated with {len(cls)} observations")


@app.command("monitor.drift")
def monitor_drift(generation: str, baseline: str = "GEN_001") -> None:
    current = pd.read_csv(settings.storage.models_dir / generation / "feature_importance.csv")
    base = pd.read_csv(settings.storage.models_dir / baseline / "feature_importance.csv")
    drift_dir = settings.storage.reports_dir / generation
    run_drift_analysis(
        base_frame=base.set_index("feature"),
        current_frame=current.set_index("feature"),
        output_dir=drift_dir,
    )
    typer.echo(f"Drift report saved to {drift_dir}")


@app.command("train")
def train(
    generation: str = "GEN_001",
    cv: str = "blocked",
    n_iters: int = 10,
    dataset_id: Optional[str] = None,
    model: str = "gradient_boosting",
) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    cache_dir = _cache_dir_for(dataset_id)
    features, _ = build_feature_matrix(price, cache_dir)
    labels, _ = fixed_horizon_returns(price, settings.training.horizon_bars, output_dir=cache_dir)
    features, labels = features.align(labels, join="inner", axis=0)
    model_path = train_generation(generation, features, labels, model_name=model, cv_scheme=cv, max_iter=n_iters)
    typer.echo(f"Model saved to {model_path}")


@app.command("backtest")
def backtest(
    strategy: str = "ema_crossover",
    generation: str = "GEN_001",
    dataset_id: Optional[str] = None,
) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    cache_dir = _cache_dir_for(dataset_id)
    features, _ = build_feature_matrix(price, cache_dir)
    returns = _compute_returns(price)
    if strategy == "ema_crossover":
        signals = ema_crossover.generate_signals(features)
    elif strategy == "ml_classifier":
        model_path = settings.storage.models_dir / generation / "model.joblib"
        signals = _model_signals(model_path, features)
    else:
        raise typer.Exit(f"Unknown strategy '{strategy}'")
    backtest_dir = settings.storage.backtests_dir / generation
    metrics = run_walk_forward(signals, returns, generation, backtest_dir)
    generation_kpi_table([metrics], backtest_dir / "kpis.csv")
    typer.echo(f"Backtest metrics saved: {metrics}")


@app.command("pipeline.run")
def pipeline_run(
    symbols: str,
    timeframe: str = "1h",
    start: str = "2020-01-01",
    end: str = "2020-12-31",
    adjust: bool = True,
    generation: str = "GEN_001",
    cv: str = "blocked",
    n_iters: int = 10,
    model: str = "gradient_boosting",
    strategy: str = "ml_classifier",
) -> None:
    """End-to-end pipeline run for one generation."""

    typer.echo(settings.simulation_warning)
    symbol_list = _parse_symbols(symbols)
    price, metadata = fetch_ohlcv(symbol_list, timeframe, start, end, adjust)
    registry.register_dataset(metadata.dataset_id, metadata.__dict__)

    cache_dir = _cache_dir_for(metadata.dataset_id)
    features, _ = build_feature_matrix(price, cache_dir)
    labels, _ = fixed_horizon_returns(price, settings.training.horizon_bars, output_dir=cache_dir)
    features, labels = features.align(labels, join="inner", axis=0)

    model_path = train_generation(
        generation,
        features,
        labels,
        model_name=model,
        cv_scheme=cv,
        max_iter=n_iters,
    )

    returns = _compute_returns(price)
    if strategy == "ema_crossover":
        signals = ema_crossover.generate_signals(features)
    elif strategy == "ml_classifier":
        signals = _model_signals(model_path, features)
    else:
        raise typer.Exit(f"Unknown strategy '{strategy}'")

    backtest_dir = settings.storage.backtests_dir / generation
    metrics = run_walk_forward(signals, returns, generation, backtest_dir)
    generation_kpi_table([metrics], backtest_dir / "kpis.csv")

    run_id = datetime.utcnow().strftime("RUN_%Y%m%d%H%M%S")
    registry.register_run(
        run_id,
        {
            "dataset_id": metadata.dataset_id,
            "symbols": symbol_list,
            "generation": generation,
            "model": model,
            "strategy": strategy,
            "metrics": metrics.__dict__,
        },
    )

    typer.echo(
        " | ".join(
            [
                f"Run ID: {run_id}",
                f"Dataset: {metadata.dataset_id}",
                f"Generation: {generation}",
                f"Model saved to {model_path}",
                f"Backtest CAGR: {metrics.cagr:.2%}",
            ]
        )
    )


@app.command("report.run")
def report_run(generation: str = "GEN_001") -> None:
    path = build_report(generation)
    typer.echo(f"Report generated at {path}")


@app.command("report.generations")
def report_generations() -> None:
    path = build_generation_timeline(registry.list_generations())
    typer.echo(f"Timeline generated at {path}")


@app.command("report.compare")
def report_compare(gens: str) -> None:
    selected = gens.split(",")
    path = build_generation_timeline(selected)
    typer.echo(f"Comparison report at {path}")


@app.command("registry.list")
def registry_list() -> None:
    typer.echo(registry.snapshot())


@app.command("registry.show")
def registry_show(run_id: str) -> None:
    snapshot = registry.snapshot()
    typer.echo(snapshot.get("runs", {}).get(run_id, {}))


if __name__ == "__main__":
    app()
