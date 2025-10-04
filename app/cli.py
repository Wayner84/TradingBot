"""Command line interface for the simulator."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from app.backtest import (
    BacktraderConfig,
    generation_kpi_table,
    run_backtrader_strategy,
    run_walk_forward,
)
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


def _filter_price_frame(
    frame: pd.DataFrame,
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    price = frame.copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=False)
    if symbol:
        if "symbol" not in price.columns:
            raise typer.BadParameter(
                "Symbol filtering requested but dataset lacks a 'symbol' column"
            )
        mask = price["symbol"].str.upper() == symbol.upper()
        price = price.loc[mask]
    elif "symbol" in price.columns and price["symbol"].nunique() > 1:
        default_symbol = price["symbol"].iloc[0]
        price = price.loc[price["symbol"] == default_symbol]
    if start:
        price = price.loc[price["timestamp"] >= pd.to_datetime(start)]
    if end:
        price = price.loc[price["timestamp"] <= pd.to_datetime(end)]
    if price.empty:
        raise typer.BadParameter("No price data available after applying filters")
    return price.sort_values("timestamp").reset_index(drop=True)


@app.command("data.pull")
def data_pull(
    symbols: str,
    timeframe: str = "1h",
    start: str = "2020-01-01",
    end: str = "2020-12-31",
    adjust: bool = True,
) -> None:
    symbol_list = [sym.strip().upper() for sym in symbols.split(",")]
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
    features, path = build_feature_matrix(price, settings.storage.cache_dir)
    typer.echo(f"Features saved to {path}")


@app.command("labels.build")
def labels_build(dataset_id: Optional[str] = None, horizon: int = 24) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    cls, reg = fixed_horizon_returns(price, horizon, output_dir=settings.storage.cache_dir)
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
    symbol: Optional[str] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    price = _filter_price_frame(price, symbol=symbol, start=train_start, end=train_end)
    features, _ = build_feature_matrix(price, settings.storage.cache_dir)
    labels, _ = fixed_horizon_returns(price, settings.training.horizon_bars)
    features, labels = features.align(labels, join="inner", axis=0)
    model_path = train_generation(generation, features, labels, model_name=model, cv_scheme=cv, max_iter=n_iters)
    typer.echo(f"Model saved to {model_path}")


@app.command("backtest")
def backtest(
    strategy: str = "ema_crossover",
    generation: str = "GEN_001",
    dataset_id: Optional[str] = None,
    engine: str = "walkforward",
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> None:
    dataset_id = dataset_id or next(iter(registry.list_datasets()), None)
    if dataset_id is None:
        raise typer.Exit("No dataset available")
    price = _load_price_data(settings.storage.data_dir / f"{dataset_id}.parquet")
    price = _filter_price_frame(price, symbol=symbol, start=start, end=end)
    engine = engine.lower()
    base_dir = settings.storage.backtests_dir / generation
    backtest_dir = base_dir / engine
    if engine == "walkforward":
        features, _ = build_feature_matrix(price, settings.storage.cache_dir)
        returns = price.set_index("timestamp")["close"].pct_change().dropna()
        if strategy == "ema_crossover":
            signals = ema_crossover.generate_signals(features)
        else:
            signals = features["returns_1"].apply(lambda x: 1 if x > 0 else -1)
        metrics = run_walk_forward(signals, returns, generation, backtest_dir)
    elif engine == "backtrader":
        config = BacktraderConfig()
        metrics = run_backtrader_strategy(price, strategy, generation, backtest_dir, config=config)
    else:
        raise typer.BadParameter("Unsupported backtest engine. Use 'walkforward' or 'backtrader'.")
    generation_kpi_table([metrics], backtest_dir / "kpis.csv")
    typer.echo(f"Backtest metrics saved: {metrics}")


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
