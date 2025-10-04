"""Backtrader-based backtesting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type

import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd

from app.core.io import ensure_directory, save_json
from app.core.types import BacktestMetrics

from .engine import performance_metrics


@dataclass
class BacktraderConfig:
    """Configuration for running Backtrader simulations."""

    initial_cash: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0
    stake: float = 1.0


class EMACrossoverStrategy(bt.Strategy):
    params = dict(fast=5, slow=21, stake=1.0)

    def __init__(self) -> None:  # pragma: no cover - behavior defined by Backtrader
        self.fast = bt.ind.EMA(period=self.params.fast)
        self.slow = bt.ind.EMA(period=self.params.slow)
        self.crossover = bt.ind.CrossOver(self.fast, self.slow)

    def next(self) -> None:  # pragma: no cover - behavior defined by Backtrader
        if not self.position and self.crossover > 0:
            self.buy(size=self.params.stake)
        elif self.position and self.crossover < 0:
            self.close()


class BuyAndHoldStrategy(bt.Strategy):
    params = dict(stake=1.0)

    def next(self) -> None:  # pragma: no cover - behavior defined by Backtrader
        if not self.position:
            self.buy(size=self.params.stake)


STRATEGY_REGISTRY: Dict[str, Type[bt.Strategy]] = {
    "ema_crossover": EMACrossoverStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
}


class PandasPriceData(bt.feeds.PandasData):
    params = (
        ("datetime", "timestamp"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
    )


def _prepare_price_frame(price: pd.DataFrame) -> pd.DataFrame:
    frame = price.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
    frame = frame.sort_values("timestamp")
    columns = ["open", "high", "low", "close", "volume"]
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for Backtrader feed: {', '.join(missing)}")
    return frame.set_index("timestamp")


def _select_strategy(name: str) -> Type[bt.Strategy]:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown Backtrader strategy '{name}'") from exc


def run_backtrader_strategy(
    price: pd.DataFrame,
    strategy: str,
    generation_id: str,
    output_dir: Path,
    config: BacktraderConfig | None = None,
) -> BacktestMetrics:
    """Execute a Backtrader strategy and persist analytics."""

    ensure_directory(output_dir)
    config = config or BacktraderConfig()

    prepared = _prepare_price_frame(price)
    data = PandasPriceData(dataname=prepared)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    strategy_cls = _select_strategy(strategy)
    cerebro.addstrategy(strategy_cls, stake=config.stake)
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission)
    if config.slippage:
        cerebro.broker.set_slippage_perc(config.slippage)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    returns_dict = strat.analyzers.timereturn.get_analysis()
    if returns_dict:
        index = pd.to_datetime(list(returns_dict.keys()))
        returns_series = pd.Series(list(returns_dict.values()), index=index)
        returns_series = returns_series.sort_index()
        returns_series.iloc[0] = 0.0
    else:
        first_index = prepared.index[0]
        returns_series = pd.Series([0.0], index=[first_index])

    equity_curve = (1 + returns_series).cumprod()
    metrics_dict = performance_metrics(equity_curve)

    final_value = float(cerebro.broker.getvalue())
    summary = {
        "generation_id": generation_id,
        **metrics_dict,
        "final_value": final_value,
        "starting_cash": config.initial_cash,
        "total_return": final_value / config.initial_cash - 1,
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    equity_curve.plot(ax=ax, label="Equity")
    (equity_curve.cummax() - equity_curve).plot(ax=ax, label="Drawdown")
    ax.legend()
    ax.set_title(f"Backtrader Equity - {generation_id}")
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    save_json(output_dir / "backtest_metrics.json", summary)

    return BacktestMetrics(generation_id=generation_id, **metrics_dict)

