"""Walk-forward backtesting engine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.io import ensure_directory, save_json
from app.core.types import BacktestMetrics


@dataclass
class WalkForwardConfig:
    train_window: int = 500
    test_window: int = 100


def performance_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    returns = equity_curve.pct_change().fillna(0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    periods = len(returns)
    cagr = (1 + total_return) ** (252 / periods) - 1 if periods else 0.0
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)
    downside = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / (downside.std() + 1e-6)
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_dd = drawdown.min()
    calmar = -cagr / max_dd if max_dd != 0 else 0.0
    wins = (returns > 0).mean()
    expectancy = returns.mean()
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum() + 1e-6)
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate": float(wins),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
    }


def run_walk_forward(
    signals: pd.Series,
    returns: pd.Series,
    generation_id: str,
    output_dir: Path,
) -> BacktestMetrics:
    ensure_directory(output_dir)
    signals = signals.reindex(returns.index).fillna(0)
    strategy_returns = signals.shift().fillna(0) * returns
    equity_curve = (1 + strategy_returns).cumprod()
    metrics = performance_metrics(equity_curve)

    fig, ax = plt.subplots(figsize=(8, 4))
    equity_curve.plot(ax=ax, label="Equity")
    (equity_curve.cummax() - equity_curve).plot(ax=ax, label="Drawdown")
    ax.legend()
    ax.set_title(f"Equity & Drawdown - {generation_id}")
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    save_json(output_dir / "backtest_metrics.json", metrics)
    return BacktestMetrics(generation_id=generation_id, **metrics)


def generation_kpi_table(metrics_list: List[BacktestMetrics], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    table = pd.DataFrame([vars(m) for m in metrics_list])
    table.to_csv(output_path, index=False)
