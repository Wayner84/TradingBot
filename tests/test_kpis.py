from __future__ import annotations

import pandas as pd

from app.backtest.engine import performance_metrics


def test_performance_metrics() -> None:
    returns = pd.Series([1.0, 1.02, 1.05, 1.1])
    metrics = performance_metrics(returns)
    assert metrics["cagr"] > 0
    assert metrics["sharpe"] >= 0
    assert metrics["max_drawdown"] <= 0
