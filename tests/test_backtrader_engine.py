from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest.backtrader_engine import BacktraderConfig, run_backtrader_strategy


def _synthetic_price_frame(periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    trend = np.linspace(0, 5, periods)
    noise = np.sin(np.linspace(0, 8, periods))
    close = 100 + trend + noise
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close + rng.normal(0, 0.2, periods),
            "high": close + np.abs(rng.normal(0, 0.3, periods)),
            "low": close - np.abs(rng.normal(0, 0.3, periods)),
            "close": close,
            "volume": rng.integers(1_000, 5_000, periods),
        }
    )
    return frame


def test_backtrader_strategy_generates_metrics(tmp_path: Path) -> None:
    price = _synthetic_price_frame()
    config = BacktraderConfig(initial_cash=10_000, commission=0.0, stake=10)
    metrics = run_backtrader_strategy(
        price=price,
        strategy="ema_crossover",
        generation_id="GEN_TEST",
        output_dir=tmp_path,
        config=config,
    )

    assert metrics.generation_id == "GEN_TEST"
    assert isinstance(metrics.cagr, float)
    assert (tmp_path / "backtest_metrics.json").exists()
    assert (tmp_path / "equity_curve.png").exists()
