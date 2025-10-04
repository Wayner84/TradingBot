from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.execution.simulator import ExecutionConfig, execute_orders


def test_execution_fees_slippage(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="H")
    price = pd.DataFrame(
        {
            "timestamp": idx,
            "close": 100.0,
        }
    )
    signals = pd.Series([1, -1, 1, 0, 1], index=idx)
    config = ExecutionConfig(fee_bps=10, slippage_bps=20, spread_bps=0)
    journal = execute_orders(price, signals, config, tmp_path)
    assert not journal.empty
    assert (journal["fee"] > 0).all()
    assert (journal["slippage"] > 0).all()
