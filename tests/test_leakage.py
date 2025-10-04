from __future__ import annotations

import pandas as pd
from pathlib import Path

from app.features.engineer import build_feature_matrix
from app.labels.fixed_horizon import fixed_horizon_returns
from app.labels.leakage import check_no_lookahead


def test_no_leakage(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="H")
    price = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100 + pd.Series(range(100)),
            "high": 100 + pd.Series(range(100)) + 1,
            "low": 100 + pd.Series(range(100)) - 1,
            "close": 100 + pd.Series(range(100)),
            "volume": 1000,
            "symbol": "TEST",
        }
    )
    features, _ = build_feature_matrix(price, tmp_path)
    labels, _ = fixed_horizon_returns(price, horizon=1)
    features, labels = features.align(labels, join="inner", axis=0)
    assert check_no_lookahead(features, labels)
