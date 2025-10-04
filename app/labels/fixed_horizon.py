"""Fixed horizon return labeling."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from app.core.io import ensure_directory


def fixed_horizon_returns(
    price_frame: pd.DataFrame,
    horizon: int,
    threshold: float = 0.0,
    output_dir: Path | None = None,
) -> Tuple[pd.Series, pd.Series]:
    price_frame = price_frame.copy()
    if "symbol" not in price_frame.columns:
        price_frame["symbol"] = "__SINGLE__"

    price_frame = price_frame.sort_values(["symbol", "timestamp"])
    classification_blocks: Dict[str, pd.Series] = {}
    regression_blocks: Dict[str, pd.Series] = {}

    for symbol, group in price_frame.groupby("symbol", sort=True):
        indexed = group.set_index("timestamp")
        future_price = indexed["close"].shift(-horizon)
        returns = future_price / indexed["close"] - 1
        cls = (returns > threshold).astype(int) - (returns < -threshold).astype(int)
        classification_blocks[symbol] = cls.dropna()
        regression_blocks[symbol] = returns.dropna()

    if classification_blocks:
        classification = pd.concat(classification_blocks, names=["symbol", "timestamp"]).sort_index()
    else:
        classification = pd.Series(dtype="int64")

    if regression_blocks:
        regression = pd.concat(regression_blocks, names=["symbol", "timestamp"]).sort_index()
    else:
        regression = pd.Series(dtype="float64")

    if output_dir is not None:
        ensure_directory(output_dir)
        classification.to_frame("label").to_parquet(output_dir / "labels_classification.parquet")
        regression.to_frame("target").to_parquet(output_dir / "labels_regression.parquet")

    return classification, regression
