"""Fixed horizon return labeling."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.core.io import ensure_directory


def fixed_horizon_returns(
    price_frame: pd.DataFrame,
    horizon: int,
    threshold: float = 0.0,
    output_dir: Path | None = None,
) -> Tuple[pd.Series, pd.Series]:
    price_frame = price_frame.sort_values("timestamp").set_index("timestamp")
    future_price = price_frame["close"].shift(-horizon)
    returns = future_price / price_frame["close"] - 1
    classification = (returns > threshold).astype(int) - (returns < -threshold).astype(int)
    regression = returns
    if output_dir is not None:
        ensure_directory(output_dir)
        classification.to_frame("label").to_parquet(output_dir / "labels_classification.parquet")
        regression.to_frame("target").to_parquet(output_dir / "labels_regression.parquet")
    return classification.dropna(), regression.dropna()
