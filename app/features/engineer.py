"""Feature engineering pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from app.core.io import ensure_directory
from .indicators import build_indicators


def build_feature_matrix(price_frame: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, Path]:
    ensure_directory(output_dir)
    price_frame = price_frame.sort_values("timestamp")
    price_frame = price_frame.set_index("timestamp")
    features = build_indicators(price_frame)
    features["returns_1"] = price_frame["close"].pct_change()
    features["volatility_14"] = price_frame["close"].pct_change().rolling(14).std()
    features = features.dropna().replace([pd.NA, pd.NaT], 0).fillna(0)
    path = output_dir / "features.parquet"
    try:
        features.to_parquet(path)
    except Exception:
        path = output_dir / "features.csv"
        features.to_csv(path)
    return features, path
