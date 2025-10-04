"""Feature engineering pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from app.core.io import ensure_directory
from .indicators import build_indicators


def build_feature_matrix(price_frame: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, Path]:
    ensure_directory(output_dir)
    price_frame = price_frame.copy()
    if "symbol" not in price_frame.columns:
        price_frame["symbol"] = "__SINGLE__"

    grouped = price_frame.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=True)
    feature_blocks: Dict[str, pd.DataFrame] = {}

    for symbol, group in grouped:
        indexed = group.set_index("timestamp")
        block = build_indicators(indexed)
        block["returns_1"] = indexed["close"].pct_change()
        block["volatility_14"] = indexed["close"].pct_change().rolling(14).std()
        block = block.dropna().replace([pd.NA, pd.NaT], 0).fillna(0)
        feature_blocks[symbol] = block

    if not feature_blocks:
        features = pd.DataFrame()
    else:
        features = pd.concat(feature_blocks, names=["symbol", "timestamp"]).sort_index()

    path = output_dir / "features.parquet"
    try:
        features.to_parquet(path)
    except Exception:
        path = output_dir / "features.csv"
        features.to_csv(path)
    return features, path
