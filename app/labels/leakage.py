"""Leakage detection helpers."""
from __future__ import annotations

import pandas as pd


def check_no_lookahead(features: pd.DataFrame, labels: pd.Series) -> bool:
    aligned = features.index.intersection(labels.index)
    if aligned.empty:
        return True
    future_rows = labels.index.difference(features.index)
    return future_rows.empty
