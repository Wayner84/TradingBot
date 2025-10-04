"""Dataset validation and quality checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class ValidationResult:
    missing_pct: float
    duplicate_pct: float
    max_gap: int
    is_valid: bool
    details: Dict[str, float]


def validate_dataset(frame: pd.DataFrame) -> ValidationResult:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp")
    missing_pct = frame.isna().mean().mean()
    duplicate_pct = frame.duplicated(subset=["timestamp", "symbol"]).mean()
    gaps = (
        frame.groupby("symbol")["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    )
    max_gap = int(gaps.max())
    is_valid = missing_pct < 0.05 and duplicate_pct < 0.01
    details = {
        "missing_pct": float(missing_pct),
        "duplicate_pct": float(duplicate_pct),
        "max_gap_minutes": float(max_gap),
    }
    return ValidationResult(missing_pct, duplicate_pct, max_gap, is_valid, details)
