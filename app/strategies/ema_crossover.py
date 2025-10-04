"""EMA crossover rule-based strategy."""
from __future__ import annotations

import pandas as pd


def generate_signals(features: pd.DataFrame, fast: str = "ema_5", slow: str = "ema_21") -> pd.Series:
    signal = (features[fast] > features[slow]).astype(int) - (
        features[fast] < features[slow]
    ).astype(int)
    return signal.rename("signal")
