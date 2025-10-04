"""Mean reversion strategy using z-score."""
from __future__ import annotations

import pandas as pd


def generate_signals(price_frame: pd.DataFrame, window: int = 20, threshold: float = 1.5) -> pd.Series:
    rolling_mean = price_frame["close"].rolling(window).mean()
    rolling_std = price_frame["close"].rolling(window).std()
    zscore = (price_frame["close"] - rolling_mean) / rolling_std
    signal = (zscore < -threshold).astype(int) - (zscore > threshold).astype(int)
    return signal.rename("signal")
