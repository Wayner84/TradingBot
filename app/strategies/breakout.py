"""Breakout strategy based on rolling highs and lows."""
from __future__ import annotations

import pandas as pd


def generate_signals(price_frame: pd.DataFrame, window: int = 20) -> pd.Series:
    highs = price_frame["high"].rolling(window).max()
    lows = price_frame["low"].rolling(window).min()
    signal = (price_frame["close"] > highs.shift()).astype(int) - (
        price_frame["close"] < lows.shift()
    ).astype(int)
    return signal.rename("signal")
