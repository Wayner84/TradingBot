"""Technical indicators for feature engineering."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import pandas_ta as ta
except Exception:  # pragma: no cover
    ta = None


IndicatorConfig = Dict[str, Iterable[int]]


def _safe_indicator(frame: pd.DataFrame, column: str, func, *args, **kwargs) -> pd.Series:
    try:
        return func(frame[column], *args, **kwargs)
    except Exception:
        return pd.Series(index=frame.index, dtype="float64")


def ema(frame: pd.DataFrame, periods: Iterable[int], price_col: str = "close") -> pd.DataFrame:
    out = {}
    for period in periods:
        out[f"ema_{period}"] = frame[price_col].ewm(span=period, adjust=False).mean()
    return pd.DataFrame(out, index=frame.index)


def sma(frame: pd.DataFrame, periods: Iterable[int], price_col: str = "close") -> pd.DataFrame:
    out = {}
    for period in periods:
        out[f"sma_{period}"] = frame[price_col].rolling(window=period, min_periods=period).mean()
    return pd.DataFrame(out, index=frame.index)


def rsi(frame: pd.DataFrame, periods: Iterable[int], price_col: str = "close") -> pd.DataFrame:
    out = {}
    delta = frame[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for period in periods:
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return pd.DataFrame(out, index=frame.index)


def macd(frame: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = frame["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = frame["close"].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=frame.index,
    )


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift()).abs()
    low_close = (frame["low"] - frame["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean().rename(f"atr_{period}")


def build_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    ema_frame = ema(frame, periods=[5, 21, 55])
    sma_frame = sma(frame, periods=[10, 50, 100])
    rsi_frame = rsi(frame, periods=[14])
    macd_frame = macd(frame)
    atr_series = atr(frame)
    features = pd.concat(
        [ema_frame, sma_frame, rsi_frame, macd_frame, atr_series], axis=1
    ).dropna()
    return features
