"""Data ingestion from yfinance with caching."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover - fallback
    yf = None

from app.core.config import settings
from app.core.io import save_parquet
from app.core.types import DatasetMetadata


def _hash_params(symbols: Iterable[str], timeframe: str, start: str, end: str, adjusted: bool) -> str:
    key = "|".join([" ".join(sorted(symbols)), timeframe, start, end, str(adjusted)])
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _mock_dataset(symbol: str, start: str, end: str, freq: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq=freq, inclusive="both")
    prices = np.cumsum(np.random.normal(0, 1, len(idx))) + 100
    frame = pd.DataFrame(
        {
            "open": prices + np.random.normal(0, 0.5, len(idx)),
            "high": prices + np.abs(np.random.normal(0, 0.8, len(idx))),
            "low": prices - np.abs(np.random.normal(0, 0.8, len(idx))),
            "close": prices + np.random.normal(0, 0.5, len(idx)),
            "volume": np.random.randint(1_000, 10_000, len(idx)),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    frame["symbol"] = symbol
    return frame.reset_index()


def fetch_ohlcv(
    symbols: Iterable[str],
    timeframe: str,
    start: str,
    end: str,
    adjusted: bool = True,
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Fetch OHLCV data and persist to parquet cache."""

    symbols = list(symbols)
    dataset_hash = _hash_params(symbols, timeframe, start, end, adjusted)
    dataset_id = f"OHLCV_{timeframe}_{dataset_hash}"
    path = settings.storage.data_dir / f"{dataset_id}.parquet"

    if path.exists():
        frame = pd.read_parquet(path)
    else:
        frames = []
        if yf is None:
            for symbol in symbols:
                frames.append(_mock_dataset(symbol, start, end, timeframe))
        else:  # pragma: no cover - network calls
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    interval=timeframe,
                    start=start,
                    end=end,
                    auto_adjust=adjusted,
                )
                hist = hist.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                ).reset_index()
                hist["symbol"] = symbol
                frames.append(hist)
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.dropna().sort_values(["symbol", "Date" if "Date" in frame.columns else "timestamp"])
        if "Date" in frame.columns:
            frame = frame.rename(columns={"Date": "timestamp"})
        save_parquet(path, frame, metadata={"dataset_id": dataset_id})

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        symbols=symbols,
        timeframe=timeframe,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        adjusted=adjusted,
        source="yfinance" if yf else "mock",
        path=path,
    )
    return frame, metadata
