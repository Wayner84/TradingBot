"""I/O utilities for storing artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_parquet(path: Path, frame: pd.DataFrame, metadata: Dict[str, Any] | None = None) -> None:
    ensure_directory(path.parent)
    if metadata:
        frame = frame.copy()
        for key, value in metadata.items():
            frame.attrs[key] = value
    frame.to_parquet(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def artifact_path(base: Path, *parts: str) -> Path:
    return ensure_directory(base.joinpath(*parts[:-1])) / parts[-1]
