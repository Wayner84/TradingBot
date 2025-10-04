"""Machine learning based classifier strategy."""
from __future__ import annotations

import joblib
import pandas as pd

from pathlib import Path


def load_model(model_path: Path):
    return joblib.load(model_path)


def generate_signals(features: pd.DataFrame, model_path: Path) -> pd.Series:
    model = load_model(model_path)
    proba = model.predict_proba(features)[:, 1]
    signal = pd.Series((proba > 0.5).astype(int) * 2 - 1, index=features.index, name="signal")
    return signal
