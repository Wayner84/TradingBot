"""Typed dataclasses used across the simulator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DatasetMetadata:
    dataset_id: str
    symbols: List[str]
    timeframe: str
    start: datetime
    end: datetime
    adjusted: bool
    source: str
    path: Path


@dataclass
class GenerationInfo:
    generation_id: str
    run_id: str
    created_at: datetime
    dataset_id: str
    model_path: Path
    model_type: str
    params: Dict[str, float]


@dataclass
class BacktestMetrics:
    generation_id: str
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    expectancy: float
    profit_factor: float


FeatureFrame = pd.DataFrame
LabelSeries = pd.Series
