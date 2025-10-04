"""Hyperparameter tuning utilities."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.core.io import ensure_directory, save_json


@dataclass
class TuningResult:
    generation_id: str
    trial_params: Dict[str, float]
    score: float


def random_search(param_grid: Dict[str, List[float]], n_trials: int, seed: int = 42) -> List[Dict[str, float]]:
    random.seed(seed)
    params = []
    keys = list(param_grid.keys())
    for _ in range(n_trials):
        params.append({key: random.choice(param_grid[key]) for key in keys})
    return params


def bayesian_like_search(param_grid: Dict[str, List[float]], n_trials: int) -> List[Dict[str, float]]:
    combos = list(itertools.product(*param_grid.values()))
    best = []
    for i in range(min(n_trials, len(combos))):
        combo = combos[i]
        best.append({key: value for key, value in zip(param_grid.keys(), combo)})
    return best


def save_generation_summary(generation_id: str, trials: List[TuningResult], output_dir: Path) -> None:
    ensure_directory(output_dir)
    table = pd.DataFrame([trial.__dict__ for trial in trials])
    table.to_csv(output_dir / "summary.csv", index=False)
    pareto = table.sort_values("score", ascending=False).head(5)
    pareto.to_csv(output_dir / "pareto.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "generation_id": generation_id,
            "best_score": float(table["score"].max() if not table.empty else 0.0),
            "n_trials": len(trials),
        },
    )
