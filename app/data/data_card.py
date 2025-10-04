"""Data card generation utilities."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from app.core.io import save_json
from app.data.validation import ValidationResult, validate_dataset


def build_data_card(frame: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    validation = validate_dataset(frame)
    metrics = asdict(validation)
    metrics.pop("details")
    metrics.update(validation.details)
    save_json(output_dir / "data_card.json", metrics)
    return metrics
