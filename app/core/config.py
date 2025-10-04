"""Configuration management for the day-trading-simulator project."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, BaseSettings, Field, validator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class StoragePaths(BaseModel):
    """Container for frequently used storage paths."""

    data_dir: Path = Field(default_factory=lambda: _project_root() / "data")
    cache_dir: Path = Field(default_factory=lambda: _project_root() / "cache")
    models_dir: Path = Field(default_factory=lambda: _project_root() / "models")
    backtests_dir: Path = Field(default_factory=lambda: _project_root() / "backtests")
    reports_dir: Path = Field(default_factory=lambda: _project_root() / "reports")
    registry_path: Path = Field(default_factory=lambda: _project_root() / "registry.json")

    def ensure(self) -> None:
        for path in [
            self.data_dir,
            self.cache_dir,
            self.models_dir,
            self.backtests_dir,
            self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class QualityThresholds(BaseModel):
    max_missing_pct: float = 0.05
    max_duplicate_pct: float = 0.01
    max_gap_minutes: int = 240
    max_outlier_pct: float = 0.05


class DriftThresholds(BaseModel):
    max_psi: float = 0.25
    max_ks: float = 0.2


class TrainingConfig(BaseModel):
    default_cv: str = "blocked"
    n_splits: int = 5
    horizon_bars: int = 24


class RiskConfig(BaseModel):
    max_leverage: float = 3.0
    daily_loss_limit: float = 0.05
    circuit_breaker_pct: float = 0.1


class Settings(BaseSettings):
    """Project level settings loaded from environment variables or YAML."""

    simulation_warning: str = (
        "SIMULATION ONLY: No live trading. For research and education purposes."
    )
    timezone: str = "UTC"
    random_seed: int = 42
    storage: StoragePaths = Field(default_factory=StoragePaths)
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    drift: DriftThresholds = Field(default_factory=DriftThresholds)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    config_path: Optional[Path] = None

    class Config:
        env_prefix = "DTS_"
        env_nested_delimiter = "__"

    @validator("storage", pre=True)
    def _validate_storage(cls, value: Any) -> StoragePaths:
        if isinstance(value, dict):
            return StoragePaths(**value)
        if isinstance(value, StoragePaths):
            return value
        raise TypeError("storage must be a mapping or StoragePaths instance")

    @validator("config_path", pre=True)
    def _parse_config_path(cls, value: Any) -> Optional[Path]:
        if value is None or value == "":
            return None
        return Path(value)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        data: Dict[str, Any] = {}
        if config_path and config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        instance = cls(**data, config_path=config_path)
        instance.storage.ensure()
        return instance


settings = Settings.load()
