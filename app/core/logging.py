"""Logging utilities for the simulator."""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

from .config import settings


def configure_logging(log_level: int = logging.INFO, log_path: Optional[Path] = None) -> Logger:
    """Configure root logger with console and optional file handlers."""

    logger = logging.getLogger("day_trading_simulator")
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path is None:
        log_path = settings.storage.reports_dir / "simulator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(settings.simulation_warning)
    return logger


logger = configure_logging()
