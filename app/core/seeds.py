"""Deterministic seeding utilities."""
from __future__ import annotations

import os
import random
from typing import Iterator

import numpy as np

from .config import settings


def seed_everything(seed: int | None = None) -> int:
    actual_seed = seed if seed is not None else settings.random_seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    os.environ["PYTHONHASHSEED"] = str(actual_seed)
    return actual_seed


def seed_sequence(start: int | None = None) -> Iterator[int]:
    base = seed_everything(start)
    while True:
        base += 1
        yield base
