"""Lightweight experiment registry."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import settings
from .io import ensure_directory


class Registry:
    """Stores metadata about runs, datasets, and generations."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.storage.registry_path
        ensure_directory(self.path.parent)
        if not self.path.exists():
            self._write({"datasets": {}, "runs": {}, "generations": {}})

    def _read(self) -> Dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: Dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    def register_dataset(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        data = self._read()
        data["datasets"][dataset_id] = metadata
        self._write(data)

    def register_run(self, run_id: str, payload: Dict[str, Any]) -> None:
        data = self._read()
        payload.setdefault("created_at", datetime.utcnow().isoformat())
        data["runs"][run_id] = payload
        self._write(data)

    def register_generation(self, generation_id: str, payload: Dict[str, Any]) -> None:
        data = self._read()
        payload.setdefault("created_at", datetime.utcnow().isoformat())
        data["generations"][generation_id] = payload
        self._write(data)

    def list_datasets(self) -> List[str]:
        return list(self._read()["datasets"].keys())

    def list_runs(self) -> List[str]:
        return list(self._read()["runs"].keys())

    def list_generations(self) -> List[str]:
        return sorted(self._read()["generations"].keys())

    def get_generation(self, generation_id: str) -> Dict[str, Any]:
        return self._read()["generations"][generation_id]

    def snapshot(self) -> Dict[str, Any]:
        return self._read()


registry = Registry()
