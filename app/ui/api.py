"""FastAPI for managing simulator operations."""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

from app.core.registry import registry
from app.reporting.builder import build_generation_timeline, build_report

app = FastAPI(title="Day Trading Simulator API", description="Simulation only - no live trading")


@app.get("/warning")
def warning() -> dict:
    return {"message": "SIMULATION ONLY - NO LIVE TRADING"}


@app.get("/datasets")
def list_datasets() -> List[str]:
    return registry.list_datasets()


@app.get("/runs")
def list_runs() -> List[str]:
    return registry.list_runs()


@app.get("/generations")
def list_generations() -> List[str]:
    return registry.list_generations()


@app.get("/reports/{generation_id}")
def get_report(generation_id: str) -> dict:
    try:
        path = build_report(generation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"path": str(path)}


@app.post("/reports/timeline")
def timeline() -> dict:
    path = build_generation_timeline(registry.list_generations())
    return {"path": str(path)}
