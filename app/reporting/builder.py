"""Reporting utilities that stitch generation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, select_autoescape

from app.core.config import settings
from app.core.io import ensure_directory


env = Environment(autoescape=select_autoescape())


def _load_json(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(generation_id: str) -> Path:
    report_dir = ensure_directory(settings.storage.reports_dir / generation_id)
    data_card = _load_json(settings.storage.data_dir / "data_card.json")
    drift_metrics = _load_json(settings.storage.reports_dir / generation_id / "drift_metrics.json")
    model_card = _load_json(settings.storage.models_dir / generation_id / "model_card.json")
    cv_metrics = _load_json(settings.storage.models_dir / generation_id / "cv_metrics.json")
    backtest_metrics = _load_json(settings.storage.backtests_dir / generation_id / "backtest_metrics.json")

    template = env.from_string(
        """
        <html><head><title>{{ generation_id }} Report</title></head>
        <body>
        <h1>{{ warning }}</h1>
        <h2>Generation {{ generation_id }}</h2>
        <h3>Data Quality</h3>
        <pre>{{ data_card | tojson(indent=2) }}</pre>
        <h3>Drift Metrics</h3>
        <pre>{{ drift_metrics | tojson(indent=2) }}</pre>
        <h3>Model Card</h3>
        <pre>{{ model_card | tojson(indent=2) }}</pre>
        <h3>CV Metrics</h3>
        <pre>{{ cv_metrics | tojson(indent=2) }}</pre>
        <h3>Backtest KPIs</h3>
        <pre>{{ backtest_metrics | tojson(indent=2) }}</pre>
        </body></html>
        """
    )
    html = template.render(
        generation_id=generation_id,
        warning=settings.simulation_warning,
        data_card=data_card,
        drift_metrics=drift_metrics,
        model_card=model_card,
        cv_metrics=cv_metrics,
        backtest_metrics=backtest_metrics,
    )
    output_path = report_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_generation_timeline(generations: List[str]) -> Path:
    timeline_dir = ensure_directory(settings.storage.reports_dir / "generations")
    metrics = []
    for gen in generations:
        backtest = _load_json(settings.storage.backtests_dir / gen / "backtest_metrics.json")
        if not backtest:
            continue
        backtest["generation_id"] = gen
        metrics.append(backtest)
    template = env.from_string(
        """
        <html><head><title>Generation Timeline</title></head>
        <body>
        <h1>Generation Improvement Timeline</h1>
        {% for metric in metrics %}
        <h3>{{ metric.generation_id }}</h3>
        <pre>{{ metric | tojson(indent=2) }}</pre>
        {% endfor %}
        </body></html>
        """
    )
    html = template.render(metrics=metrics)
    output_path = timeline_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
