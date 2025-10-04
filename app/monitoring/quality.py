"""Data quality monitoring producing data cards and HTML reports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from app.core.io import ensure_directory, save_json


def _compute_quality_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["symbol", "timestamp"])
    missingness = frame.isna().mean() * 100
    duplicate_pct = frame.duplicated(subset=["symbol", "timestamp"]).mean() * 100
    gaps = frame.groupby("symbol")["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    outlier_threshold = frame[["open", "high", "low", "close"]].mean() + 4 * frame[
        ["open", "high", "low", "close"]
    ].std()
    outliers = (
        (frame[["open", "high", "low", "close"]] > outlier_threshold)
        | (frame[["open", "high", "low", "close"]] < -outlier_threshold)
    ).mean()

    return {
        "missing_pct": float(missingness.mean()),
        "duplicate_pct": float(duplicate_pct),
        "max_consecutive_gap_min": float(gaps.max()),
        "outlier_pct": float(outliers.mean() * 100),
    }


def _quality_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    ensure_directory(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp")
    frame[["open", "close"]].plot(ax=axes[0, 0], title="Price")
    frame["volume"].plot(ax=axes[0, 1], title="Volume", color="tab:orange")
    frame[["open", "close"]].pct_change().abs().rolling(50).mean().plot(
        ax=axes[1, 0], title="Abs Returns (rolling)", color="tab:green"
    )
    frame[["open", "high", "low", "close"]].describe().T[["mean"]].plot(
        kind="bar", ax=axes[1, 1], title="Price Stats", color="tab:red"
    )
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = output_dir / "data_quality.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    html_path = output_dir / "data_quality.html"
    html_content = """
    <html><head><title>Data Quality</title></head>
    <body>
    <h1>Data Quality Overview</h1>
    <img src="data_quality.png" alt="Data quality charts" />
    </body></html>
    """
    html_path.write_text(html_content, encoding="utf-8")


def run_quality_checks(frame: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    ensure_directory(output_dir)
    metrics = _compute_quality_metrics(frame)
    save_json(output_dir / "data_card.json", metrics)
    _quality_plot(frame, output_dir)
    return metrics
