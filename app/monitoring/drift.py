"""Feature and label drift monitoring."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from app.core.io import ensure_directory, save_json


def population_stability_index(base: pd.Series, current: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(base.quantile(quantiles))
    base_counts, _ = np.histogram(base, bins=cuts)
    current_counts, _ = np.histogram(current, bins=cuts)
    base_perc = np.where(base_counts == 0, 1e-6, base_counts / base_counts.sum())
    current_perc = np.where(
        current_counts == 0, 1e-6, current_counts / current_counts.sum()
    )
    psi = np.sum((current_perc - base_perc) * np.log(current_perc / base_perc))
    return float(psi)


def kolmogorov_smirnov(base: pd.Series, current: pd.Series) -> float:
    return float(stats.ks_2samp(base, current, alternative="two-sided").statistic)


def feature_importance_drift(base: pd.Series, current: pd.Series) -> float:
    aligned = base.reindex(current.index).fillna(0)
    return float(np.abs(aligned - current).mean())


def compute_drift(
    base_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    base_importance: pd.Series | None = None,
    current_importance: pd.Series | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    metrics = []
    for column in current_frame.columns:
        if column not in base_frame.columns:
            continue
        base_col = base_frame[column].dropna()
        current_col = current_frame[column].dropna()
        if base_col.empty or current_col.empty:
            continue
        psi = population_stability_index(base_col, current_col)
        ks = kolmogorov_smirnov(base_col, current_col)
        metrics.append({"feature": column, "psi": psi, "ks": ks})
    df_metrics = pd.DataFrame(metrics)

    fi_drift = 0.0
    if base_importance is not None and current_importance is not None:
        fi_drift = feature_importance_drift(base_importance, current_importance)

    summary = {
        "max_psi": float(df_metrics["psi"].max() if not df_metrics.empty else 0.0),
        "max_ks": float(df_metrics["ks"].max() if not df_metrics.empty else 0.0),
        "feature_importance_drift": fi_drift,
    }
    return df_metrics, summary


def _render_html(df_metrics: pd.DataFrame, summary: Dict[str, float], output_dir: Path) -> None:
    ensure_directory(output_dir)
    html_path = output_dir / "drift_report.html"
    table_html = df_metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    html_path.write_text(
        f"""
        <html><head><title>Drift Report</title></head>
        <body>
        <h1>Drift Summary</h1>
        <p>Max PSI: {summary['max_psi']:.4f} | Max KS: {summary['max_ks']:.4f}</p>
        <p>Feature Importance Drift: {summary['feature_importance_drift']:.4f}</p>
        {table_html}
        </body></html>
        """,
        encoding="utf-8",
    )

    if not df_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_metrics.set_index("feature")["psi"].plot(kind="bar", ax=ax, title="PSI by Feature")
        ax.axhline(0.25, color="red", linestyle="--", label="Warning")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "drift_psi.png", dpi=150)
        plt.close(fig)


def run_drift_analysis(
    base_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    output_dir: Path,
    base_importance: pd.Series | None = None,
    current_importance: pd.Series | None = None,
) -> Dict[str, float]:
    ensure_directory(output_dir)
    df_metrics, summary = compute_drift(
        base_frame,
        current_frame,
        base_importance=base_importance,
        current_importance=current_importance,
    )
    save_json(output_dir / "drift_metrics.json", summary)
    _render_html(df_metrics, summary, output_dir)
    return summary
