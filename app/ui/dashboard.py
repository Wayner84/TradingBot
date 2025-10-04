"""Streamlit dashboard for monitoring simulation artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import settings
from app.core.registry import registry

st.set_page_config(page_title="Day Trading Simulator", layout="wide")
st.sidebar.warning("SIMULATION ONLY - NO LIVE TRADING")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def data_quality_tab() -> None:
    st.header("Data Quality")
    card = load_json(settings.storage.data_dir / "data_card.json")
    st.json(card)
    png = settings.storage.data_dir / "data_quality.png"
    if png.exists():
        st.image(str(png))


def drift_tab() -> None:
    st.header("Drift Monitoring")
    for gen in registry.list_generations():
        st.subheader(gen)
        metrics = load_json(settings.storage.reports_dir / gen / "drift_metrics.json")
        st.json(metrics)
        img = settings.storage.reports_dir / gen / "drift_psi.png"
        if img.exists():
            st.image(str(img))


def generations_tab() -> None:
    st.header("Generational Performance")
    gens = registry.list_generations()
    for gen in gens:
        st.subheader(gen)
        cv = load_json(settings.storage.models_dir / gen / "cv_metrics.json")
        st.json(cv)
        backtest = load_json(settings.storage.backtests_dir / gen / "backtest_metrics.json")
        st.json(backtest)
        curve = settings.storage.backtests_dir / gen / "equity_curve.png"
        if curve.exists():
            st.image(str(curve))
        report = settings.storage.reports_dir / gen / "index.html"
        if report.exists():
            st.write(f"[Open Report]({report.as_uri()})")


def trades_tab() -> None:
    st.header("Trades Journal")
    journal_path = settings.storage.reports_dir / "trades_journal.csv"
    if journal_path.exists():
        df = pd.read_csv(journal_path)
        st.dataframe(df)
        st.metric("Expectancy", f"{df['pnl'].mean():.4f}")
    else:
        st.info("No trades journal available yet")


def main() -> None:
    tabs = st.tabs(["Data Quality", "Drift", "Generations", "Trades"])
    with tabs[0]:
        data_quality_tab()
    with tabs[1]:
        drift_tab()
    with tabs[2]:
        generations_tab()
    with tabs[3]:
        trades_tab()


if __name__ == "__main__":
    main()
