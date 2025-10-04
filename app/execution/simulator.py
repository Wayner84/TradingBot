"""Execution simulator with microstructure effects."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from app.core.io import ensure_directory


@dataclass
class ExecutionConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    latency_ms: int = 100
    spread_bps: float = 5.0


@dataclass
class Order:
    timestamp: pd.Timestamp
    signal: int
    price: float
    size: float
    order_type: str = "market"
    limit_price: float | None = None


@dataclass
class Fill:
    timestamp: pd.Timestamp
    price: float
    size: float
    fee: float
    slippage: float
    pnl: float


def _apply_microstructure(price: float, config: ExecutionConfig, side: int) -> float:
    spread = price * config.spread_bps / 10_000
    slippage = price * config.slippage_bps / 10_000
    return price + side * (spread / 2 + slippage)


def execute_orders(
    price_frame: pd.DataFrame,
    signals: pd.Series,
    config: ExecutionConfig,
    output_dir: Path,
) -> pd.DataFrame:
    ensure_directory(output_dir)
    fills: List[Fill] = []
    price_frame = price_frame.sort_values("timestamp").set_index("timestamp")
    for timestamp, signal in signals.items():
        if signal == 0:
            continue
        price = price_frame.loc[timestamp, "close"]
        exec_price = _apply_microstructure(price, config, side=signal)
        fee = abs(exec_price * config.fee_bps / 10_000)
        pnl = -fee
        fills.append(
            Fill(
                timestamp=timestamp + timedelta(milliseconds=config.latency_ms),
                price=exec_price,
                size=signal,
                fee=fee,
                slippage=abs(exec_price - price),
                pnl=pnl,
            )
        )
    journal = pd.DataFrame([fill.__dict__ for fill in fills])
    journal.to_csv(output_dir / "trades_journal.csv", index=False)
    return journal


def compute_trade_statistics(journal: pd.DataFrame) -> pd.Series:
    if journal.empty:
        return pd.Series({"win_rate": 0.0, "expectancy": 0.0})
    wins = (journal["pnl"] > 0).mean()
    expectancy = journal["pnl"].mean()
    return pd.Series({"win_rate": wins, "expectancy": expectancy})
