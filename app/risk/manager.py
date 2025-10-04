"""Risk management utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class RiskParameters:
    vol_target: float = 0.02
    max_leverage: float = 3.0
    per_symbol_cap: float = 0.2
    atr_stop_multiplier: float = 3.0
    atr_take_multiplier: float = 6.0
    trailing_stop_pct: float = 0.03
    daily_loss_limit: float = 0.05
    circuit_breaker_pct: float = 0.1


def position_size(volatility: float, risk_params: RiskParameters) -> float:
    if volatility <= 0:
        return 0.0
    size = risk_params.vol_target / volatility
    return min(size, risk_params.max_leverage)


def apply_daily_loss_limit(equity_curve: pd.Series, risk_params: RiskParameters) -> bool:
    drawdown = equity_curve.pct_change().cumsum().min()
    return abs(drawdown) >= risk_params.daily_loss_limit


def atr_stop_levels(price: float, atr: float, risk_params: RiskParameters) -> Tuple[float, float]:
    stop = price - risk_params.atr_stop_multiplier * atr
    take = price + risk_params.atr_take_multiplier * atr
    return stop, take
