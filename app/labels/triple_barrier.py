"""Simplified triple barrier labeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from app.core.io import ensure_directory


@dataclass
class TripleBarrierConfig:
    profit_taking: float
    stop_loss: float
    max_holding: int


def triple_barrier(
    price_frame: pd.DataFrame,
    config: TripleBarrierConfig,
    output_dir: Path | None = None,
) -> Tuple[pd.Series, pd.Series]:
    price_frame = price_frame.sort_values("timestamp").set_index("timestamp")
    close = price_frame["close"]
    outcomes = []
    returns = []
    for idx, price in close.iteritems():
        window = close.loc[idx : idx + pd.Timedelta(minutes=config.max_holding * 60)]
        pt_hit = window[window >= price * (1 + config.profit_taking)]
        sl_hit = window[window <= price * (1 - config.stop_loss)]
        if not pt_hit.empty:
            outcomes.append(1)
            returns.append(pt_hit.iloc[0] / price - 1)
        elif not sl_hit.empty:
            outcomes.append(-1)
            returns.append(sl_hit.iloc[0] / price - 1)
        else:
            final = window.iloc[-1]
            ret = final / price - 1
            outcomes.append(0 if abs(ret) < config.profit_taking else int(np.sign(ret)))
            returns.append(ret)
    classification = pd.Series(outcomes, index=close.index)
    regression = pd.Series(returns, index=close.index)
    if output_dir:
        ensure_directory(output_dir)
        classification.to_frame("label").to_parquet(output_dir / "triple_barrier_labels.parquet")
        regression.to_frame("target").to_parquet(output_dir / "triple_barrier_targets.parquet")
    return classification, regression
