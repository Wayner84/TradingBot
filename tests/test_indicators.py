from __future__ import annotations

import pandas as pd

from app.features.indicators import ema, sma


def test_indicator_alignment() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="H")
    price = pd.DataFrame({"close": range(50)}, index=idx)
    ema_df = ema(price, periods=[5])
    sma_df = sma(price, periods=[5])
    assert ema_df.index.equals(price.index)
    assert sma_df.index.equals(price.index)
    assert not ema_df.isna().all().all()
    assert not sma_df.isna().all().all()
