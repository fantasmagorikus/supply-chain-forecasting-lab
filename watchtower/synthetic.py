from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticSpec:
    start_date: str
    periods: int
    freq: str
    seed: int


def generate_series(spec: SyntheticSpec) -> pd.DataFrame:
    # Reproducible random generator for the noise term.
    rng = np.random.default_rng(spec.seed)
    # Build the time index for the series.
    dates = pd.date_range(start=spec.start_date, periods=spec.periods, freq=spec.freq)
    t = np.arange(spec.periods)

    # Simple demand signal = base + trend + seasonal patterns + noise.
    base = 120.0
    trend = 0.04 * t
    weekly = 12.0 * np.sin(2 * np.pi * t / 7)
    yearly = 18.0 * np.sin(2 * np.pi * t / 365)
    noise = rng.normal(0.0, 6.0, size=spec.periods)

    demand = base + trend + weekly + yearly + noise
    # Keep demand values positive.
    demand = np.maximum(demand, 1.0)

    # Prophet expects columns named ds (date) and y (value).
    df = pd.DataFrame({"ds": dates, "y": np.round(demand, 2)})
    return df


def default_spec() -> SyntheticSpec:
    return SyntheticSpec(
        start_date=str(date.today().replace(day=1, month=1)),
        periods=730,
        freq="D",
        seed=42,
    )
