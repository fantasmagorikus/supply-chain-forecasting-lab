from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    mape: float


def load_series(path: str | Path, date_col: str = "ds", value_col: str = "y") -> pd.DataFrame:
    df = pd.read_csv(path)

    date_candidates = [date_col, "ds", "date", "timestamp"]
    value_candidates = [value_col, "y", "demand", "value", "qty", "quantity"]

    found_date = next((col for col in date_candidates if col in df.columns), None)
    found_value = next((col for col in value_candidates if col in df.columns), None)

    if not found_date or not found_value:
        raise ValueError(
            f"Expected columns '{date_col}' and '{value_col}'. "
            f"Found: {', '.join(df.columns)}"
        )

    # Prophet expects columns named ds (date) and y (value).
    df = df.rename(columns={found_date: "ds", found_value: "y"})
    # Coerce types and remove invalid rows.
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    return df[["ds", "y"]]


def split_train_test(df: pd.DataFrame, test_periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_periods <= 0:
        raise ValueError("test_periods must be greater than 0")
    if test_periods >= len(df):
        raise ValueError("test_periods must be smaller than the dataset length")

    # Use the last N rows as the holdout set.
    train = df.iloc[:-test_periods].copy()
    test = df.iloc[-test_periods:].copy()
    return train, test


def fit_model(df: pd.DataFrame, seasonality_mode: str = "multiplicative") -> Prophet:
    # Configure seasonalities; multiplicative handles proportional swings.
    model = Prophet(
        seasonality_mode=seasonality_mode,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(df)
    return model


def forecast_future(model: Prophet, periods: int, freq: str) -> pd.DataFrame:
    # Extend the timeline and let Prophet predict future values.
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)


def evaluate_forecast(test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> ForecastMetrics:
    # Align predictions with actuals and compute MAE/MAPE.
    merged = test_df.merge(forecast_df[["ds", "yhat"]], on="ds", how="left")
    y_true = merged["y"].to_numpy()
    y_pred = merged["yhat"].to_numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return ForecastMetrics(mae=mae, mape=mape)


def save_forecast_csv(forecast_df: pd.DataFrame, out_path: str | Path) -> None:
    # Persist only the columns needed for reporting.
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    out_df = forecast_df[[col for col in cols if col in forecast_df.columns]].copy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def plot_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot observed history and predicted forecast.
    ax.plot(history_df["ds"], history_df["y"], label="history", color="black", linewidth=1.2)
    ax.plot(forecast_df["ds"], forecast_df["yhat"], label="forecast", color="#2d6cdf")

    if {"yhat_lower", "yhat_upper"} <= set(forecast_df.columns):
        # Add confidence bounds if available.
        ax.fill_between(
            forecast_df["ds"],
            forecast_df["yhat_lower"],
            forecast_df["yhat_upper"],
            color="#2d6cdf",
            alpha=0.2,
            label="confidence",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
