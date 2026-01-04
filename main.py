#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from watchtower.forecast import (
    evaluate_forecast,
    fit_model,
    forecast_future,
    load_series,
    plot_forecast,
    save_forecast_csv,
    split_train_test,
)
from watchtower.synthetic import SyntheticSpec, default_spec, generate_series


def run_generate(args: argparse.Namespace) -> None:
    # Build the synthetic data spec from CLI inputs.
    spec = SyntheticSpec(
        start_date=args.start_date,
        periods=args.periods,
        freq=args.freq,
        seed=args.seed,
    )
    df = generate_series(spec)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")


def run_forecast(args: argparse.Namespace) -> None:
    df = load_series(args.data_path, date_col=args.date_col, value_col=args.value_col)

    if args.test_periods:
        # Optional holdout evaluation before final training.
        train_df, test_df = split_train_test(df, args.test_periods)
        model = fit_model(train_df, seasonality_mode=args.seasonality_mode)
        test_forecast = forecast_future(model, periods=len(test_df), freq=args.freq)
        metrics = evaluate_forecast(test_df, test_forecast)
        print(f"Holdout MAE: {metrics.mae:.2f}")
        print(f"Holdout MAPE: {metrics.mape:.2f}%")

    # Train on all data and forecast the requested horizon.
    model = fit_model(df, seasonality_mode=args.seasonality_mode)
    forecast_df = forecast_future(model, periods=args.horizon, freq=args.freq)

    if args.out_forecast:
        save_forecast_csv(forecast_df, args.out_forecast)
        print(f"Saved forecast -> {args.out_forecast}")

    if args.plot_path:
        plot_forecast(df, forecast_df, args.plot_path)
        print(f"Saved plot -> {args.plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supply Chain Watchtower CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate synthetic demand data.")
    defaults = default_spec()
    gen.add_argument("--out-path", default="data/synthetic_demand.csv", help="CSV output path.")
    gen.add_argument("--start-date", default=defaults.start_date, help="Start date (YYYY-MM-DD).")
    gen.add_argument("--periods", type=int, default=defaults.periods, help="Number of rows.")
    gen.add_argument("--freq", default=defaults.freq, help="Pandas date freq (e.g., D, W).")
    gen.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    gen.set_defaults(func=run_generate)

    forecast = subparsers.add_parser("forecast", help="Train Prophet and forecast demand.")
    forecast.add_argument("--data-path", default="data/synthetic_demand.csv", help="Input CSV path.")
    forecast.add_argument("--date-col", default="ds", help="Date column name.")
    forecast.add_argument("--value-col", default="y", help="Value column name.")
    forecast.add_argument("--freq", default="D", help="Date frequency.")
    forecast.add_argument("--horizon", type=int, default=30, help="Forecast horizon.")
    forecast.add_argument(
        "--test-periods",
        type=int,
        default=0,
        help="Holdout periods for MAE/MAPE evaluation.",
    )
    forecast.add_argument(
        "--seasonality-mode",
        choices=["additive", "multiplicative"],
        default="multiplicative",
        help="Prophet seasonality mode.",
    )
    forecast.add_argument("--plot-path", default="", help="Save forecast plot PNG.")
    forecast.add_argument("--out-forecast", default="", help="Save forecast CSV.")
    forecast.set_defaults(func=run_forecast)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
