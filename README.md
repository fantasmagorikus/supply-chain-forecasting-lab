# Project 1: Supply Chain Watchtower

Forecast demand time series with Prophet. n8n should only trigger the CLI; all logic stays in Python.

## Layout
- `data/`: input CSVs and synthetic data.
- `outputs/`: forecast plots and exported forecasts.
- `watchtower/`: Python module (forecasting utilities).
- `main.py`: CLI entry point.

## Setup
1) Create a virtualenv:
   ```bash
   cd /path/to/project-1-supply-chain-watchtower
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2) Verify dependencies, install if missing:
   ```bash
   python -c "import prophet" || pip install -r requirements.txt
   python -c "import pandas, numpy, matplotlib"
   ```

## Generate synthetic data
```bash
python main.py generate --out-path data/synthetic_demand.csv --start-date 2023-01-01 --periods 730 --freq D
```

## Run forecast + evaluation
```bash
python main.py forecast \
  --data-path data/synthetic_demand.csv \
  --horizon 30 \
  --test-periods 60 \
  --plot-path outputs/forecast.png \
  --out-forecast outputs/forecast.csv
```

## n8n Orchestration
See `n8n/README.md` and import `n8n/watchtower_orchestration.json`.

## Notes
- Input CSV expects `ds` (date) and `y` (value). If yours differ, pass `--date-col` and `--value-col`.
- `--test-periods` runs a holdout evaluation and prints MAE/MAPE.
