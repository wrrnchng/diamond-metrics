# Diamond Metrics

A command-line tool that fetches MLB historical game data, trains machine learning models for totals, moneylines, and run lines, and surfaces positive expected value (+EV) bets using manual decimal odds and quarter-Kelly stake sizing.

## Requirements

- Python 3.10+
- Internet access (MLB Stats API and `mlb-statsapi` for schedules)

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python run.py --train        # first run: fetch data and train models
python run.py --predict      # interactive odds entry and recommendations
```

**First run:** CSV data and trained models are not included in the repository. You must run `--train` before `--predict`.

## Usage

| Flag | Description |
|------|-------------|
| `--train` | Fetch/update historical data and retrain models |
| `--predict` | Load models, fetch today's schedule, prompt for odds, print +EV bets |
| `--start YYYY-MM-DD` | Start date for a custom fetch range (use with `--train`) |
| `--end YYYY-MM-DD` | End date for a custom fetch range (use with `--train`) |
| `--append` | Append a custom date range to the existing cache instead of replacing it |
| `--bankroll N` | Bankroll for Kelly sizing (default: 1000) |

If neither `--train` nor `--predict` is passed, both run in sequence.

### Examples

```bash
python run.py --train
python run.py --predict
python run.py --predict --bankroll 500
python run.py --train --start 2024-01-01 --end 2024-12-31 --append
```

During `--predict`, you select games from today's schedule and enter decimal odds for moneyline, run line, and over/under markets. The tool returns bets where model probability implies EV above 5%, with recommended stake sizes.

## Project Layout

```
mlb_betting_predictor/
├── run.py                  # CLI entry point
├── requirements.txt
├── ARCHITECTURE.md         # Technical reference
├── src/
│   ├── data_fetcher.py     # MLB Stats API fetch and CSV cache
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── predict.py
├── data/                   # Generated CSV cache (gitignored)
└── models/                 # Trained model artifacts (gitignored)
```

## Documentation

For architecture, data schema, model details, calibration, and known limitations, see [ARCHITECTURE.md](ARCHITECTURE.md).
