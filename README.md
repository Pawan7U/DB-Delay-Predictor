# 🚆 Verspätung! — Predicting Deutsche Bahn Train Delays

A data science project tackling one of Germany's most universally relatable
problems: **trains that don't arrive on time**. Deutsche Bahn's punctuality
(or lack thereof) is a running national joke — this project explores *why*
trains get delayed and builds machine learning models to predict it.

> ⚠️ **About the data**: DB does not publish an open, row-level dataset of
> individual train delays, so this project uses a **synthetic dataset**
> generated to reflect real, well-documented statistical patterns in German
> rail punctuality (e.g. long-distance ICE/IC trains historically running
> on-time roughly 55-70% of the time, delays worsening in winter and during
> track construction ["Bauarbeiten"], and DB's own >6-minute "on time"
> threshold). The generator (`src/generate_data.py`) is fully documented and
> parameterized, so you can swap in real data (e.g. from the
> [DB Open Data Portal](https://data.deutschebahn.com/) or scraped
> [Zugfinder](https://www.zugfinder.net/) data) with the same pipeline.

## The problem

Every commuter in Germany has a story about a delayed ICE, a cancelled
regional train, or a "Anschlusszug fällt leider aus" announcement. This
project asks two concrete, answerable questions:

1. **Will this train be delayed?** (classification)
2. **If so, by how many minutes?** (regression)

...and along the way, explores *what actually drives delays*: train type,
weather, season, time of day, and track construction.

## Project structure

```
db-delay-predictor/
├── data/
│   └── db_delays.csv            # generated dataset (created by generate_data.py)
├── src/
│   ├── generate_data.py         # synthetic dataset generator
│   ├── eda.py                   # exploratory analysis + plots
│   └── train_model.py           # trains classifier + regressor
├── models/                      # saved trained models (.joblib)
├── outputs/figures/             # generated PNG charts
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/db-delay-predictor.git
cd db-delay-predictor
pip install -r requirements.txt
```

## Usage

Run the three stages in order:

```bash
# 1. Generate the dataset (8,000 simulated trips across 10 major routes, 2023-2024)
python src/generate_data.py --n 8000 --seed 42 --out data/db_delays.csv

# 2. Explore it — prints a summary and saves charts to outputs/figures/
python src/eda.py --data data/db_delays.csv --out outputs/figures

# 3. Train the models — saves classifier + regressor to models/
python src/train_model.py --data data/db_delays.csv
```

## Dataset

Each row is one simulated train departure with:

| Column | Description |
|---|---|
| `date`, `weekday`, `month`, `hour` | When the train departed |
| `scheduled_departure` | Full scheduled departure timestamp |
| `origin`, `destination` | Route endpoints (10 major German routes) |
| `train_type` | `ICE`, `IC`, `RE`, or `S-Bahn` |
| `weather` | `clear`, `rain`, `snow`, `storm` |
| `construction_work` | Whether trackwork ("Bauarbeiten") was active |
| `is_rush_hour`, `is_weekend` | Context flags |
| `delay_minutes` | Actual delay in minutes (target for regression) |
| `is_delayed` | `True` if delay > 5 minutes (target for classification) |

## Key findings (EDA)

- **Long-distance trains are the least punctual.** In the simulated data,
  ICE trains run on time only ~46% of the time, IC ~66%, vs. ~84-90% for
  RE/S-Bahn — mirroring the real, well-documented pattern that DB's
  long-distance fleet struggles with punctuality far more than regional
  and commuter services.
- **Snow is the biggest weather disruptor**, roughly doubling average delay
  vs. clear weather, driven by overhead-line icing and speed restrictions.
- **Track construction ("Bauarbeiten") reliably adds delay** regardless of
  train type — a very real and constant feature of the German network.
- **Rush hour and Friday-evening travel** add congestion delay on top of
  weather and construction effects.

See `outputs/figures/` for the full set of charts (delay by train type,
weather, month, hour of day, construction, and on-time rate by route).

## Modeling results

Two RandomForest models are trained on an 80/20 train/test split, using
one-hot encoded route/weather/train-type features plus time and construction
flags:

| Model | Metric | Score |
|---|---|---|
| Classifier (delayed vs. on-time) | Accuracy | ~0.77 |
| | Precision | ~0.60 |
| | Recall | ~0.64 |
| | F1 | ~0.62 |
| | ROC-AUC | ~0.79 |
| Regressor (delay minutes, log-transformed target) | MAE | ~2.95 min |
| | R² | ~0.07 |

**Honest takeaway**: predicting *whether* a train will be delayed is fairly
tractable (ROC-AUC ~0.79), but predicting the *exact number of minutes* is
much harder — real-world delay severity is dominated by one-off disruptions
(signal failures, medical emergencies, etc.) that are inherently hard to
model from scheduling features alone. This mirrors real operational
forecasting challenges at railways worldwide, not just DB.

Feature importance plots (`outputs/figures/classifier_feature_importance.png`
and `regressor_feature_importance.png`) show `train_type`, `weather`, and
`construction_work` as the dominant predictive signals.

## Extending this project

- Swap in real historical data from the [DB Open Data Portal](https://data.deutschebahn.com/)
- Add live delay lookups via the [DB APIs](https://developers.deutschebahn.com/)
- Try gradient boosting (XGBoost/LightGBM) or a quantile regressor to predict
  a delay *range* instead of a point estimate
- Build a small Streamlit/Flask app for interactive "will my train be late?" queries

## License

MIT — see [LICENSE](LICENSE).
