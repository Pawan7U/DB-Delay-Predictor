"""
generate_data.py
-----------------
Generates a realistic *synthetic* dataset of German long-distance and
regional train departures, modeled on well-known, publicly documented
patterns in Deutsche Bahn (DB) punctuality reporting:

- ICE/IC (long distance) has historically hovered around 60-70% on-time
  (DB defines "on time" as < 6 minutes late for long-distance trains).
- Regional trains (RE/S-Bahn) are usually punctual more often, but still
  suffer during bad weather / construction ("Bauarbeiten").
- Delays worsen in winter (snow/ice on overhead lines), during long-distance
  disruption events, and on routes with active trackwork.
- Rush hour and Friday/Sunday evening travel add congestion delay.

NOTE: This is simulated data for learning/portfolio purposes. It is
statistically realistic but not real DB operational data (DB does not
publish a raw row-level open dataset of individual train delays).

Usage:
    python src/generate_data.py --n 8000 --seed 42 --out data/db_delays.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


ROUTES = [
    ("Berlin Hbf", "Hamburg Hbf", "ICE", 1.00),
    ("Munich Hbf", "Frankfurt(Main) Hbf", "ICE", 1.15),
    ("Cologne Hbf", "Berlin Hbf", "ICE", 1.30),
    ("Frankfurt(Main) Hbf", "Stuttgart Hbf", "IC", 1.05),
    ("Hamburg Hbf", "Hannover Hbf", "IC", 0.95),
    ("Leipzig Hbf", "Dresden Hbf", "RE", 1.00),
    ("Munich Hbf", "Augsburg Hbf", "RE", 0.85),
    ("Berlin Hbf", "Potsdam Hbf", "S-Bahn", 0.80),
    ("Frankfurt(Main) Hbf", "Mainz Hbf", "S-Bahn", 0.75),
    ("Cologne Hbf", "Bonn Hbf", "RE", 0.85),
]

TRAIN_TYPE_BASE_DELAY = {
    # mean delay in minutes under baseline conditions (clear weather,
    # no construction, off-peak, weekday) before route/condition multipliers
    "ICE": 3.6,
    "IC": 2.9,
    "RE": 1.8,
    "S-Bahn": 1.0,
}

WEATHER_OPTIONS = ["clear", "rain", "snow", "storm"]
WEATHER_WEIGHTS_BY_MONTH = {
    # month -> probs for [clear, rain, snow, storm]
    1: [0.35, 0.30, 0.30, 0.05], 2: [0.40, 0.30, 0.25, 0.05],
    3: [0.55, 0.30, 0.10, 0.05], 4: [0.55, 0.35, 0.02, 0.08],
    5: [0.65, 0.30, 0.00, 0.05], 6: [0.65, 0.25, 0.00, 0.10],
    7: [0.60, 0.25, 0.00, 0.15], 8: [0.60, 0.25, 0.00, 0.15],
    9: [0.60, 0.30, 0.00, 0.10], 10: [0.50, 0.35, 0.05, 0.10],
    11: [0.40, 0.35, 0.20, 0.05], 12: [0.35, 0.30, 0.30, 0.05],
}

WEATHER_DELAY_MULT = {"clear": 1.0, "rain": 1.4, "snow": 2.4, "storm": 1.9}


def simulate(n, seed=42):
    rng = np.random.default_rng(seed)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days

    rows = []
    for _ in range(n):
        day_offset = rng.integers(0, total_days)
        date = start_date + timedelta(days=int(day_offset))
        month = date.month
        weekday = date.weekday()  # 0=Mon .. 6=Sun

        origin, destination, train_type, distance_factor = ROUTES[
            rng.integers(0, len(ROUTES))
        ]

        # scheduled hour of departure, weighted toward commute peaks for
        # regional trains and spread out for long distance
        if train_type in ("RE", "S-Bahn"):
            hour = int(rng.choice(
                list(range(5, 23)),
                p=_peaked_hour_weights()
            ))
        else:
            hour = int(rng.integers(5, 23))

        scheduled_dep = date.replace(hour=hour, minute=int(rng.integers(0, 60)))

        # weather draw depends on month
        weather = rng.choice(WEATHER_OPTIONS, p=WEATHER_WEIGHTS_BY_MONTH[month])

        # construction work ("Bauarbeiten") - roughly 18% of trips affected,
        # more common on ICE/IC trunk routes in summer maintenance season
        construction_prob = 0.22 if month in (5, 6, 7, 8, 9) else 0.14
        construction = rng.random() < construction_prob

        # rush hour flag
        is_rush_hour = hour in (7, 8, 17, 18, 19)

        # weekend / Friday evening effect
        is_friday_evening = (weekday == 4 and hour >= 15)
        is_weekend = weekday >= 5

        base = TRAIN_TYPE_BASE_DELAY[train_type] * distance_factor

        multiplier = 1.0
        multiplier *= WEATHER_DELAY_MULT[weather]
        if construction:
            multiplier *= 1.55
        if is_rush_hour:
            multiplier *= 1.25
        if is_friday_evening:
            multiplier *= 1.15
        if is_weekend:
            multiplier *= 0.85  # less congestion on weekends

        mean_delay = base * multiplier
        shape = max(1.6, mean_delay / 1.4)
        scale = 1.4
        delay_minutes = rng.gamma(shape=shape, scale=scale)

        # small chance of a major disruption (signal failure, medical
        # emergency on board, etc.) causing a big delay spike
        if rng.random() < 0.02:
            delay_minutes += rng.uniform(20, 70)

        delay_minutes = float(np.round(max(0.0, delay_minutes), 1))

        # DB's official long-distance punctuality threshold is 6 minutes;
        # regional/S-Bahn commonly measured at 4 minutes. We standardize
        # on a single interpretable label: delayed if > 5 minutes.
        is_delayed = delay_minutes > 5.0

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "weekday": date.strftime("%A"),
            "month": month,
            "scheduled_departure": scheduled_dep.strftime("%Y-%m-%d %H:%M"),
            "hour": hour,
            "origin": origin,
            "destination": destination,
            "train_type": train_type,
            "weather": weather,
            "construction_work": construction,
            "is_rush_hour": is_rush_hour,
            "is_weekend": is_weekend,
            "delay_minutes": delay_minutes,
            "is_delayed": is_delayed,
        })

    return pd.DataFrame(rows)


def _peaked_hour_weights():
    hours = list(range(5, 23))
    weights = np.array([
        1, 2, 5, 6, 3, 2, 2, 2, 2, 2, 3, 5, 6, 3, 2, 2, 1, 1
    ], dtype=float)
    return weights / weights.sum()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DB delay dataset")
    parser.add_argument("--n", type=int, default=8000, help="number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default="data/db_delays.csv", help="output CSV path")
    args = parser.parse_args()

    df = simulate(args.n, args.seed)
    df.sort_values("scheduled_departure", inplace=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
