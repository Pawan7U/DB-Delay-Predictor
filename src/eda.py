"""
eda.py
------
Exploratory data analysis on the DB delay dataset. Produces PNG figures
into outputs/figures/ summarizing the main drivers of delay.

Usage:
    python src/eda.py --data data/db_delays.csv --out outputs/figures
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_delay_by_train_type(df, out_dir):
    plt.figure(figsize=(7, 5))
    order = df.groupby("train_type")["delay_minutes"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="train_type", y="delay_minutes", order=order, showfliers=False)
    plt.title("Delay Distribution by Train Type")
    plt.xlabel("Train Type")
    plt.ylabel("Delay (minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_by_train_type.png"), dpi=150)
    plt.close()


def plot_delay_by_weather(df, out_dir):
    plt.figure(figsize=(7, 5))
    order = ["clear", "rain", "storm", "snow"]
    sns.barplot(data=df, x="weather", y="delay_minutes", order=order, estimator="mean", errorbar=("ci", 95))
    plt.title("Average Delay by Weather Condition")
    plt.xlabel("Weather")
    plt.ylabel("Mean Delay (minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_by_weather.png"), dpi=150)
    plt.close()


def plot_delay_by_month(df, out_dir):
    plt.figure(figsize=(8, 5))
    monthly = df.groupby("month")["delay_minutes"].mean().reindex(range(1, 13))
    monthly.plot(kind="line", marker="o")
    plt.title("Average Delay by Month (Seasonality)")
    plt.xlabel("Month")
    plt.ylabel("Mean Delay (minutes)")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_by_month.png"), dpi=150)
    plt.close()


def plot_delay_by_hour(df, out_dir):
    plt.figure(figsize=(8, 5))
    hourly = df.groupby("hour")["delay_minutes"].mean()
    hourly.plot(kind="bar", color="steelblue")
    plt.title("Average Delay by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Mean Delay (minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_by_hour.png"), dpi=150)
    plt.close()


def plot_construction_effect(df, out_dir):
    plt.figure(figsize=(6, 5))
    sns.barplot(data=df, x="construction_work", y="delay_minutes", estimator="mean", errorbar=("ci", 95))
    plt.title("Effect of Trackwork (Bauarbeiten) on Delay")
    plt.xlabel("Construction Work Active")
    plt.ylabel("Mean Delay (minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_by_construction.png"), dpi=150)
    plt.close()


def plot_on_time_rate_by_route(df, out_dir):
    plt.figure(figsize=(9, 6))
    df["route"] = df["origin"] + " -> " + df["destination"]
    rate = (1 - df.groupby("route")["is_delayed"].mean()).sort_values()
    rate = rate * 100
    rate.plot(kind="barh", color="seagreen")
    plt.title("On-Time Rate (%) by Route")
    plt.xlabel("On-Time Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "on_time_rate_by_route.png"), dpi=150)
    plt.close()


def print_summary(df):
    print("=== Dataset summary ===")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Overall on-time rate: {(1 - df['is_delayed'].mean()) * 100:.1f}%")
    print(f"Mean delay: {df['delay_minutes'].mean():.2f} min | Median: {df['delay_minutes'].median():.2f} min")
    print()
    print("On-time rate by train type:")
    print((1 - df.groupby('train_type')['is_delayed'].mean()).sort_values(ascending=False).mul(100).round(1))


def main():
    parser = argparse.ArgumentParser(description="Run EDA on DB delay dataset")
    parser.add_argument("--data", type=str, default="data/db_delays.csv")
    parser.add_argument("--out", type=str, default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)

    print_summary(df)

    plot_delay_by_train_type(df, args.out)
    plot_delay_by_weather(df, args.out)
    plot_delay_by_month(df, args.out)
    plot_delay_by_hour(df, args.out)
    plot_construction_effect(df, args.out)
    plot_on_time_rate_by_route(df, args.out)

    print(f"\nSaved figures to {args.out}/")


if __name__ == "__main__":
    main()
