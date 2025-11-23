import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Weather Data Visualizer")
    p.add_argument("--input", "-i", default="weather.csv", help="Path to input CSV (default: weather.csv)")
    p.add_argument("--out-dir", "-o", default=".", help="Output directory for plots and cleaned CSV")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(); useful in headless environments")
    return p.parse_args()


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["Date"], infer_datetime_format=True)
    except FileNotFoundError:
        print(f"Error: input file not found: {path}")
        sys.exit(1)
    except ValueError:
       
        df = pd.read_csv(path)
    return df


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Error: missing required columns in CSV: {missing}")
        sys.exit(1)


def make_plot_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    in_path = args.input
    out_dir = args.out_dir
    show_plots = not args.no_show

    make_plot_dir(out_dir)

    df = safe_read_csv(in_path)

    required = ["Date", "Temperature", "Humidity", "Rainfall"]
    ensure_columns(df, required)

   
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Humidity"] = pd.to_numeric(df["Humidity"], errors="coerce")
    df["Rainfall"] = pd.to_numeric(df["Rainfall"], errors="coerce").fillna(0)

    if df["Temperature"].notna().any():
        df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())
    else:
        df["Temperature"] = df["Temperature"].fillna(0)

    if df["Humidity"].notna().any():
        df["Humidity"] = df["Humidity"].fillna(df["Humidity"].mean())
    else:
        df["Humidity"] = df["Humidity"].fillna(0)

    df = df[["Date", "Temperature", "Humidity", "Rainfall"]]

    print("\n--- Cleaned Data Head ---")
    print(df.head().to_string(index=False))

    daily_mean = float(df["Temperature"].mean())
    daily_max = float(df["Temperature"].max())
    daily_min = float(df["Temperature"].min())
    std_dev = float(df["Temperature"].std(ddof=0))

    print("\n--- Temperature Statistics ---")
    print(f"Mean: {daily_mean:.2f}")
    print(f"Max: {daily_max:.2f}")
    print(f"Min: {daily_min:.2f}")
    print(f"Std Dev: {std_dev:.2f}")

    df["Month"] = df["Date"].dt.month
    monthly_stats = df.groupby("Month")["Temperature"].agg(["mean", "min", "max", "std"])  

    print("\n--- Monthly Temperature Stats ---")
    print(monthly_stats)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["Date"], df["Temperature"], marker=".")
    ax1.set_title("Daily Temperature Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (°C)")
    fig1.tight_layout()
    t1 = Path(out_dir) / "temperature_trend.png"
    fig1.savefig(t1, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig1)

    monthly_rainfall = df.groupby("Month")["Rainfall"].sum()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(monthly_rainfall.index, monthly_rainfall.values)
    ax2.set_title("Monthly Rainfall Total")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Rainfall (mm)")
    fig2.tight_layout()
    t2 = Path(out_dir) / "monthly_rainfall.png"
    fig2.savefig(t2, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(df["Temperature"], df["Humidity"], alpha=0.6)
    ax3.set_title("Humidity vs Temperature")
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Humidity (%)")
    fig3.tight_layout()
    t3 = Path(out_dir) / "humidity_vs_temperature.png"
    fig3.savefig(t3, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig3)

    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(1, 2, 1)
    ax4.plot(df["Date"], df["Temperature"], color="tab:orange")
    ax4.set_title("Daily Temperature")

    ax5 = fig4.add_subplot(1, 2, 2)
    ax5.bar(monthly_rainfall.index, monthly_rainfall.values, color="tab:blue")
    ax5.set_title("Monthly Rainfall")

    fig4.tight_layout()
    t4 = Path(out_dir) / "combined_plot.png"
    fig4.savefig(t4, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig4)

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    df["Season"] = df["Month"].map(season_map)
    season_stats = df.groupby("Season")[ ["Temperature", "Rainfall", "Humidity"] ].mean()

    print("\n--- Seasonal Averages ---")
    print(season_stats)

    cleaned_csv = Path(out_dir) / "cleaned_weather_data.csv"
    df.to_csv(cleaned_csv, index=False)

    summary_text = (
        f"Weather Data Analysis Summary\n\n"
        f"Temperature:\n"
        f"- Mean temperature: {daily_mean:.2f}°C\n"
        f"- Highest temperature: {daily_max:.2f}°C\n"
        f"- Lowest temperature: {daily_min:.2f}°C\n"
        f"- Standard deviation: {std_dev:.2f}\n\n"
        f"Rainfall:\n"
        f"- Monthly rainfall total saved to: {t2}\n\n"
        f"Humidity:\n"
        f"- Scatter plot saved to: {t3}\n\n"
        f"Seasonal Averages saved to cleaned CSV: {cleaned_csv}\n"
    )

    summary_file = Path(out_dir) / "weather_summary.txt"
    summary_file.write_text(summary_text)

    print(f"\nSaved cleaned CSV to: {cleaned_csv}")
    print(f"Saved plots to: {Path(out_dir).resolve()}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()