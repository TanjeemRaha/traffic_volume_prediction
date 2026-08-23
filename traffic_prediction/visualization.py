"""Generate exploratory and model-evaluation figures."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import os
from pathlib import Path

from traffic_prediction.paths import PROJECT_ROOT

MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic_prediction.features import (
    add_interpretation_columns,
    make_chronological_split,
)
from traffic_prediction.paths import (
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    create_project_directories,
)


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _save_figure(figure: plt.Figure, filename: str) -> Path:
    path = FIGURES_DIR / filename
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def create_eda_outputs(
    cleaned: pd.DataFrame, include_held_out_periods: bool = False
) -> list[Path]:
    """Save EDA tables, plots, and a training-partition summary report."""

    create_project_directories()
    source = cleaned.sort_values("date_time").reset_index(drop=True)
    if include_held_out_periods:
        eda_source = source
        scope_note = "the full cleaned record (retrospective description)"
    else:
        split = make_chronological_split(pd.to_datetime(source["date_time"]))
        eda_source = source.iloc[split.train_indices].copy()
        scope_note = "the model-development training period only"
    data = add_interpretation_columns(eda_source)
    data["date_time"] = pd.to_datetime(data["date_time"])
    generated: list[Path] = []

    hourly = (
        data.groupby("hour")["traffic_volume"]
        .agg(mean="mean", median="median", lower_quartile=lambda x: x.quantile(0.25),
             upper_quartile=lambda x: x.quantile(0.75), observations="size")
        .reset_index()
    )
    hourly_path = METRICS_DIR / "traffic_pattern_by_hour.csv"
    hourly.to_csv(hourly_path, index=False)
    generated.append(hourly_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(hourly["hour"], hourly["mean"], marker="o", label="Mean")
    axis.plot(hourly["hour"], hourly["median"], linestyle="--", label="Median")
    axis.fill_between(
        hourly["hour"],
        hourly["lower_quartile"],
        hourly["upper_quartile"],
        alpha=0.18,
        label="Middle 50% of observations",
    )
    axis.set(
        title="Observed I-94 traffic volume by hour of day",
        xlabel="Hour of day (local time)",
        ylabel="Traffic volume (vehicles/hour)",
        xticks=range(0, 24, 2),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    generated.append(_save_figure(figure, "01_traffic_by_hour.png"))

    daily = (
        data.groupby("day_name")["traffic_volume"]
        .agg(mean="mean", median="median", observations="size")
        .reindex(DAY_ORDER)
        .reset_index()
    )
    daily_path = METRICS_DIR / "traffic_pattern_by_weekday.csv"
    daily.to_csv(daily_path, index=False)
    generated.append(daily_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(daily["day_name"], daily["mean"], color="#3978a8")
    axis.set(
        title="Average observed traffic volume by day of week",
        xlabel="Day of week",
        ylabel="Mean traffic volume (vehicles/hour)",
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    generated.append(_save_figure(figure, "02_traffic_by_weekday.png"))

    heatmap_data = data.pivot_table(
        index="day_name", columns="hour", values="traffic_volume", aggfunc="mean"
    ).reindex(index=DAY_ORDER, columns=range(24))
    figure, axis = plt.subplots(figsize=(11, 5))
    image = axis.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    axis.set(
        title="Mean traffic volume for each hour of the week",
        xlabel="Hour of day",
        ylabel="Day of week",
        xticks=np.arange(24),
        yticks=np.arange(7),
        yticklabels=DAY_ORDER,
    )
    figure.colorbar(image, ax=axis, label="Vehicles/hour")
    generated.append(_save_figure(figure, "03_hour_week_heatmap.png"))

    monthly = (
        data.groupby("month")["traffic_volume"]
        .agg(mean="mean", median="median", observations="size")
        .reindex(range(1, 13))
        .reset_index()
    )
    monthly_path = METRICS_DIR / "traffic_pattern_by_month.csv"
    monthly.to_csv(monthly_path, index=False)
    generated.append(monthly_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(monthly["month"], monthly["mean"], marker="o", color="#378b5b")
    axis.set(
        title="Average observed traffic volume by month",
        xlabel="Month",
        ylabel="Mean traffic volume (vehicles/hour)",
        xticks=range(1, 13),
        xticklabels=MONTH_NAMES,
    )
    axis.grid(alpha=0.25)
    generated.append(_save_figure(figure, "04_traffic_by_month.png"))

    monthly_timeline = (
        data.set_index("date_time")["traffic_volume"].resample("MS").mean().dropna()
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(monthly_timeline.index, monthly_timeline.values, color="#6d4b9b")
    axis.set(
        title="Monthly mean traffic volume through the observation period",
        xlabel="Calendar month",
        ylabel="Mean traffic volume (vehicles/hour)",
    )
    axis.grid(alpha=0.25)
    generated.append(_save_figure(figure, "05_monthly_traffic_timeline.png"))

    weather_long = data.assign(
        weather_condition=data["weather_main"].fillna("Unknown").str.split("|")
    ).explode("weather_condition")
    weather_long["weather_condition"] = weather_long["weather_condition"].str.strip()
    weather_summary = (
        weather_long.groupby("weather_condition")["traffic_volume"]
        .agg(mean="mean", median="median", observations="size")
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    weather_path = METRICS_DIR / "traffic_pattern_by_weather.csv"
    weather_summary.to_csv(weather_path, index=False)
    generated.append(weather_path)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        weather_summary["weather_condition"],
        weather_summary["mean"],
        color="#4b859b",
    )
    axis.set(
        title="Mean traffic volume by reported weather condition (descriptive)",
        xlabel="Reported broad weather condition",
        ylabel="Mean traffic volume (vehicles/hour)",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    generated.append(_save_figure(figure, "05b_traffic_by_weather_condition.png"))

    weekend_summary = (
        data.groupby("is_weekend")["traffic_volume"]
        .agg(mean="mean", median="median", observations="size")
        .reindex([False, True])
    )
    weekday_mean = float(weekend_summary.loc[False, "mean"])
    weekend_mean = float(weekend_summary.loc[True, "mean"])
    busiest_hour = int(hourly.loc[hourly["mean"].idxmax(), "hour"])
    busiest_day = str(daily.loc[daily["mean"].idxmax(), "day_name"])
    highest_month = int(monthly.loc[monthly["mean"].idxmax(), "month"])

    summary = f"""# Exploratory traffic summary

This report describes patterns in {scope_note}. It does not prove that the
calendar or weather *caused* a traffic change. Validation and final-test target
values are excluded from the default exploratory analysis.

- Observation period: {data['date_time'].min():%Y-%m-%d} to {data['date_time'].max():%Y-%m-%d}
- Unique observed hours: {len(data):,}
- Highest mean-volume hour: {busiest_hour:02d}:00
- Day with the highest mean hourly volume: {busiest_day}
- Month with the highest mean hourly volume: {MONTH_NAMES[highest_month - 1]}
- Weekday mean: {weekday_mean:,.0f} vehicles/hour
- Weekend mean: {weekend_mean:,.0f} vehicles/hour

Use the hour-of-week heatmap to see commute patterns more clearly. The weather
chart is descriptive: weather is tied to season and time of day, and a mean
difference is not a causal weather effect. Some hours list multiple conditions
and therefore appear in more than one weather bar. Missing hours are not
interpolated in this project.
"""
    summary_path = REPORTS_DIR / "eda_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    generated.append(summary_path)
    return generated


def create_model_plots() -> list[Path]:
    """Create performance and interpretation plots from saved model outputs."""

    create_project_directories()
    metrics_path = METRICS_DIR / "model_comparison.csv"
    importance_path = METRICS_DIR / "permutation_importance.csv"
    predictions_path = PREDICTIONS_DIR / "final_test_predictions.csv"
    for required in (metrics_path, importance_path, predictions_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Model output is missing: {required}. Run step 3 first."
            )

    metrics = pd.read_csv(metrics_path)
    importance = pd.read_csv(importance_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["date_time"])
    generated: list[Path] = []

    validation = metrics.loc[metrics["partition"].eq("validation")].sort_values(
        "MAE_vehicles_per_hour"
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(validation["model"], validation["MAE_vehicles_per_hour"], color="#3978a8")
    axis.invert_yaxis()
    axis.set(
        title="Model comparison on the validation period",
        xlabel="MAE (vehicles/hour; lower is better)",
        ylabel="Model",
    )
    axis.grid(axis="x", alpha=0.25)
    generated.append(_save_figure(figure, "06_validation_model_comparison.png"))

    start = predictions["date_time"].min()
    trace = predictions.loc[predictions["date_time"] < start + pd.Timedelta(days=14)]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        trace["date_time"], trace["actual_traffic_volume"], label="Observed", linewidth=1.4
    )
    axis.plot(
        trace["date_time"],
        trace["predicted_traffic_volume"],
        label="Predicted",
        linewidth=1.2,
        alpha=0.85,
    )
    axis.set(
        title="Final-test traffic: first 14 calendar days",
        xlabel="Date and time",
        ylabel="Traffic volume (vehicles/hour)",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    generated.append(_save_figure(figure, "07_test_actual_vs_predicted_trace.png"))

    sample = predictions.iloc[
        np.linspace(0, len(predictions) - 1, min(3_000, len(predictions)), dtype=int)
    ]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        sample["actual_traffic_volume"],
        sample["predicted_traffic_volume"],
        s=10,
        alpha=0.25,
    )
    upper = float(
        max(sample["actual_traffic_volume"].max(), sample["predicted_traffic_volume"].max())
    )
    axis.plot([0, upper], [0, upper], linestyle="--", color="black", label="Perfect prediction")
    axis.set(
        title="Observed versus predicted final-test volume",
        xlabel="Observed vehicles/hour",
        ylabel="Predicted vehicles/hour",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    generated.append(_save_figure(figure, "08_test_observed_vs_predicted.png"))

    error_by_hour = pd.read_csv(METRICS_DIR / "error_by_hour.csv")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        error_by_hour["hour"],
        error_by_hour["MAE_vehicles_per_hour"],
        marker="o",
        color="#aa4c43",
    )
    axis.set(
        title="Final-test prediction error by hour",
        xlabel="Hour of day",
        ylabel="MAE (vehicles/hour)",
        xticks=range(0, 24, 2),
    )
    axis.grid(alpha=0.25)
    generated.append(_save_figure(figure, "09_test_error_by_hour.png"))

    top_importance = importance.head(12).sort_values(
        "importance_MAE_increase_mean", ascending=True
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        top_importance["feature"],
        top_importance["importance_MAE_increase_mean"],
        color="#378b5b",
    )
    axis.set(
        title="Permutation importance on a final-test sample",
        xlabel="Increase in MAE after shuffling (larger = more predictive)",
        ylabel="Original feature",
    )
    axis.grid(axis="x", alpha=0.25)
    generated.append(_save_figure(figure, "10_permutation_importance.png"))
    return generated
