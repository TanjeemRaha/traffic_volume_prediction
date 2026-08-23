"""Regression metrics and grouped traffic-error summaries."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    actual: Iterable[float], predicted: Iterable[float]
) -> dict[str, float]:
    """Calculate metrics with units that are meaningful for traffic volume."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    absolute_error = np.abs(actual_array - predicted_array)
    denominator = np.abs(actual_array).sum()
    r_squared = (
        float(r2_score(actual_array, predicted_array))
        if len(actual_array) >= 2
        else float("nan")
    )
    return {
        "MAE_vehicles_per_hour": float(mean_absolute_error(actual_array, predicted_array)),
        "RMSE_vehicles_per_hour": float(
            np.sqrt(mean_squared_error(actual_array, predicted_array))
        ),
        "R_squared": r_squared,
        "WAPE_percent": float(100 * absolute_error.sum() / denominator)
        if denominator > 0
        else float("nan"),
    }


def grouped_error_summary(
    predictions: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    """Summarize prediction errors by group."""

    required = {group_column, "actual_traffic_volume", "predicted_traffic_volume"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError("Grouped error summary is missing: " + ", ".join(missing))

    rows: list[dict[str, object]] = []
    for group_value, group in predictions.groupby(group_column, dropna=False, sort=True):
        metrics = regression_metrics(
            group["actual_traffic_volume"], group["predicted_traffic_volume"]
        )
        rows.append(
            {
                group_column: group_value,
                "observations": len(group),
                **metrics,
                "mean_error_actual_minus_predicted": float(
                    (
                        group["actual_traffic_volume"]
                        - group["predicted_traffic_volume"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)
