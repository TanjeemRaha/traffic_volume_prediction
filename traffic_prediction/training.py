"""Train simple models, select on 2017, then evaluate on the 2018 holdout."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from traffic_prediction.evaluation import grouped_error_summary, regression_metrics
from traffic_prediction.features import (
    MODEL_FEATURES,
    add_interpretation_columns,
    create_features,
    make_chronological_split,
)
from traffic_prediction.modeling import (
    RANDOM_SEED,
    WEATHER_FREE_MODEL_NAMES,
    build_models,
    predict_nonnegative,
)
from traffic_prediction.paths import (
    BEST_MODEL_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    create_project_directories,
)


def _negative_clipped_mae(estimator, features: pd.DataFrame, target: pd.Series) -> float:
    """Permutation-importance scorer consistent with nonnegative count outputs."""

    predicted = predict_nonnegative(estimator, features)
    return -float(np.mean(np.abs(np.asarray(target, dtype=float) - predicted)))


def _format_period(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"


def _write_model_report(
    path: Path,
    best_name: str,
    validation_results: pd.DataFrame,
    test_metrics: dict[str, float],
    split_method: str,
    train_period: str,
    validation_period: str,
    test_period: str,
    travel_period_errors: pd.DataFrame,
    selection_reason: str,
    uses_weather: bool,
) -> None:
    comparison_columns = [
        "model",
        "MAE_vehicles_per_hour",
        "RMSE_vehicles_per_hour",
        "R_squared",
        "WAPE_percent",
    ]
    comparison_text = validation_results[comparison_columns].round(3).to_string(index=False)
    hardest_period = travel_period_errors.sort_values(
        "MAE_vehicles_per_hour", ascending=False
    ).iloc[0]
    lowest_rmse_model = validation_results.sort_values(
        "RMSE_vehicles_per_hour"
    ).iloc[0]
    selected_validation = validation_results.loc[
        validation_results["model"].eq(best_name)
    ].iloc[0]
    if str(lowest_rmse_model["model"]) != best_name:
        metric_tradeoff = (
            f"The {lowest_rmse_model['model']} had the lowest validation RMSE "
            f"({lowest_rmse_model['RMSE_vehicles_per_hour']:.1f} versus "
            f"{selected_validation['RMSE_vehicles_per_hour']:.1f} vehicles/hour for "
            "the selected model). MAE was the predeclared selection metric, but "
            "an agency that prioritizes avoiding occasional large errors could "
            "reasonably choose a different rule before rerunning the experiment."
        )
    else:
        metric_tradeoff = (
            "The selected model also had the lowest validation RMSE, so the MAE "
            "and large-error criteria agreed in this run."
        )
    weather_candidates = validation_results.loc[
        ~validation_results["model"].isin(WEATHER_FREE_MODEL_NAMES)
    ].sort_values("MAE_vehicles_per_hour")
    if uses_weather:
        input_finding = (
            "The selected method uses calendar and weather inputs. For a future "
            "hour, its weather fields must come from a weather forecast."
        )
    else:
        strongest_weather_model = weather_candidates.iloc[0]
        difference = (
            float(strongest_weather_model["MAE_vehicles_per_hour"])
            - float(
                validation_results.loc[
                    validation_results["model"].eq(best_name),
                    "MAE_vehicles_per_hour",
                ].iloc[0]
            )
        )
        if difference >= 0:
            weather_comparison = f"{difference:.1f} vehicles/hour worse than the winner"
        else:
            weather_comparison = (
                f"{abs(difference):.1f} vehicles/hour lower, but below the "
                "predeclared neural-network complexity threshold"
            )
        winner_scope = (
            "constant history-only"
            if best_name == "Global median baseline"
            else "schedule-only"
        )
        input_finding = (
            f"The selected method is {winner_scope} and does not use weather. The "
            f"strongest calendar-and-weather candidate was "
            f"{strongest_weather_model['model']} at "
            f"{strongest_weather_model['MAE_vehicles_per_hour']:.1f} validation "
            f"MAE, {weather_comparison}. This does "
            "not prove that weather is irrelevant; it means the tested weather "
            "models did not improve the predeclared selection metric."
        )

    report = f"""# Model results: I-94 hourly traffic volume

## Final outcome

The **{best_name}** was selected before the final holdout was evaluated within
this run.
Selection rule: {selection_reason}

- Final-test MAE: {test_metrics['MAE_vehicles_per_hour']:.1f} vehicles/hour
- Final-test RMSE: {test_metrics['RMSE_vehicles_per_hour']:.1f} vehicles/hour
- Final-test R-squared: {test_metrics['R_squared']:.3f}
- Final-test WAPE: {test_metrics['WAPE_percent']:.1f}%

MAE means that a typical prediction missed the observed hourly count by about
{test_metrics['MAE_vehicles_per_hour']:.0f} vehicles. RMSE is larger when a model
makes occasional large errors, which matters around demand peaks.

## Chronological experiment

- Split rule: {split_method}
- Training: {train_period}
- Validation/model choice: {validation_period}
- Final chronological holdout: {test_period}

## Validation comparison

```text
{comparison_text}
```

{metric_tradeoff}

## Verification status

The code enforces the train-then-validate-then-holdout order on every run.
During software implementation, the workflow was rerun after a runtime-motivated
LinearSVR solver adjustment, so these 2018 figures are an out-of-time
reproducibility demonstration, not a preregistered never-before-seen
confirmatory test. Freeze the code and use newly collected later data (or a new
local station dataset) for a future confirmatory assessment.

The largest test MAE among the defined daily periods occurred during
**{hardest_period['travel_period']}**
({hardest_period['MAE_vehicles_per_hour']:.1f} vehicles/hour). This is useful for
transportation planning because an acceptable overall error can hide weaker
performance during operationally important hours.

## Correct interpretation

{input_finding}

This is a traffic-volume estimate for one westbound I-94 count station. Weather
relationships are associations, not causal estimates. The data omit incidents,
work zones, events, lane availability, speed, and upstream conditions, so
unusual congestion cannot always be anticipated.

The model should not be transferred to another road or direction without local
data and a new chronological evaluation.
"""
    path.write_text(report, encoding="utf-8")


def train_and_evaluate(
    cleaned: pd.DataFrame,
    include_neural_network: bool = False,
    fast_mode: bool = False,
) -> dict[str, object]:
    """Run model selection, final testing, interpretation, and artifact saving."""

    create_project_directories()
    cleaned = cleaned.copy()
    cleaned["date_time"] = pd.to_datetime(cleaned["date_time"], errors="coerce")
    if cleaned["date_time"].isna().any():
        raise ValueError("Training data contains an invalid date_time value.")
    cleaned = cleaned.sort_values("date_time").reset_index(drop=True)
    origin = cleaned["date_time"].min().normalize()
    features = create_features(cleaned, origin=origin)
    target = pd.to_numeric(cleaned["traffic_volume"], errors="raise").astype(float)
    if not np.isfinite(target).all() or target.lt(0).any():
        raise ValueError(
            "Training traffic_volume values must be finite, nonnegative numbers."
        )
    split = make_chronological_split(cleaned["date_time"])

    X_train = features.iloc[split.train_indices]
    y_train = target.iloc[split.train_indices]
    X_validation = features.iloc[split.validation_indices]
    y_validation = target.iloc[split.validation_indices]

    models = build_models(
        include_neural_network=include_neural_network, fast_mode=fast_mode
    )
    validation_rows: list[dict[str, object]] = []

    print(f"Chronological split: {split.method}")
    print(f"Training observations:   {len(split.train_indices):,}")
    print(f"Validation observations: {len(split.validation_indices):,}")
    print(f"Final-test observations: {len(split.test_indices):,}\n")

    for model_name, model_template in models.items():
        print(f"Training {model_name}...")
        start = time.perf_counter()
        candidate = clone(model_template)
        candidate.fit(X_train, y_train)
        predicted = predict_nonnegative(candidate, X_validation)
        elapsed = time.perf_counter() - start
        metrics = regression_metrics(y_validation, predicted)
        validation_rows.append(
            {
                "model": model_name,
                "partition": "validation",
                **metrics,
                "fit_and_predict_seconds": elapsed,
            }
        )
        print(
            f"  validation MAE = {metrics['MAE_vehicles_per_hour']:.1f} "
            f"vehicles/hour ({elapsed:.1f} seconds)"
        )
        del candidate

    validation_results = pd.DataFrame(validation_rows).sort_values(
        "MAE_vehicles_per_hour"
    )
    best_name = str(validation_results.iloc[0]["model"])
    selection_reason = "lowest validation MAE"
    if best_name == "Small neural network (MLP)":
        classical_results = validation_results.loc[
            validation_results["model"].ne("Small neural network (MLP)")
        ]
        best_classical = classical_results.iloc[0]
        neural_mae = float(validation_results.iloc[0]["MAE_vehicles_per_hour"])
        classical_mae = float(best_classical["MAE_vehicles_per_hour"])
        neural_improvement = (classical_mae - neural_mae) / classical_mae
        if neural_improvement < 0.02:
            best_name = str(best_classical["model"])
            selection_reason = (
                "the neural network improved validation MAE by only "
                f"{100 * neural_improvement:.2f}%, below the predeclared 2% "
                "complexity threshold; the best classical model was retained"
            )
        else:
            selection_reason = (
                "lowest validation MAE, with a "
                f"{100 * neural_improvement:.2f}% improvement over the best "
                "classical model (above the predeclared 2% threshold)"
            )
    print(f"\nSelected on validation data: {best_name}")

    train_validation_indices = np.concatenate(
        [split.train_indices, split.validation_indices]
    )
    final_model = clone(models[best_name])
    final_model.fit(
        features.iloc[train_validation_indices], target.iloc[train_validation_indices]
    )

    X_test = features.iloc[split.test_indices]
    y_test = target.iloc[split.test_indices]
    test_predictions = predict_nonnegative(final_model, X_test)
    test_metrics = regression_metrics(y_test, test_predictions)
    print(
        "Final-test MAE: "
        f"{test_metrics['MAE_vehicles_per_hour']:.1f} vehicles/hour"
    )

    metrics_table = pd.concat(
        [
            validation_results,
            pd.DataFrame(
                [
                    {
                        "model": best_name,
                        "partition": "final_test",
                        **test_metrics,
                        "fit_and_predict_seconds": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics_path = METRICS_DIR / "model_comparison.csv"
    metrics_table.to_csv(metrics_path, index=False)

    prediction_context = add_interpretation_columns(
        cleaned.iloc[split.test_indices].reset_index(drop=True)
    )
    prediction_table = pd.DataFrame(
        {
            "date_time": prediction_context["date_time"],
            "actual_traffic_volume": y_test.to_numpy(),
            "predicted_traffic_volume": test_predictions,
            "residual_actual_minus_predicted": y_test.to_numpy() - test_predictions,
            "absolute_error": np.abs(y_test.to_numpy() - test_predictions),
            "hour": prediction_context["hour"],
            "day_name": prediction_context["day_name"],
            "month": prediction_context["month"],
            "is_weekend": prediction_context["is_weekend"],
            "travel_period": prediction_context["travel_period"],
            "holiday_name": prediction_context["holiday_name"],
            "weather_main": prediction_context["weather_main"],
        }
    )
    prediction_path = PREDICTIONS_DIR / "final_test_predictions.csv"
    prediction_table.to_csv(prediction_path, index=False)

    group_summaries: dict[str, pd.DataFrame] = {}
    for group_column, filename in (
        ("travel_period", "error_by_travel_period.csv"),
        ("hour", "error_by_hour.csv"),
        ("is_weekend", "error_by_weekend.csv"),
        ("month", "error_by_month.csv"),
    ):
        summary = grouped_error_summary(prediction_table, group_column)
        summary.to_csv(METRICS_DIR / filename, index=False)
        group_summaries[group_column] = summary

    sample_size = min(2_000, len(X_test))
    sample_positions = np.linspace(0, len(X_test) - 1, sample_size, dtype=int)
    importance = permutation_importance(
        final_model,
        X_test.iloc[sample_positions],
        y_test.iloc[sample_positions],
        scoring=_negative_clipped_mae,
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_MAE_increase_mean": importance.importances_mean,
            "importance_standard_deviation": importance.importances_std,
        }
    ).sort_values("importance_MAE_increase_mean", ascending=False)
    importance_path = METRICS_DIR / "permutation_importance.csv"
    importance_table.to_csv(importance_path, index=False)

    uses_weather = best_name not in WEATHER_FREE_MODEL_NAMES
    if best_name == "Global median baseline":
        inputs_used = ["no scenario input; constant fitted training median"]
        prediction_task = "Constant historical-median reference estimate"
    elif best_name == "Historical hour-of-week baseline":
        inputs_used = ["local date/time converted to hour_of_week"]
        prediction_task = "Schedule-only hour-of-week traffic-volume estimate"
    else:
        inputs_used = MODEL_FEATURES
        prediction_task = (
            "Same-hour traffic-volume estimation using calendar information and "
            "observed or forecast weather"
        )

    artifact = {
        "model": final_model,
        "model_name": best_name,
        "feature_origin": origin.isoformat(),
        "model_features": MODEL_FEATURES,
        "prediction_task": prediction_task,
        "inputs_used": inputs_used,
        "uses_weather": uses_weather,
        "selected_using": selection_reason,
        "split_method": split.method,
        "training_through": split.validation_end.isoformat(),
        "target_training_quantiles": {
            "25_percent": float(target.iloc[train_validation_indices].quantile(0.25)),
            "50_percent": float(target.iloc[train_validation_indices].quantile(0.50)),
            "75_percent": float(target.iloc[train_validation_indices].quantile(0.75)),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(artifact, BEST_MODEL_PATH)

    periods = {
        "train": _format_period(
            cleaned.iloc[split.train_indices[0]]["date_time"],
            cleaned.iloc[split.train_indices[-1]]["date_time"],
        ),
        "validation": _format_period(
            cleaned.iloc[split.validation_indices[0]]["date_time"],
            cleaned.iloc[split.validation_indices[-1]]["date_time"],
        ),
        "test": _format_period(
            cleaned.iloc[split.test_indices[0]]["date_time"],
            cleaned.iloc[split.test_indices[-1]]["date_time"],
        ),
    }
    experiment_path = REPORTS_DIR / "experiment_metadata.json"
    experiment_path.write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "split_method": split.method,
                "periods": periods,
                "selected_model": best_name,
                "selection_reason": selection_reason,
                "neural_network_included": include_neural_network,
                "fast_mode": fast_mode,
                "software_versions": artifact["software_versions"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = REPORTS_DIR / "model_results.md"
    _write_model_report(
        report_path,
        best_name,
        validation_results,
        test_metrics,
        split.method,
        periods["train"],
        periods["validation"],
        periods["test"],
        group_summaries["travel_period"],
        selection_reason,
        uses_weather,
    )

    return {
        "best_model_name": best_name,
        "test_metrics": test_metrics,
        "metrics_path": metrics_path,
        "predictions_path": prediction_path,
        "importance_path": importance_path,
        "model_path": BEST_MODEL_PATH,
        "report_path": report_path,
    }
