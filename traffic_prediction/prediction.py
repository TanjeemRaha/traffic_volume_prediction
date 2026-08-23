"""Load the saved model and estimate traffic for one user-supplied hour."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import math
import re
from pathlib import Path

import joblib
import pandas as pd

from traffic_prediction.features import create_features
from traffic_prediction.modeling import predict_nonnegative
from traffic_prediction.paths import (
    BEST_MODEL_PATH,
    PREDICTIONS_DIR,
    create_project_directories,
)


def predict_one_hour(
    date_time: str,
    temp_celsius: float,
    rain_mm: float = 0.0,
    snow_mm: float = 0.0,
    cloud_percent: float = 40.0,
    weather_main: str = "Clouds",
    holiday_name: str = "None",
    model_path: str | Path = BEST_MODEL_PATH,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Predict vehicles/hour from calendar and observed or forecast weather."""

    artifact_path = Path(model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run `python 03_train_models.py` first."
        )
    artifact = joblib.load(artifact_path)
    required_keys = {
        "model",
        "model_name",
        "feature_origin",
        "target_training_quantiles",
        "uses_weather",
        "inputs_used",
        "training_through",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise ValueError(
            "The saved model artifact is incompatible; missing: "
            + ", ".join(missing_keys)
        )

    date_text = str(date_time)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", date_text) is None:
        raise ValueError(
            "date_time must use the exact local format `YYYY-MM-DD HH:MM`, "
            "for example `2019-01-07 08:00`."
        )
    try:
        timestamp = pd.to_datetime(
            date_text, format="%Y-%m-%d %H:%M", errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "date_time must use the exact local format `YYYY-MM-DD HH:MM`, "
            "for example `2019-01-07 08:00`."
        ) from exc
    if timestamp.minute != 0 or timestamp.second != 0:
        raise ValueError("date_time must be aligned to the start of an hour (minute 00).")
    if timestamp.tzinfo is not None:
        raise ValueError(
            "date_time must be local clock time without a timezone offset, matching "
            "the source data (for example `2019-01-07 08:00`)."
        )
    uses_weather = bool(artifact["uses_weather"])
    if uses_weather:
        numeric_weather = (temp_celsius, rain_mm, snow_mm, cloud_percent)
        if not all(math.isfinite(float(value)) for value in numeric_weather):
            raise ValueError(
                "Temperature, rain, snow, and cloud values must be finite numbers."
            )
        if not 0 <= rain_mm <= 500:
            raise ValueError("Rain must be between 0 and 500 mm for the hour.")
        if not 0 <= snow_mm <= 200:
            raise ValueError("Snow must be between 0 and 200 mm for the hour.")
        if not 0 <= cloud_percent <= 100:
            raise ValueError("Cloud percentage must be between 0 and 100.")
        if not -93.15 <= temp_celsius <= 66.85:
            raise ValueError(
                "Temperature must be supplied in Celsius, between -93.15 and 66.85."
            )

    model_temp = float(temp_celsius) if uses_weather else 0.0
    model_rain = float(rain_mm) if uses_weather else 0.0
    model_snow = float(snow_mm) if uses_weather else 0.0
    model_cloud = float(cloud_percent) if uses_weather else 0.0
    model_weather_main = weather_main if uses_weather else "Unknown"

    supplied = pd.DataFrame(
        [
            {
                "date_time": timestamp,
                "holiday_name": holiday_name,
                "temp_celsius": model_temp,
                "rain_1h": model_rain,
                "snow_1h": model_snow,
                "clouds_all": model_cloud,
                "weather_main": model_weather_main,
            }
        ]
    )
    features = create_features(supplied, origin=artifact["feature_origin"])
    prediction = float(predict_nonnegative(artifact["model"], features)[0])

    quantiles = artifact["target_training_quantiles"]
    if prediction < quantiles["25_percent"]:
        historical_band = "lower quarter of historical hourly traffic"
    elif prediction > quantiles["75_percent"]:
        historical_band = "upper quarter of historical hourly traffic"
    else:
        historical_band = "middle half of historical hourly traffic"

    result: dict[str, object] = {
        "date_time": timestamp,
        "predicted_traffic_volume": round(prediction, 1),
        "unit": "vehicles/hour",
        "historical_volume_band": historical_band,
        "model_name": artifact["model_name"],
        "inputs_used": ", ".join(artifact["inputs_used"]),
        "uses_weather": uses_weather,
        "temp_celsius": float(temp_celsius),
        "rain_mm": float(rain_mm),
        "snow_mm": float(snow_mm),
        "cloud_percent": float(cloud_percent),
        "weather_main": weather_main,
        "holiday_name": holiday_name,
    }
    training_from = pd.Timestamp(artifact["feature_origin"])
    training_through = pd.Timestamp(artifact["training_through"])
    if timestamp < training_from:
        result["extrapolation_note"] = (
            f"Scenario is before the historical model period beginning "
            f"{training_from:%Y-%m-%d}; treat it as backward extrapolation."
        )
    elif timestamp > training_through:
        result["extrapolation_note"] = (
            f"Scenario is after the model-fit period ending "
            f"{training_through:%Y-%m-%d}; treat it as an extrapolative scenario "
            "and monitor accuracy."
        )
    else:
        result["extrapolation_note"] = "Scenario is within the historical model-fit period."
    if artifact["model_name"] == "Global median baseline":
        result["ignored_inputs"] = "all scenario inputs, including date/time"
    elif not uses_weather:
        result["ignored_inputs"] = (
            "temperature, rain, snow, clouds, weather condition, and holiday"
        )
    else:
        result["ignored_inputs"] = "None"

    create_project_directories()
    destination = (
        Path(output_path)
        if output_path is not None
        else PREDICTIONS_DIR / "example_prediction.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(destination, index=False)
    result["saved_to"] = destination
    return result
