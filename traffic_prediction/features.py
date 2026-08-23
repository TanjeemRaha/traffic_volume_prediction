"""Feature engineering and chronological data partitioning."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic_prediction.data import DataValidationError


WEATHER_CONDITIONS = (
    "clear",
    "clouds",
    "drizzle",
    "fog",
    "haze",
    "mist",
    "rain",
    "smoke",
    "snow",
    "squall",
    "thunderstorm",
)

NUMERIC_FEATURES = [
    "temp_celsius",
    "rain_1h_log",
    "snow_1h_log",
    "clouds_all",
    "days_since_start",
    "day_of_year_sin",
    "day_of_year_cos",
    *[f"weather_{condition}" for condition in WEATHER_CONDITIONS],
    "weather_other",
]

CATEGORICAL_FEATURES = [
    "hour_of_week",
    "month",
    "holiday_name",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_features(
    cleaned: pd.DataFrame, origin: pd.Timestamp | str | None = None
) -> pd.DataFrame:
    """Build target-free calendar and weather features.

    ``origin`` establishes the reference date for the trend feature.
    """

    required = {
        "date_time",
        "holiday_name",
        "temp_celsius",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
    }
    missing = sorted(required.difference(cleaned.columns))
    if missing:
        raise DataValidationError(
            "Feature engineering needs these cleaned column(s): " + ", ".join(missing)
        )
    if cleaned.empty:
        raise DataValidationError("Feature engineering received an empty table.")

    timestamps = pd.to_datetime(cleaned["date_time"], errors="coerce")
    if timestamps.isna().any():
        raise DataValidationError("Some cleaned date_time values could not be parsed.")

    origin_timestamp = (
        pd.Timestamp(origin) if origin is not None else timestamps.min().normalize()
    )
    features = pd.DataFrame(index=cleaned.index)

    day_of_week = timestamps.dt.dayofweek
    hour = timestamps.dt.hour
    day_of_year = timestamps.dt.dayofyear

    features["hour_of_week"] = (day_of_week * 24 + hour).astype(str)
    features["month"] = timestamps.dt.month.astype(str)
    features["holiday_name"] = cleaned["holiday_name"].fillna("None").astype(str)

    features["temp_celsius"] = pd.to_numeric(
        cleaned["temp_celsius"], errors="coerce"
    )
    rain = pd.to_numeric(cleaned["rain_1h"], errors="coerce").clip(lower=0)
    snow = pd.to_numeric(cleaned["snow_1h"], errors="coerce").clip(lower=0)
    features["rain_1h_log"] = np.log1p(rain)
    features["snow_1h_log"] = np.log1p(snow)
    features["clouds_all"] = pd.to_numeric(cleaned["clouds_all"], errors="coerce")
    features["days_since_start"] = (
        timestamps - origin_timestamp
    ).dt.total_seconds() / 86_400.0
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    weather_text = cleaned["weather_main"].fillna("Unknown").astype(str).str.lower()
    known_weather = pd.Series(False, index=cleaned.index)
    for condition in WEATHER_CONDITIONS:
        flag = weather_text.str.split("|", regex=False).apply(
            lambda parts, item=condition: item in {part.strip() for part in parts}
        )
        features[f"weather_{condition}"] = flag.astype(int)
        known_weather = known_weather | flag
    features["weather_other"] = (~known_weather).astype(int)

    return features.loc[:, MODEL_FEATURES]


def add_interpretation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add temporal groups used in plots and error summaries."""

    result = frame.copy()
    timestamps = pd.to_datetime(result["date_time"])
    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["day_name"] = timestamps.dt.day_name()
    result["month"] = timestamps.dt.month
    result["is_weekend"] = result["day_of_week"].ge(5)

    hour = result["hour"]
    result["travel_period"] = np.select(
        [
            hour.between(0, 5),
            hour.between(6, 9),
            hour.between(10, 14),
            hour.between(15, 18),
        ],
        ["Overnight", "Morning commute", "Midday", "Evening commute"],
        default="Evening",
    )
    return result


@dataclass(frozen=True)
class ChronologicalSplit:
    """Row indices and metadata for chronological partitions."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    method: str
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def make_chronological_split(timestamps: pd.Series) -> ChronologicalSplit:
    """Use the final two calendar years for validation and test when possible.

    For the UCI data this means training through 2016, validation in 2017, and
    final testing in 2018. Small synthetic or replacement datasets fall back to
    ordered 70/15/15 row partitions.
    """

    parsed = pd.Series(pd.to_datetime(timestamps, errors="coerce")).reset_index(drop=True)
    if parsed.isna().any():
        raise DataValidationError("Cannot split data containing invalid timestamps.")
    if len(parsed) < 20:
        raise DataValidationError(
            "At least 20 chronological observations are required for train, "
            "validation, and test partitions."
        )
    if not parsed.is_monotonic_increasing:
        raise DataValidationError(
            "Rows must be sorted by date_time before chronological splitting."
        )
    if parsed.duplicated().any():
        raise DataValidationError(
            "Chronological splitting requires one row per timestamp. Run the "
            "cleaning step to aggregate duplicate weather records first."
        )

    years = sorted(int(year) for year in parsed.dt.year.unique())
    positions = np.arange(len(parsed))
    if len(years) >= 3:
        validation_year = years[-2]
        test_year = years[-1]
        train_mask = parsed.dt.year < validation_year
        validation_mask = parsed.dt.year.eq(validation_year)
        test_mask = parsed.dt.year.eq(test_year)

        if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) >= 20:
            train_indices = positions[train_mask.to_numpy()]
            validation_indices = positions[validation_mask.to_numpy()]
            test_indices = positions[test_mask.to_numpy()]
            return ChronologicalSplit(
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                method=(
                    f"calendar years: train before {validation_year}, "
                    f"validate {validation_year}, test {test_year}"
                ),
                train_end=parsed.iloc[train_indices[-1]],
                validation_end=parsed.iloc[validation_indices[-1]],
            )

    train_end_position = max(1, int(len(parsed) * 0.70))
    validation_end_position = max(train_end_position + 1, int(len(parsed) * 0.85))
    validation_end_position = min(validation_end_position, len(parsed) - 1)
    train_indices = positions[:train_end_position]
    validation_indices = positions[train_end_position:validation_end_position]
    test_indices = positions[validation_end_position:]
    return ChronologicalSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        method="ordered row proportions: 70% train, 15% validation, 15% test",
        train_end=parsed.iloc[train_indices[-1]],
        validation_end=parsed.iloc[validation_indices[-1]],
    )
