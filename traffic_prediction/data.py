"""Data acquisition, validation, and hourly aggregation for UCI I-94 data."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import certifi

from traffic_prediction.paths import (
    CLEANED_DATA_PATH,
    QUALITY_REPORT_PATH,
    RAW_CSV_PATH,
    RAW_DATA_DIR,
    RAW_GZIP_PATH,
    create_project_directories,
)


DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/492/"
    "metro%2Binterstate%2Btraffic%2Bvolume"
)
DATASET_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro%2Binterstate%2Btraffic%2Bvolume.zip"
)

EXPECTED_COLUMNS = {
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
}

NO_HOLIDAY_VALUES = {"", "none", "nan", "nat", "null"}


class DataValidationError(ValueError):
    """Raised when the supplied CSV cannot safely be used by this project."""


def _find_existing_raw_file() -> Path | None:
    for candidate in (RAW_GZIP_PATH, RAW_CSV_PATH):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_dataset(force: bool = False) -> Path:
    """Download and extract the official UCI dataset."""

    create_project_directories()
    existing = _find_existing_raw_file()
    if existing is not None and not force:
        print(f"Raw data already exists: {existing}")
        return existing

    temporary_zip = RAW_DATA_DIR / ".uci_i94_download.zip"
    request = urllib.request.Request(
        DATASET_DOWNLOAD_URL,
        headers={"User-Agent": "I94-Traffic-Prediction/1.0"},
    )

    print("Downloading the official UCI Metro Interstate Traffic Volume data...")
    try:
        verified_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            request, timeout=90, context=verified_context
        ) as response:
            with temporary_zip.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        with zipfile.ZipFile(temporary_zip) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".csv.gz", ".csv"))
            ]
            if not members:
                raise DataValidationError(
                    "The downloaded UCI ZIP did not contain a CSV file. "
                    f"Please check {DATASET_PAGE}."
                )

            member = sorted(members, key=lambda item: not item.lower().endswith(".csv.gz"))[0]
            destination = RAW_GZIP_PATH if member.lower().endswith(".gz") else RAW_CSV_PATH
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    except Exception as exc:
        raise RuntimeError(
            "The dataset could not be downloaded automatically. Check your internet "
            f"connection, or manually download it from {DATASET_PAGE} and place the "
            f"CSV or CSV.GZ file in {RAW_DATA_DIR}."
        ) from exc
    finally:
        temporary_zip.unlink(missing_ok=True)

    print(f"Saved raw data to: {destination}")
    return destination


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load a raw CSV and verify that all documented UCI columns are present."""

    raw_path = Path(path) if path is not None else _find_existing_raw_file()
    if raw_path is None or not raw_path.exists():
        raise FileNotFoundError(
            "Raw data was not found. Run `python 01_download_and_clean.py` first."
        )

    frame = pd.read_csv(raw_path, low_memory=False)
    missing_columns = sorted(EXPECTED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise DataValidationError(
            "The CSV is missing required column(s): "
            + ", ".join(missing_columns)
            + ". Confirm that this is UCI dataset 492."
        )
    return frame


def _normalise_text(series: pd.Series, missing_label: str) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isna() | values.eq(""), missing_label)


def _combine_text(values: pd.Series, missing_label: str = "Unknown") -> str:
    unique_values = {
        str(value).strip()
        for value in values.dropna()
        if str(value).strip() and str(value).strip().lower() not in {"nan", "null"}
    }
    unique_values.discard(missing_label)
    return "|".join(sorted(unique_values)) if unique_values else missing_label


def _holiday_for_hour(values: pd.Series) -> str:
    holidays = sorted(
        {
            str(value).strip()
            for value in values.dropna()
            if str(value).strip().lower() not in NO_HOLIDAY_VALUES
        }
    )
    return holidays[0] if holidays else "None"


def _plain_value(value: Any) -> Any:
    """Convert NumPy/pandas scalar values so JSON can store them."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def clean_traffic_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw observations into validated hourly records and an audit report."""

    missing_columns = sorted(EXPECTED_COLUMNS.difference(raw.columns))
    if missing_columns:
        raise DataValidationError(
            "Cannot clean the data because these columns are missing: "
            + ", ".join(missing_columns)
        )

    frame = raw.loc[:, sorted(EXPECTED_COLUMNS)].copy()
    rows_received = len(frame)
    exact_duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    frame["date_time"] = pd.to_datetime(
        frame["date_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "traffic_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_datetime_rows = int(frame["date_time"].isna().sum())
    valid_target = np.isfinite(frame["traffic_volume"]) & frame["traffic_volume"].ge(0)
    invalid_target_rows = int((~valid_target).sum())
    frame = frame.loc[
        frame["date_time"].notna()
        & valid_target
    ].copy()
    if frame.empty:
        raise DataValidationError(
            "No usable rows remain after checking timestamps and traffic-volume "
            "targets. Confirm that the correct UCI CSV was supplied."
        )

    frame["holiday"] = _normalise_text(frame["holiday"], "None")
    frame["weather_main"] = _normalise_text(frame["weather_main"], "Unknown")
    frame["weather_description"] = _normalise_text(
        frame["weather_description"], "Unknown"
    )

    duplicate_timestamp_rows = int(frame.duplicated("date_time", keep=False).sum())
    duplicate_timestamps = int(
        frame.loc[frame.duplicated("date_time", keep=False), "date_time"].nunique()
    )
    target_counts = frame.groupby("date_time", sort=False)["traffic_volume"].nunique()
    conflicting_target_timestamps = int((target_counts > 1).sum())
    if conflicting_target_timestamps:
        raise DataValidationError(
            f"Found {conflicting_target_timestamps} timestamp(s) with conflicting "
            "traffic-volume targets. The project stops instead of guessing which "
            "target is correct."
        )

    invalid_masks = {
        "temperature_outside_180_to_340_kelvin": frame["temp"].notna()
        & ~frame["temp"].between(180, 340),
        "negative_or_over_500_mm_rain": frame["rain_1h"].notna()
        & ~frame["rain_1h"].between(0, 500),
        "negative_or_over_200_mm_snow": frame["snow_1h"].notna()
        & ~frame["snow_1h"].between(0, 200),
        "cloud_cover_outside_0_to_100_percent": frame["clouds_all"].notna()
        & ~frame["clouds_all"].between(0, 100),
    }
    missing_weather_before_imputation = {
        column: int(frame[column].isna().sum())
        for column in ("temp", "rain_1h", "snow_1h", "clouds_all")
    }
    plausibility_counts = {
        label: int(mask.sum()) for label, mask in invalid_masks.items()
    }
    for column, label in (
        ("temp", "temperature_outside_180_to_340_kelvin"),
        ("rain_1h", "negative_or_over_500_mm_rain"),
        ("snow_1h", "negative_or_over_200_mm_snow"),
        ("clouds_all", "cloud_cover_outside_0_to_100_percent"),
    ):
        frame.loc[invalid_masks[label], column] = np.nan

    hourly = (
        frame.groupby("date_time", as_index=False, sort=True)
        .agg(
            holiday_name=("holiday", _holiday_for_hour),
            temp_kelvin=("temp", "median"),
            rain_1h=("rain_1h", "median"),
            snow_1h=("snow_1h", "median"),
            clouds_all=("clouds_all", "median"),
            weather_main=("weather_main", _combine_text),
            weather_description=("weather_description", _combine_text),
            traffic_volume=("traffic_volume", "first"),
        )
        .sort_values("date_time")
        .reset_index(drop=True)
    )

    date_keys = hourly["date_time"].dt.normalize()
    named_holidays = hourly.loc[hourly["holiday_name"].ne("None")].copy()
    named_holidays["calendar_date"] = named_holidays["date_time"].dt.normalize()
    holiday_by_date = named_holidays.groupby("calendar_date")["holiday_name"].first()
    hourly["holiday_name"] = date_keys.map(holiday_by_date).fillna("None")

    hourly["temp_celsius"] = hourly["temp_kelvin"] - 273.15
    hourly = hourly.drop(columns="temp_kelvin")
    hourly["traffic_volume"] = hourly["traffic_volume"].round().astype(int)

    expected_hours = 0
    missing_hours = 0
    time_gaps = 0
    if not hourly.empty:
        time_span_seconds = (
            hourly["date_time"].max() - hourly["date_time"].min()
        ).total_seconds()
        expected_hours = int(time_span_seconds // 3_600) + 1
        missing_hours = expected_hours - int(hourly["date_time"].nunique())
        time_gaps = int(
            (hourly["date_time"].diff().dt.total_seconds() > 3_600).sum()
        )

    report: dict[str, Any] = {
        "source": DATASET_PAGE,
        "rows_received": rows_received,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "rows_with_invalid_datetime_removed": invalid_datetime_rows,
        "rows_with_invalid_target_removed": invalid_target_rows,
        "rows_in_duplicate_timestamps_before_aggregation": duplicate_timestamp_rows,
        "duplicate_timestamps_aggregated": duplicate_timestamps,
        "conflicting_target_timestamps": conflicting_target_timestamps,
        "plausibility_issues_replaced_with_missing": plausibility_counts,
        "missing_or_unparseable_weather_values": missing_weather_before_imputation,
        "clean_hourly_rows": len(hourly),
        "start_time": hourly["date_time"].min() if not hourly.empty else None,
        "end_time": hourly["date_time"].max() if not hourly.empty else None,
        "expected_hours_between_start_and_end": expected_hours,
        "unobserved_hours_between_start_and_end": missing_hours,
        "number_of_time_gaps": time_gaps,
        "remaining_missing_values_by_column": {
            column: int(value) for column, value in hourly.isna().sum().items()
        },
        "notes": [
            "Duplicate timestamps were aggregated before chronological splitting.",
            "Missing traffic hours were not invented or interpolated.",
            "Weather values outside broad physical bounds became missing; model "
            "imputation is fitted using training data only.",
            "Holiday labels were expanded to every observed hour on that date.",
            "Timestamps are kept timezone-naive because the source does not provide "
            "an unambiguous UTC offset.",
        ],
    }
    return hourly, report


def save_cleaned_data(
    cleaned: pd.DataFrame,
    report: dict[str, Any],
    data_path: Path = CLEANED_DATA_PATH,
    report_path: Path = QUALITY_REPORT_PATH,
) -> None:
    """Save cleaned observations and their audit trail."""

    create_project_directories()
    cleaned.to_csv(data_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    serialisable_report = {
        key: (
            {nested_key: _plain_value(nested_value) for nested_key, nested_value in value.items()}
            if isinstance(value, dict)
            else [_plain_value(item) for item in value]
            if isinstance(value, list)
            else _plain_value(value)
        )
        for key, value in report.items()
    }
    report_path.write_text(
        json.dumps(serialisable_report, indent=2), encoding="utf-8"
    )


def load_cleaned_data(path: str | Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load cleaned hourly observations from disk."""

    cleaned_path = Path(path)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            "Cleaned data was not found. Run `python 01_download_and_clean.py` first."
        )
    frame = pd.read_csv(cleaned_path, parse_dates=["date_time"])
    return frame.sort_values("date_time").reset_index(drop=True)


def prepare_data(force_download: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download, clean, and persist the dataset."""

    raw_path = download_dataset(force=force_download)
    raw = load_raw_data(raw_path)
    cleaned, report = clean_traffic_data(raw)
    save_cleaned_data(cleaned, report)
    return cleaned, report
