"""Focused tests for the raw-to-hourly data-cleaning rules."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from traffic_prediction.data import DataValidationError, clean_traffic_data


def _raw_row(date_time: str, **changes: object) -> dict[str, object]:
    """Build one minimal row with all columns required by the UCI cleaner."""

    row: dict[str, object] = {
        "holiday": "None",
        "temp": 280.0,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 20.0,
        "weather_main": "Clear",
        "weather_description": "sky is clear",
        "date_time": date_time,
        "traffic_volume": 100,
    }
    row.update(changes)
    return row


class CleanTrafficDataTests(unittest.TestCase):
    def test_collapses_duplicate_hours_expands_holiday_and_flags_invalid_weather(self):
        holiday_row = _raw_row(
            "2017-01-02 00:00:00",
            holiday="New Years Day",
            temp=280.0,
        )
        raw = pd.DataFrame(
            [
                holiday_row,
                holiday_row.copy(),
                _raw_row(
                    "2017-01-02 00:00:00",
                    holiday="None",
                    temp=282.0,
                    weather_main="Clouds",
                    weather_description="few clouds",
                ),
                _raw_row(
                    "2017-01-02 01:00:00",
                    temp=100.0,
                    rain_1h=-1.0,
                    snow_1h=201.0,
                    clouds_all=101.0,
                    traffic_volume=125,
                ),
                _raw_row("2017-01-02 02:00:00", traffic_volume=150),
                _raw_row("2017-01-03 00:00:00", traffic_volume=175),
            ]
        )

        cleaned, report = clean_traffic_data(raw)

        self.assertEqual(len(cleaned), 4)
        self.assertTrue(cleaned["date_time"].is_unique)
        self.assertTrue(cleaned["date_time"].is_monotonic_increasing)
        self.assertEqual(report["exact_duplicate_rows_removed"], 1)
        self.assertEqual(report["rows_in_duplicate_timestamps_before_aggregation"], 2)
        self.assertEqual(report["duplicate_timestamps_aggregated"], 1)

        first_hour = cleaned.iloc[0]
        self.assertEqual(first_hour["weather_main"], "Clear|Clouds")
        self.assertEqual(first_hour["weather_description"], "few clouds|sky is clear")
        self.assertAlmostEqual(first_hour["temp_celsius"], 281.0 - 273.15)
        self.assertEqual(first_hour["traffic_volume"], 100)

        january_second = cleaned["date_time"].dt.date == pd.Timestamp("2017-01-02").date()
        self.assertEqual(
            cleaned.loc[january_second, "holiday_name"].tolist(),
            ["New Years Day", "New Years Day", "New Years Day"],
        )
        self.assertEqual(cleaned.iloc[-1]["holiday_name"], "None")

        invalid_hour = cleaned.loc[
            cleaned["date_time"].eq(pd.Timestamp("2017-01-02 01:00:00"))
        ].iloc[0]
        for column in ("temp_celsius", "rain_1h", "snow_1h", "clouds_all"):
            self.assertTrue(np.isnan(invalid_hour[column]), column)

        issue_counts = report["plausibility_issues_replaced_with_missing"]
        self.assertEqual(issue_counts["temperature_outside_180_to_340_kelvin"], 1)
        self.assertEqual(issue_counts["negative_or_over_500_mm_rain"], 1)
        self.assertEqual(issue_counts["negative_or_over_200_mm_snow"], 1)
        self.assertEqual(issue_counts["cloud_cover_outside_0_to_100_percent"], 1)

    def test_rejects_conflicting_targets_for_the_same_hour(self):
        raw = pd.DataFrame(
            [
                _raw_row("2017-01-02 08:00:00", traffic_volume=100),
                _raw_row(
                    "2017-01-02 08:00:00",
                    traffic_volume=200,
                    weather_main="Rain",
                ),
            ]
        )

        with self.assertRaisesRegex(DataValidationError, "conflicting"):
            clean_traffic_data(raw)

    def test_removes_positive_infinite_target_and_reports_it(self):
        raw = pd.DataFrame(
            [
                _raw_row("2017-01-02 08:00:00", traffic_volume=np.inf),
                _raw_row("2017-01-02 09:00:00", traffic_volume=250),
            ]
        )

        cleaned, report = clean_traffic_data(raw)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["traffic_volume"], 250)
        self.assertTrue(np.isfinite(cleaned["traffic_volume"]).all())
        self.assertEqual(report["rows_with_invalid_target_removed"], 1)


if __name__ == "__main__":
    unittest.main()
