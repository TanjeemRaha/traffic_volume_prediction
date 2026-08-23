"""Tests for target-free feature engineering and chronological partitions."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from traffic_prediction.data import DataValidationError
from traffic_prediction.features import (
    MODEL_FEATURES,
    create_features,
    make_chronological_split,
)


class FeatureConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cleaned = pd.DataFrame(
            {
                "date_time": pd.to_datetime(
                    ["2020-01-06 08:00:00", "2020-07-07 19:00:00"]
                ),
                "holiday_name": ["None", "Independence Day"],
                "temp_celsius": [-5.0, 25.0],
                "rain_1h": [0.0, 3.0],
                "snow_1h": [2.0, 0.0],
                "clouds_all": [10.0, 90.0],
                "weather_main": ["Rain|Clouds", "Alien weather"],
                "traffic_volume": [1_000, 2_000],
            }
        )

    def test_constructs_expected_calendar_weather_and_transformed_features(self):
        features = create_features(self.cleaned, origin="2020-01-01 00:00:00")

        self.assertEqual(features.columns.tolist(), MODEL_FEATURES)
        self.assertEqual(features.loc[0, "hour_of_week"], "8")
        self.assertEqual(features.loc[0, "month"], "1")
        self.assertAlmostEqual(features.loc[0, "days_since_start"], 5 + 8 / 24)
        self.assertEqual(features.loc[0, "weather_rain"], 1)
        self.assertEqual(features.loc[0, "weather_clouds"], 1)
        self.assertEqual(features.loc[0, "weather_other"], 0)
        self.assertEqual(features.loc[1, "weather_other"], 1)
        self.assertAlmostEqual(features.loc[1, "rain_1h_log"], np.log1p(3.0))
        self.assertAlmostEqual(features.loc[0, "snow_1h_log"], np.log1p(2.0))

    def test_target_values_cannot_change_features(self):
        changed_target = self.cleaned.copy()
        changed_target["traffic_volume"] = [999_999, -50]

        original_features = create_features(self.cleaned, origin="2020-01-01")
        changed_features = create_features(changed_target, origin="2020-01-01")

        assert_frame_equal(original_features, changed_features)
        self.assertNotIn("traffic_volume", original_features.columns)


class ChronologicalSplitTests(unittest.TestCase):
    def test_uses_last_two_well_populated_years_for_validation_and_test(self):
        timestamps = pd.Series(
            np.concatenate(
                (
                    pd.date_range("2016-01-01", periods=24, freq="h").to_numpy(),
                    pd.date_range("2017-01-01", periods=24, freq="h").to_numpy(),
                    pd.date_range("2018-01-01", periods=24, freq="h").to_numpy(),
                )
            )
        )

        split = make_chronological_split(timestamps)

        self.assertIn("validate 2017, test 2018", split.method)
        self.assertEqual(len(split.train_indices), 24)
        self.assertEqual(len(split.validation_indices), 24)
        self.assertEqual(len(split.test_indices), 24)
        self.assertTrue((timestamps.iloc[split.train_indices].dt.year == 2016).all())
        self.assertTrue((timestamps.iloc[split.validation_indices].dt.year == 2017).all())
        self.assertTrue((timestamps.iloc[split.test_indices].dt.year == 2018).all())
        self._assert_strict_chronology(timestamps, split)

    def test_fallback_split_preserves_order_and_keeps_every_row_once(self):
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=40, freq="h"))

        split = make_chronological_split(timestamps)

        self.assertIn("70% train", split.method)
        self.assertEqual(
            (len(split.train_indices), len(split.validation_indices), len(split.test_indices)),
            (28, 6, 6),
        )
        combined = np.concatenate(
            (split.train_indices, split.validation_indices, split.test_indices)
        )
        np.testing.assert_array_equal(combined, np.arange(len(timestamps)))
        self._assert_strict_chronology(timestamps, split)

    def test_rejects_unsorted_timestamps_instead_of_silently_leaking(self):
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=20, freq="h"))
        timestamps.iloc[[5, 6]] = timestamps.iloc[[6, 5]].to_numpy()

        with self.assertRaisesRegex(DataValidationError, "sorted"):
            make_chronological_split(timestamps)

    def test_rejects_duplicate_timestamps_before_partitioning(self):
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=20, freq="h"))
        timestamps.iloc[10] = timestamps.iloc[9]
        self.assertTrue(timestamps.is_monotonic_increasing)

        with self.assertRaisesRegex(DataValidationError, "one row per timestamp"):
            make_chronological_split(timestamps)

    def _assert_strict_chronology(self, timestamps, split) -> None:
        train_times = timestamps.iloc[split.train_indices]
        validation_times = timestamps.iloc[split.validation_indices]
        test_times = timestamps.iloc[split.test_indices]
        self.assertLess(train_times.max(), validation_times.min())
        self.assertLess(validation_times.max(), test_times.min())
        self.assertTrue(
            set(split.train_indices).isdisjoint(split.validation_indices)
        )
        self.assertTrue(set(split.train_indices).isdisjoint(split.test_indices))
        self.assertTrue(
            set(split.validation_indices).isdisjoint(split.test_indices)
        )


if __name__ == "__main__":
    unittest.main()
