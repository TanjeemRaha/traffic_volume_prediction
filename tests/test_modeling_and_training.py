"""Tests for simple baselines, physical predictions, and the full training path."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import traffic_prediction.training as training
from traffic_prediction.features import MODEL_FEATURES
from traffic_prediction.modeling import (
    GlobalMedianRegressor,
    HistoricalHourOfWeekRegressor,
    WEATHER_FREE_MODEL_NAMES,
    predict_nonnegative,
)
from traffic_prediction.prediction import predict_one_hour


class _FixedPredictionModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([-25.0, 0.0, 12.5])[: len(features)]


class BaselineAndPredictionTests(unittest.TestCase):
    def test_hour_of_week_baseline_falls_back_to_training_global_median(self):
        training_features = pd.DataFrame({"hour_of_week": ["1", "1", "2"]})
        target = pd.Series([10.0, 30.0, 100.0])
        model = HistoricalHourOfWeekRegressor().fit(training_features, target)

        predictions = model.predict(
            pd.DataFrame({"hour_of_week": ["1", "2", "unseen"]})
        )

        np.testing.assert_allclose(predictions, [20.0, 100.0, 30.0])

    def test_global_median_baseline_uses_only_fitted_target(self):
        model = GlobalMedianRegressor().fit(
            pd.DataFrame({"ignored": [1, 2, 3, 4]}),
            pd.Series([1.0, 9.0, 5.0, 3.0]),
        )

        predictions = model.predict(pd.DataFrame({"anything": [100, 200]}))

        np.testing.assert_allclose(predictions, [4.0, 4.0])

    def test_prediction_helper_clips_physically_impossible_negative_counts(self):
        features = pd.DataFrame({"placeholder": [1, 2, 3]})

        predictions = predict_nonnegative(_FixedPredictionModel(), features)

        np.testing.assert_allclose(predictions, [0.0, 0.0, 12.5])


class SavedArtifactPredictionTests(unittest.TestCase):
    def test_requires_exact_zero_padded_top_of_hour_local_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "baseline.joblib"
            self._save_hour_of_week_artifact(artifact_path, uses_weather=False)

            invalid_values = (
                "2019-01-07 08:30",
                "2019-01-07 08:00:00",
                "2019/01/07 08:00",
                "2019-1-7 8:00",
                "2019-01-07T08:00",
            )
            for invalid_value in invalid_values:
                with self.subTest(date_time=invalid_value):
                    with self.assertRaisesRegex(ValueError, "date_time"):
                        predict_one_hour(
                            date_time=invalid_value,
                            temp_celsius=10.0,
                            model_path=artifact_path,
                            output_path=root / "must_not_be_written.csv",
                        )

            result = predict_one_hour(
                date_time="2019-01-07 08:00",
                temp_celsius=10.0,
                model_path=artifact_path,
                output_path=root / "valid_prediction.csv",
            )
            self.assertEqual(result["date_time"], pd.Timestamp("2019-01-07 08:00"))

    def test_weather_validation_and_reporting_follow_saved_model_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weather_free_path = root / "weather_free.joblib"
            weather_required_path = root / "weather_required.joblib"
            self._save_hour_of_week_artifact(weather_free_path, uses_weather=False)
            self._save_hour_of_week_artifact(weather_required_path, uses_weather=True)

            ignored_weather_result = predict_one_hour(
                date_time="2019-01-07 08:00",
                temp_celsius=np.nan,
                rain_mm=np.inf,
                snow_mm=-99.0,
                cloud_percent=999.0,
                weather_main="not used",
                holiday_name="also not used",
                model_path=weather_free_path,
                output_path=root / "weather_free_prediction.csv",
            )

            self.assertFalse(ignored_weather_result["uses_weather"])
            self.assertEqual(
                ignored_weather_result["inputs_used"],
                "local date/time converted to hour_of_week",
            )
            self.assertIn("temperature, rain, snow", ignored_weather_result["ignored_inputs"])
            self.assertEqual(ignored_weather_result["predicted_traffic_volume"], 100.0)

            with self.assertRaisesRegex(ValueError, "finite numbers"):
                predict_one_hour(
                    date_time="2019-01-07 08:00",
                    temp_celsius=np.nan,
                    model_path=weather_required_path,
                    output_path=root / "weather_required_prediction.csv",
                )

    @staticmethod
    def _save_hour_of_week_artifact(path: Path, uses_weather: bool) -> None:
        model = HistoricalHourOfWeekRegressor().fit(
            pd.DataFrame({"hour_of_week": ["8", "9"]}),
            pd.Series([100.0, 200.0]),
        )
        joblib.dump(
            {
                "model": model,
                "model_name": "Historical hour-of-week baseline",
                "feature_origin": "2016-01-01T00:00:00",
                "target_training_quantiles": {
                    "25_percent": 100.0,
                    "50_percent": 150.0,
                    "75_percent": 200.0,
                },
                "uses_weather": uses_weather,
                "inputs_used": ["local date/time converted to hour_of_week"],
                "training_through": "2018-12-31T23:00:00",
            },
            path,
        )


class FastEndToEndTrainingTests(unittest.TestCase):
    def test_training_rejects_nonfinite_and_negative_targets_before_model_fit(self):
        for invalid_target in (np.nan, np.inf, -1.0):
            with self.subTest(traffic_volume=invalid_target):
                cleaned = self._synthetic_cleaned_data()
                cleaned["traffic_volume"] = cleaned["traffic_volume"].astype(float)
                cleaned.loc[0, "traffic_volume"] = invalid_target

                with patch.object(
                    training, "create_project_directories", return_value=None
                ), self.assertRaisesRegex(ValueError, "finite, nonnegative"):
                    training.train_and_evaluate(cleaned, fast_mode=True)

    def test_fast_training_writes_model_and_chronological_holdout_outputs(self):
        cleaned = self._synthetic_cleaned_data()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            metrics_dir = temporary_root / "metrics"
            predictions_dir = temporary_root / "predictions"
            reports_dir = temporary_root / "reports"
            model_path = temporary_root / "models" / "best_model.joblib"
            for directory in (
                metrics_dir,
                predictions_dir,
                reports_dir,
                model_path.parent,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            with patch.multiple(
                training,
                METRICS_DIR=metrics_dir,
                PREDICTIONS_DIR=predictions_dir,
                REPORTS_DIR=reports_dir,
                BEST_MODEL_PATH=model_path,
            ), patch.object(training, "create_project_directories", return_value=None):
                result = training.train_and_evaluate(cleaned, fast_mode=True)

            self.assertTrue(Path(result["model_path"]).is_file())
            self.assertTrue(Path(result["metrics_path"]).is_file())
            self.assertTrue(Path(result["predictions_path"]).is_file())
            self.assertTrue(Path(result["importance_path"]).is_file())
            self.assertTrue(Path(result["report_path"]).is_file())

            artifact = joblib.load(result["model_path"])
            self.assertEqual(artifact["selected_using"], "lowest validation MAE")
            self.assertIn("validate 2017, test 2018", artifact["split_method"])
            self.assertEqual(pd.Timestamp(artifact["training_through"]).year, 2017)
            self.assertIn("model", artifact)
            self.assertIsInstance(artifact["uses_weather"], bool)
            self.assertIsInstance(artifact["inputs_used"], list)
            expected_uses_weather = result["best_model_name"] not in WEATHER_FREE_MODEL_NAMES
            self.assertEqual(artifact["uses_weather"], expected_uses_weather)
            if expected_uses_weather:
                self.assertEqual(artifact["inputs_used"], MODEL_FEATURES)
            elif result["best_model_name"] == "Global median baseline":
                self.assertIn("constant fitted training median", artifact["inputs_used"][0])
            else:
                self.assertEqual(
                    artifact["inputs_used"],
                    ["local date/time converted to hour_of_week"],
                )

            experiment_metadata = json.loads(
                (reports_dir / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                experiment_metadata["selection_reason"], artifact["selected_using"]
            )
            self.assertEqual(
                experiment_metadata["selected_model"], result["best_model_name"]
            )

            comparison = pd.read_csv(result["metrics_path"])
            self.assertEqual((comparison["partition"] == "final_test").sum(), 1)
            self.assertGreaterEqual((comparison["partition"] == "validation").sum(), 5)
            final_test_row = comparison.loc[comparison["partition"] == "final_test"]
            self.assertEqual(final_test_row.iloc[0]["model"], result["best_model_name"])

            predictions = pd.read_csv(result["predictions_path"])
            self.assertEqual(len(predictions), 48)
            self.assertTrue((predictions["predicted_traffic_volume"] >= 0).all())
            self.assertTrue(
                pd.to_datetime(predictions["date_time"]).dt.year.eq(2018).all()
            )

    @staticmethod
    def _synthetic_cleaned_data() -> pd.DataFrame:
        yearly_frames: list[pd.DataFrame] = []
        for year in (2016, 2017, 2018):
            timestamps = pd.date_range(f"{year}-01-01", periods=48, freq="h")
            hour = timestamps.hour.to_numpy()
            yearly_frames.append(
                pd.DataFrame(
                    {
                        "date_time": timestamps,
                        "holiday_name": ["None"] * len(timestamps),
                        "temp_celsius": 2.0 + hour * 0.5,
                        "rain_1h": np.where(hour % 7 == 0, 1.0, 0.0),
                        "snow_1h": np.zeros(len(timestamps)),
                        "clouds_all": (hour * 4) % 101,
                        "weather_main": np.where(hour % 7 == 0, "Rain", "Clear"),
                        "weather_description": ["synthetic"] * len(timestamps),
                        "traffic_volume": (
                            700
                            + 80 * hour
                            + 250 * ((hour >= 7) & (hour <= 9))
                            + (year - 2016) * 25
                        ),
                    }
                )
            )
        return pd.concat(yearly_frames, ignore_index=True)


if __name__ == "__main__":
    unittest.main()
