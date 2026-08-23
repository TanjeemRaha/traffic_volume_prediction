"""Model definitions from statistical baselines to nonlinear estimators."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

from traffic_prediction.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


RANDOM_SEED = 42
WEATHER_FREE_MODEL_NAMES = {
    "Global median baseline",
    "Historical hour-of-week baseline",
}


class GlobalMedianRegressor(RegressorMixin, BaseEstimator):
    """Predict the training-period median for every hour."""

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        del X
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "median_"):
            raise RuntimeError("The global-median baseline has not been fitted.")
        return np.full(len(X), self.median_, dtype=float)


class HistoricalHourOfWeekRegressor(RegressorMixin, BaseEstimator):
    """Predict the training median for each of the 168 hours in a week."""

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if "hour_of_week" not in X.columns:
            raise ValueError("Historical baseline requires the hour_of_week feature.")
        training = pd.DataFrame(
            {
                "hour_of_week": X["hour_of_week"].astype(str).to_numpy(),
                "target": np.asarray(y, dtype=float),
            }
        )
        self.medians_ = training.groupby("hour_of_week")["target"].median().to_dict()
        self.global_median_ = float(training["target"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "medians_"):
            raise RuntimeError("The hour-of-week baseline has not been fitted.")
        predictions = X["hour_of_week"].astype(str).map(self.medians_)
        return predictions.fillna(self.global_median_).to_numpy(dtype=float)


def _make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("fill_missing", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=None,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def build_models(
    include_neural_network: bool = False, fast_mode: bool = False
) -> OrderedDict[str, BaseEstimator]:
    """Build the candidate estimators used in model validation."""

    models: OrderedDict[str, BaseEstimator] = OrderedDict()
    models["Global median baseline"] = GlobalMedianRegressor()
    models["Historical hour-of-week baseline"] = HistoricalHourOfWeekRegressor()
    models["Linear regression"] = Pipeline(
        [
            ("prepare_features", _make_preprocessor(scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]
    )
    linear_svm = Pipeline(
        [
            ("prepare_features", _make_preprocessor(scale_numeric=True)),
            (
                "regressor",
                LinearSVR(
                    C=1.0,
                    epsilon=0.0,
                    dual="auto",
                    loss="squared_epsilon_insensitive",
                    max_iter=50_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    models["Linear support vector regression"] = TransformedTargetRegressor(
        regressor=linear_svm,
        transformer=StandardScaler(),
    )
    models["Random forest"] = Pipeline(
        [
            ("prepare_features", _make_preprocessor(scale_numeric=False)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=60 if fast_mode else 250,
                    min_samples_leaf=2,
                    max_features=0.7,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    if include_neural_network:
        neural_pipeline = Pipeline(
            [
                ("prepare_features", _make_preprocessor(scale_numeric=True)),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=80 if fast_mode else 250,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        models["Small neural network (MLP)"] = TransformedTargetRegressor(
            regressor=neural_pipeline,
            transformer=StandardScaler(),
        )

    return models


def predict_nonnegative(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Predict traffic counts and enforce the physical lower bound of zero."""

    raw_predictions = np.asarray(model.predict(features), dtype=float)
    return np.clip(raw_predictions, a_min=0.0, a_max=None)
