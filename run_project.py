"""Run the complete traffic-volume workflow."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import argparse

from traffic_prediction.data import prepare_data
from traffic_prediction.prediction import predict_one_hour
from traffic_prediction.training import train_and_evaluate
from traffic_prediction.visualization import create_eda_outputs, create_model_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete I-94 ML project.")
    parser.add_argument(
        "--include-neural-network",
        action="store_true",
        help="Also compare an optional small neural network.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer model iterations for a quick run.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the UCI raw file again.",
    )
    args = parser.parse_args()

    print("\n=== Step 1 of 4: download and clean ===")
    cleaned, _ = prepare_data(force_download=args.force_download)

    print("\n=== Step 2 of 4: explore traffic patterns ===")
    create_eda_outputs(cleaned)

    print("\n=== Step 3 of 4: train and evaluate models ===")
    result = train_and_evaluate(
        cleaned,
        include_neural_network=args.include_neural_network,
        fast_mode=args.fast,
    )
    create_model_plots()

    print("\n=== Step 4 of 4: example future-hour estimate ===")
    prediction = predict_one_hour(
        date_time="2019-01-07 08:00",
        temp_celsius=-5.0,
        rain_mm=0.0,
        snow_mm=0.0,
        cloud_percent=75.0,
        weather_main="Clouds",
        holiday_name="None",
    )

    print("\n=== Project complete ===")
    print(f"Selected model: {result['best_model_name']}")
    print(
        "Final-test MAE: "
        f"{result['test_metrics']['MAE_vehicles_per_hour']:.1f} vehicles/hour"
    )
    print(
        "Example estimate: "
        f"{prediction['predicted_traffic_volume']:,.1f} vehicles/hour"
    )
    print(f"Example inputs actually used: {prediction['inputs_used']}")
    print("Open outputs/reports/model_results.md for the model results.")


if __name__ == "__main__":
    main()
