"""Train, compare, and save traffic-volume models."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import argparse

from traffic_prediction.data import load_cleaned_data
from traffic_prediction.training import train_and_evaluate
from traffic_prediction.visualization import create_model_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simple traffic-volume models and evaluate the selected model."
    )
    parser.add_argument(
        "--include-neural-network",
        action="store_true",
        help="Also try a small MLP; classical models remain the default.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer tree/neural iterations for a quick run.",
    )
    args = parser.parse_args()

    cleaned = load_cleaned_data()
    result = train_and_evaluate(
        cleaned,
        include_neural_network=args.include_neural_network,
        fast_mode=args.fast,
    )
    figures = create_model_plots()

    print("\nModeling workflow complete")
    print(f"  Selected model: {result['best_model_name']}")
    print(
        "  Final-test MAE: "
        f"{result['test_metrics']['MAE_vehicles_per_hour']:.1f} vehicles/hour"
    )
    print(f"  Report: {result['report_path']}")
    print(f"  Saved model: {result['model_path']}")
    print(f"  Model figures created: {len(figures)}")


if __name__ == "__main__":
    main()
