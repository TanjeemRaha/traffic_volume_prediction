"""Estimate traffic volume for a user-defined scenario."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import argparse

from traffic_prediction.prediction import predict_one_hour


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate I-94 westbound traffic for one hour."
    )
    parser.add_argument("--date-time", default="2019-01-07 08:00")
    parser.add_argument("--temp-celsius", type=float, default=-5.0)
    parser.add_argument("--rain-mm", type=float, default=0.0)
    parser.add_argument("--snow-mm", type=float, default=0.0)
    parser.add_argument("--cloud-percent", type=float, default=75.0)
    parser.add_argument(
        "--weather-main",
        default="Clouds",
        help="Examples: Clear, Clouds, Rain, Snow, Mist, Thunderstorm",
    )
    parser.add_argument(
        "--holiday-name", default="None", help="Use None for an ordinary date."
    )
    args = parser.parse_args()

    result = predict_one_hour(
        date_time=args.date_time,
        temp_celsius=args.temp_celsius,
        rain_mm=args.rain_mm,
        snow_mm=args.snow_mm,
        cloud_percent=args.cloud_percent,
        weather_main=args.weather_main,
        holiday_name=args.holiday_name,
    )
    print("Traffic estimate complete")
    print(f"  Date/time: {result['date_time']}")
    print(
        "  Predicted traffic: "
        f"{result['predicted_traffic_volume']:,.1f} vehicles/hour"
    )
    print(f"  Relative level: {result['historical_volume_band']}")
    print(f"  Model: {result['model_name']}")
    print(f"  Inputs used by this model: {result['inputs_used']}")
    print(f"  Saved to: {result['saved_to']}")
    if result["uses_weather"]:
        print("  Reminder: use observed weather or a weather forecast for that hour.")
    else:
        print(f"  Ignored scenario inputs: {result['ignored_inputs']}")
    print(f"  Note: {result['extrapolation_note']}")


if __name__ == "__main__":
    main()
