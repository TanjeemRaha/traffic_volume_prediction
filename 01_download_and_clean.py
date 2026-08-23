"""Download and clean the UCI traffic dataset."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import argparse

from traffic_prediction.data import prepare_data
from traffic_prediction.paths import CLEANED_DATA_PATH, QUALITY_REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download UCI I-94 data and create one clean row per observed hour."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download a fresh copy even when raw data already exists.",
    )
    args = parser.parse_args()

    cleaned, report = prepare_data(force_download=args.force_download)
    print("\nCleaning complete")
    print(f"  Clean hourly observations: {len(cleaned):,}")
    print(f"  Observation period: {cleaned['date_time'].min()} to {cleaned['date_time'].max()}")
    print(
        "  Duplicate timestamps combined: "
        f"{report['duplicate_timestamps_aggregated']:,}"
    )
    print(
        "  Unobserved hours left as gaps: "
        f"{report['unobserved_hours_between_start_and_end']:,}"
    )
    print(f"  Cleaned data: {CLEANED_DATA_PATH}")
    print(f"  Quality audit: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
