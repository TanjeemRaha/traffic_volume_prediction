"""Generate exploratory traffic tables and figures."""

# Code by Tanjeem Farhana Raha

from __future__ import annotations

import argparse

from traffic_prediction.data import load_cleaned_data
from traffic_prediction.visualization import create_eda_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore traffic patterns without opening held-out targets."
    )
    parser.add_argument(
        "--include-held-out-periods",
        action="store_true",
        help=(
            "Use the full record for retrospective description. Only do this "
            "after completing and freezing the model-evaluation workflow."
        ),
    )
    args = parser.parse_args()
    cleaned = load_cleaned_data()
    generated = create_eda_outputs(
        cleaned, include_held_out_periods=args.include_held_out_periods
    )
    print("Exploratory analysis complete. Key files created:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
