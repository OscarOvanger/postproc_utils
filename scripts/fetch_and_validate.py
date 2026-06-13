"""Fetch and validate data for a date range."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import build_feature_vector, run_leakage_audit  # noqa: E402
from src.data_store import TRAIN_CITIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        type=str,
        default="data/fresh_validation/features_fresh.parquet",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    dates = pd.date_range(start, end, freq="D")

    print(f"Fetching data for {len(dates)} dates: {args.start} to {args.end}")
    print(f"Cities: {len(TRAIN_CITIES)}")

    results = []
    leakage_failures = 0

    for event_date in dates:
        date_str = str(event_date.date())
        for city in TRAIN_CITIES:
            print(f"\n{city} / {date_str}:")

            features = build_feature_vector(city, date_str)
            if features is None:
                print("  SKIP: insufficient data")
                continue

            clean = run_leakage_audit(city, date_str, features)
            if not clean:
                leakage_failures += 1
                print("  LEAKAGE DETECTED — excluding from validation")
                continue

            features["city"] = city
            features["date"] = date_str
            results.append(features)

    if results:
        df = pd.DataFrame(results)
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        attempted = len(dates) * len(TRAIN_CITIES)
        print("\n=== FETCH SUMMARY ===")
        print(f"Total city-dates attempted: {attempted}")
        print(f"Successful: {len(results)}")
        print(f"Leakage failures: {leakage_failures}")
        print(f"Coverage: {len(results) / attempted * 100:.1f}%")
        print(f"Saved to: {output_path}")

        city_counts = df.groupby("city").size()
        for city in TRAIN_CITIES:
            n = int(city_counts.get(city, 0))
            print(f"  {city}: {n}/{len(dates)} dates ({n / len(dates) * 100:.0f}%)")
    else:
        print("\nNo data fetched. Check data sources.")

    if leakage_failures > 0:
        print(f"\nWARNING: {leakage_failures} city-dates excluded due to leakage.")


if __name__ == "__main__":
    main()
