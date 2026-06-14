"""Build fresh validation market parquet from lean snapshots."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_store import TRAIN_CITIES  # noqa: E402

LEAN_PATH = PROJECT_ROOT / "data" / "market_lean" / "lean_snapshots.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fresh_validation" / "market_fresh.parquet"
FRESH_CUTOFF = date(2026, 6, 2)


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(LEAN_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--after", type=str, default=FRESH_CUTOFF.isoformat())
    args = parser.parse_args()

    lean_path = Path(args.input)
    if not lean_path.exists():
        raise FileNotFoundError(
            f"Missing {lean_path}. Run scripts/extract_lean_snapshots.py first."
        )

    df = pd.read_parquet(lean_path)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["city"] = df["city"].map(_city_key)
    if "source_city_folder" not in df.columns:
        df["source_city_folder"] = df["city"]

    cutoff = date.fromisoformat(args.after)
    fresh = df.loc[df["event_date"] > cutoff].copy()
    fresh = fresh.loc[fresh["city"].isin(TRAIN_CITIES)].copy()

    if fresh.empty:
        print(f"No rows with event_date > {cutoff.isoformat()}")
        return

    fresh["partition"] = "fresh_validation"
    fresh["event_date"] = pd.to_datetime(fresh["event_date"])
    fresh["snapshot_time_local"] = pd.to_datetime(fresh["snapshot_time_local"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(output_path, index=False)

    n_city_days = int(fresh.groupby(["source_city_folder", "event_date"]).ngroups)
    print(f"Saved {len(fresh)} rows to {output_path}")
    print(
        f"  Date range: {fresh['event_date'].min().date()} to "
        f"{fresh['event_date'].max().date()}"
    )
    print(f"  City-days: {n_city_days}")
    print(f"  Cities: {sorted(fresh['city'].unique())}")


if __name__ == "__main__":
    main()
