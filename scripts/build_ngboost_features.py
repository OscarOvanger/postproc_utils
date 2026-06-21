"""Build NGBoost feature tables for all train cities and lead times."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trackj.build_ngboost_features import (  # noqa: E402
    LEAD_FEATURE_COLUMNS,
    LEAD_TIMES,
    NGBOOST_DIR,
    build_feature_table,
    run_verification,
)
from src.trackj.fetch_gfs_herbie import clear_herbie_cache  # noqa: E402

CITIES = [
    "austin",
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "san_francisco",
]
START_DATE = date(2021, 1, 1)
MIN_RESUME_ROWS = 1000


def _latest_cli_end_date(city: str) -> date:
    cli_path = PROJECT_ROOT / "data" / "trackj" / city / "cli_target.parquet"
    if not cli_path.exists():
        return date.today()
    cli = pd.read_parquet(cli_path)
    cli["date"] = pd.to_datetime(cli["date"], errors="coerce")
    valid = cli[cli["tmax_f"].notna()]
    if valid.empty:
        return date.today()
    return valid["date"].max().date()


def _print_column_coverage(city: str, lead_time: str, features: pd.DataFrame) -> None:
    feature_cols = LEAD_FEATURE_COLUMNS[lead_time]
    counts = {col: int(features[col].notna().sum()) for col in feature_cols if col in features.columns}
    print(f"  Non-null counts [{city}/{lead_time}]: {counts}")


def _should_skip_build(out_path: Path, resume: bool) -> tuple[bool, int]:
    if not resume or not out_path.exists():
        return False, 0
    try:
        n_rows = len(pd.read_parquet(out_path))
    except Exception:
        return False, 0
    if n_rows > MIN_RESUME_ROWS:
        return True, n_rows
    return False, n_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NGBoost feature tables.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all feature tables even if parquet already exists (>1000 rows).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = not args.force

    coverage_rows: list[dict] = []
    row_counts: dict[str, dict[str, int]] = {city: {} for city in CITIES}

    for city in CITIES:
        end_date = _latest_cli_end_date(city)
        print(f"\n=== {city} (through {end_date}) ===")
        date_range = ""
        for lead_time in LEAD_TIMES:
            out_dir = NGBOOST_DIR / "features" / city
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{lead_time}.parquet"

            skip, n_existing = _should_skip_build(out_path, resume)
            if skip:
                print(f"Skipping {city} {lead_time}: already built ({n_existing} rows)")
                row_counts[city][lead_time] = n_existing
                if n_existing > 0:
                    existing = pd.read_parquet(out_path)
                    if not existing.empty and "date" in existing.columns:
                        date_range = f"{existing['date'].min()} to {existing['date'].max()}"
                continue

            features = build_feature_table(
                city,
                lead_time,
                START_DATE,
                end_date,
                no_fetch=False,
                fetch_gfs=True,
            )
            features.to_parquet(out_path, index=False)
            clear_herbie_cache()
            row_counts[city][lead_time] = len(features)
            if not features.empty:
                date_range = f"{features['date'].min()} to {features['date'].max()}"
            _print_column_coverage(city, lead_time, features)

        coverage_rows.append(
            {
                "city": city,
                "t1 rows": row_counts[city].get("t1", 0),
                "t2 rows": row_counts[city].get("t2", 0),
                "t3 rows": row_counts[city].get("t3", 0),
                "date range": date_range,
            }
        )

    summary = pd.DataFrame(coverage_rows)
    print("\n=== COVERAGE SUMMARY ===")
    print(summary.to_string(index=False))

    run_verification(CITIES)


if __name__ == "__main__":
    main()
