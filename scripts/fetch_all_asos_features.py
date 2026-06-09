"""Fetch/build morning ASOS features for all Track-J cities."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trackj.build_asos_features import build_asos_features  # noqa: E402

DEFAULT_START_DATE = date(2023, 1, 1)
DEFAULT_END_DATE = date.today() - timedelta(days=1)
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
RAW_DIR = TRACKJ_DIR / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"


def load_city_config() -> dict[str, dict]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_cli_target(output_dir: Path, city: str) -> pd.DataFrame | None:
    path = output_dir / city / "cli_target.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def print_coverage(output_dir: Path, cities: list[str]) -> None:
    rows = []
    feature_cols = [
        "temp_10am",
        "temp_mean_00_10",
        "temp_max_so_far_00_10",
        "dewpoint_10am",
        "rh_mean_00_10",
        "pressure_10am",
        "wind_u_mean_00_10",
        "wind_v_mean_00_10",
        "cloud_cover_mean_00_10",
        "temp_lag1",
    ]
    for city in cities:
        path = output_dir / city / "asos_features.parquet"
        if not path.exists():
            rows.append({"city": city, "rows": 0, "complete_rows": 0})
            continue
        df = pd.read_parquet(path)
        present = [column for column in feature_cols if column in df.columns]
        rows.append(
            {
                "city": city,
                "rows": int(df.shape[0]),
                "complete_rows": int(df[present].notna().all(axis=1).sum()) if present else 0,
            }
        )
    print("\nASOS feature coverage:")
    print(pd.DataFrame(rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=TRACKJ_DIR)
    parser.add_argument("--no-fetch", action="store_true", help="Use cached raw files only.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--cities",
        nargs="*",
        help="Optional city keys to process. Defaults to all configured cities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    configs = load_city_config()
    selected_cities = args.cities or list(configs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for city in selected_cities:
        city_config = configs[city]
        print(f"Fetching ASOS features {city}...")
        build_asos_features(
            city_config,
            start_date,
            end_date,
            args.raw_dir,
            args.output_dir,
            no_fetch=args.no_fetch,
            target_df=load_cli_target(args.output_dir, city),
            sleep_seconds=args.sleep_seconds,
        )

    print_coverage(args.output_dir, list(configs))


if __name__ == "__main__":
    main()
