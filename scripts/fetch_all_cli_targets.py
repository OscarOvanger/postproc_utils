"""Fetch NWS CLI target Tmax data for all Track-J cities."""

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

from src.trackj.fetch_cli_target import fetch_cli_target  # noqa: E402

DEFAULT_START_DATE = date(2023, 1, 1)
DEFAULT_END_DATE = date.today() - timedelta(days=1)
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
RAW_DIR = TRACKJ_DIR / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"


def load_city_config() -> dict[str, dict]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def print_coverage(output_dir: Path, cities: list[str]) -> None:
    rows = []
    for city in cities:
        path = output_dir / city / "cli_target.parquet"
        if not path.exists():
            rows.append({"city": city, "rows": 0, "complete_tmax": 0})
            continue
        df = pd.read_parquet(path, columns=["date", "tmax_f"])
        rows.append(
            {
                "city": city,
                "rows": int(df.shape[0]),
                "complete_tmax": int(pd.to_numeric(df["tmax_f"], errors="coerce").notna().sum()),
            }
        )
    print("\nCLI target coverage:")
    print(pd.DataFrame(rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=TRACKJ_DIR)
    parser.add_argument("--no-fetch", action="store_true", help="Use cached raw files only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    configs = load_city_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for city, city_config in configs.items():
        print(f"Fetching CLI target {city}...")
        fetch_cli_target(
            city_config,
            start_date,
            end_date,
            args.raw_dir,
            args.output_dir,
            no_fetch=args.no_fetch,
        )

    print_coverage(args.output_dir, list(configs))


if __name__ == "__main__":
    main()
