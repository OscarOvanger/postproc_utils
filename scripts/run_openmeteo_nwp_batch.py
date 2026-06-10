"""Run open-meteo NWP Tmax forecast batch fetch for all train cities."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_openmeteo_nwp import (  # noqa: E402
    fetch_openmeteo_tmax_batch,
    print_openmeteo_coverage_table,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "trackb" / "openmeteo_nwp_raw.parquet"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 6, 9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch open-meteo NWP Tmax forecasts for all train cities.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch all city-dates ignoring checkpoint resume.",
    )
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    forecasts = fetch_openmeteo_tmax_batch(
        config,
        START_DATE,
        END_DATE,
        output_path=OUTPUT_PATH,
        sleep_seconds=0.5,
        force_refresh=args.force_refresh,
    )
    summary = print_openmeteo_coverage_table(forecasts, config, trackj_dir=PROJECT_ROOT / "data" / "trackj")
    bad_mae = summary[
        summary["MAE vs actual Tmax"].notna() & (summary["MAE vs actual Tmax"] > 5.0)
    ]
    if not bad_mae.empty:
        print("\nHALTING: One or more city/model pairs have MAE > 5°F. Investigate before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
