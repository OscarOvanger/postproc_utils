"""Run NWS Tmax forecast batch fetch for all train cities."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_nws_forecast import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    fetch_nws_tmax_forecast_batch,
    print_coverage_table,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 6, 9)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dates = pd.date_range(START_DATE, END_DATE, freq="D").strftime("%Y-%m-%d")
    forecasts = fetch_nws_tmax_forecast_batch(
        config,
        dates,
        issued_before_hour=22,
        output_path=DEFAULT_OUTPUT_PATH,
        sleep_seconds=1.0,
        checkpoint_every=50,
    )
    print_coverage_table(forecasts, config, trackj_dir=PROJECT_ROOT / "data" / "trackj")


if __name__ == "__main__":
    main()
