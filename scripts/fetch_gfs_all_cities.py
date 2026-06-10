"""Fetch GFS afternoon features for all train cities (resume via per-day CSV cache)."""

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

from trackj.fetch_gfs_herbie import build_gfs_features  # noqa: E402
from trackj.fetch_nws_forecast import TRAIN_CITIES  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 6, 9)


def gfs_raw_dir(city_config: dict) -> Path:
    station = str(city_config["nws_station"]).lower()
    if station == "kaus":
        return PROJECT_ROOT / "data" / "raw" / "gfs_kaus"
    return PROJECT_ROOT / "data" / "raw" / f"gfs_{station}"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dates = pd.date_range(START_DATE, END_DATE, freq="D").strftime("%Y-%m-%d")
    for city in TRAIN_CITIES:
        city_config = config[city]
        raw_dir = gfs_raw_dir(city_config)
        cached = len(list(raw_dir.glob(f"{city_config['nws_station'].lower()}_gfs_*.csv"))) if raw_dir.exists() else 0
        print(f"GFS {city}: {cached} cached files in {raw_dir}")
        if cached > 1000:
            print(f"  Skipping fetch (cache sufficient)")
            continue
        print(f"  Fetching GFS for {city}...")
        build_gfs_features(dates, raw_dir=raw_dir, fetch=True, city_config=city_config)


if __name__ == "__main__":
    main()
