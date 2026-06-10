"""Fetch GFS afternoon features for all train cities (resume via per-day CSV cache)."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_gfs_herbie import fetch_gfs_for_date  # noqa: E402
from trackj.fetch_nws_forecast import TRAIN_CITIES  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 6, 9)
MAX_WORKERS = 4
PROGRESS_EVERY = 100


def gfs_raw_dir(city_config: dict) -> Path:
    station = str(city_config["nws_station"]).lower()
    if station == "kaus":
        return PROJECT_ROOT / "data" / "raw" / "gfs_kaus"
    return PROJECT_ROOT / "data" / "raw" / f"gfs_{station}"


def _count_ok_files(raw_dir: Path, station: str) -> tuple[int, int]:
    if not raw_dir.exists():
        return 0, 0
    files = list(raw_dir.glob(f"{station.lower()}_gfs_*.csv"))
    ok = 0
    for path in files:
        row = pd.read_csv(path).iloc[0]
        if row.get("gfs_parse_status") == "ok":
            ok += 1
    return ok, len(files)


def fetch_city(city: str, city_config: dict, dates: list[date]) -> None:
    raw_dir = gfs_raw_dir(city_config)
    station = str(city_config["nws_station"]).lower()
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok_before, total_before = _count_ok_files(raw_dir, station)
    print(f"Fetching {city}... {total_before} files cached ({ok_before} ok).", flush=True)

    def _fetch_one(target_date: date) -> tuple[date, str]:
        _, audit = fetch_gfs_for_date(target_date, raw_dir=raw_dir, city_config=city_config)
        return target_date, str(audit.get("gfs_parse_status", ""))

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, d): d for d in dates}
        for future in as_completed(futures):
            completed += 1
            if completed % PROGRESS_EVERY == 0 or completed == len(dates):
                ok_now, total_now = _count_ok_files(raw_dir, station)
                print(
                    f"  {city}: {completed}/{len(dates)} dates processed, "
                    f"{total_now} files cached ({ok_now} ok).",
                    flush=True,
                )
            future.result()

    ok_after, total_after = _count_ok_files(raw_dir, station)
    print(f"GFS {city} done: {total_after} files cached ({ok_after} ok).", flush=True)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dates = pd.date_range(START_DATE, END_DATE, freq="D").date.tolist()
    for city in TRAIN_CITIES:
        if city == "austin":
            city_config = config[city]
            raw_dir = gfs_raw_dir(city_config)
            station = str(city_config["nws_station"]).lower()
            ok, total = _count_ok_files(raw_dir, station)
            print(f"GFS {city}: {total} cached files ({ok} ok) in {raw_dir}")
            if total > 1000:
                print("  Skipping fetch (cache sufficient)")
            continue
        fetch_city(city, config[city], dates)


if __name__ == "__main__":
    main()
