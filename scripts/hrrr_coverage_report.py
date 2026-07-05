#!/usr/bin/env python3
"""Report HRRR v2 cache coverage per city (2021-01-01 through 2026-06-25)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fetch_hrrr_all_cities import HRRR_DATA_DIR, HRRR_STATIONS, PARQUET_PATH  # noqa: E402

DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date(2026, 6, 25)


def cached_dates_for_city(city: str, start: date, end: date) -> set[str]:
    city_dir = HRRR_DATA_DIR / city
    if not city_dir.exists():
        return set()
    dates: set[str] = set()
    for path in sorted(city_dir.glob(f"hrrr_{city}_*.csv")):
        try:
            df = pd.read_csv(path, usecols=["date"])
        except (OSError, ValueError, KeyError):
            continue
        for d in df["date"].astype(str):
            try:
                dt = date.fromisoformat(d)
            except ValueError:
                continue
            if start <= dt <= end:
                dates.add(d)
    return dates


def count_pending(city: str, start: date, end: date) -> int:
    expected = pd.date_range(start, end, freq="D").date
    cached = cached_dates_for_city(city, start, end)
    return sum(1 for d in expected if str(d) not in cached)


def main() -> None:
    parser = argparse.ArgumentParser(description="HRRR v2 coverage report")
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=DEFAULT_END.isoformat())
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    total_days = (end - start).days + 1

    print(f"HRRR v2 coverage: {start} .. {end} ({total_days} days/city)")
    print(f"Data dir: {HRRR_DATA_DIR}")
    if PARQUET_PATH.exists():
        size_kb = PARQUET_PATH.stat().st_size // 1024
        print(f"Combined parquet: {PARQUET_PATH.name} ({size_kb} KB)")
    print()
    header = f"{'City':<16} {'Cached':>8} {'Pending':>8} {'First':>12} {'Last':>12} {'Months':>7}"
    print(header)
    print("-" * len(header))

    total_pending = 0
    for city in sorted(HRRR_STATIONS):
        cached = cached_dates_for_city(city, start, end)
        pending = total_days - len(cached)
        total_pending += pending
        first = min(cached) if cached else "—"
        last = max(cached) if cached else "—"
        n_months = len(list((HRRR_DATA_DIR / city).glob("hrrr_*.csv"))) if (HRRR_DATA_DIR / city).exists() else 0
        print(f"{city:<16} {len(cached):8d} {pending:8d} {str(first):>12} {str(last):>12} {n_months:7d}")

    print("-" * len(header))
    print(f"{'TOTAL':<16} {'':>8} {total_pending:8d}")
    if total_pending > 0:
        print(f"\nResume: ./scripts/run_hrrr_v2_fetch_foreground.sh")


if __name__ == "__main__":
    main()
