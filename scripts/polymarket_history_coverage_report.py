#!/usr/bin/env python3
"""Report Polymarket order-book snapshot coverage for backtest precondition checks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time as dt_time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest.common import (  # noqa: E402
    ENTRY_WINDOW_END,
    ENTRY_WINDOW_START,
    POLY_CITIES,
    SNAPSHOTS_DIR,
    city_timezone,
    has_entry_window_snapshot,
    load_day_snapshot,
)

DEFAULT_FLOOR_DATE = "2026-02-03"
MIN_ENTRY_WINDOW_TOTAL = 60
MIN_ENTRY_WINDOW_CITIES = 6
MAX_ENTRY_WINDOW_CITY_SHARE = 0.50


def scan_city_coverage(city: str, floor_date: str) -> dict[str, object]:
    city_dir = SNAPSHOTS_DIR / city
    if not city_dir.exists():
        return {
            "city": city,
            "n_dates": 0,
            "n_entry_window": 0,
            "earliest_date": None,
            "latest_date": None,
            "entry_window_dates": [],
        }

    floor = pd.Timestamp(floor_date)
    dates = sorted(path.stem for path in city_dir.glob("*.parquet"))
    entry_window_dates: list[str] = []

    for date_str in dates:
        if pd.Timestamp(date_str) < floor:
            continue
        frame = load_day_snapshot(city, date_str)
        if frame is None or frame.empty:
            continue
        if has_entry_window_snapshot(frame, city, date_str):
            entry_window_dates.append(date_str)

    eligible = [d for d in dates if pd.Timestamp(d) >= floor]
    return {
        "city": city,
        "n_dates": len(eligible),
        "n_entry_window": len(entry_window_dates),
        "earliest_date": min(eligible) if eligible else None,
        "latest_date": max(eligible) if eligible else None,
        "entry_window_dates": entry_window_dates,
    }


def evaluate_coverage_gate(df: pd.DataFrame) -> tuple[bool, str]:
    """Return whether entry-window coverage meets backtest preconditions."""
    total_entry = int(df["n_entry_window"].sum())
    cities_with_data = int((df["n_entry_window"] > 0).sum())
    if total_entry == 0:
        return False, "no entry-window coverage"

    max_city = int(df["n_entry_window"].max())
    max_share = max_city / total_entry
    failures: list[str] = []
    if total_entry < MIN_ENTRY_WINDOW_TOTAL:
        failures.append(f"need ≥{MIN_ENTRY_WINDOW_TOTAL} entry-window city-dates (have {total_entry})")
    if cities_with_data < MIN_ENTRY_WINDOW_CITIES:
        failures.append(
            f"need ≥{MIN_ENTRY_WINDOW_CITIES} cities represented (have {cities_with_data})"
        )
    if max_share > MAX_ENTRY_WINDOW_CITY_SHARE:
        top_city = str(df.loc[df["n_entry_window"].idxmax(), "city"])
        failures.append(
            f"no city may exceed {MAX_ENTRY_WINDOW_CITY_SHARE:.0%} of total "
            f"({top_city} has {max_city}/{total_entry} = {max_share:.1%})"
        )
    if failures:
        return False, "; ".join(failures)
    return True, (
        f"{total_entry} entry-window city-dates across {cities_with_data} cities "
        f"(max city share {max_share:.1%})"
    )


def print_coverage_failure_details(df: pd.DataFrame) -> None:
    print("\nPer-city entry-window eligible dates:")
    header = f"{'City':<16} {'EntryWin':>9}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        print(f"{row['city']:<16} {int(row['n_entry_window']):>9d}")
    total_entry = int(df["n_entry_window"].sum())
    print("-" * len(header))
    print(f"{'TOTAL':<16} {total_entry:>9d}")


def run_report(floor_date: str = DEFAULT_FLOOR_DATE) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, object]] = []
    for city in POLY_CITIES:
        rows.append(scan_city_coverage(city, floor_date))
    df = pd.DataFrame(rows)
    passed, _detail = evaluate_coverage_gate(df)
    return df, passed


def print_report(df: pd.DataFrame, floor_date: str) -> None:
    print(f"\n=== Polymarket history coverage (entry window {ENTRY_WINDOW_START}-{ENTRY_WINDOW_END} local) ===")
    print(f"Floor date: {floor_date}")
    print(f"Snapshots dir: {SNAPSHOTS_DIR}")
    print()
    header = f"{'City':<16} {'Dates':>7} {'EntryWin':>9} {'Earliest':>12} {'Latest':>12}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        print(
            f"{row['city']:<16} {int(row['n_dates']):>7d} {int(row['n_entry_window']):>9d} "
            f"{row['earliest_date'] or '—':>12} {row['latest_date'] or '—':>12}"
        )
    total_entry = int(df["n_entry_window"].sum())
    cities_with_data = int((df["n_entry_window"] > 0).sum())
    print("-" * len(header))
    print(f"{'TOTAL':<16} {int(df['n_dates'].sum()):>7d} {total_entry:>9d}")
    print(f"\nCities with entry-window data: {cities_with_data} / {len(POLY_CITIES)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket history coverage report")
    parser.add_argument("--floor-date", default=DEFAULT_FLOOR_DATE)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    df, passed = run_report(args.floor_date)
    print_report(df, args.floor_date)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(args.json_out, orient="records", indent=2)
        print(f"\nWrote {args.json_out}")

    ok, detail = evaluate_coverage_gate(df)
    if not ok:
        print_coverage_failure_details(df)
        print(f"\nFAIL: {detail}")
        print(
            "\nPolymarket order-book backfill has not completed or was never run. "
            "Check logs/polymarket_backfill_*.log before proceeding. "
            "Do not run a backtest on synthetic/assumed prices."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
