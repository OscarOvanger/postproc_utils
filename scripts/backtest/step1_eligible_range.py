#!/usr/bin/env python3
"""Step 1: determine eligible (city, date) intersections for backtest."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_ngboost as ng  # noqa: E402

from backtest.common import (  # noqa: E402
    ELIGIBLE_DATES_CSV,
    MODEL_PATH_FILE,
    POLY_CITIES,
    REPORTS_DIR,
    features_eligible_cached,
    has_entry_window_snapshot,
    load_day_snapshot,
    load_wu_targets,
    skip_if_exists,
)

SUMMARY_MD = REPORTS_DIR / "backtest_eligible_dates_summary.md"
MIN_ELIGIBLE_TOTAL = 60


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute eligible backtest city-dates")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if skip_if_exists(ELIGIBLE_DATES_CSV, args.force, "step1"):
        return

    if not MODEL_PATH_FILE.exists():
        print(f"ERROR: run step0 first — missing {MODEL_PATH_FILE}")
        sys.exit(1)

    wu = load_wu_targets()
    rows: list[dict[str, str]] = []
    t0 = time.time()
    n_checked = 0

    for city in POLY_CITIES:
        city_dir = PROJECT_ROOT / "data" / "polymarket_history" / "snapshots" / city
        if not city_dir.exists():
            print(f"  {city}: no snapshot directory")
            continue
        dates = sorted(p.stem for p in city_dir.glob("*.parquet"))
        if dates:
            warm_start = date_cls.fromisoformat(dates[0])
            warm_end = date_cls.fromisoformat(dates[-1])
            print(f"  {city}: warming ASOS/Open-Meteo cache {dates[0]}..{dates[-1]}")
            ng.load_temp_early_morning(city, warm_start, warm_end)
            ng.load_openmeteo_tmax(city, warm_start, warm_end)

        for date_str in dates:
            n_checked += 1
            if n_checked % 50 == 0:
                elapsed = time.time() - t0
                print(f"  checked {n_checked} city-dates ({elapsed:.1f}s), eligible so far: {len(rows)}")

            frame = load_day_snapshot(city, date_str)
            if frame is None or not has_entry_window_snapshot(frame, city, date_str):
                continue

            wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
            if wu_row.empty or not bool(wu_row.iloc[0]["reliable"]):
                continue
            if not pd.notna(wu_row.iloc[0]["wunderground_tmax"]):
                continue

            if not features_eligible_cached(city, date_str):
                continue

            rows.append({"city": city, "date": date_str})

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ELIGIBLE_DATES_CSV, index=False)

    summary_lines = ["# Backtest eligible dates summary\n"]
    if df.empty:
        summary_lines.append("No eligible city-dates found.\n")
        total = 0
    else:
        total = len(df)
        summary_lines.append(f"**Total eligible city-dates:** {total}\n")
        summary_lines.append("\n| City | Count | Earliest | Latest |\n|------|------:|----------|--------|\n")
        for city in POLY_CITIES:
            sub = df[df["city"] == city]
            if sub.empty:
                summary_lines.append(f"| {city} | 0 | — | — |\n")
            else:
                summary_lines.append(
                    f"| {city} | {len(sub)} | {sub['date'].min()} | {sub['date'].max()} |\n"
                )

    SUMMARY_MD.write_text("".join(summary_lines), encoding="utf-8")
    print(f"\nWrote {len(df)} eligible rows to {ELIGIBLE_DATES_CSV}")
    print(f"Wrote summary to {SUMMARY_MD}")

    if total <= MIN_ELIGIBLE_TOTAL:
        print(
            f"\nWARNING: only {total} eligible city-dates (threshold {MIN_ELIGIBLE_TOTAL}). "
            "Backtest sample is thin — read results with wider confidence intervals."
        )
    else:
        print(f"Eligible total {total} > {MIN_ELIGIBLE_TOTAL} — sample size OK.")


if __name__ == "__main__":
    main()
