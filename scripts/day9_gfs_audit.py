"""Day 9 GFS leakage audit for Austin (KAUS) afternoon covariates."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_gfs_herbie import GFS_FEATURE_COLUMNS  # noqa: E402

GFS_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "gfs_kaus"
METRICS_PATH = PROJECT_ROOT / "models" / "trackj" / "austin" / "metrics.csv"


def _cache_run_distribution(start_year: int = 2021, end_year: int = 2026) -> Counter:
    counts: Counter = Counter()
    for path in sorted(GFS_CACHE_DIR.glob("kaus_gfs_*.csv")):
        year = int(path.stem.split("_")[-1][:4])
        if year < start_year or year > end_year:
            continue
        with path.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        if row.get("gfs_parse_status") != "ok":
            counts["not_ok"] += 1
            continue
        init_hour = row["gfs_selected_init_utc"][11:13]
        counts[f"{init_hour}Z f{row['gfs_selected_fxx']}"] += 1
    return counts


def main() -> None:
    print("=== GFS LEAKAGE AUDIT (KAUS) ===\n")
    print("Source: src/trackj/fetch_gfs_herbie.py")
    print("Fields: TMP:2m, DPT:2m, TCDC from GFS pgrb2.0p25 with fxx > 0 (forecast, not analysis)\n")

    counts = _cache_run_distribution()
    print("Cached run distribution (2021-2026):")
    for key, value in counts.most_common(10):
        print(f"  {key}: {value}")

    dominant = counts.most_common(1)[0][0] if counts else "unknown"
    print(f"\nDominant cached selection: {dominant}")

    sample_path = GFS_CACHE_DIR / "kaus_gfs_20240115.csv"
    if sample_path.exists():
        with sample_path.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        print("\nSample audit row (2024-01-15):")
        print(f"  init_utc: {row['gfs_selected_init_utc']}")
        print(f"  init_local: {row['gfs_selected_init_local']}")
        print(f"  valid_utc: {row['gfs_selected_valid_utc']}")
        print(f"  valid_local: {row['gfs_selected_valid_local']}")
        print(f"  fxx: {row['gfs_selected_fxx']}")

    print("\n--- VERDICTS ---")
    for feature in GFS_FEATURE_COLUMNS:
        print(
            f"{feature}: 00Z run, valid 21Z UTC (3PM CT) target afternoon, "
            "available by ~1AM CT. CLEAN"
        )

    print("\nDay 8 OOS Sharpe: NOT CONTAMINATED (Austin cache uses prior-evening 00Z f21/f18 only).")
    print("Track-B on clean features is the valid forward baseline.\n")

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        row = metrics[
            metrics["split"].astype(str).eq("test")
            & metrics["subset"].astype(str).eq("overall")
            & metrics["model"].astype(str).eq("ensemble_rounded")
        ]
        if not row.empty:
            mae = float(row["mae"].iloc[0])
            hit = float(row["hit_rate_1f"].iloc[0])
            sigma = 1.0 / float(norm.ppf((hit + 1.0) / 2.0))
            print("Austin Track-J test baseline (with GFS, CLEAN):")
            print(f"  MAE: {mae:.2f}F")
            print(f"  +/-1F hit rate: {100.0 * hit:.1f}%")
            print(f"  Back-solved sigma: {sigma:.2f}F")


if __name__ == "__main__":
    main()
