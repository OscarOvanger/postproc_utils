"""Extract 10AM/10:05AM snapshots + settlement from full 5-min CSVs.

Reads from: historic_tmax_market_data/<city>/*.csv
Writes to: data/market_lean/lean_snapshots.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_splits import CSV_PATTERN, discover_city_csvs  # noqa: E402
from src.data_store import TRAIN_CITIES  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "historic_tmax_market_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "market_lean"
OUTPUT_PATH = OUTPUT_DIR / "lean_snapshots.parquet"

DATE_COL = "event_date"
TIME_COL = "snapshot_time_local"


def extract_city(city_folder: str, csv_path: Path) -> pd.DataFrame | None:
    """Extract lean snapshots for one city."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)[:5]}...")

    if TIME_COL not in df.columns:
        print(f"  Cannot identify timestamp column. Columns: {df.columns.tolist()}")
        return None

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date

    df["hour"] = df[TIME_COL].dt.hour
    df["minute"] = df[TIME_COL].dt.minute

    mask_10am = (df["hour"] == 10) & (df["minute"] == 0)
    mask_1005 = (df["hour"] == 10) & (df["minute"] == 5)
    lean = df[mask_10am | mask_1005].copy()

    settle_cols = [c for c in df.columns if "resolved" in c.lower() or "settled" in c.lower()]
    if settle_cols:
        print(f"  Settlement columns found: {settle_cols}")
    else:
        print("  No settlement columns found — will need to add from NWS CLI data")

    lean["city"] = city_folder
    lean["source_city_folder"] = city_folder
    lean["snapshot_label"] = lean["minute"].map({0: "10:00", 5: "10:05"})

    print(f"  Extracted {len(lean)} lean rows")
    return lean


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        return

    city_csvs = discover_city_csvs(INPUT_DIR)
    all_lean: list[pd.DataFrame] = []

    for city_folder in sorted(TRAIN_CITIES):
        if city_folder not in city_csvs:
            print(f"\n{city_folder}: no CSV found")
            continue
        print(f"\n{city_folder}:")
        lean = extract_city(city_folder, city_csvs[city_folder])
        if lean is not None and not lean.empty:
            all_lean.append(lean)

    if not all_lean:
        print("No data extracted.")
        return

    combined = pd.concat(all_lean, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== LEAN SNAPSHOT SUMMARY ===")
    print(f"Total rows: {len(combined)}")
    print(f"Date range: {combined[DATE_COL].min()} to {combined[DATE_COL].max()}")
    print(f"Cities: {combined['city'].nunique()}")
    print(f"Saved to: {OUTPUT_PATH}")

    for city in sorted(combined["city"].unique()):
        city_df = combined[combined["city"] == city]
        n_dates = city_df[DATE_COL].nunique()
        max_date = city_df[DATE_COL].max()
        print(f"  {city}: {n_dates} dates through {max_date}")


if __name__ == "__main__":
    main()
