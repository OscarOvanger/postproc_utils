"""Build train/test parquet splits from raw Tmax CSV exports."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from snapshot_stability import load_or_create_frozen_k  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "historic_tmax_market_data"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
CSV_PATTERN = "*tmax_kalshi*5min_same_day.csv"
HOLDOUT_CITIES = ["miami", "denver", "minneapolis"]
TIME_HOLDOUT_DAYS = 15


def discover_city_csvs(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Return a mapping from city folder name to that city's raw CSV path."""
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    city_csvs: dict[str, Path] = {}
    for city_dir in sorted(path for path in raw_data_dir.iterdir() if path.is_dir()):
        matches = sorted(city_dir.glob(CSV_PATTERN))
        if not matches:
            print(f"WARNING: no raw CSV found for {city_dir.name}; skipping")
            continue
        if len(matches) > 1:
            raise ValueError(f"Expected one CSV in {city_dir}, found {len(matches)}")
        city_csvs[city_dir.name] = matches[0]

    if not city_csvs:
        raise FileNotFoundError(f"No {CSV_PATTERN} files found under {raw_data_dir}")
    return city_csvs


def load_city_frame(city_folder: str, csv_path: Path) -> pd.DataFrame:
    """Load one city CSV and normalize date/timestamp columns used for splits."""
    df = pd.read_csv(csv_path)
    required = {"event_date", "snapshot_time_local"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if "city" not in df.columns:
        df["city"] = city_folder.replace("_", " ").title()

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    df["source_city_folder"] = city_folder
    return df


def assign_cities(city_csvs: dict[str, Path]) -> tuple[list[str], list[str], list[str]]:
    """Assign discovered city folders to train and fixed location-holdout sets."""
    city_by_key = {city.casefold(): city for city in city_csvs}
    holdout_keys = {city.casefold() for city in HOLDOUT_CITIES}
    missing_holdouts = [
        city for city in HOLDOUT_CITIES if city.casefold() not in city_by_key
    ]
    holdout_cities = [
        city_by_key[city.casefold()]
        for city in HOLDOUT_CITIES
        if city.casefold() in city_by_key
    ]
    train_cities = sorted(
        city for city in city_csvs if city.casefold() not in holdout_keys
    )
    return train_cities, holdout_cities, missing_holdouts


def cutoff_date_for_train_cities(
    frames: dict[str, pd.DataFrame],
    train_cities: list[str],
) -> object:
    """Return the rolling OOS cutoff date computed from all train cities."""
    if not train_cities:
        raise ValueError("At least one train city is required to compute a time split")

    train_dates = pd.concat(
        [frames[city]["event_date"].dropna() for city in train_cities],
        ignore_index=True,
    )
    if train_dates.empty:
        raise ValueError("Train city frames do not contain any event_date values")
    max_date = max(train_dates)
    return max_date - timedelta(days=TIME_HOLDOUT_DAYS)


def add_partition(df: pd.DataFrame, partition: str) -> pd.DataFrame:
    """Return a copy of a dataframe with a partition label column."""
    out = df.copy()
    out["partition"] = partition
    return out


def empty_partition_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty partition dataframe with the expected output columns."""
    return pd.DataFrame(columns=columns)


def output_columns_for_frames(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Return the union of loaded columns plus the partition label column."""
    columns: list[str] = []
    for df in frames.values():
        for column in df.columns:
            if column not in columns:
                columns.append(column)
    if "partition" not in columns:
        columns.append("partition")
    return columns


def concat_partition(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Concatenate partition frames and keep a stable output schema."""
    if not frames:
        return empty_partition_frame(columns)
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def summarize_partition(name: str, df: pd.DataFrame) -> dict[str, int | str]:
    """Build one row of the printed partition summary table."""
    return {
        "Partition name": name,
        "N cities": int(df["source_city_folder"].nunique()) if not df.empty else 0,
        "N days": int(df["event_date"].nunique()) if not df.empty else 0,
        "N rows": int(len(df)),
    }


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet with a helpful dependency error."""
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing parquet requires a pandas parquet engine. Install pyarrow "
            "or fastparquet, then rerun scripts/build_splits.py."
        ) from exc


def main() -> None:
    """Build all requested parquet split files and print split diagnostics."""
    city_csvs = discover_city_csvs()
    train_cities, holdout_cities, missing_holdouts = assign_cities(city_csvs)

    print("train_cities:")
    for city in train_cities:
        print(f"  - {city}")

    print("holdout_cities:")
    for city in holdout_cities:
        print(f"  - {city}")
    for city in missing_holdouts:
        print(f"WARNING: configured holdout city not found in data folder: {city}")

    frames = {city: load_city_frame(city, path) for city, path in city_csvs.items()}
    output_columns = output_columns_for_frames(frames)
    cutoff_date = cutoff_date_for_train_cities(frames, train_cities)
    print(f"time_split_cutoff_date: {cutoff_date}")

    threshold_frames: list[pd.DataFrame] = []
    time_holdout_frames: list[pd.DataFrame] = []
    location_holdout_frames: list[pd.DataFrame] = []
    true_holdout_frames: list[pd.DataFrame] = []

    for city in train_cities:
        city_df = frames[city]
        threshold_frames.append(
            add_partition(city_df[city_df["event_date"] <= cutoff_date], "threshold_opt")
        )
        time_holdout_frames.append(
            add_partition(city_df[city_df["event_date"] > cutoff_date], "time_holdout")
        )

    for city in holdout_cities:
        city_df = frames[city]
        location_holdout_frames.append(add_partition(city_df, "location_holdout"))

        # TRUE HOLDOUT - do not load or inspect this file until final reporting
        true_holdout_frames.append(
            add_partition(city_df[city_df["event_date"] > cutoff_date], "true_holdout")
        )

    partitions = {
        "threshold_opt": concat_partition(threshold_frames, output_columns),
        "time_holdout": concat_partition(time_holdout_frames, output_columns),
        "location_holdout": concat_partition(location_holdout_frames, output_columns),
        "true_holdout": concat_partition(true_holdout_frames, output_columns),
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in partitions.items():
        write_parquet(df, SPLIT_DIR / f"{name}.parquet")

    summary = pd.DataFrame(
        [summarize_partition(name, df) for name, df in partitions.items()]
    )
    print("\npartition summary:")
    print(summary.to_string(index=False))
    frozen_k = load_or_create_frozen_k(SPLIT_DIR, force_recompute=True)
    print(f"\nfrozen_k recomputed: {frozen_k}")


if __name__ == "__main__":
    main()
