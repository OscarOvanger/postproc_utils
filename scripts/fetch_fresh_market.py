"""Fetch fresh market snapshots for validation (kept separate from splits)."""

from __future__ import annotations

import argparse
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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_splits import discover_city_csvs, load_city_frame  # noqa: E402
from src.data_store import TRAIN_CITIES  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "historic_tmax_market_data"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fresh_validation" / "market_fresh.parquet"
HF_REPO_ID = "oovanger/MCP_datset"


def _existing_split_max() -> date | None:
    path = SPLIT_DIR / "time_holdout.parquet"
    if not path.exists():
        return None
    return pd.to_datetime(pd.read_parquet(path)["event_date"]).max().date()


def _load_local_market(start: date, end: date) -> pd.DataFrame:
    if not RAW_DATA_DIR.exists():
        return pd.DataFrame()
    city_csvs = discover_city_csvs(RAW_DATA_DIR)
    frames: list[pd.DataFrame] = []
    for city in TRAIN_CITIES:
        if city not in city_csvs:
            continue
        df = load_city_frame(city, city_csvs[city])
        mask = (df["event_date"] >= start) & (df["event_date"] <= end)
        subset = df.loc[mask].copy()
        if not subset.empty:
            frames.append(subset)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_hf_market(start: date, end: date) -> pd.DataFrame:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError:
        return pd.DataFrame()

    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    city_csvs = discover_city_csvs(RAW_DATA_DIR) if any(RAW_DATA_DIR.iterdir()) else {}
    frames: list[pd.DataFrame] = []
    for city in TRAIN_CITIES:
        if city in city_csvs:
            continue
        hf_path = f"data/{city}"
        try:
            listing = []
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
            matches = [f for f in files if f.startswith(f"data/{city}/") and "tmax_kalshi" in f]
            if not matches:
                continue
            local = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=matches[0],
                local_dir=PROJECT_ROOT / "data" / "cache" / "hf_raw_market",
            )
            df = load_city_frame(city, Path(local))
            mask = (df["event_date"] >= start) & (df["event_date"] <= end)
            subset = df.loc[mask].copy()
            if not subset.empty:
                frames.append(subset)
        except Exception as exc:
            print(f"  HF download skipped for {city}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2026-06-02")
    parser.add_argument("--end", type=str, default="2026-06-11")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    split_max = _existing_split_max()

    local_df = _load_local_market(start, end)
    if local_df.empty:
        hf_df = _load_hf_market(start, end)
        market_df = hf_df
    else:
        market_df = local_df
        hf_df = _load_hf_market(start, end)
        if not hf_df.empty:
            market_df = pd.concat([market_df, hf_df], ignore_index=True).drop_duplicates()

    latest_existing = split_max
    if market_df.empty:
        latest_in_raw = None
        new_dates: list[str] = []
        n_city_days = 0
    else:
        market_df["event_date"] = pd.to_datetime(market_df["event_date"]).dt.date
        latest_in_raw = market_df["event_date"].max()
        all_dates = sorted(market_df["event_date"].unique())
        if split_max is not None:
            new_dates = [d.isoformat() for d in all_dates if d > split_max]
        else:
            new_dates = [d.isoformat() for d in all_dates]
        n_city_days = int(
            market_df.groupby(["source_city_folder", "event_date"]).ngroups
        )

    print("Data update status:")
    print(f"  Latest date in existing market_df: {latest_existing}")
    print(f"  Latest date in raw/HF fetch: {latest_in_raw}")
    print(f"  New dates available: {new_dates}")
    print(f"  N new city-days: {len(new_dates) * len(TRAIN_CITIES) if new_dates else 0}")

    if market_df.empty:
        print("\nNo new data found. Fresh validation skipped.")
        return

    if not new_dates:
        print(
            "\nNo dates strictly after existing split max. "
            "Saving in-range market data for separate validation file."
        )

    market_df = market_df.copy()
    market_df["partition"] = "fresh_validation"
    market_df["event_date"] = pd.to_datetime(market_df["event_date"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(market_df)} rows to {output_path}")
    print(f"  Date range: {market_df['event_date'].min().date()} to {market_df['event_date'].max().date()}")
    print(f"  City-days in file: {n_city_days}")


if __name__ == "__main__":
    main()
