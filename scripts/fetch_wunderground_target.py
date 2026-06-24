"""Fetch Wunderground-equivalent daily Tmax from IEM ASOS hourly METAR data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.poly_trading_pipeline import POLYMARKET_CITIES  # noqa: E402
from src.trackj.build_asos_features import fetch_asos_range, load_cached_asos  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
RAW_ROOT = PROJECT_ROOT / "data" / "trackj" / "raw"

DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date(2026, 6, 23)
MIN_READINGS = 12
TMPF_MIN = -30.0
TMPF_MAX = 140.0


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _daily_targets_from_asos(
    asos_df: pd.DataFrame,
    city: str,
    station: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if asos_df.empty:
        return pd.DataFrame(
            columns=["city", "date", "station", "wunderground_tmax", "n_readings", "reliable"]
        )

    df = asos_df.copy()
    df["date"] = df["date"].astype(str)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")

    rows: list[dict] = []
    for date_str, group in df.groupby("date"):
        day = pd.Timestamp(date_str).date()
        if day < start_date or day > end_date:
            continue
        tmpf = group["tmpf"].dropna()
        n_readings = int(tmpf.shape[0])
        wunderground_tmax = np.nan
        if n_readings > 0:
            daily_max = float(tmpf.max())
            if TMPF_MIN <= daily_max <= TMPF_MAX:
                wunderground_tmax = daily_max
        rows.append(
            {
                "city": city,
                "date": date_str,
                "station": station,
                "wunderground_tmax": wunderground_tmax,
                "n_readings": n_readings,
                "reliable": n_readings >= MIN_READINGS,
            }
        )

    return pd.DataFrame(rows)


def fetch_city_targets(
    city: str,
    city_config: dict,
    start_date: date,
    end_date: date,
    overwrite: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    station = city_config["nws_station"]
    raw_dir = RAW_ROOT / city / "asos"
    print(f"\n=== {city} ({station}) ===")
    fetch_asos_range(
        city_config,
        start_date,
        end_date,
        raw_dir,
        overwrite=overwrite,
        sleep_seconds=sleep_seconds,
    )
    asos_df = load_cached_asos(raw_dir, station, start_date, end_date)
    targets = _daily_targets_from_asos(asos_df, city, station, start_date, end_date)
    if targets.empty:
        print(f"WARNING: no ASOS daily targets for {city}")
        return targets

    n_total = len(targets)
    n_unreliable = int((~targets["reliable"]).sum())
    n_missing = int(targets["wunderground_tmax"].isna().sum())
    print(
        f"{city}: {n_total} days, {n_unreliable} unreliable (<{MIN_READINGS} readings), "
        f"{n_missing} missing tmax, "
        f"range {targets['date'].min()} .. {targets['date'].max()}"
    )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wunderground-equivalent Tmax from IEM ASOS.")
    parser.add_argument("--start", type=str, default=DEFAULT_START.isoformat())
    parser.add_argument("--end", type=str, default=DEFAULT_END.isoformat())
    parser.add_argument("--overwrite", action="store_true", help="Re-fetch ASOS even if cached.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--city", type=str, default=None, help="Single city slug.")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    config = _load_config()
    cities = [args.city] if args.city else list(POLYMARKET_CITIES)

    frames: list[pd.DataFrame] = []
    for city in cities:
        if city not in config:
            raise KeyError(f"City {city!r} not in {CONFIG_PATH}")
        frames.append(
            fetch_city_targets(
                city,
                config[city],
                start_date,
                end_date,
                overwrite=args.overwrite,
                sleep_seconds=args.sleep_seconds,
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["city", "date"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")

    summary = (
        out.groupby("city")
        .agg(
            n_days=("date", "count"),
            unreliable_pct=("reliable", lambda s: round(100.0 * (~s.astype(bool)).mean(), 1)),
            missing_tmax=("wunderground_tmax", lambda s: int(s.isna().sum())),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
