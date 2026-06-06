"""Build the standalone HTML market-explorer payload for the research log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORT_DIR / "market_explorer_payload.js"
EXPLORER_COLUMNS = {
    "city",
    "event_date",
    "snapshot_time_local",
    "bucket_label",
    "bucket_type",
    "bucket_lower_inclusive_f",
    "bucket_upper_inclusive_f",
    "yes_mid_close",
    "bucket_resolved_to_one_dollars",
    "minutes_before_tmax",
}


def shannon_entropy(probabilities: list[float]) -> float:
    """Return Shannon entropy for positive finite probabilities."""
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    return float(-(probs * np.log2(probs)).sum())


def load_market_df() -> pd.DataFrame:
    """Load normalized explorer rows from non-duplicating split parquets."""
    frames: list[pd.DataFrame] = []
    for name in ["threshold_opt", "time_holdout", "location_holdout"]:
        path = SPLIT_DIR / f"{name}.parquet"
        df = pd.read_parquet(path, columns=[*EXPLORER_COLUMNS])
        frames.append(df)
        print(f"loaded {name}: {len(df):,} rows", flush=True)

    if not frames:
        raise FileNotFoundError(f"No split parquet files found under {SPLIT_DIR}")
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_explorer_df(market_df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted dataframe ready to serialize by city/date/snapshot."""
    df = market_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date.astype(str)
    df["_snapshot_sort"] = pd.to_datetime(df["snapshot_time_local"].astype(str), utc=True)
    df["_snapshot_label"] = df["snapshot_time_local"].map(
        lambda value: pd.Timestamp(value).strftime("%H:%M")
    )
    df["_bucket_group"] = np.select(
        [
            df["bucket_type"].eq("LESS_THAN"),
            df["bucket_type"].eq("RANGE"),
            df["bucket_type"].eq("GREATER_THAN"),
        ],
        [0, 1, 2],
        default=3,
    )
    df["_bucket_bound"] = np.select(
        [
            df["bucket_type"].eq("LESS_THAN"),
            df["bucket_type"].eq("RANGE"),
            df["bucket_type"].eq("GREATER_THAN"),
        ],
        [
            pd.to_numeric(df["bucket_upper_inclusive_f"], errors="coerce"),
            pd.to_numeric(df["bucket_lower_inclusive_f"], errors="coerce"),
            pd.to_numeric(df["bucket_lower_inclusive_f"], errors="coerce"),
        ],
        default=np.inf,
    )
    return df.sort_values(
        [
            "city",
            "event_date",
            "_snapshot_sort",
            "_bucket_group",
            "_bucket_bound",
            "bucket_label",
        ]
    )


def build_payload(explorer_df: pd.DataFrame) -> dict:
    """Build the compact JSON-serializable explorer payload."""
    with open(SPLIT_DIR / "frozen_k.json", "r", encoding="utf-8") as handle:
        frozen_k = int(json.load(handle)["k"])

    cities: dict[str, dict] = {}
    for city, city_df in explorer_df.groupby("city", sort=True):
        print(f"serializing {city}", flush=True)
        dates: dict[str, list[dict]] = {}
        for event_date, day_df in city_df.groupby("event_date", sort=True):
            snapshots: list[dict] = []
            for _, snapshot_df in day_df.groupby("_snapshot_sort", sort=True):
                prices = snapshot_df["yes_mid_close"].astype(float).round(4).tolist()
                modal_idx = int(np.nanargmax(prices)) if prices else 0
                minutes_before_tmax = snapshot_df["minutes_before_tmax"].iloc[0]
                snapshots.append(
                    {
                        "time": str(snapshot_df["_snapshot_label"].iloc[0]),
                        "minutesBeforeTmax": (
                            None
                            if pd.isna(minutes_before_tmax)
                            else float(minutes_before_tmax)
                        ),
                        "labels": snapshot_df["bucket_label"].astype(str).tolist(),
                        "prices": prices,
                        "winners": snapshot_df[
                            "bucket_resolved_to_one_dollars"
                        ].astype(bool).tolist(),
                        "modalBucket": str(snapshot_df["bucket_label"].iloc[modal_idx]),
                        "modeProb": float(prices[modal_idx]) if prices else None,
                        "entropy": round(shannon_entropy(prices), 4),
                    }
                )
            dates[str(event_date)] = snapshots
        cities[str(city)] = {"dates": dates}
        print(f"serialized {city}: {len(dates):,} dates", flush=True)

    return {"frozenK": frozen_k, "cities": cities}


def main() -> None:
    """Build and write reports/market_explorer_payload.js."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    market_df = load_market_df()
    explorer_df = prepare_explorer_df(market_df)
    payload = build_payload(explorer_df)
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    safe_payload = payload_json.replace("</", "<\\/")
    OUTPUT_PATH.write_text(
        f"window.MARKET_EXPLORER_DATA={safe_payload};\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
