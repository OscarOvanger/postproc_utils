"""Generate Track-B ensemble forecasts for Kalshi IS and OOS partitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.data_store import TRAIN_CITIES, load_features  # noqa: E402
from src.snapshot_stability import assert_no_true_holdout  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
MODEL_DIR = PROJECT_ROOT / "models" / "trackb"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _date_key(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _load_partition(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    df = pd.read_parquet(path)
    assert_no_true_holdout(df)
    return df


def _partition_days() -> pd.DataFrame:
    threshold_opt = _load_partition("threshold_opt")
    time_holdout = _load_partition("time_holdout")
    city_col = "source_city_folder" if "source_city_folder" in threshold_opt.columns else "city"
    rows = []
    for partition_name, frame in [("threshold_opt", threshold_opt), ("time_holdout", time_holdout)]:
        days = frame[[city_col, "event_date"]].drop_duplicates().copy()
        days["city"] = days[city_col].map(_city_key)
        days["event_date"] = pd.to_datetime(days["event_date"]).dt.strftime("%Y-%m-%d")
        days["partition"] = partition_name
        rows.append(days[["city", "event_date", "partition"]])
    days = pd.concat(rows, ignore_index=True)
    return days[days["city"].isin(TRAIN_CITIES)]


def _ensemble_predict(models: list, x: np.ndarray) -> int:
    preds = [model.predict(x)[0] for model in models]
    return int(round(float(np.mean(preds))))


def _load_city_models(city: str) -> tuple[list, list[str]]:
    city_dir = MODEL_DIR / city
    models = [
        joblib.load(city_dir / "ridge.joblib"),
        joblib.load(city_dir / "huber.joblib"),
        joblib.load(city_dir / "lightgbm.joblib"),
    ]
    with open(city_dir / "feature_cols.json", encoding="utf-8") as handle:
        feature_cols = json.load(handle)
    return models, feature_cols


def generate_forecasts() -> pd.DataFrame:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        city_config = json.load(handle)

    partition_days = _partition_days()
    records: list[dict[str, object]] = []

    for city in TRAIN_CITIES:
        models, feature_cols = _load_city_models(city)
        try:
            feat_df = load_features(city, source="local")
        except FileNotFoundError:
            feat_df = load_features(city, source="hf")
        feat_df = feat_df.copy()
        feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.strftime("%Y-%m-%d")
        sigma = float(city_config[city]["trackb_sigma_f"])

        city_days = partition_days[partition_days["city"].eq(city)]
        for _, day in city_days.iterrows():
            event_date = str(day["event_date"])
            row = feat_df[feat_df["date"].eq(event_date)]
            if row.empty:
                continue
            x = row[feature_cols].values
            if np.any(np.isnan(x)):
                continue
            trackb_tmax_f = _ensemble_predict(models, x)
            nws_tmax_f = (
                float(row["nws_tmax_forecast_f"].values[0])
                if "nws_tmax_forecast_f" in row.columns and pd.notna(row["nws_tmax_forecast_f"].values[0])
                else None
            )
            records.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "trackb_tmax_f": trackb_tmax_f,
                    "trackb_sigma_f": sigma,
                    "nws_tmax_f": nws_tmax_f,
                    "model_type": "track_b",
                    "partition": day["partition"],
                }
            )

    forecasts = pd.DataFrame.from_records(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_PATH, index=False)
    return forecasts


def print_coverage_table(forecasts: pd.DataFrame, partition_days: pd.DataFrame) -> list[str]:
    low_oos_cities: list[str] = []
    print(f"{'City':<16} | {'IS days':>7} | {'IS coverage %':>13} | {'OOS days':>8} | {'OOS coverage %':>14}")
    print("-" * 16 + "-|-" + "-" * 7 + "-|-" + "-" * 13 + "-|-" + "-" * 8 + "-|-" + "-" * 14)
    forecast_keys = set(
        zip(
            forecasts["city"].map(_city_key),
            pd.to_datetime(forecasts["event_date"]).dt.strftime("%Y-%m-%d"),
            forecasts["partition"],
        )
    )
    for city in TRAIN_CITIES:
        for part_label, part_name in [("IS", "threshold_opt"), ("OOS", "time_holdout")]:
            pass
        is_days = partition_days[(partition_days["city"].eq(city)) & (partition_days["partition"].eq("threshold_opt"))]
        oos_days = partition_days[(partition_days["city"].eq(city)) & (partition_days["partition"].eq("time_holdout"))]
        is_total = len(is_days)
        oos_total = len(oos_days)
        is_hit = sum((city, _date_key(d), "threshold_opt") in forecast_keys for d in is_days["event_date"])
        oos_hit = sum((city, _date_key(d), "time_holdout") in forecast_keys for d in oos_days["event_date"])
        is_pct = 100.0 * is_hit / is_total if is_total else 0.0
        oos_pct = 100.0 * oos_hit / oos_total if oos_total else 0.0
        if oos_pct < 70.0:
            low_oos_cities.append(city)
        print(
            f"{city:<16} | {is_hit:7d} | {is_pct:12.1f}% | {oos_hit:8d} | {oos_pct:13.1f}%"
        )
    if low_oos_cities:
        print(f"\nWARNING: OOS coverage < 70% for: {', '.join(low_oos_cities)}")
    return low_oos_cities


def main() -> None:
    partition_days = _partition_days()
    forecasts = generate_forecasts()
    print(f"\nSaved {len(forecasts)} forecasts to {OUTPUT_PATH}\n")
    print("=== Track-B Forecast Coverage ===")
    print_coverage_table(forecasts, partition_days)


if __name__ == "__main__":
    main()
