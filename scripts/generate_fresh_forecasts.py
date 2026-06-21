"""Generate Track-B ensemble forecasts for fresh validation features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_store import TRAIN_CITIES  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
MODEL_DIR = PROJECT_ROOT / "models" / "trackb"
FEATURES_PATH = PROJECT_ROOT / "data" / "fresh_validation" / "features_fresh.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fresh_validation" / "forecasts_fresh.parquet"


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


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


def _ensemble_predict(models: list, x: np.ndarray) -> int:
    preds = [model.predict(x)[0] for model in models]
    return int(round(float(np.mean(preds))))


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run scripts/fetch_and_validate.py first."
        )

    with open(CONFIG_PATH, encoding="utf-8") as handle:
        city_config = json.load(handle)

    features = pd.read_parquet(FEATURES_PATH)
    features["city"] = features["city"].map(_city_key)
    features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")

    parquet_cache: dict[str, pd.DataFrame] = {}

    def _row_features(city: str, event_date: str, row: pd.Series) -> pd.Series | None:
        merged = row.copy()
        for col in feature_cols:
            if col in merged.index and pd.notna(merged[col]):
                continue
            if city not in parquet_cache:
                path = PROJECT_ROOT / "data" / "trackb" / city / "features.parquet"
                parquet_cache[city] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            cached = parquet_cache[city]
            if not cached.empty:
                cached = cached.copy()
                cached["_date"] = pd.to_datetime(cached["date"]).dt.strftime("%Y-%m-%d")
                match = cached[cached["_date"].eq(event_date)]
                if not match.empty and col in match.columns:
                    merged[col] = match.iloc[0][col]
        missing = [col for col in feature_cols if col not in merged.index]
        if missing:
            return None
        values = merged[feature_cols]
        if values.isna().any():
            return None
        return values

    records: list[dict[str, object]] = []
    for city in TRAIN_CITIES:
        city_rows = features[features["city"].eq(city)]
        if city_rows.empty:
            print(f"{city}: no fresh features")
            continue
        models, feature_cols = _load_city_models(city)
        sigma = float(city_config[city]["trackb_sigma_f"])
        n_ok = 0
        for _, row in city_rows.iterrows():
            event_date = str(row["date"])
            values = _row_features(city, event_date, row)
            if values is None:
                continue
            x = values.values.astype(float).reshape(1, -1)
            trackb_tmax_f = _ensemble_predict(models, x)
            nws_tmax_f = (
                float(row["nws_tmax_forecast_f"])
                if "nws_tmax_forecast_f" in row.index and pd.notna(row["nws_tmax_forecast_f"])
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
                    "partition": "fresh_validation",
                }
            )
            n_ok += 1
        print(f"{city}: {n_ok}/{len(city_rows)} forecasts generated")

    if not records:
        print("No forecasts generated.")
        return

    forecasts = pd.DataFrame.from_records(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(forecasts)} forecasts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
