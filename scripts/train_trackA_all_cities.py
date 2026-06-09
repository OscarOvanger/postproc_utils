"""Train Track-A models for non-holdout cities and regenerate Track-J forecasts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.track_j import generate_forecasts  # noqa: E402
from src.trackj.build_trackA_table import TRACK_A_COVARIATES  # noqa: E402

TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
MODEL_DIR = PROJECT_ROOT / "models" / "trackj"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
FORECASTS_PATH = PROJECT_ROOT / "data" / "track_j" / "forecasts.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
HOLDOUT_CITIES = {"miami", "denver", "minneapolis"}
TRAIN_CITIES = [
    "austin",
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "oklahoma_city",
    "philadelphia",
    "phoenix",
    "san_francisco",
]
TRACKA_MODEL_FILES = ["ridge.joblib", "huber.joblib", "lightgbm.joblib"]


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def _sigma_from_hit_rate(hit_rate_1f: float) -> float:
    if not math.isfinite(hit_rate_1f) or hit_rate_1f <= 0 or hit_rate_1f >= 1:
        return float("nan")
    return float(1.0 / norm.ppf((hit_rate_1f + 1.0) / 2.0))


def _split_city(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = df.sort_values("date_dt").copy()
    train = df[df["date_dt"] < pd.Timestamp("2024-01-01")]
    val = df[(df["date_dt"] >= pd.Timestamp("2024-01-01")) & (df["date_dt"] < pd.Timestamp("2025-01-01"))]
    test = df[df["date_dt"] >= pd.Timestamp("2025-01-01")]
    if train.empty or val.empty or test.empty:
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        print("  Warning: insufficient calendar split history; using 70/15/15 chronological split.")
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:], "chronological_70_15_15"
    return train, val, test, "calendar"


def _fit_models(train: pd.DataFrame) -> dict[str, object]:
    x_train = train[TRACK_A_COVARIATES]
    y_train = pd.to_numeric(train["tmax_f"], errors="coerce")
    models: dict[str, object] = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "huber": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, max_iter=1000)),
        "lightgbm": LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=0,
            verbosity=-1,
        ),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def _ensemble_predict(models: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack(
        [model.predict(frame[TRACK_A_COVARIATES]) for model in models.values()]
    )
    return np.rint(preds.mean(axis=1))


def _evaluate(test: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
    y = pd.to_numeric(test["tmax_f"], errors="coerce").to_numpy(dtype=float)
    err = preds - y
    hit_rate_1f = float((np.abs(err) <= 1.0).mean()) if y.size else float("nan")
    hit_rate_2f = float((np.abs(err) <= 2.0).mean()) if y.size else float("nan")
    return {
        "mae_f": float(np.abs(err).mean()) if y.size else float("nan"),
        "hit_rate_1f": hit_rate_1f,
        "hit_rate_2f": hit_rate_2f,
        "sigma_f": _sigma_from_hit_rate(hit_rate_1f),
    }


def _partition_days() -> pd.DataFrame:
    frames = []
    for name in ("threshold_opt.parquet", "time_holdout.parquet"):
        df = pd.read_parquet(SPLIT_DIR / name, columns=["city", "event_date"])
        day_df = df.drop_duplicates().copy()
        day_df["city"] = day_df["city"].astype(str).str.lower().str.replace(" ", "_")
        day_df["event_date"] = pd.to_datetime(day_df["event_date"]).dt.strftime("%Y-%m-%d")
        frames.append(day_df)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _load_trained_models(city: str) -> dict[str, object] | None:
    city_dir = MODEL_DIR / city
    paths = [city_dir / name for name in TRACKA_MODEL_FILES]
    if not all(path.exists() for path in paths):
        return None
    return {
        "ridge": joblib.load(city_dir / "ridge.joblib"),
        "huber": joblib.load(city_dir / "huber.joblib"),
        "lightgbm": joblib.load(city_dir / "lightgbm.joblib"),
    }


def regenerate_forecasts_with_tracka(trained_cities: list[str]) -> pd.DataFrame:
    forecasts = generate_forecasts()
    if not trained_cities:
        return forecasts

    tracka = pd.read_parquet(TRACKJ_DIR / "all_cities_trackA.parquet")
    tracka["city"] = tracka["city"].astype(str).str.lower().str.replace(" ", "_")
    tracka["event_date"] = pd.to_datetime(tracka["date"]).dt.strftime("%Y-%m-%d")
    partition_days = _partition_days()

    for city in trained_cities:
        models = _load_trained_models(city)
        if models is None:
            continue
        city_days = partition_days[partition_days["city"].eq(city)]
        city_features = city_days.merge(
            tracka[["city", "event_date", *TRACK_A_COVARIATES]],
            on=["city", "event_date"],
            how="left",
        )
        complete = city_features[TRACK_A_COVARIATES].notna().all(axis=1)
        if complete.any():
            preds = _ensemble_predict(models, city_features.loc[complete])
            pred_map = dict(zip(city_features.loc[complete, "event_date"], preds))
        else:
            pred_map = {}
        mask = forecasts["city"].eq(city)
        forecasts.loc[mask, "model_type"] = "track_a"
        forecasts.loc[mask, "track_j_tmax_f"] = forecasts.loc[mask, "event_date"].map(pred_map)
        config = _load_config()
        sigma = config.get(city, {}).get("sigma_f", np.nan)
        forecasts.loc[mask & forecasts["track_j_tmax_f"].notna(), "track_j_sigma_f"] = sigma
        forecasts.loc[mask, "city_coverage_flag"] = forecasts.loc[mask, "track_j_tmax_f"].notna()

    forecasts.to_parquet(FORECASTS_PATH, index=False)
    return forecasts


def main() -> None:
    all_tracka_path = TRACKJ_DIR / "all_cities_trackA.parquet"
    if not all_tracka_path.exists() or all_tracka_path.stat().st_size == 0:
        print("Track-A all-cities table is missing or empty; multi-city training is deferred.")
        return

    config = _load_config()
    trained_cities: list[str] = []
    summary_rows: list[dict[str, object]] = []

    for city in TRAIN_CITIES:
        if city in HOLDOUT_CITIES or city == "austin":
            continue
        table_path = TRACKJ_DIR / city / "trackA_table.parquet"
        if not table_path.exists():
            print(f"Training {city}... skipped. Missing {table_path}")
            continue
        df = pd.read_parquet(table_path).dropna(subset=["tmax_f", *TRACK_A_COVARIATES]).copy()
        if df.shape[0] < 200:
            print(f"Training {city}... skipped. Fewer than 200 complete rows ({df.shape[0]}).")
            continue
        df["date_dt"] = pd.to_datetime(df["date"])
        train, val, test, split_method = _split_city(df)
        print(f"Training {city}...", end=" ", flush=True)
        models = _fit_models(train)
        preds = _ensemble_predict(models, test)
        metrics = _evaluate(test, preds)
        metrics.update(
            {
                "n_train": int(train.shape[0]),
                "n_val": int(val.shape[0]),
                "n_test": int(test.shape[0]),
                "split_method": split_method,
            }
        )
        city_model_dir = MODEL_DIR / city
        city_model_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, city_model_dir / f"{name}.joblib")
        with open(TRACKJ_DIR / city / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
        config.setdefault(city, {})["sigma_f"] = metrics["sigma_f"]
        trained_cities.append(city)
        summary_rows.append({"City": city, "Model": "track_a", **metrics})
        print(f"done. Hit rate: {metrics['hit_rate_1f'] * 100:0.1f}%")

    _save_config(config)
    forecasts = regenerate_forecasts_with_tracka(trained_cities)

    print("\nFinal Track-J/Track-A coverage table:")
    rows = []
    for row in summary_rows:
        city = str(row["City"])
        n_covered = int(forecasts[forecasts["city"].eq(city)]["city_coverage_flag"].sum())
        rows.append(
            {
                "City": city,
                "Model": row["Model"],
                "Sigma": row["sigma_f"],
                "+/-1F hit rate": row["hit_rate_1f"],
                "N Kalshi days covered": n_covered,
            }
        )
    austin = forecasts[forecasts["city"].eq("austin")]
    rows.insert(
        0,
        {
            "City": "austin",
            "Model": "track_j",
            "Sigma": _load_config().get("austin", {}).get("sigma_f", np.nan),
            "+/-1F hit rate": 0.591,
            "N Kalshi days covered": int(austin["city_coverage_flag"].sum()),
        },
    )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:0.3f}"))


if __name__ == "__main__":
    main()
