"""Track-J forecast interface and bucket probability conversion."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKA_PATH = PROJECT_ROOT / "data" / "trackj" / "all_cities_trackA.parquet"
FORECASTS_PATH = PROJECT_ROOT / "data" / "track_j" / "forecasts.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_ROOT = PROJECT_ROOT / "models" / "trackj"
AUSTIN_PREDICTIONS_PATH = MODEL_ROOT / "austin" / "test_predictions.parquet"


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _event_date_key(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def load_city_config() -> dict[str, dict]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def update_city_sigma(city: str, sigma_f: float) -> None:
    config = load_city_config()
    config.setdefault(city, {})["sigma_f"] = float(sigma_f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def compute_austin_sigma() -> tuple[float, float]:
    """Return (MAE, sigma) for Austin Track-J test ensemble_rounded metrics."""
    metrics_path = MODEL_ROOT / "austin" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Austin metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    rows = metrics[
        metrics["split"].astype(str).str.lower().eq("test")
        & metrics["subset"].astype(str).str.lower().eq("overall")
        & metrics["model"].astype(str).str.lower().eq("ensemble_rounded")
    ]
    if rows.empty:
        raise ValueError("Could not find test/overall ensemble_rounded row in Austin metrics.csv")
    row = rows.iloc[0]
    mae = float(row["mae"])
    hit_rate_1f = float(row["hit_rate_1f"])
    sigma = 1.0 / float(norm.ppf((hit_rate_1f + 1.0) / 2.0))
    print(f"Austin Track-J: MAE = {mae:.2f}F, Gaussian sigma = {sigma:.2f}F")
    update_city_sigma("austin", sigma)
    return mae, sigma


def load_austin_prediction_table() -> pd.DataFrame:
    """Load Austin precomputed Track-J predictions once for vectorized forecast generation."""
    if not AUSTIN_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing Austin predictions file: {AUSTIN_PREDICTIONS_PATH}")
    df = pd.read_parquet(AUSTIN_PREDICTIONS_PATH, columns=["date", "pred_ensemble_rounded"])
    df = df.rename(columns={"pred_ensemble_rounded": "track_j_tmax_f"})
    df["city"] = "austin"
    df["event_date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["track_j_tmax_f"] = pd.to_numeric(df["track_j_tmax_f"], errors="coerce")
    return df[["city", "event_date", "track_j_tmax_f"]].dropna(subset=["event_date", "track_j_tmax_f"])


def _partition_days() -> pd.DataFrame:
    frames = []
    for name in ("threshold_opt.parquet", "time_holdout.parquet"):
        df = pd.read_parquet(SPLIT_DIR / name, columns=["city", "event_date"])
        day_df = df.drop_duplicates().copy()
        day_df["city_key"] = day_df["city"].map(_city_key)
        day_df["event_date"] = pd.to_datetime(day_df["event_date"]).dt.strftime("%Y-%m-%d")
        frames.append(day_df[["city_key", "event_date"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(["city_key", "event_date"])


def _tracka_lookup(tracka: pd.DataFrame) -> set[tuple[str, str]]:
    keys = tracka[["city", "date"]].copy()
    keys["city"] = keys["city"].map(_city_key)
    keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    return set(map(tuple, keys[["city", "date"]].itertuples(index=False, name=None)))


def _city_statuses(tracka: pd.DataFrame, partition_days: pd.DataFrame) -> pd.DataFrame:
    tracka_counts = tracka.groupby(tracka["city"].map(_city_key)).size().rename("trackA_rows")
    available = _tracka_lookup(tracka)
    rows = []
    for city, group in partition_days.groupby("city_key", sort=True):
        n_partition = int(group.shape[0])
        covered = sum((city, date_value) in available for date_value in group["event_date"])
        n_tracka = int(tracka_counts.get(city, 0))
        if city == "austin":
            status = "track_j"
        elif n_tracka >= 500:
            status = "trainable"
        else:
            status = "insufficient_data"
        rows.append(
            {
                "City": city,
                "N trackA rows": n_tracka,
                "Status": status,
                "In partition N days": n_partition,
                "Coverage %": 100.0 * covered / n_partition if n_partition else math.nan,
            }
        )
    return pd.DataFrame(rows)


def generate_forecasts() -> pd.DataFrame:
    """Generate Track-J forecast rows for threshold_opt + time_holdout city-days."""
    _, austin_sigma = compute_austin_sigma()
    tracka = pd.read_parquet(TRACKA_PATH)
    partition_days = _partition_days()
    available = _tracka_lookup(tracka)
    coverage = _city_statuses(tracka, partition_days)
    austin_predictions = load_austin_prediction_table()
    austin_prediction_lookup = {
        str(row.event_date): float(row.track_j_tmax_f)
        for row in austin_predictions.itertuples(index=False)
    }
    print("\nCity | N trackA rows | Status | In partition N days | Coverage %")
    print(coverage.to_string(index=False, float_format=lambda value: f"{value:0.1f}"))

    records: list[dict[str, object]] = []
    missing_austin: list[str] = []
    for row in partition_days.itertuples(index=False):
        city = str(row.city_key)
        event_date = str(row.event_date)
        has_tracka = (city, event_date) in available
        forecast = np.nan
        sigma = np.nan
        model_type = "track_a_pending"
        coverage_flag = False

        if city == "austin":
            model_type = "track_j"
            if event_date in austin_prediction_lookup:
                forecast = austin_prediction_lookup[event_date]
                sigma = float(austin_sigma)
                coverage_flag = True
            else:
                missing_austin.append(event_date)
        else:
            city_rows = int(coverage.loc[coverage["City"].eq(city), "N trackA rows"].iloc[0])
            if city_rows < 500:
                model_type = "insufficient_data"

        records.append(
            {
                "city": city,
                "event_date": event_date,
                "track_j_tmax_f": forecast,
                "track_j_sigma_f": sigma,
                "model_type": model_type,
                "city_coverage_flag": bool(coverage_flag),
            }
        )

    forecasts = pd.DataFrame.from_records(records).sort_values(["city", "event_date"])
    FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECASTS_PATH, index=False)
    austin_days = forecasts[forecasts["city"].eq("austin")]
    austin_covered = int(austin_days["city_coverage_flag"].sum())
    pending_other = int(forecasts[forecasts["model_type"].eq("track_a_pending")]["city"].nunique())
    print(
        f"\nTrack-J forecasts: Austin {austin_covered} days, "
        f"other cities: {pending_other} pending training"
    )
    if missing_austin:
        print("Missing Austin Track-J prediction dates:", ", ".join(sorted(set(missing_austin))))
    print(f"Saved forecasts to {FORECASTS_PATH}")
    return forecasts


def bucket_probs_from_point_forecast(
    tmax_forecast_f: float,
    sigma_f: float,
    buckets: pd.DataFrame,
) -> dict[str, float]:
    """
    Convert a Track-J point forecast to per-bucket probabilities using a Gaussian CDF.

    buckets: DataFrame with columns bucket_label, bucket_type,
             bucket_lower_inclusive_f, bucket_upper_inclusive_f.
    """
    if pd.isna(tmax_forecast_f) or pd.isna(sigma_f) or float(sigma_f) <= 0:
        raise ValueError("tmax_forecast_f and positive sigma_f are required")
    required = {
        "bucket_label",
        "bucket_type",
        "bucket_lower_inclusive_f",
        "bucket_upper_inclusive_f",
    }
    missing = required.difference(buckets.columns)
    if missing:
        raise ValueError(f"buckets is missing required columns: {sorted(missing)}")

    probs: dict[str, float] = {}
    forecast = float(tmax_forecast_f)
    sigma = float(sigma_f)
    bucket_defs = buckets[list(required)].drop_duplicates("bucket_label")
    for bucket in bucket_defs.itertuples(index=False):
        label = str(bucket.bucket_label)
        bucket_type = str(bucket.bucket_type)
        lower = pd.to_numeric(bucket.bucket_lower_inclusive_f, errors="coerce")
        upper = pd.to_numeric(bucket.bucket_upper_inclusive_f, errors="coerce")
        if bucket_type == "RANGE":
            prob = norm.cdf((float(upper) - forecast) / sigma) - norm.cdf((float(lower) - forecast) / sigma)
        elif bucket_type == "LESS_THAN":
            prob = norm.cdf((float(upper) - forecast) / sigma)
        elif bucket_type == "GREATER_THAN":
            prob = 1.0 - norm.cdf((float(lower) - forecast) / sigma)
        else:
            raise ValueError(f"Unsupported bucket_type: {bucket_type}")
        probs[label] = max(float(prob), 0.0)

    total = sum(probs.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Computed bucket probabilities do not have a positive finite sum")
    return {label: prob / total for label, prob in probs.items()}


def get_track_j_bucket_probs(
    city: str,
    event_date: str,
    forecasts_df: pd.DataFrame,
    day_snapshot_df: pd.DataFrame,
) -> dict[str, float] | None:
    """
    Look up Track-J forecast for (city, date), extract bucket definitions, and return probabilities.
    """
    city_key = _city_key(city)
    event_key = _event_date_key(event_date)
    forecasts = forecasts_df.copy()
    forecasts["city"] = forecasts["city"].map(_city_key)
    forecasts["event_date"] = pd.to_datetime(forecasts["event_date"]).dt.strftime("%Y-%m-%d")
    row = forecasts[forecasts["city"].eq(city_key) & forecasts["event_date"].eq(event_key)]
    if row.empty:
        return None
    row = row.iloc[0]
    if not bool(row.get("city_coverage_flag", False)) or pd.isna(row["track_j_tmax_f"]) or pd.isna(row["track_j_sigma_f"]):
        return None
    buckets = day_snapshot_df[
        [
            "bucket_label",
            "bucket_type",
            "bucket_lower_inclusive_f",
            "bucket_upper_inclusive_f",
        ]
    ].drop_duplicates("bucket_label")
    return bucket_probs_from_point_forecast(
        float(row["track_j_tmax_f"]),
        float(row["track_j_sigma_f"]),
        buckets,
    )


if __name__ == "__main__":
    generate_forecasts()
