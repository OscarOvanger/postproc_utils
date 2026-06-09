"""Track-J point forecast loading and prediction helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "trackj"
AUSTIN_PREDICTIONS_PATH = MODEL_ROOT / "austin" / "test_predictions.parquet"
AUSTIN_METRICS_PATH = MODEL_ROOT / "austin" / "metrics.csv"


def _normalise_date(value) -> object:
    return pd.Timestamp(value).date()


@lru_cache(maxsize=1)
def load_austin_predictions() -> pd.DataFrame:
    """
    Load pre-computed Track-J test predictions for Austin.

    Normalised columns:
      date: datetime.date
      track_j_tmax_f: float
      city: "austin"
    """
    df = pd.read_parquet(AUSTIN_PREDICTIONS_PATH)
    print("Austin predictions columns:", df.columns.tolist())
    print("Austin predictions date range:", df["date"].min(), "to", df["date"].max())
    print("Austin predictions rows:", len(df))

    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {AUSTIN_PREDICTIONS_PATH}")
    if "pred_ensemble_rounded" not in df.columns:
        raise ValueError("Missing pred_ensemble_rounded column in Austin Track-J predictions")

    normalised = df[["date", "pred_ensemble_rounded"]].copy()
    normalised["date"] = pd.to_datetime(normalised["date"], errors="coerce").dt.date
    normalised["track_j_tmax_f"] = pd.to_numeric(normalised["pred_ensemble_rounded"], errors="coerce")
    normalised["city"] = "austin"
    return normalised[["city", "date", "track_j_tmax_f"]].dropna(subset=["date", "track_j_tmax_f"])


@lru_cache(maxsize=1)
def load_austin_sigma() -> float:
    """
    Load Austin Track-J hit rate and compute Gaussian sigma.

    sigma = 1 / Phi_inv((hit_rate_1f + 1) / 2)
    """
    from scipy.stats import norm

    metrics = pd.read_csv(AUSTIN_METRICS_PATH)
    print("Austin metrics:", metrics)
    rows = metrics[
        metrics["split"].astype(str).str.lower().eq("test")
        & metrics["subset"].astype(str).str.lower().eq("overall")
        & metrics["model"].astype(str).str.lower().eq("ensemble_rounded")
    ]
    if rows.empty:
        rows = metrics[
            metrics["split"].astype(str).str.lower().eq("test")
            & metrics["subset"].astype(str).str.lower().eq("overall")
        ]
    if rows.empty or "hit_rate_1f" not in rows.columns:
        raise ValueError(f"Could not find test/overall hit_rate_1f in {AUSTIN_METRICS_PATH}")
    hit_rate = float(rows["hit_rate_1f"].iloc[0])
    sigma = 1.0 / norm.ppf((hit_rate + 1.0) / 2.0)
    print(f"Austin sigma: {sigma:.3f}F (from hit_rate_1f={hit_rate:.3f})")
    return float(sigma)


def predict_tmax(city: str, target_date, feature_row=None) -> dict | None:
    """
    For Austin, look up the pre-computed Track-J prediction for target_date.

    Other cities return None until city-specific models are trained.
    feature_row is accepted for API compatibility and ignored for Austin.
    """
    if str(city) != "austin":
        return None

    target_key = _normalise_date(target_date)
    preds = load_austin_predictions()
    row = preds[preds["date"].eq(target_key)]
    if row.empty:
        return None
    return {
        "city": "austin",
        "date": target_key,
        "track_j_tmax_f": float(row["track_j_tmax_f"].iloc[0]),
        "track_j_sigma_f": load_austin_sigma(),
        "model_type": "track_j",
    }
