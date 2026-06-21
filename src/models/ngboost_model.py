"""NGBoost heteroscedastic Gaussian models for Track-J lead-time forecasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import CRPScore
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CITIES = [
    "austin",
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "san_francisco",
]
LEAD_TIMES = ["t1", "t2", "t3"]
FEATURES_ROOT = PROJECT_ROOT / "data" / "ngboost" / "features"
MODELS_ROOT = PROJECT_ROOT / "models" / "ngboost"
CORR_ROOT = PROJECT_ROOT / "data" / "ngboost" / "correlation"
META_COLS = frozenset({"date", "tmax_f"})

TRAIN_START = pd.Timestamp("2021-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
MIN_FEATURE_PARQUET_ROWS = 1000


def feature_parquet_path(city: str, lead_time: str) -> Path:
    return FEATURES_ROOT / city / f"{lead_time}.parquet"


def model_dir(city: str, lead_time: str) -> Path:
    return MODELS_ROOT / city / lead_time


def load_feature_table(city: str, lead_time: str) -> pd.DataFrame:
    path = feature_parquet_path(city, lead_time)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature parquet: {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tmax_f"] = pd.to_numeric(df["tmax_f"], errors="coerce")
    df = df[df["tmax_f"].notna()].reset_index(drop=True)
    return df


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = df["date"]
    train = df[(dates >= TRAIN_START) & (dates <= TRAIN_END)].copy()
    val = df[dates.dt.year == 2025].copy()
    test = df[dates.dt.year >= 2026].copy()
    return train, val, test


def select_feature_columns(df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_features = [col for col in df.columns if col not in META_COLS]
    dropped = [col for col in all_features if train_df[col].isna().all()]
    feature_cols = [col for col in all_features if col not in dropped]
    return feature_cols, dropped


def build_ngboost_regressor() -> NGBRegressor:
    return NGBRegressor(
        Dist=Normal,
        Score=CRPScore,
        Base=DecisionTreeRegressor(max_depth=4, min_samples_leaf=10),
        n_estimators=500,
        learning_rate=0.05,
        natural_gradient=True,
        verbose=False,
    )


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[NGBRegressor, int]:
    model = build_ngboost_regressor()
    if len(y_val) == 0:
        model.fit(X_train, y_train)
    else:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            Y_val=y_val,
            early_stopping_rounds=30,
        )
    best_n = getattr(model, "best_val_loss_itr", None)
    if best_n is None:
        best_n = model.n_estimators
    return model, int(best_n)


def predict_mu_sigma(model: NGBRegressor, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dist = model.pred_dist(X)
    mu = np.asarray(dist.params["loc"], dtype=float)
    sigma = np.asarray(dist.params["scale"], dtype=float)
    return mu, sigma


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    z = (y_arr - mu_arr) / sigma_arr
    crps = sigma_arr * (
        z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def evaluate_predictions(
    y: np.ndarray | pd.Series,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    return {
        "mae": float(np.mean(np.abs(mu_arr - y_arr))),
        "crps": gaussian_crps(y_arr, mu_arr, sigma_arr),
        "mean_sigma": float(np.mean(sigma_arr)),
        "std_sigma": float(np.std(sigma_arr, ddof=1)) if len(sigma_arr) > 1 else 0.0,
        "min_sigma": float(np.min(sigma_arr)),
        "max_sigma": float(np.max(sigma_arr)),
        "mean_mu": float(np.mean(mu_arr)),
    }


def sanity_warnings(metrics: dict[str, float], city: str, lead_time: str) -> None:
    mae = metrics.get("mae", float("nan"))
    mean_sigma = metrics.get("mean_sigma", float("nan"))
    if np.isfinite(mean_sigma) and (mean_sigma < 0.5 or mean_sigma > 15.0):
        print(
            f"  WARNING [{city}/{lead_time}]: mean_sigma={mean_sigma:.2f} "
            "(likely miscalibrated; expected 0.5–15)"
        )
    if np.isfinite(mae) and mae > 5.0:
        print(f"  WARNING [{city}/{lead_time}]: val_MAE={mae:.2f} (poor forecast; expected <5.0)")


def save_model_bundle(
    city: str,
    lead_time: str,
    model: NGBRegressor,
    feature_cols: list[str],
    dropped: list[str],
    summary: dict[str, Any],
) -> Path:
    out_dir = model_dir(city, lead_time)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "feature_cols.json", "w", encoding="utf-8") as handle:
        json.dump(feature_cols, handle, indent=2)
        handle.write("\n")
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return out_dir


def load_trained_model(city: str, lead_time: str) -> tuple[NGBRegressor, list[str]]:
    out_dir = model_dir(city, lead_time)
    model = joblib.load(out_dir / "model.joblib")
    with open(out_dir / "feature_cols.json", encoding="utf-8") as handle:
        feature_cols = json.load(handle)
    return model, feature_cols


def train_city_lead(city: str, lead_time: str) -> dict[str, Any]:
    df = load_feature_table(city, lead_time)
    train_df, val_df, test_df = split_by_year(df)
    feature_cols, dropped = select_feature_columns(df, train_df)

    X_train = train_df[feature_cols]
    y_train = train_df["tmax_f"]
    X_val = val_df[feature_cols]
    y_val = val_df["tmax_f"]

    model, best_n = fit_model(X_train, y_train, X_val, y_val)

    train_mu, train_sigma = predict_mu_sigma(model, X_train)
    train_metrics = evaluate_predictions(y_train, train_mu, train_sigma)

    val_metrics: dict[str, float | None] = {
        "mae": None,
        "crps": None,
        "mean_sigma": None,
        "std_sigma": None,
        "min_sigma": None,
        "max_sigma": None,
        "mean_mu": None,
    }
    if len(y_val) > 0:
        val_mu, val_sigma = predict_mu_sigma(model, X_val)
        val_metrics = evaluate_predictions(y_val, val_mu, val_sigma)
        sanity_warnings(val_metrics, city, lead_time)

    summary: dict[str, Any] = {
        "city": city,
        "lead_time": lead_time,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": len(feature_cols),
        "features_dropped": dropped,
        "val_crps": round(val_metrics["crps"], 4) if val_metrics["crps"] is not None else None,
        "val_mae": round(val_metrics["mae"], 4) if val_metrics["mae"] is not None else None,
        "val_mean_sigma": round(val_metrics["mean_sigma"], 4) if val_metrics["mean_sigma"] is not None else None,
        "val_std_sigma": round(val_metrics["std_sigma"], 4) if val_metrics["std_sigma"] is not None else None,
        "val_mean_mu": round(val_metrics["mean_mu"], 4) if val_metrics["mean_mu"] is not None else None,
        "train_crps": round(train_metrics["crps"], 4),
        "best_n_estimators": best_n,
    }
    save_model_bundle(city, lead_time, model, feature_cols, dropped, summary)

    # Per-model stdout diagnostics
    if val_metrics["mae"] is not None:
        print(
            f"{city:18s} {lead_time:3s}  n_train={summary['n_train']:4d}  n_val={summary['n_val']:3d}  "
            f"val_MAE={val_metrics['mae']:.3f}  val_CRPS={val_metrics['crps']:.3f}  "
            f"mean_sigma={val_metrics['mean_sigma']:.3f}  "
            f"min_sigma={val_metrics['min_sigma']:.3f}  max_sigma={val_metrics['max_sigma']:.3f}  "
            f"best_n_estimators={best_n}"
        )
    else:
        print(
            f"{city:18s} {lead_time:3s}  n_train={summary['n_train']:4d}  n_val={summary['n_val']:3d}  "
            f"val_MAE=n/a  val_CRPS=n/a  mean_sigma=n/a  "
            f"min_sigma=n/a  max_sigma=n/a  best_n_estimators={best_n}"
        )

    return summary


def _innovation_variances(cov: np.ndarray) -> tuple[float, float, float]:
    r2_1 = float(cov[0, 0])
    r2_2 = float(cov[1, 1] - cov[1, 0] ** 2 / cov[0, 0])
    inv_block = np.linalg.inv(cov[:2, :2])
    r2_3 = float(cov[2, 2] - cov[2, :2] @ inv_block @ cov[:2, 2])
    return r2_1, r2_2, r2_3


def estimate_lead_correlation(city: str) -> dict[str, Any] | None:
    """Estimate shrunk cross-lead error correlation on 2025 validation overlap."""
    val_frames: dict[str, pd.DataFrame] = {}
    for lead_time in LEAD_TIMES:
        df = load_feature_table(city, lead_time)
        _, val_df, _ = split_by_year(df)
        if val_df.empty:
            return None
        val_frames[lead_time] = val_df

    common_dates = set(val_frames["t1"]["date"])
    for lead_time in LEAD_TIMES[1:]:
        common_dates &= set(val_frames[lead_time]["date"])
    if not common_dates:
        return None

    common_sorted = sorted(common_dates)
    errors: list[np.ndarray] = []
    for lead_time in LEAD_TIMES:
        model, feature_cols = load_trained_model(city, lead_time)
        val_df = val_frames[lead_time]
        val_df = val_df[val_df["date"].isin(common_sorted)].sort_values("date")
        X_val = val_df[feature_cols]
        y_val = val_df["tmax_f"].to_numpy(dtype=float)
        mu, sigma = predict_mu_sigma(model, X_val)
        sigma_safe = np.maximum(sigma, 1e-8)
        e = (y_val - mu) / sigma_safe
        errors.append(e)

    E = np.column_stack(errors)
    lw = LedoitWolf().fit(E)
    cov_shrunk = lw.covariance_
    stds = np.sqrt(np.maximum(np.diag(cov_shrunk), 1e-12))
    corr = cov_shrunk / np.outer(stds, stds)

    r2_1, r2_2, r2_3 = _innovation_variances(cov_shrunk)

    CORR_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(CORR_ROOT / f"{city}_R.npy", cov_shrunk)
    innov = {
        "city": city,
        "n_val_overlap": int(len(common_sorted)),
        "r2_1": round(r2_1, 6),
        "r2_2": round(r2_2, 6),
        "r2_3": round(r2_3, 6),
        "R_diag": [round(float(corr[i, i]), 4) for i in range(3)],
        "R_01": round(float(corr[0, 1]), 4),
        "R_02": round(float(corr[0, 2]), 4),
        "R_12": round(float(corr[1, 2]), 4),
    }
    with open(CORR_ROOT / f"{city}_innov.json", "w", encoding="utf-8") as handle:
        json.dump(innov, handle, indent=2)
        handle.write("\n")

    return innov


def preflight_feature_parquets() -> list[str]:
    """Return list of missing or undersized feature parquet paths."""
    missing: list[str] = []
    for city in CITIES:
        for lead_time in LEAD_TIMES:
            path = feature_parquet_path(city, lead_time)
            if not path.exists():
                missing.append(f"{path} (not found)")
                continue
            try:
                n_rows = len(pd.read_parquet(path, columns=["tmax_f"]))
            except Exception as exc:
                missing.append(f"{path} (unreadable: {exc})")
                continue
            if n_rows < MIN_FEATURE_PARQUET_ROWS:
                missing.append(f"{path} ({n_rows} rows < {MIN_FEATURE_PARQUET_ROWS})")
    return missing
