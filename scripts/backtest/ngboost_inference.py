"""Point-in-time NGBoost bucket probability inference for backtest."""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date as date_cls, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import re

import train_ngboost as ng  # noqa: E402
from scipy.stats import norm, t as student_t  # noqa: E402


def parse_snapshot_bucket(label: str) -> dict[str, object]:
    """Parse Polymarket snapshot bucket strings (e.g. <=55, >=74, 84-85)."""
    text = str(label).strip()
    if not text or text.startswith("Will "):
        raise ValueError(f"Unable to parse bucket label: {label!r}")

    le = re.match(r"^<=(\d+)$", text)
    if le:
        return {"type": "LESS_THAN", "lower": None, "upper": int(le.group(1))}

    ge = re.match(r"^>=(\d+)$", text)
    if ge:
        return {"type": "GREATER_THAN", "lower": int(ge.group(1)), "upper": None}

    rng = re.match(r"^(\d+)-(\d+)$", text)
    if rng:
        return {"type": "RANGE", "lower": int(rng.group(1)), "upper": int(rng.group(2))}

    from src.polymarket_api import parse_bucket_label  # noqa: WPS433

    return parse_bucket_label(text)


class NgBoostBacktestModels:
    """Load NGBoost artifacts from a backtest model directory (read-only)."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        config_path = model_dir / "model_config.json"
        with open(config_path, encoding="utf-8") as handle:
            self.config: dict[str, Any] = json.load(handle)
        self.model = joblib.load(model_dir / "ngboost_global.pkl")
        self.scaler = joblib.load(model_dir / "feature_scaler.pkl")
        self.lgb_model = joblib.load(model_dir / self.config.get("stage1_model", "lgb_stage1.pkl"))
        self.feature_cols: list[str] = list(self.config["feature_columns"])
        self.stage1_cols = [c for c in self.feature_cols if c != "lgb_tmax_pred"]
        self.fill_medians: dict[str, float] = dict(self.config.get("nan_fill_medians", {}))
        self.distribution = str(self.config.get("distribution", "gaussian"))
        self.sigma_k = float(self.config.get("sigma_calibration_k", 1.0))

    @classmethod
    def from_path_file(cls, path_file: Path) -> "NgBoostBacktestModels":
        text = path_file.read_text(encoding="utf-8").strip()
        return cls(Path(text))


@lru_cache(maxsize=1)
def _models_singleton(model_dir_str: str) -> NgBoostBacktestModels:
    return NgBoostBacktestModels(Path(model_dir_str))


def build_backtest_features(city: str, event_date: str) -> pd.DataFrame | None:
    """Build feature vector using only data available before 10am entry on event_date."""
    if city not in ng.STATION_META:
        return None

    hrrr = ng.load_hrrr_city(city)
    hrrr["date"] = pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d")
    if event_date not in set(hrrr["date"]):
        return None

    wu = ng._load_wu_all()
    wu = wu[wu["reliable"].astype(bool)].copy()
    wu["date"] = pd.to_datetime(wu["date"]).dt.strftime("%Y-%m-%d")
    wu_city = wu[(wu["city"] == city) & (wu["date"] < event_date)].copy()

    target_dt = date_cls.fromisoformat(event_date)
    d1 = (target_dt - timedelta(days=1)).isoformat()
    d2 = (target_dt - timedelta(days=2)).isoformat()

    def wu_on(d: str) -> float:
        row = wu_city[wu_city["date"] == d]
        return float(row["wunderground_tmax"].iloc[0]) if len(row) else np.nan

    def hrrr_on(d: str) -> float:
        row = hrrr[hrrr["date"] == d]
        return float(row["hrrr_tmax"].iloc[0]) if len(row) else np.nan

    hrrr_row = hrrr[hrrr["date"] == event_date].iloc[0]
    d3s = [(target_dt - timedelta(days=i)).isoformat() for i in range(1, 4)]
    d7s = [(target_dt - timedelta(days=i)).isoformat() for i in range(1, 8)]
    wu_d1, hrrr_d1 = wu_on(d1), hrrr_on(d1)

    meta = ng.STATION_META[city]
    doy = pd.Timestamp(event_date).dayofyear

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        asos = ng.load_temp_early_morning(city, target_dt, target_dt)
        om = ng.load_openmeteo_tmax(city, target_dt, target_dt)
        if om.empty:
            om = ng.fetch_openmeteo_tmax(city, meta, target_dt, target_dt)

    if asos.empty or om.empty:
        return None

    required_lags = [wu_on(d1), wu_on(d2)]
    if any(not np.isfinite(v) for v in required_lags):
        return None

    feat = {
        "hrrr_tmax": float(hrrr_row["hrrr_tmax"]),
        "peak_cloud_cover": float(hrrr_row["peak_cloud_cover"]),
        "peak_solar_flux": float(hrrr_row["peak_solar_flux"]),
        "snow_depth": float(hrrr_row["snow_depth"]),
        "temp_early_morning": float(asos["temp_early_morning"].iloc[0]),
        "nwp_tmax_openmeteo": float(om["nwp_tmax_openmeteo"].iloc[0]),
        "tmax_lag1": wu_on(d1),
        "tmax_lag2": wu_on(d2),
        "tmax_roll3": float(np.nanmean([wu_on(d) for d in d3s])),
        "tmax_roll7": float(np.nanmean([wu_on(d) for d in d7s])),
        "hrrr_error_lag1": wu_d1 - hrrr_d1 if np.isfinite(wu_d1) and np.isfinite(hrrr_d1) else np.nan,
        "station_id": ng.STATION_ID_MAP[city],
        "latitude": float(meta["lat"]),
        "elevation": float(meta["elevation_ft"]),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
    }
    return pd.DataFrame([feat])


def _two_piece_cdf(mu: float, sigma: float, ratio: float, x: float) -> float:
    """Two-piece normal CDF with mode mu, left scale r*sigma, right scale sigma."""
    if ratio < 1.0:
        raise ValueError(f"two_piece ratio must be >= 1, got {ratio}")
    if ratio == 1.0:
        return float(norm.cdf(x, loc=mu, scale=sigma))

    s1 = ratio * sigma
    s2 = sigma
    denom = s1 + s2
    w_left = 2.0 * s1 / denom
    w_right = 2.0 * s2 / denom
    mass_below_mu = s1 / denom

    if x <= mu:
        return w_left * float(norm.cdf((x - mu) / s1))
    return mass_below_mu + w_right * (float(norm.cdf((x - mu) / s2)) - 0.5)


def two_piece_ratio_for_date(config: dict[str, Any], event_date: str) -> float | None:
    """Return configured down-side sigma ratio for summer months, else None."""
    ratio = config.get("two_piece_sigma_down_ratio")
    if ratio is None:
        return None
    months = config.get("two_piece_months", [6, 7, 8])
    month = int(str(event_date)[5:7])
    if month in months:
        return float(ratio)
    return None


def describe_two_piece_mode(config: dict[str, Any]) -> str:
    ratio = config.get("two_piece_sigma_down_ratio")
    months = config.get("two_piece_months", [6, 7, 8])
    if ratio is None:
        return "bucket distribution: symmetric Gaussian"
    return f"bucket distribution: two-piece Gaussian (ratio_down={ratio}, months={months})"


def _cdf(
    mu: float,
    sigma: float,
    x: float,
    distribution: str,
    df_val: float | None,
    ratio_down: float | None = None,
) -> float:
    if distribution == "two_piece_gaussian":
        if df_val is not None:
            raise ValueError("two_piece_gaussian requires df_val=None")
        if ratio_down is None:
            raise ValueError("two_piece_gaussian requires ratio_down")
        return _two_piece_cdf(mu, sigma, ratio_down, x)
    if ratio_down is not None and ratio_down != 1.0:
        if distribution == "student_t":
            raise ValueError("two_piece ratio_down is not supported with student_t")
        return _two_piece_cdf(mu, sigma, ratio_down, x)
    if distribution == "student_t" and df_val is not None:
        return float(student_t.cdf(x, df=df_val, loc=mu, scale=sigma))
    return float(norm.cdf(x, loc=mu, scale=sigma))


def market_bucket_probability(
    label: str,
    mu: float,
    sigma: float,
    distribution: str = "gaussian",
    df_val: float | None = None,
    ratio_down: float | None = None,
) -> float:
    """Probability mass for a Polymarket bucket label under the model distribution."""
    parsed = parse_snapshot_bucket(str(label))
    btype = parsed["type"]
    if btype == "LESS_THAN":
        upper = int(parsed["upper"])
        return _cdf(mu, sigma, upper + 0.5, distribution, df_val, ratio_down)
    if btype == "GREATER_THAN":
        lower = int(parsed["lower"])
        return 1.0 - _cdf(mu, sigma, lower - 0.5, distribution, df_val, ratio_down)
    if btype == "RANGE":
        lo, hi = int(parsed["lower"]), int(parsed["upper"])
        return (
            _cdf(mu, sigma, hi + 0.5, distribution, df_val, ratio_down)
            - _cdf(mu, sigma, lo - 0.5, distribution, df_val, ratio_down)
        )
    return 0.0


def predict_mu_sigma(
    models: NgBoostBacktestModels,
    city: str,
    event_date: str,
) -> tuple[float, float] | None:
    """Return calibrated (mu, sigma) for one city-date."""
    feat_df = build_backtest_features(city, event_date)
    if feat_df is None:
        return None

    feat_df = ng.apply_saved_median_fill(feat_df, models.fill_medians, list(models.fill_medians.keys()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        feat_df["lgb_tmax_pred"] = float(models.lgb_model.predict(feat_df[models.stage1_cols])[0])

    X = ng.transform_features(models.scaler, feat_df, models.feature_cols)
    mu, sigma, _df_vals = ng.predict_dist_params(models.model, X)
    mu_f = float(mu[0])
    sigma_f = float(ng.apply_sigma_calibration(sigma, models.sigma_k)[0])
    return mu_f, sigma_f


def predict_bucket_probs_from_mu(
    models: NgBoostBacktestModels,
    city: str,
    event_date: str,
    bucket_labels: list[str],
    mu: float,
    ratio_down: float | None = None,
) -> dict[str, float] | None:
    """Bucket probabilities using a supplied mu (e.g. bias-adjusted)."""
    mu_sigma = predict_mu_sigma(models, city, event_date)
    if mu_sigma is None:
        return None
    _raw_mu, sigma_f = mu_sigma

    feat_df = build_backtest_features(city, event_date)
    if feat_df is None:
        return None
    feat_df = ng.apply_saved_median_fill(feat_df, models.fill_medians, list(models.fill_medians.keys()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        feat_df["lgb_tmax_pred"] = float(models.lgb_model.predict(feat_df[models.stage1_cols])[0])
    X = ng.transform_features(models.scaler, feat_df, models.feature_cols)
    _mu, _sigma, df_vals = ng.predict_dist_params(models.model, X)
    df_f = float(df_vals[0]) if df_vals is not None else None

    probs = {
        label: market_bucket_probability(
            label, mu, sigma_f, models.distribution, df_f, ratio_down=ratio_down
        )
        for label in bucket_labels
    }
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def predict_bucket_probs(
    models: NgBoostBacktestModels,
    city: str,
    event_date: str,
    bucket_labels: list[str],
    ratio_down: float | None = None,
) -> dict[str, float] | None:
    mu_sigma = predict_mu_sigma(models, city, event_date)
    if mu_sigma is None:
        return None
    mu_f, sigma_f = mu_sigma

    feat_df = build_backtest_features(city, event_date)
    if feat_df is None:
        return None
    feat_df = ng.apply_saved_median_fill(feat_df, models.fill_medians, list(models.fill_medians.keys()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        feat_df["lgb_tmax_pred"] = float(models.lgb_model.predict(feat_df[models.stage1_cols])[0])
    X = ng.transform_features(models.scaler, feat_df, models.feature_cols)
    _mu, _sigma, df_vals = ng.predict_dist_params(models.model, X)
    df_f = float(df_vals[0]) if df_vals is not None else None

    probs = {
        label: market_bucket_probability(
            label, mu_f, sigma_f, models.distribution, df_f, ratio_down=ratio_down
        )
        for label in bucket_labels
    }
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs
