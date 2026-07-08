"""Live NGBoost forecast for autotrader. Replaces Track-B fetch_forecast."""

from __future__ import annotations

import csv as csv_mod
import io
import json
import sys
import time
import warnings
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import train_ngboost as ng  # noqa: E402
except ImportError as exc:
    raise ImportError(
        "train_ngboost could not be imported; ensure scripts/ is on PYTHONPATH "
        "and ngboost/lightgbm dependencies are installed."
    ) from exc

ICAO_MAP = {
    "atlanta": "KATL",
    "austin": "KAUS",
    "chicago": "KORD",
    "dallas": "KDAL",
    "houston": "KHOU",
    "los_angeles": "KLAX",
    "miami": "KMIA",
    "new_york": "KLGA",
    "san_francisco": "KSFO",
    "seattle": "KSEA",
}


class NgBoostLiveModels:
    """Load NGBoost artifacts for live prediction."""

    def __init__(self, model_dir: Path | None = None) -> None:
        if model_dir is None:
            model_dir = PROJECT_ROOT / "models" / "ngboost_v2"
        config_path = model_dir / "model_config.json"
        with open(config_path, encoding="utf-8") as handle:
            self.config = json.load(handle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(model_dir / "ngboost_global.pkl")
            self.scaler = joblib.load(model_dir / "feature_scaler.pkl")
            self.lgb_model = joblib.load(
                model_dir / self.config.get("stage1_model", "lgb_stage1.pkl")
            )
        self.feature_cols = list(self.config["feature_columns"])
        self.stage1_cols = [c for c in self.feature_cols if c != "lgb_tmax_pred"]
        self.fill_medians = dict(self.config.get("nan_fill_medians", {}))
        self.sigma_k = float(self.config.get("sigma_calibration_k", 1.0))
        self.cities = list(self.config.get("cities", []))


def ensure_hrrr(city: str, event_date: str) -> bool:
    """Fetch HRRR for event_date if not cached. Returns True if available."""
    from fetch_hrrr_all_cities import (  # noqa: WPS433
        HRRR_STATIONS,
        _init_download_pool,
        _shutdown_download_pool,
        fetch_hrrr_for_date,
        load_monthly_cache,
        monthly_cache_path,
        write_monthly_row,
    )

    if city not in HRRR_STATIONS:
        return False

    target = date_cls.fromisoformat(event_date)
    today = date_cls.today()
    if target < today - timedelta(days=1) or target > today + timedelta(days=14):
        return False

    hrrr = ng.load_hrrr_city(city)
    cached = set(pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d"))
    if event_date in cached:
        return True

    try:
        _init_download_pool(8)
        try:
            row = fetch_hrrr_for_date(HRRR_STATIONS[city], target)
        finally:
            _shutdown_download_pool()
        tmax = row.get("hrrr_tmax")
        if tmax is None or (isinstance(tmax, float) and np.isnan(tmax)):
            return False
        path = monthly_cache_path(city, target)
        cache = load_monthly_cache(path)
        write_monthly_row(cache, path, row)
        return True
    except Exception as exc:
        print(f"  HRRR fetch failed for {city}/{event_date}: {exc}")
        return False


def ensure_wu_current(city: str, event_date: str) -> bool:
    """Ensure WU targets include lag dates needed through event_date - 1."""
    wu = ng._load_wu_all()
    wu_city = wu[wu["city"] == city].copy()
    wu_city["date_str"] = pd.to_datetime(wu_city["date"]).dt.strftime("%Y-%m-%d")
    existing = set(wu_city["date_str"])

    target = date_cls.fromisoformat(event_date)
    needed_through = target - timedelta(days=1)
    needed_dates = {
        (target - timedelta(days=i)).isoformat() for i in range(1, 8)
    }
    needed_dates.add(needed_through.isoformat())

    missing_dates = sorted(d for d in needed_dates if d not in existing)
    if not missing_dates:
        return True

    icao = ICAO_MAP.get(city)
    if not icao:
        return False

    new_rows: list[dict[str, Any]] = []
    for current in missing_dates:
        current_dt = date_cls.fromisoformat(current)
        next_day = current_dt + timedelta(days=1)
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={icao}&data=tmpf&tz=UTC&format=onlycomma"
            f"&year1={current_dt.year}&month1={current_dt.month}&day1={current_dt.day}"
            f"&year2={next_day.year}&month2={next_day.month}&day2={next_day.day}"
        )
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            reader = csv_mod.DictReader(io.StringIO(response.text))
            temps: list[float] = []
            for row in reader:
                val = row.get("tmpf", "M").strip()
                if val not in ("M", ""):
                    try:
                        temps.append(float(val))
                    except ValueError:
                        continue
            if temps:
                tmax = round(max(temps))
                new_rows.append(
                    {
                        "city": city,
                        "date": current,
                        "wunderground_tmax": tmax,
                        "reliable": True,
                    }
                )
                print(f"  WU backfill: {city} {current} -> {tmax}F")
        except Exception as exc:
            print(f"  WU fetch failed: {city} {current}: {exc}")
        time.sleep(1.0)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        existing_df = pd.read_parquet(ng.WU_PATH)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["city", "date"], keep="last")
        combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
        combined.to_parquet(ng.WU_PATH, index=False)
        ng._WU_CACHE = None
        print(f"  WU targets updated: {len(new_rows)} rows appended")

    return True


def build_live_features(city: str, event_date: str) -> pd.DataFrame | None:
    """Build feature vector for live prediction."""
    if city not in ng.STATION_META:
        return None

    if not ensure_hrrr(city, event_date):
        print(f"  {city}: HRRR unavailable for {event_date}")
        return None

    if not ensure_wu_current(city, event_date):
        print(f"  {city}: WU history incomplete")

    hrrr = ng.load_hrrr_city(city)
    hrrr["date"] = pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d")
    if event_date not in set(hrrr["date"]):
        print(f"  {city}: no HRRR row for {event_date}")
        return None

    wu = ng._load_wu_all()
    wu = wu[wu["reliable"].astype(bool)].copy()
    wu["date"] = pd.to_datetime(wu["date"]).dt.strftime("%Y-%m-%d")
    wu_city = wu[wu["city"] == city]

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

    required_lags = [wu_on(d1), wu_on(d2)]
    if any(not np.isfinite(v) for v in required_lags):
        print(f"  {city}: missing WU lags for {d1} or {d2}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        asos = ng.load_temp_early_morning(city, target_dt, target_dt)
        om = ng.load_openmeteo_tmax(city, target_dt, target_dt)
        if om.empty:
            om = ng.fetch_openmeteo_tmax(city, meta, target_dt, target_dt)

    if asos.empty:
        print(f"  {city}: no ASOS early morning data")
        return None
    if om.empty:
        print(f"  {city}: no OpenMeteo data")
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
        "hrrr_error_lag1": (
            wu_d1 - hrrr_d1 if np.isfinite(wu_d1) and np.isfinite(hrrr_d1) else np.nan
        ),
        "station_id": ng.STATION_ID_MAP[city],
        "latitude": float(meta["lat"]),
        "elevation": float(meta["elevation_ft"]),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
    }
    return pd.DataFrame([feat])


def predict_ngboost_from_features(
    models: NgBoostLiveModels,
    feat_df: pd.DataFrame,
) -> tuple[float, float]:
    """Return calibrated (mu, sigma) from a pre-built feature row."""
    feat_df = ng.apply_saved_median_fill(
        feat_df, models.fill_medians, list(models.fill_medians.keys())
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        feat_df["lgb_tmax_pred"] = float(
            models.lgb_model.predict(feat_df[models.stage1_cols])[0]
        )

    X = ng.transform_features(models.scaler, feat_df, models.feature_cols)
    mu, sigma, _df_vals = ng.predict_dist_params(models.model, X)
    mu_f = float(mu[0])
    sigma_f = float(ng.apply_sigma_calibration(sigma, models.sigma_k)[0])
    return mu_f, sigma_f


def predict_ngboost(
    models: NgBoostLiveModels,
    city: str,
    event_date: str,
) -> tuple[float, float] | None:
    """Return calibrated (mu, sigma) for one city-date using live data."""
    feat_df = build_live_features(city, event_date)
    if feat_df is None:
        return None
    return predict_ngboost_from_features(models, feat_df)


def ngboost_bucket_probs(
    mu: float,
    sigma: float,
    bucket_labels: list[str],
    ratio_down: float | None = None,
) -> dict[str, float]:
    """Bucket probabilities from NGBoost (mu, sigma) using Gaussian or two-piece CDF."""
    from backtest.ngboost_inference import market_bucket_probability  # noqa: WPS433

    probs: dict[str, float] = {}
    for label in bucket_labels:
        try:
            probs[label] = market_bucket_probability(
                label, mu, sigma, "gaussian", None, ratio_down=ratio_down
            )
        except ValueError:
            continue

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs
