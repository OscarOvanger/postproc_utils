"""Train NGBoost distributional regression on HRRR v2 + Wunderground Tmax targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, time as dt_time, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy.optimize import brentq
from scipy.stats import kstest, norm, probplot, t as student_t
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from urllib3.util.retry import Retry

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None  # type: ignore[misc, assignment]

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import CRPScore, LogScore
except ImportError:
    print("ngboost is not installed. Install with:\n  pip install ngboost")
    sys.exit(1)

try:
    from ngboost.distns import T as TDist

    HAS_TDIST = True
except ImportError:
    try:
        from ngboost.distns import T_uncensored as TDist  # type: ignore[attr-defined]

        HAS_TDIST = True
    except ImportError:
        HAS_TDIST = False
        TDist = None  # type: ignore[misc, assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_openmeteo_nwp import (  # noqa: E402
    DEFAULT_OUTPUT_PATH as OPENMETEO_PARQUET_PATH,
    fetch_openmeteo_tmax as _fetch_openmeteo_tmax_api,
    make_session as openmeteo_session,
)

HRRR_ROOT = PROJECT_ROOT / "data" / "hrrr_v2"
WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
ASOS_CACHE_DIR = PROJECT_ROOT / "data" / "asos_cache"
OPENMETEO_CACHE_DIR = PROJECT_ROOT / "data" / "openmeteo_cache"
OPENMETEO_MODELS = ("ecmwf_ifs025", "gfs_seamless")
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ASOS_FETCH_SLEEP_SECONDS = 1.1

DEFAULT_CITIES = ["houston", "los_angeles", "austin", "dallas", "chicago"]
TARGET = "wunderground_tmax"

TRAIN_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-12-31")
TEST_START = pd.Timestamp("2026-01-01")

STATION_META: dict[str, dict[str, float | str]] = {
    "houston": {"station": "KHOU", "lat": 29.65, "lon": -95.28, "elevation_ft": 46, "tz": "America/Chicago"},
    "los_angeles": {"station": "KLAX", "lat": 33.94, "lon": -118.41, "elevation_ft": 126, "tz": "America/Los_Angeles"},
    "austin": {"station": "KAUS", "lat": 30.20, "lon": -97.67, "elevation_ft": 542, "tz": "America/Chicago"},
    "dallas": {"station": "KDAL", "lat": 32.85, "lon": -96.85, "elevation_ft": 487, "tz": "America/Chicago"},
    "chicago": {"station": "KORD", "lat": 41.97, "lon": -87.91, "elevation_ft": 672, "tz": "America/Chicago"},
    "san_francisco": {"station": "KSFO", "lat": 37.62, "lon": -122.38, "elevation_ft": 13, "tz": "America/Los_Angeles"},
    "seattle": {"station": "KSEA", "lat": 47.45, "lon": -122.31, "elevation_ft": 433, "tz": "America/Los_Angeles"},
    "new_york": {"station": "KLGA", "lat": 40.78, "lon": -73.87, "elevation_ft": 22, "tz": "America/New_York"},
    "miami": {"station": "KMIA", "lat": 25.79, "lon": -80.29, "elevation_ft": 8, "tz": "America/New_York"},
    "atlanta": {"station": "KATL", "lat": 33.64, "lon": -84.43, "elevation_ft": 1026, "tz": "America/New_York"},
}

STATION_ID_MAP: dict[str, int] = {
    "houston": 0,
    "los_angeles": 1,
    "austin": 2,
    "dallas": 3,
    "chicago": 4,
    "san_francisco": 5,
    "seattle": 6,
    "new_york": 7,
    "miami": 8,
    "atlanta": 9,
}

FEATURE_COLS_GLOBAL: list[str] = [
    "hrrr_tmax",
    "peak_cloud_cover",
    "peak_solar_flux",
    "snow_depth",
    "temp_early_morning",
    "nwp_tmax_openmeteo",
    "tmax_lag1",
    "tmax_lag2",
    "tmax_roll3",
    "tmax_roll7",
    "hrrr_error_lag1",
    "station_id",
    "latitude",
    "elevation",
    "doy_sin",
    "doy_cos",
    "lgb_tmax_pred",
]

FEATURE_COLS_STAGE1: list[str] = [c for c in FEATURE_COLS_GLOBAL if c != "lgb_tmax_pred"]
FEATURE_COLS_PER_CITY: list[str] = [c for c in FEATURE_COLS_GLOBAL if c != "station_id"]

LAG_COLS = ["tmax_lag1", "tmax_lag2", "tmax_roll3", "tmax_roll7", "hrrr_error_lag1"]
MEDIAN_FILL_COLS = ["temp_early_morning", "nwp_tmax_openmeteo"]

COVERAGE_LEVELS = [
    (50, 0.6745),
    (80, 1.2816),
    (90, 1.6449),
    (95, 1.96),
]

BUCKET_EDGES = list(range(20, 121, 2))

PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": 3, "learning_rate": 0.01, "minibatch_frac": 1.0, "label": "d3_lr01"},
    {"max_depth": 4, "learning_rate": 0.01, "minibatch_frac": 1.0, "label": "d4_lr01"},
    {"max_depth": 3, "learning_rate": 0.01, "minibatch_frac": 0.8, "label": "d3_lr01_mb08"},
    {"max_depth": 4, "learning_rate": 0.01, "minibatch_frac": 0.8, "label": "d4_lr01_mb08"},
    {"max_depth": 3, "learning_rate": 0.02, "minibatch_frac": 0.8, "label": "d3_lr02_mb08"},
]

DEFAULT_HPARAMS: dict[str, Any] = {
    "max_depth": 4,
    "learning_rate": 0.01,
    "minibatch_frac": 1.0,
    "label": "d4_lr01",
}

GRID_N_ESTIMATORS = 800
FINAL_N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 50
T_CRPS_SAMPLES = 100

_WU_CACHE: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NGBoost on HRRR v2 + Wunderground targets.")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=DEFAULT_CITIES,
        help="Cities to include (default: 5 ready cities)",
    )
    parser.add_argument("--output-dir", default="models/ngboost", help="Directory for model artifacts")
    parser.add_argument(
        "--report-dir",
        default="reports/ngboost_calibration",
        help="Directory for calibration plots",
    )
    parser.add_argument(
        "--skip-per-city",
        action="store_true",
        help="Skip per-city model training (saves time)",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip hyperparameter grid and reuse saved/default hyperparameters",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Load saved model and print per-city TEST set metrics (2026+); no training",
    )
    return parser.parse_args()


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def _load_wu_all() -> pd.DataFrame:
    global _WU_CACHE
    if _WU_CACHE is None:
        if not WU_PATH.exists():
            print(f"Missing Wunderground targets: {WU_PATH}")
            sys.exit(1)
        _WU_CACHE = _normalize_date(pd.read_parquet(WU_PATH))
    return _WU_CACHE


def load_hrrr_city(city: str) -> pd.DataFrame:
    city_dir = HRRR_ROOT / city
    if not city_dir.exists():
        print(f"No HRRR data directory for {city}: {city_dir}")
        sys.exit(1)
    paths = sorted(city_dir.glob(f"hrrr_{city}_*.csv"))
    if not paths:
        print(f"No HRRR CSV files for {city} in {city_dir}")
        sys.exit(1)
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = _normalize_date(df)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


def load_wu_city(city: str) -> pd.DataFrame:
    wu = _load_wu_all()
    return wu[wu["city"].astype(str).eq(city)].copy()


def _make_asos_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MCP_Project/1.0 (research)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _asos_cache_path(city: str) -> Path:
    return ASOS_CACHE_DIR / f"{city}_temp_early_morning.csv"


def _cache_covers_range(cache_path: Path, start_date: date, end_date: date) -> bool:
    if not cache_path.exists():
        return False
    cached = pd.read_csv(cache_path)
    if cached.empty or "date" not in cached.columns:
        return False
    dates = pd.to_datetime(cached["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    return dates.min().date() <= start_date and dates.max().date() >= end_date


def _daily_temp_early_morning(obs: pd.DataFrame) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame(columns=["date", "temp_early_morning"])

    df = obs.copy()
    df["valid_dt"] = pd.to_datetime(df["valid"], errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df[df["valid_dt"].notna()].copy()

    rows: list[dict[str, float | str]] = []
    for date_str, group in df.groupby(df["valid_dt"].dt.strftime("%Y-%m-%d")):
        window = group[
            (group["valid_dt"].dt.time >= dt_time(9, 0))
            & (group["valid_dt"].dt.time <= dt_time(10, 30))
        ]
        if window.empty:
            rows.append({"date": date_str, "temp_early_morning": np.nan})
            continue
        before_10 = window[window["valid_dt"].dt.time <= dt_time(10, 0)]
        if before_10.empty:
            rows.append({"date": date_str, "temp_early_morning": np.nan})
            continue
        best_row = before_10.sort_values("valid_dt").iloc[-1]
        rows.append({"date": date_str, "temp_early_morning": float(best_row["tmpf"])})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "temp_early_morning"])
    out["date"] = out["date"].astype(str)
    return out.sort_values("date").reset_index(drop=True)


def fetch_asos_temp_early_morning(
    station: str,
    start_date: date,
    end_date: date,
    *,
    city: str,
    timezone: str,
) -> pd.DataFrame:
    """Fetch morning ASOS temperature from IEM and cache per city."""
    cache_path = _asos_cache_path(city)
    if _cache_covers_range(cache_path, start_date, end_date):
        cached = pd.read_csv(cache_path)
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        mask = (pd.to_datetime(cached["date"]) >= pd.Timestamp(start_date)) & (
            pd.to_datetime(cached["date"]) <= pd.Timestamp(end_date)
        )
        print(f"  ASOS {city}: loaded cache {cache_path.name} ({int(mask.sum())} rows in range)")
        return cached.loc[mask, ["date", "temp_early_morning"]].reset_index(drop=True)

    end_exclusive = end_date + timedelta(days=1)
    params = {
        "station": station,
        "data": "tmpf",
        "tz": timezone,
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "year1": str(start_date.year),
        "month1": str(start_date.month),
        "day1": str(start_date.day),
        "year2": str(end_exclusive.year),
        "month2": str(end_exclusive.month),
        "day2": str(end_exclusive.day),
        "report_type": "3",
    }

    session = _make_asos_session()
    response = session.get(IEM_ASOS_URL, params=params, timeout=30)
    response.raise_for_status()
    text = response.text
    if not text.strip() or text.startswith("ERROR"):
        raise RuntimeError(f"IEM ASOS fetch failed for {station}: {text[:200]}")

    obs = pd.read_csv(StringIO(text), na_values=["M", "null", ""], keep_default_na=True)
    daily = _daily_temp_early_morning(obs)

    all_dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    daily = pd.DataFrame({"date": all_dates}).merge(daily, on="date", how="left")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache_path, index=False)
    print(f"  ASOS {city}: fetched {len(obs)} obs -> {len(daily)} daily rows -> {cache_path.name}")
    time.sleep(ASOS_FETCH_SLEEP_SECONDS)
    return daily


def load_temp_early_morning(city: str, start_date: date, end_date: date) -> pd.DataFrame:
    meta = STATION_META[city]
    return fetch_asos_temp_early_morning(
        station=str(meta["station"]),
        start_date=start_date,
        end_date=end_date,
        city=city,
        timezone=str(meta["tz"]),
    )


def _openmeteo_cache_path(city: str) -> Path:
    return OPENMETEO_CACHE_DIR / f"{city}_tmax.csv"


def _merge_openmeteo_model_frames(ecmwf_df: pd.DataFrame, gfs_df: pd.DataFrame) -> pd.DataFrame:
    ecmwf = ecmwf_df.rename(columns={"nwp_tmax_forecast_f": "ecmwf_tmax"})[["date", "ecmwf_tmax"]]
    gfs = gfs_df.rename(columns={"nwp_tmax_forecast_f": "gfs_tmax"})[["date", "gfs_tmax"]]
    merged = ecmwf.merge(gfs, on="date", how="outer")
    ecmwf_vals = pd.to_numeric(merged["ecmwf_tmax"], errors="coerce")
    gfs_vals = pd.to_numeric(merged["gfs_tmax"], errors="coerce")
    merged["nwp_tmax_openmeteo"] = ecmwf_vals.where(ecmwf_vals.notna(), gfs_vals)
    return merged[["date", "nwp_tmax_openmeteo"]].sort_values("date").reset_index(drop=True)


def _load_openmeteo_parquet(city: str, start_date: date, end_date: date) -> pd.DataFrame | None:
    path = PROJECT_ROOT / OPENMETEO_PARQUET_PATH
    if not path.exists():
        return None
    raw = pd.read_parquet(path)
    if raw.empty or "city" not in raw.columns:
        return None

    def _city_model_frame(model: str) -> pd.DataFrame:
        city_rows = raw[
            raw["city"].astype(str).eq(city) & raw["model_used"].astype(str).eq(model)
        ].copy()
        if city_rows.empty:
            return pd.DataFrame(columns=["date", "nwp_tmax_forecast_f"])
        city_rows["date"] = pd.to_datetime(city_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        city_rows["nwp_tmax_forecast_f"] = pd.to_numeric(
            city_rows.get("nwp_tmax_forecast_f"), errors="coerce"
        )
        return city_rows[["date", "nwp_tmax_forecast_f"]]

    merged = _merge_openmeteo_model_frames(
        _city_model_frame("ecmwf_ifs025"),
        _city_model_frame("gfs_seamless"),
    )
    if merged.empty:
        return None
    mask = (pd.to_datetime(merged["date"]) >= pd.Timestamp(start_date)) & (
        pd.to_datetime(merged["date"]) <= pd.Timestamp(end_date)
    )
    subset = merged.loc[mask, ["date", "nwp_tmax_openmeteo"]].reset_index(drop=True)
    return subset if not subset.empty else None


def fetch_openmeteo_tmax(
    city: str,
    station_meta: dict[str, float | str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch daily Tmax forecasts from Open-Meteo (ECMWF-first, GFS fallback)."""
    lat = float(station_meta["lat"])
    lon = float(station_meta["lon"])
    session = openmeteo_session()
    ecmwf = _fetch_openmeteo_tmax_api(
        lat, lon, start_date, end_date, "ecmwf_ifs025", session=session, sleep_seconds=0.5
    )
    gfs = _fetch_openmeteo_tmax_api(
        lat, lon, start_date, end_date, "gfs_seamless", session=session, sleep_seconds=0.5
    )
    return _merge_openmeteo_model_frames(ecmwf, gfs)


def load_openmeteo_tmax(city: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load Open-Meteo Tmax from per-city CSV cache, parquet, or API."""
    cache_path = _openmeteo_cache_path(city)
    if _cache_covers_range(cache_path, start_date, end_date):
        cached = pd.read_csv(cache_path)
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        mask = (pd.to_datetime(cached["date"]) >= pd.Timestamp(start_date)) & (
            pd.to_datetime(cached["date"]) <= pd.Timestamp(end_date)
        )
        print(f"  Open-Meteo {city}: loaded cache {cache_path.name} ({int(mask.sum())} rows in range)")
        return cached.loc[mask, ["date", "nwp_tmax_openmeteo"]].reset_index(drop=True)

    parquet_frame = _load_openmeteo_parquet(city, start_date, end_date)
    if parquet_frame is not None:
        dates = pd.to_datetime(parquet_frame["date"], errors="coerce")
        if dates.min().date() <= start_date and dates.max().date() >= end_date:
            print(f"  Open-Meteo {city}: loaded from {OPENMETEO_PARQUET_PATH.name}")
            new_rows = parquet_frame
        else:
            new_rows = fetch_openmeteo_tmax(city, STATION_META[city], start_date, end_date)
    else:
        print(f"  Open-Meteo {city}: fetching {start_date}..{end_date}")
        new_rows = fetch_openmeteo_tmax(city, STATION_META[city], start_date, end_date)

    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    print(f"  Open-Meteo {city}: saved cache {cache_path.name} ({len(combined)} rows)")

    mask = (pd.to_datetime(combined["date"]) >= pd.Timestamp(start_date)) & (
        pd.to_datetime(combined["date"]) <= pd.Timestamp(end_date)
    )
    return combined.loc[mask, ["date", "nwp_tmax_openmeteo"]].reset_index(drop=True)


def build_city_features(city: str) -> pd.DataFrame:
    if city not in STATION_META:
        print(f"Unknown city: {city}. Must be one of: {sorted(STATION_META)}")
        sys.exit(1)

    hrrr = load_hrrr_city(city)
    wu = load_wu_city(city)
    merged = hrrr.merge(wu, on="date", how="inner", suffixes=("_hrrr", "_wu"))
    merged = merged[merged["reliable"].astype(bool)].copy()
    merged = merged.sort_values("date").reset_index(drop=True)

    dates = pd.to_datetime(merged["date"], errors="coerce")
    start_date = dates.min().date()
    end_date = dates.max().date()
    asos = load_temp_early_morning(city, start_date, end_date)
    merged = merged.merge(asos, on="date", how="left")
    om = load_openmeteo_tmax(city, start_date, end_date)
    merged = merged.merge(om, on="date", how="left")

    tmax = pd.to_numeric(merged[TARGET], errors="coerce")
    merged["tmax_lag1"] = tmax.shift(1)
    merged["tmax_lag2"] = tmax.shift(2)
    merged["tmax_roll3"] = tmax.rolling(3, min_periods=3).mean().shift(1)
    merged["tmax_roll7"] = tmax.rolling(7, min_periods=7).mean().shift(1)
    error = tmax - pd.to_numeric(merged["hrrr_tmax"], errors="coerce")
    merged["hrrr_error_lag1"] = error.shift(1)

    meta = STATION_META[city]
    doy_dates = pd.to_datetime(merged["date"])
    doy = doy_dates.dt.dayofyear
    merged["latitude"] = float(meta["lat"])
    merged["elevation"] = float(meta["elevation_ft"])
    merged["station_id"] = STATION_ID_MAP[city]
    merged["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    merged["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    merged["city"] = city
    return merged


def assemble_dataset(cities: list[str]) -> pd.DataFrame:
    frames = []
    for city in cities:
        print(f"Building features for {city}...")
        city_df = build_city_features(city)
        print(f"  {city}: {len(city_df)} rows after HRRR+WU merge (reliable only)")
        frames.append(city_df)
    return pd.concat(frames, ignore_index=True)


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = [TARGET, *LAG_COLS]
    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    before = len(out)
    out = out.dropna(subset=required)
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} rows with NaN in target or lag features")
    return out.reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    train_df = df[dates <= TRAIN_END].copy()
    val_df = df[(dates >= VAL_START) & (dates <= VAL_END)].copy()
    test_df = df[dates >= TEST_START].copy()

    print("\n=== Split row counts per city ===")
    header = f"{'city':<16} {'train':>7} {'val':>7} {'test':>7}"
    print(header)
    print("-" * len(header))
    for city in sorted(df["city"].unique()):
        c_train = int(train_df["city"].eq(city).sum())
        c_val = int(val_df["city"].eq(city).sum())
        c_test = int(test_df["city"].eq(city).sum())
        print(f"{city:<16} {c_train:7d} {c_val:7d} {c_test:7d}")
    print(
        f"{'TOTAL':<16} {len(train_df):7d} {len(val_df):7d} {len(test_df):7d}"
    )
    return train_df, val_df, test_df


def fill_median_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    medians: dict[str, float] = {}
    for col in cols:
        median_val = float(np.nanmedian(pd.to_numeric(train_df[col], errors="coerce")))
        if not np.isfinite(median_val):
            median_val = 0.0
            print(f"  WARNING: {col} has no finite train values; filling with 0.0")
        medians[col] = median_val

    filled: dict[str, pd.DataFrame] = {}
    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = frame.copy()
        for col in cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians[col])
        filled[name] = out
    return filled["train"], filled["val"], filled["test"], medians


def _build_lgb_stage1() -> Any:
    if not HAS_LIGHTGBM or lgb is None:
        print("lightgbm is not installed. Install with:\n  pip install lightgbm")
        sys.exit(1)
    return lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="mae",
        random_state=42,
        verbose=-1,
    )


def train_stage1_lgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cities: list[str],
    output_dir: Path,
) -> tuple[Any, float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train LightGBM point forecast and append lgb_tmax_pred to all splits."""
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET]
    y_val = val_df[TARGET]

    print("\n=== Stage 1: LightGBM Point Forecast ===")
    cv_estimator = _build_lgb_stage1()
    lgb_pred_train_cv = cross_val_predict(
        cv_estimator, X_train, y_train, cv=5, method="predict"
    )

    lgb_model = _build_lgb_stage1()
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    lgb_pred_val = lgb_model.predict(X_val)
    lgb_pred_test = lgb_model.predict(X_test)

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    train_out["lgb_tmax_pred"] = lgb_pred_train_cv
    val_out["lgb_tmax_pred"] = lgb_pred_val
    test_out["lgb_tmax_pred"] = lgb_pred_test

    y_val_arr = y_val.to_numpy(dtype=float)
    val_mae = float(np.mean(np.abs(lgb_pred_val - y_val_arr)))
    val_rmse = float(np.sqrt(np.mean((lgb_pred_val - y_val_arr) ** 2)))
    print(f"Val MAE: {val_mae:.2f} F")
    print(f"Val RMSE: {val_rmse:.2f} F")

    header = f"{'city':<16} {'MAE':>7}"
    print(header)
    print("-" * len(header))
    for city in cities:
        c_val = val_out[val_out["city"].eq(city)]
        if len(c_val) == 0:
            continue
        y_c = c_val[TARGET].to_numpy(dtype=float)
        pred_c = c_val["lgb_tmax_pred"].to_numpy(dtype=float)
        print(f"{city:<16} {float(np.mean(np.abs(pred_c - y_c))):7.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgb_model, output_dir / "lgb_stage1.pkl")

    return lgb_model, val_mae, val_rmse, train_out, val_out, test_out


def fit_feature_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    scaled = scaler.transform(df[feature_cols])
    return pd.DataFrame(scaled, columns=feature_cols, index=df.index)


def assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    train_dates = set(pd.to_datetime(train_df["date"]))
    val_dates = set(pd.to_datetime(val_df["date"]))
    test_dates = set(pd.to_datetime(test_df["date"]))

    assert not (test_dates & train_dates), "Test dates appear in train"
    assert not (test_dates & val_dates), "Test dates appear in val"
    assert not (val_dates & train_dates), "Val dates appear in train"
    assert TARGET not in feature_cols, "Target column appears in feature matrix"
    # Lag features verified by construction: shift(1)/shift(2) and rolling().shift(1)
    # hrrr_error_lag1 uses error.shift(1), not shift(0)
    print("Leakage assertions passed.")


def load_saved_hparams(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        saved = config.get("hyperparameters")
        if saved:
            return {
                "max_depth": int(saved["max_depth"]),
                "learning_rate": float(saved["learning_rate"]),
                "minibatch_frac": float(saved["minibatch_frac"]),
                "label": str(saved.get("label", "saved")),
            }
    return dict(DEFAULT_HPARAMS)


def build_ngboost_model(
    hparams: dict[str, Any],
    dist: type = Normal,
    n_estimators: int = FINAL_N_ESTIMATORS,
    verbose: bool = True,
) -> NGBRegressor:
    score = CRPScore if dist is Normal else LogScore
    use_t = HAS_TDIST and dist is TDist
    base_kwargs: dict[str, Any] = {"max_depth": int(hparams["max_depth"]), "random_state": 42}
    if use_t:
        base_kwargs["min_samples_leaf"] = 10
    base_learner = DecisionTreeRegressor(**base_kwargs)
    return NGBRegressor(
        Dist=dist,
        Score=score,
        Base=base_learner,
        n_estimators=n_estimators,
        learning_rate=float(hparams["learning_rate"]),
        minibatch_frac=float(hparams["minibatch_frac"]),
        natural_gradient=not use_t,
        verbose=verbose,
        random_state=42,
    )


def fit_ngboost_model(
    model: NGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[NGBRegressor, int]:
    if len(y_val) == 0:
        model.fit(X_train, y_train)
    else:
        try:
            model.fit(
                X_train,
                y_train,
                X_val=X_val,
                Y_val=y_val,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
        except TypeError:
            model.fit(
                X_train,
                y_train,
                val_data=(X_val, y_val),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
    best_n = getattr(model, "best_val_loss_itr", None)
    if best_n is None:
        best_n = model.n_estimators
    return model, int(best_n)


def distribution_name(dist: type | None) -> str:
    if dist is Normal:
        return "gaussian"
    if HAS_TDIST and dist is TDist:
        return "student_t"
    name = getattr(dist, "__name__", str(dist))
    if name in {"T", "T_uncensored"}:
        return "student_t"
    return "gaussian"


def model_distribution_name(model: NGBRegressor) -> str:
    return distribution_name(model.Dist)


def predict_dist_params(
    model: NGBRegressor,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dist = model.pred_dist(X)
    mu = np.asarray(dist.params["loc"], dtype=float)
    sigma = np.maximum(np.asarray(dist.params["scale"], dtype=float), 1e-8)
    df_vals: np.ndarray | None = None
    if model_distribution_name(model) == "student_t":
        if hasattr(dist, "df") and dist.df is not None:
            df_vals = np.maximum(np.asarray(dist.df, dtype=float), 1e-8)
        elif "df" in dist.params:
            df_vals = np.maximum(np.asarray(dist.params["df"], dtype=float), 1e-8)
    return mu, sigma, df_vals


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    z = (y_arr - mu_arr) / sigma_arr
    crps = sigma_arr * (
        z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def predict_mu_sigma(model: NGBRegressor, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma, _ = predict_dist_params(model, X)
    return mu, sigma


def apply_sigma_calibration(sigma: np.ndarray, sigma_k: float) -> np.ndarray:
    return np.maximum(np.asarray(sigma, dtype=float), 1e-8) * float(sigma_k)


def _sigma_cal_coverage_fraction(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: float,
    level: float = 0.90,
) -> float:
    z = norm.ppf((1.0 + level) / 2.0)
    lower = mu - z * k * sigma
    upper = mu + z * k * sigma
    return float(np.mean((y >= lower) & (y <= upper)))


def fit_sigma_calibration_k(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    level: float = 0.90,
    tol: float = 0.005,
) -> float:
    """Find scalar k in [1.0, 3.0] so empirical coverage at `level` matches nominal."""
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-8)

    def coverage_error(k: float) -> float:
        return _sigma_cal_coverage_fraction(y_arr, mu_arr, sigma_arr, k, level=level) - level

    lo, hi = 1.0, 3.0
    err_lo = coverage_error(lo)
    if abs(err_lo) <= tol:
        return lo
    err_hi = coverage_error(hi)
    if err_hi < 0:
        print(f"WARNING: sigma calibration k capped at {hi} (coverage still below {level:.0%})")
        return hi
    if err_lo > 0:
        return lo

    k = brentq(coverage_error, lo, hi, xtol=1e-4)
    return float(k)


def _compute_coverage_dict(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    distribution: str,
    df_vals: np.ndarray | None = None,
    sigma_k: float = 1.0,
) -> dict[int, float]:
    sigma_eff = apply_sigma_calibration(sigma, sigma_k)
    coverage: dict[int, float] = {}
    for nominal, z_alpha in COVERAGE_LEVELS:
        if distribution == "student_t":
            assert df_vals is not None
            t_crit = student_t.ppf((1.0 + nominal / 100.0) / 2.0, df=df_vals)
            lower = mu - t_crit * sigma_eff
            upper = mu + t_crit * sigma_eff
        else:
            lower = mu - z_alpha * sigma_eff
            upper = mu + z_alpha * sigma_eff
        coverage[nominal] = 100.0 * float(np.mean((y >= lower) & (y <= upper)))
    return coverage


def _print_coverage_table(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    title: str,
    distribution: str,
    df_vals: np.ndarray | None = None,
    sigma_k: float = 1.0,
) -> dict[int, float]:
    coverage = _compute_coverage_dict(y, mu, sigma, distribution, df_vals, sigma_k=sigma_k)
    print(f"\n{title}")
    print(f"{'Nominal':>8} {'Empirical':>10} {'Gap':>8}")
    for nominal in sorted(coverage):
        empirical = coverage[nominal]
        gap = empirical - nominal
        print(f"{nominal:>7}% {empirical:9.1f}% {gap:+7.1f}%")
    return coverage


def ensemble_crps(observation: float, forecasts: np.ndarray) -> float:
    forecasts = np.asarray(forecasts, dtype=float)
    term1 = np.mean(np.abs(forecasts - observation))
    term2 = 0.5 * np.mean(np.abs(forecasts[:, None] - forecasts[None, :]))
    return float(term1 - term2)


def student_t_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, df: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    df_arr = np.maximum(np.asarray(df, dtype=float), 1e-8)
    crps_vals = [
        ensemble_crps(
            yi,
            student_t.rvs(df=di, loc=mi, scale=si, size=T_CRPS_SAMPLES, random_state=42 + i),
        )
        for i, (yi, mi, si, di) in enumerate(zip(y_arr, mu_arr, sigma_arr, df_arr))
    ]
    return float(np.mean(crps_vals))


def eval_model_crps(model: NGBRegressor, X: pd.DataFrame, y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    mu, sigma, df_vals = predict_dist_params(model, X)
    if model_distribution_name(model) == "student_t":
        if df_vals is None:
            raise ValueError("Student-t model missing degrees of freedom")
        return student_t_crps(y_arr, mu, sigma, df_vals)
    return gaussian_crps(y_arr, mu, sigma)


def run_hyperparam_grid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], int, float, StandardScaler]:
    scaler = fit_feature_scaler(train_df[feature_cols])
    X_train = transform_features(scaler, train_df, feature_cols)
    X_val = transform_features(scaler, val_df, feature_cols)
    y_train = train_df[TARGET]
    y_val = val_df[TARGET]

    print("\n=== Hyperparameter Grid (global, Normal) ===")
    header = f"{'Label':<16} {'Depth':>5} {'LR':>6} {'MB':>5} {'Rounds':>6} {'Val CRPS':>9}"
    print(header)
    print("-" * len(header))

    best_cfg: dict[str, Any] | None = None
    best_n = 0
    best_crps = float("inf")

    for cfg in PARAM_GRID:
        model = build_ngboost_model(cfg, dist=Normal, n_estimators=GRID_N_ESTIMATORS, verbose=False)
        model, rounds = fit_ngboost_model(model, X_train, y_train, X_val, y_val)
        val_crps = eval_model_crps(model, X_val, y_val)
        print(
            f"{cfg['label']:<16} {cfg['max_depth']:5d} {cfg['learning_rate']:6.2f} "
            f"{cfg['minibatch_frac']:5.1f} {rounds:6d} {val_crps:9.4f}"
        )
        if val_crps < best_crps:
            best_crps = val_crps
            best_cfg = dict(cfg)
            best_n = rounds

    assert best_cfg is not None
    best_cfg["n_estimators_used"] = best_n
    print(f"\nBest grid config: {best_cfg['label']} (val CRPS {best_crps:.4f}, rounds {best_n})")
    return best_cfg, best_n, best_crps, scaler


def train_global(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    hparams: dict[str, Any],
    dist: type = Normal,
    scaler: StandardScaler | None = None,
    verbose: bool = True,
) -> tuple[NGBRegressor, int, float, StandardScaler]:
    if scaler is None:
        scaler = fit_feature_scaler(train_df[feature_cols])
    X_train = transform_features(scaler, train_df, feature_cols)
    y_train = train_df[TARGET]
    X_val = transform_features(scaler, val_df, feature_cols)
    y_val = val_df[TARGET]
    dist_label = distribution_name(dist)
    print(f"\n=== Training global model ({dist_label}) ===")
    model = build_ngboost_model(hparams, dist=dist, n_estimators=FINAL_N_ESTIMATORS, verbose=verbose)
    model, best_n = fit_ngboost_model(model, X_train, y_train, X_val, y_val)
    val_crps = eval_model_crps(model, X_val, y_val) if len(y_val) else float("nan")
    return model, best_n, val_crps, scaler


def train_per_city(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cities: list[str],
    feature_cols: list[str],
    hparams: dict[str, Any],
    dist: type = Normal,
    verbose: bool = True,
) -> tuple[dict[str, NGBRegressor], dict[str, int], float, dict[str, StandardScaler]]:
    models: dict[str, NGBRegressor] = {}
    best_ns: dict[str, int] = {}
    scalers: dict[str, StandardScaler] = {}
    city_crps: list[float] = []
    dist_label = distribution_name(dist)

    print(f"\n=== Training per-city models ({dist_label}) ===")
    for city in cities:
        c_train = train_df[train_df["city"].eq(city)]
        c_val = val_df[val_df["city"].eq(city)]
        if len(c_train) == 0:
            print(f"  {city}: skip (no train rows)")
            continue
        scaler = fit_feature_scaler(c_train[feature_cols])
        scalers[city] = scaler
        X_train = transform_features(scaler, c_train, feature_cols)
        y_train = c_train[TARGET]
        X_val = transform_features(scaler, c_val, feature_cols)
        y_val = c_val[TARGET]
        print(f"  {city}: train={len(c_train)}, val={len(c_val)}")
        model = build_ngboost_model(hparams, dist=dist, n_estimators=FINAL_N_ESTIMATORS, verbose=verbose)
        model, best_n = fit_ngboost_model(model, X_train, y_train, X_val, y_val)
        models[city] = model
        best_ns[city] = best_n
        if len(y_val) > 0:
            crps = eval_model_crps(model, X_val, y_val)
            city_crps.append(crps)
            print(f"    val CRPS: {crps:.4f}")

    mean_crps = float(np.mean(city_crps)) if city_crps else float("nan")
    return models, best_ns, mean_crps, scalers


def pick_winner(
    global_crps: float,
    per_city_crps: float,
    skip_per_city: bool,
    has_per_city_models: bool,
) -> tuple[str, float]:
    if skip_per_city or not has_per_city_models or not np.isfinite(per_city_crps):
        return "global", global_crps
    if per_city_crps < global_crps:
        return "per_city", per_city_crps
    return "global", global_crps


def pick_distribution_winner(
    gaussian_crps: float,
    student_t_crps: float,
) -> tuple[str, float]:
    if student_t_crps < gaussian_crps:
        return "student_t", student_t_crps
    return "gaussian", gaussian_crps


def train_scope_models(
    scope: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cities: list[str],
    hparams: dict[str, Any],
    dist: type,
    verbose: bool = True,
) -> tuple[
    NGBRegressor | dict[str, NGBRegressor],
    int | dict[str, int],
    float,
    StandardScaler | dict[str, StandardScaler],
    list[str],
]:
    if scope == "global":
        model, best_n, crps, scaler = train_global(
            train_df, val_df, FEATURE_COLS_GLOBAL, hparams, dist=dist, verbose=verbose
        )
        return model, best_n, crps, scaler, FEATURE_COLS_GLOBAL
    models, best_ns, crps, scalers = train_per_city(
        train_df, val_df, cities, FEATURE_COLS_PER_CITY, hparams, dist=dist, verbose=verbose
    )
    return models, best_ns, crps, scalers, FEATURE_COLS_PER_CITY


def resolution_bucket(y: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(y, dtype=float) / 2.0) * 2.0


def bucket_probs(
    mu: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray | None = None,
    distribution: str = "gaussian",
) -> np.ndarray:
    """Return (n_samples, n_buckets) probability matrix."""
    n = len(mu)
    n_buckets = len(BUCKET_EDGES) - 1 + 2
    probs = np.zeros((n, n_buckets), dtype=float)
    for i in range(n):
        m, s = mu[i], sigma[i]
        if distribution == "student_t":
            assert df is not None
            di = df[i]
            probs[i, 0] = student_t.cdf(20.0, df=di, loc=m, scale=s)
            for j in range(len(BUCKET_EDGES) - 1):
                lo, hi = BUCKET_EDGES[j], BUCKET_EDGES[j + 1]
                probs[i, j + 1] = student_t.cdf(hi, df=di, loc=m, scale=s) - student_t.cdf(
                    lo, df=di, loc=m, scale=s
                )
            probs[i, -1] = 1.0 - student_t.cdf(120.0, df=di, loc=m, scale=s)
        else:
            probs[i, 0] = norm.cdf(20.0, loc=m, scale=s)
            for j in range(len(BUCKET_EDGES) - 1):
                lo, hi = BUCKET_EDGES[j], BUCKET_EDGES[j + 1]
                probs[i, j + 1] = norm.cdf(hi, loc=m, scale=s) - norm.cdf(lo, loc=m, scale=s)
            probs[i, -1] = 1.0 - norm.cdf(120.0, loc=m, scale=s)
    return probs


def actual_bucket_index(y: np.ndarray) -> np.ndarray:
    """Map each observation to bucket index (0=lower tail, -1=upper tail)."""
    y_arr = np.asarray(y, dtype=float)
    idx = np.zeros(len(y_arr), dtype=int)
    for i, val in enumerate(y_arr):
        if val < 20:
            idx[i] = 0
        elif val >= 120:
            idx[i] = len(BUCKET_EDGES)  # upper tail index
        else:
            idx[i] = 1 + int((val - 20) // 2)
    return idx


def modal_bucket_hit_rate(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray | None = None,
    distribution: str = "gaussian",
) -> float:
    probs = bucket_probs(mu, sigma, df=df, distribution=distribution)
    modal_idx = np.argmax(probs, axis=1)
    actual_idx = actual_bucket_index(y)
    return float(np.mean(modal_idx == actual_idx))


def weighted_brier_score(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray | None = None,
    distribution: str = "gaussian",
) -> float:
    probs = bucket_probs(mu, sigma, df=df, distribution=distribution)
    actual_idx = actual_bucket_index(y)
    n_buckets = probs.shape[1]
    hits = np.zeros_like(probs)
    for i, k in enumerate(actual_idx):
        hits[i, k] = 1.0
    return float(np.mean((probs - hits) ** 2))


def _collect_predictions(
    model: NGBRegressor | dict[str, NGBRegressor],
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    scaler: StandardScaler | None,
    per_city_scalers: dict[str, StandardScaler] | None,
    distribution: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if model_type == "global":
        assert scaler is not None and isinstance(model, NGBRegressor)
        X_val = transform_features(scaler, val_df, feature_cols)
        y_val = val_df[TARGET].to_numpy(dtype=float)
        mu, sigma, df_vals = predict_dist_params(model, X_val)
        return y_val, mu, sigma, df_vals

    assert per_city_scalers is not None and isinstance(model, dict)
    mu_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    df_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for city, city_model in model.items():
        c_val = val_df[val_df["city"].eq(city)]
        if len(c_val) == 0:
            continue
        city_scaler = per_city_scalers[city]
        X_val = transform_features(city_scaler, c_val, feature_cols)
        m, s, d = predict_dist_params(city_model, X_val)
        mu_parts.append(m)
        sigma_parts.append(s)
        if d is not None:
            df_parts.append(d)
        y_parts.append(c_val[TARGET].to_numpy(dtype=float))
    mu = np.concatenate(mu_parts)
    sigma = np.concatenate(sigma_parts)
    y_val = np.concatenate(y_parts)
    df_vals = np.concatenate(df_parts) if df_parts else None
    if distribution == "student_t" and df_vals is None:
        raise ValueError("Student-t predictions missing degrees of freedom")
    return y_val, mu, sigma, df_vals


def run_calibration(
    model: NGBRegressor | dict[str, NGBRegressor],
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    report_dir: Path,
    distribution: str,
    scaler: StandardScaler | None = None,
    per_city_scalers: dict[str, StandardScaler] | None = None,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)

    y_val, mu, sigma, df_vals = _collect_predictions(
        model, val_df, feature_cols, model_type, scaler, per_city_scalers, distribution
    )

    sigma_k = 1.0
    if distribution == "gaussian":
        sigma_k = fit_sigma_calibration_k(y_val, mu, sigma, level=0.90)
        print(f"\nSigma calibration: k={sigma_k:.4f} (fitted on val set for 90% coverage)")

    sigma_cal = apply_sigma_calibration(sigma, sigma_k)

    results: dict[str, Any] = {
        "distribution": distribution,
        "sigma_calibration_k": sigma_k,
    }

    # 1. PIT histogram (calibrated scale)
    if distribution == "student_t":
        assert df_vals is not None
        u = student_t.cdf(y_val, df=df_vals, loc=mu, scale=sigma_cal)
    else:
        u = norm.cdf(y_val, loc=mu, scale=sigma_cal)
    ks_stat, ks_p = kstest(u, "uniform")
    results["pit_ks_stat"] = float(ks_stat)
    results["pit_ks_p"] = float(ks_p)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(u, bins=10, range=(0, 1), density=True, edgecolor="black", alpha=0.7)
    ax.axhline(1.0, color="red", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT value u = F(y)")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT Histogram ({distribution}, KS={ks_stat:.3f}, p={ks_p:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "pit_histogram.png", dpi=150)
    plt.close(fig)
    print(f"PIT KS test: statistic={ks_stat:.3f}, p={ks_p:.3f}")

    # 2. Q-Q plot (calibrated scale)
    z = (y_val - mu) / sigma_cal
    fig, ax = plt.subplots(figsize=(6, 4))
    if distribution == "student_t" and df_vals is not None:
        ref_df = float(np.median(df_vals))
        probplot(z, dist=student_t, sparams=(ref_df,), plot=ax)
        ax.set_title(f"Q-Q Plot: Standardized Residuals vs t(df={ref_df:.1f})")
    else:
        probplot(z, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot: Standardized Residuals vs N(0,1)")
    fig.tight_layout()
    fig.savefig(report_dir / "qq_plot.png", dpi=150)
    plt.close(fig)

    # 3. Coverage: raw then calibrated
    results["coverage_raw"] = _print_coverage_table(
        y_val, mu, sigma, "=== Coverage (raw) ===", distribution, df_vals, sigma_k=1.0
    )
    results["coverage"] = _print_coverage_table(
        y_val,
        mu,
        sigma,
        f"=== Coverage (calibrated, k={sigma_k:.2f}) ===",
        distribution,
        df_vals,
        sigma_k=sigma_k,
    )

    # 4. Bucket reliability diagram (calibrated scale)
    probs = bucket_probs(mu, sigma_cal, df=df_vals, distribution=distribution)
    actual_idx = actual_bucket_index(y_val)
    pred_probs: list[float] = []
    hits: list[float] = []
    for i in range(len(y_val)):
        for k_idx in range(probs.shape[1]):
            pred_probs.append(probs[i, k_idx])
            hits.append(1.0 if actual_idx[i] == k_idx else 0.0)
    pred_probs_arr = np.array(pred_probs)
    hits_arr = np.array(hits)

    bin_edges = np.linspace(0, 1, 11)
    bin_centers: list[float] = []
    bin_observed: list[float] = []
    for b in range(10):
        mask = (pred_probs_arr >= bin_edges[b]) & (pred_probs_arr < bin_edges[b + 1])
        if b == 9:
            mask = (pred_probs_arr >= bin_edges[b]) & (pred_probs_arr <= bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
        bin_observed.append(float(hits_arr[mask].mean()))

    brier = weighted_brier_score(y_val, mu, sigma_cal, df=df_vals, distribution=distribution)
    results["bucket_brier"] = brier

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(bin_centers, bin_observed, s=50, zorder=3)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Bucket Reliability (Brier={brier:.4f}, k={sigma_k:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "bucket_reliability.png", dpi=150)
    plt.close(fig)
    print(f"Bucket Brier score: {brier:.4f}")

    results["val_mae"] = float(np.mean(np.abs(mu - y_val)))
    if distribution == "student_t" and df_vals is not None:
        results["val_crps"] = student_t_crps(y_val, mu, sigma_cal, df_vals)
    else:
        results["val_crps"] = gaussian_crps(y_val, mu, sigma_cal)
    results["modal_bucket_hr"] = modal_bucket_hit_rate(
        y_val, mu, sigma_cal, df=df_vals, distribution=distribution
    )
    return results


def print_per_city_metrics(
    model: NGBRegressor | dict[str, NGBRegressor],
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    cities: list[str],
    distribution: str,
    scaler: StandardScaler | None = None,
    per_city_scalers: dict[str, StandardScaler] | None = None,
    sigma_k: float = 1.0,
) -> None:
    print("\n=== Per-city validation metrics ===")
    header = f"{'city':<16} {'MAE':>7} {'CRPS':>8} {'modal_hr':>9}"
    print(header)
    print("-" * len(header))
    for city in cities:
        c_val = val_df[val_df["city"].eq(city)]
        if len(c_val) == 0:
            continue
        if model_type == "global":
            assert scaler is not None and isinstance(model, NGBRegressor)
            m = model
            X = transform_features(scaler, c_val, feature_cols)
        else:
            if (
                not isinstance(model, dict)
                or city not in model
                or per_city_scalers is None
                or city not in per_city_scalers
            ):
                continue
            m = model[city]
            X = transform_features(per_city_scalers[city], c_val, feature_cols)
        y = c_val[TARGET].to_numpy(dtype=float)
        mu, sigma, df_vals = predict_dist_params(m, X)
        sigma_cal = apply_sigma_calibration(sigma, sigma_k)
        mae = float(np.mean(np.abs(mu - y)))
        if distribution == "student_t" and df_vals is not None:
            crps = student_t_crps(y, mu, sigma_cal, df_vals)
        else:
            crps = gaussian_crps(y, mu, sigma_cal)
        modal_hr = 100.0 * modal_bucket_hit_rate(
            y, mu, sigma_cal, df=df_vals, distribution=distribution
        )
        print(f"{city:<16} {mae:7.2f} {crps:8.4f} {modal_hr:8.1f}%")


def save_artifacts(
    scope: str,
    distribution: str,
    global_model: NGBRegressor | None,
    per_city_models: dict[str, NGBRegressor] | None,
    output_dir: Path,
    cities: list[str],
    feature_cols: list[str],
    winner_crps: float,
    hparams: dict[str, Any],
    best_n_global: int | None,
    best_ns_per_city: dict[str, int] | None,
    global_scaler: StandardScaler | None,
    per_city_scalers: dict[str, StandardScaler] | None,
    fill_medians: dict[str, float] | None = None,
    stage1_val_mae: float | None = None,
    stage1_val_rmse: float | None = None,
    sigma_calibration_k: float | None = None,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_path = ""

    if scope == "global":
        assert global_model is not None and global_scaler is not None
        saved_path = str(output_dir / "ngboost_global.pkl")
        joblib.dump(global_model, saved_path)
        joblib.dump(global_scaler, output_dir / "feature_scaler.pkl")
    else:
        assert per_city_models is not None and per_city_scalers is not None
        for city, city_model in per_city_models.items():
            path = output_dir / f"ngboost_{city}.pkl"
            joblib.dump(city_model, path)
            joblib.dump(per_city_scalers[city], output_dir / f"feature_scaler_{city}.pkl")
        saved_path = str(output_dir / "ngboost_{city}.pkl (per city)")

    config: dict[str, Any] = {
        "model_type": scope,
        "distribution": distribution,
        "feature_columns": feature_cols,
        "cities": cities,
        "train_dates": ["2021-01-01", "2024-12-31"],
        "val_dates": ["2025-01-01", "2025-12-31"],
        "val_crps": round(winner_crps, 4),
        "hyperparameters": {
            "max_depth": hparams["max_depth"],
            "learning_rate": hparams["learning_rate"],
            "minibatch_frac": hparams["minibatch_frac"],
            "label": hparams.get("label", "unknown"),
            "n_estimators_used": best_n_global if scope == "global" else None,
        },
        "station_id_map": STATION_ID_MAP,
        "n_estimators_used": best_n_global if scope == "global" else None,
        "scaler_path": "feature_scaler.pkl" if scope == "global" else None,
        "stage1_model": "lgb_stage1.pkl",
    }
    if stage1_val_mae is not None:
        config["stage1_val_mae"] = round(stage1_val_mae, 4)
    if stage1_val_rmse is not None:
        config["stage1_val_rmse"] = round(stage1_val_rmse, 4)
    if sigma_calibration_k is not None:
        config["sigma_calibration_k"] = round(sigma_calibration_k, 6)
    if fill_medians:
        config["nan_fill_medians"] = {k: round(v, 6) for k, v in fill_medians.items()}
    if scope == "per_city" and best_ns_per_city:
        config["per_city_n_estimators"] = best_ns_per_city
        config["scaler_paths"] = {city: f"feature_scaler_{city}.pkl" for city in per_city_models}
        config["hyperparameters"]["per_city_n_estimators"] = best_ns_per_city

    config_path = output_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return saved_path


def print_summary(
    cities: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    scope: str,
    distribution: str,
    hparams: dict[str, Any],
    best_n: int | dict[str, int],
    winner_crps: float,
    cal_results: dict[str, Any],
    saved_path: str,
    report_dir: Path,
    stage1_val_mae: float | None = None,
    stage1_val_rmse: float | None = None,
    sigma_calibration_k: float | None = None,
) -> None:
    cov = cal_results.get("coverage", {})
    cov_raw = cal_results.get("coverage_raw", {})
    print("\n=== NGBoost Training Summary ===")
    print(f"Cities: {', '.join(cities)}")
    print(f"Train samples: {n_train} (2021-2024)")
    print(f"Val samples: {n_val} (2025)")
    print(f"Test samples: {n_test} (2026+)")
    if isinstance(best_n, dict):
        rounds_str = ", ".join(f"{city}={n}" for city, n in sorted(best_n.items()))
        print(
            f"Best hyperparameters: depth={hparams['max_depth']}, "
            f"lr={hparams['learning_rate']}, minibatch_frac={hparams['minibatch_frac']} "
            f"({hparams.get('label', 'unknown')}); rounds per city: {rounds_str}"
        )
    else:
        print(
            f"Best hyperparameters: depth={hparams['max_depth']}, "
            f"lr={hparams['learning_rate']}, minibatch_frac={hparams['minibatch_frac']} "
            f"({hparams.get('label', 'unknown')}), rounds={best_n}"
        )
    print(f"Model scope winner: {scope}")
    print(f"Distribution winner: {distribution}")
    print(f"Val CRPS: {winner_crps:.4f}")
    if stage1_val_mae is not None:
        print(f"Stage-1 LightGBM val MAE: {stage1_val_mae:.2f} F")
    if stage1_val_rmse is not None:
        print(f"Stage-1 LightGBM val RMSE: {stage1_val_rmse:.2f} F")
    print(f"NGBoost val MAE (with LGB anchor): {cal_results['val_mae']:.2f} F")
    if stage1_val_mae is not None:
        mae_improvement = stage1_val_mae - cal_results["val_mae"]
        print(f"MAE improvement from LGB anchor: {mae_improvement:.2f} F")
    print(
        f"PIT KS test: statistic={cal_results['pit_ks_stat']:.3f}, "
        f"p={cal_results['pit_ks_p']:.3f}"
    )
    if sigma_calibration_k is not None:
        print(f"Sigma calibration k: {sigma_calibration_k:.4f}")
    if cov_raw:
        print(
            f"Coverage 90% (raw / calibrated): "
            f"{cov_raw.get(90, 0):.1f}% / {cov.get(90, 0):.1f}%"
        )
    print(
        f"Coverage 50/80/90/95 (calibrated): "
        f"{cov.get(50, 0):.1f} / {cov.get(80, 0):.1f} / "
        f"{cov.get(90, 0):.1f} / {cov.get(95, 0):.1f} %"
    )
    print(f"Bucket Brier score: {cal_results['bucket_brier']:.4f}")
    print(f"Model saved to: {saved_path}")
    print(f"Calibration plots saved to: {report_dir}/")


def load_saved_artifacts(output_dir: Path) -> tuple[NGBRegressor, StandardScaler, dict[str, Any]]:
    config_path = output_dir / "model_config.json"
    model_path = output_dir / "ngboost_global.pkl"
    scaler_path = output_dir / "feature_scaler.pkl"

    if not config_path.exists():
        print(f"Missing model config: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    model_type = config.get("model_type", "global")
    if model_type != "global":
        print(
            f"--eval-test currently supports global models only (saved type: {model_type})."
        )
        sys.exit(1)

    if not model_path.exists():
        print(f"Missing model file: {model_path}")
        sys.exit(1)
    if not scaler_path.exists():
        print(f"Missing scaler file: {scaler_path}")
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, config


def apply_saved_median_fill(
    df: pd.DataFrame,
    fill_medians: dict[str, float],
    cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        median_val = float(fill_medians.get(col, 0.0))
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(median_val)
    return out


def empirical_coverage(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal_pct: float,
    df_vals: np.ndarray | None = None,
    distribution: str = "gaussian",
) -> float:
    z_alpha = dict(COVERAGE_LEVELS).get(int(nominal_pct))
    if distribution == "student_t":
        if df_vals is None:
            raise ValueError("Student-t coverage requires degrees of freedom")
        t_crit = student_t.ppf((1.0 + nominal_pct / 100.0) / 2.0, df=df_vals)
        lower = mu - t_crit * sigma
        upper = mu + t_crit * sigma
    else:
        assert z_alpha is not None
        lower = mu - z_alpha * sigma
        upper = mu + z_alpha * sigma
    return 100.0 * float(np.mean((y >= lower) & (y <= upper)))


def compute_split_metrics(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    df_vals: np.ndarray | None,
    distribution: str,
    sigma_k: float = 1.0,
) -> dict[str, float]:
    sigma_cal = apply_sigma_calibration(sigma, sigma_k)
    mae = float(np.mean(np.abs(mu - y)))
    if distribution == "student_t" and df_vals is not None:
        crps = student_t_crps(y, mu, sigma_cal, df_vals)
    else:
        crps = gaussian_crps(y, mu, sigma_cal)
    cov90 = empirical_coverage(y, mu, sigma_cal, 90.0, df_vals=df_vals, distribution=distribution)
    modal_hr = 100.0 * modal_bucket_hit_rate(
        y, mu, sigma_cal, df=df_vals, distribution=distribution
    )
    return {"mae": mae, "crps": crps, "cov90": cov90, "modal_hr": modal_hr}


def print_test_eval_table(
    model: NGBRegressor,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cities: list[str],
    distribution: str,
    scaler: StandardScaler,
    sigma_k: float = 1.0,
) -> None:
    print("\n=== TEST set metrics (2026+) ===")
    header = (
        f"{'City':<16} {'N rows':>7} {'MAE':>7} {'CRPS':>8} "
        f"{'90% Cov':>9} {'Modal HR':>9}"
    )
    print(header)
    print("-" * len(header))

    overall_y: list[np.ndarray] = []
    overall_mu: list[np.ndarray] = []
    overall_sigma: list[np.ndarray] = []
    overall_df: list[np.ndarray] = []

    for city in cities:
        c_test = test_df[test_df["city"].eq(city)]
        n_rows = len(c_test)
        if n_rows == 0:
            print(f"{city:<16} {n_rows:7d} {'—':>7} {'—':>8} {'—':>9} {'—':>9}")
            continue

        X = transform_features(scaler, c_test, feature_cols)
        y = c_test[TARGET].to_numpy(dtype=float)
        mu, sigma, df_vals = predict_dist_params(model, X)
        metrics = compute_split_metrics(y, mu, sigma, df_vals, distribution, sigma_k=sigma_k)
        print(
            f"{city:<16} {n_rows:7d} {metrics['mae']:7.2f} {metrics['crps']:8.4f} "
            f"{metrics['cov90']:8.1f}% {metrics['modal_hr']:8.1f}%"
        )
        overall_y.append(y)
        overall_mu.append(mu)
        overall_sigma.append(sigma)
        if df_vals is not None:
            overall_df.append(df_vals)

    if overall_y:
        y_all = np.concatenate(overall_y)
        mu_all = np.concatenate(overall_mu)
        sigma_all = np.concatenate(overall_sigma)
        df_all = np.concatenate(overall_df) if overall_df else None
        overall = compute_split_metrics(y_all, mu_all, sigma_all, df_all, distribution, sigma_k=sigma_k)
        print("-" * len(header))
        print(
            f"{'OVERALL':<16} {len(y_all):7d} {overall['mae']:7.2f} {overall['crps']:8.4f} "
            f"{overall['cov90']:8.1f}% {overall['modal_hr']:8.1f}%"
        )
    else:
        print("\nNo test rows available (2026+).")


def run_eval_test(output_dir: Path) -> None:
    model, scaler, config = load_saved_artifacts(output_dir)
    cities = list(config.get("cities", DEFAULT_CITIES))
    feature_cols = list(config.get("feature_columns", FEATURE_COLS_GLOBAL))
    distribution = str(config.get("distribution", "gaussian"))
    fill_medians = dict(config.get("nan_fill_medians", {}))

    stage1_path = output_dir / config.get("stage1_model", "lgb_stage1.pkl")
    if not stage1_path.exists():
        print(f"Missing stage-1 model: {stage1_path}")
        sys.exit(1)
    lgb_model = joblib.load(stage1_path)
    stage1_cols = [c for c in feature_cols if c != "lgb_tmax_pred"]

    print(f"=== NGBoost TEST evaluation: {', '.join(cities)} ===")
    print(f"Model: {output_dir / 'ngboost_global.pkl'}")
    print(f"Stage-1: {stage1_path}")
    print(f"Distribution: {distribution}")

    df = assemble_dataset(cities)
    df = drop_incomplete_rows(df)
    _train_df, _val_df, test_df = temporal_split(df)

    fill_cols = list(fill_medians.keys()) if fill_medians else MEDIAN_FILL_COLS
    if fill_medians:
        test_df = apply_saved_median_fill(test_df, fill_medians, fill_cols)
        print(f"Applied saved NaN fill medians: {fill_medians}")

    missing_cols = [c for c in stage1_cols if c not in test_df.columns]
    if missing_cols:
        print(f"Missing stage-1 feature columns in test data: {missing_cols}")
        sys.exit(1)

    test_df = test_df.copy()
    test_df["lgb_tmax_pred"] = lgb_model.predict(test_df[stage1_cols])

    sigma_k = float(config.get("sigma_calibration_k", 1.0))
    if sigma_k != 1.0:
        print(f"Sigma calibration k: {sigma_k:.4f}")

    print_test_eval_table(model, test_df, feature_cols, cities, distribution, scaler, sigma_k=sigma_k)


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir

    if args.eval_test:
        run_eval_test(output_dir)
        return

    cities = args.cities
    report_dir = PROJECT_ROOT / args.report_dir

    print(f"=== NGBoost Training: {', '.join(cities)} ===\n")

    df = assemble_dataset(cities)
    df = drop_incomplete_rows(df)
    train_df, val_df, test_df = temporal_split(df)
    train_df, val_df, test_df, fill_medians = fill_median_features(
        train_df, val_df, test_df, MEDIAN_FILL_COLS
    )
    print(f"\nNaN fill medians (train only): {fill_medians}")

    _lgb_model, stage1_val_mae, stage1_val_rmse, train_df, val_df, test_df = train_stage1_lgb(
        train_df,
        val_df,
        test_df,
        FEATURE_COLS_STAGE1,
        cities,
        output_dir,
    )

    feature_cols_global = FEATURE_COLS_GLOBAL
    assert_no_leakage(train_df, val_df, test_df, feature_cols_global)

    if args.skip_grid:
        hparams = load_saved_hparams(output_dir)
        print(f"Skipping hyperparameter grid; using config: {hparams['label']}")
    else:
        hparams, _grid_rounds, _grid_crps, _grid_scaler = run_hyperparam_grid(
            train_df, val_df, feature_cols_global
        )

    # (d) Global vs per-city with best hyperparameters (Normal dist)
    global_model, best_n_global, global_crps, global_scaler = train_global(
        train_df, val_df, feature_cols_global, hparams, dist=Normal, verbose=True
    )
    print(f"Global model val CRPS: {global_crps:.4f}")

    per_city_models: dict[str, NGBRegressor] = {}
    best_ns_per_city: dict[str, int] = {}
    per_city_scalers: dict[str, StandardScaler] = {}
    per_city_mean_crps = float("nan")

    if not args.skip_per_city:
        per_city_models, best_ns_per_city, per_city_mean_crps, per_city_scalers = train_per_city(
            train_df, val_df, cities, FEATURE_COLS_PER_CITY, hparams, dist=Normal, verbose=True
        )
        print(f"Per-city model val CRPS (mean across cities): {per_city_mean_crps:.4f}")
    else:
        print("Skipping per-city model training (--skip-per-city)")

    scope, scope_crps = pick_winner(
        global_crps,
        per_city_mean_crps,
        args.skip_per_city,
        bool(per_city_models),
    )
    print(f"Model scope winner: {scope} (CRPS {scope_crps:.4f})")

    # (e) Distribution comparison on scope winner (reuse trained Normal models)
    if scope == "global":
        gaussian_model: NGBRegressor | dict[str, NGBRegressor] = global_model
        gaussian_best_n: int | dict[str, int] = best_n_global
        gaussian_scaler: StandardScaler | dict[str, StandardScaler] = global_scaler
        gaussian_feature_cols = feature_cols_global
    else:
        gaussian_model = per_city_models
        gaussian_best_n = best_ns_per_city
        gaussian_scaler = per_city_scalers
        gaussian_feature_cols = FEATURE_COLS_PER_CITY
    gaussian_crps = scope_crps
    print(f"Gaussian val CRPS: {gaussian_crps:.4f}")

    final_model: NGBRegressor | dict[str, NGBRegressor] = gaussian_model
    final_distribution = "gaussian"
    final_crps = gaussian_crps
    final_scaler = gaussian_scaler
    final_feature_cols = gaussian_feature_cols
    final_best_n = gaussian_best_n

    if HAS_TDIST and TDist is not None:
        try:
            t_model, t_best_n, t_crps, t_scaler, t_feature_cols = train_scope_models(
                scope, train_df, val_df, cities, hparams, TDist, verbose=True
            )
            print(f"Student-t val CRPS: {t_crps:.4f}")
            final_distribution, final_crps = pick_distribution_winner(gaussian_crps, t_crps)
            print(f"Distribution winner: {final_distribution}")
            if final_distribution == "student_t":
                final_model = t_model
                final_scaler = t_scaler
                final_feature_cols = t_feature_cols
                final_best_n = t_best_n
        except Exception as exc:
            print(f"WARNING: Student-t training failed ({exc}); using Gaussian only.")
    else:
        print("WARNING: Student-t distribution unavailable in ngboost; using Gaussian only.")

    cal_feature_cols = final_feature_cols
    global_scaler_final = final_scaler if scope == "global" and isinstance(final_scaler, StandardScaler) else None
    per_city_scalers_final = (
        final_scaler if scope == "per_city" and isinstance(final_scaler, dict) else None
    )

    # (f) Calibration on final winning model
    cal_results = run_calibration(
        final_model,
        val_df,
        cal_feature_cols,
        scope,
        report_dir,
        final_distribution,
        scaler=global_scaler_final,
        per_city_scalers=per_city_scalers_final,
    )
    sigma_calibration_k = float(cal_results.get("sigma_calibration_k", 1.0))
    print_per_city_metrics(
        final_model,
        val_df,
        cal_feature_cols,
        scope,
        cities,
        final_distribution,
        scaler=global_scaler_final,
        per_city_scalers=per_city_scalers_final,
        sigma_k=sigma_calibration_k,
    )

    global_model_save = final_model if scope == "global" and isinstance(final_model, NGBRegressor) else None
    per_city_models_save = final_model if scope == "per_city" and isinstance(final_model, dict) else None
    best_n_global_save = final_best_n if scope == "global" and isinstance(final_best_n, int) else None
    best_ns_save = final_best_n if scope == "per_city" and isinstance(final_best_n, dict) else None

    saved_path = save_artifacts(
        scope=scope,
        distribution=final_distribution,
        global_model=global_model_save,
        per_city_models=per_city_models_save,
        output_dir=output_dir,
        cities=cities,
        feature_cols=cal_feature_cols,
        winner_crps=final_crps,
        hparams=hparams,
        best_n_global=best_n_global_save,
        best_ns_per_city=best_ns_save,
        global_scaler=global_scaler_final,
        per_city_scalers=per_city_scalers_final,
        fill_medians=fill_medians,
        stage1_val_mae=stage1_val_mae,
        stage1_val_rmse=stage1_val_rmse,
        sigma_calibration_k=sigma_calibration_k,
    )

    print_summary(
        cities=cities,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        scope=scope,
        distribution=final_distribution,
        hparams=hparams,
        best_n=final_best_n,
        winner_crps=final_crps,
        cal_results=cal_results,
        saved_path=saved_path,
        report_dir=report_dir,
        stage1_val_mae=stage1_val_mae,
        stage1_val_rmse=stage1_val_rmse,
        sigma_calibration_k=sigma_calibration_k,
    )


if __name__ == "__main__":
    main()
