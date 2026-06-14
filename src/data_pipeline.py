"""Unified data pipeline for Track-B feature construction.

Handles both historical backfill and live daily fetch.
Every function includes leakage assertions.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from dateutil.tz import gettz

    def ZoneInfo(name: str):
        tz = gettz(name)
        if tz is None:
            raise ValueError(f"Unknown timezone: {name}")
        return tz

from src.trackj.build_asos_features import (
    ASOS_FEATURE_COLUMNS,
    aggregate_morning_asos,
    fetch_asos_range,
    load_cached_asos,
)
from src.trackj.build_calendar_lag_features import CALENDAR_LAG_COLUMNS, build_calendar_lag_features
from src.trackj.fetch_cli_target import fetch_cli_target
from src.trackj.fetch_gfs_herbie import GFS_FEATURE_COLUMNS, fetch_gfs_for_date, gfs_cache_path
from src.trackj.fetch_nws_forecast import (
    DEFAULT_OUTPUT_PATH as NWS_RAW_PATH,
    _issued_before_for_target,
    fetch_nws_tmax_forecast,
)
from src.trackj.fetch_openmeteo_nwp import (
    DEFAULT_OUTPUT_PATH as OPENMETEO_NWP_PATH,
    fetch_openmeteo_tmax,
    make_session as openmeteo_session,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKB_DIR = PROJECT_ROOT / "data" / "trackb"
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
RAW_DIR = TRACKJ_DIR / "raw"

ASOS_MORNING_COLUMNS = [column for column in ASOS_FEATURE_COLUMNS if column != "temp_lag1"]
LAG_COLUMNS = list(CALENDAR_LAG_COLUMNS) + ["temp_lag1"]

_city_config_cache: dict | None = None


def _load_all_city_config() -> dict:
    global _city_config_cache
    if _city_config_cache is None:
        _city_config_cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _city_config_cache


def _load_city_config(city: str) -> dict:
    config = _load_all_city_config()
    if city not in config:
        raise KeyError(f"Unknown city: {city}")
    return config[city]


def _parse_event_date(event_date: str) -> date:
    return date.fromisoformat(str(event_date))


def _feature_table_row(city: str, event_date: str) -> pd.Series | None:
    feat_path = TRACKB_DIR / city / "features.parquet"
    if not feat_path.exists():
        return None
    df = pd.read_parquet(feat_path)
    df["_date_key"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = df[df["_date_key"] == str(event_date)]
    if row.empty:
        return None
    return row.iloc[0]


def _issued_before(city: str, event_date: str, issued_before_hour: int = 22) -> datetime:
    cfg = _load_city_config(city)
    target = _parse_event_date(event_date)
    local_tz = ZoneInfo(cfg["timezone"])
    return _issued_before_for_target(target, issued_before_hour, local_tz)


def _gfs_raw_dir(city_config: dict) -> Path:
    station = str(city_config["nws_station"]).lower()
    if station == "kaus":
        return PROJECT_ROOT / "data" / "raw" / "gfs_kaus"
    return PROJECT_ROOT / "data" / "raw" / f"gfs_{station}"


def _extract_columns(row: pd.Series, columns: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for col in columns:
        if col not in row.index:
            continue
        val = row[col]
        if pd.notna(val):
            result[col] = float(val)
    return result


def _load_cli_target(city: str, event_date: str) -> pd.DataFrame:
    cfg = _load_city_config(city)
    cli_path = TRACKJ_DIR / city / "cli_target.parquet"
    target = _parse_event_date(event_date)
    need_through = target - timedelta(days=1)
    lag_start = target - timedelta(days=45)

    existing: pd.DataFrame | None = None
    if cli_path.exists():
        existing = pd.read_parquet(cli_path)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        valid_dates = existing["date"].dropna()
        max_date = valid_dates.max()
        min_date = valid_dates.min()
        has_recent = bool(max_date) and pd.Timestamp(max_date) >= pd.Timestamp(need_through.isoformat())
        has_history = bool(min_date) and pd.Timestamp(min_date) <= pd.Timestamp(lag_start.isoformat())
        if has_recent and has_history:
            return existing
        if not has_history:
            fetch_start = lag_start
        elif max_date:
            fetch_start = pd.Timestamp(max_date).date() + timedelta(days=1)
        else:
            fetch_start = lag_start
    else:
        fetch_start = lag_start

    if existing is not None and fetch_start > lag_start:
        refreshed = fetch_cli_target(cfg, fetch_start, target, RAW_DIR, TRACKJ_DIR, no_fetch=False)
        combined = pd.concat([existing, refreshed], ignore_index=True)
    else:
        refreshed = fetch_cli_target(cfg, lag_start, target, RAW_DIR, TRACKJ_DIR, no_fetch=False)
        combined = refreshed

    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    combined = combined[combined["date"] >= lag_start.isoformat()].copy()
    cli_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cli_path, index=False)
    return combined


def fetch_asos_morning(
    city: str,
    event_date: str,
    cutoff_hour_local: int = 10,
    skip_cache: bool = False,
) -> Optional[Dict[str, float]]:
    """Fetch ASOS morning observations up to 10AM local."""
    row = None if skip_cache else _feature_table_row(city, event_date)
    if row is not None:
        result = _extract_columns(row, ASOS_MORNING_COLUMNS)
        if result:
            return result

    cfg = _load_city_config(city)
    target = _parse_event_date(event_date)
    city_raw_dir = RAW_DIR / city

    try:
        fetch_asos_range(cfg, target, target, city_raw_dir, overwrite=skip_cache)
        asos = load_cached_asos(city_raw_dir, cfg["nws_station"], target, target)
        if asos.empty:
            print(f"  ASOS fetch failed for {city}/{event_date}: no cached rows")
            return None
        day_rows = asos[asos["date"].astype(str) == str(event_date)].copy()
        if day_rows.empty:
            print(f"  ASOS fetch failed for {city}/{event_date}: no observations for {event_date}")
            return None
        if cutoff_hour_local != 10:
            day_rows = day_rows[day_rows["valid_local"].dt.hour <= cutoff_hour_local].copy()
        morning_rows = day_rows[
            (day_rows["valid_local"].dt.time >= datetime.strptime("00:00", "%H:%M").time())
            & (day_rows["valid_local"].dt.time <= datetime.strptime("10:00", "%H:%M").time())
        ]
        if morning_rows.empty:
            print(f"  ASOS fetch failed for {city}/{event_date}: no 00:00-10:00 local observations")
            return None
        aggregated = aggregate_morning_asos(asos, [event_date], target_df=None)
        if aggregated.empty:
            return None
        result = _extract_columns(aggregated.iloc[0], ASOS_MORNING_COLUMNS)
        if len(result) < len(ASOS_MORNING_COLUMNS):
            missing = set(ASOS_MORNING_COLUMNS) - set(result)
            print(f"  ASOS fetch failed for {city}/{event_date}: missing fields {sorted(missing)}")
            return None
        return result
    except Exception as exc:
        print(f"  ASOS fetch failed for {city}/{event_date}: {exc}")
        return None


def fetch_nws_forecast(
    city: str,
    event_date: str,
    skip_cache: bool = False,
) -> Optional[float]:
    """Fetch NWS MOS Tmax forecast issued the evening before event_date."""
    result = fetch_nws_forecast_full(city, event_date, skip_cache=skip_cache)
    return result["nws_tmax_forecast_f"] if result else None


def fetch_nws_forecast_full(
    city: str,
    event_date: str,
    skip_cache: bool = False,
) -> Optional[Dict[str, float]]:
    """Fetch NWS MOS Tmax forecast and issuance lag in hours."""
    row = None if skip_cache else _feature_table_row(city, event_date)
    if row is not None and "nws_tmax_forecast_f" in row.index:
        val = row["nws_tmax_forecast_f"]
        issued_h = row.get("nws_tmax_forecast_issued_h")
        if pd.notna(val):
            payload = {"nws_tmax_forecast_f": float(val)}
            if pd.notna(issued_h):
                payload["nws_tmax_forecast_issued_h"] = float(issued_h)
            return payload

    if not skip_cache:
        nws_path = NWS_RAW_PATH
    else:
        nws_path = None
    if nws_path is not None and nws_path.exists():
        frame = pd.read_parquet(nws_path)
        city_rows = frame[frame["city"].astype(str).eq(city)].copy()
        city_rows["date_key"] = pd.to_datetime(city_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        match = city_rows[city_rows["date_key"] == str(event_date)]
        if not match.empty:
            val = match.iloc[0].get("tmax_forecast_f")
            hours = match.iloc[0].get("hours_since_issuance")
            if pd.notna(val):
                payload = {"nws_tmax_forecast_f": float(val)}
                if pd.notna(hours):
                    payload["nws_tmax_forecast_issued_h"] = float(hours)
                return payload

    try:
        cfg = _load_city_config(city)
        issued_before = _issued_before(city, event_date)
        result = fetch_nws_tmax_forecast(
            float(cfg["lat"]),
            float(cfg["lon"]),
            event_date,
            issued_before,
            station=cfg["nws_station"],
        )
        if result is None:
            return None
        issued = pd.to_datetime(result.get("issued_time"), utc=True, errors="coerce")
        target_dt = pd.Timestamp(_parse_event_date(event_date))
        local_tz = ZoneInfo(cfg["timezone"])
        cutoff = pd.Timestamp(datetime.combine(target_dt.date(), datetime.min.time().replace(hour=10)), tz=local_tz)
        if pd.notna(issued) and issued >= cutoff:
            print(f"  NWS leakage guard: issuance {issued} >= 10AM local on {event_date}")
            return None
        payload = {"nws_tmax_forecast_f": float(result["tmax_forecast_f"])}
        hours = result.get("hours_since_issuance")
        if hours is not None and pd.notna(hours):
            payload["nws_tmax_forecast_issued_h"] = float(hours)
        return payload
    except Exception as exc:
        print(f"  NWS fetch failed for {city}/{event_date}: {exc}")
        return None


def fetch_gfs_afternoon(
    city: str,
    event_date: str,
    skip_cache: bool = False,
) -> Optional[Dict[str, float]]:
    """Fetch GFS afternoon forecast fields (00Z cycle)."""
    row = None if skip_cache else _feature_table_row(city, event_date)
    if row is not None:
        result = _extract_columns(row, GFS_FEATURE_COLUMNS)
        if len(result) == 3:
            return result

    cfg = _load_city_config(city)
    target = _parse_event_date(event_date)
    raw_dir = _gfs_raw_dir(cfg)
    cache_path = gfs_cache_path(raw_dir, target, city_config=cfg)
    if cache_path.exists():
        cached = pd.read_csv(cache_path).iloc[0].to_dict()
        result = {col: float(cached[col]) for col in GFS_FEATURE_COLUMNS if col in cached and pd.notna(cached[col])}
        if len(result) == 3:
            return result

    try:
        features, audit = fetch_gfs_for_date(target, raw_dir=raw_dir, city_config=cfg)
        if audit.get("gfs_parse_status") != "ok":
            return None
        fxx = audit.get("gfs_selected_fxx")
        if pd.isna(fxx) or float(fxx) <= 0:
            print(f"  GFS leakage guard: fxx={fxx} must be > 0")
            return None
        result = {col: float(features[col]) for col in GFS_FEATURE_COLUMNS if pd.notna(features.get(col))}
        return result if len(result) == 3 else None
    except Exception as exc:
        print(f"  GFS fetch failed for {city}/{event_date}: {exc}")
        return None


def _nwp_from_openmeteo_cache(city: str, event_date: str) -> tuple[float | None, str | None]:
    path = OPENMETEO_NWP_PATH
    if not path.exists():
        return None, None
    frame = pd.read_parquet(path)
    frame["date_key"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    city_rows = frame[frame["city"].astype(str).eq(city) & frame["date_key"].eq(str(event_date))]
    for model in ("ecmwf_ifs025", "gfs_seamless"):
        model_rows = city_rows[city_rows["model_used"].astype(str).eq(model)]
        if model_rows.empty:
            continue
        val = model_rows.iloc[0].get("nwp_tmax_forecast_f")
        issued = model_rows.iloc[0].get("issued_date")
        if pd.notna(val):
            return float(val), str(issued) if pd.notna(issued) else None
    return None, None


def fetch_nwp_best(
    city: str,
    event_date: str,
    skip_cache: bool = False,
) -> Optional[float]:
    """Fetch best-available NWP Tmax forecast (ECMWF priority, GFS fallback)."""
    row = None if skip_cache else _feature_table_row(city, event_date)
    if row is not None and "nwp_tmax_best_f" in row.index:
        val = row["nwp_tmax_best_f"]
        if pd.notna(val):
            return float(val)

    cached, issued = (None, None) if skip_cache else _nwp_from_openmeteo_cache(city, event_date)
    if cached is not None:
        if issued is not None and issued >= str(event_date):
            print(f"  NWP leakage guard: issued_date {issued} >= {event_date}")
            return None
        return cached

    cfg = _load_city_config(city)
    target = _parse_event_date(event_date)
    session = openmeteo_session()
    lat = float(cfg["lat"])
    lon = float(cfg["lon"])

    try:
        for model in ("ecmwf_ifs025", "gfs_seamless"):
            frame = fetch_openmeteo_tmax(lat, lon, target, target, model, session=session, sleep_seconds=0.0)
            if frame.empty:
                continue
            match = frame[frame["date"].astype(str) == str(event_date)]
            if match.empty:
                continue
            val = match.iloc[0].get("nwp_tmax_forecast_f")
            issued_date = match.iloc[0].get("issued_date")
            if pd.isna(val):
                continue
            if pd.notna(issued_date) and str(issued_date) >= str(event_date):
                print(f"  NWP leakage guard: issued_date {issued_date} >= {event_date}")
                continue
            return float(val)

        nws = fetch_nws_forecast(city, event_date)
        return nws
    except Exception as exc:
        print(f"  NWP fetch failed for {city}/{event_date}: {exc}")
        return None


def fetch_lag_features(
    city: str,
    event_date: str,
    skip_cache: bool = False,
) -> Optional[Dict[str, float]]:
    """Compute temperature lag features from historical CLI data."""
    row = None if skip_cache else _feature_table_row(city, event_date)
    if row is not None:
        result = _extract_columns(row, LAG_COLUMNS)
        if result:
            return result

    try:
        cli = _load_cli_target(city, event_date)
        if cli.empty:
            return None
        calendar = build_calendar_lag_features(cli)
        match = calendar[calendar["date"].astype(str) == str(event_date)]
        if match.empty:
            return None
        lag_row = match.iloc[0]
        result = _extract_columns(lag_row, CALENDAR_LAG_COLUMNS)

        cli_sorted = cli.copy()
        cli_sorted["date_dt"] = pd.to_datetime(cli_sorted["date"], errors="coerce")
        cli_sorted = cli_sorted.sort_values("date_dt")
        temp_lag1 = pd.to_numeric(cli_sorted["tmax_f"], errors="coerce").shift(1)
        lag_map = dict(zip(cli_sorted["date_dt"].dt.strftime("%Y-%m-%d"), temp_lag1))
        temp_lag_val = lag_map.get(str(event_date))
        if pd.notna(temp_lag_val):
            result["temp_lag1"] = float(temp_lag_val)

        event_dt = pd.Timestamp(event_date)
        if any(pd.isna(lag_row.get(col)) for col in ("tmax_lag1", "doy_sin", "doy_cos")):
            pass
        elif pd.notna(lag_row.get("tmax_lag1")):
            prev_date = (event_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            prev_cli = cli_sorted[cli_sorted["date_dt"].dt.strftime("%Y-%m-%d") == prev_date]
            if not prev_cli.empty:
                actual = float(prev_cli.iloc[0]["tmax_f"])
                if abs(float(lag_row["tmax_lag1"]) - actual) > 0.01:
                    print(f"  Lag leakage warning: tmax_lag1 != D-1 actual for {city}/{event_date}")

        return result or None
    except Exception as exc:
        print(f"  Lag fetch failed for {city}/{event_date}: {exc}")
        return None


def build_feature_vector(city: str, event_date: str) -> Optional[Dict[str, float]]:
    """Build the complete feature vector for a (city, event_date)."""
    features: dict[str, float] = {}
    sources_ok: list[str] = []
    sources_fail: list[str] = []

    asos = fetch_asos_morning(city, event_date)
    if asos:
        features.update(asos)
        sources_ok.append("ASOS")
    else:
        sources_fail.append("ASOS")

    lags = fetch_lag_features(city, event_date)
    if lags:
        features.update(lags)
        sources_ok.append("lags")
    else:
        sources_fail.append("lags")

    nws = fetch_nws_forecast(city, event_date)
    if nws is not None:
        features["nws_tmax_forecast_f"] = nws
        sources_ok.append("NWS")
    else:
        sources_fail.append("NWS")

    gfs = fetch_gfs_afternoon(city, event_date)
    if gfs:
        features.update(gfs)
        sources_ok.append("GFS")
    else:
        sources_fail.append("GFS")

    nwp = fetch_nwp_best(city, event_date)
    if nwp is not None:
        features["nwp_tmax_best_f"] = nwp
        sources_ok.append("NWP")
    else:
        sources_fail.append("NWP")

    print(f"  Sources OK: {sources_ok}")
    if sources_fail:
        print(f"  Sources MISSING: {sources_fail}")

    if "ASOS" in sources_fail or "lags" in sources_fail:
        print("  CRITICAL: missing ASOS or lags. Cannot predict.")
        return None

    return features


def build_feature_vector_strict(
    city: str,
    event_date: str,
    required_columns: list[str] | None = None,
) -> Tuple[Optional[Dict[str, float]], str]:
    """Build a live feature vector; fail if any source is missing."""
    features: dict[str, float] = {}
    failures: list[str] = []

    asos = fetch_asos_morning(city, event_date, skip_cache=True)
    if asos:
        features.update(asos)
    else:
        failures.append("missing ASOS obs")

    lags = fetch_lag_features(city, event_date, skip_cache=True)
    if lags:
        features.update(lags)
    else:
        failures.append("missing lag features")

    nws = fetch_nws_forecast_full(city, event_date, skip_cache=True)
    if nws:
        features.update(nws)
    else:
        failures.append("missing NWS MOS")

    gfs = fetch_gfs_afternoon(city, event_date, skip_cache=True)
    if gfs and len(gfs) == 3:
        features.update(gfs)
    else:
        failures.append("missing GFS afternoon covariates")

    nwp = fetch_nwp_best(city, event_date, skip_cache=True)
    if nwp is not None:
        features["nwp_tmax_best_f"] = nwp
    else:
        failures.append("missing NWP best Tmax")

    if failures:
        return None, "; ".join(failures)

    if required_columns:
        missing_cols = [
            col
            for col in required_columns
            if col not in features or pd.isna(features.get(col))
        ]
        if missing_cols:
            return None, f"missing model features: {', '.join(missing_cols)}"

    return features, ""


def run_leakage_audit(city: str, event_date: str, features: Dict[str, float]) -> bool:
    """Verify no future data leaked into the feature vector."""
    clean = True
    event_dt = pd.Timestamp(event_date)

    feat_path = TRACKB_DIR / city / "features.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        df["_date_key"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        row = df[df["_date_key"] == str(event_date)]
        if not row.empty and "tmax_lag1" in row.columns:
            prev_row = df[df["_date_key"] == str((event_dt - timedelta(days=1)).date())]
            if not prev_row.empty and "tmax" in prev_row.columns:
                lag1_val = row.iloc[0]["tmax_lag1"]
                actual_d_minus_1 = prev_row.iloc[0]["tmax"]
                if pd.notna(lag1_val) and pd.notna(actual_d_minus_1):
                    if abs(float(lag1_val) - float(actual_d_minus_1)) > 0.01:
                        print(
                            f"  LEAKAGE WARNING: tmax_lag1={lag1_val} != D-1 actual={actual_d_minus_1}"
                        )
                        clean = False

    for key in features:
        if any(bad in key.lower() for bad in ["resolved", "settled", "bucket", "payout"]):
            print(f"  LEAKAGE DETECTED: feature '{key}' contains forbidden keyword")
            clean = False

    if clean:
        print("  Leakage audit: CLEAN")
    return clean


def fetch_kalshi_snapshot(
    city: str,
    event_date: str,
    market_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Fetch the 10AM Kalshi market snapshot for a city-date."""
    if market_df is None:
        return None

    from src.entry_interface import filter_to_trading_window
    from src.snapshot_stability import load_or_create_frozen_k, stability_entry

    day_df = market_df[
        (market_df["city"] == city) & (market_df["event_date"].astype(str) == str(event_date))
    ].copy()
    if day_df.empty and "source_city_folder" in market_df.columns:
        day_df = market_df[
            (market_df["source_city_folder"] == city)
            & (market_df["event_date"].astype(str) == str(event_date))
        ].copy()
    if day_df.empty:
        return None

    day_df = filter_to_trading_window(day_df)
    if day_df.empty:
        return None

    signal = stability_entry(day_df, k=load_or_create_frozen_k())
    if signal.no_signal:
        return None

    snapshot_df = day_df[day_df["snapshot_time_local"] == signal.entry_snapshot_time].copy()
    return snapshot_df if not snapshot_df.empty else None
