"""Build multi-lead-time feature tables for NGBoost distributional regression."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
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

from .build_asos_features import (
    ASOS_FEATURE_COLUMNS,
    aggregate_morning_asos,
    fetch_asos_range,
    latest_value_at_or_before,
    load_cached_asos,
    row_cloud_cover,
    wind_components,
)
from .build_calendar_lag_features import CALENDAR_LAG_COLUMNS, build_calendar_lag_features
from .build_trackB_features import (
    _build_nwp_best_column,
    _gfs_raw_dir,
    _load_nws_forecasts,
    _load_openmeteo_nwp,
)
from .fetch_cli_target import fetch_cli_target
from .fetch_gfs_herbie import (
    GFS_12Z_COLUMNS,
    GFS_FEATURE_COLUMNS,
    GFS_T1_COLUMNS,
    build_gfs_features,
    build_gfs_features_custom,
    fetch_gfs_12z_nowcast,
    fetch_gfs_t1_afternoon,
)
from .fetch_nws_forecast import _issued_before_for_target
from .fetch_openmeteo_nwp import DEFAULT_OUTPUT_PATH as OPENMETEO_NWP_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKB_DIR = PROJECT_ROOT / "data" / "trackb"
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
NGBOOST_DIR = PROJECT_ROOT / "data" / "ngboost"
RAW_DIR = TRACKJ_DIR / "raw"

LEAD_TIMES = ["t1", "t2", "t3"]
ASOS_MORNING_COLUMNS = [column for column in ASOS_FEATURE_COLUMNS if column != "temp_lag1"]
AFTERNOON_ASOS_COLUMNS = [
    "temp_14",
    "temp_max_so_far_00_14",
    "dewpoint_14",
    "rh_14",
    "pressure_14",
    "wind_u_14",
    "wind_v_14",
    "cloud_cover_14",
    "cloud_cover_mean_10_14",
    "warming_rate_10_14",
]
NWS_COLUMNS = ["nws_tmax_forecast_f"]

T1_COLUMNS = [
    "doy_sin",
    "doy_cos",
    "tmax_lag1",
    "tmax_lag2",
    "tmax_lag3",
    "tmax_lag7",
    "tmax_rollmean_7",
    "tmax_rollmean_30",
    "nws_tmax_forecast_f",
    "nwp_tmax_best_f",
    *GFS_T1_COLUMNS,
]
T2_COLUMNS = [
    *T1_COLUMNS[:8],
    "nws_tmax_forecast_f",
    "nwp_tmax_best_f",
    *ASOS_MORNING_COLUMNS,
    *GFS_FEATURE_COLUMNS,
]
T3_COLUMNS = [
    *T2_COLUMNS,
    *AFTERNOON_ASOS_COLUMNS,
    *GFS_12Z_COLUMNS,
]

LEAD_FEATURE_COLUMNS = {
    "t1": T1_COLUMNS,
    "t2": T2_COLUMNS,
    "t3": T3_COLUMNS,
}

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


def _asos_raw_dir(city: str) -> Path:
    return RAW_DIR / city


def asos_daily_max(asos_df: pd.DataFrame, date_str: str) -> float | None:
    """ASOS daily max tmpf for a full local calendar day."""
    if asos_df.empty:
        return None
    day_rows = asos_df[asos_df["date"].astype(str) == str(date_str)]
    if day_rows.empty:
        return None
    tmpf = pd.to_numeric(day_rows["tmpf"], errors="coerce").dropna()
    if tmpf.empty:
        return None
    daily_max = float(tmpf.max())
    if daily_max < -30 or daily_max > 140:
        return None
    return daily_max


def build_asos_daily_max_map(asos_df: pd.DataFrame) -> dict[str, float]:
    if asos_df.empty:
        return {}
    result: dict[str, float] = {}
    for date_str, group in asos_df.groupby(asos_df["date"].astype(str)):
        value = asos_daily_max(asos_df, str(date_str))
        if value is not None:
            result[str(date_str)] = value
    return result


def aggregate_afternoon_asos(
    asos: pd.DataFrame,
    target_dates: pd.Series | list[str],
    cutoff_hour: int = 14,
    morning_temp_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate ASOS observations from 00:00 through cutoff_hour local on event date D."""
    dates = pd.Series(target_dates, dtype="string").dropna().drop_duplicates().sort_values()
    cutoff_time = f"{cutoff_hour:02d}:00"
    if asos.empty:
        return pd.DataFrame({"date": dates, **{column: np.nan for column in AFTERNOON_ASOS_COLUMNS}})
    df = asos.copy()
    for column in ["tmpf", "dwpf", "relh", "drct", "sknt", "mslp"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["cloud_cover"] = df.apply(row_cloud_cover, axis=1)
    df["wind_u"], df["wind_v"] = wind_components(df["drct"], df["sknt"])
    df = df[
        (df["valid_local"].dt.time >= datetime.strptime("00:00", "%H:%M").time())
        & (df["valid_local"].dt.time <= datetime.strptime(cutoff_time, "%H:%M").time())
    ].copy()
    grouped = {key: group.sort_values("valid_local") for key, group in df.groupby("date")}
    rows: list[dict] = []
    for date_value in dates:
        group = grouped.get(str(date_value), pd.DataFrame(columns=df.columns))
        temp_10am = morning_temp_map.get(str(date_value)) if morning_temp_map else None
        if temp_10am is None and not group.empty:
            temp_10am = latest_value_at_or_before(group, str(date_value), "tmpf", cutoff_time="10:00")
        temp_14 = latest_value_at_or_before(group, str(date_value), "tmpf", cutoff_time=cutoff_time) if not group.empty else None
        cloud_10_14 = group[
            group["valid_local"].dt.time >= datetime.strptime("10:00", "%H:%M").time()
        ] if not group.empty else pd.DataFrame()
        rows.append(
            {
                "date": str(date_value),
                "temp_14": temp_14,
                "temp_max_so_far_00_14": group["tmpf"].max() if not group.empty else np.nan,
                "dewpoint_14": latest_value_at_or_before(group, str(date_value), "dwpf", cutoff_time=cutoff_time) if not group.empty else np.nan,
                "rh_14": latest_value_at_or_before(group, str(date_value), "relh", cutoff_time=cutoff_time) if not group.empty else np.nan,
                "pressure_14": latest_value_at_or_before(group, str(date_value), "mslp", cutoff_time=cutoff_time) if not group.empty else np.nan,
                "wind_u_14": latest_value_at_or_before(group, str(date_value), "wind_u", cutoff_time=cutoff_time) if not group.empty else np.nan,
                "wind_v_14": latest_value_at_or_before(group, str(date_value), "wind_v", cutoff_time=cutoff_time) if not group.empty else np.nan,
                "cloud_cover_14": latest_value_at_or_before(group, str(date_value), "cloud_cover", cutoff_time=cutoff_time) if not group.empty else np.nan,
                "cloud_cover_mean_10_14": cloud_10_14["cloud_cover"].mean() if not cloud_10_14.empty else np.nan,
                "warming_rate_10_14": (
                    (float(temp_14) - float(temp_10am)) / 4.0
                    if pd.notna(temp_14) and pd.notna(temp_10am)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)[["date", *AFTERNOON_ASOS_COLUMNS]]


def _cli_tmax_valid(cli: pd.DataFrame, check_str: str) -> bool:
    row = cli[cli["date"].astype(str) == check_str]
    if row.empty:
        return False
    return pd.notna(pd.to_numeric(row.iloc[0]["tmax_f"], errors="coerce"))


def _cli_tmax(cli: pd.DataFrame, check_str: str) -> float | None:
    row = cli[cli["date"].astype(str) == check_str]
    if row.empty:
        return None
    val = pd.to_numeric(row.iloc[0]["tmax_f"], errors="coerce")
    return float(val) if pd.notna(val) else None


def apply_tmax_lag1_override(
    merged: pd.DataFrame,
    lead_time: str,
    cli: pd.DataFrame,
    asos_daily_max_map: dict[str, float],
) -> pd.DataFrame:
    """Override tmax_lag1 per lead-time leakage rules."""
    result = merged.copy()
    lag1_values: list[float | None] = []
    for _, row in result.iterrows():
        event_str = str(row["date"])
        event_dt = pd.Timestamp(event_str)
        prev_str = (event_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        if lead_time == "t1":
            lag1 = asos_daily_max_map.get(prev_str)
        else:
            if _cli_tmax_valid(cli, prev_str):
                lag1 = _cli_tmax(cli, prev_str)
            else:
                lag1 = asos_daily_max_map.get(prev_str)
        lag1_values.append(lag1)
    result["tmax_lag1"] = lag1_values
    return result


def _filter_nws_for_lead(nws: pd.DataFrame, city_config: dict, issued_before_hour: int = 20) -> pd.DataFrame:
    if nws.empty:
        return nws
    local_tz = ZoneInfo(str(city_config["timezone"]))
    filtered = nws.copy()
    issued = pd.to_datetime(filtered.get("issued_time"), utc=True, errors="coerce")
    keep: list[bool] = []
    for idx in filtered.index:
        target_dt = pd.to_datetime(filtered.at[idx, "date"], errors="coerce")
        if pd.isna(target_dt):
            keep.append(False)
            continue
        cutoff = _issued_before_for_target(target_dt.date(), issued_before_hour, local_tz)
        issue = issued.at[idx] if idx in issued.index else pd.NaT
        keep.append(pd.notna(issue) and issue < cutoff)
    filtered = filtered.loc[keep].copy()
    return filtered[["date", *NWS_COLUMNS, "issued_time"]]


def assert_no_leakage(
    merged: pd.DataFrame,
    city: str,
    lead_time: str,
    city_config: dict,
    asos_daily_max_map: dict[str, float],
    cli: pd.DataFrame | None = None,
) -> bool:
    """Verify no future data leaked into features. Prints warnings; returns False if any fail."""
    if merged.empty:
        return True
    clean = True
    local_tz = ZoneInfo(str(city_config["timezone"]))
    forbidden_morning = set(ASOS_MORNING_COLUMNS)
    forbidden_afternoon = set(AFTERNOON_ASOS_COLUMNS)

    if lead_time == "t1":
        leaked = forbidden_morning | forbidden_afternoon | set(GFS_FEATURE_COLUMNS) | set(GFS_12Z_COLUMNS)
        for col in leaked:
            if col in merged.columns and merged[col].notna().any():
                print(f"  LEAKAGE WARNING [{city}/{lead_time}]: forbidden column {col} has non-null values")
                clean = False
        for _, row in merged.iterrows():
            event_str = str(row["date"])
            prev_str = (pd.Timestamp(event_str) - timedelta(days=1)).strftime("%Y-%m-%d")
            lag1 = row.get("tmax_lag1")
            asos_max = asos_daily_max_map.get(prev_str)
            cli_val = _cli_tmax(cli, prev_str) if cli is not None else None
            if pd.notna(lag1) and asos_max is not None and abs(float(lag1) - asos_max) > 0.01:
                print(f"  LEAKAGE WARNING [{city}/{lead_time}]: tmax_lag1 != ASOS D-1 max on {event_str}")
                clean = False
            if (
                pd.notna(lag1)
                and cli_val is not None
                and asos_max is not None
                and abs(float(lag1) - cli_val) < 0.01
                and abs(float(lag1) - asos_max) > 0.01
            ):
                print(f"  LEAKAGE WARNING [{city}/{lead_time}]: tmax_lag1 equals CLI D-1 but not ASOS on {event_str}")
                clean = False

    if lead_time == "t2":
        for col in forbidden_afternoon | set(GFS_12Z_COLUMNS):
            if col in merged.columns and merged[col].notna().any():
                print(f"  LEAKAGE WARNING [{city}/{lead_time}]: forbidden column {col} has non-null values")
                clean = False

    if "nws_tmax_forecast_f" in merged.columns and "issued_time" in merged.columns:
        issued = pd.to_datetime(merged["issued_time"], utc=True, errors="coerce")
        for _, row in merged.iterrows():
            target_dt = pd.Timestamp(str(row["date"]))
            cutoff = _issued_before_for_target(target_dt.date(), 20, local_tz)
            issue = issued.loc[row.name] if row.name in issued.index else pd.NaT
            if pd.notna(issue) and issue >= cutoff:
                print(f"  LEAKAGE WARNING [{city}/{lead_time}]: NWS issued after D-1 20:00 on {row['date']}")
                clean = False

    if "nwp_tmax_best_f" in merged.columns and "nwp_issued_date" in merged.columns:
        target = pd.to_datetime(merged["date"], errors="coerce")
        nwp_mask = merged["nwp_tmax_best_f"].notna()
        if nwp_mask.any():
            issued_dates = pd.to_datetime(merged.loc[nwp_mask, "nwp_issued_date"], errors="coerce")
            target_sub = target.loc[nwp_mask]
            if (issued_dates.dt.date >= target_sub.dt.date).any():
                print(f"  LEAKAGE WARNING [{city}/{lead_time}]: NWP issued_date >= target date")
                clean = False

    sample = merged.sample(n=min(10, len(merged)), random_state=42)
    for _, row in sample.iterrows():
        lag1 = row.get("tmax_lag1")
        target = row.get("tmax_f")
        if pd.notna(lag1) and pd.notna(target) and abs(float(lag1) - float(target)) < 0.01:
            print(f"  LEAKAGE WARNING [{city}/{lead_time}]: tmax_lag1 == target on {row['date']}")
            clean = False

    if clean:
        print(f"  Leakage audit [{city}/{lead_time}]: CLEAN")
    return clean


def _shared_city_data(
    city: str,
    start_date: date,
    end_date: date,
    no_fetch: bool = False,
) -> dict:
    """Load shared inputs for all lead times for one city."""
    city_config = _load_city_config(city)
    trackj_city_dir = TRACKJ_DIR / city
    lag_start = start_date - timedelta(days=45)
    cli_path = trackj_city_dir / "cli_target.parquet"
    if no_fetch and cli_path.exists():
        cli_target = pd.read_parquet(cli_path)
    else:
        cli_target = fetch_cli_target(
            city_config, lag_start, end_date, RAW_DIR, TRACKJ_DIR, no_fetch=no_fetch
        )
    cli_target["date"] = pd.to_datetime(cli_target["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    city_asos_dir = _asos_raw_dir(city)
    if not no_fetch:
        fetch_asos_range(city_config, lag_start, end_date, city_asos_dir, sleep_seconds=1.1)
    asos = load_cached_asos(city_asos_dir, city_config["nws_station"], lag_start, end_date)

    calendar_lags = build_calendar_lag_features(cli_target)
    asos_daily_max_map = build_asos_daily_max_map(asos)

    event_dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    morning_asos = aggregate_morning_asos(asos, event_dates, target_df=None)
    morning_temp_map = {
        str(row["date"]): float(row["temp_10am"])
        for _, row in morning_asos.iterrows()
        if pd.notna(row.get("temp_10am"))
    }
    afternoon_asos = aggregate_afternoon_asos(
        asos, event_dates, cutoff_hour=14, morning_temp_map=morning_temp_map
    )

    nws_path = TRACKB_DIR / "nws_forecasts_raw.parquet"
    nws = _load_nws_forecasts(nws_path, city)
    nws = _filter_nws_for_lead(nws, city_config, issued_before_hour=20)

    nwp_path = OPENMETEO_NWP_PATH
    ecmwf_nwp = _load_openmeteo_nwp(nwp_path, city, "ecmwf_ifs025")
    gfs_nwp = _load_openmeteo_nwp(nwp_path, city, "gfs_seamless")

    gfs_raw = _gfs_raw_dir(city_config, PROJECT_ROOT / "data" / "raw")

    return {
        "city_config": city_config,
        "cli_target": cli_target,
        "calendar_lags": calendar_lags,
        "asos_daily_max_map": asos_daily_max_map,
        "morning_asos": morning_asos,
        "afternoon_asos": afternoon_asos,
        "nws": nws,
        "ecmwf_nwp": ecmwf_nwp,
        "gfs_nwp": gfs_nwp,
        "gfs_raw": gfs_raw,
        "event_dates": event_dates,
    }


def build_t1_features(
    city: str,
    event_date: str,
    city_config: dict | None = None,
    shared: dict | None = None,
) -> dict | None:
    shared = shared or _shared_city_data(city, date.fromisoformat(event_date), date.fromisoformat(event_date))
    city_config = city_config or shared["city_config"]
    calendar = shared["calendar_lags"]
    row = calendar[calendar["date"].astype(str) == str(event_date)]
    if row.empty:
        return None
    features = row.iloc[0][CALENDAR_LAG_COLUMNS].to_dict()
    merged = pd.DataFrame([{"date": event_date, **features}])
    merged = apply_tmax_lag1_override(merged, "t1", shared["cli_target"], shared["asos_daily_max_map"])
    features = merged.iloc[0].to_dict()

    nws_row = shared["nws"][shared["nws"]["date"].astype(str) == str(event_date)]
    if not nws_row.empty:
        features["nws_tmax_forecast_f"] = float(nws_row.iloc[0]["nws_tmax_forecast_f"])

    nwp_base = pd.DataFrame([{"date": event_date}])
    nwp_merged = _build_nwp_best_column(nwp_base, shared["ecmwf_nwp"], shared["gfs_nwp"])
    if pd.notna(nwp_merged.iloc[0].get("nwp_tmax_best_f")):
        features["nwp_tmax_best_f"] = float(nwp_merged.iloc[0]["nwp_tmax_best_f"])

    gfs_features, _ = fetch_gfs_t1_afternoon(
        date.fromisoformat(event_date),
        raw_dir=shared["gfs_raw"],
        city_config=city_config,
    )
    features.update({k: gfs_features[k] for k in GFS_T1_COLUMNS if k in gfs_features})
    return features


def build_t2_features(
    city: str,
    event_date: str,
    city_config: dict | None = None,
    shared: dict | None = None,
) -> dict | None:
    shared = shared or _shared_city_data(city, date.fromisoformat(event_date), date.fromisoformat(event_date))
    features = build_t1_features(city, event_date, city_config=city_config, shared=shared)
    if features is None:
        return None
    for col in GFS_T1_COLUMNS:
        features.pop(col, None)

    morning = shared["morning_asos"][shared["morning_asos"]["date"].astype(str) == str(event_date)]
    if not morning.empty:
        for col in ASOS_MORNING_COLUMNS:
            val = morning.iloc[0].get(col)
            if pd.notna(val):
                features[col] = float(val)

    from .fetch_gfs_herbie import fetch_gfs_for_date

    gfs_features, _ = fetch_gfs_for_date(
        date.fromisoformat(event_date),
        raw_dir=shared["gfs_raw"],
        cutoff_hour=10,
        city_config=shared["city_config"],
    )
    features.update({k: gfs_features[k] for k in GFS_FEATURE_COLUMNS if k in gfs_features})

    merged = pd.DataFrame([{"date": event_date, **features, "tmax_f": np.nan}])
    merged = apply_tmax_lag1_override(merged, "t2", shared["cli_target"], shared["asos_daily_max_map"])
    return merged.iloc[0].drop(labels=["tmax_f"]).to_dict()


def build_t3_features(
    city: str,
    event_date: str,
    city_config: dict | None = None,
    shared: dict | None = None,
) -> dict | None:
    shared = shared or _shared_city_data(city, date.fromisoformat(event_date), date.fromisoformat(event_date))
    features = build_t2_features(city, event_date, city_config=city_config, shared=shared)
    if features is None:
        return None

    afternoon = shared["afternoon_asos"][shared["afternoon_asos"]["date"].astype(str) == str(event_date)]
    if not afternoon.empty:
        for col in AFTERNOON_ASOS_COLUMNS:
            val = afternoon.iloc[0].get(col)
            if pd.notna(val):
                features[col] = float(val)

    gfs_12z, _ = fetch_gfs_12z_nowcast(
        date.fromisoformat(event_date),
        raw_dir=shared["gfs_raw"],
        cutoff_hour=14,
        city_config=shared["city_config"],
    )
    features.update({k: gfs_12z[k] for k in GFS_12Z_COLUMNS if k in gfs_12z})
    return features


def build_feature_table(
    city: str,
    lead_time: str,
    start_date: date,
    end_date: date,
    no_fetch: bool = False,
    fetch_gfs: bool = True,
) -> pd.DataFrame:
    """Build feature table for one city and lead time."""
    if lead_time not in LEAD_TIMES:
        raise ValueError(f"Unknown lead_time: {lead_time}")

    print(f"Building {city} {lead_time}: {start_date} to {end_date}")
    shared = _shared_city_data(city, start_date, end_date, no_fetch=no_fetch)
    cli = shared["cli_target"].copy()
    cli["date"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    valid_cli = cli[cli["tmax_f"].notna()].copy()
    valid_cli = valid_cli[
        (pd.to_datetime(valid_cli["date"]) >= pd.Timestamp(start_date))
        & (pd.to_datetime(valid_cli["date"]) <= pd.Timestamp(end_date))
    ]
    base = valid_cli[["date", "tmax_f"]].copy()

    merged = base.merge(shared["calendar_lags"], on="date", how="left")
    merged = apply_tmax_lag1_override(merged, lead_time, cli, shared["asos_daily_max_map"])

    nws = shared["nws"]
    merged = merged.merge(nws[["date", *NWS_COLUMNS, "issued_time"]], on="date", how="left")
    merged = _build_nwp_best_column(merged, shared["ecmwf_nwp"], shared["gfs_nwp"])

    if lead_time in ("t2", "t3"):
        morning = shared["morning_asos"][["date", *ASOS_MORNING_COLUMNS]]
        merged = merged.merge(morning, on="date", how="left")

    if lead_time == "t3":
        afternoon = shared["afternoon_asos"][["date", *AFTERNOON_ASOS_COLUMNS]]
        merged = merged.merge(afternoon, on="date", how="left")

    dates = merged["date"].tolist()
    city_config = shared["city_config"]
    gfs_raw = shared["gfs_raw"]

    if lead_time == "t1":
        gfs_df = build_gfs_features_custom(
            dates,
            fetch_gfs_t1_afternoon,
            GFS_T1_COLUMNS,
            raw_dir=gfs_raw,
            fetch=fetch_gfs,
            city_config=city_config,
            cache_suffix="_t1",
        )
        merged = merged.merge(gfs_df, on="date", how="left")
    elif lead_time == "t2":
        gfs_df, _ = build_gfs_features(
            dates,
            raw_dir=gfs_raw,
            fetch=fetch_gfs,
            cutoff_hour=10,
            city_config=city_config,
        )
        merged = merged.merge(gfs_df, on="date", how="left")
    elif lead_time == "t3":
        gfs_df, _ = build_gfs_features(
            dates,
            raw_dir=gfs_raw,
            fetch=fetch_gfs,
            cutoff_hour=10,
            city_config=city_config,
        )
        merged = merged.merge(gfs_df, on="date", how="left")
        gfs_12z_df = build_gfs_features_custom(
            dates,
            fetch_gfs_12z_nowcast,
            GFS_12Z_COLUMNS,
            raw_dir=gfs_raw,
            fetch=fetch_gfs,
            city_config=city_config,
            cache_suffix="_t3_12z",
        )
        merged = merged.merge(gfs_12z_df, on="date", how="left")

    feature_cols = LEAD_FEATURE_COLUMNS[lead_time]
    output_cols = ["date", "tmax_f", *feature_cols]
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    result = merged[output_cols].sort_values("date").reset_index(drop=True)

    assert_no_leakage(result, city, lead_time, city_config, shared["asos_daily_max_map"], cli=cli)
    print(f"  {city} {lead_time}: {len(result)} rows, {result['date'].min()} to {result['date'].max()}")
    return result


def compare_t2_to_trackb(city: str, sample_date: str | None = None, tolerance: float = 1e-4) -> None:
    """Print comparison of t2 features vs Track-B features.parquet."""
    trackb_path = TRACKB_DIR / city / "features.parquet"
    ngboost_path = NGBOOST_DIR / "features" / city / "t2.parquet"
    if not trackb_path.exists():
        print(f"  Track-B features not found for {city}; skipping comparison")
        return
    if not ngboost_path.exists():
        print(f"  NGBoost t2 features not found for {city}; skipping comparison")
        return
    trackb = pd.read_parquet(trackb_path)
    ngboost = pd.read_parquet(ngboost_path)
    trackb["date"] = pd.to_datetime(trackb["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ngboost["date"] = pd.to_datetime(ngboost["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if sample_date is None:
        sample_date = str(ngboost["date"].iloc[len(ngboost) // 2])
    tb_row = trackb[trackb["date"] == sample_date]
    ng_row = ngboost[ngboost["date"] == sample_date]
    if tb_row.empty or ng_row.empty:
        print(f"  No matching row for {city} on {sample_date}")
        return
    print(f"  t2 vs Track-B [{city}] on {sample_date}:")
    for col in T2_COLUMNS:
        ng_val = ng_row.iloc[0].get(col)
        tb_col = "tmax" if col == "tmax_f" else col
        tb_val = tb_row.iloc[0].get(tb_col)
        if pd.isna(ng_val) and pd.isna(tb_val):
            continue
        if pd.isna(ng_val) or pd.isna(tb_val):
            print(f"    {col}: ngboost={ng_val} trackb={tb_val} (one missing)")
            continue
        diff = abs(float(ng_val) - float(tb_val))
        status = "OK" if diff <= tolerance else f"DIFF={diff:.6f}"
        if diff > tolerance:
            print(f"    {col}: ngboost={ng_val} trackb={tb_val} {status}")


def run_verification(cities: list[str]) -> None:
    """Run post-build sanity checks (print only)."""
    print("\n=== NGBOOST FEATURE VERIFICATION ===")
    for city in cities:
        t1_path = NGBOOST_DIR / "features" / city / "t1.parquet"
        t2_path = NGBOOST_DIR / "features" / city / "t2.parquet"
        t3_path = NGBOOST_DIR / "features" / city / "t3.parquet"

        if t1_path.exists():
            t1 = pd.read_parquet(t1_path)
            leaked = [c for c in ASOS_MORNING_COLUMNS + AFTERNOON_ASOS_COLUMNS if c in t1.columns]
            print(f"  t1 [{city}]: ASOS morning cols present = {leaked} (expect none)")

        if t3_path.exists():
            t3 = pd.read_parquet(t3_path)
            has_14 = "temp_14" in t3.columns and "temp_max_so_far_00_14" in t3.columns
            print(f"  t3 [{city}]: temp_14 cols present = {has_14}")

        compare_t2_to_trackb(city)

        for lead, path in [("t1", t1_path), ("t2", t2_path), ("t3", t3_path)]:
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            sample = df.sample(n=min(10, len(df)), random_state=42)
            leaks = 0
            for _, row in sample.iterrows():
                if pd.notna(row.get("tmax_lag1")) and pd.notna(row.get("tmax_f")):
                    if abs(float(row["tmax_lag1"]) - float(row["tmax_f"])) < 0.01:
                        leaks += 1
            print(f"  lag1==target sample leaks [{city}/{lead}]: {leaks}/10")

        trackb_path = TRACKB_DIR / city / "features.parquet"
        if t2_path.exists() and trackb_path.exists():
            t2_n = len(pd.read_parquet(t2_path))
            tb_n = len(pd.read_parquet(trackb_path))
            print(f"  row counts [{city}]: t2={t2_n} trackb={tb_n}")
