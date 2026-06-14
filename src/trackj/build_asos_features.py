from __future__ import annotations

import calendar
import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .hf_data_store import sync_city_to_hf


IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ASOS_FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "mslp", "alti", "vsby", "gust", "feel", "p01i", "skyc1", "skyc2", "skyc3", "skyc4", "metar"]
ASOS_FEATURE_COLUMNS = [
    "temp_10am",
    "temp_mean_00_10",
    "temp_max_so_far_00_10",
    "dewpoint_10am",
    "rh_mean_00_10",
    "pressure_10am",
    "wind_u_mean_00_10",
    "wind_v_mean_00_10",
    "cloud_cover_mean_00_10",
    "temp_lag1",
]


HTTP_USER_AGENT = "postproc_utils/1.0 (research)"
LIVE_FETCH_TIMEOUT_SECONDS = 10


def make_session() -> requests.Session:
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.headers.update({"User-Agent": HTTP_USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def month_starts(start_date: date, end_date: date) -> list[date]:
    cursor = date(start_date.year, start_date.month, 1)
    starts = []
    while cursor <= end_date:
        starts.append(cursor)
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
    return starts


def month_window(month_start: date, start_date: date, end_date: date) -> tuple[datetime, datetime]:
    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    next_month = date(month_start.year + (month_start.month == 12), 1 if month_start.month == 12 else month_start.month + 1, 1)
    return datetime.combine(max(month_start, start_date), datetime.min.time()), datetime.combine(min(next_month, end_date + timedelta(days=1)), datetime.min.time())


def raw_path_for_month(raw_dir: Path, station: str, month_start: date) -> Path:
    return raw_dir / f"{station.lower()}_asos_{month_start:%Y%m}.csv"


def fetch_asos_range(city_config: dict, start_date: date, end_date: date, raw_dir: Path, overwrite: bool = False, sleep_seconds: float = 1.1) -> list[Path]:
    session = make_session()
    station = city_config["nws_station"]
    paths = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for month_start in month_starts(start_date, end_date):
        path = raw_path_for_month(raw_dir, station, month_start)
        fetched = False
        if not path.exists() or overwrite:
            start_dt, end_dt = month_window(month_start, start_date, end_date)
            params = [
                ("station", station),
                ("tz", city_config["timezone"]),
                ("format", "onlycomma"),
                ("missing", "null"),
                ("trace", "null"),
                ("year1", str(start_dt.year)), ("month1", str(start_dt.month)), ("day1", str(start_dt.day)), ("hour1", "0"), ("minute1", "0"),
                ("year2", str(end_dt.year)), ("month2", str(end_dt.month)), ("day2", str(end_dt.day)), ("hour2", "0"), ("minute2", "0"),
            ]
            params.extend(("data", field) for field in ASOS_FIELDS)
            response = session.get(IEM_ASOS_URL, params=params, timeout=LIVE_FETCH_TIMEOUT_SECONDS)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
            row_count = max(response.text.count("\n") - 1, 0)
            print(
                f"ASOS {city_config['city']} {month_start:%Y-%m}: fetched {row_count} rows "
                f"from {response.url}"
            )
            fetched = True
        paths.append(path)
        if not fetched:
            print(f"ASOS {city_config['city']} {month_start:%Y-%m}: cached {path}")
        if fetched and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return paths


def load_cached_asos(raw_dir: Path, station: str, start_date: date, end_date: date) -> pd.DataFrame:
    frames = []
    months = {m.strftime("%Y%m") for m in month_starts(start_date, end_date)}
    for path in sorted(raw_dir.glob(f"{station.lower()}_asos_*.csv")):
        if path.stem.rsplit("_", 1)[-1] not in months:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip() or text.startswith("ERROR"):
            continue
        frame = pd.read_csv(StringIO(text), na_values=["null", "M", ""], keep_default_na=True)
        if not frame.empty:
            frame["raw_file"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station", "valid", *ASOS_FIELDS])
    df = pd.concat(frames, ignore_index=True)
    df["valid_local"] = pd.to_datetime(df["valid"], errors="coerce")
    df = df[df["valid_local"].notna()].copy()
    df["date"] = df["valid_local"].dt.strftime("%Y-%m-%d")
    return df


def cloud_code_to_fraction(code: object) -> float | None:
    if pd.isna(code):
        return None
    return {"CLR": 0.0, "SKC": 0.0, "NSC": 0.0, "NCD": 0.0, "FEW": 0.125, "SCT": 0.375, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}.get(str(code).strip().upper())


def row_cloud_cover(row: pd.Series) -> float | None:
    values = [cloud_code_to_fraction(row.get(field)) for field in ("skyc1", "skyc2", "skyc3", "skyc4")]
    numeric = [value for value in values if value is not None]
    return max(numeric) if numeric else None


def wind_components(drct: pd.Series, sknt: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(pd.to_numeric(drct, errors="coerce"))
    speed = pd.to_numeric(sknt, errors="coerce")
    u = (-speed * np.sin(radians)).mask(speed == 0, 0.0)
    v = (-speed * np.cos(radians)).mask(speed == 0, 0.0)
    return u, v


def latest_value_at_or_before(group: pd.DataFrame, date_value: str, column: str, cutoff_time: str = "10:00") -> float | None:
    candidates = group[group["valid_local"] <= pd.Timestamp(f"{date_value} {cutoff_time}")][["valid_local", column]].dropna()
    return None if candidates.empty else float(candidates.sort_values("valid_local").iloc[-1][column])


def aggregate_morning_asos(asos: pd.DataFrame, target_dates: pd.Series | list[str], target_df: pd.DataFrame | None = None) -> pd.DataFrame:
    dates = pd.Series(target_dates, dtype="string").dropna().drop_duplicates().sort_values()
    if asos.empty:
        return pd.DataFrame({"date": dates, **{column: np.nan for column in ASOS_FEATURE_COLUMNS}})
    df = asos.copy()
    for column in ["tmpf", "dwpf", "relh", "drct", "sknt", "mslp"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["cloud_cover"] = df.apply(row_cloud_cover, axis=1)
    df["wind_u"], df["wind_v"] = wind_components(df["drct"], df["sknt"])
    df = df[(df["valid_local"].dt.time >= datetime.strptime("00:00", "%H:%M").time()) & (df["valid_local"].dt.time <= datetime.strptime("10:00", "%H:%M").time())].copy()
    grouped = {key: group.sort_values("valid_local") for key, group in df.groupby("date")}
    lag_map = {}
    if target_df is not None and "tmax_f" in target_df.columns:
        t = target_df.copy()
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t = t.sort_values("date")
        lag_map = dict(zip(t["date"].dt.strftime("%Y-%m-%d"), pd.to_numeric(t["tmax_f"], errors="coerce").shift(1)))
    rows = []
    for date_value in dates:
        group = grouped.get(str(date_value), pd.DataFrame(columns=df.columns))
        rows.append({
            "date": str(date_value),
            "temp_10am": latest_value_at_or_before(group, str(date_value), "tmpf") if not group.empty else np.nan,
            "temp_mean_00_10": group["tmpf"].mean() if not group.empty else np.nan,
            "temp_max_so_far_00_10": group["tmpf"].max() if not group.empty else np.nan,
            "dewpoint_10am": latest_value_at_or_before(group, str(date_value), "dwpf") if not group.empty else np.nan,
            "rh_mean_00_10": group["relh"].mean() if not group.empty else np.nan,
            "pressure_10am": latest_value_at_or_before(group, str(date_value), "mslp") if not group.empty else np.nan,
            "wind_u_mean_00_10": group["wind_u"].mean() if not group.empty else np.nan,
            "wind_v_mean_00_10": group["wind_v"].mean() if not group.empty else np.nan,
            "cloud_cover_mean_00_10": group["cloud_cover"].mean() if not group.empty else np.nan,
            "temp_lag1": lag_map.get(str(date_value), np.nan),
        })
    return pd.DataFrame(rows)[["date", *ASOS_FEATURE_COLUMNS]]


def build_asos_features(
    city_config: dict,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    output_dir: Path,
    no_fetch: bool = False,
    target_df: pd.DataFrame | None = None,
    sleep_seconds: float = 1.1,
) -> pd.DataFrame:
    city = city_config["city"]
    city_raw_dir = Path(raw_dir) / city / "asos"
    if not no_fetch:
        fetch_asos_range(
            city_config,
            start_date,
            end_date,
            city_raw_dir,
            sleep_seconds=sleep_seconds,
        )
        if os.environ.get("TRACKJ_SKIP_HF_SYNC", "0") == "1":
            print(f"HF raw sync skipped for {city} ASOS data (TRACKJ_SKIP_HF_SYNC=1)")
        else:
            try:
                sync_city_to_hf(city, raw_dir)
            except Exception as exc:
                print(f"Warning: HF raw sync skipped for {city} ASOS data: {exc}")
    asos = load_cached_asos(city_raw_dir, city_config["nws_station"], start_date, end_date)
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    features = aggregate_morning_asos(asos, dates, target_df=target_df)
    city_output = Path(output_dir) / city
    city_output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(city_output / "asos_features.parquet", index=False)
    return features
