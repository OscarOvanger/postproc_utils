from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from dateutil.tz import gettz

    def ZoneInfo(name: str):
        tz = gettz(name)
        if tz is None:
            raise ValueError(f"Unknown timezone: {name}")
        return tz

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
DEFAULT_OUTPUT_PATH = Path("data/trackb/nws_forecasts_raw.parquet")
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
NBE_START = date(2020, 7, 23)
NBS_START = date(2018, 11, 7)


def make_session() -> requests.Session:
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.headers.update({"User-Agent": "mcp-trackj-nws-forecast/1.0 (research)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parse_issued_before(issued_before: str | datetime) -> datetime:
    if isinstance(issued_before, datetime):
        return issued_before if issued_before.tzinfo else issued_before.replace(tzinfo=ZoneInfo("UTC"))
    parsed = pd.to_datetime(issued_before, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse issued_before: {issued_before}")
    return parsed.to_pydatetime()


def _mos_model_for_date(target_date: date) -> str:
    if target_date >= NBE_START:
        return "NBE"
    if target_date >= NBS_START:
        return "NBS"
    return "GFS"


def _fetch_iem_mos_table(station: str, model: str, start_dt: datetime, end_dt: datetime, session: requests.Session) -> pd.DataFrame:
    params = {
        "station": station,
        "model": model,
        "year1": start_dt.year,
        "month1": start_dt.month,
        "day1": start_dt.day,
        "hour1": start_dt.hour,
        "year2": end_dt.year,
        "month2": end_dt.month,
        "day2": end_dt.day,
        "hour2": end_dt.hour,
    }
    response = session.get(IEM_MOS_URL, params=params, timeout=90)
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.upper().startswith("ERROR"):
        return pd.DataFrame()
    from io import StringIO

    frame = pd.read_csv(StringIO(text), na_values=["", "M", "null"], keep_default_na=True)
    if frame.empty:
        return frame
    frame["runtime"] = pd.to_datetime(frame["runtime"], utc=True, errors="coerce")
    frame["ftime"] = pd.to_datetime(frame["ftime"], utc=True, errors="coerce")
    return frame.dropna(subset=["runtime", "ftime"])


def _extract_tmax_from_mos(frame: pd.DataFrame, target_date: date, issued_before: datetime) -> dict | None:
    if frame.empty or "txn" not in frame.columns:
        return None
    cutoff = issued_before if issued_before.tzinfo else issued_before.replace(tzinfo=ZoneInfo("UTC"))
    eligible = frame[frame["runtime"] < cutoff].copy()
    if eligible.empty:
        return None
    eligible["ftime_date"] = eligible["ftime"].dt.date
    day_rows = eligible[eligible["ftime_date"].eq(target_date)].copy()
    if day_rows.empty:
        return None
    midnight_rows = day_rows[day_rows["ftime"].dt.hour.eq(0)]
    selection = midnight_rows if not midnight_rows.empty else day_rows
    latest_runtime = selection["runtime"].max()
    latest_day = selection[selection["runtime"].eq(latest_runtime)].sort_values("ftime")
    row = latest_day.iloc[0]
    tmax = pd.to_numeric(row.get("txn"), errors="coerce")
    if pd.isna(tmax):
        afternoon = latest_day[latest_day["ftime"].dt.hour.ge(12)]
        if not afternoon.empty:
            tmax = pd.to_numeric(afternoon["tmp"], errors="coerce").max()
    if pd.isna(tmax):
        return None
    issued_time = latest_runtime.isoformat().replace("+00:00", "Z")
    hours_since = (cutoff - latest_runtime).total_seconds() / 3600.0
    return {
        "tmax_forecast_f": float(tmax),
        "issued_time": issued_time,
        "valid_date": target_date.isoformat(),
        "hours_since_issuance": float(hours_since),
    }


def _fetch_live_nws_tmax(lat: float, lon: float, target_date: date, issued_before: datetime, session: requests.Session) -> dict | None:
    response = session.get(NWS_POINTS_URL.format(lat=lat, lon=lon), timeout=60, allow_redirects=True)
    if response.status_code >= 400:
        return None
    payload = response.json()
    forecast_url = payload.get("properties", {}).get("forecast")
    if not forecast_url:
        return None
    forecast_response = session.get(forecast_url, timeout=60)
    forecast_response.raise_for_status()
    periods = forecast_response.json().get("properties", {}).get("periods", [])
    target_str = target_date.isoformat()
    for period in periods:
        start = pd.to_datetime(period.get("startTime"), utc=True, errors="coerce")
        if pd.isna(start) or start.date() != target_date:
            continue
        if not period.get("isDaytime", True):
            continue
        temp = period.get("temperature")
        if temp is None:
            continue
        issued = pd.to_datetime(forecast_response.json().get("properties", {}).get("updateTime"), utc=True, errors="coerce")
        if pd.isna(issued) or issued >= pd.Timestamp(issued_before):
            return None
        hours_since = (pd.Timestamp(issued_before) - issued).total_seconds() / 3600.0
        return {
            "tmax_forecast_f": float(temp),
            "issued_time": issued.isoformat().replace("+00:00", "Z"),
            "valid_date": target_str,
            "hours_since_issuance": float(hours_since),
        }
    return None


def fetch_nws_tmax_forecast(
    lat: float,
    lon: float,
    target_date: date | str,
    issued_before: str | datetime,
    *,
    station: str | None = None,
    session: requests.Session | None = None,
) -> dict | None:
    """Fetch NWS Tmax forecast for lat/lon valid on target_date, issued strictly before issued_before."""
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    issued_cutoff = _parse_issued_before(issued_before)
    session = session or make_session()
    today = datetime.now(ZoneInfo("UTC")).date()
    if target_date >= today and station is None:
        return _fetch_live_nws_tmax(lat, lon, target_date, issued_cutoff, session)
    if station is None:
        return None
    model = _mos_model_for_date(target_date)
    query_start = datetime.combine(target_date - timedelta(days=3), datetime.min.time()).replace(tzinfo=ZoneInfo("UTC"))
    query_end = datetime.combine(target_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=ZoneInfo("UTC"))
    frame = _fetch_iem_mos_table(station, model, query_start, query_end, session)
    if frame.empty and model == "NBE":
        frame = _fetch_iem_mos_table(station, "NBS", query_start, query_end, session)
    return _extract_tmax_from_mos(frame, target_date, issued_cutoff)


def _issued_before_for_target(target_date: date, issued_before_hour: int, local_tz: ZoneInfo) -> datetime:
    prior_day = target_date - timedelta(days=1)
    local_cutoff = datetime.combine(prior_day, datetime.min.time().replace(hour=issued_before_hour), tzinfo=local_tz)
    return local_cutoff.astimezone(ZoneInfo("UTC"))


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["city", "date", "tmax_forecast_f", "issued_time", "valid_date", "hours_since_issuance", "station"])


def _save_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop_duplicates(subset=["city", "date"], keep="last").sort_values(["city", "date"]).to_parquet(path, index=False)


def _month_starts(start_date: date, end_date: date) -> list[date]:
    cursor = date(start_date.year, start_date.month, 1)
    starts = []
    while cursor <= end_date:
        starts.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return starts


def _load_month_mos_frame(station: str, month_start: date, session: requests.Session) -> pd.DataFrame:
    if month_start.month == 12:
        next_month = date(month_start.year + 1, 1, 1)
    else:
        next_month = date(month_start.year, month_start.month + 1, 1)
    models = ["NBE", "NBS"] if month_start >= NBE_START else ["NBS", "GFS"]
    query_start = datetime.combine(month_start - timedelta(days=3), datetime.min.time()).replace(tzinfo=ZoneInfo("UTC"))
    query_end = datetime.combine(next_month + timedelta(days=1), datetime.min.time()).replace(tzinfo=ZoneInfo("UTC"))
    for model in models:
        frame = _fetch_iem_mos_table(station, model, query_start, query_end, session)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def fetch_nws_tmax_forecast_batch(
    city_config: dict,
    target_dates: pd.Series | list[str],
    issued_before_hour: int = 22,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sleep_seconds: float = 1.0,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Fetch NWS Tmax forecasts for train cities and target dates with resume support."""
    dates = sorted(pd.to_datetime(pd.Series(target_dates).dropna().drop_duplicates()).dt.date.tolist())
    if not dates:
        return pd.DataFrame()
    session = make_session()
    existing = _load_checkpoint(output_path)
    if not existing.empty:
        success_mask = existing["tmax_forecast_f"].notna()
        done = set(zip(existing.loc[success_mask, "city"], existing.loc[success_mask, "date"]))
        rows = existing.loc[success_mask].to_dict("records")
    else:
        done = set()
        rows = []
    date_set = set(dates)
    fetched_since_save = 0
    api_calls = 0
    for city in TRAIN_CITIES:
        if city not in city_config:
            continue
        cfg = city_config[city]
        station = str(cfg["nws_station"])
        local_tz = ZoneInfo(str(cfg["timezone"]))
        for month_start in _month_starts(dates[0], dates[-1]):
            month_dates = [
                day
                for day in dates
                if day.year == month_start.year and day.month == month_start.month and (city, day.isoformat()) not in done
            ]
            if not month_dates:
                continue
            frame = _load_month_mos_frame(station, month_start, session)
            api_calls += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            for target_date in month_dates:
                issued_before = _issued_before_for_target(target_date, issued_before_hour, local_tz)
                result = _extract_tmax_from_mos(frame, target_date, issued_before)
                if result is None:
                    result = fetch_nws_tmax_forecast(
                        float(cfg["lat"]),
                        float(cfg["lon"]),
                        target_date,
                        issued_before,
                        station=station,
                        session=session,
                    )
                    api_calls += 1
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                if not result:
                    continue
                rows.append(
                    {
                        "city": city,
                        "date": target_date.isoformat(),
                        "station": station,
                        "tmax_forecast_f": result["tmax_forecast_f"],
                        "issued_time": result["issued_time"],
                        "valid_date": result["valid_date"],
                        "hours_since_issuance": result["hours_since_issuance"],
                    }
                )
                done.add((city, target_date.isoformat()))
                fetched_since_save += 1
                if fetched_since_save >= checkpoint_every:
                    _save_checkpoint(output_path, pd.DataFrame(rows))
                    fetched_since_save = 0
                    print(f"Checkpoint saved: {output_path} ({len(rows)} rows)")
            print(f"NWS {city} {month_start:%Y-%m}: {len(month_dates)} dates processed")

    print(f"NWS batch complete: {api_calls} API calls, {len(rows)} rows fetched")
    final = pd.DataFrame(rows).drop_duplicates(subset=["city", "date"], keep="last").sort_values(["city", "date"])
    _save_checkpoint(output_path, final)
    return final


def print_coverage_table(
    forecasts: pd.DataFrame,
    city_config: dict,
    trackj_dir: Path = Path("data/trackj"),
    train_start: date = date(2021, 1, 1),
    train_end: date = date(2024, 12, 31),
) -> pd.DataFrame:
    """Print and return coverage table with mean abs error vs CLI Tmax."""
    summary_rows: list[dict] = []
    for city in TRAIN_CITIES:
        if city not in city_config:
            continue
        city_forecasts = forecasts[forecasts["city"].eq(city)].copy()
        n_requested = int(city_forecasts["date"].nunique())
        n_fetched = int(city_forecasts["tmax_forecast_f"].notna().sum())
        coverage = 100.0 * n_fetched / n_requested if n_requested else 0.0
        mae = float("nan")
        cli_path = trackj_dir / city / "cli_target.parquet"
        if cli_path.exists():
            cli = pd.read_parquet(cli_path)
            cli["date"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            joined = city_forecasts.merge(cli[["date", "tmax_f"]], on="date", how="inner")
            joined = joined[joined["tmax_forecast_f"].notna() & joined["tmax_f"].notna()]
            if not joined.empty:
                mae = float((joined["tmax_forecast_f"] - joined["tmax_f"]).abs().mean())
        train_dates = pd.to_datetime(city_forecasts["date"], errors="coerce")
        train_mask = train_dates.dt.date.ge(train_start) & train_dates.dt.date.le(train_end)
        train_subset = city_forecasts.loc[train_mask]
        train_cov_pct = 100.0 * train_subset["tmax_forecast_f"].notna().mean() if not train_subset.empty else 0.0
        summary_rows.append(
            {
                "City": city,
                "N dates requested": n_requested,
                "N fetched": n_fetched,
                "Coverage %": round(coverage, 1),
                "Mean abs error vs actual Tmax": round(mae, 2) if mae == mae else None,
                "2021-2024 coverage %": round(train_cov_pct, 1),
            }
        )
    summary = pd.DataFrame(summary_rows)
    print("\n=== NWS FORECAST COVERAGE TABLE ===")
    print(summary.to_string(index=False))
    for _, row in summary.iterrows():
        if row["2021-2024 coverage %"] < 50.0:
            print(f"FLAG: {row['City']} has <50% NWS coverage for 2021-2024; use Groups 1-2 only for training.")
    return summary
