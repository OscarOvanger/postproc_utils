from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
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

import numpy as np
import pandas as pd


CENTRAL_TZ = ZoneInfo("America/Chicago")
KAUS_LAT = 30.1975
KAUS_LON = -97.6664
KAUS_STATION = "KAUS"
DEFAULT_RAW_DIR = Path("data/raw/gfs_kaus")
GFS_START_DATE = date(2021, 1, 1)
GFS_FEATURE_COLUMNS = ["gfs_t2m_afternoon", "gfs_dewpoint_afternoon", "gfs_cloudcover_afternoon"]
GFS_AUDIT_COLUMNS = [
    "date",
    "gfs_selected_init_utc",
    "gfs_selected_init_local",
    "gfs_selected_valid_utc",
    "gfs_selected_valid_local",
    "gfs_selected_fxx",
    "gfs_selected_source",
    "gfs_grid_lat",
    "gfs_grid_lon",
    "gfs_parse_status",
    "gfs_parse_warning",
]


@dataclass(frozen=True)
class GfsCandidate:
    init_utc: datetime
    fxx: int
    label: str

    @property
    def valid_utc(self) -> datetime:
        return self.init_utc + timedelta(hours=self.fxx)

    def init_local(self, local_tz: ZoneInfo = CENTRAL_TZ) -> datetime:
        return self.init_utc.astimezone(local_tz)

    def valid_local(self, local_tz: ZoneInfo = CENTRAL_TZ) -> datetime:
        return self.valid_utc.astimezone(local_tz)


def _city_station(city_config: dict | None) -> str:
    return str((city_config or {}).get("nws_station", KAUS_STATION)).lower()


def _city_lat(city_config: dict | None) -> float:
    return float((city_config or {}).get("lat", KAUS_LAT))


def _city_lon(city_config: dict | None) -> float:
    return float((city_config or {}).get("lon", KAUS_LON))


def _city_tz(city_config: dict | None) -> ZoneInfo:
    return ZoneInfo(str((city_config or {}).get("timezone", "America/Chicago")))


GFS_AVAILABILITY_LAG_HOURS = 6


def permissible_gfs_candidates(
    target_date: date,
    cutoff_hour: int = 10,
    local_tz: ZoneInfo = CENTRAL_TZ,
    availability_lag_hours: int = GFS_AVAILABILITY_LAG_HOURS,
) -> list[GfsCandidate]:
    """Return newest permissible operational runs available before the morning feature cutoff."""
    target_00z = datetime(target_date.year, target_date.month, target_date.day, 0, tzinfo=timezone.utc)
    target_06z = target_00z + timedelta(hours=6)
    prev_12z = target_00z - timedelta(hours=12)
    candidates = [
        GfsCandidate(target_06z, 15, "target_day_06z_f15"),
        GfsCandidate(target_06z, 12, "target_day_06z_f12"),
        GfsCandidate(target_06z, 11, "target_day_06z_f11"),
        GfsCandidate(target_00z, 21, "target_day_00z_f21"),
        GfsCandidate(target_00z, 18, "target_day_00z_f18"),
        GfsCandidate(prev_12z, 33, "previous_day_12z_f33"),
        GfsCandidate(prev_12z, 30, "previous_day_12z_f30"),
    ]
    cutoff_local = datetime.combine(target_date, time(cutoff_hour, 0), tzinfo=local_tz)
    lag = timedelta(hours=availability_lag_hours)
    return [
        candidate
        for candidate in candidates
        if candidate.init_local(local_tz) + lag <= cutoff_local
    ]


def gfs_cache_path(raw_dir: Path, target_date: date, city_config: dict | None = None) -> Path:
    station = _city_station(city_config)
    return raw_dir / f"{station}_gfs_{target_date:%Y%m%d}.csv"


def _select_nearest(ds, lat: float = KAUS_LAT, lon: float = KAUS_LON):
    ds = _normalize_dataset(ds)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lon_values = ds[lon_name]
    selected_lon = lon % 360 if float(lon_values.max()) > 180 else lon
    if len(ds[lat_name].dims) == 2 or len(ds[lon_name].dims) == 2:
        distance = (ds[lat_name] - lat) ** 2 + (ds[lon_name] - selected_lon) ** 2
        indexes = np.unravel_index(int(distance.argmin().values), distance.shape)
        return ds.isel({dim: index for dim, index in zip(distance.dims, indexes)})
    return ds.sel({lat_name: lat, lon_name: selected_lon}, method="nearest")


def _normalize_dataset(ds):
    if isinstance(ds, list):
        for item in ds:
            if getattr(item, "data_vars", None):
                return item
        if ds:
            return ds[0]
    return ds


def _first_data_variable(ds) -> str:
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("Herbie returned no data variables")
    return data_vars[0]


def _extract_scalar(ds, lat: float = KAUS_LAT, lon: float = KAUS_LON) -> tuple[float, float | None, float | None]:
    point = _select_nearest(ds, lat, lon)
    var_name = _first_data_variable(point)
    value = float(point[var_name].values)
    grid_lat = float(point["latitude"].values) if "latitude" in point.coords else None
    grid_lon = float(point["longitude"].values) if "longitude" in point.coords else None
    if grid_lon is not None and grid_lon > 180:
        grid_lon -= 360
    return value, grid_lat, grid_lon


def kelvin_to_f(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return (float(value) - 273.15) * 9 / 5 + 32 if float(value) > 150 else float(value)


def normalize_cloud(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric / 100.0 if numeric > 1.0 else numeric


def _download_candidate(
    candidate: GfsCandidate,
    lat: float = KAUS_LAT,
    lon: float = KAUS_LON,
    local_tz: ZoneInfo = CENTRAL_TZ,
) -> tuple[dict, dict]:
    from herbie import Herbie

    init_naive_utc = candidate.init_utc.replace(tzinfo=None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        h = Herbie(init_naive_utc, model="gfs", product="pgrb2.0p25", fxx=candidate.fxx)
        selected_source = getattr(h, "source", None) or ""
        tmp_ds = h.xarray(":TMP:2 m above", remove_grib=False)
        dpt_ds = h.xarray(":DPT:2 m above", remove_grib=False)
        cloud_ds = h.xarray(":TCDC:entire atmosphere", remove_grib=False)

    tmp_value, grid_lat, grid_lon = _extract_scalar(tmp_ds, lat=lat, lon=lon)
    dpt_value, _, _ = _extract_scalar(dpt_ds, lat=lat, lon=lon)
    cloud_value, _, _ = _extract_scalar(cloud_ds, lat=lat, lon=lon)

    features = {
        "gfs_t2m_afternoon": kelvin_to_f(tmp_value),
        "gfs_dewpoint_afternoon": kelvin_to_f(dpt_value),
        "gfs_cloudcover_afternoon": normalize_cloud(cloud_value),
    }
    audit = {
        "gfs_selected_init_utc": candidate.init_utc.isoformat().replace("+00:00", "Z"),
        "gfs_selected_init_local": candidate.init_local(local_tz).strftime("%Y-%m-%d %H:%M:%S%z"),
        "gfs_selected_valid_utc": candidate.valid_utc.isoformat().replace("+00:00", "Z"),
        "gfs_selected_valid_local": candidate.valid_local(local_tz).strftime("%Y-%m-%d %H:%M:%S%z"),
        "gfs_selected_fxx": candidate.fxx,
        "gfs_selected_source": selected_source,
        "gfs_grid_lat": grid_lat,
        "gfs_grid_lon": grid_lon,
        "gfs_parse_status": "ok",
        "gfs_parse_warning": "",
    }
    return features, audit


def fetch_gfs_for_date(
    target_date: date,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    cutoff_hour: int = 10,
    city_config: dict | None = None,
) -> tuple[dict, dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = gfs_cache_path(raw_dir, target_date, city_config=city_config)
    if path.exists() and not overwrite:
        row = pd.read_csv(path).iloc[0].to_dict()
        if row.get("gfs_parse_status") == "ok" or target_date < GFS_START_DATE:
            features = {column: row.get(column) for column in GFS_FEATURE_COLUMNS}
            audit = {column: row.get(column, "") for column in GFS_AUDIT_COLUMNS if column != "date"}
            return features, audit

    if target_date < GFS_START_DATE:
        features = {column: np.nan for column in GFS_FEATURE_COLUMNS}
        audit = {
            "gfs_selected_init_utc": "",
            "gfs_selected_init_local": "",
            "gfs_selected_valid_utc": "",
            "gfs_selected_valid_local": "",
            "gfs_selected_fxx": np.nan,
            "gfs_selected_source": "",
            "gfs_grid_lat": np.nan,
            "gfs_grid_lon": np.nan,
            "gfs_parse_status": "missing_gfs",
            "gfs_parse_warning": "GFS pgrb2.0p25 archive not attempted before 2021-01-01",
        }
    else:
        warnings: list[str] = []
        features = {column: np.nan for column in GFS_FEATURE_COLUMNS}
        audit = {
            "gfs_selected_init_utc": "",
            "gfs_selected_init_local": "",
            "gfs_selected_valid_utc": "",
            "gfs_selected_valid_local": "",
            "gfs_selected_fxx": np.nan,
            "gfs_selected_source": "",
            "gfs_grid_lat": np.nan,
            "gfs_grid_lon": np.nan,
            "gfs_parse_status": "missing_gfs",
            "gfs_parse_warning": "",
        }
        local_tz = _city_tz(city_config)
        for candidate in permissible_gfs_candidates(target_date, cutoff_hour=cutoff_hour, local_tz=local_tz):
            try:
                features, audit = _download_candidate(
                    candidate,
                    lat=_city_lat(city_config),
                    lon=_city_lon(city_config),
                    local_tz=local_tz,
                )
                missing = [column for column in GFS_FEATURE_COLUMNS if pd.isna(features[column])]
                if missing:
                    warnings.append(f"{candidate.label}: missing {','.join(missing)}")
                    continue
                break
            except Exception as exc:
                warnings.append(f"{candidate.label}: {type(exc).__name__}: {exc}")
        if audit["gfs_parse_status"] != "ok":
            audit["gfs_parse_warning"] = "; ".join(warnings)[:2000]

    row = {"date": target_date.isoformat(), **features, **audit}
    pd.DataFrame([row]).to_csv(path, index=False)
    return features, audit


def build_gfs_features(
    target_dates: pd.Series | list[str],
    raw_dir: Path = DEFAULT_RAW_DIR,
    fetch: bool = True,
    overwrite: bool = False,
    cutoff_hour: int = 10,
    city_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(pd.Series(target_dates).dropna().drop_duplicates().sort_values()).dt.date
    feature_rows: list[dict] = []
    audit_rows: list[dict] = []
    for target_date in dates:
        if fetch:
            features, audit = fetch_gfs_for_date(
                target_date,
                raw_dir=raw_dir,
                overwrite=overwrite,
                cutoff_hour=cutoff_hour,
                city_config=city_config,
            )
        else:
            path = gfs_cache_path(raw_dir, target_date, city_config=city_config)
            if path.exists():
                row = pd.read_csv(path).iloc[0].to_dict()
                features = {column: row.get(column) for column in GFS_FEATURE_COLUMNS}
                audit = {column: row.get(column, "") for column in GFS_AUDIT_COLUMNS if column != "date"}
            else:
                features = {column: np.nan for column in GFS_FEATURE_COLUMNS}
                audit = {
                    "gfs_selected_init_utc": "",
                    "gfs_selected_init_local": "",
                    "gfs_selected_valid_utc": "",
                    "gfs_selected_valid_local": "",
                    "gfs_selected_fxx": np.nan,
                    "gfs_selected_source": "",
                    "gfs_grid_lat": np.nan,
                    "gfs_grid_lon": np.nan,
                    "gfs_parse_status": "missing_gfs_cache",
                    "gfs_parse_warning": "cached GFS feature file not found",
                }
        feature_rows.append({"date": target_date.isoformat(), **features})
        audit_rows.append({"date": target_date.isoformat(), **audit})
        print(f"GFS {target_date}: {audit.get('gfs_parse_status')} {audit.get('gfs_selected_init_utc', '')}", flush=True)
    return pd.DataFrame(feature_rows), pd.DataFrame(audit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch KAUS GFS covariates with Herbie.")
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y-%m-%d")
    build_gfs_features(dates, raw_dir=args.raw_dir, fetch=True, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
