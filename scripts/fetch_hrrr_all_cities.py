"""Fetch HRRR 10Z multi-covariate forecasts for 10 US cities via Herbie."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import os
import sys
import threading
import warnings
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_gfs_herbie import (  # noqa: E402
    _extract_scalar,
    _normalize_dataset,
    clear_herbie_cache,
    kelvin_to_f,
)

HRRR_STATIONS = {
    "austin": {"station": "KAUS", "lat": 30.1975, "lon": -97.6664, "tz": "America/Chicago"},
    "houston": {"station": "KHOU", "lat": 29.6454, "lon": -95.2789, "tz": "America/Chicago"},
    "dallas": {"station": "KDAL", "lat": 32.8471, "lon": -96.8518, "tz": "America/Chicago"},
    "chicago": {"station": "KORD", "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago"},
    "los_angeles": {"station": "KLAX", "lat": 33.9425, "lon": -118.4081, "tz": "America/Los_Angeles"},
    "san_francisco": {"station": "KSFO", "lat": 37.6213, "lon": -122.3790, "tz": "America/Los_Angeles"},
    "seattle": {"station": "KSEA", "lat": 47.4502, "lon": -122.3088, "tz": "America/Los_Angeles"},
    "new_york": {"station": "KLGA", "lat": 40.7772, "lon": -73.8726, "tz": "America/New_York"},
    "miami": {"station": "KMIA", "lat": 25.7932, "lon": -80.2906, "tz": "America/New_York"},
    "atlanta": {"station": "KATL", "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York"},
}

HRRR_DATA_DIR = PROJECT_ROOT / "data" / "hrrr_v2"
PARQUET_PATH = HRRR_DATA_DIR / "hrrr_all_cities.parquet"
FXX_RANGE = range(1, 19)
PEAK_HOUR_MIN, PEAK_HOUR_MAX = 13, 17
INIT_HOUR_UTC = 10

# Defaults tuned for ~16 GB free disk: ~20 concurrent subset GRIBs ≈ 100–300 MB scratch peak.
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_DAY_WORKERS = 8
DEFAULT_FXX_WORKERS = 18
CACHE_FLUSH_EVERY = 20

# Per-fxx xarray pulls (wind merged; snow only at fxx=1).
FXX_FIELD_SEARCHES = (
    ("tmp_2m", ":TMP:2 m above ground:"),
    ("wind", ":UGRD:10 m|:VGRD:10 m"),
    ("tcdc", ":TCDC:entire atmosphere:"),
    ("dswrf", ":DSWRF:surface:"),
)
SNOW_FIELD = ":SNOD:surface:"

CSV_COLUMNS = [
    "date",
    "hrrr_tmax",
    "peak_cloud_cover",
    "peak_solar_flux",
    "peak_wind_speed",
    "snow_depth",
    "n_hours_fetched",
    "n_peak_hours",
    "grid_lat",
    "grid_lon",
]
PARQUET_COLUMNS = [
    "city",
    "date",
    "station",
    "hrrr_tmax",
    "peak_cloud_cover",
    "peak_solar_flux",
    "peak_wind_speed",
    "snow_depth",
    "n_hours_fetched",
    "n_peak_hours",
    "grid_lat",
    "grid_lon",
]

logger = logging.getLogger(__name__)

_grib_semaphore: threading.Semaphore | None = None
_download_executor: ThreadPoolExecutor | None = None


def suppress_noisy_output() -> None:
    warnings.filterwarnings("ignore")
    for name in ("herbie", "xarray", "cfgrib", "eccodes"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _init_download_pool(max_concurrent: int) -> None:
    global _grib_semaphore, _download_executor
    _grib_semaphore = threading.Semaphore(max_concurrent)
    _download_executor = ThreadPoolExecutor(max_workers=max_concurrent)


def _shutdown_download_pool() -> None:
    global _download_executor
    if _download_executor is not None:
        _download_executor.shutdown(wait=True, cancel_futures=False)
        _download_executor = None


def _city_label(city: str) -> str:
    return city.replace("_", " ").title()


def monthly_cache_path(city: str, target_date: date) -> Path:
    return HRRR_DATA_DIR / city / f"hrrr_{city}_{target_date:%Y%m}.csv"


def load_monthly_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.read_csv(path)


def date_in_cache(cache_df: pd.DataFrame, date_str: str) -> bool:
    if cache_df.empty or "date" not in cache_df.columns:
        return False
    return date_str in cache_df["date"].astype(str).values


def write_monthly_row(cache_df: pd.DataFrame, path: Path, row: dict) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    date_str = str(row["date"])
    if date_in_cache(cache_df, date_str):
        cache_df = cache_df[cache_df["date"].astype(str) != date_str]
    new_row = pd.DataFrame([row])
    cache_df = new_row if cache_df.empty else pd.concat([cache_df, new_row], ignore_index=True)
    cache_df = cache_df.sort_values("date").reset_index(drop=True)
    cache_df.to_csv(path, index=False)
    return cache_df


def _extract_wind_mps(ds, lat: float, lon: float) -> tuple[float | None, float | None]:
    items = ds if isinstance(ds, list) else [ds]
    u_val: float | None = None
    v_val: float | None = None
    for item in items:
        if item is None or not getattr(item, "data_vars", None):
            continue
        for var_name in list(item.data_vars):
            single_ds = item[[var_name]]
            value, _, _ = _extract_scalar(single_ds, lat=lat, lon=lon)
            if value is None:
                continue
            upper = var_name.upper()
            if "UGRD" in upper or upper.startswith("U"):
                u_val = value
            elif "VGRD" in upper or upper.startswith("V"):
                v_val = value
    return u_val, v_val


def _download_fxx_fields(
    init_naive_utc: datetime,
    fxx: int,
    lat: float,
    lon: float,
) -> tuple[dict[str, float | None], float | None, float | None]:
    """Download all HRRR fields for one fxx from a single Herbie instance."""
    from herbie import Herbie

    assert _grib_semaphore is not None
    values: dict[str, float | None] = {
        "tmp_2m": None,
        "tcdc": None,
        "dswrf": None,
        "wind_u": None,
        "wind_v": None,
        "snow_depth": None,
    }
    grid_lat: float | None = None
    grid_lon: float | None = None

    searches = list(FXX_FIELD_SEARCHES)
    if fxx == 1:
        searches = searches + [("snow_depth", SNOW_FIELD)]

    with _grib_semaphore:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            h = Herbie(init_naive_utc, model="hrrr", product="sfc", fxx=fxx)
            for idx, (field_name, search_string) in enumerate(searches):
                remove = idx == len(searches) - 1
                try:
                    ds = h.xarray(search_string, remove_grib=remove)
                    if field_name == "wind":
                        u_val, v_val = _extract_wind_mps(ds, lat, lon)
                        values["wind_u"] = u_val
                        values["wind_v"] = v_val
                        if grid_lat is None and u_val is not None:
                            point = _normalize_dataset(ds)
                            _, grid_lat, grid_lon = _extract_scalar(point, lat=lat, lon=lon)
                    else:
                        value, glat, glon = _extract_scalar(_normalize_dataset(ds), lat=lat, lon=lon)
                        values[field_name] = value
                        if grid_lat is None and glat is not None:
                            grid_lat = glat
                            grid_lon = glon
                except Exception as exc:
                    logger.debug("%s fxx=%d field=%s: %s", init_naive_utc.date(), fxx, field_name, exc)

    return values, grid_lat, grid_lon


def _submit_fxx_download(
    init_naive_utc: datetime,
    fxx: int,
    lat: float,
    lon: float,
) -> Future:
    assert _download_executor is not None
    return _download_executor.submit(_download_fxx_fields, init_naive_utc, fxx, lat, lon)


def fetch_hrrr_for_date(cfg: dict, target_date: date) -> dict:
    """Download HRRR 10Z forecast hours and compute daily covariates for target_date."""
    lat = float(cfg["lat"])
    lon = float(cfg["lon"])
    local_tz = ZoneInfo(cfg["tz"])

    init_utc = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        INIT_HOUR_UTC,
        tzinfo=timezone.utc,
    )
    init_naive_utc = init_utc.replace(tzinfo=None)

    futures = {
        _submit_fxx_download(init_naive_utc, fxx, lat, lon): fxx for fxx in FXX_RANGE
    }
    fxx_results: dict[int, tuple[dict[str, float | None], float | None, float | None]] = {}
    for future in as_completed(futures):
        fxx = futures[future]
        fxx_results[fxx] = future.result()

    tmp_values: list[float] = []
    peak_tcdc: list[float] = []
    peak_dswrf: list[float] = []
    peak_wind: list[float] = []
    grid_lat: float | None = None
    grid_lon: float | None = None
    snow_depth = 0.0
    n_hours_fetched = 0

    for fxx in FXX_RANGE:
        hour_values, point_lat, point_lon = fxx_results[fxx]
        valid_utc = init_utc + timedelta(hours=fxx)
        valid_local = valid_utc.astimezone(local_tz)
        is_peak = PEAK_HOUR_MIN <= valid_local.hour <= PEAK_HOUR_MAX

        if fxx == 1:
            snod = hour_values.get("snow_depth")
            if snod is not None and not pd.isna(snod):
                snow_depth = float(snod)

        tmp_k = hour_values.get("tmp_2m")
        if tmp_k is not None and not pd.isna(tmp_k):
            tmp_f = kelvin_to_f(float(tmp_k))
            if tmp_f is not None and not pd.isna(tmp_f):
                tmp_values.append(float(tmp_f))
                n_hours_fetched += 1
                if grid_lat is None and point_lat is not None:
                    grid_lat = point_lat
                    grid_lon = point_lon

        if is_peak:
            tcdc = hour_values.get("tcdc")
            if tcdc is not None and not pd.isna(tcdc):
                peak_tcdc.append(float(tcdc))
            dswrf = hour_values.get("dswrf")
            if dswrf is not None and not pd.isna(dswrf):
                peak_dswrf.append(float(dswrf))
            u = hour_values.get("wind_u")
            v = hour_values.get("wind_v")
            if u is not None and v is not None and not pd.isna(u) and not pd.isna(v):
                peak_wind.append(math.sqrt(float(u) ** 2 + float(v) ** 2) * 2.237)

    def _mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else np.nan

    return {
        "date": target_date.isoformat(),
        "hrrr_tmax": max(tmp_values) if tmp_values else np.nan,
        "peak_cloud_cover": _mean_or_nan(peak_tcdc),
        "peak_solar_flux": _mean_or_nan(peak_dswrf),
        "peak_wind_speed": _mean_or_nan(peak_wind),
        "snow_depth": snow_depth,
        "n_hours_fetched": n_hours_fetched,
        "n_peak_hours": len(peak_tcdc),
        "grid_lat": grid_lat if grid_lat is not None else np.nan,
        "grid_lon": grid_lon if grid_lon is not None else np.nan,
    }


def fetch_city(
    city: str,
    cfg: dict,
    dates: list[date],
    overwrite: bool,
    day_workers: int,
) -> None:
    station = cfg["station"]
    label = _city_label(city)
    fetched = 0
    skipped = 0
    bar_desc = f"{label} ({station})"
    monthly_caches: dict[str, pd.DataFrame] = {}
    cache_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    flush_lock = threading.Lock()
    fetched_since_flush = 0

    def _process_day(target_date: date) -> str:
        nonlocal fetched_since_flush
        cache_path = monthly_cache_path(city, target_date)
        month_key = str(cache_path)
        date_str = target_date.isoformat()

        with cache_locks[month_key]:
            if month_key not in monthly_caches:
                monthly_caches[month_key] = load_monthly_cache(cache_path)
            cache_df = monthly_caches[month_key]
            if date_in_cache(cache_df, date_str) and not overwrite:
                return "skipped"

        row = fetch_hrrr_for_date(cfg, target_date)

        with cache_locks[month_key]:
            cache_df = monthly_caches.get(month_key, load_monthly_cache(cache_path))
            monthly_caches[month_key] = write_monthly_row(cache_df, cache_path, row)

        with flush_lock:
            fetched_since_flush += 1
            if fetched_since_flush >= CACHE_FLUSH_EVERY:
                clear_herbie_cache(verbose=False)
                fetched_since_flush = 0
        return "fetched"

    with tqdm(
        total=len(dates),
        desc=bar_desc,
        unit="day",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=day_workers) as day_executor:
            futures = {day_executor.submit(_process_day, d): d for d in dates}
            for future in as_completed(futures):
                result = future.result()
                if result == "skipped":
                    skipped += 1
                else:
                    fetched += 1
                pbar.set_postfix(
                    fetched=fetched,
                    skipped=skipped,
                    day_w=day_workers,
                    refresh=False,
                )
                pbar.update(1)

    clear_herbie_cache(verbose=False)
    tqdm.write(f"✓ {label} done: {fetched} fetched, {skipped} skipped from cache.")


def load_city_frames(cities: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for city in cities:
        cfg = HRRR_STATIONS[city]
        city_dir = HRRR_DATA_DIR / city
        if not city_dir.exists():
            continue
        for path in sorted(city_dir.glob(f"hrrr_{city}_*.csv")):
            frame = pd.read_csv(path)
            frame["city"] = city
            frame["station"] = cfg["station"]
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=PARQUET_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_combined_parquet(cities: list[str]) -> pd.DataFrame:
    combined = load_city_frames(cities)
    if combined.empty:
        print("No HRRR cache files found; skipping parquet write.", flush=True)
        return combined

    combined = combined.drop_duplicates(subset=["city", "date"], keep="last")
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
    out = combined[PARQUET_COLUMNS].copy()

    HRRR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PARQUET_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {PARQUET_PATH}", flush=True)
    return out


def print_summary(combined: pd.DataFrame) -> None:
    if combined.empty:
        print("\n=== SUMMARY ===\n(no data)", flush=True)
        return

    summary = (
        combined.groupby("city")
        .agg(
            n_days=("date", "count"),
            n_valid=("hrrr_tmax", lambda s: int(s.notna().sum())),
            date_min=("date", "min"),
            date_max=("date", "max"),
            mean_tmax=("hrrr_tmax", "mean"),
        )
        .reset_index()
    )
    summary["mean_tmax"] = summary["mean_tmax"].round(2)
    print("\n=== SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch HRRR 10Z covariates for 10 US cities.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--city", type=str, default=None, help="Single city slug.")
    parser.add_argument(
        "--overwrite",
        "--force",
        action="store_true",
        dest="overwrite",
        help="Re-fetch dates already in cache (alias: --force).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Debug mode: fetch from GRIB, print row dict(s) to stdout, do not write CSV.",
    )
    parser.add_argument(
        "--day-workers",
        type=int,
        default=int(os.environ.get("HRRR_DAY_WORKERS", DEFAULT_DAY_WORKERS)),
        help=f"Parallel calendar days per city (default {DEFAULT_DAY_WORKERS}).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=int(os.environ.get("HRRR_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT)),
        help=(
            f"Max simultaneous GRIB downloads; caps disk scratch (default {DEFAULT_MAX_CONCURRENT})."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("HRRR_WORKERS", DEFAULT_FXX_WORKERS)),
        help="Deprecated alias kept for shell scripts; use --max-concurrent instead.",
    )
    return parser.parse_args()


def run_test_fetch(city: str, dates: list[date], max_concurrent: int) -> None:
    """Fetch dates from GRIB and print daily rows (bypasses CSV cache)."""
    cfg = HRRR_STATIONS[city]
    rows: list[dict] = []
    _init_download_pool(max_concurrent)
    try:
        for target_date in dates:
            rows.append(fetch_hrrr_for_date(cfg, target_date))
    finally:
        _shutdown_download_pool()
        clear_herbie_cache(verbose=False)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    out = sys.__stdout__
    for row in rows:
        target_date = row["date"]
        out.write(f"\n=== {city} {target_date} ===\n")
        for key, value in row.items():
            out.write(f"  {key}: {value}\n")
        wind = row.get("peak_wind_speed")
        has_wind = wind is not None and not (isinstance(wind, float) and np.isnan(wind))
        out.write(f"  peak_wind_speed non-NaN: {has_wind}\n")
        if "soil_moisture" in row:
            out.write("  WARNING: soil_moisture still present in output dict\n")
    out.flush()


def main() -> None:
    suppress_noisy_output()
    args = parse_args()
    day_workers = max(1, min(args.day_workers, 12))
    max_concurrent = max(2, min(args.max_concurrent or args.workers, 36))

    if args.city is not None and args.city not in HRRR_STATIONS:
        valid = ", ".join(sorted(HRRR_STATIONS))
        raise ValueError(f"Unknown city {args.city!r}. Valid cities: {valid}")

    if args.test and args.city is None:
        raise ValueError("--test requires --city")

    cities = [args.city] if args.city else list(HRRR_STATIONS)
    dates = pd.date_range(args.start, args.end, freq="D").date.tolist()

    if args.test:
        run_test_fetch(cities[0], dates, max_concurrent)
        return

    print(
        f"HRRR 10Z fetch: {day_workers} day-workers, {max_concurrent} max concurrent GRIB downloads",
        flush=True,
    )
    print(
        f"  Disk budget: ~{max_concurrent * 8} MB transient GRIB scratch (CSV output ~200 MB total)",
        flush=True,
    )

    _init_download_pool(max_concurrent)
    try:
        for city in cities:
            fetch_city(
                city,
                HRRR_STATIONS[city],
                dates,
                overwrite=args.overwrite,
                day_workers=day_workers,
            )
    finally:
        _shutdown_download_pool()
        clear_herbie_cache(verbose=False)

    combined = build_combined_parquet(cities if args.city else list(HRRR_STATIONS))
    print_summary(combined)


if __name__ == "__main__":
    main()
