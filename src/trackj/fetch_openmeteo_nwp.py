from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .fetch_nws_forecast import TRAIN_CITIES

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_OUTPUT_PATH = Path("data/trackb/openmeteo_nwp_raw.parquet")
NWP_MODELS = ("ecmwf_ifs025", "gfs_seamless")
LIVE_CUTOFF_DAYS = 7


def make_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MCP_trading_research oscar@utexas.edu"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _month_starts(start_date: date, end_date: date) -> list[date]:
    cursor = date(start_date.year, start_date.month, 1)
    starts: list[date] = []
    while cursor <= end_date:
        starts.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return starts


def _month_end(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year, 12, 31)
    return date(month_start.year, month_start.month + 1, 1) - timedelta(days=1)


def _live_cutoff_date() -> date:
    return datetime.utcnow().date() - timedelta(days=LIVE_CUTOFF_DAYS)


def _fetch_openmeteo_range(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    model: str,
    session: requests.Session,
) -> pd.DataFrame:
    """Fetch one date range from historical or live open-meteo forecast API."""
    cutoff = _live_cutoff_date()
    if end_date <= cutoff:
        base_url = HISTORICAL_FORECAST_URL
    elif start_date > cutoff:
        base_url = LIVE_FORECAST_URL
    else:
        hist_end = cutoff
        hist_df = _fetch_openmeteo_range(lat, lon, start_date, hist_end, model, session)
        live_df = _fetch_openmeteo_range(lat, lon, hist_end + timedelta(days=1), end_date, model, session)
        return pd.concat([hist_df, live_df], ignore_index=True)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "models": model,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "temperature_unit": "fahrenheit",
    }
    try:
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"WARNING: open-meteo {model} {start_date}..{end_date} failed: {exc}")
        return pd.DataFrame(columns=["date", "nwp_tmax_forecast_f", "model_used", "issued_date", "valid_date"])

    times = payload.get("daily", {}).get("time", [])
    temps = payload.get("daily", {}).get("temperature_2m_max", [])
    rows: list[dict] = []
    for day_str, temp in zip(times, temps):
        valid_date = date.fromisoformat(day_str)
        issued_date = valid_date - timedelta(days=1)
        rows.append(
            {
                "date": day_str,
                "nwp_tmax_forecast_f": float(temp) if temp is not None else float("nan"),
                "model_used": model,
                "issued_date": issued_date.isoformat(),
                "valid_date": valid_date.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def fetch_openmeteo_tmax(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    model: str,
    *,
    session: requests.Session | None = None,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """Fetch open-meteo daily Tmax forecasts for a lat/lon and date range."""
    if start_date > end_date:
        return pd.DataFrame(columns=["date", "nwp_tmax_forecast_f", "model_used", "issued_date", "valid_date"])
    session = session or make_session()
    frames: list[pd.DataFrame] = []
    for month_start in _month_starts(start_date, end_date):
        chunk_start = max(start_date, month_start)
        chunk_end = min(end_date, _month_end(month_start))
        chunk = _fetch_openmeteo_range(lat, lon, chunk_start, chunk_end, model, session)
        frames.append(chunk)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    if not frames:
        return pd.DataFrame(columns=["date", "nwp_tmax_forecast_f", "model_used", "issued_date", "valid_date"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["city", "date", "nwp_tmax_forecast_f", "model_used", "issued_date", "valid_date"])


def _save_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop_duplicates(subset=["city", "date", "model_used"], keep="last").sort_values(
        ["city", "model_used", "date"]
    ).to_parquet(path, index=False)


def fetch_openmeteo_tmax_batch(
    city_config: dict,
    start_date: date,
    end_date: date,
    models: tuple[str, ...] = NWP_MODELS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sleep_seconds: float = 0.5,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch open-meteo Tmax forecasts for all train cities with per-city checkpointing."""
    session = make_session()
    existing = _load_checkpoint(output_path)
    if force_refresh:
        rows: list[dict] = []
        done: set[tuple[str, str, str]] = set()
    elif not existing.empty:
        success_mask = existing["nwp_tmax_forecast_f"].notna()
        done = set(
            zip(
                existing.loc[success_mask, "city"].astype(str),
                existing.loc[success_mask, "date"].astype(str),
                existing.loc[success_mask, "model_used"].astype(str),
            )
        )
        rows = existing.to_dict("records")
    else:
        rows = []
        done = set()

    api_calls = 0
    for city in TRAIN_CITIES:
        if city not in city_config:
            continue
        cfg = city_config[city]
        lat = float(cfg["lat"])
        lon = float(cfg["lon"])
        city_rows_before = len(rows)
        for model in models:
            if (city, start_date.isoformat(), model) in done and (city, end_date.isoformat(), model) in done:
                # coarse skip if endpoints already fetched for this city/model
                city_model_done = sum(1 for c, d, m in done if c == city and m == model)
                expected_days = (end_date - start_date).days + 1
                if city_model_done >= expected_days * 0.95:
                    print(f"Skipping {city} {model}: already fetched ({city_model_done} rows)")
                    continue
            frame = fetch_openmeteo_tmax(lat, lon, start_date, end_date, model, session=session, sleep_seconds=sleep_seconds)
            api_calls += len(_month_starts(start_date, end_date))
            for record in frame.to_dict("records"):
                key = (city, str(record["date"]), str(record["model_used"]))
                if key in done:
                    continue
                rows.append(
                    {
                        "city": city,
                        "date": record["date"],
                        "nwp_tmax_forecast_f": record["nwp_tmax_forecast_f"],
                        "model_used": record["model_used"],
                        "issued_date": record["issued_date"],
                        "valid_date": record["valid_date"],
                    }
                )
                if pd.notna(record["nwp_tmax_forecast_f"]):
                    done.add(key)
        _save_checkpoint(output_path, pd.DataFrame(rows))
        city_added = len(rows) - city_rows_before
        print(f"open-meteo {city}: checkpoint saved ({city_added} new rows, {len(rows)} total)")

    print(f"open-meteo batch complete: ~{api_calls} API calls, {len(rows)} rows")
    final = pd.DataFrame(rows).drop_duplicates(subset=["city", "date", "model_used"], keep="last").sort_values(
        ["city", "model_used", "date"]
    )
    _save_checkpoint(output_path, final)
    return final


def print_openmeteo_coverage_table(
    forecasts: pd.DataFrame,
    city_config: dict,
    trackj_dir: Path = Path("data/trackj"),
    train_start: date = date(2021, 1, 1),
    train_end: date = date(2024, 12, 31),
) -> pd.DataFrame:
    """Print per-city coverage and MAE vs CLI Tmax, grouped by model."""
    summary_rows: list[dict] = []
    for city in TRAIN_CITIES:
        if city not in city_config:
            continue
        for model in sorted(forecasts["model_used"].dropna().unique()):
            city_model = forecasts[forecasts["city"].eq(city) & forecasts["model_used"].eq(model)].copy()
            n_requested = int(city_model["date"].nunique())
            n_fetched = int(city_model["nwp_tmax_forecast_f"].notna().sum())
            coverage = 100.0 * n_fetched / n_requested if n_requested else 0.0
            mae = float("nan")
            cli_path = trackj_dir / city / "cli_target.parquet"
            if cli_path.exists():
                cli = pd.read_parquet(cli_path)
                cli["date"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                joined = city_model.merge(cli[["date", "tmax_f"]], on="date", how="inner")
                joined = joined[joined["nwp_tmax_forecast_f"].notna() & joined["tmax_f"].notna()]
                if not joined.empty:
                    mae = float((joined["nwp_tmax_forecast_f"] - joined["tmax_f"]).abs().mean())
            train_dates = pd.to_datetime(city_model["date"], errors="coerce")
            train_mask = train_dates.dt.date.ge(train_start) & train_dates.dt.date.le(train_end)
            train_subset = city_model.loc[train_mask]
            train_cov_pct = 100.0 * train_subset["nwp_tmax_forecast_f"].notna().mean() if not train_subset.empty else 0.0
            summary_rows.append(
                {
                    "City": city,
                    "Model": model,
                    "N dates": n_requested,
                    "N fetched": n_fetched,
                    "Coverage %": round(coverage, 1),
                    "MAE vs actual Tmax": round(mae, 2) if mae == mae else None,
                    "2021-2024 coverage %": round(train_cov_pct, 1),
                }
            )
    summary = pd.DataFrame(summary_rows)
    print("\n=== OPEN-METEO NWP COVERAGE TABLE ===")
    print(summary.to_string(index=False))
    for _, row in summary.iterrows():
        mae_val = row["MAE vs actual Tmax"]
        if mae_val is not None and mae_val == mae_val and mae_val > 5.0:
            print(
                f"ERROR: {row['City']} {row['Model']} MAE {mae_val}°F exceeds 5°F; "
                "possible wrong horizon or reanalysis contamination."
            )
    return summary
