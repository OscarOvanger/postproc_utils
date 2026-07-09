#!/usr/bin/env python3
"""Backfill rolling-bias residuals: raw NGBoost mu minus WU actual."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_ngboost as ng  # noqa: E402
from fetch_hrrr_all_cities import (  # noqa: E402
    HRRR_STATIONS,
    _init_download_pool,
    _shutdown_download_pool,
    fetch_hrrr_for_date,
    load_monthly_cache,
    monthly_cache_path,
    write_monthly_row,
)
from src.ngboost_live_forecast import (  # noqa: E402
    ICAO_MAP,
    NgBoostLiveModels,
    build_live_features,
    ensure_wu_current,
    last_feature_fail_reason,
    predict_ngboost_from_features,
)
import src.ngboost_live_forecast as ngboost_live  # noqa: E402
from src.rolling_bias import (  # noqa: E402
    SNAPSHOT_PATH,
    load_residuals_df,
    save_residuals_and_snapshot,
)

DEPLOY_CITIES = [
    "atlanta",
    "austin",
    "chicago",
    "dallas",
    "houston",
    "miami",
    "new_york",
    "seattle",
]
WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "ngboost_v2"
HRRR_TIMEOUT_SEC = 120


def ensure_hrrr_backfill(city: str, event_date: str, timeout: int = HRRR_TIMEOUT_SEC) -> bool:
    """Like ensure_hrrr but allows historical dates and uses a longer timeout."""
    if city not in HRRR_STATIONS:
        return False

    target = date.fromisoformat(event_date)
    hrrr = ng.load_hrrr_city(city)
    cached = set(pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d"))
    if event_date in cached:
        return True

    try:
        _init_download_pool(8)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(fetch_hrrr_for_date, HRRR_STATIONS[city], target)
            try:
                row = future.result(timeout=timeout)
            except FuturesTimeout:
                print(f"  HRRR fetch timed out for {city}/{event_date} ({timeout}s)")
                return False
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            _shutdown_download_pool()
        tmax = row.get("hrrr_tmax")
        if tmax is None or (isinstance(tmax, float) and np.isnan(tmax)):
            print(f"  HRRR unavailable (null tmax) for {city}/{event_date}")
            return False
        path = monthly_cache_path(city, target)
        cache = load_monthly_cache(path)
        write_monthly_row(cache, path, row)
        print(f"  HRRR fetched: {city} {event_date} tmax={tmax}")
        return True
    except Exception as exc:
        print(f"  HRRR fetch failed for {city}/{event_date}: {exc}")
        return False


def fetch_wu_actual_iem(icao: str, date_str: str) -> int | None:
    """Fetch max hourly METAR temperature from IEM ASOS."""
    current = date.fromisoformat(date_str)
    next_day = current + timedelta(days=1)
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={icao}&data=tmpf&tz=UTC&format=onlycomma"
        f"&year1={current.year}&month1={current.month}&day1={current.day}"
        f"&year2={next_day.year}&month2={next_day.month}&day2={next_day.day}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        temps: list[float] = []
        for row in reader:
            val = row.get("tmpf", "M").strip()
            if val not in ("M", ""):
                try:
                    temps.append(float(val))
                except ValueError:
                    continue
        if not temps:
            return None
        return round(max(temps))
    except Exception as exc:
        print(f"  IEM fetch failed for {icao} on {date_str}: {exc}")
        return None


def load_wu_actual(city: str, date_str: str, wu_df: pd.DataFrame) -> float | None:
    """Prefer parquet WU actual; fall back to IEM with 1s sleep."""
    if not wu_df.empty:
        match = wu_df[(wu_df["city"] == city) & (wu_df["date"] == date_str)]
        if not match.empty:
            val = match.iloc[0].get("wunderground_tmax")
            if val is not None and np.isfinite(float(val)):
                return float(val)

    icao = ICAO_MAP.get(city)
    if not icao:
        return None
    actual = fetch_wu_actual_iem(icao, date_str)
    time.sleep(1.0)
    return float(actual) if actual is not None else None


def _asos_date_present(city: str, date_str: str) -> bool:
    path = ng._asos_cache_path(city)
    if not path.exists():
        return False
    cached = pd.read_csv(path)
    if cached.empty or "date" not in cached.columns:
        return False
    cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = cached[cached["date"] == date_str]
    if row.empty:
        return False
    val = pd.to_numeric(row.iloc[0].get("temp_early_morning"), errors="coerce")
    return bool(np.isfinite(val))


def ensure_asos_for_date(city: str, date_str: str) -> bool:
    """Force-fetch ASOS early-morning temp when cache has span holes."""
    if _asos_date_present(city, date_str):
        return True
    meta = ng.STATION_META[city]
    target = date.fromisoformat(date_str)
    # _cache_covers_range is true when min/max span the day even if the day
    # itself is missing — temporarily hide the cache so fetch re-pulls.
    cache_path = ng._asos_cache_path(city)
    backup = cache_path.with_suffix(".csv.bak_backfill")
    moved = False
    try:
        if cache_path.exists():
            cache_path.rename(backup)
            moved = True
        ng.fetch_asos_temp_early_morning(
            station=str(meta["station"]),
            start_date=target,
            end_date=target,
            city=city,
            timezone=str(meta["tz"]),
        )
    except Exception as exc:
        print(f"  ASOS fetch failed for {city}/{date_str}: {exc}")
        return False
    finally:
        if moved and backup.exists():
            if cache_path.exists():
                new_rows = pd.read_csv(cache_path)
                old_rows = pd.read_csv(backup)
                combined = (
                    pd.concat([old_rows, new_rows], ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                combined.to_csv(cache_path, index=False)
                backup.unlink(missing_ok=True)
            else:
                backup.rename(cache_path)
    ok = _asos_date_present(city, date_str)
    if not ok:
        print(f"  ASOS still missing after fetch for {city}/{date_str}")
    return ok


def ensure_openmeteo_for_date(city: str, date_str: str) -> None:
    """Best-effort Open-Meteo cache fill for a single date."""
    target = date.fromisoformat(date_str)
    om = ng.load_openmeteo_tmax(city, target, target)
    if om is not None and not om.empty:
        val = pd.to_numeric(om.iloc[0].get("nwp_tmax_openmeteo"), errors="coerce")
        if np.isfinite(val):
            return
    try:
        meta = ng.STATION_META[city]
        fetched = ng.fetch_openmeteo_tmax(city, meta, target, target)
        cache_path = ng._openmeteo_cache_path(city)
        if cache_path.exists():
            existing = pd.read_csv(cache_path)
            existing["date"] = pd.to_datetime(
                existing["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            combined = pd.concat([existing, fetched], ignore_index=True)
        else:
            combined = fetched
        combined = (
            combined.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path, index=False)
        print(f"  Open-Meteo backfill: {city} {date_str}")
    except Exception as exc:
        print(f"  Open-Meteo fetch failed for {city}/{date_str}: {exc}")


def date_range(start: str, end: str) -> list[str]:
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    if end_d < start_d:
        raise ValueError(f"--end {end} is before --start {start}")
    out: list[str] = []
    cur = start_d
    while cur <= end_d:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill rolling-bias residuals (raw NGBoost mu - WU actual)"
    )
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-08")
    args = parser.parse_args()

    dates = date_range(args.start, args.end)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    print(
        f"Backfill residuals {args.start} → {args.end} "
        f"({len(dates)} days × {len(DEPLOY_CITIES)} cities)"
    )
    print(f"  yesterday={yesterday} model={MODEL_DIR}")

    # Allow historical HRRR via monkeypatch (ensure_hrrr rejects dates < today-1).
    ngboost_live.ensure_hrrr = ensure_hrrr_backfill

    models = NgBoostLiveModels(MODEL_DIR)
    existing = load_residuals_df()
    existing_keys = set(zip(existing["city"].astype(str), existing["date"].astype(str)))
    print(f"  existing residual rows: {len(existing)}")

    if WU_PATH.exists():
        wu_df = pd.read_parquet(WU_PATH)
        wu_df["city"] = wu_df["city"].astype(str)
        wu_df["date"] = pd.to_datetime(wu_df["date"]).dt.strftime("%Y-%m-%d")
    else:
        wu_df = pd.DataFrame(columns=["city", "date", "wunderground_tmax"])

    new_rows: list[dict] = []
    skipped_existing = 0
    skipped_other = 0

    for event_date in dates:
        for city in DEPLOY_CITIES:
            key = (city, event_date)
            if key in existing_keys:
                skipped_existing += 1
                continue

            print(f"\n=== {city} {event_date} ===")
            if not ensure_hrrr_backfill(city, event_date, timeout=HRRR_TIMEOUT_SEC):
                print(f"  SKIP: HRRR unavailable")
                skipped_other += 1
                continue

            ensure_wu_current(city, event_date)
            if not ensure_asos_for_date(city, event_date):
                print(f"  SKIP: ASOS early-morning unavailable")
                skipped_other += 1
                continue
            ensure_openmeteo_for_date(city, event_date)

            feat = build_live_features(city, event_date)
            if feat is None:
                reason = last_feature_fail_reason(city, event_date) or "unknown"
                print(f"  SKIP: features failed ({reason})")
                skipped_other += 1
                continue

            mu, _sigma = predict_ngboost_from_features(models, feat)
            if not np.isfinite(mu):
                print(f"  SKIP: non-finite mu={mu}")
                skipped_other += 1
                continue

            wu_actual = load_wu_actual(city, event_date, wu_df)
            if wu_actual is None:
                print(f"  SKIP: WU actual unavailable")
                skipped_other += 1
                continue

            residual = float(mu) - float(wu_actual)
            row = {
                "city": city,
                "date": event_date,
                "forecast": float(mu),
                "wu_actual": float(wu_actual),
                "residual": residual,
            }
            new_rows.append(row)
            existing_keys.add(key)
            print(
                f"  ADD: mu={mu:.3f} wu={wu_actual:.0f} residual={residual:+.3f}"
            )

    print(f"\n=== SUMMARY ===")
    print(f"  added: {len(new_rows)}")
    print(f"  skipped existing: {skipped_existing}")
    print(f"  skipped other: {skipped_other}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        per_city = new_df.groupby("city").size().to_dict()
        for city in DEPLOY_CITIES:
            print(f"  +{per_city.get(city, 0)} {city}")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
        save_residuals_and_snapshot(combined)
        print(f"  wrote {len(combined)} rows → residuals + snapshot")
        final = combined
    else:
        print("  nothing to write")
        final = existing

    print("\n=== Latest residual date per city ===")
    latest = final.groupby("city")["date"].max().sort_index()
    for city, d in latest.items():
        print(f"  {city}: {d}")

    print(f"\n=== Rolling bias snapshot ({SNAPSHOT_PATH}) ===")
    if SNAPSHOT_PATH.exists():
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        for city in sorted(snapshot):
            entry = snapshot[city]
            print(
                f"  {city}: ewma={entry.get('ewma')} "
                f"n_obs={entry.get('n_obs')} as_of={entry.get('as_of_date')}"
            )
    else:
        print("  snapshot missing")

    if not final.empty:
        max_date = str(final["date"].max())
        if max_date > args.end:
            raise SystemExit(
                f"LOOKAHEAD FAIL: max residual date {max_date} > --end {args.end}"
            )
        if max_date > yesterday:
            raise SystemExit(
                f"LOOKAHEAD FAIL: max residual date {max_date} > yesterday {yesterday}"
            )
        print(f"\n  lookahead OK: max residual date={max_date} <= yesterday={yesterday}")


if __name__ == "__main__":
    main()
