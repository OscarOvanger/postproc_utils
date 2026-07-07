"""Shared helpers for the Polymarket backtest pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scan_modal_buckets import compute_midpoint  # noqa: E402
from src.polymarket_api import parse_bucket_label  # noqa: E402
from src.poly_trading_pipeline import poly_taker_fee  # noqa: E402
from src.sizing import daily_cap_from_bankroll, effective_probability  # noqa: E402

import train_ngboost as ng  # noqa: E402

import train_ngboost as ng  # noqa: E402

POLY_CITIES: list[str] = sorted(ng.STATION_META.keys())
HRRR_PARQUET = PROJECT_ROOT / "data" / "hrrr_v2" / "hrrr_all_cities.parquet"
WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
POLY_HISTORY_DIR = PROJECT_ROOT / "data" / "polymarket_history"
SNAPSHOTS_DIR = POLY_HISTORY_DIR / "snapshots"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRADES_DIR = PROJECT_ROOT / "data" / "backtest_trades"
EQUITY_DIR = PROJECT_ROOT / "data" / "backtest_equity"
DEPLOY_CONFIG_PATH = PROJECT_ROOT / "config" / "deploy_config.json"
MODEL_PATH_FILE = REPORTS_DIR / "backtest_model_path.txt"
ELIGIBLE_DATES_CSV = REPORTS_DIR / "backtest_eligible_dates.csv"

LIVE_MODEL_DIR = PROJECT_ROOT / "models" / "ngboost"
TRACKB_DIR = PROJECT_ROOT / "models" / "trackb"

ENTRY_WINDOW_START = dt_time(9, 55)
ENTRY_WINDOW_END = dt_time(10, 10)
ENTRY_TARGET_TIME = dt_time(10, 2, 30)

MIN_ENTRY_ASK = 0.35
MAX_ENTRY_ASK = 0.60
MAKER_TICK = 0.01
MODAL_CONTRACTS = 5
PRICE_FLOOR = 0.10
PROFIT_TARGET = 0.15

INITIAL_BANKROLL_USD = 100.0
ELIMINATION_USD = 70.0
MAX_POSITION_PCT = 0.30
MODAL_DAILY_CAP_USD = 6.0

TRACKB_POLY_MAP: dict[str, str | None] = {
    "austin": "austin",
    "chicago": "chicago_midway",
    "houston": "houston",
    "los_angeles": "los_angeles",
    "new_york": "new_york_city",
    "san_francisco": "san_francisco",
    "dallas": None,
    "seattle": None,
    "miami": None,
    "atlanta": None,
}

TRACKB_ONLY_CITIES = ["oklahoma_city", "philadelphia", "phoenix"]

REGIONS: dict[str, list[str]] = {
    "gulf_south": ["houston", "austin", "dallas", "miami", "atlanta"],
    "west_coast": ["los_angeles", "san_francisco", "seattle"],
    "northeast_midwest": ["new_york", "chicago"],
}

CITY_TO_REGION: dict[str, str] = {
    city: region for region, cities in REGIONS.items() for city in cities
}


def city_timezone(city: str) -> str:
    return str(ng.STATION_META[city]["tz"])


def entry_timestamps(city: str, date_str: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (window_start, window_end, target_entry) as UTC timestamps."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(city_timezone(city))
    local_date = datetime.fromisoformat(date_str).date()
    window_start = pd.Timestamp(datetime.combine(local_date, ENTRY_WINDOW_START, tzinfo=tz)).tz_convert("UTC")
    window_end = pd.Timestamp(datetime.combine(local_date, ENTRY_WINDOW_END, tzinfo=tz)).tz_convert("UTC")
    target = pd.Timestamp(datetime.combine(local_date, ENTRY_TARGET_TIME, tzinfo=tz)).tz_convert("UTC")
    return window_start, window_end, target


def assert_no_lookahead(
    df: pd.DataFrame,
    ts_col: str,
    entry_ts: pd.Timestamp,
    *,
    label: str = "",
) -> tuple[pd.DataFrame, int]:
    """Filter rows to timestamps <= entry_ts; return filtered df and excluded count."""
    if df.empty:
        return df, 0
    ts = pd.to_datetime(df[ts_col], utc=True)
    entry = pd.Timestamp(entry_ts).tz_convert("UTC") if pd.Timestamp(entry_ts).tzinfo else pd.Timestamp(entry_ts, tz="UTC")
    excluded = int((ts > entry).sum())
    if excluded:
        prefix = f"{label}: " if label else ""
        print(f"  LOOKAHEAD FILTER: {prefix}excluded {excluded} rows after {entry.isoformat()}")
    return df.loc[ts <= entry].copy(), excluded


def load_day_snapshot(city: str, date_str: str) -> pd.DataFrame | None:
    path = SNAPSHOTS_DIR / city / f"{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def snapshot_in_entry_window(frame: pd.DataFrame, city: str, date_str: str) -> pd.DataFrame:
    window_start, window_end, _ = entry_timestamps(city, date_str)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[(ts >= window_start) & (ts <= window_end)].copy()


def has_entry_window_snapshot(frame: pd.DataFrame, city: str, date_str: str) -> bool:
    return not snapshot_in_entry_window(frame, city, date_str).empty


def select_entry_snapshot(
    frame: pd.DataFrame,
    city: str,
    date_str: str,
) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    """Pick snapshot nearest target entry time within 09:55-10:10 local."""
    window_start, window_end, target = entry_timestamps(city, date_str)
    filtered, excluded = assert_no_lookahead(frame, "timestamp", window_end, label="entry_window")
    windowed = snapshot_in_entry_window(filtered, city, date_str)
    if windowed.empty:
        return windowed, target, excluded

    ts = pd.to_datetime(windowed["timestamp"], utc=True)
    nearest_idx = (ts - target).abs().idxmin()
    entry_ts = ts.loc[nearest_idx]
    snap_rows = windowed[windowed["timestamp"] == entry_ts].copy()
    return snap_rows, entry_ts, excluded


def quotes_at_entry(snap_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per bucket at entry snapshot with midpoint."""
    out = snap_rows.copy()
    out = out[~out["bucket"].astype(str).str.startswith("Will ")]
    out["midpoint"] = out.apply(
        lambda row: compute_midpoint(row.get("best_bid"), row.get("best_ask")),
        axis=1,
    )
    return out.dropna(subset=["midpoint"])


def compute_modal_bucket(snap_rows: pd.DataFrame) -> pd.Series | None:
    quotes = quotes_at_entry(snap_rows)
    if quotes.empty:
        return None
    idx = quotes["midpoint"].astype(float).idxmax()
    return quotes.loc[idx]


def intraday_snapshots_after_entry(
    frame: pd.DataFrame,
    city: str,
    date_str: str,
    entry_ts: pd.Timestamp,
) -> pd.DataFrame:
    """All snapshots after entry through end of local event day."""
    from zoneinfo import ZoneInfo

    filtered, _ = assert_no_lookahead(frame, "timestamp", pd.Timestamp(frame["timestamp"].max()), label="intraday")
    tz = ZoneInfo(city_timezone(city))
    local_date = datetime.fromisoformat(date_str).date()
    day_end = pd.Timestamp(datetime.combine(local_date, dt_time(23, 59, 59), tzinfo=tz)).tz_convert("UTC")
    entry = pd.Timestamp(entry_ts).tz_convert("UTC")
    ts = pd.to_datetime(filtered["timestamp"], utc=True)
    return filtered.loc[(ts > entry) & (ts <= day_end)].sort_values("timestamp")


def check_profit_target_exit(
    intraday: pd.DataFrame,
    bucket: str,
    entry_price: float,
    target: float = PROFIT_TARGET,
) -> tuple[bool, float | None]:
    bucket_rows = intraday[intraday["bucket"].astype(str) == str(bucket)].copy()
    for _, row in bucket_rows.iterrows():
        bid = row.get("best_bid")
        if bid is not None and float(bid) >= entry_price + target:
            return True, round(entry_price + target, 4)
    return False, None


def _parse_snapshot_bucket(label: str) -> dict[str, object]:
    import re

    text = str(label).strip()
    le = re.match(r"^<=(\d+)$", text)
    if le:
        return {"type": "LESS_THAN", "lower": None, "upper": int(le.group(1))}
    ge = re.match(r"^>=(\d+)$", text)
    if ge:
        return {"type": "GREATER_THAN", "lower": int(ge.group(1)), "upper": None}
    rng = re.match(r"^(\d+)-(\d+)$", text)
    if rng:
        return {"type": "RANGE", "lower": int(rng.group(1)), "upper": int(rng.group(2))}
    return parse_bucket_label(text)


def temp_in_bucket(tmax: float, label: str) -> bool:
    try:
        parsed = _parse_snapshot_bucket(label)
    except ValueError:
        return False
    t = int(round(tmax))
    btype = parsed["type"]
    if btype == "RANGE":
        return int(parsed["lower"]) <= t <= int(parsed["upper"])
    if btype == "LESS_THAN":
        return t <= int(parsed["upper"])
    if btype == "GREATER_THAN":
        return t >= int(parsed["lower"])
    return False


def settlement_pnl(
    *,
    n_contracts: float,
    entry_price: float,
    won: bool,
    is_taker: bool = False,
) -> float:
    per = (1.0 - entry_price) if won else (-entry_price)
    gross = per * n_contracts
    fee = poly_taker_fee(int(n_contracts), entry_price) if is_taker else 0.0
    return round(gross - fee, 4)


def profit_target_pnl(n_contracts: float, entry_price: float, exit_price: float) -> float:
    return round((exit_price - entry_price) * n_contracts, 4)


def load_wu_targets() -> pd.DataFrame:
    df = pd.read_parquet(WU_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["city"] = df["city"].astype(str)
    return df


def load_hrrr_all() -> pd.DataFrame:
    df = pd.read_parquet(HRRR_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["city"] = df["city"].astype(str)
    return df


def features_eligible_cached(city: str, event_date: str) -> bool:
    """Check NGBoost feature availability using local caches only (no API fetch)."""
    from datetime import date as date_cls, timedelta

    hrrr = ng.load_hrrr_city(city)
    hrrr["date"] = pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d")
    if event_date not in set(hrrr["date"]):
        return False

    wu = ng._load_wu_all()
    wu = wu[wu["reliable"].astype(bool)].copy()
    wu["date"] = pd.to_datetime(wu["date"]).dt.strftime("%Y-%m-%d")
    wu_city = wu[(wu["city"] == city) & (wu["date"] < event_date)]

    target_dt = date_cls.fromisoformat(event_date)
    d1 = (target_dt - timedelta(days=1)).isoformat()
    d2 = (target_dt - timedelta(days=2)).isoformat()

    def wu_on(d: str) -> float:
        row = wu_city[wu_city["date"] == d]
        return float(row["wunderground_tmax"].iloc[0]) if len(row) else np.nan

    if not np.isfinite(wu_on(d1)) or not np.isfinite(wu_on(d2)):
        return False

    asos = ng.load_temp_early_morning(city, target_dt, target_dt)
    om = ng.load_openmeteo_tmax(city, target_dt, target_dt)
    if asos.empty or om.empty:
        return False

    hrrr_row = hrrr[hrrr["date"] == event_date].iloc[0]
    for col in ["hrrr_tmax", "peak_cloud_cover", "peak_solar_flux", "snow_depth"]:
        val = hrrr_row.get(col)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return False
    return True


def output_exists(path: Path, min_rows: int = 1) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".jsonl":
        count = sum(1 for _ in path.open(encoding="utf-8") if _.strip())
        return count >= min_rows
    if path.suffix == ".csv":
        try:
            return len(pd.read_csv(path)) >= min_rows
        except Exception:
            return False
    return path.stat().st_size > 0


def skip_if_exists(path: Path, force: bool, label: str) -> bool:
    if force:
        return False
    if output_exists(path):
        print(f"SKIP {label}: {path} already exists (use --force to recompute)")
        return True
    return False


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def configure_output_tag(tag: str) -> None:
    """Switch trade/equity output directories (e.g. tag='v5' → backtest_trades_v5)."""
    global TRADES_DIR, EQUITY_DIR
    suffix = f"_{tag}" if tag else ""
    TRADES_DIR = PROJECT_ROOT / "data" / f"backtest_trades{suffix}"
    EQUITY_DIR = PROJECT_ROOT / "data" / f"backtest_equity{suffix}"


def filter_eligible_by_date(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Filter eligible city-dates to an inclusive date window."""
    out = df.copy()
    out["date"] = out["date"].astype(str)
    if start:
        out = out[out["date"] >= start]
    if end:
        out = out[out["date"] <= end]
    return out.reset_index(drop=True)


def load_trading_config() -> dict[str, Any]:
    with open(DEPLOY_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def daily_budget_ngboost(bankroll_usd: float, config: dict[str, Any] | None = None) -> float:
    cfg = config or load_trading_config()
    return daily_cap_from_bankroll(bankroll_usd, cfg)


def max_trades_per_day(config: dict[str, Any] | None = None) -> int:
    cfg = config or load_trading_config()
    return int(cfg.get("max_trades_per_day", 2))


def shrinkage_lambda(config: dict[str, Any] | None = None) -> float:
    cfg = config or load_trading_config()
    return float(cfg.get("shrinkage_lambda", 1.0))


def print_trackb_mapping_table() -> None:
    print("\n=== TrackB ↔ Polymarket city mapping ===")
    print(f"{'Polymarket':<16} {'TrackB':<18} {'Notes'}")
    print("-" * 60)
    notes = {
        "chicago": "KORD vs KMDW",
        "new_york": "KLGA vs NYC",
    }
    for city in POLY_CITIES:
        trackb = TRACKB_POLY_MAP.get(city)
        note = notes.get(city, "overlap" if trackb else "NGBoost only")
        print(f"{city:<16} {trackb or '—':<18} {note}")
    for city in TRACKB_ONLY_CITIES:
        print(f"{'—':<16} {city:<18} TrackB only, no Polymarket")
