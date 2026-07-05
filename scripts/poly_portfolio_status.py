#!/usr/bin/env python3
"""Extended Polymarket portfolio report: balance, trades, open orders, forecasts, PnL.

Usage:
  .venv/bin/python scripts/poly_portfolio_status.py
  .venv/bin/python scripts/poly_portfolio_status.py --no-forecasts

Bid/ask at entry: from auto_trader state when available; otherwise N/A.
Settled PnL uses Wunderground actuals from wunderground_targets.parquet.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poly_order_status import (  # noqa: E402
    _normalize_orders,
    _order_id,
    _to_float,
    _token_id,
    fetch_gamma_token_labels,
    fetch_open_orders,
    load_posted_orders,
    load_token_labels,
)
from src.polymarket_api import (  # noqa: E402
    CLOB_HOST,
    DEFAULT_MARKETS_PATH,
    ORDER_LOG_PATH,
    PolymarketClient,
    _parse_order_book_sides,
    load_markets_map,
    parse_bucket_label,
)
from src.poly_trading_pipeline import poly_taker_fee  # noqa: E402

WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
AUTO_STATE_GLOB = "auto_trader_state_*.json"
PAPER_LOG = PROJECT_ROOT / "logs" / "poly_paper_trades.jsonl"
DEFAULT_STARTING_BANKROLL = 100.0

WU_CITY_MAP: dict[str, str] = {
    "new_york_city": "new_york",
    "new_york": "new_york",
    "chicago_midway": "chicago",
    "chicago": "chicago",
    "los_angeles": "los_angeles",
    "san_francisco": "san_francisco",
    "houston": "houston",
    "austin": "austin",
    "dallas": "dallas",
    "seattle": "seattle",
    "atlanta": "atlanta",
    "miami": "miami",
}

TRACKB_CITY_MAP: dict[str, str | None] = {
    "austin": "austin",
    "houston": "houston",
    "los_angeles": "los_angeles",
    "san_francisco": "san_francisco",
    "new_york": "new_york_city",
    "new_york_city": "new_york_city",
    "chicago": "chicago_midway",
    "chicago_midway": "chicago_midway",
    "dallas": None,
    "seattle": None,
}

NGBOOST_CITIES = {
    "austin", "houston", "los_angeles", "dallas", "chicago",
    "san_francisco", "seattle", "new_york", "miami", "atlanta",
}


@dataclass
class TokenMeta:
    token_id: str
    city: str
    city_display: str
    bucket_label: str
    event_date: str


@dataclass
class TradeRecord:
    order_id: str
    placed_at: str
    city: str
    bucket_label: str
    event_date: str
    n_contracts: float
    entry_price: float
    fill_price: float | None
    is_taker: bool
    bid_at_entry: float | None = None
    ask_at_entry: float | None = None
    status: str = "unknown"
    winning_bucket: str | None = None
    won: bool | None = None
    pnl_usd: float | None = None
    trackb_f: int | None = None
    ngboost_mu: float | None = None
    actual_tmax_f: float | None = None


@dataclass
class HoldingRow:
    label: str
    shares: float
    entry_price: float
    cost_usd: float
    mark_bid: float | None
    mark_value: float
    unrealized_usd: float


@dataclass
class OpenPosition:
    kind: str  # resting_order | held_shares
    order_id: str | None
    placed_at: str | None
    city: str
    bucket_label: str
    event_date: str
    n_contracts: float
    entry_price: float | None
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    trackb_f: int | None = None
    ngboost_mu: float | None = None
    modal_bucket: str | None = None
    is_modal: bool | None = None


@contextlib.contextmanager
def _suppress_output():
    """Silence noisy fetch logs from train_ngboost / run_daily_trade."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _paper_city_keys(city_slug: str) -> list[str]:
    keys = [TRACKB_CITY_MAP.get(city_slug), city_slug, WU_CITY_MAP.get(city_slug, city_slug)]
    if city_slug == "new_york":
        keys.append("new_york_city")
    if city_slug == "new_york_city":
        keys.append("new_york")
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def wu_city_slug(city: str) -> str:
    return WU_CITY_MAP.get(city, city)


def infer_event_date(record: dict[str, Any], meta: TokenMeta | None, today: str) -> str:
    if meta and meta.event_date:
        return meta.event_date
    ts = str(record.get("timestamp", ""))
    if len(ts) >= 10:
        return ts[:10]
    return today


def order_dates_from_posted(posted: dict[str, dict[str, Any]]) -> set[str]:
    dates: set[str] = set()
    for record in posted.values():
        ts = record.get("timestamp")
        if ts:
            dates.add(str(ts)[:10])
    dates.add(str(date.today()))
    return dates


def enrich_token_index_for_dates(index: dict[str, TokenMeta], event_dates: set[str]) -> None:
    today = str(date.today())
    for event_date in sorted(event_dates):
        with _suppress_output():
            enrich_token_index_from_scan(
                index,
                event_date,
                include_closed=event_date < today,
            )
        time.sleep(0.05)


def _city_slug_from_label(label: str) -> str:
    text = label.lower()
    for display, slug in [
        ("san francisco", "san_francisco"),
        ("los angeles", "los_angeles"),
        ("new york", "new_york"),
        ("oklahoma city", "oklahoma_city"),
        ("chicago", "chicago"),
    ]:
        if display in text:
            return slug.replace(" ", "_")
    parts = text.split()
    return parts[0] if parts else "unknown"


def _parse_label_parts(label: str) -> tuple[str, str, str]:
    """Parse 'City bucket' label into (city_display, city_slug, bucket)."""
    if " " not in label:
        return label, _city_slug_from_label(label), label
    # bucket labels often contain °F and digits; city is leading words before temp
    tokens = label.split()
    for idx, tok in enumerate(tokens):
        if any(ch.isdigit() for ch in tok) or "°" in tok or tok.startswith("<") or tok.startswith(">"):
            city_display = " ".join(tokens[:idx]).strip()
            bucket = " ".join(tokens[idx:]).strip()
            return city_display, _city_slug_from_label(city_display), bucket
    return label, _city_slug_from_label(label), ""


def build_token_index(
    *,
    event_dates: set[str],
    refresh_labels: bool,
) -> dict[str, TokenMeta]:
    index: dict[str, TokenMeta] = {}

    cached = load_markets_map(DEFAULT_MARKETS_PATH)
    if cached:
        for market in cached.get("markets", []):
            city = str(market.get("city", ""))
            city_display = str(market.get("city_display") or city.replace("_", " ").title())
            event_date = str(market.get("event_date", ""))
            for bucket in market.get("buckets", []):
                token = str(bucket.get("token_id", ""))
                if not token:
                    continue
                index[token] = TokenMeta(
                    token_id=token,
                    city=city,
                    city_display=city_display,
                    bucket_label=str(bucket.get("label", "")),
                    event_date=event_date,
                )

    if refresh_labels:
        for event_date in sorted(event_dates):
            try:
                labels = fetch_gamma_token_labels(event_date)
            except Exception:
                continue
            for token, label in labels.items():
                if token in index:
                    continue
                city_display, city_slug, bucket = _parse_label_parts(label)
                index[token] = TokenMeta(
                    token_id=token,
                    city=city_slug,
                    city_display=city_display.title() if city_display else city_slug,
                    bucket_label=bucket or label,
                    event_date=event_date,
                )
            time.sleep(0.1)

        today = str(date.today())
        if today not in event_dates:
            try:
                for token, label in fetch_gamma_token_labels(today).items():
                    if token not in index:
                        city_display, city_slug, bucket = _parse_label_parts(label)
                        index[token] = TokenMeta(
                            token_id=token,
                            city=city_slug,
                            city_display=city_display.title() if city_display else city_slug,
                            bucket_label=bucket or label,
                            event_date=today,
                        )
            except Exception:
                pass

    return index


def enrich_token_index_from_scan(
    index: dict[str, TokenMeta],
    event_date: str,
    *,
    include_closed: bool = False,
) -> None:
    """Fill missing labels using scan_modal_buckets discovery (fast, weather cities)."""
    try:
        from scan_modal_buckets import TARGET_CITIES, discover_markets
    except ImportError:
        return

    discovered = discover_markets(event_date, include_closed=include_closed)
    for slug, _display in TARGET_CITIES:
        market = discovered.get(slug)
        if not market:
            continue
        for bucket in market.get("buckets", []):
            token = str(bucket.get("token_id", ""))
            if not token:
                continue
            index[token] = TokenMeta(
                token_id=token,
                city=slug,
                city_display=_display,
                bucket_label=str(bucket.get("label", "")),
                event_date=event_date,
            )


def load_auto_trader_entry_books() -> dict[str, dict[str, float | None]]:
    """order_id -> {best_bid_at_entry, best_ask_at_entry, maker_entry_price}."""
    out: dict[str, dict[str, float | None]] = {}

    def _merge(order_id: str, pos: dict[str, Any]) -> None:
        out[str(order_id)] = {
            "best_bid_at_entry": _to_float(pos.get("best_bid_at_entry")),
            "best_ask_at_entry": _to_float(pos.get("best_ask_at_entry")),
            "maker_entry_price": _to_float(pos.get("maker_entry_price")),
        }

    for path in sorted((PROJECT_ROOT / "logs").glob(AUTO_STATE_GLOB)):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for pos in state.get("positions", []):
            oid = pos.get("order_id")
            if oid:
                _merge(str(oid), pos)

    if PAPER_LOG.exists():
        with open(PAPER_LOG, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for pos in row.get("positions", []):
                    oid = pos.get("order_id")
                    if oid:
                        _merge(str(oid), pos)
                    token = pos.get("yes_token_id") or pos.get("token_id")
                    if token:
                        out[f"token:{token}"] = {
                            "best_bid_at_entry": _to_float(pos.get("best_bid_at_entry")),
                            "best_ask_at_entry": _to_float(pos.get("best_ask_at_entry")),
                            "maker_entry_price": _to_float(pos.get("maker_entry_price")),
                        }
                for trade in row.get("trades", []):
                    token = trade.get("yes_token_id") or trade.get("token_id")
                    if not token:
                        continue
                    out[f"token:{token}"] = {
                        "best_bid_at_entry": _to_float(trade.get("best_bid")),
                        "best_ask_at_entry": _to_float(trade.get("best_ask")),
                        "maker_entry_price": _to_float(trade.get("maker_entry_price")),
                    }

    return out


def lookup_entry_book(
    entry_books: dict[str, dict[str, float | None]],
    *,
    order_id: str,
    token: str,
) -> dict[str, float | None]:
    if order_id in entry_books:
        return entry_books[order_id]
    token_key = f"token:{token}"
    if token_key in entry_books:
        return entry_books[token_key]
    return {}


def load_token_entry_prices(
    posted: dict[str, dict[str, Any]],
    token_index: dict[str, TokenMeta],
) -> dict[str, float]:
    """Latest log price per token for open-event holdings."""
    today = str(date.today())
    latest: dict[str, tuple[str, float]] = {}
    for record in sorted(posted.values(), key=lambda r: r.get("timestamp", "")):
        token = str(record.get("token_id", ""))
        price = _to_float(record.get("price"))
        if not token or price is None:
            continue
        meta = token_index.get(token)
        event_date = infer_event_date(record, meta, today)
        if event_date < today:
            continue
        ts = str(record.get("timestamp", ""))
        latest[token] = (ts, price)
    return {token: price for token, (_, price) in latest.items()}


def load_paper_forecasts() -> dict[tuple[str, str], dict[str, int]]:
    """(event_date, city_key) -> {trackb, trackb_raw} from poly_paper_trades.jsonl."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    if not PAPER_LOG.exists():
        return out
    with open(PAPER_LOG, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_date = str(row.get("date", ""))
            raw = row.get("raw_forecasts") or {}
            adj = row.get("forecasts") or raw
            for city, val in adj.items():
                try:
                    trackb = int(val)
                except (TypeError, ValueError):
                    continue
                raw_val = raw.get(city, val)
                try:
                    trackb_raw = int(raw_val)
                except (TypeError, ValueError):
                    trackb_raw = trackb
                out[(event_date, str(city))] = {
                    "trackb": trackb,
                    "trackb_raw": trackb_raw,
                }
    return out


def build_modal_buckets(
    event_date: str,
    token_index: dict[str, TokenMeta],
    book_cache: dict[str, tuple[float | None, float | None]],
) -> dict[str, str]:
    """city_slug -> modal bucket label (highest midpoint) for one event date."""
    by_city: dict[str, list[TokenMeta]] = {}
    for meta in token_index.values():
        if meta.event_date != event_date:
            continue
        by_city.setdefault(meta.city, []).append(meta)

    modals: dict[str, str] = {}
    for city, metas in by_city.items():
        best_label: str | None = None
        best_mid = -1.0
        for meta in metas:
            bid, ask = fetch_book_cached(meta.token_id, book_cache, fetch=True)
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2
            elif ask is not None:
                mid = ask
            elif bid is not None:
                mid = bid
            else:
                continue
            if mid > best_mid:
                best_mid = mid
                best_label = meta.bucket_label
        if best_label:
            modals[city] = best_label
    return modals


def load_wu_targets() -> pd.DataFrame:
    if not WU_PATH.exists():
        return pd.DataFrame(columns=["city", "date", "wunderground_tmax"])
    df = pd.read_parquet(WU_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["city"] = df["city"].astype(str)
    df["wunderground_tmax"] = pd.to_numeric(df["wunderground_tmax"], errors="coerce")
    return df


def temp_in_bucket(tmax: float, label: str) -> bool:
    parsed = parse_bucket_label(label)
    t = int(round(tmax))
    btype = parsed["type"]
    if btype == "RANGE":
        return int(parsed["lower"]) <= t <= int(parsed["upper"])
    if btype == "LESS_THAN":
        return t <= int(parsed["upper"])
    if btype == "GREATER_THAN":
        return t >= int(parsed["lower"])
    return False


def winning_bucket_for_city(
    wu: pd.DataFrame,
    city: str,
    event_date: str,
    bucket_labels: list[str],
) -> tuple[str | None, float | None]:
    wu_city = wu_city_slug(city)
    row = wu[(wu["city"] == wu_city) & (wu["date"] == event_date)]
    if row.empty:
        return None, None
    tmax = float(row.iloc[0]["wunderground_tmax"])
    if not np.isfinite(tmax):
        return None, None
    for label in bucket_labels:
        if temp_in_bucket(tmax, label):
            return label, tmax
    return None, tmax


def settlement_pnl(
    *,
    n_contracts: float,
    entry_price: float,
    won: bool,
    is_taker: bool,
) -> float:
    per = (1.0 - entry_price) if won else (-entry_price)
    gross = per * n_contracts
    fee = poly_taker_fee(int(n_contracts), entry_price) if is_taker else 0.0
    return round(gross - fee, 4)


def compute_sharpe(pnls: list[float], capitals: list[float]) -> float | None:
    if len(pnls) < 2:
        return None
    returns = [p / c if c > 0 else 0.0 for p, c in zip(pnls, capitals)]
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return None
    return float(np.mean(returns) / std)


def fetch_trackb_forecast(city_slug: str, event_date: str) -> int | None:
    trackb_city = TRACKB_CITY_MAP.get(city_slug)
    if not trackb_city:
        return None
    try:
        with _suppress_output():
            from run_daily_trade import fetch_forecast, load_city_config, load_deploy_config

            deploy = load_deploy_config(PROJECT_ROOT / "config" / "deploy_config.json")
            city_config = load_city_config(deploy)
            if trackb_city not in deploy.get("cities", []):
                return None
            mini = {**deploy, "cities": [trackb_city]}
            forecasts, _, _ = fetch_forecast(mini, event_date, city_config)
        return forecasts.get(trackb_city)
    except Exception:
        return None


class _NgBoostModels:
    def __init__(self) -> None:
        import joblib

        output_dir = PROJECT_ROOT / "models" / "ngboost"
        self.model = joblib.load(output_dir / "ngboost_global.pkl")
        self.scaler = joblib.load(output_dir / "feature_scaler.pkl")
        self.lgb_model = joblib.load(output_dir / "lgb_stage1.pkl")
        with open(output_dir / "model_config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        self.feature_cols = config["feature_columns"]
        self.stage1_cols = [c for c in self.feature_cols if c != "lgb_tmax_pred"]
        self.fill_medians = config.get("nan_fill_medians", {})


def _ensure_hrrr_for_event_date(city_slug: str, event_date: str) -> bool:
    """Best-effort single-day HRRR fetch when cache lacks a near-term event date."""
    from datetime import date as date_cls, timedelta

    from fetch_hrrr_all_cities import (
        HRRR_STATIONS,
        _init_download_pool,
        _shutdown_download_pool,
        fetch_hrrr_for_date,
        load_monthly_cache,
        monthly_cache_path,
        write_monthly_row,
    )
    import train_ngboost as ng

    if city_slug not in HRRR_STATIONS:
        return False
    target = date_cls.fromisoformat(event_date)
    today = date_cls.today()
    if target < today - timedelta(days=1) or target > today + timedelta(days=14):
        return False

    hrrr = ng.load_hrrr_city(city_slug)
    cached = set(pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d"))
    if event_date in cached:
        return True

    try:
        with _suppress_output():
            _init_download_pool(8)
            try:
                row = fetch_hrrr_for_date(HRRR_STATIONS[city_slug], target)
            finally:
                _shutdown_download_pool()
        tmax = row.get("hrrr_tmax")
        if tmax is None or (isinstance(tmax, float) and np.isnan(tmax)):
            return False
        path = monthly_cache_path(city_slug, target)
        cache = load_monthly_cache(path)
        write_monthly_row(cache, path, row)
        return True
    except Exception:
        return False


def explain_ngboost_unavailable(city_slug: str, event_date: str) -> str:
    """Short reason string when fetch_ngboost_forecast returns None."""
    if city_slug not in NGBOOST_CITIES:
        return "city not in NGBOOST_CITIES"
    try:
        import train_ngboost as ng
        from datetime import date as date_cls, timedelta

        hrrr = ng.load_hrrr_city(city_slug)
        hrrr["date"] = pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d")
        if event_date not in set(hrrr["date"]):
            last = hrrr["date"].max() if len(hrrr) else "none"
            return f"no HRRR row for {event_date} (cache ends {last})"
        target_dt = date_cls.fromisoformat(event_date)
        with _suppress_output():
            asos = ng.load_temp_early_morning(city_slug, target_dt, target_dt)
            om = ng.load_openmeteo_tmax(city_slug, target_dt, target_dt)
            if om.empty:
                om = ng.fetch_openmeteo_tmax(city_slug, ng.STATION_META[city_slug], target_dt, target_dt)
        if asos.empty:
            return "ASOS early-morning temp unavailable"
        if om.empty:
            return "Open-Meteo NWP unavailable"
        return "model inference failed"
    except Exception as exc:
        return str(exc)


def fetch_ngboost_forecast(
    city_slug: str,
    event_date: str,
    *,
    models: _NgBoostModels | None = None,
) -> float | None:
    if city_slug not in NGBOOST_CITIES:
        return None
    try:
        import train_ngboost as ng
        from datetime import date as date_cls, timedelta

        if models is None:
            with _suppress_output(), warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                models = _NgBoostModels()

        _ensure_hrrr_for_event_date(city_slug, event_date)
        hrrr = ng.load_hrrr_city(city_slug)
        hrrr["date"] = pd.to_datetime(hrrr["date"]).dt.strftime("%Y-%m-%d")
        if event_date not in set(hrrr["date"]):
            return None

        wu = ng._load_wu_all()
        wu = wu[wu["reliable"].astype(bool)].copy()
        wu["date"] = pd.to_datetime(wu["date"]).dt.strftime("%Y-%m-%d")
        wu_city = wu[wu["city"] == city_slug]

        target_dt = date_cls.fromisoformat(event_date)
        d1 = (target_dt - timedelta(days=1)).isoformat()
        d2 = (target_dt - timedelta(days=2)).isoformat()

        def wu_on(d: str) -> float:
            row = wu_city[wu_city["date"] == d]
            return float(row["wunderground_tmax"].iloc[0]) if len(row) else np.nan

        def hrrr_on(d: str) -> float:
            row = hrrr[hrrr["date"] == d]
            return float(row["hrrr_tmax"].iloc[0]) if len(row) else np.nan

        hrrr_row = hrrr[hrrr["date"] == event_date].iloc[0]
        d3s = [(target_dt - timedelta(days=i)).isoformat() for i in range(1, 4)]
        d7s = [(target_dt - timedelta(days=i)).isoformat() for i in range(1, 8)]
        wu_d1, hrrr_d1 = wu_on(d1), hrrr_on(d1)

        meta = ng.STATION_META[city_slug]
        doy = pd.Timestamp(event_date).dayofyear

        with _suppress_output():
            asos = ng.load_temp_early_morning(city_slug, target_dt, target_dt)
            om = ng.load_openmeteo_tmax(city_slug, target_dt, target_dt)
            if om.empty:
                om = ng.fetch_openmeteo_tmax(city_slug, meta, target_dt, target_dt)

        if asos.empty or om.empty:
            return None

        feat = {
            "hrrr_tmax": float(hrrr_row["hrrr_tmax"]),
            "peak_cloud_cover": float(hrrr_row["peak_cloud_cover"]),
            "peak_solar_flux": float(hrrr_row["peak_solar_flux"]),
            "snow_depth": float(hrrr_row["snow_depth"]),
            "temp_early_morning": float(asos["temp_early_morning"].iloc[0]),
            "nwp_tmax_openmeteo": float(om["nwp_tmax_openmeteo"].iloc[0]),
            "tmax_lag1": wu_on(d1),
            "tmax_lag2": wu_on(d2),
            "tmax_roll3": float(np.nanmean([wu_on(d) for d in d3s])),
            "tmax_roll7": float(np.nanmean([wu_on(d) for d in d7s])),
            "hrrr_error_lag1": wu_d1 - hrrr_d1 if np.isfinite(wu_d1) and np.isfinite(hrrr_d1) else np.nan,
            "station_id": ng.STATION_ID_MAP[city_slug],
            "latitude": float(meta["lat"]),
            "elevation": float(meta["elevation_ft"]),
            "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
            "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        }
        df = pd.DataFrame([feat])
        df = ng.apply_saved_median_fill(df, models.fill_medians, list(models.fill_medians.keys()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df["lgb_tmax_pred"] = float(models.lgb_model.predict(df[models.stage1_cols])[0])
        X = ng.transform_features(models.scaler, df, models.feature_cols)
        mu, _, _ = ng.predict_dist_params(models.model, X)
        return round(float(mu[0]), 1)
    except Exception:
        return None


class ForecastCache:
    def __init__(self, *, use_live: bool, paper: dict[tuple[str, str], dict[str, int]]):
        self.use_live = use_live
        self.paper = paper
        self.trackb: dict[tuple[str, str], int | None] = {}
        self.ngboost: dict[tuple[str, str], float | None] = {}
        self._ng_models: _NgBoostModels | None = None
        self._today = str(date.today())

    def _paper_trackb(self, city: str, event_date: str) -> int | None:
        for key_city in _paper_city_keys(city):
            paper_key = (event_date, key_city)
            if paper_key in self.paper:
                return self.paper[paper_key].get("trackb")
        return None

    def trackb_for(self, city: str, event_date: str) -> int | None:
        key = (event_date, city)
        if key in self.trackb:
            return self.trackb[key]
        val = self._paper_trackb(city, event_date)
        if val is None and self.use_live and event_date >= self._today:
            val = fetch_trackb_forecast(city, event_date)
        self.trackb[key] = val
        return val

    def ngboost_for(self, city: str, event_date: str) -> float | None:
        key = (city, event_date)
        if key in self.ngboost:
            return self.ngboost[key]
        val = None
        if self.use_live and event_date >= self._today:
            if self._ng_models is None:
                try:
                    with _suppress_output(), warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        self._ng_models = _NgBoostModels()
                except Exception:
                    self._ng_models = None
            if self._ng_models is not None:
                val = fetch_ngboost_forecast(city, event_date, models=self._ng_models)
        self.ngboost[key] = val
        return val


def fetch_book_silent(token_id: str) -> tuple[float | None, float | None]:
    """Fetch order book without printing on 404 (closed/expired markets)."""
    import requests

    try:
        response = requests.get(
            f"{CLOB_HOST}/book",
            params={"token_id": token_id},
            timeout=20,
        )
        response.raise_for_status()
        return _parse_order_book_sides(response.json())
    except Exception:
        return None, None


def fetch_book_cached(
    token: str,
    cache: dict[str, tuple[float | None, float | None]],
    *,
    fetch: bool = True,
) -> tuple[float | None, float | None]:
    if not token or not fetch:
        return None, None
    if token not in cache:
        cache[token] = fetch_book_silent(token)
        time.sleep(0.05)
    return cache[token]


def fmt_price(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"${value:.2f}"


def fmt_pnl(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:.2f}"


def collect_trades(
    client: PolymarketClient,
    posted: dict[str, dict[str, Any]],
    open_orders: list[dict[str, Any]],
    token_index: dict[str, TokenMeta],
    entry_books: dict[str, dict[str, float | None]],
    wu: pd.DataFrame,
    forecast_cache: ForecastCache,
    book_cache: dict[str, tuple[float | None, float | None]],
    modal_by_city: dict[str, str],
) -> tuple[list[TradeRecord], list[OpenPosition]]:
    settled: list[TradeRecord] = []
    open_rows: list[OpenPosition] = []
    today = str(date.today())

    open_by_id: dict[str, dict[str, Any]] = {}
    open_ids: set[str] = set()
    for order in open_orders:
        oid = _order_id(order)
        if oid:
            open_by_id[oid] = order
            open_ids.add(oid)

    buckets_by_city_date: dict[tuple[str, str], list[str]] = {}
    for meta in token_index.values():
        key = (meta.city, meta.event_date)
        buckets_by_city_date.setdefault(key, [])
        if meta.bucket_label not in buckets_by_city_date[key]:
            buckets_by_city_date[key].append(meta.bucket_label)

    for order_id, record in sorted(posted.items(), key=lambda x: x[1].get("timestamp", "")):
        token = str(record.get("token_id", ""))
        meta = token_index.get(token)
        event_date = infer_event_date(record, meta, today)
        city = meta.city if meta else "unknown"
        bucket = meta.bucket_label if meta else "?"
        city_display = meta.city_display if meta else city
        is_past = event_date < today

        is_taker = not bool(record.get("post_only", True))
        original = _to_float(record.get("size")) or 0.0
        entry_price = _to_float(record.get("price"))
        matched = 0.0
        state = "unknown"

        book = lookup_entry_book(entry_books, order_id=order_id, token=token)
        bid_at = book.get("best_bid_at_entry")
        ask_at = book.get("best_ask_at_entry")
        modal_bucket = modal_by_city.get(city)
        is_modal = (
            modal_bucket is not None
            and bucket not in ("?", "")
            and modal_bucket == bucket
        )

        if order_id in open_ids:
            oo = open_by_id.get(order_id, {})
            state = "open"
            matched = _to_float(oo.get("size_matched")) or 0.0
            original = _to_float(oo.get("original_size") or oo.get("size")) or original
            entry_price = _to_float(oo.get("price")) or entry_price
        else:
            status_token = None if is_past else (token or None)
            status_info = client.get_order_status(order_id, token_id=status_token)
            state = str(status_info.get("status", "unknown"))
            matched = _to_float(status_info.get("size_matched")) or 0.0
            original = _to_float(status_info.get("original_size")) or original
            entry_price = _to_float(status_info.get("fill_price")) or entry_price

        if state == "open" and order_id in open_ids and matched <= 0 and not is_past:
            best_bid, best_ask = fetch_book_cached(token, book_cache, fetch=True)
            mid = (
                (best_bid + best_ask) / 2
                if best_bid is not None and best_ask is not None
                else best_ask
            )
            open_rows.append(
                OpenPosition(
                    kind="resting_order",
                    order_id=order_id,
                    placed_at=str(record.get("timestamp", "")),
                    city=city_display,
                    bucket_label=bucket,
                    event_date=event_date,
                    n_contracts=original,
                    entry_price=_to_float(record.get("price")),
                    best_bid=best_bid,
                    best_ask=best_ask,
                    midpoint=mid,
                    trackb_f=forecast_cache.trackb_for(city, event_date),
                    ngboost_mu=forecast_cache.ngboost_for(city, event_date),
                    modal_bucket=modal_bucket,
                    is_modal=is_modal,
                )
            )
            continue

        if matched <= 0:
            continue

        trade = TradeRecord(
            order_id=order_id,
            placed_at=str(record.get("timestamp", "")),
            city=city_display,
            bucket_label=bucket,
            event_date=event_date,
            n_contracts=matched,
            entry_price=entry_price or 0.0,
            fill_price=_to_float(record.get("price")),
            is_taker=is_taker,
            bid_at_entry=bid_at,
            ask_at_entry=ask_at,
            status=state,
            trackb_f=forecast_cache.trackb_for(city, event_date),
            ngboost_mu=forecast_cache.ngboost_for(city, event_date),
        )

        winner, actual = winning_bucket_for_city(
            wu, city, event_date, buckets_by_city_date.get((city, event_date), [bucket])
        )
        trade.actual_tmax_f = actual
        trade.winning_bucket = winner

        if actual is not None and entry_price is not None and bucket not in ("?", ""):
            trade.won = temp_in_bucket(actual, bucket)
            trade.pnl_usd = settlement_pnl(
                n_contracts=matched,
                entry_price=entry_price,
                won=trade.won,
                is_taker=is_taker,
            )
            settled.append(trade)
        elif is_past:
            settled.append(trade)
        else:
            best_bid, best_ask = fetch_book_cached(token, book_cache, fetch=True)
            mid = (
                (best_bid + best_ask) / 2
                if best_bid is not None and best_ask is not None
                else best_ask
            )
            open_rows.append(
                OpenPosition(
                    kind="held_shares",
                    order_id=order_id,
                    placed_at=str(record.get("timestamp", "")),
                    city=city_display,
                    bucket_label=bucket,
                    event_date=event_date,
                    n_contracts=matched,
                    entry_price=entry_price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    midpoint=mid,
                    trackb_f=trade.trackb_f,
                    ngboost_mu=trade.ngboost_mu,
                    modal_bucket=modal_bucket,
                    is_modal=is_modal,
                )
            )

    seen_tokens = {str(r.get("token_id", "")) for r in posted.values()}
    for token, meta in token_index.items():
        if token in seen_tokens or meta.event_date < today:
            continue
        try:
            shares = client.get_conditional_balance(token)
        except Exception:
            shares = 0.0
        if shares <= 0:
            continue
        best_bid, best_ask = fetch_book_cached(token, book_cache, fetch=True)
        if best_bid is None and best_ask is None:
            continue
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else best_ask
        open_rows.append(
            OpenPosition(
                kind="held_shares",
                order_id=None,
                placed_at=None,
                city=meta.city_display,
                bucket_label=meta.bucket_label,
                event_date=meta.event_date,
                n_contracts=shares,
                entry_price=None,
                best_bid=best_bid,
                best_ask=best_ask,
                midpoint=mid,
                trackb_f=forecast_cache.trackb_for(meta.city, meta.event_date),
                ngboost_mu=forecast_cache.ngboost_for(meta.city, meta.event_date),
                modal_bucket=modal_by_city.get(meta.city),
                is_modal=(
                    modal_by_city.get(meta.city) == meta.bucket_label
                    if modal_by_city.get(meta.city)
                    else None
                ),
            )
        )

    return settled, open_rows


def print_portfolio(
    holdings: list[HoldingRow],
    cash: float,
) -> tuple[float, float, float]:
    cost_basis = sum(row.cost_usd for row in holdings)
    mark_value = sum(row.mark_value for row in holdings)
    unrealized = sum(row.unrealized_usd for row in holdings)

    print("\n=== Portfolio (pUSD) ===", flush=True)
    print(f"Cash:                  ${cash:.2f}")
    print(f"Open positions cost:   ${cost_basis:.2f}  (capital deployed at entry)")
    print(f"Open positions mark:   ${mark_value:.2f}  (current bid × shares)")
    print(f"Unrealized on open:    {fmt_pnl(unrealized)}")
    print(f"Total portfolio:       ${cash + mark_value:.2f}  (cash + mark; excludes unsettled PnL)")

    if holdings:
        print("\nOpen YES positions:")
        for row in holdings:
            pct = ""
            if row.cost_usd > 0:
                pct_chg = 100.0 * row.unrealized_usd / row.cost_usd
                pct = f" ({pct_chg:+.1f}%)"
            mark_str = fmt_price(row.mark_bid) if row.mark_bid is not None else "N/A"
            print(
                f"  {row.label}: {row.shares:.1f} sh | "
                f"paid {fmt_price(row.entry_price)} → cost {fmt_price(row.cost_usd)} | "
                f"bid now {mark_str} → mark {fmt_price(row.mark_value)} | "
                f"unrealized {fmt_pnl(row.unrealized_usd)}{pct}"
            )
    else:
        print("\nOpen YES positions: none")
    return cash, mark_value, cost_basis


def collect_holdings(
    client: PolymarketClient,
    tokens: set[str],
    token_index: dict[str, TokenMeta],
    book_cache: dict[str, tuple[float | None, float | None]],
    token_entry_prices: dict[str, float],
) -> list[HoldingRow]:
    today = str(date.today())
    holdings: list[HoldingRow] = []
    for token in sorted(tokens):
        meta = token_index.get(token)
        if meta and meta.event_date < today:
            continue
        try:
            shares = client.get_conditional_balance(token)
        except Exception:
            shares = 0.0
        if shares <= 0:
            continue
        best_bid, best_ask = fetch_book_cached(token, book_cache, fetch=True)
        if best_bid is None and best_ask is None:
            continue
        mark = best_bid if best_bid is not None else best_ask
        entry_price = token_entry_prices.get(token, 0.0)
        cost = entry_price * shares
        mark_value = (mark or 0.0) * shares
        if meta:
            label = f"{meta.city_display} {meta.bucket_label} ({meta.event_date})"
        else:
            label = f"token {token[:12]}..."
        holdings.append(
            HoldingRow(
                label=label,
                shares=shares,
                entry_price=entry_price,
                cost_usd=cost,
                mark_bid=mark,
                mark_value=mark_value,
                unrealized_usd=mark_value - cost,
            )
        )
    return holdings


def print_settled_trades(trades: list[TradeRecord]) -> None:
    resolved = [t for t in trades if t.pnl_usd is not None]
    pending = [t for t in trades if t.pnl_usd is None]

    print(f"\n=== Settled / closed trades ({len(resolved)} with PnL) ===")
    if not resolved:
        print("  (none settled yet)")
    else:
        header = (
            f"{'Placed':<20} {'City':<14} {'Bucket':<10} {'Qty':>4} {'Entry':>6} "
            f"{'Bid@':>6} {'Ask@':>6} {'Winner':<10} {'PnL':>8} {'TrackB':>7} {'NGB μ':>7}"
        )
        print(header)
        print("-" * len(header))
        for t in resolved:
            print(
                f"{t.placed_at[:19]:<20} {t.city:<14} {t.bucket_label:<10} {t.n_contracts:4.0f} "
                f"{fmt_price(t.entry_price):>6} {fmt_price(t.bid_at_entry):>6} "
                f"{fmt_price(t.ask_at_entry):>6} {str(t.winning_bucket or '?'):<10} "
                f"{fmt_pnl(t.pnl_usd):>8} "
                f"{str(t.trackb_f) if t.trackb_f is not None else '—':>7} "
                f"{f'{t.ngboost_mu:.1f}' if t.ngboost_mu is not None else '—':>7}"
            )
        print(
            "  TrackB/NGB: from paper log when saved; live NGB only for today's open event. "
            "Bid@/Ask@ from order-time logs when available."
        )

    if pending:
        print(f"\n=== Filled but unsettled ({len(pending)}) ===")
        for t in pending:
            print(
                f"  {t.placed_at[:19]} {t.city} {t.bucket_label} "
                f"{t.n_contracts:.0f} @ {fmt_price(t.entry_price)} event={t.event_date}"
            )


def print_open_positions(rows: list[OpenPosition]) -> None:
    print(f"\n=== Open positions / resting orders ({len(rows)}) ===")
    if not rows:
        print("  (none)")
        return
    header = (
        f"{'Type':<8} {'Placed':<20} {'City':<14} {'Bucket':<10} {'Modal?':>6} "
        f"{'Mkt modal':<10} {'Qty':>4} {'Our bid':>8} {'Bid now':>8} {'Ask now':>8} "
        f"{'Mid':>7} {'TrackB':>7} {'NGB μ':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        placed = (row.placed_at or "")[:19] or "—"
        modal_flag = (
            "YES" if row.is_modal else ("no" if row.is_modal is False else "—")
        )
        mkt_modal = row.modal_bucket or "—"
        print(
            f"{row.kind:<8} {placed:<20} {row.city:<14} {row.bucket_label:<10} "
            f"{modal_flag:>6} {mkt_modal:<10} {row.n_contracts:4.0f} "
            f"{fmt_price(row.entry_price):>8} {fmt_price(row.best_bid):>8} "
            f"{fmt_price(row.best_ask):>8} {fmt_price(row.midpoint):>7} "
            f"{str(row.trackb_f) if row.trackb_f is not None else '—':>7} "
            f"{f'{row.ngboost_mu:.1f}' if row.ngboost_mu is not None else '—':>7}"
        )


def print_pnl_summary(
    trades: list[TradeRecord],
    *,
    cash: float,
    open_cost: float,
    open_mark: float,
    starting_bankroll: float = DEFAULT_STARTING_BANKROLL,
) -> None:
    resolved = [t for t in trades if t.pnl_usd is not None]
    account_total = cash + open_cost
    total_pnl = account_total - starting_bankroll

    print("\n=== PnL summary ===")
    print(f"Starting bankroll:       ${starting_bankroll:.2f}")
    print(f"Cash now:                ${cash:.2f}")
    print(f"Open positions (cost):   ${open_cost:.2f}")
    print(f"Account total:           ${account_total:.2f}  (cash + open cost)")
    print(f"Total PnL vs start:      {fmt_pnl(total_pnl)}")
    print(f"  (mark-to-market total: ${cash + open_mark:.2f}, "
          f"unrealized on open {fmt_pnl(open_mark - open_cost)})")

    if not resolved:
        print("Settled trades:          0 with PnL recorded")
        return

    pnls = [float(t.pnl_usd) for t in resolved if t.pnl_usd is not None]
    capitals = [t.n_contracts * t.entry_price for t in resolved]
    wins = sum(1 for t in resolved if t.won)
    realized = sum(pnls)
    sharpe = compute_sharpe(pnls, capitals)

    print(f"Settled trades:          {len(resolved)}")
    print(f"Win rate:                {wins}/{len(resolved)} ({100.0 * wins / len(resolved):.1f}%)")
    print(f"Realized PnL (settled):  {fmt_pnl(realized)}")
    print(f"Avg PnL/trade:           {fmt_pnl(realized / len(resolved))}")
    if sharpe is not None:
        print(f"Sharpe (per-trade return on capital): {sharpe:.3f}")
    else:
        print("Sharpe:                  N/A (need 2+ settled trades)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended Polymarket portfolio status with trades, forecasts, and PnL."
    )
    parser.add_argument("--date", default=str(date.today()), help="Default event date for labels")
    parser.add_argument(
        "--fetch-labels",
        action="store_true",
        help="Fetch extra token labels from Gamma API (slower)",
    )
    parser.add_argument(
        "--no-forecasts",
        action="store_true",
        help="Skip live TrackB/NGBoost forecast fetches (use paper log only)",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=DEFAULT_STARTING_BANKROLL,
        help="Starting bankroll for total PnL (default: 100)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    client = PolymarketClient()
    posted_all = load_posted_orders(ORDER_LOG_PATH)
    open_orders = fetch_open_orders(client)
    open_ids = {_order_id(o) for o in open_orders}
    open_ids.discard(None)

    event_dates = order_dates_from_posted(posted_all)
    token_ids = {str(r.get("token_id", "")) for r in posted_all.values() if r.get("token_id")}
    for order in open_orders:
        tok = _token_id(order)
        if tok:
            token_ids.add(tok)

    token_index = build_token_index(
        event_dates=event_dates,
        refresh_labels=args.fetch_labels,
    )
    enrich_token_index_for_dates(token_index, event_dates)
    simple_labels = load_token_labels(
        event_date=args.date,
        token_ids=token_ids,
        refresh=args.fetch_labels,
    )
    today = str(date.today())
    for token, label in simple_labels.items():
        if token in token_index:
            continue
        city_display, city_slug, bucket = _parse_label_parts(label)
        order_date = args.date
        for record in posted_all.values():
            if str(record.get("token_id", "")) == token:
                order_date = str(record.get("timestamp", ""))[:10] or order_date
                break
        token_index[token] = TokenMeta(
            token_id=token,
            city=city_slug,
            city_display=city_display.title() if city_display else city_slug,
            bucket_label=bucket or label,
            event_date=order_date,
        )

    # Backfill event_date on index entries from order log when scan missed a token.
    for record in posted_all.values():
        token = str(record.get("token_id", ""))
        if not token or token not in token_index:
            continue
        meta = token_index[token]
        order_date = str(record.get("timestamp", ""))[:10]
        if meta.event_date == today and order_date and order_date != today:
            token_index[token] = TokenMeta(
                token_id=meta.token_id,
                city=meta.city,
                city_display=meta.city_display,
                bucket_label=meta.bucket_label,
                event_date=order_date,
            )

    entry_books = load_auto_trader_entry_books()
    token_entry_prices = load_token_entry_prices(posted_all, token_index)
    wu = load_wu_targets()
    paper_forecasts = load_paper_forecasts()
    forecast_cache = ForecastCache(use_live=not args.no_forecasts, paper=paper_forecasts)
    book_cache: dict[str, tuple[float | None, float | None]] = {}

    modal_by_city = build_modal_buckets(today, token_index, book_cache)

    cash = client.get_balance()
    holding_tokens: set[str] = set()
    for record in posted_all.values():
        tok = str(record.get("token_id", ""))
        if not tok:
            continue
        meta = token_index.get(tok)
        event_date = infer_event_date(record, meta, today)
        if event_date >= today:
            holding_tokens.add(tok)
    for order in open_orders:
        tok = _token_id(order)
        if tok:
            holding_tokens.add(tok)
    holdings = collect_holdings(
        client, holding_tokens, token_index, book_cache, token_entry_prices
    )
    print_portfolio(holdings, cash)

    settled, open_rows = collect_trades(
        client,
        posted_all,
        open_orders,
        token_index,
        entry_books,
        wu,
        forecast_cache,
        book_cache,
        modal_by_city,
    )
    print_settled_trades(settled)
    print_open_positions(open_rows)
    open_cost = sum(row.cost_usd for row in holdings)
    open_mark = sum(row.mark_value for row in holdings)
    print_pnl_summary(
        settled,
        cash=cash,
        open_cost=open_cost,
        open_mark=open_mark,
        starting_bankroll=args.bankroll,
    )
    print()


if __name__ == "__main__":
    main()
