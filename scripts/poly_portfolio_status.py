#!/usr/bin/env python3
"""Extended Polymarket portfolio report: balance, trades, open orders, forecasts, PnL.

Usage:
  .venv/bin/python scripts/poly_portfolio_status.py
  .venv/bin/python scripts/poly_portfolio_status.py --days 7 --no-forecasts
  .venv/bin/python scripts/poly_portfolio_status.py --all-orders   # full history (slow)

Bid/ask at entry: from auto_trader state when available; otherwise N/A.
Settled PnL uses Wunderground actuals from wunderground_targets.parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    _parse_log_timestamp,
    _to_float,
    _token_id,
    fetch_gamma_token_labels,
    fetch_open_orders,
    load_posted_orders,
    load_token_labels,
)
from src.polymarket_api import (  # noqa: E402
    DEFAULT_MARKETS_PATH,
    ORDER_LOG_PATH,
    PolymarketClient,
    fetch_order_book_http,
    load_markets_map,
    parse_bucket_label,
)
from src.poly_trading_pipeline import poly_taker_fee  # noqa: E402

WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
AUTO_STATE_GLOB = "auto_trader_state_*.json"
PAPER_LOG = PROJECT_ROOT / "logs" / "poly_paper_trades.jsonl"

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


def enrich_token_index_from_scan(index: dict[str, TokenMeta], event_date: str) -> None:
    """Fill missing labels using scan_modal_buckets discovery (fast, weather cities)."""
    try:
        from scan_modal_buckets import TARGET_CITIES, discover_markets
    except ImportError:
        return

    discovered = discover_markets(event_date)
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
    for path in sorted((PROJECT_ROOT / "logs").glob(AUTO_STATE_GLOB)):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for pos in state.get("positions", []):
            oid = pos.get("order_id")
            if not oid:
                continue
            out[str(oid)] = {
                "best_bid_at_entry": _to_float(pos.get("best_bid_at_entry")),
                "best_ask_at_entry": _to_float(pos.get("best_ask_at_entry")),
                "maker_entry_price": _to_float(pos.get("maker_entry_price")),
            }
    return out


def load_paper_forecasts() -> dict[tuple[str, str], dict[str, int]]:
    """(event_date, trackb_city) -> {raw, adjusted} from poly_paper_trades.jsonl."""
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
            adj = row.get("forecasts") or {}
            for city, val in adj.items():
                key = (event_date, str(city))
                out[key] = {
                    "trackb": int(val),
                    "trackb_raw": int(raw.get(city, val)),
                }
    return out


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
    row = wu[(wu["city"] == city) & (wu["date"] == event_date)]
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
        from run_daily_trade import fetch_forecast, load_city_config, load_deploy_config

        deploy = load_deploy_config()
        city_config = load_city_config()
        if trackb_city not in deploy.get("cities", []):
            return None
        mini = {**deploy, "cities": [trackb_city]}
        forecasts, _, _ = fetch_forecast(mini, event_date, city_config)
        return forecasts.get(trackb_city)
    except Exception:
        return None


def fetch_ngboost_forecast(city_slug: str, event_date: str) -> float | None:
    if city_slug not in NGBOOST_CITIES:
        return None
    try:
        import joblib

        import train_ngboost as ng
        from datetime import date as date_cls, timedelta

        output_dir = PROJECT_ROOT / "models" / "ngboost"
        model = joblib.load(output_dir / "ngboost_global.pkl")
        scaler = joblib.load(output_dir / "feature_scaler.pkl")
        lgb_model = joblib.load(output_dir / "lgb_stage1.pkl")
        with open(output_dir / "model_config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        feature_cols = config["feature_columns"]
        stage1_cols = [c for c in feature_cols if c != "lgb_tmax_pred"]
        fill_medians = config.get("nan_fill_medians", {})
        sigma_k = float(config.get("sigma_calibration_k", 1.0))

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
        asos = ng.load_temp_early_morning(city_slug, target_dt, target_dt)
        om = ng.load_openmeteo_tmax(city_slug, target_dt, target_dt)

        feat = {
            "hrrr_tmax": float(hrrr_row["hrrr_tmax"]),
            "peak_cloud_cover": float(hrrr_row["peak_cloud_cover"]),
            "peak_solar_flux": float(hrrr_row["peak_solar_flux"]),
            "snow_depth": float(hrrr_row["snow_depth"]),
            "temp_early_morning": float(asos["temp_early_morning"].iloc[0]) if len(asos) else np.nan,
            "nwp_tmax_openmeteo": float(om["nwp_tmax_openmeteo"].iloc[0]) if len(om) else np.nan,
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
        df = ng.apply_saved_median_fill(df, fill_medians, list(fill_medians.keys()))
        df["lgb_tmax_pred"] = float(lgb_model.predict(df[stage1_cols])[0])
        X = ng.transform_features(scaler, df, feature_cols)
        mu, _, _ = ng.predict_dist_params(model, X)
        return round(float(mu[0]), 1)
    except Exception:
        return None


class ForecastCache:
    def __init__(self, *, use_live: bool, paper: dict[tuple[str, str], dict[str, int]]):
        self.use_live = use_live
        self.paper = paper
        self.trackb: dict[tuple[str, str], int | None] = {}
        self.ngboost: dict[tuple[str, str], float | None] = {}

    def trackb_for(self, city: str, event_date: str) -> int | None:
        key = (event_date, TRACKB_CITY_MAP.get(city) or city)
        if key in self.trackb:
            return self.trackb[key]
        paper_key = (event_date, str(key[1]))
        if paper_key in self.paper:
            val = self.paper[paper_key].get("trackb")
            self.trackb[key] = val
            return val
        val = fetch_trackb_forecast(city, event_date) if self.use_live else None
        self.trackb[key] = val
        return val

    def ngboost_for(self, city: str, event_date: str) -> float | None:
        key = (city, event_date)
        if key in self.ngboost:
            return self.ngboost[key]
        val = fetch_ngboost_forecast(city, event_date) if self.use_live else None
        self.ngboost[key] = val
        return val


def fetch_book_cached(token: str, cache: dict[str, tuple[float | None, float | None]]) -> tuple[float | None, float | None]:
    if not token:
        return None, None
    if token not in cache:
        cache[token] = fetch_order_book_http(token)
        time.sleep(0.05)
    return cache[token]


def filter_posted_orders(
    posted: dict[str, dict[str, Any]],
    *,
    open_ids: set[str],
    days: int | None,
    all_orders: bool,
) -> dict[str, dict[str, Any]]:
    if all_orders:
        return posted
    cutoff = datetime.now().timestamp() - (days or 14) * 86400
    filtered: dict[str, dict[str, Any]] = {}
    for order_id, record in posted.items():
        if order_id in open_ids:
            filtered[order_id] = record
            continue
        if _parse_log_timestamp(record.get("timestamp")) >= cutoff:
            filtered[order_id] = record
    return filtered


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

    # bucket labels per city/date for winner lookup
    buckets_by_city_date: dict[tuple[str, str], list[str]] = {}
    for meta in token_index.values():
        key = (meta.city, meta.event_date)
        buckets_by_city_date.setdefault(key, [])
        if meta.bucket_label not in buckets_by_city_date[key]:
            buckets_by_city_date[key].append(meta.bucket_label)

    for order_id, record in sorted(posted.items(), key=lambda x: x[1].get("timestamp", "")):
        token = str(record.get("token_id", ""))
        meta = token_index.get(token)
        city = meta.city if meta else "unknown"
        bucket = meta.bucket_label if meta else "?"
        event_date = meta.event_date if meta else today
        city_display = meta.city_display if meta else city

        is_taker = not bool(record.get("post_only", True))
        original = _to_float(record.get("size")) or 0.0
        entry_price = _to_float(record.get("price"))
        matched = 0.0
        state = "unknown"

        book = entry_books.get(order_id, {})
        bid_at = book.get("best_bid_at_entry")
        ask_at = book.get("best_ask_at_entry")

        if order_id in open_ids:
            oo = open_by_id.get(order_id, {})
            state = "open"
            matched = _to_float(oo.get("size_matched")) or 0.0
            original = _to_float(oo.get("original_size") or oo.get("size")) or original
            entry_price = _to_float(oo.get("price")) or entry_price
        else:
            status_info = client.get_order_status(order_id, token_id=token or None)
            state = str(status_info.get("status", "unknown"))
            matched = _to_float(status_info.get("size_matched")) or 0.0
            original = _to_float(status_info.get("original_size")) or original
            entry_price = _to_float(status_info.get("fill_price")) or entry_price

        if state == "open" and order_id in open_ids and matched <= 0:
            best_bid, best_ask = fetch_book_cached(token, book_cache)
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

        if winner is not None and actual is not None and entry_price is not None:
            trade.won = temp_in_bucket(actual, bucket)
            trade.pnl_usd = settlement_pnl(
                n_contracts=matched,
                entry_price=entry_price,
                won=trade.won,
                is_taker=is_taker,
            )
            settled.append(trade)
        elif event_date >= today:
            best_bid, best_ask = fetch_book_cached(token, book_cache)
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
                )
            )
        else:
            settled.append(trade)

    # held shares without a posted order record
    seen_tokens = {str(r.get("token_id", "")) for r in posted.values()}
    for token, meta in token_index.items():
        if token in seen_tokens:
            continue
        try:
            shares = client.get_conditional_balance(token)
        except Exception:
            shares = 0.0
        if shares <= 0:
            continue
        best_bid, best_ask = fetch_book_cached(token, book_cache)
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
            )
        )

    return settled, open_rows


def print_portfolio(
    client: PolymarketClient,
    holdings: list[tuple[str, float, float | None]],
    cash: float,
) -> tuple[float, float]:
    mark_value = sum(shares * (mark or 0.0) for _, shares, mark in holdings)

    print("\n=== Portfolio (pUSD) ===")
    print(f"Cash (pUSD):           ${cash:.2f}")
    print(f"Positions (mark bid):  ${mark_value:.2f}")
    print(f"Total portfolio:       ${cash + mark_value:.2f}")
    if holdings:
        print("\nHeld YES shares:")
        for label, shares, mark in holdings:
            print(f"  {label}: {shares:.1f} sh @ {fmt_price(mark)} = {fmt_price((mark or 0) * shares)}")
    else:
        print("\nHeld YES shares: none")
    return cash, mark_value


def collect_holdings(
    client: PolymarketClient,
    tokens: set[str],
    token_index: dict[str, TokenMeta],
    book_cache: dict[str, tuple[float | None, float | None]],
) -> list[tuple[str, float, float | None]]:
    today = str(date.today())
    holdings: list[tuple[str, float, float | None]] = []
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
        best_bid, best_ask = fetch_book_cached(token, book_cache)
        if best_bid is None and best_ask is None:
            continue
        mark = best_bid if best_bid is not None else best_ask
        if meta:
            label = f"{meta.city_display} {meta.bucket_label} ({meta.event_date})"
        else:
            label = f"token {token[:12]}..."
        holdings.append((label, shares, mark))
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
        f"{'Type':<8} {'Placed':<20} {'City':<14} {'Bucket':<10} {'Qty':>4} "
        f"{'Our bid':>8} {'Bid now':>8} {'Ask now':>8} {'Mid':>7} "
        f"{'TrackB':>7} {'NGB μ':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        placed = (row.placed_at or "")[:19] or "—"
        print(
            f"{row.kind:<8} {placed:<20} {row.city:<14} {row.bucket_label:<10} "
            f"{row.n_contracts:4.0f} {fmt_price(row.entry_price):>8} "
            f"{fmt_price(row.best_bid):>8} {fmt_price(row.best_ask):>8} "
            f"{fmt_price(row.midpoint):>7} "
            f"{str(row.trackb_f) if row.trackb_f is not None else '—':>7} "
            f"{f'{row.ngboost_mu:.1f}' if row.ngboost_mu is not None else '—':>7}"
        )


def print_pnl_summary(trades: list[TradeRecord]) -> None:
    resolved = [t for t in trades if t.pnl_usd is not None]
    if not resolved:
        print("\n=== PnL summary ===")
        print("  No settled trades with PnL yet.")
        return

    pnls = [float(t.pnl_usd) for t in resolved if t.pnl_usd is not None]
    capitals = [t.n_contracts * t.entry_price for t in resolved]
    wins = sum(1 for t in resolved if t.won)
    total = sum(pnls)
    sharpe = compute_sharpe(pnls, capitals)

    print("\n=== PnL summary ===")
    print(f"Settled trades:  {len(resolved)}")
    print(f"Win rate:        {wins}/{len(resolved)} ({100.0 * wins / len(resolved):.1f}%)")
    print(f"Total PnL:       {fmt_pnl(total)}")
    print(f"Avg PnL/trade:   {fmt_pnl(total / len(resolved))}")
    if sharpe is not None:
        print(f"Sharpe (per-trade return on capital): {sharpe:.3f}")
    else:
        print("Sharpe:          N/A (need 2+ settled trades)")


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
        "--days",
        type=int,
        default=14,
        help="Look back N days for order status checks (default: 14)",
    )
    parser.add_argument(
        "--all-orders",
        action="store_true",
        help="Check status for all logged orders (slow)",
    )
    parser.add_argument(
        "--no-forecasts",
        action="store_true",
        help="Skip live TrackB/NGBoost forecast fetches (use paper log only)",
    )
    args = parser.parse_args()

    client = PolymarketClient()
    posted_all = load_posted_orders(ORDER_LOG_PATH)
    open_orders = fetch_open_orders(client)
    open_ids = {_order_id(o) for o in open_orders}
    open_ids.discard(None)
    posted = filter_posted_orders(
        posted_all,
        open_ids=open_ids,
        days=args.days,
        all_orders=args.all_orders,
    )

    event_dates: set[str] = {args.date}
    for record in posted.values():
        ts = record.get("timestamp")
        if ts:
            try:
                event_dates.add(datetime.fromisoformat(str(ts)).date().isoformat())
            except ValueError:
                pass
    event_dates.add(str(date.today()))

    token_ids = {str(r.get("token_id", "")) for r in posted.values() if r.get("token_id")}
    for order in open_orders:
        tok = _token_id(order)
        if tok:
            token_ids.add(tok)

    token_index = build_token_index(
        event_dates=event_dates,
        refresh_labels=args.fetch_labels,
    )
    enrich_token_index_from_scan(token_index, str(date.today()))
    simple_labels = load_token_labels(
        event_date=args.date,
        token_ids=token_ids,
        refresh=args.fetch_labels,
    )
    for token, label in simple_labels.items():
        if token not in token_index:
            city_display, city_slug, bucket = _parse_label_parts(label)
            token_index[token] = TokenMeta(
                token_id=token,
                city=city_slug,
                city_display=city_display.title() if city_display else city_slug,
                bucket_label=bucket or label,
                event_date=args.date,
            )

    entry_books = load_auto_trader_entry_books()
    wu = load_wu_targets()
    paper_forecasts = load_paper_forecasts()
    forecast_cache = ForecastCache(use_live=not args.no_forecasts, paper=paper_forecasts)
    book_cache: dict[str, tuple[float | None, float | None]] = {}

    cash = client.get_balance()
    holding_tokens: set[str] = set()
    for record in posted.values():
        tok = str(record.get("token_id", ""))
        if tok:
            holding_tokens.add(tok)
    for order in open_orders:
        tok = _token_id(order)
        if tok:
            holding_tokens.add(tok)
    holdings = collect_holdings(client, holding_tokens, token_index, book_cache)
    print_portfolio(client, holdings, cash)

    settled, open_rows = collect_trades(
        client,
        posted,
        open_orders,
        token_index,
        entry_books,
        wu,
        forecast_cache,
        book_cache,
    )
    if args.all_orders and posted is not posted_all:
        settled_all, _ = collect_trades(
            client,
            posted_all,
            open_orders,
            token_index,
            entry_books,
            wu,
            forecast_cache,
            book_cache,
        )
        settled = settled_all
    print_settled_trades(settled)
    print_open_positions(open_rows)
    print_pnl_summary(settled)
    print()


if __name__ == "__main__":
    main()
