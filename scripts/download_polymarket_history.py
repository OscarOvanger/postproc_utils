#!/usr/bin/env python3
"""Download historical Polymarket Tmax orderbook data from the Resolved Markets API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from polymarket_recon import (  # noqa: E402
    EVENT_TITLE_RE,
    STATION_REGISTRY,
    TMAX_QUESTION_RE,
    TMAX_TITLE_RE,
    city_to_slug,
    parse_bucket,
    parse_city,
    parse_event_date,
)

API_BASE = "https://api.resolvedmarkets.com"
CLOB_API = "https://clob.polymarket.com"
API_KEY_PATH = PROJECT_ROOT / "config" / "resolved_markets_key.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "polymarket_history"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
MARKETS_INDEX_PATH = OUTPUT_DIR / "markets_index.json"
RESOLUTIONS_PATH = OUTPUT_DIR / "resolutions.csv"
TRADING_SNAPSHOTS_PATH = OUTPUT_DIR / "trading_snapshots.parquet"

TARGET_CITIES = [
    "houston",
    "los_angeles",
    "austin",
    "dallas",
    "chicago",
    "san_francisco",
    "seattle",
    "new_york",
    "miami",
    "atlanta",
]

# Resolved Markets uses Polymarket subcategory labels for daily temperature cities.
CITY_SUBCATEGORIES: dict[str, str] = {
    "houston": "Houston",
    "los_angeles": "Los Angeles",
    "austin": "Austin",
    "dallas": "Dallas",
    "chicago": "Chicago",
    "san_francisco": "San Francisco",
    "seattle": "Seattle",
    "new_york": "Nyc",
    "miami": "Miami",
    "atlanta": "Atlanta",
}

SUBCATEGORY_TO_SLUG = {v.lower(): k for k, v in CITY_SUBCATEGORIES.items()}

SLUG_NORMALIZE = {
    "chicago_midway": "chicago",
    "new_york_city": "new_york",
    "nyc": "new_york",
}

CITY_SLUG_OVERRIDES = {
    "chicago": "chicago",
    "chicago midway": "chicago",
    "new york": "new_york",
    "new york city": "new_york",
    "nyc": "new_york",
}

EXCLUDE_KEYWORDS = (
    "hurricane",
    "precipitation",
    "rainfall",
    "earthquake",
    "arctic",
    "climate",
    "tweet",
    "tornado",
    "snowfall",
    "sea ice",
    "temperature increase",
    "warming",
)

REQUEST_SLEEP_SECONDS = 0.1
RATE_LIMIT_SLEEP_SECONDS = 30.0
MAX_RETRIES = 3
SNAPSHOT_PAGE_LIMIT = 5000
DOWNSAMPLE_MINUTES = 5

TRADING_TIME_LABELS: list[tuple[str, str]] = [
    ("entry_6am_ct", "America/Chicago"),
    ("entry_10am_local", "local"),
    ("midday_12pm_local", "local"),
    ("afternoon_3pm_local", "local"),
]


def load_api_key() -> str:
    env_key = os.environ.get("RESOLVED_MARKETS_API_KEY", "").strip()
    if env_key:
        return env_key
    if API_KEY_PATH.exists():
        key = API_KEY_PATH.read_text(encoding="utf-8").strip()
        if key:
            return key
    print(
        "Missing Resolved Markets API key.\n"
        "Set RESOLVED_MARKETS_API_KEY or create config/resolved_markets_key.txt",
        file=sys.stderr,
    )
    sys.exit(1)


def normalize_city_slug(slug: str | None) -> str | None:
    if not slug:
        return None
    slug = slug.strip().lower()
    override = CITY_SLUG_OVERRIDES.get(slug.replace("_", " "))
    if override:
        return override
    slug = city_to_slug(slug.replace("_", " "))
    return SLUG_NORMALIZE.get(slug, slug)


def slug_from_subcategory(subcategory: str) -> str | None:
    return SUBCATEGORY_TO_SLUG.get(subcategory.strip().lower())


def slug_from_crypto(crypto: str) -> str | None:
    return slug_from_subcategory(crypto) or normalize_city_slug(crypto)


def parse_city_from_market(market: dict[str, Any]) -> str | None:
    crypto = str(market.get("crypto") or "")
    if crypto:
        slug = slug_from_crypto(crypto)
        if slug:
            return slug

    question = str(market.get("question") or market.get("label") or "")
    subcategory = str(market.get("subcategory") or "")
    title = question
    if subcategory and subcategory.lower() not in {"weather", "temperature", "daily-temperature"}:
        title = f"highest temperature in {subcategory} on {question}"
    slug = parse_city(title, question)
    if slug is None and subcategory:
        slug = slug_from_subcategory(subcategory) or normalize_city_slug(subcategory)
    return normalize_city_slug(slug)


def bucket_label_from_market(market: dict[str, Any]) -> str:
    question = str(market.get("question") or "")
    label = str(market.get("label") or market.get("groupItemTitle") or "")
    parsed = parse_bucket(label) or parse_bucket(question)
    if parsed and parsed["type"] == "RANGE":
        low, high = parsed["bounds"]
        return f"{low}-{high}"
    if parsed and parsed["type"] == "LESS_THAN" and parsed["bounds"][1] is not None:
        return f"<={parsed['bounds'][1]}"
    if parsed and parsed["type"] == "GREATER_THAN" and parsed["bounds"][0] is not None:
        return f">={parsed['bounds'][0]}"

    for text in (label, question):
        if not text:
            continue
        between = re.search(r"between\s*(\d+)\s*[-–]\s*(\d+)", text, re.I)
        if between:
            return f"{between.group(1)}-{between.group(2)}"
        or_below = re.search(r"(\d+)\s*°?\s*F?\s*or\s+below", text, re.I)
        if or_below:
            return f"<={or_below.group(1)}"
        or_higher = re.search(r"(\d+)\s*°?\s*F?\s*or\s+(?:above|higher)", text, re.I)
        if or_higher:
            return f">={or_higher.group(1)}"
        range_match = re.search(r"(\d+)\s*(?:°|degrees?)?\s*F?\s*(?:to|-)\s*(\d+)", text, re.I)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"

    return label[:32] or "unknown"


def is_excluded_market(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in EXCLUDE_KEYWORDS)


def is_tmax_market(market: dict[str, Any]) -> bool:
    timeframe = str(market.get("timeframe") or "").lower()
    crypto = str(market.get("crypto") or "")
    if timeframe == "daily" and slug_from_crypto(crypto) in TARGET_CITIES:
        return True

    question = str(market.get("question") or market.get("label") or "")
    if not question:
        return False
    if is_excluded_market(question):
        return False
    if TMAX_TITLE_RE.search(question) or EVENT_TITLE_RE.search(question):
        return True
    if TMAX_QUESTION_RE.search(question):
        return True
    if re.search(r"(?i)will the (high|highest) temperature", question):
        return True
    if re.search(r"(?i)temperature in .+ be ", question):
        return True
    return False


def parse_market_date(market: dict[str, Any]) -> str | None:
    end_date = market.get("end_date") or market.get("endDate") or market.get("end_date_iso")
    if end_date:
        try:
            return pd.to_datetime(end_date, utc=True).date().isoformat()
        except (TypeError, ValueError):
            pass

    question = str(market.get("question") or market.get("label") or "")
    year_hint = end_date or market.get("first_seen") or market.get("last_seen")
    parsed = parse_event_date(question, str(year_hint) if year_hint else None)
    if parsed:
        return parsed
    match = EVENT_TITLE_RE.search(question)
    if match:
        return parse_event_date(match.group(2), str(year_hint) if year_hint else None)

    for field in ("last_seen", "first_seen"):
        value = market.get(field)
        if value:
            try:
                return pd.to_datetime(value, utc=True).date().isoformat()
            except (TypeError, ValueError):
                continue
    return None


def _first_present(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def normalize_market(raw: dict[str, Any]) -> dict[str, Any]:
    token_ids = _first_present(raw, "tokenIds", "token_ids") or []
    if isinstance(token_ids, str):
        try:
            token_ids = json.loads(token_ids)
        except json.JSONDecodeError:
            token_ids = [token_ids]
    outcomes = _first_present(raw, "outcomes") or []
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = [outcomes]

    yes_index = 0
    if outcomes and str(outcomes[0]).lower() not in {"yes", "up"}:
        yes_index = 1 if len(token_ids) > 1 else 0

    condition_id = _first_present(raw, "conditionId", "condition_id", "market_id", "id")
    snapshot_count = _first_present(raw, "snapshot_count", "snapshots") or 0
    try:
        snapshot_count = int(snapshot_count)
    except (TypeError, ValueError):
        snapshot_count = 0

    return {
        "condition_id": str(condition_id) if condition_id else None,
        "slug": str(_first_present(raw, "slug", "market_slug") or ""),
        "question": str(_first_present(raw, "question", "label") or ""),
        "token_ids": [str(t) for t in token_ids],
        "token_id_yes": str(token_ids[yes_index]) if token_ids else None,
        "outcomes": [str(o) for o in outcomes],
        "end_date": _first_present(raw, "end_date", "endDate", "end_date_iso"),
        "is_live": bool(_first_present(raw, "is_live", "isLive", "active")),
        "snapshot_count": snapshot_count,
        "subcategory": str(_first_present(raw, "subcategory", "crypto") or ""),
        "category": str(_first_present(raw, "category") or "weather"),
        "timeframe": str(_first_present(raw, "timeframe") or ""),
        "crypto": str(_first_present(raw, "crypto") or ""),
        "first_seen": _first_present(raw, "first_seen"),
        "last_seen": _first_present(raw, "last_seen"),
        "resolved": not bool(_first_present(raw, "is_live", "isLive", "active")),
    }


_clob_cache: dict[str, dict[str, Any]] = {}
CLOB_CACHE_PATH = OUTPUT_DIR / "clob_cache.json"
CLOB_WORKERS = 8


def load_clob_cache() -> None:
    global _clob_cache
    if CLOB_CACHE_PATH.exists():
        try:
            _clob_cache = json.loads(CLOB_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _clob_cache = {}


def save_clob_cache() -> None:
    if not _clob_cache:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLOB_CACHE_PATH.write_text(json.dumps(_clob_cache, indent=2) + "\n", encoding="utf-8")


def enrich_market_from_clob(market: dict[str, Any]) -> dict[str, Any]:
    """Fill question/token metadata from Polymarket CLOB when RM history rows omit it."""
    if market.get("question"):
        return market
    condition_id = market.get("condition_id") or market.get("market_id")
    if not condition_id:
        return market
    if condition_id in _clob_cache:
        return {**market, **_clob_cache[condition_id]}

    try:
        response = requests.get(f"{CLOB_API}/markets/{condition_id}", timeout=30)
        if not response.ok:
            return market
        payload = response.json()
    except requests.RequestException:
        return market

    tokens = payload.get("tokens") or []
    token_ids = [str(t.get("token_id")) for t in tokens if t.get("token_id")]
    yes_index = 0
    for idx, token in enumerate(tokens):
        if str(token.get("outcome", "")).lower() in {"yes", "up"}:
            yes_index = idx
            break

    enriched = {
        "question": str(payload.get("question") or ""),
        "slug": str(payload.get("market_slug") or market.get("slug") or ""),
        "end_date": payload.get("end_date_iso") or market.get("end_date"),
        "token_ids": token_ids,
        "token_id_yes": token_ids[yes_index] if token_ids else market.get("token_id_yes"),
        "label": str(payload.get("groupItemTitle") or market.get("label") or ""),
    }
    _clob_cache[condition_id] = enriched
    return {**market, **enriched}


def enrich_markets_parallel(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not markets:
        return []
    load_clob_cache()
    with ThreadPoolExecutor(max_workers=CLOB_WORKERS) as pool:
        enriched = list(pool.map(enrich_market_from_clob, [dict(m) for m in markets]))
    save_clob_cache()
    return enriched


class ResolvedMarketsClient:
    def __init__(self, api_key: str) -> None:
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "MCP_Project/download_polymarket_history (research)",
                "X-API-Key": api_key,
                "Authorization": f"Bearer {api_key}",
            }
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        allow_404: bool = False,
    ) -> dict[str, Any] | list[Any] | None:
        url = f"{API_BASE}{path}"
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=60)
                if response.status_code == 404 and allow_404:
                    return None
                if response.status_code == 429:
                    print(
                        f"Rate limited (429) on {url}; sleeping {RATE_LIMIT_SLEEP_SECONDS:.0f}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})",
                        flush=True,
                    )
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                    continue
                if response.status_code == 403:
                    print(
                        f"403 Forbidden: {url}\n  Body: {response.text[:500]}",
                        file=sys.stderr,
                    )
                    response.raise_for_status()
                response.raise_for_status()
                remaining = response.headers.get("X-RateLimit-Remaining")
                if remaining is not None and int(remaining) < 10:
                    print(f"  Rate limit remaining: {remaining}", flush=True)
                time.sleep(REQUEST_SLEEP_SECONDS)
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                body = ""
                if exc.response is not None:
                    body = exc.response.text[:500]
                print(
                    f"Request failed: {url} params={params}\n  {exc}\n  Body: {body}",
                    file=sys.stderr,
                )
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS if "429" in str(exc) else 2.0)
        if last_error:
            raise last_error
        return None

    def fetch_live_weather(self) -> list[dict[str, Any]]:
        payload = self.get("/v1/markets/live", {"category": "weather"})
        return _extract_market_list(payload)

    def fetch_history_weather(self) -> list[dict[str, Any]]:
        payload = self.get("/v1/markets/history", {"category": "weather"})
        markets = _extract_market_list(payload)
        if markets:
            return markets
        return self._fetch_history_recent_paginated()

    def _fetch_history_recent_paginated(self) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        offset = 0
        limit = 500
        while True:
            payload = self.get(
                "/v1/markets/history/recent",
                {"category": "weather", "limit": limit, "offset": offset},
            )
            batch = _extract_market_list(payload)
            if not batch:
                break
            markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return markets

    def fetch_snapshots(
        self,
        condition_id: str,
        *,
        limit: int = SNAPSHOT_PAGE_LIMIT,
        side: str = "UP",
        order: str = "ASC",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self.get(
                f"/v1/markets/{condition_id}/snapshots",
                {
                    "limit": limit,
                    "offset": offset,
                    "side": side,
                    "order": order,
                },
                allow_404=True,
            )
            if payload is None:
                break
            if not isinstance(payload, dict):
                print(f"Unexpected snapshot payload type for {condition_id}: {type(payload)}")
                break
            batch = payload.get("data") or []
            if not isinstance(batch, list):
                print(f"Unexpected snapshot data for {condition_id}: {payload.keys()}")
                break
            rows.extend(batch)
            total_raw = payload.get("total", len(rows))
            try:
                total = int(total_raw)
            except (TypeError, ValueError):
                total = len(rows)
            if len(batch) < limit or offset + len(batch) >= total:
                break
            offset += len(batch)
        return rows


def _extract_market_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("markets", "data", "results"):
            if key in payload and isinstance(payload[key], list):
                return [row for row in payload[key] if isinstance(row, dict)]
        if payload.get("conditionId") or payload.get("condition_id") or payload.get("market_id"):
            return [payload]
    return []


def discover_markets(
    client: ResolvedMarketsClient,
    *,
    cities: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    city_slugs = set(cities or TARGET_CITIES)
    live = client.fetch_live_weather()
    history_weather = client.fetch_history_weather()

    merged: dict[str, dict[str, Any]] = {}
    raw_all = live + history_weather
    for raw in raw_all:
        norm = normalize_market(raw)
        cid = norm.get("condition_id")
        if not cid:
            continue
        if norm.get("timeframe") == "daily":
            city = slug_from_crypto(norm.get("crypto") or norm.get("subcategory") or "")
            if city not in city_slugs:
                continue
        record = {**norm, **raw}
        if cid not in merged:
            merged[cid] = record
        else:
            merged[cid].update({k: v for k, v in record.items() if v not in (None, "", 0)})
    return list(merged.values()), raw_all


def build_events(
    markets: list[dict[str, Any]],
    *,
    cities: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cities_filter = set(cities or TARGET_CITIES)
    candidates = [raw for raw in markets if is_tmax_market(raw)]
    enriched_markets = enrich_markets_parallel(candidates)

    tmax_markets: list[dict[str, Any]] = []
    for record in enriched_markets:
        city = parse_city_from_market(record)
        event_date = parse_market_date(record)
        if city is None or event_date is None:
            continue
        if city not in cities_filter:
            continue
        if start_date and event_date < start_date.isoformat():
            continue
        if end_date and event_date > end_date.isoformat():
            continue
        norm = normalize_market(record)
        norm["city"] = city
        norm["date"] = event_date
        norm["bucket"] = bucket_label_from_market(record)
        tmax_markets.append(norm)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for market in tmax_markets:
        key = (market["city"], market["date"])
        grouped.setdefault(key, []).append(market)

    events: list[dict[str, Any]] = []
    for (city, event_date), buckets in sorted(grouped.items()):
        events.append(
            {
                "city": city,
                "date": event_date,
                "buckets": sorted(buckets, key=lambda b: b.get("bucket", "")),
            }
        )

    dates = [m["date"] for m in tmax_markets if m.get("date")]
    resolved = sum(1 for m in tmax_markets if m.get("resolved"))
    active = sum(1 for m in tmax_markets if m.get("is_live"))
    summary = {
        "total_weather": len(markets),
        "tmax_markets": len(tmax_markets),
        "events": len(events),
        "cities": sorted({e["city"] for e in events}),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "resolved": resolved,
        "active": active,
    }
    return events, summary


def save_markets_index(
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    raw_markets: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_cities": TARGET_CITIES,
        "summary": summary,
        "events": events,
        "raw_markets": raw_markets[:500],
    }
    MARKETS_INDEX_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_discovery_summary(summary: dict[str, Any]) -> None:
    print(f"Found {summary['total_weather']} total weather markets")
    print(f"  Tmax markets: {summary['tmax_markets']}")
    print(f"  Cities: {summary['cities']}")
    print(f"  Date range: {summary['date_min']} to {summary['date_max']}")
    print(f"  Resolved: {summary['resolved']}, Active: {summary['active']}")


def snapshot_rows_to_df(rows: list[dict[str, Any]], bucket: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "bucket", "best_bid", "best_ask", "midpoint", "bid_depth", "ask_depth"]
        )
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["best_bid"] = pd.to_numeric(frame.get("best_bid"), errors="coerce")
    frame["best_ask"] = pd.to_numeric(frame.get("best_ask"), errors="coerce")
    frame["midpoint"] = pd.to_numeric(
        frame.get("mid_price", frame.get("midpoint")), errors="coerce"
    )
    frame["bid_depth"] = pd.to_numeric(frame.get("bid_depth_total"), errors="coerce")
    frame["ask_depth"] = pd.to_numeric(frame.get("ask_depth_total"), errors="coerce")
    frame["bucket"] = bucket
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.sort_values("timestamp")
    frame = frame.set_index("timestamp")
    frame = frame.resample(f"{DOWNSAMPLE_MINUTES}min").last().dropna(how="all")
    frame = frame.reset_index()
    return frame[
        ["timestamp", "bucket", "best_bid", "best_ask", "midpoint", "bid_depth", "ask_depth"]
    ]


def download_event_snapshots(
    client: ResolvedMarketsClient,
    event: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[pd.DataFrame | None, int]:
    city = event["city"]
    event_date = event["date"]
    out_path = SNAPSHOTS_DIR / city / f"{event_date}.parquet"
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        return existing, len(existing)

    frames: list[pd.DataFrame] = []
    total_raw = 0
    for bucket_info in event.get("buckets", []):
        condition_id = bucket_info.get("condition_id")
        bucket = bucket_info.get("bucket", "unknown")
        if not condition_id:
            continue
        if bucket_info.get("snapshot_count", 0) == 0:
            continue
        rows = client.fetch_snapshots(condition_id)
        if not rows:
            continue
        total_raw += len(rows)
        frames.append(snapshot_rows_to_df(rows, bucket))

    if not frames:
        return None, 0

    combined = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "bucket"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return combined, total_raw


def infer_winning_bucket(event: dict[str, Any], frame: pd.DataFrame | None) -> tuple[str | None, str | None]:
    if frame is None or frame.empty:
        return None, None
    last_ts = frame["timestamp"].max()
    last_rows = frame[frame["timestamp"] == last_ts]
    winners = last_rows[
        (last_rows["best_bid"] >= 0.95)
        | (last_rows["midpoint"] >= 0.95)
        | (last_rows["best_ask"] >= 0.95)
    ]
    if winners.empty:
        winners = last_rows.sort_values("midpoint", ascending=False).head(1)
    if winners.empty:
        return None, None
    bucket = str(winners.iloc[0]["bucket"])
    condition_id = None
    for bucket_info in event.get("buckets", []):
        if bucket_info.get("bucket") == bucket:
            condition_id = bucket_info.get("condition_id")
            break
    return bucket, condition_id


def build_resolutions(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str | None]] = []
    for event in events:
        path = SNAPSHOTS_DIR / event["city"] / f"{event['date']}.parquet"
        frame = pd.read_parquet(path) if path.exists() else None
        winning_bucket, condition_id = infer_winning_bucket(event, frame)
        rows.append(
            {
                "city": event["city"],
                "date": event["date"],
                "winning_bucket": winning_bucket,
                "condition_id": condition_id,
            }
        )
    return pd.DataFrame(rows)


def city_timezone(city: str) -> str:
    for key in (city, SLUG_NORMALIZE.get(city, city), f"{city}_city"):
        if key in STATION_REGISTRY:
            return str(STATION_REGISTRY[key]["timezone"])
    if (PROJECT_ROOT / "config" / "city_config.json").exists():
        cfg = json.loads((PROJECT_ROOT / "config" / "city_config.json").read_text())
        if city in cfg and "timezone" in cfg[city]:
            return str(cfg[city]["timezone"])
    return "America/Chicago"


def target_timestamp(event_date: str, hour: int, tz_name: str) -> pd.Timestamp:
    local_date = datetime.fromisoformat(event_date).date()
    tz = ZoneInfo(tz_name)
    local_dt = datetime(local_date.year, local_date.month, local_date.day, hour, 0, tzinfo=tz)
    return pd.Timestamp(local_dt).tz_convert("UTC")


def extract_trading_snapshots(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        path = SNAPSHOTS_DIR / event["city"] / f"{event['date']}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        local_tz = city_timezone(event["city"])

        targets: list[tuple[str, pd.Timestamp | None]] = []
        for label, tz_kind in TRADING_TIME_LABELS:
            if tz_kind == "local":
                hour = 10 if "10am" in label else 12 if "12pm" in label else 15
                targets.append((label, target_timestamp(event["date"], hour, local_tz)))
            else:
                targets.append((label, target_timestamp(event["date"], 6, tz_kind)))

        last_ts = frame["timestamp"].max()
        targets.append(("pre_settlement", last_ts))

        for label, target_ts in targets:
            if target_ts is None:
                continue
            eligible = frame[frame["timestamp"] <= target_ts]
            if eligible.empty:
                continue
            snap_ts = eligible["timestamp"].max()
            snap_rows = eligible[eligible["timestamp"] == snap_ts]
            for _, row in snap_rows.iterrows():
                rows.append(
                    {
                        "city": event["city"],
                        "date": event["date"],
                        "snapshot_time": snap_ts.isoformat(),
                        "snapshot_label": label,
                        "bucket": row["bucket"],
                        "best_bid": row["best_bid"],
                        "best_ask": row["best_ask"],
                        "midpoint": row["midpoint"],
                    }
                )
    return pd.DataFrame(rows)


def dir_size_mb(path: Path) -> float:
    total = 0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024 * 1024)


def run_explore(client: ResolvedMarketsClient) -> None:
    print("=== Resolved Markets API Explore ===\n")
    health = client.get("/health", allow_404=True)
    if health:
        print(f"Health: {health.get('status', health)}")
    live = client.fetch_live_weather()
    history = client.fetch_history_weather()
    print(f"Weather markets: live={len(live)}, history={len(history)}")
    markets, _raw = discover_markets(client)
    events, summary = build_events(markets)
    print_discovery_summary(summary)

    print("\nSample Tmax markets:")
    shown = 0
    for market in markets:
        if not is_tmax_market(market):
            continue
        record = enrich_market_from_clob(dict(market))
        city = parse_city_from_market(record)
        event_date = parse_market_date(record)
        bucket = bucket_label_from_market(record)
        print(f"  - city={city} date={event_date} bucket={bucket}")
        print(f"    Q: {record.get('question') or record.get('label')}")
        shown += 1
        if shown >= 3:
            break

    sample_id = None
    for event in events:
        for bucket in event.get("buckets", []):
            if bucket.get("snapshot_count", 0) > 0 and bucket.get("condition_id"):
                sample_id = bucket["condition_id"]
                break
        if sample_id:
            break
    if sample_id is None:
        for market in markets:
            norm = normalize_market(market)
            if norm.get("snapshot_count", 0) > 0 and norm.get("condition_id"):
                sample_id = norm["condition_id"]
                break

    if sample_id:
        payload = client.get(
            f"/v1/markets/{sample_id}/snapshots",
            {"limit": 5, "offset": 0, "side": "UP", "order": "ASC"},
        )
        print(f"\nRaw snapshot response for {sample_id}:")
        print(json.dumps(payload, indent=2)[:4000])
        if isinstance(payload, dict) and payload.get("data"):
            row = payload["data"][0]
            print("\nInferred column mapping:")
            mapping = {
                "timestamp": row.get("timestamp"),
                "best_bid": row.get("best_bid"),
                "best_ask": row.get("best_ask"),
                "midpoint": row.get("mid_price", row.get("midpoint")),
                "bid_depth": row.get("bid_depth_total"),
                "ask_depth": row.get("ask_depth_total"),
            }
            for key, value in mapping.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo market with snapshots found for sample fetch.")


def run_download(
    client: ResolvedMarketsClient,
    *,
    cities: list[str] | None,
    start_date: date | None,
    end_date: date | None,
    force: bool = False,
    build_trading_only: bool = False,
) -> None:
    if build_trading_only:
        if MARKETS_INDEX_PATH.exists():
            index = json.loads(MARKETS_INDEX_PATH.read_text(encoding="utf-8"))
            events = index.get("events", [])
        else:
            markets, _raw = discover_markets(client, cities=cities)
            events, _summary = build_events(
                markets, cities=cities, start_date=start_date, end_date=end_date
            )
    else:
        markets, raw_markets = discover_markets(client, cities=cities)
        events, summary = build_events(
            markets, cities=cities, start_date=start_date, end_date=end_date
        )
        save_markets_index(events, summary, raw_markets)
        print_discovery_summary(summary)

    if not events:
        print("No Tmax events matched filters.")
        return

    total_snapshots = 0
    if not build_trading_only:
        total = len(events)
        downloaded = 0
        skipped = 0
        total_snapshots = 0
        for idx, event in enumerate(events, start=1):
            out_path = SNAPSHOTS_DIR / event["city"] / f"{event['date']}.parquet"
            if out_path.exists() and not force:
                print(f"Skipping {event['city']} {event['date']} (cached) ({idx}/{total})")
                skipped += 1
                frame = pd.read_parquet(out_path)
                total_snapshots += len(frame)
                continue
            print(f"Downloading {event['city']} {event['date']}... ({idx}/{total})")
            frame, raw_count = download_event_snapshots(client, event, force=force)
            if frame is None:
                print(f"  No snapshots for {event['city']} {event['date']}")
                continue
            downloaded += 1
            total_snapshots += len(frame)
            print(f"  Saved {len(frame)} rows ({raw_count} raw ticks)")

    resolutions = build_resolutions(events)
    resolutions.to_csv(RESOLUTIONS_PATH, index=False)
    trading = extract_trading_snapshots(events)
    trading.to_parquet(TRADING_SNAPSHOTS_PATH, index=False)

    resolved_count = int(resolutions["winning_bucket"].notna().sum())
    storage_mb = dir_size_mb(OUTPUT_DIR)
    dates = [e["date"] for e in events]
    snapshot_rows = sum(
        len(pd.read_parquet(p))
        for p in SNAPSHOTS_DIR.rglob("*.parquet")
        if p.is_file()
    ) if SNAPSHOTS_DIR.exists() else total_snapshots
    print("\n=== Download Complete ===")
    print(f"Markets downloaded: {len(events)} events")
    print(f"Total snapshots: {snapshot_rows}")
    print(f"Date range: {min(dates)} to {max(dates)}")
    print(f"Cities covered: {sorted({e['city'] for e in events})}")
    print(f"Resolved markets: {resolved_count}")
    print(f"Storage used: {storage_mb:.1f} MB")
    print(f"Trading snapshots saved to: {TRADING_SNAPSHOTS_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Polymarket Tmax history from Resolved Markets API."
    )
    parser.add_argument("--explore", action="store_true", help="Explore API; no bulk download")
    parser.add_argument("--force", action="store_true", help="Re-download cached parquet files")
    parser.add_argument("--build-trading-only", action="store_true", help="Rebuild trading/resolution outputs")
    parser.add_argument("--city", nargs="+", default=None, help="Limit to city slug(s)")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key()
    client = ResolvedMarketsClient(api_key)
    cities = args.city
    if cities:
        unknown = [c for c in cities if normalize_city_slug(c) not in TARGET_CITIES]
        if unknown:
            print(f"Unknown cities (expected one of {TARGET_CITIES}): {unknown}", file=sys.stderr)
            sys.exit(1)
        cities = [normalize_city_slug(c) or c for c in cities]

    if args.explore:
        run_explore(client)
        return

    run_download(
        client,
        cities=cities,
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        build_trading_only=args.build_trading_only,
    )


if __name__ == "__main__":
    main()
