#!/usr/bin/env python3
"""Discover and document Polymarket daily high temperature markets."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "polymarket_recon"
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
IEM_CLI_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"

TRAIN_SLUGS = {
    "austin",
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "oklahoma_city",
    "philadelphia",
    "phoenix",
    "san_francisco",
}

CITY_SLUG_MAP = {
    "austin": "austin",
    "chicago": "chicago_midway",
    "chicago midway": "chicago_midway",
    "houston": "houston",
    "los angeles": "los_angeles",
    "new york": "new_york_city",
    "new york city": "new_york_city",
    "nyc": "new_york_city",
    "oklahoma city": "oklahoma_city",
    "philadelphia": "philadelphia",
    "phoenix": "phoenix",
    "san francisco": "san_francisco",
    "miami": "miami",
    "denver": "denver",
    "minneapolis": "minneapolis",
    "dallas": "dallas",
    "seattle": "seattle",
    "atlanta": "atlanta",
    "detroit": "detroit",
    "boston": "boston",
    "washington dc": "washington_dc",
    "washington d.c.": "washington_dc",
    "las vegas": "las_vegas",
    "nashville": "nashville",
    "portland": "portland",
    "charlotte": "charlotte",
    "san antonio": "san_antonio",
    "san diego": "san_diego",
    "indianapolis": "indianapolis",
    "columbus": "columbus",
    "jacksonville": "jacksonville",
    "memphis": "memphis",
    "sacramento": "sacramento",
    "tampa": "tampa",
    "salt lake city": "salt_lake_city",
    "kansas city": "kansas_city",
    "st louis": "st_louis",
    "st. louis": "st_louis",
    "pittsburgh": "pittsburgh",
    "raleigh": "raleigh",
    "milwaukee": "milwaukee",
    "cincinnati": "cincinnati",
    "orlando": "orlando",
    "new orleans": "new_orleans",
    "dc": "washington_dc",
    "nyc": "new_york_city",
}

STATION_REGISTRY: dict[str, dict[str, Any]] = {
    "austin": {"station": "KAUS", "lat": 30.1945, "lon": -97.6699, "timezone": "America/Chicago"},
    "chicago_midway": {"station": "KMDW", "lat": 41.7861, "lon": -87.7522, "timezone": "America/Chicago"},
    "houston": {"station": "KIAH", "lat": 29.9844, "lon": -95.3414, "timezone": "America/Chicago"},
    "los_angeles": {"station": "KLAX", "lat": 33.9382, "lon": -118.3886, "timezone": "America/Los_Angeles"},
    "new_york_city": {"station": "KLGA", "lat": 40.7769, "lon": -73.8740, "timezone": "America/New_York"},
    "oklahoma_city": {"station": "KOKC", "lat": 35.3931, "lon": -97.6007, "timezone": "America/Chicago"},
    "philadelphia": {"station": "KPHL", "lat": 39.8721, "lon": -75.2411, "timezone": "America/New_York"},
    "phoenix": {"station": "KPHX", "lat": 33.4373, "lon": -112.0078, "timezone": "America/Phoenix"},
    "san_francisco": {"station": "KSFO", "lat": 37.6197, "lon": -122.3647, "timezone": "America/Los_Angeles"},
    "miami": {"station": "KMIA", "lat": 25.7959, "lon": -80.2870, "timezone": "America/New_York"},
    "denver": {"station": "KDEN", "lat": 39.8561, "lon": -104.6737, "timezone": "America/Denver"},
    "minneapolis": {"station": "KMSP", "lat": 44.8831, "lon": -93.2289, "timezone": "America/Chicago"},
    "dallas": {"station": "KDFW", "lat": 32.8998, "lon": -97.0403, "timezone": "America/Chicago"},
    "seattle": {"station": "KSEA", "lat": 47.4502, "lon": -122.3088, "timezone": "America/Los_Angeles"},
    "atlanta": {"station": "KATL", "lat": 33.6407, "lon": -84.4277, "timezone": "America/New_York"},
    "detroit": {"station": "KDTW", "lat": 42.2162, "lon": -83.3554, "timezone": "America/New_York"},
    "boston": {"station": "KBOS", "lat": 42.3656, "lon": -71.0096, "timezone": "America/New_York"},
    "washington_dc": {"station": "KDCA", "lat": 38.8512, "lon": -77.0402, "timezone": "America/New_York"},
    "las_vegas": {"station": "KLAS", "lat": 36.0840, "lon": -115.1537, "timezone": "America/Los_Angeles"},
    "nashville": {"station": "KBNA", "lat": 36.1245, "lon": -86.6782, "timezone": "America/Chicago"},
    "portland": {"station": "KPDX", "lat": 45.5898, "lon": -122.5951, "timezone": "America/Los_Angeles"},
    "charlotte": {"station": "KCLT", "lat": 35.2144, "lon": -80.9431, "timezone": "America/New_York"},
    "san_antonio": {"station": "KSAT", "lat": 29.5337, "lon": -98.4698, "timezone": "America/Chicago"},
    "san_diego": {"station": "KSAN", "lat": 32.7336, "lon": -117.1831, "timezone": "America/Los_Angeles"},
    "indianapolis": {"station": "KIND", "lat": 39.7173, "lon": -86.2944, "timezone": "America/New_York"},
    "columbus": {"station": "KCMH", "lat": 40.0022, "lon": -82.8914, "timezone": "America/New_York"},
    "jacksonville": {"station": "KJAX", "lat": 30.4941, "lon": -81.6879, "timezone": "America/New_York"},
    "memphis": {"station": "KMEM", "lat": 35.0424, "lon": -89.9767, "timezone": "America/Chicago"},
    "sacramento": {"station": "KSMF", "lat": 38.6954, "lon": -121.5908, "timezone": "America/Los_Angeles"},
    "tampa": {"station": "KTPA", "lat": 27.9756, "lon": -82.5325, "timezone": "America/New_York"},
    "salt_lake_city": {"station": "KSLC", "lat": 40.7884, "lon": -111.9778, "timezone": "America/Denver"},
    "kansas_city": {"station": "KMCI", "lat": 39.2976, "lon": -94.7139, "timezone": "America/Chicago"},
    "st_louis": {"station": "KSTL", "lat": 38.7487, "lon": -90.3700, "timezone": "America/Chicago"},
    "pittsburgh": {"station": "KPIT", "lat": 40.4915, "lon": -80.2329, "timezone": "America/New_York"},
    "raleigh": {"station": "KRDU", "lat": 35.8776, "lon": -78.7875, "timezone": "America/New_York"},
    "milwaukee": {"station": "KMKE", "lat": 42.9472, "lon": -87.8966, "timezone": "America/Chicago"},
    "cincinnati": {"station": "KCVG", "lat": 39.0488, "lon": -84.6678, "timezone": "America/New_York"},
    "orlando": {"station": "KMCO", "lat": 28.4312, "lon": -81.3081, "timezone": "America/New_York"},
    "new_orleans": {"station": "KMSY", "lat": 29.9911, "lon": -90.2592, "timezone": "America/Chicago"},
}

TMAX_TITLE_RE = re.compile(
    r"(?i)highest temperature in .+ on "
)
TMAX_QUESTION_RE = re.compile(
    r"(?i)(highest temperature|high temperature|daily high)"
)
EVENT_TITLE_RE = re.compile(
    r"(?i)highest temperature in (.+?) on ([A-Za-z]+ \d{1,2})\??"
)
DATE_RE = re.compile(r"(?:on\s+)?(\w+ \d{1,2}(?:,?\s*\d{4})?)")
RANGE_BUCKET_RE = re.compile(r"(\d+)\s*(?:°|degrees?)?\s*(?:to|-)\s*(\d+)", re.I)
LESS_THAN_RE = re.compile(r"(?:less than|under|below)\s*(\d+)", re.I)
GREATER_THAN_RE = re.compile(
    r"(?:(?:greater|more) than|over|above)\s*(\d+)", re.I
)
OR_BELOW_RE = re.compile(r"(?i)(\d+)\s*°?\s*or\s+below")
OR_ABOVE_RE = re.compile(r"(?i)(\d+)\s*°?\s*or\s+above")

RESOLUTION_PATTERNS = {
    "nws": re.compile(r"(?i)\b(?:national weather service|nws)\b"),
    "noaa": re.compile(r"(?i)\bnoaa\b"),
    "wunderground": re.compile(r"(?i)weather underground"),
    "accuweather": re.compile(r"(?i)accuweather"),
    "cli": re.compile(r"(?i)(?:climate report|\bcli\b)"),
    "asos": re.compile(r"(?i)\basos\b"),
    "station_code": re.compile(r"\bK[A-Z]{3}\b"),
}

_session: requests.Session | None = None


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MCP_Project/polymarket_recon (research)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = make_session()
    return _session


def gamma_get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{GAMMA_API}{path}"
    response = get_session().get(url, params=params or {}, timeout=30)
    response.raise_for_status()
    time.sleep(0.2)
    return response.json()


def clob_get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{CLOB_API}{path}"
    response = get_session().get(url, params=params or {}, timeout=30)
    response.raise_for_status()
    time.sleep(0.1)
    return response.json()


def iem_get(url: str, params: dict[str, Any]) -> requests.Response:
    response = get_session().get(url, params=params, timeout=20)
    response.raise_for_status()
    time.sleep(0.5)
    return response


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_section(
    name: str,
    fn: Callable[[], Any],
    results: dict[str, Any],
    key: str,
) -> Any:
    print(f"\n{'=' * 60}\nSECTION: {name}\n{'=' * 60}")
    try:
        value = fn()
        results[key] = value
        return value
    except Exception as exc:
        print(f"WARNING: Section '{name}' failed: {exc}")
        results[key] = {"error": str(exc)}
        return None


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def city_to_slug(city_name: str) -> str:
    normalized = city_name.strip().lower()
    slug = CITY_SLUG_MAP.get(normalized)
    if slug:
        return slug
    return normalized.replace(" ", "_").replace(".", "")


def parse_event_date(text: str, year_hint: str | None = None) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    date_str = match.group(1).strip()
    if re.search(r"\d{4}", date_str):
        for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
            try:
                return datetime.strptime(date_str.replace(",", " ").strip(), fmt.replace(",", "")).date().isoformat()
            except ValueError:
                try:
                    return datetime.strptime(date_str, fmt).date().isoformat()
                except ValueError:
                    continue
    year = date.today().year
    if year_hint:
        try:
            year = int(str(year_hint)[:4])
        except ValueError:
            pass
    for fmt in ("%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(date_str.split(",")[0].strip(), fmt).replace(year=year)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def parse_city(title: str, question: str = "") -> str | None:
    match = EVENT_TITLE_RE.search(title)
    if match:
        return city_to_slug(match.group(1).strip())
    combined = f"{title} {question}"
    city_match = re.search(r"(?i)(?:in|for)\s+([A-Za-z .]+?)\s+(?:on|,|\?)", combined)
    if city_match:
        return city_to_slug(city_match.group(1).strip())
    return None


def parse_bucket(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()

    or_below = OR_BELOW_RE.match(text)
    if or_below:
        return {"type": "LESS_THAN", "bounds": [None, int(or_below.group(1))]}

    or_above = OR_ABOVE_RE.match(text)
    if or_above:
        return {"type": "GREATER_THAN", "bounds": [int(or_above.group(1)), None]}

    less_than = LESS_THAN_RE.search(text)
    if less_than:
        return {"type": "LESS_THAN", "bounds": [None, int(less_than.group(1))]}

    greater_than = GREATER_THAN_RE.search(text)
    if greater_than:
        return {"type": "GREATER_THAN", "bounds": [int(greater_than.group(1)), None]}

    range_match = RANGE_BUCKET_RE.search(text)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        return {"type": "RANGE", "bounds": [low, high]}

    range_to = re.match(r"(?i)(\d+)\s*°?\s*to\s+(\d+)", text)
    if range_to:
        return {
            "type": "RANGE",
            "bounds": [int(range_to.group(1)), int(range_to.group(2))],
        }
    return None


def is_tmax_event(event: dict[str, Any]) -> bool:
    title = str(event.get("title", ""))
    if "temperature increase" in title.lower():
        return False
    if TMAX_TITLE_RE.search(title) or EVENT_TITLE_RE.search(title):
        return True
    for market in event.get("markets") or []:
        question = str(market.get("question", ""))
        if TMAX_QUESTION_RE.search(question):
            return True
    return False


def extract_market_record(market: dict[str, Any]) -> dict[str, Any]:
    token_ids = _parse_json_field(market.get("clobTokenIds")) or []
    outcome_prices = _parse_json_field(market.get("outcomePrices")) or []
    outcomes = _parse_json_field(market.get("outcomes")) or []

    yes_index = 0
    if outcomes and str(outcomes[0]).lower() != "yes":
        yes_index = 1 if len(token_ids) > 1 else 0

    label = market.get("groupItemTitle") or market.get("question") or ""
    bucket = parse_bucket(str(label)) or parse_bucket(str(market.get("question", "")))

    return {
        "market_id": market.get("id"),
        "question": market.get("question"),
        "conditionId": market.get("conditionId"),
        "clobTokenIds": token_ids,
        "yes_token_id": str(token_ids[yes_index]) if token_ids else None,
        "outcomePrices": outcome_prices,
        "volume": _to_float(market.get("volume")),
        "volume24hr": _to_float(market.get("volume24hr")),
        "liquidity": _to_float(market.get("liquidity")),
        "outcomes": outcomes,
        "description": market.get("description"),
        "acceptingOrders": market.get("acceptingOrders"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "minimum_tick_size": market.get("orderPriceMinTickSize"),
        "bucket_label": label,
        "bucket_type": bucket["type"] if bucket else None,
        "bucket_bounds": bucket["bounds"] if bucket else None,
    }


def extract_event_record(event: dict[str, Any]) -> dict[str, Any]:
    title = str(event.get("title", ""))
    year_hint = event.get("endDate") or event.get("endDateIso") or event.get("eventDate")
    parsed_title = EVENT_TITLE_RE.search(title)
    if parsed_title:
        city = city_to_slug(parsed_title.group(1).strip())
        event_date = parse_event_date(parsed_title.group(2), str(year_hint) if year_hint else None)
    else:
        city = parse_city(title)
        event_date = parse_event_date(title, str(year_hint) if year_hint else None)

    markets = [extract_market_record(m) for m in (event.get("markets") or [])]

    range_bounds = []
    for m in markets:
        if m.get("bucket_type") == "RANGE" and m.get("bucket_bounds"):
            low, high = m["bucket_bounds"]
            if low is not None and high is not None:
                range_bounds.append((low, high))

    bucket_widths = [high - low + 1 for low, high in range_bounds]
    min_bucket = min((b[0] for b in range_bounds), default=None)
    max_bucket = max((b[1] for b in range_bounds), default=None)

    total_volume = sum(m.get("volume") or 0 for m in markets)
    accepting = any(m.get("acceptingOrders") for m in markets) and not event.get("closed")

    return {
        "event_id": event.get("id"),
        "event_slug": event.get("slug"),
        "event_title": title,
        "description": event.get("description"),
        "resolutionSource": event.get("resolutionSource"),
        "startDate": event.get("startDate"),
        "endDate": event.get("endDate"),
        "createdAt": event.get("createdAt"),
        "active": event.get("active"),
        "closed": event.get("closed"),
        "archived": event.get("archived"),
        "neg_risk": bool(event.get("negRisk", event.get("enableNegRisk", False))),
        "city": city,
        "event_date": event_date,
        "n_buckets": len(markets),
        "bucket_width": int(round(sum(bucket_widths) / len(bucket_widths))) if bucket_widths else None,
        "min_bucket": min_bucket,
        "max_bucket": max_bucket,
        "total_volume": total_volume,
        "accepting_orders": accepting,
        "markets": markets,
    }


def paginate_gamma_events(
    extra_params: dict[str, Any] | None = None,
    label: str = "events",
) -> list[dict[str, Any]]:
    all_events: list[dict[str, Any]] = []
    offset = 0
    limit = 100
    page = 0
    params = {"limit": limit, "offset": offset}
    if extra_params:
        params.update(extra_params)

    while True:
        params["offset"] = offset
        page += 1
        batch = gamma_get("/events", params)
        if not batch:
            break
        all_events.extend(batch)
        if page % 10 == 0:
            print(f"  {label}: fetched {len(all_events)} events (page {page})...")
        if len(batch) < limit:
            break
        offset += limit
    return all_events


def section1_discover_tags(results: dict[str, Any]) -> dict[str, Any]:
    matching_tags: list[dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        tags = gamma_get("/tags", {"limit": limit, "offset": offset})
        if not tags:
            break
        for tag in tags:
            label = str(tag.get("label", ""))
            slug = str(tag.get("slug", ""))
            if re.search(r"(?i)weather|temperature|climate", label) or re.search(
                r"(?i)weather|temperature|climate", slug
            ):
                entry = {
                    "id": tag.get("id"),
                    "label": label,
                    "slug": slug,
                }
                matching_tags.append(entry)
                print(f"  Tag: id={entry['id']} label={entry['label']} slug={entry['slug']}")
        if len(tags) < limit:
            break
        offset += limit

    weather_tag_id = None
    for slug_pref in ("highest-temperature", "daily-temperature", "weather"):
        for tag in matching_tags:
            if tag["slug"] == slug_pref or tag["label"].strip().lower() == slug_pref.replace("-", " "):
                weather_tag_id = tag["id"]
                break
        if weather_tag_id is not None:
            break
    if weather_tag_id is None and matching_tags:
        weather_tag_id = matching_tags[0]["id"]

    print(f"\nSelected weather_tag_id: {weather_tag_id}")
    return {"weather_tag_id": weather_tag_id, "weather_tags": matching_tags}


def section2_enumerate_events(results: dict[str, Any]) -> dict[str, Any]:
    weather_tag_id = results.get("weather_tag_id")
    seen_ids: set[str | int] = set()
    raw_events: list[dict[str, Any]] = []

    if weather_tag_id:
        print("Strategy A: tag-based discovery...")
        for params, label in [
            ({"tag_id": weather_tag_id, "active": "true", "closed": "false"}, "active"),
            ({"tag_id": weather_tag_id}, "all"),
        ]:
            batch = paginate_gamma_events(params, label=f"tag/{label}")
            for event in batch:
                eid = event.get("id")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    raw_events.append(event)
            print(f"  tag/{label}: {len(batch)} events")

    print("Strategy B: keyword fallback scan (active events)...")
    keyword_events = paginate_gamma_events(
        {"active": "true", "closed": "false"},
        label="keyword_scan",
    )
    added = 0
    for event in keyword_events:
        eid = event.get("id")
        if eid in seen_ids:
            continue
        if not is_tmax_event(event):
            continue
        seen_ids.add(eid)
        raw_events.append(event)
        added += 1
    print(f"  keyword fallback added {added} events")

    tmax_events = [e for e in raw_events if is_tmax_event(e)]
    print(f"\nTotal raw events: {len(raw_events)}, Tmax-filtered: {len(tmax_events)}")

    records = [extract_event_record(e) for e in tmax_events]
    save_json(OUTPUT_DIR / "all_events.json", records)

    summary_rows = []
    for rec in records:
        summary_rows.append(
            {
                "event_id": rec["event_id"],
                "event_slug": rec["event_slug"],
                "event_title": rec["event_title"],
                "city": rec["city"],
                "event_date": rec["event_date"],
                "n_buckets": rec["n_buckets"],
                "bucket_width": rec["bucket_width"],
                "min_bucket": rec["min_bucket"],
                "max_bucket": rec["max_bucket"],
                "neg_risk": rec["neg_risk"],
                "total_volume": rec["total_volume"],
                "created_at": rec["createdAt"],
                "accepting_orders": rec["accepting_orders"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("event_date", na_position="last")
        print("\nEvents summary (sorted by event_date):")
        print(summary_df.to_string(index=False))
    save_csv(summary_df, OUTPUT_DIR / "events_summary.csv")

    n_active = sum(1 for r in records if r.get("active") and not r.get("closed"))
    n_closed = sum(1 for r in records if r.get("closed"))
    cities = sorted({r["city"] for r in records if r.get("city")})

    return {
        "records": records,
        "summary_df": summary_df,
        "n_events_found": len(records),
        "n_active_events": n_active,
        "n_closed_events": n_closed,
        "n_unique_cities": len(cities),
        "cities": cities,
    }


def section3_horizon_analysis(section2: dict[str, Any]) -> dict[str, Any]:
    records = section2.get("records") or []
    rows = []
    for rec in records:
        if not rec.get("city") or not rec.get("event_date") or not rec.get("createdAt"):
            continue
        try:
            event_dt = date.fromisoformat(str(rec["event_date"]))
            created_dt = pd.to_datetime(rec["createdAt"], utc=True).date()
            horizon = (event_dt - created_dt).days
        except (ValueError, TypeError):
            continue
        rows.append(
            {
                "event_id": rec["event_id"],
                "city": rec["city"],
                "event_date": rec["event_date"],
                "created_at": rec["createdAt"],
                "horizon_days": horizon,
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        horizon_df = pd.DataFrame(
            columns=[
                "city",
                "n_dates",
                "earliest_date",
                "latest_date",
                "min_horizon",
                "max_horizon",
                "median_horizon",
            ]
        )
    else:
        grouped = detail_df.groupby("city").agg(
            n_dates=("event_date", "nunique"),
            earliest_date=("event_date", "min"),
            latest_date=("event_date", "max"),
            min_horizon=("horizon_days", "min"),
            max_horizon=("horizon_days", "max"),
            median_horizon=("horizon_days", "median"),
        ).reset_index()
        horizon_df = grouped

    print("\nHorizon analysis by city:")
    print(horizon_df.to_string(index=False))
    save_csv(horizon_df, OUTPUT_DIR / "horizon_analysis.csv")

    return {
        "detail": detail_df.to_dict("records") if not detail_df.empty else [],
        "summary": horizon_df.to_dict("records") if not horizon_df.empty else [],
    }


def _pick_sample_markets(records: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    active_pool = [
        r
        for r in records
        if r.get("accepting_orders") and not r.get("closed") and r.get("markets")
    ]
    closed_pool = sorted(
        [r for r in records if r.get("closed") and r.get("markets")],
        key=lambda r: str(r.get("endDate") or ""),
        reverse=True,
    )

    def pick_distinct(pool: list[dict], n: int) -> list[dict]:
        picked: list[dict] = []
        seen_cities: set[str] = set()
        for rec in pool:
            city = rec.get("city") or "unknown"
            if city in seen_cities and len(picked) >= n:
                continue
            picked.append(rec)
            seen_cities.add(city)
            if len(picked) >= n:
                break
        if len(picked) < n:
            for rec in pool:
                if rec not in picked:
                    picked.append(rec)
                if len(picked) >= n:
                    break
        return picked[:n]

    return pick_distinct(active_pool, 3), pick_distinct(closed_pool, 3)


def _test_price_history(token_id: str) -> dict[str, Any]:
    fidelities = [1, 5, 15, 60, 360, 720]
    working: list[int] = []
    earliest = latest = None
    max_points = 0

    for fidelity in fidelities:
        try:
            payload = clob_get(
                "/prices-history",
                {"market": token_id, "interval": "max", "fidelity": fidelity},
            )
            history = payload.get("history") or []
            if history:
                working.append(fidelity)
                timestamps = [entry["t"] for entry in history if "t" in entry]
                if timestamps:
                    earliest = min(timestamps) if earliest is None else min(earliest, min(timestamps))
                    latest = max(timestamps) if latest is None else max(latest, max(timestamps))
                max_points = max(max_points, len(history))
        except Exception as exc:
            print(f"    fidelity={fidelity} failed: {exc}")

    lookback_days = None
    if earliest and latest:
        lookback_days = (latest - earliest) / 86400.0

    return {
        "working_fidelities": working,
        "earliest_timestamp": earliest,
        "latest_timestamp": latest,
        "n_points_max": max_points,
        "lookback_days": lookback_days,
    }


def _fetch_order_book(token_id: str) -> dict[str, Any]:
    book = clob_get("/book", {"token_id": token_id})
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    def _price(level: dict) -> float:
        return float(level.get("price", 0))

    def _size(level: dict) -> float:
        return float(level.get("size", 0))

    best_bid = max((_price(b) for b in bids), default=None)
    best_ask = min((_price(a) for a in asks if _price(a) > 0), default=None)
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None

    return {
        "n_bids": len(bids),
        "n_asks": len(asks),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "total_bid_size": sum(_size(b) for b in bids),
        "total_ask_size": sum(_size(a) for a in asks),
    }


def section4_price_history(section2: dict[str, Any]) -> list[dict[str, Any]]:
    records = section2.get("records") or []
    active_samples, closed_samples = _pick_sample_markets(records)
    results: list[dict[str, Any]] = []

    for label, samples in [("active", active_samples), ("closed", closed_samples)]:
        print(f"\nTesting {label} markets ({len(samples)} samples)...")
        for rec in samples:
            market = rec["markets"][0]
            token_id = market.get("yes_token_id")
            if not token_id:
                continue
            print(f"  {rec.get('city')} {rec.get('event_date')} token={token_id[:20]}...")
            entry: dict[str, Any] = {
                "status": label,
                "city": rec.get("city"),
                "event_date": rec.get("event_date"),
                "event_id": rec.get("event_id"),
                "market_question": market.get("question"),
                "yes_token_id": token_id,
            }
            entry.update(_test_price_history(token_id))
            if label == "active":
                try:
                    entry["order_book"] = _fetch_order_book(token_id)
                except Exception as exc:
                    entry["order_book"] = {"error": str(exc)}
            results.append(entry)

    save_json(OUTPUT_DIR / "price_history_test.json", results)

    all_working = [f for r in results for f in r.get("working_fidelities", [])]
    min_fidelity = min(all_working) if all_working else None
    max_lookback = max(
        (r.get("lookback_days") or 0 for r in results),
        default=0,
    )
    print(f"\nPrice history summary:")
    print(f"  Min working fidelity: {min_fidelity} minutes")
    print(f"  Max lookback: {max_lookback:.1f} days")
    print(f"  Samples tested: {len(results)}")

    return results


def _resolve_city_station(city: str) -> dict[str, Any]:
    if city in STATION_REGISTRY:
        return STATION_REGISTRY[city].copy()
    slug = city_to_slug(city)
    if slug in STATION_REGISTRY:
        cfg = STATION_REGISTRY[slug].copy()
        cfg["city"] = slug
        return cfg

    if CITY_CONFIG_PATH.exists():
        config = json.loads(CITY_CONFIG_PATH.read_text(encoding="utf-8"))
        if slug in config:
            c = config[slug]
            return {
                "station": c["nws_station"],
                "lat": c["lat"],
                "lon": c["lon"],
                "timezone": c["timezone"],
            }
    return {}


def _check_cli(station: str) -> dict[str, Any]:
    pil = f"CLI{station[-3:]}"
    try:
        response = iem_get(IEM_CLI_URL, {"pil": pil, "fmt": "text", "limit": 5})
        text = response.text.strip()
        available = bool(text) and not text.upper().startswith("ERROR")
        sample_count = text.count("CLIMATE SUMMARY") if available else 0
        return {
            "available": available,
            "sample_count": sample_count,
            "pil": pil,
            "date_range": None,
        }
    except Exception as exc:
        return {"available": False, "sample_count": 0, "error": str(exc)}


def _check_asos(station: str) -> dict[str, Any]:
    try:
        response = iem_get(
            IEM_ASOS_URL,
            {
                "station": station,
                "data": "tmpf",
                "year1": 2026,
                "month1": 6,
                "day1": 15,
                "year2": 2026,
                "month2": 6,
                "day2": 16,
                "tz": "Etc/UTC",
                "format": "onlycomma",
                "latlon": "no",
                "elev": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
                "report_type": 3,
            },
        )
        text = response.text.strip()
        if not text or text.upper().startswith("ERROR"):
            return {"available": False, "n_obs": 0}
        n_obs = max(text.count("\n") - 1, 0)
        return {"available": n_obs > 0, "n_obs": n_obs}
    except Exception as exc:
        return {"available": False, "n_obs": 0, "error": str(exc)}


def _check_mos(station: str) -> dict[str, Any]:
    try:
        response = iem_get(
            IEM_MOS_URL,
            {
                "station": station,
                "model": "NBS",
                "year1": 2026,
                "month1": 6,
                "day1": 15,
                "hour1": 0,
                "year2": 2026,
                "month2": 6,
                "day2": 16,
                "hour2": 23,
                "tz": "UTC",
                "format": "csv",
            },
        )
        text = response.text.strip()
        if not text or text.upper().startswith("ERROR"):
            return {"available": False, "models_available": []}
        frame = pd.read_csv(StringIO(text))
        models = sorted(frame["model"].dropna().unique().tolist()) if "model" in frame.columns else ["NBS"]
        return {"available": not frame.empty, "models_available": models}
    except Exception as exc:
        return {"available": False, "models_available": [], "error": str(exc)}


def _check_openmeteo(lat: float, lon: float) -> dict[str, Any]:
    try:
        response = get_session().get(
            OPENMETEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "models": "ecmwf_ifs025,gfs_seamless",
                "timezone": "auto",
                "past_days": 7,
                "forecast_days": 7,
            },
            timeout=20,
        )
        response.raise_for_status()
        time.sleep(0.3)
        payload = response.json()
        daily = payload.get("daily") or {}
        n_days = len(daily.get("time") or [])
        ecmwf_available = "temperature_2m_max_ecmwf_ifs025" in daily and n_days > 0
        gfs_available = "temperature_2m_max_gfs_seamless" in daily and n_days > 0
        return {
            "ecmwf_available": ecmwf_available,
            "gfs_available": gfs_available,
            "n_days": n_days,
            "models_available": list(daily.keys()),
        }
    except Exception as exc:
        return {
            "ecmwf_available": False,
            "gfs_available": False,
            "n_days": 0,
            "error": str(exc),
        }


def _compute_tier(row: dict[str, Any]) -> str:
    sources = [
        row.get("cli_available"),
        row.get("asos_available"),
        row.get("mos_available"),
        row.get("ecmwf_available"),
        row.get("gfs_available"),
    ]
    n_available = sum(1 for s in sources if s)
    n_missing = 5 - n_available
    has_model = row.get("has_trained_model")

    if has_model and n_missing == 0:
        return "Tier 1"
    if n_missing == 0:
        return "Tier 2"
    if n_missing <= 2:
        return "Tier 3"
    return "Tier 4"


def section5_data_availability(section2: dict[str, Any]) -> list[dict[str, Any]]:
    cities = section2.get("cities") or []
    rows: list[dict[str, Any]] = []

    print(f"\nChecking data availability for {len(cities)} cities...")
    for city in cities:
        print(f"  {city}...")
        station_cfg = _resolve_city_station(city)
        if not station_cfg:
            rows.append(
                {
                    "city": city,
                    "station": None,
                    "lat": None,
                    "lon": None,
                    "timezone": None,
                    "cli_available": False,
                    "asos_available": False,
                    "mos_available": False,
                    "ecmwf_available": False,
                    "gfs_available": False,
                    "has_trained_model": city in TRAIN_SLUGS,
                    "model_city_name": city if city in TRAIN_SLUGS else None,
                    "data_quality_tier": "Tier 4",
                }
            )
            continue

        station = station_cfg["station"]
        lat = station_cfg["lat"]
        lon = station_cfg["lon"]
        tz = station_cfg.get("timezone")

        cli = _check_cli(station)
        asos = _check_asos(station)
        mos = _check_mos(station)
        om = _check_openmeteo(lat, lon)

        row = {
            "city": city,
            "station": station,
            "lat": lat,
            "lon": lon,
            "timezone": tz,
            "cli_available": cli.get("available", False),
            "asos_available": asos.get("available", False),
            "mos_available": mos.get("available", False),
            "ecmwf_available": om.get("ecmwf_available", False),
            "gfs_available": om.get("gfs_available", False),
            "has_trained_model": city in TRAIN_SLUGS,
            "model_city_name": city if city in TRAIN_SLUGS else None,
        }
        row["data_quality_tier"] = _compute_tier(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nData availability:")
    print(df.to_string(index=False))
    save_csv(df, OUTPUT_DIR / "city_data_availability.csv")
    return rows


def _analyze_resolution_text(text: str) -> dict[str, Any]:
    flags = {key: bool(pat.search(text)) for key, pat in RESOLUTION_PATTERNS.items()}
    conflicting = flags["wunderground"] or flags["accuweather"]
    nws_cli = (
        flags["nws"] or flags["noaa"] or flags["cli"]
    ) and not conflicting
    if flags["station_code"] and not conflicting:
        nws_cli = True

    detected = [key for key, val in flags.items() if val and key != "station_code"]
    return {
        "nws_cli_match": nws_cli,
        "flags": flags,
        "detected_sources": detected,
        "resolution_text_snippet": text[:500] if text else "",
    }


def section6_resolution_sources(section2: dict[str, Any]) -> dict[str, Any]:
    records = section2.get("records") or []
    event_results: list[dict[str, Any]] = []
    city_stats: dict[str, dict[str, int]] = {}

    for rec in records:
        parts = [
            str(rec.get("description") or ""),
            str(rec.get("resolutionSource") or ""),
            str(rec.get("event_title") or ""),
        ]
        for market in rec.get("markets") or []:
            parts.append(str(market.get("description") or ""))

        combined = "\n".join(parts)
        analysis = _analyze_resolution_text(combined)
        city = rec.get("city") or "unknown"

        entry = {
            "event_id": rec.get("event_id"),
            "city": city,
            "event_date": rec.get("event_date"),
            "event_title": rec.get("event_title"),
            "nws_cli_match": analysis["nws_cli_match"],
            "detected_sources": analysis["detected_sources"],
            "resolution_text_snippet": analysis["resolution_text_snippet"],
        }
        event_results.append(entry)

        stats = city_stats.setdefault(city, {"total": 0, "nws_match": 0})
        stats["total"] += 1
        if analysis["nws_cli_match"]:
            stats["nws_match"] += 1

    city_summary = []
    flagged_cities = []
    for city, stats in sorted(city_stats.items()):
        match_rate = stats["nws_match"] / stats["total"] if stats["total"] else 0
        flagged = match_rate < 0.5
        if flagged:
            flagged_cities.append(city)
        city_summary.append(
            {
                "city": city,
                "n_events": stats["total"],
                "nws_cli_matches": stats["nws_match"],
                "match_rate": round(match_rate, 3),
                "flagged": flagged,
            }
        )

    payload = {
        "events": event_results,
        "city_summary": city_summary,
        "flagged_cities": flagged_cities,
    }
    save_json(OUTPUT_DIR / "resolution_sources.json", payload)

    n_match = sum(1 for e in event_results if e["nws_cli_match"])
    print(f"\nResolution sources: NWS CLI confirmed for {n_match}/{len(event_results)} events")
    if flagged_cities:
        print(f"  Flagged cities (majority non-NWS): {flagged_cities}")

    return payload


def _build_cities_summary(section2: dict[str, Any]) -> list[dict[str, Any]]:
    records = section2.get("records") or []
    by_city: dict[str, list] = {}
    for rec in records:
        city = rec.get("city")
        if not city:
            continue
        by_city.setdefault(city, []).append(rec)

    summary = []
    for city, events in sorted(by_city.items()):
        volumes = [e.get("total_volume") or 0 for e in events]
        widths = [e.get("bucket_width") for e in events if e.get("bucket_width")]
        summary.append(
            {
                "city": city,
                "n_events": len(events),
                "total_volume": sum(volumes),
                "median_bucket_width": int(round(float(pd.Series(widths).median()))) if widths else None,
                "has_trained_model": city in TRAIN_SLUGS,
            }
        )
    return summary


def print_final_summary(results: dict[str, Any]) -> None:
    section2 = results.get("section2") or {}
    section3 = results.get("section3") or {}
    section4 = results.get("section4") or []
    section5 = results.get("section5") or []
    section6 = results.get("section6") or {}

    cities = section2.get("cities") or []
    tier1 = sum(1 for r in section5 if r.get("data_quality_tier") == "Tier 1")
    tier2 = sum(1 for r in section5 if r.get("data_quality_tier") == "Tier 2")
    trained_cities = sum(1 for r in section5 if r.get("has_trained_model"))

    dates = [
        r.get("event_date")
        for r in (section2.get("records") or [])
        if r.get("event_date")
    ]
    earliest = min(dates) if dates else "n/a"
    latest = max(dates) if dates else "n/a"

    horizons = [r.get("horizon_days") for r in section3.get("detail", []) if r.get("horizon_days") is not None]
    min_horizon = min(horizons) if horizons else "n/a"
    max_horizon = max(horizons) if horizons else "n/a"

    all_fidelities = [f for r in section4 for f in r.get("working_fidelities", [])]
    min_fidelity = min(all_fidelities) if all_fidelities else "n/a"
    lookbacks = [r.get("lookback_days") for r in section4 if r.get("lookback_days")]
    max_lookback = max(lookbacks) if lookbacks else "n/a"

    events = section6.get("events") or []
    n_match = sum(1 for e in events if e.get("nws_cli_match"))
    n_cities_res = len(section6.get("city_summary") or [])

    today = date.today().isoformat()
    active_today = sum(
        1
        for r in (section2.get("records") or [])
        if r.get("event_date") == today and r.get("accepting_orders")
    )

    print(f"\n{'=' * 60}")
    print("FINAL RECON SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total events found: {section2.get('n_events_found', 0)}")
    print(f"Unique cities: {len(cities)} ({', '.join(cities) if cities else 'none'})")
    print(f"Cities with trained models: {trained_cities}")
    print(f"Cities Tier 1 (model + full data): {tier1}")
    print(f"Cities trainable (Tier 2): {tier2}")
    print(f"Active events today: {active_today}")
    print(f"Date range: {earliest} to {latest}")
    print(f"Market horizon: {min_horizon} to {max_horizon} days before event")
    print(f"Price history: available at fidelity {min_fidelity} minutes, going back {max_lookback} days")
    print(f"Resolution source: NWS CLI confirmed for {n_match}/{len(events)} events ({n_cities_res} cities)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"run_timestamp": datetime.now(timezone.utc).isoformat()}

    s1 = run_section("Discover weather tag", lambda: section1_discover_tags(results), results, "section1")
    if s1:
        results["weather_tag_id"] = s1.get("weather_tag_id")
        results["weather_tags"] = s1.get("weather_tags")

    s2 = run_section(
        "Enumerate Tmax events",
        lambda: section2_enumerate_events(results),
        results,
        "section2",
    )

    s3 = run_section(
        "Market horizon analysis",
        lambda: section3_horizon_analysis(s2 or {}),
        results,
        "section3",
    )

    s4 = run_section(
        "Historical price data availability",
        lambda: section4_price_history(s2 or {}),
        results,
        "section4",
    )

    s5 = run_section(
        "NWS/ASOS data availability",
        lambda: section5_data_availability(s2 or {}),
        results,
        "section5",
    )

    s6 = run_section(
        "Resolution source verification",
        lambda: section6_resolution_sources(s2 or {}),
        results,
        "section6",
    )

    master = {
        "run_timestamp": results["run_timestamp"],
        "weather_tag_id": results.get("weather_tag_id"),
        "n_events_found": (s2 or {}).get("n_events_found", 0),
        "n_unique_cities": (s2 or {}).get("n_unique_cities", 0),
        "n_active_events": (s2 or {}).get("n_active_events", 0),
        "n_closed_events": (s2 or {}).get("n_closed_events", 0),
        "cities_summary": _build_cities_summary(s2 or {}),
        "horizon_analysis": (s3 or {}).get("summary", []),
        "price_history_test": s4 or [],
        "data_availability": s5 or [],
        "resolution_sources": (s6 or {}).get("city_summary", []),
    }
    save_json(OUTPUT_DIR / "recon_results.json", master)
    results["master"] = master

    print_final_summary(results)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
