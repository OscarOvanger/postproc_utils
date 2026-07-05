#!/usr/bin/env python3
"""Scan Polymarket Tmax modal buckets for target cities (read-only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import (  # noqa: E402
    BUCKET_FROM_QUESTION_RE,
    EVENT_TITLE_RE,
    GAMMA_API,
    _month_day_label,
    _parse_event_date,
    _parse_order_book_sides,
    build_clob_client,
    parse_bucket_label,
)
from src.poly_trading_pipeline import WEATHER_TAG_ID, _build_http_session  # noqa: E402

TEMP_KEYWORD_RE = re.compile(r"(?i)(temperature|\bhigh\b|degrees)")
MIN_MIDPOINT = 0.35
MAX_MIDPOINT = 0.60
N_CONTRACTS = 5
MAKER_TICK = 0.01

# slug -> display name (sorted alphabetically by display name in output)
TARGET_CITIES: list[tuple[str, str]] = [
    ("austin", "Austin"),
    ("atlanta", "Atlanta"),
    ("chicago", "Chicago"),
    ("dallas", "Dallas"),
    ("houston", "Houston"),
    ("los_angeles", "Los Angeles"),
    ("miami", "Miami"),
    ("new_york", "New York"),
    ("san_francisco", "San Francisco"),
    ("seattle", "Seattle"),
]

CITY_SEARCH_ALIASES: dict[str, list[str]] = {
    "austin": ["austin"],
    "atlanta": ["atlanta"],
    "chicago": ["chicago"],
    "dallas": ["dallas"],
    "houston": ["houston"],
    "los_angeles": ["los angeles"],
    "miami": ["miami"],
    "new_york": ["new york", "new york city", "nyc"],
    "san_francisco": ["san francisco"],
    "seattle": ["seattle"],
}


@dataclass
class BucketQuote:
    label: str
    token_id: str
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None


@dataclass
class CityScan:
    slug: str
    display_name: str
    status: str = "ok"
    buckets: list[BucketQuote] = field(default_factory=list)
    modal: BucketQuote | None = None
    runner_up: BucketQuote | None = None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _extract_bucket_label(market: dict[str, Any]) -> str:
    group_title = market.get("groupItemTitle")
    if group_title:
        return str(group_title)
    question = str(market.get("question", ""))
    match = BUCKET_FROM_QUESTION_RE.search(question)
    if match:
        return match.group(1).strip()
    return question


def _title_matches_city(title: str, slug: str) -> bool:
    title_lower = title.lower()
    for alias in CITY_SEARCH_ALIASES[slug]:
        if len(alias) <= 3:
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", title_lower):
                return True
        elif alias in title_lower:
            return True
    return False


def _text_has_date(text: str, event_date: str) -> bool:
    month_day = _month_day_label(event_date)
    if month_day.lower() in text.lower():
        return True
    parsed = date.fromisoformat(event_date)
    for fmt in (
        parsed.strftime("%B %d, %Y"),
        parsed.strftime("%B %d %Y"),
        parsed.strftime("%b %d, %Y"),
        parsed.strftime("%b %d"),
        event_date,
    ):
        if fmt.lower() in text.lower():
            return True
    return False


def _is_tmax_market_text(question: str, description: str) -> bool:
    combined = f"{question} {description}"
    return bool(TEMP_KEYWORD_RE.search(combined))


def _event_matches_date(event: dict[str, Any], event_date: str) -> bool:
    title = str(event.get("title", ""))
    description = str(event.get("description", ""))
    year_hint = event.get("eventDate") or event.get("endDate")

    parsed = EVENT_TITLE_RE.search(title)
    if parsed:
        try:
            if _parse_event_date(parsed.group(2), year_hint=str(year_hint) if year_hint else None) == event_date:
                return True
        except ValueError:
            pass

    end_iso = str(event.get("endDateIso") or event.get("endDate") or "")
    if end_iso.startswith(event_date):
        return True

    return _text_has_date(f"{title} {description}", event_date)


def _ingest_event_buckets(
    event: dict[str, Any],
    *,
    event_date: str,
    slug: str,
    display_name: str,
    buckets: dict[str, dict[str, Any]],
    include_closed: bool = False,
) -> None:
    title = str(event.get("title", ""))
    description = str(event.get("description", ""))
    if not _is_tmax_market_text(title, description):
        return
    if not _event_matches_date(event, event_date):
        return
    if not _title_matches_city(title, slug):
        return

    entry = buckets.setdefault(
        slug,
        {
            "slug": slug,
            "display_name": display_name,
            "buckets": [],
            "seen_tokens": set(),
        },
    )

    for market in event.get("markets") or []:
        if not include_closed and (
            market.get("closed") or market.get("acceptingOrders") is False
        ):
            continue

        question = str(market.get("question", ""))
        market_desc = str(market.get("description", ""))
        if not _is_tmax_market_text(question, market_desc):
            continue

        try:
            label = _extract_bucket_label(market)
            parse_bucket_label(label)
        except ValueError:
            continue

        token_ids = _parse_json_field(market.get("clobTokenIds"))
        outcomes = _parse_json_field(market.get("outcomes"))
        if not token_ids:
            continue

        yes_index = 0
        if outcomes and str(outcomes[0]).lower() != "yes":
            yes_index = 1 if len(token_ids) > 1 else 0
        token_id = str(token_ids[yes_index])
        if token_id in entry["seen_tokens"]:
            continue
        entry["seen_tokens"].add(token_id)
        entry["buckets"].append({"label": label, "token_id": token_id})


def _discover_from_gamma_search(
    session: requests.Session,
    *,
    event_date: str,
    include_closed: bool = False,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    month_day = _month_day_label(event_date)

    for slug, display_name in TARGET_CITIES:
        query = f"Highest temperature in {display_name} on {month_day}"
        try:
            response = session.get(
                f"{GAMMA_API}/public-search",
                params={"q": query},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"WARNING: Gamma search failed for {display_name}: {exc}", file=sys.stderr)
            continue

        for event in response.json().get("events", []):
            _ingest_event_buckets(
                event,
                event_date=event_date,
                slug=slug,
                display_name=display_name,
                buckets=buckets,
                include_closed=include_closed,
            )
        time.sleep(0.15)

    return buckets


def _discover_from_weather_tag(
    session: requests.Session,
    *,
    event_date: str,
    include_closed: bool = False,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    offset = 0

    while True:
        params = {
            "tag_id": WEATHER_TAG_ID,
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": offset,
        }
        try:
            response = session.get(f"{GAMMA_API}/events", params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"WARNING: Gamma events fetch failed: {exc}", file=sys.stderr)
            break

        batch = response.json()
        if not batch:
            break

        for event in batch:
            title = str(event.get("title", ""))
            if not _event_matches_date(event, event_date):
                continue
            for slug, display_name in TARGET_CITIES:
                if _title_matches_city(title, slug):
                    _ingest_event_buckets(
                        event,
                        event_date=event_date,
                        slug=slug,
                        display_name=display_name,
                        buckets=buckets,
                        include_closed=include_closed,
                    )

        if len(batch) < 100:
            break
        offset += 100
        time.sleep(0.15)

    return buckets


def discover_markets(
    event_date: str,
    *,
    include_closed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return slug -> {display_name, buckets: [{label, token_id}]}."""
    session = _build_http_session()
    merged: dict[str, dict[str, Any]] = {}

    for source in (
        _discover_from_gamma_search(
            session, event_date=event_date, include_closed=include_closed
        ),
        _discover_from_weather_tag(
            session, event_date=event_date, include_closed=include_closed
        ),
    ):
        for slug, entry in source.items():
            if slug not in merged:
                merged[slug] = {
                    "slug": slug,
                    "display_name": entry["display_name"],
                    "buckets": list(entry["buckets"]),
                    "seen_tokens": set(entry["seen_tokens"]),
                }
                continue
            existing = merged[slug]
            for bucket in entry["buckets"]:
                token_id = bucket["token_id"]
                if token_id in existing["seen_tokens"]:
                    continue
                existing["seen_tokens"].add(token_id)
                existing["buckets"].append(bucket)

    for entry in merged.values():
        entry.pop("seen_tokens", None)
    return merged


def compute_midpoint(best_bid: float | None, best_ask: float | None) -> float | None:
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    if best_ask is not None:
        return best_ask
    return None


def fetch_bucket_quotes(client: Any, buckets: list[dict[str, str]]) -> list[BucketQuote]:
    quotes: list[BucketQuote] = []
    for bucket in buckets:
        token_id = bucket["token_id"]
        try:
            book = client.get_order_book(token_id)
            best_bid, best_ask = _parse_order_book_sides(book)
        except Exception as exc:
            print(
                f"  WARNING: order book failed for {bucket['label']!r}: {exc}",
                file=sys.stderr,
            )
            best_bid, best_ask = None, None

        midpoint = compute_midpoint(best_bid, best_ask)
        if midpoint is None:
            continue

        quotes.append(
            BucketQuote(
                label=bucket["label"],
                token_id=token_id,
                best_bid=best_bid,
                best_ask=best_ask,
                midpoint=midpoint,
            )
        )
        time.sleep(0.1)
    return quotes


def scan_city(client: Any, slug: str, display_name: str, market: dict[str, Any] | None) -> CityScan:
    if market is None or not market.get("buckets"):
        return CityScan(slug=slug, display_name=display_name, status="no_market")

    quotes = fetch_bucket_quotes(client, market["buckets"])
    if not quotes:
        return CityScan(slug=slug, display_name=display_name, status="no_liquidity")

    ranked = sorted(quotes, key=lambda row: row.midpoint or -1.0, reverse=True)
    scan = CityScan(
        slug=slug,
        display_name=display_name,
        status="ok",
        buckets=ranked,
        modal=ranked[0],
        runner_up=ranked[1] if len(ranked) > 1 else None,
    )
    return scan


def format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.2f}"


def in_range(midpoint: float | None) -> bool:
    return midpoint is not None and MIN_MIDPOINT <= midpoint <= MAX_MIDPOINT


def print_table(scans: list[CityScan], event_date: str) -> None:
    print(f"\nModal bucket scan for {event_date}")
    print(
        f"{'City':<16} {'Modal Bucket':<14} {'Best Bid':>9} {'Best Ask':>9} "
        f"{'Midpoint':>9} {'In Range':>9}"
    )
    print("-" * 72)

    for scan in scans:
        if scan.status == "no_market":
            print(f"{scan.display_name:<16} {'NO MARKET':<14} {'':>9} {'':>9} {'':>9} {'':>9}")
            continue
        if scan.status == "no_liquidity":
            print(
                f"{scan.display_name:<16} {'NO LIQUIDITY':<14} {'':>9} {'':>9} {'':>9} {'':>9}"
            )
            continue

        modal = scan.modal
        assert modal is not None
        range_flag = "YES" if in_range(modal.midpoint) else "NO"
        print(
            f"{scan.display_name:<16} {modal.label:<14} "
            f"{format_price(modal.best_bid):>9} {format_price(modal.best_ask):>9} "
            f"{format_price(modal.midpoint):>9} {range_flag:>9}"
        )


def print_recommendations(scans: list[CityScan]) -> None:
    in_range_scans = [
        scan
        for scan in scans
        if scan.status == "ok" and scan.modal is not None and in_range(scan.modal.midpoint)
    ]
    if not in_range_scans:
        print("\nNo in-range modal buckets.")
        return

    print(f"\nIn-range modal buckets ({MIN_MIDPOINT:.2f}–{MAX_MIDPOINT:.2f}):")
    for scan in in_range_scans:
        modal = scan.modal
        assert modal is not None
        if modal.best_ask is None:
            print(f"  {scan.display_name}: SKIP (no best ask on modal bucket)")
            continue

        maker_price = round(modal.best_ask - MAKER_TICK, 2)
        total_cost = round(N_CONTRACTS * maker_price, 2)
        print(
            f"  {scan.display_name} ({modal.label}): "
            f"maker buy @ {format_price(maker_price)} x {N_CONTRACTS} "
            f"= {format_price(total_cost)}"
        )
        if scan.runner_up is not None:
            print(
                f"    2nd bucket: {scan.runner_up.label} "
                f"mid={format_price(scan.runner_up.midpoint)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan Polymarket modal Tmax bucket prices for target cities."
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Event date (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()

    event_date = args.date
    try:
        date.fromisoformat(event_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid --date {event_date!r}: {exc}") from exc

    print(f"Initializing Polymarket CLOB client...")
    client = build_clob_client()

    print(f"Discovering Tmax markets for {event_date}...")
    discovered = discover_markets(event_date)

    scans: list[CityScan] = []
    for slug, display_name in TARGET_CITIES:
        market = discovered.get(slug)
        print(f"  {display_name}: {len(market['buckets']) if market else 0} buckets found")
        scans.append(scan_city(client, slug, display_name, market))

    print_table(scans, event_date)
    print_recommendations(scans)


if __name__ == "__main__":
    main()
