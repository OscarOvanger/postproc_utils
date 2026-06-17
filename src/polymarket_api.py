"""Polymarket CLOB v2 API wrapper for Tmax trading."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

from py_clob_client_v2 import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    ClobClient,
    OrderArgs,
    OrderPayload,
    OrderType,
    PartialCreateOrderOptions,
    Side,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "config" / "polymarket_credentials.json"
DEFAULT_MARKETS_PATH = PROJECT_ROOT / "config" / "polymarket_markets.json"
ORDER_LOG_PATH = PROJECT_ROOT / "logs" / "poly_orders.jsonl"
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CHAIN_ID = 137
SIGNATURE_TYPE = 3  # POLY_1271 deposit-wallet signing

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
    "houston": "houston",
    "los angeles": "los_angeles",
    "new york": "new_york_city",
    "new york city": "new_york_city",
    "nyc": "new_york_city",
    "oklahoma city": "oklahoma_city",
    "philadelphia": "philadelphia",
    "phoenix": "phoenix",
    "san francisco": "san_francisco",
}

CITY_DISPLAY_NAMES = {
    "austin": "Austin",
    "chicago_midway": "Chicago",
    "houston": "Houston",
    "los_angeles": "Los Angeles",
    "new_york_city": "New York City",
    "oklahoma_city": "Oklahoma City",
    "philadelphia": "Philadelphia",
    "phoenix": "Phoenix",
    "san_francisco": "San Francisco",
}

TEMP_QUESTION_RE = re.compile(r"(?i)(temperature|high temp)")
EVENT_TITLE_RE = re.compile(
    r"(?i)highest temperature in (.+?) on ([A-Za-z]+ \d{1,2})\??"
)
BUCKET_FROM_QUESTION_RE = re.compile(
    r"(?i)be (.+?) on [A-Za-z]+ \d{1,2}"
)


def load_credentials(credentials_path: Path | None = None) -> dict[str, str]:
    """Load Polymarket credentials from env vars or JSON file."""
    env_map = {
        "private_key": "POLY_PRIVATE_KEY",
        "api_key": "POLY_API_KEY",
        "api_secret": "POLY_API_SECRET",
        "api_passphrase": "POLY_API_PASSPHRASE",
        "funder": "POLY_FUNDER",
    }
    creds: dict[str, str] = {}
    for key, env_name in env_map.items():
        value = os.environ.get(env_name)
        if value:
            creds[key] = value

    if len(creds) == len(env_map):
        return creds

    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    if not path.exists():
        missing = [env_map[k] for k in env_map if k not in creds]
        raise FileNotFoundError(
            f"Missing Polymarket credentials. Set env vars {missing} or create {path}"
        )

    with open(path, encoding="utf-8") as handle:
        file_creds = json.load(handle)

    for key in env_map:
        creds.setdefault(key, file_creds.get(key, ""))

    missing_values = [key for key, value in creds.items() if not value]
    if missing_values:
        raise ValueError(f"Polymarket credentials missing keys: {missing_values}")
    return creds


def polymarket_city_to_slug(city_name: str) -> str:
    """Map Polymarket display names to internal city slugs."""
    normalized = city_name.strip().lower()
    slug = CITY_SLUG_MAP.get(normalized)
    if slug:
        return slug
    slug = normalized.replace(" ", "_")
    print(f"WARNING: Unknown Polymarket city '{city_name}' -> slug '{slug}' (no_model)")
    return slug


def _normalize_bucket_label(label: str) -> str:
    text = label.strip()
    text = re.sub(r"\s*°F\b", "°", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*F\b", "°", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)^between\s+", "", text)
    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1° to \2°", text)
    text = text.replace("  ", " ")
    return text


def parse_bucket_label(label: str) -> dict[str, Any]:
    """Parse Polymarket outcome labels into structured bucket info."""
    text = _normalize_bucket_label(label)

    less_than = re.match(r"(?i)(\d+)\s*°?\s*or\s+below", text)
    if less_than:
        return {"type": "LESS_THAN", "lower": None, "upper": int(less_than.group(1))}

    greater_than = re.match(r"(?i)(\d+)\s*°?\s*or\s+above", text)
    if greater_than:
        return {"type": "GREATER_THAN", "lower": int(greater_than.group(1)), "upper": None}

    range_match = re.match(r"(?i)(\d+)\s*°?\s*to\s+(\d+)", text)
    if range_match:
        return {
            "type": "RANGE",
            "lower": int(range_match.group(1)),
            "upper": int(range_match.group(2)),
        }

    raise ValueError(f"Unable to parse bucket label: {label!r}")


def _parse_event_date(month_day: str, year_hint: str | None = None) -> str:
    year = int(year_hint[:4]) if year_hint else date.today().year
    parsed = datetime.strptime(f"{month_day} {year}", "%B %d %Y")
    return parsed.date().isoformat()


def _parse_event_title(title: str, year_hint: str | None = None) -> tuple[str, str] | None:
    match = EVENT_TITLE_RE.search(title)
    if not match:
        return None
    city_display = match.group(1).strip()
    event_date = _parse_event_date(match.group(2), year_hint=year_hint)
    return city_display, event_date


def _extract_bucket_label(market: dict[str, Any]) -> str:
    group_title = market.get("groupItemTitle")
    if group_title:
        return str(group_title)
    question = str(market.get("question", ""))
    match = BUCKET_FROM_QUESTION_RE.search(question)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Unable to extract bucket label from market: {question!r}")


def _terminal_cursor(cursor: str | None) -> bool:
    return not cursor or cursor == "LTE="


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_midpoint(response: Any) -> float | None:
    if isinstance(response, dict):
        return _to_float(response.get("mid"))
    return _to_float(response)


def _book_sides(book: dict[str, Any]) -> tuple[float | None, float | None]:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = _to_float(bids[0]["price"]) if bids else None
    best_ask = _to_float(asks[0]["price"]) if asks else None
    return best_bid, best_ask


def _gamma_market_for_condition(condition_id: str) -> dict[str, Any] | None:
    response = requests.get(
        f"{GAMMA_API}/markets",
        params={"condition_ids": condition_id},
        timeout=20,
    )
    response.raise_for_status()
    markets = response.json()
    if not markets:
        return None
    return markets[0]


def _month_day_label(event_date: str) -> str:
    parsed = date.fromisoformat(event_date)
    return f"{parsed.strftime('%B')} {parsed.day}"


def _ingest_gamma_event(
    event: dict[str, Any],
    grouped: dict[tuple[str, str], dict[str, Any]],
    *,
    event_date: str | None = None,
    active_only: bool = False,
) -> None:
    title = str(event.get("title", ""))
    if not TEMP_QUESTION_RE.search(title):
        return

    year_hint = event.get("eventDate") or event.get("endDate")
    parsed = _parse_event_title(title, year_hint=str(year_hint) if year_hint else None)
    if parsed is None:
        return
    city_display, parsed_date = parsed
    if event_date and parsed_date != event_date:
        return

    if active_only and (event.get("closed") or not event.get("active", True)):
        return

    city_slug = polymarket_city_to_slug(city_display)
    key = (city_slug, parsed_date)
    condition_id = event.get("negRiskMarketID") or str(event.get("id", ""))
    entry = grouped.setdefault(
        key,
        {
            "condition_id": condition_id,
            "city": city_slug,
            "city_display": city_display,
            "event_date": parsed_date,
            "neg_risk": bool(event.get("negRisk", event.get("enableNegRisk", True))),
            "tick_size": "0.01",
            "fee_rate_bps": 0,
            "closed": bool(event.get("closed", False)),
            "accepting_orders": bool(event.get("active", True))
            and not bool(event.get("closed", False)),
            "model_status": "ok" if city_slug in TRAIN_SLUGS else "no_model",
            "buckets": [],
            "question": title,
        },
    )

    for market in event.get("markets", []):
        if active_only and (market.get("closed") or market.get("acceptingOrders") is False):
            continue
        try:
            label = _extract_bucket_label(market)
            parsed_bucket = parse_bucket_label(label)
        except ValueError:
            continue

        token_ids = market.get("clobTokenIds") or []
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        outcomes = market.get("outcomes") or []
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        yes_index = 0
        if outcomes and str(outcomes[0]).lower() != "yes":
            yes_index = 1 if len(token_ids) > 1 else 0
        if not token_ids:
            continue

        tick_size = str(market.get("orderPriceMinTickSize", entry["tick_size"]))
        entry["tick_size"] = tick_size
        token_id = str(token_ids[yes_index])
        if any(bucket["token_id"] == token_id for bucket in entry["buckets"]):
            continue
        entry["buckets"].append(
            {
                "token_id": token_id,
                "condition_id": str(market.get("conditionId", "")),
                "label": label,
                "bucket_type": parsed_bucket["type"],
                "lower_f": parsed_bucket["lower"],
                "upper_f": parsed_bucket["upper"],
                "midpoint": None,
                "best_bid": None,
                "best_ask": None,
                "tick_size": tick_size,
                "neg_risk": bool(market.get("negRisk", entry["neg_risk"])),
                "accepting_orders": bool(market.get("acceptingOrders", True)),
                "closed": bool(market.get("closed", False)),
            }
        )


def _discover_from_gamma_city_queries(
    *,
    event_date: str,
    active_only: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    month_day = _month_day_label(event_date)
    for slug, display_name in CITY_DISPLAY_NAMES.items():
        query = f"Highest temperature in {display_name} on {month_day}"
        response = requests.get(
            f"{GAMMA_API}/public-search",
            params={"q": query},
            timeout=30,
        )
        response.raise_for_status()
        for event in response.json().get("events", []):
            _ingest_gamma_event(
                event,
                grouped,
                event_date=event_date,
                active_only=active_only,
            )
    return grouped


def _discover_from_gamma_search(
    *,
    event_date: str | None = None,
    active_only: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Discover Tmax events via Gamma public search."""
    if event_date:
        return _discover_from_gamma_city_queries(
            event_date=event_date,
            active_only=active_only,
        )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    page = 0
    while page < 200:
        response = requests.get(
            f"{GAMMA_API}/public-search",
            params={"q": "Highest temperature", "page": page},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events", [])
        if not events:
            break

        for event in events:
            _ingest_gamma_event(
                event,
                grouped,
                event_date=event_date,
                active_only=active_only,
            )

        pagination = payload.get("pagination") or {}
        if not pagination.get("hasMore"):
            break
        page += 1

    return grouped


def _discover_from_clob_sampling(
    client: ClobClient,
    *,
    event_date: str | None = None,
    active_only: bool = False,
    max_pages: int = 100,
    seen_condition_ids: set[str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Paginate CLOB simplified sampling markets and enrich via Gamma."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    seen = set(seen_condition_ids or ())
    cursor = "MA=="
    pages = 0
    stale_pages = 0

    while pages < max_pages and not _terminal_cursor(cursor):
        page = client.get_sampling_simplified_markets(next_cursor=cursor)
        pages += 1
        page_hits = 0
        for clob_market in page.get("data", []):
            condition_id = clob_market.get("condition_id")
            if not condition_id or condition_id in seen:
                continue

            gamma_market = _gamma_market_for_condition(condition_id)
            seen.add(condition_id)
            if gamma_market is None:
                continue

            question = str(gamma_market.get("question", ""))
            if not TEMP_QUESTION_RE.search(question):
                continue

            event_title = question
            if "highest temperature in" not in question.lower():
                group_title = gamma_market.get("groupItemTitle")
                if group_title:
                    city_match = re.search(
                        r"(?i)highest temperature in (.+?) on",
                        str(gamma_market.get("description", "")),
                    )
                    if city_match:
                        event_title = (
                            f"Highest temperature in {city_match.group(1)} on "
                            f"{gamma_market.get('endDateIso', '')}"
                        )

            year_hint = gamma_market.get("endDateIso") or gamma_market.get("endDate")
            parsed = _parse_event_title(event_title, year_hint=str(year_hint) if year_hint else None)
            if parsed is None:
                city_match = re.search(r"(?i)in (.+?) be ", question)
                date_match = re.search(r"(?i)on ([A-Za-z]+ \d{1,2})", question)
                if not city_match or not date_match:
                    continue
                city_display = city_match.group(1).strip()
                parsed_date = _parse_event_date(
                    date_match.group(1),
                    year_hint=str(year_hint) if year_hint else None,
                )
            else:
                city_display, parsed_date = parsed

            if event_date and parsed_date != event_date:
                continue

            if active_only and (
                clob_market.get("closed")
                or clob_market.get("accepting_orders") is False
            ):
                continue

            city_slug = polymarket_city_to_slug(city_display)
            key = (city_slug, parsed_date)
            try:
                label = _extract_bucket_label(gamma_market)
                parsed_bucket = parse_bucket_label(label)
            except ValueError:
                continue

            token_ids = gamma_market.get("clobTokenIds") or []
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            outcomes = gamma_market.get("outcomes") or []
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            yes_index = 0
            if outcomes and str(outcomes[0]).lower() != "yes":
                yes_index = 1 if len(token_ids) > 1 else 0
            if not token_ids:
                tokens = clob_market.get("tokens") or []
                yes_tokens = [t for t in tokens if str(t.get("outcome", "")).lower() == "yes"]
                token_id = yes_tokens[0]["token_id"] if yes_tokens else (
                    tokens[0]["token_id"] if tokens else None
                )
            else:
                token_id = token_ids[yes_index]

            if not token_id:
                continue

            entry = grouped.setdefault(
                key,
                {
                    "condition_id": str(
                        gamma_market.get("negRiskMarketID") or condition_id
                    ),
                    "city": city_slug,
                    "city_display": city_display,
                    "event_date": parsed_date,
                    "neg_risk": bool(clob_market.get("neg_risk", gamma_market.get("negRisk", True))),
                    "tick_size": str(gamma_market.get("orderPriceMinTickSize", "0.01")),
                    "fee_rate_bps": 0,
                    "closed": bool(clob_market.get("closed", False)),
                    "accepting_orders": bool(clob_market.get("accepting_orders", True)),
                    "model_status": "ok" if city_slug in TRAIN_SLUGS else "no_model",
                    "buckets": [],
                    "question": f"Highest temperature in {city_display} on {parsed_date}?",
                },
            )

            if any(b["token_id"] == str(token_id) for b in entry["buckets"]):
                continue

            entry["buckets"].append(
                {
                    "token_id": str(token_id),
                    "condition_id": str(condition_id),
                    "label": label,
                    "bucket_type": parsed_bucket["type"],
                    "lower_f": parsed_bucket["lower"],
                    "upper_f": parsed_bucket["upper"],
                    "midpoint": None,
                    "best_bid": None,
                    "best_ask": None,
                    "tick_size": entry["tick_size"],
                    "neg_risk": entry["neg_risk"],
                    "accepting_orders": entry["accepting_orders"],
                    "closed": entry["closed"],
                }
            )
            page_hits += 1

        if page_hits == 0:
            stale_pages += 1
        else:
            stale_pages = 0
        if event_date and grouped and stale_pages >= 5:
            break

        cursor = page.get("next_cursor", "")
        time.sleep(0.02)

    return grouped


def _merge_grouped_markets(
    *sources: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sources:
        for key, market in source.items():
            if key not in merged:
                merged[key] = market
                continue
            existing = merged[key]
            seen = {b["token_id"] for b in existing["buckets"]}
            for bucket in market["buckets"]:
                if bucket["token_id"] not in seen:
                    existing["buckets"].append(bucket)
                    seen.add(bucket["token_id"])
    return list(merged.values())


def _attach_prices(client: ClobClient, markets: list[dict[str, Any]]) -> None:
    for market in markets:
        if market.get("closed"):
            continue
        sample_token = None
        for bucket in market["buckets"]:
            if bucket.get("closed") or bucket.get("accepting_orders") is False:
                continue
            token_id = bucket["token_id"]
            sample_token = sample_token or token_id
            try:
                midpoint = _parse_midpoint(client.get_midpoint(token_id))
                book = client.get_order_book(token_id)
                best_bid, best_ask = _book_sides(book)
                bucket["midpoint"] = midpoint
                bucket["best_bid"] = best_bid
                bucket["best_ask"] = best_ask
                if book.get("tick_size"):
                    bucket["tick_size"] = str(book["tick_size"])
                if book.get("neg_risk") is not None:
                    bucket["neg_risk"] = bool(book["neg_risk"])
            except Exception as exc:
                print(f"  Price fetch failed for {market['city']} {bucket['label']}: {exc}")

        if sample_token:
            try:
                market["fee_rate_bps"] = int(client.get_fee_rate_bps(sample_token))
            except Exception:
                market["fee_rate_bps"] = 0


def _bucket_width(market: dict[str, Any]) -> int | None:
    ranges = [
        b
        for b in market.get("buckets", [])
        if b.get("bucket_type") == "RANGE"
        and b.get("lower_f") is not None
        and b.get("upper_f") is not None
    ]
    if not ranges:
        return None
    widths = [int(r["upper_f"]) - int(r["lower_f"]) + 1 for r in ranges]
    return int(round(sum(widths) / len(widths)))


def discover_tmax_markets(
    client: ClobClient,
    *,
    event_date: str | None = None,
    active_only: bool = False,
    fetch_prices: bool = True,
) -> dict[str, Any]:
    """Discover and aggregate Polymarket Tmax markets."""
    gamma_grouped = _discover_from_gamma_search(
        event_date=event_date,
        active_only=active_only,
    )
    seen_condition_ids: set[str] = set()
    for market in gamma_grouped.values():
        if market.get("condition_id"):
            seen_condition_ids.add(str(market["condition_id"]))
        for bucket in market.get("buckets", []):
            if bucket.get("condition_id"):
                seen_condition_ids.add(str(bucket["condition_id"]))

    clob_grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if not event_date or len(gamma_grouped) < len(TRAIN_SLUGS):
        clob_grouped = _discover_from_clob_sampling(
            client,
            event_date=event_date,
            active_only=active_only,
            seen_condition_ids=seen_condition_ids,
            max_pages=10 if event_date else 100,
        )
    markets = _merge_grouped_markets(gamma_grouped, clob_grouped)
    markets = [market for market in markets if market.get("buckets")]
    markets.sort(key=lambda row: (row["event_date"], row["city"]))

    if fetch_prices:
        _attach_prices(client, markets)

    city_slug_map = {
        market["city_display"]: market["city"] for market in markets
    }
    return {
        "markets": markets,
        "city_slug_map": city_slug_map,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_markets_map(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_markets_map(path: Path | None = None) -> dict[str, Any] | None:
    target = path or DEFAULT_MARKETS_PATH
    if not target.exists():
        return None
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def _markets_fetched_today(payload: dict[str, Any]) -> bool:
    fetched_at = payload.get("fetched_at")
    if not fetched_at:
        return False
    try:
        fetched_date = datetime.fromisoformat(fetched_at).date()
    except ValueError:
        return False
    return fetched_date == date.today()


def _find_market(
    payload: dict[str, Any],
    *,
    condition_id: str | None = None,
    city: str | None = None,
    event_date: str | None = None,
) -> dict[str, Any] | None:
    for market in payload.get("markets", []):
        if condition_id and market.get("condition_id") == condition_id:
            return market
        for bucket in market.get("buckets", []):
            if condition_id and bucket.get("condition_id") == condition_id:
                return market
        if city and event_date:
            if market.get("city") == city and market.get("event_date") == event_date:
                return market
    return None


def _append_order_log(record: dict[str, Any]) -> None:
    ORDER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ORDER_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def build_clob_client(credentials: dict[str, str] | None = None) -> ClobClient:
    creds = credentials or load_credentials()
    return ClobClient(
        host=CLOB_HOST,
        chain_id=CHAIN_ID,
        key=creds["private_key"],
        creds=ApiCreds(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            api_passphrase=creds["api_passphrase"],
        ),
        signature_type=SIGNATURE_TYPE,
        funder=creds["funder"],
    )


class PolymarketClient:
    """API wrapper for Polymarket Tmax trading."""

    def __init__(self, credentials_path: str | None = None):
        path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS_PATH
        self._credentials_path = path
        self._credentials = load_credentials(path if path.exists() else None)
        self.client = build_clob_client(self._credentials)
        self._fee_cache: dict[str, float] = {}
        self._markets_path = DEFAULT_MARKETS_PATH

    def get_balance(self) -> float:
        result = self.client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw = result.get("balance", 0) if isinstance(result, dict) else 0
        return float(raw) / 1_000_000

    def discover_and_cache(
        self,
        *,
        event_date: str | None = None,
        active_only: bool = False,
        fetch_prices: bool = True,
    ) -> dict[str, Any]:
        payload = discover_tmax_markets(
            self.client,
            event_date=event_date,
            active_only=active_only,
            fetch_prices=fetch_prices,
        )
        save_markets_map(self._markets_path, payload)
        return payload

    def fetch_tmax_markets(self, event_date: str) -> list[dict[str, Any]]:
        cached = load_markets_map(self._markets_path)
        if cached is None or not _markets_fetched_today(cached):
            cached = self.discover_and_cache(event_date=event_date, fetch_prices=True)

        markets = [
            market
            for market in cached.get("markets", [])
            if market.get("event_date") == event_date
        ]
        for market in markets:
            self.fetch_bucket_prices(market["condition_id"], market=market)
        return markets

    def fetch_bucket_prices(
        self,
        condition_id: str,
        *,
        market: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if market is None:
            cached = load_markets_map(self._markets_path)
            if cached is None:
                raise KeyError(f"No cached markets map for condition_id={condition_id}")
            market = _find_market(cached, condition_id=condition_id)
            if market is None:
                raise KeyError(f"Unknown Polymarket market: {condition_id}")

        prices: dict[str, float] = {}
        for bucket in market.get("buckets", []):
            token_id = bucket["token_id"]
            midpoint = _parse_midpoint(self.client.get_midpoint(token_id))
            if midpoint is not None:
                bucket["midpoint"] = midpoint
                prices[str(bucket["label"])] = midpoint
        return prices

    def get_fee_rate(self, token_id: str) -> float:
        if token_id in self._fee_cache:
            return self._fee_cache[token_id]
        fee_bps = float(self.client.get_fee_rate_bps(token_id))
        fee_rate = fee_bps / 10_000.0
        self._fee_cache[token_id] = fee_rate
        return fee_rate

    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        *,
        tick_size: str = "0.01",
        neg_risk: bool = True,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        order_side = Side.BUY if side.upper() == "BUY" else Side.SELL
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            side=order_side,
            size=size,
        )
        options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "token_id": token_id,
            "side": side.upper(),
            "price": price,
            "size": size,
            "tick_size": tick_size,
            "neg_risk": neg_risk,
            "dry_run": dry_run,
        }

        try:
            if dry_run:
                signed = self.client.create_order(order_args, options)
                record["status"] = "signed"
                record["signed_order"] = str(signed)
            else:
                response = self.client.create_and_post_order(
                    order_args=order_args,
                    options=options,
                    order_type=OrderType.GTC,
                )
                record["status"] = "posted"
                record["response"] = response
                if isinstance(response, dict):
                    record["order_id"] = response.get("orderID") or response.get("id")
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        _append_order_log(record)
        return record

    def get_open_positions(self) -> list[dict[str, Any]]:
        orders = self.client.get_open_orders()
        simplified: list[dict[str, Any]] = []
        for order in orders:
            if isinstance(order, dict):
                simplified.append(
                    {
                        "token_id": order.get("asset_id") or order.get("token_id"),
                        "side": order.get("side"),
                        "price": _to_float(order.get("price")),
                        "size": _to_float(order.get("original_size") or order.get("size")),
                        "order_id": order.get("id") or order.get("orderID"),
                    }
                )
        return simplified

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self.client.cancel_order(OrderPayload(orderID=order_id))
