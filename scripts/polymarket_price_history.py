#!/usr/bin/env python3
"""Investigate Polymarket historical price/trade data sources for Tmax backtesting.

Run with the project venv so authenticated CLOB calls work:
  .venv/bin/python scripts/polymarket_price_history.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore

import matplotlib.pyplot as plt
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from polymarket_recon import STATION_REGISTRY, save_csv, save_json  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "polymarket_history"
RAW_DIR = OUTPUT_DIR / "raw_responses"
SECTION2B_RAW_DIR = RAW_DIR / "section2b"
CREDENTIALS_PATH = PROJECT_ROOT / "config" / "polymarket_credentials.json"
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKB_MODEL_DIR = PROJECT_ROOT / "models" / "trackb"
EVENTS_PATH = PROJECT_ROOT / "data" / "polymarket_recon" / "all_events.json"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

TIER1_CITIES = frozenset(
    {
        "austin",
        "chicago_midway",
        "houston",
        "los_angeles",
        "new_york_city",
        "san_francisco",
    }
)

FIVE_MIN = 300
END_CURSOR = "LTE="
MAX_TRADE_PAGES = 5
MAX_AUTH_TRADE_PAGES = 10
AUTH_BUILD_SLEEP = 0.5
MAX_ACTIVITY_OFFSET = 10_000
VIABILITY_LOOKBACK_DAYS = 7

CLOB_HOST = CLOB_API
CLOB_CHAIN_ID = 137
CLOB_SIGNATURE_TYPE = 3

CREDENTIALS_TEMPLATE = """{
    "private_key": "0x...",
    "api_key": "...",
    "api_secret": "...",
    "api_passphrase": "...",
    "funder": "0x..."
}
"""

_session: requests.Session | None = None


def _normalize_bucket_label(label: str) -> str:
    text = label.strip()
    text = re.sub(r"\s*°F\b", "°", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*F\b", "°", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)^between\s+", "", text)
    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1° to \2°", text)
    return text.replace("  ", " ")


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MCP_Project/polymarket_price_history (research)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = make_session()
    return _session


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


def _log_response_debug(response: requests.Response, label: str) -> None:
    body = response.text[:500]
    print(f"  DEBUG [{label}]: status={response.status_code} body={body!r}")


def throttled_get(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    label: str = "",
    accept_statuses: frozenset[int] | None = None,
) -> tuple[requests.Response | None, Any | None]:
    """GET with rate limiting and 429 exponential backoff."""
    backoff = 5.0
    params = params or {}
    for attempt in range(4):
        try:
            response = get_session().get(url, params=params, timeout=45)
            time.sleep(0.35)
            if response.status_code == 429:
                print(f"  Rate limited ({label}); sleeping {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if accept_statuses and response.status_code in accept_statuses:
                try:
                    return response, response.json()
                except json.JSONDecodeError:
                    return response, None
            if not response.ok:
                _log_response_debug(response, label)
                return response, None
            try:
                return response, response.json()
            except json.JSONDecodeError:
                _log_response_debug(response, label)
                return response, None
        except requests.RequestException as exc:
            print(f"  Request error ({label}): {exc}")
            time.sleep(backoff)
            backoff *= 2
    return None, None


def _to_int_ts(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _time_span_days(earliest: int | None, latest: int | None) -> float | None:
    if earliest is None or latest is None:
        return None
    return (latest - earliest) / 86400.0


def _city_timezone(city: str) -> ZoneInfo:
    cfg = STATION_REGISTRY.get(city, {})
    tz_name = cfg.get("timezone", "UTC")
    return ZoneInfo(tz_name)


def event_window_timestamps(city: str, event_date_str: str) -> tuple[int, int]:
    """Return (start, end) unix seconds: 7 days before event through end of event day local."""
    event_date = date.fromisoformat(event_date_str)
    tz = _city_timezone(city)
    start_dt = datetime.combine(event_date - timedelta(days=7), datetime.min.time(), tzinfo=tz)
    end_dt = datetime.combine(event_date + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def load_events() -> list[dict[str, Any]]:
    with open(EVENTS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def pick_modal_market(event: dict[str, Any]) -> dict[str, Any] | None:
    markets = [m for m in event.get("markets") or [] if m.get("yes_token_id")]
    if not markets:
        return None
    return max(markets, key=lambda m: m.get("volume") or 0)


def _pick_distinct(events: list[dict[str, Any]], n: int, used_cities: set[str]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda e: -(e.get("total_volume") or 0)):
        city = event.get("city")
        if not city or city in used_cities:
            continue
        if not pick_modal_market(event):
            continue
        used_cities.add(city)
        picked.append(event)
        if len(picked) >= n:
            break
    return picked


def build_test_market_record(event: dict[str, Any], category: str) -> dict[str, Any]:
    market = pick_modal_market(event)
    if not market:
        raise ValueError(f"No modal market for event {event.get('event_id')}")
    return {
        "category": category,
        "event_id": event.get("event_id"),
        "event_slug": event.get("event_slug"),
        "condition_id": market.get("conditionId"),
        "city": event.get("city"),
        "event_date": event.get("event_date"),
        "yes_token_id": market.get("yes_token_id"),
        "question": market.get("question"),
        "bucket_label": market.get("bucket_label"),
        "bucket_volume": market.get("volume"),
    }


def section1_pick_test_markets() -> list[dict[str, Any]]:
    today = date.today()
    tomorrow = today + timedelta(days=1)
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    old_cutoff = today - timedelta(days=35)

    events = load_events()
    tier1 = [e for e in events if e.get("city") in TIER1_CITIES]

    active_pool = [
        e
        for e in tier1
        if not e.get("closed")
        and e.get("accepting_orders")
        and e.get("event_date") in {today.isoformat(), tomorrow.isoformat()}
    ]
    recent_pool = [
        e
        for e in tier1
        if e.get("closed")
        and e.get("event_date") in {yesterday.isoformat(), two_days_ago.isoformat()}
    ]
    old_pool = [
        e
        for e in tier1
        if e.get("closed") and e.get("event_date") and e["event_date"] <= old_cutoff.isoformat()
    ]

    used: set[str] = set()
    picks: list[dict[str, Any]] = []
    for category, pool, n in [
        ("ACTIVE", active_pool, 2),
        ("RECENTLY_CLOSED", recent_pool, 2),
        ("OLD_CLOSED", old_pool, 2),
    ]:
        for event in _pick_distinct(pool, n, used):
            picks.append(build_test_market_record(event, category))

    print(f"\nSelected {len(picks)} test markets:")
    for idx, rec in enumerate(picks, 1):
        print(
            f"  [{idx}] {rec['category']} | {rec['city']} {rec['event_date']} | "
            f"event_id={rec['event_id']} | condition_id={rec['condition_id']} | "
            f"yes_token_id={rec['yes_token_id'][:24]}... | {rec['question']}"
        )

    save_json(OUTPUT_DIR / "test_markets.json", picks)
    return picks


def _normalize_trade_row(row: dict[str, Any]) -> dict[str, Any] | None:
    ts = _to_int_ts(row.get("timestamp") or row.get("match_time") or row.get("t"))
    price = _to_float(row.get("price") or row.get("p"))
    size = _to_float(row.get("size"))
    if ts is None or price is None:
        return None
    return {
        "timestamp": ts,
        "price": price,
        "size": size or 0.0,
        "side": row.get("side"),
        "asset": row.get("asset") or row.get("asset_id"),
    }


def _extract_trades_payload(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [t for t in payload if isinstance(t, dict)]
    if isinstance(payload, dict):
        for key in ("data", "trades", "history"):
            if isinstance(payload.get(key), list):
                return [t for t in payload[key] if isinstance(t, dict)]
    return []


def _summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [_normalize_trade_row(t) for t in trades]
    normalized = [t for t in normalized if t]
    if not normalized:
        return {
            "n_trades": 0,
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "time_span_days": None,
            "sample_trades": [],
        }
    timestamps = [t["timestamp"] for t in normalized]
    earliest = min(timestamps)
    latest = max(timestamps)
    return {
        "n_trades": len(normalized),
        "earliest_timestamp": earliest,
        "latest_timestamp": latest,
        "time_span_days": _time_span_days(earliest, latest),
        "sample_trades": normalized[:3],
    }


def _paginate_clob_trades(
    base_params: dict[str, Any],
    *,
    label: str,
    save_raw: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    all_trades: list[dict[str, Any]] = []
    cursor = "MA=="
    raw_first: dict[str, Any] | None = None

    for page in range(MAX_TRADE_PAGES):
        params = {**base_params, "limit": 500, "next_cursor": cursor}
        response, payload = throttled_get(f"{CLOB_API}/trades", params, label=f"{label} p{page+1}")
        if save_raw and page == 0:
            raw_first = {
                "status_code": response.status_code if response else None,
                "params": params,
                "payload": payload,
                "body": response.text[:500] if response is not None else None,
            }
        if payload is None:
            break
        batch = _extract_trades_payload(payload)
        all_trades.extend(batch)
        if isinstance(payload, dict):
            cursor = str(payload.get("next_cursor") or END_CURSOR)
            if cursor == END_CURSOR or not batch:
                break
        else:
            break

    return all_trades, raw_first


def section2_test_clob_trades(test_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, market in enumerate(test_markets, 1):
        print(f"Section 2: Testing {market['city']} {market['event_date']}... [{idx}/{len(test_markets)}]")
        entry: dict[str, Any] = {
            "category": market["category"],
            "city": market["city"],
            "event_date": market["event_date"],
            "event_id": market["event_id"],
            "yes_token_id": market["yes_token_id"],
            "condition_id": market["condition_id"],
        }

        trades, raw_first = _paginate_clob_trades(
            {"asset_id": market["yes_token_id"]},
            label=f"clob_trades asset {market['city']}",
            save_raw=(idx == 1),
        )
        endpoint_used = "asset_id"
        if not trades:
            trades, raw_market = _paginate_clob_trades(
                {"market": market["condition_id"]},
                label=f"clob_trades market {market['city']}",
            )
            endpoint_used = "market"
            if idx == 1 and raw_first is None:
                raw_first = raw_market

        if idx == 1 and raw_first:
            save_json(RAW_DIR / "clob_trades_sample.json", raw_first)

        summary = _summarize_trades(trades)
        entry.update(summary)
        entry["endpoint_used"] = endpoint_used
        entry["rate_limit_issues"] = False
        results.append(entry)

        if summary["sample_trades"]:
            print("  Sample trades:")
            for t in summary["sample_trades"]:
                print(f"    ts={t['timestamp']} price={t['price']} size={t['size']} side={t['side']}")

    save_json(OUTPUT_DIR / "clob_trades_test.json", results)
    return results


def load_polymarket_credentials() -> dict[str, str] | None:
    """Load credentials from POLYMARKET_* / POLY_* env vars or JSON file."""
    env_groups = [
        {
            "private_key": "POLYMARKET_PRIVATE_KEY",
            "api_key": "POLYMARKET_API_KEY",
            "api_secret": "POLYMARKET_API_SECRET",
            "api_passphrase": "POLYMARKET_API_PASSPHRASE",
            "funder": "POLYMARKET_FUNDER",
        },
        {
            "private_key": "POLY_PRIVATE_KEY",
            "api_key": "POLY_API_KEY",
            "api_secret": "POLY_API_SECRET",
            "api_passphrase": "POLY_API_PASSPHRASE",
            "funder": "POLY_FUNDER",
        },
    ]
    creds: dict[str, str] = {}
    for env_map in env_groups:
        creds = {key: os.environ.get(env_name, "") for key, env_name in env_map.items()}
        if all(creds.values()):
            return creds
        creds = {}

    if CREDENTIALS_PATH.exists():
        with open(CREDENTIALS_PATH, encoding="utf-8") as handle:
            file_creds = json.load(handle)
        for key in ("private_key", "api_key", "api_secret", "api_passphrase", "funder"):
            creds[key] = str(file_creds.get(key, "")).strip()
        if all(creds.values()):
            return creds

    return None


def write_credentials_template() -> None:
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CREDENTIALS_PATH.exists():
        CREDENTIALS_PATH.write_text(CREDENTIALS_TEMPLATE, encoding="utf-8")
    print(f"Created credentials template at {CREDENTIALS_PATH}")
    print("Fill in private_key, api_key, api_secret, api_passphrase, and funder, then re-run.")


def _sign_clob_request(api_secret: str, method: str, path: str, body: str = "") -> tuple[str, str]:
    timestamp = str(int(time.time()))
    message = timestamp + method.upper() + path + body
    signature = hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return timestamp, signature


def _auth_headers(creds: dict[str, str], method: str, path: str, body: str = "") -> dict[str, str]:
    timestamp, signature = _sign_clob_request(creds["api_secret"], method, path, body)
    return {
        "POLY-API-KEY": creds["api_key"],
        "POLY-TIMESTAMP": timestamp,
        "POLY-SIGNATURE": signature,
        "POLY-PASSPHRASE": creds["api_passphrase"],
    }


def try_build_sdk_client(creds: dict[str, str]) -> tuple[Any | None, str | None]:
    try:
        from py_clob_client_v2 import ApiCreds, ClobClient
    except ImportError:
        return None, (
            "py_clob_client_v2 not installed. Install with:\n"
            "  pip install 'git+https://github.com/Polymarket/py-clob-client-v2.git@1.0.1rc1'"
        )

    client = ClobClient(
        host=CLOB_HOST,
        chain_id=CLOB_CHAIN_ID,
        key=creds["private_key"],
        signature_type=CLOB_SIGNATURE_TYPE,
        funder=creds["funder"],
        creds=ApiCreds(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            api_passphrase=creds["api_passphrase"],
        ),
    )
    return client, None


def verify_auth(creds: dict[str, str], client: Any | None) -> tuple[bool, str]:
    if client is not None:
        try:
            client.get_api_keys()
            return True, "Auth OK"
        except Exception as exc:
            return False, f"SDK auth failed: {exc}"

    for path in ("/auth/api-keys", "/data/trades"):
        headers = _auth_headers(creds, "GET", path)
        params: dict[str, Any] = {"limit": 1}
        if path == "/data/trades":
            params["maker_address"] = creds["funder"]
        try:
            response = get_session().get(
                f"{CLOB_HOST}{path}",
                headers=headers,
                params=params,
                timeout=30,
            )
            time.sleep(0.35)
            if response.ok:
                return True, "Auth OK"
        except requests.RequestException as exc:
            return False, f"Raw auth check failed on {path}: {exc}"
    return False, "Auth verification failed for SDK and raw HTTP"


def authenticated_clob_get(
    creds: dict[str, str],
    path: str,
    params: dict[str, Any] | None = None,
    *,
    label: str = "",
) -> tuple[requests.Response | None, Any | None]:
    params = params or {}
    headers = _auth_headers(creds, "GET", path)
    backoff = 5.0
    for attempt in range(4):
        try:
            response = get_session().get(
                f"{CLOB_HOST}{path}",
                headers=headers,
                params=params,
                timeout=45,
            )
            time.sleep(0.35)
            if response.status_code == 429:
                print(f"  Rate limited ({label}); sleeping {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if not response.ok:
                _log_response_debug(response, label)
                return response, None
            try:
                return response, response.json()
            except json.JSONDecodeError:
                _log_response_debug(response, label)
                return response, None
        except requests.RequestException as exc:
            print(f"  Auth request error ({label}): {exc}")
            time.sleep(backoff)
            backoff *= 2
    return None, None


def _sdk_fetch_trades(
    client: Any,
    *,
    yes_token_id: str,
    condition_id: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    raw_first: dict[str, Any] | None = None
    if not hasattr(client, "get_trades_paginated"):
        return [], "sdk_none", None

    try:
        from py_clob_client_v2.clob_types import TradeParams
    except ImportError:
        return [], "sdk_none", None

    all_trades: list[dict[str, Any]] = []
    for label, params in [
        ("sdk_asset_id", TradeParams(asset_id=yes_token_id)),
        ("sdk_market", TradeParams(market=condition_id)),
        ("sdk_asset_and_market", TradeParams(asset_id=yes_token_id, market=condition_id)),
    ]:
        cursor = None
        for page in range(MAX_AUTH_TRADE_PAGES):
            try:
                payload = client.get_trades_paginated(params=params, next_cursor=cursor)
            except Exception as exc:
                raw_first = {"method": "get_trades_paginated", "params": str(params), "error": str(exc)}
                break
            if raw_first is None and page == 0:
                raw_first = {"method": "get_trades_paginated", "params": str(params), "payload": payload}

            batch = payload.get("trades") if isinstance(payload, dict) else []
            batch = batch or []
            all_trades.extend(batch)
            cursor = payload.get("next_cursor") if isinstance(payload, dict) else END_CURSOR
            if not batch or cursor == END_CURSOR:
                break
        if all_trades:
            return all_trades, label, raw_first

    if raw_first is None:
        raw_first = {
            "method": "get_trades_paginated",
            "note": "Authenticated CLOB /trades returns only this wallet's fills, not global market tape.",
            "payload": {"trades": [], "next_cursor": END_CURSOR, "count": 0},
        }
    return [], "sdk_user_trades_empty", raw_first


def _paginate_auth_clob_trades(
    creds: dict[str, str],
    base_params: dict[str, Any],
    *,
    label: str,
    client: Any | None = None,
    yes_token_id: str | None = None,
    condition_id: str | None = None,
    save_raw: bool = False,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    if client is not None and yes_token_id and condition_id:
        trades, endpoint, raw_first = _sdk_fetch_trades(
            client,
            yes_token_id=yes_token_id,
            condition_id=condition_id,
        )
        if trades:
            return trades, endpoint, raw_first
        if raw_first and endpoint == "sdk_user_trades_empty":
            return [], endpoint, raw_first

    all_trades: list[dict[str, Any]] = []
    raw_first: dict[str, Any] | None = None
    endpoint_used = "auth_http_asset_id"

    for path, param_list in [
        ("/trades", [base_params]),
        (
            "/data/trades",
            [
                {**base_params, "maker_address": creds["funder"]},
                {"market": base_params.get("market"), "maker_address": creds["funder"]},
            ],
        ),
    ]:
        for params in param_list:
            if not any(params.get(k) for k in ("asset_id", "market")):
                continue
            cursor = "MA=="
            endpoint_used = (
                "auth_http_asset_id"
                if params.get("asset_id")
                else "auth_http_market"
            )
            if path == "/data/trades":
                endpoint_used = f"auth_data_trades_{endpoint_used.split('_')[-1]}"

            for page in range(MAX_AUTH_TRADE_PAGES):
                page_params = {**params, "limit": 500, "next_cursor": cursor}
                response, payload = authenticated_clob_get(
                    creds,
                    path,
                    page_params,
                    label=f"{label} {path} p{page+1}",
                )
                if save_raw and page == 0 and raw_first is None:
                    raw_first = {
                        "path": path,
                        "status_code": response.status_code if response else None,
                        "params": page_params,
                        "payload": payload,
                        "body": response.text[:500] if response is not None else None,
                    }
                if payload is None:
                    break
                batch = _extract_trades_payload(payload)
                all_trades.extend(batch)
                if isinstance(payload, dict):
                    cursor = str(payload.get("next_cursor") or END_CURSOR)
                    if cursor == END_CURSOR or not batch:
                        break
                else:
                    break
            if all_trades:
                return all_trades, endpoint_used, raw_first

    if not base_params.get("market") and condition_id:
        return _paginate_auth_clob_trades(
            creds,
            {"market": condition_id},
            label=f"{label} fallback_market",
            client=None,
            save_raw=save_raw,
        )

    return all_trades, endpoint_used, raw_first


def section2b_authenticated_trades(
    test_markets: list[dict[str, Any]],
    unauth_results: list[dict[str, Any]],
) -> dict[str, Any]:
    SECTION2B_RAW_DIR.mkdir(parents=True, exist_ok=True)
    creds = load_polymarket_credentials()
    if creds is None:
        write_credentials_template()
        return {
            "auth_ok": False,
            "error": "credentials_missing",
            "results": [],
            "deep_history": False,
        }

    client, sdk_error = try_build_sdk_client(creds)
    if sdk_error:
        print(sdk_error)

    auth_ok, auth_msg = verify_auth(creds, client)
    print(auth_msg if auth_ok else f"Auth FAILED: {auth_msg}")
    if not auth_ok:
        save_json(
            SECTION2B_RAW_DIR / "auth_failure.json",
            {"auth_ok": False, "message": auth_msg, "sdk_error": sdk_error},
        )
        return {
            "auth_ok": False,
            "error": auth_msg,
            "results": [],
            "deep_history": False,
        }

    unauth_by_event = {str(r.get("event_id")): r for r in unauth_results}
    results: list[dict[str, Any]] = []

    for idx, market in enumerate(test_markets, 1):
        print(
            f"Section 2b: Authenticated trades {market['city']} {market['event_date']}... "
            f"[{idx}/{len(test_markets)}]"
        )
        trades, endpoint_used, raw_first = _paginate_auth_clob_trades(
            creds,
            {"asset_id": market["yes_token_id"]},
            label=f"auth trades {market['city']}",
            client=client,
            yes_token_id=market["yes_token_id"],
            condition_id=market["condition_id"],
            save_raw=(idx == 1),
        )
        if not trades and client is None:
            trades, endpoint_used, raw_market = _paginate_auth_clob_trades(
                creds,
                {"market": market["condition_id"]},
                label=f"auth trades market {market['city']}",
                client=None,
                yes_token_id=market["yes_token_id"],
                condition_id=market["condition_id"],
            )
            if idx == 1 and raw_first is None:
                raw_first = raw_market

        summary = _summarize_trades(trades)
        unauth = unauth_by_event.get(str(market.get("event_id")), {})
        entry = {
            "category": market["category"],
            "city": market["city"],
            "event_date": market["event_date"],
            "event_id": market["event_id"],
            "yes_token_id": market["yes_token_id"],
            "condition_id": market["condition_id"],
            **summary,
            "endpoint_used": endpoint_used,
            "trade_scope": "authenticated_wallet_only",
            "unauth_n_trades": unauth.get("n_trades", 0),
            "unauth_status": "401 Unauthorized" if unauth.get("n_trades", 0) == 0 else "ok",
        }
        results.append(entry)

        if idx == 1 and raw_first:
            save_json(SECTION2B_RAW_DIR / "authenticated_trades_sample.json", raw_first)

        if summary["sample_trades"]:
            print("  Sample trades:")
            for t in summary["sample_trades"]:
                print(f"    ts={t['timestamp']} price={t['price']} size={t['size']} side={t['side']}")
        else:
            print("  No trades returned.")

    save_json(OUTPUT_DIR / "authenticated_trades_test.json", results)
    closed_spans = [
        r["time_span_days"]
        for r in results
        if r.get("category") in ("RECENTLY_CLOSED", "OLD_CLOSED") and r.get("time_span_days")
    ]
    max_closed = max(closed_spans) if closed_spans else 0.0
    deep_history = max_closed > VIABILITY_LOOKBACK_DAYS
    return {
        "auth_ok": True,
        "results": results,
        "max_closed_lookback_days": max_closed,
        "deep_history": deep_history,
        "creds": creds,
        "client": client,
    }


def evaluate_auth_decision_gate(auth_section: dict[str, Any]) -> dict[str, Any]:
    if not auth_section.get("auth_ok"):
        print("\nAUTH DID NOT UNLOCK DEEPER HISTORY")
        print(f"Reason: {auth_section.get('error', 'authentication failed')}")
        print(
            "Recommendation: Proceed with paper trading validation. Historical backtest on "
            "Polymarket is not feasible with public/auth APIs. The Track-B model is "
            "exchange-agnostic; Kalshi backtest remains the primary validation source."
        )
        return {"deep_history": False, "max_lookback_days": 0.0}

    max_lookback = float(auth_section.get("max_closed_lookback_days") or 0.0)
    if auth_section.get("deep_history"):
        print("\nDEEP HISTORY AVAILABLE - proceeding to build dataset")
        print(f"Max closed-market lookback from authenticated /trades: {max_lookback:.2f} days")
        return {"deep_history": True, "max_lookback_days": max_lookback}

    print("\nAUTH DID NOT UNLOCK DEEPER HISTORY")
    print(f"Max lookback achieved on closed markets: {max_lookback:.2f} days")
    print(
        "Recommendation: Proceed with paper trading validation. Historical backtest on "
        "Polymarket is not feasible with public/auth APIs. The Track-B model is "
        "exchange-agnostic; Kalshi backtest remains the primary validation source."
    )
    return {"deep_history": False, "max_lookback_days": max_lookback}


def _event_buckets_df(event: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for market in event.get("markets") or []:
        bucket_type = market.get("bucket_type")
        bounds = market.get("bucket_bounds") or [None, None]
        if not bucket_type or not market.get("bucket_label"):
            continue
        lower, upper = bounds if len(bounds) == 2 else (None, None)
        rows.append(
            {
                "bucket_label": market["bucket_label"],
                "bucket_type": bucket_type,
                "bucket_lower_inclusive_f": lower,
                "bucket_upper_inclusive_f": upper,
            }
        )
    return pd.DataFrame(rows)


def _load_trackb_forecast(city: str, event_date: str) -> tuple[float, float] | None:
    features_path = PROJECT_ROOT / "data" / "trackb" / city / "features.parquet"
    model_dir = TRACKB_MODEL_DIR / city
    if not features_path.exists() or not model_dir.exists():
        return None
    try:
        import joblib
        import numpy as np
    except ImportError:
        return None

    features = pd.read_parquet(features_path)
    features["_date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")
    row = features[features["_date"].eq(event_date)]
    if row.empty:
        return None

    with open(model_dir / "feature_cols.json", encoding="utf-8") as handle:
        feature_cols = json.load(handle)
    values = row.iloc[0]
    if any(col not in values.index or pd.isna(values[col]) for col in feature_cols):
        return None

    models = [
        joblib.load(model_dir / "ridge.joblib"),
        joblib.load(model_dir / "huber.joblib"),
        joblib.load(model_dir / "lightgbm.joblib"),
    ]
    x = values[feature_cols].values.astype(float).reshape(1, -1)
    preds = [model.predict(x)[0] for model in models]
    tmax_pred = float(np.mean(preds))

    if CITY_CONFIG_PATH.exists():
        city_config = json.loads(CITY_CONFIG_PATH.read_text(encoding="utf-8"))
        sigma = float(city_config.get(city, {}).get("trackb_sigma_f", 0))
    else:
        sigma = 0.0
    if sigma <= 0:
        return None
    return tmax_pred, sigma


def _market_price_for_bucket(
    city: str,
    event_date: str,
    yes_token_id: str,
) -> float | None:
    start_ts, end_ts = event_window_timestamps(city, event_date)
    _, payload = throttled_get(
        f"{CLOB_API}/prices-history",
        {
            "market": yes_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": 60,
        },
        label=f"sanity price {city}",
    )
    history = (payload or {}).get("history") if isinstance(payload, dict) else []
    if history:
        return _to_float(history[-1].get("p"))
    return None


def shallow_model_sanity_check(test_markets: list[dict[str, Any]]) -> dict[str, Any]:
    print("\nRunning quick model-vs-market sanity check on closed test markets...")
    try:
        from models.track_j import bucket_probs_from_point_forecast
    except ImportError:
        try:
            from src.models.track_j import bucket_probs_from_point_forecast
        except ImportError as exc:
            print(f"Skipping sanity check: could not import Track-B helpers ({exc})")
            return {"skipped": True, "reason": str(exc)}

    events = {str(e["event_id"]): e for e in load_events()}
    comparisons: list[dict[str, Any]] = []

    for market in test_markets:
        if market.get("category") not in ("RECENTLY_CLOSED", "OLD_CLOSED"):
            continue
        event = events.get(str(market["event_id"]))
        if not event:
            continue
        forecast = _load_trackb_forecast(market["city"], market["event_date"])
        if forecast is None:
            print(f"  {market['city']} {market['event_date']}: no Track-B forecast available")
            continue
        tmax_pred, sigma = forecast
        buckets = _event_buckets_df(event)
        if buckets.empty:
            continue
        try:
            probs = bucket_probs_from_point_forecast(tmax_pred, sigma, buckets)
        except ValueError as exc:
            print(f"  {market['city']} {market['event_date']}: bucket prob error ({exc})")
            continue

        for mkt in event.get("markets") or []:
            label = mkt.get("bucket_label")
            yes_token = mkt.get("yes_token_id")
            if not label or not yes_token or label not in probs:
                continue
            market_price = _market_price_for_bucket(market["city"], market["event_date"], yes_token)
            if market_price is None:
                continue
            comparisons.append(
                {
                    "city": market["city"],
                    "event_date": market["event_date"],
                    "bucket_label": label,
                    "model_prob": probs[label],
                    "market_price": market_price,
                }
            )

    if len(comparisons) < 2:
        print("Insufficient model/market pairs for correlation.")
        result = {
            "skipped": True,
            "reason": "insufficient_pairs",
            "n_pairs": len(comparisons),
        }
        save_json(OUTPUT_DIR / "model_vs_market_sanity.json", result)
        return result

    frame = pd.DataFrame(comparisons)
    corr = float(frame["model_prob"].corr(frame["market_price"]))
    mae = float((frame["model_prob"] - frame["market_price"]).abs().mean())
    bias = float((frame["model_prob"] - frame["market_price"]).mean())
    print(f"Model-vs-market correlation: {corr:.3f}")
    print(f"Mean absolute error: {mae:.3f}")
    print(f"Signed bias (model - market): {bias:+.3f}")

    result = {
        "n_pairs": len(frame),
        "correlation": corr,
        "mean_abs_error": mae,
        "signed_bias": bias,
        "pairs_sample": comparisons[:20],
    }
    save_json(OUTPUT_DIR / "model_vs_market_sanity.json", result)
    return result


def section6_build_authenticated_dataset(auth_section: dict[str, Any]) -> dict[str, Any] | None:
    creds = auth_section.get("creds")
    client = auth_section.get("client")
    if not creds:
        return None

    events = [
        e
        for e in load_events()
        if e.get("city") in TIER1_CITIES and e.get("closed") and e.get("event_date")
    ]
    progress = _load_progress()
    completed = set(progress.get("completed") or [])

    city_frames: dict[str, list[pd.DataFrame]] = {city: [] for city in TIER1_CITIES}
    coverage_rows: list[dict[str, Any]] = []
    total_buckets = sum(len(e.get("markets") or []) for e in events)
    done = 0

    for event in sorted(events, key=lambda e: (e.get("city", ""), e.get("event_date", ""))):
        city = event["city"]
        event_date = event["event_date"]
        start_ts, end_ts = event_window_timestamps(city, event_date)
        event_frames: list[pd.DataFrame] = []

        for market in event.get("markets") or []:
            done += 1
            bucket_label = market.get("bucket_label") or market.get("question")
            yes_token_id = market.get("yes_token_id")
            condition_id = market.get("conditionId")
            if not yes_token_id or not condition_id or not bucket_label:
                continue

            key = f"auth|{city}|{event_date}|{bucket_label}"
            print(
                f"Section 6 (auth): fetching {city} {event_date} bucket {bucket_label} "
                f"[{done}/{total_buckets}]"
            )
            if key in completed:
                continue

            trades, _, _ = _paginate_auth_clob_trades(
                creds,
                {"asset_id": yes_token_id},
                label=f"auth build {city}",
                client=client,
                yes_token_id=yes_token_id,
                condition_id=condition_id,
            )
            time.sleep(AUTH_BUILD_SLEEP)

            bucket_df = trades_to_ohlcv(trades, start_ts, end_ts)
            if bucket_df.empty:
                coverage_rows.append(
                    {
                        "city": city,
                        "event_date": event_date,
                        "n_buckets": len(event.get("markets") or []),
                        "n_snapshots": 0,
                        "first_snapshot": None,
                        "last_snapshot": None,
                        "source": "authenticated_clob_trades",
                        "bucket_label": bucket_label,
                    }
                )
                completed.add(key)
                progress["completed"] = sorted(completed)
                _save_progress(progress)
                continue

            bucket_df["event_date"] = event_date
            bucket_df["bucket_label"] = bucket_label
            event_frames.append(
                bucket_df[
                    [
                        "event_date",
                        "bucket_label",
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "n_trades",
                    ]
                ]
            )
            city_frames[city].append(event_frames[-1])
            completed.add(key)
            progress["completed"] = sorted(completed)
            _save_progress(progress)

        if event_frames:
            event_df = pd.concat(event_frames, ignore_index=True)
            coverage_rows.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "n_buckets": event_df["bucket_label"].nunique(),
                    "n_snapshots": len(event_df),
                    "first_snapshot": int(event_df["timestamp"].min()),
                    "last_snapshot": int(event_df["timestamp"].max()),
                    "source": "authenticated_clob_trades",
                    "bucket_label": None,
                }
            )

    saved_any = False
    for city in sorted(TIER1_CITIES):
        parts = city_frames.get(city) or []
        if not parts:
            continue
        city_df = pd.concat(parts, ignore_index=True)
        out_path = OUTPUT_DIR / city / "prices.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        city_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(city_df)} rows -> {out_path}")
        saved_any = True

    if coverage_rows:
        save_csv(pd.DataFrame(coverage_rows), OUTPUT_DIR / "coverage_summary.csv")

    if not saved_any:
        print("WARNING: Authenticated build produced no parquet data.")
        return {"built": False, "source": "authenticated_clob_trades"}

    return {
        "built": True,
        "source": "authenticated_clob_trades",
        "n_events": len(events),
    }


def _paginate_data_activity(
    params: dict[str, Any],
    *,
    label: str,
    yes_token_id: str | None = None,
    save_raw: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    all_rows: list[dict[str, Any]] = []
    offset = 0
    limit = 500
    raw_first: dict[str, Any] | None = None

    while offset <= MAX_ACTIVITY_OFFSET:
        page_params = {**params, "limit": limit, "offset": offset}
        response, payload = throttled_get(f"{DATA_API}/activity", page_params, label=f"{label} o{offset}")
        if save_raw and offset == 0:
            raw_first = {
                "status_code": response.status_code if response else None,
                "params": page_params,
                "payload": payload,
                "body": response.text[:500] if response is not None else None,
            }
        if payload is None:
            break
        batch = payload if isinstance(payload, list) else _extract_trades_payload(payload)
        if yes_token_id:
            batch = [r for r in batch if str(r.get("asset", "")) == str(yes_token_id)]
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    return all_rows, raw_first


def section3_test_data_activity(test_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, market in enumerate(test_markets, 1):
        print(f"Section 3: Testing {market['city']} {market['event_date']}... [{idx}/{len(test_markets)}]")
        base_params = {
            "market": market["condition_id"],
            "type": "TRADE",
            "sortBy": "TIMESTAMP",
            "sortDirection": "ASC",
        }
        trades, raw_first = _paginate_data_activity(
            base_params,
            label=f"activity full {market['city']}",
            yes_token_id=market["yes_token_id"],
            save_raw=(idx == 1),
        )

        start_ts, end_ts = event_window_timestamps(market["city"], market["event_date"])
        windowed, _ = _paginate_data_activity(
            {**base_params, "start": start_ts, "end": end_ts},
            label=f"activity window {market['city']}",
            yes_token_id=market["yes_token_id"],
        )

        if idx == 1 and raw_first:
            save_json(RAW_DIR / "data_api_activity_sample.json", raw_first)

        full_summary = _summarize_trades(trades)
        window_summary = _summarize_trades(windowed)
        entry = {
            "category": market["category"],
            "city": market["city"],
            "event_date": market["event_date"],
            "condition_id": market["condition_id"],
            "yes_token_id": market["yes_token_id"],
            "full_query": full_summary,
            "windowed_query": {
                **window_summary,
                "start": start_ts,
                "end": end_ts,
            },
            "rate_limit_issues": False,
        }
        results.append(entry)

        sample = window_summary["sample_trades"] or full_summary["sample_trades"]
        if sample:
            print("  Sample trades:")
            for t in sample:
                print(f"    ts={t['timestamp']} price={t['price']} size={t['size']} side={t['side']}")

    save_json(OUTPUT_DIR / "data_api_activity_test.json", results)
    return results


def _paginate_data_trades(
    condition_id: str,
    *,
    yes_token_id: str,
    label: str,
    save_raw: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    all_rows: list[dict[str, Any]] = []
    offset = 0
    limit = 500
    raw_first: dict[str, Any] | None = None

    while offset <= MAX_ACTIVITY_OFFSET:
        params = {
            "market": condition_id,
            "limit": limit,
            "offset": offset,
            "takerOnly": "false",
        }
        response, payload = throttled_get(f"{DATA_API}/trades", params, label=f"{label} o{offset}")
        if save_raw and offset == 0:
            raw_first = {
                "status_code": response.status_code if response else None,
                "params": params,
                "payload": payload,
                "body": response.text[:500] if response is not None else None,
            }
        if payload is None:
            break
        batch = payload if isinstance(payload, list) else _extract_trades_payload(payload)
        batch = [r for r in batch if str(r.get("asset", "")) == str(yes_token_id)]
        all_rows.extend(batch)
        raw_batch = payload if isinstance(payload, list) else _extract_trades_payload(payload)
        if len(raw_batch) < limit:
            break
        offset += limit

    return all_rows, raw_first


def section3b_test_data_trades(test_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Probe public Data API /trades (market filter, no user required)."""
    results: list[dict[str, Any]] = []
    for idx, market in enumerate(test_markets, 1):
        print(f"Section 3b: Testing Data API /trades {market['city']} {market['event_date']}... [{idx}/{len(test_markets)}]")
        trades, raw_first = _paginate_data_trades(
            market["condition_id"],
            yes_token_id=market["yes_token_id"],
            label=f"data trades {market['city']}",
            save_raw=(idx == 1),
        )
        if idx == 1 and raw_first:
            save_json(RAW_DIR / "data_api_trades_sample.json", raw_first)

        summary = _summarize_trades(trades)
        entry = {
            "category": market["category"],
            "city": market["city"],
            "event_date": market["event_date"],
            "condition_id": market["condition_id"],
            "yes_token_id": market["yes_token_id"],
            **summary,
            "rate_limit_issues": False,
        }
        results.append(entry)
        if summary["sample_trades"]:
            print("  Sample trades:")
            for t in summary["sample_trades"]:
                print(f"    ts={t['timestamp']} price={t['price']} size={t['size']} side={t['side']}")

    save_json(OUTPUT_DIR / "data_api_trades_test.json", results)
    return results


def section4_test_prices_history(test_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, market in enumerate(test_markets, 1):
        print(f"Section 4: Testing {market['city']} {market['event_date']}... [{idx}/{len(test_markets)}]")
        start_ts, end_ts = event_window_timestamps(market["city"], market["event_date"])
        entry: dict[str, Any] = {
            "category": market["category"],
            "city": market["city"],
            "event_date": market["event_date"],
            "yes_token_id": market["yes_token_id"],
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelities_tested": [],
        }

        for fidelity in (60, 5):
            params = {
                "market": market["yes_token_id"],
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": fidelity,
            }
            response, payload = throttled_get(
                f"{CLOB_API}/prices-history",
                params,
                label=f"prices-history f{fidelity} {market['city']}",
            )
            if idx == 1 and fidelity == 60 and payload is not None:
                save_json(
                    RAW_DIR / "prices_history_explicit_sample.json",
                    {
                        "status_code": response.status_code if response else None,
                        "params": params,
                        "payload": payload,
                    },
                )

            history = []
            if isinstance(payload, dict):
                history = payload.get("history") or []
            points = [h for h in history if isinstance(h, dict) and "t" in h]
            n_points = len(points)
            earliest = min((int(p["t"]) for p in points), default=None)
            latest = max((int(p["t"]) for p in points), default=None)
            fidelity_result = {
                "fidelity": fidelity,
                "n_points": n_points,
                "earliest_t": earliest,
                "latest_t": latest,
                "time_span_days": _time_span_days(earliest, latest),
            }
            entry["fidelities_tested"].append(fidelity_result)
            if fidelity == 60:
                entry.update(
                    {
                        "n_points": n_points,
                        "earliest_t": earliest,
                        "latest_t": latest,
                        "time_span_days": _time_span_days(earliest, latest),
                    }
                )
            if n_points == 0:
                break

        results.append(entry)

    save_json(OUTPUT_DIR / "prices_history_explicit_test.json", results)
    return results


def _category_lookback(rows: list[dict[str, Any]], category: str, field: str = "time_span_days") -> float | None:
    values = [
        r.get(field)
        for r in rows
        if r.get("category") == category and r.get(field) is not None
    ]
    return max(values) if values else None


def section5_compare_sources(
    clob_trades: list[dict[str, Any]],
    data_activity: list[dict[str, Any]],
    data_trades: list[dict[str, Any]],
    prices_history: list[dict[str, Any]],
) -> dict[str, Any]:
    def closed_lookbacks(rows: list[dict[str, Any]], field: str) -> float | None:
        recent = _category_lookback(rows, "RECENTLY_CLOSED", field)
        old = _category_lookback(rows, "OLD_CLOSED", field)
        if recent is None and old is None:
            return None
        return max(v for v in (recent, old) if v is not None)

    activity_rows = []
    for row in data_activity:
        window = row.get("windowed_query") or {}
        full = row.get("full_query") or {}
        span = window.get("time_span_days") or full.get("time_span_days")
        activity_rows.append({**row, "time_span_days": span, "n_trades": window.get("n_trades") or full.get("n_trades")})

    comparison = [
        {
            "source": "clob_trades",
            "active_lookback": _category_lookback(clob_trades, "ACTIVE"),
            "closed_lookback": closed_lookbacks(clob_trades, "time_span_days"),
            "old_closed_lookback": _category_lookback(clob_trades, "OLD_CLOSED"),
            "resolution": "trade-level",
            "pagination": "next_cursor (max 5 pages in test)",
            "rate_limit_issues": any(r.get("rate_limit_issues") for r in clob_trades),
        },
        {
            "source": "data_api_activity",
            "active_lookback": _category_lookback(activity_rows, "ACTIVE"),
            "closed_lookback": closed_lookbacks(activity_rows, "time_span_days"),
            "old_closed_lookback": _category_lookback(activity_rows, "OLD_CLOSED"),
            "resolution": "trade-level",
            "pagination": "offset/limit (max offset 10000); requires user param",
            "rate_limit_issues": any(r.get("rate_limit_issues") for r in data_activity),
        },
        {
            "source": "data_api_trades",
            "active_lookback": _category_lookback(data_trades, "ACTIVE"),
            "closed_lookback": closed_lookbacks(data_trades, "time_span_days"),
            "old_closed_lookback": _category_lookback(data_trades, "OLD_CLOSED"),
            "resolution": "trade-level",
            "pagination": "offset/limit (max offset 10000)",
            "rate_limit_issues": any(r.get("rate_limit_issues") for r in data_trades),
        },
        {
            "source": "clob_prices_history",
            "active_lookback": _category_lookback(prices_history, "ACTIVE"),
            "closed_lookback": closed_lookbacks(prices_history, "time_span_days"),
            "old_closed_lookback": _category_lookback(prices_history, "OLD_CLOSED"),
            "resolution": "5-60 min snapshots",
            "pagination": "single request per fidelity",
            "rate_limit_issues": False,
        },
    ]

    print("\nSource comparison:")
    header = (
        f"{'source':<22} {'active':>8} {'closed':>8} {'old_closed':>11} "
        f"{'resolution':<18} {'pagination':<28} {'rate_limit':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in comparison:
        print(
            f"{row['source']:<22} "
            f"{(row['active_lookback'] or 0):>8.2f} "
            f"{(row['closed_lookback'] or 0):>8.2f} "
            f"{(row['old_closed_lookback'] or 0):>11.2f} "
            f"{row['resolution']:<18} "
            f"{row['pagination']:<28} "
            f"{str(row['rate_limit_issues']):>10}"
        )

    def score(row: dict[str, Any]) -> tuple[float, float]:
        closed_depth = row.get("closed_lookback") or 0
        old_depth = row.get("old_closed_lookback") or 0
        res_bonus = 1.0 if "trade" in row.get("resolution", "") else 0.0
        return (max(closed_depth, old_depth), res_bonus)

    ranked = sorted(comparison, key=score, reverse=True)
    recommendation = ranked[0]["source"]
    print(f"\nRecommended source for backtest dataset: {recommendation}")

    closed_spans = []
    for source_rows in (clob_trades, activity_rows, data_trades, prices_history):
        for cat in ("RECENTLY_CLOSED", "OLD_CLOSED"):
            span = _category_lookback(source_rows, cat)
            if span is not None:
                closed_spans.append(span)
    viable = any(s >= VIABILITY_LOOKBACK_DAYS for s in closed_spans)
    if not viable:
        print(
            f"\nWARNING: No source returned >={VIABILITY_LOOKBACK_DAYS} days on closed markets. "
            "Skipping Sections 6 and 7."
        )
    else:
        print(
            f"\nViability gate PASSED: at least one closed-market test has "
            f">={VIABILITY_LOOKBACK_DAYS} days of history."
        )

    payload = {
        "comparison": comparison,
        "recommendation": recommendation,
        "viable_for_build": viable,
    }
    save_json(OUTPUT_DIR / "source_comparison.json", payload)
    return payload


def trades_to_ohlcv(trades: list[dict[str, Any]], start_ts: int, end_ts: int) -> pd.DataFrame:
    """Convert trade list to 5-minute OHLCV snapshots with forward-filled close."""
    normalized = [_normalize_trade_row(t) for t in trades]
    normalized = [t for t in normalized if t]
    if not normalized:
        return pd.DataFrame(columns=["timestamp", "yes_price", "volume", "open", "high", "low", "close"])

    frame = pd.DataFrame(normalized).sort_values("timestamp")
    grid_start = (start_ts // FIVE_MIN) * FIVE_MIN
    grid_end = ((end_ts // FIVE_MIN) + 1) * FIVE_MIN
    grid = list(range(grid_start, grid_end + 1, FIVE_MIN))

    rows: list[dict[str, Any]] = []
    last_close: float | None = None
    trade_idx = 0
    n_trades = len(frame)

    for bucket_start in grid:
        bucket_end = bucket_start + FIVE_MIN
        bucket_trades: list[dict[str, Any]] = []
        while trade_idx < n_trades and frame.iloc[trade_idx]["timestamp"] < bucket_end:
            ts = int(frame.iloc[trade_idx]["timestamp"])
            if ts >= bucket_start:
                bucket_trades.append(frame.iloc[trade_idx].to_dict())
            trade_idx += 1

        if bucket_trades:
            prices = [t["price"] for t in bucket_trades]
            vol = sum(t["size"] for t in bucket_trades)
            close = prices[-1]
            last_close = close
            rows.append(
                {
                    "timestamp": bucket_start,
                    "yes_price": close,
                    "volume": vol,
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": close,
                    "n_trades": len(bucket_trades),
                }
            )
        elif last_close is not None:
            rows.append(
                {
                    "timestamp": bucket_start,
                    "yes_price": last_close,
                    "volume": 0.0,
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "n_trades": 0,
                }
            )

    return pd.DataFrame(rows)


def prices_history_to_ohlcv(history: list[dict[str, Any]], start_ts: int, end_ts: int) -> pd.DataFrame:
    """Resample price history points to 5-minute grid with forward fill."""
    points = []
    for item in history:
        ts = _to_int_ts(item.get("t"))
        price = _to_float(item.get("p"))
        if ts is not None and price is not None:
            points.append({"timestamp": ts, "yes_price": price})
    if not points:
        return pd.DataFrame(columns=["timestamp", "yes_price", "volume"])

    frame = pd.DataFrame(points).sort_values("timestamp")
    grid_start = (start_ts // FIVE_MIN) * FIVE_MIN
    grid_end = ((end_ts // FIVE_MIN) + 1) * FIVE_MIN
    grid = list(range(grid_start, grid_end + 1, FIVE_MIN))

    rows: list[dict[str, Any]] = []
    last_price: float | None = None
    idx = 0
    n = len(frame)

    for bucket_start in grid:
        bucket_end = bucket_start + FIVE_MIN
        bucket_price: float | None = None
        while idx < n and frame.iloc[idx]["timestamp"] < bucket_end:
            if frame.iloc[idx]["timestamp"] >= bucket_start:
                bucket_price = float(frame.iloc[idx]["yes_price"])
            idx += 1
        if bucket_price is not None:
            last_price = bucket_price
            rows.append({"timestamp": bucket_start, "yes_price": bucket_price, "volume": 0.0})
        elif last_price is not None:
            rows.append({"timestamp": bucket_start, "yes_price": last_price, "volume": 0.0})

    return pd.DataFrame(rows)


def _load_progress() -> dict[str, Any]:
    path = OUTPUT_DIR / "fetch_progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": []}


def _save_progress(progress: dict[str, Any]) -> None:
    save_json(OUTPUT_DIR / "fetch_progress.json", progress)


def _fetch_trades_for_bucket(
    source: str,
    *,
    condition_id: str,
    yes_token_id: str,
    city: str,
    start_ts: int,
    end_ts: int,
) -> list[dict[str, Any]]:
    all_trades: list[dict[str, Any]] = []
    chunk_seconds = 2 * 86400
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + chunk_seconds, end_ts)
        if source == "data_api_activity":
            params = {
                "market": condition_id,
                "type": "TRADE",
                "sortBy": "TIMESTAMP",
                "sortDirection": "ASC",
                "start": chunk_start,
                "end": chunk_end,
            }
            batch, _ = _paginate_data_activity(
                params,
                label=f"build activity {city}",
                yes_token_id=yes_token_id,
            )
            all_trades.extend(batch)
        elif source == "data_api_trades":
            offset = 0
            while offset <= MAX_ACTIVITY_OFFSET:
                params = {
                    "market": condition_id,
                    "limit": 500,
                    "offset": offset,
                }
                _, payload = throttled_get(f"{DATA_API}/trades", params, label=f"build trades {city}")
                if payload is None:
                    break
                batch = payload if isinstance(payload, list) else _extract_trades_payload(payload)
                batch = [r for r in batch if str(r.get("asset", "")) == str(yes_token_id)]
                batch = [
                    r
                    for r in batch
                    if start_ts <= (_to_int_ts(r.get("timestamp")) or 0) <= end_ts
                ]
                all_trades.extend(batch)
                if len(batch) < 500:
                    break
                offset += 500
        elif source == "clob_trades":
            for param_key, value in (("asset_id", yes_token_id), ("market", condition_id)):
                trades, _ = _paginate_clob_trades(
                    {param_key: value, "after": str(chunk_start), "before": str(chunk_end)},
                    label=f"build clob {city}",
                )
                if trades:
                    all_trades.extend(trades)
                    break
        chunk_start = chunk_end
    return all_trades


def section6_build_dataset(source_comparison: dict[str, Any]) -> dict[str, Any] | None:
    if not source_comparison.get("viable_for_build"):
        return None

    recommendation = source_comparison.get("recommendation", "data_api_activity")
    build_source = recommendation
    if recommendation == "clob_prices_history":
        fetch_mode = "prices"
        source_label = "clob_prices_history"
    elif recommendation == "clob_trades":
        fetch_mode = "trades"
        source_label = "clob_trades"
    else:
        fetch_mode = "trades"
        source_label = "data_api_activity"
        # If activity returned nothing in tests, fall back to data_api /trades for build.
        activity_test = OUTPUT_DIR / "data_api_activity_test.json"
        trades_test = OUTPUT_DIR / "data_api_trades_test.json"
        if activity_test.exists():
            activity_rows = json.loads(activity_test.read_text(encoding="utf-8"))
            if all((r.get("n_trades") or 0) == 0 for r in activity_rows) and trades_test.exists():
                trades_rows = json.loads(trades_test.read_text(encoding="utf-8"))
                if any((r.get("n_trades") or 0) > 0 for r in trades_rows):
                    build_source = "data_api_trades"
                    source_label = "data_api_trades"

    events = [
        e
        for e in load_events()
        if e.get("city") in TIER1_CITIES and e.get("closed") and e.get("event_date")
    ]
    progress = _load_progress()
    completed = set(progress.get("completed") or [])

    all_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    total_buckets = sum(len(e.get("markets") or []) for e in events)
    done = 0

    for event in sorted(events, key=lambda e: (e.get("city", ""), e.get("event_date", ""))):
        city = event["city"]
        event_date = event["event_date"]
        start_ts, end_ts = event_window_timestamps(city, event_date)
        event_frames: list[pd.DataFrame] = []

        for market in event.get("markets") or []:
            done += 1
            bucket_label = market.get("bucket_label") or market.get("question")
            yes_token_id = market.get("yes_token_id")
            condition_id = market.get("conditionId")
            if not yes_token_id or not condition_id or not bucket_label:
                continue

            key = f"{city}|{event_date}|{bucket_label}"
            print(f"Section 6: fetching {city} {event_date} bucket {bucket_label} [{done}/{total_buckets}]")
            if key in completed:
                parquet_path = OUTPUT_DIR / city / f"{event_date}_{market.get('market_id', 'bucket')}.parquet"
                if parquet_path.exists():
                    event_frames.append(pd.read_parquet(parquet_path))
                continue

            if fetch_mode == "prices":
                params = {
                    "market": yes_token_id,
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "fidelity": 5,
                }
                _, payload = throttled_get(
                    f"{CLOB_API}/prices-history",
                    params,
                    label=f"build prices {city}",
                )
                history = (payload or {}).get("history") if isinstance(payload, dict) else []
                bucket_df = prices_history_to_ohlcv(history or [], start_ts, end_ts)
            else:
                trades = _fetch_trades_for_bucket(
                    build_source,
                    condition_id=condition_id,
                    yes_token_id=yes_token_id,
                    city=city,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                bucket_df = trades_to_ohlcv(trades, start_ts, end_ts)

            if bucket_df.empty:
                coverage_rows.append(
                    {
                        "city": city,
                        "event_date": event_date,
                        "n_buckets": len(event.get("markets") or []),
                        "n_snapshots": 0,
                        "first_snapshot": None,
                        "last_snapshot": None,
                        "source": source_label,
                        "bucket_label": bucket_label,
                    }
                )
                completed.add(key)
                continue

            bucket_df["city"] = city
            bucket_df["event_date"] = event_date
            bucket_df["bucket_label"] = bucket_label
            bucket_df["source"] = source_label
            event_frames.append(bucket_df)

            out_path = OUTPUT_DIR / city / f"{event_date}_{market.get('market_id', 'bucket')}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bucket_df.to_parquet(out_path, index=False)
            completed.add(key)
            progress["completed"] = sorted(completed)
            _save_progress(progress)

        if event_frames:
            event_df = pd.concat(event_frames, ignore_index=True)
            all_frames.append(event_df)
            coverage_rows.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "n_buckets": event_df["bucket_label"].nunique(),
                    "n_snapshots": len(event_df),
                    "first_snapshot": int(event_df["timestamp"].min()),
                    "last_snapshot": int(event_df["timestamp"].max()),
                    "source": source_label,
                    "bucket_label": None,
                }
            )

    if not all_frames:
        print("WARNING: Section 6 produced no snapshot data.")
        return {"built": False, "source": source_label}

    full_df = pd.concat(all_frames, ignore_index=True)
    for city in sorted(full_df["city"].unique()):
        city_df = full_df[full_df["city"] == city].copy()
        city_path = OUTPUT_DIR / city / "snapshots.parquet"
        city_path.parent.mkdir(parents=True, exist_ok=True)
        city_df.to_parquet(city_path, index=False)
        print(f"  Saved {len(city_df)} rows -> {city_path}")

    coverage_df = pd.DataFrame(coverage_rows)
    save_csv(coverage_df, OUTPUT_DIR / "coverage_summary.csv")
    return {"built": True, "source": source_label, "n_rows": len(full_df), "n_events": len(events)}


def section7_compare_kalshi(build_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not build_result or not build_result.get("built"):
        return None

    poly_frames = []
    for city_dir in sorted(OUTPUT_DIR.iterdir()):
        if not city_dir.is_dir() or city_dir.name not in TIER1_CITIES:
            continue
        snap_path = city_dir / "snapshots.parquet"
        if snap_path.exists():
            poly_frames.append(pd.read_parquet(snap_path))

    if not poly_frames:
        print("WARNING: No Polymarket parquet files found for comparison.")
        return None

    poly = pd.concat(poly_frames, ignore_index=True)
    poly["event_date"] = pd.to_datetime(poly["event_date"]).dt.strftime("%Y-%m-%d")
    poly["bucket_norm"] = poly["bucket_label"].astype(str).map(_normalize_bucket_label)
    poly["snap_time"] = pd.to_datetime(poly["timestamp"], unit="s", utc=True)

    kalshi_parts = []
    for name in ("threshold_opt.parquet", "time_holdout.parquet"):
        path = SPLIT_DIR / name
        if path.exists():
            kalshi_parts.append(pd.read_parquet(path))
    if not kalshi_parts:
        print("WARNING: Kalshi split parquets not found; skipping comparison.")
        return None

    kalshi = pd.concat(kalshi_parts, ignore_index=True)
    kalshi["event_date"] = pd.to_datetime(kalshi["event_date"]).dt.strftime("%Y-%m-%d")
    kalshi["city_key"] = kalshi["source_city_folder"].astype(str)
    kalshi["bucket_norm"] = kalshi["bucket_label"].astype(str).map(_normalize_bucket_label)
    kalshi["snap_time"] = pd.to_datetime(kalshi["snapshot_time_local"], utc=True)

    poly_pairs = set(zip(poly["city"], poly["event_date"]))
    kalshi_pairs = set(zip(kalshi["city_key"], kalshi["event_date"]))
    overlap_pairs = sorted(poly_pairs & kalshi_pairs)
    print(f"\nOverlapping (city, event_date) pairs: {len(overlap_pairs)}")

    if not overlap_pairs:
        result = {
            "n_overlap_pairs": 0,
            "n_aligned_points": 0,
            "correlation": None,
            "mean_abs_diff": None,
            "signed_bias": None,
            "message": "No overlapping city-dates between Polymarket and Kalshi datasets.",
        }
        save_json(OUTPUT_DIR / "poly_vs_kalshi.json", result)
        print(result["message"])
        return result

    aligned_parts: list[pd.DataFrame] = []
    for city, event_date in overlap_pairs:
        p = poly[(poly["city"] == city) & (poly["event_date"] == event_date)].copy()
        k = kalshi[(kalshi["city_key"] == city) & (kalshi["event_date"] == event_date)].copy()
        if p.empty or k.empty:
            continue
        for bucket in p["bucket_norm"].unique():
            p_bucket = p[p["bucket_norm"] == bucket].sort_values("snap_time")
            k_bucket = k[k["bucket_norm"] == bucket].sort_values("snap_time")
            if p_bucket.empty or k_bucket.empty:
                continue
            merged = pd.merge_asof(
                p_bucket,
                k_bucket[["snap_time", "yes_mid_close"]],
                on="snap_time",
                direction="nearest",
                tolerance=pd.Timedelta("5min"),
            )
            merged = merged.dropna(subset=["yes_mid_close", "yes_price"])
            if not merged.empty:
                aligned_parts.append(merged)

    if not aligned_parts:
        result = {
            "n_overlap_pairs": len(overlap_pairs),
            "n_aligned_points": 0,
            "correlation": None,
            "mean_abs_diff": None,
            "signed_bias": None,
            "message": "Overlap pairs found but no bucket-timestamp alignment within 5 minutes.",
        }
        save_json(OUTPUT_DIR / "poly_vs_kalshi.json", result)
        print(result["message"])
        return result

    aligned = pd.concat(aligned_parts, ignore_index=True)
    corr = float(aligned["yes_price"].corr(aligned["yes_mid_close"]))
    mad = float((aligned["yes_price"] - aligned["yes_mid_close"]).abs().mean())
    bias = float((aligned["yes_price"] - aligned["yes_mid_close"]).mean())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "poly_vs_kalshi_scatter.png"
    plt.figure(figsize=(7, 7))
    plt.scatter(aligned["yes_mid_close"], aligned["yes_price"], alpha=0.25, s=8)
    plt.xlabel("Kalshi yes_mid_close")
    plt.ylabel("Polymarket yes_price")
    plt.title("Polymarket vs Kalshi prices (aligned bucket-snapshots)")
    lims = [0, 1]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    result = {
        "n_overlap_pairs": len(overlap_pairs),
        "overlap_pairs_sample": overlap_pairs[:20],
        "n_aligned_points": int(len(aligned)),
        "correlation": corr,
        "mean_abs_diff": mad,
        "signed_bias": bias,
        "figure_path": str(fig_path),
    }
    save_json(OUTPUT_DIR / "poly_vs_kalshi.json", result)

    print("\nPolymarket vs Kalshi summary:")
    print(f"  Overlap pairs: {len(overlap_pairs)}")
    print(f"  Aligned points: {len(aligned)}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Mean absolute difference: {mad:.4f}")
    print(f"  Signed bias (poly - kalshi): {bias:+.4f}")
    print(f"  Scatter plot: {fig_path}")
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SECTION2B_RAW_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    test_markets = run_section(
        "Pick test markets",
        section1_pick_test_markets,
        results,
        "section1",
    ) or []

    clob_trades = run_section(
        "Test CLOB /trades",
        lambda: section2_test_clob_trades(test_markets),
        results,
        "section2",
    ) or []

    auth_section = run_section(
        "Authenticated CLOB /trades (Section 2b)",
        lambda: section2b_authenticated_trades(test_markets, clob_trades),
        results,
        "section2b",
    ) or {"auth_ok": False, "deep_history": False}

    auth_gate = evaluate_auth_decision_gate(auth_section)
    results["auth_decision_gate"] = auth_gate

    if auth_gate.get("deep_history"):
        run_section(
            "Build authenticated historical dataset",
            lambda: section6_build_authenticated_dataset(auth_section),
            results,
            "section6_auth",
        )
    else:
        run_section(
            "Model-vs-market sanity check",
            lambda: shallow_model_sanity_check(test_markets),
            results,
            "sanity_check",
        )

    data_activity = run_section(
        "Test Data API /activity",
        lambda: section3_test_data_activity(test_markets),
        results,
        "section3",
    ) or []

    data_trades = run_section(
        "Test Data API /trades",
        lambda: section3b_test_data_trades(test_markets),
        results,
        "section3b",
    ) or []

    prices_history = run_section(
        "Test CLOB /prices-history explicit window",
        lambda: section4_test_prices_history(test_markets),
        results,
        "section4",
    ) or []

    source_comparison = run_section(
        "Determine best data source",
        lambda: section5_compare_sources(clob_trades, data_activity, data_trades, prices_history),
        results,
        "section5",
    ) or {"viable_for_build": False}

    build_result = run_section(
        "Build historical dataset",
        lambda: section6_build_dataset(source_comparison),
        results,
        "section6",
    )

    run_section(
        "Compare Polymarket vs Kalshi",
        lambda: section7_compare_kalshi(build_result if isinstance(build_result, dict) else None),
        results,
        "section7",
    )

    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
