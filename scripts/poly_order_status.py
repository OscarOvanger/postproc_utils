#!/usr/bin/env python3
"""Print Polymarket balance, open orders, fills, and ask-to-bid spread."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import (  # noqa: E402
    DEFAULT_MARKETS_PATH,
    EVENT_TITLE_RE,
    GAMMA_API,
    ORDER_LOG_PATH,
    PolymarketClient,
    _parse_event_date,
    fetch_order_book_http,
    load_markets_map,
)
from src.poly_trading_pipeline import (  # noqa: E402
    WEATHER_TAG_ID,
    _build_http_session,
    _extract_bucket_label,
    _parse_json_field,
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_orders(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("data", "orders"):
            if key in payload and isinstance(payload[key], list):
                return [row for row in payload[key] if isinstance(row, dict)]
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _order_id(order: dict[str, Any]) -> str | None:
    value = order.get("id") or order.get("orderID")
    return str(value) if value else None


def _token_id(order: dict[str, Any]) -> str | None:
    value = order.get("asset_id") or order.get("token_id")
    return str(value) if value else None


def load_posted_orders(log_path: Path) -> dict[str, dict[str, Any]]:
    """Return latest posted order record per order_id from poly_orders.jsonl."""
    posted: dict[str, dict[str, Any]] = {}
    if not log_path.exists():
        return posted
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "posted":
                continue
            order_id = record.get("order_id")
            if not order_id:
                response = record.get("response")
                if isinstance(response, dict):
                    order_id = response.get("orderID") or response.get("id")
            if not order_id:
                continue
            posted[str(order_id)] = record
    return posted


def fetch_gamma_token_labels(event_date: str) -> dict[str, str]:
    """Build token_id -> 'City bucket' from Gamma weather events for one date."""
    labels: dict[str, str] = {}
    session = _build_http_session()
    offset = 0
    while True:
        params = {
            "tag_id": WEATHER_TAG_ID,
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": offset,
        }
        response = session.get(f"{GAMMA_API}/events", params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        for event in batch:
            title = str(event.get("title", ""))
            match = EVENT_TITLE_RE.search(title)
            if not match:
                continue
            year_hint = event.get("eventDate") or event.get("endDate")
            try:
                parsed_date = _parse_event_date(
                    match.group(2),
                    year_hint=str(year_hint) if year_hint else None,
                )
            except ValueError:
                continue
            if parsed_date != event_date:
                continue
            city = match.group(1).strip()
            for market in event.get("markets") or []:
                token_ids = _parse_json_field(market.get("clobTokenIds"))
                outcomes = _parse_json_field(market.get("outcomes"))
                if not token_ids:
                    continue
                yes_index = 0
                if outcomes and str(outcomes[0]).lower() != "yes":
                    yes_index = 1 if len(token_ids) > 1 else 0
                token = str(token_ids[yes_index])
                try:
                    bucket = _extract_bucket_label(market)
                except Exception:
                    bucket = str(market.get("question", ""))[:24]
                labels[token] = f"{city} {bucket}".strip()
        if len(batch) < 100:
            break
        offset += 100
    return labels


def load_token_labels(
    *,
    event_date: str,
    token_ids: set[str],
    refresh: bool,
) -> dict[str, str]:
    """Map token_id -> 'City bucket' label."""
    labels: dict[str, str] = {}
    cached = load_markets_map(DEFAULT_MARKETS_PATH)
    if cached:
        for market in cached.get("markets", []):
            city = str(market.get("city_display") or market.get("city") or "")
            for bucket in market.get("buckets", []):
                token = str(bucket.get("token_id", ""))
                if token:
                    labels[token] = f"{city} {bucket.get('label', '')}".strip()

    missing = {token for token in token_ids if token and token not in labels}
    if missing and refresh:
        try:
            labels.update(fetch_gamma_token_labels(event_date))
        except Exception as exc:
            print(f"WARNING: could not fetch Gamma labels: {exc}", file=sys.stderr)

    for token in token_ids:
        if token and token not in labels:
            labels[token] = f"token {token[:12]}..."
    return labels


def fetch_open_orders(client: PolymarketClient) -> list[dict[str, Any]]:
    clob = client.client
    if hasattr(clob, "get_open_orders"):
        return _normalize_orders(clob.get_open_orders())
    if hasattr(clob, "get_orders"):
        return _normalize_orders(clob.get_orders())
    raise RuntimeError("CLOB client has no get_open_orders() or get_orders()")


def format_cents(spread: float | None) -> str:
    if spread is None:
        return "N/A"
    return f"{spread * 100:.1f}c"


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Polymarket balance, open orders, fills, and ask-bid spread."
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Event date for token label lookup (default: today)",
    )
    parser.add_argument(
        "--no-fetch-labels",
        action="store_true",
        help="Skip Gamma lookup for human-readable city/bucket names",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Only show fills for posted orders in the last N days (default: 7)",
    )
    args = parser.parse_args()

    client = PolymarketClient()
    posted_by_id = load_posted_orders(ORDER_LOG_PATH)
    open_orders = fetch_open_orders(client)
    open_ids = {_order_id(order) for order in open_orders}
    open_ids.discard(None)

    token_ids: set[str] = set()
    for order in open_orders:
        token = _token_id(order)
        if token:
            token_ids.add(token)
    for record in posted_by_id.values():
        token = record.get("token_id")
        if token:
            token_ids.add(str(token))

    labels = load_token_labels(
        event_date=args.date,
        token_ids=token_ids,
        refresh=not args.no_fetch_labels,
    )

    print_header("Balance")
    try:
        pusd = client.get_balance()
        print(f"pUSD (cash): ${pusd:.2f}")
    except Exception as exc:
        print(f"Could not fetch balance: {exc}")

    holdings: list[tuple[str, float]] = []
    for token in sorted(token_ids):
        try:
            shares = client.get_conditional_balance(token)
        except Exception:
            shares = 0.0
        if shares > 0:
            holdings.append((labels.get(token, token[:12]), shares))
    if holdings:
        print("YES shares held:")
        for label, shares in holdings:
            print(f"  {label}: {shares:.2f}")
    else:
        print("YES shares held: none")

    print_header(f"Open orders ({len(open_orders)})")
    if not open_orders:
        print("  (none)")
    else:
        print(
            f"{'Label':<28} {'Side':<5} {'Our bid':>8} {'Best ask':>9} "
            f"{'Spread':>8} {'Size':>6} {'Matched':>8}"
        )
        print("-" * 82)
        for order in sorted(open_orders, key=lambda row: labels.get(_token_id(row) or "", "")):
            token = _token_id(order) or "?"
            label = labels.get(token, token[:18])
            side = str(order.get("side", "?"))
            our_price = _to_float(order.get("price"))
            original = _to_float(order.get("original_size") or order.get("size")) or 0.0
            matched = _to_float(order.get("size_matched")) or 0.0

            best_bid, best_ask = None, None
            spread = None
            if side.upper() == "BUY" and token != "?":
                best_bid, best_ask = fetch_order_book_http(token)
                time.sleep(0.1)
                if our_price is not None and best_ask is not None:
                    spread = best_ask - our_price

            our_s = f"${our_price:.2f}" if our_price is not None else "N/A"
            ask_s = f"${best_ask:.2f}" if best_ask is not None else "N/A"
            print(
                f"{label:<28} {side:<5} {our_s:>8} {ask_s:>9} "
                f"{format_cents(spread):>8} {original:>6.1f} {matched:>8.1f}"
            )
            if side.upper() == "BUY" and best_bid is not None:
                print(f"  best bid: ${best_bid:.2f}")

    cutoff = datetime.now().timestamp() - args.days * 86400
    recent_posted = {
        oid: record
        for oid, record in posted_by_id.items()
        if _parse_log_timestamp(record.get("timestamp")) >= cutoff
    }

    filled_rows: list[dict[str, Any]] = []
    for order_id, record in sorted(
        recent_posted.items(),
        key=lambda item: item[1].get("timestamp", ""),
        reverse=True,
    ):
        token = str(record.get("token_id", ""))
        status = client.get_order_status(order_id, token_id=token or None)
        state = status.get("status", "unknown")
        matched = _to_float(status.get("size_matched")) or 0.0
        original = _to_float(status.get("original_size")) or _to_float(record.get("size")) or 0.0

        if state == "open" and order_id in open_ids:
            continue
        if state == "open" and matched <= 0:
            continue
        if state not in {"filled", "partial"} and matched <= 0:
            continue

        filled_rows.append(
            {
                "order_id": order_id,
                "label": labels.get(token, token[:18]),
                "side": str(record.get("side", "?")),
                "price": _to_float(record.get("price")),
                "fill_price": _to_float(status.get("fill_price")),
                "matched": matched,
                "original": original,
                "status": state,
                "timestamp": record.get("timestamp", ""),
            }
        )

    print_header(f"Filled / partial ({len(filled_rows)} in last {args.days}d)")
    if not filled_rows:
        print("  (none)")
    else:
        for row in filled_rows:
            price = row["fill_price"] if row["fill_price"] is not None else row["price"]
            price_s = f"${price:.2f}" if price is not None else "N/A"
            print(
                f"  {row['label']}: {row['side']} {row['matched']:.1f}/"
                f"{row['original']:.1f} @ {price_s} [{row['status']}]"
            )
            print(f"    id: {row['order_id'][:20]}...  placed: {row['timestamp']}")

    print()


def _parse_log_timestamp(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main()
