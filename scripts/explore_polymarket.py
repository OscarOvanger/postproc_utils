#!/usr/bin/env python3
"""Discover and map Polymarket Tmax markets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import (  # noqa: E402
    DEFAULT_MARKETS_PATH,
    _bucket_width,
    build_clob_client,
    discover_tmax_markets,
    load_credentials,
    save_markets_map,
)


def _market_status(market: dict) -> str:
    if market.get("closed"):
        return "closed"
    if market.get("accepting_orders"):
        return "active"
    return "inactive"


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Polymarket Tmax Market Structure",
        "",
        f"Fetched at: {payload.get('fetched_at', '')}",
        "",
        "## Summary",
        "",
        "| City | Date | Buckets | Width | Status | Fee (bps) | Model |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for market in sorted(
        payload.get("markets", []),
        key=lambda row: (row.get("event_date", ""), row.get("city", "")),
    ):
        width = _bucket_width(market)
        width_str = str(width) if width is not None else "n/a"
        lines.append(
            f"| {market.get('city_display', market.get('city', ''))} "
            f"| {market.get('event_date', '')} "
            f"| {len(market.get('buckets', []))} "
            f"| {width_str} "
            f"| {_market_status(market)} "
            f"| {market.get('fee_rate_bps', 0)} "
            f"| {market.get('model_status', 'ok')} |"
        )

    lines.extend(["", "## City Slug Map", ""])
    for display, slug in sorted(payload.get("city_slug_map", {}).items()):
        lines.append(f"- {display} -> `{slug}`")

    lines.extend(["", "## Market Details", ""])
    for market in payload.get("markets", []):
        lines.append(
            f"### {market.get('city_display')} — {market.get('event_date')}"
        )
        lines.append("")
        lines.append(f"- Condition ID: `{market.get('condition_id')}`")
        lines.append(f"- Question: {market.get('question', '')}")
        lines.append(f"- Neg risk: {market.get('neg_risk')}")
        lines.append(f"- Tick size: {market.get('tick_size')}")
        lines.append(f"- Model status: {market.get('model_status', 'ok')}")
        lines.append("")
        lines.append("| Bucket | Type | Lower | Upper | Mid | Bid | Ask |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for bucket in market.get("buckets", []):
            lines.append(
                f"| {bucket.get('label', '')} "
                f"| {bucket.get('bucket_type', '')} "
                f"| {bucket.get('lower_f', '')} "
                f"| {bucket.get('upper_f', '')} "
                f"| {bucket.get('midpoint', '')} "
                f"| {bucket.get('best_bid', '')} "
                f"| {bucket.get('best_ask', '')} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary_table(payload: dict) -> None:
    print(
        f"{'City':<18} | {'Date':<12} | {'N buckets':>9} | "
        f"{'Bucket width':>12} | {'Status':<8} | {'Fee (bps)':>9}"
    )
    print("-" * 84)
    for market in sorted(
        payload.get("markets", []),
        key=lambda row: (row.get("event_date", ""), row.get("city", "")),
    ):
        width = _bucket_width(market)
        width_str = str(width) if width is not None else "n/a"
        print(
            f"{market.get('city_display', market.get('city', '')):<18} | "
            f"{market.get('event_date', ''):<12} | "
            f"{len(market.get('buckets', [])):>9} | "
            f"{width_str:>12} | "
            f"{_market_status(market):<8} | "
            f"{market.get('fee_rate_bps', 0):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Polymarket Tmax markets")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_MARKETS_PATH),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "polymarket_structure.md"),
    )
    parser.add_argument("--date", type=str, default=None, help="Filter to event date")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only include markets accepting orders",
    )
    args = parser.parse_args()

    print("Initializing Polymarket client...")
    credentials = load_credentials()
    client = build_clob_client(credentials)

    print("Discovering Tmax markets...")
    payload = discover_tmax_markets(
        client,
        event_date=args.date,
        active_only=args.active_only,
        fetch_prices=True,
    )

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    save_markets_map(json_path, payload)
    _write_markdown(md_path, payload)

    print(f"\nFound {len(payload.get('markets', []))} Tmax markets")
    print(f"Saved JSON: {json_path}")
    print(f"Saved markdown: {md_path}")
    print()
    _print_summary_table(payload)


if __name__ == "__main__":
    main()
