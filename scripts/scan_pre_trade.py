#!/usr/bin/env python3
"""Pre-trade scan: 10-city modal buckets + TrackB + NGBoost + WU High."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scan_modal_buckets import (  # noqa: E402
    TARGET_CITIES,
    discover_markets,
    scan_city,
)
from poly_portfolio_status import (  # noqa: E402
    _NgBoostModels,
    explain_ngboost_unavailable,
    fetch_ngboost_forecast,
    fetch_trackb_forecast,
)
from src.polymarket_api import build_clob_client  # noqa: E402
from src.wunderground_forecast import fetch_wu_high, load_city_stations  # noqa: E402


def format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def format_int(value: int | float | None) -> str:
    if value is None:
        return "—"
    return str(int(round(value)))


def print_summary_table(
    scans,
    *,
    event_date: str,
    trackb: dict[str, int | None],
    ngboost: dict[str, float | None],
    wu_high: dict[str, int | None],
    verbose: bool,
) -> None:
    print(f"\nPre-trade scan for {event_date}")
    header = (
        f"{'City':<16} {'Modal bucket':<14} {'Bid':>7} {'Ask':>7} {'Mid':>7} "
        f"{'TrackB':>7} {'NGB μ':>7} {'WU High':>8}"
    )
    print(header)
    print("-" * len(header))

    for scan in scans:
        if scan.status != "ok" or scan.modal is None:
            status = scan.status.upper().replace("_", " ")
            print(
                f"{scan.display_name:<16} {status:<14} {'':>7} {'':>7} {'':>7} "
                f"{format_int(trackb.get(scan.slug)):>7} "
                f"{format_int(ngboost.get(scan.slug)):>7} "
                f"{format_int(wu_high.get(scan.slug)):>8}"
            )
            continue

        modal = scan.modal
        print(
            f"{scan.display_name:<16} {modal.label:<14} "
            f"{format_price(modal.best_bid):>7} {format_price(modal.best_ask):>7} "
            f"{format_price(modal.midpoint):>7} "
            f"{format_int(trackb.get(scan.slug)):>7} "
            f"{format_int(ngboost.get(scan.slug)):>7} "
            f"{format_int(wu_high.get(scan.slug)):>8}"
        )

        if verbose and scan.buckets:
            for bucket in scan.buckets:
                if bucket.label == modal.label:
                    continue
                print(
                    f"  {'':<16} {bucket.label:<14} "
                    f"{format_price(bucket.best_bid):>7} {format_price(bucket.best_ask):>7} "
                    f"{format_price(bucket.midpoint):>7}"
                )

    stations = load_city_stations()
    print("\nNotes:")
    print("  TrackB: dallas, seattle, miami, atlanta may show — if not in deploy_config cities.")
    print("  NGBoost μ: requires HRRR row for event date (auto-fetched for today/tomorrow).")
    print("  WU High: Polymarket resolution station pages (see src/wunderground_forecast.py WU_PAGE_BY_CITY).")


def main() -> None:
    parser = argparse.ArgumentParser(description="10-city pre-trade modal bucket scan.")
    parser.add_argument("--date", default=str(date.today()), help="Event date YYYY-MM-DD")
    parser.add_argument("--verbose", action="store_true", help="Print all buckets per city")
    parser.add_argument("--no-forecasts", action="store_true", help="Markets only (skip TrackB/NGB/WU)")
    args = parser.parse_args()

    event_date = args.date
    try:
        date.fromisoformat(event_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid --date {event_date!r}: {exc}") from exc

    print("Initializing Polymarket CLOB client...")
    client = build_clob_client()

    print(f"Discovering Tmax markets for {event_date}...")
    discovered = discover_markets(event_date)

    scans = []
    for slug, display_name in TARGET_CITIES:
        market = discovered.get(slug)
        n_buckets = len(market["buckets"]) if market else 0
        print(f"  {display_name}: {n_buckets} buckets found")
        scans.append(scan_city(client, slug, display_name, market))

    trackb: dict[str, int | None] = {}
    ngboost: dict[str, float | None] = {}
    wu_high: dict[str, int | None] = {}

    if not args.no_forecasts:
        ng_models = None
        try:
            ng_models = _NgBoostModels()
        except Exception:
            ng_models = None

        for slug, display_name in TARGET_CITIES:
            trackb[slug] = fetch_trackb_forecast(slug, event_date)
            ngboost[slug] = fetch_ngboost_forecast(slug, event_date, models=ng_models)
            if ngboost[slug] is None:
                reason = explain_ngboost_unavailable(slug, event_date)
                print(f"  WARNING: NGBoost unavailable for {display_name}: {reason}", file=sys.stderr)
            wu = fetch_wu_high(slug)
            wu_high[slug] = wu
            if wu is None:
                print(f"  WARNING: WU High scrape failed for {display_name}", file=sys.stderr)

    print_summary_table(
        scans,
        event_date=event_date,
        trackb=trackb,
        ngboost=ngboost,
        wu_high=wu_high,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
