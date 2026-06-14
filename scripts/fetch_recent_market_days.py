"""Fetch and append market data for a specific date range.

Wraps the Codex fetch module but only pulls requested dates and merges
into existing per-city CSVs instead of overwriting the full history.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEX_FETCH = Path(
    "/Users/oscaro/Documents/Codex/2026-06-01/hey-i-want-you-to-use/work/"
    "fetch_all_city_tmax_market_data.py"
)
OUT_ROOT = PROJECT_ROOT / "historic_tmax_market_data"

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


def _load_codex():
    import sys
    import types

    if "zoneinfo" not in sys.modules:
        try:
            from backports.zoneinfo import ZoneInfo  # type: ignore
        except ImportError:
            from dateutil.tz import gettz

            def ZoneInfo(name: str):
                tz = gettz(name)
                if tz is None:
                    raise ValueError(f"Unknown timezone: {name}")
                return tz

        zoneinfo_mod = types.ModuleType("zoneinfo")
        zoneinfo_mod.ZoneInfo = ZoneInfo
        sys.modules["zoneinfo"] = zoneinfo_mod

    spec = importlib.util.spec_from_file_location("codex_fetch", CODEX_FETCH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {CODEX_FETCH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_path(city: dict) -> Path:
    city_dir = OUT_ROOT / city["slug"]
    return city_dir / city.get(
        "csv_name", f"{city['slug']}_tmax_kalshi_5min_same_day.csv"
    )


def _load_existing_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def fetch_city_dates(
    codex, city: dict, start: date, end: date, merge_each: bool = True, paper_live: bool = False
) -> list[dict]:
    """Fetch market rows for one city between start and end (inclusive)."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(city["tz"])
    markets_by_date, _, _, _ = codex.load_kalshi_markets(city["series"])
    target_dates = {d for d in markets_by_date if start <= d <= end}
    if not target_dates:
        print(f"  {city['slug']}: no markets in range")
        return []

    text_cache: dict = {}
    all_rows: list[dict] = []
    min_coverage = 0.70

    path = _csv_path(city)
    fieldnames, existing = _load_existing_rows(path)
    existing_dates = {r.get("event_date") for r in existing if r.get("event_date")}

    for d in sorted(target_dates, reverse=True):
        if d > datetime.now(tz).date():
            continue
        if d.isoformat() in existing_dates:
            print(f"  {city['slug']}: skip {d} (already in CSV)")
            continue
        print(f"  {city['slug']}: fetching {d}", flush=True)
        try:
            nws = codex.best_cli_record(city["pil"], d, tz, text_cache)
        except Exception as exc:
            print(f"    skip: nws_fetch_failed: {exc}")
            continue
        if not nws:
            if not paper_live:
                print("    skip: no_exact_nws_cli_record")
                continue
            tmax_dt = datetime.combine(d, dtime(16, 0), tzinfo=tz)
            nws = {
                "tmax_f": None,
                "tmax_time_local": tmax_dt,
                "station_text": "",
                "report_issue_utc": None,
                "product_id": "",
                "report_excerpt_maximum": "",
            }
        else:
            tmax_dt = nws["tmax_time_local"]
        market_opens = [
            codex.parse_kalshi_dt(m.get("open_time")).astimezone(tz)
            for m in markets_by_date[d]
            if codex.parse_kalshi_dt(m.get("open_time"))
        ]
        if not market_opens:
            print("    skip: no_kalshi_open_time")
            continue

        previous_midnight = datetime.combine(d - timedelta(days=1), dtime(0, 0), tz)
        start_dt = codex.ceil_to_next_5min(max(min(market_opens), previous_midnight))
        end_dt = tmax_dt.replace(second=0, microsecond=0)
        end_dt -= timedelta(minutes=end_dt.minute % 5)
        if end_dt < start_dt:
            print("    skip: no_5min_mark_between_market_open_and_tmax")
            continue

        markets = markets_by_date[d]
        tickers = [m["ticker"] for m in markets if m.get("ticker")]
        try:
            candles_by_ticker = codex.get_event_candles(
                tickers, start_dt, end_dt + timedelta(minutes=1)
            )
        except Exception as exc:
            print(f"    skip: kalshi_fetch_failed: {exc}")
            continue

        winning = [m for m in markets if codex.bucket_contains(m, nws["tmax_f"])] if nws.get("tmax_f") is not None else []
        if nws.get("tmax_f") is not None and len(winning) != 1:
            print(f"    skip: winning_bucket_count_{len(winning)}")
            continue
        winning_market = winning[0] if winning else {}
        winning_label = ""
        if winning_market:
            _, winning_label, _, _, _, _ = codex.bucket_metadata(winning_market)

        day_rows: list[dict] = []
        has_count = 0
        snapshots = list(codex.five_minute_times(start_dt, end_dt))
        expected = len(snapshots) * len(markets)
        for snapshot_dt in snapshots:
            for m in markets:
                candle = codex.latest_candle_at(
                    candles_by_ticker.get(m["ticker"], []), snapshot_dt
                )
                has_candle = candle is not None
                if has_candle:
                    has_count += 1
                bucket_type, bucket_label, lo, hi, yes_condition, no_condition = (
                    codex.bucket_metadata(m)
                )
                yb = codex.dollars_to_float(codex.cval(candle, "yes_bid", "close_dollars"))
                ya = codex.dollars_to_float(codex.cval(candle, "yes_ask", "close_dollars"))
                yes_mid = (yb + ya) / 2 if yb is not None and ya is not None else None
                no_bid = 1 - ya if ya is not None else None
                no_ask = 1 - yb if yb is not None else None
                no_mid = (
                    (no_bid + no_ask) / 2
                    if no_bid is not None and no_ask is not None
                    else None
                )
                candle_ts = candle.get("end_period_ts") if candle else None
                candle_dt = (
                    datetime.fromtimestamp(candle_ts, timezone.utc).astimezone(tz)
                    if candle_ts
                    else None
                )
                is_winning = bool(winning_market) and m["ticker"] == winning_market["ticker"]
                day_rows.append(
                    {
                        "event_date": d.isoformat(),
                        "city": city["city"],
                        "station_text": nws["station_text"],
                        "nws_pil": city["pil"],
                        "nws_tmax_f": nws["tmax_f"],
                        "nws_tmax_time_local": tmax_dt.isoformat(),
                        "nws_tmax_time_local_hhmm": tmax_dt.strftime("%H:%M"),
                        "nws_report_issue_utc": (
                            nws["report_issue_utc"].isoformat()
                            if nws["report_issue_utc"]
                            else ""
                        ),
                        "nws_product_id": nws["product_id"],
                        "nws_report_excerpt_maximum": nws["report_excerpt_maximum"],
                        "series_ticker": city["series"],
                        "event_ticker": m.get("event_ticker", ""),
                        "market_ticker": m.get("ticker", ""),
                        "market_status": m.get("status", ""),
                        "market_title": m.get("title", ""),
                        "bucket_label": bucket_label,
                        "bucket_type": bucket_type,
                        "bucket_lower_inclusive_f": "" if lo is None else lo,
                        "bucket_upper_inclusive_f": "" if hi is None else hi,
                        "bucket_yes_condition": yes_condition,
                        "bucket_no_condition": no_condition,
                        "bucket_resolved_to_one_dollars": str(is_winning).lower(),
                        "contract_resolved_side": codex.settlement_side(m, is_winning),
                        "kalshi_settlement_value_dollars": m.get("settlement_value_dollars") or "",
                        "winning_market_ticker": winning_market.get("ticker", ""),
                        "winning_bucket_label": winning_label,
                        "snapshot_time_local": snapshot_dt.isoformat(),
                        "snapshot_time_utc": snapshot_dt.astimezone(timezone.utc).isoformat(),
                        "snapshot_phase": (
                            "previous_day" if snapshot_dt.date() < d else "event_day"
                        ),
                        "minutes_before_tmax": int(
                            (tmax_dt - snapshot_dt).total_seconds() // 60
                        ),
                        "kalshi_candle_end_time_local": (
                            candle_dt.isoformat() if candle_dt else ""
                        ),
                        "kalshi_candle_end_time_utc": (
                            candle_dt.astimezone(timezone.utc).isoformat() if candle_dt else ""
                        ),
                        "has_snapshot_candle": str(has_candle).lower(),
                        "yes_bid_close": codex.fmt(yb),
                        "yes_ask_close": codex.fmt(ya),
                        "yes_mid_close": codex.fmt(yes_mid),
                        "no_bid_close": codex.fmt(no_bid),
                        "no_ask_close": codex.fmt(no_ask),
                        "no_mid_close": codex.fmt(no_mid),
                        "last_trade_price_close": codex.cval(candle, "price", "close_dollars"),
                        "last_trade_price_previous": codex.cval(
                            candle, "price", "previous_dollars"
                        ),
                        "volume_contracts": candle.get("volume_fp", "") if candle else "",
                        "open_interest_contracts": (
                            candle.get("open_interest_fp", "") if candle else ""
                        ),
                    }
                )

        coverage = has_count / expected if expected else 0
        if coverage < min_coverage:
            print(f"    skip: low coverage {coverage:.2%}")
            continue
        print(f"    retained {d}: {len(day_rows)} rows, coverage {coverage:.2%}")
        all_rows.extend(day_rows)
        if merge_each:
            merge_and_write(city, day_rows)
            existing_dates.add(d.isoformat())

    return all_rows


def merge_and_write(city: dict, new_rows: list[dict]) -> dict:
    path = _csv_path(city)
    fieldnames, existing = _load_existing_rows(path)
    if new_rows:
        new_dates = {r["event_date"] for r in new_rows}
        existing = [r for r in existing if r.get("event_date") not in new_dates]
        merged = existing + new_rows
        if not fieldnames:
            fieldnames = list(new_rows[0].keys())
        else:
            for row in new_rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
    else:
        merged = existing

    dates = sorted({r["event_date"] for r in merged if r.get("event_date")})
    return {
        "city": city["slug"],
        "rows": len(merged),
        "dates": len(dates),
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "new_rows": len(new_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-06-03")
    parser.add_argument("--end", default="2026-06-12")
    parser.add_argument("--train-only", action="store_true", default=True)
    parser.add_argument("--city", type=str, default=None, help="Single city slug")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    codex = _load_codex()

    print(f"Fetching market data for {start} to {end}")
    summaries = []
    for city in codex.CITIES:
        if args.train_only and city["slug"] not in TRAIN_SLUGS:
            continue
        if args.city and city["slug"] != args.city:
            continue
        print(f"\n=== {city['city']} ===")
        new_rows = fetch_city_dates(codex, city, start, end, merge_each=True)
        summary = merge_and_write(city, [])  # report final state
        summary["new_rows"] = len(new_rows)
        summaries.append(summary)
        print(f"  merged: {summary}")

    print("\n=== DATA AVAILABILITY AFTER FETCH ===")
    print(f"{'City':18} | {'Rows':>6} | {'Dates':>5} | {'End date':>10} | New rows")
    for s in summaries:
        print(
            f"{s['city']:18} | {s['rows']:6d} | {s['dates']:5d} | "
            f"{s['end']:>10} | +{s['new_rows']}"
        )


if __name__ == "__main__":
    main()
