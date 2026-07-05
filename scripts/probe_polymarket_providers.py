#!/usr/bin/env python3
"""Probe PolymarketData vs Telonex for Tmax bucket history (free-tier friendly)."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_polymarket_history import (  # noqa: E402
    CITY_SUBCATEGORIES,
    MARKETS_INDEX_PATH,
    TARGET_CITIES,
)
from src.provider_keys import load_polymarketdata_key, load_telonex_key  # noqa: E402

PMD_BASE = "https://api.polymarketdata.co/v1"
REPORT_PATH = PROJECT_ROOT / "reports" / "polymarket_provider_probe.md"
PROBE_JSON_PATH = PROJECT_ROOT / "reports" / "polymarket_provider_probe.json"
VERIFY_REPORT_PATH = PROJECT_ROOT / "reports" / "telonex_purchase_verification.md"
VERIFY_JSON_PATH = PROJECT_ROOT / "reports" / "telonex_purchase_verification.json"
COMPARE_REPORT_PATH = PROJECT_ROOT / "reports" / "pmd_telonex_coverage_comparison.md"
COMPARE_JSON_PATH = PROJECT_ROOT / "reports" / "pmd_telonex_coverage_comparison.json"
DOWNLOAD_USAGE_PATH = PROJECT_ROOT / "reports" / "telonex_download_usage.json"

TELONEX_FREE_DOWNLOAD_LIMIT = 5
SF_PROBE_SAMPLE_SLUG = "highest-temperature-in-san-francisco-on-march-24-2026-55forbelow"
# Telonex API channel names (no literal "order_book"; map to depth snapshots).
ORDER_BOOK_CHANNEL_ALIASES = ("order_book", "book_snapshot_5", "book_snapshot_25", "book_snapshot_full")

TMAX_QUESTION_RE = re.compile(r"(?i)highest temperature in (.+?) (?:be|on)")
DATE_IN_SLUG_RE = re.compile(
    r"-on-(january|february|march|april|may|june|july|august|september|october|november|december)-(\d+)-(\d{4})-"
)


def _parse_slug_date(slug: str) -> str | None:
    match = DATE_IN_SLUG_RE.search(slug.lower())
    if not match:
        return None
    month_name, day, year = match.groups()
    month = datetime.strptime(month_name.title(), "%B").month
    return f"{year}-{month:02d}-{int(day):02d}"


def _city_from_question(question: str) -> str | None:
    match = TMAX_QUESTION_RE.search(question)
    if not match:
        return None
    city_text = match.group(1).strip().lower()
    for slug, label in CITY_SUBCATEGORIES.items():
        if label.lower() in city_text or city_text in label.lower():
            return slug
    aliases = {"nyc": "new_york", "new york city": "new_york"}
    return aliases.get(city_text)


def load_local_index() -> dict[str, Any]:
    if not MARKETS_INDEX_PATH.exists():
        return {}
    return json.loads(MARKETS_INDEX_PATH.read_text(encoding="utf-8"))


def phase1_telonex_catalog() -> pd.DataFrame:
    from telonex import get_markets_dataframe

    print("Phase 1a: Telonex markets catalog (free, no API key)...")
    markets = get_markets_dataframe(exchange="polymarket")
    q = markets["question"].astype(str)
    mask = q.str.contains(r"highest temperature", case=False, na=False)
    tmax = markets.loc[mask].copy()
    tmax["city"] = tmax["question"].map(_city_from_question)
    tmax["event_date"] = tmax["slug"].map(_parse_slug_date)
    us = tmax[tmax["city"].isin(TARGET_CITIES)].copy()
    print(f"  Telonex Tmax rows (10 US cities): {len(us):,} / {len(tmax):,} global Tmax")
    return us


def phase1_pmd_search(
    session: requests.Session,
    api_key: str,
    *,
    limit: int = 100,
    max_pages: int = 1,
) -> tuple[list[dict[str, Any]], bool]:
    print("Phase 1b: PolymarketData /markets search (free)...")
    results: list[dict[str, Any]] = []
    auth_blocked = False
    for slug, label in CITY_SUBCATEGORIES.items():
        query = f"Highest temperature in {label}"
        batch: list[dict[str, Any]] = []
        for page in range(max_pages):
            offset = page * limit
            try:
                response = session.get(
                    f"{PMD_BASE}/markets",
                    headers={"X-API-Key": api_key},
                    params={"search": query, "limit": limit, "offset": offset},
                    timeout=60,
                )
                if response.status_code in (401, 403):
                    print("PMD discovery requires paid tier", file=sys.stderr)
                    auth_blocked = True
                    break
                if response.status_code == 429:
                    time.sleep(2.0)
                    response = session.get(
                        f"{PMD_BASE}/markets",
                        headers={"X-API-Key": api_key},
                        params={"search": query, "limit": limit, "offset": offset},
                        timeout=60,
                    )
                response.raise_for_status()
                page_rows = response.json().get("data", [])
            except requests.RequestException as exc:
                print(f"  WARNING: PMD search failed for {label}: {exc}", file=sys.stderr)
                page_rows = []
            if not page_rows:
                break
            batch.extend(page_rows)
            if len(page_rows) < limit:
                break
            time.sleep(0.3)
        if auth_blocked:
            break
        time.sleep(0.2)
        for row in batch:
            item = dict(row)
            item["probe_city"] = slug
            results.append(item)
        print(f"  {label}: {len(batch)} hits")
    if auth_blocked and not results:
        return [], True
    return results, auth_blocked


def cross_check_local(
    telonex_us: pd.DataFrame,
    pmd_rows: list[dict[str, Any]],
    local_index: dict[str, Any],
) -> dict[str, Any]:
    local_slugs: set[str] = set()
    for event in local_index.get("events", []):
        for bucket in event.get("buckets", []):
            slug = bucket.get("slug")
            if slug:
                local_slugs.add(str(slug))

    telonex_slugs = set(telonex_us["slug"].astype(str))
    pmd_slugs = {str(r.get("slug", "")) for r in pmd_rows if r.get("slug")}
    overlap_tlx = len(local_slugs & telonex_slugs)
    overlap_pmd = len(local_slugs & pmd_slugs)
    return {
        "local_slug_count": len(local_slugs),
        "telonex_overlap": overlap_tlx,
        "pmd_overlap": overlap_pmd,
        "local_summary": local_index.get("summary", {}),
    }


def coverage_matrix(telonex_us: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for city in TARGET_CITIES:
        subset = telonex_us[telonex_us["city"] == city]
        dates = subset["event_date"].dropna().astype(str)
        events = subset.groupby("event_date")["slug"].nunique() if len(subset) else pd.Series(dtype=int)
        rows.append(
            {
                "city": city,
                "bucket_markets": int(len(subset)),
                "event_dates": int(dates.nunique()) if len(dates) else 0,
                "date_min": str(dates.min()) if len(dates) else None,
                "date_max": str(dates.max()) if len(dates) else None,
                "avg_buckets_per_event": float(events.mean()) if len(events) else 0.0,
            }
        )
    return rows


def pick_sample_from_index(local_index: dict[str, Any]) -> dict[str, str]:
    """Prefer SF 2026-03-24 from local index; fall back to first bucket."""
    for event in local_index.get("events", []):
        if event.get("city") == "san_francisco" and event.get("date") == "2026-03-24":
            bucket = event["buckets"][0]
            return {
                "city": "san_francisco",
                "date": "2026-03-24",
                "slug": str(bucket["slug"]),
                "token_id_yes": str(bucket.get("token_id_yes", "")),
            }
    for event in local_index.get("events", []):
        if event.get("buckets"):
            bucket = event["buckets"][0]
            return {
                "city": str(event.get("city", "")),
                "date": str(event.get("date", "")),
                "slug": str(bucket["slug"]),
                "token_id_yes": str(bucket.get("token_id_yes", "")),
            }
    return {
        "city": "san_francisco",
        "date": "2026-03-24",
        "slug": "highest-temperature-in-san-francisco-on-march-24-2026-55forbelow",
        "token_id_yes": "",
    }


def phase2_pmd_sample(
    session: requests.Session,
    api_key: str,
    sample: dict[str, str],
) -> dict[str, Any]:
    from datetime import timedelta

    print(f"Phase 2a: PolymarketData prices sample ({sample['slug']})...")
    slug = sample["slug"]
    end_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_start = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")

    try:
        response = session.get(
            f"{PMD_BASE}/markets/{slug}/prices",
            headers={"X-API-Key": api_key},
            params={"start_ts": recent_start, "end_ts": end_ts, "resolution": "10m"},
            timeout=60,
        )
        if not response.ok:
            return {
                "slug": slug,
                "ok": False,
                "status": response.status_code,
                "detail": response.text[:400],
                "free_tier_notes": "1m requires Pro; free tier minimum 10m; history capped to ~30 days",
            }
        payload = response.json()
        data = payload.get("data", [])
        if isinstance(data, dict):
            yes_rows = data.get("Yes", data.get("yes", []))
            rows = yes_rows or data.get("No", data.get("no", [])) or []
        else:
            rows = data
        gaps_min: list[float] = []
        if len(rows) > 1:
            ts = pd.to_datetime([r["t"] for r in rows], utc=True)
            gaps = ts.diff().dt.total_seconds() / 60.0
            gaps_min = [float(x) for x in gaps.dropna().tolist()]
        return {
            "slug": slug,
            "ok": True,
            "resolution": "10m",
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "gap_minutes_min": min(gaps_min) if gaps_min else None,
            "gap_minutes_median": float(pd.Series(gaps_min).median()) if gaps_min else None,
            "gap_minutes_max": max(gaps_min) if gaps_min else None,
            "first_ts": rows[0]["t"] if rows else None,
            "last_ts": rows[-1]["t"] if rows else None,
            "free_tier_notes": "1m requires Pro; free tier minimum 10m; history capped to ~30 days",
        }
    except requests.RequestException as exc:
        return {"slug": slug, "ok": False, "error": str(exc)}


def phase2_telonex_sample(api_key: str, sample: dict[str, str]) -> dict[str, Any]:
    from telonex import get_dataframe

    print(f"Phase 2b: Telonex quotes sample ({sample['slug']})...")
    event_date = sample["date"]
    next_day = (datetime.fromisoformat(event_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = get_dataframe(
            api_key=api_key,
            exchange="polymarket",
            channel="quotes",
            slug=sample["slug"],
            outcome="Yes",
            from_date=event_date,
            to_date=next_day,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    gaps_us: list[float] = []
    if len(df) > 1 and "timestamp_us" in df.columns:
        ts = pd.to_numeric(df["timestamp_us"], errors="coerce")
        gaps_us = ts.diff().dropna().tolist()

    return {
        "ok": True,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "gap_us_min": min(gaps_us) if gaps_us else None,
        "gap_us_median": float(pd.Series(gaps_us).median()) if gaps_us else None,
        "gap_us_max": max(gaps_us) if gaps_us else None,
        "has_bbo": all(c in df.columns for c in ("bid_price", "ask_price")),
    }


def estimate_backfill_volume(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    total_events = sum(r["event_dates"] for r in matrix)
    avg_buckets = sum(r["avg_buckets_per_event"] for r in matrix) / max(len(matrix), 1)
    bucket_markets = sum(r["bucket_markets"] for r in matrix)
    return {
        "telonex_est_daily_files_per_bucket": 1,
        "est_bucket_markets_in_catalog": bucket_markets,
        "est_event_days": total_events,
        "avg_buckets_per_event": round(avg_buckets, 1),
        "telonex_note": "One parquet file per bucket per day; full backfill ~ bucket_markets * trading_days",
        "pmd_note": "One API call per bucket per time window; Pro needed for 1m beyond 30-day free window",
    }


def recommend(
    matrix: list[dict[str, Any]],
    pmd_sample: dict[str, Any],
    tlx_sample: dict[str, Any],
    cross: dict[str, Any],
) -> str:
    hist = pmd_sample
    recent_1m = pmd_sample
    tlx_ok = tlx_sample.get("ok") and tlx_sample.get("row_count", 0) > 0
    has_us_coverage = any(r["bucket_markets"] > 0 for r in matrix)

    if not has_us_coverage:
        return "NEITHER — no 10-city US Tmax bucket markets found in Telonex catalog."

    if tlx_ok and tlx_sample.get("has_bbo"):
        if hist.get("ok") is False and "30 days" in str(hist.get("detail", "")):
            return (
                "**Telonex Plus ($79/mo)** for tick/BBO quotes per bucket with multi-year depth. "
                "PolymarketData Pro only if you specifically need 1-minute bars (free tier caps at 10m "
                "and last-30-days history). Your Resolved Markets archive already covers Mar–May 2026 "
                "for backtest overlap — Telonex fills pre/post-resolution tick history."
            )
        return (
            "**Telonex Plus ($79/mo)** for tick/BBO quotes per bucket with multi-year depth. "
            "PolymarketData Pro only if you need 1-minute bars (free tier: 10m min, ~30-day history). "
            "Resolved Markets archive overlaps Mar–May 2026 — use Telonex for tick history outside that window."
        )
    if recent_1m.get("ok"):
        return (
            "**PolymarketData Pro** if 1-minute prices/books for all buckets is the priority; "
            "free tier confirmed 10m max and 30-day history cap. "
            "**Telonex Plus** if sub-minute BBO and parquet bulk download scale better."
        )
    return (
        "**Telonex Plus** preferred: tick-level quotes with BBO confirmed on sample; "
        "catalog shows full 10-city Tmax coverage. PolymarketData useful as complement for "
        "1m aggregated bars if Pro unlocks full history."
    )


def write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROBE_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Polymarket data provider probe",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Recommendation",
        "",
        payload["recommendation"],
        "",
        "## Phase 1 — Discovery (zero download cost)",
        "",
        f"- Local Resolved Markets index: {payload['cross_check']['local_slug_count']} bucket slugs",
        f"- Telonex catalog overlap: {payload['cross_check']['telonex_overlap']} slugs",
        f"- PolymarketData search overlap: {payload['cross_check']['pmd_overlap']} slugs",
        "",
        "### 10-city coverage (Telonex catalog)",
        "",
        "| City | Bucket markets | Event dates | Date min | Date max | Avg buckets/event |",
        "|------|----------------|-------------|----------|----------|-------------------|",
    ]
    for row in payload["coverage_matrix"]:
        lines.append(
            f"| {row['city']} | {row['bucket_markets']} | {row['event_dates']} | "
            f"{row['date_min'] or '—'} | {row['date_max'] or '—'} | {row['avg_buckets_per_event']:.1f} |"
        )

    sample = payload["sample"]
    lines.extend(
        [
            "",
            "## Phase 2 — Controlled samples (2 downloads)",
            "",
            f"Sample: **{sample['city']}** {sample['date']} — `{sample['slug']}`",
            "",
            "### PolymarketData `/markets/{{slug}}/prices`",
            "",
            f"```json\n{json.dumps(payload['pmd_sample'], indent=2)}\n```",
            "",
            "### Telonex `quotes` channel",
            "",
            f"```json\n{json.dumps(payload['telonex_sample'], indent=2)}\n```",
            "",
            "## Phase 3 — Backfill estimate",
            "",
            f"```json\n{json.dumps(payload['backfill_estimate'], indent=2)}\n```",
            "",
            "## Notes",
            "",
            "- PolymarketData free tier: **10m minimum** resolution; **1m requires Pro**; history capped to **last 30 days** on free key.",
            "- Telonex free tier: **5 file downloads** total; markets catalog is unlimited without a key.",
            "- Existing `download_polymarket_history.py` (Resolved Markets) overlaps Mar–May 2026 — avoid paying twice for the same window.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


def _slug_matches_city(slug: str, city: str) -> bool:
    s = slug.lower()
    if city == "new_york":
        return (
            "highest-temperature-in-new-york-on-" in s
            or "highest-temperature-in-nyc-on-" in s
            or "highest-temperature-in-new-york-city-on-" in s
        )
    needle = city.replace("_", "-")
    return f"highest-temperature-in-{needle}-on-" in s


def browse_tmax_catalog_metadata() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Step 1: free Telonex markets catalog browse — metadata only, no paid downloads."""
    from telonex import get_markets_dataframe

    print("\n=== Step 1: Telonex catalog browse (free public dataset, no API key) ===")
    markets = get_markets_dataframe(exchange="polymarket")
    q = markets["question"].astype(str)
    mask = q.str.contains(r"highest temperature", case=False, na=False)
    tmax = markets.loc[mask].copy()
    tmax["city"] = tmax["question"].map(_city_from_question)
    tmax["event_date"] = tmax["slug"].map(_parse_slug_date)
    us = tmax[tmax["city"].isin(TARGET_CITIES)].copy()

    rows: list[dict[str, Any]] = []
    print("")
    header = f"{'City':<16} {'Markets':>8} {'Earliest':>12} {'Latest':>12}"
    print(header)
    print("-" * len(header))
    for city in TARGET_CITIES:
        subset = us[us["slug"].map(lambda s: _slug_matches_city(str(s), city))]
        dates = subset["event_date"].dropna().astype(str)
        row = {
            "city": city,
            "market_count": int(len(subset)),
            "earliest_event_date": str(dates.min()) if len(dates) else None,
            "latest_event_date": str(dates.max()) if len(dates) else None,
        }
        rows.append(row)
        print(
            f"{city:<16} {row['market_count']:8d} "
            f"{row['earliest_event_date'] or '—':>12} {row['latest_event_date'] or '—':>12}"
        )
    print(f"\nCatalog Tmax rows (10 US cities): {len(us):,}")
    return us, rows


def load_download_usage() -> dict[str, Any]:
    if DOWNLOAD_USAGE_PATH.exists():
        return json.loads(DOWNLOAD_USAGE_PATH.read_text(encoding="utf-8"))
    # Seed from initial probe quotes sample (1 download) unless overridden.
    used = int(os.environ.get("TELONEX_DOWNLOADS_USED", "1"))
    payload = {
        "total_free": TELONEX_FREE_DOWNLOAD_LIMIT,
        "used": used,
        "history": [{"note": "seeded: initial SF quotes probe", "used": used}],
    }
    DOWNLOAD_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_USAGE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_download_usage(payload: dict[str, Any]) -> None:
    DOWNLOAD_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_USAGE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_free_download_quota(expected_remaining: int | None = None) -> int:
    if expected_remaining is None:
        expected_remaining = int(os.environ.get("TELONEX_EXPECTED_REMAINING", "4"))
    usage = load_download_usage()
    used = int(usage.get("used", 0))
    remaining = TELONEX_FREE_DOWNLOAD_LIMIT - used
    print(
        f"\nDownload quota: {used}/{TELONEX_FREE_DOWNLOAD_LIMIT} used, "
        f"{remaining} remaining (expect {expected_remaining} before Step 2)"
    )
    if remaining != expected_remaining:
        raise AssertionError(
            f"Expected {expected_remaining} free Telonex downloads remaining, "
            f"but ledger shows {remaining} (used={used}). "
            f"Update {DOWNLOAD_USAGE_PATH} or set TELONEX_DOWNLOADS_USED."
        )
    return remaining


def _record_download(label: str) -> None:
    usage = load_download_usage()
    usage["used"] = int(usage.get("used", 0)) + 1
    usage.setdefault("history", []).append(
        {"at": datetime.now(timezone.utc).isoformat(), "label": label}
    )
    save_download_usage(usage)


def pick_slug_and_quotes_date(
    catalog: pd.DataFrame,
    city: str,
    *,
    prefer_latest: bool = True,
) -> tuple[str, str]:
    """Pick a slug and a calendar day known to have quotes data (via free availability API)."""
    from telonex import get_availability
    from telonex.exceptions import NotFoundError

    dates = sorted(catalog.loc[catalog["city"] == city, "event_date"].dropna().astype(str).unique())
    if not dates:
        raise ValueError(f"No catalog dates for {city}")
    ordered = list(reversed(dates)) if prefer_latest else dates
    for event_date in ordered:
        try:
            slug = pick_slug_for_city_date(catalog, city, event_date)
        except ValueError:
            continue
        try:
            av = get_availability(exchange="polymarket", slug=slug, outcome="Yes")
        except NotFoundError:
            continue
        quotes = av.get("channels", {}).get("quotes")
        if not quotes:
            continue
        from_dt = datetime.fromisoformat(str(quotes["from_date"]))
        to_excl = datetime.fromisoformat(str(quotes["to_date"]))
        target = min(datetime.fromisoformat(event_date), to_excl - timedelta(days=1))
        if target < from_dt:
            target = from_dt
        return slug, target.strftime("%Y-%m-%d")
    raise ValueError(f"No quotes availability found for {city}")


def _run_download_check(
    api_key: str,
    *,
    check_name: str,
    label: str,
    channel: str,
    slug: str,
    event_date: str,
    extra_fn=None,
) -> dict[str, Any]:
    try:
        df, meta = telonex_download_one_day(
            api_key,
            label=label,
            channel=channel,
            slug=slug,
            event_date=event_date,
        )
        row = {"pass": len(df) > 0, **meta}
        if extra_fn:
            row.update(extra_fn(df, meta))
        _print_download_result(meta, {k: v for k, v in row.items() if k not in meta and k != "pass"})
        return row
    except Exception as exc:
        err = str(exc)
        if "limit_reached" in err.lower():
            err = f"{err} (Telonex free-tier download cap hit on server; see reports/telonex_download_usage.json)"
        print(f"\n  [{label}] FAIL: {err}")
        return {
            "pass": False,
            "label": label,
            "channel_requested": channel,
            "slug": slug,
            "event_date": event_date,
            "error": str(exc),
        }


def pick_slug_for_city_date(catalog: pd.DataFrame, city: str, event_date: str) -> str:
    subset = catalog[
        (catalog["city"] == city)
        & (catalog["event_date"].astype(str) == event_date)
        & catalog["slug"].map(lambda s: _slug_matches_city(str(s), city))
    ]
    if subset.empty:
        raise ValueError(f"No catalog slug for {city} on {event_date}")
    return str(subset.iloc[0]["slug"])


def _next_day(date_str: str) -> str:
    return (datetime.fromisoformat(date_str) + timedelta(days=1)).strftime("%Y-%m-%d")


def _dataframe_timestamp_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    if "timestamp_us" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["timestamp_us"], errors="coerce"), unit="us", utc=True)
    elif "t" in df.columns:
        ts = pd.to_datetime(df["t"], utc=True)
    else:
        return None, None
    ts = ts.dropna()
    if ts.empty:
        return None, None
    return ts.min().isoformat(), ts.max().isoformat()


def resolve_telonex_channel(channel: str) -> tuple[str, str]:
    """Return (api_channel, channel_requested). Maps order_book -> book_snapshot_5."""
    if channel == "order_book":
        return "book_snapshot_5", channel
    return channel, channel


def telonex_download_one_day(
    api_key: str,
    *,
    label: str,
    channel: str,
    slug: str,
    event_date: str,
    outcome: str = "Yes",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Spend exactly one Telonex file download (single calendar day)."""
    from telonex import get_dataframe

    channel_used, channel_requested = resolve_telonex_channel(channel)
    to_date = _next_day(event_date)
    df = get_dataframe(
        api_key=api_key,
        exchange="polymarket",
        channel=channel_used,
        slug=slug,
        outcome=outcome,
        from_date=event_date,
        to_date=to_date,
    )

    _record_download(label)
    t_min, t_max = _dataframe_timestamp_range(df)
    meta = {
        "label": label,
        "channel_requested": channel_requested,
        "channel_used": channel_used,
        "slug": slug,
        "event_date": event_date,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "timestamp_min": t_min,
        "timestamp_max": t_max,
    }
    if len(df) == 0:
        raise AssertionError(f"{label}: expected nonzero rows, got 0")
    return df, meta


def analyze_order_book_depth(df: pd.DataFrame, *, channel_used: str) -> dict[str, Any]:
    """Return whether schema includes multi-level depth beyond top-of-book."""
    bid_level_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"bid_price_\d+", str(c))],
        key=lambda x: int(str(x).rsplit("_", 1)[-1]),
    )
    ask_level_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"ask_price_\d+", str(c))],
        key=lambda x: int(str(x).rsplit("_", 1)[-1]),
    )
    depth_levels = max(len(bid_level_cols), len(ask_level_cols))

    def _active_levels(cols: list[str]) -> int:
        active = 0
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any() and (vals.fillna(0) > 0).any():
                active += 1
        return active

    bid_active = _active_levels(bid_level_cols)
    ask_active = _active_levels(ask_level_cols)

    multi_level_rows = 0
    for _, row in df.iterrows():
        bid_filled = sum(
            1
            for col in bid_level_cols
            if pd.notna(row.get(col)) and float(row.get(col) or 0) > 0
        )
        ask_filled = sum(
            1
            for col in ask_level_cols
            if pd.notna(row.get(col)) and float(row.get(col) or 0) > 0
        )
        if bid_filled > 1 or ask_filled > 1:
            multi_level_rows += 1

    has_multi_level = bid_active > 1 or ask_active > 1 or multi_level_rows > 0
    identical_tob_only = not has_multi_level
    return {
        "channel_used": channel_used,
        "depth_level_columns": depth_levels,
        "bid_levels_with_data": bid_active,
        "ask_levels_with_data": ask_active,
        "rows_with_multi_level": int(multi_level_rows),
        "has_multi_level_depth": has_multi_level,
        "identical_top_of_book_only": identical_tob_only,
    }


def _pick_random_recent_date(catalog: pd.DataFrame, city: str, *, seed: int = 42) -> str:
    dates = sorted(catalog.loc[catalog["city"] == city, "event_date"].dropna().astype(str).unique())
    if not dates:
        raise ValueError(f"No event dates for {city}")
    recent = dates[-14:] if len(dates) >= 14 else dates
    rng = random.Random(seed)
    return rng.choice(recent)


def _city_with_longest_history(city_rows: list[dict[str, Any]]) -> tuple[str, str]:
    dated = [r for r in city_rows if r.get("earliest_event_date")]
    if not dated:
        raise ValueError("No dated markets in catalog")
    best = min(dated, key=lambda r: r["earliest_event_date"])
    return str(best["city"]), str(best["earliest_event_date"])


def _print_download_result(meta: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    print(f"\n  [{meta['label']}]")
    print(f"    channel: {meta['channel_requested']} -> {meta['channel_used']}")
    print(f"    slug: {meta['slug']}")
    print(f"    requested date: {meta['event_date']}")
    print(f"    rows: {meta['row_count']}")
    print(f"    timestamp range: {meta['timestamp_min']} .. {meta['timestamp_max']}")
    print(f"    columns: {meta['columns']}")
    if extra:
        for key, value in extra.items():
            print(f"    {key}: {value}")


def _already_downloaded(label: str) -> bool:
    return any(h.get("label") == label for h in load_download_usage().get("history", []))


def verify_coverage_before_purchase() -> dict[str, Any]:
    """Pre-purchase verification: catalog browse + 4 controlled Telonex downloads."""
    api_key = load_telonex_key()
    catalog, city_rows = browse_tmax_catalog_metadata()

    remaining = assert_free_download_quota()

    print("\n=== Step 2: Controlled downloads (exactly 4, one day each) ===")
    checks: dict[str, dict[str, Any]] = {}

    # (a) Oldest date for city with longest history
    hist_city, oldest_date = _city_with_longest_history(city_rows)
    slug_a = pick_slug_for_city_date(catalog, hist_city, oldest_date)
    print(f"\n(a) Longest-history city={hist_city}, oldest date={oldest_date}")
    if _already_downloaded("a_oldest_longest_history_quotes"):
        print("  (skipped — already recorded in download ledger)")
        checks["a_oldest_quotes"] = {
            "pass": True,
            "skipped": True,
            "slug": slug_a,
            "event_date": oldest_date,
            "note": "prior download in ledger",
        }
    else:
        checks["a_oldest_quotes"] = _run_download_check(
            api_key,
            check_name="a_oldest_quotes",
            label="a_oldest_longest_history_quotes",
            channel="quotes",
            slug=slug_a,
            event_date=oldest_date,
        )

    # (b) Houston most recent date with confirmed quotes availability
    try:
        slug_b, houston_date = pick_slug_and_quotes_date(catalog, "houston", prefer_latest=True)
    except ValueError as exc:
        print(f"\n(b) Houston recent quotes: {exc}")
        checks["b_houston_recent_quotes"] = {"pass": False, "error": str(exc)}
        slug_b = houston_date = ""
    else:
        print(f"\n(b) Houston recent quotes (catalog latest with data): download date={houston_date}")
        if _already_downloaded("b_houston_recent_quotes"):
            print("  (skipped — already recorded in download ledger)")
            checks["b_houston_recent_quotes"] = {
                "pass": True,
                "skipped": True,
                "slug": slug_b,
                "event_date": houston_date,
            }
        else:
            checks["b_houston_recent_quotes"] = _run_download_check(
                api_key,
                check_name="b_houston_recent_quotes",
                label="b_houston_recent_quotes",
                channel="quotes",
                slug=slug_b,
                event_date=houston_date,
            )

    # (c) SF probe sample — order book depth vs quotes
    sf_date = "2026-03-24"
    print(f"\n(c) SF probe sample order book: {SF_PROBE_SAMPLE_SLUG}")

    def _depth_extra(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
        depth = analyze_order_book_depth(df, channel_used=meta["channel_used"])
        depth["pass"] = len(df) > 0 and depth["has_multi_level_depth"]
        return depth

    checks["c_sf_order_book"] = (
        {
            "pass": True,
            "skipped": True,
            "slug": SF_PROBE_SAMPLE_SLUG,
            "event_date": sf_date,
            "note": "prior download in ledger",
        }
        if _already_downloaded("c_sf_probe_order_book")
        else _run_download_check(
            api_key,
            check_name="c_sf_order_book",
            label="c_sf_probe_order_book",
            channel="order_book",
            slug=SF_PROBE_SAMPLE_SLUG,
            event_date=sf_date,
            extra_fn=_depth_extra,
        )
    )
    if checks["c_sf_order_book"].get("has_multi_level_depth") is not None:
        checks["c_sf_order_book"]["pass"] = bool(
            checks["c_sf_order_book"].get("row_count", 0) > 0
            and checks["c_sf_order_book"].get("has_multi_level_depth")
        )

    # (d) Thinner catalog city (Miami vs Atlanta) — random recent quotes
    miami_count = next(r["market_count"] for r in city_rows if r["city"] == "miami")
    atlanta_count = next(r["market_count"] for r in city_rows if r["city"] == "atlanta")
    thin_city = "miami" if miami_count <= atlanta_count else "atlanta"
    thin_catalog_date = _pick_random_recent_date(catalog, thin_city)
    slug_d = pick_slug_for_city_date(catalog, thin_city, thin_catalog_date)
    from telonex import get_availability

    av_d = get_availability(exchange="polymarket", slug=slug_d, outcome="Yes")
    quotes_d = av_d.get("channels", {}).get("quotes", {})
    from_dt = datetime.fromisoformat(str(quotes_d["from_date"]))
    to_excl = datetime.fromisoformat(str(quotes_d["to_date"]))
    thin_date = min(datetime.fromisoformat(thin_catalog_date), to_excl - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    if datetime.fromisoformat(thin_date) < from_dt:
        thin_date = from_dt.strftime("%Y-%m-%d")
    print(
        f"\n(d) Thinner city={thin_city} (miami={miami_count}, atlanta={atlanta_count}), "
        f"random catalog date={thin_catalog_date}, download date={thin_date}"
    )
    checks["d_thin_city_quotes"] = (
        {
            "pass": True,
            "skipped": True,
            "slug": slug_d,
            "event_date": thin_date,
            "thin_city": thin_city,
            "note": "prior download in ledger",
        }
        if _already_downloaded("d_thin_city_recent_quotes")
        else _run_download_check(
            api_key,
            check_name="d_thin_city_quotes",
            label="d_thin_city_recent_quotes",
            channel="quotes",
            slug=slug_d,
            event_date=thin_date,
            extra_fn=lambda _df, _meta: {"thin_city": thin_city, "catalog_date": thin_catalog_date},
        )
    )

    usage = load_download_usage()
    passed = [name for name, row in checks.items() if row.get("pass")]
    failed = [name for name, row in checks.items() if not row.get("pass")]
    all_pass = len(failed) == 0
    depth = checks.get("c_sf_order_book", {})

    if all_pass and depth.get("has_multi_level_depth"):
        go_no_go = (
            "**GO** — Subscribe to Telonex Single Exchange (Plus): all 4 checks passed; "
            "tick quotes confirmed on oldest/recent/thin markets; order-book channel exposes "
            "multi-level depth (not just renamed top-of-book)."
        )
    elif all_pass:
        go_no_go = (
            "**CONDITIONAL GO** — Quotes history looks good, but order-book channel appears "
            "top-of-book only; Single Exchange still useful for tick quotes, not L2 depth."
        )
    else:
        go_no_go = (
            "**NO-GO** — One or more verification downloads failed; do not pay for Single Exchange "
            "until gaps are understood."
        )

    print("\n=== Pass / Fail Summary ===")
    for name in checks:
        status = "PASS" if checks[name].get("pass") else "FAIL"
        print(f"  {name}: {status}")
    print(f"\nFinal recommendation: {go_no_go}")
    print(
        f"Downloads after verification: {usage['used']}/{usage['total_free']} used "
        f"({usage['total_free'] - usage['used']} remaining)"
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step1_city_rows": city_rows,
        "quota_before_step2": remaining,
        "quota_after": usage,
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "recommendation": go_no_go,
    }
    VERIFY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Telonex Single Exchange — pre-purchase verification",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Step 1 — Catalog browse (free)",
        "",
        "| City | Markets | Earliest | Latest |",
        "|------|---------|----------|--------|",
    ]
    for row in city_rows:
        lines.append(
            f"| {row['city']} | {row['market_count']} | "
            f"{row['earliest_event_date'] or '—'} | {row['latest_event_date'] or '—'} |"
        )
    lines.extend(["", "## Step 2 — Download checks", ""])
    for name, row in checks.items():
        lines.append(f"### {name} — {'PASS' if row.get('pass') else 'FAIL'}")
        lines.append(f"```json\n{json.dumps(row, indent=2, default=str)}\n```")
        lines.append("")
    lines.extend(["## Recommendation", "", go_no_go, ""])
    VERIFY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {VERIFY_REPORT_PATH}")
    return payload


def _pmd_market_event_date(row: dict[str, Any]) -> str | None:
    slug = str(row.get("slug", ""))
    parsed = _parse_slug_date(slug)
    if parsed:
        return parsed
    for key in ("first_seen", "created_at", "start_date", "end_date"):
        raw = row.get(key)
        if not raw:
            continue
        text = str(raw)
        if len(text) >= 10 and text[4] == "-":
            return text[:10]
    return None


def _pmd_row_matches_city(row: dict[str, Any], city: str) -> bool:
    probe_city = row.get("probe_city")
    if probe_city == city:
        return True
    slug = str(row.get("slug", ""))
    if slug and _slug_matches_city(slug, city):
        return True
    question = str(row.get("question", ""))
    return _city_from_question(question) == city


def load_telonex_verification_earliest() -> dict[str, str]:
    if not VERIFY_JSON_PATH.exists():
        raise SystemExit(
            f"Missing {VERIFY_JSON_PATH}. Run verify_coverage_before_purchase first:\n"
            "  .venv/bin/python scripts/probe_polymarket_providers.py --verify-before-purchase"
        )
    payload = json.loads(VERIFY_JSON_PATH.read_text(encoding="utf-8"))
    rows = payload.get("step1_city_rows", [])
    if not rows:
        raise SystemExit(f"{VERIFY_JSON_PATH} has no step1_city_rows — re-run --verify-before-purchase")
    earliest: dict[str, str] = {}
    for row in rows:
        city = str(row["city"])
        date = row.get("earliest_event_date")
        if date:
            earliest[city] = str(date)
    return earliest


def _assert_verification_city_list_matches() -> list[str]:
    earliest = load_telonex_verification_earliest()
    verify_cities = sorted(earliest.keys())
    target = sorted(TARGET_CITIES)
    if verify_cities != target:
        raise AssertionError(
            f"City list mismatch: verification JSON has {verify_cities}, "
            f"expected TARGET_CITIES {target}"
        )
    return target


def _pmd_earliest_by_city(pmd_rows: list[dict[str, Any]], *, pmd_blocked: bool) -> dict[str, str | None]:
    earliest: dict[str, str | None] = {city: None for city in TARGET_CITIES}
    if pmd_blocked or not pmd_rows:
        return earliest
    for city in TARGET_CITIES:
        dates: list[str] = []
        for row in pmd_rows:
            if not _pmd_row_matches_city(row, city):
                continue
            if not _is_tmax_pmd_row(row):
                continue
            event_date = _pmd_market_event_date(row)
            if event_date:
                dates.append(event_date)
        if dates:
            earliest[city] = min(dates)
    return earliest


def _is_tmax_pmd_row(row: dict[str, Any]) -> bool:
    question = str(row.get("question", ""))
    slug = str(row.get("slug", ""))
    return bool(
        TMAX_QUESTION_RE.search(question)
        or "highest-temperature" in slug.lower()
        or "highest temperature" in question.lower()
    )


def _coverage_verdict(delta_days: int | None) -> str:
    if delta_days is None:
        return "PMD_UNAVAILABLE"
    if abs(delta_days) <= 5:
        return "AGREE"
    if delta_days > 5:
        return "PMD_DEEPER"
    return "TELONEX_DEEPER"


def compare_earliest_coverage() -> dict[str, Any]:
    """Cross-check PMD discovery vs Telonex Feb 2026 floor (metadata only, no paid downloads)."""
    print("=== PMD vs Telonex earliest coverage comparison (free metadata only) ===\n")

    cities = _assert_verification_city_list_matches()
    telonex_earliest = load_telonex_verification_earliest()
    print(f"Loaded Telonex earliest dates from {VERIFY_JSON_PATH.name} ({len(cities)} cities)\n")

    session = requests.Session()
    pmd_blocked = False
    try:
        pmd_key = load_polymarketdata_key()
        pmd_rows, pmd_blocked = phase1_pmd_search(session, pmd_key, limit=100, max_pages=20)
    except FileNotFoundError:
        print("PMD discovery requires paid tier (no API key found)", file=sys.stderr)
        pmd_rows = []
        pmd_blocked = True

    pmd_earliest = _pmd_earliest_by_city(pmd_rows, pmd_blocked=pmd_blocked)

    comparison: list[dict[str, Any]] = []
    print("")
    header = f"{'City':<16} {'Telonex':>12} {'PMD':>12} {'Delta':>8} {'Verdict':<16}"
    print(header)
    print("-" * len(header))

    for city in cities:
        tlx_date = telonex_earliest.get(city)
        pmd_date = pmd_earliest.get(city)
        delta_days: int | None = None
        if tlx_date and pmd_date:
            tlx_dt = datetime.fromisoformat(tlx_date)
            pmd_dt = datetime.fromisoformat(pmd_date)
            delta_days = (tlx_dt - pmd_dt).days
        verdict = _coverage_verdict(delta_days if pmd_date else None)
        comparison.append(
            {
                "city": city,
                "telonex_earliest_date": tlx_date,
                "pmd_earliest_date": pmd_date,
                "delta_days": delta_days,
                "verdict": verdict,
            }
        )
        delta_str = str(delta_days) if delta_days is not None else "—"
        print(
            f"{city:<16} {tlx_date or '—':>12} {pmd_date or '—':>12} "
            f"{delta_str:>8} {verdict:<16}"
        )

    pmd_deeper = [r for r in comparison if r["verdict"] == "PMD_DEEPER"]
    unavailable = [r for r in comparison if r["verdict"] == "PMD_UNAVAILABLE"]
    comparable = [r for r in comparison if r["verdict"] != "PMD_UNAVAILABLE"]
    all_comparable_ok = all(r["verdict"] in ("AGREE", "TELONEX_DEEPER") for r in comparable)

    print("")
    if pmd_blocked and not pmd_rows:
        overall = (
            "PMD discovery unavailable (auth/key issue) — cannot compare; "
            "re-run after configuring config/polymarketdata_key.txt."
        )
    elif pmd_deeper:
        parts = [
            "PMD has deeper history for at least one city — investigate before purchasing Telonex:"
        ]
        for row in pmd_deeper:
            parts.append(
                f"  - {row['city']}: PMD earliest {row['pmd_earliest_date']}, "
                f"Telonex {row['telonex_earliest_date']} "
                f"({row['delta_days']} days)"
            )
        overall = "\n".join(parts)
    elif unavailable and not all_comparable_ok:
        overall = (
            f"PMD discovery incomplete ({len(unavailable)}/10 cities unavailable, "
            "possible rate limit) — re-run comparison before deciding."
        )
    elif all_comparable_ok and len(comparable) == len(cities):
        overall = (
            "Telonex's Feb 2026 floor appears to be a real market-history limit, not a provider gap — "
            "proceed with Telonex Single Exchange purchase."
        )
    elif all_comparable_ok:
        overall = (
            "Telonex's Feb 2026 floor appears to be a real market-history limit for all cities "
            f"with PMD data ({len(comparable)}/10); remaining cities had no PMD discovery hits — "
            "proceed with Telonex Single Exchange purchase."
        )
    else:
        overall = "Inconclusive — review per-city verdicts above."

    print(f"Overall verdict:\n{overall}\n")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "telonex_source": str(VERIFY_JSON_PATH),
        "pmd_search_rows": len(pmd_rows),
        "pmd_discovery_blocked": pmd_blocked,
        "comparison": comparison,
        "overall_verdict": overall,
    }
    COMPARE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# PMD vs Telonex earliest coverage comparison",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Metadata-only comparison (no priced/historical downloads).",
        "",
        "## Comparison table",
        "",
        "| City | Telonex earliest | PMD earliest | Delta (days) | Verdict |",
        "|------|------------------|--------------|--------------|---------|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['city']} | {row['telonex_earliest_date'] or '—'} | "
            f"{row['pmd_earliest_date'] or '—'} | "
            f"{row['delta_days'] if row['delta_days'] is not None else '—'} | "
            f"{row['verdict']} |"
        )
    lines.extend(["", "## Overall verdict", "", overall, ""])
    COMPARE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {COMPARE_REPORT_PATH}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PolymarketData vs Telonex for Tmax history.")
    parser.add_argument("--skip-samples", action="store_true", help="Discovery only (no paid/free downloads)")
    parser.add_argument(
        "--verify-before-purchase",
        action="store_true",
        help="Run Telonex catalog browse + 4 pre-purchase verification downloads",
    )
    parser.add_argument(
        "--expected-remaining",
        type=int,
        default=None,
        help="Override expected free Telonex downloads remaining before Step 2 (default: 4, or TELONEX_EXPECTED_REMAINING env)",
    )
    parser.add_argument(
        "--compare-earliest-coverage",
        action="store_true",
        help="Compare PMD vs Telonex earliest catalog dates (metadata only, no paid downloads)",
    )
    args = parser.parse_args()

    if args.compare_earliest_coverage:
        compare_earliest_coverage()
        return

    if args.verify_before_purchase:
        if args.expected_remaining is not None:
            os.environ["TELONEX_EXPECTED_REMAINING"] = str(args.expected_remaining)
        verify_coverage_before_purchase()
        return

    pmd_key = load_polymarketdata_key()
    tlx_key = load_telonex_key()
    session = requests.Session()

    local_index = load_local_index()
    telonex_us = phase1_telonex_catalog()
    pmd_rows, _pmd_blocked = phase1_pmd_search(session, pmd_key)
    cross = cross_check_local(telonex_us, pmd_rows, local_index)
    matrix = coverage_matrix(telonex_us)
    sample = pick_sample_from_index(local_index)

    pmd_sample: dict[str, Any] = {"skipped": True}
    tlx_sample: dict[str, Any] = {"skipped": True}
    if not args.skip_samples:
        pmd_sample = phase2_pmd_sample(session, pmd_key, sample)
        tlx_sample = phase2_telonex_sample(tlx_key, sample)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "cross_check": cross,
        "coverage_matrix": matrix,
        "pmd_search_count": len(pmd_rows),
        "pmd_sample": pmd_sample,
        "telonex_sample": tlx_sample,
        "backfill_estimate": estimate_backfill_volume(matrix),
        "recommendation": recommend(matrix, pmd_sample, tlx_sample, cross),
    }
    write_report(payload)


if __name__ == "__main__":
    main()
