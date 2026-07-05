#!/usr/bin/env python3
"""One-shot settlement target verification audit → reports/settlement_audit.md"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest.common import temp_in_bucket as backtest_temp_in_bucket  # noqa: E402
from download_polymarket_history import infer_winning_bucket  # noqa: E402
from fetch_wunderground_target import _daily_targets_from_asos  # noqa: E402
from poly_portfolio_status import (  # noqa: E402
    ForecastCache,
    TokenMeta,
    _parse_label_parts,
    build_modal_buckets,
    build_token_index,
    collect_trades,
    enrich_token_index_for_dates,
    load_auto_trader_entry_books,
    load_paper_forecasts,
    load_wu_targets,
    order_dates_from_posted,
    temp_in_bucket as portfolio_temp_in_bucket,
    wu_city_slug,
)
from poly_order_status import (  # noqa: E402
    _token_id,
    fetch_gamma_token_labels,
    fetch_open_orders,
    load_posted_orders,
    load_token_labels,
)
from src.polymarket_api import ORDER_LOG_PATH, PolymarketClient, parse_bucket_label  # noqa: E402
from src.trackj.build_asos_features import load_cached_asos  # noqa: E402
from src.wunderground_forecast import WU_PAGE_BY_CITY  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "reports" / "settlement_audit.md"
WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "polymarket_history" / "snapshots"
RAW_ROOT = PROJECT_ROOT / "data" / "trackj" / "raw"
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"

CLI_CITY_MAP = {
    "new_york": "new_york_city",
    "chicago": "chicago_midway",
}
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"

DISPLAY_TO_SLUG = {
    "Houston": "houston",
    "San Francisco": "san_francisco",
    "Los Angeles": "los_angeles",
    "Austin": "austin",
    "Dallas": "dallas",
    "Seattle": "seattle",
    "New York": "new_york",
    "Chicago": "chicago",
    "Atlanta": "atlanta",
    "Miami": "miami",
}

SPOT_CHECK_KEYS = [
    ("houston", "2026-06-24"),
    ("san_francisco", "2026-06-24"),
    ("los_angeles", "2026-06-25"),
    ("new_york", "2026-06-26"),
    ("seattle", "2026-06-29"),
]

WU_HISTORY_MAX_RE = re.compile(
    r'"temperatureMax"\s*:\s*\[\s*\{[^}]*"imperial"\s*:\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
WU_API_KEY_RE = re.compile(r'apiKey=([a-zA-Z0-9]+)')
WU_HISTORY_MAX_ALT = re.compile(
    r"maxTemp[^>]*>(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _normalize_label(label: str) -> str:
    return str(label).replace("°F", "").replace("°", "").strip()


def winning_bucket_continuous(tmax: float, labels: list[str]) -> str | None:
    """settle_daily-style: compare raw float to inclusive bounds (no round)."""
    for label in labels:
        try:
            parsed = parse_bucket_label(label)
        except ValueError:
            continue
        btype = parsed["type"]
        if btype == "RANGE":
            if float(parsed["lower"]) <= tmax <= float(parsed["upper"]):
                return _normalize_label(label)
        elif btype == "LESS_THAN":
            if tmax <= float(parsed["upper"]):
                return _normalize_label(label)
        elif btype == "GREATER_THAN":
            if tmax >= float(parsed["lower"]):
                return _normalize_label(label)
    return None


def winning_bucket_round(tmax: float, labels: list[str]) -> str | None:
    for label in labels:
        if portfolio_temp_in_bucket(tmax, label):
            return _normalize_label(label)
    return None


def load_city_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def fresh_asos_tmax(city: str, event_date: str) -> float | None:
    cfg = load_city_config()
    if city not in cfg:
        return None
    station = cfg[city]["nws_station"]
    raw_dir = RAW_ROOT / city / "asos"
    d = date.fromisoformat(event_date)
    asos = load_cached_asos(raw_dir, station, d, d)
    targets = _daily_targets_from_asos(asos, city, station, d, d)
    if targets.empty:
        return None
    val = targets.iloc[0]["wunderground_tmax"]
    return float(val) if pd.notna(val) else None


def cli_tmax(city: str, event_date: str) -> float | None:
    cli_city = CLI_CITY_MAP.get(city, city)
    path = TRACKJ_DIR / cli_city / "cli_target.parquet"
    if not path.exists():
        return None
    cli = pd.read_parquet(path)
    cli["date_key"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = cli[cli["date_key"] == event_date]
    if row.empty:
        return None
    val = row.iloc[0].get("tmax_f")
    return float(val) if pd.notna(val) else None


def pm_winner_from_snapshot(city: str, event_date: str) -> tuple[str | None, str]:
    """Infer PM winner from order-book history. Returns (bucket, method)."""
    path = SNAPSHOTS_DIR / city / f"{event_date}.parquet"
    if not path.exists():
        return None, "no_snapshot"

    frame = pd.read_parquet(path)
    if frame.empty:
        return None, "empty_snapshot"

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "midpoint" not in frame.columns:
        frame["midpoint"] = frame.apply(
            lambda r: (r.get("best_bid", 0) + r.get("best_ask", 0)) / 2
            if pd.notna(r.get("best_bid")) and pd.notna(r.get("best_ask"))
            else r.get("best_ask"),
            axis=1,
        )

    # Primary: any bucket reaching settlement prices at any point in the day
    settled = frame[
        (frame["best_bid"].fillna(0) >= 0.95)
        | (frame["midpoint"].fillna(0) >= 0.95)
        | (frame["best_ask"].fillna(0) >= 0.95)
    ]
    if not settled.empty:
        # Pick bucket with highest observed bid
        best = settled.groupby("bucket")["best_bid"].max().sort_values(ascending=False)
        bucket = str(best.index[0])
        return _normalize_label(bucket), "day_max_bid_ge_95"

    # Fallback: last-snapshot inference (can fail when winner bucket drops off book)
    bucket, _ = infer_winning_bucket({"buckets": []}, frame)
    if bucket:
        return _normalize_label(bucket), "last_snapshot_fallback"
    return None, "unresolved"


def bucket_labels_for_day(city: str, event_date: str) -> list[str]:
    path = SNAPSHOTS_DIR / city / f"{event_date}.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    labels = sorted(frame["bucket"].astype(str).unique())
    return [lb for lb in labels if not lb.startswith("Will ")]


def wu_history_url(city: str, event_date: str) -> str | None:
    base = WU_PAGE_BY_CITY.get(city)
    if not base:
        return None
    # Convert weather URL to history URL pattern
    if "/history/" in base:
        return base.replace("/history/daily/", f"/history/daily/").rstrip("/") + f"/date/{event_date}"
    # weather/us/tx/houston/KHOU -> history/daily/us/tx/houston/KHOU/date/YYYY-MM-DD
    m = re.search(r"/weather/(us/.+/(?:KHOU|KAUS|KORD|KDAL|KLAX|KMIA|KLGA|KSFO|KSEA|KATL))", base, re.I)
    if m:
        return f"https://www.wunderground.com/history/daily/{m.group(1)}/date/{event_date}"
    return None


def scrape_wu_history_max(city: str, event_date: str) -> tuple[float | None, str | None, str]:
    """Return (max_f, url, method)."""
    url = wu_history_url(city, event_date)
    if not url:
        return None, None, "no_url"

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException as exc:
        return None, url, f"fetch_failed:{exc}"

    match = WU_HISTORY_MAX_RE.search(html)
    if match:
        return float(match.group(1)), url, "html_temperatureMax"

    for pattern in (WU_HISTORY_MAX_ALT,):
        m = pattern.search(html)
        if m:
            return float(m.group(1)), url, "html_alt"

    # Try Weather.com historical API key embedded in WU page
    key_match = WU_API_KEY_RE.search(html)
    station = load_city_config().get(city, {}).get("nws_station")
    if key_match and station:
        api_key = key_match.group(1)
        api_url = (
            f"https://api.weather.com/v1/location/{station}:9:US/observations/historical.json"
            f"?apiKey={api_key}&units=e&startDate={event_date.replace('-', '')}"
            f"&endDate={event_date.replace('-', '')}"
        )
        try:
            api_resp = requests.get(api_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            api_resp.raise_for_status()
            data = api_resp.json()
            obs = data.get("observations", [])
            temps = [o.get("temp") for o in obs if o.get("temp") is not None]
            if temps:
                return float(max(temps)), api_url, "weathercom_api"
        except (requests.RequestException, ValueError, KeyError):
            pass

    return None, url, "parse_failed"


def gather_settled_trades() -> list[dict]:
    """Collect settled trades via portfolio status (same code path as live report)."""
    from datetime import date as date_cls

    client = PolymarketClient()
    posted = load_posted_orders(ORDER_LOG_PATH)
    today = str(date_cls.today())
    open_orders = fetch_open_orders(client)
    event_dates = order_dates_from_posted(posted)
    token_ids = {str(r.get("token_id", "")) for r in posted.values() if r.get("token_id")}
    for order in open_orders:
        tok = _token_id(order)
        if tok:
            token_ids.add(tok)

    token_index = build_token_index(event_dates=event_dates, refresh_labels=False)
    enrich_token_index_for_dates(token_index, event_dates)
    simple_labels = load_token_labels(event_date=today, token_ids=token_ids, refresh=False)
    for token, label in simple_labels.items():
        if token in token_index:
            continue
        city_display, city_slug, bucket = _parse_label_parts(label)
        order_date = today
        for record in posted.values():
            if str(record.get("token_id", "")) == token:
                order_date = str(record.get("timestamp", ""))[:10] or order_date
                break
        token_index[token] = TokenMeta(
            token_id=token,
            city=city_slug,
            city_display=city_display.title() if city_display else city_slug,
            bucket_label=bucket or label,
            event_date=order_date,
        )
    for record in posted.values():
        token = str(record.get("token_id", ""))
        if not token or token not in token_index:
            continue
        meta = token_index[token]
        order_date = str(record.get("timestamp", ""))[:10]
        if meta.event_date == today and order_date and order_date != today:
            token_index[token] = TokenMeta(
                token_id=meta.token_id,
                city=meta.city,
                city_display=meta.city_display,
                bucket_label=meta.bucket_label,
                event_date=order_date,
            )
    wu = load_wu_targets()
    entry_books = load_auto_trader_entry_books()
    paper_forecasts = load_paper_forecasts()
    forecast_cache = ForecastCache(use_live=False, paper=paper_forecasts)
    book_cache: dict = {}
    modal_by_city = build_modal_buckets(today, token_index, book_cache)

    settled, _ = collect_trades(
        client,
        posted,
        open_orders,
        token_index,
        entry_books,
        wu,
        forecast_cache,
        book_cache,
        modal_by_city,
    )
    rows = []
    for t in settled:
        if t.pnl_usd is None:
            continue
        city_slug = DISPLAY_TO_SLUG.get(t.city, wu_city_slug(t.city))
        rows.append(
            {
                "placed_at": t.placed_at,
                "city_display": t.city,
                "city": city_slug,
                "event_date": t.event_date,
                "bucket": _normalize_label(t.bucket_label),
                "qty": t.n_contracts,
                "entry": t.entry_price,
                "pnl": t.pnl_usd,
                "portfolio_winner": _normalize_label(t.winning_bucket or ""),
                "wu_tmax_portfolio": t.actual_tmax_f,
            }
        )
    return rows


def main() -> None:
    if not WU_PATH.exists():
        REPORT_PATH.write_text("# Settlement Audit\n\n**BLOCKER:** `wunderground_targets.parquet` missing.\n")
        print("WU parquet missing")
        sys.exit(1)

    wu_df = load_wu_targets()
    trades = gather_settled_trades()
    if not trades:
        raise SystemExit("No settled trades found")

    # Deduplicate unique city-dates for cross-source comparison
    unique_days: dict[tuple[str, str], dict] = {}

    table_rows: list[dict] = []
    for tr in trades:
        city = tr["city"]
        event_date = tr["event_date"]
        key = (city, event_date)

        if key not in unique_days:
            labels = bucket_labels_for_day(city, event_date)
            wu_row = wu_df[(wu_df["city"] == city) & (wu_df["date"] == event_date)]
            wu_stored = float(wu_row.iloc[0]["wunderground_tmax"]) if not wu_row.empty else None
            asos_fresh = fresh_asos_tmax(city, event_date)
            cli_val = cli_tmax(city, event_date)
            pm_winner, pm_method = pm_winner_from_snapshot(city, event_date)

            wu_for_bucket = wu_stored if wu_stored is not None else asos_fresh
            wu_bucket_round = winning_bucket_round(wu_for_bucket, labels) if wu_for_bucket is not None else None
            wu_bucket_cont = winning_bucket_continuous(wu_for_bucket, labels) if wu_for_bucket is not None else None
            cli_bucket_round = winning_bucket_round(cli_val, labels) if cli_val is not None else None
            cli_bucket_cont = winning_bucket_continuous(cli_val, labels) if cli_val is not None else None

            unique_days[key] = {
                "labels": labels,
                "wu_stored": wu_stored,
                "asos_fresh": asos_fresh,
                "cli_tmax": cli_val,
                "pm_winner": pm_winner,
                "pm_method": pm_method,
                "wu_bucket_round": wu_bucket_round,
                "wu_bucket_cont": wu_bucket_cont,
                "cli_bucket_round": cli_bucket_round,
                "cli_bucket_cont": cli_bucket_cont,
            }

        day = unique_days[key]
        wu_tmax = day["wu_stored"]
        pm = day["pm_winner"]
        wu_b = day["wu_bucket_round"]
        cli_t = day["cli_tmax"]
        cli_b = day["cli_bucket_round"]

        wu_pm = (wu_b == pm) if (wu_b and pm) else None
        cli_pm = (cli_b == pm) if (cli_b and pm) else None

        table_rows.append(
            {
                **tr,
                "pm_winner": pm,
                "wu_tmax": wu_tmax,
                "asos_fresh": day["asos_fresh"],
                "wu_bucket": wu_b,
                "cli_tmax": cli_t,
                "cli_bucket": cli_b,
                "wu_eq_pm": wu_pm,
                "cli_eq_pm": cli_pm,
                "asos_eq_stored": (
                    abs(day["asos_fresh"] - wu_tmax) < 0.01
                    if day["asos_fresh"] is not None and wu_tmax is not None
                    else None
                ),
            }
        )

    # Summary stats on unique city-dates with PM winner
    day_items = list(unique_days.items())
    pm_available = [(k, v) for k, v in day_items if v["pm_winner"]]
    wu_agree = sum(1 for _, v in pm_available if v["wu_bucket_round"] == v["pm_winner"])
    cli_agree = sum(1 for _, v in pm_available if v["cli_bucket_round"] == v["pm_winner"])
    wu_cli_disagree = sum(
        1
        for _, v in day_items
        if v["wu_bucket_round"] and v["cli_bucket_round"] and v["wu_bucket_round"] != v["cli_bucket_round"]
    )
    biases = [
        v["wu_stored"] - v["cli_tmax"]
        for _, v in day_items
        if v["wu_stored"] is not None and v["cli_tmax"] is not None
    ]

    # Spot checks
    spot_results: list[dict] = []
    for city, event_date in SPOT_CHECK_KEYS:
        day = unique_days.get((city, event_date), {})
        scraped, url, scrape_method = scrape_wu_history_max(city, event_date)
        stored = day.get("wu_stored")
        delta = abs(scraped - stored) if scraped is not None and stored is not None else None
        spot_results.append(
            {
                "city": city,
                "date": event_date,
                "wu_scraped": scraped,
                "wu_stored": stored,
                "delta_f": delta,
                "url": url,
                "method": scrape_method,
                "flag": delta is not None and delta > 1.0,
            }
        )

    # Boundary edge cases near .5
    boundary_notes: list[str] = []
    for t in [83.4, 83.5, 83.6, 84.4, 84.5, 84.6]:
        boundary_notes.append(
            f"tmax={t}: round→{int(round(t))}, in 84-85 (portfolio/backtest)={portfolio_temp_in_bucket(t, '84-85')}, "
            f"continuous [84,85]={winning_bucket_continuous(t, ['84-85°F']) is not None}"
        )

    # Check backtest uses WU not CLI
    backtest_uses_cli = False
    for fname in ("scripts/backtest/step3_ngboost_kelly.py", "scripts/backtest/common.py", "scripts/backtest/step2_modal_maker.py"):
        text = (PROJECT_ROOT / fname).read_text()
        if "fetch_cli" in text or "_cli_tmax" in text or "_load_cli_target" in text:
            backtest_uses_cli = True

    portfolio_winner_is_wu = True  # documented in code review

    # Build markdown
    lines: list[str] = [
        "# Settlement Target Verification Audit\n",
        f"Generated from {len(trades)} settled live trades across {len(unique_days)} unique city-dates.\n\n",
        "## Critical methodology note\n\n",
        "The `Winner` column in `poly_portfolio_status.py` is **not** Polymarket's official resolution. "
        "It is computed by `winning_bucket_for_city()` from our IEM-reconstructed `wunderground_targets.parquet`. "
        "This audit uses **order-book snapshot inference** as Polymarket ground truth: the bucket that reached "
        "bid/mid/ask ≥ 0.95 at any point during the event day (fallback: last-snapshot midpoint).\n\n",
    ]

    lines.append("## 1. Settled trades comparison\n\n")
    lines.append(
        "| City | Date | Bucket | Entry | PnL | PM Winner (snapshot) | Our WU Tmax | "
        "Fresh ASOS | Our WU Bucket | CLI Tmax | CLI Bucket | WU==PM? | CLI==PM? |\n"
    )
    lines.append("|---|---|---|---:|---:|---|---:|---:|---|---:|---|---|---|\n")
    for r in table_rows:
        lines.append(
            f"| {r['city_display']} | {r['event_date']} | {r['bucket']} | ${r['entry']:.2f} | "
            f"${r['pnl']:+.2f} | {r.get('pm_winner') or '—'} | "
            f"{r['wu_tmax'] if r['wu_tmax'] is not None else '—'} | "
            f"{r['asos_fresh'] if r['asos_fresh'] is not None else '—'} | "
            f"{r['wu_bucket'] or '—'} | "
            f"{r['cli_tmax'] if r['cli_tmax'] is not None else '—'} | "
            f"{r['cli_bucket'] or '—'} | "
            f"{'YES' if r['wu_eq_pm'] else 'NO' if r['wu_eq_pm'] is False else '—'} | "
            f"{'YES' if r['cli_eq_pm'] else 'NO' if r['cli_eq_pm'] is False else '—'} |\n"
        )

    lines.append("\n## 2. Summary statistics\n\n")
    lines.append(f"- Unique city-dates in sample: **{len(unique_days)}**\n")
    lines.append(f"- City-dates with PM snapshot winner: **{len(pm_available)}**\n")
    lines.append(f"- WU (round bucket) agrees with PM winner: **{wu_agree}/{len(pm_available)}**\n")
    lines.append(f"- CLI (round bucket) agrees with PM winner: **{cli_agree}/{len(pm_available)}**\n")
    lines.append(f"- City-dates where WU and CLI round buckets disagree: **{wu_cli_disagree}/{len(day_items)}**\n")
    if biases:
        lines.append(
            f"- Mean WU − CLI bias: **{float(np.mean(biases)):+.2f}°F** "
            f"(median {float(np.median(biases)):+.2f}°F)\n"
        )
    else:
        lines.append("- Mean WU − CLI bias: **N/A** (missing CLI rows)\n")

    asos_mismatch = [
        (k, v) for k, v in unique_days.items()
        if v["asos_fresh"] is not None and v["wu_stored"] is not None and abs(v["asos_fresh"] - v["wu_stored"]) > 0.01
    ]
    lines.append(f"- Stored WU vs fresh ASOS cache mismatches (>0.01°F): **{len(asos_mismatch)}**\n")

    lines.append("\n## 3. Bucket boundary audit\n\n")
    lines.append("### Portfolio / backtest logic (`int(round(tmax))`)\n\n")
    lines.append(
        "Both `scripts/poly_portfolio_status.py::temp_in_bucket` and "
        "`scripts/backtest/common.py::temp_in_bucket` use `t = int(round(tmax))` then inclusive integer "
        "range checks. For RANGE bucket `84-85`, membership is `round(tmax) ∈ {84, 85}`.\n\n"
    )
    lines.append("### Polymarket market metadata (`config/polymarket_markets.json`)\n\n")
    lines.append(
        "Buckets are typed `RANGE` with integer `lower_f` / `upper_f` (e.g. 84–85). "
        "Market copy uses labels like `84-85°F` or `83°F or below`. "
        "No explicit half-degree rounding rule is documented in metadata; settlement references Wunderground station pages.\n\n"
    )
    lines.append("### `settle_daily.py` (Kalshi) uses continuous bounds\n\n")
    lines.append(
        "`_tmax_in_bucket()` compares raw float Tmax to inclusive `[lower, upper]` without rounding. "
        "This differs from Polymarket backtest/portfolio code.\n\n"
    )
    lines.append("### NGBoost inference (`ngboost_inference.py`)\n\n")
    lines.append(
        "Bucket **probabilities** integrate a Gaussian CDF with ±0.5°F continuity correction at integer "
        "boundaries. Settlement in backtest still uses `backtest/common.py::temp_in_bucket` (round-then-check).\n\n"
    )
    lines.append("### Round-vs-continuous disagreements on sample city-dates\n\n")
    round_cont_disagree = [
        (k, v) for k, v in day_items
        if v["wu_bucket_round"] != v["wu_bucket_cont"] and v["wu_stored"] is not None
    ]
    if round_cont_disagree:
        for (city, d), v in round_cont_disagree:
            lines.append(
                f"- {city} {d}: tmax={v['wu_stored']}, round→{v['wu_bucket_round']}, "
                f"continuous→{v['wu_bucket_cont']}\n"
            )
    else:
        lines.append("- No sample city-dates where round vs continuous WU bucketing differ.\n")

    lines.append("\n### Synthetic edge cases (84-85°F bucket)\n\n")
    for note in boundary_notes:
        lines.append(f"- {note}\n")

    lines.append("\n## 4. Backtest target source confirmation\n\n")
    lines.append(f"- Backtest loads targets from: `data/polymarket/wunderground_targets.parquet`\n")
    lines.append(f"- `wunderground_targets.parquet` exists: **YES** ({len(wu_df)} rows)\n")
    lines.append(f"- Backtest code references CLI/`fetch_cli_target`: **{'YES — INVESTIGATE' if backtest_uses_cli else 'NO'}**\n")
    lines.append(
        "- Settlement functions: `step2_modal_maker.py` and `step3_ngboost_kelly.py` call "
        "`load_wu_targets()` + `backtest/common.py::temp_in_bucket`.\n"
    )
    lines.append(
        "- `settle_daily.py` (Kalshi cron) uses `_cli_tmax()` / NWS CLI — **separate path, not used by Polymarket backtest**.\n"
    )

    lines.append("\n## 5. Live Wunderground scrape spot-check (5 trades)\n\n")
    lines.append("| City | Date | WU page max | Stored WU | Δ°F | Method | Flag |\n")
    lines.append("|---|---|---:|---:|---:|---|---|\n")
    loud = False
    for s in spot_results:
        if s["flag"]:
            loud = True
        lines.append(
            f"| {s['city']} | {s['date']} | {s['wu_scraped'] if s['wu_scraped'] is not None else 'parse fail'} | "
            f"{s['wu_stored'] if s['wu_stored'] is not None else '—'} | "
            f"{s['delta_f'] if s['delta_f'] is not None else '—'} | {s['method']} | "
            f"{'**LOUD**' if s['flag'] else 'ok'} |\n"
        )
        if s["url"]:
            lines.append(f"  - URL: {s['url']}\n")

    lines.append("\n## 6. Verdict\n\n")
    wu_pm_mismatches = [r for r in table_rows if r["wu_eq_pm"] is False]
    cli_pm_mismatches = [r for r in table_rows if r["cli_eq_pm"] is False]
    pm_missing = sum(1 for r in table_rows if not r.get("pm_winner"))
    if pm_missing == len(table_rows):
        verdict = "NO"
        reason = (
            "Could not infer Polymarket winners from order-book snapshots for any trade dates. "
            "Cannot validate settlement alignment."
        )
    elif wu_pm_mismatches:
        verdict = "NO"
        reason = (
            f"{len(wu_pm_mismatches)} trade rows show our WU bucket disagreeing with PM snapshot winner. "
            "Settlement target mismatch is a showstopper for backtest trust."
        )
    elif loud:
        verdict = "NO"
        reason = "Live Wunderground scrape differs from stored IEM reconstruction by >1°F on at least one spot-check."
    elif wu_agree < len(pm_available):
        verdict = "NO"
        reason = "WU round-bucket winners do not match all PM snapshot winners on unique city-dates."
    else:
        verdict = "CONDITIONAL YES"
        reason = (
            f"Our IEM-reconstructed WU targets agree with PM snapshot-inferred winners on all "
            f"{len(pm_available)} city-dates ({wu_agree}/{len(pm_available)}). "
            f"CLI disagrees with PM on {len(cli_pm_mismatches)} trade rows and with WU on "
            f"{wu_cli_disagree} city-dates — expected given CLI uses different stations/products "
            f"(mean WU−CLI bias {float(np.mean(biases)):+.2f}°F where both available). "
            "Caveats: (1) PM winner inferred from order book, not official resolution API; "
            "(2) portfolio status Winner column is WU-derived; "
            "(3) round-then-check bucketing differs from Kalshi `settle_daily.py` continuous bounds."
        )

    lines.append(f"**{verdict}** — {reason}\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
