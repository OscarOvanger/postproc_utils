#!/usr/bin/env python3
"""Forward validation: reconstruct v5b trades on Jul 1-3 2026 using Telonex + Gamma."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backfill_polymarket_history import backfill_city_date, load_catalog  # noqa: E402
from backtest.common import (  # noqa: E402
    load_day_snapshot,
    quotes_at_entry,
    select_entry_snapshot,
)
from download_polymarket_history import SNAPSHOTS_DIR  # noqa: E402
from retroactive_paper_test import WuActualCache, bucket_settles_yes, settlement_pnl_usd  # noqa: E402
from run_daily_trade import (  # noqa: E402
    build_feature_vector_strict,
    load_city_config,
    load_deploy_config,
    load_models,
    predict_tmax_strict,
)
from src.polymarket_api import GAMMA_API, parse_bucket_label  # noqa: E402
from src.poly_trading_pipeline import (  # noqa: E402
    POLYMARKET_CITIES,
    _extract_bucket_label,
    _match_poly_city,
    _parse_event_date_from_title,
    _parse_json_field,
    _to_float,
    apply_wunderground_bias,
    compute_edge,
    compute_maker_entry_price,
    load_wunderground_bias,
    select_trades_poly,
    size_positions_poly,
)
from src.provider_keys import load_telonex_key  # noqa: E402
from src.rolling_bias import compute_rolling_bias  # noqa: E402

GAMMA_DIR = PROJECT_ROOT / "data" / "polymarket_history"
DEFAULT_DATES = ["2026-07-01", "2026-07-02", "2026-07-03"]
DEFAULT_BANKROLL = 86.63
TELONEX_CHANNEL = "book_snapshot_5"

POLY_TO_TRACKB: dict[str, str] = {
    "austin": "austin",
    "chicago": "chicago_midway",
    "houston": "houston",
    "los_angeles": "los_angeles",
    "new_york": "new_york_city",
    "san_francisco": "san_francisco",
}

CITY_DISPLAY: dict[str, str] = {
    "atlanta": "Atlanta",
    "austin": "Austin",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "houston": "Houston",
    "los_angeles": "Los Angeles",
    "miami": "Miami",
    "new_york": "New York",
    "san_francisco": "San Francisco",
    "seattle": "Seattle",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telonex retro validation for v5b (Jul 1-3 2026).")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="Event dates YYYY-MM-DD (default: Jul 1-3 2026)",
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Phase 2: use cached Telonex parquets only",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "gamma", "telonex", "validate"),
        default="all",
        help="Run one phase or all (default: all)",
    )
    return parser.parse_args()


def _month_day_label(event_date: str) -> str:
    dt = date.fromisoformat(event_date)
    return f"{dt.strftime('%B')} {dt.day}"


def _normalize_bucket_key(label: str) -> str:
    text = str(label).replace("°F", "").replace("°", "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def _city_search_name(city: str) -> str:
    if city == "new_york":
        return "New York"
    if city == "los_angeles":
        return "Los Angeles"
    if city == "san_francisco":
        return "San Francisco"
    return CITY_DISPLAY.get(city, city.replace("_", " ").title())


def _gamma_structure_path(event_date: str) -> Path:
    return GAMMA_DIR / f"gamma_structure_{event_date}.json"


def _parse_market(market: dict[str, Any], city: str, event_date: str, condition_id: str) -> dict[str, Any] | None:
    try:
        label = _extract_bucket_label(market)
        parsed = parse_bucket_label(label)
    except ValueError:
        return None

    token_ids = _parse_json_field(market.get("clobTokenIds"))
    outcome_prices = _parse_json_field(market.get("outcomePrices"))
    outcomes = _parse_json_field(market.get("outcomes"))
    if not token_ids:
        return None

    yes_index = 0
    if outcomes and str(outcomes[0]).lower() != "yes":
        yes_index = 1 if len(token_ids) > 1 else 0

    gamma_price = _to_float(outcome_prices[yes_index]) if outcome_prices else None
    tick_raw = market.get("orderPriceMinTickSize", "0.01")
    tick_size = str(tick_raw)

    return {
        "city": city,
        "event_date": event_date,
        "bucket_label": label,
        "bucket_key": _normalize_bucket_key(label),
        "bucket_type": parsed["type"],
        "bucket_lower_inclusive_f": parsed["lower"],
        "bucket_upper_inclusive_f": parsed["upper"],
        "question": str(market.get("question", "")),
        "slug": str(market.get("slug", "")),
        "yes_token_id": str(token_ids[yes_index]),
        "condition_id": str(market.get("conditionId") or condition_id),
        "gamma_price": gamma_price,
        "tick_size": tick_size,
        "outcome_prices": outcome_prices,
        "closed": bool(market.get("closed")),
    }


def fetch_gamma_structure(event_date: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    """Phase 1: Gamma event + bucket metadata for all 10 Polymarket cities."""
    sess = session or requests.Session()
    md = _month_day_label(event_date)
    year_hint = event_date[:4]
    cities: dict[str, Any] = {}

    for city in POLYMARKET_CITIES:
        query = f"Highest temperature in {_city_search_name(city)} on {md}"
        try:
            resp = sess.get(f"{GAMMA_API}/public-search", params={"q": query}, timeout=30)
            resp.raise_for_status()
            events = resp.json().get("events", [])
        except requests.RequestException as exc:
            print(f"  WARNING: Gamma search failed for {city}: {exc}")
            continue

        match = None
        for event in events:
            title = str(event.get("title", ""))
            parsed_date = _parse_event_date_from_title(title, year_hint=year_hint)
            if parsed_date == event_date and _match_poly_city(title) == city:
                match = event
                break
        if match is None:
            print(f"  WARNING: no Gamma event for {city} on {event_date}")
            continue

        event_id = match.get("id")
        try:
            full = sess.get(f"{GAMMA_API}/events/{event_id}", timeout=30).json()
        except requests.RequestException as exc:
            print(f"  WARNING: Gamma event fetch failed {city}: {exc}")
            continue

        condition_id = str(full.get("negRiskMarketID") or full.get("id", ""))
        markets: list[dict[str, Any]] = []
        for market in full.get("markets") or []:
            row = _parse_market(market, city, event_date, condition_id)
            if row is not None:
                markets.append(row)

        cities[city] = {
            "event_id": event_id,
            "title": full.get("title"),
            "slug": full.get("slug"),
            "closed": full.get("closed"),
            "markets": markets,
        }
        time.sleep(0.15)

    payload = {"event_date": event_date, "fetched_at": datetime.now(timezone.utc).isoformat(), "cities": cities}
    path = _gamma_structure_path(event_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Wrote {path} ({sum(len(v['markets']) for v in cities.values())} markets)")
    return payload


def load_gamma_structure(event_date: str) -> dict[str, Any]:
    path = _gamma_structure_path(event_date)
    if not path.exists():
        raise FileNotFoundError(f"Missing Gamma structure: {path}. Run phase gamma first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _slug_lookup(gamma: dict[str, Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for city, block in gamma.get("cities", {}).items():
        for market in block.get("markets", []):
            out[(city, market["bucket_key"])] = market["slug"]
    return out


def _ensure_telonex_snapshots(
    event_date: str,
    gamma: dict[str, Any],
    *,
    skip_fetch: bool,
    api_key: str | None,
    catalog: pd.DataFrame | None,
) -> list[str]:
    """Phase 2: ensure city-day parquets exist; return warnings."""
    warnings: list[str] = []
    slug_map = _slug_lookup(gamma)

    for city in POLYMARKET_CITIES:
        out_path = SNAPSHOTS_DIR / city / f"{event_date}.parquet"
        if out_path.exists():
            continue
        if skip_fetch:
            warnings.append(f"{city}/{event_date}: no cached snapshot (--skip-fetch)")
            continue
        if api_key is None:
            warnings.append(f"{city}/{event_date}: no Telonex key, cannot fetch")
            continue

        city_block = gamma.get("cities", {}).get(city)
        if not city_block:
            warnings.append(f"{city}/{event_date}: no Gamma structure")
            continue

        buckets: list[dict[str, str]] = []
        for market in city_block.get("markets", []):
            slug = market.get("slug") or slug_map.get((city, market["bucket_key"]))
            if not slug:
                continue
            buckets.append({"slug": slug, "bucket": _normalize_bucket_key(market["bucket_label"])})

        if not buckets:
            warnings.append(f"{city}/{event_date}: no bucket slugs")
            continue

        ok, status = backfill_city_date(
            api_key,
            city,
            event_date,
            buckets,
            channel=TELONEX_CHANNEL,
            force=False,
            show_progress=False,
        )
        if not ok:
            warnings.append(f"{city}/{event_date}: Telonex fetch failed ({status})")

    return warnings


@dataclass
class PriceQuote:
    city: str
    event_date: str
    bucket_label: str
    best_bid: float | None
    best_ask: float | None
    source: str  # telonex | gamma_fallback
    entry_ts: str | None = None


def _quotes_from_snapshot(city: str, event_date: str, gamma_markets: list[dict[str, Any]]) -> list[PriceQuote]:
    frame = load_day_snapshot(city, event_date)
    if frame is None or frame.empty:
        return []

    snap_rows, entry_ts, _ = select_entry_snapshot(frame, city, event_date)
    if snap_rows.empty:
        return []

    quotes = quotes_at_entry(snap_rows)
    by_bucket = {_normalize_bucket_key(str(r["bucket"])): r for _, r in quotes.iterrows()}
    out: list[PriceQuote] = []
    for market in gamma_markets:
        key = market["bucket_key"]
        row = by_bucket.get(key)
        if row is None:
            continue
        out.append(
            PriceQuote(
                city=city,
                event_date=event_date,
                bucket_label=market["bucket_label"],
                best_bid=_to_float(row.get("best_bid")),
                best_ask=_to_float(row.get("best_ask")),
                source="telonex",
                entry_ts=str(entry_ts),
            )
        )
    return out


def _gamma_fallback_quotes(city: str, event_date: str, gamma_markets: list[dict[str, Any]]) -> list[PriceQuote]:
    out: list[PriceQuote] = []
    for market in gamma_markets:
        gp = _to_float(market.get("gamma_price"))
        if gp is None:
            continue
        out.append(
            PriceQuote(
                city=city,
                event_date=event_date,
                bucket_label=market["bucket_label"],
                best_bid=None,
                best_ask=gp,
                source="gamma_fallback",
            )
        )
    return out


def build_market_df(
    event_date: str,
    gamma: dict[str, Any],
    *,
    skip_fetch: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Phase 2: Telonex BBO at ~10:10 local, with Gamma fallback flagged."""
    quality: dict[str, Any] = {
        "telonex_buckets": 0,
        "gamma_fallback_buckets": 0,
        "missing_cities": [],
        "warnings": [],
    }
    rows: list[dict[str, Any]] = []

    for city in POLYMARKET_CITIES:
        block = gamma.get("cities", {}).get(city)
        if not block:
            quality["missing_cities"].append(city)
            continue

        markets = block.get("markets", [])
        quotes = _quotes_from_snapshot(city, event_date, markets)
        quote_by_key = {_normalize_bucket_key(q.bucket_label): q for q in quotes}

        if not quotes:
            if SNAPSHOTS_DIR.joinpath(city, f"{event_date}.parquet").exists():
                quality["warnings"].append(f"{city}: snapshot exists but no entry-window quotes")
            else:
                quality["warnings"].append(
                    f"{city}: WARNING — using Gamma settlement prices (NOT valid 10:10 AM BBO)"
                )
            quotes = _gamma_fallback_quotes(city, event_date, markets)
            quote_by_key = {_normalize_bucket_key(q.bucket_label): q for q in quotes}

        for market in markets:
            key = market["bucket_key"]
            q = quote_by_key.get(key)
            if q is None:
                continue

            if q.source == "telonex":
                quality["telonex_buckets"] += 1
            else:
                quality["gamma_fallback_buckets"] += 1

            gamma_price = _to_float(market.get("gamma_price"))
            best_bid = q.best_bid
            best_ask = q.best_ask
            tick_size = float(market.get("tick_size") or 0.01)
            market_price = best_ask if best_ask is not None else (gamma_price or 0.0)
            spread = (
                round(best_ask - best_bid, 4)
                if best_bid is not None and best_ask is not None
                else None
            )

            rows.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "bucket_label": market["bucket_label"],
                    "bucket_type": market["bucket_type"],
                    "bucket_lower_inclusive_f": market["bucket_lower_inclusive_f"],
                    "bucket_upper_inclusive_f": market["bucket_upper_inclusive_f"],
                    "gamma_price": gamma_price,
                    "yes_mid_close": gamma_price,
                    "market_price": float(market_price),
                    "yes_bid_close": best_bid,
                    "yes_ask_close": best_ask,
                    "spread": spread,
                    "tick_size": str(market.get("tick_size", "0.01")),
                    "yes_token_id": market["yes_token_id"],
                    "condition_id": market["condition_id"],
                    "price_source": q.source,
                }
            )

    return pd.DataFrame(rows), quality


def fetch_poly_forecasts(
    poly_config: dict[str, Any],
    event_date: str,
    city_config: dict[str, Any],
) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    model_dir = PROJECT_ROOT / poly_config["model_dir"]
    forecasts: dict[str, int] = {}
    reasons: dict[str, str] = {}
    notes: dict[str, str] = {}

    for poly_city, trackb_city in POLY_TO_TRACKB.items():
        try:
            models, feature_cols = load_models(trackb_city, model_dir)
        except FileNotFoundError:
            reasons[poly_city] = "missing model artifacts"
            continue

        features, fail_reason = build_feature_vector_strict(trackb_city, event_date, feature_cols)
        if features is None:
            reasons[poly_city] = fail_reason or "feature build failed"
            continue

        pred = predict_tmax_strict(models, feature_cols, features)
        if pred is None:
            reasons[poly_city] = "prediction failed"
            continue
        forecasts[poly_city] = pred

    return forecasts, reasons, notes


def _poly_city_config(city_config: dict[str, Any]) -> dict[str, Any]:
    """Inject trackb sigma into poly city entries used by compute_edge."""
    out = dict(city_config)
    for poly_city, trackb_city in POLY_TO_TRACKB.items():
        if trackb_city not in city_config or poly_city not in out:
            continue
        sigma = city_config[trackb_city].get("trackb_sigma_f")
        if sigma is not None:
            out[poly_city] = {**out[poly_city], "trackb_sigma_f": sigma}
    return out


def apply_forecast_adjustments(
    forecasts: dict[str, int],
    event_date: str,
    poly_config: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    meta: dict[str, Any] = {}
    wu_bias = load_wunderground_bias()
    raw, adjusted, bias_applied = apply_wunderground_bias(forecasts, wu_bias)
    meta["raw_forecasts"] = raw
    meta["bias_applied"] = bias_applied

    halflife = int(poly_config.get("rolling_bias_halflife_days", 20))
    max_corr = float(poly_config.get("max_rolling_correction_f", 1.5))
    rolling_applied: dict[str, float] = {}
    final = dict(adjusted)
    for city in list(final.keys()):
        rolling = compute_rolling_bias(city, event_date, halflife, max_correction_f=max_corr)
        final[city] = int(round(final[city] - rolling))
        rolling_applied[city] = rolling
    meta["rolling_bias_applied"] = rolling_applied
    meta["forecasts"] = final
    return final, meta


def run_validation_day(
    event_date: str,
    bankroll: float,
    poly_config: dict[str, Any],
    city_config: dict[str, Any],
    gamma: dict[str, Any],
    wu_cache: WuActualCache,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Phase 3 for one date. Returns (trades with settlement, day_meta, updated_bankroll)."""
    market_df, price_quality = build_market_df(event_date, gamma, skip_fetch=True)

    forecasts, forecast_reasons, _notes = fetch_poly_forecasts(poly_config, event_date, city_config)
    if not forecasts:
        return [], {"event_date": event_date, "abort": "no_forecasts", "price_quality": price_quality}, bankroll

    forecasts, forecast_meta = apply_forecast_adjustments(forecasts, event_date, poly_config)
    market_reasons = {city: "no Polymarket market for date" for city in POLYMARKET_CITIES}
    for city in gamma.get("cities", {}):
        if city in market_df["city"].astype(str).unique():
            market_reasons.pop(city, None)

    edges, edge_reasons, _sanity = compute_edge(
        market_df,
        forecasts,
        _poly_city_config(city_config),
        poly_config,
        {**market_reasons, **forecast_reasons},
        event_date,
    )
    selected, _reasons = select_trades_poly(edges, poly_config, edge_reasons)
    sized = size_positions_poly(selected, bankroll, poly_config)

    settled: list[dict[str, Any]] = []
    for trade in sized:
        city = trade["city"]
        actual, actual_src = wu_cache.get(city, event_date)
        won: bool | None = None
        pnl: float | None = None
        pending = False
        if actual is not None:
            try:
                bucket = parse_bucket_label(trade["bucket_label"])
                won = bucket_settles_yes(actual, bucket)
                entry = float(trade.get("maker_entry_price") or trade["market_price"])
                pnl = settlement_pnl_usd(trade["n_contracts"], entry, won)
            except ValueError:
                pending = True
        else:
            pending = True

        price_src = "telonex"
        mrow = market_df[
            (market_df["city"] == city)
            & (market_df["bucket_label"].astype(str) == str(trade["bucket_label"]))
        ]
        if not mrow.empty:
            price_src = str(mrow.iloc[0].get("price_source", "telonex"))

        settled.append(
            {
                **trade,
                "event_date": event_date,
                "forecast_f": forecasts.get(city),
                "actual_tmax_f": actual,
                "actual_source": actual_src,
                "won": won,
                "pnl": pnl,
                "pending": pending,
                "price_source": price_src,
            }
        )

    day_pnl = sum(t["pnl"] for t in settled if t.get("pnl") is not None)
    return settled, {
        "event_date": event_date,
        "forecasts": forecasts,
        "forecast_meta": forecast_meta,
        "price_quality": price_quality,
        "n_edges": len(edges),
        "n_selected": len(sized),
    }, bankroll + day_pnl


def _fmt_money(value: float, *, signed: bool = True) -> str:
    if signed and value >= 0:
        return f"+${value:.2f}"
    if signed:
        return f"-${abs(value):.2f}"
    return f"${value:.2f}"


def _normalize_bucket_display(label: str) -> str:
    return label.replace("°F", "").replace("°", "").strip()


def print_report(
    trades: list[dict[str, Any]],
    *,
    dates: list[str],
    start_bankroll: float,
    end_bankroll: float,
    quality: dict[str, Any],
) -> None:
    n_contracts = 5
    assert n_contracts == 5
    print("=== Telonex Retroactive Validation (Jul 1-3) ===")
    print(f"Bankroll: ${start_bankroll:.2f} | Strategy: v5b | Sizing: flat {n_contracts}")
    print()

    header = (
        f"{'Date':<10} | {'City':<13} | {'Bucket':<9} | {'Forecast':>8} | "
        f"{'Entry$':>6} | {'Edge':>6} | {'Actual':>6} | {'Win':>3} | {'PnL':>8}"
    )
    print(header)
    print("-" * len(header))

    if not trades:
        print("(no trades selected)")
    else:
        for t in trades:
            fc = f"{t['forecast_f']}F" if t.get("forecast_f") is not None else "—"
            actual = f"{t['actual_tmax_f']}F" if t.get("actual_tmax_f") is not None else "—"
            entry = float(t.get("maker_entry_price") or t.get("market_price") or 0)
            edge = t.get("edge")
            edge_s = f"{edge:+.3f}" if edge is not None else "—"
            if t.get("price_source") == "gamma_fallback":
                edge_s += "*"
            win = "PND" if t.get("pending") else ("YES" if t.get("won") else "NO")
            pnl_s = "pending" if t.get("pending") or t.get("pnl") is None else _fmt_money(float(t["pnl"]))
            print(
                f"{t['event_date']:<10} | {t['city']:<13} | "
                f"{_normalize_bucket_display(t['bucket_label']):<9} | {fc:>8} | "
                f"${entry:.2f} | {edge_s:>6} | {actual:>6} | {win:>3} | {pnl_s:>8}"
            )

    settled = [t for t in trades if not t.get("pending") and t.get("pnl") is not None]
    wins = [t for t in settled if t.get("won")]
    losses = [t for t in settled if t.get("won") is False]
    total_pnl = sum(float(t["pnl"]) for t in settled)
    mean_pnl = total_pnl / len(settled) if settled else None
    win_rate = len(wins) / len(settled) if settled else None

    print()
    print("Summary:")
    print(f"  Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"  Total trades: {len(trades)}")
    print(f"  Wins: {len(wins)} / Losses: {len(losses)}")
    if win_rate is not None:
        print(f"  Win rate: {win_rate:.1%}")
    else:
        print("  Win rate: —")
    print(f"  Total PnL: {_fmt_money(total_pnl)}")
    if mean_pnl is not None:
        print(f"  Mean PnL/trade: {_fmt_money(mean_pnl)}")
    else:
        print("  Mean PnL/trade: —")
    print(f"  Bankroll trajectory: ${start_bankroll:.2f} -> ${end_bankroll:.2f}")

    print()
    print("Telonex data quality:")
    print(f"  Tokens with BBO at ~10:10 AM local: {quality['telonex_bbo']}/{quality['total_buckets']}")
    print(f"  Tokens using Gamma fallback: {quality['gamma_fallback']}")
    if quality.get("missing"):
        print(f"  Missing data: {quality['missing']}")
    if quality.get("warnings"):
        print("  Warnings:")
        for w in quality["warnings"]:
            print(f"    - {w}")
    if quality["gamma_fallback"]:
        print(
            "\n  * Edge marked with * used Gamma settlement prices — "
            "NOT valid for forward validation."
        )


def main() -> None:
    args = _parse_args()
    dates = sorted(set(args.dates))
    phase = args.phase

    deploy = load_deploy_config(PROJECT_ROOT / "config" / "deploy_config.json")
    poly_config = {
        **deploy,
        "cities": list(POLYMARKET_CITIES),
        "basket_boundary_margin_f": 0.0,
    }
    city_config = load_city_config(poly_config)

    api_key: str | None = None
    catalog: pd.DataFrame | None = None
    if phase in ("all", "telonex") and not args.skip_fetch:
        try:
            api_key = load_telonex_key()
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")

    all_quality: dict[str, Any] = {
        "telonex_bbo": 0,
        "gamma_fallback": 0,
        "total_buckets": 0,
        "missing": [],
        "warnings": [],
    }
    all_trades: list[dict[str, Any]] = []
    start_bankroll = float(args.bankroll)
    bankroll = start_bankroll

    wu_cache = WuActualCache.load(city_config)

    for event_date in dates:
        print(f"\n=== {event_date} ===")

        if phase in ("all", "gamma"):
            print("Phase 1: Gamma structure")
            fetch_gamma_structure(event_date)
        elif phase == "validate":
            if not _gamma_structure_path(event_date).exists():
                print(f"  Missing {_gamma_structure_path(event_date)} — run phase gamma first")
                continue

        gamma = load_gamma_structure(event_date)

        if phase in ("all", "telonex"):
            print("Phase 2: Telonex snapshots")
            if api_key and catalog is None and not args.skip_fetch:
                try:
                    catalog = load_catalog(show_progress=False)
                except Exception as exc:
                    print(f"  WARNING: Telonex catalog load failed: {exc}")
            warns = _ensure_telonex_snapshots(
                event_date,
                gamma,
                skip_fetch=args.skip_fetch,
                api_key=api_key,
                catalog=catalog,
            )
            all_quality["warnings"].extend(warns)

        if phase in ("all", "validate"):
            if phase != "validate":
                print("Phase 3: v5b pipeline + settlement")
            _, pq = build_market_df(event_date, gamma, skip_fetch=args.skip_fetch)
            all_quality["telonex_bbo"] += pq["telonex_buckets"]
            all_quality["gamma_fallback"] += pq["gamma_fallback_buckets"]
            all_quality["total_buckets"] += pq["telonex_buckets"] + pq["gamma_fallback_buckets"]
            all_quality["missing"].extend(pq["missing_cities"])

            day_trades, _meta, bankroll = run_validation_day(
                event_date,
                bankroll,
                poly_config,
                city_config,
                gamma,
                wu_cache,
            )
            all_trades.extend(day_trades)

    wu_cache.save()

    if phase in ("all", "validate"):
        print_report(
            all_trades,
            dates=dates,
            start_bankroll=start_bankroll,
            end_bankroll=bankroll,
            quality=all_quality,
        )


if __name__ == "__main__":
    main()
