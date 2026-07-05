"""Shared Polymarket forecast/edge/sizing pipeline for daily and auto trading."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

from run_daily_trade import (  # noqa: E402
    _wait_for_market_open,
    fetch_forecast,
    load_city_config,
    load_deploy_config,
)
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402
from src.polymarket_api import (  # noqa: E402
    EVENT_TITLE_RE,
    GAMMA_API,
    _parse_event_date,
    fetch_order_book_http,
    parse_bucket_label,
)
from src.rolling_bias import compute_rolling_bias  # noqa: E402
from src.sizing import daily_cap_from_bankroll, effective_probability, has_edge  # noqa: E402

BIAS_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_bias.json"
WEATHER_TAG_ID = "104596"
POLYMARKET_CITIES = [
    "atlanta",
    "austin",
    "chicago",
    "dallas",
    "houston",
    "los_angeles",
    "miami",
    "new_york",
    "san_francisco",
    "seattle",
]
HRRR_CITIES = [
    "austin",
    "houston",
    "dallas",
    "chicago",
    "los_angeles",
    "san_francisco",
    "seattle",
    "new_york",
    "miami",
    "atlanta",
]
POLY_CITY_ALIASES: dict[str, list[str]] = {
    "atlanta": ["atlanta"],
    "austin": ["austin"],
    "chicago": ["chicago"],
    "dallas": ["dallas"],
    "houston": ["houston"],
    "los_angeles": ["los angeles", "la"],
    "miami": ["miami"],
    "new_york": ["new york", "new york city", "nyc"],
    "san_francisco": ["san francisco", "sf"],
    "seattle": ["seattle"],
}
POLY_PRICE_FLOOR = 0.10
BUCKET_FROM_QUESTION_RE = re.compile(r"(?i)be (.+?) on [A-Za-z]+ \d{1,2}")
POLY_TAKER_FEE_RATE = 0.05
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "deploy_config.json"


def poly_maker_fee(n_contracts: int, price: float) -> float:
    """Polymarket maker fee in pUSD. Zero for post_only orders."""
    return 0.0


def poly_taker_fee(n_contracts: int, price: float) -> float:
    """Polymarket weather taker fee in pUSD (worst-case if not maker fill)."""
    return round(n_contracts * POLY_TAKER_FEE_RATE * price * (1 - price), 5)


def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _title_matches_city(title: str, city: str) -> bool:
    title_lower = title.lower()
    for alias in POLY_CITY_ALIASES[city]:
        if len(alias) <= 3:
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", title_lower):
                return True
        elif alias in title_lower:
            return True
    return False


def _match_poly_city(title: str) -> str | None:
    for city in POLYMARKET_CITIES:
        if _title_matches_city(title, city):
            return city
    return None


def load_wunderground_bias() -> dict[str, dict[str, float | int]]:
    if not BIAS_PATH.exists():
        print(f"WARNING: No Wunderground bias file found at {BIAS_PATH}. Using zero bias.")
        return {}
    with open(BIAS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def apply_wunderground_bias(
    forecasts: dict[str, int],
    wunderground_bias: dict[str, dict[str, float | int]],
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    raw_forecasts = dict(forecasts)
    adjusted: dict[str, int] = {}
    bias_applied: dict[str, float] = {}
    for city, tmax_cli in forecasts.items():
        bias = float(wunderground_bias.get(city, {}).get("median_bias", 0.0))
        tmax_wu = int(round(tmax_cli - bias))
        adjusted[city] = tmax_wu
        bias_applied[city] = bias
        print(f"  {city}: Predicted Tmax: {tmax_cli}F (CLI-calibrated)")
        print(f"         Adjusted Tmax: {tmax_wu}F (Wunderground, bias={bias:+.1f})")
    return raw_forecasts, adjusted, bias_applied


def _range_bucket_midpoints(labels: list[str]) -> list[tuple[str, float]]:
    """Parse RANGE labels like '84-85' → (label, midpoint) sorted by midpoint."""
    parsed: list[tuple[str, float]] = []
    for label in labels:
        match = re.match(r"^(\d+)-(\d+)$", str(label).strip())
        if not match:
            continue
        lo, hi = int(match.group(1)), int(match.group(2))
        parsed.append((str(label), (lo + hi) / 2.0))
    return sorted(parsed, key=lambda x: x[1])


def nearest_boundary_distance_f(
    mu: float,
    range_buckets: list[tuple[str, int, int]],
) -> tuple[float, str | None, str | None]:
    """Return (min_distance, lower_label, upper_label) at interior boundaries."""
    if len(range_buckets) < 2:
        return float("inf"), None, None
    ordered = sorted(range_buckets, key=lambda b: (b[1] + b[2]) / 2.0)
    best_dist = float("inf")
    best_pair: tuple[str | None, str | None] = (None, None)
    for i in range(len(ordered) - 1):
        lo_label, _lo1, hi1 = ordered[i]
        hi_label, lo2, _hi2 = ordered[i + 1]
        boundary = (hi1 + lo2) / 2.0
        dist = abs(mu - boundary)
        if dist < best_dist:
            best_dist = dist
            best_pair = (lo_label, hi_label)
    return best_dist, best_pair[0], best_pair[1]


def basket_companion_label(
    mu: float,
    best_label: str,
    range_buckets: list[tuple[str, int, int]],
    margin_f: float,
) -> str | None:
    """If mu within margin of a boundary, return the adjacent bucket across it."""
    if margin_f <= 0 or len(range_buckets) < 2:
        return None
    dist, lo_label, hi_label = nearest_boundary_distance_f(mu, range_buckets)
    if dist > margin_f or lo_label is None or hi_label is None:
        return None
    if best_label == lo_label:
        return hi_label
    if best_label == hi_label:
        return lo_label
    return None


def _parse_event_date_from_title(title: str, year_hint: str | None = None) -> str | None:
    match = EVENT_TITLE_RE.search(title)
    if not match:
        return None
    try:
        return _parse_event_date(match.group(2), year_hint=year_hint)
    except ValueError:
        return None


def _extract_bucket_label(market: dict[str, Any]) -> str:
    group_title = market.get("groupItemTitle")
    if group_title:
        return str(group_title)
    question = str(market.get("question", ""))
    match = BUCKET_FROM_QUESTION_RE.search(question)
    if match:
        return match.group(1).strip()
    return question


def _paginate_gamma_events(
    session: requests.Session,
    event_date: str,
) -> list[dict[str, Any]]:
    """Fetch all active Tmax events for a date via weather tag_id."""
    all_events: list[dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        params = {
            "tag_id": WEATHER_TAG_ID,
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }
        try:
            response = session.get(f"{GAMMA_API}/events", params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"ERROR: Gamma API request failed: {exc}")
            raise SystemExit(1) from exc

        batch = response.json()
        if not batch:
            break

        for event in batch:
            title = str(event.get("title", ""))
            year_hint = event.get("eventDate") or event.get("endDate")
            parsed_date = _parse_event_date_from_title(
                title,
                str(year_hint) if year_hint else None,
            )
            if parsed_date != event_date:
                continue
            if _match_poly_city(title) is None:
                continue
            all_events.append(event)

        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)

    return all_events


def compute_maker_entry_price(
    *,
    best_bid: float | None,
    best_ask: float | None,
    gamma_price: float | None,
    tick_size: float,
) -> float | None:
    """Compute maker GTC limit price: one tick inside ask, or join bid."""
    if best_ask is not None:
        maker_entry = best_ask - tick_size
        if best_bid is not None and maker_entry <= best_bid:
            maker_entry = best_bid
        return round(maker_entry, 4)
    if gamma_price is not None:
        return round(gamma_price, 4)
    return None


def fetch_market(
    config: dict[str, Any],
    event_date: str,
    *,
    wait_for_open: bool = True,
    raise_on_no_market: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch Polymarket bucket snapshots for Tier 1 cities."""
    if wait_for_open and not _wait_for_market_open(event_date):
        raise SystemExit("Markets not available pre-open")

    print("\n--- fetch_market ---")
    session = _build_http_session()
    reasons: dict[str, str] = {city: "no Polymarket market for date" for city in config["cities"]}
    rows: list[dict[str, Any]] = []

    events = _paginate_gamma_events(session, event_date)
    if not events:
        tomorrow = (date.fromisoformat(event_date) + timedelta(days=1)).isoformat()
        message = (
            f"No Polymarket Tmax events found for {event_date}. "
            "Markets may not be open yet. "
            f"Try --date {tomorrow}."
        )
        if raise_on_no_market:
            print(message)
            raise SystemExit(1)
        print(f"WARNING: {message}")
        return pd.DataFrame(), reasons

    for event in events:
        title = str(event.get("title", ""))
        city = _match_poly_city(title)
        if city is None:
            continue

        print(f"  Fetching market: {city}")
        condition_id = str(event.get("negRiskMarketID") or event.get("id", ""))

        for market in event.get("markets") or []:
            if market.get("closed") or market.get("acceptingOrders") is False:
                continue

            question = str(market.get("question", ""))
            try:
                label = _extract_bucket_label(market)
                parsed_bucket = parse_bucket_label(label)
            except ValueError:
                print(f"  WARNING: could not parse bucket: {question!r}")
                continue

            token_ids = _parse_json_field(market.get("clobTokenIds"))
            outcome_prices = _parse_json_field(market.get("outcomePrices"))
            outcomes = _parse_json_field(market.get("outcomes"))
            if not token_ids:
                continue

            yes_index = 0
            if outcomes and str(outcomes[0]).lower() != "yes":
                yes_index = 1 if len(token_ids) > 1 else 0

            yes_token_id = str(token_ids[yes_index])
            gamma_price = _to_float(outcome_prices[yes_index]) if outcome_prices else None
            tick_size = str(market.get("orderPriceMinTickSize", "0.01"))

            if gamma_price is None:
                continue

            best_bid: float | None = None
            best_ask: float | None = None
            spread: float | None = None
            if gamma_price >= POLY_PRICE_FLOOR:
                best_bid, best_ask = fetch_order_book_http(yes_token_id)
                time.sleep(0.15)
                if best_bid is None and best_ask is None:
                    print(
                        f"  WARNING: empty order book for {city} "
                        f"{label!r}, using gamma {gamma_price:.4f}"
                    )
                elif best_bid is not None and best_ask is not None:
                    spread = round(best_ask - best_bid, 4)

            market_price = best_ask if best_ask is not None else gamma_price

            rows.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "bucket_label": label,
                    "bucket_type": parsed_bucket["type"],
                    "bucket_lower_inclusive_f": parsed_bucket["lower"],
                    "bucket_upper_inclusive_f": parsed_bucket["upper"],
                    "gamma_price": float(gamma_price),
                    "yes_mid_close": float(gamma_price),
                    "market_price": float(market_price),
                    "yes_bid_close": best_bid,
                    "yes_ask_close": best_ask,
                    "spread": spread,
                    "tick_size": tick_size,
                    "yes_token_id": yes_token_id,
                    "condition_id": str(market.get("conditionId") or condition_id),
                }
            )

        if any(row["city"] == city for row in rows):
            reasons.pop(city, None)

    market_df = pd.DataFrame(rows)
    for city in config["cities"]:
        if city in reasons:
            continue
        if market_df.empty or city not in set(market_df["city"].astype(str)):
            reasons[city] = "no market data"

    return market_df, reasons


def compute_edge(
    market_df: pd.DataFrame,
    forecasts: dict[str, int],
    city_config: dict[str, Any],
    config: dict[str, Any],
    market_reasons: dict[str, str],
    event_date: str,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Compute best tradeable bucket per city using Polymarket prices."""
    print("\n--- compute_edge ---")
    edges: list[dict[str, Any]] = []
    reasons = dict(market_reasons)
    sanity_rows: dict[str, list[dict[str, Any]]] = {}
    price_floor = POLY_PRICE_FLOOR
    edge_threshold = float(config["edge_threshold"])

    for city in config["cities"]:
        if city in reasons:
            continue
        if city not in forecasts:
            reasons[city] = "no forecast"
            continue

        day_df = market_df[
            (market_df["city"].astype(str) == city)
            & (market_df["event_date"].astype(str) == event_date)
        ].copy()
        if day_df.empty:
            reasons[city] = "no market data"
            continue

        buckets = day_df[
            [
                "bucket_label",
                "bucket_type",
                "bucket_lower_inclusive_f",
                "bucket_upper_inclusive_f",
            ]
        ].drop_duplicates("bucket_label")
        tmax_pred = forecasts[city]
        sigma = float(city_config[city]["trackb_sigma_f"])
        probs = bucket_probs_from_point_forecast(tmax_pred, sigma, buckets)

        city_sanity: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        shrinkage_lambda = float(config.get("shrinkage_lambda", 1.0))

        for bucket_label, model_prob in probs.items():
            entry_rows = day_df[day_df["bucket_label"].astype(str).eq(str(bucket_label))]
            if entry_rows.empty:
                continue
            row = entry_rows.iloc[0]
            gamma_price = _to_float(row.get("gamma_price"))
            best_bid = _to_float(row.get("yes_bid_close"))
            best_ask = _to_float(row.get("yes_ask_close"))
            spread = _to_float(row.get("spread"))
            tick_size = float(row.get("tick_size", 0.01))
            entry_price = float(row["market_price"])
            maker_entry_price = compute_maker_entry_price(
                best_bid=best_bid,
                best_ask=best_ask,
                gamma_price=gamma_price,
                tick_size=tick_size,
            )
            p_eff = effective_probability(
                float(model_prob), entry_price, shrinkage_lambda
            )
            edge = p_eff - entry_price
            passes_guardrail = entry_price >= price_floor and has_edge(
                p_eff, entry_price, 0.0
            )

            if entry_price < price_floor:
                status = "skip (below floor)"
            elif not passes_guardrail:
                status = "skip"
            elif edge >= edge_threshold:
                status = "passes E*"
            else:
                status = "skip"

            city_sanity.append(
                {
                    "bucket_label": str(bucket_label),
                    "model_prob": float(model_prob),
                    "effective_prob": float(p_eff),
                    "gamma_price": gamma_price,
                    "market_price": entry_price,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "edge": edge,
                    "status": status,
                }
            )

            if not passes_guardrail:
                continue

            candidate = {
                "city": city,
                "bucket_label": str(bucket_label),
                "model_prob": float(model_prob),
                "effective_prob": float(p_eff),
                "gamma_price": gamma_price,
                "market_price": entry_price,
                "edge": edge,
                "side": "YES",
                "yes_token_id": str(row["yes_token_id"]),
                "condition_id": str(row["condition_id"]),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "tick_size": tick_size,
                "maker_entry_price": maker_entry_price,
            }
            if best is None or candidate["edge"] > best["edge"]:
                best = candidate

        sanity_rows[city] = city_sanity

        if best is None:
            reasons[city] = "no bucket passes guardrails"
            continue

        for row in city_sanity:
            if row["bucket_label"] == best["bucket_label"]:
                row["status"] = "SELECTED"

        edges.append(best)
        print(
            f"  {city}: {best['bucket_label']} edge={best['edge']:+.3f} "
            f"@ ${best['market_price']:.2f}"
        )

        margin = float(config.get("basket_boundary_margin_f", 1.0))
        range_buckets = [
            (str(r["bucket_label"]), int(r["bucket_lower_inclusive_f"]), int(r["bucket_upper_inclusive_f"]))
            for _, r in buckets.iterrows()
            if str(r["bucket_type"]) == "RANGE"
            and pd.notna(r["bucket_lower_inclusive_f"])
            and pd.notna(r["bucket_upper_inclusive_f"])
        ]
        companion_label = basket_companion_label(
            float(tmax_pred), best["bucket_label"], range_buckets, margin
        )
        if companion_label and companion_label in probs:
            entry_rows = day_df[day_df["bucket_label"].astype(str).eq(companion_label)]
            if not entry_rows.empty:
                row = entry_rows.iloc[0]
                gamma_price = _to_float(row.get("gamma_price"))
                best_bid = _to_float(row.get("yes_bid_close"))
                best_ask = _to_float(row.get("yes_ask_close"))
                spread = _to_float(row.get("spread"))
                tick_size = float(row.get("tick_size", 0.01))
                entry_price = float(row["market_price"])
                maker_entry_price = compute_maker_entry_price(
                    best_bid=best_bid,
                    best_ask=best_ask,
                    gamma_price=gamma_price,
                    tick_size=tick_size,
                )
                model_prob = float(probs[companion_label])
                p_eff = effective_probability(model_prob, entry_price, shrinkage_lambda)
                edge = p_eff - entry_price
                passes_guardrail = entry_price >= price_floor and has_edge(p_eff, entry_price, 0.0)
                if passes_guardrail and edge >= edge_threshold:
                    companion = {
                        "city": city,
                        "bucket_label": companion_label,
                        "model_prob": model_prob,
                        "effective_prob": float(p_eff),
                        "gamma_price": gamma_price,
                        "market_price": entry_price,
                        "edge": edge,
                        "side": "YES",
                        "yes_token_id": str(row["yes_token_id"]),
                        "condition_id": str(row["condition_id"]),
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "tick_size": tick_size,
                        "maker_entry_price": maker_entry_price,
                        "basket_companion": True,
                    }
                    edges.append(companion)
                    for sanity_row in city_sanity:
                        if sanity_row["bucket_label"] == companion_label:
                            sanity_row["status"] = "SELECTED (basket)"
                    print(
                        f"  {city}: basket {companion_label} edge={edge:+.3f} "
                        f"@ ${entry_price:.2f}"
                    )

    return edges, reasons, sanity_rows


def select_trades_poly(
    edges: list[dict[str, Any]],
    config: dict[str, Any],
    reasons: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Apply edge_threshold selection and rank by edge (no OOS exclusions)."""
    print("\n--- select_trades ---")
    threshold = float(config["edge_threshold"])
    selected: list[dict[str, Any]] = []

    for edge_row in sorted(edges, key=lambda row: row["edge"], reverse=True):
        city = edge_row["city"]
        if edge_row["edge"] < threshold:
            reasons[city] = (
                f"edge below threshold ({edge_row['edge']:.3f} < {threshold:.3f})"
            )
            continue
        selected.append(edge_row)

    max_trades = int(config.get("max_trades_per_day", 2))
    selected = selected[:max_trades]

    return selected, reasons


def size_positions_poly(
    trades: list[dict[str, Any]],
    bankroll: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply flat sizing and daily loss cap with Polymarket fee model."""
    print("\n--- size_positions ---")
    n_default = int(config["n_contracts_default"])
    n_reduced = int(config["n_contracts_reduced"])
    threshold = float(config["bankroll_reduction_threshold"])
    daily_cap = daily_cap_from_bankroll(bankroll, config)
    n_contracts = n_reduced if bankroll < threshold else n_default

    sized: list[dict[str, Any]] = []
    for trade in trades:
        maker_price = float(trade.get("maker_entry_price") or trade["market_price"])
        sized.append(
            {
                **trade,
                "n_contracts": n_contracts,
                "capital_at_risk": round(n_contracts * maker_price, 4),
                "maker_fee": poly_maker_fee(n_contracts, maker_price),
                "potential_taker_fee": poly_taker_fee(n_contracts, float(trade["market_price"])),
            }
        )

    while sized:
        total_cap = sum(t["capital_at_risk"] for t in sized)
        if total_cap <= daily_cap:
            break
        dropped = sized.pop()
        print(f"  Dropped {dropped['city']} (cap trim): edge={dropped['edge']:.3f}")

    return sized


def build_poly_config(config_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_deploy_config(config_path or DEFAULT_CONFIG_PATH)
    poly_config = {**config, "cities": list(POLYMARKET_CITIES)}
    city_config = load_city_config(poly_config)
    return poly_config, city_config


def prepare_poly_trades(
    event_date: str,
    bankroll: float,
    config_path: Path | None = None,
    *,
    wait_for_open: bool = True,
    raise_on_no_market: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run forecast -> market -> edge -> select -> size pipeline.

    Returns (sized_trades, metadata).
    """
    poly_config, city_config = build_poly_config(config_path)
    wunderground_bias = load_wunderground_bias()

    print("\n--- PHASE 1: Pre-fetch features ---")
    forecasts, forecast_reasons, forecast_notes = fetch_forecast(
        poly_config, event_date, city_config
    )
    n_forecasts = len(forecasts)
    print(f"\nFeature coverage: {n_forecasts}/{len(poly_config['cities'])} cities")

    metadata: dict[str, Any] = {
        "event_date": event_date,
        "bankroll": bankroll,
        "poly_config": poly_config,
        "city_config": city_config,
        "forecast_notes": forecast_notes,
        "forecast_reasons": forecast_reasons,
        "forecasts": {},
        "raw_forecasts": {},
        "bias_applied": {},
        "market_df": pd.DataFrame(),
        "market_reasons": {},
        "edges": [],
        "sanity_rows": {},
        "all_reasons": {},
        "skipped_edges": [],
        "abort_reason": None,
    }

    if n_forecasts == 0:
        metadata["abort_reason"] = "no_forecasts"
        metadata["all_reasons"] = dict(forecast_reasons)
        return [], metadata

    for city, pred in sorted(forecasts.items()):
        note = forecast_notes.get(city, "")
        if note:
            print(f"  {city}: {pred}F ({note})")
    for city, reason in sorted(forecast_reasons.items()):
        print(f"  {city}: SKIP ({reason})")

    print("\n--- Wunderground bias adjustment ---")
    raw_forecasts, forecasts, bias_applied = apply_wunderground_bias(
        forecasts, wunderground_bias
    )
    metadata["forecasts"] = forecasts
    metadata["raw_forecasts"] = raw_forecasts
    metadata["bias_applied"] = bias_applied

    halflife = int(poly_config.get("rolling_bias_halflife_days", 20))
    rolling_applied: dict[str, float] = {}
    print("\n--- Rolling bias correction ---")
    for city in list(forecasts.keys()):
        rolling = compute_rolling_bias(city, event_date, halflife)
        forecasts[city] = int(round(forecasts[city] - rolling))
        rolling_applied[city] = rolling
        print(f"  {city}: rolling bias correction {rolling:+.2f}F → {forecasts[city]}F")
    metadata["rolling_bias_applied"] = rolling_applied
    metadata["forecasts"] = forecasts

    print("\n--- PHASE 2: Fetch Polymarket snapshot ---")
    market_df, market_reasons = fetch_market(
        poly_config,
        event_date,
        wait_for_open=wait_for_open,
        raise_on_no_market=raise_on_no_market,
    )
    metadata["market_df"] = market_df
    metadata["market_reasons"] = market_reasons

    if market_df.empty and not raise_on_no_market:
        all_reasons = {**market_reasons, **forecast_reasons}
        metadata["all_reasons"] = all_reasons
        metadata["abort_reason"] = "no_markets"
        return [], metadata

    print("\n--- PHASE 3: Compute edge, select, size ---")
    all_reasons = {**market_reasons, **forecast_reasons}
    edges, edge_reasons, sanity_rows = compute_edge(
        market_df,
        forecasts,
        city_config,
        poly_config,
        all_reasons,
        event_date,
    )
    all_reasons.update(edge_reasons)
    metadata["edges"] = edges
    metadata["sanity_rows"] = sanity_rows

    selected, all_reasons = select_trades_poly(edges, poly_config, all_reasons)
    sized_trades = size_positions_poly(selected, bankroll, poly_config)
    skipped_edges = [
        row for row in edges if row["city"] not in {t["city"] for t in sized_trades}
    ]
    metadata["all_reasons"] = all_reasons
    metadata["skipped_edges"] = skipped_edges

    return sized_trades, metadata
