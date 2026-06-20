"""Daily trading pipeline for Polymarket. Run at 10:00 AM CT."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_daily_trade import (  # noqa: E402
    _now_ct,
    _wait_for_market_open,
    fetch_forecast,
    load_city_config,
    load_deploy_config,
)
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402
from src.polymarket_api import (  # noqa: E402
    CLOB_HOST,
    EVENT_TITLE_RE,
    GAMMA_API,
    _parse_event_date,
    parse_bucket_label,
)
from src.sizing import has_edge  # noqa: E402

POLY_PAPER_LOG = PROJECT_ROOT / "logs" / "poly_paper_trades.jsonl"
BIAS_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_bias.json"
WEATHER_TAG_ID = "104596"
# Station-matched cities only (Polymarket resolution station = model training station).
POLYMARKET_CITIES = ["austin", "houston", "los_angeles", "san_francisco"]
# NYC excluded: Polymarket uses KLGA (LaGuardia), model trained on KNYC
# Chicago excluded: Polymarket uses KORD (O'Hare), model trained on KMDW
# These cities will be added back after retraining on correct stations.
POLY_CITY_ALIASES: dict[str, list[str]] = {
    "austin": ["austin"],
    "houston": ["houston"],
    "los_angeles": ["los angeles", "la"],
    "san_francisco": ["san francisco", "sf"],
}
POLY_PRICE_FLOOR = 0.10  # Lower than Kalshi 0.15; Polymarket buckets are wider-spaced
BUCKET_FROM_QUESTION_RE = re.compile(
    r"(?i)be (.+?) on [A-Za-z]+ \d{1,2}"
)
POLY_TAKER_FEE_RATE = 0.05


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


def _load_wunderground_bias() -> dict[str, dict[str, float | int]]:
    if not BIAS_PATH.exists():
        print(f"WARNING: No Wunderground bias file found at {BIAS_PATH}. Using zero bias.")
        return {}
    with open(BIAS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _apply_wunderground_bias(
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


def _fetch_clob_midpoint(session: requests.Session, token_id: str) -> float | None:
    try:
        response = session.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return _to_float(payload.get("mid"))
        return _to_float(payload)
    except requests.RequestException:
        return None


def _fetch_clob_book(
    session: requests.Session,
    token_id: str,
) -> tuple[float | None, float | None, float | None]:
    try:
        response = session.get(
            f"{CLOB_HOST}/book",
            params={"token_id": token_id},
            timeout=20,
        )
        response.raise_for_status()
        book = response.json()
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = _to_float(bids[0]["price"]) if bids else None
        best_ask = _to_float(asks[0]["price"]) if asks else None
        spread = None
        if best_bid is not None and best_ask is not None:
            spread = round(best_ask - best_bid, 4)
        return best_bid, best_ask, spread
    except (requests.RequestException, KeyError, IndexError, TypeError):
        return None, None, None


def fetch_market(
    config: dict[str, Any],
    event_date: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch Polymarket bucket snapshots for Tier 1 cities."""
    if not _wait_for_market_open(event_date):
        raise SystemExit("Markets not available pre-open")

    print("\n--- fetch_market ---")
    session = _build_http_session()
    reasons: dict[str, str] = {city: "no Polymarket market for date" for city in config["cities"]}
    rows: list[dict[str, Any]] = []

    events = _paginate_gamma_events(session, event_date)
    if not events:
        tomorrow = (date.fromisoformat(event_date) + timedelta(days=1)).isoformat()
        print(
            f"No Polymarket Tmax events found for {event_date}. "
            "Markets may not be open yet. "
            f"Try --date {tomorrow}."
        )
        raise SystemExit(1)

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

            midpoint = _fetch_clob_midpoint(session, yes_token_id)
            time.sleep(0.1)
            if midpoint is None:
                if gamma_price is not None:
                    print(
                        f"  WARNING: CLOB midpoint unavailable for {city} "
                        f"{label!r}, using Gamma price {gamma_price:.4f}"
                    )
                    midpoint = gamma_price
                else:
                    continue

            best_bid, best_ask, spread = _fetch_clob_book(session, yes_token_id)

            rows.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "bucket_label": label,
                    "bucket_type": parsed_bucket["type"],
                    "bucket_lower_inclusive_f": parsed_bucket["lower"],
                    "bucket_upper_inclusive_f": parsed_bucket["upper"],
                    "yes_mid_close": float(midpoint),
                    "yes_bid_close": best_bid,
                    "yes_ask_close": best_ask,
                    "spread": spread,
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

        for bucket_label, model_prob in probs.items():
            entry_rows = day_df[day_df["bucket_label"].astype(str).eq(str(bucket_label))]
            if entry_rows.empty:
                continue
            row = entry_rows.iloc[0]
            entry_price = float(row["yes_mid_close"])
            edge = float(model_prob) - entry_price
            # Polymarket maker orders: fee_per_contract = 0
            passes_guardrail = entry_price >= price_floor and has_edge(
                model_prob, entry_price, 0.0
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
                    "market_price": entry_price,
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
                "market_price": entry_price,
                "edge": edge,
                "side": "YES",
                "yes_token_id": str(row["yes_token_id"]),
                "condition_id": str(row["condition_id"]),
                "best_bid": row.get("yes_bid_close"),
                "best_ask": row.get("yes_ask_close"),
                "spread": row.get("spread"),
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
    daily_cap = float(config["daily_loss_cap"])
    n_contracts = n_reduced if bankroll < threshold else n_default

    sized: list[dict[str, Any]] = []
    for trade in trades:
        price = float(trade["market_price"])
        sized.append(
            {
                **trade,
                "n_contracts": n_contracts,
                "capital_at_risk": round(n_contracts * price, 4),
                "maker_fee": poly_maker_fee(n_contracts, price),
                "potential_taker_fee": poly_taker_fee(n_contracts, price),
            }
        )

    while sized:
        total_cap = sum(t["capital_at_risk"] for t in sized)
        if total_cap <= daily_cap:
            break
        dropped = sized.pop()
        print(f"  Dropped {dropped['city']} (cap trim): edge={dropped['edge']:.3f}")

    return sized


def log_decision_poly(decision: dict[str, Any]) -> None:
    POLY_PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(POLY_PAPER_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, default=str) + "\n")
    print(f"\nDecision log appended to {POLY_PAPER_LOG}")
    print(json.dumps(decision, indent=2, default=str))


def _print_market_diagnostics(
    market_df: pd.DataFrame,
    cities: list[str],
    event_date: str,
    market_reasons: dict[str, str],
) -> None:
    """Print bucket price distribution to diagnose zero-trade runs."""
    print("\n=== MARKET DIAGNOSTICS ===")
    for city in cities:
        if city in market_reasons:
            print(f"  {city}: SKIP ({market_reasons[city]})")
            continue
        day_df = market_df[
            (market_df["city"].astype(str) == city)
            & (market_df["event_date"].astype(str) == event_date)
        ].copy()
        if day_df.empty:
            print(f"  {city}: no market data")
            continue

        prices = pd.to_numeric(day_df["yes_mid_close"], errors="coerce").dropna()
        if prices.empty:
            print(f"  {city}: no valid prices")
            continue

        n_buckets = len(prices)
        n_above_floor = int((prices >= POLY_PRICE_FLOOR).sum())
        modal_idx = prices.idxmax()
        modal_row = day_df.loc[modal_idx]
        modal_bucket = str(modal_row["bucket_label"])
        modal_price = float(prices.max())

        print(f"  {city}: {n_buckets} buckets, {n_above_floor} above ${POLY_PRICE_FLOOR:.2f} floor")
        print(f"    Modal: {modal_bucket} @ ${modal_price:.3f}")
        print(
            f"    Price distribution: min=${prices.min():.3f} "
            f"median=${prices.median():.3f} max=${prices.max():.3f}"
        )
        if n_above_floor <= 1:
            print(
                f"    WARNING: Only {n_above_floor} bucket(s) above floor. "
                "Market may be too concentrated for trading."
            )


def _print_sanity_check(
    sanity_rows: dict[str, list[dict[str, Any]]],
    forecasts: dict[str, int],
    raw_forecasts: dict[str, int] | None,
    city_config: dict[str, Any],
    market_reasons: dict[str, str],
) -> None:
    print("\n=== SANITY CHECK: Model vs Polymarket Prices ===")
    for city in POLYMARKET_CITIES:
        if city in market_reasons or city not in forecasts:
            continue
        rows = sanity_rows.get(city)
        if not rows:
            continue

        tmax_pred = forecasts[city]
        raw_tmax = raw_forecasts.get(city, tmax_pred) if raw_forecasts else tmax_pred
        sigma = float(city_config[city]["trackb_sigma_f"])
        if raw_forecasts and raw_tmax != tmax_pred:
            print(
                f"\n{city.replace('_', ' ').title()} "
                f"(Tmax: {raw_tmax}F CLI -> {tmax_pred}F WU-adjusted, sigma: {sigma:.2f})"
            )
        else:
            print(
                f"\n{city.replace('_', ' ').title()} "
                f"(Tmax forecast: {tmax_pred}F, sigma: {sigma:.2f})"
            )
        print(f"  {'Bucket':<12} {'Model_P':>8} {'Market_P':>9} {'Edge':>7}  Status")
        sum_model = 0.0
        sum_market = 0.0
        for row in rows:
            sum_model += row["model_prob"]
            sum_market += row["market_price"]
            print(
                f"  {row['bucket_label']:<12} "
                f"{row['model_prob']:>8.3f} "
                f"{row['market_price']:>9.3f} "
                f"{row['edge']:>+7.3f}  {row['status']}"
            )
        print(f"  {'Sum:':<12} {sum_model:>8.3f} {sum_market:>9.3f}")


def daily_risk_report_poly(
    decision: dict[str, Any],
    skipped_edges: list[dict[str, Any]],
    mode: str,
    *,
    sanity_rows: dict[str, list[dict[str, Any]]] | None = None,
    forecasts: dict[str, int] | None = None,
    city_config: dict[str, Any] | None = None,
    market_reasons: dict[str, str] | None = None,
) -> None:
    event_date = decision["date"]
    bankroll = decision["bankroll"]
    n_cities = len(decision.get("cities_attempted", []))
    n_forecast = decision["n_cities_with_forecast"]
    n_trades = decision["n_trades_selected"]
    total_cap = decision["total_capital_at_risk"]
    daily_cap = decision.get("daily_loss_cap", 6.0)
    no_signal = decision.get("no_signal_reasons", {})

    print(f"\n=== DAILY RISK REPORT — {event_date} ({mode.upper()}/POLYMARKET) ===")
    print(f"Bankroll:           ${bankroll:.2f}")
    print(f"Trades selected:    {n_trades} / {n_cities} cities")
    print(f"Total cap at risk:  ${total_cap:.2f} / ${daily_cap:.2f} daily cap")
    coverage_notes = [
        f"{city}: {reason}"
        for city, reason in sorted(no_signal.items())
        if city not in {t["city"] for t in decision.get("trades", [])}
    ]
    print(
        f"Forecast coverage:  {n_forecast} / {n_cities} cities"
        + (f" ({', '.join(coverage_notes[:3])})" if coverage_notes else "")
    )
    print()

    for idx, trade in enumerate(decision.get("trades", []), start=1):
        print(
            f"Trade {idx}: {trade['city']} | {trade['bucket_label']} | "
            f"edge={trade['edge']:+.3f} | {trade['n_contracts']} contracts "
            f"@ ${trade['market_price']:.2f} (maker GTC) | "
            f"maker_fee=${trade.get('maker_fee', 0.0):.2f}"
        )

    if skipped_edges:
        print()
        for row in skipped_edges:
            city = row["city"]
            if city in {t["city"] for t in decision.get("trades", [])}:
                continue
            reason = no_signal.get(city, "")
            if "below threshold" in reason or "edge below" in reason:
                print(
                    f"Skipped: {city} (edge={row['edge']:.3f} < "
                    f"E*={decision.get('edge_threshold', 0.037):.3f})"
                )
            elif reason:
                print(f"Skipped: {city} ({reason})")

    if sanity_rows and forecasts and city_config and market_reasons is not None:
        _print_sanity_check(
            sanity_rows,
            forecasts,
            decision.get("raw_forecasts"),
            city_config,
            market_reasons,
        )

    if mode == "paper":
        print("\n** PAPER MODE — no orders placed **")
        print("** To place manually: review edges above, enter on Polymarket UI **")
    print("===")


def _print_header(event_date: str, bankroll: float, edge_threshold: float) -> None:
    cities_str = ", ".join(POLYMARKET_CITIES)
    now_ct = _now_ct().strftime("%H:%M:%S")
    print(f"\n=== POLYMARKET PAPER TRADE: {event_date} ===")
    print("Exchange: Polymarket (CLOB, Polygon chain 137)")
    print("Fee model: Maker zero (post_only GTC)")
    print(f"Cities: 4 station-matched ({cities_str})")
    print(f"Price floor: ${POLY_PRICE_FLOOR:.2f}")
    print(f"Edge threshold: E*={edge_threshold:.3f}")
    print(f"Bankroll: ${bankroll:.2f}")
    print(f"Run time: {now_ct} CT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Track-B trading pipeline (Polymarket)")
    parser.add_argument("--date", type=str, default=str(date.today()))
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "deploy_config.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass duplicate check in poly_paper_trades.jsonl",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Fetch Polymarket snapshot and print diagnostics only (no forecasts/trades)",
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="Build forecasts and exit without market fetch or trading",
    )
    parser.add_argument(
        "--live-confirm",
        action="store_true",
        help="Required to actually place live orders (otherwise stub only)",
    )
    parser.add_argument(
        "--cancel-unfilled",
        action="store_true",
        help="Cancel all open GTC orders and exit (requires credentials)",
    )
    args = parser.parse_args()

    config = load_deploy_config(Path(args.config))
    poly_config = {**config, "cities": list(POLYMARKET_CITIES)}
    city_config = load_city_config(poly_config)
    wunderground_bias = _load_wunderground_bias()
    event_date = args.date
    bankroll = args.bankroll
    edge_threshold = float(config["edge_threshold"])

    _print_header(event_date, bankroll, edge_threshold)
    print(f"Mode: {args.mode.upper()}")

    if args.cancel_unfilled:
        from src.polymarket_api import PolymarketClient  # noqa: E402

        print("\n--- Cancel unfilled GTC orders ---")
        PolymarketClient().cancel_unfilled_orders(event_date=event_date)
        return

    if args.fetch_only:
        print("\n--- Fetch Polymarket snapshot (fetch-only) ---")
        market_df, market_reasons = fetch_market(poly_config, event_date)
        _print_market_diagnostics(
            market_df, POLYMARKET_CITIES, event_date, market_reasons
        )
        return

    if POLY_PAPER_LOG.exists() and not args.force:
        with open(POLY_PAPER_LOG, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("date") == event_date and entry.get("mode") == args.mode:
                    print(f"Decision log already exists for {event_date}. Skipping.")
                    print("Use --force to override.")
                    return

    if args.force:
        print("--force: overriding duplicate check, will append new entry")

    print("\n--- PHASE 1: Pre-fetch features ---")
    forecasts, forecast_reasons, forecast_notes = fetch_forecast(
        poly_config, event_date, city_config
    )
    n_forecasts = len(forecasts)
    print(f"\nFeature coverage: {n_forecasts}/{len(poly_config['cities'])} cities")
    if n_forecasts == 0:
        print("ABORT: 0 cities have forecast coverage. Fix data sources.")
        decision = {
            "date": event_date,
            "mode": args.mode,
            "exchange": "polymarket",
            "bankroll": bankroll,
            "cities_attempted": poly_config["cities"],
            "n_cities_eligible": len(poly_config["cities"]),
            "n_cities_with_forecast": 0,
            "n_trades_selected": 0,
            "edge_threshold": edge_threshold,
            "daily_loss_cap": float(config["daily_loss_cap"]),
            "fee_model": "maker_zero",
            "trades": [],
            "total_capital_at_risk": 0,
            "daily_loss_cap_remaining": float(config["daily_loss_cap"]),
            "no_signal_cities": sorted(poly_config["cities"]),
            "no_signal_reasons": {**forecast_reasons},
            "forecast_notes": forecast_notes,
        }
        log_decision_poly(decision)
        daily_risk_report_poly(decision, [], args.mode)
        return

    for city, pred in sorted(forecasts.items()):
        note = forecast_notes.get(city, "")
        if note:
            print(f"  {city}: {pred}F ({note})")
    for city, reason in sorted(forecast_reasons.items()):
        print(f"  {city}: SKIP ({reason})")

    print("\n--- Wunderground bias adjustment ---")
    raw_forecasts, forecasts, bias_applied = _apply_wunderground_bias(
        forecasts, wunderground_bias
    )

    if args.prefetch_only:
        print("\n--prefetch-only: stopping after feature build.")
        return

    print("\n--- PHASE 2: Fetch Polymarket snapshot ---")
    market_df, market_reasons = fetch_market(poly_config, event_date)
    _print_market_diagnostics(
        market_df, POLYMARKET_CITIES, event_date, market_reasons
    )

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

    selected, all_reasons = select_trades_poly(edges, poly_config, all_reasons)
    sized_trades = size_positions_poly(selected, bankroll, poly_config)
    skipped_edges = [
        row for row in edges if row["city"] not in {t["city"] for t in sized_trades}
    ]

    if args.mode == "live":
        if args.live_confirm:
            from src.polymarket_api import PolymarketClient  # noqa: E402

            poly_client = PolymarketClient()
            live_trades: list[dict[str, Any]] = []
            for trade in sized_trades:
                result = poly_client.place_order(
                    token_id=trade["yes_token_id"],
                    side="YES",
                    price=trade["market_price"],
                    size=float(trade["n_contracts"]),
                    dry_run=False,
                    post_only=True,
                )
                if result.get("status") == "rejected_would_cross":
                    print(
                        f"  {trade['city']}: order rejected (would cross) "
                        f"@ ${trade['market_price']:.4f}"
                    )
                    all_reasons[trade["city"]] = "maker order rejected (would cross)"
                    continue
                order_id = result.get("order_id")
                if order_id:
                    print(f"  {trade['city']}: posted order {order_id}")
                live_trades.append({**trade, "order_result": result})
            sized_trades = live_trades
        else:
            for trade in sized_trades:
                print(
                    f"  LIVE: would place order {trade['city']} | "
                    f"{trade['bucket_label']} | {trade['n_contracts']} contracts "
                    f"@ ${trade['market_price']:.2f} (post_only GTC)"
                )

    total_cap = round(sum(t["capital_at_risk"] for t in sized_trades), 2)
    daily_cap = float(config["daily_loss_cap"])
    no_signal_cities = sorted(
        city for city in poly_config["cities"]
        if city not in {t["city"] for t in sized_trades}
    )

    decision = {
        "date": event_date,
        "mode": args.mode,
        "exchange": "polymarket",
        "bankroll": bankroll,
        "cities_attempted": poly_config["cities"],
        "n_cities_eligible": len(poly_config["cities"]),
        "n_cities_with_forecast": len(forecasts),
        "n_trades_selected": len(sized_trades),
        "edge_threshold": edge_threshold,
        "daily_loss_cap": daily_cap,
        "fee_model": "maker_zero",
        "trades": sized_trades,
        "total_capital_at_risk": total_cap,
        "daily_loss_cap_remaining": round(max(daily_cap - total_cap, 0.0), 2),
        "no_signal_cities": no_signal_cities,
        "no_signal_reasons": {
            city: all_reasons[city] for city in no_signal_cities if city in all_reasons
        },
        "forecast_notes": forecast_notes,
        "raw_forecasts": raw_forecasts,
        "wunderground_bias_applied": bias_applied,
        "price_floor": POLY_PRICE_FLOOR,
    }

    log_decision_poly(decision)
    daily_risk_report_poly(
        decision,
        skipped_edges,
        args.mode,
        sanity_rows=sanity_rows,
        forecasts=forecasts,
        city_config=city_config,
        market_reasons=market_reasons,
    )


if __name__ == "__main__":
    main()
