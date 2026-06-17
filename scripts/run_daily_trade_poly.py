"""Daily trading pipeline for Polymarket. Run at 10:00 AM CT."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

import pandas as pd

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
    _wait_for_market_open,
    fetch_forecast,
    load_city_config,
    load_deploy_config,
    select_trades,
)
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402
from src.polymarket_api import (  # noqa: E402
    PolymarketClient,
    polymarket_maker_fee,
    polymarket_taker_fee,
)
from src.sizing import has_edge  # noqa: E402

POLY_PAPER_LOG = PROJECT_ROOT / "logs" / "poly_paper_trades.jsonl"


def compute_maker_price(best_bid: float, best_ask: float, tick_size: float) -> float:
    """Place order at midpoint, ensuring it stays on the bid side."""
    midpoint = (best_bid + best_ask) / 2
    maker_price = math.floor(midpoint / tick_size) * tick_size
    if maker_price >= best_ask:
        maker_price = best_ask - tick_size
    if maker_price < best_bid:
        maker_price = best_bid
    return round(maker_price, 4)


def apply_maker_pricing(
    trades: list[dict[str, Any]],
    client: PolymarketClient,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Attach maker limit prices from live order book."""
    priced: list[dict[str, Any]] = []
    skip_reasons: dict[str, str] = {}
    for trade in trades:
        token_id = str(trade["token_id"])
        tick_size = float(trade.get("tick_size", "0.01"))
        best_bid, best_ask = client.get_best_bid_ask(token_id)
        if best_bid is None or best_ask is None:
            skip_reasons[trade["city"]] = "no book for maker pricing"
            print(f"  {trade['city']}: SKIP (no book for maker pricing)")
            continue
        entry_price = compute_maker_price(best_bid, best_ask, tick_size)
        priced.append(
            {
                **trade,
                "entry_price": entry_price,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }
        )
        print(
            f"  {trade['city']}: maker price ${entry_price:.4f} "
            f"(bid=${best_bid:.2f}, ask=${best_ask:.2f})"
        )
    return priced, skip_reasons


def fetch_market_poly(
    client: PolymarketClient,
    config: dict[str, Any],
    event_date: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, Any]]]:
    """Fetch Polymarket bucket snapshots for all cities."""
    if not _wait_for_market_open(event_date):
        raise SystemExit("Markets not available pre-open")

    print("\n--- fetch_market_poly ---")
    reasons: dict[str, str] = {}
    market_lookup: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    poly_markets = client.fetch_tmax_markets(event_date)
    by_city = {market["city"]: market for market in poly_markets}

    for city in config["cities"]:
        print(f"  Fetching market: {city}")
        market = by_city.get(city)
        if market is None:
            reasons[city] = "no Polymarket market for date"
            continue
        if not market.get("buckets"):
            reasons[city] = "no bucket data"
            continue

        market_lookup[city] = market
        prices = client.fetch_bucket_prices(market["condition_id"], market=market)
        if not prices:
            reasons[city] = "no live prices"
            continue

        for bucket in market["buckets"]:
            midpoint = bucket.get("midpoint")
            if midpoint is None:
                continue
            rows.append(
                {
                    "city": city,
                    "event_date": event_date,
                    "condition_id": market["condition_id"],
                    "bucket_label": bucket["label"],
                    "bucket_type": bucket["bucket_type"],
                    "bucket_lower_inclusive_f": bucket["lower_f"],
                    "bucket_upper_inclusive_f": bucket["upper_f"],
                    "yes_mid_close": float(midpoint),
                    "token_id": bucket["token_id"],
                    "tick_size": bucket.get("tick_size", market.get("tick_size", "0.01")),
                    "neg_risk": bucket.get("neg_risk", market.get("neg_risk", True)),
                    "fee_rate_bps": market.get("fee_rate_bps", 0),
                }
            )

    market_df = pd.DataFrame(rows)
    for city in config["cities"]:
        if city in reasons:
            continue
        if market_df.empty or city not in set(market_df["city"].astype(str)):
            reasons[city] = "no market data"

    return market_df, reasons, market_lookup


def compute_edge_poly(
    market_df: pd.DataFrame,
    forecasts: dict[str, int],
    city_config: dict[str, Any],
    config: dict[str, Any],
    market_reasons: dict[str, str],
    event_date: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Compute best tradeable bucket per city using Polymarket prices."""
    print("\n--- compute_edge_poly ---")
    edges: list[dict[str, Any]] = []
    reasons = dict(market_reasons)
    price_floor = float(config["price_floor"])

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

        best: dict[str, Any] | None = None
        for bucket_label, model_prob in probs.items():
            entry_rows = day_df[day_df["bucket_label"].astype(str).eq(str(bucket_label))]
            if entry_rows.empty:
                continue
            row = entry_rows.iloc[0]
            entry_price = float(row["yes_mid_close"])
            token_id = str(row["token_id"])
            maker_fee = polymarket_maker_fee(1, entry_price)
            taker_fee_per_contract = polymarket_taker_fee(1, entry_price)
            edge = float(model_prob) - entry_price
            if entry_price < price_floor or not has_edge(model_prob, entry_price, maker_fee):
                continue
            candidate = {
                "city": city,
                "bucket_label": str(bucket_label),
                "model_prob": float(model_prob),
                "market_price": entry_price,
                "edge": edge,
                "side": "YES",
                "token_id": token_id,
                "condition_id": str(row["condition_id"]),
                "tick_size": str(row["tick_size"]),
                "neg_risk": bool(row["neg_risk"]),
                "maker_fee": maker_fee,
                "taker_fee_per_contract": taker_fee_per_contract,
            }
            if best is None or candidate["edge"] > best["edge"]:
                best = candidate

        if best is None:
            reasons[city] = "no bucket passes guardrails"
            continue

        edges.append(best)
        print(
            f"  {city}: {best['bucket_label']} edge={best['edge']:+.3f} "
            f"@ ${best['market_price']:.2f}"
        )

    return edges, reasons


def size_positions_poly(
    trades: list[dict[str, Any]],
    bankroll: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply flat sizing and daily loss cap with Polymarket fee model."""
    print("\n--- size_positions_poly ---")
    n_default = int(config["n_contracts_default"])
    n_reduced = int(config["n_contracts_reduced"])
    threshold = float(config["bankroll_reduction_threshold"])
    daily_cap = float(config["daily_loss_cap"])
    n_contracts = n_reduced if bankroll < threshold else n_default

    sized: list[dict[str, Any]] = []
    for trade in trades:
        price = float(trade.get("entry_price", trade["market_price"]))
        fee = polymarket_maker_fee(n_contracts, price)
        taker_fee = polymarket_taker_fee(n_contracts, price)
        sized.append(
            {
                **trade,
                "n_contracts": n_contracts,
                "capital_at_risk": round(n_contracts * price, 4),
                "fee": fee,
                "fee_type": "maker",
                "taker_fee": taker_fee,
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


def daily_risk_report_poly(
    decision: dict[str, Any],
    skipped_edges: list[dict[str, Any]],
    mode: str,
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
        order_price = trade.get("entry_price", trade["market_price"])
        print(
            f"Trade {idx}: {trade['city']} | {trade['bucket_label']} | "
            f"edge={trade['edge']:+.3f} | {trade['n_contracts']} contracts "
            f"@ ${order_price:.2f} (maker GTC) | fee=${trade['fee']:.2f}"
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

    if mode == "paper":
        print("\n** PAPER MODE — no orders placed **")
        print("** To place manually: review edges above, enter on Polymarket UI **")
    print("===")


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
        "--prefetch-only",
        action="store_true",
        help="Build forecasts and exit without market fetch or trading",
    )
    parser.add_argument(
        "--cancel-unfilled",
        action="store_true",
        help="Cancel all open GTC orders and exit",
    )
    args = parser.parse_args()

    config = load_deploy_config(Path(args.config))
    city_config = load_city_config(config)
    event_date = args.date
    bankroll = args.bankroll
    poly_client = PolymarketClient()

    print(f"\n=== DAILY TRADE (POLYMARKET): {event_date} ({args.mode.upper()}) ===")

    if args.cancel_unfilled:
        print("\n--- Cancel unfilled GTC orders ---")
        poly_client.cancel_unfilled_orders(event_date=event_date)
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
        config, event_date, city_config
    )
    n_forecasts = len(forecasts)
    print(f"\nFeature coverage: {n_forecasts}/{len(config['cities'])} cities")
    if n_forecasts == 0:
        print("ABORT: 0 cities have forecast coverage. Fix data sources.")
        decision = {
            "date": event_date,
            "mode": args.mode,
            "exchange": "polymarket",
            "bankroll": bankroll,
            "cities_attempted": config["cities"],
            "n_cities_eligible": len(config["cities"]),
            "n_cities_with_forecast": 0,
            "n_trades_selected": 0,
            "edge_threshold": float(config["edge_threshold"]),
            "daily_loss_cap": float(config["daily_loss_cap"]),
            "trades": [],
            "total_capital_at_risk": 0,
            "daily_loss_cap_remaining": float(config["daily_loss_cap"]),
            "no_signal_cities": sorted(config["cities"]),
            "no_signal_reasons": {**forecast_reasons},
            "forecast_notes": forecast_notes,
        }
        log_decision_poly(decision)
        daily_risk_report_poly(decision, [], args.mode)
        return

    for city, pred in sorted(forecasts.items()):
        note = forecast_notes.get(city, "")
        print(f"  {city}: {pred}F{' (' + note + ')' if note else ''}")
    for city, reason in sorted(forecast_reasons.items()):
        print(f"  {city}: SKIP ({reason})")

    if args.prefetch_only:
        print("\n--prefetch-only: stopping after feature build.")
        return

    print("\n--- PHASE 2: Fetch Polymarket snapshot ---")
    market_df, market_reasons, market_lookup = fetch_market_poly(
        poly_client, config, event_date
    )

    print("\n--- PHASE 3: Compute edge, select, size ---")
    all_reasons = {**market_reasons, **forecast_reasons}
    edges, edge_reasons = compute_edge_poly(
        market_df,
        forecasts,
        city_config,
        config,
        all_reasons,
        event_date,
    )
    all_reasons.update(edge_reasons)

    selected, all_reasons = select_trades(edges, config, all_reasons)

    print("\n--- apply_maker_pricing ---")
    priced_trades, pricing_skip_reasons = apply_maker_pricing(selected, poly_client)
    all_reasons.update(pricing_skip_reasons)

    sized_trades = size_positions_poly(priced_trades, bankroll, config)
    skipped_edges = [
        row for row in edges if row["city"] not in {t["city"] for t in sized_trades}
    ]

    if args.mode == "live":
        live_trades: list[dict[str, Any]] = []
        for trade in sized_trades:
            result = poly_client.place_order(
                token_id=trade["token_id"],
                side="YES",
                price=trade["entry_price"],
                size=float(trade["n_contracts"]),
                tick_size=trade.get("tick_size", "0.01"),
                neg_risk=bool(trade.get("neg_risk", True)),
                dry_run=False,
                post_only=True,
            )
            if result.get("status") == "rejected_would_cross":
                print(
                    f"  {trade['city']}: order rejected (would cross) "
                    f"@ ${trade['entry_price']:.4f}"
                )
                all_reasons[trade["city"]] = "maker order rejected (would cross)"
                continue
            order_id = result.get("order_id")
            if order_id:
                print(f"  {trade['city']}: posted order {order_id}")
            live_trades.append({**trade, "order_result": result})
        sized_trades = live_trades

    total_cap = round(sum(t["capital_at_risk"] for t in sized_trades), 2)
    daily_cap = float(config["daily_loss_cap"])
    no_signal_cities = sorted(
        city for city in config["cities"] if city not in {t["city"] for t in sized_trades}
    )

    decision = {
        "date": event_date,
        "mode": args.mode,
        "exchange": "polymarket",
        "bankroll": bankroll,
        "cities_attempted": config["cities"],
        "n_cities_eligible": len(config["cities"]),
        "n_cities_with_forecast": len(forecasts),
        "n_trades_selected": len(sized_trades),
        "edge_threshold": float(config["edge_threshold"]),
        "daily_loss_cap": daily_cap,
        "trades": sized_trades,
        "total_capital_at_risk": total_cap,
        "daily_loss_cap_remaining": round(max(daily_cap - total_cap, 0.0), 2),
        "no_signal_cities": no_signal_cities,
        "no_signal_reasons": {
            city: all_reasons[city] for city in no_signal_cities if city in all_reasons
        },
        "forecast_notes": forecast_notes,
    }

    log_decision_poly(decision)
    daily_risk_report_poly(decision, skipped_edges, args.mode)


if __name__ == "__main__":
    main()
