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

from run_daily_trade import _now_ct, load_deploy_config  # noqa: E402
from src.poly_trading_pipeline import (  # noqa: E402
    POLYMARKET_CITIES,
    POLY_PRICE_FLOOR,
    build_poly_config,
    fetch_market,
    prepare_poly_trades,
)

POLY_PAPER_LOG = PROJECT_ROOT / "logs" / "poly_paper_trades.jsonl"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_bid_ask(
    best_bid: float | None,
    best_ask: float | None,
) -> str:
    def _fmt(value: float | None) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "-"
        return f"{value:.2f}"

    return f"{_fmt(best_bid)}/{_fmt(best_ask)}"


def _format_spread_cents(spread: float | None) -> str:
    if spread is None:
        return "-"
    return f"{int(round(spread * 100))}c"


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

        prices = pd.to_numeric(day_df["gamma_price"], errors="coerce").dropna()
        if prices.empty:
            print(f"  {city}: no valid prices")
            continue

        n_buckets = len(prices)
        n_above_floor = int((prices >= POLY_PRICE_FLOOR).sum())
        modal_idx = prices.idxmax()
        modal_row = day_df.loc[modal_idx]
        modal_bucket = str(modal_row["bucket_label"])
        modal_price = float(modal_row.get("market_price", prices.max()))
        modal_bid = _to_float(modal_row.get("yes_bid_close"))
        modal_ask = _to_float(modal_row.get("yes_ask_close"))
        modal_spread = _to_float(modal_row.get("spread"))

        print(f"  {city}: {n_buckets} buckets, {n_above_floor} above ${POLY_PRICE_FLOOR:.2f} floor")
        if modal_bid is not None and modal_ask is not None:
            print(
                f"    Modal: {modal_bucket} @ ${modal_price:.3f} "
                f"(bid={modal_bid:.2f}, ask={modal_ask:.2f}, "
                f"spread={_format_spread_cents(modal_spread)})"
            )
        else:
            print(f"    Modal: {modal_bucket} @ ${modal_price:.3f} (no order book)")
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
        print(f"  {'Bucket':<12} {'Model_P':>8} {'Mkt_P':>7} {'Bid/Ask':>10} {'Edge':>7}  Status")
        sum_model = 0.0
        sum_market = 0.0
        for row in rows:
            sum_model += row["model_prob"]
            sum_market += row["market_price"]
            bid_ask = _format_bid_ask(row.get("best_bid"), row.get("best_ask"))
            print(
                f"  {row['bucket_label']:<12} "
                f"{row['model_prob']:>8.3f} "
                f"{row['market_price']:>7.3f} "
                f"{bid_ask:>10} "
                f"{row['edge']:>+7.3f}  {row['status']}"
            )
        print(f"  {'Sum:':<12} {sum_model:>8.3f} {sum_market:>7.3f}")


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
        bid = trade.get("best_bid")
        ask = trade.get("best_ask")
        spread = trade.get("spread")
        bid_str = f"${bid:.2f}" if bid is not None else "-"
        ask_str = f"${ask:.2f}" if ask is not None else "-"
        spread_str = _format_spread_cents(spread)
        print(
            f"Trade {idx}: {trade['city']} | {trade['bucket_label']} | "
            f"edge={trade['edge']:+.3f} | {trade['n_contracts']} @ "
            f"${trade['market_price']:.2f} (ask={ask_str}, bid={bid_str}, "
            f"spread={spread_str}) | maker GTC | "
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


def _build_decision(
    *,
    event_date: str,
    mode: str,
    bankroll: float,
    poly_config: dict[str, Any],
    sized_trades: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    edge_threshold = float(poly_config["edge_threshold"])
    daily_cap = float(poly_config["daily_loss_cap"])
    forecasts = metadata.get("forecasts") or {}
    all_reasons = metadata.get("all_reasons") or {}
    total_cap = round(sum(t["capital_at_risk"] for t in sized_trades), 2)
    no_signal_cities = sorted(
        city for city in poly_config["cities"]
        if city not in {t["city"] for t in sized_trades}
    )
    return {
        "date": event_date,
        "mode": mode,
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
        "forecast_notes": metadata.get("forecast_notes", {}),
        "raw_forecasts": metadata.get("raw_forecasts", {}),
        "wunderground_bias_applied": metadata.get("bias_applied", {}),
        "price_floor": POLY_PRICE_FLOOR,
    }


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

    config_path = Path(args.config)
    poly_config, city_config = build_poly_config(config_path)
    event_date = args.date
    bankroll = args.bankroll
    edge_threshold = float(poly_config["edge_threshold"])

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

    if args.prefetch_only:
        from src.poly_trading_pipeline import load_wunderground_bias, apply_wunderground_bias  # noqa: E402
        from run_daily_trade import fetch_forecast  # noqa: E402

        print("\n--- PHASE 1: Pre-fetch features ---")
        forecasts, forecast_reasons, forecast_notes = fetch_forecast(
            poly_config, event_date, city_config
        )
        print(f"\nFeature coverage: {len(forecasts)}/{len(poly_config['cities'])} cities")
        if forecasts:
            print("\n--- Wunderground bias adjustment ---")
            apply_wunderground_bias(forecasts, load_wunderground_bias())
        print("\n--prefetch-only: stopping after feature build.")
        return

    sized_trades, metadata = prepare_poly_trades(
        event_date,
        bankroll,
        config_path,
    )

    if metadata.get("abort_reason") == "no_forecasts":
        print("ABORT: 0 cities have forecast coverage. Fix data sources.")
        decision = _build_decision(
            event_date=event_date,
            mode=args.mode,
            bankroll=bankroll,
            poly_config=poly_config,
            sized_trades=[],
            metadata=metadata,
        )
        log_decision_poly(decision)
        daily_risk_report_poly(decision, [], args.mode)
        return

    market_df = metadata["market_df"]
    market_reasons = metadata["market_reasons"]
    _print_market_diagnostics(
        market_df, POLYMARKET_CITIES, event_date, market_reasons
    )

    if args.mode == "live":
        if args.live_confirm:
            from src.polymarket_api import PolymarketClient  # noqa: E402

            poly_client = PolymarketClient()
            live_trades: list[dict[str, Any]] = []
            all_reasons = metadata["all_reasons"]
            for trade in sized_trades:
                result = poly_client.place_order(
                    token_id=trade["yes_token_id"],
                    side="YES",
                    price=float(trade.get("maker_entry_price") or trade["market_price"]),
                    size=float(trade["n_contracts"]),
                    tick_size=str(trade.get("tick_size", "0.01")),
                    dry_run=False,
                    post_only=True,
                )
                if result.get("status") == "rejected_would_cross":
                    maker_px = trade.get("maker_entry_price") or trade["market_price"]
                    print(
                        f"  {trade['city']}: order rejected (would cross) "
                        f"@ ${maker_px:.4f}"
                    )
                    all_reasons[trade["city"]] = "maker order rejected (would cross)"
                    continue
                order_id = result.get("order_id")
                if order_id:
                    print(f"  {trade['city']}: posted order {order_id}")
                live_trades.append({**trade, "order_result": result})
            sized_trades = live_trades
            metadata["all_reasons"] = all_reasons
        else:
            for trade in sized_trades:
                maker_px = trade.get("maker_entry_price") or trade["market_price"]
                print(
                    f"  LIVE: would place order {trade['city']} | "
                    f"{trade['bucket_label']} | {trade['n_contracts']} contracts "
                    f"@ ${maker_px:.2f} (post_only GTC)"
                )

    decision = _build_decision(
        event_date=event_date,
        mode=args.mode,
        bankroll=bankroll,
        poly_config=poly_config,
        sized_trades=sized_trades,
        metadata=metadata,
    )
    log_decision_poly(decision)
    daily_risk_report_poly(
        decision,
        metadata["skipped_edges"],
        args.mode,
        sanity_rows=metadata["sanity_rows"],
        forecasts=metadata["forecasts"],
        city_config=city_config,
        market_reasons=market_reasons,
    )


if __name__ == "__main__":
    main()
