"""Daily trading pipeline. Run at 10:00 AM CT."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

import joblib
import numpy as np
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

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from dateutil.tz import gettz

    def ZoneInfo(name: str):
        tz = gettz(name)
        if tz is None:
            raise ValueError(f"Unknown timezone: {name}")
        return tz

from build_splits import discover_city_csvs, load_city_frame  # noqa: E402
from fetch_recent_market_days import TRAIN_SLUGS, _load_codex, fetch_city_dates  # noqa: E402
from src.data_pipeline import _load_cli_target, build_feature_vector_strict, fetch_kalshi_snapshot  # noqa: E402
from src.entry_interface import filter_to_trading_window  # noqa: E402
from src.fees import taker_fee  # noqa: E402
from src.kalshi_api import place_order  # noqa: E402
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402
from src.sizing import has_edge, taker_fee_cents  # noqa: E402
from src.snapshot_stability import load_or_create_frozen_k, stability_entry  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "historic_tmax_market_data"
CT = ZoneInfo("America/Chicago")
FORECAST_SANITY_DELTA_F = 25.0


def load_deploy_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_city_config(config: dict[str, Any]) -> dict[str, Any]:
    sigma_path = PROJECT_ROOT / config["sigma_source"]
    with open(sigma_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_models(city: str, model_dir: Path) -> tuple[list, list[str]]:
    base = model_dir / city
    models = [
        joblib.load(base / "ridge.joblib"),
        joblib.load(base / "huber.joblib"),
        joblib.load(base / "lightgbm.joblib"),
    ]
    with open(base / "feature_cols.json", encoding="utf-8") as handle:
        feature_cols = json.load(handle)
    return models, feature_cols


def predict_tmax_strict(
    models: list,
    feature_cols: list[str],
    feature_row: dict[str, float],
) -> int | None:
    values = []
    for col in feature_cols:
        val = feature_row.get(col)
        if val is None or pd.isna(val):
            return None
        values.append(float(val))
    x = np.array(values, dtype=float).reshape(1, -1)
    preds = [model.predict(x)[0] for model in models]
    return int(round(float(np.mean(preds))))


def _now_ct() -> datetime:
    return datetime.now(tz=CT)


def _wait_for_market_open(event_date: str) -> bool:
    """Wait until 10:05 AM CT on event day, retrying every 60s until 10:10 AM."""
    target = date.fromisoformat(event_date)
    today_ct = _now_ct().date()
    if target != today_ct:
        return True

    deadline = datetime.combine(target, dtime(10, 10), tzinfo=CT)
    ready_after = datetime.combine(target, dtime(10, 5), tzinfo=CT)

    while _now_ct() < ready_after:
        now = _now_ct()
        if now >= deadline:
            print("Markets not available pre-open (past 10:10 AM CT). Aborting.")
            return False
        print(f"Before 10:05 AM CT ({now.strftime('%H:%M:%S')} CT). Waiting 60s...")
        time.sleep(60)

    return True


def _codex_city_map() -> dict[str, dict]:
    codex = _load_codex()
    return {city["slug"]: city for city in codex.CITIES if city["slug"] in TRAIN_SLUGS}


def fetch_market(
    config: dict[str, Any],
    event_date: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pull market bucket snapshots for all cities."""
    if not _wait_for_market_open(event_date):
        raise SystemExit("Markets not available pre-open")

    target = date.fromisoformat(event_date)
    city_map = _codex_city_map()
    reasons: dict[str, str] = {}

    print("\n--- fetch_market ---")
    codex = _load_codex()
    for city in config["cities"]:
        print(f"  Fetching market: {city}")
        city_def = city_map.get(city)
        if city_def is None:
            reasons[city] = "unknown city in Codex map"
            continue
        try:
            rows = fetch_city_dates(codex, city_def, target, target, merge_each=True, paper_live=True, force_refresh=True)
            if not rows:
                reasons[city] = "no market data"
        except Exception as exc:
            reasons[city] = f"market fetch error: {exc}"

    frames: list[pd.DataFrame] = []
    if RAW_DATA_DIR.exists():
        city_csvs = discover_city_csvs(RAW_DATA_DIR)
        for city in config["cities"]:
            csv_path = city_csvs.get(city)
            if csv_path is None:
                if city not in reasons:
                    reasons[city] = "no local market CSV"
                continue
            df = load_city_frame(city, csv_path)
            df["event_date_key"] = pd.to_datetime(df["event_date"]).astype(str)
            day = df[df["event_date_key"] == event_date].copy()
            if day.empty:
                if city not in reasons:
                    reasons[city] = "no market rows for date"
                continue
            day["city"] = city
            frames.append(day)

    market_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not market_df.empty:
        market_df["event_date"] = pd.to_datetime(market_df["event_date"])
        for col in ("yes_mid_close", "yes_bid_close", "yes_ask_close"):
            if col in market_df.columns:
                market_df[col] = pd.to_numeric(market_df[col], errors="coerce")

    for city in config["cities"]:
        if city in reasons:
            continue
        if market_df.empty:
            reasons[city] = "no market data"
            continue
        snapshot = fetch_kalshi_snapshot(city, event_date, market_df)
        if snapshot is None or snapshot.empty:
            reasons[city] = "stability not met"

    return market_df, reasons


def fetch_forecast(
    config: dict[str, Any],
    event_date: str,
    city_config: dict[str, Any],
) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    """Generate Track-B ensemble forecasts for all cities."""
    print("\n--- fetch_forecast ---")
    print("  Pre-warming CLI history for lag features...")
    for city in config["cities"]:
        try:
            _load_cli_target(city, event_date)
        except Exception as exc:
            print(f"    CLI pre-warm failed for {city}: {exc}")

    model_dir = PROJECT_ROOT / config["model_dir"]
    forecasts: dict[str, int] = {}
    reasons: dict[str, str] = {}
    notes: dict[str, str] = {}

    for city in config["cities"]:
        print(f"\n  {city}")
        try:
            models, feature_cols = load_models(city, model_dir)
        except FileNotFoundError:
            reasons[city] = "missing model artifacts"
            continue

        features, fail_reason = build_feature_vector_strict(city, event_date, feature_cols)
        if features is None:
            print(f"    {fail_reason}")
            reasons[city] = fail_reason
            continue

        pred = predict_tmax_strict(models, feature_cols, features)
        if pred is None:
            reasons[city] = "prediction failed (NaN features)"
            continue

        nws = features.get("nws_tmax_forecast_f")
        if nws is not None and abs(pred - nws) > FORECAST_SANITY_DELTA_F:
            notes[city] = f"forecast sanity warning: pred={pred}F vs NWS={nws:.0f}F"

        sigma = float(city_config[city]["trackb_sigma_f"])
        print(f"    Predicted Tmax: {pred}F (sigma={sigma:.2f})")
        forecasts[city] = pred

    return forecasts, reasons, notes


def _city_day_market(market_df: pd.DataFrame, city: str, event_date: str) -> pd.DataFrame:
    if market_df.empty:
        return market_df
    mask = market_df["city"].astype(str).str.lower().str.replace(" ", "_") == city
    day = market_df[mask].copy()
    day = day[day["event_date"].astype(str).str[:10] == event_date]
    return day


def compute_edge(
    market_df: pd.DataFrame,
    forecasts: dict[str, int],
    city_config: dict[str, Any],
    config: dict[str, Any],
    market_reasons: dict[str, str],
    event_date: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Compute best tradeable bucket per city."""
    print("\n--- compute_edge ---")
    edges: list[dict[str, Any]] = []
    reasons = dict(market_reasons)

    price_floor = float(config["price_floor"])
    for city in config["cities"]:
        if city in reasons:
            continue
        if city not in forecasts:
            if city not in reasons:
                reasons[city] = "no forecast"
            continue

        day_df = _city_day_market(market_df, city, event_date)
        if day_df.empty:
            reasons[city] = "no market data"
            continue

        day_df = filter_to_trading_window(day_df)
        stability = stability_entry(day_df, k=load_or_create_frozen_k())
        if stability.no_signal:
            reasons[city] = "stability not met"
            continue

        snapshot = day_df[
            pd.to_datetime(day_df["snapshot_time_local"]).eq(stability.entry_snapshot_time)
        ]
        buckets = snapshot[
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
            entry_rows = snapshot[snapshot["bucket_label"].astype(str).eq(str(bucket_label))]
            if entry_rows.empty:
                continue
            entry_price = float(entry_rows["yes_mid_close"].iloc[0])
            fee_per_contract = taker_fee_cents(1, entry_price) / 100.0
            edge = float(model_prob) - entry_price
            if entry_price < price_floor or not has_edge(model_prob, entry_price, fee_per_contract):
                continue
            candidate = {
                "city": city,
                "bucket_label": str(bucket_label),
                "model_prob": float(model_prob),
                "market_price": entry_price,
                "edge": edge,
                "side": "YES",
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


def select_trades(
    edges: list[dict[str, Any]],
    config: dict[str, Any],
    reasons: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Apply edge_threshold selection and rank by edge."""
    print("\n--- select_trades ---")
    threshold = float(config["edge_threshold"])
    excluded = set(config.get("excluded_cities_oos", []))
    selected: list[dict[str, Any]] = []

    for edge_row in sorted(edges, key=lambda row: row["edge"], reverse=True):
        city = edge_row["city"]
        if city in excluded:
            reasons[city] = "excluded OOS city"
            continue
        if edge_row["edge"] < threshold:
            reasons[city] = f"edge below threshold ({edge_row['edge']:.3f} < {threshold:.3f})"
            continue
        selected.append(edge_row)

    return selected, reasons


def size_positions(
    trades: list[dict[str, Any]],
    bankroll: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply flat sizing, drawdown scaling, and daily loss cap."""
    from src.sizing import assert_poly_order_notional, daily_cap_from_bankroll, poly_contracts_for_price

    print("\n--- size_positions ---")
    daily_cap = daily_cap_from_bankroll(bankroll, config)

    sized: list[dict[str, Any]] = []
    for trade in trades:
        price = float(trade["market_price"])
        n_contracts = poly_contracts_for_price(price)
        assert_poly_order_notional(n_contracts, price)
        fee_cents = taker_fee(n_contracts, price)
        sized_trade = {
            **trade,
            "n_contracts": n_contracts,
            "capital_at_risk": round(n_contracts * price, 4),
            "fee": round(fee_cents / 100.0, 4),
        }
        sized.append(sized_trade)

    while sized:
        total_cap = sum(t["capital_at_risk"] for t in sized)
        if total_cap <= daily_cap:
            break
        dropped = sized.pop()
        print(f"  Dropped {dropped['city']} (cap trim): edge={dropped['edge']:.3f}")

    return sized


def log_decision(
    decision: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Append structured decision log to JSONL."""
    log_dir = PROJECT_ROOT / config["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "paper_trades.jsonl"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, default=str) + "\n")
    print(f"\nDecision log appended to {log_path}")
    print(json.dumps(decision, indent=2, default=str))


def daily_risk_report(
    decision: dict[str, Any],
    skipped_edges: list[dict[str, Any]],
    mode: str,
) -> None:
    """Print human-readable daily risk summary."""
    event_date = decision["date"]
    bankroll = decision["bankroll"]
    n_cities = len(decision.get("cities_attempted", []))
    n_forecast = decision["n_cities_with_forecast"]
    n_trades = decision["n_trades_selected"]
    total_cap = decision["total_capital_at_risk"]
    daily_cap = decision.get("daily_loss_cap", 6.0)
    no_signal = decision.get("no_signal_reasons", {})

    print(f"\n=== DAILY RISK REPORT — {event_date} ({mode.upper()}) ===")
    print(f"Bankroll:           ${bankroll:.2f}")
    print(f"Trades selected:    {n_trades} / {n_cities} cities")
    print(f"Total cap at risk:  ${total_cap:.2f} / ${daily_cap:.2f} daily cap")
    coverage_notes = [
        f"{city}: {reason}" for city, reason in sorted(no_signal.items()) if city not in {
            t["city"] for t in decision.get("trades", [])
        }
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
            f"@ ${trade['market_price']:.2f} | fee=${trade['fee']:.2f}"
        )

    if skipped_edges:
        print()
        for row in skipped_edges:
            city = row["city"]
            if city in {t["city"] for t in decision.get("trades", [])}:
                continue
            reason = no_signal.get(city, "")
            if "below threshold" in reason or "edge below" in reason:
                print(f"Skipped: {city} (edge={row['edge']:.3f} < E*={decision.get('edge_threshold', 0.037):.3f})")
            elif reason:
                print(f"Skipped: {city} ({reason})")

    if mode == "paper":
        print("\n** PAPER MODE — no orders placed **")
        print("** To place manually: review edges above, enter on exchange UI **")
    print("===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Track-B trading pipeline")
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
        help="Bypass duplicate check in paper_trades.jsonl",
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="Build forecasts and exit without market fetch or trading",
    )
    args = parser.parse_args()

    config = load_deploy_config(Path(args.config))
    city_config = load_city_config(config)
    event_date = args.date
    bankroll = args.bankroll

    print(f"\n=== DAILY TRADE: {event_date} ({args.mode.upper()}) ===")

    log_dir = PROJECT_ROOT / config["log_dir"]
    log_path = log_dir / "paper_trades.jsonl"
    if log_path.exists() and not args.force:
        with open(log_path, encoding="utf-8") as handle:
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
        log_decision(decision, config)
        daily_risk_report(decision, [], args.mode)
        return

    for city, pred in sorted(forecasts.items()):
        note = forecast_notes.get(city, "")
        print(f"  {city}: {pred}F{' (' + note + ')' if note else ''}")
    for city, reason in sorted(forecast_reasons.items()):
        print(f"  {city}: SKIP ({reason})")

    if args.prefetch_only:
        print("\n--prefetch-only: stopping after feature build.")
        return

    print("\n--- PHASE 2: Fetch market snapshot ---")
    market_df, market_reasons = fetch_market(config, event_date)

    print("\n--- PHASE 3: Compute edge, select, size ---")
    all_reasons = {**market_reasons, **forecast_reasons}
    edges, edge_reasons = compute_edge(
        market_df, forecasts, city_config, config, all_reasons, event_date
    )
    all_reasons.update(edge_reasons)

    selected, all_reasons = select_trades(edges, config, all_reasons)
    sized_trades = size_positions(selected, bankroll, config)

    skipped_edges = [
        row for row in edges if row["city"] not in {t["city"] for t in sized_trades}
    ]

    if args.mode == "live":
        bankroll_cents = int(bankroll * 100)
        daily_spent = 0
        live_trades: list[dict[str, Any]] = []
        for trade in sized_trades:
            series = city_config[trade["city"]]["kalshi_series"]
            result = place_order(
                market_ticker=f"{series}-{event_date}",
                side="YES",
                price=trade["market_price"],
                contracts=trade["n_contracts"],
                model_prob=trade["model_prob"],
                edge=trade["edge"],
                bankroll_cents=bankroll_cents,
                daily_spent_cents=daily_spent,
                paper_trade=False,
            )
            daily_spent += int(result.get("position_value_cents", 0))
            live_trades.append({**trade, **result})
        sized_trades = live_trades

    total_cap = round(sum(t["capital_at_risk"] for t in sized_trades), 2)
    daily_cap = float(config["daily_loss_cap"])
    no_signal_cities = sorted(
        city for city in config["cities"] if city not in {t["city"] for t in sized_trades}
    )

    decision = {
        "date": event_date,
        "mode": args.mode,
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
        "no_signal_reasons": {city: all_reasons[city] for city in no_signal_cities if city in all_reasons},
        "forecast_notes": forecast_notes,
    }

    log_decision(decision, config)
    daily_risk_report(decision, skipped_edges, args.mode)


if __name__ == "__main__":
    main()
