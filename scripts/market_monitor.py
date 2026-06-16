"""Intraday exit monitor for open Kalshi Tmax positions.

Runs every 5 minutes during market hours, fetches current bucket prices,
evaluates exit rules, and logs/notifies when triggered.

Add to crontab for 5-min monitoring during market hours:
  */5 10-23 * * 1-6 cd ~/MCP_Project && python scripts/market_monitor.py
Or for a specific date:
  */5 10-23 * * * cd ~/MCP_Project && python scripts/market_monitor.py --date 2026-06-16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
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
from src.entry_interface import filter_to_trading_window  # noqa: E402
from src.fees import taker_fee  # noqa: E402
from src.kalshi_api import place_order  # noqa: E402
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "historic_tmax_market_data"
CT = ZoneInfo("America/Chicago")
PROFIT_TARGET_CENTS = 15
MODEL_PROB_EXIT = True
STOP_LOSS_CENTS = 10
EDGE_THRESHOLD = 0.037
PRICE_FLOOR = 0.15
LOG_DIR = PROJECT_ROOT / "logs"
MONITOR_LOG = LOG_DIR / "monitor_events.jsonl"
PAPER_LOG = LOG_DIR / "paper_trades.jsonl"
EXIT_SIGNALS_LOG = LOG_DIR / "exit_signals.jsonl"
SETTLEMENT_LOG = LOG_DIR / "settlements.jsonl"

TEST_POSITION = {
    "city": "houston",
    "bucket_label": "84-85",
    "model_prob": 0.42,
    "market_price": 0.35,
    "edge": 0.07,
    "side": "YES",
    "n_contracts": 5,
    "capital_at_risk": 1.75,
    "fee": 0.01,
}

EXIT_RULE_LABELS = {
    "profit_target_15c": "PROFIT TARGET 15c",
    "profit_target_model": "PROFIT TARGET MODEL",
    "stop_loss_10c": "STOP LOSS 10c",
}


def _now_ct() -> datetime:
    return datetime.now(tz=CT)


def _timestamp_ct() -> str:
    return _now_ct().isoformat()


def _price_cents(price: float) -> int:
    return int(round(float(price) * 100))


def _round_price(price: float) -> float:
    return round(float(price), 4)


def load_deploy_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_city_config(config: dict[str, Any]) -> dict[str, Any]:
    sigma_path = PROJECT_ROOT / config["sigma_source"]
    with open(sigma_path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_decision(event_date: str, mode: str) -> dict | None:
    match = None
    with open(PAPER_LOG, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("date") == event_date and entry.get("mode", "paper") == mode:
                match = entry
    return match


def _load_exited_keys(event_date: str) -> set[tuple[str, str]]:
    exited: set[tuple[str, str]] = set()
    if not EXIT_SIGNALS_LOG.exists():
        return exited
    with open(EXIT_SIGNALS_LOG, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("event_date") != event_date:
                continue
            city = entry.get("city")
            bucket = entry.get("bucket_label")
            if city and bucket:
                exited.add((str(city), str(bucket)))
    return exited


def _load_settled_keys(event_date: str) -> set[tuple[str, str]]:
    settled: set[tuple[str, str]] = set()
    if not SETTLEMENT_LOG.exists():
        return settled
    with open(SETTLEMENT_LOG, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("date") != event_date:
                continue
            for trade in entry.get("trades", []):
                city = trade.get("city")
                bucket = trade.get("bucket_label")
                if city and bucket:
                    settled.add((str(city), str(bucket)))
    return settled


def load_open_positions(event_date: str, mode: str = "paper") -> list[dict]:
    """Load today's open positions from the paper trade log."""
    if not PAPER_LOG.exists():
        raise SystemExit(f"Paper trade log not found at {PAPER_LOG}")

    decision = _load_decision(event_date, mode)
    if decision is None:
        return []

    if decision.get("settled"):
        return []

    trades = list(decision.get("trades", []))
    if not trades:
        return []

    exited = _load_exited_keys(event_date)
    settled = _load_settled_keys(event_date)
    skip_keys = exited | settled

    open_positions: list[dict] = []
    for trade in trades:
        city = str(trade.get("city", ""))
        bucket = str(trade.get("bucket_label", ""))
        if not city or not bucket:
            continue
        if (city, bucket) in skip_keys:
            continue
        open_positions.append(trade)

    return open_positions


def _codex_city_map() -> dict[str, dict]:
    codex = _load_codex()
    return {city["slug"]: city for city in codex.CITIES if city["slug"] in TRAIN_SLUGS}


def _latest_bucket_prices(city: str, event_date: str) -> dict[str, float]:
    city_csvs = discover_city_csvs(RAW_DATA_DIR)
    csv_path = city_csvs.get(city)
    if csv_path is None:
        raise FileNotFoundError(f"no local market CSV for {city}")

    df = load_city_frame(city, csv_path)
    df["event_date_key"] = pd.to_datetime(df["event_date"]).astype(str)
    day = df[df["event_date_key"] == event_date].copy()
    if day.empty:
        raise ValueError(f"no market rows for {city} on {event_date}")

    day["snapshot_time_local"] = pd.to_datetime(day["snapshot_time_local"])
    latest_time = day["snapshot_time_local"].max()
    snapshot = day[day["snapshot_time_local"].eq(latest_time)].copy()

    prices: dict[str, float] = {}
    for _, row in snapshot.iterrows():
        label = str(row.get("bucket_label", ""))
        if not label:
            continue
        val = pd.to_numeric(row.get("yes_mid_close"), errors="coerce")
        if pd.isna(val):
            continue
        prices[label] = _round_price(float(val))
    if not prices:
        raise ValueError(f"no yes_mid_close prices for {city} on {event_date}")
    return prices


def fetch_current_prices(cities: list[str], event_date: str) -> dict[str, dict[str, float]]:
    """Fetch current Kalshi bucket prices for each city."""
    target = date.fromisoformat(event_date)
    city_map = _codex_city_map()
    codex = _load_codex()
    result: dict[str, dict[str, float]] = {}

    for city in cities:
        city_def = city_map.get(city)
        if city_def is None:
            print(f"  WARNING: unknown city in Codex map: {city}")
            continue
        try:
            fetch_city_dates(
                codex,
                city_def,
                target,
                target,
                merge_each=True,
                paper_live=True,
                force_refresh=True,
            )
            result[city] = _latest_bucket_prices(city, event_date)
        except Exception as exc:
            print(f"  WARNING: price fetch failed for {city}: {exc}")

    return result


def check_exit_conditions(
    position: dict,
    current_prices: dict[str, float],
) -> dict | None:
    """Evaluate exit rules for one open position."""
    bucket_label = str(position.get("bucket_label", ""))
    current_price = current_prices.get(bucket_label)
    if current_price is None:
        return None

    entry_price = _round_price(position.get("market_price", 0.0))
    n_contracts = int(position.get("n_contracts", 0) or 0)
    model_prob = _round_price(position.get("model_prob", 0.0))
    gain_cents = round((current_price - entry_price) * 100, 2)
    exit_fee = taker_fee(n_contracts, current_price)
    net_gain = round(gain_cents * n_contracts - exit_fee, 2)

    exit_rule: str | None = None
    if gain_cents >= PROFIT_TARGET_CENTS:
        exit_rule = "profit_target_15c"
    elif MODEL_PROB_EXIT and current_price >= model_prob:
        exit_rule = "profit_target_model"
    elif gain_cents <= -STOP_LOSS_CENTS:
        exit_rule = "stop_loss_10c"

    if exit_rule is None:
        return None

    return {
        "city": str(position.get("city", "")),
        "bucket_label": bucket_label,
        "entry_price": entry_price,
        "current_price": current_price,
        "gain_cents": gain_cents,
        "net_gain_cents": net_gain,
        "exit_fee_cents": exit_fee,
        "n_contracts": n_contracts,
        "model_prob": model_prob,
        "exit_rule": exit_rule,
        "action": "SELL",
    }


def _city_day_frame(city: str, event_date: str) -> pd.DataFrame:
    city_csvs = discover_city_csvs(RAW_DATA_DIR)
    csv_path = city_csvs.get(city)
    if csv_path is None:
        return pd.DataFrame()
    df = load_city_frame(city, csv_path)
    df["event_date_key"] = pd.to_datetime(df["event_date"]).astype(str)
    day = df[df["event_date_key"] == event_date].copy()
    if day.empty:
        return day
    day["snapshot_time_local"] = pd.to_datetime(day["snapshot_time_local"])
    latest_time = day["snapshot_time_local"].max()
    return day[day["snapshot_time_local"].eq(latest_time)].copy()


def check_new_entries(
    event_date: str,
    current_prices: dict[str, dict[str, float]],
    existing_cities: set[str],
    decision: dict | None,
    deploy_config: dict[str, Any],
    city_config: dict[str, Any],
) -> list[dict]:
    """Flag informational new-entry opportunities (no auto-entry in v1)."""
    if decision is None:
        return []

    forecasts = decision.get("forecasts")
    if not isinstance(forecasts, dict) or not forecasts:
        return []

    opportunities: list[dict] = []
    for city in deploy_config.get("cities", []):
        if city in existing_cities:
            continue
        tmax = forecasts.get(city)
        if tmax is None:
            continue
        city_prices = current_prices.get(city)
        if not city_prices:
            continue

        day_df = _city_day_frame(city, event_date)
        if day_df.empty:
            continue
        day_df = filter_to_trading_window(day_df)
        if day_df.empty:
            continue

        buckets = day_df[
            [
                "bucket_label",
                "bucket_type",
                "bucket_lower_inclusive_f",
                "bucket_upper_inclusive_f",
            ]
        ].drop_duplicates("bucket_label")
        sigma = float(city_config.get(city, {}).get("trackb_sigma_f", 0))
        if sigma <= 0:
            continue

        probs = bucket_probs_from_point_forecast(float(tmax), sigma, buckets)
        best: dict | None = None
        for bucket_label, model_prob in probs.items():
            price = city_prices.get(str(bucket_label))
            if price is None:
                continue
            edge = round(float(model_prob) - price, 4)
            if price < PRICE_FLOOR or edge < EDGE_THRESHOLD:
                continue
            candidate = {
                "city": city,
                "bucket_label": str(bucket_label),
                "edge": edge,
                "current_price": price,
                "model_prob": _round_price(float(model_prob)),
            }
            if best is None or candidate["edge"] > best["edge"]:
                best = candidate
        if best is not None:
            opportunities.append(best)

    return opportunities


def log_monitor_event(event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**event, "timestamp_ct": _timestamp_ct()}
    with open(MONITOR_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def log_exit_signal(signal: dict, event_date: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**signal, "event_date": event_date, "timestamp_ct": _timestamp_ct()}
    with open(EXIT_SIGNALS_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def send_notification(message: str) -> None:
    user_key = os.environ.get("PUSHOVER_USER_KEY")
    api_token = os.environ.get("PUSHOVER_API_TOKEN")
    if user_key and api_token:
        try:
            import requests

            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={"token": api_token, "user": user_key, "message": message},
                timeout=15,
            )
            response.raise_for_status()
            return
        except Exception as exc:
            print(f"[ALERT] Pushover failed ({exc}): {message}")
            return
    print(f"[ALERT] {message}")


def _format_status_line(
    position: dict,
    city_prices: dict[str, float] | None,
    signal: dict | None,
) -> str:
    city = str(position.get("city", ""))
    bucket = str(position.get("bucket_label", ""))
    model_prob = position.get("model_prob", 0.0)
    entry_price = _round_price(position.get("market_price", 0.0))

    if city_prices is None:
        return f"SKIP:   {city:<12} | {bucket:<6} | no market data"

    current_price = city_prices.get(bucket)
    if current_price is None:
        return f"SKIP:   {city:<12} | {bucket:<6} | bucket not in snapshot"

    gain_cents = round((current_price - entry_price) * 100, 2)
    if signal is not None:
        rule_label = EXIT_RULE_LABELS.get(signal["exit_rule"], signal["exit_rule"])
        return (
            f"EXIT:   {city:<12} | {bucket:<6} | "
            f"now {_price_cents(current_price):>2}c "
            f"(entry {_price_cents(entry_price):>2}c, {gain_cents:+.0f}c) | {rule_label}"
        )

    return (
        f"HOLD:   {city:<12} | {bucket:<6} | "
        f"now {_price_cents(current_price):>2}c "
        f"(entry {_price_cents(entry_price):>2}c, {gain_cents:+.0f}c) | "
        f"model {_price_cents(model_prob):>2}c"
    )


def _format_exit_alert(signal: dict) -> str:
    return (
        f"EXIT SIGNAL: {signal['city']} {signal['bucket_label']} "
        f"@ {_price_cents(signal['current_price'])}c "
        f"(entry {_price_cents(signal['entry_price'])}c, "
        f"gain {signal['gain_cents']:+.0f}c, rule: {signal['exit_rule']})"
    )


def _execute_live_exit(
    signal: dict,
    event_date: str,
    city_config: dict[str, Any],
) -> None:
    city = signal["city"]
    series = city_config.get(city, {}).get("kalshi_series")
    if not series:
        print(f"  LIVE EXIT SKIP: missing kalshi_series for {city}")
        return
    try:
        place_order(
            market_ticker=f"{series}-{event_date}",
            side="NO",
            price=_round_price(1.0 - signal["current_price"]),
            contracts=int(signal["n_contracts"]),
            model_prob=float(signal.get("model_prob", 0.0)),
            edge=0.0,
            bankroll_cents=0,
            daily_spent_cents=0,
            paper_trade=False,
        )
    except Exception as exc:
        print(f"  LIVE EXIT FAILED: {signal['city']} {signal['bucket_label']} — {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor open Tmax positions for exit signals")
    parser.add_argument("--date", type=str, default=str(date.today()))
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "deploy_config.json"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Check only; no logs or alerts")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use hardcoded test position against live market data",
    )
    parser.add_argument(
        "--execute-exits",
        action="store_true",
        help="Place live exit orders (requires --mode live; default is notify only)",
    )
    args = parser.parse_args()

    event_date = args.date
    now = _now_ct()
    header = f"=== MARKET MONITOR: {event_date} {now.strftime('%H:%M')} CT ==="
    print(header)

    if not args.test and not PAPER_LOG.exists():
        raise SystemExit(f"Paper trade log not found at {PAPER_LOG}")

    deploy_config = load_deploy_config(Path(args.config))
    city_config = load_city_config(deploy_config)
    decision = None if args.test else _load_decision(event_date, args.mode)

    if args.test:
        positions = [TEST_POSITION]
    else:
        positions = load_open_positions(event_date, mode=args.mode)

    if not positions:
        print(f"No open positions for {event_date}")
        return

    print(f"Positions: {len(positions)} open\n")

    cities_with_positions = sorted({str(p.get("city", "")) for p in positions if p.get("city")})
    current_prices = fetch_current_prices(cities_with_positions, event_date)
    if not current_prices:
        print("ERROR: failed to fetch current prices for all cities")
        raise SystemExit(1)

    status_rows: list[str] = []
    summary_positions: list[dict] = []
    exit_signals: list[dict] = []

    for position in positions:
        city = str(position.get("city", ""))
        city_prices = current_prices.get(city)
        if city_prices is None:
            print(f"  WARNING: no prices for {city}")
            status_rows.append(_format_status_line(position, None, None))
            summary_positions.append(
                {
                    "city": city,
                    "bucket_label": position.get("bucket_label"),
                    "status": "no_prices",
                }
            )
            continue

        signal = check_exit_conditions(position, city_prices)
        status_rows.append(_format_status_line(position, city_prices, signal))

        bucket = str(position.get("bucket_label", ""))
        entry_price = _round_price(position.get("market_price", 0.0))
        current_price = city_prices.get(bucket)
        gain_cents = None
        if current_price is not None:
            gain_cents = round((current_price - entry_price) * 100, 2)

        summary_positions.append(
            {
                "city": city,
                "bucket_label": bucket,
                "entry_price": entry_price,
                "current_price": current_price,
                "gain_cents": gain_cents,
                "status": "exit" if signal else "hold",
                "exit_rule": signal.get("exit_rule") if signal else None,
            }
        )

        if signal is None:
            continue

        alert = _format_exit_alert(signal)
        exit_signals.append(signal)
        if not args.dry_run:
            log_exit_signal(signal, event_date)
            send_notification(alert)

    for row in status_rows:
        print(row)

    if not args.dry_run:
        log_monitor_event(
            {
                "event_date": event_date,
                "mode": args.mode,
                "n_positions": len(positions),
                "positions": summary_positions,
                "exit_signals": exit_signals,
            }
        )

    existing_cities = {str(p.get("city", "")) for p in positions}
    new_entries = check_new_entries(
        event_date,
        current_prices,
        existing_cities,
        decision,
        deploy_config,
        city_config,
    )
    if new_entries:
        print()
        for entry in new_entries:
            line = (
                f"New entries: {entry['city']} {entry['bucket_label']} "
                f"edge={entry['edge']:.3f} @ {_price_cents(entry['current_price'])}c"
            )
            print(line)

    print("\n===")

    if (
        args.mode == "live"
        and args.execute_exits
        and not args.dry_run
        and exit_signals
    ):
        for signal in exit_signals:
            _execute_live_exit(signal, event_date, city_config)


if __name__ == "__main__":
    main()
