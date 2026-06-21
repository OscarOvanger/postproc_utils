"""Market exchange API wrapper. Paper mode by default."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from dateutil.tz import gettz

    def ZoneInfo(name: str):
        tz = gettz(name)
        if tz is None:
            raise ValueError(f"Unknown timezone: {name}")
        return tz

PAPER_TRADE = True
LIVE_LOG_DIR = Path("data/live")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "city_config.json"


def fetch_market_snapshot(series_ticker: str, event_date: date) -> pd.DataFrame:
    """Fetch current bucket prices for a city's Tmax market."""
    raise NotImplementedError(
        "Market API integration pending. "
        "Need API keys from MCP Slack channel. "
        "For paper trading, use historical snapshots from market_df."
    )


def fetch_nws_forecast_live(city: str) -> float:
    """Fetch the most recent NWS Tmax forecast for today."""
    from src.trackj.fetch_nws_forecast import _issued_before_for_target, fetch_nws_tmax_forecast

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if city not in config:
        raise KeyError(f"Unknown city: {city}")
    cfg = config[city]
    target = date.today()
    issued_before = _issued_before_for_target(
        target, issued_before_hour=22, local_tz=ZoneInfo(cfg["timezone"])
    )
    result = fetch_nws_tmax_forecast(
        float(cfg["lat"]),
        float(cfg["lon"]),
        target,
        issued_before,
        station=cfg["nws_station"],
    )
    if result is None:
        raise RuntimeError(f"No NWS forecast available for {city} on {target}")
    return float(result["tmax_forecast_f"])


def place_order(
    market_ticker: str,
    side: str,
    price: float,
    contracts: int,
    model_prob: float,
    edge: float,
    bankroll_cents: int,
    daily_spent_cents: int,
    order_type: str = "limit",
    paper_trade: bool = True,
) -> dict:
    """Place an order with full validation."""
    fee_cents = math.ceil(0.07 * contracts * price * (1 - price))
    position_value_cents = int(contracts * price * 100)

    checks = {
        "price_above_floor": price >= 0.15,
        "edge_above_guardrail": edge > 2 * (fee_cents / (contracts * 100)),
        "contracts_within_max": contracts <= 8,
        "position_within_30pct": position_value_cents <= 0.30 * bankroll_cents,
        "daily_cap_not_exceeded": (daily_spent_cents + position_value_cents) <= 600,
        "bankroll_above_elimination": bankroll_cents > 7000,
    }
    all_passed = all(checks.values())

    decision = {
        "timestamp": datetime.now().isoformat(),
        "market_ticker": market_ticker,
        "side": side,
        "price": price,
        "contracts": contracts,
        "model_prob": model_prob,
        "edge": edge,
        "fee_cents": fee_cents,
        "position_value_cents": position_value_cents,
        "bankroll_cents": bankroll_cents,
        "daily_spent_cents": daily_spent_cents,
        "checks": checks,
        "all_checks_passed": all_passed,
        "paper_trade": paper_trade,
        "action": "TRADE" if all_passed else "SKIP",
    }

    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LIVE_LOG_DIR / "paper_orders.jsonl"
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision) + "\n")

    if not all_passed:
        failed = [key for key, value in checks.items() if not value]
        print(f"  SKIP: {market_ticker} — failed checks: {failed}")
        return {"status": "skipped", "reason": failed, **decision}

    if paper_trade or PAPER_TRADE:
        print(f"  PAPER TRADE: {market_ticker} {side} {contracts}x @ ${price:.2f}")
        return {"status": "paper_filled", **decision}

    raise RuntimeError("Live trading not enabled. Set paper_trade=True.")
