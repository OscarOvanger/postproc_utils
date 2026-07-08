"""Automated Polymarket Tmax trading loop with entry, monitoring, and exit."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.poly_trading_pipeline import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_poly_config,
    compute_maker_entry_price,
    fetch_market,
    poly_taker_fee,
    prepare_poly_trades,
)
from src.polymarket_api import (  # noqa: E402
    CLOB_HOST,
    PolymarketClient,
    fetch_order_book_http,
)
from src.sizing import (  # noqa: E402
    assert_poly_order_notional,
    daily_cap_from_bankroll,
    poly_contracts_for_price,
)

CT = ZoneInfo("America/Chicago")
STATE_DIR = PROJECT_ROOT / "logs"
BANKROLL_FILE = STATE_DIR / "current_bankroll.txt"
POLY_PAPER_LOG = STATE_DIR / "poly_paper_trades.jsonl"
PROFIT_TARGET = 0.15
ENTRY_TIMEOUT_MIN = 60
MONITOR_END_HOUR = 22
ENTRY_HOUR = 10
ENTRY_MINUTE = 5
BOOK_FETCH_DELAY_SEC = 0.2

MODAL_MAKER_CITIES = ["houston", "los_angeles"]
MIN_ENTRY_PRICE = 0.35
MAX_ENTRY_PRICE = 0.60
EXIT_THRESHOLD = 0.18
N_CONTRACTS = 5
STRATEGY_NAME = "modal_maker_18c"


def send_pushover(title: str, message: str) -> None:
    """Send push notification. Fail silently if not configured."""
    user_key = os.environ.get("PUSHOVER_USER_KEY") or os.environ.get("PUSHOVER_USER", "")
    api_token = os.environ.get("PUSHOVER_API_TOKEN") or os.environ.get("PUSHOVER_TOKEN", "")
    if not user_key or not api_token:
        print(f"  [PUSHOVER not configured] {title}: {message}")
        return
    try:
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode(
            {
                "token": api_token,
                "user": user_key,
                "title": title,
                "message": message,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=data,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)
    except TimeoutError as exc:
        print(f"  Pushover WARN (timeout; message may still deliver): {exc}")
    except Exception as exc:
        err = str(exc).lower()
        if "timed out" in err or "timeout" in err:
            print(f"  Pushover WARN (timeout; message may still deliver): {exc}")
        else:
            print(f"  Pushover failed: {exc}")


def git_freeze_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_deploy_cities() -> list[str]:
    poly_config, _ = build_poly_config(DEFAULT_CONFIG_PATH)
    return list(poly_config.get("cities", []))


def read_bankroll_file() -> float | None:
    if not BANKROLL_FILE.exists():
        return None
    try:
        return float(BANKROLL_FILE.read_text().strip())
    except ValueError:
        return None


def normalize_strategy_name(strategy: str | None) -> str | None:
    if strategy == "trackb":
        return "ngboost"
    return strategy


def operative_daily_cap(bankroll: float) -> float:
    poly_config, _ = build_poly_config(DEFAULT_CONFIG_PATH)
    cap = daily_cap_from_bankroll(bankroll, poly_config)
    print(
        f"operative daily cap at ${bankroll:.2f} = ${cap:.2f} "
        f"(source: sizing.daily_cap_from_bankroll)"
    )
    return cap


def emit_config_echo(mode: str, bankroll: float, strategy: str = "ngboost") -> str:
    poly_config, _ = build_poly_config(DEFAULT_CONFIG_PATH)
    cities = poly_config.get("cities", [])
    model_dir = poly_config.get("model_dir", "models/ngboost_v2")
    daily_cap = daily_cap_from_bankroll(bankroll, poly_config)
    lines = [
        "CONFIG ECHO (frozen D3)",
        f"  commit={git_freeze_hash()}",
        f"  mode={mode}",
        f"  strategy={strategy}",
        f"  model_dir={model_dir}",
        f"  bankroll=${bankroll:.2f}",
        f"  operative_daily_cap=${daily_cap:.2f}",
        f"  signal={poly_config.get('signal')}",
        f"  shrinkage_lambda={poly_config.get('shrinkage_lambda')}",
        f"  convective_cloud_skip_threshold={poly_config.get('convective_cloud_skip_threshold')}",
        f"  cities={len(cities)}: {', '.join(cities)}",
        f"  pace_fallback_enabled={poly_config.get('pace_fallback_enabled')}",
        f"  budget_divisor={poly_config.get('budget_divisor')}",
        f"  edge_threshold={poly_config.get('edge_threshold')}",
        f"  max_trades_per_day={poly_config.get('max_trades_per_day')}",
    ]
    print(
        f"operative daily cap at ${bankroll:.2f} = ${daily_cap:.2f} "
        f"(source: sizing.daily_cap_from_bankroll)"
    )
    message = "\n".join(lines)
    print(message)
    return message


def run_preflight(mode: str) -> bool:
    """Verify CLOB reachability and auth before live order placement."""
    print("\n--- PREFLIGHT ---")
    try:
        import requests

        resp = requests.get(f"{CLOB_HOST}/", timeout=10)
        resp.raise_for_status()
        print(f"  CLOB reachable ({resp.status_code})")
    except Exception as exc:
        print(f"  PREFLIGHT FAIL: CLOB unreachable: {exc}")
        send_pushover("Autotrader preflight failed", f"CLOB unreachable: {exc}")
        return False

    if mode != "live":
        print("  Auth check skipped (paper mode)")
        return True

    try:
        PolymarketClient().get_balance()
        print("  CLOB auth OK")
        return True
    except Exception as exc:
        print(f"  PREFLIGHT FAIL: CLOB auth error: {exc}")
        send_pushover("Autotrader preflight failed", f"CLOB auth error: {exc}")
        return False


def state_path(date_str: str) -> Path:
    return STATE_DIR / f"auto_trader_state_{date_str}.json"


def load_state(date_str: str) -> dict[str, Any] | None:
    path = state_path(date_str)
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
        if "strategy" in state:
            state["strategy"] = normalize_strategy_name(state["strategy"])
        if "daily_loss_cap" in state and "daily_budget_cap" not in state:
            state["daily_budget_cap"] = state.pop("daily_loss_cap")
        return state
    return None


def save_state(state: dict[str, Any]) -> None:
    path = state_path(state["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, default=str)
    tmp.rename(path)


def append_trade_log(state: dict[str, Any], event: dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.now(CT).isoformat(),
        **event,
    }
    state.setdefault("trades_log", []).append(entry)


def get_best_bid_ask(token_id: str) -> tuple[float | None, float | None]:
    """Fetch real order book via public HTTP. No credentials needed."""
    try:
        return fetch_order_book_http(token_id)
    except Exception as exc:
        print(f"  WARNING: book fetch failed for {token_id[:20]}...: {exc}")
        return None, None


def _get_poly_client():
    from src.polymarket_api import PolymarketClient

    return PolymarketClient()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _is_modal_strategy(state: dict[str, Any]) -> bool:
    return state.get("strategy") == STRATEGY_NAME


def _tick_float(tick_size: str) -> float:
    return float(tick_size)


def _fresh_edge_ok(
    trade: dict[str, Any],
    maker_price: float,
    poly_config: dict[str, Any],
) -> tuple[bool, float]:
    model_prob = float(trade["model_prob"])
    edge = model_prob - maker_price
    if trade.get("entry_mode") == "fallback":
        min_edge = float(poly_config.get("pace_fallback_min_edge", 0.01))
        return edge >= min_edge, edge
    threshold = float(poly_config["edge_threshold"])
    return edge >= threshold, edge


def _order_error_text(result: dict[str, Any]) -> str:
    return str(result.get("error") or result.get("status") or "unknown")


def _retryable_order_failure(result: dict[str, Any]) -> bool:
    status = str(result.get("status", ""))
    err = _order_error_text(result).lower()
    if status in ("rejected_would_cross", "error"):
        return True
    return any(token in err for token in ("post", "cross", "min size", "400", "invalid amount"))


def place_live_entry_with_fresh_book(
    trade: dict[str, Any],
    *,
    poly_config: dict[str, Any],
) -> tuple[dict[str, Any], float, int, float | None, float | None]:
    """Fetch fresh book, recheck edge, post maker entry with one retry."""
    token_id = trade["yes_token_id"]
    tick = _tick_float(str(trade.get("tick_size", "0.01")))
    maker_price = 0.0
    n_contracts = 0
    best_bid: float | None = None
    best_ask: float | None = None
    last_result: dict[str, Any] = {"status": "error", "error": "no_attempt"}

    for attempt in (1, 2):
        best_bid, best_ask = get_best_bid_ask(token_id)
        maker_price = compute_maker_entry_price(
            best_bid=best_bid,
            best_ask=best_ask,
            gamma_price=None,
            tick_size=tick,
        )
        if maker_price is None:
            return (
                {"status": "error", "error": "no_maker_price"},
                0.0,
                0,
                best_bid,
                best_ask,
            )

        ok, edge = _fresh_edge_ok(trade, float(maker_price), poly_config)
        if not ok:
            threshold = (
                poly_config.get("pace_fallback_min_edge")
                if trade.get("entry_mode") == "fallback"
                else poly_config["edge_threshold"]
            )
            err = f"fresh edge {edge:.4f} < {threshold} at ${maker_price:.2f}"
            return (
                {"status": "cancelled", "error": err},
                float(maker_price),
                0,
                best_bid,
                best_ask,
            )

        n_contracts = poly_contracts_for_price(float(maker_price))
        assert_poly_order_notional(n_contracts, float(maker_price))
        last_result = place_maker_entry(
            token_id=token_id,
            price=float(maker_price),
            size=n_contracts,
            tick_size=str(trade.get("tick_size", "0.01")),
        )
        if last_result.get("status") in ("posted", "signed"):
            return last_result, float(maker_price), n_contracts, best_bid, best_ask

        if attempt == 1 and _retryable_order_failure(last_result):
            print(
                f"  RETRY: {trade['city']} after {_order_error_text(last_result)}"
            )
            time.sleep(BOOK_FETCH_DELAY_SEC)
            continue
        break

    return last_result, float(maker_price), n_contracts, best_bid, best_ask


def place_maker_entry(
    *,
    token_id: str,
    price: float,
    size: int,
    tick_size: str = "0.01",
    neg_risk: bool = True,
) -> dict[str, Any]:
    client = _get_poly_client()
    return client.place_order(
        token_id=token_id,
        side="YES",
        price=price,
        size=float(size),
        tick_size=tick_size,
        neg_risk=neg_risk,
        dry_run=False,
        post_only=True,
    )


def place_maker_exit(
    *,
    token_id: str,
    price: float,
    size: int,
    tick_size: str = "0.01",
    neg_risk: bool = True,
) -> dict[str, Any]:
    client = _get_poly_client()
    return client.place_order(
        token_id=token_id,
        side="SELL",
        price=price,
        size=float(size),
        tick_size=tick_size,
        neg_risk=neg_risk,
        dry_run=False,
        post_only=True,
    )


def place_taker_exit(
    *,
    token_id: str,
    price: float,
    size: int,
    tick_size: str = "0.01",
    neg_risk: bool = True,
) -> dict[str, Any]:
    client = _get_poly_client()
    return client.place_taker_sell(
        token_id=token_id,
        price=price,
        size=float(size),
        tick_size=tick_size,
        neg_risk=neg_risk,
        dry_run=False,
    )


def compute_bucket_midpoint(
    best_bid: float | None,
    best_ask: float | None,
    fallback: float | None,
) -> float | None:
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    if best_ask is not None:
        return best_ask
    return fallback


def select_modal_trades(event_date: str, bankroll: float) -> list[dict[str, Any]]:
    """Fetch markets and select modal-bucket trades for MODAL_MAKER_CITIES."""
    print("\n--- PHASE 1: fetch_market (modal maker) ---")
    config = {"cities": list(MODAL_MAKER_CITIES)}
    market_df, reasons = fetch_market(
        config,
        event_date,
        wait_for_open=False,
        raise_on_no_market=False,
    )

    if market_df.empty:
        for city, reason in sorted(reasons.items()):
            print(f"  {city}: SKIP ({reason})")
        return []

    eligible: list[dict[str, Any]] = []
    for city in MODAL_MAKER_CITIES:
        if city in reasons:
            print(f"  {city}: SKIP ({reasons[city]})")
            continue

        city_df = market_df[market_df["city"].astype(str) == city].copy()
        if city_df.empty:
            print(f"  {city}: SKIP (no market data)")
            continue

        city_df["_midpoint"] = city_df.apply(
            lambda row: compute_bucket_midpoint(
                _to_float(row.get("yes_bid_close")),
                _to_float(row.get("yes_ask_close")),
                _to_float(row.get("gamma_price")),
            ),
            axis=1,
        )
        city_df = city_df.dropna(subset=["_midpoint"])
        if city_df.empty:
            print(f"  {city}: SKIP (no bucket prices)")
            continue

        modal_idx = city_df["_midpoint"].idxmax()
        modal_row = city_df.loc[modal_idx]
        best_bid = _to_float(modal_row.get("yes_bid_close"))
        best_ask = _to_float(modal_row.get("yes_ask_close"))
        modal_mid = float(modal_row["_midpoint"])
        bucket_label = str(modal_row["bucket_label"])

        print(
            f"  {city}: modal={bucket_label} mid={modal_mid:.3f} "
            f"bid={best_bid} ask={best_ask}"
        )

        if best_ask is None:
            print(f"  {city}: SKIP (no best_ask for modal bucket)")
            continue
        if best_ask < MIN_ENTRY_PRICE or best_ask > MAX_ENTRY_PRICE:
            print(
                f"  {city}: SKIP modal bucket price {best_ask:.2f} out of range "
                f"[{MIN_ENTRY_PRICE:.2f}, {MAX_ENTRY_PRICE:.2f}]"
            )
            continue

        tick_size = float(modal_row.get("tick_size", 0.01))
        maker_entry_price = compute_maker_entry_price(
            best_bid=best_bid,
            best_ask=best_ask,
            gamma_price=_to_float(modal_row.get("gamma_price")),
            tick_size=tick_size,
        )
        if maker_entry_price is None:
            print(f"  {city}: SKIP (could not compute maker entry price)")
            continue

        exit_target_price = round(maker_entry_price + EXIT_THRESHOLD, 2)
        n_contracts = poly_contracts_for_price(maker_entry_price)
        assert_poly_order_notional(n_contracts, maker_entry_price)
        eligible.append(
            {
                "city": city,
                "bucket_label": bucket_label,
                "yes_token_id": str(modal_row["yes_token_id"]),
                "condition_id": str(modal_row.get("condition_id", "")),
                "modal_bucket_probability": round(modal_mid, 4),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "maker_entry_price": maker_entry_price,
                "exit_target_price": exit_target_price,
                "tick_size": str(modal_row.get("tick_size", "0.01")),
                "n_contracts": n_contracts,
                "capital_at_risk": round(n_contracts * maker_entry_price, 4),
            }
        )

    daily_cap = operative_daily_cap(bankroll)
    eligible.sort(key=lambda trade: trade["maker_entry_price"])
    selected: list[dict[str, Any]] = []
    cumulative = 0.0
    for trade in eligible:
        if cumulative + trade["capital_at_risk"] > daily_cap:
            print(f"  Skipped {trade['city']} (daily cap ${daily_cap:.2f})")
            continue
        selected.append(trade)
        cumulative += trade["capital_at_risk"]

    print(f"\n--- PHASE 3: size and cap ---")
    print(f"  Selected: {len(selected)} trades (cap used ${cumulative:.2f})")
    return selected


def place_exit_order_for_position(
    state: dict[str, Any],
    pos: dict[str, Any],
    size: int | float,
    *,
    replace: bool = False,
    notify_on_fill: bool = False,
) -> bool:
    """Place or replace a maker sell at exit_target_price."""
    placed_size = pos.get("exit_order_size")
    if pos.get("exit_order_id") and not replace:
        if placed_size is None or placed_size == size:
            return True

    if replace and pos.get("exit_order_id"):
        exit_id = str(pos["exit_order_id"])
        if not exit_id.startswith("paper"):
            try:
                _get_poly_client().cancel_order(exit_id)
            except Exception as exc:
                print(f"  Cancel sell failed: {exc}")
        pos["exit_order_id"] = None
        pos["exit_order_size"] = None

    exit_price = float(pos["exit_target_price"])
    token_id = str(pos["yes_token_id"])
    tick_size = str(pos.get("tick_size", "0.01"))
    sell_size = int(size)

    if state["mode"] == "paper":
        print(f"  PAPER: would place sell at ${exit_price:.2f} x {sell_size}")
        pos["exit_order_id"] = "paper_exit"
        pos["exit_order_size"] = sell_size
        return True

    result = place_maker_exit(
        token_id=token_id,
        price=exit_price,
        size=sell_size,
        tick_size=tick_size,
    )
    if result.get("status") == "rejected_would_cross":
        print(f"  sell order would cross, holding to settlement.")
        return False
    if result.get("status") == "error":
        print(f"  EXIT ORDER ERROR: {pos['city']}: {result.get('error')}")
        return False

    pos["exit_order_id"] = result.get("order_id")
    pos["exit_order_size"] = sell_size
    print(
        f"  EXIT PLACED: {pos['city']} @ ${exit_price:.2f} "
        f"| order={pos['exit_order_id']}"
    )

    if notify_on_fill:
        entry_price = float(pos.get("fill_price") or pos.get("maker_entry_price"))
        send_pushover(
            f"ENTRY FILLED: {pos['city']}",
            f"{pos['bucket_label']} @ ${entry_price:.2f}, "
            f"exit order at ${exit_price:.2f}",
        )
    return True


def initialize_day_modal_maker(state: dict[str, Any]) -> dict[str, Any]:
    """Fetch modal buckets, filter, size, place entry + exit maker orders."""
    mode = state["mode"]
    now = datetime.now(CT)
    state["strategy"] = STRATEGY_NAME

    sized_trades = select_modal_trades(state["date"], float(state["bankroll"]))
    if not sized_trades:
        print("  WARNING: no eligible modal-maker trades.")
        state["phase"] = "monitoring"
        state["entry_time"] = now.strftime("%H:%M:%S")
        append_trade_log(state, {"event": "init_abort", "reason": "no_eligible_trades"})
        return state

    print(f"\n--- PHASE 4-5: place entry and exit orders ---")
    positions: list[dict[str, Any]] = []
    for trade in sized_trades:
        assert int(trade["n_contracts"]) >= 5
        assert float(trade["n_contracts"]) * float(trade["maker_entry_price"]) >= 1.0
        pos: dict[str, Any] = {
            "city": trade["city"],
            "bucket_label": trade["bucket_label"],
            "yes_token_id": trade["yes_token_id"],
            "condition_id": trade.get("condition_id"),
            "modal_bucket_probability": trade["modal_bucket_probability"],
            "n_contracts": trade["n_contracts"],
            "maker_entry_price": trade["maker_entry_price"],
            "exit_target_price": trade["exit_target_price"],
            "best_bid_at_entry": trade.get("best_bid"),
            "best_ask_at_entry": trade.get("best_ask"),
            "capital_at_risk": trade["capital_at_risk"],
            "tick_size": trade.get("tick_size", "0.01"),
            "order_id": None,
            "exit_order_id": None,
            "exit_order_size": None,
            "fill_price": None,
            "fill_time": None,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "exit_fee": 0.0,
            "pnl": None,
            "monitoring_log": [],
            "status": "pending_entry",
        }

        if mode == "paper":
            pos["order_id"] = "paper_buy"
            pos["status"] = "pending_entry"
            print(
                f"  PAPER ENTRY: {trade['city']} {trade['bucket_label']} "
                f"@ ${trade['maker_entry_price']:.2f} "
                f"| modal={trade['modal_bucket_probability']:.3f}"
            )
            place_exit_order_for_position(state, pos, trade["n_contracts"])
        else:
            order_result = place_maker_entry(
                token_id=trade["yes_token_id"],
                price=float(trade["maker_entry_price"]),
                size=int(trade["n_contracts"]),
                tick_size=str(trade.get("tick_size", "0.01")),
            )
            if order_result.get("status") == "rejected_would_cross":
                print(
                    f"  REJECTED: {trade['city']} would cross at "
                    f"${trade['maker_entry_price']:.2f}"
                )
                pos["status"] = "cancelled"
                pos["exit_reason"] = "rejected_would_cross"
            elif order_result.get("status") == "error":
                err = _order_error_text(order_result)
                print(f"  ERROR: {trade['city']} entry failed: {err}")
                pos["status"] = "cancelled"
                pos["exit_reason"] = err
            else:
                pos["status"] = "pending_entry"
                pos["order_id"] = order_result.get("order_id")
                print(
                    f"  ENTRY PLACED: {trade['city']} {trade['bucket_label']} "
                    f"@ ${trade['maker_entry_price']:.2f} | order={pos['order_id']}"
                )
                if pos.get("order_id"):
                    place_exit_order_for_position(state, pos, trade["n_contracts"])

        positions.append(pos)
        append_trade_log(
            state,
            {
                "event": "entry",
                "city": trade["city"],
                "bucket_label": trade["bucket_label"],
                "status": pos["status"],
                "maker_entry_price": pos["maker_entry_price"],
                "exit_target_price": pos["exit_target_price"],
            },
        )

    state["positions"] = positions
    state["phase"] = (
        "entries_placed"
        if any(p["status"] == "pending_entry" for p in positions)
        else "monitoring"
    )
    state["entry_time"] = now.strftime("%H:%M:%S")
    state["daily_capital_at_risk"] = round(
        sum(
            p["capital_at_risk"]
            for p in positions
            if p["status"] in ("pending_entry", "filled")
        ),
        4,
    )

    send_pushover(
        f"Modal-maker: {len(positions)} entries ({state['mode']})",
        "\n".join(
            f"{p['city']} {p['bucket_label']} @ ${p['maker_entry_price']:.2f} "
            f"[{p['status']}]"
            for p in positions
        ),
    )
    return state


def initialize_day(state: dict[str, Any]) -> dict[str, Any]:
    """Run forecast, fetch market, compute edges, place entries."""
    date_str = state["date"]
    mode = state["mode"]
    bankroll = state["bankroll"]
    now = datetime.now(CT)

    sized_trades, metadata = prepare_poly_trades(
        date_str,
        bankroll,
        raise_on_no_market=False,
    )

    if metadata.get("abort_reason") == "no_forecasts":
        print("  ABORT: 0 cities have forecast coverage.")
        state["phase"] = "monitoring"
        state["entry_time"] = now.strftime("%H:%M:%S")
        append_trade_log(state, {"event": "init_abort", "reason": "no_forecasts"})
        return state

    if metadata.get("abort_reason") == "no_markets":
        print("  WARNING: no active Polymarket markets found.")
        state["phase"] = "monitoring"
        state["entry_time"] = now.strftime("%H:%M:%S")
        append_trade_log(state, {"event": "init_abort", "reason": "no_markets"})
        return state

    # Persist forecast metadata for settlement/residual computation
    state["raw_forecasts"] = metadata.get("raw_forecasts", {})
    state["forecasts_after_bias"] = metadata.get("forecasts", {})
    state["rolling_bias_applied"] = metadata.get("rolling_bias_applied", {})
    state["wunderground_bias_applied"] = metadata.get("bias_applied", {})
    if metadata.get("ngboost_sigmas"):
        state["ngboost_sigmas"] = metadata["ngboost_sigmas"]

    no_signal = metadata.get("no_signal_cities", {})
    for city, reason in sorted(no_signal.items()):
        print(f"  NO SIGNAL: {city} (reason: {reason})")

    signal = metadata.get("poly_config", {}).get("signal", "track_b_flat")
    state["signal"] = signal
    if signal == "ngboost":
        state["wu_adjusted_forecasts"] = metadata.get("raw_forecasts", {})
    else:
        wu_adjusted_forecasts = {}
        for city, forecast_val in metadata.get("forecasts", {}).items():
            rolling = metadata.get("rolling_bias_applied", {}).get(city, 0.0)
            wu_adjusted_forecasts[city] = int(round(float(forecast_val) + float(rolling)))
        state["wu_adjusted_forecasts"] = wu_adjusted_forecasts

    print(f"  Selected: {len(sized_trades)} trades")

    poly_config = metadata.get("poly_config", build_poly_config(DEFAULT_CONFIG_PATH)[0])
    state["daily_budget_cap"] = daily_cap_from_bankroll(bankroll, poly_config)

    positions: list[dict[str, Any]] = []
    for trade in sized_trades:
        assert int(trade["n_contracts"]) >= 5
        assert float(trade["n_contracts"]) * float(trade["maker_entry_price"]) >= 1.0
        pos = {
            "city": trade["city"],
            "bucket_label": trade["bucket_label"],
            "yes_token_id": trade["yes_token_id"],
            "condition_id": trade.get("condition_id"),
            "model_prob": trade["model_prob"],
            "edge": trade["edge"],
            "entry_mode": trade.get("entry_mode", "primary"),
            "n_contracts": trade["n_contracts"],
            "maker_entry_price": trade["maker_entry_price"],
            "best_bid_at_entry": trade.get("best_bid"),
            "best_ask_at_entry": trade.get("best_ask"),
            "capital_at_risk": trade["capital_at_risk"],
            "tick_size": trade.get("tick_size", "0.01"),
            "order_id": None,
            "fill_price": None,
            "fill_time": None,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "exit_fee": None,
            "pnl": None,
            "monitoring_log": [],
            "status": "pending_entry",
        }

        if mode == "paper":
            pos["status"] = "filled"
            pos["fill_price"] = trade["maker_entry_price"]
            pos["fill_time"] = now.strftime("%H:%M:%S")
            print(
                f"  PAPER ENTRY: {trade['city']} {trade['bucket_label']} "
                f"@ ${trade['maker_entry_price']:.2f} | edge={trade['edge']:+.3f}"
            )
        else:
            order_result, fresh_price, n_contracts, best_bid, best_ask = (
                place_live_entry_with_fresh_book(trade, poly_config=poly_config)
            )
            pos["maker_entry_price"] = fresh_price
            pos["n_contracts"] = n_contracts
            pos["capital_at_risk"] = round(n_contracts * fresh_price, 4)
            pos["best_bid_at_entry"] = best_bid
            pos["best_ask_at_entry"] = best_ask
            status = order_result.get("status", "")
            if status in ("posted", "signed"):
                pos["status"] = "pending_entry"
                pos["order_id"] = order_result.get("order_id")
                print(
                    f"  ENTRY PLACED: {trade['city']} {trade['bucket_label']} "
                    f"@ ${fresh_price:.2f} x{n_contracts} | order={pos['order_id']}"
                )
            else:
                err = _order_error_text(order_result)
                print(f"  ERROR: {trade['city']} entry failed: {err}")
                pos["status"] = "cancelled"
                pos["exit_reason"] = err

        positions.append(pos)
        append_trade_log(
            state,
            {
                "event": "entry",
                "city": trade["city"],
                "bucket_label": trade["bucket_label"],
                "status": pos["status"],
                "maker_entry_price": pos["maker_entry_price"],
                "entry_mode": pos.get("entry_mode", "primary"),
            },
        )

    state["positions"] = positions
    state["phase"] = (
        "entries_placed"
        if any(p["status"] == "pending_entry" for p in positions)
        else "monitoring"
    )
    state["entry_time"] = now.strftime("%H:%M:%S")
    state["daily_capital_at_risk"] = round(
        sum(
            p["capital_at_risk"]
            for p in positions
            if p["status"] in ("pending_entry", "filled")
        ),
        4,
    )

    push_lines = []
    for city, reason in sorted(no_signal.items()):
        push_lines.append(f"NO SIGNAL: {city} (reason: {reason})")
    for p in positions:
        push_lines.append(
            f"{p['city']} {p['bucket_label']} @ ${p['maker_entry_price']:.2f} "
            f"edge={p['edge']:+.3f} mode={p.get('entry_mode', 'primary')} [{p['status']}]"
        )
    send_pushover(
        f"Auto-trader: {len(positions)} entries ({state['mode']})",
        "\n".join(push_lines),
    )
    return state


def _apply_entry_fill(
    state: dict[str, Any],
    pos: dict[str, Any],
    *,
    fill_price: float | None,
    size_matched: float | None,
    now: datetime,
    event: str = "entry_filled",
    skip_pushover: bool = False,
) -> None:
    """Transition a pending entry to filled, including partial fills."""
    matched = size_matched or pos.get("n_contracts")
    if matched and matched > 0:
        pos["n_contracts"] = int(matched) if matched == int(matched) else matched
    entry_price = float(fill_price or pos.get("maker_entry_price") or 0.0)
    pos["status"] = "filled"
    pos["fill_price"] = entry_price
    pos["fill_time"] = now.strftime("%H:%M:%S")
    pos["exit_reason"] = None
    pos["capital_at_risk"] = round(pos["n_contracts"] * entry_price, 4)
    print(
        f"  FILLED: {pos['city']} {pos['bucket_label']} "
        f"@ ${entry_price:.3f} x {pos['n_contracts']}"
    )
    append_trade_log(
        state,
        {
            "event": event,
            "city": pos["city"],
            "fill_price": pos["fill_price"],
            "n_contracts": pos["n_contracts"],
            "order_id": pos.get("order_id"),
        },
    )
    if not skip_pushover:
        send_pushover(
            f"Entry filled: {pos['city']}",
            f"{pos['bucket_label']} @ ${entry_price:.3f} x {pos['n_contracts']}",
        )


def _sync_modal_exit_after_fill(state: dict[str, Any], pos: dict[str, Any]) -> None:
    """Ensure exit sell matches filled size; defer if not yet placed."""
    filled_size = pos["n_contracts"]
    placed_size = pos.get("exit_order_size")
    if not pos.get("exit_order_id"):
        place_exit_order_for_position(
            state,
            pos,
            filled_size,
            notify_on_fill=True,
        )
    elif placed_size is not None and placed_size != filled_size:
        place_exit_order_for_position(
            state,
            pos,
            filled_size,
            replace=True,
            notify_on_fill=True,
        )
    elif pos.get("exit_order_id"):
        entry_price = float(pos.get("fill_price") or pos.get("maker_entry_price"))
        exit_price = float(pos["exit_target_price"])
        send_pushover(
            f"ENTRY FILLED: {pos['city']}",
            f"{pos['bucket_label']} @ ${entry_price:.2f}, "
            f"exit order at ${exit_price:.2f}",
        )


def _apply_modal_entry_fill(
    state: dict[str, Any],
    pos: dict[str, Any],
    *,
    fill_price: float | None,
    size_matched: float | None,
    now: datetime,
    event: str = "entry_filled",
) -> None:
    _apply_entry_fill(
        state,
        pos,
        fill_price=fill_price,
        size_matched=size_matched,
        now=now,
        event=event,
        skip_pushover=True,
    )
    _sync_modal_exit_after_fill(state, pos)


def _entry_has_fill(status: dict[str, Any]) -> bool:
    matched = status.get("size_matched") or 0.0
    return status.get("status") in ("filled", "partial") or matched > 0


def _exit_has_fill(status: dict[str, Any]) -> bool:
    matched = status.get("size_matched") or 0.0
    return status.get("status") in ("filled", "partial") or matched > 0


def _cancel_order_safe(order_id: str | None) -> None:
    if not order_id or str(order_id).startswith("paper"):
        return
    try:
        _get_poly_client().cancel_order(str(order_id))
    except Exception as exc:
        print(f"  Cancel failed: {exc}")


def check_pending_entries(state: dict[str, Any]) -> dict[str, Any]:
    """Check fill status of pending entry orders. Live mode only."""
    if state["mode"] == "paper":
        return state

    client = _get_poly_client()
    now = datetime.now(CT)

    for pos in state["positions"]:
        if pos["status"] != "pending_entry" or not pos.get("order_id"):
            continue

        status = client.get_order_status(
            str(pos["order_id"]),
            token_id=str(pos["yes_token_id"]),
        )
        if _entry_has_fill(status):
            should_cancel_remainder = status.get("status") in ("open", "partial")
            _apply_entry_fill(
                state,
                pos,
                fill_price=status.get("fill_price"),
                size_matched=status.get("size_matched"),
                now=now,
            )
            if should_cancel_remainder and pos.get("order_id"):
                try:
                    client.cancel_order(str(pos["order_id"]))
                except Exception as exc:
                    print(f"  Cancel remainder failed: {exc}")

    entry_time_str = state.get("entry_time") or "10:05:00"
    entry_dt = datetime.strptime(
        f"{state['date']} {entry_time_str}",
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=CT)
    elapsed_min = (now - entry_dt).total_seconds() / 60

    if elapsed_min > ENTRY_TIMEOUT_MIN:
        for pos in state["positions"]:
            if pos["status"] != "pending_entry":
                continue

            status = client.get_order_status(
                str(pos["order_id"]),
                token_id=str(pos["yes_token_id"]),
            )
            if _entry_has_fill(status):
                print(
                    f"  TIMEOUT skipped: {pos['city']} already filled "
                    f"({status.get('size_matched')} contracts)"
                )
                _apply_entry_fill(
                    state,
                    pos,
                    fill_price=status.get("fill_price"),
                    size_matched=status.get("size_matched"),
                    now=now,
                    event="entry_filled_on_timeout",
                )
                continue

            print(f"  TIMEOUT: cancelling {pos['city']} {pos['bucket_label']}")
            _cancel_order_safe(pos.get("order_id"))
            pos["status"] = "cancelled"
            pos["exit_reason"] = "entry_timeout"
            append_trade_log(
                state,
                {
                    "event": "entry_timeout",
                    "city": pos["city"],
                    "order_id": pos.get("order_id"),
                },
            )
            send_pushover(
                f"Entry timeout: {pos['city']}",
                f"Cancelled {pos['bucket_label']} after {ENTRY_TIMEOUT_MIN}min",
            )

    if not any(p["status"] == "pending_entry" for p in state["positions"]):
        state["phase"] = "monitoring"

    state["daily_capital_at_risk"] = round(
        sum(
            p["capital_at_risk"]
            for p in state["positions"]
            if p["status"] in ("pending_entry", "filled")
        ),
        4,
    )
    return state


def check_pending_entries_modal(state: dict[str, Any]) -> dict[str, Any]:
    """Check fill status and 60-min timeout for modal maker entries."""
    if state["mode"] == "paper":
        return state

    client = _get_poly_client()
    now = datetime.now(CT)

    for pos in state["positions"]:
        if pos["status"] != "pending_entry" or not pos.get("order_id"):
            continue

        status = client.get_order_status(
            str(pos["order_id"]),
            token_id=str(pos["yes_token_id"]),
        )
        if _entry_has_fill(status):
            should_cancel_remainder = status.get("status") in ("open", "partial")
            _apply_modal_entry_fill(
                state,
                pos,
                fill_price=status.get("fill_price"),
                size_matched=status.get("size_matched"),
                now=now,
            )
            if should_cancel_remainder and pos.get("order_id"):
                _cancel_order_safe(pos.get("order_id"))

    entry_time_str = state.get("entry_time") or "10:05:00"
    entry_dt = datetime.strptime(
        f"{state['date']} {entry_time_str}",
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=CT)
    elapsed_min = (now - entry_dt).total_seconds() / 60

    if elapsed_min > ENTRY_TIMEOUT_MIN:
        for pos in state["positions"]:
            if pos["status"] != "pending_entry":
                continue

            status = client.get_order_status(
                str(pos["order_id"]),
                token_id=str(pos["yes_token_id"]),
            )
            if _entry_has_fill(status):
                print(
                    f"  TIMEOUT skipped: {pos['city']} already filled "
                    f"({status.get('size_matched')} contracts)"
                )
                _apply_modal_entry_fill(
                    state,
                    pos,
                    fill_price=status.get("fill_price"),
                    size_matched=status.get("size_matched"),
                    now=now,
                    event="entry_filled_on_timeout",
                )
                continue

            print(f"  TIMEOUT: cancelling {pos['city']} {pos['bucket_label']}")
            _cancel_order_safe(pos.get("order_id"))
            _cancel_order_safe(pos.get("exit_order_id"))
            pos["exit_order_id"] = None
            pos["exit_order_size"] = None
            pos["status"] = "cancelled"
            pos["exit_reason"] = "entry_timeout"
            append_trade_log(
                state,
                {
                    "event": "entry_timeout",
                    "city": pos["city"],
                    "order_id": pos.get("order_id"),
                    "exit_order_id": pos.get("exit_order_id"),
                },
            )
            send_pushover(
                f"Entry timeout: {pos['city']}",
                f"{pos['bucket_label']} (60min, both orders cancelled)",
            )

    if not any(p["status"] == "pending_entry" for p in state["positions"]):
        state["phase"] = "monitoring"

    state["daily_capital_at_risk"] = round(
        sum(
            p["capital_at_risk"]
            for p in state["positions"]
            if p["status"] in ("pending_entry", "filled")
        ),
        4,
    )
    return state


def check_exit_conditions(state: dict[str, Any]) -> dict[str, Any]:
    """Check profit target for all filled positions."""
    now = datetime.now(CT)
    filled_positions = [p for p in state["positions"] if p["status"] == "filled"]
    for idx, pos in enumerate(filled_positions):
        if idx > 0:
            time.sleep(BOOK_FETCH_DELAY_SEC)

        best_bid, best_ask = get_best_bid_ask(str(pos["yes_token_id"]))
        log_entry = {
            "time": now.strftime("%H:%M"),
            "best_bid": best_bid,
            "best_ask": best_ask,
        }
        pos.setdefault("monitoring_log", []).append(log_entry)
        if len(pos["monitoring_log"]) > 10:
            pos["monitoring_log"] = pos["monitoring_log"][-10:]

        if best_bid is None:
            print(f"  {pos['city']}: no bid available, skipping")
            continue

        entry = float(pos["fill_price"] or pos["maker_entry_price"])
        gain = best_bid - entry
        print(
            f"  {pos['city']} {pos['bucket_label']}: "
            f"entry=${entry:.2f} bid=${best_bid:.2f} "
            f"gain={gain:+.3f} target={PROFIT_TARGET:.2f}"
        )

        if gain < PROFIT_TARGET:
            continue

        print(
            f"  *** EXIT SIGNAL: {pos['city']} {pos['bucket_label']} "
            f"gain={gain:+.3f} >= {PROFIT_TARGET}"
        )

        if state["mode"] == "paper":
            pos["status"] = "exited"
            pos["exit_price"] = best_bid
            pos["exit_time"] = now.strftime("%H:%M:%S")
            pos["exit_reason"] = "profit_target_15c"
            pos["exit_fee"] = 0.0
            pos["pnl"] = round(pos["n_contracts"] * (best_bid - entry), 4)
            print(
                f"  PAPER EXIT: {pos['city']} @ ${best_bid:.2f} "
                f"PnL=${pos['pnl']:.4f}"
            )
        else:
            pos["status"] = "exit_triggered"
            exit_result = place_taker_exit(
                token_id=str(pos["yes_token_id"]),
                price=best_bid,
                size=int(pos["n_contracts"]),
                tick_size=str(pos.get("tick_size", "0.01")),
            )
            if exit_result.get("status") == "error":
                print(f"  EXIT FAILED: {exit_result}")
                pos["status"] = "filled"
                continue

            pos["status"] = "exited"
            pos["exit_price"] = best_bid
            pos["exit_time"] = now.strftime("%H:%M:%S")
            pos["exit_reason"] = "profit_target_15c"
            taker_fee = poly_taker_fee(int(pos["n_contracts"]), best_bid)
            pos["exit_fee"] = taker_fee
            pos["pnl"] = round(
                pos["n_contracts"] * (best_bid - entry) - taker_fee,
                4,
            )
            print(
                f"  LIVE EXIT: {pos['city']} @ ${best_bid:.2f} "
                f"fee=${taker_fee:.4f} PnL=${pos['pnl']:.4f}"
            )

        append_trade_log(
            state,
            {
                "event": "exit",
                "city": pos["city"],
                "exit_price": pos["exit_price"],
                "pnl": pos["pnl"],
                "reason": pos["exit_reason"],
            },
        )
        send_pushover(
            f"EXIT: {pos['city']} {pos['bucket_label']}",
            f"Entry ${entry:.2f} -> Exit ${best_bid:.2f}\n"
            f"Gain: {gain:+.3f} | PnL: ${pos.get('pnl', 0):.4f}\n"
            f"Reason: profit_target_15c",
        )

    return state


def check_modal_exit_orders(state: dict[str, Any]) -> dict[str, Any]:
    """Monitor maker sell orders for modal maker filled positions."""
    now = datetime.now(CT)
    client = _get_poly_client() if state["mode"] == "live" else None
    filled_positions = [p for p in state["positions"] if p["status"] == "filled"]

    for idx, pos in enumerate(filled_positions):
        if idx > 0:
            time.sleep(BOOK_FETCH_DELAY_SEC)

        if not pos.get("exit_order_id"):
            place_exit_order_for_position(state, pos, pos["n_contracts"])
            continue

        if state["mode"] == "live" and client is not None:
            exit_status = client.get_order_status(str(pos["exit_order_id"]))
            if _exit_has_fill(exit_status):
                entry = float(pos.get("fill_price") or pos.get("maker_entry_price"))
                exit_price = float(pos["exit_target_price"])
                pos["status"] = "exited"
                pos["exit_price"] = exit_price
                pos["exit_time"] = now.strftime("%H:%M:%S")
                pos["exit_reason"] = "maker_exit_18c"
                pos["exit_fee"] = 0.0
                pos["pnl"] = round(EXIT_THRESHOLD * pos["n_contracts"], 4)
                print(
                    f"  MAKER EXIT: {pos['city']} @ ${entry:.2f} -> "
                    f"${exit_price:.2f} PnL=${pos['pnl']:.4f}"
                )
                append_trade_log(
                    state,
                    {
                        "event": "exit",
                        "city": pos["city"],
                        "exit_price": pos["exit_price"],
                        "pnl": pos["pnl"],
                        "reason": pos["exit_reason"],
                    },
                )
                send_pushover(
                    f"MAKER EXIT: {pos['city']} {pos['bucket_label']}",
                    f"@ ${entry:.2f} -> ${exit_price:.2f}, PnL: ${pos['pnl']:.4f}",
                )
                continue

        best_bid, best_ask = get_best_bid_ask(str(pos["yes_token_id"]))
        log_entry = {
            "time": now.strftime("%H:%M"),
            "best_bid": best_bid,
            "best_ask": best_ask,
        }
        pos.setdefault("monitoring_log", []).append(log_entry)
        if len(pos["monitoring_log"]) > 10:
            pos["monitoring_log"] = pos["monitoring_log"][-10:]

        entry = float(pos.get("fill_price") or pos.get("maker_entry_price"))
        print(
            f"  {pos['city']} {pos['bucket_label']}: "
            f"entry=${entry:.2f} bid={best_bid} ask={best_ask} "
            f"exit_target=${pos['exit_target_price']:.2f}"
        )

    return state


def end_of_day(state: dict[str, Any]) -> dict[str, Any]:
    """Summarize and close out the day."""
    if state["phase"] == "done":
        return state

    n_filled = sum(1 for p in state["positions"] if p["status"] == "filled")
    n_exited = sum(1 for p in state["positions"] if p["status"] == "exited")
    n_cancelled = sum(1 for p in state["positions"] if p["status"] == "cancelled")
    total_pnl = sum(
        (p.get("pnl") or 0) for p in state["positions"] if p["status"] == "exited"
    )

    for pos in state["positions"]:
        if pos["status"] == "filled":
            pos["exit_reason"] = "settlement_pending"

    summary = (
        f"Day summary ({state['mode']}):\n"
        f"  Exited: {n_exited} (PnL: ${total_pnl:.4f})\n"
        f"  Settling: {n_filled}\n"
        f"  Cancelled: {n_cancelled}"
    )
    print(summary)
    append_trade_log(state, {"event": "end_of_day", "summary": summary})
    send_pushover(f"Day complete: {state['date']}", summary)

    state["phase"] = "done"
    return state


def end_of_day_modal(state: dict[str, Any]) -> dict[str, Any]:
    """Summarize modal maker day; hold unsettled positions to settlement."""
    if state["phase"] == "done":
        return state

    n_settling = sum(1 for p in state["positions"] if p["status"] == "filled")
    n_exited = sum(1 for p in state["positions"] if p["status"] == "exited")
    n_cancelled = sum(1 for p in state["positions"] if p["status"] == "cancelled")
    total_pnl = sum(
        (p.get("pnl") or 0) for p in state["positions"] if p["status"] == "exited"
    )

    for pos in state["positions"]:
        if pos["status"] == "filled":
            pos["status"] = "settlement_pending"
            pos["exit_reason"] = "settlement_pending"

    summary = (
        f"Exited: {n_exited} (PnL: ${total_pnl:.4f}), "
        f"Settling: {n_settling}, Cancelled: {n_cancelled}"
    )
    print(summary)
    append_trade_log(state, {"event": "end_of_day", "summary": summary})
    send_pushover(f"Day complete: {state['date']}", summary)

    state["phase"] = "done"
    return state


def _canonical_strategy(cli_strategy: str) -> str:
    if cli_strategy == "trackb":
        print("  strategy alias 'trackb' -> 'ngboost' (naming legacy)")
        return "ngboost"
    return cli_strategy


def _resolve_strategy(args: argparse.Namespace, state: dict[str, Any]) -> str:
    cli_strategy = _canonical_strategy(args.strategy)
    persisted = normalize_strategy_name(state.get("strategy"))
    if persisted == STRATEGY_NAME:
        return "modal_maker"
    if persisted and persisted not in (STRATEGY_NAME, "ngboost"):
        return cli_strategy
    if state.get("phase") != "uninitialized" and persisted is None:
        return "ngboost"
    if (
        state.get("phase") != "uninitialized"
        and cli_strategy == "modal_maker"
        and persisted != STRATEGY_NAME
    ):
        print(
            f"  WARNING: persisted strategy={persisted!r} differs from "
            f"CLI --strategy={cli_strategy!r}; using persisted state."
        )
        return "ngboost"
    return cli_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Polymarket Tmax trader")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument(
        "--strategy",
        default="ngboost",
        choices=["ngboost", "trackb", "modal_maker"],
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip entry-time wait (for dry-run testing)",
    )
    args = parser.parse_args()

    now = datetime.now(CT)
    date_str = args.date or now.strftime("%Y-%m-%d")
    bankroll = args.bankroll
    if bankroll is None:
        bankroll = read_bankroll_file()
    if bankroll is None:
        bankroll = 100.0
        print(f"  WARNING: no bankroll file; defaulting to ${bankroll:.2f}")
    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S')} CT] "
        f"auto_trader_poly ({args.mode}, {args.strategy}) bankroll=${bankroll:.2f}"
    )

    state = load_state(date_str)
    if state is None:
        cities = (
            list(MODAL_MAKER_CITIES)
            if args.strategy == "modal_maker"
            else load_deploy_cities()
        )
        state = {
            "date": date_str,
            "mode": args.mode,
            "bankroll": bankroll,
            "phase": "uninitialized",
            "entry_time": None,
            "positions": [],
            "daily_capital_at_risk": 0.0,
            "daily_budget_cap": operative_daily_cap(bankroll),
            "trades_log": [],
            "last_tick": now.isoformat(),
            "cities": cities,
            "strategy": STRATEGY_NAME if args.strategy == "modal_maker" else "ngboost",
        }

    strategy = _resolve_strategy(args, state)
    if strategy == "ngboost" and state.get("strategy") != STRATEGY_NAME:
        state["strategy"] = "ngboost"
    state["last_tick"] = now.isoformat()

    if state["phase"] == "done":
        print("  Day complete, nothing to do.")
        save_state(state)
        return

    current_hour = now.hour
    current_minute = now.minute

    if state["phase"] == "uninitialized":
        if not args.force and (
            current_hour < ENTRY_HOUR
            or (current_hour == ENTRY_HOUR and current_minute < ENTRY_MINUTE)
        ):
            print(
                f"  [{now.strftime('%H:%M')}] Waiting for "
                f"{ENTRY_HOUR}:{ENTRY_MINUTE:02d} CT"
            )
            save_state(state)
            return

        config_message = emit_config_echo(
            state["mode"], float(state["bankroll"]), strategy=strategy
        )
        send_pushover(f"Autotrader config ({date_str})", config_message)
        if not run_preflight(state["mode"]):
            state["phase"] = "monitoring"
            state["entry_time"] = now.strftime("%H:%M:%S")
            append_trade_log(state, {"event": "init_abort", "reason": "preflight_failed"})
            save_state(state)
            return

        print(f"\n=== AUTO-TRADER INITIALIZE: {date_str} ({state['mode']}) ===")
        if strategy == "modal_maker":
            state = initialize_day_modal_maker(state)
        else:
            state = initialize_day(state)
        save_state(state)

        log_entry = {
            "date": date_str,
            "mode": state["mode"],
            "exchange": "polymarket",
            "source": "auto_trader",
            "strategy": state.get("strategy", strategy),
            "bankroll": state["bankroll"],
            "n_trades": len(
                [p for p in state["positions"] if p["status"] != "cancelled"]
            ),
            "positions": state["positions"],
        }
        POLY_PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(POLY_PAPER_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry, default=str) + "\n")
        return

    if current_hour >= MONITOR_END_HOUR:
        print(f"\n=== AUTO-TRADER END OF DAY: {date_str} ===")
        if _is_modal_strategy(state):
            state = end_of_day_modal(state)
        else:
            state = end_of_day(state)
        save_state(state)
        return

    print(f"  [{now.strftime('%H:%M')}] Monitoring tick ({state['mode']})")

    if any(p["status"] == "pending_entry" for p in state["positions"]):
        if _is_modal_strategy(state):
            state = check_pending_entries_modal(state)
        else:
            state = check_pending_entries(state)

    if _is_modal_strategy(state):
        state = check_modal_exit_orders(state)
    else:
        state = check_exit_conditions(state)

    active = [
        p
        for p in state["positions"]
        if p["status"]
        in ("pending_entry", "filled", "exit_triggered", "settlement_pending")
    ]
    if not active and state["phase"] != "uninitialized":
        print("  All positions resolved. Marking done.")
        if _is_modal_strategy(state):
            state = end_of_day_modal(state)
        else:
            state = end_of_day(state)

    save_state(state)


if __name__ == "__main__":
    main()
