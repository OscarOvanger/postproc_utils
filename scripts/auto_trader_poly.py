"""Automated Polymarket Tmax trading loop with entry, monitoring, and exit."""

from __future__ import annotations

import argparse
import json
import os
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
    POLYMARKET_CITIES,
    poly_taker_fee,
    prepare_poly_trades,
)
from src.polymarket_api import fetch_order_book_http  # noqa: E402

CT = ZoneInfo("America/Chicago")
STATE_DIR = PROJECT_ROOT / "logs"
POLY_PAPER_LOG = STATE_DIR / "poly_paper_trades.jsonl"
PROFIT_TARGET = 0.15
ENTRY_TIMEOUT_MIN = 60
MONITOR_END_HOUR = 22
ENTRY_HOUR = 10
ENTRY_MINUTE = 5
DAILY_LOSS_CAP = 6.0
BOOK_FETCH_DELAY_SEC = 0.2


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
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"  Pushover failed: {exc}")


def state_path(date_str: str) -> Path:
    return STATE_DIR / f"auto_trader_state_{date_str}.json"


def load_state(date_str: str) -> dict[str, Any] | None:
    path = state_path(date_str)
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
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

    print(f"  Selected: {len(sized_trades)} trades")

    positions: list[dict[str, Any]] = []
    for trade in sized_trades:
        pos = {
            "city": trade["city"],
            "bucket_label": trade["bucket_label"],
            "yes_token_id": trade["yes_token_id"],
            "condition_id": trade.get("condition_id"),
            "model_prob": trade["model_prob"],
            "edge": trade["edge"],
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
            order_result = place_maker_entry(
                token_id=trade["yes_token_id"],
                price=float(trade["maker_entry_price"] or trade["market_price"]),
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
                print(f"  ERROR: {trade['city']} entry failed: {order_result.get('error')}")
                pos["status"] = "cancelled"
                pos["exit_reason"] = "entry_error"
            else:
                pos["status"] = "pending_entry"
                pos["order_id"] = order_result.get("order_id")
                print(
                    f"  ENTRY PLACED: {trade['city']} {trade['bucket_label']} "
                    f"@ ${trade['maker_entry_price']:.2f} | order={pos['order_id']}"
                )

        positions.append(pos)
        append_trade_log(
            state,
            {
                "event": "entry",
                "city": trade["city"],
                "bucket_label": trade["bucket_label"],
                "status": pos["status"],
                "maker_entry_price": pos["maker_entry_price"],
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
        f"Auto-trader: {len(positions)} entries ({state['mode']})",
        "\n".join(
            f"{p['city']} {p['bucket_label']} @ ${p['maker_entry_price']:.2f} "
            f"edge={p['edge']:+.3f} [{p['status']}]"
            for p in positions
        ),
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
    send_pushover(
        f"Entry filled: {pos['city']}",
        f"{pos['bucket_label']} @ ${entry_price:.3f} x {pos['n_contracts']}",
    )


def _entry_has_fill(status: dict[str, Any]) -> bool:
    matched = status.get("size_matched") or 0.0
    return status.get("status") in ("filled", "partial") or matched > 0


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
            if pos.get("order_id"):
                try:
                    client.cancel_order(str(pos["order_id"]))
                except Exception as exc:
                    print(f"  Cancel failed: {exc}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Polymarket Tmax trader")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--date", default=None)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()

    now = datetime.now(CT)
    date_str = args.date or now.strftime("%Y-%m-%d")
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')} CT] auto_trader_poly ({args.mode})")

    state = load_state(date_str)
    if state is None:
        state = {
            "date": date_str,
            "mode": args.mode,
            "bankroll": args.bankroll,
            "phase": "uninitialized",
            "entry_time": None,
            "positions": [],
            "daily_capital_at_risk": 0.0,
            "daily_loss_cap": DAILY_LOSS_CAP,
            "trades_log": [],
            "last_tick": now.isoformat(),
            "cities": list(POLYMARKET_CITIES),
        }

    state["last_tick"] = now.isoformat()

    if state["phase"] == "done":
        print("  Day complete, nothing to do.")
        save_state(state)
        return

    current_hour = now.hour
    current_minute = now.minute

    if state["phase"] == "uninitialized":
        if current_hour < ENTRY_HOUR or (
            current_hour == ENTRY_HOUR and current_minute < ENTRY_MINUTE
        ):
            print(
                f"  [{now.strftime('%H:%M')}] Waiting for "
                f"{ENTRY_HOUR}:{ENTRY_MINUTE:02d} CT"
            )
            save_state(state)
            return

        print(f"\n=== AUTO-TRADER INITIALIZE: {date_str} ({state['mode']}) ===")
        state = initialize_day(state)
        save_state(state)

        log_entry = {
            "date": date_str,
            "mode": state["mode"],
            "exchange": "polymarket",
            "source": "auto_trader",
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
        state = end_of_day(state)
        save_state(state)
        return

    print(f"  [{now.strftime('%H:%M')}] Monitoring tick ({state['mode']})")

    if any(p["status"] == "pending_entry" for p in state["positions"]):
        state = check_pending_entries(state)

    state = check_exit_conditions(state)

    active = [
        p
        for p in state["positions"]
        if p["status"] in ("pending_entry", "filled", "exit_triggered")
    ]
    if not active and state["phase"] != "uninitialized":
        print("  All positions resolved. Marking done.")
        state = end_of_day(state)

    save_state(state)


if __name__ == "__main__":
    main()
