"""Risk monitoring for the MCP 60-day challenge."""

from __future__ import annotations

import json
from pathlib import Path

ACCOUNT_STATE_PATH = Path("data/live/account_state.json")
INITIAL_BANKROLL = 10_000
ELIMINATION_THRESHOLD = 7_000
DAILY_LOSS_CAP = 600
MIN_TRADES_60D = 80
CHALLENGE_DAYS = 60


def init_account_state() -> dict:
    """Initialise a fresh account state."""
    state = {
        "bankroll_cents": INITIAL_BANKROLL,
        "trades_placed": 0,
        "days_elapsed": 0,
        "start_date": None,
        "cumulative_pnl_cents": 0,
        "peak_bankroll_cents": INITIAL_BANKROLL,
        "max_drawdown_cents": 0,
        "daily_history": [],
    }
    save_account_state(state)
    return state


def load_account_state() -> dict:
    """Load account state from disk."""
    if not ACCOUNT_STATE_PATH.exists():
        return init_account_state()
    with open(ACCOUNT_STATE_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def save_account_state(state: dict) -> None:
    """Save account state to disk."""
    ACCOUNT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCOUNT_STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, default=str)


def update_after_day(state: dict, daily_pnl_cents: int, n_trades: int, event_date: str) -> dict:
    """Update account state after a trading day."""
    state["bankroll_cents"] += daily_pnl_cents
    state["cumulative_pnl_cents"] += daily_pnl_cents
    state["trades_placed"] += n_trades
    state["days_elapsed"] += 1

    if state["start_date"] is None:
        state["start_date"] = event_date

    if state["bankroll_cents"] > state["peak_bankroll_cents"]:
        state["peak_bankroll_cents"] = state["bankroll_cents"]
    dd = state["bankroll_cents"] - state["peak_bankroll_cents"]
    if dd < state["max_drawdown_cents"]:
        state["max_drawdown_cents"] = dd

    state["daily_history"].append(
        {
            "date": event_date,
            "pnl_cents": daily_pnl_cents,
            "n_trades": n_trades,
            "bankroll_cents": state["bankroll_cents"],
        }
    )

    save_account_state(state)
    return state


def check_pace(trades_placed: int, days_elapsed: int) -> str:
    """Check if on pace for 80 trades in 60 days."""
    if days_elapsed == 0:
        return "ON_PACE"
    remaining_days = CHALLENGE_DAYS - days_elapsed
    remaining_trades = MIN_TRADES_60D - trades_placed

    if remaining_days <= 0:
        return "COMPLETE"
    required_remaining_rate = remaining_trades / remaining_days

    if required_remaining_rate <= 0:
        return "ON_PACE"
    if required_remaining_rate <= 2.0:
        return "ON_PACE"
    if required_remaining_rate <= 3.0:
        return "BEHIND_PACE"
    return "CRITICAL"


def check_drawdown(bankroll_cents: int) -> str:
    """Check drawdown status."""
    if bankroll_cents > 9000:
        return "SAFE"
    if bankroll_cents > 8000:
        return "CAUTION"
    if bankroll_cents > 7000:
        return "DANGER"
    return "ELIMINATED"


def recommended_contracts(bankroll_cents: int, base: int = 5) -> int:
    """Adjust contract count based on drawdown status."""
    status = check_drawdown(bankroll_cents)
    if status == "SAFE":
        return base
    if status == "CAUTION":
        return max(base - 2, 1)
    if status == "DANGER":
        return 1
    return 0


def daily_risk_report(state: dict) -> str:
    """Generate human-readable daily risk report."""
    pace = check_pace(state["trades_placed"], state["days_elapsed"])
    dd_status = check_drawdown(state["bankroll_cents"])
    rec_contracts = recommended_contracts(state["bankroll_cents"])

    lines = [
        "=" * 50,
        "DAILY RISK REPORT",
        "=" * 50,
        f"Day: {state['days_elapsed']} / {CHALLENGE_DAYS}",
        f"Bankroll: ${state['bankroll_cents'] / 100:.2f}",
        f"Cumulative PnL: ${state['cumulative_pnl_cents'] / 100:.2f}",
        f"Trades placed: {state['trades_placed']} / {MIN_TRADES_60D} min",
        f"Max drawdown: ${state['max_drawdown_cents'] / 100:.2f}",
        f"Pace status: {pace}",
        f"Drawdown status: {dd_status}",
        f"Recommended contracts: {rec_contracts}",
    ]

    if pace == "BEHIND_PACE":
        lines.append("ACTION: Lower edge threshold by 50% tomorrow")
    if pace == "CRITICAL":
        lines.append("ACTION: Trade ALL eligible cities tomorrow")
    if dd_status == "DANGER":
        lines.append("ACTION: Reduce to 1 contract per trade")
    if dd_status == "ELIMINATED":
        lines.append("ACTION: STOP TRADING — account eliminated")

    lines.append("=" * 50)
    report = "\n".join(lines)
    print(report)
    return report
