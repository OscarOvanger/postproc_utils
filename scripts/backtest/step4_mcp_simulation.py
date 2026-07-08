#!/usr/bin/env python3
"""Step 4: MCP constraint equity simulation for all backtest variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest.common as bc  # noqa: E402

VARIANTS = [
    "modal_maker_hold_to_settlement",
    "modal_maker_profit_target_15c",
    "ngboost_kelly_hold_to_settlement",
    "ngboost_kelly_profit_target_15c",
]
FLAT_VARIANT = "ngboost_flat_hold_to_settlement"


def trade_log_path(variant: str) -> Path:
    if variant.startswith("modal_maker_"):
        return bc.TRADES_DIR / f"modal_maker_{variant.removeprefix('modal_maker_')}.jsonl"
    if variant.startswith("ngboost_flat_"):
        return bc.TRADES_DIR / f"ngboost_flat_{variant.removeprefix('ngboost_flat_')}.jsonl"
    return bc.TRADES_DIR / f"ngboost_kelly_{variant.removeprefix('ngboost_kelly_')}.jsonl"


def fit_contracts(
    desired: int,
    entry_price: float,
    bankroll_usd: float,
    daily_spent_usd: float,
    daily_cap_usd: float,
) -> int:
    if desired <= 0 or entry_price <= 0:
        return 0
    max_pos_usd = bankroll_usd * bc.MAX_POSITION_PCT
    max_by_position = int(max_pos_usd / entry_price) if entry_price > 0 else 0
    remaining_budget = daily_cap_usd - daily_spent_usd
    max_by_budget = int(remaining_budget / entry_price) if entry_price > 0 else 0
    return max(0, min(desired, max_by_position, max_by_budget))


def scale_pnl(original_pnl: float, original_contracts: int, fitted: int) -> float:
    if original_contracts <= 0 or fitted <= 0:
        return 0.0
    return round(original_pnl * (fitted / original_contracts), 4)


def simulate_variant(variant: str, config: dict, force: bool) -> None:
    out_path = bc.EQUITY_DIR / f"{variant}.csv"
    if bc.skip_if_exists(out_path, force, f"step4/{variant}"):
        return

    log_path = trade_log_path(variant)
    if not log_path.exists():
        print(f"ERROR: missing trade log {log_path}")
        sys.exit(1)

    trades = bc.read_jsonl(log_path)
    if not trades:
        print(f"ERROR: empty trade log {log_path}")
        sys.exit(1)

    is_modal = variant.startswith("modal_maker_")
    is_flat = variant.startswith("ngboost_flat_")
    df = pd.DataFrame(trades)
    dates = sorted(df["date"].unique()) if "date" in df.columns else []

    bankroll = bc.INITIAL_BANKROLL_USD
    eliminated = False
    rows: list[dict] = []

    for date_str in dates:
        if bankroll <= bc.ELIMINATION_USD:
            eliminated = True
            rows.append({
                "date": date_str,
                "bankroll_usd": bankroll,
                "daily_pnl_usd": 0.0,
                "eliminated": True,
                "n_trades": 0,
            })
            break

        day_cap = bc.daily_budget_ngboost(bankroll, config)
        day_trades = df[(df["date"] == date_str) & (df["traded"] == True)]  # noqa: E712
        daily_spent = 0.0
        day_pnl = 0.0
        n_trades = 0

        for _, trade in day_trades.iterrows():
            entry_price = float(trade["entry_price"])
            original_n = int(trade.get("n_contracts", 5))
            if is_modal:
                desired = 5
            elif is_flat:
                from src.sizing import poly_contracts_for_price

                desired = poly_contracts_for_price(entry_price)
            else:
                desired = original_n
            fitted = fit_contracts(desired, entry_price, bankroll, daily_spent, day_cap)
            if fitted <= 0:
                continue
            pnl = scale_pnl(float(trade["pnl_usd"]), original_n, fitted)
            daily_spent += fitted * entry_price
            day_pnl += pnl
            n_trades += 1

        bankroll += day_pnl
        rows.append({
            "date": date_str,
            "bankroll_usd": round(bankroll, 4),
            "daily_pnl_usd": round(day_pnl, 4),
            "eliminated": bankroll <= bc.ELIMINATION_USD,
            "n_trades": n_trades,
        })
        if bankroll <= bc.ELIMINATION_USD:
            eliminated = True
            break

    out_df = pd.DataFrame(rows)
    bc.EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"{variant}: {len(rows)} days, final bankroll ${bankroll:.2f}, "
        f"eliminated={eliminated} → {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP equity simulation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="", help="Output suffix, e.g. v5")
    parser.add_argument(
        "--include-flat",
        action="store_true",
        help="Also simulate ngboost_flat_hold_to_settlement",
    )
    parser.add_argument(
        "--flat-only",
        action="store_true",
        help="Simulate only ngboost_flat_hold_to_settlement",
    )
    args = parser.parse_args()

    if args.output_tag:
        bc.configure_output_tag(args.output_tag)

    config = bc.load_trading_config()
    if args.flat_only:
        variants = [FLAT_VARIANT]
    else:
        variants = list(VARIANTS)
        if args.include_flat or args.output_tag in ("v5", "v5b"):
            variants.append(FLAT_VARIANT)

    for variant in variants:
        simulate_variant(variant, config, args.force)


if __name__ == "__main__":
    main()
