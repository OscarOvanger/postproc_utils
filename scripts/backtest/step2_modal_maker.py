#!/usr/bin/env python3
"""Step 2: modal_maker strategy on real Polymarket order-book data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest.common as bc  # noqa: E402

EXIT_VARIANTS = ["hold_to_settlement", "profit_target_15c"]


def simulate_city_date(
    city: str,
    date_str: str,
    exit_variant: str,
    wu: pd.DataFrame,
) -> dict:
    frame = bc.load_day_snapshot(city, date_str)
    if frame is None:
        raise ValueError(f"Missing snapshot for {city} {date_str}")

    snap_rows, entry_ts, excluded = bc.select_entry_snapshot(frame, city, date_str)
    base = {
        "city": city,
        "date": date_str,
        "strategy": "modal_maker",
        "exit_variant": exit_variant,
        "entry_ts": entry_ts.isoformat(),
        "lookahead_excluded_rows": excluded,
    }

    if snap_rows.empty:
        return {**base, "traded": False, "no_trade_reason": "no_entry_window_snapshot"}

    modal = bc.compute_modal_bucket(snap_rows)
    if modal is None:
        return {**base, "traded": False, "no_trade_reason": "no_modal_bucket"}

    bucket = str(modal["bucket"])
    best_ask = modal.get("best_ask")
    if best_ask is None or not (bc.MIN_ENTRY_ASK <= float(best_ask) <= bc.MAX_ENTRY_ASK):
        return {
            **base,
            "traded": False,
            "no_trade_reason": "ask_out_of_range",
            "bucket": bucket,
            "best_ask": best_ask,
        }

    entry_price = round(float(best_ask) - bc.MAKER_TICK, 4)
    n_contracts = bc.MODAL_CONTRACTS

    wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
    actual = float(wu_row.iloc[0]["wunderground_tmax"])
    won = bc.temp_in_bucket(actual, bucket)

    exit_type = "settlement"
    exit_price = 1.0 if won else 0.0
    pnl = bc.settlement_pnl(n_contracts=n_contracts, entry_price=entry_price, won=won)

    if exit_variant == "profit_target_15c":
        intraday = bc.intraday_snapshots_after_entry(frame, city, date_str, entry_ts)
        hit, target_price = bc.check_profit_target_exit(intraday, bucket, entry_price)
        if hit and target_price is not None:
            exit_type = "profit_target_15c"
            exit_price = target_price
            pnl = bc.profit_target_pnl(n_contracts, entry_price, target_price)

    return {
        **base,
        "traded": True,
        "bucket": bucket,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_type": exit_type,
        "n_contracts": n_contracts,
        "won": won if exit_type == "settlement" else pnl > 0,
        "pnl_usd": pnl,
        "actual_tmax": actual,
    }


def run_variant(exit_variant: str, eligible: pd.DataFrame, force: bool) -> None:
    out_path = bc.TRADES_DIR / f"modal_maker_{exit_variant}.jsonl"
    if bc.skip_if_exists(out_path, force, f"step2/{exit_variant}"):
        return

    wu = bc.load_wu_targets()
    records: list[dict] = []
    t0 = time.time()

    for i, row in eligible.iterrows():
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{exit_variant}] {i + 1}/{len(eligible)} ({elapsed:.1f}s)")
        records.append(simulate_city_date(str(row["city"]), str(row["date"]), exit_variant, wu))

    bc.write_jsonl(out_path, records)
    traded = sum(1 for r in records if r.get("traded"))
    print(f"Wrote {len(records)} rows ({traded} trades) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modal maker backtest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="", help="Output suffix, e.g. v5")
    args = parser.parse_args()

    if args.output_tag:
        bc.configure_output_tag(args.output_tag)

    if not bc.ELIGIBLE_DATES_CSV.exists():
        print(f"ERROR: run step1 first — missing {bc.ELIGIBLE_DATES_CSV}")
        sys.exit(1)

    eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
    if eligible.empty:
        print("ERROR: no eligible city-dates from step1")
        sys.exit(1)

    for variant in EXIT_VARIANTS:
        run_variant(variant, eligible, args.force)


if __name__ == "__main__":
    main()
