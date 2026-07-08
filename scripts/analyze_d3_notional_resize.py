#!/usr/bin/env python3
"""Compare D3 pace-amendment trades at flat-5 vs $1-min-notional sizing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sizing import poly_contracts_for_price  # noqa: E402

TRADES_DIR = PROJECT_ROOT / "data" / "analysis" / "pace_amendment_d3_trades"
WINDOWS = ("early", "middle", "late")
CHEAP_THRESHOLD = 0.20


def settlement_pnl(n_contracts: int, entry_price: float, won: bool) -> float:
    per = (1.0 - entry_price) if won else (-entry_price)
    return round(per * n_contracts, 4)


def analyze_window(window: str) -> dict:
    path = TRADES_DIR / f"d3_{window}_trades.jsonl"
    trades = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trades.append(json.loads(line))

    traded = [t for t in trades if t.get("traded")]
    cheap = [t for t in traded if float(t["entry_price"]) < CHEAP_THRESHOLD]

    orig_pnl = sum(float(t["pnl_usd"]) for t in traded)
    orig_cost = sum(float(t["cost_usd"]) for t in traded)
    orig_edge_w = sum(float(t["edge"]) * float(t["cost_usd"]) for t in traded)
    orig_realized_edge = orig_edge_w / orig_cost if orig_cost else 0.0

    resized_pnl = 0.0
    resized_cost = 0.0
    resized_edge_w = 0.0
    for t in traded:
        price = float(t["entry_price"])
        n_old = int(t["n_contracts"])
        n_new = poly_contracts_for_price(price)
        won = bool(t.get("won"))
        pnl_new = settlement_pnl(n_new, price, won)
        cost_new = n_new * price
        resized_pnl += pnl_new
        resized_cost += cost_new
        resized_edge_w += float(t["edge"]) * cost_new

    resized_realized_edge = resized_edge_w / resized_cost if resized_cost else 0.0

    return {
        "window": window,
        "n_trades": len(traded),
        "n_cheap": len(cheap),
        "cheap_pct": 100.0 * len(cheap) / len(traded) if traded else 0.0,
        "orig_pnl": orig_pnl,
        "resized_pnl": resized_pnl,
        "pnl_delta": resized_pnl - orig_pnl,
        "orig_realized_edge": orig_realized_edge,
        "resized_realized_edge": resized_realized_edge,
        "edge_delta": resized_realized_edge - orig_realized_edge,
    }


def main() -> None:
    print("D3 pace-amendment: flat-5 vs $1-min-notional resize")
    print(f"(cheap = entry_price < ${CHEAP_THRESHOLD:.2f})\n")
    print(
        f"{'window':<8} {'trades':>6} {'cheap':>6} {'%cheap':>7} "
        f"{'PnL flat5':>10} {'PnL resize':>11} {'dPnL':>8} "
        f"{'edge flat5':>10} {'edge resize':>11} {'dEdge':>8}"
    )
    print("-" * 96)
    for window in WINDOWS:
        r = analyze_window(window)
        print(
            f"{r['window']:<8} {r['n_trades']:6d} {r['n_cheap']:6d} {r['cheap_pct']:6.1f}% "
            f"{r['orig_pnl']:10.2f} {r['resized_pnl']:11.2f} {r['pnl_delta']:+8.2f} "
            f"{r['orig_realized_edge']:10.4f} {r['resized_realized_edge']:11.4f} "
            f"{r['edge_delta']:+8.4f}"
        )


if __name__ == "__main__":
    main()
