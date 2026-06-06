"""Kalshi fee utilities for backtesting Tmax bucket trades."""

from __future__ import annotations

import math


def taker_fee(C: float, P: float) -> float:
    """Kalshi taker fee in cents. C = contracts, P = yes_mid_close."""
    return math.ceil(0.07 * C * P * (1 - P))


def maker_fee(C: float, P: float) -> float:
    """Kalshi maker fee in cents (since April 2025, ~1/4 of taker)."""
    return math.ceil(0.0175 * C * P * (1 - P))


def net_pnl(
    gross_pnl_cents: float,
    C: float,
    P: float,
    order_type: str = "taker",
) -> float:
    """Return PnL in cents after fees. order_type: 'taker' or 'maker'."""
    if order_type == "taker":
        fee = taker_fee(C, P)
    elif order_type == "maker":
        fee = maker_fee(C, P)
    else:
        raise ValueError("order_type must be 'taker' or 'maker'")
    return gross_pnl_cents - fee


def _raw_taker_fee(C: float, P: float) -> float:
    """Return the unrounded Kalshi taker fee in cents."""
    return 0.07 * C * P * (1 - P)


def _raw_maker_fee(C: float, P: float) -> float:
    """Return the unrounded Kalshi maker fee in cents."""
    return 0.0175 * C * P * (1 - P)


if __name__ == "__main__":
    contracts = 100
    probabilities = [0.10, 0.25, 0.50, 0.75, 0.90]

    print(f"Fee sanity check at C={contracts} contracts")
    print("P     raw_taker_c  taker_fee_c  raw_maker_c  maker_fee_c")
    for probability in probabilities:
        print(
            f"{probability:0.2f}  "
            f"{_raw_taker_fee(contracts, probability):11.4f}  "
            f"{taker_fee(contracts, probability):11.0f}  "
            f"{_raw_maker_fee(contracts, probability):11.4f}  "
            f"{maker_fee(contracts, probability):11.0f}"
        )
    print("Worst raw fee occurs at P=0.50: taker=1.75c, maker≈0.44c.")
