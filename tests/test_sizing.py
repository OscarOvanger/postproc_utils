import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sizing import (
    contracts_from_kelly,
    contracts_with_daily_cap,
    full_kelly,
    half_kelly,
    has_edge,
    portfolio_kelly,
    taker_fee_cents,
)


def test_kelly_zero_when_no_edge():
    assert full_kelly(0.30, 0.30) == 0.0
    assert full_kelly(0.20, 0.30) == 0.0


def test_kelly_positive_when_edge():
    f = full_kelly(0.50, 0.30)
    assert f > 0
    assert f == 0.08


def test_kelly_cap_enforced():
    f = full_kelly(0.90, 0.10)
    assert f <= 0.08


def test_half_kelly():
    f = half_kelly(0.50, 0.30)
    assert f == full_kelly(0.50, 0.30) * 0.5


def test_has_edge():
    fee = 0.01
    assert has_edge(0.40, 0.30, fee) is True
    assert has_edge(0.31, 0.30, fee) is False


def test_contracts_from_kelly():
    n = contracts_from_kelly(0.04, 10000, 0.40)
    assert n == 10


def test_daily_cap():
    n = contracts_with_daily_cap(0.08, 10000, 0.30, 500, 600)
    assert n <= 3
    n2 = contracts_with_daily_cap(0.08, 10000, 0.30, 600, 600)
    assert n2 == 0


def test_portfolio_kelly_single_edge():
    probs = np.array([0.50, 0.30, 0.20])
    prices = np.array([0.30, 0.35, 0.35])
    result = portfolio_kelly(probs, prices, fee_per_contract=0.01)
    assert result["fractions"][0] > 0
    assert result["best_bucket"] == 0


def test_taker_fee():
    assert taker_fee_cents(1, 0.30) == 1
    assert taker_fee_cents(100, 0.50) == 2


if __name__ == "__main__":
    test_kelly_zero_when_no_edge()
    test_kelly_positive_when_edge()
    test_kelly_cap_enforced()
    test_half_kelly()
    test_has_edge()
    test_contracts_from_kelly()
    test_daily_cap()
    test_portfolio_kelly_single_edge()
    test_taker_fee()
    print("All sizing tests passed.")
