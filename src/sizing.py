import math

import numpy as np
from scipy.optimize import minimize


def full_kelly(p: float, c: float, cap: float = 0.08) -> float:
    """Kelly fraction for a binary bet.

    Args:
        p: estimated probability of winning (model probability)
        c: cost per contract in dollars (YES mid price, e.g. 0.35)
        cap: maximum fraction of bankroll (default 8%)

    Returns:
        Fraction of bankroll to wager, in [0, cap].
    """
    if p <= c or p <= 0 or p >= 1 or c <= 0 or c >= 1:
        return 0.0
    f = (p - c) / (1 - c)
    return min(max(f, 0.0), cap)


def half_kelly(p: float, c: float, cap: float = 0.08) -> float:
    """Half-Kelly: half the full Kelly fraction."""
    return full_kelly(p, c, cap) * 0.5


def portfolio_kelly(
    bucket_probs: np.ndarray,
    bucket_prices: np.ndarray,
    fee_per_contract: float,
    cap_total: float = 0.15,
) -> dict:
    """Multi-bucket Kelly for mutually exclusive outcomes.

    Maximises E[log(1 + sum_j f_j * (payoff_j - 1))] where payoff_j
    for bucket j winning = 1/c_j, and payoff for losing = 0 (lose stake).

    Args:
        bucket_probs: array of model probabilities per bucket (sum ~1)
        bucket_prices: array of YES mid prices per bucket
        fee_per_contract: fee in dollars per contract
        cap_total: max total fraction across all buckets (default 15%)

    Returns:
        dict with keys "fractions" (array), "best_bucket" (int),
        "expected_log_growth" (float)
    """
    k = len(bucket_probs)
    assert k == len(bucket_prices)

    net_prices = bucket_prices + fee_per_contract

    def neg_expected_log_growth(f):
        total = 0.0
        for win_k in range(k):
            wealth = 1.0
            for j in range(k):
                if j == win_k:
                    wealth += f[j] * (1.0 / net_prices[j] - 1.0)
                else:
                    wealth -= f[j]
            if wealth <= 0:
                return 1e10
            total += bucket_probs[win_k] * np.log(wealth)
        p_none = max(0, 1.0 - np.sum(bucket_probs))
        if p_none > 0:
            wealth_none = 1.0 - np.sum(f)
            if wealth_none <= 0:
                return 1e10
            total += p_none * np.log(wealth_none)
        return -total

    bounds = [(0, cap_total) for _ in range(k)]
    constraints = [{"type": "ineq", "fun": lambda f: cap_total - np.sum(f)}]
    x0 = np.zeros(k)

    result = minimize(
        neg_expected_log_growth,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    fractions = np.maximum(result.x, 0)
    best_bucket = int(np.argmax(fractions))

    return {
        "fractions": fractions,
        "best_bucket": best_bucket,
        "expected_log_growth": -result.fun,
    }


def has_edge(p: float, c: float, fee: float) -> bool:
    """Check if estimated edge exceeds the fee guardrail."""
    return (p - c) > 2 * fee


def effective_probability(
    model_prob: float,
    market_price: float,
    shrinkage_lambda: float = 1.0,
) -> float:
    """Blend model probability toward market price before edge/sizing decisions."""
    return shrinkage_lambda * model_prob + (1.0 - shrinkage_lambda) * market_price


def daily_cap_from_bankroll(bankroll: float, config: dict) -> float:
    """Anti-cyclic daily budget with optional hard ceiling fallback."""
    budget_floor = float(config.get("budget_floor", 70.0))
    budget_divisor = float(config.get("budget_divisor", 4.0))
    budget_cap_bankroll = float(config.get("budget_cap_bankroll", float("inf")))
    hard_cap = float(config.get("daily_loss_cap", 6.0))
    dynamic = (min(bankroll, budget_cap_bankroll) - budget_floor) / budget_divisor
    return max(0.0, min(dynamic, hard_cap))


def taker_fee_cents(contracts: int, price: float) -> int:
    """Exchange taker fee in cents."""
    return math.ceil(0.07 * contracts * price * (1 - price))


def contracts_from_kelly(f: float, bankroll_cents: int, c: float) -> int:
    """Convert Kelly fraction to integer contract count.

    Args:
        f: Kelly fraction (e.g. 0.04)
        bankroll_cents: current bankroll in cents
        c: YES mid price in dollars
    """
    if f <= 0 or c <= 0:
        return 0
    return max(int(f * bankroll_cents / (c * 100)), 0)


def contracts_with_daily_cap(
    f: float,
    bankroll_cents: int,
    c: float,
    daily_spent_cents: int,
    daily_cap_cents: int = 600,
) -> int:
    """Like contracts_from_kelly but enforces a daily loss cap.

    Args:
        daily_spent_cents: capital already at risk today (cents)
        daily_cap_cents: max daily capital at risk (default $6 = 600c)
    """
    n = contracts_from_kelly(f, bankroll_cents, c)
    cost_cents = int(n * c * 100)
    remaining = daily_cap_cents - daily_spent_cents
    if remaining <= 0:
        return 0
    if cost_cents > remaining:
        n = max(int(remaining / (c * 100)), 0)
    return n
