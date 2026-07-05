"""Multi-city Kelly allocation with regional caps for HRRR-NGBoost backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class BetCandidate:
    city: str
    bucket: str
    prob: float
    cost: float  # maker entry price (best_ask - 0.01)
    region: str


@dataclass
class KellyAllocation:
    fractions: np.ndarray
    contracts: list[int]
    regional_cap_bound: bool = False
    expected_log_growth: float = 0.0
    bets: list[BetCandidate] = field(default_factory=list)


def _neg_log_growth(
    f: np.ndarray,
    probs: np.ndarray,
    costs: np.ndarray,
    city_groups: list[list[int]],
) -> float:
    """Expected log growth for independent city groups with mutually exclusive buckets per city."""
    n = len(probs)
    total = 0.0

    # Enumerate outcomes: each city resolves to one of its buckets or none
    # Approximate via independent expectation per city group (standard portfolio Kelly extension)
    for group in city_groups:
        if not group:
            continue
        group_probs = probs[group]
        group_f = f[group]
        group_costs = costs[group]

        group_total = 0.0
        for win_idx in range(len(group)):
            wealth = 1.0
            for j_local, j_global in enumerate(group):
                fj = group_f[j_local]
                if j_local == win_idx:
                    cj = group_costs[j_local]
                    if cj <= 0 or cj >= 1:
                        return 1e10
                    wealth += fj * (1.0 / cj - 1.0)
                else:
                    wealth -= fj
            if wealth <= 0:
                return 1e10
            group_total += group_probs[win_idx] * np.log(wealth)

        p_none = max(0.0, 1.0 - float(np.sum(group_probs)))
        if p_none > 0:
            wealth_none = 1.0 - float(np.sum(group_f))
            if wealth_none <= 0:
                return 1e10
            group_total += p_none * np.log(wealth_none)
        total += group_total

    return -total


def allocate_ngboost_kelly(
    bets: list[BetCandidate],
    bankroll_usd: float,
    daily_budget_usd: float,
    max_position_pct: float = 0.30,
    regional_cap_pct: float = 0.60,
    regions: dict[str, list[str]] | None = None,
) -> KellyAllocation:
    """Solve multi-bet Kelly with position and regional caps."""
    if not bets or daily_budget_usd <= 0 or bankroll_usd <= 0:
        return KellyAllocation(fractions=np.array([]), contracts=[], bets=bets)

    n = len(bets)
    probs = np.array([b.prob for b in bets], dtype=float)
    costs = np.array([b.cost for b in bets], dtype=float)
    cap_total = min(daily_budget_usd / bankroll_usd, 1.0)
    per_bet_cap = max_position_pct

    # Group bet indices by city (mutually exclusive within city)
    city_to_indices: dict[str, list[int]] = {}
    for i, bet in enumerate(bets):
        city_to_indices.setdefault(bet.city, []).append(i)
    city_groups = list(city_to_indices.values())

    bounds = [(0.0, min(per_bet_cap, cap_total)) for _ in range(n)]
    constraints: list[dict] = [{"type": "ineq", "fun": lambda f, ct=cap_total: ct - np.sum(f)}]

    result = minimize(
        lambda f: _neg_log_growth(f, probs, costs, city_groups),
        np.zeros(n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    fractions = np.maximum(result.x, 0.0)
    regional_cap_bound = False

    if regions:
        budget_fraction = cap_total
        for _region, cities in regions.items():
            region_idx = [i for i, b in enumerate(bets) if b.city in cities]
            if not region_idx:
                continue
            region_sum = float(np.sum(fractions[region_idx]))
            region_cap = regional_cap_pct * budget_fraction
            if region_sum > region_cap and region_sum > 0:
                scale = region_cap / region_sum
                fractions[region_idx] *= scale
                regional_cap_bound = True

    contracts: list[int] = []
    for i, bet in enumerate(bets):
        frac = fractions[i]
        if frac <= 0 or bet.cost <= 0:
            contracts.append(0)
            continue
        stake_usd = frac * bankroll_usd
        contracts.append(max(int(stake_usd / bet.cost), 0))

    return KellyAllocation(
        fractions=fractions,
        contracts=contracts,
        regional_cap_bound=regional_cap_bound,
        expected_log_growth=-result.fun if result.success else 0.0,
        bets=bets,
    )
