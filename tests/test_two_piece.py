import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest.ngboost_inference import _two_piece_cdf, market_bucket_probability  # noqa: E402


def test_two_piece_monotone():
    mu, sigma, ratio = 80.0, 2.0, 1.3
    s1 = ratio * sigma
    s2 = sigma
    grid = np.linspace(mu - 10 * s1, mu + 10 * s2, 500)
    cdf_vals = [_two_piece_cdf(mu, sigma, ratio, float(x)) for x in grid]
    assert np.all(np.diff(cdf_vals) >= -1e-12)


def test_two_piece_tails():
    mu, sigma, ratio = 80.0, 2.0, 1.3
    s1 = ratio * sigma
    s2 = sigma
    assert _two_piece_cdf(mu, sigma, ratio, mu - 10 * s1) < 1e-6
    assert 1.0 - _two_piece_cdf(mu, sigma, ratio, mu + 10 * s2) < 1e-6


def test_two_piece_r_equals_one_matches_gaussian():
    mu, sigma = 75.5, 1.8
    xs = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 200)
    for x in xs:
        got = _two_piece_cdf(mu, sigma, 1.0, float(x))
        expected = norm.cdf(x, loc=mu, scale=sigma)
        assert abs(got - expected) < 1e-12


def test_two_piece_mass_at_mode():
    mu, sigma, ratio = 82.0, 2.5, 1.4
    s1 = ratio * sigma
    s2 = sigma
    assert abs(_two_piece_cdf(mu, sigma, ratio, mu) - s1 / (s1 + s2)) < 1e-12


def test_ratio_down_none_matches_gaussian_bucket_prob():
    label = "80-81"
    mu, sigma = 80.5, 2.0
    base = market_bucket_probability(label, mu, sigma, "gaussian", None, ratio_down=None)
    same = market_bucket_probability(label, mu, sigma, "gaussian", None, ratio_down=1.0)
    assert abs(base - same) < 1e-15


if __name__ == "__main__":
    test_two_piece_monotone()
    test_two_piece_tails()
    test_two_piece_r_equals_one_matches_gaussian()
    test_two_piece_mass_at_mode()
    test_ratio_down_none_matches_gaussian_bucket_prob()
    print("All two-piece tests passed.")
