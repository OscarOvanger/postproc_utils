"""Run all seven baselines in-sample and produce the full statistics table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINES_DIR = SRC_DIR / "baselines"
for path in (SRC_DIR, BASELINES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backtest_utils import full_stats_table  # noqa: E402
from frozen_params import load_frozen_params  # noqa: E402
from snapshot_stability import SPLIT_DIR, load_or_create_frozen_k  # noqa: E402
from implied_favorite import evaluate_implied_favorite  # noqa: E402
from implied_distribution_copy import evaluate_distribution_copy  # noqa: E402
from sell_longshots import evaluate_sell_longshots  # noqa: E402
from make_the_market import evaluate_make_the_market  # noqa: E402
from mode_prob_threshold import evaluate_mode_prob_threshold  # noqa: E402
from entropy_threshold import evaluate_entropy_threshold  # noqa: E402
from momentum_threshold import evaluate_momentum_threshold  # noqa: E402

OPT_DIR = SPLIT_DIR / "optimisation"


def count_n_variants() -> int:
    """Total parameter combinations tried across all threshold grid searches."""
    paths = [
        OPT_DIR / "mode_prob_t_grid.parquet",
        OPT_DIR / "entropy_h_grid.parquet",
        OPT_DIR / "momentum_grid.parquet",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing optimisation grid parquet(s); run the threshold optimisers first: "
            + ", ".join(missing)
        )
    return sum(len(pd.read_parquet(path)) for path in paths)


def require_frozen_params(params: dict, required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise KeyError(
            "Missing frozen parameter(s) in frozen_params.json; run the threshold "
            f"optimisers first: {', '.join(missing)}"
        )


def main() -> None:
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    frozen_k = load_or_create_frozen_k()
    frozen_params = load_frozen_params()
    require_frozen_params(frozen_params, ["t_star", "h_star", "d_star", "w_star"])
    print(f"Loaded threshold_opt with frozen k = {frozen_k}")
    print(f"Frozen params: {frozen_params}\n")

    favorite_results = evaluate_implied_favorite(threshold_opt, k=frozen_k)
    distribution_results = evaluate_distribution_copy(threshold_opt, k=frozen_k)
    longshot_trades, _ = evaluate_sell_longshots(threshold_opt, k=frozen_k)
    make_market_results = evaluate_make_the_market(threshold_opt)
    mode_prob_results = evaluate_mode_prob_threshold(
        threshold_opt, t_star=float(frozen_params["t_star"])
    )
    entropy_results = evaluate_entropy_threshold(
        threshold_opt, h_star=float(frozen_params["h_star"])
    )
    momentum_results = evaluate_momentum_threshold(
        threshold_opt,
        d_star=float(frozen_params["d_star"]),
        w=int(frozen_params["w_star"]),
    )

    results_dict = {
        "implied_favorite": favorite_results,
        "distribution_copy": distribution_results,
        "sell_longshots": longshot_trades,
        "make_the_market": make_market_results,
        "mode_prob_threshold": mode_prob_results,
        "entropy_threshold": entropy_results,
        "momentum_threshold": momentum_results,
    }

    n_variants = count_n_variants()
    stats_table = full_stats_table(results_dict, n_variants=n_variants)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(stats_table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    results_dir = SPLIT_DIR / "smoke_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results_dict.items():
        frame.to_parquet(results_dir / f"{name}_IS.parquet", index=False)
    stats_table.to_csv(results_dir / "full_stats_table_IS.csv", index=False)
    print(f"\nSaved smoke-test results to {results_dir}")

    print(
        "\nNOTE: above results are IN-SAMPLE (threshold_opt).\n"
        "      Do not interpret as OOS performance."
    )


if __name__ == "__main__":
    main()
