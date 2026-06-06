"""Run all three baselines in-sample on threshold_opt and save smoke results."""

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

from snapshot_stability import SPLIT_DIR, load_or_create_frozen_k  # noqa: E402
from backtest_utils import print_summary_table  # noqa: E402
from implied_favorite import evaluate_implied_favorite  # noqa: E402
from implied_distribution_copy import evaluate_distribution_copy  # noqa: E402
from sell_longshots import evaluate_sell_longshots  # noqa: E402


def main() -> None:
    """Evaluate every baseline on the in-sample partition and persist results."""
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    frozen_k = load_or_create_frozen_k()
    print(f"Loaded threshold_opt with frozen k = {frozen_k}\n")

    favorite_results = evaluate_implied_favorite(threshold_opt, k=frozen_k)
    distribution_results = evaluate_distribution_copy(threshold_opt, k=frozen_k)
    longshot_trades, _ = evaluate_sell_longshots(threshold_opt, k=frozen_k)

    print_summary_table("implied_favorite", favorite_results)
    print_summary_table("implied_distribution_copy", distribution_results)
    print_summary_table("sell_longshots", longshot_trades)

    results_dir = SPLIT_DIR / "smoke_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    favorite_results.to_parquet(results_dir / "implied_favorite_IS.parquet", index=False)
    distribution_results.to_parquet(
        results_dir / "distribution_copy_IS.parquet", index=False
    )
    longshot_trades.to_parquet(results_dir / "sell_longshots_IS.parquet", index=False)
    print(f"\nSaved smoke-test results to {results_dir}")

    print(
        "\nNOTE: above results are IN-SAMPLE (threshold_opt partition).\n"
        "      Do not interpret as OOS performance."
    )


if __name__ == "__main__":
    main()
