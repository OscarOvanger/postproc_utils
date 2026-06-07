"""Plot cumulative IS and OOS net PnL for selected baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest_utils import cumulative_pnl_plot  # noqa: E402
from snapshot_stability import SPLIT_DIR  # noqa: E402

BASELINES = [
    "implied_favorite",
    "distribution_copy",
    "sell_longshots",
    "make_the_market",
    "mode_prob_threshold",
    "entropy_threshold",
    "momentum_threshold",
]
PLOT_BASELINES = [
    "implied_favorite",
    "make_the_market",
    "momentum_threshold",
    "entropy_threshold",
]
IS_DIR = SPLIT_DIR / "smoke_test_results"
OOS_DIR = SPLIT_DIR / "oos_results"


def load_results(results_dir: Path, suffix: str) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for baseline in BASELINES:
        path = results_dir / f"{baseline}_{suffix}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing result parquet: {path}")
        results[baseline] = pd.read_parquet(path)
    return results


def save_plot(results: dict[str, pd.DataFrame], title: str, output_path: Path) -> None:
    fig = cumulative_pnl_plot(results, PLOT_BASELINES, title=title, capital=100.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=160)


def main() -> None:
    oos_results = load_results(OOS_DIR, "OOS")
    is_results = load_results(IS_DIR, "IS")
    save_plot(
        oos_results,
        "OOS cumulative net PnL",
        OOS_DIR / "cumulative_pnl_OOS.png",
    )
    save_plot(
        is_results,
        "IS cumulative net PnL",
        OOS_DIR / "cumulative_pnl_IS.png",
    )
    print(f"Saved OOS cumulative PnL plot to {OOS_DIR / 'cumulative_pnl_OOS.png'}")
    print(f"Saved IS cumulative PnL plot to {OOS_DIR / 'cumulative_pnl_IS.png'}")


if __name__ == "__main__":
    main()
