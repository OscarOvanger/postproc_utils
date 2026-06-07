"""Run all seven frozen baselines on the time_holdout OOS partition."""

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

from backtest_utils import (  # noqa: E402
    bootstrap_sharpe,
    bucket_decile_breakdown,
    daily_returns,
    full_stats_table,
    is_oos_comparison,
    sharpe_stats,
)
from frozen_params import load_frozen_params  # noqa: E402
from snapshot_stability import SPLIT_DIR, assert_no_true_holdout, load_or_create_frozen_k  # noqa: E402
from implied_favorite import evaluate_implied_favorite  # noqa: E402
from implied_distribution_copy import evaluate_distribution_copy  # noqa: E402
from sell_longshots import evaluate_sell_longshots  # noqa: E402
from make_the_market import evaluate_make_the_market  # noqa: E402
from mode_prob_threshold import evaluate_mode_prob_threshold  # noqa: E402
from entropy_threshold import evaluate_entropy_threshold  # noqa: E402
from momentum_threshold import evaluate_momentum_threshold  # noqa: E402

OPT_DIR = SPLIT_DIR / "optimisation"
OOS_PATH = SPLIT_DIR / "time_holdout.parquet"
TRUE_HOLDOUT_NAME = "true_holdout"
OOS_DIR = SPLIT_DIR / "oos_results"
RAW_DATA_DIR = PROJECT_ROOT / "historic_tmax_market_data"
CSV_PATTERN = "*/*tmax_kalshi*5min_same_day.csv"


def assert_not_true_holdout_path(path: Path) -> None:
    """Crash if a caller tries to pass the protected true-holdout file."""
    if TRUE_HOLDOUT_NAME in path.name or TRUE_HOLDOUT_NAME in path.as_posix():
        raise AssertionError("true_holdout.parquet must not be loaded for OOS evaluation")


def load_time_holdout(path: Path = OOS_PATH) -> pd.DataFrame:
    assert_not_true_holdout_path(path)
    partition = pd.read_parquet(path)
    if "partition" not in partition.columns:
        raise AssertionError("time_holdout must contain a partition column")
    labels = set(partition["partition"].dropna().astype(str).unique())
    if labels != {"time_holdout"}:
        raise AssertionError(f"Expected only time_holdout rows, found {sorted(labels)}")
    assert_no_true_holdout(partition)
    return partition


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


def add_entry_alias(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add entry_snapshot_time without removing existing entry_time compatibility."""
    df = results_df.copy()
    if "entry_snapshot_time" not in df.columns:
        if "entry_time" in df.columns:
            df["entry_snapshot_time"] = df["entry_time"]
        elif "crossing_snapshot_time" in df.columns:
            df["entry_snapshot_time"] = df["crossing_snapshot_time"]
        else:
            df["entry_snapshot_time"] = pd.NaT
    if "fee_cents" not in df.columns and "total_fee_cents" in df.columns:
        df["fee_cents"] = df["total_fee_cents"]
    if "side" not in df.columns:
        df["side"] = "YES"
    if "bucket_label" not in df.columns and "resolved_bucket_label" in df.columns:
        df["bucket_label"] = df["resolved_bucket_label"]
    if "entry_price" not in df.columns:
        df["entry_price"] = pd.NA
    if "signal_value" not in df.columns:
        df["signal_value"] = pd.NA
    if "resolved_correctly" not in df.columns:
        df["resolved_correctly"] = pd.NA
    return df


def load_market_df() -> pd.DataFrame:
    csv_paths = sorted(RAW_DATA_DIR.glob(CSV_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {RAW_DATA_DIR / CSV_PATTERN}")

    required_columns = {
        "event_date",
        "city",
        "snapshot_time_local",
        "bucket_label",
        "yes_mid_close",
    }
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        available = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in available if column in required_columns]
        df = pd.read_csv(path, usecols=usecols)
        if "city" not in df.columns:
            df["city"] = path.parent.name.replace("_", " ").title()
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
        df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
        df["source_city_folder"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(frame.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))


def main() -> None:
    time_holdout = load_time_holdout()
    time_holdout = time_holdout.copy()
    time_holdout["event_date"] = pd.to_datetime(time_holdout["event_date"]).dt.date
    start_date = time_holdout["event_date"].min()
    end_date = time_holdout["event_date"].max()
    train_cities = time_holdout["source_city_folder"].nunique()
    day_equivalents = time_holdout[["source_city_folder", "event_date"]].drop_duplicates()

    frozen_k = load_or_create_frozen_k()
    frozen_params = load_frozen_params()
    require_frozen_params(frozen_params, ["t_star", "h_star", "d_star", "w_star"])

    print(f"Loaded time_holdout OOS partition: {start_date} to {end_date}")
    print(f"Train cities: {train_cities}; day-equivalents: {len(day_equivalents)}")
    print(f"Frozen k: {frozen_k}")
    print(f"Frozen params: {frozen_params}\n")

    favorite_results = evaluate_implied_favorite(time_holdout, k=frozen_k)
    distribution_results = evaluate_distribution_copy(time_holdout, k=frozen_k)
    longshot_trades, _ = evaluate_sell_longshots(time_holdout, k=frozen_k)
    make_market_results = evaluate_make_the_market(time_holdout)
    mode_prob_results = evaluate_mode_prob_threshold(
        time_holdout, t_star=float(frozen_params["t_star"])
    )
    entropy_results = evaluate_entropy_threshold(
        time_holdout, h_star=float(frozen_params["h_star"])
    )
    momentum_results = evaluate_momentum_threshold(
        time_holdout,
        d_star=float(frozen_params["d_star"]),
        w=int(frozen_params["w_star"]),
    )

    results_dict = {
        "implied_favorite": add_entry_alias(favorite_results),
        "distribution_copy": add_entry_alias(distribution_results),
        "sell_longshots": add_entry_alias(longshot_trades),
        "make_the_market": add_entry_alias(make_market_results),
        "mode_prob_threshold": add_entry_alias(mode_prob_results),
        "entropy_threshold": add_entry_alias(entropy_results),
        "momentum_threshold": add_entry_alias(momentum_results),
    }

    n_variants = count_n_variants()
    stats_table = full_stats_table(results_dict, n_variants=n_variants)

    OOS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in results_dict.items():
        frame.to_parquet(OOS_DIR / f"{name}_OOS.parquet", index=False)
    stats_path = OOS_DIR / "full_stats_table_OOS.csv"
    stats_table.to_csv(stats_path, index=False)
    print_frame("OOS Full Stats Table", stats_table)

    comparison = is_oos_comparison(
        str(SPLIT_DIR / "smoke_test_results" / "full_stats_table_IS.csv"),
        str(stats_path),
    )
    comparison.to_csv(OOS_DIR / "is_oos_comparison.csv", index=False)
    print_frame("IS vs OOS Comparison", comparison)

    entropy_oos = pd.read_parquet(OOS_DIR / "entropy_threshold_OOS.parquet")
    entropy_returns = daily_returns(entropy_oos)
    entropy_stats = sharpe_stats(entropy_returns)
    entropy_boot = bootstrap_sharpe(entropy_returns, n_boot=2000)
    bootstrap_table = pd.DataFrame(
        [
            {
                "method": "Lo-2002 parametric",
                "sharpe_mean": entropy_stats["sharpe_annual"],
                "sharpe_se": entropy_stats["sharpe_se"],
                "ci_low": entropy_stats["sharpe_ci_low"],
                "ci_high": entropy_stats["sharpe_ci_high"],
            },
            {
                "method": "Circular block bootstrap",
                "sharpe_mean": entropy_boot["sharpe_boot_mean"],
                "sharpe_se": entropy_boot["sharpe_boot_se"],
                "ci_low": entropy_boot["sharpe_boot_ci_low"],
                "ci_high": entropy_boot["sharpe_boot_ci_high"],
            },
        ]
    )
    print_frame("Entropy Threshold OOS Sharpe CI", bootstrap_table)

    market_df = load_market_df()
    entropy_decile = bucket_decile_breakdown(entropy_oos, market_df)
    entropy_decile.to_csv(OOS_DIR / "entropy_threshold_decile_OOS.csv", index=False)
    print_frame("Entropy Threshold OOS Decile Breakdown", entropy_decile)

    favorite_oos = pd.read_parquet(OOS_DIR / "implied_favorite_OOS.parquet")
    favorite_decile = bucket_decile_breakdown(favorite_oos, market_df)
    favorite_decile.to_csv(OOS_DIR / "implied_favorite_decile_OOS.csv", index=False)
    print_frame("Implied Favorite OOS Decile Breakdown", favorite_decile)

    print(
        "\nOOS evaluation complete on time_holdout partition.\n"
        "True holdout (true_holdout.parquet) has NOT been loaded."
    )


if __name__ == "__main__":
    main()
