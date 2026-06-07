"""Build a consolidated in-sample and OOS baseline summary table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest_utils import _day_index_columns  # noqa: E402
from snapshot_stability import SPLIT_DIR  # noqa: E402

IS_DIR = SPLIT_DIR / "smoke_test_results"
OOS_DIR = SPLIT_DIR / "oos_results"
IS_STATS_PATH = IS_DIR / "full_stats_table_IS.csv"
OOS_STATS_PATH = OOS_DIR / "full_stats_table_OOS.csv"
SUMMARY_PATH = OOS_DIR / "full_summary_is_oos.csv"
POSITIVE_OOS_BASELINES = [
    "implied_favorite",
    "make_the_market",
    "momentum_threshold",
]


def require_columns(frame: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def daily_pnl(results_df: pd.DataFrame) -> pd.Series:
    """Return daily net PnL, with no-signal rows contributing zero."""
    required = {"event_date", "net_pnl_cents"}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"results parquet missing required columns: {sorted(missing)}")

    df = results_df.copy()
    index_cols = _day_index_columns(df)
    if "no_signal" in df.columns:
        no_signal_mask = df["no_signal"].fillna(False).astype(bool)
    else:
        no_signal_mask = pd.Series(False, index=df.index)
    pnl = pd.to_numeric(df["net_pnl_cents"], errors="coerce")
    df["_pnl"] = pnl.where(~no_signal_mask, 0.0).fillna(0.0)

    grouped = df.groupby(index_cols, sort=True)["_pnl"].sum().reset_index()
    grouped["event_date"] = pd.to_datetime(grouped["event_date"])
    return grouped.groupby("event_date", sort=True)["_pnl"].sum()


def top_three_day_concentration(results_df: pd.DataFrame) -> float:
    pnl = daily_pnl(results_df)
    total = float(pnl.sum())
    if total == 0:
        return float("nan")
    top_three = float(pnl.sort_values(ascending=False).head(3).sum())
    return 100.0 * top_three / total


def build_summary_table(is_stats: pd.DataFrame, oos_stats: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Baseline",
        "Sharpe",
        "Sharpe_CI_low",
        "Sharpe_CI_high",
        "PSR_0",
        "N_trades",
        "NoSignal_pct",
        "SR_deflated",
    }
    require_columns(is_stats, IS_STATS_PATH, required)
    require_columns(oos_stats, OOS_STATS_PATH, required)

    merged = is_stats.merge(
        oos_stats,
        on="Baseline",
        how="outer",
        suffixes=("_IS", "_OOS"),
        validate="one_to_one",
    )
    summary = pd.DataFrame(
        {
            "Baseline": merged["Baseline"],
            "IS_Sharpe": merged["Sharpe_IS"],
            "IS_CI_low": merged["Sharpe_CI_low_IS"],
            "IS_CI_high": merged["Sharpe_CI_high_IS"],
            "IS_PSR0": merged["PSR_0_IS"],
            "IS_n_trades": merged["N_trades_IS"],
            "IS_NoSig_pct": merged["NoSignal_pct_IS"],
            "OOS_Sharpe": merged["Sharpe_OOS"],
            "OOS_CI_low": merged["Sharpe_CI_low_OOS"],
            "OOS_CI_high": merged["Sharpe_CI_high_OOS"],
            "OOS_PSR0": merged["PSR_0_OOS"],
            "OOS_n_trades": merged["N_trades_OOS"],
            "OOS_NoSig_pct": merged["NoSignal_pct_OOS"],
            "IS_SR_deflated": merged["SR_deflated_IS"],
            "OOS_SR_deflated": merged["SR_deflated_OOS"],
        }
    )
    summary["Sharpe_decay"] = summary["IS_Sharpe"] - summary["OOS_Sharpe"]
    return summary.sort_values("OOS_Sharpe", ascending=False).reset_index(drop=True)


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(frame.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))


def main() -> None:
    is_stats = pd.read_csv(IS_STATS_PATH)
    oos_stats = pd.read_csv(OOS_STATS_PATH)
    summary = build_summary_table(is_stats, oos_stats)
    OOS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    rho = summary["IS_Sharpe"].corr(summary["OOS_Sharpe"], method="spearman")
    print_frame("Full IS + OOS Summary", summary)
    print(f"\nSpearman rank correlation (IS Sharpe vs OOS Sharpe): {rho:0.4f}")

    for baseline in POSITIVE_OOS_BASELINES:
        path = OOS_DIR / f"{baseline}_OOS.parquet"
        results_df = pd.read_parquet(path)
        concentration = top_three_day_concentration(results_df)
        print(
            f"Top-3-day PnL concentration for {baseline}: "
            f"{concentration:0.1f}% of total OOS PnL"
        )

    print(f"\nSaved consolidated summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
