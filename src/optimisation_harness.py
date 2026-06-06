"""Walk-forward optimisation harness for threshold baselines."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
BASELINES_DIR = SRC_DIR / "baselines"
for path in (SRC_DIR, BASELINES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backtest_utils import daily_returns, sharpe_stats  # noqa: E402
from entry_interface import TradeSignal, filter_to_trading_window  # noqa: E402
from fees import taker_fee  # noqa: E402
from snapshot_stability import SPLIT_DIR, assert_no_true_holdout  # noqa: E402

OPT_DIR = SPLIT_DIR / "optimisation"


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    city_col = (
        "source_city_folder"
        if "source_city_folder" in partition_df.columns
        else "city"
    )
    return [city_col, "event_date"]


def _resolved_correctly(day_df: pd.DataFrame, bucket_label: str) -> bool:
    entry_rows = day_df[day_df["bucket_label"].astype(str) == str(bucket_label)]
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    return bool(resolved_values[0])


def evaluate_signal_fn(
    partition_df: pd.DataFrame,
    signal_fn: Callable[..., TradeSignal],
    params: dict,
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Evaluate an arbitrary signal function over every city-date day."""
    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    records: list[dict] = []

    for _, day_df in df.groupby(group_cols, sort=True):
        city = (
            str(day_df["city"].iloc[0])
            if "city" in day_df.columns
            else str(day_df[group_cols[0]].iloc[0])
        )
        event_date = str(day_df["event_date"].dropna().iloc[0])
        day_df = filter_to_trading_window(day_df)
        if day_df.empty:
            records.append(
                {
                    "event_date": event_date,
                    "city": city,
                    "entry_time": pd.NaT,
                    "bucket_label": "",
                    "side": "YES",
                    "entry_price": np.nan,
                    "signal_value": np.nan,
                    "no_signal": True,
                    "gross_pnl_cents": np.nan,
                    "fee_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "resolved_correctly": np.nan,
                }
            )
            continue
        signal = signal_fn(day_df, **params)

        if signal.no_signal:
            records.append(
                {
                    "event_date": signal.event_date,
                    "city": city,
                    "entry_time": pd.NaT,
                    "bucket_label": "",
                    "side": signal.side,
                    "entry_price": np.nan,
                    "signal_value": np.nan,
                    "no_signal": True,
                    "gross_pnl_cents": np.nan,
                    "fee_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "resolved_correctly": np.nan,
                }
            )
            continue

        resolved_correctly = _resolved_correctly(day_df, signal.bucket_label)
        entry_price = float(signal.entry_price)
        gross_pnl_cents = (
            (1.0 - entry_price) * 100.0 if resolved_correctly else -entry_price * 100.0
        )
        fee_cents = float(taker_fee(contracts, entry_price))
        records.append(
            {
                "event_date": signal.event_date,
                "city": city,
                "entry_time": signal.entry_snapshot_time,
                "bucket_label": signal.bucket_label,
                "side": signal.side,
                "entry_price": entry_price,
                "signal_value": float(signal.signal_value),
                "no_signal": False,
                "gross_pnl_cents": gross_pnl_cents,
                "fee_cents": fee_cents,
                "net_pnl_cents": gross_pnl_cents - fee_cents,
                "resolved_correctly": resolved_correctly,
            }
        )

    return pd.DataFrame.from_records(records)


def _param_combinations(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _annualised_sharpe(results_df: pd.DataFrame) -> float:
    return float(sharpe_stats(daily_returns(results_df))["sharpe_annual"])


def _sequential_folds(dates: list, n_folds: int) -> list[list]:
    dates = sorted(dates)
    if n_folds < 1 or not dates:
        return []
    fold_size = max(1, len(dates) // n_folds)
    folds: list[list] = []
    start = 0
    for fold_idx in range(n_folds):
        if fold_idx == n_folds - 1:
            block = dates[start:]
        else:
            block = dates[start : start + fold_size]
            start += fold_size
        if block:
            folds.append(block)
    return folds


def walk_forward_optimise(
    partition_df: pd.DataFrame,
    signal_fn: Callable[..., TradeSignal],
    param_grid: dict,
    n_folds: int = 3,
    min_is_days: int = 20,
) -> dict:
    """
    Walk-forward cross-validation on partition_df by event_date.
    """
    assert_no_true_holdout(partition_df)
    df = partition_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    unique_dates = sorted(df["event_date"].dropna().unique())
    folds = _sequential_folds(unique_dates, n_folds=n_folds)
    combos = _param_combinations(param_grid)

    oos_frames: list[pd.DataFrame] = []
    fold_params: list[dict] = []
    fold_sharpes_is: list[float] = []
    fold_sharpes_oos: list[float] = []

    for oos_dates in folds:
        oos_start = min(oos_dates)
        is_dates = [d for d in unique_dates if d < oos_start]
        if len(is_dates) < min_is_days:
            continue

        is_df = df[df["event_date"].isin(is_dates)]
        oos_df = df[df["event_date"].isin(oos_dates)]

        best_params: dict | None = None
        best_sharpe = float("-inf")
        for params in combos:
            is_results = evaluate_signal_fn(is_df, signal_fn, params)
            sharpe = _annualised_sharpe(is_results)
            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        if best_params is None:
            continue

        oos_results = evaluate_signal_fn(oos_df, signal_fn, best_params)
        oos_sharpe = _annualised_sharpe(oos_results)
        oos_frames.append(oos_results)
        fold_params.append(best_params)
        fold_sharpes_is.append(best_sharpe)
        fold_sharpes_oos.append(oos_sharpe)

    oos_results_df = (
        pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    )
    signal_name = getattr(signal_fn, "__name__", "threshold_signal")
    if signal_name.endswith("_signal"):
        signal_name = signal_name[: -len("_signal")]
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_params_path = OPT_DIR / f"{signal_name}_wf_fold_params.json"
    with open(fold_params_path, "w", encoding="utf-8") as handle:
        json.dump(fold_params, handle, indent=2)
        handle.write("\n")
    return {
        "oos_results_df": oos_results_df,
        "oos_sharpe_stats": sharpe_stats(daily_returns(oos_results_df)),
        "fold_params": fold_params,
        "fold_params_path": str(fold_params_path),
        "fold_sharpes_is": fold_sharpes_is,
        "fold_sharpes_oos": fold_sharpes_oos,
    }


def plot_optimisation_curve(
    grid_df: pd.DataFrame,
    param_col: str,
    sharpe_col: str = "sharpe",
    frozen_value: float | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot Sharpe vs parameter value; returns Figure without calling show()."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    plot_df = grid_df.sort_values(param_col)
    ax.plot(plot_df[param_col], plot_df[sharpe_col], color="#4878CF", linewidth=1.8)
    if frozen_value is not None and pd.notna(frozen_value):
        ax.axvline(frozen_value, color="#8A8A8A", linestyle="--", linewidth=1.2)
        frozen_row = plot_df.loc[plot_df[param_col] == frozen_value]
        if not frozen_row.empty and {"sharpe_ci_low", "sharpe_ci_high"}.issubset(
            frozen_row.columns
        ):
            row = frozen_row.iloc[0]
            ax.axhspan(
                row["sharpe_ci_low"],
                row["sharpe_ci_high"],
                color="#4878CF",
                alpha=0.15,
            )
    ax.set_xlabel(param_col)
    ax.set_ylabel("Sharpe")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _walk_forward_with_evaluator(
    partition_df: pd.DataFrame,
    signal_name: str,
    evaluate_fn: Callable[..., pd.DataFrame],
    param_grid: dict,
    n_folds: int = 3,
    min_is_days: int = 20,
) -> dict:
    """Fast script helper that uses each baseline's vectorised evaluate wrapper."""
    assert_no_true_holdout(partition_df)
    df = partition_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    unique_dates = sorted(df["event_date"].dropna().unique())
    folds = _sequential_folds(unique_dates, n_folds=n_folds)
    combos = _param_combinations(param_grid)

    oos_frames: list[pd.DataFrame] = []
    fold_params: list[dict] = []
    fold_sharpes_is: list[float] = []
    fold_sharpes_oos: list[float] = []

    for oos_dates in folds:
        oos_start = min(oos_dates)
        is_dates = [d for d in unique_dates if d < oos_start]
        if len(is_dates) < min_is_days:
            continue

        is_df = df[df["event_date"].isin(is_dates)]
        oos_df = df[df["event_date"].isin(oos_dates)]
        best_params: dict | None = None
        best_sharpe = float("-inf")
        for params in combos:
            sharpe = _annualised_sharpe(evaluate_fn(is_df, **params))
            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        if best_params is None:
            continue

        oos_results = evaluate_fn(oos_df, **best_params)
        oos_frames.append(oos_results)
        fold_params.append(best_params)
        fold_sharpes_is.append(best_sharpe)
        fold_sharpes_oos.append(_annualised_sharpe(oos_results))

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_params_path = OPT_DIR / f"{signal_name}_wf_fold_params.json"
    with open(fold_params_path, "w", encoding="utf-8") as handle:
        json.dump(fold_params, handle, indent=2)
        handle.write("\n")
    oos_results_df = (
        pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    )
    return {
        "oos_results_df": oos_results_df,
        "oos_sharpe_stats": sharpe_stats(daily_returns(oos_results_df)),
        "fold_params": fold_params,
        "fold_params_path": str(fold_params_path),
        "fold_sharpes_is": fold_sharpes_is,
        "fold_sharpes_oos": fold_sharpes_oos,
    }


def _walk_forward_with_summaries(
    partition_df: pd.DataFrame,
    signal_name: str,
    summary_fn: Callable[[pd.DataFrame], list[dict]],
    summary_eval_fn: Callable[..., pd.DataFrame],
    param_grid: dict,
    n_folds: int = 3,
    min_is_days: int = 20,
) -> dict:
    """Fast script helper for baselines with reusable per-day summaries."""
    assert_no_true_holdout(partition_df)
    df = partition_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    unique_dates = sorted(df["event_date"].dropna().unique())
    folds = _sequential_folds(unique_dates, n_folds=n_folds)
    combos = _param_combinations(param_grid)

    oos_frames: list[pd.DataFrame] = []
    fold_params: list[dict] = []
    fold_sharpes_is: list[float] = []
    fold_sharpes_oos: list[float] = []

    for oos_dates in folds:
        oos_start = min(oos_dates)
        is_dates = [d for d in unique_dates if d < oos_start]
        if len(is_dates) < min_is_days:
            continue

        is_summaries = summary_fn(df[df["event_date"].isin(is_dates)])
        oos_summaries = summary_fn(df[df["event_date"].isin(oos_dates)])
        best_params: dict | None = None
        best_sharpe = float("-inf")
        for params in combos:
            sharpe = _annualised_sharpe(summary_eval_fn(is_summaries, **params))
            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        if best_params is None:
            continue

        oos_results = summary_eval_fn(oos_summaries, **best_params)
        oos_frames.append(oos_results)
        fold_params.append(best_params)
        fold_sharpes_is.append(best_sharpe)
        fold_sharpes_oos.append(_annualised_sharpe(oos_results))

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_params_path = OPT_DIR / f"{signal_name}_wf_fold_params.json"
    with open(fold_params_path, "w", encoding="utf-8") as handle:
        json.dump(fold_params, handle, indent=2)
        handle.write("\n")
    oos_results_df = (
        pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    )
    return {
        "oos_results_df": oos_results_df,
        "oos_sharpe_stats": sharpe_stats(daily_returns(oos_results_df)),
        "fold_params": fold_params,
        "fold_params_path": str(fold_params_path),
        "fold_sharpes_is": fold_sharpes_is,
        "fold_sharpes_oos": fold_sharpes_oos,
    }


if __name__ == "__main__":
    from entropy_threshold import (  # noqa: E402
        _day_summaries as entropy_day_summaries,
        _evaluate_from_summaries as evaluate_entropy_summaries,
    )
    from mode_prob_threshold import (  # noqa: E402
        _day_summaries as mode_prob_day_summaries,
        _evaluate_from_summaries as evaluate_mode_prob_summaries,
    )
    from momentum_threshold import (  # noqa: E402
        _day_summaries as momentum_day_summaries,
        _evaluate_from_summaries as evaluate_momentum_summaries,
    )

    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    OPT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (
            "mode_prob",
            mode_prob_day_summaries,
            evaluate_mode_prob_summaries,
            {"t_star": np.arange(0.55, 0.92, 0.01).tolist()},
        ),
        (
            "entropy",
            entropy_day_summaries,
            evaluate_entropy_summaries,
            {"h_star": np.arange(0.30, 1.85, 0.05).tolist()},
        ),
        (
            "momentum",
            momentum_day_summaries,
            evaluate_momentum_summaries,
            {
                "d_star": np.arange(0.02, 0.22, 0.02).tolist(),
                "w": [2, 4, 6, 8, 12, 16, 24],
            },
        ),
    ]

    for name, summary_fn, summary_eval_fn, param_grid in configs:
        print(f"\n=== Walk-forward: {name} ===")
        result = _walk_forward_with_summaries(
            threshold_opt,
            signal_name=name,
            summary_fn=summary_fn,
            summary_eval_fn=summary_eval_fn,
            param_grid=param_grid,
        )
        print(f"Saved fold params to {result['fold_params_path']}")

        for fold_idx, (is_sharpe, oos_sharpe, params) in enumerate(
            zip(
                result["fold_sharpes_is"],
                result["fold_sharpes_oos"],
                result["fold_params"],
            ),
            start=1,
        ):
            print(
                f"  Fold {fold_idx}: IS Sharpe={is_sharpe:0.3f}, "
                f"OOS Sharpe={oos_sharpe:0.3f}, params={params}"
            )
