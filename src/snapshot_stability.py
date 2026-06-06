"""Snapshot-stability entry rule and optimization utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from entry_interface import TradeSignal, filter_to_trading_window, make_entry_rule
from fees import net_pnl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FROZEN_K_PATH = SPLIT_DIR / "frozen_k.json"
CONTRACTS = 1


def assert_no_true_holdout(partition_df: pd.DataFrame) -> None:
    """Assert that a partition dataframe does not contain true holdout rows."""
    if "partition" not in partition_df.columns:
        raise AssertionError("partition_df must contain a partition column")
    if partition_df["partition"].eq("true_holdout").any():
        raise AssertionError("partition_df must not contain true_holdout rows")


def compute_modal_bucket(day_df: pd.DataFrame, snapshot_time: pd.Timestamp) -> str:
    """
    At a given snapshot, return the bucket_label with the highest yes_mid_close.
    """
    required = {"snapshot_time_local", "bucket_label", "yes_mid_close"}
    missing = required.difference(day_df.columns)
    if missing:
        raise ValueError(f"day_df is missing required columns: {sorted(missing)}")

    snapshot = day_df[pd.to_datetime(day_df["snapshot_time_local"]) == snapshot_time]
    if snapshot.empty:
        raise ValueError(f"snapshot_time {snapshot_time} is not present in day_df")
    best_idx = snapshot["yes_mid_close"].astype(float).idxmax()
    return str(snapshot.loc[best_idx, "bucket_label"])


def _event_date_value(day_df: pd.DataFrame) -> str:
    """Return the single event_date value for a day dataframe as a string."""
    if "event_date" not in day_df.columns:
        raise ValueError("day_df must contain event_date")
    event_dates = day_df["event_date"].dropna().unique()
    if len(event_dates) != 1:
        raise ValueError("day_df must contain exactly one event_date")
    return str(event_dates[0])


def _modal_row(day_df: pd.DataFrame, snapshot_time: pd.Timestamp) -> pd.Series:
    """Return the modal bucket row at one snapshot."""
    snapshot = day_df[pd.to_datetime(day_df["snapshot_time_local"]) == snapshot_time]
    if snapshot.empty:
        raise ValueError(f"snapshot_time {snapshot_time} is not present in day_df")
    return snapshot.loc[snapshot["yes_mid_close"].astype(float).idxmax()]


@make_entry_rule
def stability_entry(day_df: pd.DataFrame, k: int) -> TradeSignal:
    """
    Entry rule for point-in-time baselines (implied-favorite, distribution-copy).

    Returns a TradeSignal at the first snapshot where the modal bucket
    has been unchanged for k consecutive 5-min snapshots.
    If the mode never stabilizes across the full day, returns a
    TradeSignal with no_signal=True.

    Assumes day_df has already been filtered to the 10AM trading window
    by the caller via filter_to_trading_window().
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    sorted_day = day_df.sort_values("snapshot_time_local").copy()
    sorted_day["snapshot_time_local"] = pd.to_datetime(sorted_day["snapshot_time_local"])
    snapshot_times = list(sorted_day["snapshot_time_local"].dropna().drop_duplicates())
    event_date = _event_date_value(sorted_day)

    if not snapshot_times:
        raise ValueError("day_df must contain at least one snapshot_time_local")

    modal_buckets: list[str] = []
    for snapshot_time in snapshot_times:
        modal_bucket = compute_modal_bucket(sorted_day, snapshot_time)
        modal_buckets.append(modal_bucket)
        if len(modal_buckets) >= k and len(set(modal_buckets[-k:])) == 1:
            row = _modal_row(sorted_day, snapshot_time)
            return TradeSignal(
                event_date=event_date,
                entry_snapshot_time=snapshot_time,
                bucket_label=str(row["bucket_label"]),
                side="YES",
                entry_price=float(row["yes_mid_close"]),
                signal_name="snapshot_stability",
                signal_value=float(row["yes_mid_close"]),
            )

    final_snapshot_time = snapshot_times[-1]
    return TradeSignal(
        event_date=event_date,
        entry_snapshot_time=final_snapshot_time,
        bucket_label="",
        side="YES",
        entry_price=float("nan"),
        signal_name="snapshot_stability",
        signal_value=float("nan"),
        no_signal=True,
    )


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    """Return columns that uniquely identify one city-date trading day."""
    city_col = "source_city_folder" if "source_city_folder" in partition_df.columns else "city"
    return [city_col, "event_date"]


def _resolved_yes_for_signal(day_df: pd.DataFrame, signal: TradeSignal) -> float:
    """Return 1.0 when the signal's YES bucket resolved to one dollar, else 0.0."""
    entry_rows = day_df[day_df["bucket_label"].astype(str) == signal.bucket_label]
    if entry_rows.empty:
        raise ValueError(f"bucket_label {signal.bucket_label} not found in day_df")
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved_values) != 1:
        raise ValueError(f"bucket_label {signal.bucket_label} has inconsistent resolution")
    return float(resolved_values[0])


def _sharpe(pnls: list[float]) -> float:
    """Return the sample Sharpe ratio for a sequence of trade PnLs."""
    if len(pnls) < 2:
        return float("nan")
    values = np.asarray(pnls, dtype=float)
    std = values.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(values.mean() / std)


def optimise_k(partition_df: pd.DataFrame, k_grid: list[int] | None = None) -> int:
    """
    Find the value of k in k_grid that maximises Sharpe net of taker
    fees on partition_df (pass threshold_opt partition only).

    k_grid defaults to [1, 2, 3, 4, 5, 6].
    Prints a table of k vs. N_trades, Sharpe, % no-signal days.
    Returns the best k.
    """
    assert_no_true_holdout(partition_df)
    if k_grid is None:
        k_grid = [1, 2, 3, 4, 5, 6]

    required = {
        "event_date",
        "snapshot_time_local",
        "bucket_label",
        "bucket_resolved_to_one_dollars",
        "yes_mid_close",
    }
    missing = required.difference(partition_df.columns)
    if missing:
        raise ValueError(f"partition_df is missing required columns: {sorted(missing)}")

    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    day_groups = [
        (key, filter_to_trading_window(day_df))
        for key, day_df in df.groupby(group_cols, sort=True)
    ]
    total_days = len(day_groups)

    results: list[dict[str, float | int]] = []
    for k in k_grid:
        pnls: list[float] = []
        no_signal_days = 0
        for _, day_df in day_groups:
            if day_df.empty:
                no_signal_days += 1
                continue
            signal = stability_entry(day_df, k=k)
            if signal.no_signal:
                no_signal_days += 1
                continue

            resolved_yes = _resolved_yes_for_signal(day_df, signal)
            gross_pnl_cents = 100 * CONTRACTS * (resolved_yes - signal.entry_price)
            pnl = net_pnl(
                gross_pnl_cents,
                C=CONTRACTS,
                P=signal.entry_price,
                order_type="taker",
            )
            pnls.append(pnl)

        sharpe = _sharpe(pnls)
        no_signal_pct = 100 * no_signal_days / total_days if total_days else float("nan")
        results.append(
            {
                "k": k,
                "N_trades": len(pnls),
                "Sharpe": sharpe,
                "% no-signal days": no_signal_pct,
            }
        )

    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))

    sortable = table.dropna(subset=["Sharpe"])
    if sortable.empty:
        raise ValueError("No finite Sharpe values were produced for k_grid")
    return int(sortable.sort_values(["Sharpe", "k"], ascending=[False, True]).iloc[0]["k"])


def load_or_create_frozen_k(
    split_dir: Path = SPLIT_DIR,
    force_recompute: bool = False,
) -> int:
    """
    Return the frozen snapshot-stability k, persisting it to frozen_k.json.

    If ``frozen_k.json`` already exists under ``split_dir`` and
    ``force_recompute`` is False, load and return its ``k`` value. Otherwise,
    load ``threshold_opt.parquet`` from the same directory, run
    :func:`optimise_k` on it, write ``{"k": <int>}`` to ``frozen_k.json``,
    and return the optimized k.

    This is the single source of truth for k. Baseline modules must call this
    (or read the persisted file) and must never re-optimise k themselves.
    """
    frozen_k_path = split_dir / "frozen_k.json"
    if frozen_k_path.exists() and not force_recompute:
        with open(frozen_k_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return int(payload["k"])

    threshold_opt = pd.read_parquet(split_dir / "threshold_opt.parquet")
    best_k = optimise_k(threshold_opt)
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(frozen_k_path, "w", encoding="utf-8") as handle:
        json.dump({"k": int(best_k)}, handle)
    return int(best_k)


if __name__ == "__main__":
    frozen_k = load_or_create_frozen_k(force_recompute=True)
    print(f"NEW frozen_k = {frozen_k}  (was 4 with overnight window)")
    print(f"Persisted to {FROZEN_K_PATH}")
