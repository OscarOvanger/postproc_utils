"""Momentum threshold baseline: enter when delta_mode >= d_star."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entry_interface import TradeSignal, filter_to_trading_window, make_entry_rule  # noqa: E402
from fees import maker_fee, taker_fee  # noqa: E402
from frozen_params import load_frozen_params, save_frozen_params  # noqa: E402
from snapshot_stability import (  # noqa: E402
    SPLIT_DIR,
    _event_date_value,
    assert_no_true_holdout,
)
from backtest_utils import (  # noqa: E402
    _n_trades_and_no_signal,
    daily_returns,
    print_summary_table,
    sharpe_stats,
)

OPT_DIR = SPLIT_DIR / "optimisation"


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    city_col = (
        "source_city_folder"
        if "source_city_folder" in partition_df.columns
        else "city"
    )
    return [city_col, "event_date"]


def _mode_prob_at_snapshot(snapshot: pd.DataFrame) -> float:
    return float(snapshot["yes_mid_close"].astype(float).max())


def compute_delta_mode(
    day_df: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    w: int,
) -> float:
    """
    mode_prob at snapshot_time minus mode_prob w snapshots earlier.

    Returns NaN if fewer than w prior snapshots exist for that day.
    """
    sorted_day = day_df.sort_values("snapshot_time_local").copy()
    sorted_day["snapshot_time_local"] = pd.to_datetime(sorted_day["snapshot_time_local"])
    snapshot_times = list(sorted_day["snapshot_time_local"].dropna().drop_duplicates())
    try:
        idx = snapshot_times.index(pd.Timestamp(snapshot_time))
    except ValueError:
        return float("nan")
    if idx < w:
        return float("nan")

    current_snapshot = sorted_day[sorted_day["snapshot_time_local"] == snapshot_time]
    prior_time = snapshot_times[idx - w]
    prior_snapshot = sorted_day[sorted_day["snapshot_time_local"] == prior_time]
    return _mode_prob_at_snapshot(current_snapshot) - _mode_prob_at_snapshot(prior_snapshot)


@make_entry_rule
def momentum_signal(day_df: pd.DataFrame, d_star: float, w: int) -> TradeSignal:
    """
    Return TradeSignal at first snapshot where delta_mode >= d_star.

    Enter on the modal bucket at that snapshot.
    signal_name: "momentum_threshold"
    signal_value: delta_mode at entry snapshot
    """
    sorted_day = day_df.sort_values("snapshot_time_local").copy()
    sorted_day["snapshot_time_local"] = pd.to_datetime(sorted_day["snapshot_time_local"])
    event_date = _event_date_value(sorted_day)
    snapshot_times = list(sorted_day["snapshot_time_local"].dropna().drop_duplicates())

    for snapshot_time in snapshot_times:
        delta_mode = compute_delta_mode(sorted_day, snapshot_time, w=w)
        if pd.notna(delta_mode) and delta_mode >= d_star:
            snapshot = sorted_day[sorted_day["snapshot_time_local"] == snapshot_time]
            modal_row = snapshot.loc[snapshot["yes_mid_close"].astype(float).idxmax()]
            return TradeSignal(
                event_date=event_date,
                entry_snapshot_time=snapshot_time,
                bucket_label=str(modal_row["bucket_label"]),
                side="YES",
                entry_price=float(modal_row["yes_mid_close"]),
                signal_name="momentum_threshold",
                signal_value=float(delta_mode),
                no_signal=False,
            )

    return TradeSignal(
        event_date=event_date,
        entry_snapshot_time=snapshot_times[-1],
        bucket_label="",
        side="YES",
        entry_price=float("nan"),
        signal_name="momentum_threshold",
        signal_value=float("nan"),
        no_signal=True,
    )


def _resolved_correctly(day_df: pd.DataFrame, bucket_label: str) -> bool:
    entry_rows = day_df[day_df["bucket_label"].astype(str) == str(bucket_label)]
    if entry_rows.empty:
        raise ValueError(f"bucket_label {bucket_label} not found in day_df")
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved_values) != 1:
        raise ValueError(f"bucket_label {bucket_label} has inconsistent resolution")
    return bool(resolved_values[0])


def _evaluate_momentum_fast(
    partition_df: pd.DataFrame,
    d_star: float,
    w: int,
    order_type: str,
    contracts: float,
) -> pd.DataFrame:
    """Evaluate momentum threshold from one compact modal-row scan per day."""
    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    fee_fn = taker_fee if order_type == "taker" else maker_fee

    records: list[dict] = []
    for _, raw_day_df in df.groupby(group_cols, sort=True):
        city = (
            str(raw_day_df["city"].iloc[0])
            if "city" in raw_day_df.columns
            else str(raw_day_df[group_cols[0]].iloc[0])
        )
        event_date = str(raw_day_df["event_date"].dropna().iloc[0])
        day_df = filter_to_trading_window(raw_day_df)
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

        modal_idx = day_df.groupby("snapshot_time_local")["yes_mid_close"].idxmax()
        modal_rows = day_df.loc[modal_idx].sort_values("snapshot_time_local").copy()
        modal_rows["mode_prob"] = modal_rows["yes_mid_close"].astype(float)
        modal_rows["delta_mode"] = modal_rows["mode_prob"] - modal_rows["mode_prob"].shift(w)
        entry_rows = modal_rows[modal_rows["delta_mode"] >= d_star]
        if entry_rows.empty:
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

        entry_row = entry_rows.iloc[0]
        bucket_label = str(entry_row["bucket_label"])
        resolved_correctly = _resolved_correctly(day_df, bucket_label)
        entry_price = float(entry_row["yes_mid_close"])
        gross_pnl_cents = (
            (1.0 - entry_price) * 100.0 if resolved_correctly else -entry_price * 100.0
        )
        fee_cents = float(fee_fn(contracts, entry_price))
        records.append(
            {
                "event_date": event_date,
                "city": city,
                "entry_time": entry_row["snapshot_time_local"],
                "bucket_label": bucket_label,
                "side": "YES",
                "entry_price": entry_price,
                "signal_value": float(entry_row["delta_mode"]),
                "no_signal": False,
                "gross_pnl_cents": gross_pnl_cents,
                "fee_cents": fee_cents,
                "net_pnl_cents": gross_pnl_cents - fee_cents,
                "resolved_correctly": resolved_correctly,
            }
        )

    return pd.DataFrame.from_records(records)


def _day_summaries(partition_df: pd.DataFrame) -> list[dict]:
    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    summaries: list[dict] = []
    for _, raw_day_df in df.groupby(group_cols, sort=True):
        city = (
            str(raw_day_df["city"].iloc[0])
            if "city" in raw_day_df.columns
            else str(raw_day_df[group_cols[0]].iloc[0])
        )
        event_date = str(raw_day_df["event_date"].dropna().iloc[0])
        day_df = filter_to_trading_window(raw_day_df)
        if day_df.empty:
            summary = pd.DataFrame()
        else:
            modal_idx = day_df.groupby("snapshot_time_local")["yes_mid_close"].idxmax()
            summary = (
                day_df.loc[modal_idx, ["snapshot_time_local", "bucket_label", "yes_mid_close"]]
                .rename(columns={"yes_mid_close": "mode_prob"})
                .sort_values("snapshot_time_local")
                .reset_index(drop=True)
            )
        summaries.append({"event_date": event_date, "city": city, "summary": summary, "day_df": day_df})
    return summaries


def _evaluate_from_summaries(
    day_summaries: list[dict],
    d_star: float,
    w: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for item in day_summaries:
        event_date = item["event_date"]
        city = item["city"]
        summary = item["summary"]
        day_df = item["day_df"]
        no_signal_record = {
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
        if summary.empty:
            records.append(no_signal_record)
            continue
        working = summary.copy()
        working["mode_prob"] = working["mode_prob"].astype(float)
        working["delta_mode"] = working["mode_prob"] - working["mode_prob"].shift(w)
        entry_rows = working[working["delta_mode"] >= d_star]
        if entry_rows.empty:
            records.append(no_signal_record)
            continue
        entry_row = entry_rows.iloc[0]
        bucket_label = str(entry_row["bucket_label"])
        entry_price = float(entry_row["mode_prob"])
        resolved_correctly = _resolved_correctly(day_df, bucket_label)
        gross_pnl_cents = (
            (1.0 - entry_price) * 100.0 if resolved_correctly else -entry_price * 100.0
        )
        fee_cents = float(taker_fee(1.0, entry_price))
        records.append(
            {
                "event_date": event_date,
                "city": city,
                "entry_time": entry_row["snapshot_time_local"],
                "bucket_label": bucket_label,
                "side": "YES",
                "entry_price": entry_price,
                "signal_value": float(entry_row["delta_mode"]),
                "no_signal": False,
                "gross_pnl_cents": gross_pnl_cents,
                "fee_cents": fee_cents,
                "net_pnl_cents": gross_pnl_cents - fee_cents,
                "resolved_correctly": resolved_correctly,
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_momentum_threshold(
    partition_df: pd.DataFrame,
    d_star: float,
    w: int,
    order_type: str = "taker",
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Run momentum_signal on every (city, event_date). Same schema as implied_favorite."""
    if order_type not in {"taker", "maker"}:
        raise ValueError("order_type must be 'taker' or 'maker'")
    return _evaluate_momentum_fast(partition_df, d_star, w, order_type, contracts)


def _grid_row_stats(results_df: pd.DataFrame, d_star: float, w: int) -> dict:
    returns = daily_returns(results_df)
    stats = sharpe_stats(returns)
    n_trades, n_no_signal = _n_trades_and_no_signal(results_df)
    n_days = n_trades + n_no_signal
    no_signal_pct = 100.0 * n_no_signal / n_days if n_days else float("nan")
    return {
        "d_star": d_star,
        "w": w,
        "sharpe": stats["sharpe_annual"],
        "n_trades": n_trades,
        "no_signal_pct": no_signal_pct,
        "sharpe_se": stats["sharpe_se"],
        "sharpe_ci_low": stats["sharpe_ci_low"],
        "sharpe_ci_high": stats["sharpe_ci_high"],
    }


def _print_sharpe_heatmap(grid_df: pd.DataFrame) -> None:
    pivot = grid_df.pivot(index="w", columns="d_star", values="sharpe")
    d_cols = [f"{c:0.2f}" for c in pivot.columns]
    header = "w\\d  " + "  ".join(f"{c:>6s}" for c in d_cols)
    print(header)
    for w_val, row in pivot.iterrows():
        cells = "  ".join(
            f"{v:6.3f}" if pd.notna(v) else "   nan" for v in row.values
        )
        print(f"{int(w_val):>3d}  {cells}")


def optimise_momentum(
    partition_df: pd.DataFrame,
    d_grid: list[float] | None = None,
    w_grid: list[int] | None = None,
) -> dict:
    """Joint grid search over d_star and w; save momentum_grid.parquet."""
    assert_no_true_holdout(partition_df)
    if d_grid is None:
        d_grid = np.arange(0.02, 0.22, 0.02).tolist()
    if w_grid is None:
        w_grid = [2, 4, 6, 8, 12, 16, 24]

    day_summaries = _day_summaries(partition_df)
    rows = []
    for w in w_grid:
        for d in d_grid:
            results = _evaluate_from_summaries(day_summaries, d_star=float(d), w=int(w))
            rows.append(_grid_row_stats(results, float(d), int(w)))

    grid_df = pd.DataFrame(rows)
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    grid_df.to_parquet(OPT_DIR / "momentum_grid.parquet", index=False)

    _print_sharpe_heatmap(grid_df)

    sortable = grid_df.dropna(subset=["sharpe"])
    if sortable.empty:
        raise ValueError("No finite Sharpe values produced for momentum grid")
    best = sortable.sort_values(
        ["sharpe", "d_star", "w"], ascending=[False, True, True]
    ).iloc[0]
    best_dict = {
        "d_star": float(best["d_star"]),
        "w_star": int(best["w"]),
        "sharpe": float(best["sharpe"]),
        "n_trades": int(best["n_trades"]),
        "no_signal_pct": float(best["no_signal_pct"]),
        "sharpe_se": float(best["sharpe_se"]),
        "sharpe_ci_low": float(best["sharpe_ci_low"]),
        "sharpe_ci_high": float(best["sharpe_ci_high"]),
    }
    save_frozen_params({"d_star": best_dict["d_star"], "w_star": best_dict["w_star"]})
    return best_dict


if __name__ == "__main__":
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    best = optimise_momentum(threshold_opt)
    print(f"\nFROZEN d_star = {best['d_star']:0.2f}, w_star = {best['w_star']}")
    save_frozen_params({"d_star": best["d_star"], "w_star": best["w_star"]})

    frozen = load_frozen_params()
    results = evaluate_momentum_threshold(
        threshold_opt,
        d_star=float(frozen["d_star"]),
        w=int(frozen["w_star"]),
    )
    print_summary_table("momentum_threshold", results)
