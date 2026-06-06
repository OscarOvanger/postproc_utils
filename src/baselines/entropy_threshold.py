"""Entropy threshold baseline: enter when distribution entropy <= h_star."""

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


def compute_entropy(snapshot_prices: pd.Series) -> float:
    """
    Shannon entropy of yes_mid_close values normalised to sum to 1.

    Return 0.0 if only one bucket has non-zero price.
    """
    prices = pd.to_numeric(snapshot_prices, errors="coerce").fillna(0.0).astype(float)
    positive = prices[prices > 0]
    if positive.shape[0] <= 1:
        return 0.0
    probs = positive / positive.sum()
    probs = probs.clip(lower=1e-10, upper=1.0)
    return float(-(probs * np.log(probs)).sum())


@make_entry_rule
def entropy_signal(day_df: pd.DataFrame, h_star: float) -> TradeSignal:
    """
    Return TradeSignal at first snapshot where H <= h_star.

    Enter on the modal bucket at that snapshot.
    signal_name: "entropy_threshold"
    signal_value: entropy H at entry snapshot
    """
    sorted_day = day_df.sort_values("snapshot_time_local").copy()
    sorted_day["snapshot_time_local"] = pd.to_datetime(sorted_day["snapshot_time_local"])
    event_date = _event_date_value(sorted_day)
    snapshot_times = list(sorted_day["snapshot_time_local"].dropna().drop_duplicates())

    for snapshot_time in snapshot_times:
        snapshot = sorted_day[sorted_day["snapshot_time_local"] == snapshot_time]
        entropy = compute_entropy(snapshot["yes_mid_close"])
        if entropy <= h_star:
            modal_row = snapshot.loc[snapshot["yes_mid_close"].astype(float).idxmax()]
            return TradeSignal(
                event_date=event_date,
                entry_snapshot_time=snapshot_time,
                bucket_label=str(modal_row["bucket_label"]),
                side="YES",
                entry_price=float(modal_row["yes_mid_close"]),
                signal_name="entropy_threshold",
                signal_value=entropy,
                no_signal=False,
            )

    return TradeSignal(
        event_date=event_date,
        entry_snapshot_time=snapshot_times[-1],
        bucket_label="",
        side="YES",
        entry_price=float("nan"),
        signal_name="entropy_threshold",
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


def _evaluate_entropy_fast(
    partition_df: pd.DataFrame,
    h_star: float,
    order_type: str,
    contracts: float,
) -> pd.DataFrame:
    """Evaluate entropy threshold from one compact snapshot summary per day."""
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

        summary = _entropy_summary(day_df)
        entry_rows = summary[summary["entropy"] <= h_star]
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
        entry_price = float(entry_row["entry_price"])
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
                "signal_value": float(entry_row["entropy"]),
                "no_signal": False,
                "gross_pnl_cents": gross_pnl_cents,
                "fee_cents": fee_cents,
                "net_pnl_cents": gross_pnl_cents - fee_cents,
                "resolved_correctly": resolved_correctly,
            }
        )

    return pd.DataFrame.from_records(records)


def _entropy_summary(day_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per snapshot with modal bucket, modal price, and entropy."""
    working = day_df.copy()
    working["yes_mid_close"] = pd.to_numeric(
        working["yes_mid_close"], errors="coerce"
    ).fillna(0.0)
    modal_idx = working.groupby("snapshot_time_local")["yes_mid_close"].idxmax()
    summary = (
        working.loc[modal_idx, ["snapshot_time_local", "bucket_label", "yes_mid_close"]]
        .rename(columns={"yes_mid_close": "entry_price"})
        .sort_values("snapshot_time_local")
        .reset_index(drop=True)
    )

    positive = working[working["yes_mid_close"] > 0].copy()
    if positive.empty:
        summary["entropy"] = 0.0
        return summary
    sums = positive.groupby("snapshot_time_local")["yes_mid_close"].transform("sum")
    positive["_p"] = (positive["yes_mid_close"] / sums).clip(lower=1e-10, upper=1.0)
    positive["_entropy_term"] = -(positive["_p"] * np.log(positive["_p"]))
    entropy = positive.groupby("snapshot_time_local")["_entropy_term"].sum()
    nonzero_counts = positive.groupby("snapshot_time_local")["yes_mid_close"].size()
    entropy = entropy.where(nonzero_counts > 1, 0.0)
    summary = summary.merge(
        entropy.rename("entropy"),
        left_on="snapshot_time_local",
        right_index=True,
        how="left",
    )
    summary["entropy"] = summary["entropy"].fillna(0.0)
    return summary


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
        summary = _entropy_summary(day_df) if not day_df.empty else pd.DataFrame()
        summaries.append({"event_date": event_date, "city": city, "summary": summary, "day_df": day_df})
    return summaries


def _evaluate_from_summaries(day_summaries: list[dict], h_star: float) -> pd.DataFrame:
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
        entry_rows = summary[summary["entropy"] <= h_star]
        if entry_rows.empty:
            records.append(no_signal_record)
            continue
        entry_row = entry_rows.iloc[0]
        bucket_label = str(entry_row["bucket_label"])
        entry_price = float(entry_row["entry_price"])
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
                "signal_value": float(entry_row["entropy"]),
                "no_signal": False,
                "gross_pnl_cents": gross_pnl_cents,
                "fee_cents": fee_cents,
                "net_pnl_cents": gross_pnl_cents - fee_cents,
                "resolved_correctly": resolved_correctly,
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_entropy_threshold(
    partition_df: pd.DataFrame,
    h_star: float,
    order_type: str = "taker",
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Run entropy_signal on every (city, event_date). Same schema as implied_favorite."""
    if order_type not in {"taker", "maker"}:
        raise ValueError("order_type must be 'taker' or 'maker'")
    return _evaluate_entropy_fast(partition_df, h_star, order_type, contracts)


def _grid_row_stats(results_df: pd.DataFrame, h_star: float) -> dict:
    returns = daily_returns(results_df)
    stats = sharpe_stats(returns)
    n_trades, n_no_signal = _n_trades_and_no_signal(results_df)
    n_days = n_trades + n_no_signal
    no_signal_pct = 100.0 * n_no_signal / n_days if n_days else float("nan")
    return {
        "h_star": h_star,
        "sharpe": stats["sharpe_annual"],
        "n_trades": n_trades,
        "no_signal_pct": no_signal_pct,
        "sharpe_se": stats["sharpe_se"],
        "sharpe_ci_low": stats["sharpe_ci_low"],
        "sharpe_ci_high": stats["sharpe_ci_high"],
    }


def optimise_h_star(
    partition_df: pd.DataFrame,
    h_grid: list[float] | None = None,
) -> dict:
    """Grid search over h_grid; save entropy_h_grid.parquet; return best row."""
    assert_no_true_holdout(partition_df)
    if h_grid is None:
        h_grid = np.arange(0.30, 1.85, 0.05).tolist()

    day_summaries = _day_summaries(partition_df)
    rows = []
    for h in h_grid:
        results = _evaluate_from_summaries(day_summaries, h_star=float(h))
        rows.append(_grid_row_stats(results, float(h)))

    grid_df = pd.DataFrame(rows)
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    grid_df.to_parquet(OPT_DIR / "entropy_h_grid.parquet", index=False)

    print(grid_df[["h_star", "sharpe", "n_trades", "no_signal_pct"]].to_string(
        index=False, float_format=lambda v: f"{v:0.4f}"
    ))

    sortable = grid_df.dropna(subset=["sharpe"])
    if sortable.empty:
        raise ValueError("No finite Sharpe values produced for h_grid")
    best = sortable.sort_values(["sharpe", "h_star"], ascending=[False, True]).iloc[0]
    best_dict = {
        "h_star": float(best["h_star"]),
        "sharpe": float(best["sharpe"]),
        "n_trades": int(best["n_trades"]),
        "no_signal_pct": float(best["no_signal_pct"]),
        "sharpe_se": float(best["sharpe_se"]),
        "sharpe_ci_low": float(best["sharpe_ci_low"]),
        "sharpe_ci_high": float(best["sharpe_ci_high"]),
    }
    save_frozen_params({"h_star": best_dict["h_star"]})
    return best_dict


if __name__ == "__main__":
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    best = optimise_h_star(threshold_opt)
    print(f"\nFROZEN h_star = {best['h_star']:0.2f}")
    save_frozen_params({"h_star": best["h_star"]})

    frozen = load_frozen_params()
    h_star = float(frozen["h_star"])
    results = evaluate_entropy_threshold(threshold_opt, h_star=h_star)
    print_summary_table("entropy_threshold", results)
