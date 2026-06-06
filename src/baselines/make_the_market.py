"""Make-the-market baseline: quote one cent inside the spread on the modal bucket."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entry_interface import TradeSignal, filter_to_trading_window  # noqa: E402
from fees import maker_fee  # noqa: E402
from snapshot_stability import SPLIT_DIR, _event_date_value  # noqa: E402
from backtest_utils import print_summary_table  # noqa: E402


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    """Return columns that uniquely identify one city-date trading day."""
    city_col = (
        "source_city_folder"
        if "source_city_folder" in partition_df.columns
        else "city"
    )
    return [city_col, "event_date"]


def _scan_for_fill(day_df: pd.DataFrame) -> tuple[TradeSignal, int]:
    """Scan intraday snapshots; return (TradeSignal, n_requotes)."""
    sorted_day = day_df.sort_values("snapshot_time_local").copy()
    sorted_day["snapshot_time_local"] = pd.to_datetime(sorted_day["snapshot_time_local"])
    event_date = _event_date_value(sorted_day)

    n_requotes = 0
    previous_modal: str | None = None
    final_snapshot_time: pd.Timestamp | None = None

    for snapshot_time, snapshot in sorted_day.groupby(
        "snapshot_time_local", sort=True
    ):
        final_snapshot_time = pd.Timestamp(snapshot_time)
        modal_row = snapshot.loc[snapshot["yes_mid_close"].astype(float).idxmax()]
        modal_bucket = str(modal_row["bucket_label"])
        if previous_modal is not None and modal_bucket != previous_modal:
            n_requotes += 1
        previous_modal = modal_bucket

        yes_bid_close = float(modal_row["yes_bid_close"])
        quoted_bid = yes_bid_close + 0.01
        volume = float(modal_row.get("volume_contracts", 0) or 0)

        if volume > 0 and quoted_bid >= yes_bid_close:
            return (
                TradeSignal(
                    event_date=event_date,
                    entry_snapshot_time=final_snapshot_time,
                    bucket_label=modal_bucket,
                    side="YES",
                    entry_price=quoted_bid,
                    signal_name="make_the_market",
                    signal_value=quoted_bid,
                    no_signal=False,
                ),
                n_requotes,
            )

    if final_snapshot_time is None:
        raise ValueError("day_df must contain at least one snapshot_time_local")
    return (
        TradeSignal(
            event_date=event_date,
            entry_snapshot_time=final_snapshot_time,
            bucket_label="",
            side="YES",
            entry_price=float("nan"),
            signal_name="make_the_market",
            signal_value=float("nan"),
            no_signal=True,
        ),
        n_requotes,
    )


def make_the_market_signal(day_df: pd.DataFrame) -> TradeSignal:
    """
    Scan the full intraday snapshot series for the first maker fill.

    Returns TradeSignal with no_signal=True if no fill occurs.
    signal_name: "make_the_market"
    signal_value: entry_price at fill
    """
    signal, _ = _scan_for_fill(day_df)
    return signal


def _resolved_correctly(day_df: pd.DataFrame, bucket_label: str) -> bool:
    """Return True when the given bucket resolved to one dollar (YES won)."""
    entry_rows = day_df[day_df["bucket_label"].astype(str) == str(bucket_label)]
    if entry_rows.empty:
        raise ValueError(f"bucket_label {bucket_label} not found in day_df")
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved_values) != 1:
        raise ValueError(f"bucket_label {bucket_label} has inconsistent resolution")
    return bool(resolved_values[0])


def evaluate_make_the_market(
    partition_df: pd.DataFrame,
    contracts: float = 1.0,
) -> pd.DataFrame:
    """
    Run make_the_market_signal on every (city, event_date).

    Returns results dataframe with same schema as evaluate_implied_favorite
    plus fill_snapshot_time and n_requotes columns.
    """
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
                    "fill_snapshot_time": pd.NaT,
                    "n_requotes": 0,
                }
            )
            continue
        signal, n_requotes = _scan_for_fill(day_df)

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
                    "fill_snapshot_time": pd.NaT,
                    "n_requotes": n_requotes,
                }
            )
            continue

        resolved_correctly = _resolved_correctly(day_df, signal.bucket_label)
        entry_price = float(signal.entry_price)
        if resolved_correctly:
            gross_pnl_cents = (1.0 - entry_price) * 100.0
        else:
            gross_pnl_cents = -entry_price * 100.0
        fee_cents = float(maker_fee(contracts, entry_price))
        net_pnl_cents = gross_pnl_cents - fee_cents

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
                "net_pnl_cents": net_pnl_cents,
                "resolved_correctly": resolved_correctly,
                "fill_snapshot_time": signal.entry_snapshot_time,
                "n_requotes": n_requotes,
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    results = evaluate_make_the_market(threshold_opt)
    print_summary_table("make_the_market", results)
    mean_requotes = float(pd.to_numeric(results["n_requotes"], errors="coerce").mean())
    fill_rate = 100.0 * float((~results["no_signal"].fillna(False)).mean())
    print(f"Mean n_requotes/day: {mean_requotes:0.2f}")
    print(f"Fill rate: {fill_rate:0.1f}%")
