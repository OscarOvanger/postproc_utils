"""Implied-favorite baseline: buy YES on the stabilized modal bucket."""

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
from snapshot_stability import (  # noqa: E402
    SPLIT_DIR,
    load_or_create_frozen_k,
    stability_entry,
)
from backtest_utils import daily_returns, sharpe_stats  # noqa: E402


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    """Return columns that uniquely identify one city-date trading day."""
    city_col = (
        "source_city_folder"
        if "source_city_folder" in partition_df.columns
        else "city"
    )
    return [city_col, "event_date"]


@make_entry_rule
def implied_favorite_signal(day_df: pd.DataFrame, k: int) -> TradeSignal:
    """
    Entry rule for the implied-favorite baseline.

    Uses :func:`stability_entry` to determine the entry snapshot. If
    ``stability_entry`` returns ``no_signal=True``, this function also returns a
    ``no_signal=True`` signal.

    On a valid entry the returned :class:`TradeSignal` carries the modal bucket
    at the entry snapshot, ``side="YES"``, ``entry_price`` equal to the modal
    bucket's ``yes_mid_close`` at the entry snapshot, ``signal_name`` set to
    ``"implied_favorite"``, and ``signal_value`` equal to that same
    ``yes_mid_close`` (the mode probability).
    """
    signal = stability_entry(day_df, k=k)
    if signal.no_signal:
        return TradeSignal(
            event_date=signal.event_date,
            entry_snapshot_time=signal.entry_snapshot_time,
            bucket_label=signal.bucket_label,
            side="YES",
            entry_price=signal.entry_price,
            signal_name="implied_favorite",
            signal_value=signal.signal_value,
            no_signal=True,
        )

    return TradeSignal(
        event_date=signal.event_date,
        entry_snapshot_time=signal.entry_snapshot_time,
        bucket_label=signal.bucket_label,
        side="YES",
        entry_price=signal.entry_price,
        signal_name="implied_favorite",
        signal_value=signal.entry_price,
        no_signal=False,
    )


def _resolved_correctly(day_df: pd.DataFrame, bucket_label: str) -> bool:
    """Return True when the given bucket resolved to one dollar (YES won)."""
    entry_rows = day_df[day_df["bucket_label"].astype(str) == str(bucket_label)]
    if entry_rows.empty:
        raise ValueError(f"bucket_label {bucket_label} not found in day_df")
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved_values) != 1:
        raise ValueError(f"bucket_label {bucket_label} has inconsistent resolution")
    return bool(resolved_values[0])


def evaluate_implied_favorite(
    partition_df: pd.DataFrame,
    k: int,
    order_type: str = "taker",
    contracts: float = 1.0,
) -> pd.DataFrame:
    """
    Run :func:`implied_favorite_signal` on every (city, event_date) day.

    Returns a results dataframe with one row per trading day and columns:
    ``event_date``, ``city``, ``entry_time``, ``bucket_label``, ``side``,
    ``entry_price``, ``signal_value``, ``no_signal``, ``gross_pnl_cents``,
    ``fee_cents``, ``net_pnl_cents``, ``resolved_correctly``.

    PnL is per contract in cents. For a placed trade, gross PnL is
    ``(1 - entry_price) * 100`` when the entered bucket resolved to one dollar
    and ``-entry_price * 100`` otherwise. The fee is the taker or maker fee on
    ``entry_price`` (per ``order_type``), and net PnL is gross minus fee. On a
    no-signal day every PnL column is NaN and ``no_signal`` is True.
    """
    if order_type not in {"taker", "maker"}:
        raise ValueError("order_type must be 'taker' or 'maker'")

    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    fee_fn = taker_fee if order_type == "taker" else maker_fee

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
        signal = implied_favorite_signal(day_df, k=k)

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
        if resolved_correctly:
            gross_pnl_cents = (1.0 - entry_price) * 100.0
        else:
            gross_pnl_cents = -entry_price * 100.0
        fee_cents = float(fee_fn(contracts, entry_price))
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
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    frozen_k = load_or_create_frozen_k()
    results = evaluate_implied_favorite(threshold_opt, k=frozen_k)

    n_days = len(results)
    n_no_signal = int(results["no_signal"].sum())
    n_trades = n_days - n_no_signal
    mean_net_pnl = float(results["net_pnl_cents"].mean())
    stats = sharpe_stats(daily_returns(results))

    print(f"implied_favorite smoke test (frozen k = {frozen_k})")
    print(f"  N days        : {n_days}")
    print(f"  N trades      : {n_trades}")
    print(f"  N no-signal   : {n_no_signal}")
    print(f"  mean net PnL  : {mean_net_pnl:0.4f} cents")
    print(f"  Sharpe annual : {stats['sharpe_annual']:0.4f}")
