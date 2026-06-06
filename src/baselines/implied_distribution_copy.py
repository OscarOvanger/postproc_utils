"""Implied-distribution-copy baseline: bet every bucket in market proportion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entry_interface import TradeSignal, filter_to_trading_window, make_entry_rule  # noqa: E402
from fees import taker_fee  # noqa: E402
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
def _single_bucket_signal(
    day_df: pd.DataFrame,
    entry_snapshot_time: pd.Timestamp,
    event_date: str,
    bucket_label: str,
    entry_price: float,
    weight: float,
) -> TradeSignal:
    """Build one validated YES :class:`TradeSignal` for a single bucket leg."""
    return TradeSignal(
        event_date=event_date,
        entry_snapshot_time=entry_snapshot_time,
        bucket_label=bucket_label,
        side="YES",
        entry_price=float(entry_price),
        signal_name="implied_distribution_copy",
        signal_value=float(weight),
        no_signal=False,
    )


def distribution_copy_signal(day_df: pd.DataFrame, k: int) -> list[TradeSignal]:
    """
    Entry rule for the implied-distribution-copy baseline.

    Uses :func:`stability_entry` to find the entry snapshot (the same snapshot
    as implied-favorite, ensuring comparability). If there is no stability
    signal, returns a single :class:`TradeSignal` with ``no_signal=True``.

    On a valid entry, returns one ``TradeSignal`` per bucket whose fee-adjusted
    price ``yes_mid_close - taker_fee(1, yes_mid_close) / 100`` is strictly
    positive. Each leg's ``signal_value`` is its proportional weight::

        weight_i = max(yes_mid_close_i - fee_i, 0)
                   / sum(max(yes_mid_close_j - fee_j, 0) for all j)

    The inner single-leg builder is decorated with :func:`make_entry_rule`;
    this dispatcher is not, because it emits multiple simultaneous trades.
    """
    entry = stability_entry(day_df, k=k)
    if entry.no_signal:
        return [
            TradeSignal(
                event_date=entry.event_date,
                entry_snapshot_time=entry.entry_snapshot_time,
                bucket_label=entry.bucket_label,
                side="YES",
                entry_price=entry.entry_price,
                signal_name="implied_distribution_copy",
                signal_value=entry.signal_value,
                no_signal=True,
            )
        ]

    day = day_df.copy()
    day["snapshot_time_local"] = pd.to_datetime(day["snapshot_time_local"])
    snapshot = day[day["snapshot_time_local"] == entry.entry_snapshot_time].copy()

    prices = snapshot["yes_mid_close"].astype(float)
    fees = prices.apply(lambda price: taker_fee(1, price) / 100.0)
    adjusted = (prices - fees).clip(lower=0.0)
    total = float(adjusted.sum())

    signals: list[TradeSignal] = []
    if total <= 0:
        return [
            TradeSignal(
                event_date=entry.event_date,
                entry_snapshot_time=entry.entry_snapshot_time,
                bucket_label=entry.bucket_label,
                side="YES",
                entry_price=entry.entry_price,
                signal_name="implied_distribution_copy",
                signal_value=entry.signal_value,
                no_signal=True,
            )
        ]

    for idx, adj in adjusted.items():
        if adj <= 0:
            continue
        weight = float(adj) / total
        signals.append(
            _single_bucket_signal(
                day,
                entry_snapshot_time=entry.entry_snapshot_time,
                event_date=entry.event_date,
                bucket_label=str(snapshot.loc[idx, "bucket_label"]),
                entry_price=float(prices.loc[idx]),
                weight=weight,
            )
        )

    return signals


def evaluate_distribution_copy(
    partition_df: pd.DataFrame,
    k: int,
    total_outlay_cents: float = 100.0,
) -> pd.DataFrame:
    """
    Run :func:`distribution_copy_signal` on every (city, event_date) day.

    The ``total_outlay_cents`` budget is allocated across bucket legs in
    proportion to each leg's weight (``signal_value``)::

        contracts_i = (weight_i * total_outlay_cents) / (entry_price_i * 100)

    Returns a results dataframe with one row per trading day and columns:
    ``event_date``, ``city``, ``entry_time``, ``no_signal``,
    ``n_buckets_traded``, ``total_outlay_cents``, ``total_fee_cents``,
    ``gross_pnl_cents``, ``net_pnl_cents``, ``resolved_bucket_label``,
    ``resolved_bucket_weight``.

    Gross PnL is the winnings from the resolved bucket minus the cost of all
    other (losing) buckets; net PnL subtracts the summed taker fees. The
    ``resolved_bucket_weight`` is the weight assigned to the winning bucket (0
    when that bucket received no allocation); under market efficiency it should
    approximate the winning bucket's entry price.
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
                    "no_signal": True,
                    "n_buckets_traded": 0,
                    "total_outlay_cents": np.nan,
                    "total_fee_cents": np.nan,
                    "gross_pnl_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "resolved_bucket_label": "",
                    "resolved_bucket_weight": np.nan,
                }
            )
            continue
        signals = distribution_copy_signal(day_df, k=k)

        if len(signals) == 1 and signals[0].no_signal:
            records.append(
                {
                    "event_date": signals[0].event_date,
                    "city": city,
                    "entry_time": pd.NaT,
                    "no_signal": True,
                    "n_buckets_traded": 0,
                    "total_outlay_cents": np.nan,
                    "total_fee_cents": np.nan,
                    "gross_pnl_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "resolved_bucket_label": "",
                    "resolved_bucket_weight": np.nan,
                }
            )
            continue

        entry_time = signals[0].entry_snapshot_time
        event_date = signals[0].event_date

        resolved_lookup = (
            day_df.drop_duplicates("bucket_label")
            .set_index(day_df.drop_duplicates("bucket_label")["bucket_label"].astype(str))[
                "bucket_resolved_to_one_dollars"
            ]
            .astype(bool)
        )

        gross_pnl_cents = 0.0
        total_fee_cents = 0.0
        resolved_bucket_label = ""
        resolved_bucket_weight = 0.0

        for leg in signals:
            weight = float(leg.signal_value)
            price = float(leg.entry_price)
            contracts = (weight * total_outlay_cents) / (price * 100.0)
            total_fee_cents += float(taker_fee(contracts, price))

            won = bool(resolved_lookup.get(str(leg.bucket_label), False))
            if won:
                gross_pnl_cents += contracts * (1.0 - price) * 100.0
                resolved_bucket_label = str(leg.bucket_label)
                resolved_bucket_weight = weight
            else:
                gross_pnl_cents += -contracts * price * 100.0

        if not resolved_bucket_label:
            won_rows = resolved_lookup[resolved_lookup]
            if not won_rows.empty:
                resolved_bucket_label = str(won_rows.index[0])
                resolved_bucket_weight = 0.0

        net_pnl_cents = gross_pnl_cents - total_fee_cents

        records.append(
            {
                "event_date": event_date,
                "city": city,
                "entry_time": entry_time,
                "no_signal": False,
                "n_buckets_traded": len(signals),
                "total_outlay_cents": float(total_outlay_cents),
                "total_fee_cents": total_fee_cents,
                "gross_pnl_cents": gross_pnl_cents,
                "net_pnl_cents": net_pnl_cents,
                "resolved_bucket_label": resolved_bucket_label,
                "resolved_bucket_weight": resolved_bucket_weight,
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    frozen_k = load_or_create_frozen_k()
    results = evaluate_distribution_copy(threshold_opt, k=frozen_k)

    n_days = len(results)
    n_no_signal = int(results["no_signal"].sum())
    n_trades = n_days - n_no_signal
    mean_net_pnl = float(results["net_pnl_cents"].mean())
    stats = sharpe_stats(daily_returns(results, capital=100.0))

    print(f"implied_distribution_copy smoke test (frozen k = {frozen_k})")
    print(f"  N days        : {n_days}")
    print(f"  N trades      : {n_trades}")
    print(f"  N no-signal   : {n_no_signal}")
    print(f"  mean net PnL  : {mean_net_pnl:0.4f} cents")
    print(f"  Sharpe annual : {stats['sharpe_annual']:0.4f}")
