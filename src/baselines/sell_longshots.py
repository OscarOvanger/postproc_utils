"""Sell-longshots baseline: fade buckets that cross below a low YES price."""

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


def find_longshot_crossings(
    day_df: pd.DataFrame,
    k: int,
    price_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Find every bucket's first downward crossing of ``price_threshold``.

    A crossing occurs at snapshot ``t`` when ``yes_mid_close`` is strictly below
    ``price_threshold`` at ``t`` and was at or above ``price_threshold`` at the
    immediately preceding snapshot ``t-1`` (a strict first cross, not merely any
    snapshot below the threshold). Only crossings at or after the snapshot-
    stability entry snapshot (:func:`stability_entry`) qualify, and per bucket
    only the earliest qualifying crossing is kept.

    If ``stability_entry`` returns ``no_signal=True`` for the day, an empty
    dataframe is returned (no trades that day). Otherwise the result has one row
    per qualifying first crossing with columns: ``bucket_label``,
    ``crossing_snapshot_time``, ``crossing_price``, ``stability_entry_time``,
    ``minutes_after_stability_entry``, ``bucket_resolved_to_one_dollars``.
    """
    columns = [
        "bucket_label",
        "crossing_snapshot_time",
        "crossing_price",
        "stability_entry_time",
        "minutes_after_stability_entry",
        "bucket_resolved_to_one_dollars",
    ]

    entry = stability_entry(day_df, k=k)
    if entry.no_signal:
        return pd.DataFrame(columns=columns)

    entry_time = entry.entry_snapshot_time
    day = day_df.copy()
    day["snapshot_time_local"] = pd.to_datetime(day["snapshot_time_local"])

    records: list[dict] = []
    for bucket_label, bucket_df in day.groupby("bucket_label", sort=False):
        series = bucket_df.sort_values("snapshot_time_local")
        prices = series["yes_mid_close"].astype(float).to_numpy()
        times = series["snapshot_time_local"].to_numpy()
        resolved = bool(
            series["bucket_resolved_to_one_dollars"].astype(bool).iloc[0]
        )

        for i in range(1, len(prices)):
            if prices[i] < price_threshold and prices[i - 1] >= price_threshold:
                crossing_time = pd.Timestamp(times[i])
                if crossing_time < entry_time:
                    continue
                minutes_after = (
                    crossing_time - entry_time
                ).total_seconds() / 60.0
                records.append(
                    {
                        "bucket_label": str(bucket_label),
                        "crossing_snapshot_time": crossing_time,
                        "crossing_price": float(prices[i]),
                        "stability_entry_time": entry_time,
                        "minutes_after_stability_entry": float(minutes_after),
                        "bucket_resolved_to_one_dollars": resolved,
                    }
                )
                break

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]


@make_entry_rule
def _single_longshot_signal(
    day_df: pd.DataFrame,
    event_date: str,
    entry_snapshot_time: pd.Timestamp,
    bucket_label: str,
    entry_price: float,
    yes_price: float,
) -> TradeSignal:
    """Build one validated NO :class:`TradeSignal` for a longshot crossing."""
    return TradeSignal(
        event_date=event_date,
        entry_snapshot_time=entry_snapshot_time,
        bucket_label=bucket_label,
        side="NO",
        entry_price=float(entry_price),
        signal_name="sell_longshots",
        signal_value=float(yes_price),
        no_signal=False,
    )


def sell_longshots_signals(
    day_df: pd.DataFrame,
    k: int,
    price_threshold: float = 0.10,
) -> list[TradeSignal]:
    """
    Convert :func:`find_longshot_crossings` output into trade signals.

    For each crossing the signal sells YES by buying NO: ``side="NO"``,
    ``entry_price = 1 - yes_mid_close`` at the crossing snapshot (the NO cost),
    ``signal_name="sell_longshots"``, and ``signal_value`` equal to the
    ``yes_mid_close`` being faded. If there are no crossings, a single
    :class:`TradeSignal` with ``no_signal=True`` is returned.
    """
    crossings = find_longshot_crossings(day_df, k=k, price_threshold=price_threshold)
    event_date = str(day_df["event_date"].dropna().iloc[0])

    if crossings.empty:
        final_snapshot = pd.to_datetime(day_df["snapshot_time_local"]).dropna().max()
        return [
            TradeSignal(
                event_date=event_date,
                entry_snapshot_time=final_snapshot,
                bucket_label="",
                side="NO",
                entry_price=float("nan"),
                signal_name="sell_longshots",
                signal_value=float("nan"),
                no_signal=True,
            )
        ]

    signals: list[TradeSignal] = []
    for _, crossing in crossings.iterrows():
        yes_price = float(crossing["crossing_price"])
        no_price = 1.0 - yes_price
        signals.append(
            _single_longshot_signal(
                day_df,
                event_date=event_date,
                entry_snapshot_time=crossing["crossing_snapshot_time"],
                bucket_label=str(crossing["bucket_label"]),
                entry_price=no_price,
                yes_price=yes_price,
            )
        )
    return signals


def evaluate_sell_longshots(
    partition_df: pd.DataFrame,
    k: int,
    price_threshold: float = 0.10,
    order_type: str = "taker",
    contracts: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run :func:`sell_longshots_signals` on every (city, event_date) day.

    Returns a ``(trade_level, day_level)`` tuple of dataframes.

    The trade-level frame has one row per (city, event_date, bucket_label)
    longshot trade (or one no-signal row for days without crossings) with
    columns: ``event_date``, ``city``, ``bucket_label``,
    ``crossing_snapshot_time``, ``crossing_price``, ``side``, ``entry_price``,
    ``no_signal``, ``gross_pnl_cents``, ``fee_cents``, ``net_pnl_cents``,
    ``resolved_correctly``. We buy NO at ``entry_price``; the trade resolves
    correctly when ``bucket_resolved_to_one_dollars`` is False (the YES we sold
    did not win). Gross PnL is ``(1 - entry_price) * 100`` when correct and
    ``-entry_price * 100`` otherwise; the fee is on ``entry_price``.

    The day-level frame aggregates per (city, event_date) with columns:
    ``event_date``, ``city``, ``n_longshots_sold``, ``total_fee_cents``,
    ``gross_pnl_cents``, ``net_pnl_cents``, ``any_resolved_against_us``.
    """
    if order_type not in {"taker", "maker"}:
        raise ValueError("order_type must be 'taker' or 'maker'")

    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    fee_fn = taker_fee if order_type == "taker" else maker_fee

    trade_records: list[dict] = []
    day_records: list[dict] = []

    for _, day_df in df.groupby(group_cols, sort=True):
        city = (
            str(day_df["city"].iloc[0])
            if "city" in day_df.columns
            else str(day_df[group_cols[0]].iloc[0])
        )
        event_date = str(day_df["event_date"].dropna().iloc[0])
        day_df = filter_to_trading_window(day_df)

        if day_df.empty:
            trade_records.append(
                {
                    "event_date": event_date,
                    "city": city,
                    "bucket_label": "",
                    "crossing_snapshot_time": pd.NaT,
                    "crossing_price": np.nan,
                    "side": "NO",
                    "entry_price": np.nan,
                    "no_signal": True,
                    "gross_pnl_cents": np.nan,
                    "fee_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "resolved_correctly": np.nan,
                }
            )
            day_records.append(
                {
                    "event_date": event_date,
                    "city": city,
                    "n_longshots_sold": 0,
                    "total_fee_cents": np.nan,
                    "gross_pnl_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "any_resolved_against_us": False,
                }
            )
            continue

        crossings = find_longshot_crossings(
            day_df, k=k, price_threshold=price_threshold
        )
        resolved_lookup = (
            day_df.drop_duplicates("bucket_label")
            .assign(_bl=lambda frame: frame["bucket_label"].astype(str))
            .set_index("_bl")["bucket_resolved_to_one_dollars"]
            .astype(bool)
        )

        if crossings.empty:
            trade_records.append(
                {
                    "event_date": event_date,
                    "city": city,
                    "bucket_label": "",
                    "crossing_snapshot_time": pd.NaT,
                    "crossing_price": np.nan,
                    "side": "NO",
                    "entry_price": np.nan,
                    "no_signal": True,
                    "gross_pnl_cents": np.nan,
                    "fee_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "resolved_correctly": np.nan,
                }
            )
            day_records.append(
                {
                    "event_date": event_date,
                    "city": city,
                    "n_longshots_sold": 0,
                    "total_fee_cents": np.nan,
                    "gross_pnl_cents": np.nan,
                    "net_pnl_cents": np.nan,
                    "any_resolved_against_us": False,
                }
            )
            continue

        day_gross = 0.0
        day_fee = 0.0
        day_net = 0.0
        any_against = False

        for _, crossing in crossings.iterrows():
            bucket_label = str(crossing["bucket_label"])
            yes_price = float(crossing["crossing_price"])
            entry_price = 1.0 - yes_price
            yes_won = bool(resolved_lookup.get(bucket_label, False))
            resolved_correctly = not yes_won

            if resolved_correctly:
                gross_pnl_cents = (1.0 - entry_price) * 100.0
            else:
                gross_pnl_cents = -entry_price * 100.0
                any_against = True

            fee_cents = float(fee_fn(contracts, entry_price))
            net_pnl_cents = gross_pnl_cents - fee_cents

            day_gross += gross_pnl_cents
            day_fee += fee_cents
            day_net += net_pnl_cents

            trade_records.append(
                {
                    "event_date": event_date,
                    "city": city,
                    "bucket_label": bucket_label,
                    "crossing_snapshot_time": crossing["crossing_snapshot_time"],
                    "crossing_price": yes_price,
                    "side": "NO",
                    "entry_price": entry_price,
                    "no_signal": False,
                    "gross_pnl_cents": gross_pnl_cents,
                    "fee_cents": fee_cents,
                    "net_pnl_cents": net_pnl_cents,
                    "resolved_correctly": resolved_correctly,
                }
            )

        day_records.append(
            {
                "event_date": event_date,
                "city": city,
                "n_longshots_sold": int(len(crossings)),
                "total_fee_cents": day_fee,
                "gross_pnl_cents": day_gross,
                "net_pnl_cents": day_net,
                "any_resolved_against_us": bool(any_against),
            }
        )

    return pd.DataFrame.from_records(trade_records), pd.DataFrame.from_records(
        day_records
    )


if __name__ == "__main__":
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    frozen_k = load_or_create_frozen_k()
    trades, day_summary = evaluate_sell_longshots(threshold_opt, k=frozen_k)

    n_days = len(day_summary)
    n_no_trade_days = int((day_summary["n_longshots_sold"] == 0).sum())
    n_trade_days = n_days - n_no_trade_days
    mean_net_pnl = float(trades["net_pnl_cents"].mean())
    stats = sharpe_stats(daily_returns(trades, capital=100.0))

    crossings_per_day = day_summary["n_longshots_sold"]

    print(f"sell_longshots smoke test (frozen k = {frozen_k})")
    print(f"  N days              : {n_days}")
    print(f"  N days with trades  : {n_trade_days}")
    print(f"  N no-trade days     : {n_no_trade_days}")
    print(f"  mean net PnL/trade  : {mean_net_pnl:0.4f} cents")
    print(f"  Sharpe annual       : {stats['sharpe_annual']:0.4f}")
    print(f"  mean crossings/day  : {float(crossings_per_day.mean()):0.4f}")

    all_minutes: list[float] = []
    df_all = threshold_opt.copy()
    df_all["snapshot_time_local"] = pd.to_datetime(df_all["snapshot_time_local"])
    group_cols = _day_group_columns(df_all)
    for _, day_df in df_all.groupby(group_cols, sort=True):
        crossings = find_longshot_crossings(day_df, k=frozen_k)
        if not crossings.empty:
            all_minutes.extend(crossings["minutes_after_stability_entry"].tolist())

    minutes_series = pd.Series(all_minutes, dtype=float)
    print("  crossing-time distribution (minutes after stability entry):")
    if minutes_series.empty:
        print("    (no crossings)")
    else:
        print(minutes_series.describe().to_string())
