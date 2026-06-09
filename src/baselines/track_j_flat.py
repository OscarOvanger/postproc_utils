"""Track-J flat baseline: buy YES on the highest Track-J probability bucket."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from entry_interface import TradeSignal, filter_to_trading_window, make_entry_rule  # noqa: E402
from fees import net_pnl, taker_fee  # noqa: E402
from snapshot_stability import compute_modal_bucket, stability_entry  # noqa: E402
from src.models.track_j import get_track_j_bucket_probs  # noqa: E402


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    city_col = "source_city_folder" if "source_city_folder" in partition_df.columns else "city"
    return [city_col, "event_date"]


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _date_key(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _resolved_correctly(day_df: pd.DataFrame, bucket_label: str) -> bool:
    entry_rows = day_df[day_df["bucket_label"].astype(str).eq(str(bucket_label))]
    if entry_rows.empty:
        raise ValueError(f"bucket_label {bucket_label} not found in day_df")
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved_values) != 1:
        raise ValueError(f"bucket_label {bucket_label} has inconsistent resolution")
    return bool(resolved_values[0])


def _forecast_lookup(
    city: str,
    event_date: str,
    forecasts_df: pd.DataFrame,
) -> tuple[float, bool]:
    forecasts = forecasts_df.copy()
    forecasts["city"] = forecasts["city"].map(_city_key)
    forecasts["event_date"] = pd.to_datetime(forecasts["event_date"]).dt.strftime("%Y-%m-%d")
    row = forecasts[
        forecasts["city"].eq(_city_key(city))
        & forecasts["event_date"].eq(_date_key(event_date))
    ]
    if row.empty:
        return np.nan, False
    row = row.iloc[0]
    return float(row["track_j_tmax_f"]) if pd.notna(row["track_j_tmax_f"]) else np.nan, bool(row.get("city_coverage_flag", False))


@make_entry_rule
def track_j_flat_signal(
    day_df: pd.DataFrame,
    k: int,
    forecasts_df: pd.DataFrame,
) -> TradeSignal:
    """
    Entry rule for Track-J flat sizing.

    Uses the stability entry snapshot, then buys YES on the bucket with the
    highest Track-J probability. Missing forecasts return no_signal=True and
    never fall back to implied-favorite.
    """
    signal = stability_entry(day_df, k=k)
    if signal.no_signal:
        return TradeSignal(
            event_date=signal.event_date,
            entry_snapshot_time=signal.entry_snapshot_time,
            bucket_label=signal.bucket_label,
            side="YES",
            entry_price=signal.entry_price,
            signal_name="track_j_flat",
            signal_value=np.nan,
            no_signal=True,
        )

    city = day_df["city"].iloc[0] if "city" in day_df.columns else day_df.get("source_city_folder", pd.Series([""])).iloc[0]
    snapshot = day_df[pd.to_datetime(day_df["snapshot_time_local"]).eq(signal.entry_snapshot_time)]
    probs = get_track_j_bucket_probs(city, signal.event_date, forecasts_df, snapshot)
    if probs is None:
        return TradeSignal(
            event_date=signal.event_date,
            entry_snapshot_time=signal.entry_snapshot_time,
            bucket_label="",
            side="YES",
            entry_price=np.nan,
            signal_name="track_j_flat",
            signal_value=np.nan,
            no_signal=True,
        )

    chosen_bucket = max(probs, key=probs.get)
    entry_rows = snapshot[snapshot["bucket_label"].astype(str).eq(str(chosen_bucket))]
    if entry_rows.empty:
        raise ValueError(f"Track-J selected bucket {chosen_bucket} not found at entry snapshot")
    entry_price = float(entry_rows["yes_mid_close"].iloc[0])
    return TradeSignal(
        event_date=signal.event_date,
        entry_snapshot_time=signal.entry_snapshot_time,
        bucket_label=str(chosen_bucket),
        side="YES",
        entry_price=entry_price,
        signal_name="track_j_flat",
        signal_value=float(probs[chosen_bucket]),
        no_signal=False,
    )


def evaluate_track_j_flat(
    partition_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    k: int,
    order_type: str = "taker",
    contracts: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate Track-J flat over each city-date in the partition.

    Missing Track-J forecasts remain explicit no_signal rows.
    """
    if order_type != "taker":
        raise ValueError("track_j_flat currently supports order_type='taker'")
    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    records: list[dict[str, object]] = []

    for _, raw_day_df in df.groupby(group_cols, sort=True):
        city = str(raw_day_df["city"].iloc[0]) if "city" in raw_day_df.columns else str(raw_day_df[group_cols[0]].iloc[0])
        event_date = str(raw_day_df["event_date"].dropna().iloc[0])
        track_j_tmax_f, forecast_available = _forecast_lookup(city, event_date, forecasts_df)
        day_df = filter_to_trading_window(raw_day_df)
        if day_df.empty:
            records.append(
                {
                    "event_date": event_date,
                    "city": _city_key(city),
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
                    "track_j_tmax_f": track_j_tmax_f,
                    "market_modal_bucket": "",
                    "agrees_with_market": np.nan,
                }
            )
            continue

        signal = track_j_flat_signal(day_df, k=k, forecasts_df=forecasts_df)
        market_modal_bucket = ""
        if pd.notna(signal.entry_snapshot_time):
            market_modal_bucket = compute_modal_bucket(day_df, signal.entry_snapshot_time)

        if signal.no_signal:
            records.append(
                {
                    "event_date": signal.event_date,
                    "city": _city_key(city),
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
                    "track_j_tmax_f": track_j_tmax_f,
                    "market_modal_bucket": market_modal_bucket if forecast_available else "",
                    "agrees_with_market": np.nan,
                }
            )
            continue

        resolved_correctly = _resolved_correctly(day_df, signal.bucket_label)
        entry_price = float(signal.entry_price)
        gross_pnl_cents = 100.0 * contracts * ((1.0 if resolved_correctly else 0.0) - entry_price)
        net = net_pnl(gross_pnl_cents, C=contracts, P=entry_price, order_type=order_type)
        records.append(
            {
                "event_date": signal.event_date,
                "city": _city_key(city),
                "entry_time": signal.entry_snapshot_time,
                "bucket_label": signal.bucket_label,
                "side": signal.side,
                "entry_price": entry_price,
                "signal_value": float(signal.signal_value),
                "no_signal": False,
                "gross_pnl_cents": gross_pnl_cents,
                "fee_cents": float(taker_fee(contracts, entry_price)),
                "net_pnl_cents": float(net),
                "resolved_correctly": resolved_correctly,
                "track_j_tmax_f": track_j_tmax_f,
                "market_modal_bucket": market_modal_bucket,
                "agrees_with_market": str(signal.bucket_label) == str(market_modal_bucket),
            }
        )

    return pd.DataFrame.from_records(records)
