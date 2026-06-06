"""Entry-rule interface and look-ahead guard for baseline strategies."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import pandas as pd


ENTRY_FLOOR_LOCAL = datetime.time(10, 0, 0)
# No baseline may enter before this local time.
# Rationale: pre-10AM entries reflect overnight price discovery
# in a different market regime; all baselines use the 10AM-to-
# resolution window for comparability.


@dataclass
class TradeSignal:
    """A point-in-time trade entry decision for one event date."""

    event_date: str
    entry_snapshot_time: pd.Timestamp
    bucket_label: str
    side: Literal["YES", "NO"]
    entry_price: float
    signal_name: str
    signal_value: float
    no_signal: bool = False


def make_entry_rule(fn):
    """
    Decorator enforcing the entry-rule contract.

    fn must accept (day_df: pd.DataFrame, **params) -> TradeSignal
    where day_df is all rows for a single event_date, sorted by
    snapshot_time_local ascending.
    Raises if fn returns a TradeSignal whose entry_snapshot_time does
    not exist in day_df (look-ahead guard).
    """

    @wraps(fn)
    def wrapper(day_df: pd.DataFrame, **params) -> TradeSignal:
        if "snapshot_time_local" not in day_df.columns:
            raise ValueError("day_df must contain snapshot_time_local")
        if day_df.empty:
            raise ValueError("day_df must contain at least one row")

        signal = fn(day_df, **params)
        if not isinstance(signal, TradeSignal):
            raise TypeError(f"{fn.__name__} must return a TradeSignal")
        if not isinstance(signal.entry_snapshot_time, pd.Timestamp):
            raise TypeError("entry_snapshot_time must be a pandas Timestamp")

        valid_times = pd.to_datetime(day_df["snapshot_time_local"]).dropna()
        if signal.entry_snapshot_time not in set(valid_times):
            raise ValueError(
                "entry_snapshot_time must be an existing snapshot_time_local in day_df"
            )
        if not signal.no_signal and signal.entry_snapshot_time.time() < ENTRY_FLOOR_LOCAL:
            raise ValueError(
                f"Entry at {signal.entry_snapshot_time.time()} violates "
                f"10AM floor. Filter day_df before calling entry rule."
            )
        return signal

    return wrapper


def filter_to_trading_window(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only rows where snapshot_time_local >= 10:00 AM local time.
    All entry-rule functions must call this at the top before any
    signal computation.
    """
    if "snapshot_time_local" not in day_df.columns:
        raise ValueError("day_df must contain snapshot_time_local")
    if day_df.empty:
        return day_df.copy()

    df = day_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    if "event_date" in df.columns and df["event_date"].notna().any():
        floor_date = pd.to_datetime(df["event_date"].dropna().iloc[0])
        first_snapshot = pd.Timestamp(df["snapshot_time_local"].dropna().iloc[0])
        if first_snapshot.tzinfo is not None and floor_date.tzinfo is None:
            floor_date = floor_date.tz_localize(first_snapshot.tzinfo)
    else:
        floor_date = pd.Timestamp(df["snapshot_time_local"].dropna().iloc[0])
    floor = floor_date.replace(
        hour=ENTRY_FLOOR_LOCAL.hour,
        minute=ENTRY_FLOOR_LOCAL.minute,
        second=ENTRY_FLOOR_LOCAL.second,
        microsecond=0,
    )
    return df[df["snapshot_time_local"] >= floor].copy()
