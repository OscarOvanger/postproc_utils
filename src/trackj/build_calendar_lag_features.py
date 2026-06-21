from __future__ import annotations

import numpy as np
import pandas as pd


CALENDAR_LAG_COLUMNS = ["doy_sin", "doy_cos", "tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_lag7", "tmax_rollmean_7", "tmax_rollmean_30"]


def build_calendar_lag_features(cli_target_df: pd.DataFrame) -> pd.DataFrame:
    df = cli_target_df[["date", "tmax_f"]].copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"].notna()].drop_duplicates("date_dt").sort_values("date_dt")
    tmax = pd.to_numeric(df["tmax_f"], errors="coerce")

    # Reindex to a complete daily date range so positional shift == date shift.
    # This fills gap dates with NaN tmax, making .shift(n) mean "n days ago".
    full_idx = pd.date_range(df["date_dt"].min(), df["date_dt"].max(), freq="D")
    daily = pd.DataFrame({"date_dt": full_idx})
    daily = daily.merge(
        df[["date_dt"]].assign(tmax_f=tmax.values),
        on="date_dt",
        how="left",
    )
    daily = daily.sort_values("date_dt").reset_index(drop=True)

    # Calendar features
    doy = daily["date_dt"].dt.dayofyear
    days_in_year = np.where(daily["date_dt"].dt.is_leap_year, 366, 365)
    daily["doy_sin"] = np.sin(2 * np.pi * doy / days_in_year)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / days_in_year)

    # Lag features: .shift(n) on a complete daily index = n calendar days
    t = daily["tmax_f"]
    daily["tmax_lag1"] = t.shift(1)
    daily["tmax_lag2"] = t.shift(2)
    daily["tmax_lag3"] = t.shift(3)
    daily["tmax_lag7"] = t.shift(7)

    # Rolling means on the shifted series (yesterday and prior),
    # with min_periods reduced to tolerate small gaps within the window
    shifted = t.shift(1)
    daily["tmax_rollmean_7"] = shifted.rolling(7, min_periods=4).mean()
    daily["tmax_rollmean_30"] = shifted.rolling(30, min_periods=15).mean()

    # Format output
    daily["date"] = daily["date_dt"].dt.strftime("%Y-%m-%d")

    # Return only the dates that were in the original input
    # (including the placeholder for today if it was added)
    input_dates = set(df["date_dt"].dt.strftime("%Y-%m-%d"))
    result = daily[daily["date"].isin(input_dates)].copy()

    return result[["date", *CALENDAR_LAG_COLUMNS]]
