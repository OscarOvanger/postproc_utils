from __future__ import annotations

import numpy as np
import pandas as pd


CALENDAR_LAG_COLUMNS = ["doy_sin", "doy_cos", "tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_lag7", "tmax_rollmean_7", "tmax_rollmean_30"]


def build_calendar_lag_features(cli_target_df: pd.DataFrame) -> pd.DataFrame:
    result = cli_target_df[["date", "tmax_f"]].copy()
    result["date_dt"] = pd.to_datetime(result["date"], errors="coerce")
    result = result[result["date_dt"].notna()].sort_values("date_dt")
    day_of_year = result["date_dt"].dt.dayofyear
    days_in_year = np.where(result["date_dt"].dt.is_leap_year, 366, 365)
    result["doy_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    result["doy_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)
    shifted = pd.to_numeric(result["tmax_f"], errors="coerce").shift(1)
    result["tmax_lag1"] = shifted
    result["tmax_lag2"] = result["tmax_f"].shift(2)
    result["tmax_lag3"] = result["tmax_f"].shift(3)
    result["tmax_lag7"] = result["tmax_f"].shift(7)
    result["tmax_rollmean_7"] = shifted.rolling(7, min_periods=7).mean()
    result["tmax_rollmean_30"] = shifted.rolling(30, min_periods=30).mean()
    result["date"] = result["date_dt"].dt.strftime("%Y-%m-%d")
    return result[["date", *CALENDAR_LAG_COLUMNS]]
