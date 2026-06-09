from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .build_asos_features import ASOS_FEATURE_COLUMNS, build_asos_features
from .build_calendar_lag_features import CALENDAR_LAG_COLUMNS, build_calendar_lag_features
from .fetch_cli_target import fetch_cli_target


TRACK_A_COVARIATES = ASOS_FEATURE_COLUMNS + CALENDAR_LAG_COLUMNS


def build_trackA_table(
    city_config: dict,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    output_dir: Path,
    no_fetch: bool = False,
    sleep_seconds: float = 1.1,
) -> pd.DataFrame:
    city_output = Path(output_dir) / city_config["city"]
    cli_path = city_output / "cli_target.parquet"
    asos_path = city_output / "asos_features.parquet"
    if no_fetch and cli_path.exists():
        cli_target = pd.read_parquet(cli_path)
    else:
        cli_target = fetch_cli_target(
            city_config,
            start_date,
            end_date,
            raw_dir,
            output_dir,
            no_fetch=no_fetch,
        )
    if no_fetch and asos_path.exists():
        asos = pd.read_parquet(asos_path)
    else:
        asos = build_asos_features(
            city_config,
            start_date,
            end_date,
            raw_dir,
            output_dir,
            no_fetch=no_fetch,
            target_df=cli_target,
            sleep_seconds=sleep_seconds,
        )
    calendar_lags = build_calendar_lag_features(cli_target)
    joined = cli_target[["date", "tmax_f"]].merge(asos, on="date", how="inner").merge(calendar_lags, on="date", how="inner")
    final = joined.dropna(subset=["tmax_f", *TRACK_A_COVARIATES]).sort_values("date").copy()
    city_output.mkdir(parents=True, exist_ok=True)
    final.to_parquet(city_output / "trackA_table.parquet", index=False)
    return final[["date", "tmax_f", *TRACK_A_COVARIATES]]
