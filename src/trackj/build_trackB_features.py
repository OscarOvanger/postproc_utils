from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .build_asos_features import ASOS_FEATURE_COLUMNS, build_asos_features
from .build_calendar_lag_features import CALENDAR_LAG_COLUMNS, build_calendar_lag_features
from .fetch_cli_target import fetch_cli_target
from .fetch_gfs_herbie import GFS_FEATURE_COLUMNS, build_gfs_features
from .fetch_openmeteo_nwp import DEFAULT_OUTPUT_PATH as OPENMETEO_NWP_PATH

ASOS_MORNING_COLUMNS = [column for column in ASOS_FEATURE_COLUMNS if column != "temp_lag1"]
GROUP2_COLUMNS = list(CALENDAR_LAG_COLUMNS) + ["temp_lag1"]
NWS_COLUMNS = ["nws_tmax_forecast_f", "nws_tmax_forecast_issued_h"]
NWP_BEST_COLUMNS = ["nwp_tmax_best_f", "nwp_is_ecmwf"]
TRACKB_BASE_COLUMNS = ASOS_MORNING_COLUMNS + GROUP2_COLUMNS
TRACKB_ALWAYS_COLUMNS = TRACKB_BASE_COLUMNS


def _gfs_raw_dir(city_config: dict, raw_root: Path) -> Path:
    station = str(city_config["nws_station"]).lower()
    if station == "kaus":
        return raw_root / "gfs_kaus"
    return raw_root / f"gfs_{station}"


def _load_nws_forecasts(nws_path: Path, city: str) -> pd.DataFrame:
    if not nws_path.exists():
        return pd.DataFrame(columns=["date", *NWS_COLUMNS])
    frame = pd.read_parquet(nws_path)
    city_rows = frame[frame["city"].astype(str).eq(city)].copy()
    city_rows["date"] = pd.to_datetime(city_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    city_rows["nws_tmax_forecast_f"] = pd.to_numeric(city_rows.get("tmax_forecast_f"), errors="coerce")
    city_rows["nws_tmax_forecast_issued_h"] = pd.to_numeric(city_rows.get("hours_since_issuance"), errors="coerce")
    return city_rows[["date", *NWS_COLUMNS, "issued_time"]]


def _load_openmeteo_nwp(path: Path, city: str, model: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "nwp_tmax_forecast_f", "issued_date", "valid_date"])
    frame = pd.read_parquet(path)
    city_rows = frame[frame["city"].astype(str).eq(city) & frame["model_used"].astype(str).eq(model)].copy()
    city_rows["date"] = pd.to_datetime(city_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    city_rows["nwp_tmax_forecast_f"] = pd.to_numeric(city_rows.get("nwp_tmax_forecast_f"), errors="coerce")
    return city_rows[["date", "nwp_tmax_forecast_f", "issued_date", "valid_date"]]


def _build_nwp_best_column(merged: pd.DataFrame, ecmwf_df: pd.DataFrame, gfs_df: pd.DataFrame) -> pd.DataFrame:
    """Priority: ECMWF IFS -> GFS seamless -> NWS MOS."""
    result = merged.copy()
    result["date_key"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ecmwf = ecmwf_df.rename(
        columns={
            "date": "date_key",
            "nwp_tmax_forecast_f": "ecmwf_tmax",
            "issued_date": "ecmwf_issued",
            "valid_date": "ecmwf_valid",
        }
    )
    gfs = gfs_df.rename(
        columns={
            "date": "date_key",
            "nwp_tmax_forecast_f": "gfs_tmax",
            "issued_date": "gfs_issued",
            "valid_date": "gfs_valid",
        }
    )
    result = result.merge(ecmwf[["date_key", "ecmwf_tmax", "ecmwf_issued", "ecmwf_valid"]], on="date_key", how="left")
    result = result.merge(gfs[["date_key", "gfs_tmax", "gfs_issued", "gfs_valid"]], on="date_key", how="left")

    nws_vals = pd.to_numeric(result.get("nws_tmax_forecast_f"), errors="coerce")
    ecmwf_vals = pd.to_numeric(result.get("ecmwf_tmax"), errors="coerce")
    gfs_vals = pd.to_numeric(result.get("gfs_tmax"), errors="coerce")

    result["nwp_tmax_best_f"] = ecmwf_vals
    result["nwp_is_ecmwf"] = ecmwf_vals.notna().astype(float)
    result["nwp_source"] = pd.Series([None] * len(result), index=result.index, dtype=object)
    result.loc[ecmwf_vals.notna(), "nwp_source"] = "ecmwf"
    result["nwp_issued_date"] = result.get("ecmwf_issued")
    result["nwp_valid_date"] = result.get("ecmwf_valid")

    gfs_mask = result["nwp_tmax_best_f"].isna() & gfs_vals.notna()
    result.loc[gfs_mask, "nwp_tmax_best_f"] = gfs_vals[gfs_mask]
    result.loc[gfs_mask, "nwp_is_ecmwf"] = 0.0
    result.loc[gfs_mask, "nwp_source"] = "gfs_seamless"
    result.loc[gfs_mask, "nwp_issued_date"] = result.loc[gfs_mask, "gfs_issued"]
    result.loc[gfs_mask, "nwp_valid_date"] = result.loc[gfs_mask, "gfs_valid"]

    nws_mask = result["nwp_tmax_best_f"].isna() & nws_vals.notna()
    result.loc[nws_mask, "nwp_tmax_best_f"] = nws_vals[nws_mask]
    result.loc[nws_mask, "nwp_is_ecmwf"] = 0.0
    result.loc[nws_mask, "nwp_source"] = "nws_mos"
    issued = pd.to_datetime(result.loc[nws_mask, "issued_time"], utc=True, errors="coerce")
    result.loc[nws_mask, "nwp_issued_date"] = issued.dt.strftime("%Y-%m-%d")
    result.loc[nws_mask, "nwp_valid_date"] = result.loc[nws_mask, "date_key"]

    drop_cols = [c for c in ("date_key", "ecmwf_tmax", "gfs_tmax", "ecmwf_issued", "gfs_issued", "ecmwf_valid", "gfs_valid") if c in result.columns]
    return result.drop(columns=drop_cols, errors="ignore")


def print_nwp_source_distribution(merged: pd.DataFrame, city: str) -> dict:
    """Summarize NWP source mix per city (requires nwp_source column before export)."""
    n_rows = len(merged)
    if n_rows == 0:
        return {"City": city, "% ecmwf": 0.0, "% gfs_seamless": 0.0, "% nws_mos": 0.0, "% missing": 100.0}
    source = merged.get("nwp_source", pd.Series(index=merged.index, dtype=object))
    row = {
        "City": city,
        "% ecmwf": round(100.0 * source.eq("ecmwf").sum() / n_rows, 1),
        "% gfs_seamless": round(100.0 * source.eq("gfs_seamless").sum() / n_rows, 1),
        "% nws_mos": round(100.0 * source.eq("nws_mos").sum() / n_rows, 1),
        "% missing": round(100.0 * merged["nwp_tmax_best_f"].isna().mean(), 1),
    }
    if row["% missing"] > 10.0:
        print(f"FLAG: {city} nwp_tmax_best_f missing {row['% missing']}% (>10%)")
    return row


def assert_no_leakage(merged: pd.DataFrame, city_config: dict) -> None:
    """Assert Track-B feature table satisfies leakage constraints."""
    if merged.empty:
        return
    dates = pd.to_datetime(merged["date"], errors="coerce")
    if dates.isna().any():
        raise AssertionError("Track-B table contains invalid dates")
    required_lags = ("tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_lag7", "temp_lag1")
    missing_lags = [column for column in required_lags if column not in merged.columns]
    if missing_lags:
        raise AssertionError(f"Missing lag features (min lag >= 1 required): {missing_lags}")
    if "nws_tmax_forecast_f" in merged.columns and "issued_time" in merged.columns:
        issued = pd.to_datetime(merged["issued_time"], utc=True, errors="coerce")
        target = pd.to_datetime(merged["date"], errors="coerce")
        valid = issued.notna() & target.notna()
        if valid.any() and (issued[valid].dt.date >= target[valid].dt.date).any():
            raise AssertionError("NWS forecast issued_time must be strictly before target_date")
    if "nwp_tmax_best_f" in merged.columns:
        target = pd.to_datetime(merged["date"], errors="coerce")
        nwp_valid = merged["nwp_tmax_best_f"].notna()
        if nwp_valid.any():
            if "nwp_valid_date" in merged.columns:
                valid_dates = pd.to_datetime(merged.loc[nwp_valid, "nwp_valid_date"], errors="coerce")
                target_sub = target.loc[nwp_valid]
                if (valid_dates.dt.strftime("%Y-%m-%d") != target_sub.dt.strftime("%Y-%m-%d")).any():
                    raise AssertionError("NWP valid_date must equal target date")
            if "nwp_issued_date" in merged.columns:
                issued_dates = pd.to_datetime(merged.loc[nwp_valid, "nwp_issued_date"], errors="coerce")
                target_sub = target.loc[nwp_valid]
                if (issued_dates.dt.date >= target_sub.dt.date).any():
                    raise AssertionError("NWP issued_date must be strictly before target date")


def build_trackB_features(
    city_config: dict,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    output_dir: Path,
    nws_forecasts_path: Path,
    trackj_dir: Path | None = None,
    openmeteo_nwp_path: Path | None = None,
    include_gfs: bool = True,
    no_fetch: bool = True,
) -> pd.DataFrame:
    city = city_config["city"]
    trackj_city_dir = Path(trackj_dir or Path("data/trackj")) / city
    city_output = Path(output_dir) / city
    city_output.mkdir(parents=True, exist_ok=True)

    cli_path = trackj_city_dir / "cli_target.parquet"
    if no_fetch and cli_path.exists():
        cli_target = pd.read_parquet(cli_path)
    else:
        cli_target = fetch_cli_target(city_config, start_date, end_date, raw_dir, trackj_city_dir.parent, no_fetch=no_fetch)

    asos_path = trackj_city_dir / "asos_features.parquet"
    if no_fetch and asos_path.exists():
        asos = pd.read_parquet(asos_path)
    else:
        asos = build_asos_features(
            city_config,
            start_date,
            end_date,
            raw_dir,
            trackj_city_dir.parent,
            no_fetch=no_fetch,
            target_df=cli_target,
        )

    calendar_lags = build_calendar_lag_features(cli_target)
    asos_subset = asos[["date", *ASOS_MORNING_COLUMNS, "temp_lag1"]]
    base = (
        cli_target[["date", "tmax_f"]]
        .merge(asos_subset, on="date", how="inner")
        .merge(calendar_lags, on="date", how="inner")
    )

    nws = _load_nws_forecasts(nws_forecasts_path, city)
    merged = base.merge(nws[["date", *NWS_COLUMNS, "issued_time"]], on="date", how="left")

    if include_gfs:
        gfs_raw = _gfs_raw_dir(city_config, Path("data/raw"))
        gfs_features, _ = build_gfs_features(
            merged["date"],
            raw_dir=gfs_raw,
            fetch=False,
            city_config=city_config,
        )
        merged = merged.merge(gfs_features, on="date", how="left")

    nwp_path = Path(openmeteo_nwp_path or OPENMETEO_NWP_PATH)
    ecmwf_nwp = _load_openmeteo_nwp(nwp_path, city, "ecmwf_ifs025")
    gfs_nwp = _load_openmeteo_nwp(nwp_path, city, "gfs_seamless")
    merged = _build_nwp_best_column(merged, ecmwf_nwp, gfs_nwp)

    assert_no_leakage(merged, city_config)
    print_nwp_source_distribution(merged, city)

    final = merged.copy()
    final.insert(0, "city", city)
    final = final.rename(columns={"tmax_f": "tmax"})
    drop_internal = {"issued_time", "nwp_source", "nwp_issued_date", "nwp_valid_date"}
    feature_cols = [column for column in final.columns if column not in {"city", "date", "tmax", *drop_internal}]
    final = final[["city", "date", "tmax", *feature_cols, "issued_time", "nwp_source", "nwp_issued_date", "nwp_valid_date"]]
    final = final.drop(columns=list(drop_internal), errors="ignore")
    final = final.sort_values("date")
    final.to_parquet(city_output / "features.parquet", index=False)
    return final


def summarize_trackB_table(city: str, features: pd.DataFrame) -> dict:
    n_rows = len(features)
    feature_cols = [column for column in features.columns if column not in {"city", "date", "tmax"}]
    missing_pct = {column: round(100.0 * features[column].isna().mean(), 1) for column in feature_cols}
    nws_cov = round(100.0 * features.get("nws_tmax_forecast_f", pd.Series(dtype=float)).notna().mean(), 1) if n_rows else 0.0
    nwp_cov = round(100.0 * features.get("nwp_tmax_best_f", pd.Series(dtype=float)).notna().mean(), 1) if n_rows else 0.0
    gfs_cols = [column for column in GFS_FEATURE_COLUMNS if column in features.columns]
    gfs_cov = (
        round(100.0 * features[gfs_cols].notna().all(axis=1).mean(), 1)
        if gfs_cols and n_rows
        else 0.0
    )
    return {
        "City": city,
        "N rows": n_rows,
        "N features": len(feature_cols),
        "NWS coverage %": nws_cov,
        "NWP best coverage %": nwp_cov,
        "GFS coverage %": gfs_cov,
        "Missing % per feature": ", ".join(f"{k}:{v}" for k, v in missing_pct.items()),
    }
