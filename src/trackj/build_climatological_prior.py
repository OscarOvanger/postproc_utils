"""Build climatological prior N(mu_clim, sigma_clim^2) per city and day-of-year."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch_cli_target import fetch_cli_target

CITIES = [
    "austin",
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "san_francisco",
]
CLI_START = date(2010, 1, 1)
CLI_END = date(2024, 12, 31)
DOY_SMOOTH_WINDOW = 15
SUMMER_DOY_START = 152
SUMMER_DOY_END = 243
WINTER_DOY_LOW = 335
WINTER_DOY_HIGH = 59

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
NGBOOST_DIR = PROJECT_ROOT / "data" / "ngboost"
RAW_DIR = TRACKJ_DIR / "raw"
CLI_HISTORY_DIR = NGBOOST_DIR / "cli_history"
CLI_STAGING_DIR = NGBOOST_DIR / "cli_staging"
PRIOR_PATH = NGBOOST_DIR / "climatological_prior.parquet"

_city_config_cache: dict | None = None


def _load_all_city_config() -> dict:
    global _city_config_cache
    if _city_config_cache is None:
        _city_config_cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _city_config_cache


def _load_city_config(city: str) -> dict:
    config = _load_all_city_config()
    if city not in config:
        raise KeyError(f"Unknown city: {city}")
    return config[city]


def fetch_cli_history(
    city: str,
    start_date: date,
    end_date: date,
    no_fetch: bool = False,
) -> pd.DataFrame:
    """Fetch CLI Tmax history and save to data/ngboost/cli_history/{city}.parquet."""
    city_config = _load_city_config(city)
    CLI_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CLI_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    cli = fetch_cli_target(
        city_config,
        start_date,
        end_date,
        RAW_DIR,
        CLI_STAGING_DIR,
        no_fetch=no_fetch,
    )
    cli["date"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    valid = cli[cli["tmax_f"].notna()].copy()
    out_cols = [c for c in ("date", "tmax_f", "tmin_f", "source_product_id") if c in valid.columns]
    history = valid[out_cols].sort_values("date")
    history.to_parquet(CLI_HISTORY_DIR / f"{city}.parquet", index=False)
    print(f"  {city}: {len(history)} CLI records saved to cli_history")
    return history


def compute_raw_doy_stats(cli_df: pd.DataFrame) -> pd.DataFrame:
    """Per-DOY mean, std, and count of tmax_f; reindex to DOY 1-366."""
    df = cli_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"].notna() & df["tmax_f"].notna()].copy()
    df["tmax_f"] = pd.to_numeric(df["tmax_f"], errors="coerce")
    df = df[df["tmax_f"].notna()]
    df["doy"] = df["date_dt"].dt.dayofyear

    city_std = float(df["tmax_f"].std(ddof=1)) if len(df) >= 2 else 0.0
    if city_std == 0.0 or np.isnan(city_std):
        city_std = 1.0

    grouped = df.groupby("doy")["tmax_f"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "mu_clim", "std": "sigma_clim", "count": "n_obs"}
    )
    full_doy = pd.DataFrame({"doy": np.arange(1, 367)})
    stats = full_doy.merge(grouped, on="doy", how="left")
    stats["n_obs"] = stats["n_obs"].fillna(0).astype(int)
    sparse = stats["n_obs"] < 2
    stats.loc[sparse, "sigma_clim"] = city_std
    return stats.sort_values("doy").reset_index(drop=True)


def smooth_doy_stats(doy_df: pd.DataFrame, window: int = DOY_SMOOTH_WINDOW) -> pd.DataFrame:
    """Apply centered rolling smooth to mu_clim and sigma_clim along DOY."""
    result = doy_df.sort_values("doy").copy()
    result["mu_clim"] = result["mu_clim"].rolling(window=window, center=True, min_periods=1).mean()
    result["sigma_clim"] = result["sigma_clim"].rolling(window=window, center=True, min_periods=1).mean()

    raw_mu = doy_df.sort_values("doy")["mu_clim"]
    raw_sigma = doy_df.sort_values("doy")["sigma_clim"]
    result["mu_clim"] = result["mu_clim"].fillna(raw_mu)
    result["sigma_clim"] = result["sigma_clim"].fillna(raw_sigma)

    if result["mu_clim"].isna().any():
        result["mu_clim"] = result["mu_clim"].fillna(result["mu_clim"].mean())
    if result["sigma_clim"].isna().any():
        fallback_sigma = float(raw_sigma.dropna().mean()) if raw_sigma.notna().any() else 1.0
        result["sigma_clim"] = result["sigma_clim"].fillna(fallback_sigma)

    result["sigma_clim"] = result["sigma_clim"].clip(lower=0.5)
    return result


def build_climatological_prior(
    cities: list[str],
    start_date: date,
    end_date: date,
    no_fetch: bool = False,
) -> pd.DataFrame:
    """Build and save climatological prior for all cities."""
    NGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    for city in cities:
        print(f"Building climatological prior for {city}...")
        history = fetch_cli_history(city, start_date, end_date, no_fetch=no_fetch)
        raw_stats = compute_raw_doy_stats(history)
        smoothed = smooth_doy_stats(raw_stats)
        smoothed.insert(0, "city", city)
        rows.append(smoothed[["city", "doy", "mu_clim", "sigma_clim", "n_obs"]])

    prior = pd.concat(rows, ignore_index=True)
    prior["doy"] = prior["doy"].astype(int)
    prior["n_obs"] = prior["n_obs"].astype(int)
    prior.to_parquet(PRIOR_PATH, index=False)
    print(f"Saved climatological prior: {PRIOR_PATH} ({len(prior)} rows)")
    return prior


def print_summary_table(prior_df: pd.DataFrame) -> pd.DataFrame:
    """Print per-city summary of sigma_clim and total observations."""
    summary_rows: list[dict] = []
    for city in prior_df["city"].unique():
        city_df = prior_df[prior_df["city"] == city]
        sigmas = city_df["sigma_clim"]
        summary_rows.append(
            {
                "city": city,
                "mean sigma": round(float(sigmas.mean()), 2),
                "min sigma": round(float(sigmas.min()), 2),
                "max sigma": round(float(sigmas.max()), 2),
                "total obs": int(city_df["n_obs"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    print("\n=== CLIMATOLOGICAL PRIOR SUMMARY ===")
    print(summary.to_string(index=False))
    return summary


def _is_summer_doy(doy: int) -> bool:
    return SUMMER_DOY_START <= doy <= SUMMER_DOY_END


def _is_winter_doy(doy: int) -> bool:
    return doy >= WINTER_DOY_LOW or doy <= WINTER_DOY_HIGH


def run_sanity_checks(prior_df: pd.DataFrame) -> None:
    """Print sanity checks (no assertions)."""
    print("\n=== CLIMATOLOGICAL PRIOR SANITY CHECKS ===")

    nan_mu = int(prior_df["mu_clim"].isna().sum())
    nan_sigma = int(prior_df["sigma_clim"].isna().sum())
    nan_nobs = int(prior_df["n_obs"].isna().sum())
    nan_status = "PASS" if nan_mu == 0 and nan_sigma == 0 and nan_nobs == 0 else "FAIL"
    print(f"  No NaN values: {nan_status} (mu={nan_mu}, sigma={nan_sigma}, n_obs={nan_nobs})")

    outside = prior_df[(prior_df["sigma_clim"] < 5) | (prior_df["sigma_clim"] > 12)]
    pct_outside = 100.0 * len(outside) / len(prior_df) if len(prior_df) else 0.0
    print(f"  sigma_clim outside 5-12F: {pct_outside:.1f}% of rows ({len(outside)} / {len(prior_df)})")
    if pct_outside > 50:
        print("  FLAG: majority of sigma_clim values outside expected 5-12F range")

    for city in prior_df["city"].unique():
        city_df = prior_df[prior_df["city"] == city]
        summer = city_df[city_df["doy"].map(_is_summer_doy)]
        winter = city_df[city_df["doy"].map(_is_winter_doy)]
        summer_mu = float(summer["mu_clim"].mean()) if not summer.empty else float("nan")
        winter_mu = float(winter["mu_clim"].mean()) if not winter.empty else float("nan")
        summer_sigma = float(summer["sigma_clim"].mean()) if not summer.empty else float("nan")
        winter_sigma = float(winter["sigma_clim"].mean()) if not winter.empty else float("nan")
        seasonal_ok = summer_mu > winter_mu if not (np.isnan(summer_mu) or np.isnan(winter_mu)) else False
        sigma_ok = summer_sigma < winter_sigma if not (np.isnan(summer_sigma) or np.isnan(winter_sigma)) else False
        print(
            f"  {city}: summer mu={summer_mu:.1f}F winter mu={winter_mu:.1f}F "
            f"(seasonal cycle {'OK' if seasonal_ok else 'check'})"
        )
        print(
            f"         summer sigma={summer_sigma:.2f}F winter sigma={winter_sigma:.2f}F "
            f"(summer < winter: {'yes' if sigma_ok else 'no'})"
        )
