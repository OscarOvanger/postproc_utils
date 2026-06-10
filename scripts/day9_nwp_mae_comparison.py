"""Compare NWS MOS vs best-available NWP MAE on 2026 Kalshi dates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_nws_forecast import TRAIN_CITIES  # noqa: E402

TRACKB_DIR = PROJECT_ROOT / "data" / "trackb"
TIME_HOLDOUT_PATH = PROJECT_ROOT / "data" / "splits" / "time_holdout.parquet"
OPENMETEO_PATH = TRACKB_DIR / "openmeteo_nwp_raw.parquet"


def _dominant_source(city: str, dates: pd.Series) -> str:
    if not OPENMETEO_PATH.exists():
        return "nws_mos"
    raw = pd.read_parquet(OPENMETEO_PATH)
    raw = raw[raw["city"].eq(city)].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    date_set = set(pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m-%d"))
    subset = raw[raw["date"].isin(date_set)]
    ecmwf = int((subset["model_used"].eq("ecmwf_ifs025") & subset["nwp_tmax_forecast_f"].notna()).sum())
    gfs = int((subset["model_used"].eq("gfs_seamless") & subset["nwp_tmax_forecast_f"].notna()).sum())
    if ecmwf >= gfs and ecmwf > 0:
        return "ecmwf"
    if gfs > 0:
        return "gfs_seamless"
    return "nws_mos"


def main() -> None:
    if not TIME_HOLDOUT_PATH.exists():
        raise FileNotFoundError(f"Missing {TIME_HOLDOUT_PATH}")
    holdout = pd.read_parquet(TIME_HOLDOUT_PATH)
    holdout["event_date"] = pd.to_datetime(holdout["event_date"], errors="coerce")
    kalshi_2026 = holdout[holdout["event_date"].dt.year.eq(2026)].copy()
    if kalshi_2026.empty:
        kalshi_2026 = holdout.copy()

    rows: list[dict] = []
    for city in TRAIN_CITIES:
        feat_path = TRACKB_DIR / city / "features.parquet"
        if not feat_path.exists():
            continue
        features = pd.read_parquet(feat_path)
        features["date"] = pd.to_datetime(features["date"], errors="coerce")
        city_dates = kalshi_2026[kalshi_2026["city"].astype(str).str.replace(" ", "_").eq(city.replace("_", " "))]["event_date"]
        if city_dates.empty:
            city_key = city
            city_dates = kalshi_2026[kalshi_2026.get("city", pd.Series(dtype=object)).astype(str).str.lower().str.replace(" ", "_").eq(city_key)]["event_date"]
        eval_dates = pd.to_datetime(city_dates, errors="coerce").dropna().unique()
        subset = features[features["date"].isin(eval_dates)].copy()
        if subset.empty:
            subset = features[features["date"].dt.year.eq(2026)].copy()
        subset = subset[subset["tmax"].notna()]
        if subset.empty:
            continue
        nws_mae = (subset["nws_tmax_forecast_f"] - subset["tmax"]).abs().mean()
        nwp_mae = (subset["nwp_tmax_best_f"] - subset["tmax"]).abs().mean()
        improvement = nws_mae - nwp_mae if nws_mae == nws_mae and nwp_mae == nwp_mae else float("nan")
        rows.append(
            {
                "City": city,
                "NWS MOS MAE": round(nws_mae, 2) if nws_mae == nws_mae else None,
                "NWP best MAE": round(nwp_mae, 2) if nwp_mae == nwp_mae else None,
                "Improvement (°F)": round(improvement, 2) if improvement == improvement else None,
                "Source used": _dominant_source(city, subset["date"]),
            }
        )
        if improvement == improvement and improvement <= 0:
            print(f"NOTE: {city} NWP best did not improve over NWS MOS on 2026 Kalshi dates.")

    summary = pd.DataFrame(rows)
    print("\n=== 2026 KALSHI NWP MAE COMPARISON ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
