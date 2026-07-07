#!/usr/bin/env python3
"""Bootstrap rolling bias residuals from NGBoost mu vs WU actuals."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_ngboost as ng  # noqa: E402
from backtest.ngboost_inference import NgBoostBacktestModels  # noqa: E402
from src.poly_trading_pipeline import load_wunderground_bias  # noqa: E402
from src.rolling_bias import save_residuals_and_snapshot  # noqa: E402

WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
MODEL_PATH_FILE = PROJECT_ROOT / "reports" / "backtest_model_path.txt"
DEFAULT_HALFLIFE = 20
DEFAULT_MIN_OBS = 5
LAG_COLS = ["tmax_lag1", "tmax_lag2"]


def predict_mu_batch(models: NgBoostBacktestModels, feat_df: pd.DataFrame) -> np.ndarray:
    """Vectorized mu for rows with complete lag features."""
    df = feat_df.copy()
    df = ng.apply_saved_median_fill(df, models.fill_medians, list(models.fill_medians.keys()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df["lgb_tmax_pred"] = models.lgb_model.predict(df[models.stage1_cols])
    complete = df[LAG_COLS + ["hrrr_tmax"]].notna().all(axis=1)
    mu = np.full(len(df), np.nan, dtype=float)
    if not complete.any():
        return mu
    X = ng.transform_features(models.scaler, df.loc[complete], models.feature_cols)
    pred_mu, _sigma, _df = ng.predict_dist_params(models.model, X)
    mu[complete.to_numpy()] = pred_mu
    return mu


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize rolling bias from NGBoost history")
    parser.add_argument("--halflife", type=int, default=DEFAULT_HALFLIFE)
    parser.add_argument("--min-obs", type=int, default=DEFAULT_MIN_OBS)
    parser.add_argument("--model-path-file", default=str(MODEL_PATH_FILE))
    args = parser.parse_args()

    models = NgBoostBacktestModels.from_path_file(Path(args.model_path_file))
    wu_bias = load_wunderground_bias()
    rows: list[dict] = []
    t0 = time.time()

    for city in sorted(ng.STATION_META):
        print(f"Building residuals for {city}...", flush=True)
        city_df = ng.build_city_features(city)
        city_df = ng.drop_incomplete_rows(city_df)
        if city_df.empty:
            continue
        city_df["date"] = pd.to_datetime(city_df["date"]).dt.strftime("%Y-%m-%d")
        mu = predict_mu_batch(models, city_df)
        for i, row in city_df.iterrows():
            m = float(mu[i]) if np.isfinite(mu[i]) else np.nan
            actual = float(row[ng.TARGET])
            if not np.isfinite(m) or not np.isfinite(actual):
                continue
            static_bias = float(wu_bias.get(city, {}).get("median_bias", 0.0))
            corrected_mu = m - static_bias
            rows.append(
                {
                    "city": city,
                    "date": str(row["date"]),
                    "forecast": corrected_mu,
                    "wu_actual": actual,
                    "residual": corrected_mu - actual,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Computed {len(df)} residuals across {df['city'].nunique()} cities in {time.time() - t0:.1f}s")
    save_residuals_and_snapshot(df, halflife_days=args.halflife, min_obs=args.min_obs)
    print(f"Wrote residuals and snapshot (halflife={args.halflife})")

    snapshot_path = PROJECT_ROOT / "data" / "polymarket" / "rolling_bias.json"
    with open(snapshot_path, encoding="utf-8") as handle:
        snapshot = json.load(handle)
    print("\nPer-city EWMA (after static WU bias correction):")
    for city in sorted(snapshot):
        ewma = snapshot[city]["ewma"]
        flag = " *** EXCEEDS ±1.5F" if abs(ewma) > 1.5 else ""
        print(f"  {city}: {ewma:+.4f}F (n={snapshot[city]['n_obs']}){flag}")


if __name__ == "__main__":
    main()
