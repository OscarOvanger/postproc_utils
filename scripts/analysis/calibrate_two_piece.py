#!/usr/bin/env python3
"""Calibrate two-piece Gaussian down-side ratio from 2025 validation residuals only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_ngboost as ng  # noqa: E402
from backtest.ngboost_inference import NgBoostBacktestModels, _two_piece_cdf  # noqa: E402

MODEL_PATH_FILE = PROJECT_ROOT / "reports" / "backtest_model_path.txt"
OUTPUT_CSV = PROJECT_ROOT / "data" / "analysis" / "two_piece_calibration.csv"
SUMMER_MONTHS = {6, 7, 8}
COVERAGE_LEVELS = [50, 80, 90]
MIN_SUMMER_SAMPLES = 300
MIN_R_HAT = 1.05


def assert_val_dates_only(dates: pd.Series) -> None:
    years = pd.to_datetime(dates).dt.year.unique().tolist()
    if any(year != 2025 for year in years):
        raise SystemExit(f"Refusing to run: validation dates must be 2025 only, got years {years}")
    if any(pd.to_datetime(dates) >= pd.Timestamp("2026-01-01")):
        raise SystemExit("Refusing to run: found 2026+ validation dates")


def two_piece_quantile(mu: float, sigma: float, ratio: float, p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile probability out of range: {p}")
    if ratio == 1.0:
        return float(norm.ppf(p, loc=mu, scale=sigma))

    s1 = ratio * sigma
    s2 = sigma
    denom = s1 + s2
    w_left = 2.0 * s1 / denom
    mass_below_mu = s1 / denom
    w_right = 2.0 * s2 / denom

    if p <= mass_below_mu:
        return mu + s1 * float(norm.ppf(p / w_left))
    inner = (p - mass_below_mu) / w_right + 0.5
    return mu + s2 * float(norm.ppf(inner))


def pit_values(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    ratio: float | None,
) -> np.ndarray:
    if ratio is None or ratio == 1.0:
        return norm.cdf(y, loc=mu, scale=sigma)
    return np.array(
        [_two_piece_cdf(float(m), float(s), ratio, float(v)) for m, s, v in zip(mu, sigma, y)]
    )


def central_coverage(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal_pct: int,
    ratio: float | None,
) -> float:
    alpha = nominal_pct / 100.0
    p_lo = (1.0 - alpha) / 2.0
    p_hi = 1.0 - (1.0 - alpha) / 2.0
    lowers = np.zeros_like(y, dtype=float)
    uppers = np.zeros_like(y, dtype=float)
    for i, (m, s, obs) in enumerate(zip(mu, sigma, y)):
        if ratio is None or ratio == 1.0:
            z = dict(ng.COVERAGE_LEVELS).get(nominal_pct)
            assert z is not None
            lowers[i] = m - z * s
            uppers[i] = m + z * s
        else:
            lowers[i] = two_piece_quantile(float(m), float(s), ratio, p_lo)
            uppers[i] = two_piece_quantile(float(m), float(s), ratio, p_hi)
        _ = obs
    return 100.0 * float(np.mean((y >= lowers) & (y <= uppers)))


def compute_r_hat(z: np.ndarray) -> tuple[float, float, int, int]:
    z_neg = z[z < 0]
    z_pos = z[z > 0]
    if len(z_neg) < 2 or len(z_pos) < 2:
        return float("nan"), float("nan"), len(z_neg), len(z_pos)
    std_ratio = float(np.std(z_neg, ddof=1) / np.std(z_pos, ddof=1))
    robust_ratio = float(np.mean(np.abs(z_neg)) / np.mean(np.abs(z_pos)))
    return std_ratio, robust_ratio, len(z_neg), len(z_pos)


def load_validation_predictions() -> tuple[pd.DataFrame, NgBoostBacktestModels]:
    models = NgBoostBacktestModels.from_path_file(MODEL_PATH_FILE)
    cities = list(models.config.get("cities", ng.DEFAULT_CITIES))
    dataset = ng.drop_incomplete_rows(ng.assemble_dataset(cities))
    _train_df, val_df, _test_df = ng.temporal_split(dataset)
    assert_val_dates_only(val_df["date"])

    val_df = val_df.copy()
    fill_cols = list(models.fill_medians.keys())
    val_df = ng.apply_saved_median_fill(val_df, models.fill_medians, fill_cols)
    stage1_cols = [c for c in models.feature_cols if c != "lgb_tmax_pred"]
    val_df["lgb_tmax_pred"] = models.lgb_model.predict(val_df[stage1_cols])

    X_val = ng.transform_features(models.scaler, val_df, models.feature_cols)
    mu, sigma, _df_vals = ng.predict_dist_params(models.model, X_val)
    sigma_cal = ng.apply_sigma_calibration(sigma, models.sigma_k)

    out = val_df[["date", "city", ng.TARGET]].copy()
    out["mu"] = mu
    out["sigma"] = sigma_cal
    out["y"] = out[ng.TARGET].astype(float)
    out["month"] = pd.to_datetime(out["date"]).dt.month
    return out, models


def diagnostics_block(
    subset_name: str,
    frame: pd.DataFrame,
    ratio: float | None,
) -> list[dict[str, Any]]:
    y = frame["y"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    sigma = frame["sigma"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    pit_gauss = pit_values(y, mu, sigma, None)
    ks_g = kstest(pit_gauss, "uniform")
    rows.append(
        {
            "subset": subset_name,
            "metric": "pit_ks_stat_gaussian",
            "value": float(ks_g.statistic),
            "detail": "",
        }
    )
    rows.append(
        {
            "subset": subset_name,
            "metric": "pit_ks_pvalue_gaussian",
            "value": float(ks_g.pvalue),
            "detail": "",
        }
    )

    if ratio is not None and ratio != 1.0:
        pit_two = pit_values(y, mu, sigma, ratio)
        ks_t = kstest(pit_two, "uniform")
        rows.append(
            {
                "subset": subset_name,
                "metric": "pit_ks_stat_two_piece",
                "value": float(ks_t.statistic),
                "detail": f"ratio={ratio:.4f}",
            }
        )
        rows.append(
            {
                "subset": subset_name,
                "metric": "pit_ks_pvalue_two_piece",
                "value": float(ks_t.pvalue),
                "detail": f"ratio={ratio:.4f}",
            }
        )

    for level in COVERAGE_LEVELS:
        cov_g = central_coverage(y, mu, sigma, level, None)
        rows.append(
            {
                "subset": subset_name,
                "metric": f"coverage_{level}_gaussian",
                "value": cov_g,
                "detail": "",
            }
        )
        if ratio is not None and ratio != 1.0:
            cov_t = central_coverage(y, mu, sigma, level, ratio)
            rows.append(
                {
                    "subset": subset_name,
                    "metric": f"coverage_{level}_two_piece",
                    "value": cov_t,
                    "detail": f"ratio={ratio:.4f}",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate two-piece sigma down ratio on 2025 val only")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path")
    args = parser.parse_args()

    print("=== TWO-PIECE GAUSSIAN CALIBRATION (2025 validation only) ===")
    print(f"Model path file: {MODEL_PATH_FILE}")
    val_preds, models = load_validation_predictions()
    print(f"Loaded {len(val_preds)} validation rows across {val_preds['city'].nunique()} cities")
    print(f"Date range: {val_preds['date'].min()} to {val_preds['date'].max()}")

    z = (val_preds["y"] - val_preds["mu"]) / val_preds["sigma"]
    summer = val_preds[val_preds["month"].isin(SUMMER_MONTHS)].copy()
    nonsummer = val_preds[~val_preds["month"].isin(SUMMER_MONTHS)].copy()

    summer_z = (summer["y"] - summer["mu"]) / summer["sigma"]
    r_hat, r_robust, n_neg, n_pos = compute_r_hat(summer_z.to_numpy(dtype=float))
    n_summer = len(summer)

    print("\n=== POOLED SUMMER 2025 (Jun-Aug) ===")
    print(f"n_city_days={n_summer} | n_neg={n_neg} | n_pos={n_pos}")
    print(f"r_hat (std ratio):     {r_hat:.4f}")
    print(f"r_robust (|z| ratio):  {r_robust:.4f}")

    rows: list[dict[str, Any]] = []
    rows.append({"subset": "summer_2025", "metric": "n_city_days", "value": n_summer, "detail": ""})
    rows.append({"subset": "summer_2025", "metric": "n_neg", "value": n_neg, "detail": ""})
    rows.append({"subset": "summer_2025", "metric": "n_pos", "value": n_pos, "detail": ""})
    rows.append({"subset": "summer_2025", "metric": "r_hat_pooled", "value": r_hat, "detail": "std ratio"})
    rows.append(
        {
            "subset": "summer_2025",
            "metric": "r_hat_robust",
            "value": r_robust,
            "detail": "mean |z| ratio",
        }
    )

    print("\n=== PER-CITY SUMMER r_hat ===")
    print(f"{'city':<16} {'n':>5} {'r_hat':>8} {'r_robust':>10} {'n_neg':>6} {'n_pos':>6}")
    for city in sorted(summer["city"].unique()):
        sub = summer[summer["city"] == city]
        z_city = (sub["y"] - sub["mu"]) / sub["sigma"]
        city_r, city_rr, city_neg, city_pos = compute_r_hat(z_city.to_numpy(dtype=float))
        print(
            f"{city:<16} {len(sub):5d} {city_r:8.4f} {city_rr:10.4f} "
            f"{city_neg:6d} {city_pos:6d}"
        )
        rows.append(
            {
                "subset": f"summer_2025_{city}",
                "metric": "r_hat",
                "value": city_r,
                "detail": f"n={len(sub)}",
            }
        )

    nonsummer_z = (nonsummer["y"] - nonsummer["mu"]) / nonsummer["sigma"]
    ns_r, ns_rr, ns_neg, ns_pos = compute_r_hat(nonsummer_z.to_numpy(dtype=float))
    print("\n=== NON-SUMMER 2025 CONTROL ===")
    print(f"n_city_days={len(nonsummer)} | r_hat={ns_r:.4f} | r_robust={ns_rr:.4f}")
    rows.append({"subset": "nonsummer_2025", "metric": "r_hat_pooled", "value": ns_r, "detail": ""})
    rows.append(
        {"subset": "nonsummer_2025", "metric": "r_hat_robust", "value": ns_rr, "detail": ""}
    )

    if not np.isfinite(r_hat) or n_summer < MIN_SUMMER_SAMPLES or r_hat < MIN_R_HAT:
        print(
            "\nSKEW NOT SUPPORTED BY VALIDATION DATA - leave two_piece_sigma_down_ratio null"
        )
        if n_summer < MIN_SUMMER_SAMPLES:
            print(f"  reason: summer subsample has {n_summer} city-days (< {MIN_SUMMER_SAMPLES})")
        elif not np.isfinite(r_hat):
            print("  reason: insufficient positive/negative residuals for r_hat")
        else:
            print(f"  reason: pooled summer r_hat={r_hat:.4f} < {MIN_R_HAT}")
        return

    rows.extend(diagnostics_block("summer_2025", summer, r_hat))
    rows.extend(diagnostics_block("nonsummer_2025", nonsummer, r_hat))

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote calibration results to {out_path}")

    summer_diag = out_df[out_df["subset"] == "summer_2025"]
    print("\n=== SUMMER PIT / COVERAGE (gaussian vs two-piece) ===")
    for metric in [
        "pit_ks_stat_gaussian",
        "pit_ks_pvalue_gaussian",
        "pit_ks_stat_two_piece",
        "pit_ks_pvalue_two_piece",
        "coverage_50_gaussian",
        "coverage_50_two_piece",
        "coverage_80_gaussian",
        "coverage_80_two_piece",
        "coverage_90_gaussian",
        "coverage_90_two_piece",
    ]:
        hit = summer_diag[summer_diag["metric"] == metric]
        if not hit.empty:
            print(f"  {metric}: {float(hit['value'].iloc[0]):.4f}")

    print(
        f"\nRECOMMENDED two_piece_sigma_down_ratio: {r_hat:.4f} "
        "(set manually in config/deploy_config.json; not auto-applied)"
    )


if __name__ == "__main__":
    main()
