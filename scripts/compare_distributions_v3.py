#!/usr/bin/env python3
"""Compare Gaussian vs Student-t NGBoost calibration for v3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_ngboost as ng  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "models" / "ngboost_v3"
REPORT_DIR = PROJECT_ROOT / "reports" / "ngboost_calibration_v3"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "ngboost_v3_summary.md"
V2_CONFIG = PROJECT_ROOT / "models" / "ngboost_v2" / "model_config.json"

ALL_CITIES = list(ng.STATION_META.keys())
T_CRPS_EVAL_SAMPLES = 500


def student_t_crps_eval(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray,
    n_samples: int = T_CRPS_EVAL_SAMPLES,
) -> float:
    """Monte Carlo CRPS with more draws than train_ngboost default."""
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    df_arr = np.maximum(np.asarray(df, dtype=float), 1e-8)
    crps_vals = [
        ng.ensemble_crps(
            yi,
            student_t.rvs(df=di, loc=mi, scale=si, size=n_samples, random_state=42 + i),
        )
        for i, (yi, mi, si, di) in enumerate(zip(y_arr, mu_arr, sigma_arr, df_arr))
    ]
    return float(np.mean(crps_vals))


def reliability_high_prob_bin(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    df: np.ndarray | None = None,
    distribution: str = "gaussian",
    lo: float = 0.75,
    hi: float = 0.85,
) -> dict[str, float]:
    """Observed hit rate for predictions in [lo, hi] probability bin."""
    probs = ng.bucket_probs(mu, sigma, df=df, distribution=distribution)
    actual_idx = ng.actual_bucket_index(y)
    pred_probs: list[float] = []
    hits: list[float] = []
    for i in range(len(y)):
        for k_idx in range(probs.shape[1]):
            p = probs[i, k_idx]
            if lo <= p < hi:
                pred_probs.append(p)
                hits.append(1.0 if actual_idx[i] == k_idx else 0.0)
    if not hits:
        return {"n": 0, "mean_pred": float("nan"), "observed": float("nan"), "gap": float("nan")}
    mean_pred = float(np.mean(pred_probs))
    observed = float(np.mean(hits))
    return {"n": len(hits), "mean_pred": mean_pred, "observed": observed, "gap": observed - mean_pred}


def coverage_90_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    z = norm.ppf(0.95)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return float(np.mean((y >= lower) & (y <= upper)))


def coverage_90_student_t(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray,
) -> float:
    vals: list[float] = []
    for j in range(len(y)):
        lo = student_t.ppf(0.05, df=df[j], loc=mu[j], scale=sigma[j])
        hi = student_t.ppf(0.95, df=df[j], loc=mu[j], scale=sigma[j])
        vals.append(float(lo <= y[j] <= hi))
    return float(np.mean(vals))


def load_hparams() -> dict:
    config_path = OUTPUT_DIR / "model_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hp = config.get("hyperparameters", {})
        return {
            "max_depth": int(hp.get("max_depth", 4)),
            "learning_rate": float(hp.get("learning_rate", 0.01)),
            "minibatch_frac": float(hp.get("minibatch_frac", 1.0)),
            "label": str(hp.get("label", "loaded")),
        }
    return {"max_depth": 4, "learning_rate": 0.01, "minibatch_frac": 1.0, "label": "default"}


def apply_stage1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lgb_path = OUTPUT_DIR / "lgb_stage1.pkl"
    if not lgb_path.exists():
        raise FileNotFoundError(f"Missing {lgb_path} — run main v3 training first")
    lgb_model = joblib.load(lgb_path)
    stage1_cols = ng.FEATURE_COLS_STAGE1
    for split in (train_df, val_df, test_df):
        split["lgb_tmax_pred"] = lgb_model.predict(split[stage1_cols])
    print(f"Loaded stage-1 LGB from {lgb_path}")
    return train_df, val_df, test_df


def load_or_train_gaussian(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hparams: dict,
) -> tuple[object, float, object, float]:
    config_path = OUTPUT_DIR / "model_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("distribution") == "gaussian" and (OUTPUT_DIR / "ngboost_global.pkl").exists():
            model = joblib.load(OUTPUT_DIR / "ngboost_global.pkl")
            scaler = joblib.load(OUTPUT_DIR / "feature_scaler.pkl")
            X_val = ng.transform_features(scaler, val_df, ng.FEATURE_COLS_GLOBAL)
            val_crps = ng.eval_model_crps(model, X_val, val_df[ng.TARGET])
            print(f"Loaded Gaussian winner from v3 artifacts (val CRPS {val_crps:.4f})")
            return model, val_crps, scaler, float(config.get("sigma_calibration_k", 1.0))

    print("\n--- Training Gaussian ---")
    model, _, val_crps, scaler = ng.train_global(
        train_df, val_df, ng.FEATURE_COLS_GLOBAL, hparams, dist=ng.Normal, verbose=False
    )
    print(f"Gaussian val CRPS: {val_crps:.4f}")
    return model, val_crps, scaler, 1.0


def per_city_test_metrics(
    test_df: pd.DataFrame,
    g_model: object,
    g_scaler: object,
    g_sigma_k: float,
    t_model: object,
    t_scaler: object,
) -> list[dict]:
    X_test_g = ng.transform_features(g_scaler, test_df, ng.FEATURE_COLS_GLOBAL)
    X_test_t = ng.transform_features(t_scaler, test_df, ng.FEATURE_COLS_GLOBAL)
    g_mu, g_sigma, _ = ng.predict_dist_params(g_model, X_test_g)
    t_mu, t_sigma, t_df = ng.predict_dist_params(t_model, X_test_t)
    g_sigma_cal = ng.apply_sigma_calibration(g_sigma, g_sigma_k)
    y_test = test_df[ng.TARGET].to_numpy(dtype=float)
    cities_col = test_df["city"].to_numpy()

    rows: list[dict] = []
    for city in ALL_CITIES:
        mask = cities_col == city
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        yc = y_test[mask]
        gc_mu, gc_sig = g_mu[mask], g_sigma_cal[mask]
        tc_mu, tc_sig, tc_df = t_mu[mask], t_sigma[mask], t_df[mask]

        g_mae = float(np.mean(np.abs(gc_mu - yc)))
        t_mae = float(np.mean(np.abs(tc_mu - yc)))
        g_crps = ng.gaussian_crps(yc, gc_mu, gc_sig)
        t_crps = student_t_crps_eval(yc, tc_mu, tc_sig, tc_df)
        g_cov = coverage_90_gaussian(yc, gc_mu, gc_sig)
        t_cov = coverage_90_student_t(yc, tc_mu, tc_sig, tc_df)
        g_bhr = ng.modal_bucket_hit_rate(yc, gc_mu, gc_sig)
        t_bhr = ng.modal_bucket_hit_rate(yc, tc_mu, tc_sig, df=tc_df, distribution="student_t")

        rows.append(
            {
                "city": city,
                "n": n,
                "g_mae": g_mae,
                "t_mae": t_mae,
                "g_crps": g_crps,
                "t_crps": t_crps,
                "g_90cov": g_cov,
                "t_90cov": t_cov,
                "g_bkt_hr": g_bhr,
                "t_bkt_hr": t_bhr,
            }
        )
    return rows


def classify_city(row: dict) -> str:
    """Well-calibrated if both distributions within 5pp of 90% coverage."""
    flags: list[str] = []
    for prefix in ("g", "t"):
        cov = row[f"{prefix}_90cov"]
        if abs(cov - 0.90) > 0.05:
            flags.append(f"{prefix}_90cov")
    if row["city"] in ("san_francisco", "miami", "seattle"):
        return "focus_" + ("ok" if not flags else "poor")
    return "ok" if not flags else "poor"


def write_summary_md(
    v2_config: dict,
    g_val_crps: float,
    t_val_crps: float,
    g_cal: dict,
    t_cal: dict,
    g_cal_test: dict,
    t_cal_test: dict,
    g_test_crps: float,
    t_test_crps: float,
    median_df_val: float,
    median_df_test: float,
    rel_val: dict,
    rel_test: dict,
    per_city: list[dict],
) -> None:
    crps_winner = "student_t" if t_val_crps < g_val_crps else "gaussian"
    brier_winner = "student_t" if t_cal.get("bucket_brier", 999) < g_cal.get("bucket_brier", 999) else "gaussian"

    lines = [
        "# NGBoost v3 Summary\n\n",
        "## 1. v2 baseline\n\n",
        f"- Distribution: **{v2_config.get('distribution', '?')}**\n",
        f"- Val CRPS: **{v2_config.get('val_crps', '?')}**\n",
        f"- Sigma calibration k: **{v2_config.get('sigma_calibration_k', '?')}**\n",
        "- Known issues: fat Q-Q tails beyond ±2.5σ, overconfidence in 0.75–0.85 predicted-prob bin, "
        "PIT KS p≈0.001, SF ~79.9% nominal-90% coverage, 20.1% backtest win rate when disagreeing with market.\n\n",
        "## 2. v3 Gaussian vs Student-t\n\n",
        "| Metric | Gaussian | Student-t |\n",
        "|--------|----------|----------|\n",
        f"| Val CRPS | {g_val_crps:.4f} | {t_val_crps:.4f} |\n",
        f"| Test CRPS (500-draw t) | {g_test_crps:.4f} | {t_test_crps:.4f} |\n",
        f"| Val PIT KS (p) | {g_cal.get('pit_ks_stat', float('nan')):.3f} ({g_cal.get('pit_ks_p', float('nan')):.4f}) | "
        f"{t_cal.get('pit_ks_stat', float('nan')):.3f} ({t_cal.get('pit_ks_p', float('nan')):.4f}) |\n",
        f"| Test PIT KS (p) | {g_cal_test.get('pit_ks_stat', float('nan')):.3f} ({g_cal_test.get('pit_ks_p', float('nan')):.4f}) | "
        f"{t_cal_test.get('pit_ks_stat', float('nan')):.3f} ({t_cal_test.get('pit_ks_p', float('nan')):.4f}) |\n",
    ]

    for label, g_cov, t_cov in [
        ("50%", g_cal.get("coverage", {}).get(50), t_cal.get("coverage", {}).get(50)),
        ("80%", g_cal.get("coverage", {}).get(80), t_cal.get("coverage", {}).get(80)),
        ("90%", g_cal.get("coverage", {}).get(90), t_cal.get("coverage", {}).get(90)),
        ("95%", g_cal.get("coverage", {}).get(95), t_cal.get("coverage", {}).get(95)),
    ]:
        g_s = f"{g_cov:.1f}%" if g_cov is not None else "—"
        t_s = f"{t_cov:.1f}%" if t_cov is not None else "—"
        lines.append(f"| Val {label} coverage | {g_s} | {t_s} |\n")

    for label, g_cov, t_cov in [
        ("50%", g_cal_test.get("coverage", {}).get(50), t_cal_test.get("coverage", {}).get(50)),
        ("80%", g_cal_test.get("coverage", {}).get(80), t_cal_test.get("coverage", {}).get(80)),
        ("90%", g_cal_test.get("coverage", {}).get(90), t_cal_test.get("coverage", {}).get(90)),
        ("95%", g_cal_test.get("coverage", {}).get(95), t_cal_test.get("coverage", {}).get(95)),
    ]:
        g_s = f"{g_cov:.1f}%" if g_cov is not None else "—"
        t_s = f"{t_cov:.1f}%" if t_cov is not None else "—"
        lines.append(f"| Test {label} coverage | {g_s} | {t_s} |\n")

    lines.extend(
        [
            f"| Bucket Brier (val) | {g_cal.get('bucket_brier', float('nan')):.4f} | {t_cal.get('bucket_brier', float('nan')):.4f} |\n",
            f"| Bucket Brier (test) | {g_cal_test.get('bucket_brier', float('nan')):.4f} | {t_cal_test.get('bucket_brier', float('nan')):.4f} |\n",
            f"| Modal bucket HR (val) | {100 * g_cal.get('modal_bucket_hr', 0):.1f}% | {100 * t_cal.get('modal_bucket_hr', 0):.1f}% |\n",
            f"| Modal bucket HR (test) | {100 * g_cal_test.get('modal_bucket_hr', 0):.1f}% | {100 * t_cal_test.get('modal_bucket_hr', 0):.1f}% |\n",
            f"| Median df | — | val {median_df_val:.1f}, test {median_df_test:.1f} |\n",
            f"| Reliability 0.75–0.85 bin (val) | n={rel_val['gaussian']['n']}, obs={rel_val['gaussian']['observed']:.3f} vs pred {rel_val['gaussian']['mean_pred']:.3f} | "
            f"n={rel_val['student_t']['n']}, obs={rel_val['student_t']['observed']:.3f} vs pred {rel_val['student_t']['mean_pred']:.3f} |\n",
            f"| Reliability 0.75–0.85 bin (test) | n={rel_test['gaussian']['n']}, obs={rel_test['gaussian']['observed']:.3f} | "
            f"n={rel_test['student_t']['n']}, obs={rel_test['student_t']['observed']:.3f} |\n",
            f"\n**CRPS winner (val):** {crps_winner}\n\n",
            f"**Bucket Brier winner (val):** {brier_winner}\n\n",
            "## 3. Per-city test set\n\n",
            "| City | N | G_MAE | T_MAE | G_CRPS | T_CRPS | G_90cov | T_90cov | G_BktHR | T_BktHR | Status |\n",
            "|------|--:|------:|------:|-------:|-------:|--------:|--------:|--------:|--------:|--------|\n",
        ]
    )

    for row in per_city:
        status = classify_city(row)
        lines.append(
            f"| {row['city']} | {row['n']} | {row['g_mae']:.2f} | {row['t_mae']:.2f} | "
            f"{row['g_crps']:.3f} | {row['t_crps']:.3f} | {100 * row['g_90cov']:.1f}% | "
            f"{100 * row['t_90cov']:.1f}% | {100 * row['g_bkt_hr']:.1f}% | {100 * row['t_bkt_hr']:.1f}% | {status} |\n"
        )

    focus = [r for r in per_city if r["city"] in ("san_francisco", "miami", "seattle")]
    lines.append("\n### Focus cities\n\n")
    for row in focus:
        lines.append(
            f"- **{row['city']}**: Gaussian 90% cov {100 * row['g_90cov']:.1f}%, Student-t {100 * row['t_90cov']:.1f}%; "
            f"modal bucket HR G={100 * row['g_bkt_hr']:.1f}% T={100 * row['t_bkt_hr']:.1f}%\n"
        )

    rec_dist = brier_winner if brier_winner != crps_winner else crps_winner
    if crps_winner != brier_winner:
        rec_dist = f"{brier_winner} for trading (better bucket Brier despite CRPS favoring {crps_winner})"
    else:
        rec_dist = crps_winner

    exclude = [r["city"] for r in per_city if classify_city(r).endswith("poor")]
    lines.extend(
        [
            "\n## 4. Recommendation\n\n",
            f"- **Distribution for trading:** {rec_dist}\n",
            f"- **Cities to down-weight or exclude:** {', '.join(exclude) if exclude else 'none flagged on 90% coverage rule'}\n",
            "- **Backtest model path:** not updated (v2 remains in `reports/backtest_model_path.txt` until explicitly switched)\n",
            "\nPlots: `reports/ngboost_calibration_v3/gaussian/` vs `student_t/` (val), "
            "`gaussian_test/` vs `student_t_test/` (test).\n",
        ]
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}")


def main() -> None:
    print("=== Distribution Comparison: Gaussian vs Student-t (v3) ===\n")

    if not ng.HAS_TDIST or ng.TDist is None:
        print("ERROR: Student-t not available. Upgrade ngboost.")
        sys.exit(1)

    df = ng.assemble_dataset(ALL_CITIES)
    df = ng.drop_incomplete_rows(df)
    train_df, val_df, test_df = ng.temporal_split(df)
    train_df, val_df, test_df, fill_medians = ng.fill_median_features(
        train_df, val_df, test_df, ng.MEDIAN_FILL_COLS
    )
    train_df, val_df, test_df = apply_stage1(train_df, val_df, test_df)
    hparams = load_hparams()

    g_model, g_val_crps, g_scaler, saved_sigma_k = load_or_train_gaussian(train_df, val_df, hparams)

    print("\n--- Training Student-t ---")
    t_model, _, t_val_crps, t_scaler = ng.train_global(
        train_df, val_df, ng.FEATURE_COLS_GLOBAL, hparams, dist=ng.TDist, verbose=False
    )
    print(f"Student-t val CRPS: {t_val_crps:.4f}")

    print("\n--- Gaussian calibration (val) ---")
    g_cal = ng.run_calibration(
        g_model, val_df, ng.FEATURE_COLS_GLOBAL, "global",
        REPORT_DIR / "gaussian", "gaussian", scaler=g_scaler,
    )
    g_sigma_k = float(g_cal.get("sigma_calibration_k", saved_sigma_k))

    print("\n--- Student-t calibration (val) ---")
    t_cal = ng.run_calibration(
        t_model, val_df, ng.FEATURE_COLS_GLOBAL, "global",
        REPORT_DIR / "student_t", "student_t", scaler=t_scaler,
    )

    print("\n--- Gaussian TEST calibration ---")
    g_cal_test = ng.run_calibration(
        g_model, test_df, ng.FEATURE_COLS_GLOBAL, "global",
        REPORT_DIR / "gaussian_test", "gaussian", scaler=g_scaler,
    )

    print("\n--- Student-t TEST calibration ---")
    t_cal_test = ng.run_calibration(
        t_model, test_df, ng.FEATURE_COLS_GLOBAL, "global",
        REPORT_DIR / "student_t_test", "student_t", scaler=t_scaler,
    )

    # Test CRPS with 500-draw t evaluation
    X_test_g = ng.transform_features(g_scaler, test_df, ng.FEATURE_COLS_GLOBAL)
    X_test_t = ng.transform_features(t_scaler, test_df, ng.FEATURE_COLS_GLOBAL)
    y_test = test_df[ng.TARGET].to_numpy(dtype=float)
    g_mu, g_sigma, _ = ng.predict_dist_params(g_model, X_test_g)
    t_mu, t_sigma, t_df = ng.predict_dist_params(t_model, X_test_t)
    g_sigma_cal = ng.apply_sigma_calibration(g_sigma, g_sigma_k)
    g_test_crps = ng.gaussian_crps(y_test, g_mu, g_sigma_cal)
    t_test_crps = student_t_crps_eval(y_test, t_mu, t_sigma, t_df)

    X_val_t = ng.transform_features(t_scaler, val_df, ng.FEATURE_COLS_GLOBAL)
    _, _, df_val = ng.predict_dist_params(t_model, X_val_t)
    median_df_val = float(np.median(df_val)) if df_val is not None else float("nan")
    median_df_test = float(np.median(t_df)) if t_df is not None else float("nan")

    X_val_g = ng.transform_features(g_scaler, val_df, ng.FEATURE_COLS_GLOBAL)
    g_mu_v, g_sig_v, _ = ng.predict_dist_params(g_model, X_val_g)
    t_mu_v, t_sig_v, t_df_v = ng.predict_dist_params(t_model, X_val_t)
    y_val = val_df[ng.TARGET].to_numpy(float)

    rel_val = {
        "gaussian": reliability_high_prob_bin(
            y_val, g_mu_v, ng.apply_sigma_calibration(g_sig_v, g_sigma_k), distribution="gaussian"
        ),
        "student_t": reliability_high_prob_bin(
            y_val, t_mu_v, t_sig_v, df=t_df_v, distribution="student_t"
        ),
    }
    rel_test = {
        "gaussian": reliability_high_prob_bin(
            y_test, g_mu, g_sigma_cal, distribution="gaussian"
        ),
        "student_t": reliability_high_prob_bin(
            y_test, t_mu, t_sigma, df=t_df, distribution="student_t"
        ),
    }

    per_city = per_city_test_metrics(test_df, g_model, g_scaler, g_sigma_k, t_model, t_scaler)

    print("\n=== Per-City Test Set Comparison ===")
    header = (
        f"{'City':<16} {'N':>5} {'G_MAE':>6} {'T_MAE':>6} "
        f"{'G_CRPS':>7} {'T_CRPS':>7} {'G_90cov':>7} {'T_90cov':>7} "
        f"{'G_BktHR':>7} {'T_BktHR':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in per_city:
        print(
            f"{row['city']:<16} {row['n']:5d} "
            f"{row['g_mae']:6.2f} {row['t_mae']:6.2f} "
            f"{row['g_crps']:7.3f} {row['t_crps']:7.3f} "
            f"{row['g_90cov']:7.1%} {row['t_90cov']:7.1%} "
            f"{row['g_bkt_hr']:7.1%} {row['t_bkt_hr']:7.1%}"
        )

    t_save_dir = OUTPUT_DIR / "student_t_model"
    t_save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(t_model, t_save_dir / "ngboost_global.pkl")
    joblib.dump(t_scaler, t_save_dir / "feature_scaler.pkl")
    t_config = {
        "model_type": "global",
        "distribution": "student_t",
        "feature_columns": ng.FEATURE_COLS_GLOBAL,
        "cities": ALL_CITIES,
        "val_crps": round(t_val_crps, 4),
        "hyperparameters": hparams,
        "median_df_val": round(median_df_val, 4),
        "median_df_test": round(median_df_test, 4),
        "nan_fill_medians": fill_medians,
    }
    (t_save_dir / "model_config.json").write_text(json.dumps(t_config, indent=2), encoding="utf-8")
    print(f"\nStudent-t artifacts saved to {t_save_dir}")

    comparison_json = {
        "gaussian_val_crps": g_val_crps,
        "student_t_val_crps": t_val_crps,
        "gaussian_test_crps": g_test_crps,
        "student_t_test_crps": t_test_crps,
        "gaussian_cal": g_cal,
        "student_t_cal": t_cal,
        "gaussian_cal_test": g_cal_test,
        "student_t_cal_test": t_cal_test,
        "median_df_val": median_df_val,
        "median_df_test": median_df_test,
        "reliability_val": rel_val,
        "reliability_test": rel_test,
        "per_city_test": per_city,
    }
    (REPORT_DIR / "comparison_results.json").write_text(
        json.dumps(comparison_json, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(f"Gaussian val CRPS:  {g_val_crps:.4f}")
    print(f"Student-t val CRPS: {t_val_crps:.4f}")
    print(f"Gaussian test CRPS: {g_test_crps:.4f}")
    print(f"Student-t test CRPS (500-draw): {t_test_crps:.4f}")
    print(f"Gaussian PIT KS p (val):  {g_cal.get('pit_ks_p', float('nan')):.4f}")
    print(f"Student-t PIT KS p (val): {t_cal.get('pit_ks_p', float('nan')):.4f}")
    print(f"Median Student-t df (val/test): {median_df_val:.1f} / {median_df_test:.1f}")
    crps_winner = "student_t" if t_val_crps < g_val_crps else "gaussian"
    brier_winner = "student_t" if t_cal.get("bucket_brier", 999) < g_cal.get("bucket_brier", 999) else "gaussian"
    print(f"CRPS winner: {crps_winner}")
    print(f"Bucket Brier winner (val): {brier_winner}")

    v2_config = json.loads(V2_CONFIG.read_text(encoding="utf-8")) if V2_CONFIG.exists() else {}
    write_summary_md(
        v2_config,
        g_val_crps,
        t_val_crps,
        g_cal,
        t_cal,
        g_cal_test,
        t_cal_test,
        g_test_crps,
        t_test_crps,
        median_df_val,
        median_df_test,
        rel_val,
        rel_test,
        per_city,
    )

    print("\nCheck reports/ngboost_calibration_v3/ for diagnostic plots.")


if __name__ == "__main__":
    main()
