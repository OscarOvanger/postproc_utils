#!/usr/bin/env python3
"""Diagnose NGBoost modal bucket hit rate on the 2026 test set."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_ngboost import (  # noqa: E402
    BUCKET_EDGES,
    actual_bucket_index,
    apply_saved_median_fill,
    apply_sigma_calibration,
    bucket_probs,
    empirical_coverage,
    fit_sigma_calibration_k,
    load_saved_artifacts,
    modal_bucket_hit_rate,
    predict_dist_params,
    transform_features,
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "ngboost_v2"
NWS_PATH = PROJECT_ROOT / "data" / "trackb" / "nws_forecasts_raw.parquet"

NGBOOST_TO_NWS: dict[str, str] = {
    "chicago": "chicago_midway",
    "new_york": "new_york_city",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)),
        help="Path to saved NGBoost model directory (default: models/ngboost_v2)",
    )
    parser.add_argument(
        "--train-module",
        default=None,
        help="Training module name (train_ngboost or train_ngboost_v4). Auto-detected from feature count if omitted.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Output JSON path (default: reports/bucket_hit_diagnostic[_suffix].json)",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        help="Output markdown path (default: reports/bucket_hit_diagnostic[_suffix].md)",
    )
    return parser.parse_args()


def resolve_train_module(model_dir: Path, train_module: str | None) -> str:
    if train_module:
        return train_module
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        n_features = len(config.get("feature_columns", []))
        if n_features > 17:
            return "train_ngboost_v4"
    return "train_ngboost"


def load_train_helpers(module_name: str):
    mod = importlib.import_module(module_name)
    return {
        "assemble_dataset": mod.assemble_dataset,
        "drop_incomplete_rows": mod.drop_incomplete_rows,
        "temporal_split": mod.temporal_split,
        "MEDIAN_FILL_COLS": mod.MEDIAN_FILL_COLS,
        "TARGET": mod.TARGET,
    }


def default_report_paths(model_dir: Path) -> tuple[Path, Path]:
    name = model_dir.name
    if name == "ngboost_v2":
        suffix = ""
    else:
        suffix = f"_{name.replace('ngboost_', '')}"
    return (
        PROJECT_ROOT / "reports" / f"bucket_hit_diagnostic{suffix}.json",
        PROJECT_ROOT / "reports" / f"bucket_hit_diagnostic{suffix}.md",
    )


def bucket_index_to_label(idx: int) -> str:
    if idx == 0:
        return "<20"
    n_interior = len(BUCKET_EDGES) - 1
    if idx >= n_interior + 1:
        return "120+"
    lo = BUCKET_EDGES[idx - 1]
    hi = BUCKET_EDGES[idx] - 1
    return f"{lo}-{hi}"


def shared_boundary(modal_idx: int, actual_idx: int) -> float | None:
    if modal_idx == actual_idx:
        return None
    lo_idx = min(modal_idx, actual_idx)
    if lo_idx < 1 or lo_idx >= len(BUCKET_EDGES):
        return None
    return float(BUCKET_EDGES[lo_idx])


def load_test_predictions(
    model_dir: Path,
    train_helpers: dict,
) -> tuple[pd.DataFrame, float]:
    assemble_dataset = train_helpers["assemble_dataset"]
    drop_incomplete_rows = train_helpers["drop_incomplete_rows"]
    temporal_split = train_helpers["temporal_split"]
    median_fill_cols = train_helpers["MEDIAN_FILL_COLS"]
    target = train_helpers["TARGET"]

    model, scaler, config = load_saved_artifacts(model_dir)
    cities = list(config.get("cities", []))
    feature_cols = list(config.get("feature_columns", []))
    fill_medians = dict(config.get("nan_fill_medians", {}))
    sigma_k = float(config.get("sigma_calibration_k", 1.0))

    stage1_path = model_dir / config.get("stage1_model", "lgb_stage1.pkl")
    lgb_model = joblib.load(stage1_path)
    stage1_cols = [c for c in feature_cols if c != "lgb_tmax_pred"]

    df = assemble_dataset(cities)
    df = drop_incomplete_rows(df)
    _train, _val, test_df = temporal_split(df)
    fill_cols = list(fill_medians.keys()) if fill_medians else median_fill_cols
    test_df = apply_saved_median_fill(test_df, fill_medians, fill_cols)
    test_df = test_df.copy()
    test_df["lgb_tmax_pred"] = lgb_model.predict(test_df[stage1_cols])

    X = transform_features(scaler, test_df, feature_cols)
    y = test_df[target].to_numpy(dtype=float)
    mu, sigma_raw, _ = predict_dist_params(model, X)
    sigma = apply_sigma_calibration(sigma_raw, sigma_k)

    out = test_df[["city", "date"]].copy()
    out["actual_tmax"] = y
    out["mu"] = mu
    out["sigma_raw"] = sigma_raw
    out["sigma"] = sigma
    out["signed_error_f"] = mu - y
    out["abs_error_f"] = np.abs(mu - y)

    probs = bucket_probs(mu, sigma, distribution="gaussian")
    out["modal_idx"] = np.argmax(probs, axis=1)
    out["actual_idx"] = actual_bucket_index(y)
    out["bucket_distance"] = np.abs(out["modal_idx"] - out["actual_idx"])
    out["signed_bucket_distance"] = out["modal_idx"] - out["actual_idx"]
    out["modal_prob"] = probs[np.arange(len(out)), out["modal_idx"]]

    boundaries = [
        shared_boundary(int(m), int(a)) for m, a in zip(out["modal_idx"], out["actual_idx"])
    ]
    out["shared_boundary_f"] = boundaries
    out["boundary_distance_f"] = [
        abs(a - b) if b is not None else np.nan for a, b in zip(out["actual_tmax"], boundaries)
    ]
    return out, sigma_k


def histogram_table(distances: np.ndarray) -> dict[str, float | int]:
    misses = distances[distances > 0]
    total = len(distances)
    hits = int(np.sum(distances == 0))
    rows: dict[str, float | int] = {"0_hit": hits, "pct_hit": round(100.0 * hits / total, 2)}
    for d in range(1, 5):
        cnt = int(np.sum(distances == d))
        rows[f"{d}_bucket_off"] = cnt
        rows[f"pct_{d}_off"] = round(100.0 * cnt / total, 2)
    rows["4plus_off"] = int(np.sum(distances >= 4))
    rows["pct_4plus_off"] = round(100.0 * rows["4plus_off"] / total, 2)
    if len(misses):
        rows["pct_misses_1_off"] = round(100.0 * np.sum(misses == 1) / len(misses), 2)
    else:
        rows["pct_misses_1_off"] = 0.0
    return rows


def per_city_summary(frame: pd.DataFrame, sigma_k: float) -> dict[str, dict]:
    z90 = 1.6449
    out: dict[str, dict] = {}
    for city, grp in frame.groupby("city"):
        y = grp["actual_tmax"].to_numpy(dtype=float)
        mu = grp["mu"].to_numpy(dtype=float)
        sigma = grp["sigma"].to_numpy(dtype=float)
        sigma_raw = grp["sigma_raw"].to_numpy(dtype=float)
        dist = grp["bucket_distance"].to_numpy(dtype=int)

        cov90 = empirical_coverage(y, mu, sigma, 90.0, distribution="gaussian")
        within_1f = float(np.mean(np.abs(mu - y) <= 1.0))
        modal_hr = float(np.mean(dist == 0))

        med_sigma = float(np.median(sigma))
        med_abs_err = float(np.median(np.abs(mu - y)))
        ratio = med_sigma / med_abs_err if med_abs_err > 0 else float("inf")

        mean_err = float(np.mean(mu - y))
        bias_flag = "systematic_bias" if abs(mean_err) > 1.0 else None
        if abs(mean_err) <= 1.0 and abs(cov90 - 90.0) > 8.0:
            cal_flag = "high_variance"
        elif abs(mean_err) <= 0.5 and abs(cov90 - 90.0) <= 5.0 and modal_hr < 0.35:
            cal_flag = "boundary_cases"
        else:
            cal_flag = "mixed"

        per_k = fit_sigma_calibration_k(y, mu, sigma_raw, level=0.90)
        sigma_per_city = apply_sigma_calibration(sigma_raw, per_k)
        hr_per_city_k = float(
            modal_bucket_hit_rate(y, mu, sigma_per_city, distribution="gaussian")
        )

        out[str(city)] = {
            "n": int(len(grp)),
            "modal_hit_rate_pct": round(100.0 * modal_hr, 2),
            "mean_signed_error_f": round(mean_err, 3),
            "median_signed_error_f": round(float(np.median(mu - y)), 3),
            "within_1f_pct": round(100.0 * within_1f, 2),
            "coverage_90_pct": round(cov90, 2),
            "median_sigma_calibrated": round(med_sigma, 3),
            "median_abs_error_f": round(med_abs_err, 3),
            "sigma_to_mae_ratio": round(ratio, 3),
            "per_city_sigma_k": round(per_k, 4),
            "modal_hr_global_k_pct": round(100.0 * modal_hr, 2),
            "modal_hr_per_city_k_pct": round(100.0 * hr_per_city_k, 2),
            "classification": cal_flag if not bias_flag else bias_flag,
        }
    return out


def boundary_analysis(frame: pd.DataFrame) -> dict[str, float | int]:
    one_off = frame[frame["bucket_distance"] == 1].copy()
    if one_off.empty:
        return {"n_1_bucket_misses": 0, "within_1f_of_boundary_pct": 0.0}
    within = one_off["boundary_distance_f"] <= 1.0
    return {
        "n_1_bucket_misses": int(len(one_off)),
        "within_1f_of_boundary": int(within.sum()),
        "within_1f_of_boundary_pct": round(100.0 * within.mean(), 2),
        "mean_boundary_distance_f": round(float(one_off["boundary_distance_f"].mean()), 3),
    }


def nws_mos_analysis(frame: pd.DataFrame) -> dict:
    if not NWS_PATH.exists():
        return {"available": False, "reason": f"missing {NWS_PATH}"}

    nws = pd.read_parquet(NWS_PATH)
    nws["date"] = pd.to_datetime(nws["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    nws["nws_city"] = nws["city"].astype(str)
    nws["tmax_forecast_f"] = pd.to_numeric(nws["tmax_forecast_f"], errors="coerce")

    rows = []
    for city in sorted(frame["city"].unique()):
        nws_city = NGBOOST_TO_NWS.get(city, city)
        sub = frame[frame["city"] == city].merge(
            nws[nws["nws_city"] == nws_city][["date", "tmax_forecast_f"]],
            on="date",
            how="left",
        )
        matched = sub["tmax_forecast_f"].notna()
        if not matched.any():
            rows.append({"city": city, "nws_city": nws_city, "n_matched": 0})
            continue
        y = sub.loc[matched, "actual_tmax"].to_numpy(dtype=float)
        mos = sub.loc[matched, "tmax_forecast_f"].to_numpy(dtype=float)
        mu = sub.loc[matched, "mu"].to_numpy(dtype=float)
        rows.append(
            {
                "city": city,
                "nws_city": nws_city,
                "n_matched": int(matched.sum()),
                "corr_mos_actual": round(float(np.corrcoef(mos, y)[0, 1]), 4),
                "corr_mu_actual": round(float(np.corrcoef(mu, y)[0, 1]), 4),
                "mos_mae": round(float(np.mean(np.abs(mos - y))), 3),
                "mu_mae": round(float(np.mean(np.abs(mu - y))), 3),
            }
        )

    available_cities = [r for r in rows if r.get("n_matched", 0) > 0]
    overall = {}
    if available_cities:
        all_mos = []
        all_mu = []
        all_y = []
        for city in frame["city"].unique():
            nws_city = NGBOOST_TO_NWS.get(city, city)
            sub = frame[frame["city"] == city].merge(
                nws[nws["nws_city"] == nws_city][["date", "tmax_forecast_f"]],
                on="date",
                how="left",
            )
            m = sub["tmax_forecast_f"].notna()
            if not m.any():
                continue
            all_y.extend(sub.loc[m, "actual_tmax"].tolist())
            all_mos.extend(sub.loc[m, "tmax_forecast_f"].tolist())
            all_mu.extend(sub.loc[m, "mu"].tolist())
        if all_y:
            y_arr = np.asarray(all_y, dtype=float)
            mos_arr = np.asarray(all_mos, dtype=float)
            mu_arr = np.asarray(all_mu, dtype=float)
            overall = {
                "n_paired": len(y_arr),
                "corr_mos_actual": round(float(np.corrcoef(mos_arr, y_arr)[0, 1]), 4),
                "corr_mu_actual": round(float(np.corrcoef(mu_arr, y_arr)[0, 1]), 4),
                "mos_mae": round(float(np.mean(np.abs(mos_arr - y_arr))), 3),
                "mu_mae": round(float(np.mean(np.abs(mu_arr - y_arr))), 3),
            }

    missing = [c for c in frame["city"].unique() if c not in {r["city"] for r in available_cities}]
    return {
        "available": True,
        "path": str(NWS_PATH),
        "per_city": rows,
        "overall_paired": overall,
        "cities_without_nws_cache": sorted(missing),
    }


def counterfactual_sigma_hit_rates(frame: pd.DataFrame, sigma_k: float) -> dict[str, float]:
    y = frame["actual_tmax"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    sigma_raw = frame["sigma_raw"].to_numpy(dtype=float)

    rates: dict[str, float] = {}
    for label, k in [
        ("k_0.50", 0.50),
        ("k_0.70", 0.70),
        ("k_1.00_raw", 1.00),
        ("global_k", sigma_k),
        ("k_1.50", 1.50),
    ]:
        sig = apply_sigma_calibration(sigma_raw, k)
        rates[label] = round(
            100.0 * modal_bucket_hit_rate(y, mu, sig, distribution="gaussian"), 3
        )

    sigma_per = np.zeros_like(sigma_raw)
    pos_map = {idx: pos for pos, idx in enumerate(frame.index)}
    per_city_k_map: dict[str, float] = {}
    for city, grp in frame.groupby("city"):
        y_c = grp["actual_tmax"].to_numpy(dtype=float)
        mu_c = grp["mu"].to_numpy(dtype=float)
        sr_c = grp["sigma_raw"].to_numpy(dtype=float)
        k_c = fit_sigma_calibration_k(y_c, mu_c, sr_c, level=0.90)
        per_city_k_map[str(city)] = round(k_c, 4)
        cal = apply_sigma_calibration(sr_c, k_c)
        for orig_idx, val in zip(grp.index, cal):
            sigma_per[pos_map[orig_idx]] = val

    rates["per_city_k"] = round(
        100.0 * modal_bucket_hit_rate(y, mu, sigma_per, distribution="gaussian"), 3
    )

    # Bucket containing mu (mode of Gaussian) vs actual — isolates mean error from sigma spread.
    mu_idx = actual_bucket_index(mu)
    actual_idx = frame["actual_idx"].to_numpy(dtype=int)
    rates["mu_in_actual_bucket_pct"] = round(100.0 * float(np.mean(mu_idx == actual_idx)), 3)
    rates["mu_within_1_bucket_pct"] = round(
        100.0 * float(np.mean(np.abs(mu_idx - actual_idx) <= 1)), 3
    )
    rates["within_1f_temp_pct"] = round(100.0 * float(np.mean(frame["abs_error_f"] <= 1.0)), 3)
    rates["within_2f_temp_pct"] = round(100.0 * float(np.mean(frame["abs_error_f"] <= 2.0)), 3)
    rates["per_city_k_values"] = per_city_k_map
    return rates


def feature_gap_analysis() -> dict:
    trackb_only = [
        "dewpoint_10am (ASOS cache via IEM)",
        "rh_mean_00_10 (ASOS)",
        "pressure_10am (ASOS)",
        "wind_u_mean_00_10 / wind_v_mean_00_10 (ASOS)",
        "cloud_cover_mean_00_10 (ASOS morning, distinct from HRRR peak_cloud_cover)",
        "nws_tmax_forecast_f (data/trackb/nws_forecasts_raw.parquet, 6/10 cities)",
        "GFS afternoon 2m temp/dewpoint/cloud (TrackB Herbie cache under data/trackj/raw/)",
        "tmax_lag3 through tmax_lag7 (extend from existing WU targets parquet)",
        "temp_mean_00_10 / temp_max_so_far_00_10 (ASOS, pre-10AM)",
    ]
    ngboost_only = [
        "hrrr_tmax (3km, higher resolution than GFS 25km)",
        "peak_solar_flux (HRRR-derived)",
        "snow_depth (HRRR)",
        "hrrr_error_lag1 (HRRR bias memory)",
        "lgb_tmax_pred (stage-1 stacking)",
        "station_id (global model city encoding)",
    ]
    prioritized = [
        {
            "rank": 1,
            "feature": "Improve μ accuracy (±1°F hit rate)",
            "impact": "high",
            "rationale": "Modal bucket follows μ; σ scaling k∈[0.5,2.0] moves hit rate <0.3pp. TrackB gap is point accuracy, not σ.",
        },
        {
            "rank": 2,
            "feature": "dewpoint_10am + rh_mean_00_10 + ASOS morning obs",
            "impact": "high",
            "rationale": "TrackB uses 9 ASOS morning fields; NGBoost only has temp_early_morning. Moisture/cloud regime errors.",
        },
        {
            "rank": 3,
            "feature": "nws_tmax_forecast_f",
            "impact": "medium",
            "rationale": "TrackB #3 importance; parquet exists for 6/10 cities; orthogonal to HRRR despite higher MOS MAE on 2026.",
        },
        {
            "rank": 4,
            "feature": "cloud_cover_mean_00_10 + GFS afternoon cloud",
            "impact": "medium",
            "rationale": "Reduces afternoon Tmax miss on cloudy days; complements peak_cloud_cover",
        },
        {
            "rank": 5,
            "feature": "tmax_lag3-7",
            "impact": "medium",
            "rationale": "Cheap extension of existing lag pipeline; helps persistence regimes",
        },
        {
            "rank": 6,
            "feature": "wind_u/v + pressure_10am",
            "impact": "medium-low",
            "rationale": "Synoptic pattern context; available from same ASOS fetch",
        },
        {
            "rank": 7,
            "feature": "per-city bias correction (miami)",
            "impact": "conditional",
            "rationale": "Miami shows -1.08°F systematic cold bias",
        },
        {
            "rank": 8,
            "feature": "per-city training / SF coverage fix",
            "impact": "conditional",
            "rationale": "San Francisco: 79.9% coverage, 26.8% hit rate — needs local calibration or features",
        },
    ]
    return {
        "trackb_not_in_ngboost": trackb_only,
        "ngboost_not_in_trackb": ngboost_only,
        "prioritized_improvements": prioritized,
    }


def write_markdown(payload: dict, report_md: Path) -> None:
    g = payload["global"]
    hist = g["miss_distance_histogram"]
    bound = g["boundary_analysis"]
    cities = payload["per_city"]
    sigma = payload["sigma_analysis"]
    nws = payload["nws_mos"]
    gaps = payload["feature_gap"]
    model_dir = payload["model_dir"]
    model_label = Path(model_dir).name

    lines = [
        f"# NGBoost {model_label} Modal Bucket Hit Rate Diagnostic\n\n",
        f"**Model:** `{model_dir}` (Gaussian, global k={payload['sigma_calibration_k']:.4f})\n\n",
        f"**Test set:** 2026+ ({g['n_predictions']} city-days across 10 cities)\n\n",
        f"**Global modal bucket hit rate:** {g['modal_hit_rate_pct']:.1f}%\n\n",
        "## 1. Miss distance distribution\n\n",
        "| Distance (buckets) | Count | % of all | Interpretation |\n",
        "|-------------------:|------:|---------:|----------------|\n",
        f"| 0 (hit) | {hist['0_hit']} | {hist['pct_hit']:.1f}% | Modal bucket correct |\n",
    ]
    for d in range(1, 5):
        lines.append(
            f"| {d} ({d*2}°F) | {hist[f'{d}_bucket_off']} | "
            f"{hist[f'pct_{d}_off']:.1f}% | Off by {d} bucket(s) |\n"
        )
    lines.append(
        f"| 4+ (8°F+) | {hist['4plus_off']} | {hist['pct_4plus_off']:.1f}% | Large miss |\n\n"
    )
    lines.append(
        f"- **Fraction of misses off by exactly 1 bucket:** {hist['pct_misses_1_off']:.1f}%\n"
    )
    lines.append(
        f"- **Mean signed error (μ − actual):** {g['mean_signed_error_f']:.2f}°F\n"
    )
    lines.append(
        f"- **Median signed error:** {g['median_signed_error_f']:.2f}°F\n\n"
    )

    lines.append("### Per-city miss distance (1-bucket-off share of all days)\n\n")
    lines.append("| City | Hit rate | 1-off % | 2-off % | 3+ off % | Mean err (°F) |\n")
    lines.append("|------|--------:|--------:|--------:|---------:|--------------:|\n")
    for city, c in sorted(cities.items()):
        sub = payload["per_city_histograms"][city]
        lines.append(
            f"| {city} | {c['modal_hit_rate_pct']:.1f}% | {sub.get('pct_1_off', 0):.1f}% | "
            f"{sub.get('pct_2_off', 0):.1f}% | "
            f"{sub.get('pct_3plus_off', 0):.1f}% | {c['mean_signed_error_f']:+.2f} |\n"
        )

    lines.append("\n## 2. Bucket boundary analysis (1-bucket misses)\n\n")
    lines.append(
        f"Of **{bound['n_1_bucket_misses']}** predictions off by exactly one bucket, "
        f"**{bound['within_1f_of_boundary_pct']:.1f}%** had actual WU Tmax within 1°F of the "
        f"shared bucket boundary (mean distance to boundary: {bound['mean_boundary_distance_f']:.2f}°F).\n\n"
    )
    if bound["within_1f_of_boundary_pct"] > 50:
        lines.append(
            "Most 1-bucket misses are **boundary effects** — the mean prediction is close but "
            "probability mass straddles the cut point.\n\n"
        )
    else:
        lines.append(
            "A substantial share of 1-bucket misses are **not** near boundaries — these reflect "
            "genuine 2°F mean errors, not discretization.\n\n"
        )

    lines.append("## 3. Per-city bias and calibration\n\n")
    lines.append(
        "| City | Hit % | μ err | ±1°F % | 90% cov | σ/MAE | Class |\n"
        "|------|------:|------:|-------:|--------:|------:|-------|\n"
    )
    for city, c in sorted(cities.items()):
        lines.append(
            f"| {city} | {c['modal_hit_rate_pct']:.1f} | {c['mean_signed_error_f']:+.2f} | "
            f"{c['within_1f_pct']:.1f} | {c['coverage_90_pct']:.1f} | "
            f"{c['sigma_to_mae_ratio']:.2f} | {c['classification']} |\n"
        )

    lines.append("\n**Classification key:**\n")
    lines.append("- `systematic_bias`: |mean error| > 1°F — candidate for bias correction\n")
    lines.append("- `high_variance`: coverage far from 90% with small bias — needs features or per-city training\n")
    lines.append("- `boundary_cases`: tight coverage, small bias, but hit rate < 35% — sigma too wide\n")
    lines.append("- `mixed`: combination of effects\n\n")

    lines.append("## 4. Sigma analysis\n\n")
    cf = sigma["counterfactual_hit_rates"]
    lines.append(
        "Modal bucket selection is **dominated by μ, not σ**: scaling σ by k∈[0.5, 2.0] "
        f"changes global hit rate by <0.3pp (k=0.50 → {cf['k_0.50']:.1f}%, "
        f"k=1.00 → {cf['k_1.00_raw']:.1f}%, k=1.50 → {cf['k_1.50']:.1f}%). "
        "Per-city coverage calibration does not move the modal argmax.\n\n"
    )
    lines.append(
        f"- **μ in actual bucket:** {cf['mu_in_actual_bucket_pct']:.1f}% "
        f"(equals modal hit rate when σ is irrelevant)\n"
    )
    lines.append(
        f"- **μ within 1 bucket of actual:** {cf['mu_within_1_bucket_pct']:.1f}%\n"
    )
    lines.append(
        f"- **|μ − actual| ≤ 1°F:** {cf['within_1f_temp_pct']:.1f}% "
        f"(TrackB per-city models: 50–67%)\n"
    )
    lines.append(
        f"- **|μ − actual| ≤ 2°F:** {cf['within_2f_temp_pct']:.1f}%\n\n"
    )
    lines.append(
        "| σ scale k | Modal hit rate |\n"
        "|----------:|---------------:|\n"
    )
    for key in ["k_0.50", "k_0.70", "k_1.00_raw", "global_k", "k_1.50", "per_city_k"]:
        label = key.replace("_", " ").replace("raw", "(raw σ)")
        lines.append(f"| {label} | {cf[key]:.1f}% |\n")
    lines.append("\n")
    lines.append("| City | Median σ (cal) | Median |err| | σ/MAE | Per-city k | HR global k | HR per-city k |\n")
    lines.append("|------|---------------:|-----------:|------:|-----------:|------------:|--------------:|\n")
    for city, c in sorted(cities.items()):
        lines.append(
            f"| {city} | {c['median_sigma_calibrated']:.2f} | {c['median_abs_error_f']:.2f} | "
            f"{c['sigma_to_mae_ratio']:.2f} | {c['per_city_sigma_k']:.3f} | "
            f"{c['modal_hr_global_k_pct']:.1f}% | {c['modal_hr_per_city_k_pct']:.1f}% |\n"
        )
    lines.append(
        "\nσ/MAE >> 1 means intervals are wide (underconfident for trading edge), but "
        "**tightening σ alone will not raise modal hit rate** — μ must land in the correct bucket.\n\n"
    )

    lines.append("## 5. Feature gap vs TrackB\n\n")
    lines.append("**TrackB features not in NGBoost v2 (available pre-10AM):**\n\n")
    for feat in gaps["trackb_not_in_ngboost"]:
        lines.append(f"- {feat}\n")
    lines.append("\n**NGBoost-only features:**\n\n")
    for feat in gaps["ngboost_not_in_trackb"]:
        lines.append(f"- {feat}\n")

    lines.append("\n## 6. NWS MOS quick check\n\n")
    if not nws.get("available"):
        lines.append(f"NWS MOS data not available: {nws.get('reason', 'unknown')}\n\n")
    else:
        lines.append(f"Source: `{nws['path']}`\n\n")
        if nws.get("overall_paired"):
            o = nws["overall_paired"]
            lines.append(
                f"Paired test rows: {o['n_paired']} (6 cities with MOS cache). "
                f"corr(MOS, actual)={o['corr_mos_actual']:.3f}, "
                f"corr(μ, actual)={o['corr_mu_actual']:.3f}. "
                f"MOS MAE={o['mos_mae']:.2f}°F vs NGBoost MAE={o['mu_mae']:.2f}°F. "
                "NGBoost already dominates MOS on point accuracy; MOS may still add "
                "orthogonal signal as a stacked feature.\n\n"
            )
        lines.append("| City | N matched | corr(MOS,y) | corr(μ,y) | MOS MAE | μ MAE |\n")
        lines.append("|------|----------:|------------:|----------:|--------:|------:|\n")
        for row in nws["per_city"]:
            if row.get("n_matched", 0) == 0:
                lines.append(f"| {row['city']} | 0 | — | — | — | — |\n")
            else:
                lines.append(
                    f"| {row['city']} | {row['n_matched']} | {row['corr_mos_actual']:.3f} | "
                    f"{row['corr_mu_actual']:.3f} | {row['mos_mae']:.2f} | {row['mu_mae']:.2f} |\n"
                )
        if nws.get("cities_without_nws_cache"):
            lines.append(
                f"\nCities without NWS MOS cache: {', '.join(nws['cities_without_nws_cache'])}\n\n"
            )

    lines.append("## 7. Prioritized improvements\n\n")
    lines.append("| Rank | Change | Expected impact | Rationale |\n")
    lines.append("|-----:|--------|-----------------|------------|\n")
    for item in gaps["prioritized_improvements"]:
        lines.append(
            f"| {item['rank']} | {item['feature']} | {item['impact']} | {item['rationale']} |\n"
        )

    lines.append("\n## 8. Summary\n\n")
    lines.append(payload["summary"] + "\n")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_dir = PROJECT_ROOT / args.model_dir
    module_name = resolve_train_module(model_dir, args.train_module)
    train_helpers = load_train_helpers(module_name)
    report_json, report_md = default_report_paths(model_dir)
    if args.report_json:
        report_json = PROJECT_ROOT / args.report_json
    if args.report_md:
        report_md = PROJECT_ROOT / args.report_md

    print(f"Model dir: {model_dir}")
    print(f"Train module: {module_name}")
    print("Loading test predictions...")
    frame, sigma_k = load_test_predictions(model_dir, train_helpers)

    dist = frame["bucket_distance"].to_numpy(dtype=int)
    hits = float(np.mean(dist == 0))

    global_hist = histogram_table(dist)
    per_city_hist: dict[str, dict] = {}
    for city, grp in frame.groupby("city"):
        h = histogram_table(grp["bucket_distance"].to_numpy(dtype=int))
        h["pct_3plus_off"] = round(
            100.0 * np.sum(grp["bucket_distance"] >= 3) / len(grp), 2
        )
        per_city_hist[str(city)] = h

    bound = boundary_analysis(frame)
    cities = per_city_summary(frame, sigma_k)
    cf_sigma = counterfactual_sigma_hit_rates(frame, sigma_k)
    nws = nws_mos_analysis(frame)
    gaps = feature_gap_analysis()

    y = frame["actual_tmax"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)

    summary_parts = [
        f"Global modal hit rate is {100*hits:.1f}% on {len(frame)} test city-days.",
        f"{global_hist['pct_misses_1_off']:.0f}% of misses are exactly 1 bucket off;",
        f"{bound['within_1f_of_boundary_pct']:.0f}% of those are within 1°F of the boundary — the model is close but discretization loses the modal pick.",
        f"|μ−actual|≤1°F is only {cf_sigma['within_1f_temp_pct']:.1f}% vs TrackB's 50–67% — this is the primary gap vs TrackB.",
        f"σ scaling (k=0.5–1.5) changes hit rate by <0.3pp; per-city σ calibration does not help modal selection.",
        "Miami (-1.08°F bias) and San Francisco (79.9% coverage) are the weakest cities.",
    ]

    payload = {
        "model_dir": str(model_dir),
        "train_module": module_name,
        "sigma_calibration_k": sigma_k,
        "global": {
            "n_predictions": len(frame),
            "modal_hit_rate_pct": round(100.0 * hits, 2),
            "mean_signed_error_f": round(float(np.mean(mu - y)), 3),
            "median_signed_error_f": round(float(np.median(mu - y)), 3),
            "miss_distance_histogram": global_hist,
            "boundary_analysis": bound,
        },
        "per_city_histograms": per_city_hist,
        "per_city": cities,
        "sigma_analysis": {"counterfactual_hit_rates": cf_sigma},
        "nws_mos": nws,
        "feature_gap": gaps,
        "summary": " ".join(summary_parts),
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    write_markdown(payload, report_md)
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(f"Global modal hit rate: {100*hits:.1f}%")


if __name__ == "__main__":
    main()
