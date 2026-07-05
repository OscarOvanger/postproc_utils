#!/usr/bin/env python3
"""Write v2 vs v4 NGBoost comparison report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

V2_DIR = PROJECT_ROOT / "models" / "ngboost_v2"
V4_DIR = PROJECT_ROOT / "models" / "ngboost_v4"
V2_DIAG = PROJECT_ROOT / "reports" / "bucket_hit_diagnostic.json"
V4_DIAG = PROJECT_ROOT / "reports" / "bucket_hit_diagnostic_v4.json"
OUTPUT = PROJECT_ROOT / "reports" / "ngboost_v4_comparison.md"
STATION_MD = PROJECT_ROOT / "reports" / "ngboost_calibration_v4" / "station_verification.md"
IMPORTANCE_CSV = PROJECT_ROOT / "reports" / "ngboost_calibration_v4" / "stage1_feature_importance.csv"

NEW_ASOS_COLS = [
    "dewpoint_10am",
    "rh_mean_00_10",
    "pressure_10am",
    "wind_u_mean_00_10",
    "wind_v_mean_00_10",
    "cloud_cover_mean_00_10",
]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_diag(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_eval_test(model_dir: Path, train_module: str) -> str:
    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(SCRIPTS_DIR / f"{train_module}.py"),
        "--output-dir",
        str(model_dir.relative_to(PROJECT_ROOT)),
        "--eval-test",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout + result.stderr


def parse_overall_mae(eval_output: str) -> float | None:
    for line in eval_output.splitlines():
        if line.strip().startswith("OVERALL"):
            parts = line.split()
            # Format: OVERALL N_rows MAE CRPS ...
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    continue
    return None


def pit_ks_p(model_dir: Path, train_module: str) -> float | None:
    mod = __import__(train_module, fromlist=["*"])
    model, scaler, config = mod.load_saved_artifacts(model_dir)
    cities = list(config.get("cities", []))
    feature_cols = list(config.get("feature_columns", []))
    fill_medians = dict(config.get("nan_fill_medians", {}))
    sigma_k = float(config.get("sigma_calibration_k", 1.0))

    stage1_path = model_dir / config.get("stage1_model", "lgb_stage1.pkl")
    lgb_model = joblib.load(stage1_path)
    stage1_cols = [c for c in feature_cols if c != "lgb_tmax_pred"]

    df = mod.assemble_dataset(cities)
    df = mod.drop_incomplete_rows(df)
    _train, val_df, _test = mod.temporal_split(df)
    fill_cols = list(fill_medians.keys()) if fill_medians else mod.MEDIAN_FILL_COLS
    val_df = mod.apply_saved_median_fill(val_df, fill_medians, fill_cols)
    val_df = val_df.copy()
    val_df["lgb_tmax_pred"] = lgb_model.predict(val_df[stage1_cols])

    X = mod.transform_features(scaler, val_df, feature_cols)
    y = val_df[mod.TARGET].to_numpy(dtype=float)
    mu, sigma_raw, _ = mod.predict_dist_params(model, X)
    sigma = mod.apply_sigma_calibration(sigma_raw, sigma_k)
    u = norm.cdf(y, loc=mu, scale=sigma)
    _, p_value = kstest(u, "uniform")
    return float(p_value)


def overall_coverage_90(diag: dict) -> float | None:
    cities = diag.get("per_city", {})
    if not cities:
        return None
    weights = [c["n"] for c in cities.values()]
    covs = [c["coverage_90_pct"] for c in cities.values()]
    return float(np.average(covs, weights=weights))


def trade_recommendation(hit_pct: float, within_1f: float) -> str:
    if hit_pct < 30.0:
        return "EXCLUDE"
    if hit_pct < 35.0 or within_1f < 35.0:
        return "CAUTION"
    return "TRADE"


def delta_str(v2: float | None, v4: float | None, pct: bool = False, higher_better: bool = True) -> str:
    if v2 is None or v4 is None:
        return "—"
    d = v4 - v2
    good = (d > 0) if higher_better else (d < 0)
    sign = "+" if d >= 0 else ""
    suffix = "pp" if pct else ""
    arrow = "↑" if good else "↓"
    if pct:
        return f"{sign}{d:.1f}{suffix} {arrow}"
    return f"{sign}{d:.3f} {arrow}"


def write_report(v2_cfg: dict, v4_cfg: dict, v2_diag: dict, v4_diag: dict) -> None:
    v2_global = v2_diag.get("global", {})
    v4_global = v4_diag.get("global", {})
    v2_sigma = v2_diag.get("sigma_analysis", {}).get("counterfactual_hit_rates", {})
    v4_sigma = v4_diag.get("sigma_analysis", {}).get("counterfactual_hit_rates", {})
    v2_cities = v2_diag.get("per_city", {})
    v4_cities = v4_diag.get("per_city", {})

    v2_mae = parse_overall_mae(run_eval_test(V2_DIR, "train_ngboost"))
    v4_mae = parse_overall_mae(run_eval_test(V4_DIR, "train_ngboost_v4"))

    v2_pit = pit_ks_p(V2_DIR, "train_ngboost")
    v4_pit = pit_ks_p(V4_DIR, "train_ngboost_v4")

    lines = [
        "# NGBoost v2 vs v4 Comparison\n\n",
        "v4 adds 6 TrackB ASOS morning features to the v2 17-feature global Gaussian model.\n\n",
        "## 1. Overall metrics\n\n",
        "| Metric | v2 (17 feat) | v4 (23 feat) | Delta |\n",
        "|--------|-------------:|-------------:|-------|\n",
        f"| Val CRPS | {v2_cfg.get('val_crps', '?'):.4f} | {v4_cfg.get('val_crps', '?'):.4f} | "
        f"{delta_str(v2_cfg.get('val_crps'), v4_cfg.get('val_crps'), higher_better=False)} |\n",
        f"| Test MAE | {v2_mae:.2f} | {v4_mae:.2f} | "
        f"{delta_str(v2_mae, v4_mae, higher_better=False)} |\n" if v2_mae and v4_mae else
        "| Test MAE | — | — | — |\n",
        f"| ±1°F accuracy | {v2_sigma.get('within_1f_temp_pct', '?'):.1f}% | "
        f"{v4_sigma.get('within_1f_temp_pct', '?'):.1f}% | "
        f"{delta_str(v2_sigma.get('within_1f_temp_pct'), v4_sigma.get('within_1f_temp_pct'), pct=True)} |\n",
        f"| Modal bucket HR | {v2_global.get('modal_hit_rate_pct', '?'):.1f}% | "
        f"{v4_global.get('modal_hit_rate_pct', '?'):.1f}% | "
        f"{delta_str(v2_global.get('modal_hit_rate_pct'), v4_global.get('modal_hit_rate_pct'), pct=True)} |\n",
        f"| 90% coverage (city-weighted) | {overall_coverage_90(v2_diag):.1f}% | "
        f"{overall_coverage_90(v4_diag):.1f}% | "
        f"{delta_str(overall_coverage_90(v2_diag), overall_coverage_90(v4_diag), pct=True)} |\n"
        if overall_coverage_90(v2_diag) and overall_coverage_90(v4_diag) else
        "| 90% coverage | varies | varies | — |\n",
        f"| PIT KS p | {v2_pit:.3f} | {v4_pit:.3f} | "
        f"{delta_str(v2_pit, v4_pit)} |\n\n",
    ]

    lines.append("## 2. Per-city bucket hit rate\n\n")
    lines.append("| City | v2 HR | v4 HR | Delta |\n|------|------:|------:|------:|\n")
    all_cities = sorted(set(v2_cities) | set(v4_cities))
    for city in all_cities:
        v2_hr = v2_cities.get(city, {}).get("modal_hit_rate_pct")
        v4_hr = v4_cities.get(city, {}).get("modal_hit_rate_pct")
        d = (v4_hr - v2_hr) if v2_hr is not None and v4_hr is not None else None
        d_str = f"{d:+.1f}pp" if d is not None else "—"
        lines.append(f"| {city} | {v2_hr:.1f}% | {v4_hr:.1f}% | {d_str} |\n")

    lines.append("\n## 3. Per-city ±1°F accuracy\n\n")
    lines.append("| City | v2 ±1°F | v4 ±1°F | Delta |\n|------|--------:|--------:|------:|\n")
    for city in all_cities:
        v2_w = v2_cities.get(city, {}).get("within_1f_pct")
        v4_w = v4_cities.get(city, {}).get("within_1f_pct")
        d = (v4_w - v2_w) if v2_w is not None and v4_w is not None else None
        d_str = f"{d:+.1f}pp" if d is not None else "—"
        lines.append(f"| {city} | {v2_w:.1f}% | {v4_w:.1f}% | {d_str} |\n")

    lines.append("\n## 4. Stage-1 LightGBM feature importance (v4)\n\n")
    if IMPORTANCE_CSV.exists():
        imp = pd.read_csv(IMPORTANCE_CSV)
        imp["rank"] = range(1, len(imp) + 1)
        lines.append("| Rank | Feature | Importance |\n|-----:|---------|----------:|\n")
        for _, row in imp.iterrows():
            marker = " **NEW**" if row["feature"] in NEW_ASOS_COLS else ""
            lines.append(f"| {int(row['rank'])} | {row['feature']}{marker} | {row['importance']:.0f} |\n")
        lines.append("\n**NEW** = TrackB ASOS morning feature added in v4.\n\n")
    else:
        lines.append("_Feature importance file not found._\n\n")

    lines.append("## 5. Miss distance distribution\n\n")
    v2_hist = v2_global.get("miss_distance_histogram", {})
    v4_hist = v4_global.get("miss_distance_histogram", {})
    lines.append("| Distance | v2 % | v4 % | Delta |\n|----------|-----:|-----:|------:|\n")
    for key, label in [
        ("pct_hit", "0 (hit)"),
        ("pct_1_off", "1 bucket"),
        ("pct_2_off", "2 buckets"),
        ("pct_3_off", "3 buckets"),
        ("pct_4plus_off", "4+ buckets"),
    ]:
        v2_v = v2_hist.get(key, 0)
        v4_v = v4_hist.get(key, 0)
        lines.append(f"| {label} | {v2_v:.1f}% | {v4_v:.1f}% | {v4_v - v2_v:+.1f}pp |\n")
    lines.append(
        f"\n1-bucket miss share of all misses: v2 {v2_hist.get('pct_misses_1_off', 0):.1f}% → "
        f"v4 {v4_hist.get('pct_misses_1_off', 0):.1f}%\n\n"
    )

    lines.append("## 6. Station verification\n\n")
    if STATION_MD.exists():
        lines.append(STATION_MD.read_text(encoding="utf-8"))
        lines.append("\n")
    else:
        lines.append("_Station verification report not found._\n\n")

    lines.append("## 7. Trade selection for deployment\n\n")
    lines.append(
        "Deployment parameters (not model changes):\n\n"
        "- **Default:** top-2 trades per day ranked by edge\n"
        "- **Minimum:** 1 trade per day on days with weak signals\n"
        "- **Exclude:** cities with test bucket HR < 30%\n"
        "- **Skip:** when model μ is within 0.5°F of a bucket boundary "
        "(89% of v2 1-bucket misses were boundary cases)\n\n"
    )
    lines.append("| City | v4 Bucket HR | v4 ±1°F | Recommendation |\n")
    lines.append("|------|------------:|--------:|----------------|\n")
    for city in all_cities:
        c = v4_cities.get(city, {})
        hr = c.get("modal_hit_rate_pct", 0)
        w1 = c.get("within_1f_pct", 0)
        rec = trade_recommendation(hr, w1)
        lines.append(f"| {city} | {hr:.1f}% | {w1:.1f}% | {rec} |\n")

    lines.append(
        "\n## 8. Summary\n\n"
        f"- v4 feature count: {len(v4_cfg.get('feature_columns', []))} "
        f"(+{len(NEW_ASOS_COLS)} ASOS morning features)\n"
        f"- Val CRPS: v2 {v2_cfg.get('val_crps', '?'):.4f} → v4 {v4_cfg.get('val_crps', '?'):.4f}\n"
        f"- Modal bucket HR (test): v2 {v2_global.get('modal_hit_rate_pct', '?'):.1f}% → "
        f"v4 {v4_global.get('modal_hit_rate_pct', '?'):.1f}%\n"
        f"- ±1°F accuracy: v2 {v2_sigma.get('within_1f_temp_pct', '?'):.1f}% → "
        f"v4 {v4_sigma.get('within_1f_temp_pct', '?'):.1f}%\n"
        "- Live/backtest model remains v2 until explicitly switched.\n"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-diag", action="store_true", help="Skip re-running diagnostics")
    args = parser.parse_args()

    if not args.skip_diag:
        for model_dir, module in [(V2_DIR, None), (V4_DIR, "train_ngboost_v4")]:
            cmd = [
                str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                str(SCRIPTS_DIR / "run_bucket_hit_diagnostic.py"),
                "--model-dir",
                str(model_dir.relative_to(PROJECT_ROOT)),
            ]
            if module:
                cmd.extend(["--train-module", module])
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    v2_cfg = load_config(V2_DIR / "model_config.json")
    v4_cfg = load_config(V4_DIR / "model_config.json")
    v2_diag = load_diag(V2_DIAG)
    v4_diag = load_diag(V4_DIAG)
    write_report(v2_cfg, v4_cfg, v2_diag, v4_diag)


if __name__ == "__main__":
    main()
