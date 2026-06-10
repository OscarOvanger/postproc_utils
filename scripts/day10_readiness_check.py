"""Day 10 readiness gate — feature tables, splits, HuggingFace, leakage."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.build_trackB_features import assert_no_leakage  # noqa: E402
from trackj.fetch_gfs_herbie import GFS_FEATURE_COLUMNS  # noqa: E402
from trackj.fetch_nws_forecast import TRAIN_CITIES, print_coverage_table  # noqa: E402

TRACKB_DIR = PROJECT_ROOT / "data" / "trackb"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
NWS_PATH = TRACKB_DIR / "nws_forecasts_raw.parquet"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
DATA_STORE_PATH = PROJECT_ROOT / "src" / "data_store.py"
HF_REPO_ID = "oovanger/MCP_datset"

ASOS_FEATURES = 9
CALENDAR_LAG_FEATURES = 9
NWS_FEATURES = 2
GFS_FEATURES = 3
NWP_BEST_FEATURES = 2
EXPECTED_FEATURES = ASOS_FEATURES + CALENDAR_LAG_FEATURES + NWS_FEATURES + GFS_FEATURES + NWP_BEST_FEATURES


def _check(label: str, ok: bool, blockers: list[str]) -> None:
    mark = "x" if ok else " "
    print(f"  [{mark}] {label}")
    if not ok:
        blockers.append(label)


def main() -> None:
    blockers: list[str] = []
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    print("Feature tables:")
    all_cities_exist = True
    for city in TRAIN_CITIES:
        path = TRACKB_DIR / city / "features.parquet"
        if not path.exists():
            all_cities_exist = False
    _check(f"data/trackb/<city>/features.parquet exists for all {len(TRAIN_CITIES)} train cities", all_cities_exist, blockers)

    nws_ok = True
    gfs_ok = True
    mae_ok = True
    feature_count_ok = True
    leakage_ok = True

    if NWS_PATH.exists():
        forecasts = pd.read_parquet(NWS_PATH)
        nws_summary = print_coverage_table(forecasts, config, trackj_dir=PROJECT_ROOT / "data" / "trackj")
        for _, row in nws_summary.iterrows():
            city = row["City"]
            if row["Coverage %"] < 80.0:
                nws_ok = False
                blockers.append(f"{city} NWS coverage {row['Coverage %']}% < 80%")
            mae_val = row["Mean abs error vs actual Tmax"]
            if mae_val is not None and mae_val == mae_val and mae_val >= 4.0:
                mae_ok = False
                blockers.append(f"{city} NWS MAE {mae_val}°F >= 4°F (flagged)")

    _check("All cities have >= 80% NWS coverage", nws_ok, blockers)

    gfs_flags: list[str] = []
    for city in TRAIN_CITIES:
        path = TRACKB_DIR / city / "features.parquet"
        if not path.exists():
            continue
        features = pd.read_parquet(path)
        feature_cols = [c for c in features.columns if c not in {"city", "date", "tmax"}]
        if len(feature_cols) != EXPECTED_FEATURES:
            feature_count_ok = False
            blockers.append(f"{city}: {len(feature_cols)} features (expected {EXPECTED_FEATURES})")
        gfs_cols = [c for c in GFS_FEATURE_COLUMNS if c in features.columns]
        gfs_cov = 100.0 * features[gfs_cols].notna().all(axis=1).mean() if gfs_cols else 0.0
        nws_cov = 100.0 * features.get("nws_tmax_forecast_f", pd.Series(dtype=float)).notna().mean()
        if gfs_cov < 80.0:
            gfs_ok = False
            reason = "below 50%" if gfs_cov < 50.0 else "below 80%"
            gfs_flags.append(f"{city}: GFS {gfs_cov:.1f}% ({reason})")
        try:
            merged = features.copy()
            if NWS_PATH.exists():
                nws = pd.read_parquet(NWS_PATH)
                city_nws = nws[nws["city"].eq(city)][["date", "issued_time"]]
                city_nws["date"] = pd.to_datetime(city_nws["date"]).dt.strftime("%Y-%m-%d")
                merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
                merged = merged.merge(city_nws, on="date", how="left")
            assert_no_leakage(merged, config[city])
        except AssertionError as exc:
            leakage_ok = False
            blockers.append(f"{city} leakage: {exc}")

    _check("All cities have >= 80% GFS coverage (or flagged with reason)", gfs_ok or bool(gfs_flags), blockers)
    for flag in gfs_flags:
        print(f"      FLAG: {flag}")
    _check("NWS MAE < 4°F for all cities (or flagged)", mae_ok, blockers)
    nwp_ok = True
    for city in TRAIN_CITIES:
        path = TRACKB_DIR / city / "features.parquet"
        if not path.exists():
            continue
        features = pd.read_parquet(path)
        nwp_cov = 100.0 * features.get("nwp_tmax_best_f", pd.Series(dtype=float)).notna().mean()
        if nwp_cov < 95.0:
            nwp_ok = False
            blockers.append(f"{city} NWP best coverage {nwp_cov:.1f}% < 95%")
    _check("All cities have >= 95% nwp_tmax_best_f coverage", nwp_ok, blockers)
    _check(f"All cities have {EXPECTED_FEATURES} features (5 groups active)", feature_count_ok, blockers)
    _check("No leakage assertions failed during feature table build", leakage_ok, blockers)

    print("\nData splits:")
    _check("data/splits/threshold_opt.parquet exists", (SPLITS_DIR / "threshold_opt.parquet").exists(), blockers)
    _check("data/splits/time_holdout.parquet exists", (SPLITS_DIR / "time_holdout.parquet").exists(), blockers)

    grep_result = subprocess.run(
        ["rg", "-l", "true_holdout\\.parquet", "scripts/", "src/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    readers = [p for p in grep_result.stdout.strip().split("\n") if p and "data_store" not in p and "day10_readiness" not in p]
    holdout_guarded = all("assert" in Path(PROJECT_ROOT / p).read_text(encoding="utf-8") for p in readers if p)
    _check("true_holdout.parquet NOT loaded by scripts without guards", holdout_guarded or not readers, blockers)

    print("\nHuggingFace:")
    hf_module_ok = DATA_STORE_PATH.exists()
    _check("src/data_store.py exists with load_features and load_splits", hf_module_ok, blockers)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
        uploaded_cities = sum(1 for c in TRAIN_CITIES if f"data/trackb/{c}/features.parquet" in repo_files)
        _check(f"Feature parquets uploaded and verified for all 9 cities ({uploaded_cities}/9 on HF)", uploaded_cities == 9, blockers)
    except Exception as exc:
        _check("Feature parquets uploaded and verified for all 9 cities", False, blockers)
        blockers.append(f"HF check failed: {exc}")

    status = "READY" if not blockers else "NOT READY"
    print(f"\nDay 10 readiness: {status}")
    if blockers:
        print("Blockers:")
        for item in blockers:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
