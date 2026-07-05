#!/usr/bin/env python3
"""Step 0: verify backtest preconditions before running the pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest.common import (  # noqa: E402
    HRRR_PARQUET,
    LIVE_MODEL_DIR,
    MODEL_PATH_FILE,
    POLY_CITIES,
    POLY_HISTORY_DIR,
    REPORTS_DIR,
    TRACKB_DIR,
    WU_PATH,
    print_trackb_mapping_table,
)

HRRR_MIN_ROWS = 1900
HRRR_START = "2021-01-01"
HRRR_END = "2026-07-02"
MIN_FREE_GB = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def discover_model_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not (p / "model_config.json").exists():
            raise FileNotFoundError(f"No model_config.json in {p}")
        return p

    if MODEL_PATH_FILE.exists():
        path_text = MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
        if path_text:
            p = Path(path_text)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if (p / "model_config.json").exists():
                return p

    default_v2 = PROJECT_ROOT / "models" / "ngboost_v2"
    if (default_v2 / "model_config.json").exists():
        return default_v2

    raise FileNotFoundError(
        "No backtest NGBoost model found. Train models/ngboost_v2 or set reports/backtest_model_path.txt"
    )


def check_hrrr() -> tuple[bool, str]:
    if not HRRR_PARQUET.exists():
        return False, f"Missing {HRRR_PARQUET}. Run fetch_hrrr_all_cities.py"
    df = pd.read_parquet(HRRR_PARQUET)
    df["city"] = df["city"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    missing_cities = set(POLY_CITIES) - set(df["city"].unique())
    if missing_cities:
        return False, f"Missing cities in HRRR: {sorted(missing_cities)}"
    for city in POLY_CITIES:
        sub = df[df["city"] == city]
        if len(sub) <= HRRR_MIN_ROWS:
            return False, f"{city}: only {len(sub)} rows (need >{HRRR_MIN_ROWS})"
        if sub["date"].min() > pd.Timestamp(HRRR_START) or sub["date"].max() < pd.Timestamp(HRRR_END):
            return False, f"{city}: date range {sub['date'].min().date()}–{sub['date'].max().date()}"
    return True, f"All 10 cities, >{HRRR_MIN_ROWS} rows, {HRRR_START}–{HRRR_END}"


def check_wu() -> tuple[bool, str]:
    if not WU_PATH.exists():
        return False, f"Missing {WU_PATH}. Run fetch_wunderground_target.py"
    df = pd.read_parquet(WU_PATH)
    if "reliable" not in df.columns:
        return False, "Missing `reliable` column"
    df["city"] = df["city"].astype(str)
    missing = set(POLY_CITIES) - set(df["city"].unique())
    if missing:
        return False, f"Missing cities: {sorted(missing)}"
    return True, f"10 cities, reliable column present, {len(df)} rows"


def check_polymarket_history() -> tuple[bool, str]:
    if not POLY_HISTORY_DIR.exists() or not any(POLY_HISTORY_DIR.iterdir()):
        return False, (
            "Polymarket order-book backfill has not completed or was never run. "
            "Check logs/polymarket_backfill_*.log before proceeding. "
            "Do not run a backtest on synthetic/assumed prices."
        )
    from polymarket_history_coverage_report import (  # noqa: E402
        DEFAULT_FLOOR_DATE,
        evaluate_coverage_gate,
        print_coverage_failure_details,
        print_report,
        run_report,
    )

    df, _passed = run_report(DEFAULT_FLOOR_DATE)
    print_report(df, DEFAULT_FLOOR_DATE)
    ok, detail = evaluate_coverage_gate(df)
    if not ok:
        print_coverage_failure_details(df)
        return False, (
            f"Polymarket history coverage insufficient: {detail}. "
            "Check logs/polymarket_backfill_*.log before proceeding."
        )
    return True, f"Coverage report passed ({detail})"


def check_ngboost(model_dir: Path) -> tuple[bool, str]:
    config_path = model_dir / "model_config.json"
    model_path = model_dir / "ngboost_global.pkl"
    if not config_path.exists() or not model_path.exists():
        return False, f"Incomplete model dir: {model_dir}"

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    live_model = LIVE_MODEL_DIR / "ngboost_global.pkl"
    same_as_live = False
    if live_model.exists() and model_path.resolve() == live_model.resolve():
        same_as_live = True
    elif live_model.exists():
        same_as_live = sha256_file(model_path) == sha256_file(live_model)

    lines = [
        f"model_dir={model_dir}",
        f"model_type={config.get('model_type')}",
        f"cities={config.get('cities')}",
        f"val_crps={config.get('val_crps')}",
        f"n_features={len(config.get('feature_columns', []))}",
    ]
    if same_as_live:
        lines.append("WARNING: identical to live-trading model at models/ngboost/ngboost_global.pkl")
    else:
        lines.append("Distinct from live-trading model path")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH_FILE.write_text(str(model_dir.resolve()) + "\n", encoding="utf-8")
    return True, "; ".join(lines)


def check_trackb() -> tuple[bool, str]:
    if not TRACKB_DIR.exists():
        return False, f"Missing {TRACKB_DIR}"
    cities = sorted(
        d.name for d in TRACKB_DIR.iterdir() if d.is_dir() and (d / "feature_cols.json").exists()
    )
    if not cities:
        return False, "No trained TrackB models found"
    print_trackb_mapping_table()
    return True, f"{len(cities)} cities: {', '.join(cities)}"


def check_disk() -> tuple[bool, str]:
    result = subprocess.run(["df", "-h", str(PROJECT_ROOT)], capture_output=True, text=True)
    output = result.stdout.strip()
    print("\n=== Disk space ===")
    print(output)
    warn = False
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == "/":
            avail = parts[3]
            if avail.endswith("G"):
                try:
                    if float(avail[:-1]) < MIN_FREE_GB:
                        warn = True
                except ValueError:
                    pass
    if warn:
        print(f"\nWARNING: free space under {MIN_FREE_GB}GB — backtest outputs may be large")
    return True, "disk check complete" + (" (low space warning)" if warn else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest precondition checks")
    parser.add_argument("--model-dir", default=None, help="NGBoost model directory for backtest")
    args = parser.parse_args()

    checks: list[tuple[str, tuple[bool, str]]] = []
    try:
        model_dir = discover_model_dir(args.model_dir)
    except FileNotFoundError as exc:
        model_dir = None
        checks.append(("4 NGBoost model", (False, str(exc))))

    checks.append(("1 HRRR parquet", check_hrrr()))
    checks.append(("2 WU targets", check_wu()))
    checks.append(("3 Polymarket history", check_polymarket_history()))
    if model_dir is not None:
        checks.append(("4 NGBoost model", check_ngboost(model_dir)))
    checks.append(("5 TrackB models", check_trackb()))
    checks.append(("6 Disk space", check_disk()))

    print("\n=== PRECONDITION CHECKS ===")
    print(f"{'Check':<22} {'Status':<6} Details")
    print("-" * 80)
    all_pass = True
    for name, (ok, detail) in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"{name:<22} {status:<6} {detail}")

    if not all_pass:
        print("\nSTOP: fix failing checks before proceeding to Step 1.")
        sys.exit(1)
    print(f"\nAll checks passed. Model path written to {MODEL_PATH_FILE}")
    print("Proceed with: .venv/bin/python scripts/backtest/step1_eligible_range.py")


if __name__ == "__main__":
    main()
