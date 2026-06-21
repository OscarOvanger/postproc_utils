"""Pre-fetch and validate feature sources before the 10AM trading pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("TRACKJ_SKIP_HF_SYNC", "1")

from src.data_pipeline import (  # noqa: E402
    build_feature_vector_strict,
    fetch_asos_morning,
    fetch_gfs_afternoon,
    fetch_lag_features,
    fetch_nwp_best,
    fetch_nws_forecast_full,
)
from run_daily_trade import (  # noqa: E402
    load_deploy_config,
    load_models,
    predict_tmax_strict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch and validate features")
    parser.add_argument("--date", type=str, default=str(date.today()))
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated city list (default: all from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "deploy_config.json"),
    )
    args = parser.parse_args()

    config = load_deploy_config(Path(args.config))
    event_date = args.date
    cities = args.cities.split(",") if args.cities else config["cities"]

    print(f"\n=== PREFETCH: {event_date} ===")
    print(f"Cities: {len(cities)}")
    print()

    results = []
    for city in cities:
        print(f"--- {city} ---")
        row: dict[str, str] = {"city": city}

        try:
            asos = fetch_asos_morning(city, event_date, skip_cache=True)
            row["asos"] = "OK" if asos and len(asos) >= 5 else "FAIL"
            if asos:
                row["asos_detail"] = f"{len(asos)} features"
            else:
                row["asos_detail"] = "returned None"
        except Exception as exc:
            row["asos"] = "ERROR"
            row["asos_detail"] = str(exc)[:80]

        try:
            lags = fetch_lag_features(city, event_date, skip_cache=True)
            row["lags"] = "OK" if lags and "temp_lag1" in lags else "FAIL"
            if lags:
                row["lags_detail"] = f"{len(lags)} features"
            else:
                row["lags_detail"] = "returned None"
        except Exception as exc:
            row["lags"] = "ERROR"
            row["lags_detail"] = str(exc)[:80]

        try:
            nws = fetch_nws_forecast_full(city, event_date, skip_cache=True)
            row["nws"] = "OK" if nws else "FAIL"
            if nws:
                row["nws_detail"] = f"tmax={nws.get('nws_tmax_forecast_f', '?')}F"
            else:
                row["nws_detail"] = "returned None"
        except Exception as exc:
            row["nws"] = "ERROR"
            row["nws_detail"] = str(exc)[:80]

        try:
            gfs = fetch_gfs_afternoon(city, event_date, skip_cache=True)
            row["gfs"] = "OK" if gfs and len(gfs) == 3 else "FAIL"
            if gfs:
                row["gfs_detail"] = f"t2m={gfs.get('gfs_t2m_afternoon', '?'):.1f}"
            else:
                row["gfs_detail"] = "returned None"
        except Exception as exc:
            row["gfs"] = "ERROR"
            row["gfs_detail"] = str(exc)[:80]

        try:
            nwp = fetch_nwp_best(city, event_date, skip_cache=True)
            row["nwp"] = "OK" if nwp is not None else "FAIL"
            if nwp is not None:
                row["nwp_detail"] = f"tmax={nwp:.1f}F"
            else:
                row["nwp_detail"] = "returned None"
        except Exception as exc:
            row["nwp"] = "ERROR"
            row["nwp_detail"] = str(exc)[:80]

        all_ok = all(row.get(group) == "OK" for group in ("asos", "lags", "nws", "gfs", "nwp"))
        if all_ok:
            try:
                model_dir = PROJECT_ROOT / config["model_dir"]
                models, feature_cols = load_models(city, model_dir)
                features, fail = build_feature_vector_strict(city, event_date, feature_cols)
                if features:
                    pred = predict_tmax_strict(models, feature_cols, features)
                    row["prediction"] = f"{pred}F" if pred else "NaN features"
                else:
                    row["prediction"] = f"FAIL: {fail}"
            except Exception as exc:
                row["prediction"] = f"ERROR: {str(exc)[:60]}"
        else:
            row["prediction"] = "SKIP (missing sources)"

        status_line = " | ".join(
            f"{group}:{row.get(group, '?')}" for group in ("asos", "lags", "nws", "gfs", "nwp")
        )
        print(f"  {status_line}")
        if row.get("prediction"):
            print(f"  Prediction: {row['prediction']}")
        print()

        results.append(row)

    print("=== SUMMARY ===")
    print(
        f"{'City':<20} {'ASOS':<6} {'Lags':<6} {'NWS':<6} {'GFS':<6} {'NWP':<6} {'Pred':<12}"
    )
    print("-" * 72)
    for row in results:
        print(
            f"{row['city']:<20} {row.get('asos', '?'):<6} {row.get('lags', '?'):<6} "
            f"{row.get('nws', '?'):<6} {row.get('gfs', '?'):<6} {row.get('nwp', '?'):<6} "
            f"{row.get('prediction', '?'):<12}"
        )

    n_ready = sum(
        1
        for row in results
        if row.get("prediction", "").endswith("F")
        and "FAIL" not in row.get("prediction", "")
    )
    print(f"\nReady: {n_ready}/{len(results)} cities")
    if n_ready == 0:
        print("WARNING: No cities have forecast coverage. Debug data sources.")
    elif n_ready < 5:
        print(f"WARNING: Only {n_ready}/9 cities ready. Check failing sources above.")
    else:
        print("GOOD: Sufficient coverage for trading.")

    sys.exit(0 if n_ready >= 5 else 1)


if __name__ == "__main__":
    main()
