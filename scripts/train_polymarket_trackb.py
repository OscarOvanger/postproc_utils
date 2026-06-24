"""Build Polymarket Track-B features and train Wunderground-target ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.poly_trading_pipeline import POLYMARKET_CITIES  # noqa: E402
from src.trackj.build_calendar_lag_features import (  # noqa: E402
    CALENDAR_LAG_COLUMNS,
    build_calendar_lag_features,
)
from src.trackj.build_trackB_features import build_trackB_features  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
WU_TARGETS_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
FEATURES_DIR = PROJECT_ROOT / "data" / "polymarket" / "trackb_features"
TRACKB_DATA_DIR = PROJECT_ROOT / "data" / "trackb"
TRACKJ_DATA_DIR = PROJECT_ROOT / "data" / "trackj"
NWS_PATH = TRACKB_DATA_DIR / "nws_forecasts_raw.parquet"
RAW_DIR = TRACKJ_DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models" / "polymarket_trackb"
OLD_MODEL_DIR = PROJECT_ROOT / "models" / "trackb"

FEATURE_START = date(2021, 1, 1)
FEATURE_END = date(2026, 6, 23)
TRAIN_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-12-31")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-06-23")

LAG_COLUMNS = [*CALENDAR_LAG_COLUMNS, "temp_lag1"]

META_COLS = [
    "date",
    "wunderground_tmax",
    "cli_tmax",
    "tmax",
    "tmax_f",
    "city",
    "station",
    "n_readings",
    "reliable",
    "nwp_is_ecmwf",
    "nwp_source",
    "nws_issuance_hour",
    "nws_cycle",
    "issued_time",
    "nwp_issued_date",
    "nwp_valid_date",
]

LEAKAGE_SUBSTRINGS = ("target", "actual", "resolved", "settled", "bucket")
MIN_TRAIN_ROWS = 500


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def backsolve_sigma(hit_rate_1f: float) -> float:
    if hit_rate_1f <= 0 or hit_rate_1f >= 1:
        return float("nan")
    z = norm.ppf((hit_rate_1f + 1) / 2)
    if z <= 0:
        return float("nan")
    return 1.0 / z


def bucket_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_bucket = np.floor(y_pred / 2.0) * 2.0
    true_bucket = np.floor(y_true / 2.0) * 2.0
    return float(np.mean(pred_bucket == true_bucket))


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred.astype(float) - y_true.astype(float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "sigma_errors": float(np.std(errors)),
        "hit_rate_1f": float(np.mean(np.abs(errors) <= 1.0)),
        "hit_rate_2f": float(np.mean(np.abs(errors) <= 2.0)),
        "bucket_hr": bucket_hit_rate(y_true, y_pred),
        "bias": float(np.mean(errors)),
        "n": int(len(y_true)),
    }


def ensemble_predict(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack([m.predict(X) for m in models])
    return np.round(np.mean(preds, axis=1)).astype(int)


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def _build_wu_lag_features(wu_df: pd.DataFrame) -> pd.DataFrame:
    wu = wu_df.copy()
    wu = _normalize_date_column(wu)
    wu = wu.sort_values("date").drop_duplicates("date")
    target_df = wu[["date"]].assign(tmax_f=pd.to_numeric(wu["wunderground_tmax"], errors="coerce"))
    lags = build_calendar_lag_features(target_df)
    sorted_target = target_df.sort_values("date").reset_index(drop=True)
    lag_map = dict(
        zip(
            sorted_target["date"],
            sorted_target["tmax_f"].shift(1),
        )
    )
    lags["temp_lag1"] = lags["date"].map(lag_map)
    return lags


def _load_or_build_trackb_features(city: str, config: dict) -> pd.DataFrame:
    features_path = TRACKB_DATA_DIR / city / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    print(f"{city}: building Track-B covariates via build_trackB_features (no_fetch=True)")
    return build_trackB_features(
        config[city],
        FEATURE_START,
        FEATURE_END,
        RAW_DIR,
        TRACKB_DATA_DIR,
        NWS_PATH,
        trackj_dir=TRACKJ_DATA_DIR,
        include_gfs=True,
        no_fetch=True,
    )


def build_polymarket_features(city: str, config: dict, wu_targets: pd.DataFrame) -> pd.DataFrame:
    wu_city = wu_targets[wu_targets["city"].astype(str).eq(city)].copy()
    wu_city = _normalize_date_column(wu_city)

    cli_path = TRACKJ_DATA_DIR / city / "cli_target.parquet"
    if not cli_path.exists():
        raise FileNotFoundError(f"Missing CLI target: {cli_path}")
    cli = pd.read_parquet(cli_path)
    cli = _normalize_date_column(cli)
    cli = cli[["date", "tmax_f"]].rename(columns={"tmax_f": "cli_tmax"})

    base = _load_or_build_trackb_features(city, config)
    base = _normalize_date_column(base)
    if "tmax" in base.columns and "tmax_f" not in base.columns:
        base = base.drop(columns=["tmax"], errors="ignore")

    drop_cols = [c for c in LAG_COLUMNS if c in base.columns]
    covariate_cols = [c for c in base.columns if c not in drop_cols and c not in {"city", "tmax", "tmax_f"}]
    covariates = base[covariate_cols].copy()

    wu_lags = _build_wu_lag_features(wu_city)
    merged = covariates.merge(wu_lags, on="date", how="inner")
    merged = merged.merge(
        wu_city[["date", "wunderground_tmax", "n_readings", "reliable"]],
        on="date",
        how="inner",
    )
    merged = merged.merge(cli, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _candidate_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in META_COLS
        and not any(x in c.lower() for x in LEAKAGE_SUBSTRINGS)
    ]


def _select_features(train_df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    y = pd.to_numeric(train_df["wunderground_tmax"], errors="coerce")
    retained: list[str] = []
    for col in candidate_cols:
        missing_frac = train_df[col].isna().mean()
        if missing_frac > 0.20:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce")
        valid = series.notna() & y.notna()
        if valid.sum() < 10:
            continue
        corr = series[valid].corr(y[valid])
        if corr is None or abs(corr) < 0.05:
            continue
        retained.append(col)
    return retained


def _split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    train_df = df[dates <= TRAIN_END].copy()
    val_df = df[(dates >= VAL_START) & (dates <= VAL_END)].copy()
    test_df = df[(dates >= TEST_START) & (dates <= TEST_END)].copy()
    return train_df, val_df, test_df


def _forward_fill_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    for frame in (train_df, val_df, test_df):
        frame["date"] = pd.to_datetime(frame["date"])

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.sort_values("date")
    combined[feature_cols] = combined[feature_cols].ffill()

    splits = {
        "train": combined[combined["date"] <= TRAIN_END].copy(),
        "val": combined[(combined["date"] >= VAL_START) & (combined["date"] <= VAL_END)].copy(),
        "test": combined[(combined["date"] >= TEST_START) & (combined["date"] <= TEST_END)].copy(),
    }
    dropped: dict[str, int] = {}
    cleaned: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        before = len(frame)
        cleaned[name] = frame.dropna(subset=feature_cols + ["wunderground_tmax"])
        dropped[name] = before - len(cleaned[name])
    return cleaned["train"], cleaned["val"], cleaned["test"], dropped


def _filter_reliable_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["wunderground_tmax"] = pd.to_numeric(out["wunderground_tmax"], errors="coerce")
    out["n_readings"] = pd.to_numeric(out.get("n_readings"), errors="coerce")
    out = out[out["wunderground_tmax"].notna()]
    if "n_readings" in out.columns:
        out = out[out["n_readings"] >= 12]
    elif "reliable" in out.columns:
        out = out[out["reliable"].astype(bool)]
    return out.sort_values("date")


def _load_old_models(city: str) -> tuple[list, list[str]] | None:
    city_dir = OLD_MODEL_DIR / city
    paths = [
        city_dir / "ridge.joblib",
        city_dir / "huber.joblib",
        city_dir / "lightgbm.joblib",
        city_dir / "feature_cols.json",
    ]
    if not all(p.exists() for p in paths):
        return None
    feature_cols = json.loads((city_dir / "feature_cols.json").read_text(encoding="utf-8"))
    models = [
        joblib.load(city_dir / "ridge.joblib"),
        joblib.load(city_dir / "huber.joblib"),
        joblib.load(city_dir / "lightgbm.joblib"),
    ]
    return models, feature_cols


def _evaluate_old_model_on_test(city: str, test_df: pd.DataFrame) -> dict[str, float] | None:
    loaded = _load_old_models(city)
    if loaded is None:
        print(f"{city}: old Track-B models not found, skipping comparison")
        return None
    models, feature_cols = loaded
    trackb_path = TRACKB_DATA_DIR / city / "features.parquet"
    if not trackb_path.exists():
        print(f"{city}: old features.parquet missing, skipping comparison")
        return None

    old_features = pd.read_parquet(trackb_path)
    old_features["date"] = pd.to_datetime(old_features["date"])
    test_dates = pd.to_datetime(test_df["date"])
    aligned = old_features[old_features["date"].isin(test_dates)].copy()
    aligned = aligned.merge(
        test_df[["date", "wunderground_tmax"]],
        on="date",
        how="inner",
    )
    missing_cols = [c for c in feature_cols if c not in aligned.columns]
    if missing_cols:
        print(f"{city}: old model missing feature columns {missing_cols}, skipping comparison")
        return None
    aligned = aligned.dropna(subset=feature_cols)
    if aligned.empty:
        return None

    y_true = pd.to_numeric(aligned["wunderground_tmax"], errors="coerce").to_numpy(dtype=float)
    y_pred = ensemble_predict(models, aligned[feature_cols])
    return compute_extended_metrics(y_true, y_pred)


def train_city(city: str, df: pd.DataFrame, config: dict) -> dict:
    df = _filter_reliable_rows(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    train_df, val_df, test_df = _split_dataframe(df)
    print(f"{city}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    if len(train_df) < MIN_TRAIN_ROWS:
        print(f"WARNING: {city} has only {len(train_df)} training rows (< {MIN_TRAIN_ROWS})")

    candidate_cols = _candidate_feature_cols(df)
    feature_cols = _select_features(train_df, candidate_cols)
    print(f"{city}: retained {len(feature_cols)} features")

    train_df, val_df, test_df, dropped = _forward_fill_splits(train_df, val_df, test_df, feature_cols)
    for split_name, n_drop in dropped.items():
        print(f"{city}: dropped {n_drop} rows from {split_name} after forward-fill")
    print(f"{city}: final splits train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = pd.to_numeric(train_df["wunderground_tmax"], errors="coerce").to_numpy(dtype=float)
    X_val = val_df[feature_cols]
    y_val = pd.to_numeric(val_df["wunderground_tmax"], errors="coerce").to_numpy(dtype=float)
    X_test = test_df[feature_cols]
    y_test = pd.to_numeric(test_df["wunderground_tmax"], errors="coerce").to_numpy(dtype=float)

    best_alpha, best_val_mae = None, float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(X_train, y_train)
        val_preds = pipe.predict(X_val)
        mae = float(np.mean(np.abs(val_preds - y_val)))
        if mae < best_val_mae:
            best_alpha, best_val_mae = alpha, mae

    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    ridge_pipe.fit(X_train, y_train)

    huber_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(epsilon=1.35, max_iter=1000)),
        ]
    )
    huber_pipe.fit(X_train, y_train)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    models = [ridge_pipe, huber_pipe, lgb_model]
    val_preds = ensemble_predict(models, X_val)
    test_preds = ensemble_predict(models, X_test)

    val_metrics = compute_extended_metrics(y_val, val_preds)
    test_metrics = compute_extended_metrics(y_test, test_preds)
    sigma_test = backsolve_sigma(test_metrics["hit_rate_1f"])

    old_test_metrics = _evaluate_old_model_on_test(city, test_df)

    out_dir = MODEL_DIR / city
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, out_dir / "ridge.joblib")
    joblib.dump(huber_pipe, out_dir / "huber.joblib")
    joblib.dump(lgb_model, out_dir / "lgb.joblib")
    with open(out_dir / "feature_cols.json", "w", encoding="utf-8") as handle:
        json.dump(feature_cols, handle, indent=2)
        handle.write("\n")

    metrics = {
        "city": city,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "best_ridge_alpha": best_alpha,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sigma_test": sigma_test,
        "old_cli_model_test_metrics_vs_wu": old_test_metrics,
        "old_sigma_cli": config.get(city, {}).get("trackb_sigma_f"),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
        handle.write("\n")

    return {
        "city": city,
        "metrics": metrics,
        "test_df": test_df,
    }


def build_all_features(cities: list[str], config: dict) -> dict[str, pd.DataFrame]:
    if not WU_TARGETS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {WU_TARGETS_PATH}. Run scripts/fetch_wunderground_target.py first."
        )
    wu_targets = pd.read_parquet(WU_TARGETS_PATH)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}
    for city in cities:
        print(f"\n=== Building features for {city} ===")
        features = build_polymarket_features(city, config, wu_targets)
        out_path = FEATURES_DIR / f"{city}.parquet"
        features.to_parquet(out_path, index=False)
        print(f"{city}: wrote {len(features)} rows to {out_path}")
        tables[city] = features
    return tables


def _print_main_summary(results: list[dict]) -> None:
    rows = []
    for item in results:
        m = item["metrics"]
        tm = m["test_metrics"]
        vm = m["val_metrics"]
        rows.append(
            {
                "City": item["city"],
                "Train N": m["n_train"],
                "Val N": m["n_val"],
                "Test N": m["n_test"],
                "Val MAE": vm["mae"],
                "Test MAE": tm["mae"],
                "Test RMSE": tm["rmse"],
                "Test Sigma": tm["sigma_errors"],
                "Test +-1F": tm["hit_rate_1f"],
                "Test +-2F": tm["hit_rate_2f"],
                "Bucket HR": tm["bucket_hr"],
                "Test Bias": tm["bias"],
            }
        )
    table = pd.DataFrame(rows)
    print("\n=== NEW MODEL SUMMARY ===")
    print(
        table.to_string(
            index=False,
            formatters={
                "Val MAE": "{:0.2f}".format,
                "Test MAE": "{:0.2f}".format,
                "Test RMSE": "{:0.2f}".format,
                "Test Sigma": "{:0.2f}".format,
                "Test +-1F": "{:0.1%}".format,
                "Test +-2F": "{:0.1%}".format,
                "Bucket HR": "{:0.1%}".format,
                "Test Bias": "{:0.2f}".format,
            },
        )
    )


def _print_comparison_table(results: list[dict]) -> None:
    rows = []
    for item in results:
        m = item["metrics"]
        old = m.get("old_cli_model_test_metrics_vs_wu")
        new_mae = m["test_metrics"]["mae"]
        old_mae = old["mae"] if old else float("nan")
        improvement = old_mae - new_mae if old else float("nan")
        rows.append(
            {
                "City": item["city"],
                "Old Model MAE (vs WU)": old_mae,
                "New Model MAE (vs WU)": new_mae,
                "Improvement": improvement,
                "Old Bucket HR": old["bucket_hr"] if old else float("nan"),
                "New Bucket HR": m["test_metrics"]["bucket_hr"],
            }
        )
    table = pd.DataFrame(rows)
    print("\n=== OLD (CLI) vs NEW (WU) COMPARISON (2026 test, vs WU actuals) ===")
    print(
        table.to_string(
            index=False,
            formatters={
                "Old Model MAE (vs WU)": "{:0.2f}".format,
                "New Model MAE (vs WU)": "{:0.2f}".format,
                "Improvement": "{:0.2f}".format,
                "Old Bucket HR": "{:0.1%}".format,
                "New Bucket HR": "{:0.1%}".format,
            },
        )
    )


def _print_sigma_table(results: list[dict], config: dict) -> None:
    rows = []
    for item in results:
        city = item["city"]
        rows.append(
            {
                "City": city,
                "Old Sigma (CLI)": config.get(city, {}).get("trackb_sigma_f"),
                "New Sigma (WU-trained)": item["metrics"]["sigma_test"],
            }
        )
    table = pd.DataFrame(rows)
    print("\n=== SIGMA BACK-SOLVE (test +-1F hit rate vs WU) ===")
    print(
        table.to_string(
            index=False,
            formatters={
                "Old Sigma (CLI)": "{:0.3f}".format,
                "New Sigma (WU-trained)": "{:0.3f}".format,
            },
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Polymarket Track-B on Wunderground targets.")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    config = _load_config()
    cities = [args.city] if args.city else list(POLYMARKET_CITIES)

    if args.train_only:
        tables = {}
        for city in cities:
            path = FEATURES_DIR / f"{city}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing feature table: {path}")
            tables[city] = pd.read_parquet(path)
    else:
        tables = build_all_features(cities, config)

    if args.features_only:
        print("\nFeatures built; skipping training.")
        return

    results: list[dict] = []
    for city in cities:
        print(f"\n=== Training {city} ===")
        results.append(train_city(city, tables[city], config))

    _print_main_summary(results)
    _print_comparison_table(results)
    _print_sigma_table(results, config)
    print("\nPolymarket Track-B training complete.")


if __name__ == "__main__":
    main()
