"""Train Track-B ensemble models for all 9 train cities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.trackj.build_trackA_table import TRACK_A_COVARIATES  # noqa: E402

TRACKB_DATA_DIR = PROJECT_ROOT / "data" / "trackb"
TRACKJ_DATA_DIR = PROJECT_ROOT / "data" / "trackj"
MODEL_DIR = PROJECT_ROOT / "models" / "trackb"
TRACKJ_MODEL_DIR = PROJECT_ROOT / "models" / "trackj"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"

TRAIN_CITIES = [
    "austin",
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "oklahoma_city",
    "philadelphia",
    "phoenix",
    "san_francisco",
]

META_COLS = [
    "date",
    "tmax_f",
    "city",
    "station",
    "nwp_is_ecmwf",
    "nwp_source",
    "nws_issuance_hour",
    "nws_cycle",
    "issued_time",
    "nwp_issued_date",
    "nwp_valid_date",
]

LEAKAGE_SUBSTRINGS = ("target", "actual", "resolved", "settled", "bucket")
TRACKA_MODEL_FILES = ["ridge.joblib", "huber.joblib", "lightgbm.joblib"]


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def backsolve_sigma(hit_rate_1f: float) -> float:
    """Given P(|error| <= 1) = h, solve h = 2*Phi(1/sigma) - 1."""
    if hit_rate_1f <= 0 or hit_rate_1f >= 1:
        return float("nan")
    z = norm.ppf((hit_rate_1f + 1) / 2)
    if z <= 0:
        return float("nan")
    return 1.0 / z


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "hit_rate_1f": float(np.mean(np.abs(errors) <= 1.0)),
        "hit_rate_2f": float(np.mean(np.abs(errors) <= 2.0)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "n": int(len(y_true)),
    }


def ensemble_predict(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack([m.predict(X) for m in models])
    return np.round(np.mean(preds, axis=1)).astype(int)


def _candidate_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in META_COLS
        and not any(x in c.lower() for x in LEAKAGE_SUBSTRINGS)
    ]


def _select_features(
    train_df: pd.DataFrame, candidate_cols: list[str]
) -> list[str]:
    y = pd.to_numeric(train_df["tmax_f"], errors="coerce")
    retained = []
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


def _forward_fill_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.sort_values("date")
    combined[feature_cols] = combined[feature_cols].ffill()

    splits = {
        "train": combined[combined["date"] < "2025-01-01"].copy(),
        "val": combined[(combined["date"] >= "2025-01-01") & (combined["date"] < "2026-01-01")].copy(),
        "test": combined[combined["date"] >= "2026-01-01"].copy(),
    }
    dropped = {}
    cleaned = {}
    for name, frame in splits.items():
        before = len(frame)
        cleaned[name] = frame.dropna(subset=feature_cols + ["tmax_f"])
        dropped[name] = before - len(cleaned[name])
    return cleaned["train"], cleaned["val"], cleaned["test"], dropped


def _assert_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, city: str) -> None:
    assert len(train_df) > 100, f"{city}: insufficient training data ({len(train_df)} rows)"
    train_dates = set(train_df["date"].dt.normalize())
    val_dates = set(val_df["date"].dt.normalize())
    test_dates = set(test_df["date"].dt.normalize())
    assert not train_dates & val_dates, f"{city}: train/val date overlap"
    assert not train_dates & test_dates, f"{city}: train/test date overlap"
    assert not val_dates & test_dates, f"{city}: val/test date overlap"
    assert train_df["date"].max() < val_df["date"].min(), f"{city}: train.max >= val.min"
    assert val_df["date"].max() < test_df["date"].min(), f"{city}: val.max >= test.min"
    assert (train_df["date"] < pd.Timestamp("2025-01-01")).all(), f"{city}: train dates not all < 2025"
    assert (
        (val_df["date"] >= pd.Timestamp("2025-01-01"))
        & (val_df["date"] < pd.Timestamp("2026-01-01"))
    ).all(), f"{city}: val dates not in [2025, 2026)"
    assert (test_df["date"] >= pd.Timestamp("2026-01-01")).all(), f"{city}: test dates not all >= 2026"


def _lgbm_importance(lgb_model: lgb.LGBMRegressor, feature_cols: list[str], top_n: int = 10) -> dict[str, float]:
    raw = lgb_model.booster_.feature_importance(importance_type="split")
    total = float(raw.sum()) or 1.0
    pairs = sorted(zip(feature_cols, raw / total), key=lambda x: x[1], reverse=True)[:top_n]
    return {name: float(score) for name, score in pairs}


def _load_tracka_models(city: str) -> dict[str, object] | None:
    city_dir = TRACKJ_MODEL_DIR / city
    paths = [city_dir / name for name in TRACKA_MODEL_FILES]
    if not all(path.exists() for path in paths):
        return None
    return {
        "ridge": joblib.load(city_dir / "ridge.joblib"),
        "huber": joblib.load(city_dir / "huber.joblib"),
        "lightgbm": joblib.load(city_dir / "lightgbm.joblib"),
    }


def _tracka_ensemble_predict(models: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack([model.predict(frame[TRACK_A_COVARIATES]) for model in models.values()])
    return np.rint(preds.mean(axis=1))


def _score_tracka_on_test(
    city: str,
    test_df: pd.DataFrame,
) -> dict[str, float] | None:
    """Re-score Track-A (or Track-J for Austin) on the 2026+ test window."""
    if test_df.empty:
        return None

    y_true = pd.to_numeric(test_df["tmax_f"], errors="coerce").to_numpy(dtype=float)

    if city == "austin":
        pred_path = TRACKJ_MODEL_DIR / "austin" / "test_predictions.parquet"
        if pred_path.exists():
            preds_df = pd.read_parquet(pred_path)
            date_col = "date" if "date" in preds_df.columns else "event_date"
            pred_col = None
            for candidate in (
                "pred_ensemble_rounded",
                "pred_ensemble_calibrated",
                "prediction",
                "tmax_f_pred",
                "track_j_tmax_f",
            ):
                if candidate in preds_df.columns:
                    pred_col = candidate
                    break
            if pred_col is None:
                return None
            preds_df[date_col] = pd.to_datetime(preds_df[date_col])
            merged = test_df[["date", "tmax_f"]].merge(
                preds_df[[date_col, pred_col]].rename(columns={date_col: "date", pred_col: "pred"}),
                on="date",
                how="inner",
            )
            if merged.empty:
                return None
            y_pred = np.rint(merged["pred"].to_numpy(dtype=float))
            y_aligned = merged["tmax_f"].to_numpy(dtype=float)
            return compute_metrics(y_aligned, y_pred)

        table_path = TRACKJ_DATA_DIR / "austin" / "trackA_table.parquet"
        models = _load_tracka_models("austin")
        if not table_path.exists() or models is None:
            return None
        tracka = pd.read_parquet(table_path)
        tracka["date"] = pd.to_datetime(tracka["date"])
        tracka_cols = [c for c in ["date", "tmax_f", *TRACK_A_COVARIATES] if c in tracka.columns]
        aligned = test_df[["date"]].merge(tracka[tracka_cols], on="date", how="inner")
        aligned = aligned.dropna(subset=[c for c in TRACK_A_COVARIATES if c in aligned.columns])
        if aligned.empty:
            return None
        y_pred = _tracka_ensemble_predict(models, aligned)
        y_aligned = pd.to_numeric(aligned["tmax_f"], errors="coerce").to_numpy(dtype=float)
        return compute_metrics(y_aligned, y_pred)

    table_path = TRACKJ_DATA_DIR / city / "trackA_table.parquet"
    models = _load_tracka_models(city)
    if not table_path.exists() or models is None:
        return None
    tracka = pd.read_parquet(table_path)
    tracka["date"] = pd.to_datetime(tracka["date"])
    tracka_cols = [c for c in ["date", "tmax_f", *TRACK_A_COVARIATES] if c in tracka.columns]
    aligned = test_df[["date"]].merge(tracka[tracka_cols], on="date", how="inner")
    aligned = aligned.dropna(subset=[c for c in TRACK_A_COVARIATES if c in aligned.columns])
    if aligned.empty:
        return None
    y_pred = _tracka_ensemble_predict(models, aligned)
    y_aligned = pd.to_numeric(aligned["tmax_f"], errors="coerce").to_numpy(dtype=float)
    return compute_metrics(y_aligned, y_pred)


def _train_city(city: str) -> dict:
    features_path = TRACKB_DATA_DIR / city / "features.parquet"
    df = pd.read_parquet(features_path)
    if "tmax_f" not in df.columns and "tmax" in df.columns:
        df = df.rename(columns={"tmax": "tmax_f"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["tmax_f"])

    train_df = df[df["date"] < "2025-01-01"].copy()
    val_df = df[(df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")].copy()
    test_df = df[df["date"] >= "2026-01-01"].copy()
    _assert_splits(train_df, val_df, test_df, city)
    print(f"{city}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    nws_col_present = "nws_tmax_forecast_f" in df.columns

    candidate_cols = _candidate_feature_cols(df)
    feature_cols = _select_features(train_df, candidate_cols)
    print(f"{city}: retained {len(feature_cols)} features: {feature_cols}")

    train_df, val_df, test_df, dropped = _forward_fill_splits(train_df, val_df, test_df, feature_cols)
    for split_name, n_drop in dropped.items():
        print(f"{city}: dropped {n_drop} rows from {split_name} after forward-fill")

    X_train = train_df[feature_cols]
    y_train = pd.to_numeric(train_df["tmax_f"], errors="coerce").to_numpy(dtype=float)
    X_val = val_df[feature_cols]
    y_val = pd.to_numeric(val_df["tmax_f"], errors="coerce").to_numpy(dtype=float)
    X_test = test_df[feature_cols]
    y_test = pd.to_numeric(test_df["tmax_f"], errors="coerce").to_numpy(dtype=float)

    assert "date" not in X_train.columns and "tmax_f" not in X_train.columns
    assert not X_train.isna().any().any(), f"{city}: NaN in X_train after preprocessing"

    best_alpha, best_val_mae = None, float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        mae = float(np.mean(np.abs(preds - y_val)))
        if mae < best_val_mae:
            best_alpha, best_val_mae = alpha, mae
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    ridge_pipe.fit(X_train, y_train)

    huber_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(epsilon=1.35, max_iter=1000)),
    ])
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
    assert np.issubdtype(val_preds.dtype, np.integer), f"{city}: val ensemble preds not integers"
    assert np.issubdtype(test_preds.dtype, np.integer), f"{city}: test ensemble preds not integers"

    val_metrics = compute_metrics(y_val, val_preds)
    test_metrics = compute_metrics(y_test, test_preds)
    sigma_val = backsolve_sigma(val_metrics["hit_rate_1f"])
    sigma_test = backsolve_sigma(test_metrics["hit_rate_1f"])

    if (
        np.isfinite(sigma_val)
        and np.isfinite(sigma_test)
        and sigma_test > 0
        and abs(sigma_val - sigma_test) / sigma_test > 0.30
    ):
        print(
            f"WARNING: {city} sigma_val={sigma_val:.2f} vs sigma_test={sigma_test:.2f}, "
            f"divergence={abs(sigma_val - sigma_test) / sigma_test * 100:.0f}%"
        )

    ridge_test_mae = float(np.mean(np.abs(ridge_pipe.predict(X_test) - y_test)))
    huber_test_mae = float(np.mean(np.abs(huber_pipe.predict(X_test) - y_test)))
    lgb_test_mae = float(np.mean(np.abs(lgb_model.predict(X_test) - y_test)))

    nws_raw_test_mae = float("nan")
    if nws_col_present and "nws_tmax_forecast_f" in test_df.columns:
        aligned_nws = pd.to_numeric(test_df["nws_tmax_forecast_f"], errors="coerce")
        valid = aligned_nws.notna().to_numpy()
        if valid.any():
            nws_raw_test_mae = float(
                np.mean(np.abs(aligned_nws.to_numpy()[valid] - y_test[valid]))
            )

    lgb_importance = _lgbm_importance(lgb_model, feature_cols)

    out_dir = MODEL_DIR / city
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, out_dir / "ridge.joblib")
    joblib.dump(huber_pipe, out_dir / "huber.joblib")
    joblib.dump(lgb_model, out_dir / "lightgbm.joblib")
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
        "sigma_val": sigma_val,
        "per_learner_test_mae": {
            "ridge": ridge_test_mae,
            "huber": huber_test_mae,
            "lightgbm": lgb_test_mae,
        },
        "nws_raw_test_mae": nws_raw_test_mae,
        "lgbm_feature_importance": lgb_importance,
    }
    tracka_metrics = _score_tracka_on_test(city, test_df)
    if tracka_metrics is not None:
        metrics["tracka_aligned_test_metrics"] = tracka_metrics

    metrics_dir = TRACKB_DATA_DIR / city
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
        handle.write("\n")

    print(f"\n{city} LightGBM top-10 feature importance (split, normalized):")
    for feat, score in lgb_importance.items():
        print(f"  {feat}: {score:.4f}")

    return {
        "city": city,
        "metrics": metrics,
        "tracka_metrics": tracka_metrics,
        "X_train": X_train,
        "sigma_test": sigma_test,
    }


def _print_summary_table(results: list[dict]) -> None:
    rows = []
    for item in results:
        m = item["metrics"]
        rows.append({
            "City": item["city"],
            "N train": m["n_train"],
            "N val": m["n_val"],
            "N test": m["n_test"],
            "N feat": m["n_features"],
            "Val MAE": m["val_metrics"]["mae"],
            "Test MAE": m["test_metrics"]["mae"],
            "Test ±1°F": m["test_metrics"]["hit_rate_1f"],
            "Sigma": m["sigma_test"],
        })
    table = pd.DataFrame(rows)
    print("\nTable 1: Per-city Track-B training summary")
    print(
        table.to_string(
            index=False,
            formatters={
                "Val MAE": "{:0.2f}".format,
                "Test MAE": "{:0.2f}".format,
                "Test ±1°F": "{:0.1%}".format,
                "Sigma": "{:0.2f}".format,
            },
        )
    )


def _print_comparison_table(results: list[dict]) -> None:
    rows = []
    for item in results:
        m = item["metrics"]
        ta = item.get("tracka_metrics")
        tracka_mae = ta["mae"] if ta else None
        trackb_mae = m["test_metrics"]["mae"]
        improvement = (tracka_mae - trackb_mae) if tracka_mae is not None else None
        rows.append({
            "City": item["city"],
            "Track-A MAE": f"{tracka_mae:0.2f}" if tracka_mae is not None else "N/A",
            "Track-B MAE": f"{trackb_mae:0.2f}",
            "Improvement": f"{improvement:0.2f}" if improvement is not None else "N/A",
            "Track-A ±1°F": f"{ta['hit_rate_1f']:0.1%}" if ta else "N/A",
            "Track-B ±1°F": f"{m['test_metrics']['hit_rate_1f']:0.1%}",
            "Raw NWS MAE": (
                f"{m['nws_raw_test_mae']:0.2f}"
                if np.isfinite(m["nws_raw_test_mae"])
                else "N/A"
            ),
        })
    print("\nTable 2: Track-B vs Track-A vs raw NWS comparison (2026+ test period)")
    print(pd.DataFrame(rows).to_string(index=False))


def _defensive_checks(results: list[dict]) -> None:
    for item in results:
        city = item["city"]
        sigma = item["sigma_test"]
        assert 0.5 <= sigma <= 5.0, f"{city}: sigma_test={sigma} outside [0.5, 5.0]"
        out_dir = MODEL_DIR / city
        for fname in ["ridge.joblib", "huber.joblib", "lightgbm.joblib", "feature_cols.json"]:
            assert (out_dir / fname).exists(), f"{city}: missing {fname}"

    assert len(results) == len(TRAIN_CITIES), "Not all cities trained"

    script_text = Path(__file__).read_text(encoding="utf-8")
    forbidden = "true_holdout" + ".parquet"
    load_markers = ("read_parquet", "load_parquet", "pd.read_parquet")
    assert not any(marker in script_text and forbidden in script_text for marker in load_markers)


def main() -> None:
    config = _load_config()
    results: list[dict] = []

    for city in TRAIN_CITIES:
        print(f"\n=== Training {city} ===")
        result = _train_city(city)
        results.append(result)
        m = result["metrics"]
        config.setdefault(city, {})
        config[city]["trackb_sigma_f"] = m["sigma_test"]
        config[city]["trackb_test_hit_rate_1f"] = m["test_metrics"]["hit_rate_1f"]
        config[city]["trackb_test_mae"] = m["test_metrics"]["mae"]
        config[city]["trackb_n_features"] = m["n_features"]

    _save_config(config)
    _print_summary_table(results)
    _print_comparison_table(results)
    _defensive_checks(results)
    print("\nTrack-B training complete for all 9 cities.")


if __name__ == "__main__":
    main()
