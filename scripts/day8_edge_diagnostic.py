"""Day 8 Track-J in-sample edge diagnostic and bucket edge table."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINES_DIR = SRC_DIR / "baselines"
for path in (PROJECT_ROOT, SRC_DIR, BASELINES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from entry_interface import filter_to_trading_window  # noqa: E402
from fees import taker_fee  # noqa: E402
from snapshot_stability import compute_modal_bucket, stability_entry  # noqa: E402
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_ROOT = PROJECT_ROOT / "models" / "trackj"
TRACK_J_DIR = PROJECT_ROOT / "data" / "track_j"
FORECASTS_PATH = TRACK_J_DIR / "forecasts.parquet"
AUSTIN_PREDICTIONS_PATH = MODEL_ROOT / "austin" / "test_predictions.parquet"
THRESHOLD_OPT_PATH = SPLIT_DIR / "threshold_opt.parquet"
OUTPUT_PATH = TRACK_J_DIR / "bucket_edges_IS.parquet"
DAY8_SIGMA_F = 1.21
AUSTIN_GAP_START = pd.Timestamp("2026-05-02")
AUSTIN_GAP_END = pd.Timestamp("2026-05-14")


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _date_key(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _is_austin_gap(event_date: object) -> bool:
    date_value = pd.to_datetime(event_date)
    return AUSTIN_GAP_START <= date_value <= AUSTIN_GAP_END


def _prediction_value_column(columns: list[str]) -> str | None:
    candidates = [
        "pred_ensemble_rounded",
        "track_j_tmax_f",
        "prediction",
        "pred",
        "y_pred",
        "forecast",
        "tmax_forecast_f",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _date_column(columns: list[str]) -> str | None:
    for candidate in ("event_date", "date", "target_date"):
        if candidate in columns:
            return candidate
    return None


def _normalize_prediction_frame(
    frame: pd.DataFrame,
    city: str,
    prediction_col: str,
    date_col: str,
) -> pd.DataFrame:
    predictions = frame[[date_col, prediction_col]].copy()
    predictions = predictions.rename(
        columns={date_col: "event_date", prediction_col: "track_j_tmax_f"}
    )
    predictions["city"] = _city_key(city)
    predictions["event_date"] = pd.to_datetime(
        predictions["event_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    predictions["track_j_tmax_f"] = pd.to_numeric(
        predictions["track_j_tmax_f"], errors="coerce"
    )
    predictions["track_j_sigma_f"] = DAY8_SIGMA_F
    predictions["city_coverage_flag"] = predictions["track_j_tmax_f"].notna()
    return predictions[
        [
            "city",
            "event_date",
            "track_j_tmax_f",
            "track_j_sigma_f",
            "city_coverage_flag",
        ]
    ].dropna(subset=["event_date", "track_j_tmax_f"])


def _load_austin_predictions() -> pd.DataFrame:
    if not AUSTIN_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(AUSTIN_PREDICTIONS_PATH)
    prediction_col = _prediction_value_column(list(frame.columns))
    date_col = _date_column(list(frame.columns))
    if prediction_col is None or date_col is None:
        return pd.DataFrame()
    predictions = _normalize_prediction_frame(frame, "austin", prediction_col, date_col)
    gap_mask = predictions["event_date"].map(_is_austin_gap)
    return predictions[~gap_mask].copy()


def _load_forecast_predictions(city_keys: set[str]) -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        return pd.DataFrame()
    forecasts = pd.read_parquet(FORECASTS_PATH)
    required = {"city", "event_date", "track_j_tmax_f"}
    if required.difference(forecasts.columns):
        return pd.DataFrame()
    predictions = forecasts.copy()
    predictions["city"] = predictions["city"].map(_city_key)
    predictions = predictions[predictions["city"].isin(city_keys)].copy()
    predictions["event_date"] = pd.to_datetime(
        predictions["event_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    predictions["track_j_tmax_f"] = pd.to_numeric(
        predictions["track_j_tmax_f"], errors="coerce"
    )
    predictions["track_j_sigma_f"] = DAY8_SIGMA_F
    predictions["city_coverage_flag"] = predictions["track_j_tmax_f"].notna()
    cols = [
        "city",
        "event_date",
        "track_j_tmax_f",
        "track_j_sigma_f",
        "city_coverage_flag",
    ]
    return predictions[cols].dropna(subset=["event_date", "track_j_tmax_f"])


def _load_city_prediction_parquets(city: str) -> pd.DataFrame:
    city_dir = MODEL_ROOT / city
    if not city_dir.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(city_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        prediction_col = _prediction_value_column(list(frame.columns))
        date_col = _date_column(list(frame.columns))
        if prediction_col is None or date_col is None:
            continue
        frames.append(_normalize_prediction_frame(frame, city, prediction_col, date_col))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_day8_predictions(partition_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    city_keys = {_city_key(city) for city in partition_df["city"].dropna().unique()}
    frames = [_load_austin_predictions()]
    forecast_predictions = _load_forecast_predictions(city_keys)
    if not forecast_predictions.empty:
        frames.append(forecast_predictions[~forecast_predictions["city"].eq("austin")])

    for city in sorted(city_keys - {"austin"}):
        frames.append(_load_city_prediction_parquets(city))

    predictions = (
        pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in frames)
        else pd.DataFrame(
            columns=[
                "city",
                "event_date",
                "track_j_tmax_f",
                "track_j_sigma_f",
                "city_coverage_flag",
            ]
        )
    )
    if not predictions.empty:
        predictions["city"] = predictions["city"].map(_city_key)
        predictions["event_date"] = pd.to_datetime(
            predictions["event_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        predictions["track_j_sigma_f"] = DAY8_SIGMA_F
        predictions["city_coverage_flag"] = predictions["track_j_tmax_f"].notna()
        predictions = predictions.dropna(subset=["event_date", "track_j_tmax_f"])
        predictions = predictions.drop_duplicates(["city", "event_date"], keep="first")

    partition_days = partition_df[["city", "event_date"]].drop_duplicates().copy()
    partition_days["city"] = partition_days["city"].map(_city_key)
    partition_days["event_date"] = pd.to_datetime(partition_days["event_date"]).dt.strftime("%Y-%m-%d")
    covered_cities = sorted(
        predictions.merge(partition_days, on=["city", "event_date"], how="inner")["city"].unique()
    )
    skipped = sorted(city for city in city_keys if city not in set(covered_cities))
    return predictions, covered_cities, skipped


def _prediction_for_day(predictions: pd.DataFrame, city: str, event_date: str) -> float | None:
    if predictions.empty:
        return None
    rows = predictions[
        predictions["city"].eq(_city_key(city))
        & predictions["event_date"].eq(_date_key(event_date))
    ]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[0]["track_j_tmax_f"], errors="coerce")
    return None if pd.isna(value) else float(value)


def _resolved_yes(day_df: pd.DataFrame, bucket_label: str) -> bool:
    rows = day_df[day_df["bucket_label"].astype(str).eq(str(bucket_label))]
    if rows.empty:
        raise ValueError(f"bucket_label {bucket_label} not found")
    values = rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(values) != 1:
        raise ValueError(f"bucket_label {bucket_label} has inconsistent resolution")
    return bool(values[0])


def _bucket_defs(snapshot: pd.DataFrame) -> pd.DataFrame:
    return snapshot[
        [
            "bucket_label",
            "bucket_type",
            "bucket_lower_inclusive_f",
            "bucket_upper_inclusive_f",
        ]
    ].drop_duplicates("bucket_label")


def _bucket_midpoint(row: pd.Series) -> float:
    bucket_type = str(row["bucket_type"])
    lower = pd.to_numeric(row["bucket_lower_inclusive_f"], errors="coerce")
    upper = pd.to_numeric(row["bucket_upper_inclusive_f"], errors="coerce")
    if bucket_type == "RANGE":
        return float(lower + upper) / 2.0
    if bucket_type == "LESS_THAN":
        return float(upper)
    if bucket_type == "GREATER_THAN":
        return float(lower)
    raise ValueError(f"Unsupported bucket_type: {bucket_type}")


def _closest_midpoint_bucket(forecast: float, buckets: pd.DataFrame) -> str:
    scored = buckets.copy()
    scored["_midpoint"] = scored.apply(_bucket_midpoint, axis=1)
    scored["_distance"] = (scored["_midpoint"] - float(forecast)).abs()
    return str(scored.sort_values(["_distance", "_midpoint"]).iloc[0]["bucket_label"])


def _entry_snapshot(day_df: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.Timestamp] | None:
    trading_day = filter_to_trading_window(day_df)
    if trading_day.empty:
        return None
    signal = stability_entry(trading_day, k=1)
    if signal.no_signal:
        return None
    snapshot = trading_day[
        pd.to_datetime(trading_day["snapshot_time_local"]).eq(signal.entry_snapshot_time)
    ].copy()
    market_bucket = compute_modal_bucket(trading_day, signal.entry_snapshot_time)
    return snapshot, market_bucket, signal.entry_snapshot_time


def _format_pct(value: float) -> str:
    return "nan%" if not math.isfinite(value) else f"{100.0 * value:0.1f}%"


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame[column].astype(bool).mean())


def _print_task1(
    diagnostic: pd.DataFrame,
    austin: pd.DataFrame,
    cities_with_predictions: list[str],
    cities_skipped: list[str],
) -> None:
    agree = diagnostic[diagnostic["agreement"]]
    disagree = diagnostic[~diagnostic["agreement"]]
    austin_agree = austin[austin["agreement"]]
    austin_disagree = austin[~austin["agreement"]]

    agreement_rate = _rate(diagnostic, "agreement")
    tj_agree = _rate(agree, "track_j_win")
    tj_disagree = _rate(disagree, "track_j_win")
    market_disagree = _rate(disagree, "market_win")
    increment_pp = (tj_disagree - market_disagree) * 100.0
    edge_found = bool(math.isfinite(increment_pp) and increment_pp > 0)
    verdict = "EDGE FOUND" if edge_found else "NO EDGE"
    direction = "above" if edge_found else "below"

    print("=== Track-J IS Edge Diagnostic ===")
    print(f"Cities with predictions: {cities_with_predictions}")
    print(f"Cities skipped (no predictions): {cities_skipped}")
    print()
    print("--- Aggregated across all cities with predictions ---")
    print(f"Total IS days with predictions: {len(diagnostic)}")
    print(f"Agreement rate: {_format_pct(agreement_rate)} ({len(agree)} days)")
    print(f"Track-J win rate (agreement days): {_format_pct(tj_agree)} ({len(agree)} days)")
    print(f"Track-J win rate (disagreement days): {_format_pct(tj_disagree)} ({len(disagree)} days)")
    print(f"Market win rate (disagreement days): {_format_pct(market_disagree)} ({len(disagree)} days)")
    print(f"Track-J incremental win rate on disagreement days: {increment_pp:0.1f} pp vs market")
    print()
    print("--- Austin only ---")
    print(f"Total IS days with predictions: {len(austin)}")
    print(f"Agreement rate: {_format_pct(_rate(austin, 'agreement'))} ({len(austin_agree)} days)")
    print(f"Track-J win rate (agreement days): {_format_pct(_rate(austin_agree, 'track_j_win'))} ({len(austin_agree)} days)")
    print(f"Track-J win rate (disagreement days): {_format_pct(_rate(austin_disagree, 'track_j_win'))} ({len(austin_disagree)} days)")
    print(f"Market win rate (disagreement days): {_format_pct(_rate(austin_disagree, 'market_win'))} ({len(austin_disagree)} days)")
    print()
    print(f"VERDICT: {verdict} — Track-J win rate on disagreement days")
    print(f"  is {abs(increment_pp):0.1f} pp {direction} market win rate on those same days.")
    print(f"  Proceeding to bucket edge calculation: {'YES' if edge_found else 'NO'}")


def run_edge_diagnostic(
    partition_df: pd.DataFrame,
    predictions: pd.DataFrame,
    cities_with_predictions: list[str],
    cities_skipped: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_cols = ["city", "event_date"]
    for (city, event_date), day_df in partition_df.groupby(group_cols, sort=True):
        city_key = _city_key(city)
        date_key = _date_key(event_date)
        forecast = _prediction_for_day(predictions, city_key, date_key)
        if forecast is None:
            continue
        entry = _entry_snapshot(day_df)
        if entry is None:
            continue
        snapshot, market_bucket, _ = entry
        bucket_defs = _bucket_defs(snapshot)
        bucket_probs_from_point_forecast(forecast, DAY8_SIGMA_F, bucket_defs)
        track_j_bucket = _closest_midpoint_bucket(forecast, bucket_defs)
        records.append(
            {
                "city": city_key,
                "event_date": date_key,
                "track_j_bucket": track_j_bucket,
                "market_bucket": market_bucket,
                "agreement": str(track_j_bucket) == str(market_bucket),
                "track_j_win": _resolved_yes(day_df, track_j_bucket),
                "market_win": _resolved_yes(day_df, market_bucket),
            }
        )

    diagnostic = pd.DataFrame.from_records(records)
    austin = diagnostic[diagnostic["city"].eq("austin")].copy() if not diagnostic.empty else diagnostic
    _print_task1(diagnostic, austin, cities_with_predictions, cities_skipped)
    return diagnostic


def _fee_probability_units(contracts: float, price: float) -> float:
    return float(taker_fee(contracts, price)) / (100.0 * contracts)


def run_bucket_edges(partition_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    prediction_days = 0
    tradeable_days: set[tuple[str, str]] = set()
    for (city, event_date), day_df in partition_df.groupby(["city", "event_date"], sort=True):
        city_key = _city_key(city)
        date_key = _date_key(event_date)
        forecast = _prediction_for_day(predictions, city_key, date_key)
        if forecast is None:
            continue
        entry = _entry_snapshot(day_df)
        if entry is None:
            continue
        prediction_days += 1
        snapshot, _, _ = entry
        probs = bucket_probs_from_point_forecast(forecast, DAY8_SIGMA_F, _bucket_defs(snapshot))
        day_has_tradeable = False
        for row in snapshot.drop_duplicates("bucket_label").itertuples(index=False):
            label = str(row.bucket_label)
            p = float(probs[label])
            c = float(row.yes_mid_close)
            edge = p - c
            kelly_f = (edge / (1.0 - c)) if edge > 0 and c < 1.0 else 0.0
            fee_taker = _fee_probability_units(100.0, c)
            tradeable = bool(edge > 2.0 * fee_taker and c >= 0.15)
            day_has_tradeable = day_has_tradeable or tradeable
            records.append(
                {
                    "city": city_key,
                    "event_date": date_key,
                    "bucket_label": label,
                    "p": p,
                    "c": c,
                    "edge": edge,
                    "kelly_f": kelly_f,
                    "fee_taker": fee_taker,
                    "tradeable": tradeable,
                }
            )
        if day_has_tradeable:
            tradeable_days.add((city_key, date_key))

    edges = pd.DataFrame.from_records(records)
    TRACK_J_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(OUTPUT_PATH, index=False)

    mean_edge = float(edges["edge"].mean()) if not edges.empty else float("nan")
    std_edge = float(edges["edge"].std(ddof=1)) if edges.shape[0] > 1 else float("nan")
    positive_pct = float(edges["edge"].gt(0).mean()) if not edges.empty else float("nan")
    tradeable_pct = float(edges["tradeable"].mean()) if not edges.empty else float("nan")
    tradeable_day_pct = len(tradeable_days) / prediction_days if prediction_days else float("nan")
    sufficient = bool(math.isfinite(tradeable_day_pct) and tradeable_day_pct >= 0.30)

    print()
    print("=== Track-J IS Bucket Edge Distribution ===")
    print(f"Total bucket-days computed: {len(edges)}")
    print(f"Mean edge: {mean_edge:0.3f}")
    print(f"Std edge: {std_edge:0.3f}")
    print(f"% positive edge buckets: {100.0 * positive_pct:0.1f}%")
    print(f"% tradeable (edge > 2*fee, price >= 0.15): {100.0 * tradeable_pct:0.1f}%")
    print(
        f"Days with at least one tradeable bucket: {len(tradeable_days)} "
        f"({100.0 * tradeable_day_pct:0.1f}% of prediction days)"
    )
    print()
    print(f"VERDICT: {'SIGNAL SUFFICIENT' if sufficient else 'SIGNAL TOO WEAK'}")
    if sufficient:
        print(f"  {100.0 * tradeable_day_pct:0.1f}% of days have at least one tradeable bucket.")
    else:
        print("  fewer than 30% of days have at least one tradeable bucket.")
    print("  Threshold for proceeding: 30% of prediction days.")
    return edges


def main() -> None:
    threshold_opt = pd.read_parquet(THRESHOLD_OPT_PATH)
    predictions, cities_with_predictions, cities_skipped = load_day8_predictions(threshold_opt)
    run_edge_diagnostic(threshold_opt, predictions, cities_with_predictions, cities_skipped)
    run_bucket_edges(threshold_opt, predictions)


if __name__ == "__main__":
    main()
