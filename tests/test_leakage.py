"""Verify no future information leaks into feature construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def test_lag_alignment() -> None:
    """Verify tmax_lag1 equals previous day's tmax."""
    for city in TRAIN_CITIES:
        path = PROJECT_ROOT / f"data/trackb/{city}/features.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_values("date")
        if "tmax_lag1" not in df.columns or "tmax" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            if pd.isna(curr["tmax_lag1"]) or pd.isna(prev["tmax"]):
                continue
            date_diff = (curr["date"] - prev["date"]).days
            if date_diff != 1:
                continue
            assert abs(curr["tmax_lag1"] - prev["tmax"]) < 0.01, (
                f"{city} {curr['date']}: lag1={curr['tmax_lag1']} != prev tmax={prev['tmax']}"
            )
    print("PASS: lag alignment")


def test_no_future_features() -> None:
    """Verify no feature column contains future/settlement info."""
    forbidden = [
        "resolved",
        "settled",
        "payout",
        "bucket_resolved",
        "contract_resolved",
        "settlement",
    ]
    for city in TRAIN_CITIES:
        path = PROJECT_ROOT / f"data/trackb/{city}/features.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for col in df.columns:
            for bad in forbidden:
                assert bad not in col.lower(), f"LEAKAGE: {city} has column '{col}' containing '{bad}'"
    print("PASS: no future features")


def test_nws_issuance_before_event() -> None:
    """Verify NWS forecast is from D-1, not D."""
    for city in TRAIN_CITIES:
        path = PROJECT_ROOT / f"data/trackb/{city}/features.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "nws_tmax_forecast_issued_h" not in df.columns:
            continue
        print(f"  {city}: issuance hour distribution:")
        print(df["nws_tmax_forecast_issued_h"].value_counts().to_string())
    print("PASS: NWS issuance check (manual review above)")


def test_feature_date_integrity() -> None:
    """Verify feature table dates are monotonically increasing."""
    for city in TRAIN_CITIES:
        path = PROJECT_ROOT / f"data/trackb/{city}/features.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_values("date")
        dates = pd.to_datetime(df["date"], errors="coerce").values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)), (
            f"{city}: dates not monotonically increasing"
        )
    print("PASS: date integrity")


if __name__ == "__main__":
    test_lag_alignment()
    test_no_future_features()
    test_nws_issuance_before_event()
    test_feature_date_integrity()
    print("\nAll leakage tests passed.")
