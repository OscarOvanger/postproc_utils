from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.track_j import bucket_probs_from_point_forecast


def _real_austin_buckets() -> pd.DataFrame:
    cols = [
        "city",
        "event_date",
        "bucket_label",
        "bucket_type",
        "bucket_lower_inclusive_f",
        "bucket_upper_inclusive_f",
    ]
    df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "threshold_opt.parquet", columns=cols)
    day = df[df["city"].eq("Austin")].sort_values("event_date").iloc[0]["event_date"]
    return (
        df[df["city"].eq("Austin") & df["event_date"].eq(day)]
        [
            [
                "bucket_label",
                "bucket_type",
                "bucket_lower_inclusive_f",
                "bucket_upper_inclusive_f",
            ]
        ]
        .drop_duplicates("bucket_label")
        .reset_index(drop=True)
    )


def test_probs_sum_to_one_for_real_austin_day() -> None:
    probs = bucket_probs_from_point_forecast(89.0, 1.2, _real_austin_buckets())

    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_forecast_bucket_has_highest_probability() -> None:
    probs = bucket_probs_from_point_forecast(89.0, 1.2, _real_austin_buckets())

    assert max(probs, key=probs.get) == "89-90"


def test_boundary_forecast_degrades_gracefully() -> None:
    buckets = pd.DataFrame(
        [
            {"bucket_label": "88-89", "bucket_type": "RANGE", "bucket_lower_inclusive_f": 88, "bucket_upper_inclusive_f": 89},
            {"bucket_label": "89-90", "bucket_type": "RANGE", "bucket_lower_inclusive_f": 89, "bucket_upper_inclusive_f": 90},
        ]
    )

    probs = bucket_probs_from_point_forecast(88.5, 1.2, buckets)

    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert probs["88-89"] > probs["89-90"]
    assert probs["89-90"] > 0.0


def test_tail_buckets_sum_with_interior_ranges() -> None:
    buckets = pd.DataFrame(
        [
            {"bucket_label": "<88", "bucket_type": "LESS_THAN", "bucket_lower_inclusive_f": pd.NA, "bucket_upper_inclusive_f": 88},
            {"bucket_label": "88-89", "bucket_type": "RANGE", "bucket_lower_inclusive_f": 88, "bucket_upper_inclusive_f": 89},
            {"bucket_label": "89-90", "bucket_type": "RANGE", "bucket_lower_inclusive_f": 89, "bucket_upper_inclusive_f": 90},
            {"bucket_label": ">90", "bucket_type": "GREATER_THAN", "bucket_lower_inclusive_f": 90, "bucket_upper_inclusive_f": pd.NA},
        ]
    )

    probs = bucket_probs_from_point_forecast(89.0, 1.2, buckets)

    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert probs["<88"] == pytest.approx(probs[">90"])
    assert probs["88-89"] == pytest.approx(probs["89-90"])
