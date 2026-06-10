"""Unified local / HuggingFace data access for Track-B features and splits."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HF_REPO_ID = "oovanger/MCP_datset"
HF_CACHE = Path("data/cache/hf_processed")
TRACKB_DIR = Path("data/trackb")
SPLITS_DIR = Path("data/splits")

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

SPLIT_PARTITIONS = ("threshold_opt", "time_holdout", "location_holdout", "true_holdout")


def _hf_download(hf_path: str, token: str | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for source='hf'. Install repo requirements first."
        ) from exc
    token = token or os.environ.get("HF_TOKEN")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=hf_path,
        local_dir=HF_CACHE,
        token=token,
    )
    return Path(local_path)


def load_features(city: str, source: str = "local") -> pd.DataFrame:
    """Load Track-B feature table for a train city."""
    if source not in {"local", "hf"}:
        raise ValueError(f"source must be 'local' or 'hf', got {source!r}")
    rel_path = f"data/trackb/{city}/features.parquet"
    if source == "local":
        path = TRACKB_DIR / city / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing local features: {path}")
        return pd.read_parquet(path)
    path = _hf_download(rel_path)
    return pd.read_parquet(path)


def load_splits(partition: str, source: str = "local") -> pd.DataFrame:
    """Load a data split partition (threshold_opt, time_holdout, location_holdout, true_holdout)."""
    if partition not in SPLIT_PARTITIONS:
        raise ValueError(f"partition must be one of {SPLIT_PARTITIONS}, got {partition!r}")
    if source not in {"local", "hf"}:
        raise ValueError(f"source must be 'local' or 'hf', got {source!r}")
    if partition == "true_holdout" and source == "local":
        raise AssertionError(
            "true_holdout can only be loaded in final reporting mode (source='hf')"
        )
    rel_path = f"data/splits/{partition}.parquet"
    if source == "local":
        path = SPLITS_DIR / f"{partition}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing local split: {path}")
        return pd.read_parquet(path)
    path = _hf_download(rel_path)
    return pd.read_parquet(path)
