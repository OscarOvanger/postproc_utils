"""Upload processed Track-B artifacts to HuggingFace dataset repo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_nws_forecast import TRAIN_CITIES  # noqa: E402

HF_REPO_ID = "oovanger/MCP_datset"


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _metrics_json_path() -> Path:
    csv_path = PROJECT_ROOT / "models" / "trackj" / "austin" / "metrics.csv"
    json_path = PROJECT_ROOT / "models" / "trackj" / "austin" / "metrics.json"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    return json_path


def _upload_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for city in TRAIN_CITIES:
        local = PROJECT_ROOT / "data" / "trackb" / city / "features.parquet"
        if local.exists():
            files.append((local, f"data/trackb/{city}/features.parquet"))
    nws = PROJECT_ROOT / "data" / "trackb" / "nws_forecasts_raw.parquet"
    if nws.exists():
        files.append((nws, "data/trackb/nws_forecasts_raw.parquet"))
    openmeteo = PROJECT_ROOT / "data" / "trackb" / "openmeteo_nwp_raw.parquet"
    if openmeteo.exists():
        files.append((openmeteo, "data/trackb/openmeteo_nwp_raw.parquet"))
    for partition in ("threshold_opt", "time_holdout", "location_holdout", "true_holdout"):
        local = PROJECT_ROOT / "data" / "splits" / f"{partition}.parquet"
        if local.exists():
            files.append((local, f"data/splits/{partition}.parquet"))
    test_pred = PROJECT_ROOT / "models" / "trackj" / "austin" / "test_predictions.parquet"
    if test_pred.exists():
        files.append((test_pred, "models/trackj/austin/test_predictions.parquet"))
    metrics_json = _metrics_json_path()
    if metrics_json.exists():
        files.append((metrics_json, "models/trackj/austin/metrics.json"))
    config = PROJECT_ROOT / "config" / "city_config.json"
    if config.exists():
        files.append((config, "config/city_config.json"))
    return files


def main() -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = _upload_files()
    if not files:
        print("No files to upload.")
        return
    for local_path, hf_path in files:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=token,
        )
        print(f"Uploaded: {local_path} -> HF: {hf_path} ({_mb(local_path):.2f} MB)")

    # Verify Austin features
    from huggingface_hub import hf_hub_download

    verify_hf = "data/trackb/austin/features.parquet"
    local_verify = PROJECT_ROOT / "data" / "trackb" / "austin" / "features.parquet"
    if local_verify.exists():
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=verify_hf,
            local_dir=PROJECT_ROOT / "data" / "cache" / "hf_verify",
            token=token,
            force_download=True,
        )
        local_rows = len(pd.read_parquet(local_verify))
        remote_rows = len(pd.read_parquet(downloaded))
        assert local_rows == remote_rows, f"Row count mismatch: local={local_rows} remote={remote_rows}"
        print(f"Verified: {verify_hf} row count matches local ({local_rows} rows)")


if __name__ == "__main__":
    main()
