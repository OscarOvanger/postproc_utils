"""Build Track-B feature tables for all train cities."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.build_trackB_features import build_trackB_features, summarize_trackB_table  # noqa: E402
from trackj.fetch_nws_forecast import TRAIN_CITIES  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
TRACKB_DIR = PROJECT_ROOT / "data" / "trackb"
NWS_PATH = TRACKB_DIR / "nws_forecasts_raw.parquet"
RAW_DIR = TRACKJ_DIR / "raw"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 6, 9)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summaries: list[dict] = []
    for city in TRAIN_CITIES:
        print(f"Building Track-B features for {city}...")
        features = build_trackB_features(
            config[city],
            START_DATE,
            END_DATE,
            RAW_DIR,
            TRACKB_DIR,
            NWS_PATH,
            trackj_dir=TRACKJ_DIR,
            include_gfs=True,
            no_fetch=True,
        )
        summaries.append(summarize_trackB_table(city, features))
    summary = pd.DataFrame(summaries)
    print("\n=== TRACK-B FEATURE TABLE SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
