"""Re-extract NWS forecasts from IEM month caches using current extraction logic."""

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

from trackj.fetch_nws_forecast import (  # noqa: E402
    TRAIN_CITIES,
    _extract_tmax_from_mos,
    _issued_before_for_target,
    _load_month_mos_frame,
    _month_starts,
    fetch_nws_tmax_forecast,
    make_session,
    print_coverage_table,
)
from zoneinfo import ZoneInfo

CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "trackb" / "nws_forecasts_raw.parquet"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 6, 9)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dates = sorted(pd.date_range(START_DATE, END_DATE, freq="D").date.tolist())
    session = make_session()
    rows: list[dict] = []
    for city in TRAIN_CITIES:
        if city not in config:
            continue
        cfg = config[city]
        station = str(cfg["nws_station"])
        local_tz = ZoneInfo(str(cfg["timezone"]))
        for month_start in _month_starts(dates[0], dates[-1]):
            month_dates = [d for d in dates if d.year == month_start.year and d.month == month_start.month]
            if not month_dates:
                continue
            frame = _load_month_mos_frame(station, month_start, session)
            for target_date in month_dates:
                issued_before = _issued_before_for_target(target_date, 22, local_tz)
                result = _extract_tmax_from_mos(frame, target_date, issued_before)
                if result is None:
                    result = fetch_nws_tmax_forecast(
                        float(cfg["lat"]),
                        float(cfg["lon"]),
                        target_date,
                        issued_before,
                        station=station,
                        session=session,
                    )
                if not result:
                    continue
                rows.append(
                    {
                        "city": city,
                        "date": target_date.isoformat(),
                        "station": station,
                        "tmax_forecast_f": result["tmax_forecast_f"],
                        "issued_time": result["issued_time"],
                        "valid_date": result["valid_date"],
                        "hours_since_issuance": result["hours_since_issuance"],
                    }
                )
            print(f"Reprocessed {city} {month_start:%Y-%m}: {len(month_dates)} dates")
    final = pd.DataFrame(rows).drop_duplicates(subset=["city", "date"], keep="last").sort_values(["city", "date"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(final)} rows to {OUTPUT_PATH}")
    print_coverage_table(final, config, trackj_dir=PROJECT_ROOT / "data" / "trackj")


if __name__ == "__main__":
    main()
