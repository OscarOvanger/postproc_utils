"""Diagnose NWS forecast horizon, issuance cycles, and IEM MOS uncertainty columns."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_nws_forecast import (  # noqa: E402
    TRAIN_CITIES,
    _fetch_iem_mos_table,
    _load_month_mos_frame,
    make_session,
)
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NWS_PATH = PROJECT_ROOT / "data" / "trackb" / "nws_forecasts_raw.parquet"
TRACKJ_DIR = PROJECT_ROOT / "data" / "trackj"
DIAG_PATH = PROJECT_ROOT / "diagnostics" / "nws_diagnose.txt"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"


def _lines_to_output(lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    print(text)
    DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAG_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    lines: list[str] = []
    forecasts = pd.read_parquet(NWS_PATH)
    austin = forecasts[forecasts["city"].eq("austin")].copy()
    cli_path = TRACKJ_DIR / "austin" / "cli_target.parquet"
    cli = pd.read_parquet(cli_path)
    cli["date"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    sample_dates = austin["date"].drop_duplicates().sample(5, random_state=42).tolist()
    sample = austin[austin["date"].isin(sample_dates)].merge(cli[["date", "tmax_f"]], on="date", how="left")
    lines.append("=== AUSTIN SAMPLE ROWS (5 random dates) ===")
    for _, row in sample.sort_values("date").iterrows():
        lines.append(
            f"  date={row['date']}  issued_time={row['issued_time']}  valid_date={row['valid_date']}  "
            f"forecast={row['tmax_forecast_f']}°F  actual_cli={row.get('tmax_f', 'n/a')}°F"
        )

    austin["issued"] = pd.to_datetime(austin["issued_time"], utc=True)
    austin["valid"] = pd.to_datetime(austin["valid_date"], utc=True)
    lead_hours = (austin["valid"] - austin["issued"]).dt.total_seconds() / 3600.0
    lines.append("\n=== FORECAST LEAD TIME (hours) ===")
    lines.append(f"  mean={lead_hours.mean():.2f}  min={lead_hours.min():.2f}  max={lead_hours.max():.2f}")
    for pct in (10, 25, 50, 75, 90):
        lines.append(f"  p{pct}={np.percentile(lead_hours.dropna(), pct):.2f}")
    if lead_hours.mean() > 30:
        lines.append("  DIAGNOSIS: mean lead >30h suggests multi-day horizon bug.")
    else:
        lines.append("  DIAGNOSIS: mean lead ~12-24h — horizon is next-day, not multi-day.")

    lines.append("\n=== ISSUANCE HOUR (UTC) HISTOGRAM — Austin ===")
    hour_counts = austin["issued"].dt.hour.value_counts().sort_index()
    for hour, count in hour_counts.items():
        lines.append(f"  {hour:02d}Z: {count}")

    session = make_session()
    sample_month = date(2023, 2, 1)
    frame = _load_month_mos_frame("KAUS", sample_month, session)
    lines.append("\n=== IEM MOS UNCERTAINTY COLUMNS (KAUS NBE sample month) ===")
    lines.append(f"  columns: {list(frame.columns)}")
    quantile_cols = [c for c in frame.columns if any(k in c.lower() for k in ("p10", "p90", "q10", "q90", "pct"))]
    if quantile_cols:
        lines.append(f"  quantile-like columns found: {quantile_cols}")
    else:
        lines.append("  No p10/p90 quantile columns in IEM NBE/NBS MOS export.")
        lines.append("  Spread flags (tsd/gsd/wsd) exist but are not calibrated uncertainty bounds.")
        lines.append("  Defer nws_tmax_p10_f / nws_tmax_p90_f to NBM quantiles in a later week.")

    _lines_to_output(lines)


if __name__ == "__main__":
    main()
