"""Compute CLI vs ASOS daily-max bias for Polymarket Wunderground settlement."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.trackj.build_asos_features import ASOS_FIELDS  # noqa: E402
from src.trackj.build_ngboost_features import build_asos_daily_max_map  # noqa: E402
from src.trackj.fetch_cli_target import _correction_rank, parse_file  # noqa: E402

BIAS_CITIES = ["austin", "houston", "los_angeles", "san_francisco"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "polymarket"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
DEFAULT_START_DATE = date(2023, 1, 1)


def _issue_date_from_path(path: Path) -> date | None:
    stamp = path.stem[:8]
    if len(stamp) == 8 and stamp.isdigit():
        return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
    return None


def _month_key(d: date) -> str:
    return f"{d.year:04d}{d.month:02d}"


def _month_keys_between(start_date: date, end_date: date) -> set[str]:
    cursor = date(start_date.year, start_date.month, 1)
    keys: set[str] = set()
    while cursor <= end_date:
        keys.add(_month_key(cursor))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return keys


def _load_city_config() -> dict:
    with open(CITY_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _load_cli(
    city: str,
    city_config: dict,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Load CLI Tmax from cached raw products (full history, not recent parquet slice)."""
    cli_raw_dir = PROJECT_ROOT / "data" / "trackj" / "raw" / city / "cli"
    if not cli_raw_dir.exists():
        raise FileNotFoundError(f"Missing CLI raw dir: {cli_raw_dir}")

    cli_paths = sorted(cli_raw_dir.glob("*.txt"))
    if start_date is not None or end_date is not None:
        lo = (start_date - timedelta(days=7)) if start_date else date.min
        hi = (end_date + timedelta(days=7)) if end_date else date.max
        filtered: list[Path] = []
        for path in cli_paths:
            issue_date = _issue_date_from_path(path)
            if issue_date is None or lo <= issue_date <= hi:
                filtered.append(path)
        cli_paths = filtered

    parsed = pd.DataFrame([parse_file(path, city_config) for path in cli_paths])
    if parsed.empty:
        return pd.DataFrame(columns=["date", "tmax_f"])
    parsed["issue_dt"] = pd.to_datetime(parsed["report_issue_timestamp_utc"], utc=True, errors="coerce")
    parsed["correction_rank"] = parsed["source_product_id"].map(_correction_rank)
    selected = (
        parsed[parsed["date"].notna()]
        .sort_values(["date", "correction_rank", "issue_dt", "source_product_id"])
        .groupby("date", as_index=False)
        .tail(1)
    )
    cli = selected[["date", "tmax_f"]].copy()
    cli["tmax_f"] = pd.to_numeric(cli["tmax_f"], errors="coerce")
    return cli.loc[cli["tmax_f"].notna(), ["date", "tmax_f"]].copy()


def _load_all_asos(
    raw_dirs: list[Path],
    station: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Load cached ASOS month files from root + asos/ subdir."""
    from io import StringIO

    month_keys: set[str] | None = None
    if start_date is not None and end_date is not None:
        month_keys = _month_keys_between(start_date, end_date)

    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()
    pattern = f"{station.lower()}_asos_*.csv"
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        for path in sorted(raw_dir.glob(pattern)):
            if path in seen_paths:
                continue
            if month_keys is not None:
                month_token = path.stem.rsplit("_", 1)[-1]
                if month_token not in month_keys:
                    continue
            seen_paths.add(path)
            text = path.read_text(encoding="utf-8")
            if not text.strip() or text.startswith("ERROR"):
                continue
            frame = pd.read_csv(StringIO(text), na_values=["null", "M", ""], keep_default_na=True)
            if not frame.empty:
                frame["raw_file"] = str(path)
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station", "valid", *ASOS_FIELDS])
    df = pd.concat(frames, ignore_index=True)
    df["valid_local"] = pd.to_datetime(df["valid"], errors="coerce")
    df = df[df["valid_local"].notna()].copy()
    df["date"] = df["valid_local"].dt.strftime("%Y-%m-%d")
    return df


def _compare_city(
    city: str,
    city_config: dict,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    cli = _load_cli(city, city_config[city], start_date, end_date)
    if start_date is not None:
        cli = cli[cli["date"] >= start_date.isoformat()]
    if end_date is not None:
        cli = cli[cli["date"] <= end_date.isoformat()]
    if cli.empty:
        return pd.DataFrame(columns=["city", "date", "cli_tmax", "asos_daily_max", "bias"])

    station = city_config[city]["nws_station"]
    city_raw = PROJECT_ROOT / "data" / "trackj" / "raw" / city
    asos = _load_all_asos([city_raw, city_raw / "asos"], station, start_date, end_date)
    asos_max = build_asos_daily_max_map(asos)

    rows: list[dict[str, object]] = []
    for _, row in cli.iterrows():
        day = str(row["date"])
        asos_daily_max = asos_max.get(day)
        if asos_daily_max is None:
            continue
        cli_tmax = float(row["tmax_f"])
        rows.append(
            {
                "city": city,
                "date": day,
                "cli_tmax": cli_tmax,
                "asos_daily_max": float(asos_daily_max),
                "bias": cli_tmax - float(asos_daily_max),
            }
        )
    return pd.DataFrame(rows)


def _city_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    bias = frame["bias"]
    return {
        "mean_bias": round(float(bias.mean()), 3),
        "median_bias": round(float(bias.median()), 3),
        "std": round(float(bias.std(ddof=1)), 3) if len(bias) > 1 else 0.0,
        "min": round(float(bias.min()), 3),
        "max": round(float(bias.max()), 3),
        "n_days": int(len(frame)),
    }


def _save_histogram(city: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frame["bias"], bins=30, color="#4878CF", edgecolor="white", alpha=0.85)
    ax.axvline(frame["bias"].median(), color="#E68A2E", linestyle="--", linewidth=1.2, label="Median")
    ax.axvline(0, color="#8A8A8A", linestyle=":", linewidth=0.8)
    ax.set_title(f"CLI minus ASOS daily max bias — {city.replace('_', ' ').title()}")
    ax.set_xlabel("Bias (°F): CLI Tmax − ASOS hourly max")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"wunderground_bias_{city}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Wunderground settlement bias (CLI vs ASOS max).")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=f"YYYY-MM-DD inclusive (default: {DEFAULT_START_DATE.isoformat()})",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="YYYY-MM-DD inclusive (default: yesterday)",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date) if args.start_date else DEFAULT_START_DATE
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)
    print(f"Bias window: {start_date.isoformat()} to {end_date.isoformat()}")
    city_config = _load_city_config()

    detail_frames: list[pd.DataFrame] = []
    summary: dict[str, dict[str, float | int]] = {}

    for city in BIAS_CITIES:
        frame = _compare_city(city, city_config, start_date, end_date)
        if frame.empty:
            print(f"WARNING: no overlapping CLI/ASOS days for {city}")
            continue
        detail_frames.append(frame)
        summary[city] = _city_summary(frame)
        _save_histogram(city, frame)

    if not detail_frames:
        raise SystemExit("No bias comparisons computed.")

    detail = pd.concat(detail_frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(OUTPUT_DIR / "wunderground_bias_detail.parquet", index=False)
    with open(OUTPUT_DIR / "wunderground_bias.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Wunderground Bias Summary (CLI Tmax − ASOS daily max) ===")
    print(f"{'City':<18} {'N':>6} {'Mean':>7} {'Median':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    for city in BIAS_CITIES:
        if city not in summary:
            continue
        row = summary[city]
        print(
            f"{city:<18} {row['n_days']:>6d} "
            f"{row['mean_bias']:>7.2f} {row['median_bias']:>7.2f} "
            f"{row['std']:>7.2f} {row['min']:>7.2f} {row['max']:>7.2f}"
        )
    print(f"\nSaved: {OUTPUT_DIR / 'wunderground_bias.json'}")
    print(f"Saved: {OUTPUT_DIR / 'wunderground_bias_detail.parquet'}")


if __name__ == "__main__":
    main()
