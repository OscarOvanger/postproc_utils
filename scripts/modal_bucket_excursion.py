"""Analyze modal-bucket price excursions after 10AM entry on Kalshi snapshot data."""

from __future__ import annotations

import sys
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

from snapshot_stability import compute_modal_bucket  # noqa: E402
from src.data_store import TRAIN_CITIES  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DETAIL_PATH = PROJECT_ROOT / "data" / "modal_bucket_excursion.parquet"
REPORT_DIR = PROJECT_ROOT / "reports"

PLOT_MFE = REPORT_DIR / "modal_bucket_mfe.png"
PLOT_BY_CITY = REPORT_DIR / "modal_bucket_excursion_by_city.png"
PLOT_TIME_TO_EXIT = REPORT_DIR / "modal_bucket_time_to_exit.png"
PLOT_MFE_VS_ENTRY = REPORT_DIR / "modal_bucket_mfe_vs_entry.png"

COLUMNS = [
    "city",
    "source_city_folder",
    "event_date",
    "snapshot_time_local",
    "bucket_label",
    "bucket_type",
    "bucket_lower_inclusive_f",
    "bucket_upper_inclusive_f",
    "yes_mid_close",
    "bucket_resolved_to_one_dollars",
]

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
THRESHOLD_LABELS = {t: f"{int(round(t * 100))}c" for t in THRESHOLDS}


def _to_naive_local(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_localize(None)
    return parsed


def load_snapshot_data() -> pd.DataFrame:
    """Load train-city snapshots from threshold_opt and time_holdout parquets."""
    frames: list[pd.DataFrame] = []
    for partition in ("threshold_opt", "time_holdout"):
        path = SPLIT_DIR / f"{partition}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split parquet: {path}")
        df = pd.read_parquet(path, columns=COLUMNS)
        df["partition"] = partition
        frames.append(df)
        print(f"loaded {partition}: {len(df):,} rows", flush=True)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["source_city_folder"] = out["source_city_folder"].astype(str)
    out = out[out["source_city_folder"].isin(TRAIN_CITIES)].copy()
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.date
    out["snapshot_time_local"] = _to_naive_local(out["snapshot_time_local"])
    out["yes_mid_close"] = pd.to_numeric(out["yes_mid_close"], errors="coerce")
    return out


def _filter_trading_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep same-day snapshots on or after 10:00 local."""
    snap = df["snapshot_time_local"]
    event_dates = pd.to_datetime(df["event_date"])
    same_day = snap.dt.date == event_dates.dt.date
    after_open = (snap.dt.hour > 10) | ((snap.dt.hour == 10) & (snap.dt.minute >= 0))
    return df.loc[same_day & after_open].copy()


def _settlement_for_bucket(bucket_df: pd.DataFrame) -> bool:
    resolved = bucket_df["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved) != 1:
        raise ValueError("Inconsistent bucket_resolved_to_one_dollars for modal bucket")
    return bool(resolved[0])


def _first_time_to_threshold(
    path: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    threshold: float,
) -> float | None:
    target = entry_price + threshold
    for _, row in path.iterrows():
        price = float(row["yes_mid_close"])
        if price >= target:
            snap_time = pd.Timestamp(row["snapshot_time_local"])
            return (snap_time - entry_time).total_seconds() / 60.0
    return None


def _analyze_city_day(day_df: pd.DataFrame) -> dict[str, object] | None:
    trading = _filter_trading_window(day_df)
    trading = trading.dropna(subset=["yes_mid_close"])
    if trading.empty:
        return None

    entry_time = pd.Timestamp(trading["snapshot_time_local"].min())
    entry_snapshot = trading[trading["snapshot_time_local"] == entry_time]
    if entry_snapshot.empty:
        return None

    modal_bucket = compute_modal_bucket(trading, entry_time)
    modal_rows = entry_snapshot[entry_snapshot["bucket_label"].astype(str) == modal_bucket]
    if modal_rows.empty:
        return None
    entry_price = float(modal_rows["yes_mid_close"].iloc[0])
    if not np.isfinite(entry_price):
        return None

    bucket_day = day_df[day_df["bucket_label"].astype(str) == modal_bucket].copy()
    try:
        settlement_outcome = _settlement_for_bucket(bucket_day)
    except ValueError:
        return None

    path = bucket_day[
        pd.to_datetime(bucket_day["snapshot_time_local"]) > entry_time
    ].copy()
    path = path.dropna(subset=["yes_mid_close"]).sort_values("snapshot_time_local")

    record: dict[str, object] = {
        "source_city_folder": str(day_df["source_city_folder"].iloc[0]),
        "event_date": day_df["event_date"].iloc[0],
        "entry_time": entry_time,
        "modal_bucket": modal_bucket,
        "entry_price": entry_price,
        "settlement_outcome": settlement_outcome,
        "settlement_pnl": (1.0 - entry_price) if settlement_outcome else (-entry_price),
        "n_post_entry_snapshots": len(path),
    }

    if path.empty:
        record["max_price"] = np.nan
        record["min_price"] = np.nan
        record["mfe"] = np.nan
        record["mae"] = np.nan
        for threshold in THRESHOLDS:
            label = THRESHOLD_LABELS[threshold]
            record[f"reached_{label}"] = False
            record[f"time_to_{label}_min"] = np.nan
        return record

    prices = path["yes_mid_close"].astype(float)
    max_price = float(prices.max())
    min_price = float(prices.min())
    record["max_price"] = max_price
    record["min_price"] = min_price
    record["mfe"] = max_price - entry_price
    record["mae"] = entry_price - min_price

    for threshold in THRESHOLDS:
        label = THRESHOLD_LABELS[threshold]
        target = entry_price + threshold
        reached = bool((prices >= target).any())
        record[f"reached_{label}"] = reached
        record[f"time_to_{label}_min"] = (
            _first_time_to_threshold(path, entry_time, entry_price, threshold)
            if reached
            else np.nan
        )
    return record


def build_excursion_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per city-day with excursion and settlement metrics."""
    records: list[dict[str, object]] = []
    skipped = 0
    group_cols = ["source_city_folder", "event_date"]
    for _, day_df in df.groupby(group_cols, sort=True):
        try:
            row = _analyze_city_day(day_df)
        except ValueError:
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        records.append(row)

    if skipped:
        print(f"Skipped {skipped:,} city-days (missing data or ambiguous resolution)", flush=True)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.date
    return out.sort_values(["source_city_folder", "event_date"]).reset_index(drop=True)


def table_excursion_by_threshold(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build excursion summary table for each threshold."""
    rows: list[dict[str, object]] = []
    groups = [("all", df)] if group_cols is None else df.groupby(group_cols, sort=True)

    if group_cols is None:
        group_iter = groups
    else:
        group_iter = ((name, group) for name, group in groups)

    for group_name, group in group_iter:
        if isinstance(group_name, tuple):
            group_name = group_name[0] if len(group_name) == 1 else str(group_name)
        for threshold in THRESHOLDS:
            label = THRESHOLD_LABELS[threshold]
            reached_col = f"reached_{label}"
            time_col = f"time_to_{label}_min"
            reached = group[reached_col].astype(bool)
            not_reached = group[~reached]
            median_time = group.loc[reached, time_col].median()
            rows.append(
                {
                    "group": group_name if group_cols is not None else "all",
                    "threshold": label,
                    "hit_rate": float(reached.mean()) if len(group) else np.nan,
                    "median_time_min": float(median_time) if pd.notna(median_time) else np.nan,
                    "settlement_win_rate_when_not_reached": (
                        float(not_reached["settlement_outcome"].mean())
                        if len(not_reached)
                        else np.nan
                    ),
                    "n_city_days": int(len(group)),
                    "n_not_reached": int(len(not_reached)),
                }
            )
    return pd.DataFrame(rows)


def table_settlement_15c_split(df: pd.DataFrame) -> pd.DataFrame:
    """Compare settlement outcomes when 15c threshold is reached vs not."""
    reached = df["reached_15c"].astype(bool)
    rows: list[dict[str, object]] = []
    for label, mask in (("15c NOT reached", ~reached), ("15c reached", reached)):
        subset = df.loc[mask]
        rows.append(
            {
                "segment": label,
                "n_city_days": int(len(subset)),
                "settlement_win_rate": float(subset["settlement_outcome"].mean()) if len(subset) else np.nan,
                "mean_settlement_pnl": float(subset["settlement_pnl"].mean()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def table_settlement_15c_by_city(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city, city_df in df.groupby("source_city_folder", sort=True):
        reached = city_df["reached_15c"].astype(bool)
        for segment, mask in (("NOT reached", ~reached), ("reached", reached)):
            subset = city_df.loc[mask]
            rows.append(
                {
                    "source_city_folder": city,
                    "segment": segment,
                    "n_city_days": int(len(subset)),
                    "settlement_win_rate": (
                        float(subset["settlement_outcome"].mean()) if len(subset) else np.nan
                    ),
                    "mean_settlement_pnl": (
                        float(subset["settlement_pnl"].mean()) if len(subset) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_mfe_histogram(df: pd.DataFrame, out_path: Path) -> None:
    mfe = df["mfe"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(mfe, bins=40, color="#4878CF", edgecolor="white", alpha=0.85)
    for threshold, color in ((0.15, "#d62728"), (0.25, "#ff7f0e")):
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.5, label=f"{int(threshold*100)}c")
    ax.set_xlabel("Max favorable excursion (MFE, $)")
    ax.set_ylabel("City-days")
    ax.set_title("Modal bucket MFE after 10AM entry")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_excursion_by_city(df: pd.DataFrame, out_path: Path) -> None:
    by_city = (
        df.groupby("source_city_folder", sort=False)["reached_15c"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(by_city.index, by_city.values, color="#4878CF", alpha=0.85)
    ax.set_xlabel("Fraction of days reaching +15c after 10AM entry")
    ax.set_ylabel("City")
    ax.set_title("Modal bucket +15c excursion rate by city")
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_time_to_15c(df: pd.DataFrame, out_path: Path) -> None:
    times = df.loc[df["reached_15c"].astype(bool), "time_to_15c_min"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    if times.empty:
        ax.text(0.5, 0.5, "No 15c excursions", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(times, bins=30, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Minutes after 10AM entry")
    ax.set_ylabel("City-days")
    ax.set_title("Time to first +15c excursion (when reached)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mfe_vs_entry(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["mfe", "entry_price"]).copy()
    colors = np.where(plot_df["settlement_outcome"].astype(bool), "#2ca02c", "#d62728")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot_df["entry_price"], plot_df["mfe"], c=colors, alpha=0.45, s=18, edgecolors="none")
    ax.axhline(0.15, color="#888888", linestyle="--", linewidth=1.0, label="15c threshold")
    ax.set_xlabel("Entry price (yes_mid_close at 10AM modal bucket)")
    ax.set_ylabel("Max favorable excursion (MFE, $)")
    ax.set_title("MFE vs entry price (green=settlement win, red=loss)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_caveats() -> None:
    print("\n=== Caveats ===")
    print("- Entry at yes_mid_close; real maker fill would be below best ask (conservative).")
    print("- Exit feasibility uses midpoint, not best_bid; Polymarket exits need a larger mid move.")
    print("- No fees modeled; Kalshi taker exit would reduce net profit.")
    print("- Kalshi data settles on CLI (not Wunderground).")


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%" if np.isfinite(value) else "n/a"


def _format_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"


def print_summary(df: pd.DataFrame, table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame) -> None:
    n_days = len(df)
    date_min = df["event_date"].min()
    date_max = df["event_date"].max()

    print("\n=== Modal bucket excursion summary ===")
    print(f"City-days analyzed: {n_days:,}")
    print(f"Cities: {df['source_city_folder'].nunique()}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"MFE median: {_format_num(float(df['mfe'].median()))} | MAE median: {_format_num(float(df['mae'].median()))}")

    print("\n=== Table 1: Excursion rate by threshold (aggregate) ===")
    print(f"{'Threshold':<10} | {'Hit rate':>10} | {'Median min':>12} | {'Win rate if NOT reached':>24}")
    print("-" * 64)
    agg = table1[table1["group"] == "all"] if "group" in table1.columns else table1
    for _, row in agg.iterrows():
        print(
            f"{row['threshold']:<10} | {_format_pct(row['hit_rate']):>10} | "
            f"{_format_num(row['median_time_min']):>12} | "
            f"{_format_pct(row['settlement_win_rate_when_not_reached']):>24}"
        )

    print("\n=== Table 2: +15c excursion by city (sorted descending) ===")
    city_15 = table2[table2["threshold"] == "15c"].copy()
    city_15 = city_15.sort_values("hit_rate", ascending=False)
    print(f"{'City':<20} | {'Hit rate':>10} | {'Median min':>12} | {'Win rate if NOT reached':>24}")
    print("-" * 74)
    for _, row in city_15.iterrows():
        print(
            f"{str(row['group']):<20} | {_format_pct(row['hit_rate']):>10} | "
            f"{_format_num(row['median_time_min']):>12} | "
            f"{_format_pct(row['settlement_win_rate_when_not_reached']):>24}"
        )

    print("\n=== Table 3: Settlement analysis (15c split, aggregate) ===")
    print(f"{'Segment':<18} | {'N days':>8} | {'Win rate':>10} | {'Mean PnL':>10}")
    print("-" * 54)
    for _, row in table3.iterrows():
        print(
            f"{row['segment']:<18} | {int(row['n_city_days']):>8,} | "
            f"{_format_pct(row['settlement_win_rate']):>10} | "
            f"{_format_num(row['mean_settlement_pnl'], 3):>10}"
        )


def main() -> None:
    raw = load_snapshot_data()
    excursion = build_excursion_table(raw)
    if excursion.empty:
        raise RuntimeError("No city-days remained after analysis.")

    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    excursion.to_parquet(DETAIL_PATH, index=False)
    print(f"Wrote detail table to {DETAIL_PATH}", flush=True)

    table1 = table_excursion_by_threshold(excursion, group_cols=None)
    table1["group"] = "all"
    table2 = table_excursion_by_threshold(excursion, group_cols=["source_city_folder"])
    table2 = table2.rename(columns={"group": "source_city_folder"})
    table2["group"] = table2["source_city_folder"]
    table3 = table_settlement_15c_split(excursion)

    plot_mfe_histogram(excursion, PLOT_MFE)
    plot_excursion_by_city(excursion, PLOT_BY_CITY)
    plot_time_to_15c(excursion, PLOT_TIME_TO_EXIT)
    plot_mfe_vs_entry(excursion, PLOT_MFE_VS_ENTRY)

    print_summary(excursion, table1, table2, table3)
    print_caveats()

    print(f"\nSaved plots:")
    for path in (PLOT_MFE, PLOT_BY_CITY, PLOT_TIME_TO_EXIT, PLOT_MFE_VS_ENTRY):
        print(f"  {path}")


if __name__ == "__main__":
    main()
