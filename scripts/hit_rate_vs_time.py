"""Compute and plot market modal-bucket hit rate vs time before day-end.

Settlement reference is the last available snapshot per city-day (end of trading
data), not the time of Tmax occurrence.
"""

from __future__ import annotations

import json
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

from src.data_store import TRAIN_CITIES  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
TRACKB_FORECASTS_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "hit_rate_vs_time.png"

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

BUCKET_COLS = [
    "bucket_label",
    "bucket_type",
    "bucket_lower_inclusive_f",
    "bucket_upper_inclusive_f",
]

HORIZONS_H = [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0]
BIN_WIDTH_H = 0.5
HORIZON_WINDOW_H = 0.25
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def load_city_timezones() -> dict[str, str]:
    """Return city slug -> IANA timezone from config/city_config.json."""
    with open(CITY_CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    return {slug: str(entry["timezone"]) for slug, entry in config.items()}


def _to_naive_local(series: pd.Series) -> pd.Series:
    """Parse datetimes and keep station-local wall-clock time without tz."""
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
    return out


def _filter_trading_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep same-day snapshots on or after 10:00 local (times are station-local naive)."""
    snap = df["snapshot_time_local"]
    event_dates = pd.to_datetime(df["event_date"])
    same_day = snap.dt.date == event_dates.dt.date
    after_open = (snap.dt.hour > 10) | ((snap.dt.hour == 10) & (snap.dt.minute >= 0))
    return df.loc[same_day & after_open].copy()


def _actual_bucket_for_day(day_df: pd.DataFrame) -> str:
    winners = day_df.loc[day_df["bucket_resolved_to_one_dollars"].astype(bool), "bucket_label"]
    unique = winners.astype(str).unique()
    if len(unique) != 1:
        raise ValueError(
            f"Expected one winning bucket for {day_df['source_city_folder'].iloc[0]} "
            f"{day_df['event_date'].iloc[0]}, found {len(unique)}"
        )
    return str(unique[0])


def _day_end_times(df: pd.DataFrame) -> pd.Series:
    """Last available snapshot timestamp per city-day (end of trading data)."""
    return df.groupby(["source_city_folder", "event_date"], sort=True)["snapshot_time_local"].transform("max")


def build_snapshot_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per snapshot with modal bucket, actual bucket, and hit flag."""
    filtered = _filter_trading_window(df)
    filtered = filtered.dropna(subset=["yes_mid_close"]).copy()
    filtered["yes_mid_close"] = pd.to_numeric(filtered["yes_mid_close"], errors="coerce")
    filtered = filtered.dropna(subset=["yes_mid_close"])
    filtered["day_end_time"] = _day_end_times(df)

    day_keys = ["source_city_folder", "event_date"]
    actual_map: dict[tuple[str, object], str] = {}
    for key, day_df in filtered.groupby(day_keys, sort=True):
        actual_map[key] = _actual_bucket_for_day(day_df)

    snapshot_records: list[dict[str, object]] = []
    group_cols = [*day_keys, "snapshot_time_local"]
    for key, snap_df in filtered.groupby(group_cols, sort=True):
        city, event_date, snapshot_time = key
        prices = snap_df["yes_mid_close"].astype(float)
        modal_idx = prices.idxmax()
        modal_bucket = str(snap_df.loc[modal_idx, "bucket_label"])
        actual_bucket = actual_map[(city, event_date)]
        day_end = pd.Timestamp(snap_df["day_end_time"].iloc[0])
        snapshot_ts = pd.Timestamp(snapshot_time)
        minutes_before_end = (day_end - snapshot_ts).total_seconds() / 60.0
        if minutes_before_end < 0:
            continue
        snapshot_records.append(
            {
                "source_city_folder": city,
                "event_date": event_date,
                "snapshot_time_local": snapshot_time,
                "day_end_time": day_end,
                "modal_bucket": modal_bucket,
                "actual_bucket": actual_bucket,
                "hit": int(modal_bucket == actual_bucket),
                "hours_before_settlement": minutes_before_end / 60.0,
            }
        )

    hits = pd.DataFrame(snapshot_records)
    if hits.empty:
        return hits
    hits["event_date"] = pd.to_datetime(hits["event_date"]).dt.date
    return hits.sort_values(["source_city_folder", "event_date", "snapshot_time_local"]).reset_index(
        drop=True
    )


def binned_hit_rates(hits: pd.DataFrame, *, bin_width: float = BIN_WIDTH_H) -> pd.DataFrame:
    """Aggregate hit rate into fixed-width bins on hours_before_settlement."""
    if hits.empty:
        return pd.DataFrame(columns=["bin_center", "hit_rate", "n_snapshots"])

    max_hours = hits["hours_before_settlement"].max()
    edges = np.arange(0.0, max_hours + bin_width, bin_width)
    if edges[-1] < max_hours:
        edges = np.append(edges, max_hours + bin_width)

    binned = hits.copy()
    binned["bin"] = pd.cut(
        binned["hours_before_settlement"],
        bins=edges,
        include_lowest=True,
        right=False,
    )
    agg = (
        binned.groupby("bin", observed=True)
        .agg(hit_rate=("hit", "mean"), n_snapshots=("hit", "size"))
        .reset_index()
    )
    agg["bin_center"] = agg["bin"].apply(lambda interval: interval.mid).astype(float)
    return agg.sort_values("bin_center", ascending=False).reset_index(drop=True)


def binned_hit_rates_by_city(hits: pd.DataFrame, *, bin_width: float = BIN_WIDTH_H) -> pd.DataFrame:
    """Per-city binned hit rates using the same global bin edges."""
    if hits.empty:
        return pd.DataFrame(columns=["source_city_folder", "bin_center", "hit_rate", "n_snapshots"])

    max_hours = hits["hours_before_settlement"].max()
    edges = np.arange(0.0, max_hours + bin_width, bin_width)
    if edges[-1] < max_hours:
        edges = np.append(edges, max_hours + bin_width)

    rows: list[pd.DataFrame] = []
    for city, city_hits in hits.groupby("source_city_folder", sort=True):
        city_copy = city_hits.copy()
        city_copy["bin"] = pd.cut(
            city_copy["hours_before_settlement"],
            bins=edges,
            include_lowest=True,
            right=False,
        )
        agg = (
            city_copy.groupby("bin", observed=True)
            .agg(hit_rate=("hit", "mean"), n_snapshots=("hit", "size"))
            .reset_index()
        )
        agg["bin_center"] = agg["bin"].apply(lambda interval: interval.mid).astype(float)
        agg["source_city_folder"] = city
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def hit_rate_at_horizons(
    hits: pd.DataFrame,
    horizons: list[float] | None = None,
    *,
    window: float = HORIZON_WINDOW_H,
) -> pd.DataFrame:
    """Compute hit rate in ±window hour bands around each target horizon."""
    if horizons is None:
        horizons = HORIZONS_H

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon == 0.0:
            mask = hits["hours_before_settlement"] <= window
        else:
            mask = hits["hours_before_settlement"].between(
                horizon - window,
                horizon + window,
                inclusive="both",
            )
        subset = hits.loc[mask]
        rows.append(
            {
                "hours_before_settlement": horizon,
                "hit_rate": float(subset["hit"].mean()) if not subset.empty else np.nan,
                "n_snapshots": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def _temp_in_bucket(
    temp: float,
    bucket_type: str,
    lower: float | None,
    upper: float | None,
) -> bool:
    t = int(round(temp))
    if bucket_type == "RANGE":
        return lower is not None and upper is not None and float(lower) <= t <= float(upper)
    if bucket_type == "LESS_THAN":
        return upper is not None and t <= float(upper)
    if bucket_type == "GREATER_THAN":
        return lower is not None and t >= float(lower)
    return False


def assign_bucket(temp: float, bucket_defs: pd.DataFrame) -> str | None:
    """Map a rounded Tmax forecast to the matching Kalshi bucket."""
    for _, row in bucket_defs.iterrows():
        if _temp_in_bucket(
            temp,
            str(row["bucket_type"]),
            pd.to_numeric(row.get("bucket_lower_inclusive_f"), errors="coerce"),
            pd.to_numeric(row.get("bucket_upper_inclusive_f"), errors="coerce"),
        ):
            return str(row["bucket_label"])
    return None


def load_bucket_defs(df: pd.DataFrame) -> dict[tuple[str, object], pd.DataFrame]:
    """Return bucket definitions keyed by (city, event_date)."""
    bucket_map: dict[tuple[str, object], pd.DataFrame] = {}
    for key, day_df in df.groupby(["source_city_folder", "event_date"], sort=True):
        bucket_map[key] = day_df[BUCKET_COLS].drop_duplicates("bucket_label").copy()
    return bucket_map


def build_trackb_hits(
    hits: pd.DataFrame,
    forecasts: pd.DataFrame,
    bucket_defs: dict[tuple[str, object], pd.DataFrame],
) -> pd.DataFrame:
    """Compute per-city-day Track-B bucket hits on the market snapshot universe."""
    day_universe = hits[["source_city_folder", "event_date", "actual_bucket"]].drop_duplicates()
    forecast_map = {
        (str(row.city), row.event_date): row
        for row in forecasts.itertuples(index=False)
    }

    records: list[dict[str, object]] = []
    for row in day_universe.itertuples(index=False):
        city = str(row.source_city_folder)
        event_date = row.event_date
        forecast_row = forecast_map.get((city, event_date))
        defs = bucket_defs.get((city, event_date))
        if forecast_row is None or defs is None:
            continue

        trackb_tmax_f = float(forecast_row.trackb_tmax_f)
        predicted_bucket = assign_bucket(trackb_tmax_f, defs)
        if predicted_bucket is None:
            continue

        records.append(
            {
                "source_city_folder": city,
                "event_date": event_date,
                "actual_bucket": str(row.actual_bucket),
                "predicted_bucket": predicted_bucket,
                "trackb_tmax_f": trackb_tmax_f,
                "hit": int(predicted_bucket == str(row.actual_bucket)),
            }
        )

    return pd.DataFrame(records)


def load_trackb_forecasts() -> pd.DataFrame:
    if not TRACKB_FORECASTS_PATH.exists():
        raise FileNotFoundError(f"Missing Track-B forecasts: {TRACKB_FORECASTS_PATH}")
    forecasts = pd.read_parquet(TRACKB_FORECASTS_PATH)
    forecasts["city"] = forecasts["city"].astype(str)
    forecasts["event_date"] = pd.to_datetime(forecasts["event_date"]).dt.date
    return forecasts


def trackb_hit_rate_by_city(trackb_hits: pd.DataFrame) -> pd.DataFrame:
    """Return per-city Track-B hit rates sorted descending."""
    return (
        trackb_hits.groupby("source_city_folder", sort=False)
        .agg(hit_rate=("hit", "mean"), n_days=("hit", "size"))
        .reset_index()
        .sort_values("hit_rate", ascending=False)
    )


def market_hit_at_10am(
    hits: pd.DataFrame,
    city_days: pd.DataFrame,
) -> tuple[float, int]:
    """Modal-bucket hit rate at the 10:00 AM snapshot for selected city-days."""
    keys = set(zip(city_days["source_city_folder"], city_days["event_date"]))
    ten_am = hits[
        hits.apply(
            lambda row: (row["source_city_folder"], row["event_date"]) in keys,
            axis=1,
        )
        & (pd.to_datetime(hits["snapshot_time_local"]).dt.hour == 10)
        & (pd.to_datetime(hits["snapshot_time_local"]).dt.minute == 0)
    ]
    if ten_am.empty:
        return float("nan"), 0
    return float(ten_am["hit"].mean()), int(len(ten_am))


def bootstrap_ci(values: np.ndarray, *, n_samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """Return 95% bootstrap CI for a binary hit-rate vector."""
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rates = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_rates[idx] = sample.mean()
    lower, upper = np.percentile(boot_rates, [2.5, 97.5])
    return float(lower), float(upper)


def compute_reference_hours(df: pd.DataFrame) -> dict[str, float]:
    """Median hours before day-end at 10:00 and 14:00 local across city-days."""
    day_cols = ["source_city_folder", "event_date"]
    day_ends = (
        df.groupby(day_cols, sort=True)["snapshot_time_local"]
        .max()
        .rename("day_end_time")
        .reset_index()
    )
    records: list[dict[str, float]] = []
    for row in day_ends.itertuples(index=False):
        day_end = pd.Timestamp(row.day_end_time)
        event_day = pd.Timestamp(row.event_date)
        for label, hour in (("entry_10am", 10), ("entry_2pm", 14)):
            clock_time = event_day + pd.Timedelta(hours=hour)
            hours_before = (day_end - clock_time).total_seconds() / 3600.0
            if hours_before >= 0:
                records.append({"label": label, "hours_before_settlement": hours_before})

    ref_df = pd.DataFrame(records)
    if ref_df.empty:
        return {"entry_10am": np.nan, "entry_2pm": np.nan}
    return {
        label: float(ref_df.loc[ref_df["label"] == label, "hours_before_settlement"].median())
        for label in ("entry_10am", "entry_2pm")
    }


def plot_hit_rate(
    agg: pd.DataFrame,
    by_city: pd.DataFrame,
    ref_lines: dict[str, float],
    trackb_summary: dict[str, float] | None,
    out_path: Path,
) -> None:
    """Plot aggregate and per-city hit-rate curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for city, city_df in by_city.groupby("source_city_folder", sort=True):
        city_plot = city_df.sort_values("bin_center", ascending=False)
        ax.plot(
            city_plot["bin_center"],
            city_plot["hit_rate"],
            color="#999999",
            alpha=0.35,
            linewidth=1.0,
        )

    agg_plot = agg.sort_values("bin_center", ascending=False)
    ax.plot(
        agg_plot["bin_center"],
        agg_plot["hit_rate"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Market modal bucket",
    )

    ax.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.2, label="50% hit rate")

    ref_styles = {
        "entry_10am": ("10:00 AM local", "#2ca02c"),
        "entry_2pm": ("2:00 PM local", "#ff7f0e"),
    }
    for key, (label, color) in ref_styles.items():
        value = ref_lines.get(key)
        if value is not None and np.isfinite(value):
            ax.axvline(value, color=color, linestyle=":", linewidth=1.2, label=label)

    if trackb_summary is not None:
        x_pos = trackb_summary.get("x_position", ref_lines.get("entry_10am"))
        hit_rate = trackb_summary.get("hit_rate")
        ci_lower = trackb_summary.get("ci_lower")
        ci_upper = trackb_summary.get("ci_upper")
        if x_pos is not None and hit_rate is not None and np.isfinite(hit_rate):
            yerr = None
            if ci_lower is not None and ci_upper is not None and np.isfinite(ci_lower) and np.isfinite(ci_upper):
                yerr = [[hit_rate - ci_lower], [ci_upper - hit_rate]]
            ax.errorbar(
                x_pos,
                hit_rate,
                yerr=yerr,
                fmt="D",
                color="#d62728",
                markersize=10,
                markerfacecolor="#d62728",
                markeredgecolor="white",
                markeredgewidth=1.0,
                elinewidth=2.0,
                capsize=6,
                zorder=5,
                label="Track-B model (10AM, CLI-calibrated)",
            )

    ax.set_xlabel("Hours before end of trading day")
    ax.set_ylabel("Hit rate (predicted bucket == resolved bucket)")
    ax.set_title("Market modal-bucket hit rate vs time before end of trading day")
    ax.set_ylim(0.0, 1.05)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    hits: pd.DataFrame,
    horizon_table: pd.DataFrame,
    ref_lines: dict[str, float],
    trackb_hits: pd.DataFrame | None = None,
    market_10am_hit_rate: float | None = None,
    market_10am_n: int = 0,
) -> None:
    """Print summary statistics to stdout."""
    n_days = hits[["source_city_folder", "event_date"]].drop_duplicates().shape[0]
    date_min = hits["event_date"].min()
    date_max = hits["event_date"].max()
    overall = float(hits["hit"].mean()) if not hits.empty else float("nan")

    print("\n=== Hit rate vs time summary ===")
    print(f"Snapshots: {len(hits):,}")
    print(f"City-days: {n_days:,}")
    print(f"Cities: {hits['source_city_folder'].nunique()}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Overall hit rate: {overall:.3f}")

    print("\nReference lines (median hours before day-end):")
    print(f"  10:00 AM local: {ref_lines.get('entry_10am', float('nan')):.2f} h")
    print(f"  2:00 PM local:  {ref_lines.get('entry_2pm', float('nan')):.2f} h")

    print("\nHours before day-end | Hit rate | N snapshots")
    print("-" * 48)
    for _, row in horizon_table.iterrows():
        hit_rate = row["hit_rate"]
        hit_text = f"{hit_rate:.3f}" if np.isfinite(hit_rate) else "n/a"
        print(
            f"{row['hours_before_settlement']:>23.1f} | {hit_text:>8} | "
            f"{int(row['n_snapshots']):>11,}"
        )

    if trackb_hits is not None and not trackb_hits.empty:
        trackb_rate = float(trackb_hits["hit"].mean())
        ci_lower, ci_upper = bootstrap_ci(trackb_hits["hit"].to_numpy())
        print("\n=== Track-B vs market at 10:00 AM ===")
        print("No Wunderground adjustment applied (Kalshi data settles on CLI)")
        print(
            f"Track-B hit rate: {trackb_rate * 100:.1f}% "
            f"({len(trackb_hits)} city-days)"
        )
        print(f"Track-B 95% CI: [{ci_lower * 100:.1f}%, {ci_upper * 100:.1f}%]")
        if market_10am_hit_rate is not None and np.isfinite(market_10am_hit_rate):
            print(
                f"Market modal hit rate at 10AM: {market_10am_hit_rate * 100:.1f}% "
                f"({market_10am_n} snapshots)"
            )
            edge = trackb_rate - market_10am_hit_rate
            if edge > 0:
                print(f"Track-B edge at 10AM: +{edge * 100:.1f} pp over market modal")
            else:
                print(f"Track-B vs market at 10AM: {edge * 100:.1f} pp (market leads)")
        else:
            print("Market modal hit rate at 10AM: n/a")

        print("\nPer-city Track-B hit rates:")
        print(f"{'City':<20} | {'Hit rate':>8} | {'N days':>6}")
        print("-" * 40)
        for row in trackb_hit_rate_by_city(trackb_hits).itertuples(index=False):
            print(f"{row.source_city_folder:<20} | {row.hit_rate * 100:7.1f}% | {row.n_days:6d}")


def main() -> None:
    load_city_timezones()  # validate config exists
    raw = load_snapshot_data()
    hits = build_snapshot_hits(raw)
    if hits.empty:
        raise RuntimeError("No snapshots remained after filtering; cannot compute hit rates.")

    agg = binned_hit_rates(hits)
    by_city = binned_hit_rates_by_city(hits)
    horizon_table = hit_rate_at_horizons(hits)
    ref_lines = compute_reference_hours(raw)

    forecasts = load_trackb_forecasts()
    bucket_defs = load_bucket_defs(raw)
    trackb_hits = build_trackb_hits(hits, forecasts, bucket_defs)

    trackb_summary = None
    market_10am_hit_rate = float("nan")
    market_10am_n = 0
    if not trackb_hits.empty:
        hit_rate = float(trackb_hits["hit"].mean())
        ci_lower, ci_upper = bootstrap_ci(trackb_hits["hit"].to_numpy())
        trackb_summary = {
            "x_position": ref_lines.get("entry_10am"),
            "hit_rate": hit_rate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        market_10am_hit_rate, market_10am_n = market_hit_at_10am(hits, trackb_hits)

    plot_hit_rate(agg, by_city, ref_lines, trackb_summary, OUTPUT_PATH)
    print_summary(
        hits,
        horizon_table,
        ref_lines,
        trackb_hits=trackb_hits if not trackb_hits.empty else None,
        market_10am_hit_rate=market_10am_hit_rate,
        market_10am_n=market_10am_n,
    )
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
