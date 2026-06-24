"""Realistic spread-adjusted backtest: Track-B hybrid vs modal and inflated baselines."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest_modal_maker import (  # noqa: E402
    compute_combo_metrics,
    daily_pnl_series_from_trades,
    load_partition,
)
from run_trackB_grid import (  # noqa: E402
    FORECASTS_PATH,
    _calendar_date_keys,
    _calendar_days,
    _city_key,
    _date_key,
    _forecast_lookup,
    _resolved_for_bucket,
)
from snapshot_stability import compute_modal_bucket  # noqa: E402
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402
from src.sizing import has_edge, taker_fee_cents  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
GRID_CSV = PROJECT_ROOT / "data" / "hybrid_realistic_grid.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

PLOT_EQUITY = REPORT_DIR / "hybrid_realistic_oos_equity.png"
PLOT_SPREAD = REPORT_DIR / "hybrid_realistic_spread_sensitivity.png"
PLOT_COMPARE = REPORT_DIR / "hybrid_realistic_comparison.png"

BUCKET_COLS = [
    "bucket_label",
    "bucket_type",
    "bucket_lower_inclusive_f",
    "bucket_upper_inclusive_f",
]

SPREADS = {
    "optimistic": 0.04,
    "moderate": 0.10,
    "pessimistic": 0.18,
}
EXIT_THRESHOLDS = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
MIN_ENTRIES = [0.15, 0.25, 0.35]
MAX_ENTRIES = [0.55, 0.65]
E_STARS = [0.02, 0.037, 0.05]

CONTRACTS = 5
DAILY_CAP_DOLLARS = 6.0
MIDPOINT_EXIT = 0.15

STRATEGY_LABELS = {
    "hybrid_maker": "Hybrid maker",
    "trackb_midpoint": "Track-B midpoint",
    "trackb_taker": "Track-B taker",
    "modal_maker": "Modal maker",
}


@dataclass
class MarketDay:
    city: str
    event_date: str
    signal_time: pd.Timestamp
    tmax_f: float
    sigma_f: float
    signal_mids: dict[str, float]
    bucket_defs: pd.DataFrame
    bucket_paths: dict[str, dict[str, list]]
    settlement: dict[str, bool]
    modal_bucket: str
    modal_mid: float


def _filter_trading_window(df: pd.DataFrame) -> pd.DataFrame:
    snap = df["snapshot_time_local"]
    event_dates = pd.to_datetime(df["event_date"])
    same_day = snap.dt.date == event_dates.dt.date
    after_open = (snap.dt.hour > 10) | ((snap.dt.hour == 10) & (snap.dt.minute >= 0))
    return df.loc[same_day & after_open].copy()


def _load_city_config() -> dict:
    with open(CITY_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _load_forecasts() -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(f"Missing forecasts: {FORECASTS_PATH}")
    forecasts = pd.read_parquet(FORECASTS_PATH)
    forecasts["city"] = forecasts["city"].map(_city_key)
    forecasts["event_date"] = pd.to_datetime(forecasts["event_date"]).dt.strftime("%Y-%m-%d")
    return forecasts


def build_market_days(partition_df: pd.DataFrame, forecasts: pd.DataFrame, city_config: dict) -> list[MarketDay]:
    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    days: list[MarketDay] = []
    skipped = 0

    for (city, event_date), day_df in df.groupby(["source_city_folder", "event_date"], sort=True):
        city_key = _city_key(city)
        date_key = _date_key(event_date)
        forecast = _forecast_lookup(city_key, date_key, forecasts)
        if forecast is None:
            skipped += 1
            continue
        tmax_f, sigma_f = forecast
        if pd.isna(sigma_f) or float(sigma_f) <= 0:
            sigma_f = float(city_config.get(city_key, {}).get("trackb_sigma_f", np.nan))
        if not np.isfinite(sigma_f) or sigma_f <= 0:
            skipped += 1
            continue

        trading = _filter_trading_window(day_df)
        trading = trading.dropna(subset=["yes_mid_close"])
        if trading.empty:
            skipped += 1
            continue

        signal_time = pd.Timestamp(trading["snapshot_time_local"].min())
        signal_rows = trading[trading["snapshot_time_local"] == signal_time]
        if signal_rows.empty:
            skipped += 1
            continue

        bucket_defs = signal_rows[BUCKET_COLS].drop_duplicates("bucket_label")
        signal_mids = {
            str(row["bucket_label"]): float(row["yes_mid_close"])
            for _, row in signal_rows.iterrows()
            if pd.notna(row["yes_mid_close"])
        }
        if not signal_mids:
            skipped += 1
            continue

        try:
            modal_bucket = compute_modal_bucket(trading, signal_time)
            modal_mid = float(signal_mids[str(modal_bucket)])
        except (ValueError, KeyError):
            skipped += 1
            continue

        settlement: dict[str, bool] = {}
        bucket_paths: dict[str, dict[str, list]] = {}
        for bucket_label in signal_mids:
            bucket_rows = day_df[day_df["bucket_label"].astype(str) == str(bucket_label)].copy()
            try:
                settlement[str(bucket_label)] = _resolved_for_bucket(day_df, str(bucket_label))
            except ValueError:
                skipped += 1
                settlement = {}
                break
            after = bucket_rows[pd.to_datetime(bucket_rows["snapshot_time_local"]) > signal_time]
            after = after.dropna(subset=["yes_mid_close"]).sort_values("snapshot_time_local")
            bucket_paths[str(bucket_label)] = {
                "times": [pd.Timestamp(t) for t in after["snapshot_time_local"]],
                "mids": after["yes_mid_close"].astype(float).tolist(),
            }
        if not settlement:
            continue

        days.append(
            MarketDay(
                city=city_key,
                event_date=date_key,
                signal_time=signal_time,
                tmax_f=float(tmax_f),
                sigma_f=float(sigma_f),
                signal_mids=signal_mids,
                bucket_defs=bucket_defs,
                bucket_paths=bucket_paths,
                settlement=settlement,
                modal_bucket=str(modal_bucket),
                modal_mid=modal_mid,
            )
        )

    if skipped:
        print(f"Skipped {skipped:,} city-days building market_days", flush=True)
    return days


def simulate_trade(
    strategy: str,
    spread: float,
    bucket_label: str,
    signal_mid: float,
    path_times: list[pd.Timestamp],
    path_mids: list[float],
    settlement_yes: bool,
    exit_threshold: float,
) -> dict[str, object] | None:
    if strategy in ("hybrid_maker", "modal_maker"):
        if not path_times:
            return None
        next_mid = float(path_mids[0])
        if next_mid > signal_mid + spread / 2.0:
            return None
        entry_price = signal_mid - spread / 2.0
        exit_target = entry_price + exit_threshold
        exit_fill_mid = entry_price + exit_threshold - spread / 2.0
        for mid in path_mids:
            if float(mid) >= exit_fill_mid:
                return {
                    "entry_price": entry_price,
                    "pnl": exit_threshold * CONTRACTS,
                    "exit_type": "intraday",
                }
        settle_per = (1.0 - entry_price) if settlement_yes else (-entry_price)
        return {
            "entry_price": entry_price,
            "pnl": settle_per * CONTRACTS,
            "exit_type": "settlement",
        }

    if strategy == "trackb_midpoint":
        entry_price = signal_mid
        exit_target = entry_price + MIDPOINT_EXIT
        for mid in path_mids:
            if float(mid) >= exit_target:
                return {
                    "entry_price": entry_price,
                    "pnl": MIDPOINT_EXIT * CONTRACTS,
                    "exit_type": "intraday",
                }
        settle_per = (1.0 - entry_price) if settlement_yes else (-entry_price)
        return {
            "entry_price": entry_price,
            "pnl": settle_per * CONTRACTS,
            "exit_type": "settlement",
        }

    if strategy == "trackb_taker":
        entry_price = signal_mid + spread / 2.0
        profit_target = MIDPOINT_EXIT
        for mid in path_mids:
            taker_exit_price = float(mid) - spread / 2.0
            if taker_exit_price >= entry_price + profit_target:
                gross = (taker_exit_price - entry_price) * CONTRACTS
                fee = taker_fee_cents(CONTRACTS, taker_exit_price) / 100.0
                return {
                    "entry_price": entry_price,
                    "pnl": gross - fee,
                    "exit_type": "intraday_taker",
                }
        settle_per = (1.0 - entry_price) if settlement_yes else (-entry_price)
        return {
            "entry_price": entry_price,
            "pnl": settle_per * CONTRACTS,
            "exit_type": "settlement",
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def select_trackb_bucket(
    day: MarketDay,
    min_entry: float,
    max_entry: float,
    e_star: float,
) -> tuple[str, float, float] | None:
    try:
        probs = bucket_probs_from_point_forecast(day.tmax_f, day.sigma_f, day.bucket_defs)
    except ValueError:
        return None

    best: tuple[str, float, float] | None = None
    for bucket_label, model_prob in probs.items():
        mid = day.signal_mids.get(str(bucket_label))
        if mid is None:
            continue
        if not (min_entry <= mid <= max_entry):
            continue
        fee = taker_fee_cents(1, mid) / 100.0
        if not has_edge(float(model_prob), mid, fee):
            continue
        edge = float(model_prob) - mid
        if edge <= e_star:
            continue
        if best is None or edge > best[2]:
            best = (str(bucket_label), mid, edge)
    return best


def select_modal_bucket(day: MarketDay, min_entry: float, max_entry: float) -> tuple[str, float] | None:
    mid = day.modal_mid
    if not (min_entry <= mid <= max_entry):
        return None
    return day.modal_bucket, mid


def run_strategy(
    market_days: list[MarketDay],
    calendar_dates: list[str],
    strategy: str,
    spread: float,
    min_entry: float,
    max_entry: float,
    exit_threshold: float,
    e_star: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    by_date: dict[str, list[dict[str, object]]] = {}

    for day in market_days:
        if strategy == "modal_maker":
            picked = select_modal_bucket(day, min_entry, max_entry)
            if picked is None:
                continue
            bucket_label, signal_mid = picked
            edge = np.nan
        else:
            picked = select_trackb_bucket(day, min_entry, max_entry, e_star)
            if picked is None:
                continue
            bucket_label, signal_mid, edge = picked

        path = day.bucket_paths.get(bucket_label, {"times": [], "mids": []})
        result = simulate_trade(
            strategy=strategy,
            spread=spread,
            bucket_label=bucket_label,
            signal_mid=signal_mid,
            path_times=path["times"],
            path_mids=path["mids"],
            settlement_yes=day.settlement[bucket_label],
            exit_threshold=exit_threshold,
        )
        if result is None:
            continue
        by_date.setdefault(day.event_date, []).append(
            {
                "event_date": day.event_date,
                "city": day.city,
                "bucket_label": bucket_label,
                "strategy": strategy,
                "spread": spread,
                "edge": edge,
                **result,
            }
        )

    trade_records: list[dict[str, object]] = []
    daily_pnl = np.zeros(len(calendar_dates), dtype=float)

    for idx, event_date in enumerate(calendar_dates):
        candidates = by_date.get(event_date, [])
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda r: float(r["entry_price"]))
        cum_risk = 0.0
        day_pnl = 0.0
        for row in candidates:
            risk = float(row["entry_price"]) * CONTRACTS
            if cum_risk + risk > DAILY_CAP_DOLLARS:
                break
            cum_risk += risk
            day_pnl += float(row["pnl"])
            trade_records.append(row)
        daily_pnl[idx] = day_pnl

    return pd.DataFrame.from_records(trade_records), daily_pnl


def compute_strategy_metrics(
    trades_df: pd.DataFrame,
    daily_pnl: np.ndarray,
    calendar_days: int,
) -> dict[str, float]:
    metrics = compute_combo_metrics(trades_df, daily_pnl, calendar_days)
    if len(trades_df) and "exit_type" in trades_df.columns:
        metrics["exit_rate"] = float(
            trades_df["exit_type"].isin(["intraday", "intraday_taker"]).mean()
        )
    return metrics


def run_hybrid_grid(
    market_days: list[MarketDay],
    calendar_dates: list[str],
    calendar_days: int,
    spread_label: str,
    spread: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    combos = list(product(EXIT_THRESHOLDS, MIN_ENTRIES, MAX_ENTRIES, E_STARS))
    for exit_threshold, min_entry, max_entry, e_star in combos:
        if min_entry > max_entry:
            continue
        trades, daily_pnl = run_strategy(
            market_days,
            calendar_dates,
            "hybrid_maker",
            spread,
            min_entry,
            max_entry,
            exit_threshold,
            e_star,
        )
        metrics = compute_strategy_metrics(trades, daily_pnl, calendar_days)
        rows.append(
            {
                "partition": "IS",
                "strategy": "hybrid_maker",
                "spread_label": spread_label,
                "spread": spread,
                "exit_threshold": exit_threshold,
                "min_entry": min_entry,
                "max_entry": max_entry,
                "e_star": e_star,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def best_params_for_spread(grid_df: pd.DataFrame, spread_label: str) -> pd.Series:
    subset = grid_df[grid_df["spread_label"] == spread_label]
    if subset.empty:
        raise ValueError(f"No grid rows for spread {spread_label}")
    return subset.sort_values("sharpe", ascending=False).iloc[0]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%" if np.isfinite(value) else "n/a"


def print_table1(rows: list[dict[str, object]]) -> None:
    print("\n=== Table 1: OOS comparison across spread assumptions ===")
    print(
        f"{'Spread':<6} | {'Strategy':<18} | {'OOS Sharpe':>10} | {'OOS N':>5} | "
        f"{'OOS PnL':>8} | {'Win%':>6} | {'Exit%':>6} | {'Max DD':>8}"
    )
    print("-" * 95)
    for row in rows:
        print(
            f"{row['spread_label']:<6} | {row['strategy_label']:<18} | "
            f"{row['sharpe']:>10.2f} | {int(row['n_trades']):>5} | "
            f"{row['total_pnl']:>8.2f} | {_pct(row['win_rate']):>6} | "
            f"{_pct(row['exit_rate']):>6} | {row['max_dd']:>8.2f}"
        )


def print_table2(rows: list[dict[str, object]]) -> None:
    print("\n=== Table 2: IS best parameters per spread assumption ===")
    print(
        f"{'Spread':<12} | {'exit_thr':>8} | {'min_ent':>7} | {'max_ent':>7} | "
        f"{'E*':>6} | {'IS Sharpe':>9}"
    )
    print("-" * 65)
    for row in rows:
        print(
            f"{row['spread_label']:<12} | {row['exit_threshold']:>8.2f} | "
            f"{row['min_entry']:>7.2f} | {row['max_entry']:>7.2f} | "
            f"{row['e_star']:>6.3f} | {row['sharpe']:>9.2f}"
        )


def print_key_questions(
    table1_rows: list[dict[str, object]],
    midpoint_moderate: dict[str, float],
    hybrid_moderate: dict[str, float],
) -> None:
    print("\n=== Key questions ===")
    print(
        f"1. Midpoint Track-B OOS Sharpe at 4c spread: "
        f"{next((r['sharpe'] for r in table1_rows if r['spread_label']=='optimistic' and r['strategy']=='trackb_midpoint'), float('nan')):.2f} "
        f"(prior inflated ~10.47 used calendar-dollar Sharpe on shorter logic)"
    )
    print(
        f"2. Hybrid maker vs midpoint at 10c: "
        f"{hybrid_moderate.get('sharpe', float('nan')):.2f} vs {midpoint_moderate.get('sharpe', float('nan')):.2f}"
    )
    modal_10 = next(
        (r for r in table1_rows if r["spread_label"] == "moderate" and r["strategy"] == "modal_maker"),
        {},
    )
    print(
        f"3. Track-B hybrid vs modal at 10c Sharpe: "
        f"{hybrid_moderate.get('sharpe', float('nan')):.2f} vs {modal_10.get('sharpe', float('nan')):.2f}"
    )
    pess = next(
        (r for r in table1_rows if r["spread_label"] == "pessimistic" and r["strategy"] == "hybrid_maker"),
        {},
    )
    print(f"4. Realistic Polymarket-like (18c) hybrid Sharpe: {pess.get('sharpe', float('nan')):.2f}")
    print(
        f"5. Hybrid profitable after costs at 10c? "
        f"PnL=${hybrid_moderate.get('total_pnl', float('nan')):.2f}, "
        f"Sharpe={hybrid_moderate.get('sharpe', float('nan')):.2f}"
    )


def plot_oos_equity(curves: list[tuple[str, pd.Series]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4878CF", "#E68A2E", "#d62728", "#2ca02c"]
    for idx, (label, series) in enumerate(curves):
        cum = series.cumsum()
        ax.plot(cum.index, cum.values, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("OOS equity curves at 10c spread")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_spread_sensitivity(curves: list[tuple[str, pd.Series]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c", "#4878CF", "#d62728"]
    for idx, (label, series) in enumerate(curves):
        cum = series.cumsum()
        ax.plot(cum.index, cum.values, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Hybrid maker OOS spread sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(table1_rows: list[dict[str, object]], out_path: Path) -> None:
    frame = pd.DataFrame(table1_rows)
    strategies = ["hybrid_maker", "trackb_midpoint", "trackb_taker", "modal_maker"]
    spread_labels = list(SPREADS.keys())
    x = np.arange(len(spread_labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, strategy in enumerate(strategies):
        vals = [
            float(frame[(frame["spread_label"] == s) & (frame["strategy"] == strategy)]["sharpe"].iloc[0])
            if not frame[(frame["spread_label"] == s) & (frame["strategy"] == strategy)].empty
            else 0.0
            for s in spread_labels
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=STRATEGY_LABELS[strategy])
    ax.set_xticks(x)
    ax.set_xticklabels(spread_labels)
    ax.set_ylabel("OOS Sharpe")
    ax.set_title("Strategy comparison by spread assumption")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_caveats() -> None:
    print("\n=== Caveats ===")
    print("- Kalshi data has yes_mid_close only; bid/ask simulated via fixed spread assumptions.")
    print("- Maker entry fill modeled on next 5-min snapshot if price has not run away.")
    print("- Track-B forecasts are CLI-calibrated (not Wunderground-adjusted).")
    print("- Modal/hybrid maker assumes zero fees; Track-B taker includes Kalshi taker fees.")
    print("- Zero-PnL calendar days included in Sharpe (daily dollar PnL, sqrt(252) annualized).")


def main() -> None:
    city_config = _load_city_config()
    forecasts = _load_forecasts()
    print(
        "Forecasts: CLI-calibrated Track-B from "
        f"{FORECASTS_PATH} (trackb_tmax_f + trackb_sigma_f; NOT WU-adjusted)",
        flush=True,
    )
    print(
        f"Forecast date range: {forecasts['event_date'].min()} to {forecasts['event_date'].max()}",
        flush=True,
    )

    is_raw = load_partition("threshold_opt")
    oos_raw = load_partition("time_holdout")

    is_days = build_market_days(is_raw, forecasts, city_config)
    oos_days = build_market_days(oos_raw, forecasts, city_config)
    print(f"Market days: IS={len(is_days):,}, OOS={len(oos_days):,}", flush=True)

    is_dates = _calendar_date_keys(is_raw)
    oos_dates = _calendar_date_keys(oos_raw)
    is_calendar_days = _calendar_days(is_raw)
    oos_calendar_days = _calendar_days(oos_raw)

    grid_frames: list[pd.DataFrame] = []
    for spread_label, spread in SPREADS.items():
        print(f"Running IS hybrid grid for {spread_label} ({spread:.2f})...", flush=True)
        grid_frames.append(run_hybrid_grid(is_days, is_dates, is_calendar_days, spread_label, spread))
    grid_df = pd.concat(grid_frames, ignore_index=True)
    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(GRID_CSV, index=False)
    print(f"Saved {len(grid_df)} grid rows to {GRID_CSV}", flush=True)

    table2_rows: list[dict[str, object]] = []
    table1_rows: list[dict[str, object]] = []
    equity_10c: list[tuple[str, pd.Series]] = []
    hybrid_spread_curves: list[tuple[str, pd.Series]] = []

    for spread_label, spread in SPREADS.items():
        best = best_params_for_spread(grid_df, spread_label)
        table2_rows.append(
            {
                "spread_label": spread_label,
                "exit_threshold": float(best["exit_threshold"]),
                "min_entry": float(best["min_entry"]),
                "max_entry": float(best["max_entry"]),
                "e_star": float(best["e_star"]),
                "sharpe": float(best["sharpe"]),
            }
        )

        params = {
            "min_entry": float(best["min_entry"]),
            "max_entry": float(best["max_entry"]),
            "exit_threshold": float(best["exit_threshold"]),
            "e_star": float(best["e_star"]),
        }

        for strategy in ("hybrid_maker", "trackb_midpoint", "trackb_taker", "modal_maker"):
            trades, daily_pnl = run_strategy(
                oos_days,
                oos_dates,
                strategy,
                spread,
                params["min_entry"],
                params["max_entry"],
                params["exit_threshold"],
                params["e_star"],
            )
            metrics = compute_strategy_metrics(trades, daily_pnl, oos_calendar_days)
            table1_rows.append(
                {
                    "spread_label": spread_label,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    **metrics,
                }
            )
            if spread_label == "moderate":
                equity_10c.append((STRATEGY_LABELS[strategy], daily_pnl_series_from_trades(trades, oos_dates)))
            if strategy == "hybrid_maker":
                hybrid_spread_curves.append(
                    (f"Hybrid {spread_label} ({spread:.0%})", daily_pnl_series_from_trades(trades, oos_dates))
                )

    print_table2(table2_rows)
    print_table1(table1_rows)

    midpoint_moderate = next(
        (r for r in table1_rows if r["spread_label"] == "moderate" and r["strategy"] == "trackb_midpoint"),
        {},
    )
    hybrid_moderate = next(
        (r for r in table1_rows if r["spread_label"] == "moderate" and r["strategy"] == "hybrid_maker"),
        {},
    )
    print_key_questions(table1_rows, midpoint_moderate, hybrid_moderate)
    print_caveats()

    plot_oos_equity(equity_10c, PLOT_EQUITY)
    plot_spread_sensitivity(hybrid_spread_curves, PLOT_SPREAD)
    plot_comparison(table1_rows, PLOT_COMPARE)

    print("\nSaved plots:")
    for path in (PLOT_EQUITY, PLOT_SPREAD, PLOT_COMPARE):
        print(f"  {path}")


if __name__ == "__main__":
    main()
