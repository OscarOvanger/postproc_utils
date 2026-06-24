"""Grid-search and backtest a maker-maker modal bucket strategy on Kalshi snapshots."""

from __future__ import annotations

import json
import sys
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

from run_trackB_grid import (  # noqa: E402
    FORECASTS_PATH,
    LOW_OOS_COVERAGE_CITIES,
    _calendar_date_keys,
    _calendar_days,
    apply_selection,
    generate_signals,
    run_backtest,
)
from snapshot_stability import assert_no_true_holdout, compute_modal_bucket  # noqa: E402
from src.data_store import TRAIN_CITIES  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "deploy_config.json"
GRID_CSV = PROJECT_ROOT / "data" / "modal_maker_grid_results.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

PLOT_HEATMAP = REPORT_DIR / "modal_maker_sharpe_heatmap.png"
PLOT_EQUITY = REPORT_DIR / "modal_maker_oos_equity.png"
PLOT_SHARPE_TRADES = REPORT_DIR / "modal_maker_sharpe_vs_trades.png"

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

MIN_ENTRIES = [0.15, 0.20, 0.25, 0.30, 0.35]
MAX_ENTRIES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
EXIT_THRESHOLDS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

CONTRACTS = 5
DAILY_CAP_DOLLARS = 6.0
MIN_TRADES_60D = 80
MIN_TRADES_60D_RELAXED = 60
MAX_DD_DOLLARS = 30.0


def _to_naive_local(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_localize(None)
    return parsed


def load_partition(partition: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{partition}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split parquet: {path}")
    df = pd.read_parquet(path, columns=COLUMNS)
    df["partition"] = partition
    assert_no_true_holdout(df)
    df["source_city_folder"] = df["source_city_folder"].astype(str)
    df = df[df["source_city_folder"].isin(TRAIN_CITIES)].copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["snapshot_time_local"] = _to_naive_local(df["snapshot_time_local"])
    df["yes_mid_close"] = pd.to_numeric(df["yes_mid_close"], errors="coerce")
    print(f"loaded {partition}: {len(df):,} rows", flush=True)
    return df


def _filter_trading_window(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_city_day_row(day_df: pd.DataFrame) -> dict[str, object] | None:
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

    path = bucket_day[pd.to_datetime(bucket_day["snapshot_time_local"]) > entry_time].copy()
    path = path.dropna(subset=["yes_mid_close"]).sort_values("snapshot_time_local")

    return {
        "source_city_folder": str(day_df["source_city_folder"].iloc[0]),
        "event_date": pd.Timestamp(day_df["event_date"].iloc[0]).strftime("%Y-%m-%d"),
        "entry_time": entry_time,
        "modal_bucket": modal_bucket,
        "entry_price": entry_price,
        "settlement_outcome": settlement_outcome,
        "post_entry_prices": path["yes_mid_close"].astype(float).tolist(),
        "post_entry_times": [pd.Timestamp(t) for t in path["snapshot_time_local"]],
    }


def build_city_day_candidates(partition_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    skipped = 0
    for _, day_df in partition_df.groupby(["source_city_folder", "event_date"], sort=True):
        try:
            row = _build_city_day_row(day_df)
        except ValueError:
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        records.append(row)
    if skipped:
        print(f"Skipped {skipped:,} city-days building candidates", flush=True)
    return pd.DataFrame.from_records(records)


def _trade_pnl(
    entry_price: float,
    settlement_outcome: bool,
    post_entry_prices: list[float],
    exit_threshold: float,
) -> tuple[float, str]:
    target = entry_price + exit_threshold
    if post_entry_prices and max(post_entry_prices) >= target:
        return exit_threshold * CONTRACTS, "intraday"
    per_contract = (1.0 - entry_price) if settlement_outcome else (-entry_price)
    return per_contract * CONTRACTS, "settlement"


def run_combo(
    candidates: pd.DataFrame,
    calendar_dates: list[str],
    min_entry: float,
    max_entry: float,
    exit_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if min_entry > max_entry:
        return pd.DataFrame(), np.zeros(len(calendar_dates), dtype=float)

    by_date: dict[str, list[dict[str, object]]] = {}
    for _, row in candidates.iterrows():
        entry_price = float(row["entry_price"])
        if not (min_entry <= entry_price <= max_entry):
            continue
        event_date = str(row["event_date"])
        by_date.setdefault(event_date, []).append(row.to_dict())

    trade_records: list[dict[str, object]] = []
    daily_pnl = np.zeros(len(calendar_dates), dtype=float)

    for idx, event_date in enumerate(calendar_dates):
        day_rows = by_date.get(event_date, [])
        if not day_rows:
            continue
        day_rows = sorted(day_rows, key=lambda r: float(r["entry_price"]))
        cum_risk = 0.0
        day_pnl = 0.0

        for row in day_rows:
            entry_price = float(row["entry_price"])
            capital_at_risk = entry_price * CONTRACTS
            if cum_risk + capital_at_risk > DAILY_CAP_DOLLARS:
                break
            cum_risk += capital_at_risk
            pnl, exit_type = _trade_pnl(
                entry_price,
                bool(row["settlement_outcome"]),
                list(row["post_entry_prices"]),
                exit_threshold,
            )
            day_pnl += pnl
            trade_records.append(
                {
                    "event_date": event_date,
                    "city": row["source_city_folder"],
                    "modal_bucket": row["modal_bucket"],
                    "entry_price": entry_price,
                    "exit_threshold": exit_threshold,
                    "exit_type": exit_type,
                    "pnl": pnl,
                    "settlement_outcome": bool(row["settlement_outcome"]),
                }
            )

        daily_pnl[idx] = day_pnl

    return pd.DataFrame.from_records(trade_records), daily_pnl


def compute_combo_metrics(
    trades_df: pd.DataFrame,
    daily_pnl: np.ndarray,
    calendar_days: int,
) -> dict[str, float]:
    n_trades = len(trades_df)
    total_pnl = float(daily_pnl.sum())
    proj_60d = (n_trades / calendar_days * 60.0) if calendar_days > 0 else 0.0

    if len(daily_pnl) > 1 and np.std(daily_pnl, ddof=1) > 0:
        sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl, ddof=1) * np.sqrt(252))
    elif len(daily_pnl) > 0 and np.std(daily_pnl, ddof=0) > 0:
        sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl, ddof=0) * np.sqrt(252))
    else:
        sharpe = 0.0

    cum = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    max_dd = float((cum - peak).min()) if len(cum) else 0.0

    win_rate = float((trades_df["pnl"] > 0).mean()) if n_trades else float("nan")
    if n_trades and "exit_type" in trades_df.columns:
        exit_rate = float((trades_df["exit_type"] == "intraday").mean())
    else:
        exit_rate = float("nan")
    mean_pnl = float(trades_df["pnl"].mean()) if n_trades else float("nan")

    return {
        "n_trades": n_trades,
        "proj_60d": proj_60d,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "exit_rate": exit_rate,
        "max_dd": max_dd,
        "mean_pnl_per_trade": mean_pnl,
    }


def run_grid(
    candidates: pd.DataFrame,
    calendar_dates: list[str],
    calendar_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    combos = [
        (min_e, max_e, exit_t)
        for min_e, max_e, exit_t in product(MIN_ENTRIES, MAX_ENTRIES, EXIT_THRESHOLDS)
        if min_e <= max_e
    ]
    print(f"Running {len(combos)} IS parameter combinations...", flush=True)
    for min_entry, max_entry, exit_threshold in combos:
        trades, daily_pnl = run_combo(
            candidates, calendar_dates, min_entry, max_entry, exit_threshold
        )
        metrics = compute_combo_metrics(trades, daily_pnl, calendar_days)
        rows.append(
            {
                "min_entry": min_entry,
                "max_entry": max_entry,
                "exit_threshold": exit_threshold,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def select_best_combo(grid_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    feasible = grid_df[
        (grid_df["proj_60d"] >= MIN_TRADES_60D) & (grid_df["max_dd"] > -MAX_DD_DOLLARS)
    ].copy()
    relaxed = False
    if feasible.empty:
        print(
            f"WARNING: No combo meets proj_60d>={MIN_TRADES_60D} and max_dd>${-MAX_DD_DOLLARS:.0f}; "
            f"relaxing to proj_60d>={MIN_TRADES_60D_RELAXED}",
            flush=True,
        )
        feasible = grid_df[
            (grid_df["proj_60d"] >= MIN_TRADES_60D_RELAXED) & (grid_df["max_dd"] > -MAX_DD_DOLLARS)
        ].copy()
        relaxed = True
    if feasible.empty:
        print("WARNING: No combo meets max drawdown constraint; using full grid.", flush=True)
        feasible = grid_df.copy()
        relaxed = True
    return feasible.sort_values("sharpe", ascending=False), relaxed


def combo_label(row: pd.Series) -> str:
    return (
        f"min={row['min_entry']:.2f}, max={row['max_entry']:.2f}, "
        f"exit={row['exit_threshold']:.2f}"
    )


def daily_pnl_series_from_trades(
    trades_df: pd.DataFrame, calendar_dates: list[str]
) -> pd.Series:
    if trades_df.empty:
        return pd.Series(0.0, index=pd.to_datetime(calendar_dates))
    frame = trades_df.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    daily = frame.groupby("event_date", sort=True)["pnl"].sum()
    idx = pd.to_datetime(calendar_dates)
    return daily.reindex(idx, fill_value=0.0)


def run_trackb_oos(time_holdout: pd.DataFrame, oos_dates: list[str], calendar_days: int) -> dict:
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(f"Missing Track-B forecasts: {FORECASTS_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        deploy = json.load(handle)
    edge_threshold = float(deploy["edge_threshold"])

    forecasts = pd.read_parquet(FORECASTS_PATH)
    oos_signals = generate_signals(
        time_holdout,
        forecasts,
        "track_b_flat",
        exclude_cities=LOW_OOS_COVERAGE_CITIES,
    )
    oos_sel = apply_selection(oos_signals, "edge_threshold", edge_threshold)
    trades, _, daily_pnl_cents, _ = run_backtest(oos_sel, "flat_5", oos_dates)
    daily_pnl = daily_pnl_cents / 100.0

    metrics = compute_combo_metrics(
        trades.assign(pnl=trades["net_pnl_cents"] / 100.0) if not trades.empty else trades,
        daily_pnl,
        calendar_days,
    )
    metrics["trades_df"] = trades
    metrics["daily_pnl"] = daily_pnl
    metrics["edge_threshold"] = edge_threshold
    return metrics


def plot_sharpe_heatmap(grid_df: pd.DataFrame, best_max_entry: float, out_path: Path) -> None:
    subset = grid_df[grid_df["max_entry"].eq(best_max_entry)].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(
        index="min_entry", columns="exit_threshold", values="sharpe", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("exit_threshold")
    ax.set_ylabel("min_entry")
    ax.set_title(f"IS Sharpe (max_entry={best_max_entry:.2f})")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_oos_equity(
    equity_curves: list[tuple[str, pd.Series]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4878CF", "#E68A2E", "#8A8A8A", "#d62728"]
    for idx, (label, series) in enumerate(equity_curves):
        if series.empty:
            continue
        cum = series.cumsum()
        ax.plot(cum.index, cum.values, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("OOS cumulative PnL")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sharpe_vs_trades(grid_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        grid_df["proj_60d"],
        grid_df["sharpe"],
        c=grid_df["exit_threshold"],
        cmap="viridis",
        alpha=0.65,
        s=28,
    )
    ax.axvline(MIN_TRADES_60D, color="#d62728", linestyle="--", linewidth=1.2, label="80 trades / 60d")
    ax.set_xlabel("Projected trades per 60 days")
    ax.set_ylabel("IS Sharpe (annualized, daily $ PnL)")
    ax.set_title("IS Sharpe vs projected trade count")
    fig.colorbar(scatter, ax=ax, label="exit_threshold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_caveats() -> None:
    print("\n=== Caveats ===")
    print("- Entry/exit at yes_mid_close; live maker fills differ from midpoint.")
    print("- Exit feasibility uses midpoint, not best_bid.")
    print("- Modal-maker assumes zero fees; Track-B includes Kalshi taker fees.")
    print("- Kalshi data settles on CLI (not Wunderground).")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%" if np.isfinite(value) else "n/a"


def main() -> None:
    is_raw = load_partition("threshold_opt")
    oos_raw = load_partition("time_holdout")

    is_candidates = build_city_day_candidates(is_raw)
    oos_candidates = build_city_day_candidates(oos_raw)
    print(
        f"Candidates: IS={len(is_candidates):,} city-days, OOS={len(oos_candidates):,} city-days",
        flush=True,
    )

    is_dates = _calendar_date_keys(is_raw)
    oos_dates = _calendar_date_keys(oos_raw)
    is_calendar_days = _calendar_days(is_raw)
    oos_calendar_days = _calendar_days(oos_raw)

    grid_df = run_grid(is_candidates, is_dates, is_calendar_days)
    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(GRID_CSV, index=False)
    print(f"Saved grid results to {GRID_CSV}", flush=True)

    feasible, relaxed = select_best_combo(grid_df)
    top10 = feasible.head(10)
    top3 = feasible.head(3)

    print("\n=== Table 1: Top 10 IS combinations by Sharpe ===")
    print(
        f"{'Rank':<5} | {'min':>5} | {'max':>5} | {'exit':>5} | {'N':>5} | "
        f"{'Proj/60d':>8} | {'Sharpe':>7} | {'Win%':>6} | {'Exit%':>6} | "
        f"{'PnL':>8} | {'Max DD':>8}"
    )
    print("-" * 95)
    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        print(
            f"{rank:<5} | {row['min_entry']:>5.2f} | {row['max_entry']:>5.2f} | "
            f"{row['exit_threshold']:>5.2f} | {int(row['n_trades']):>5} | "
            f"{row['proj_60d']:>8.1f} | {row['sharpe']:>7.2f} | "
            f"{_pct(row['win_rate']):>6} | {_pct(row['exit_rate']):>6} | "
            f"{row['total_pnl']:>8.2f} | {row['max_dd']:>8.2f}"
        )
    if relaxed:
        print("(Selection used relaxed trade-count constraint.)")

    print("\n=== Table 2: OOS evaluation of top 3 IS combinations ===")
    print(
        f"{'Combo':<40} | {'IS Sharpe':>9} | {'OOS Sharpe':>10} | {'OOS N':>5} | "
        f"{'OOS PnL':>8} | {'OOS Win%':>8} | {'OOS Exit%':>9} | {'OOS Max DD':>10}"
    )
    print("-" * 120)

    equity_curves: list[tuple[str, pd.Series]] = []
    for _, row in top3.iterrows():
        oos_trades, oos_daily = run_combo(
            oos_candidates,
            oos_dates,
            float(row["min_entry"]),
            float(row["max_entry"]),
            float(row["exit_threshold"]),
        )
        oos_metrics = compute_combo_metrics(oos_trades, oos_daily, oos_calendar_days)
        label = combo_label(row)
        print(
            f"{label:<40} | {row['sharpe']:>9.2f} | {oos_metrics['sharpe']:>10.2f} | "
            f"{int(oos_metrics['n_trades']):>5} | {oos_metrics['total_pnl']:>8.2f} | "
            f"{_pct(oos_metrics['win_rate']):>8} | {_pct(oos_metrics['exit_rate']):>9} | "
            f"{oos_metrics['max_dd']:>10.2f}"
        )
        equity_curves.append((f"Modal: {label}", daily_pnl_series_from_trades(oos_trades, oos_dates)))

    trackb = run_trackb_oos(oos_raw, oos_dates, oos_calendar_days)
    trackb_trades = trackb.pop("trades_df")
    trackb_daily = trackb.pop("daily_pnl")
    trackb_label = f"Track-B (E*={trackb['edge_threshold']:.3f})"

    print("\n=== Table 3: Best modal-maker vs Track-B (OOS) ===")
    print(
        f"{'Strategy':<45} | {'OOS Sharpe':>10} | {'OOS N':>5} | "
        f"{'OOS PnL':>8} | {'OOS Win%':>8} | {'OOS Max DD':>10}"
    )
    print("-" * 100)

    best_modal = top3.iloc[0] if not top3.empty else None
    if best_modal is not None:
        best_oos_trades, best_oos_daily = run_combo(
            oos_candidates,
            oos_dates,
            float(best_modal["min_entry"]),
            float(best_modal["max_entry"]),
            float(best_modal["exit_threshold"]),
        )
        best_oos = compute_combo_metrics(best_oos_trades, best_oos_daily, oos_calendar_days)
        print(
            f"{('Modal-maker: ' + combo_label(best_modal)):<45} | "
            f"{best_oos['sharpe']:>10.2f} | {int(best_oos['n_trades']):>5} | "
            f"{best_oos['total_pnl']:>8.2f} | {_pct(best_oos['win_rate']):>8} | "
            f"{best_oos['max_dd']:>10.2f}"
        )
    print(
        f"{trackb_label:<45} | {trackb['sharpe']:>10.2f} | {int(trackb['n_trades']):>5} | "
        f"{trackb['total_pnl']:>8.2f} | {_pct(trackb['win_rate']):>8} | "
        f"{trackb['max_dd']:>10.2f}"
    )

    equity_curves.append(
        (
            trackb_label,
            pd.Series(trackb_daily, index=pd.to_datetime(oos_dates)),
        )
    )

    best_max_entry = float(top3.iloc[0]["max_entry"]) if not top3.empty else float(MAX_ENTRIES[0])
    plot_sharpe_heatmap(grid_df, best_max_entry, PLOT_HEATMAP)
    plot_oos_equity(equity_curves, PLOT_EQUITY)
    plot_sharpe_vs_trades(grid_df, PLOT_SHARPE_TRADES)

    print_caveats()
    print(f"\nSaved plots:")
    for path in (PLOT_HEATMAP, PLOT_EQUITY, PLOT_SHARPE_TRADES):
        print(f"  {path}")


if __name__ == "__main__":
    main()
