"""Run fresh validation backtest for the leading Track-B combination."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.run_trackB_grid import (  # noqa: E402
    LOW_OOS_COVERAGE_CITIES,
    _calendar_date_keys,
    _calendar_days,
    apply_selection,
    compute_stats,
    generate_signals,
    run_backtest,
)
from src.snapshot_stability import assert_no_true_holdout  # noqa: E402

FRESH_DIR = PROJECT_ROOT / "data" / "fresh_validation"
MARKET_PATH = FRESH_DIR / "market_fresh.parquet"
FORECASTS_PATH = FRESH_DIR / "forecasts_fresh.parquet"
GRID_META_PATH = PROJECT_ROOT / "data" / "trackb" / "sizing_grid" / "grid_meta.json"
OOS_FORECASTS_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_PATH = FRESH_DIR / "fresh_results_expanded.json"

TOP_SIGNAL = "track_b_flat"
TOP_SIZER = "flat_5"
TOP_SELECTION = "edge_threshold"


def _parse_ci(ci_str: str) -> tuple[float, float]:
    text = str(ci_str).strip("[]")
    lo, hi = text.split(",")
    return float(lo.strip()), float(hi.strip())


def _split_max_date() -> pd.Timestamp | None:
    oos_path = SPLIT_DIR / "time_holdout.parquet"
    if not oos_path.exists():
        return None
    return pd.to_datetime(pd.read_parquet(oos_path)["event_date"]).max()


def _sharpe_from_series(
    daily_returns: np.ndarray,
    daily_pnl: np.ndarray,
) -> tuple[float, str, float]:
    bankroll = 10_000.0 + np.cumsum(daily_pnl)
    bankroll_path = np.concatenate([[10_000.0], bankroll])
    stats = compute_stats(
        daily_returns,
        daily_pnl,
        bankroll_path,
        n_trades=0,
        calendar_days=len(daily_returns),
        mean_edge=float("nan"),
    )
    return float(stats["sharpe"]), str(stats["ci"]), float(stats["max_dd_cents"])


def _run_partition_backtest(
    market_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    edge_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object], list[str], np.ndarray, np.ndarray]:
    calendar_dates = _calendar_date_keys(market_df)
    calendar_day_count = _calendar_days(market_df)
    signals = generate_signals(
        market_df,
        forecasts_df,
        TOP_SIGNAL,
        exclude_cities=LOW_OOS_COVERAGE_CITIES,
    )
    selected = apply_selection(signals.copy(), TOP_SELECTION, edge_threshold)
    trades, daily_returns, daily_pnl, bankroll_path = run_backtest(
        selected, TOP_SIZER, calendar_dates
    )
    mean_edge = float(trades["edge"].mean()) if not trades.empty else float("nan")
    stats = compute_stats(
        daily_returns,
        daily_pnl,
        bankroll_path,
        n_trades=len(trades),
        calendar_days=calendar_day_count,
        mean_edge=mean_edge,
    )
    return trades, stats, calendar_dates, daily_returns, daily_pnl


def main() -> None:
    if not MARKET_PATH.exists() or not FORECASTS_PATH.exists():
        print("Missing market_fresh.parquet or forecasts_fresh.parquet.")
        print("Run fetch_fresh_market.py and generate_fresh_forecasts.py first.")
        return

    with open(GRID_META_PATH, encoding="utf-8") as handle:
        grid_meta = json.load(handle)
    edge_threshold = float(grid_meta["e_star"])

    market = pd.read_parquet(MARKET_PATH)
    forecasts = pd.read_parquet(FORECASTS_PATH)
    assert_no_true_holdout(market)

    split_max = _split_max_date()
    if split_max is not None:
        market["event_date"] = pd.to_datetime(market["event_date"])
        before = len(market)
        market = market.loc[market["event_date"] > split_max].copy()
        if market.empty:
            print(f"No market rows strictly after split max ({split_max.date()}).")
            return
        print(
            f"Filtered fresh market to dates after {split_max.date()}: "
            f"{before} -> {len(market)} rows"
        )
        fresh_event_dates = set(market["event_date"].dt.strftime("%Y-%m-%d"))
        forecasts = forecasts.copy()
        forecasts["event_date"] = pd.to_datetime(forecasts["event_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        forecasts = forecasts.loc[forecasts["event_date"].isin(fresh_event_dates)].copy()

    trades, stats, calendar_dates, daily_returns, daily_pnl = _run_partition_backtest(
        market, forecasts, edge_threshold
    )

    fresh_dates = sorted(market["event_date"].dropna().unique())
    window_start = pd.Timestamp(fresh_dates[0]).strftime("%Y-%m-%d") if fresh_dates else "?"
    window_end = pd.Timestamp(fresh_dates[-1]).strftime("%Y-%m-%d") if fresh_dates else "?"
    n_calendar_days = len(calendar_dates)
    total_pnl = float(trades["net_pnl_cents"].sum()) if not trades.empty else 0.0
    win_rate = float(trades["resolved"].mean()) if not trades.empty else float("nan")
    ci_lo, ci_hi = _parse_ci(str(stats["ci"]))

    pnl_by_city: dict[str, dict[str, float]] = {}
    if not trades.empty:
        for city, group in trades.groupby("city"):
            pnl_by_city[str(city)] = {
                "n_trades": int(len(group)),
                "pnl_cents": float(group["net_pnl_cents"].sum()),
                "win_rate": float(group["resolved"].mean()),
            }

    fresh_results = {
        "signal": TOP_SIGNAL,
        "sizer": TOP_SIZER,
        "selection": TOP_SELECTION,
        "edge_threshold": edge_threshold,
        "n_calendar_days": n_calendar_days,
        "fresh_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in fresh_dates],
        "n_trades": int(len(trades)),
        "total_pnl_cents": total_pnl,
        "win_rate": win_rate,
        "mean_edge": stats["mean_edge"],
        "max_dd_cents": stats["max_dd_cents"],
        "sharpe": stats["sharpe"],
        "ci": stats["ci"],
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "pnl_by_city": pnl_by_city,
        "daily_pnl": [float(x) for x in daily_pnl],
    }

    print(f"\n=== FRESH VALIDATION EXPANDED ===")
    print(f"Window: {window_start} to {window_end} ({n_calendar_days} calendar days)")
    print(f"Trades: {len(trades)}")
    if np.isfinite(win_rate):
        print(f"Win rate: {win_rate * 100:.1f}%")
    print(f"PnL: {total_pnl:.0f} cents (${total_pnl / 100:.2f})")
    print(f"Sharpe (annualised): {stats['sharpe']} {stats['ci']}")
    print(f"Max drawdown: {stats['max_dd_cents']} cents")
    print(f"Mean edge on taken trades: {stats['mean_edge']}")
    print("\nPer-city breakdown:")
    print(f"  {'City':16} | Trades | PnL (c) | Win rate")
    for city in sorted(pnl_by_city):
        row = pnl_by_city[city]
        wr = row["win_rate"] * 100 if np.isfinite(row["win_rate"]) else float("nan")
        print(f"  {city:16} | {row['n_trades']:6d} | {row['pnl_cents']:7.0f} | {wr:5.1f}%")

    combined: dict[str, object] | None = None
    oos_path = SPLIT_DIR / "time_holdout.parquet"
    if oos_path.exists() and OOS_FORECASTS_PATH.exists():
        time_holdout = pd.read_parquet(oos_path)
        oos_forecasts = pd.read_parquet(OOS_FORECASTS_PATH)
        assert_no_true_holdout(time_holdout)
        oos_trades, oos_stats, oos_dates, oos_returns, oos_daily = _run_partition_backtest(
            time_holdout, oos_forecasts, edge_threshold
        )
        combined_returns = np.concatenate([oos_returns, daily_returns])
        combined_pnl = np.concatenate([oos_daily, daily_pnl])
        combined_sharpe, combined_ci, combined_max_dd = _sharpe_from_series(
            combined_returns, combined_pnl
        )
        oos_start = pd.Timestamp(oos_dates[0]).strftime("%Y-%m-%d") if oos_dates else "?"
        combined_end = window_end
        combined = {
            "window_start": oos_start,
            "window_end": combined_end,
            "n_calendar_days": len(oos_dates) + n_calendar_days,
            "n_trades": int(len(oos_trades) + len(trades)),
            "total_pnl_cents": float(oos_trades["net_pnl_cents"].sum() + total_pnl),
            "sharpe": combined_sharpe,
            "ci": combined_ci,
            "max_dd_cents": combined_max_dd,
        }
        fresh_results["combined_oos_fresh"] = combined
        print("\n=== COMBINED (OOS + FRESH) ===")
        print(
            f"Window: {oos_start} to {combined_end} "
            f"({combined['n_calendar_days']} calendar days)"
        )
        print(f"Total trades: {combined['n_trades']}")
        print(f"Total PnL: {combined['total_pnl_cents']:.0f} cents")
        print(f"Sharpe: {combined['sharpe']} {combined['ci']}")
        print(f"Max DD: {combined['max_dd_cents']:.0f} cents")

    FRESH_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(fresh_results, handle, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
