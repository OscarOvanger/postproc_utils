"""Audit Day 11 OOS Sharpe calculations for the top grid combinations."""

from __future__ import annotations

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
    INITIAL_BANKROLL_CENTS,
    LOW_OOS_COVERAGE_CITIES,
    _calendar_date_keys,
    apply_selection,
    generate_signals,
    resolve_edge_threshold,
    run_backtest,
)
from src.snapshot_stability import assert_no_true_holdout  # noqa: E402

FORECASTS_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
STATS_PATH = PROJECT_ROOT / "data" / "trackb" / "sizing_grid" / "full_stats_OOS.csv"

TOP_SIGNAL = "track_b_flat"
TOP_SIZER = "flat_5"
TOP_SELECTION = "edge_threshold"


def _sharpe_annual(returns: pd.Series) -> float:
    values = returns.dropna()
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1) * np.sqrt(252))


def _sharpe_ci(returns: pd.Series) -> tuple[float, float]:
    n = len(returns)
    sr = _sharpe_annual(returns)
    if not np.isfinite(sr) or n < 2:
        return float("nan"), float("nan")
    sr_daily = sr / np.sqrt(252)
    se = np.sqrt((1 + 0.5 * sr_daily**2) / n) * np.sqrt(252)
    return sr - 1.96 * se, sr + 1.96 * se


def _load_top_combo_trades() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    forecasts = pd.read_parquet(FORECASTS_PATH)
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    time_holdout = pd.read_parquet(SPLIT_DIR / "time_holdout.parquet")
    assert_no_true_holdout(threshold_opt)
    assert_no_true_holdout(time_holdout)

    is_signals = generate_signals(threshold_opt, forecasts, TOP_SIGNAL)
    oos_signals = generate_signals(
        time_holdout, forecasts, TOP_SIGNAL, exclude_cities=LOW_OOS_COVERAGE_CITIES
    )
    e_star = resolve_edge_threshold(is_signals, oos_signals, _calendar_date_keys(time_holdout))
    oos_sel = apply_selection(oos_signals, TOP_SELECTION, e_star)
    trades_df, _, _, _ = run_backtest(oos_sel, TOP_SIZER)
    return trades_df, time_holdout, threshold_opt, e_star


def _method_a(trades_df: pd.DataFrame) -> tuple[pd.Series, float]:
    daily_pnl = trades_df.groupby("event_date")["net_pnl_cents"].sum()
    daily_return = daily_pnl / INITIAL_BANKROLL_CENTS
    return daily_return, _sharpe_annual(daily_return)


def _method_b(trades_df: pd.DataFrame) -> tuple[pd.Series, float]:
    trade_return = trades_df["net_pnl_cents"] / (
        trades_df["contracts"] * trades_df["entry_price"] * 100
    )
    return trade_return, _sharpe_annual(trade_return)


def _method_c(trades_df: pd.DataFrame, time_holdout: pd.DataFrame) -> tuple[pd.Series, float]:
    oos_dates = sorted(pd.to_datetime(time_holdout["event_date"].unique()))
    daily_pnl = trades_df.groupby("event_date")["net_pnl_cents"].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    all_dates = pd.date_range(oos_dates[0], oos_dates[-1], freq="D")
    daily_pnl_full = daily_pnl.reindex(all_dates, fill_value=0.0)
    daily_return_full = daily_pnl_full / INITIAL_BANKROLL_CENTS
    return daily_return_full, _sharpe_annual(daily_return_full)


def _method_c_evolving_bankroll(trades_df: pd.DataFrame, time_holdout: pd.DataFrame) -> tuple[pd.Series, float]:
    """Method C with opening bankroll per calendar day (spec-correct variant)."""
    oos_dates = sorted(pd.to_datetime(time_holdout["event_date"].unique()))
    all_dates = pd.date_range(oos_dates[0], oos_dates[-1], freq="D")
    daily_pnl = trades_df.groupby("event_date")["net_pnl_cents"].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)

    bankroll = INITIAL_BANKROLL_CENTS
    returns: list[float] = []
    for day in all_dates:
        day_pnl = float(daily_pnl.get(day, 0.0))
        opening = bankroll
        returns.append(day_pnl / opening if opening > 0 else 0.0)
        bankroll += day_pnl
    series = pd.Series(returns, index=all_dates)
    return series, _sharpe_annual(series)


def _grid_method_used() -> str:
    return (
        "Method C (all calendar days): daily PnL summed across cities per calendar day, "
        "return = day_pnl / opening bankroll; no-trade days included with 0 return."
    )


def audit_top_combo() -> dict[str, object]:
    trades_df, time_holdout, _, e_star = _load_top_combo_trades()
    trades_df = trades_df.copy()
    trades_df["won"] = trades_df["resolved"].astype(bool)

    oos_dates = sorted(pd.to_datetime(time_holdout["event_date"].unique()))
    print("=== 1.1 Sharpe Calculation Method ===")
    print(f"OOS calendar days (unique event dates in partition): {len(oos_dates)}")
    print(f"OOS date range: {oos_dates[0].date()} to {oos_dates[-1].date()}")
    print(f"OOS full calendar span (inclusive): {(oos_dates[-1] - oos_dates[0]).days + 1} days")
    print(f"Total individual trades: {len(trades_df)}")
    print(f"E* threshold: {e_star:.4f}")
    print()

    daily_pnl = trades_df.groupby("event_date")["net_pnl_cents"].sum()
    ret_a, sr_a = _method_a(trades_df)
    ret_b, sr_b = _method_b(trades_df)
    ret_c, sr_c = _method_c(trades_df, time_holdout)
    ret_c_evol, sr_c_evol = _method_c_evolving_bankroll(trades_df, time_holdout)

    print(f"Method A (per trade-day only, n={len(ret_a)}): SR={sr_a:.2f}")
    print(f"Method B (per-trade, n={len(ret_b)}): SR={sr_b:.2f}")
    print(f"Method C (all calendar days, fixed $100 bankroll, n={len(ret_c)}): SR={sr_c:.2f}")
    print(f"Method C* (all calendar days, evolving bankroll, n={len(ret_c_evol)}): SR={sr_c_evol:.2f}")
    print()
    print(f"Grid script method: {_grid_method_used()}")
    if abs(sr_a - sr_c_evol) > 0.5:
        print(
            "NOTE: Pre-fix grid used trade-days-only (Method A). "
            f"Method A SR={sr_a:.2f} vs corrected evolving-bankroll SR={sr_c_evol:.2f}."
        )
    else:
        print("Grid script aligns with Method C (all calendar days).")
    print()

    print("=== 1.2 PnL Concentration ===")
    daily_pnl_sorted = daily_pnl.sort_values(ascending=False)
    total_pnl = daily_pnl.sum()
    top3_pnl = daily_pnl_sorted.head(3).sum()
    top3_pct = 100.0 * top3_pnl / total_pnl if total_pnl else 0.0
    best_day_pnl = daily_pnl_sorted.iloc[0]
    best_day_pct = 100.0 * best_day_pnl / total_pnl if total_pnl else 0.0
    print(f"Top 3 days PnL: {top3_pnl:.0f} cents ({top3_pct:.0f}% of total)")
    print(f"Top 3 days: {daily_pnl_sorted.head(3).index.tolist()}")
    print(f"Best day: {daily_pnl_sorted.index[0]}, PnL: {best_day_pnl:.0f} cents ({best_day_pct:.0f}%)")
    print(f"Winning trade-days: {(daily_pnl > 0).sum()}")
    print(f"Losing trade-days: {(daily_pnl < 0).sum()}")
    print(f"Zero trade-days: {(daily_pnl == 0).sum()}")
    print()

    print("=== 1.3 Entry Price Sanity ===")
    print(
        f"Entry price range: {trades_df['entry_price'].min():.2f} to "
        f"{trades_df['entry_price'].max():.2f}"
    )
    print(f"Entry price mean: {trades_df['entry_price'].mean():.2f}")
    cheap = trades_df[trades_df["entry_price"] < 0.10]
    print(f"Trades with entry price < $0.10: {len(cheap)}")
    print(f"Total fees paid: {trades_df['fee_cents'].sum():.0f} cents")
    print(f"Mean fee per trade: {trades_df['fee_cents'].mean():.1f} cents")
    settled_wins = trades_df[trades_df["resolved"]]
    if not settled_wins.empty:
        print(
            f"Resolved YES trades: mean entry {settled_wins['entry_price'].mean():.2f} "
            f"(not $1.00 settlement price)"
        )
    print()

    print("=== 1.4 Win Rate by Entry Price Quintile ===")
    trades_df["price_decile"] = pd.qcut(trades_df["entry_price"], 5, labels=False, duplicates="drop")
    for d in sorted(trades_df["price_decile"].dropna().unique()):
        subset = trades_df[trades_df["price_decile"] == d]
        wr = subset["won"].mean()
        mean_price = subset["entry_price"].mean()
        mean_pnl = subset["net_pnl_cents"].mean()
        print(f"Decile {d}: price={mean_price:.2f}, win_rate={wr:.1%}, mean_pnl={mean_pnl:.1f}c, n={len(subset)}")
    print()

    return {
        "trades_df": trades_df,
        "time_holdout": time_holdout,
        "daily_pnl": daily_pnl,
        "sr_a": sr_a,
        "sr_c": sr_c,
        "sr_c_evol": sr_c_evol,
        "top3_pct": top3_pct,
        "total_fees": float(trades_df["fee_cents"].sum()),
        "cheap_trades": len(cheap),
    }


def recompute_top3(stats_path: Path = STATS_PATH) -> pd.DataFrame:
    """Recompute corrected Sharpe for top 3 surviving combinations."""
    forecasts = pd.read_parquet(FORECASTS_PATH)
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    time_holdout = pd.read_parquet(SPLIT_DIR / "time_holdout.parquet")
    assert_no_true_holdout(threshold_opt)
    assert_no_true_holdout(time_holdout)

    stats = pd.read_csv(stats_path)
    surviving = stats[
        (~stats["Eliminated"])
        & (stats["Proj/60d"] >= 90)
        & (stats["Sharpe"] > 0)
    ].sort_values("Sharpe", ascending=False)
    top3 = surviving.head(3)

    e_star = resolve_edge_threshold(
        generate_signals(threshold_opt, forecasts, "track_b_flat"),
        generate_signals(time_holdout, forecasts, "track_b_flat", exclude_cities=LOW_OOS_COVERAGE_CITIES),
        _calendar_date_keys(time_holdout),
    )

    rows = []
    for _, row in top3.iterrows():
        signal, sizer, selection = row["Signal"], row["Sizer"], row["Selection"]
        oos_signals = generate_signals(
            time_holdout, forecasts, signal, exclude_cities=LOW_OOS_COVERAGE_CITIES
        )
        threshold = e_star if selection == "edge_threshold" else None
        oos_sel = apply_selection(oos_signals, selection, threshold)
        trades_df, _, _, _ = run_backtest(oos_sel, sizer)
        ret_c, sr_c = _method_c_evolving_bankroll(trades_df, time_holdout)
        ci_lo, ci_hi = _sharpe_ci(ret_c)
        rows.append(
            {
                "Combination": f"{signal} + {sizer} + {selection}",
                "Original SR": row["Sharpe"],
                "Corrected SR": round(sr_c, 2) if np.isfinite(sr_c) else float("nan"),
                "Corrected CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if np.isfinite(ci_lo) else "[nan, nan]",
                "N days": len(ret_c),
            }
        )

    result = pd.DataFrame(rows)
    print("=== 1.5 Corrected Sharpe (Method C) — Top 3 Surviving ===")
    print(result.to_string(index=False))
    print()
    return result


def print_verdict(audit: dict[str, object]) -> None:
    sr_a = float(audit["sr_a"])
    sr_c_evol = float(audit["sr_c_evol"])
    top3_pct = float(audit["top3_pct"])
    total_fees = float(audit["total_fees"])
    cheap_trades = int(audit["cheap_trades"])

    method_c_matches = abs(sr_a - sr_c_evol) < 0.5
    discrepancy = f"trade-day SR={sr_a:.2f} vs calendar-day SR={sr_c_evol:.2f}"
    concentrated = top3_pct > 70
    no_suspicious = cheap_trades == 0
    fees_ok = total_fees > 0
    all_clean = fees_ok and no_suspicious

    print("=== AUDIT SUMMARY ===")
    print(
        f"Sharpe method used: {'CORRECT (calendar days)' if method_c_matches else 'INCORRECT - ' + discrepancy}"
    )
    print(f"PnL concentration: {'CONCENTRATED' if concentrated else 'DISTRIBUTED'} ({top3_pct:.0f}% in top 3 days)")
    print(f"Entry prices: {'CLEAN' if no_suspicious else 'SUSPICIOUS'}")
    print(f"Fees applied: {'YES' if fees_ok else 'NO - CRITICAL ERROR'}")
    print(f"Overall: {'RESULTS VALID' if all_clean else 'RESULTS NEED CORRECTION'}")
    print()
    print(f"Corrected Sharpe for top combo (Method C, evolving bankroll): {sr_c_evol:.2f}")


def main() -> None:
    audit = audit_top_combo()
    recompute_top3()
    print_verdict(audit)


if __name__ == "__main__":
    main()
