"""Day 8 Track-J flat evaluation on the time_holdout partition."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINES_DIR = SRC_DIR / "baselines"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SRC_DIR, BASELINES_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backtest_utils import daily_returns, deflated_sharpe, sharpe_stats  # noqa: E402
from day8_edge_diagnostic import DAY8_SIGMA_F, load_day8_predictions  # noqa: E402
from track_j_flat import evaluate_track_j_flat  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TIME_HOLDOUT_PATH = SPLIT_DIR / "time_holdout.parquet"
TRACK_J_DIR = PROJECT_ROOT / "data" / "track_j"
OUTPUT_PATH = TRACK_J_DIR / "oos_trackj_flat_results.parquet"

BASELINE_ROWS = [
    ("make the market", -0.45, -1.98, 1.07, 1.45, -1.55, 4.45, 0.28, 0.82, -1.90),
    ("momentum threshold", -0.52, -2.04, 1.01, 1.36, -1.63, 4.36, 0.25, 0.81, -1.88),
    ("implied favorite", -0.35, -1.88, 1.17, 1.16, -1.84, 4.16, 0.33, 0.77, -1.52),
    ("mode prob threshold", -0.83, -2.36, 0.69, 0.10, -2.89, 3.09, 0.11, 0.53, -0.93),
    ("entropy threshold", 0.42, -1.11, 1.94, -0.49, -3.48, 2.51, 0.69, 0.37, 0.91),
    ("sell longshots", -0.28, -1.80, 1.25, -0.60, -3.60, 2.39, 0.36, 0.34, 0.32),
    ("distribution copy", -9.71, -11.38, -8.05, -6.05, -9.15, -2.95, 0.00, 0.00, -3.66),
]


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _format_float(value: float, digits: int = 2) -> str:
    return "nan" if not math.isfinite(value) else f"{value:0.{digits}f}"


def _load_time_holdout() -> pd.DataFrame:
    partition = pd.read_parquet(TIME_HOLDOUT_PATH)
    if "partition" not in partition.columns:
        raise AssertionError("time_holdout must contain a partition column")
    labels = set(partition["partition"].dropna().astype(str).unique())
    if labels != {"time_holdout"}:
        raise AssertionError(f"Expected only time_holdout rows, found {sorted(labels)}")
    return partition


def _summary(results: pd.DataFrame) -> dict[str, float]:
    returns = daily_returns(results)
    stats = sharpe_stats(returns)
    deflated = deflated_sharpe(returns, n_variants=1)
    no_signal = results["no_signal"].fillna(False).astype(bool)
    n_days = int(results[["city", "event_date"]].drop_duplicates().shape[0])
    n_no_signal = int(results.loc[no_signal, ["city", "event_date"]].drop_duplicates().shape[0])
    n_trades = int(results.loc[~no_signal, ["city", "event_date"]].drop_duplicates().shape[0])
    pnl = pd.to_numeric(results["net_pnl_cents"], errors="coerce").where(~no_signal, 0.0)
    mean_net = float(
        results.assign(_pnl=pnl.fillna(0.0))
        .groupby(["city", "event_date"], sort=True)["_pnl"]
        .sum()
        .mean()
    )
    return {
        "n_days": n_days,
        "n_trades": n_trades,
        "no_signal_pct": 100.0 * n_no_signal / n_days if n_days else float("nan"),
        "mean_net_pnl_c": mean_net,
        "sharpe": float(stats["sharpe_annual"]),
        "ci_low": float(stats["sharpe_ci_low"]),
        "ci_high": float(stats["sharpe_ci_high"]),
        "psr0": float(stats["PSR_0"]),
        "mintrl": float(stats["MinTRL_0"]),
        "sr_deflated": float(deflated["sr_deflated"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "sortino": float(stats["sortino_annual"]),
    }


def _print_comparison(track_j_stats: dict[str, float]) -> None:
    print()
    print("--- IS vs OOS Comparison (Track-J flat + all 7 baselines) ---")
    print("Baseline            | IS Sharpe [CI]          | OOS Sharpe [CI]         | IS PSR(0) | OOS PSR(0) | Decay")
    for name, is_sr, is_lo, is_hi, oos_sr, oos_lo, oos_hi, is_psr, oos_psr, decay in BASELINE_ROWS:
        print(
            f"{name:<19} | {is_sr:6.2f} [{is_lo:6.2f}, {is_hi:5.2f}]    | "
            f"{oos_sr:6.2f} [{oos_lo:6.2f}, {oos_hi:5.2f}]    | "
            f"{is_psr:6.2f}    | {oos_psr:7.2f}    | {decay:5.2f}"
        )
    track_j_decay = 2.59 - track_j_stats["sharpe"]
    print(
        f"{'track_j flat':<19} | {2.59:6.2f} [{1.87:6.2f}, {3.31:5.2f}]    | "
        f"{track_j_stats['sharpe']:6.2f} [{track_j_stats['ci_low']:6.2f}, {track_j_stats['ci_high']:5.2f}]    | "
        f"{'COMP':>6}    | {track_j_stats['psr0']:7.2f}    | {track_j_decay:5.2f}"
    )


def main() -> None:
    time_holdout = _load_time_holdout()
    predictions, cities_evaluated, cities_skipped = load_day8_predictions(time_holdout)
    if cities_evaluated:
        eval_partition = time_holdout[
            time_holdout["city"].map(_city_key).isin(set(cities_evaluated))
        ].copy()
    else:
        eval_partition = time_holdout.iloc[0:0].copy()

    forecasts = predictions.copy()
    forecasts["track_j_sigma_f"] = DAY8_SIGMA_F
    results = evaluate_track_j_flat(
        eval_partition,
        forecasts,
        k=1,
        order_type="taker",
        contracts=1.0,
        min_edge_fee_multiple=2.0,
        min_entry_price=0.15,
    )

    TRACK_J_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_PATH, index=False)
    stats = _summary(results) if not results.empty else {
        "n_days": 0,
        "n_trades": 0,
        "no_signal_pct": float("nan"),
        "mean_net_pnl_c": float("nan"),
        "sharpe": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "psr0": float("nan"),
        "mintrl": float("nan"),
        "sr_deflated": float("nan"),
        "max_drawdown": float("nan"),
        "sortino": float("nan"),
    }

    print("=== Track-J OOS Flat Evaluation (time_holdout) ===")
    print(f"Cities evaluated: {cities_evaluated}")
    print(f"Cities skipped (no OOS predictions): {cities_skipped}")
    print(f"Total OOS trades: {stats['n_trades']}")
    print(f"No-signal %: {_format_float(stats['no_signal_pct'], 1)}%")
    print(f"Mean net PnL (c): {_format_float(stats['mean_net_pnl_c'], 2)}")
    print(
        f"Sharpe: {_format_float(stats['sharpe'], 2)} "
        f"[95% CI: {_format_float(stats['ci_low'], 2)}, {_format_float(stats['ci_high'], 2)}]  (Lo 2002 SE)"
    )
    print(f"PSR(SR*=0): {_format_float(stats['psr0'], 2)}")
    print(f"MinTRL: {_format_float(stats['mintrl'], 0)} days")
    print(f"SR deflated: {_format_float(stats['sr_deflated'], 2)}")
    print(f"Max drawdown: {_format_float(stats['max_drawdown'], 2)}")
    print(f"Sortino: {_format_float(stats['sortino'], 2)}")

    _print_comparison(stats)

    beats_implied = bool(math.isfinite(stats["sharpe"]) and stats["sharpe"] > 1.16)
    beats_gate = bool(math.isfinite(stats["sharpe"]) and stats["sharpe"] > 1.45)
    print()
    print("KEY COMPARISON:")
    print(f"  Track-J OOS Sharpe: {_format_float(stats['sharpe'], 2)}")
    print("  Implied-favorite OOS Sharpe (minimum bar): 1.16")
    print("  Make-the-market OOS Sharpe (gate threshold): 1.45")
    print(f"  Track-J beats implied-favorite: {'YES' if beats_implied else 'NO'}")
    print(f"  Track-J beats make-the-market gate: {'YES' if beats_gate else 'NO'}")


if __name__ == "__main__":
    main()
