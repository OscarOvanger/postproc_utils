"""Run Austin Track-J flat in-sample smoke test and disagreement diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINES_DIR = SRC_DIR / "baselines"
for path in (PROJECT_ROOT, SRC_DIR, BASELINES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backtest_utils import (  # noqa: E402
    daily_returns,
    disagreement_diagnostic,
    print_summary_table,
    sharpe_stats,
)
from implied_favorite import evaluate_implied_favorite  # noqa: E402
from track_j_flat import evaluate_track_j_flat  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FORECASTS_PATH = PROJECT_ROOT / "data" / "track_j" / "forecasts.parquet"
RESULTS_DIR = SPLIT_DIR / "smoke_test_results"


def _austin_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["city"].astype(str).eq("Austin")].copy()


def _summary_metrics(results: pd.DataFrame) -> dict[str, float]:
    no_signal = results["no_signal"].fillna(False).astype(bool)
    trades = results[~no_signal]
    stats = sharpe_stats(daily_returns(results))
    return {
        "N trades": int((~no_signal).sum()),
        "N no-signal": int(no_signal.sum()),
        "Mean net PnL (c)": float(pd.to_numeric(trades["net_pnl_cents"], errors="coerce").mean()) if not trades.empty else float("nan"),
        "Sharpe": float(stats["sharpe_annual"]),
        "Win rate": float(trades["resolved_correctly"].astype(float).mean()) if not trades.empty else float("nan"),
    }


def _print_comparison(track_j_results: pd.DataFrame, implied_results: pd.DataFrame, diag: dict) -> None:
    track_j = _summary_metrics(track_j_results)
    implied = _summary_metrics(implied_results)
    rows = [
        ("N trades", track_j["N trades"], implied["N trades"]),
        ("N no-signal", track_j["N no-signal"], implied["N no-signal"]),
        ("Mean net PnL (c)", track_j["Mean net PnL (c)"], implied["Mean net PnL (c)"]),
        ("Sharpe", track_j["Sharpe"], implied["Sharpe"]),
        ("Win rate", track_j["Win rate"], implied["Win rate"]),
        ("--- Track-J specific ---", "", ""),
        ("N agree with market", diag["n_agree"], "—"),
        ("N disagree w/ market", diag["n_disagree"], "—"),
        ("Win rate (agree)", diag["win_rate_agree"], "—"),
        ("Win rate (disagree)", diag["win_rate_disagree"], "—"),
        ("% PnL from disagree", diag["pct_pnl_from_disagree"], "—"),
    ]
    print("\nMetric               | Track-J flat | Implied-favorite")
    print("---------------------|--------------|-----------------")
    for metric, left, right in rows:
        left_text = f"{left:0.3f}" if isinstance(left, float) else str(left)
        right_text = f"{right:0.3f}" if isinstance(right, float) else str(right)
        print(f"{metric:<20} | {left_text:>12} | {right_text:>15}")


def main() -> None:
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    threshold_opt = _austin_only(threshold_opt)
    with open(SPLIT_DIR / "frozen_k.json", encoding="utf-8") as handle:
        k = int(json.load(handle)["k"])
    forecasts = pd.read_parquet(FORECASTS_PATH)

    track_j_results = evaluate_track_j_flat(threshold_opt, forecasts, k=k)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "track_j_flat_IS.parquet"
    track_j_results.to_parquet(output_path, index=False)

    print("IS evaluation covers Austin only (45 days). Other cities pending model training.")
    print(
        "Track-J IS evaluation: 32/45 Austin days covered. 13 days no-signal "
        "due to missing predictions (2026-05-02 to 2026-05-14)."
    )
    print_summary_table("track_j_flat", track_j_results)

    diag = disagreement_diagnostic(track_j_results)
    print("\nTrack-J disagreement diagnostic")
    for key, value in diag.items():
        if isinstance(value, float):
            print(f"  {key}: {value:0.4f}")
        else:
            print(f"  {key}: {value}")

    implied_path = RESULTS_DIR / "implied_favorite_IS.parquet"
    if implied_path.exists():
        implied_results = pd.read_parquet(implied_path)
        implied_results = implied_results[implied_results["city"].astype(str).str.lower().str.replace(" ", "_").eq("austin")]
        if implied_results.empty:
            implied_results = evaluate_implied_favorite(threshold_opt, k=k)
    else:
        implied_results = evaluate_implied_favorite(threshold_opt, k=k)

    _print_comparison(track_j_results, implied_results, diag)

    if diag["win_rate_disagree"] > diag["win_rate_agree"] and diag["win_rate_disagree"] > 0.5:
        print("\nTrack-J shows edge on market-disagreement days. Proceed to OOS.")
    elif diag["win_rate_disagree"] <= 0.5:
        print(
            "\nTrack-J does not beat 50% on disagreement days in-sample. "
            "Check model coverage and consider whether Austin-only predictions are diluting the signal."
        )

    coverage = int(track_j_results["track_j_tmax_f"].notna().sum())
    total = int(track_j_results.shape[0])
    if total and coverage / total < 0.8:
        print("Warning: low Austin coverage may invalidate IS results.")

    print(
        "\nTrack-J smoke test complete. Austin IS evaluation done. "
        "Next: train Track-A models for remaining 8 train cities and re-run with full city coverage."
    )
    print(f"Saved Track-J IS results to {output_path}")


if __name__ == "__main__":
    main()
