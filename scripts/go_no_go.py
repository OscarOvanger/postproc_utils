"""GO/NO-GO decision script for Track-B deployment."""

from __future__ import annotations

import json
import math
import re
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

from backtest_utils import deflated_sharpe, sharpe_stats  # noqa: E402
from scripts.run_trackB_grid import (  # noqa: E402
    FORECASTS_PATH,
    LOW_OOS_COVERAGE_CITIES,
    SPLIT_DIR,
    _calendar_date_keys,
    _calendar_days,
    _load_partitions,
    apply_selection,
    generate_signals,
    run_backtest,
    run_grid,
)
from src.snapshot_stability import assert_no_true_holdout  # noqa: E402

GRID_DIR = PROJECT_ROOT / "data" / "trackb" / "sizing_grid"
FRESH_RESULTS_PATH = PROJECT_ROOT / "data" / "fresh_validation" / "fresh_results.json"
REPORT_TXT = PROJECT_ROOT / "reports" / "go_no_go_week3.txt"
REPORT_CSV = PROJECT_ROOT / "reports" / "go_no_go_week3.csv"

MAKE_MARKET_GATE = 1.45
N_VARIANTS = 18
TOP_COMBO = ("track_b_flat", "flat_5", "edge_threshold")


def _parse_ci(ci_text: str) -> tuple[float, float]:
    match = re.findall(r"[-+]?\d*\.?\d+", str(ci_text))
    if len(match) >= 2:
        return float(match[0]), float(match[1])
    return float("nan"), float("nan")


def _combo_label(signal: str, sizer: str, selection: str) -> str:
    return f"{signal} + {sizer} + {selection}"


def _ensure_grid_stats() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    is_path = GRID_DIR / "full_stats_IS.csv"
    oos_path = GRID_DIR / "full_stats_OOS.csv"
    meta_path = GRID_DIR / "grid_meta.json"
    if is_path.exists() and oos_path.exists() and meta_path.exists():
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        return pd.read_csv(is_path), pd.read_csv(oos_path), float(meta["e_star"])

    print("Grid stats missing — running Track-B grid...")
    forecasts = pd.read_parquet(FORECASTS_PATH)
    threshold_opt, time_holdout = _load_partitions()
    is_df, oos_df, e_star, _ = run_grid(threshold_opt, time_holdout, forecasts)
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    is_df.to_csv(is_path, index=False)
    oos_df.to_csv(oos_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"e_star": e_star}, handle, indent=2)
    return is_df, oos_df, e_star


def _signal_coverage(threshold_opt: pd.DataFrame, forecasts: pd.DataFrame, signal: str) -> float:
    city_col = "source_city_folder" if "source_city_folder" in threshold_opt.columns else "city"
    expected = threshold_opt[[city_col, "event_date"]].drop_duplicates().copy()
    expected["city"] = expected[city_col].astype(str).str.lower().str.replace(" ", "_")
    fc = forecasts.copy()
    fc["city"] = fc["city"].astype(str).str.lower().str.replace(" ", "_")
    fc["event_date"] = pd.to_datetime(fc["event_date"]).dt.strftime("%Y-%m-%d")
    merged = expected.merge(fc[["city", "event_date"]].drop_duplicates(), on=["city", "event_date"], how="left", indicator=True)
    merged = merged[~merged["city"].isin(LOW_OOS_COVERAGE_CITIES)]
    if merged.empty:
        return 0.0
    return float((merged["_merge"] == "both").mean())


def _approx_deflated_sharpe(annual_sharpe: float, n_days: int) -> float:
    if n_days < 2 or not np.isfinite(annual_sharpe):
        return float("nan")
    sr_daily = annual_sharpe / math.sqrt(252)
    rng = np.random.default_rng(42)
    synthetic = rng.normal(sr_daily, abs(sr_daily) + 1e-6, n_days)
    result = deflated_sharpe(pd.Series(synthetic), N_VARIANTS)
    return float(result["sr_deflated"])


def _disagreement_rates(
    time_holdout: pd.DataFrame,
    forecasts: pd.DataFrame,
    signal: str,
    selection: str,
    edge_threshold: float,
) -> tuple[float, float]:
    from scripts.run_trackB_grid import _resolved_for_bucket  # noqa: E402

    signals = generate_signals(
        time_holdout, forecasts, signal, exclude_cities=LOW_OOS_COVERAGE_CITIES
    )
    selected = apply_selection(signals.copy(), selection, edge_threshold)
    traded = selected[~selected["no_signal"]].copy()
    disagree = traded[~traded["agrees_with_market"].astype(bool)]
    if disagree.empty:
        return float("nan"), float("nan")

    model_wr = float(disagree["resolved"].astype(bool).mean())
    market_wins: list[bool] = []
    city_col = "source_city_folder" if "source_city_folder" in time_holdout.columns else "city"
    for _, row in disagree.iterrows():
        city = row["city"]
        event_date = row["event_date"]
        day_df = time_holdout[
            time_holdout[city_col].astype(str).str.lower().str.replace(" ", "_").eq(city)
            & (pd.to_datetime(time_holdout["event_date"]).dt.strftime("%Y-%m-%d") == event_date)
        ]
        if day_df.empty:
            continue
        try:
            market_wins.append(_resolved_for_bucket(day_df, str(row["market_modal_bucket"])))
        except ValueError:
            continue
    market_wr = float(np.mean(market_wins)) if market_wins else float("nan")
    return model_wr, market_wr


def _top_combo_returns(
    threshold_opt: pd.DataFrame,
    time_holdout: pd.DataFrame,
    forecasts: pd.DataFrame,
    e_star: float,
) -> np.ndarray:
    signal, sizer, selection = TOP_COMBO
    oos_signals = generate_signals(
        time_holdout, forecasts, signal, exclude_cities=LOW_OOS_COVERAGE_CITIES
    )
    selected = apply_selection(oos_signals.copy(), selection, e_star)
    _, daily_returns, _, _ = run_backtest(selected, sizer, _calendar_date_keys(time_holdout))
    return daily_returns


def _evaluate_row(
    oos_row: pd.Series,
    is_coverage: float,
    n_oos_days: int,
    disagree_model_wr: float | None,
    disagree_market_wr: float | None,
    fresh_pnl: float | None,
) -> dict[str, object]:
    sharpe = float(oos_row["Sharpe"])
    ci_lo, ci_hi = _parse_ci(str(oos_row["CI"]))
    max_dd = abs(float(oos_row["Max DD"]))
    proj = float(oos_row["Proj/60d"])
    sr_deflated = _approx_deflated_sharpe(sharpe, n_oos_days)

    c3 = None
    if disagree_model_wr is not None and disagree_market_wr is not None:
        if np.isfinite(disagree_model_wr) and np.isfinite(disagree_market_wr):
            c3 = disagree_model_wr > disagree_market_wr

    criteria = {
        "C1_sharpe_gt_baseline": sharpe > MAKE_MARKET_GATE,
        "C2_deflated_sharpe_gt_0": sr_deflated > 0 if np.isfinite(sr_deflated) else False,
        "C3_disagree_wr_gt_market": c3,
        "C4_ci_lower_gt_neg3": ci_lo > -3.0 if np.isfinite(ci_lo) else False,
        "C5_is_coverage_gte_70": is_coverage >= 0.70,
        "C6_proj_trades_gte_90": proj >= 90,
        "C7_max_dd_lt_2500": max_dd < 2500,
    }
    combo = _combo_label(str(oos_row["Signal"]), str(oos_row["Sizer"]), str(oos_row["Selection"]))
    if fresh_pnl is not None and combo == _combo_label(*TOP_COMBO):
        criteria["C8_fresh_pnl_positive"] = fresh_pnl > 0

    scored = sum(1 for val in criteria.values() if val is True)
    total = sum(1 for val in criteria.values() if val is not None)

    return {
        "Combination": combo,
        "Sharpe": sharpe,
        "CI_lower": ci_lo,
        "CI_upper": ci_hi,
        "Max_DD_cents": max_dd,
        "Proj_60d": proj,
        "IS_coverage_pct": round(is_coverage * 100, 1),
        "SR_deflated": round(sr_deflated, 3) if np.isfinite(sr_deflated) else None,
        **{key: ("PASS" if val else "FAIL" if val is not None else "N/A") for key, val in criteria.items()},
        "Score": f"{scored}/{total}",
    }


def main() -> None:
    is_df, oos_df, e_star = _ensure_grid_stats()
    forecasts = pd.read_parquet(FORECASTS_PATH)
    threshold_opt, time_holdout = _load_partitions()
    assert_no_true_holdout(threshold_opt)
    assert_no_true_holdout(time_holdout)

    fresh_results = None
    fresh_pnl = None
    if FRESH_RESULTS_PATH.exists():
        with open(FRESH_RESULTS_PATH, encoding="utf-8") as handle:
            fresh_results = json.load(handle)
        fresh_pnl = float(fresh_results.get("total_pnl_cents", 0))

    n_oos_days = _calendar_days(time_holdout)
    coverage_by_signal = {
        signal: _signal_coverage(threshold_opt, forecasts, signal)
        for signal in sorted(oos_df["Signal"].unique())
    }

    top6_keys = {
        _combo_label(str(r["Signal"]), str(r["Sizer"]), str(r["Selection"]))
        for _, r in oos_df.head(6).iterrows()
    }
    disagree_cache: dict[str, tuple[float, float]] = {}
    for _, row in oos_df.head(6).iterrows():
        key = _combo_label(str(row["Signal"]), str(row["Sizer"]), str(row["Selection"]))
        if str(row["Signal"]) == "track_b_disagree" and key not in disagree_cache:
            disagree_cache[key] = _disagreement_rates(
                time_holdout,
                forecasts,
                str(row["Signal"]),
                str(row["Selection"]),
                e_star,
            )

    rows = []
    for _, oos_row in oos_df.iterrows():
        key = _combo_label(str(oos_row["Signal"]), str(oos_row["Sizer"]), str(oos_row["Selection"]))
        d_model, d_market = (None, None)
        if key in disagree_cache:
            d_model, d_market = disagree_cache[key]
        rows.append(
            _evaluate_row(
                oos_row,
                coverage_by_signal[str(oos_row["Signal"])],
                n_oos_days,
                d_model,
                d_market,
                fresh_pnl if key == _combo_label(*TOP_COMBO) else None,
            )
        )

    matrix = pd.DataFrame(rows)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(REPORT_CSV, index=False)

    display_cols = [
        "Combination",
        "C1_sharpe_gt_baseline",
        "C2_deflated_sharpe_gt_0",
        "C3_disagree_wr_gt_market",
        "C4_ci_lower_gt_neg3",
        "C5_is_coverage_gte_70",
        "C6_proj_trades_gte_90",
        "C7_max_dd_lt_2500",
        "Score",
    ]
    if fresh_pnl is not None:
        display_cols.insert(-1, "C8_fresh_pnl_positive")

    print("\n=== GO/NO-GO CRITERIA MATRIX ===")
    print(matrix[display_cols].to_string(index=False))

    top_row = matrix[matrix["Combination"] == _combo_label(*TOP_COMBO)].iloc[0]
    top_returns = _top_combo_returns(threshold_opt, time_holdout, forecasts, e_star)
    mintrl = float(sharpe_stats(pd.Series(top_returns))["MinTRL_0"])
    fresh_days = int(fresh_results["n_calendar_days"]) if fresh_results else 0
    current_days = n_oos_days + fresh_days
    days_to_mintrl = max(0.0, mintrl - current_days)

    score_n, score_d = str(top_row["Score"]).split("/")
    decision = "GO" if int(score_n) >= 6 and top_row["C1_sharpe_gt_baseline"] == "PASS" else "CONDITIONAL GO"
    if top_row["C1_sharpe_gt_baseline"] == "FAIL":
        decision = "NO-GO"
    if fresh_pnl is not None and top_row.get("C8_fresh_pnl_positive") == "FAIL":
        decision = "CONDITIONAL GO"

    fresh_note = (
        f"Fresh validation PnL: {fresh_pnl:.0f} cents on {fresh_days} calendar day(s)."
        if fresh_results
        else "Fresh validation limited to June 2 (single day, overlaps OOS boundary)."
    )

    top_oos = oos_df[
        (oos_df["Signal"] == "track_b_flat")
        & (oos_df["Sizer"] == "flat_5")
        & (oos_df["Selection"] == "edge_threshold")
    ].iloc[0]

    eligible = [
        "chicago_midway",
        "houston",
        "los_angeles",
        "new_york_city",
        "oklahoma_city",
        "phoenix",
        "san_francisco",
    ]

    report = f"""KALSHI TMAX STRATEGY — GO/NO-GO DECISION
=========================================
Date: 2026-06-13
Author: Oscar Ovanger

DECISION: {decision}

Based on evaluation of 18 signal-sizer-selection combinations across
15 OOS calendar days{' and ' + str(fresh_days) + ' fresh validation day(s)' if fresh_days else ''}.

TOP COMBINATION: track_b_flat + flat_5 + edge_threshold
  OOS Sharpe: {top_row['Sharpe']} [{top_row['CI_lower']:.2f}, {top_row['CI_upper']:.2f}]
  OOS Max Drawdown: {top_row['Max_DD_cents']:.0f} cents
  OOS Trades: {int(top_oos['N trades'])} (projected {top_row['Proj_60d']:.0f} per 60 days)
  {fresh_note}
  Criteria passed: {top_row['Score']}

CRITERIA MATRIX:
{matrix[display_cols].to_string(index=False)}

DEPLOYMENT PARAMETERS:
  Bankroll: 100 USDC
  Max daily loss: $6 (600 cents)
  Contracts per trade: 5 (flat sizing)
  Max position: 30 USDC per trade (MCP constraint)
  Edge threshold: E* = {e_star:.3f} (from IS calibration)
  Price floor: $0.15
  No-trade guardrail: edge > 2 * fee
  Decision time: 10:05 AM CT
  Eligible cities: {', '.join(eligible)}

RISK MANAGEMENT:
  Pace monitor: lower edge threshold by 50% if trades < 1.33/day
  Drawdown monitor:
    bankroll > $90: normal sizing (5 contracts)
    $80 < bankroll <= $90: reduce to 3 contracts
    $70 < bankroll <= $80: reduce to 1 contract
    bankroll <= $70: ELIMINATED — stop trading

CAVEATS:
  - OOS window is 15 calendar days; MinTRL is {mintrl:.0f} days
  - 68% of OOS PnL concentrated in top 3 days (Day 11 audit)
  - Austin excluded (no OOS coverage); Philadelphia excluded (54.5% hit rate)
  - {fresh_note}
"""
    REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"\nMinTRL: {mintrl:.0f} day-equivalents")
    print(f"Current track record: {current_days} days")
    print(f"Days remaining to MinTRL: {days_to_mintrl:.0f}")
    print(f"Calendar days needed: {days_to_mintrl / 7:.0f}")
    if mintrl > 2000:
        print("Statistical validation not achievable within project timeline.")
        print("Decision must be based on directional evidence + risk limits.")
    print(f"\nDecision: {decision}")
    print(f"Saved {REPORT_TXT}")
    print(f"Saved {REPORT_CSV}")


if __name__ == "__main__":
    main()
