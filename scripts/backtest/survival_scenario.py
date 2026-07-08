#!/usr/bin/env python3
"""Forward survival scenarios for the deployed v5b NGBoost strategy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_PATH_FILE = PROJECT_ROOT / "reports" / "backtest_model_path.txt"
ELIGIBLE_DATES_CSV = PROJECT_ROOT / "reports" / "backtest_eligible_dates.csv"
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "polymarket_history" / "snapshots"

STARTING_BANKROLL_USD = 84.98
ELIMINATION_USD = 70.0
TARGET_BANKROLL_USD = 100.0
SCENARIO_DAYS = 49
N_CONTRACTS = 5

VARIANTS: dict[str, dict[str, int]] = {
    "current": {"n_buckets_per_city": 1, "max_trades_per_day": 2},
    "top2_buckets": {"n_buckets_per_city": 2, "max_trades_per_day": 2},
    "top3_buckets": {"n_buckets_per_city": 3, "max_trades_per_day": 3},
}

WINDOWS = ("early", "middle", "late")


@dataclass
class Candidate:
    city: str
    bucket: str
    raw_prob: float
    effective_prob: float
    entry_price: float
    edge: float


def output_paths(output_tag: str) -> tuple[Path, str]:
    suffix = f"_{output_tag}" if output_tag else ""
    return PROJECT_ROOT / "reports" / f"survival_scenario{suffix}.json", suffix


def equity_path(variant: str, window: str, suffix: str) -> Path:
    return PROJECT_ROOT / "reports" / f"survival_equity{suffix}_{variant}_{window}.csv"


def load_runtime_dependencies() -> None:
    global NgBoostBacktestModels
    global RollingBiasCache
    global bc
    global load_wunderground_bias
    global np
    global pd
    global predict_bucket_probs_from_mu
    global predict_mu_sigma
    global sharpe_stats
    global two_piece_ratio_for_date

    import numpy as _np
    import pandas as _pd

    import backtest.common as _bc
    from backtest.ngboost_inference import (
        NgBoostBacktestModels as _NgBoostBacktestModels,
    )
    from backtest.ngboost_inference import (
        predict_bucket_probs_from_mu as _predict_bucket_probs_from_mu,
    )
    from backtest.ngboost_inference import predict_mu_sigma as _predict_mu_sigma
    from backtest.ngboost_inference import (
        two_piece_ratio_for_date as _two_piece_ratio_for_date,
    )
    from backtest_utils import sharpe_stats as _sharpe_stats
    from poly_trading_pipeline import load_wunderground_bias as _load_wunderground_bias
    from rolling_bias import RollingBiasCache as _RollingBiasCache

    np = _np
    pd = _pd
    bc = _bc
    NgBoostBacktestModels = _NgBoostBacktestModels
    predict_bucket_probs_from_mu = _predict_bucket_probs_from_mu
    predict_mu_sigma = _predict_mu_sigma
    two_piece_ratio_for_date = _two_piece_ratio_for_date
    sharpe_stats = _sharpe_stats
    load_wunderground_bias = _load_wunderground_bias
    RollingBiasCache = _RollingBiasCache


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def check_prerequisites() -> None:
    missing: list[str] = []
    if not MODEL_PATH_FILE.exists():
        missing.append(
            f"ERROR: run step0 first - missing {MODEL_PATH_FILE}\n"
            "       python scripts/backtest/step0_preconditions.py"
        )
    if not ELIGIBLE_DATES_CSV.exists():
        missing.append(
            f"ERROR: run step1 first - missing {ELIGIBLE_DATES_CSV}\n"
            "       python scripts/backtest/step1_eligible_range.py"
        )
    if missing:
        print("\n".join(missing))
        sys.exit(1)

    has_snapshots = False
    if SNAPSHOTS_DIR.exists():
        for child in SNAPSHOTS_DIR.iterdir():
            if child.is_dir() and next(child.glob("*.parquet"), None) is not None:
                has_snapshots = True
                break
    if not has_snapshots:
        missing.append(
            f"ERROR: missing real Polymarket snapshot parquet files under {SNAPSHOTS_DIR}\n"
            "       Backfill real order-book history before running this scenario."
        )
    if missing:
        print("\n".join(missing))
        sys.exit(1)


def build_windows(all_dates: list[str]) -> dict[str, list[str]]:
    total = len(all_dates)
    if total < SCENARIO_DAYS:
        raise ValueError(
            f"Need at least {SCENARIO_DAYS} eligible dates; found {total}. "
            "Run step1 after completing the historical backfill."
        )

    if total >= 136:
        starts = {"early": 0, "middle": 43, "late": 87}
    else:
        starts = {
            "early": 0,
            "middle": max(0, min(total - SCENARIO_DAYS, total // 2 - 24)),
            "late": total - SCENARIO_DAYS,
        }

    windows = {
        name: all_dates[start : start + SCENARIO_DAYS]
        for name, start in starts.items()
    }
    for name, dates in windows.items():
        if len(dates) != SCENARIO_DAYS or len(set(dates)) != SCENARIO_DAYS:
            raise ValueError(f"Window {name!r} does not contain {SCENARIO_DAYS} distinct dates")
    return windows


def evaluate_city_buckets(
    city: str,
    date_str: str,
    models: NgBoostBacktestModels,
    config: dict[str, Any],
    bias_cache: RollingBiasCache,
    wu_bias: dict[str, dict[str, float | int]],
    n_buckets_per_city: int,
) -> tuple[list[Candidate], float | None]:
    frame = bc.load_day_snapshot(city, date_str)
    if frame is None:
        return [], None

    snap_rows, _entry_ts, _excluded = bc.select_entry_snapshot(frame, city, date_str)
    if snap_rows.empty:
        return [], None

    quotes = bc.quotes_at_entry(snap_rows)
    bucket_labels = [
        str(bucket)
        for bucket in quotes["bucket"].astype(str).tolist()
        if not str(bucket).startswith("Will ")
    ]
    if not bucket_labels:
        return [], None

    mu_sigma = predict_mu_sigma(models, city, date_str)
    if mu_sigma is None:
        return [], None

    raw_mu, _sigma = mu_sigma
    static_bias = float(wu_bias.get(city, {}).get("median_bias", 0.0))
    mu_wu = raw_mu - static_bias
    bias = float(np.clip(bias_cache.bias(city, date_str), -1.5, 1.5))
    mu_adj = mu_wu - bias
    ratio_down = two_piece_ratio_for_date(config, date_str)
    probs = predict_bucket_probs_from_mu(
        models, city, date_str, bucket_labels, mu_adj, ratio_down=ratio_down
    )
    if not probs:
        return [], raw_mu

    lam = bc.shrinkage_lambda(config)
    candidates: list[Candidate] = []
    for _, qrow in quotes.iterrows():
        label = str(qrow["bucket"])
        if label.startswith("Will "):
            continue
        ask = qrow.get("best_ask")
        if ask is None or not pd.notna(ask):
            continue
        entry_price = round(float(ask) - bc.MAKER_TICK, 4)
        if entry_price < bc.PRICE_FLOOR:
            continue
        raw_prob = float(probs.get(label, 0.0))
        effective_prob = bc.effective_probability(raw_prob, entry_price, lam)
        edge = effective_prob - entry_price
        if edge <= 0:
            continue
        candidates.append(
            Candidate(
                city=city,
                bucket=label,
                raw_prob=raw_prob,
                effective_prob=effective_prob,
                entry_price=entry_price,
                edge=edge,
            )
        )

    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates[:n_buckets_per_city], raw_mu


def collect_day_candidates(
    day_rows: pd.DataFrame,
    models: NgBoostBacktestModels,
    config: dict[str, Any],
    bias_cache: RollingBiasCache,
    wu_bias: dict[str, dict[str, float | int]],
    n_buckets_per_city: int,
) -> tuple[list[Candidate], dict[str, float], int]:
    candidates: list[Candidate] = []
    raw_mu_by_city: dict[str, float] = {}
    n_city_days_skipped = 0

    for _, row in day_rows.iterrows():
        city = str(row["city"])
        date_str = str(row["date"])
        cloud_cover = bc.peak_cloud_cover_for_day(city, date_str)
        if bc.convective_skip(city, date_str, cloud_cover, config):
            n_city_days_skipped += 1
            continue
        city_candidates, raw_mu = evaluate_city_buckets(
            city,
            date_str,
            models,
            config,
            bias_cache,
            wu_bias,
            n_buckets_per_city,
        )
        candidates.extend(city_candidates)
        if raw_mu is not None:
            raw_mu_by_city[city] = raw_mu

    return candidates, raw_mu_by_city, n_city_days_skipped


def select_day_trades(
    candidates: list[Candidate],
    edge_threshold: float,
    max_trades_per_day: int,
    *,
    pace_fallback_enabled: bool = False,
    pace_fallback_min_edge: float = 0.010,
    pace_fallback_target_trades: int = 2,
) -> list[tuple[Candidate, str]]:
    filtered = [candidate for candidate in candidates if candidate.edge >= edge_threshold]
    filtered.sort(key=lambda c: c.edge, reverse=True)
    selected: list[tuple[Candidate, str]] = [
        (candidate, "primary") for candidate in filtered[:max_trades_per_day]
    ]

    if pace_fallback_enabled and len(selected) < pace_fallback_target_trades:
        taken = {(c.city, c.bucket) for c, _ in selected}
        remaining = [
            candidate
            for candidate in candidates
            if (candidate.city, candidate.bucket) not in taken
            and candidate.edge > pace_fallback_min_edge
        ]
        remaining.sort(key=lambda c: c.edge, reverse=True)
        for candidate in remaining:
            if len(selected) >= pace_fallback_target_trades:
                break
            selected.append((candidate, "fallback"))

    return selected


def settle_candidate(
    candidate: Candidate,
    date_str: str,
    wu: pd.DataFrame,
) -> dict[str, Any] | None:
    wu_row = wu[(wu["city"] == candidate.city) & (wu["date"] == date_str)]
    if wu_row.empty:
        return None
    actual = float(wu_row.iloc[0]["wunderground_tmax"])
    won = bc.temp_in_bucket(actual, candidate.bucket)
    pnl = bc.settlement_pnl(
        n_contracts=N_CONTRACTS,
        entry_price=candidate.entry_price,
        won=won,
    )
    return {
        "actual_tmax": actual,
        "won": won,
        "exit_price": 1.0 if won else 0.0,
        "pnl_usd": pnl,
    }


def record_day_residuals(
    day_rows: pd.DataFrame,
    raw_mu_by_city: dict[str, float],
    bias_cache: RollingBiasCache,
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
    date_str: str,
) -> None:
    for _, row in day_rows.iterrows():
        city = str(row["city"])
        if city not in raw_mu_by_city:
            continue
        wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
        if wu_row.empty:
            continue
        actual = float(wu_row.iloc[0]["wunderground_tmax"])
        static_bias = float(wu_bias.get(city, {}).get("median_bias", 0.0))
        corrected_mu = raw_mu_by_city[city] - static_bias
        bias_cache.record(city, date_str, corrected_mu, actual)


def metrics_from_paths(
    dates: list[str],
    trades: list[dict[str, Any]],
    equity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    traded = [trade for trade in trades if trade.get("traded")]
    pnls = [float(trade.get("pnl_usd", 0.0)) for trade in traded]
    total_pnl = sum(pnls)
    wins = sum(1 for pnl in pnls if pnl > 0)
    by_date = {date_str: 0.0 for date_str in dates}
    for trade in traded:
        date_str = str(trade["date"])
        by_date[date_str] = by_date.get(date_str, 0.0) + float(trade.get("pnl_usd", 0.0))
    returns = pd.Series(
        {date_str: pnl / STARTING_BANKROLL_USD for date_str, pnl in by_date.items()},
        dtype=float,
    ).sort_index()
    stats = sharpe_stats(returns)

    bankroll_path = [STARTING_BANKROLL_USD] + [
        float(row["bankroll_usd"]) for row in equity_rows
    ]
    bankroll_series = pd.Series(bankroll_path, dtype=float)
    running_peak = bankroll_series.cummax()
    drawdown = bankroll_series - running_peak

    final_bankroll = float(bankroll_series.iloc[-1])
    min_bankroll = float(bankroll_series.min())
    max_drawdown = float(drawdown.min())
    eliminated = any(bool(row.get("eliminated")) for row in equity_rows)

    computed_edges = [float(t.get("edge", 0.0)) for t in traded]
    realized_edges = [
        (1.0 if t.get("won") else 0.0) - float(t.get("entry_price", 0.0)) for t in traded
    ]

    return {
        "trades": len(traded),
        "win_rate": wins / len(traded) if traded else 0.0,
        "total_pnl_usd": round(total_pnl, 4),
        "final_bankroll_usd": round(final_bankroll, 4),
        "sharpe": float(stats.get("sharpe_annual", float("nan"))),
        "max_drawdown_usd": round(max_drawdown, 4),
        "min_bankroll_usd": round(min_bankroll, 4),
        "eliminated": eliminated,
        "trades_per_day": len(traded) / len(dates) if dates else 0.0,
        "survived": not eliminated,
        "recovered_to_target": final_bankroll >= TARGET_BANKROLL_USD,
        "avg_computed_edge": float(np.mean(computed_edges)) if computed_edges else float("nan"),
        "avg_realized_edge": float(np.mean(realized_edges)) if realized_edges else float("nan"),
    }


def run_scenario(
    variant_name: str,
    variant_config: dict[str, int],
    window_name: str,
    dates: list[str],
    eligible: pd.DataFrame,
    models: NgBoostBacktestModels,
    config: dict[str, Any],
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    bias_cache = RollingBiasCache(
        halflife_days=int(config.get("rolling_bias_halflife_days", 20)),
        max_correction_f=float(config.get("max_rolling_correction_f", 1.5)),
    )
    bias_cache.seed_from_parquet()

    edge_threshold = float(config.get("edge_threshold", 0.037))
    bankroll = STARTING_BANKROLL_USD
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    eliminated = False
    n_city_days_skipped = 0
    n_budget_dropped = 0

    for day_index, date_str in enumerate(dates, start=1):
        day_rows = eligible[eligible["date"] == date_str]
        bankroll_before = bankroll
        day_spent = 0.0
        day_pnl = 0.0
        day_trades = 0
        raw_mu: dict[str, float] = {}

        if not eliminated:
            budget = bc.daily_budget_ngboost(bankroll, config)
            candidates, raw_mu, day_skipped = collect_day_candidates(
                day_rows,
                models,
                config,
                bias_cache,
                wu_bias,
                int(variant_config["n_buckets_per_city"]),
            )
            n_city_days_skipped += day_skipped
            selected = select_day_trades(
                candidates,
                edge_threshold,
                int(variant_config["max_trades_per_day"]),
                pace_fallback_enabled=bool(config.get("pace_fallback_enabled", False)),
                pace_fallback_min_edge=float(config.get("pace_fallback_min_edge", 0.010)),
                pace_fallback_target_trades=int(config.get("pace_fallback_target_trades", 2)),
            )

            for candidate, entry_mode in selected:
                assert N_CONTRACTS == 5
                trade_cost = N_CONTRACTS * candidate.entry_price
                if day_spent + trade_cost > budget:
                    n_budget_dropped += 1
                    continue
                settlement = settle_candidate(candidate, date_str, wu)
                if settlement is None:
                    continue
                day_spent += trade_cost
                day_pnl += float(settlement["pnl_usd"])
                day_trades += 1
                trade_record = {
                    "date": date_str,
                    "day_index": day_index,
                    "window": window_name,
                    "variant": variant_name,
                    "traded": True,
                    "entry_mode": entry_mode,
                    **asdict(candidate),
                    "n_contracts": N_CONTRACTS,
                    "cost_usd": round(trade_cost, 4),
                    "exit_type": "settlement",
                    "exit_price": settlement["exit_price"],
                    "won": settlement["won"],
                    "actual_tmax": settlement["actual_tmax"],
                    "pnl_usd": settlement["pnl_usd"],
                    "bankroll_before": round(bankroll_before, 4),
                }
                trades.append(trade_record)

            bankroll = round(bankroll + day_pnl, 4)
            if bankroll <= ELIMINATION_USD:
                eliminated = True

            for trade in trades[-day_trades:]:
                trade["bankroll_after"] = bankroll

            record_day_residuals(day_rows, raw_mu, bias_cache, wu, wu_bias, date_str)

        equity_rows.append(
            {
                "date": date_str,
                "day_index": day_index,
                "window": window_name,
                "variant": variant_name,
                "bankroll_before_usd": round(bankroll_before, 4),
                "day_pnl_usd": round(day_pnl, 4),
                "day_spent_usd": round(day_spent, 4),
                "bankroll_usd": round(bankroll, 4),
                "daily_cap_usd": round(bc.daily_budget_ngboost(bankroll_before, config), 4)
                if not eliminated or bankroll_before > ELIMINATION_USD
                else 0.0,
                "trades": day_trades,
                "eliminated": eliminated,
                "recovered_to_target": bankroll >= TARGET_BANKROLL_USD,
            }
        )

    metrics = metrics_from_paths(dates, trades, equity_rows)
    metrics["n_city_days_skipped"] = n_city_days_skipped
    metrics["n_budget_dropped"] = n_budget_dropped
    return {
        "variant": variant_name,
        "window": window_name,
        "start_date": dates[0],
        "end_date": dates[-1],
        "metrics": metrics,
        "trades": trades,
        "equity": equity_rows,
    }


def fmt_money(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}${value:.2f}"


def fmt_float(value: float) -> str:
    return f"{value:.2f}" if math.isfinite(value) else "NA"


def print_window_summary(window_name: str, dates: list[str], results: dict[str, dict[str, Any]]) -> None:
    print(f"\nWindow: {window_name} ({dates[0]} to {dates[-1]})")
    print("-" * 57)
    headers = ["Metric", *VARIANTS.keys()]
    print(f"{headers[0]:<24} {headers[1]:>10} {headers[2]:>12} {headers[3]:>12}")

    rows = [
        ("Trades", lambda m: f"{m['trades']:.0f}"),
        ("Win rate", lambda m: f"{100 * m['win_rate']:.1f}%"),
        ("Total PnL", lambda m: fmt_money(m["total_pnl_usd"], signed=True)),
        ("Final bankroll", lambda m: fmt_money(m["final_bankroll_usd"])),
        ("Sharpe (ann.)", lambda m: fmt_float(m["sharpe"])),
        ("Max drawdown", lambda m: fmt_money(m["max_drawdown_usd"])),
        ("Min bankroll", lambda m: fmt_money(m["min_bankroll_usd"])),
        ("Eliminated", lambda m: "YES" if m["eliminated"] else "NO"),
        ("Trades/day", lambda m: f"{m['trades_per_day']:.2f}"),
    ]
    for label, formatter in rows:
        values = [formatter(results[variant]["metrics"]) for variant in VARIANTS]
        print(f"{label:<24} {values[0]:>10} {values[1]:>12} {values[2]:>12}")


def print_cross_window_summary(results: dict[str, dict[str, Any]]) -> None:
    print("\n=== CROSS-WINDOW SUMMARY ===")
    print(f"{'Scenario':<22} {'Survived?':>10} {'Final $':>12} {'Sharpe':>10} {'Max DD':>10}")
    for variant in VARIANTS:
        for window in WINDOWS:
            scenario = results[window][variant]
            metrics = scenario["metrics"]
            survived = "YES" if not metrics["eliminated"] else "NO"
            print(
                f"{variant + '/' + window:<22} {survived:>10} "
                f"{fmt_money(metrics['final_bankroll_usd']):>12} "
                f"{fmt_float(metrics['sharpe']):>10} "
                f"{fmt_money(metrics['max_drawdown_usd']):>10}"
            )

    all_survivors: list[tuple[str, float]] = []
    for variant in VARIANTS:
        scenarios = [results[window][variant]["metrics"] for window in WINDOWS]
        if all(not metrics["eliminated"] for metrics in scenarios):
            worst_drawdown = min(float(metrics["max_drawdown_usd"]) for metrics in scenarios)
            all_survivors.append((variant, worst_drawdown))
    if all_survivors:
        best_variant, _worst_dd = max(all_survivors, key=lambda item: item[1])
        print(f"\nVERDICT: {best_variant} survived all 3 windows with lowest max drawdown.")
    else:
        print("\nVERDICT: No strategy survived all 3 windows.")


def write_outputs(
    report_path: Path,
    suffix: str,
    payload: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for window, variant_results in payload["results"].items():
        for variant, scenario in variant_results.items():
            pd.DataFrame(scenario["equity"]).to_csv(equity_path(variant, window, suffix), index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def apply_two_piece_ratio(config: dict[str, Any], ratio_arg: str | None) -> dict[str, Any]:
    cfg = dict(config)
    if ratio_arg is None or ratio_arg == "config":
        return cfg
    if str(ratio_arg).lower() in {"null", "none", ""}:
        cfg["two_piece_sigma_down_ratio"] = None
        return cfg
    cfg["two_piece_sigma_down_ratio"] = float(ratio_arg)
    return cfg


def scenario_comparison_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    computed = float(metrics["avg_computed_edge"])
    realized = float(metrics["avg_realized_edge"])
    return {
        "label": label,
        "n_trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "avg_computed_edge": computed,
        "avg_realized_edge": realized,
        "calibration_gap": computed - realized,
        "total_pnl": metrics["total_pnl_usd"],
        "max_drawdown": metrics["max_drawdown_usd"],
    }


def run_two_piece_eval(
    windows: dict[str, list[str]],
    eligible: pd.DataFrame,
    config: dict[str, Any],
    models: NgBoostBacktestModels,
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
    ratio_hat: float,
) -> None:
    variant_name = "current"
    variant_config = VARIANTS[variant_name]
    window_name = "late"
    dates = windows[window_name]
    window_eligible = eligible[eligible["date"].isin(dates)].copy()

    print("=== TWO-PIECE GAUSSIAN EVAL (late window, current variant) ===")
    print(f"Window: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"Calibrated ratio_down (summer only): {ratio_hat:.4f}")

    rows: list[dict[str, Any]] = []
    pit_rows: dict[str, float] = {}

    for label, ratio_value in [("null", None), ("ratio_hat", ratio_hat)]:
        cfg = apply_two_piece_ratio(config, "null" if ratio_value is None else str(ratio_value))
        scenario = run_scenario(
            variant_name,
            variant_config,
            window_name,
            dates,
            window_eligible,
            models,
            cfg,
            wu,
            wu_bias,
        )
        metrics = scenario["metrics"]
        rows.append(scenario_comparison_row(label, metrics))

        if label == "null":
            late = metrics
            if (
                late["trades"] != BASELINE_LATE_CURRENT["trades"]
                or abs(late["win_rate"] - BASELINE_LATE_CURRENT["win_rate"]) > 0.002
                or abs(late["final_bankroll_usd"] - BASELINE_LATE_CURRENT["final_bankroll_usd"]) > 0.05
            ):
                raise SystemExit(
                    "BASELINE MISMATCH for two_piece=null late/current: "
                    f"got trades={late['trades']} win={late['win_rate']:.3f} "
                    f"final=${late['final_bankroll_usd']:.2f}; expected "
                    f"{BASELINE_LATE_CURRENT['trades']} trades, "
                    f"{BASELINE_LATE_CURRENT['win_rate']:.1%} win, "
                    f"${BASELINE_LATE_CURRENT['final_bankroll_usd']:.2f} final."
                )

    print("\n=== LATE-WINDOW COMPARISON ===")
    print(
        f"{'label':<12} {'trades':>7} {'win%':>7} {'comp_edge':>10} "
        f"{'real_edge':>10} {'cal_gap':>9} {'pnl':>8} {'max_dd':>8}"
    )
    for row in rows:
        print(
            f"{row['label']:<12} {row['n_trades']:7d} "
            f"{100*row['win_rate']:6.1f}% {row['avg_computed_edge']:10.3f} "
            f"{row['avg_realized_edge']:10.3f} {row['calibration_gap']:9.3f} "
            f"{row['total_pnl']:+8.2f} {row['max_drawdown']:8.2f}"
        )

    null_row = rows[0]
    ratio_row = rows[1]
    cal_gap_improved = ratio_row["calibration_gap"] < null_row["calibration_gap"]

    cal_csv = PROJECT_ROOT / "data" / "analysis" / "two_piece_calibration.csv"
    pit_improved = False
    if cal_csv.exists():
        cal_df = pd.read_csv(cal_csv)
        summer = cal_df[cal_df["subset"] == "summer_2025"]
        if not summer.empty:
            gauss_p = float(summer[summer["metric"] == "pit_ks_pvalue_gaussian"]["value"].iloc[0])
            two_p = float(summer[summer["metric"] == "pit_ks_pvalue_two_piece"]["value"].iloc[0])
            pit_rows = {"gaussian": gauss_p, "two_piece": two_p}
            pit_improved = two_p > gauss_p

    print("\n=== DECISION RULE ===")
    if pit_rows:
        print(
            f"Validation summer PIT KS p-value: gaussian={pit_rows['gaussian']:.4f}, "
            f"two_piece={pit_rows['two_piece']:.4f}"
        )
    if cal_gap_improved and pit_improved:
        print(
            f"ENABLE two_piece_sigma_down_ratio={ratio_hat:.4f}: late-window calibration gap "
            f"shrinks ({null_row['calibration_gap']:.3f} -> {ratio_row['calibration_gap']:.3f}) "
            "and validation PIT improves."
        )
    else:
        reasons = []
        if not cal_gap_improved:
            reasons.append(
                f"late calibration gap did not shrink ({null_row['calibration_gap']:.3f} -> "
                f"{ratio_row['calibration_gap']:.3f})"
            )
        if not pit_improved:
            reasons.append("validation PIT did not improve")
        print(
            "Do NOT enable two-piece Gaussian automatically. "
            + "; ".join(reasons)
            + ". Final bankroll is not a selection criterion."
        )


def apply_convective_threshold(config: dict[str, Any], threshold_arg: str | None) -> dict[str, Any]:
    cfg = dict(config)
    if threshold_arg is None or threshold_arg == "config":
        return cfg
    if str(threshold_arg).lower() in {"null", "none", ""}:
        cfg["convective_cloud_skip_threshold"] = None
        return cfg
    cfg["convective_cloud_skip_threshold"] = float(threshold_arg)
    return cfg


def city_pnl_by_trades(trades: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for trade in trades:
        if not trade.get("traded"):
            continue
        city = str(trade["city"])
        totals[city] = totals.get(city, 0.0) + float(trade.get("pnl_usd", 0.0))
    return totals


BASELINE_LATE_CURRENT = {
    "trades": 96,
    "win_rate": 0.219,
    "final_bankroll_usd": 100.58,
}
SWEEP_THRESHOLDS: list[float | None] = [None, 0.6, 0.5, 0.4]
LATE_PNL_CITIES = ["houston", "atlanta", "miami", "chicago", "dallas"]
SWEEP_CSV = PROJECT_ROOT / "data" / "analysis" / "convective_sweep.csv"


def run_convective_sweep(
    windows: dict[str, list[str]],
    eligible: pd.DataFrame,
    config: dict[str, Any],
    models: NgBoostBacktestModels,
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
) -> None:
    variant_name = "current"
    variant_config = VARIANTS[variant_name]
    rows: list[dict[str, Any]] = []
    baseline_late_pnl: dict[str, float] | None = None

    print("=== CONVECTIVE THRESHOLD SWEEP (current variant) ===")
    for threshold in SWEEP_THRESHOLDS:
        threshold_label = "null" if threshold is None else f"{threshold:.1f}"
        cfg = apply_convective_threshold(config, "null" if threshold is None else str(threshold))
        print(f"\n--- threshold={threshold_label} ---")
        for window_name, dates in windows.items():
            scenario = run_scenario(
                variant_name,
                variant_config,
                window_name,
                dates,
                eligible[eligible["date"].isin(dates)].copy(),
                models,
                cfg,
                wu,
                wu_bias,
            )
            metrics = scenario["metrics"]
            row = {
                "window": window_name,
                "threshold": threshold_label,
                "n_trades": metrics["trades"],
                "trades_per_day": metrics["trades_per_day"],
                "win_rate": metrics["win_rate"],
                "total_pnl": metrics["total_pnl_usd"],
                "final_bankroll": metrics["final_bankroll_usd"],
                "max_drawdown": metrics["max_drawdown_usd"],
                "avg_computed_edge": metrics["avg_computed_edge"],
                "avg_realized_edge": metrics["avg_realized_edge"],
                "n_city_days_skipped": metrics.get("n_city_days_skipped", 0),
            }
            rows.append(row)
            print(
                f"{window_name}: trades={metrics['trades']} ({metrics['trades_per_day']:.2f}/day) "
                f"win={100*metrics['win_rate']:.1f}% final=${metrics['final_bankroll_usd']:.2f} "
                f"real_edge={metrics['avg_realized_edge']:.3f} skipped={metrics.get('n_city_days_skipped', 0)}"
            )

            if threshold is None and window_name == "late":
                late = metrics
                if (
                    late["trades"] != BASELINE_LATE_CURRENT["trades"]
                    or abs(late["win_rate"] - BASELINE_LATE_CURRENT["win_rate"]) > 0.002
                    or abs(late["final_bankroll_usd"] - BASELINE_LATE_CURRENT["final_bankroll_usd"]) > 0.05
                ):
                    raise SystemExit(
                        "BASELINE MISMATCH for threshold=null late/current: "
                        f"got trades={late['trades']} win={late['win_rate']:.3f} "
                        f"final=${late['final_bankroll_usd']:.2f}; expected "
                        f"{BASELINE_LATE_CURRENT['trades']} trades, "
                        f"{BASELINE_LATE_CURRENT['win_rate']:.1%} win, "
                        f"${BASELINE_LATE_CURRENT['final_bankroll_usd']:.2f} final. "
                        "Driver configuration does not reproduce the existing report."
                    )
                baseline_late_pnl = city_pnl_by_trades(scenario["trades"])

    sweep_df = pd.DataFrame(rows)
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(SWEEP_CSV, index=False)
    print(f"\nWrote sweep results to {SWEEP_CSV}")

    print("\n=== SWEEP TABLE ===")
    print(
        sweep_df.to_string(
            index=False,
            formatters={
                "trades_per_day": lambda v: f"{v:.2f}",
                "win_rate": lambda v: f"{100*v:.1f}%",
                "total_pnl": lambda v: f"{v:+.2f}",
                "final_bankroll": lambda v: f"{v:.2f}",
                "max_drawdown": lambda v: f"{v:.2f}",
                "avg_computed_edge": lambda v: f"{v:.3f}",
                "avg_realized_edge": lambda v: f"{v:.3f}",
            },
        )
    )

    if baseline_late_pnl is not None:
        print("\nLate-window per-city PnL delta vs null baseline:")
        for threshold in SWEEP_THRESHOLDS:
            if threshold is None:
                continue
            threshold_label = f"{threshold:.1f}"
            cfg = apply_convective_threshold(config, str(threshold))
            late_scenario = run_scenario(
                variant_name,
                variant_config,
                "late",
                windows["late"],
                eligible[eligible["date"].isin(windows["late"])].copy(),
                models,
                cfg,
                wu,
                wu_bias,
            )
            late_pnl = city_pnl_by_trades(late_scenario["trades"])
            print(f"  threshold={threshold_label}:")
            for city in LATE_PNL_CITIES:
                delta = late_pnl.get(city, 0.0) - baseline_late_pnl.get(city, 0.0)
                print(f"    {city}: {delta:+.2f}")

    null_late_edge = float(
        sweep_df[(sweep_df["threshold"] == "null") & (sweep_df["window"] == "late")]["avg_realized_edge"].iloc[0]
    )
    adopted: float | None = None
    adopted_late_edge = null_late_edge
    for threshold in sorted([t for t in SWEEP_THRESHOLDS if t is not None], reverse=True):
        threshold_label = f"{threshold:.1f}"
        sub = sweep_df[sweep_df["threshold"] == threshold_label]
        if sub.empty:
            continue
        late_edge = float(sub[sub["window"] == "late"]["avg_realized_edge"].iloc[0])
        min_tpd = float(sub["trades_per_day"].min())
        if late_edge > null_late_edge and min_tpd >= 1.2:
            adopted = threshold
            adopted_late_edge = late_edge
            break

    print("\n=== DECISION RULE ===")
    if adopted is not None:
        print(
            f"Adopt threshold {adopted:.1f}: improves late-window realized edge "
            f"({null_late_edge:.3f} -> {adopted_late_edge:.3f}) with trades/day >= 1.2 in all windows."
        )
    else:
        print(
            "No threshold meets the decision rule (late realized edge improvement + trades/day >= 1.2 "
            "across all windows). Do not auto-apply."
        )


JOINT_VALIDATION_CSV = PROJECT_ROOT / "data" / "analysis" / "joint_validation.csv"
JOINT_MIN_BANKROLL = 80.0
JOINT_MIN_TRADES_PER_DAY = 1.30
JOINT_TIE_EDGE = 0.003

JOINT_CONFIGS: dict[str, dict[str, Any]] = {
    "A": {"convective_cloud_skip_threshold": 0.6, "excluded_cities": []},
    "B": {"convective_cloud_skip_threshold": 0.5, "excluded_cities": []},
    "C": {
        "convective_cloud_skip_threshold": 0.6,
        "excluded_cities": ["los_angeles", "san_francisco"],
    },
    "D": {
        "convective_cloud_skip_threshold": 0.5,
        "excluded_cities": ["los_angeles", "san_francisco"],
    },
}


def build_baseline_config() -> dict[str, Any]:
    """OLD config for harness sanity anchor (lambda 0.6, no convective filter)."""
    cfg = bc.load_trading_config()
    cfg["shrinkage_lambda"] = 0.6
    cfg["shrinkage_lambda_summer"] = None
    cfg["convective_cloud_skip_threshold"] = None
    cfg["two_piece_sigma_down_ratio"] = None
    cfg["n_contracts_default"] = 5
    cfg["n_contracts_reduced"] = 5
    cfg["budget_divisor"] = 5
    cfg["budget_floor"] = 70.0
    cfg["budget_cap_bankroll"] = 100.0
    cfg["edge_threshold"] = 0.037
    cfg["max_trades_per_day"] = 2
    cfg["rolling_bias_halflife_days"] = 20
    cfg["max_rolling_correction_f"] = 1.5
    cfg["basket_boundary_margin_f"] = 0.0
    return cfg


def build_joint_config(config_id: str) -> dict[str, Any]:
    spec = JOINT_CONFIGS[config_id]
    cfg = build_baseline_config()
    cfg["shrinkage_lambda"] = 0.35
    cfg["convective_cloud_skip_threshold"] = spec["convective_cloud_skip_threshold"]
    return cfg


def filter_eligible_cities(eligible: pd.DataFrame, excluded: list[str]) -> pd.DataFrame:
    if not excluded:
        return eligible
    return eligible[~eligible["city"].isin(excluded)].copy()


def joint_validation_row(
    config_id: str,
    window_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    spec = JOINT_CONFIGS[config_id]
    threshold = spec["convective_cloud_skip_threshold"]
    excluded = spec["excluded_cities"]
    computed = float(metrics["avg_computed_edge"])
    realized = float(metrics["avg_realized_edge"])
    return {
        "config_id": config_id,
        "window": window_name,
        "threshold": threshold,
        "cities_excluded": ",".join(excluded) if excluded else "",
        "n_trades": metrics["trades"],
        "trades_per_day": metrics["trades_per_day"],
        "win_rate": metrics["win_rate"],
        "mean_computed_edge": computed,
        "mean_realized_edge": realized,
        "cal_gap": computed - realized,
        "total_pnl": metrics["total_pnl_usd"],
        "final_bankroll": metrics["final_bankroll_usd"],
        "min_bankroll": metrics["min_bankroll_usd"],
        "max_drawdown": metrics["max_drawdown_usd"],
        "n_budget_dropped": metrics.get("n_budget_dropped", 0),
        "n_convective_skipped": metrics.get("n_city_days_skipped", 0),
    }


def run_baseline_sanity_anchor(
    windows: dict[str, list[str]],
    eligible: pd.DataFrame,
    models: NgBoostBacktestModels,
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    """Reproduce published late-window baseline before the joint-validation grid."""
    variant_name = "current"
    variant_config = VARIANTS[variant_name]
    window_name = "late"
    dates = windows[window_name]
    config = build_baseline_config()

    daily_cap = bc.daily_budget_ngboost(STARTING_BANKROLL_USD, config)
    print("=== HARNESS SANITY ANCHOR (OLD config, late window) ===")
    print(
        f"Config: lambda=0.6, convective=null, two_piece=null, flat 5, "
        f"divisor=5, start=${STARTING_BANKROLL_USD:.2f}"
    )
    print(
        f"Daily cap at start bankroll: (${STARTING_BANKROLL_USD:.2f} - 70) / 5 = "
        f"${daily_cap:.2f}"
    )
    print(
        "Budget trim is whole-trade: a 5-contract trade at entry c risks 5c dollars; "
        "two trades fit only if c1 + c2 <= 0.60 at the $3.00 cap."
    )

    scenario = run_scenario(
        variant_name,
        variant_config,
        window_name,
        dates,
        eligible[eligible["date"].isin(dates)].copy(),
        models,
        config,
        wu,
        wu_bias,
    )
    metrics = scenario["metrics"]
    print(
        f"Result: trades={metrics['trades']} win={100*metrics['win_rate']:.1f}% "
        f"final=${metrics['final_bankroll_usd']:.2f} "
        f"budget_dropped={metrics.get('n_budget_dropped', 0)}"
    )

    trades_match = metrics["trades"] == BASELINE_LATE_CURRENT["trades"]
    win_match = abs(metrics["win_rate"] - BASELINE_LATE_CURRENT["win_rate"]) <= 0.002
    final_match = (
        abs(metrics["final_bankroll_usd"] - BASELINE_LATE_CURRENT["final_bankroll_usd"]) <= 0.05
    )
    if not (trades_match and win_match and final_match):
        print("\n*** BASELINE DELTA (3-below-$85 rule removed; flat 5 unconditional) ***")
        print(
            f"  Expected: {BASELINE_LATE_CURRENT['trades']} trades, "
            f"{100*BASELINE_LATE_CURRENT['win_rate']:.1f}% win, "
            f"${BASELINE_LATE_CURRENT['final_bankroll_usd']:.2f} final"
        )
        print(
            f"  Got:      {metrics['trades']} trades, "
            f"{100*metrics['win_rate']:.1f}% win, "
            f"${metrics['final_bankroll_usd']:.2f} final"
        )
        if metrics["trades"] != BASELINE_LATE_CURRENT["trades"]:
            delta_trades = metrics["trades"] - BASELINE_LATE_CURRENT["trades"]
            print(f"  Trade count delta: {delta_trades:+d}")
        if abs(metrics["final_bankroll_usd"] - BASELINE_LATE_CURRENT["final_bankroll_usd"]) > 0.05:
            delta_final = metrics["final_bankroll_usd"] - BASELINE_LATE_CURRENT["final_bankroll_usd"]
            print(f"  Final bankroll delta: ${delta_final:+.2f}")
        print("Proceeding with explicit delta noted (published report may have used 3-below-$85).")
    else:
        print("Baseline reproduced exactly.")

    return metrics


def apply_joint_decision_rule(rows: list[dict[str, Any]]) -> str | None:
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config_id"]), []).append(row)

    survivors: list[str] = []
    for config_id, config_rows in by_config.items():
        failed = False
        for row in config_rows:
            if float(row["min_bankroll"]) < JOINT_MIN_BANKROLL:
                failed = True
                break
            if float(row["trades_per_day"]) < JOINT_MIN_TRADES_PER_DAY:
                failed = True
                break
        if not failed:
            survivors.append(config_id)

    if not survivors:
        return None

    late_edges: dict[str, float] = {}
    for config_id in survivors:
        late_row = next(r for r in by_config[config_id] if r["window"] == "late")
        late_edges[config_id] = float(late_row["mean_realized_edge"])

    best_edge = max(late_edges.values())
    tier = [
        cid
        for cid in survivors
        if late_edges[cid] >= best_edge - JOINT_TIE_EDGE
    ]

    def sort_key(config_id: str) -> tuple[float, int]:
        threshold = float(JOINT_CONFIGS[config_id]["convective_cloud_skip_threshold"])
        n_excluded = len(JOINT_CONFIGS[config_id]["excluded_cities"])
        return (-threshold, n_excluded)

    tier.sort(key=sort_key)
    return tier[0]


def frozen_deploy_json(config_id: str) -> dict[str, Any]:
    spec = JOINT_CONFIGS[config_id]
    return {
        "signal": "ngboost",
        "sizer": "flat_5",
        "selection": "edge_threshold",
        "edge_threshold": 0.037,
        "n_contracts_default": 5,
        "shrinkage_lambda": 0.35,
        "shrinkage_lambda_summer": None,
        "convective_cloud_skip_threshold": spec["convective_cloud_skip_threshold"],
        "convective_skip_months": [6, 7, 8],
        "convective_skip_exempt_cities": ["san_francisco", "los_angeles", "seattle"],
        "two_piece_sigma_down_ratio": None,
        "budget_floor": 70.0,
        "budget_divisor": 5,
        "budget_cap_bankroll": 100.0,
        "max_trades_per_day": 2,
        "rolling_bias_halflife_days": 20,
        "max_rolling_correction_f": 1.5,
        "basket_boundary_margin_f": 0.0,
        "cities": [
            city
            for city in bc.load_trading_config().get("cities", [])
            if city not in spec["excluded_cities"]
        ],
    }


def run_joint_validation(
    windows: dict[str, list[str]],
    eligible: pd.DataFrame,
    models: NgBoostBacktestModels,
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
) -> None:
    variant_name = "current"
    variant_config = VARIANTS[variant_name]

    run_baseline_sanity_anchor(windows, eligible, models, wu, wu_bias)

    print("\n=== JOINT VALIDATION GRID (lambda=0.35 global, flat 5, divisor 5) ===")
    rows: list[dict[str, Any]] = []
    for config_id in sorted(JOINT_CONFIGS):
        config = build_joint_config(config_id)
        excluded = JOINT_CONFIGS[config_id]["excluded_cities"]
        threshold = JOINT_CONFIGS[config_id]["convective_cloud_skip_threshold"]
        print(f"\n--- Config {config_id}: threshold={threshold}, excluded={excluded or 'none'} ---")
        for window_name, dates in windows.items():
            window_eligible = filter_eligible_cities(
                eligible[eligible["date"].isin(dates)].copy(),
                excluded,
            )
            scenario = run_scenario(
                variant_name,
                variant_config,
                window_name,
                dates,
                window_eligible,
                models,
                config,
                wu,
                wu_bias,
            )
            row = joint_validation_row(config_id, window_name, scenario["metrics"])
            rows.append(row)
            print(
                f"  {window_name}: trades={row['n_trades']} ({row['trades_per_day']:.2f}/day) "
                f"win={100*row['win_rate']:.1f}% real_edge={row['mean_realized_edge']:.3f} "
                f"min_br=${row['min_bankroll']:.2f} budget_dropped={row['n_budget_dropped']}"
            )

    df = pd.DataFrame(rows)
    JOINT_VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(JOINT_VALIDATION_CSV, index=False)

    print("\n=== JOINT VALIDATION TABLE ===")
    display = df.copy()
    display["win_rate"] = display["win_rate"].map(lambda v: f"{100*v:.1f}%")
    for col in (
        "trades_per_day",
        "mean_computed_edge",
        "mean_realized_edge",
        "cal_gap",
        "total_pnl",
        "final_bankroll",
        "min_bankroll",
        "max_drawdown",
    ):
        display[col] = display[col].map(lambda v: f"{v:.3f}" if col != "total_pnl" else f"{v:+.2f}")
    print(display.to_string(index=False))
    print(f"\nWrote {JOINT_VALIDATION_CSV}")

    print("\n=== PER-CONFIG WORST WINDOW ===")
    for config_id in sorted(JOINT_CONFIGS):
        sub = df[df["config_id"] == config_id]
        worst_min = sub.loc[sub["min_bankroll"].idxmin()]
        worst_tpd = sub.loc[sub["trades_per_day"].idxmin()]
        print(
            f"  {config_id}: worst min_bankroll=${worst_min['min_bankroll']:.2f} "
            f"({worst_min['window']}); worst trades/day={worst_tpd['trades_per_day']:.2f} "
            f"({worst_tpd['window']})"
        )

    winner = apply_joint_decision_rule(rows)
    print("\n=== PRE-REGISTERED DECISION RULE ===")
    print(
        "Step 1 eliminate: min_bankroll < $80.00 in ANY window OR trades_per_day < 1.30 in ANY window"
    )
    print("Step 2 among survivors: rank by LATE-window mean_realized_edge, descending")
    print("Step 3 tie-break (within 0.003): higher threshold, then fewer excluded cities")
    print(
        "Trade-count note: 57 more trades in 49 days = 1.16/day minimum; "
        "1.30/day gate includes margin for live no-signal days and fill failures."
    )

    if winner is None:
        print("\nALL CONFIGS ELIMINATED in Step 1. No frozen deployment config selected.")
        return

    deploy = frozen_deploy_json(winner)
    print(f"\nFROZEN DEPLOYMENT CONFIG: {winner}")
    print(json.dumps(deploy, indent=2))


PACE_AMENDMENT_CSV = PROJECT_ROOT / "data" / "analysis" / "pace_amendment.csv"
PACE_MIN_BANKROLL = 80.0
PACE_MIN_TRADES_PER_DAY = 1.35
PACE_TIE_EDGE = 0.003
PACE_TIE_ORDER = {"D3": 0, "D1": 1, "D4": 2, "D2": 3}
CONFIG_D_EXCLUDED = ["los_angeles", "san_francisco"]

D0_LATE_REFERENCE = {
    "trades": 57,
    "win_rate": 0.211,
    "final_bankroll_usd": 95.22,
}

PACE_VARIANTS: dict[str, dict[str, Any]] = {
    "D0": {
        "edge_threshold": 0.037,
        "max_trades_per_day": 2,
        "pace_fallback_enabled": False,
    },
    "D1": {
        "edge_threshold": 0.030,
        "max_trades_per_day": 2,
        "pace_fallback_enabled": False,
    },
    "D2": {
        "edge_threshold": 0.025,
        "max_trades_per_day": 2,
        "pace_fallback_enabled": False,
    },
    "D3": {
        "edge_threshold": 0.037,
        "max_trades_per_day": 2,
        "pace_fallback_enabled": True,
        "pace_fallback_min_edge": 0.010,
        "pace_fallback_target_trades": 2,
    },
    "D4": {
        "edge_threshold": 0.037,
        "max_trades_per_day": 3,
        "pace_fallback_enabled": False,
    },
}


def build_config_d_base() -> dict[str, Any]:
    cfg = bc.load_trading_config()
    cfg["shrinkage_lambda"] = 0.35
    cfg["shrinkage_lambda_summer"] = None
    cfg["convective_cloud_skip_threshold"] = 0.5
    cfg["two_piece_sigma_down_ratio"] = None
    cfg["n_contracts_default"] = 5
    cfg["n_contracts_reduced"] = 5
    cfg["budget_divisor"] = 5
    cfg["budget_floor"] = 70.0
    cfg["budget_cap_bankroll"] = 100.0
    cfg["rolling_bias_halflife_days"] = 20
    cfg["max_rolling_correction_f"] = 1.5
    cfg["basket_boundary_margin_f"] = 0.0
    cfg["edge_threshold"] = 0.037
    cfg["max_trades_per_day"] = 2
    cfg["pace_fallback_enabled"] = False
    return cfg


def apply_pace_variant(config: dict[str, Any], variant_id: str) -> dict[str, Any]:
    spec = PACE_VARIANTS[variant_id]
    cfg = dict(config)
    cfg["edge_threshold"] = spec["edge_threshold"]
    cfg["max_trades_per_day"] = spec["max_trades_per_day"]
    cfg["pace_fallback_enabled"] = spec.get("pace_fallback_enabled", False)
    if cfg["pace_fallback_enabled"]:
        cfg["pace_fallback_min_edge"] = spec["pace_fallback_min_edge"]
        cfg["pace_fallback_target_trades"] = spec["pace_fallback_target_trades"]
    else:
        cfg.pop("pace_fallback_min_edge", None)
        cfg.pop("pace_fallback_target_trades", None)
    return cfg


def pace_fallback_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    traded = [t for t in trades if t.get("traded")]
    fallback = [t for t in traded if t.get("entry_mode") == "fallback"]
    primary = [t for t in traded if t.get("entry_mode", "primary") == "primary"]

    def _realized_edge(trade: dict[str, Any]) -> float:
        return (1.0 if trade.get("won") else 0.0) - float(trade.get("entry_price", 0.0))

    fb_wins = sum(1 for t in fallback if t.get("won"))
    return {
        "n_fallback_trades": len(fallback),
        "fallback_win_rate": fb_wins / len(fallback) if fallback else float("nan"),
        "fallback_mean_realized_edge": float(np.mean([_realized_edge(t) for t in fallback]))
        if fallback
        else float("nan"),
        "primary_mean_realized_edge": float(np.mean([_realized_edge(t) for t in primary]))
        if primary
        else float("nan"),
    }


def pace_amendment_row(
    variant_id: str,
    window_name: str,
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    computed = float(metrics["avg_computed_edge"])
    realized = float(metrics["avg_realized_edge"])
    row: dict[str, Any] = {
        "variant": variant_id,
        "window": window_name,
        "n_trades": metrics["trades"],
        "trades_per_day": metrics["trades_per_day"],
        "win_rate": metrics["win_rate"],
        "mean_computed_edge": computed,
        "mean_realized_edge": realized,
        "cal_gap": computed - realized,
        "total_pnl": metrics["total_pnl_usd"],
        "final_bankroll": metrics["final_bankroll_usd"],
        "min_bankroll": metrics["min_bankroll_usd"],
        "max_drawdown": metrics["max_drawdown_usd"],
        "n_budget_dropped": metrics.get("n_budget_dropped", 0),
        "n_convective_skipped": metrics.get("n_city_days_skipped", 0),
    }
    if variant_id == "D3":
        row.update(pace_fallback_metrics(trades))
    return row


def pace_step1_eliminated(rows: list[dict[str, Any]], variant_id: str) -> tuple[bool, list[str]]:
    sub = [r for r in rows if r["variant"] == variant_id]
    reasons: list[str] = []
    for row in sub:
        if float(row["min_bankroll"]) < PACE_MIN_BANKROLL:
            reasons.append(f"{row['window']} min_bankroll=${float(row['min_bankroll']):.2f}")
        if float(row["trades_per_day"]) < PACE_MIN_TRADES_PER_DAY:
            reasons.append(f"{row['window']} trades/day={float(row['trades_per_day']):.2f}")
    return bool(reasons), reasons


def pace_fails_only_pace_gate(rows: list[dict[str, Any]], variant_id: str) -> bool:
    sub = [r for r in rows if r["variant"] == variant_id]
    bankroll_ok = all(float(r["min_bankroll"]) >= PACE_MIN_BANKROLL for r in sub)
    pace_fail = any(float(r["trades_per_day"]) < PACE_MIN_TRADES_PER_DAY for r in sub)
    return bankroll_ok and pace_fail


def apply_pace_decision_rule(rows: list[dict[str, Any]]) -> str | None:
    survivors: list[str] = []
    for variant_id in PACE_VARIANTS:
        eliminated, _reasons = pace_step1_eliminated(rows, variant_id)
        if not eliminated:
            survivors.append(variant_id)

    if not survivors:
        return None

    late_edges = {
        variant_id: float(
            next(r for r in rows if r["variant"] == variant_id and r["window"] == "late")[
                "mean_realized_edge"
            ]
        )
        for variant_id in survivors
    }
    best_edge = max(late_edges.values())
    tier = [
        variant_id
        for variant_id in survivors
        if late_edges[variant_id] >= best_edge - PACE_TIE_EDGE
    ]
    tier.sort(key=lambda variant_id: PACE_TIE_ORDER.get(variant_id, 99))
    return tier[0]


def frozen_pace_deploy_json(variant_id: str) -> dict[str, Any]:
    base = build_config_d_base()
    spec = PACE_VARIANTS[variant_id]
    deploy: dict[str, Any] = {
        "signal": "ngboost",
        "sizer": "flat_5",
        "selection": "edge_threshold",
        "edge_threshold": spec["edge_threshold"],
        "n_contracts_default": 5,
        "shrinkage_lambda": 0.35,
        "shrinkage_lambda_summer": None,
        "convective_cloud_skip_threshold": 0.5,
        "convective_skip_months": [6, 7, 8],
        "convective_skip_exempt_cities": ["san_francisco", "los_angeles", "seattle"],
        "two_piece_sigma_down_ratio": None,
        "budget_floor": 70.0,
        "budget_divisor": 5,
        "budget_cap_bankroll": 100.0,
        "max_trades_per_day": spec["max_trades_per_day"],
        "rolling_bias_halflife_days": 20,
        "max_rolling_correction_f": 1.5,
        "basket_boundary_margin_f": 0.0,
        "price_floor": base.get("price_floor", 0.15),
        "cities": [
            city for city in base.get("cities", []) if city not in CONFIG_D_EXCLUDED
        ],
    }
    if spec.get("pace_fallback_enabled"):
        deploy["pace_fallback_enabled"] = True
        deploy["pace_fallback_min_edge"] = spec["pace_fallback_min_edge"]
        deploy["pace_fallback_target_trades"] = spec["pace_fallback_target_trades"]
    else:
        deploy["pace_fallback_enabled"] = False
    return deploy


def pace_config_diff(variant_id: str) -> dict[str, Any]:
    baseline = frozen_pace_deploy_json("D0")
    winner = frozen_pace_deploy_json(variant_id)
    diff: dict[str, Any] = {}
    for key, value in winner.items():
        if baseline.get(key) != value:
            diff[key] = value
    return diff


def run_pace_amendment(
    windows: dict[str, list[str]],
    eligible: pd.DataFrame,
    models: NgBoostBacktestModels,
    wu: pd.DataFrame,
    wu_bias: dict[str, dict[str, float | int]],
) -> None:
    variant_config = VARIANTS["current"]
    base_config = build_config_d_base()
    filtered_eligible = filter_eligible_cities(eligible, CONFIG_D_EXCLUDED)

    print("=== PACE AMENDMENT (Config D base, 5 variants x 3 windows) ===")
    print(
        f"Base: lambda=0.35, convective=0.5, exclude {CONFIG_D_EXCLUDED}, "
        f"flat 5, divisor 5, start=${STARTING_BANKROLL_USD:.2f}"
    )

    # D0 reference anchor on late window
    d0_config = apply_pace_variant(base_config, "D0")
    late_dates = windows["late"]
    d0_late = run_scenario(
        "D0",
        variant_config,
        "late",
        late_dates,
        filtered_eligible[filtered_eligible["date"].isin(late_dates)].copy(),
        models,
        d0_config,
        wu,
        wu_bias,
    )
    late_metrics = d0_late["metrics"]
    print("\n=== D0 REFERENCE ANCHOR (late window) ===")
    print(
        f"Result: trades={late_metrics['trades']} win={100*late_metrics['win_rate']:.1f}% "
        f"final=${late_metrics['final_bankroll_usd']:.2f}"
    )
    trades_match = late_metrics["trades"] == D0_LATE_REFERENCE["trades"]
    win_match = abs(late_metrics["win_rate"] - D0_LATE_REFERENCE["win_rate"]) <= 0.002
    final_match = (
        abs(late_metrics["final_bankroll_usd"] - D0_LATE_REFERENCE["final_bankroll_usd"]) <= 0.05
    )
    if not (trades_match and win_match and final_match):
        raise SystemExit(
            "D0 REFERENCE MISMATCH: expected "
            f"{D0_LATE_REFERENCE['trades']} trades, "
            f"{100*D0_LATE_REFERENCE['win_rate']:.1f}% win, "
            f"${D0_LATE_REFERENCE['final_bankroll_usd']:.2f} final; got "
            f"{late_metrics['trades']} trades, "
            f"{100*late_metrics['win_rate']:.1f}% win, "
            f"${late_metrics['final_bankroll_usd']:.2f} final. STOP."
        )
    print("D0 reference reproduced exactly.")

    rows: list[dict[str, Any]] = []

    for variant_id in PACE_VARIANTS:
        cfg = apply_pace_variant(base_config, variant_id)
        spec = PACE_VARIANTS[variant_id]
        print(
            f"\n--- {variant_id}: E*={spec['edge_threshold']}, "
            f"max/day={spec['max_trades_per_day']}, "
            f"fallback={spec.get('pace_fallback_enabled', False)} ---"
        )
        for window_name, dates in windows.items():
            if variant_id == "D0" and window_name == "late":
                scenario = d0_late
            else:
                scenario = run_scenario(
                    variant_id,
                    variant_config,
                    window_name,
                    dates,
                    filtered_eligible[filtered_eligible["date"].isin(dates)].copy(),
                    models,
                    cfg,
                    wu,
                    wu_bias,
                )
            row = pace_amendment_row(
                variant_id, window_name, scenario["metrics"], scenario["trades"]
            )
            rows.append(row)
            print(
                f"  {window_name}: trades={row['n_trades']} ({row['trades_per_day']:.2f}/day) "
                f"win={100*row['win_rate']:.1f}% real_edge={row['mean_realized_edge']:.3f} "
                f"min_br=${row['min_bankroll']:.2f} budget_dropped={row['n_budget_dropped']}"
            )
            if variant_id == "D3":
                fb_n = int(row.get("n_fallback_trades", 0))
                fb_wr = row.get("fallback_win_rate", float("nan"))
                fb_re = row.get("fallback_mean_realized_edge", float("nan"))
                pri_re = row.get("primary_mean_realized_edge", float("nan"))
                fb_wr_s = f"{100*fb_wr:.1f}%" if fb_n and math.isfinite(fb_wr) else "n/a"
                fb_re_s = f"{fb_re:.3f}" if fb_n and math.isfinite(fb_re) else "n/a"
                print(
                    f"    D3 fallback: n={fb_n} win={fb_wr_s} real_edge={fb_re_s} "
                    f"(primary real_edge={pri_re:.3f})"
                )

    df = pd.DataFrame(rows)
    PACE_AMENDMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PACE_AMENDMENT_CSV, index=False)

    print("\n=== PACE AMENDMENT TABLE ===")
    display_cols = [
        "variant",
        "window",
        "n_trades",
        "trades_per_day",
        "win_rate",
        "mean_computed_edge",
        "mean_realized_edge",
        "cal_gap",
        "total_pnl",
        "final_bankroll",
        "min_bankroll",
        "max_drawdown",
        "n_budget_dropped",
        "n_convective_skipped",
    ]
    display = df[display_cols].copy()
    display["win_rate"] = display["win_rate"].map(lambda v: f"{100*v:.1f}%")
    for col in (
        "trades_per_day",
        "mean_computed_edge",
        "mean_realized_edge",
        "cal_gap",
        "final_bankroll",
        "min_bankroll",
        "max_drawdown",
    ):
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    display["total_pnl"] = df["total_pnl"].map(lambda v: f"{v:+.2f}")
    print(display.to_string(index=False))
    print(f"\nWrote {PACE_AMENDMENT_CSV}")

    if "D3" in df["variant"].values:
        print("\n=== D3 FALLBACK BREAKDOWN (per window) ===")
        d3 = df[df["variant"] == "D3"]
        for _, row in d3.iterrows():
            fb_wr = row.get("fallback_win_rate", float("nan"))
            fb_re = row.get("fallback_mean_realized_edge", float("nan"))
            pri_re = row.get("primary_mean_realized_edge", float("nan"))
            print(
                f"  {row['window']}: n_fallback={int(row.get('n_fallback_trades', 0))} "
                f"fallback_win={100*fb_wr:.1f}% fallback_real={fb_re:.3f} "
                f"primary_real={pri_re:.3f}"
            )

    print("\n=== PER-VARIANT WORST WINDOW ===")
    for variant_id in PACE_VARIANTS:
        sub = df[df["variant"] == variant_id]
        worst_min = sub.loc[sub["min_bankroll"].idxmin()]
        worst_tpd = sub.loc[sub["trades_per_day"].idxmin()]
        print(
            f"  {variant_id}: worst min_bankroll=${worst_min['min_bankroll']:.2f} "
            f"({worst_min['window']}); worst trades/day={worst_tpd['trades_per_day']:.2f} "
            f"({worst_tpd['window']})"
        )

    winner = apply_pace_decision_rule(rows)
    print("\n=== PRE-REGISTERED DECISION RULE ===")
    print(
        "Step 1 eliminate: min_bankroll < $80.00 in ANY window OR trades_per_day < 1.35 in ANY window."
    )
    print(
        "Gate raised from 1.30 to 1.35: deployment is into Jul-Aug, deeper convective season "
        "than the May-Jun late window, so live skip rates will exceed the backtest's."
    )
    print("Step 2 among survivors: rank by LATE-window mean_realized_edge, descending")
    print("Step 3 tie-break (within 0.003): prefer D3, then D1, then D4, then D2")

    if winner is not None:
        deploy = frozen_pace_deploy_json(winner)
        print(f"\nFROZEN DEPLOYMENT CONFIG: {winner}")
        print(json.dumps(deploy, indent=2))
        diff = pace_config_diff(winner)
        if diff:
            print("\n=== CONFIG DIFF vs D0 reference ===")
            print(json.dumps(diff, indent=2))
        if winner == "D3":
            print(
                "\nLIVE WIRING: pace fallback implemented in src/poly_trading_pipeline.py "
                "(select_trades_poly); entry_mode logged per trade in auto_trader_poly.py."
            )
        else:
            print("\nLIVE WIRING: config-only change (update config/deploy_config.json).")
        return

    print("\nALL VARIANTS ELIMINATED in Step 1.")
    pace_only: list[tuple[str, float]] = []
    for variant_id in PACE_VARIANTS:
        eliminated, reasons = pace_step1_eliminated(rows, variant_id)
        if not eliminated:
            continue
        if pace_fails_only_pace_gate(rows, variant_id):
            late_edge = float(
                next(r for r in rows if r["variant"] == variant_id and r["window"] == "late")[
                    "mean_realized_edge"
                ]
            )
            pace_only.append((variant_id, late_edge))
    if pace_only:
        best_variant, best_edge = max(pace_only, key=lambda item: item[1])
        deploy = frozen_pace_deploy_json(best_variant)
        print(
            f"\nMANUAL DECISION REQUIRED: pace cannot be satisfied by strategy knobs alone; "
            f"consider operational backstop (manual trade on alerted zero-trade days) "
            f"with this config."
        )
        print(f"Best among pace-only failures: {best_variant} (late real_edge={best_edge:.3f})")
        print(json.dumps(deploy, indent=2))
    else:
        print("No variant fails only the pace gate; min_bankroll violations present. Do not relax.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 49-day survival scenarios for v5b NGBoost")
    parser.add_argument("--force", action="store_true", help="Overwrite existing survival outputs")
    parser.add_argument("--output-tag", default="", help="Output suffix, e.g. survival_v2")
    parser.add_argument(
        "--convective-threshold",
        default="config",
        help="Override convective_cloud_skip_threshold; use 'null' to disable",
    )
    parser.add_argument(
        "--convective-sweep",
        action="store_true",
        help="Run threshold sweep {null,0.6,0.5,0.4} for current variant across all windows",
    )
    parser.add_argument(
        "--two-piece-eval",
        action="store_true",
        help="Compare late-window survival with two_piece_sigma_down_ratio null vs --two-piece-ratio",
    )
    parser.add_argument(
        "--two-piece-ratio",
        default=None,
        help="Pooled summer r_hat for --two-piece-eval; loads from calibration CSV if omitted",
    )
    parser.add_argument(
        "--joint-validation",
        action="store_true",
        help="Run 4-config x 3-window joint validation with pre-registered decision rule",
    )
    parser.add_argument(
        "--pace-amendment",
        action="store_true",
        help="Run Config-D pace amendment variants with pre-registered decision rule",
    )
    args = parser.parse_args()

    check_prerequisites()
    if args.pace_amendment:
        load_runtime_dependencies()
        eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
        eligible["date"] = eligible["date"].astype(str)
        eligible["city"] = eligible["city"].astype(str)
        all_dates = sorted(eligible["date"].unique().tolist())
        windows = build_windows(all_dates)
        models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
        wu = bc.load_wu_targets()
        wu_bias = load_wunderground_bias()
        run_pace_amendment(windows, eligible, models, wu, wu_bias)
        return
    if args.joint_validation:
        load_runtime_dependencies()
        eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
        eligible["date"] = eligible["date"].astype(str)
        eligible["city"] = eligible["city"].astype(str)
        all_dates = sorted(eligible["date"].unique().tolist())
        windows = build_windows(all_dates)
        models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
        wu = bc.load_wu_targets()
        wu_bias = load_wunderground_bias()
        run_joint_validation(windows, eligible, models, wu, wu_bias)
        return
    if args.two_piece_eval:
        load_runtime_dependencies()
        eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
        eligible["date"] = eligible["date"].astype(str)
        eligible["city"] = eligible["city"].astype(str)
        all_dates = sorted(eligible["date"].unique().tolist())
        windows = build_windows(all_dates)
        config = apply_convective_threshold(bc.load_trading_config(), args.convective_threshold)
        models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
        wu = bc.load_wu_targets()
        wu_bias = load_wunderground_bias()
        ratio_hat = args.two_piece_ratio
        if ratio_hat is None:
            cal_csv = PROJECT_ROOT / "data" / "analysis" / "two_piece_calibration.csv"
            if not cal_csv.exists():
                raise SystemExit(
                    f"Missing {cal_csv}; run scripts/analysis/calibrate_two_piece.py first "
                    "or pass --two-piece-ratio"
                )
            cal_df = pd.read_csv(cal_csv)
            pooled = cal_df[
                (cal_df["subset"] == "summer_2025") & (cal_df["metric"] == "r_hat_pooled")
            ]
            if pooled.empty:
                raise SystemExit("Calibration CSV missing summer_2025 r_hat_pooled row")
            ratio_hat = float(pooled["value"].iloc[0])
        run_two_piece_eval(windows, eligible, config, models, wu, wu_bias, float(ratio_hat))
        return
    if args.convective_sweep:
        load_runtime_dependencies()
        eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
        eligible["date"] = eligible["date"].astype(str)
        eligible["city"] = eligible["city"].astype(str)
        all_dates = sorted(eligible["date"].unique().tolist())
        windows = build_windows(all_dates)
        config = apply_convective_threshold(bc.load_trading_config(), args.convective_threshold)
        models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
        wu = bc.load_wu_targets()
        wu_bias = load_wunderground_bias()
        run_convective_sweep(windows, eligible, config, models, wu, wu_bias)
        return

    report_path, suffix = output_paths(args.output_tag)
    if report_path.exists() and not args.force:
        print(f"SKIP survival scenario: {report_path} already exists (use --force to recompute)")
        return

    load_runtime_dependencies()

    eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
    if eligible.empty:
        print("ERROR: no eligible city-dates from step1")
        sys.exit(1)
    eligible["date"] = eligible["date"].astype(str)
    eligible["city"] = eligible["city"].astype(str)
    all_dates = sorted(eligible["date"].unique().tolist())
    try:
        windows = build_windows(all_dates)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    config = apply_convective_threshold(bc.load_trading_config(), args.convective_threshold)
    models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
    wu = bc.load_wu_targets()
    wu_bias = load_wunderground_bias()

    print("=== SURVIVAL BACKTEST RESULTS ===")
    print(f"Starting bankroll: ${STARTING_BANKROLL_USD:.2f} | Elimination: ${ELIMINATION_USD:.2f}")

    t0 = time.time()
    results: dict[str, dict[str, Any]] = {}
    for window_name, dates in windows.items():
        results[window_name] = {}
        for variant_name, variant_config in VARIANTS.items():
            print(f"  running {variant_name}/{window_name}...")
            results[window_name][variant_name] = run_scenario(
                variant_name,
                variant_config,
                window_name,
                dates,
                eligible[eligible["date"].isin(dates)].copy(),
                models,
                config,
                wu,
                wu_bias,
            )
        print_window_summary(window_name, dates, results[window_name])

    print_cross_window_summary(results)

    payload = {
        "metadata": {
            "starting_bankroll_usd": STARTING_BANKROLL_USD,
            "elimination_usd": ELIMINATION_USD,
            "target_bankroll_usd": TARGET_BANKROLL_USD,
            "scenario_days": SCENARIO_DAYS,
            "n_contracts": N_CONTRACTS,
            "variants": VARIANTS,
            "runtime_seconds": round(time.time() - t0, 2),
            "eligible_dates_file": str(bc.ELIGIBLE_DATES_CSV),
            "model_path_file": str(bc.MODEL_PATH_FILE),
        },
        "windows": windows,
        "results": results,
    }
    write_outputs(report_path, suffix, payload)
    print(f"\nWrote full results to {report_path}")
    print(f"Wrote equity curves to {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
