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
    from backtest_utils import sharpe_stats as _sharpe_stats
    from poly_trading_pipeline import load_wunderground_bias as _load_wunderground_bias
    from rolling_bias import RollingBiasCache as _RollingBiasCache

    np = _np
    pd = _pd
    bc = _bc
    NgBoostBacktestModels = _NgBoostBacktestModels
    predict_bucket_probs_from_mu = _predict_bucket_probs_from_mu
    predict_mu_sigma = _predict_mu_sigma
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
    probs = predict_bucket_probs_from_mu(models, city, date_str, bucket_labels, mu_adj)
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
) -> tuple[list[Candidate], dict[str, float]]:
    candidates: list[Candidate] = []
    raw_mu_by_city: dict[str, float] = {}

    for _, row in day_rows.iterrows():
        city = str(row["city"])
        date_str = str(row["date"])
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

    return candidates, raw_mu_by_city


def select_day_trades(
    candidates: list[Candidate],
    edge_threshold: float,
    max_trades_per_day: int,
) -> list[Candidate]:
    filtered = [candidate for candidate in candidates if candidate.edge >= edge_threshold]
    filtered.sort(key=lambda c: c.edge, reverse=True)
    return filtered[:max_trades_per_day]


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

    for day_index, date_str in enumerate(dates, start=1):
        day_rows = eligible[eligible["date"] == date_str]
        bankroll_before = bankroll
        day_spent = 0.0
        day_pnl = 0.0
        day_trades = 0
        raw_mu: dict[str, float] = {}

        if not eliminated:
            budget = bc.daily_budget_ngboost(bankroll, config)
            candidates, raw_mu = collect_day_candidates(
                day_rows,
                models,
                config,
                bias_cache,
                wu_bias,
                int(variant_config["n_buckets_per_city"]),
            )
            selected = select_day_trades(
                candidates,
                edge_threshold,
                int(variant_config["max_trades_per_day"]),
            )

            for candidate in selected:
                trade_cost = N_CONTRACTS * candidate.entry_price
                if day_spent + trade_cost > budget:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 49-day survival scenarios for v5b NGBoost")
    parser.add_argument("--force", action="store_true", help="Overwrite existing survival outputs")
    parser.add_argument("--output-tag", default="", help="Output suffix, e.g. survival_v2")
    args = parser.parse_args()

    check_prerequisites()
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

    config = bc.load_trading_config()
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
