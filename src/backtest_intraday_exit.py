"""Intraday exit backtest: grid-matched entry + walk-forward exit rules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from backtest_utils import sharpe_stats  # noqa: E402
from entry_interface import filter_to_trading_window  # noqa: E402
from run_trackB_grid import (  # noqa: E402
    LOW_OOS_COVERAGE_CITIES,
    _calendar_date_keys,
    _day_group_columns,
    apply_selection,
    generate_signals,
    run_backtest,
)
from snapshot_stability import assert_no_true_holdout, stability_entry  # noqa: E402
from src.sizing import taker_fee_cents  # noqa: E402

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from dateutil.tz import gettz

    def ZoneInfo(name: str):
        tz = gettz(name)
        if tz is None:
            raise ValueError(f"Unknown timezone: {name}")
        return tz

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FORECASTS_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"
OUTPUT_RESULTS = PROJECT_ROOT / "data" / "backtest_intraday_exit_results.parquet"
OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "backtest_intraday_exit_summary.csv"
REPORT_SUMMARY = PROJECT_ROOT / "reports" / "intraday_exit_summary.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

VALID_EXIT_RULES = [
    "hold_to_settlement",
    "profit_target_model",
    "profit_target_15c",
    "trailing_stop_10c",
    "time_stop_2pm",
    "combined",
]

PALETTE = {"blue": "#4878CF", "grey": "#8A8A8A"}
INITIAL_BANKROLL_CENTS = 10_000


@dataclass
class ExitResult:
    exit_time: pd.Timestamp | None
    exit_price: float | None
    exit_rule_triggered: str
    hold_duration_minutes: float | None
    entry_fee_cents: int
    exit_fee_cents: int
    pnl_cents: float
    fell_through_to_settlement: bool


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _date_key(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _load_deploy_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_city_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_split(split_name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split parquet: {path}")
    df = pd.read_parquet(path)
    assert_no_true_holdout(df)
    return df


def _day_market(partition_df: pd.DataFrame, city: str, event_date: str) -> pd.DataFrame:
    df = partition_df.copy()
    group_cols = _day_group_columns(df)
    city_col = group_cols[0]
    df[city_col] = df[city_col].map(_city_key)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    mask = df[city_col].eq(_city_key(city)) & df["event_date"].eq(_date_key(event_date))
    day = df.loc[mask].copy()
    if day.empty:
        return day
    day["snapshot_time_local"] = pd.to_datetime(day["snapshot_time_local"])
    return day


def _entry_snapshot_time(day_df: pd.DataFrame, k: int) -> pd.Timestamp | None:
    windowed = filter_to_trading_window(day_df)
    if windowed.empty:
        return None
    stability = stability_entry(windowed, k=k)
    if stability.no_signal:
        return None
    return pd.Timestamp(stability.entry_snapshot_time)


def _snapshot_prices(
    day_df: pd.DataFrame,
    bucket_label: str,
    after: pd.Timestamp,
) -> pd.DataFrame:
    windowed = filter_to_trading_window(day_df)
    windowed = windowed.loc[pd.to_datetime(windowed["snapshot_time_local"]) > after].copy()
    if windowed.empty:
        return windowed
    held = windowed[windowed["bucket_label"].astype(str).eq(str(bucket_label))].copy()
    held = held.sort_values("snapshot_time_local")
    held["yes_mid_close"] = pd.to_numeric(held["yes_mid_close"], errors="coerce")
    held = held.loc[held["yes_mid_close"].notna() & (held["yes_mid_close"] > 0)]
    return held


def _settlement_pnl(
    n_contracts: int,
    entry_price: float,
    resolved: bool,
) -> ExitResult:
    cost_cents = int(n_contracts * entry_price * 100)
    entry_fee = taker_fee_cents(n_contracts, entry_price)
    payout = n_contracts * 100 if resolved else 0
    pnl = payout - cost_cents - entry_fee
    return ExitResult(
        exit_time=None,
        exit_price=1.0 if resolved else 0.0,
        exit_rule_triggered="hold_to_settlement",
        hold_duration_minutes=None,
        entry_fee_cents=entry_fee,
        exit_fee_cents=0,
        pnl_cents=float(pnl),
        fell_through_to_settlement=True,
    )


def _intraday_pnl(
    n_contracts: int,
    entry_price: float,
    exit_price: float,
    rule_triggered: str,
    exit_time: pd.Timestamp,
    entry_time: pd.Timestamp,
) -> ExitResult:
    gross = (exit_price - entry_price) * n_contracts * 100
    entry_fee = taker_fee_cents(n_contracts, entry_price)
    exit_fee = taker_fee_cents(n_contracts, exit_price)
    hold_min = (exit_time - entry_time).total_seconds() / 60.0
    return ExitResult(
        exit_time=exit_time,
        exit_price=exit_price,
        exit_rule_triggered=rule_triggered,
        hold_duration_minutes=hold_min,
        entry_fee_cents=entry_fee,
        exit_fee_cents=exit_fee,
        pnl_cents=float(gross - entry_fee - exit_fee),
        fell_through_to_settlement=False,
    )


def _is_time_stop(snapshot_time: pd.Timestamp, city_tz_name: str) -> bool:
    ts = pd.Timestamp(snapshot_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize(city_tz_name)
    else:
        ts = ts.tz_convert(city_tz_name)
    return (ts.hour, ts.minute) >= (14, 0)


def _check_combined_at_snapshot(
    price: float,
    entry_price: float,
    model_prob: float,
    high_water_mark: float,
    snapshot_time: pd.Timestamp,
    city_tz_name: str,
) -> str | None:
    if price >= entry_price + 0.15:
        return "profit_target_15c"
    if price <= high_water_mark - 0.10:
        return "trailing_stop_10c"
    if _is_time_stop(snapshot_time, city_tz_name):
        return "time_stop_2pm"
    return None


def walk_exit_snapshots(
    day_df: pd.DataFrame,
    bucket_label: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    model_prob: float,
    rule: str,
    city_tz_name: str,
    n_contracts: int,
    resolved: bool,
) -> ExitResult:
    if rule == "hold_to_settlement":
        return _settlement_pnl(n_contracts, entry_price, resolved)

    snapshots = _snapshot_prices(day_df, bucket_label, entry_time)
    high_water_mark = entry_price

    for _, row in snapshots.iterrows():
        price = float(row["yes_mid_close"])
        snap_time = pd.Timestamp(row["snapshot_time_local"])
        high_water_mark = max(high_water_mark, price)

        triggered: str | None = None
        if rule == "profit_target_model" and price >= model_prob:
            triggered = "profit_target_model"
        elif rule == "profit_target_15c" and price >= entry_price + 0.15:
            triggered = "profit_target_15c"
        elif rule == "trailing_stop_10c" and price <= high_water_mark - 0.10:
            triggered = "trailing_stop_10c"
        elif rule == "time_stop_2pm" and _is_time_stop(snap_time, city_tz_name):
            triggered = "time_stop_2pm"
        elif rule == "combined":
            triggered = _check_combined_at_snapshot(
                price, entry_price, model_prob, high_water_mark, snap_time, city_tz_name
            )

        if triggered:
            return _intraday_pnl(
                n_contracts, entry_price, price, triggered, snap_time, entry_time
            )

    result = _settlement_pnl(n_contracts, entry_price, resolved)
    if rule == "combined":
        result = ExitResult(
            exit_time=None,
            exit_price=result.exit_price,
            exit_rule_triggered="hold_to_settlement",
            hold_duration_minutes=None,
            entry_fee_cents=result.entry_fee_cents,
            exit_fee_cents=0,
            pnl_cents=result.pnl_cents,
            fell_through_to_settlement=True,
        )
    else:
        result = ExitResult(
            exit_time=None,
            exit_price=result.exit_price,
            exit_rule_triggered="hold_to_settlement",
            hold_duration_minutes=None,
            entry_fee_cents=result.entry_fee_cents,
            exit_fee_cents=0,
            pnl_cents=result.pnl_cents,
            fell_through_to_settlement=True,
        )
    return result


def _build_day_cache(partition_df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    df = partition_df.copy()
    group_cols = _day_group_columns(df)
    city_col = group_cols[0]
    df[city_col] = df[city_col].map(_city_key)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in df.groupby([city_col, "event_date"], sort=False):
        cache[(str(key[0]), str(key[1]))] = group.copy()
    return cache


def run_intraday_exit_backtest(
    split_name: str,
    exit_rules: list[str],
    deploy_config_path: str | Path,
    city_config_path: str | Path,
    frozen_k: int | None = None,
) -> pd.DataFrame:
    deploy = _load_deploy_config(Path(deploy_config_path))
    city_config = _load_city_config(Path(city_config_path))
    edge_threshold = float(deploy["edge_threshold"])
    n_contracts = int(deploy["n_contracts_default"])

    if frozen_k is None:
        from snapshot_stability import load_or_create_frozen_k

        frozen_k = load_or_create_frozen_k()

    partition = _load_split(split_name)
    forecasts = pd.read_parquet(FORECASTS_PATH)
    exclude = LOW_OOS_COVERAGE_CITIES if split_name == "time_holdout" else None
    print(f"  Generating signals for {split_name}...")
    signals = generate_signals(partition, forecasts, "track_b_flat", exclude_cities=exclude)
    selected = apply_selection(signals, "edge_threshold", edge_threshold)
    traded = selected[~selected["no_signal"]].copy()
    print(f"  {len(traded)} entry trades after selection")

    day_cache = _build_day_cache(partition)
    entry_cache: dict[tuple[str, str], pd.Timestamp | None] = {}

    records: list[dict[str, object]] = []
    for _, sig in traded.iterrows():
        city = str(sig["city"])
        event_date = str(sig["event_date"])
        bucket_label = str(sig["entry_bucket"])
        entry_price = float(sig["entry_price"])
        model_prob = float(sig["model_prob"])
        edge = float(sig["edge"])
        resolved = bool(sig["resolved"])

        cache_key = (city, event_date)
        day_df = day_cache.get(cache_key, pd.DataFrame())
        if day_df.empty:
            continue
        if cache_key not in entry_cache:
            entry_cache[cache_key] = _entry_snapshot_time(day_df, frozen_k)
        entry_time = entry_cache[cache_key]
        if entry_time is None:
            continue

        tz_name = str(city_config.get(city, {}).get("timezone", "America/Chicago"))

        for rule in exit_rules:
            if rule not in VALID_EXIT_RULES:
                raise ValueError(f"Unknown exit rule: {rule}")
            outcome = walk_exit_snapshots(
                day_df,
                bucket_label,
                entry_time,
                entry_price,
                model_prob,
                rule,
                tz_name,
                n_contracts,
                resolved,
            )
            records.append(
                {
                    "split": split_name,
                    "city": city,
                    "event_date": event_date,
                    "exit_rule": rule,
                    "entry_time": entry_time.isoformat(),
                    "entry_price": entry_price,
                    "model_prob": model_prob,
                    "edge_at_entry": edge,
                    "exit_time": outcome.exit_time.isoformat() if outcome.exit_time else None,
                    "exit_price": outcome.exit_price,
                    "exit_rule_triggered": outcome.exit_rule_triggered,
                    "hold_duration_minutes": outcome.hold_duration_minutes,
                    "n_contracts": n_contracts,
                    "entry_fee_cents": outcome.entry_fee_cents,
                    "exit_fee_cents": outcome.exit_fee_cents,
                    "pnl_cents": outcome.pnl_cents,
                    "settlement_won": int(resolved),
                    "bucket_label": bucket_label,
                    "fell_through_to_settlement": outcome.fell_through_to_settlement,
                }
            )

    return pd.DataFrame.from_records(records)


def summarize_exit_backtest(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (split_name, rule), group in results.groupby(["split", "exit_rule"], sort=True):
        calendar_dates = sorted(group["event_date"].unique())
        n_calendar = len(calendar_dates) if calendar_dates else 1
        daily = (
            group.groupby("event_date", sort=True)["pnl_cents"]
            .sum()
            .reindex(calendar_dates, fill_value=0.0)
        )
        daily_returns = daily.to_numpy(dtype=float) / INITIAL_BANKROLL_CENTS

        n = len(daily_returns)
        if n == 0 or np.std(daily_returns, ddof=1) == 0:
            sharpe = 0.0
            ci_lo = float("nan")
            ci_hi = float("nan")
            sortino = 0.0
        else:
            std_r = float(np.std(daily_returns, ddof=1))
            sr_daily = float(np.mean(daily_returns) / std_r)
            sharpe = sr_daily * np.sqrt(252)
            se_sr = np.sqrt((1 + 0.5 * sharpe**2) / n)
            ci_lo = float(sharpe - 1.96 * se_sr)
            ci_hi = float(sharpe + 1.96 * se_sr)
            downside = daily_returns[daily_returns < 0]
            d_std = np.std(downside, ddof=1) if len(downside) > 0 else 1e-6
            sortino = float(np.mean(daily_returns) / d_std * np.sqrt(252))

        cum = daily.cumsum()
        peak = cum.cummax()
        max_dd = float((cum - peak).min()) if len(cum) else 0.0

        holds = group["hold_duration_minutes"].dropna()
        fell_through_pct = 0.0
        if rule != "hold_to_settlement":
            fell_through_pct = float(group["fell_through_to_settlement"].mean() * 100.0)

        rows.append(
            {
                "split": split_name,
                "exit_rule": rule,
                "n_trades": int(len(group)),
                "win_rate": float((group["pnl_cents"] > 0).mean()),
                "mean_pnl_cents": float(group["pnl_cents"].mean()),
                "total_pnl_cents": float(group["pnl_cents"].sum()),
                "sharpe_annual": round(sharpe, 2),
                "sharpe_ci_lo": round(ci_lo, 2) if np.isfinite(ci_lo) else None,
                "sharpe_ci_hi": round(ci_hi, 2) if np.isfinite(ci_hi) else None,
                "sortino_annual": round(sortino, 2),
                "max_drawdown_cents": round(max_dd, 1),
                "mean_hold_minutes": float(holds.mean()) if len(holds) else None,
                "median_hold_minutes": float(holds.median()) if len(holds) else None,
                "avg_trades_per_day": round(len(group) / n_calendar, 2),
                "pct_fell_through_to_settlement": round(fell_through_pct, 1),
            }
        )

    return pd.DataFrame(rows)


def plot_exit_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.groupby("exit_rule", sort=False).agg(
        sharpe_annual=("sharpe_annual", "mean"),
        sharpe_ci_lo=("sharpe_ci_lo", "mean"),
        sharpe_ci_hi=("sharpe_ci_hi", "mean"),
    ).reset_index()
    plot_df = plot_df.sort_values("sharpe_annual", ascending=False)

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, plot_df["sharpe_annual"], color=PALETTE["blue"], width=0.6)
    yerr_lo = plot_df["sharpe_annual"] - plot_df["sharpe_ci_lo"]
    yerr_hi = plot_df["sharpe_ci_hi"] - plot_df["sharpe_annual"]
    ax.errorbar(
        x,
        plot_df["sharpe_annual"],
        yerr=[yerr_lo.fillna(0), yerr_hi.fillna(0)],
        fmt="none",
        ecolor=PALETTE["grey"],
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["exit_rule"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe (annual)")
    ax.set_title("Intraday Exit Rule Comparison")
    ax.axhline(0, color=PALETTE["grey"], linestyle=":", linewidth=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exit_cumulative(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(VALID_EXIT_RULES)))
    for idx, rule in enumerate(VALID_EXIT_RULES):
        subset = results.loc[results["exit_rule"].eq(rule)]
        if subset.empty:
            continue
        daily = (
            subset.groupby("event_date", sort=True)["pnl_cents"]
            .sum()
            .cumsum()
        )
        ax.plot(
            pd.to_datetime(daily.index).to_numpy(),
            daily.to_numpy(),
            label=rule,
            color=colors[idx % len(colors)],
            linewidth=1.5,
        )
    ax.axhline(0, color=PALETTE["grey"], linestyle=":", linewidth=0.8)
    ax.set_title("Cumulative PnL by Exit Rule")
    ax.set_xlabel("Event date")
    ax.set_ylabel("Cumulative net PnL (cents)")
    ax.legend(fontsize=7, loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_hold_to_settlement(
    split_name: str,
    results: pd.DataFrame,
    deploy_config_path: Path,
) -> None:
    deploy = _load_deploy_config(deploy_config_path)
    edge_threshold = float(deploy["edge_threshold"])
    partition = _load_split(split_name)
    forecasts = pd.read_parquet(FORECASTS_PATH)
    exclude = LOW_OOS_COVERAGE_CITIES if split_name == "time_holdout" else None
    signals = generate_signals(partition, forecasts, "track_b_flat", exclude_cities=exclude)
    selected = apply_selection(signals, "edge_threshold", edge_threshold)
    calendar_dates = _calendar_date_keys(partition)
    ref_trades, _, _, _ = run_backtest(selected, "flat_5", calendar_dates)

    hold = results.loc[
        (results["split"].eq(split_name)) & (results["exit_rule"].eq("hold_to_settlement"))
    ].copy()
    if ref_trades.empty and hold.empty:
        print(f"  [{split_name}] hold_to_settlement: 0 trades (OK)")
        return

    ref_total = float(ref_trades["net_pnl_cents"].sum()) if not ref_trades.empty else 0.0
    hold_total = float(hold["pnl_cents"].sum()) if not hold.empty else 0.0
    if len(ref_trades) != len(hold) or abs(ref_total - hold_total) > 0.01:
        raise RuntimeError(
            f"hold_to_settlement mismatch on {split_name}: "
            f"grid n={len(ref_trades)} pnl={ref_total:.1f} vs "
            f"intraday n={len(hold)} pnl={hold_total:.1f}"
        )
    print(
        f"  [{split_name}] hold_to_settlement validated: "
        f"{len(hold)} trades, {hold_total:.0f} cents total PnL"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday exit backtest")
    parser.add_argument(
        "--split",
        choices=["threshold_opt", "time_holdout", "both"],
        default="both",
    )
    parser.add_argument("--rules", nargs="+", default=VALID_EXIT_RULES)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "deploy_config.json",
    )
    parser.add_argument(
        "--city-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "city_config.json",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["threshold_opt", "time_holdout"] if args.split == "both" else [args.split]

    frames: list[pd.DataFrame] = []
    for split_name in splits:
        print(f"\n=== Running intraday exit backtest: {split_name} ===")
        frame = run_intraday_exit_backtest(
            split_name,
            args.rules,
            args.deploy_config,
            args.city_config,
        )
        print(f"  {len(frame)} trade-rule rows")
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize_exit_backtest(results)

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_RESULTS, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    REPORT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_SUMMARY, index=False)

    if not args.no_validate and "hold_to_settlement" in args.rules:
        print("\n=== Validating hold_to_settlement vs grid ===")
        for split_name in splits:
            validate_hold_to_settlement(split_name, results, args.deploy_config)

    print("\n=== SUMMARY ===")
    if summary.empty:
        print("No trades.")
    else:
        display_cols = [
            "split",
            "exit_rule",
            "n_trades",
            "win_rate",
            "total_pnl_cents",
            "sharpe_annual",
            "sharpe_ci_lo",
            "sharpe_ci_hi",
            "pct_fell_through_to_settlement",
        ]
        print(summary[display_cols].to_string(index=False))

    if not args.no_figures and not summary.empty:
        plot_exit_comparison(summary, FIGURE_DIR / "intraday_exit_comparison.png")
        plot_exit_cumulative(results, FIGURE_DIR / "intraday_exit_cumulative.png")
        print(f"\nFigures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
