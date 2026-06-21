"""Extended MCP challenge simulation: Track-B + profit_target_15c over full market window."""

from __future__ import annotations

import argparse
import json
import sys
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

from backtest_intraday_exit import (  # noqa: E402
    _build_day_cache,
    _entry_snapshot_time,
    walk_exit_snapshots,
)
from backtest_utils import bootstrap_sharpe, sharpe_stats  # noqa: E402
from run_mcp_simulation import (  # noqa: E402
    INITIAL_BANKROLL_CENTS,
    ELIMINATION_CENTS,
    FLAT_CONTRACTS_DEFAULT,
    FLAT_CONTRACTS_REDUCED,
    BANKROLL_REDUCTION_THRESHOLD_CENTS,
    DAILY_LOSS_CAP_CENTS,
    MAX_POSITION_PCT,
    _fit_contracts,
    load_simulation_data,
    plot_equity_curve,
)
from run_trackB_grid import (  # noqa: E402
    LOW_OOS_COVERAGE_CITIES,
    apply_selection,
    generate_signals,
)
from snapshot_stability import load_or_create_frozen_k  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "trackb" / "extended_mcp_sim"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "deploy_config.json"
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"

TRADEABLE_CITIES = [
    "chicago_midway",
    "houston",
    "los_angeles",
    "new_york_city",
    "oklahoma_city",
    "phoenix",
    "san_francisco",
]
HOLDOUT_CITIES = {"denver", "miami", "minneapolis"}
EXCLUDED_CITIES = set(LOW_OOS_COVERAGE_CITIES)
EXIT_RULE = "profit_target_15c"
EDGE_THRESHOLD_DEFAULT = 0.037


def _load_city_config() -> dict:
    with open(CITY_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _city_timezone(city_config: dict, city: str) -> str:
    return str(city_config.get(city, {}).get("timezone", "America/Chicago"))


def run_extended_mcp_backtest(
    signals: pd.DataFrame,
    market_df: pd.DataFrame,
    calendar_dates: list[str],
    city_config: dict,
    frozen_k: int,
    exit_rule: str = EXIT_RULE,
    initial_bankroll_cents: int = INITIAL_BANKROLL_CENTS,
    elimination_cents: int = ELIMINATION_CENTS,
    flat_contracts_default: int = FLAT_CONTRACTS_DEFAULT,
    flat_contracts_reduced: int = FLAT_CONTRACTS_REDUCED,
    bankroll_reduction_threshold_cents: int = BANKROLL_REDUCTION_THRESHOLD_CENTS,
    daily_loss_cap_cents: int = DAILY_LOSS_CAP_CENTS,
    max_position_pct: float = MAX_POSITION_PCT,
) -> dict[str, object]:
    """Sequential MCP backtest with intraday profit_target_15c exits."""
    traded = signals[~signals["no_signal"]].copy()
    day_cache = _build_day_cache(market_df)
    entry_cache: dict[tuple[str, str], pd.Timestamp | None] = {}

    bankroll = float(initial_bankroll_cents)
    eliminated = False
    elimination_date: str | None = None
    trade_records: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for event_date in calendar_dates:
        if bankroll <= elimination_cents:
            eliminated = True
            elimination_date = event_date
            break

        opening_bankroll = bankroll
        base_contracts = (
            flat_contracts_reduced
            if opening_bankroll < bankroll_reduction_threshold_cents
            else flat_contracts_default
        )
        day_trades = traded[traded["event_date"].eq(event_date)].copy() if not traded.empty else traded
        day_trades = day_trades.sort_values("edge", ascending=False)

        daily_spent = 0
        day_pnl = 0.0
        n_trades = 0
        n_wins = 0

        for _, row in day_trades.iterrows():
            city = str(row["city"])
            entry_price = float(row["entry_price"])
            contracts = _fit_contracts(
                base_contracts,
                entry_price,
                int(opening_bankroll),
                daily_spent,
                max_position_pct=max_position_pct,
            )
            if contracts <= 0:
                continue

            cost_cents = int(contracts * entry_price * 100)
            daily_spent += cost_cents

            cache_key = (city, event_date)
            day_df = day_cache.get(cache_key, pd.DataFrame())
            if day_df.empty:
                continue
            if cache_key not in entry_cache:
                entry_cache[cache_key] = _entry_snapshot_time(day_df, frozen_k)
            entry_time = entry_cache[cache_key]
            if entry_time is None:
                continue

            exit_result = walk_exit_snapshots(
                day_df,
                str(row["entry_bucket"]),
                entry_time,
                entry_price,
                float(row["model_prob"]),
                exit_rule,
                _city_timezone(city_config, city),
                contracts,
                bool(row["resolved"]),
            )

            pnl_cents = float(exit_result.pnl_cents)
            exit_type = (
                "profit_target"
                if exit_result.exit_rule_triggered == "profit_target_15c"
                else "settlement"
            )
            exit_price = float(exit_result.exit_price) if exit_result.exit_price is not None else np.nan
            won = pnl_cents > 0

            day_pnl += pnl_cents
            n_trades += 1
            if won:
                n_wins += 1

            trade_records.append(
                {
                    "date": event_date,
                    "city": city,
                    "bucket": str(row["entry_bucket"]),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_type": exit_type,
                    "pnl_cents": pnl_cents,
                    "won": won,
                    "contracts": contracts,
                    "edge": float(row["edge"]),
                    "model_prob": float(row["model_prob"]),
                    "resolved": bool(row["resolved"]),
                    "entry_fee_cents": exit_result.entry_fee_cents,
                    "exit_fee_cents": exit_result.exit_fee_cents,
                    "event_date": event_date,
                    "net_pnl_cents": pnl_cents,
                }
            )

        bankroll += day_pnl
        cumulative_pnl = bankroll - initial_bankroll_cents
        daily_rows.append(
            {
                "date": event_date,
                "daily_pnl_cents": day_pnl,
                "cumulative_pnl_cents": cumulative_pnl,
                "bankroll_cents": bankroll,
                "n_trades": n_trades,
                "n_wins": n_wins,
            }
        )

        if bankroll <= elimination_cents:
            eliminated = True
            elimination_date = event_date
            break

    trades_df = pd.DataFrame.from_records(trade_records)
    daily_log = pd.DataFrame.from_records(daily_rows)
    return {
        "trades": trades_df,
        "daily_log": daily_log,
        "eliminated": eliminated,
        "elimination_date": elimination_date,
    }


def _per_city_breakdown(trades: pd.DataFrame, daily_log: pd.DataFrame) -> dict[str, dict]:
    breakdown: dict[str, dict] = {}
    if trades.empty:
        for city in TRADEABLE_CITIES:
            breakdown[city] = {
                "trades": 0,
                "wins": 0,
                "pnl_cents": 0.0,
                "sharpe_annual": None,
            }
        return breakdown

    for city in sorted(trades["city"].unique()):
        city_trades = trades[trades["city"].eq(city)]
        city_pnl = float(city_trades["pnl_cents"].sum())
        wins = int((city_trades["pnl_cents"] > 0).sum())
        city_daily = (
            city_trades.groupby("date", sort=True)["pnl_cents"].sum()
            if not city_trades.empty
            else pd.Series(dtype=float)
        )
        if len(city_daily) >= 2 and city_daily.std(ddof=1) > 0:
            sr = float(city_daily.mean() / city_daily.std(ddof=1) * np.sqrt(252))
        else:
            sr = None
        breakdown[city] = {
            "trades": int(len(city_trades)),
            "wins": wins,
            "pnl_cents": round(city_pnl, 1),
            "sharpe_annual": round(sr, 2) if sr is not None else None,
        }
    return breakdown


def build_extended_summary(
    result: dict[str, object],
    calendar_dates: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    trades: pd.DataFrame = result["trades"]
    daily_log: pd.DataFrame = result["daily_log"]
    eliminated: bool = bool(result["eliminated"])
    elimination_date: str | None = result["elimination_date"]

    n_calendar_days = len(calendar_dates)
    if daily_log.empty:
        daily_returns = np.array([], dtype=float)
        daily_pnl = np.array([], dtype=float)
        bankroll_path = np.array([INITIAL_BANKROLL_CENTS], dtype=float)
    else:
        daily_pnl = daily_log["daily_pnl_cents"].to_numpy(dtype=float)
        bankroll_path = np.concatenate([[INITIAL_BANKROLL_CENTS], daily_log["bankroll_cents"].to_numpy()])
        opening = np.concatenate([[INITIAL_BANKROLL_CENTS], daily_log["bankroll_cents"].to_numpy()[:-1]])
        daily_returns = np.divide(
            daily_pnl,
            opening,
            out=np.zeros_like(daily_pnl, dtype=float),
            where=opening > 0,
        )

    returns_series = pd.Series(daily_returns)
    n_trades = int(len(trades))
    mean_edge = float(trades["edge"].mean()) if not trades.empty and "edge" in trades.columns else float("nan")
    n_trading_days = int((daily_log["n_trades"] > 0).sum()) if not daily_log.empty else 0
    trades_per_day = n_trades / n_calendar_days if n_calendar_days > 0 else 0.0

    if len(daily_returns) < 2 or np.std(daily_returns, ddof=1) == 0:
        sharpe_annual = 0.0
        sortino_annual = 0.0
    else:
        std_return = float(np.std(daily_returns, ddof=1))
        sharpe_annual = float(np.mean(daily_returns) / std_return * np.sqrt(252))
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 1e-6
        sortino_annual = float(np.mean(daily_returns) / downside_std * np.sqrt(252))

    boot = bootstrap_sharpe(returns_series, n_boot=1000)
    sharpe_boot_ci_lo = boot.get("sharpe_boot_ci_low")
    sharpe_boot_ci_hi = boot.get("sharpe_boot_ci_high")

    peak = np.maximum.accumulate(bankroll_path)
    dd = bankroll_path - peak
    max_drawdown_cents = float(dd.min()) if len(dd) else 0.0
    peak_at_max_dd = float(peak[np.argmin(dd)]) if len(dd) else float(INITIAL_BANKROLL_CENTS)
    max_drawdown_pct = max_drawdown_cents / peak_at_max_dd if peak_at_max_dd > 0 else 0.0
    min_bankroll_cents = float(bankroll_path.min()) if len(bankroll_path) else float(INITIAL_BANKROLL_CENTS)
    final_bankroll_cents = float(bankroll_path[-1]) if len(bankroll_path) else float(INITIAL_BANKROLL_CENTS)
    total_pnl_cents = float(daily_pnl.sum()) if len(daily_pnl) else 0.0

    losses = (daily_pnl < 0).astype(int) if len(daily_pnl) else np.array([], dtype=int)
    streaks: list[int] = []
    current = 0
    for loss in losses:
        if loss:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    worst_losing_streak = max(streaks) if streaks else 0

    if not trades.empty:
        pnl_col = "pnl_cents" if "pnl_cents" in trades.columns else "net_pnl_cents"
        wins_sum = trades.loc[trades[pnl_col] > 0, pnl_col].sum()
        losses_total = trades.loc[trades[pnl_col] < 0, pnl_col].sum()
        win_rate = float((trades[pnl_col] > 0).mean())
        profit_factor = float(wins_sum / abs(losses_total)) if losses_total < 0 else float("inf")
        profit_target_count = int((trades.get("exit_type", pd.Series(dtype=str)) == "profit_target").sum())
        settlement_count = int((trades.get("exit_type", pd.Series(dtype=str)) == "settlement").sum())
    else:
        win_rate = 0.0
        profit_factor = float("nan")
        profit_target_count = 0
        settlement_count = 0

    if len(daily_pnl) > 0 and total_pnl_cents != 0:
        top3 = daily_log.nlargest(3, "daily_pnl_cents")["daily_pnl_cents"].sum()
        pnl_concentration_pct = float(100.0 * top3 / total_pnl_cents)
    else:
        pnl_concentration_pct = 0.0

    per_city = _per_city_breakdown(trades, daily_log)

    psr_0 = float("nan")
    if len(daily_returns) > 0:
        stats = sharpe_stats(returns_series)
        psr_0 = float(stats["PSR_0"])

    return {
        "start_date": start_date,
        "end_date": end_date,
        "n_calendar_days": n_calendar_days,
        "exit_rule": EXIT_RULE,
        "edge_threshold": EDGE_THRESHOLD_DEFAULT,
        "sharpe_annual": round(sharpe_annual, 2),
        "sharpe_bootstrap_ci_lo": round(float(sharpe_boot_ci_lo), 2) if pd.notna(sharpe_boot_ci_lo) else None,
        "sharpe_bootstrap_ci_hi": round(float(sharpe_boot_ci_hi), 2) if pd.notna(sharpe_boot_ci_hi) else None,
        "sortino_annual": round(sortino_annual, 2),
        "total_pnl_cents": round(total_pnl_cents, 1),
        "total_pnl_dollars": round(total_pnl_cents / 100.0, 2),
        "final_bankroll_cents": round(final_bankroll_cents, 1),
        "max_drawdown_cents": round(max_drawdown_cents, 1),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "worst_losing_streak": worst_losing_streak,
        "pnl_concentration_top3_pct": round(pnl_concentration_pct, 1),
        "eliminated": bool(eliminated),
        "elimination_date": elimination_date,
        "min_bankroll_cents": round(min_bankroll_cents, 1),
        "n_trades": n_trades,
        "n_trading_days": n_trading_days,
        "trades_per_day": round(trades_per_day, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "profit_target_exits": profit_target_count,
        "settlement_exits": settlement_count,
        "mean_edge": round(mean_edge, 4) if np.isfinite(mean_edge) else None,
        "psr_0": round(psr_0, 4) if np.isfinite(psr_0) else None,
        "per_city": per_city,
        "tradeable_cities": TRADEABLE_CITIES,
    }


def print_extended_summary(summary: dict[str, object]) -> None:
    print("\n=== EXTENDED MCP CHALLENGE SIMULATION (Track-B) ===")
    print(
        f"Period: {summary['start_date']} to {summary['end_date']} "
        f"({summary['n_calendar_days']} calendar days)"
    )
    print(f"Exit rule: {summary['exit_rule']} | Edge threshold: {summary['edge_threshold']}")

    print("\nPERFORMANCE")
    print(f"  Total PnL:          ${summary['total_pnl_dollars']:.2f}")
    print(f"  Final bankroll:     ${summary['final_bankroll_cents'] / 100:.2f}")
    ci_lo = summary.get("sharpe_bootstrap_ci_lo")
    ci_hi = summary.get("sharpe_bootstrap_ci_hi")
    if ci_lo is not None and ci_hi is not None:
        print(
            f"  Sharpe (annual):    {summary['sharpe_annual']:.2f} "
            f"[bootstrap 95% CI: {ci_lo:.2f}, {ci_hi:.2f}]"
        )
    else:
        print(f"  Sharpe (annual):    {summary['sharpe_annual']:.2f}")
    print(f"  Sortino (annual):   {summary['sortino_annual']:.2f}")
    print(f"  PnL top-3 days:     {summary['pnl_concentration_top3_pct']:.1f}% of total")

    print("\nRISK")
    print(
        f"  Max drawdown:       ${summary['max_drawdown_cents'] / 100:.2f} "
        f"({summary['max_drawdown_pct']:.1%})"
    )
    print(f"  Min bankroll:       ${summary['min_bankroll_cents'] / 100:.2f}")
    print(f"  Eliminated:         {summary['eliminated']}")
    if summary.get("elimination_date"):
        print(f"  Elimination date:   {summary['elimination_date']}")
    print(f"  Worst losing streak: {summary['worst_losing_streak']} days")

    print("\nACTIVITY")
    print(f"  Total trades:       {summary['n_trades']}")
    print(f"  Trading days:       {summary['n_trading_days']}/{summary['n_calendar_days']}")
    print(f"  Trades/day:         {summary['trades_per_day']:.2f}")
    print(
        f"  Exit mix:           {summary['profit_target_exits']} profit_target, "
        f"{summary['settlement_exits']} settlement"
    )

    print("\nQUALITY")
    print(f"  Win rate:           {summary['win_rate']:.1%}")
    pf = summary.get("profit_factor")
    print(f"  Profit factor:      {pf:.2f}" if pf is not None else "  Profit factor:      n/a")
    mean_edge = summary.get("mean_edge")
    print(f"  Mean edge:          {mean_edge:.3f}" if mean_edge is not None else "  Mean edge:          n/a")

    print("\nPER-CITY BREAKDOWN")
    per_city = summary.get("per_city", {})
    for city in TRADEABLE_CITIES:
        row = per_city.get(city, {})
        sharpe = row.get("sharpe_annual")
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "n/a"
        print(
            f"  {city:18s} trades={row.get('trades', 0):3d}  wins={row.get('wins', 0):3d}  "
            f"PnL=${row.get('pnl_cents', 0) / 100:.2f}  Sharpe={sharpe_str}"
        )


def save_outputs(
    result: dict[str, object],
    summary: dict[str, object],
    output_dir: Path,
    write_figure: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades: pd.DataFrame = result["trades"]
    daily_log: pd.DataFrame = result["daily_log"]

    trade_cols = ["date", "city", "bucket", "entry_price", "exit_price", "exit_type", "pnl_cents", "won"]
    if not trades.empty:
        trades_out = trades[trade_cols].copy()
    else:
        trades_out = pd.DataFrame(columns=trade_cols)
    trades_out.to_csv(output_dir / "trades.csv", index=False)

    daily_log.to_csv(output_dir / "daily_pnl.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if write_figure and not daily_log.empty:
        plot_equity_curve(daily_log, summary, output_dir / "equity_curve.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended MCP challenge simulation (Track-B).")
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    global EDGE_THRESHOLD_DEFAULT
    args = parse_args()
    EDGE_THRESHOLD_DEFAULT = args.edge_threshold

    if args.config.exists():
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)
        if "edge_threshold" in config:
            EDGE_THRESHOLD_DEFAULT = float(config["edge_threshold"])
        print(f"Loaded deploy config from {args.config} (edge_threshold={EDGE_THRESHOLD_DEFAULT})")
    else:
        print(f"WARNING: config not found at {args.config}; using CLI defaults.")

    market_df, forecasts_df, calendar_dates, start_date, end_date = load_simulation_data()
    market_df = market_df[market_df["city"].isin(TRADEABLE_CITIES)].copy()
    if market_df["city"].isin(HOLDOUT_CITIES).any():
        raise ValueError("Holdout city data leaked into tradeable market frame.")

    print(
        f"Simulation period: {start_date} to {end_date} "
        f"({len(calendar_dates)} calendar days)"
    )
    print(f"Tradeable cities: {', '.join(TRADEABLE_CITIES)}")
    print(f"Market rows: {len(market_df):,} | Forecast rows: {len(forecasts_df):,}")

    if forecasts_df.empty:
        raise FileNotFoundError(
            "No Track-B forecasts found. Run scripts/generate_trackB_forecasts.py first."
        )

    frozen_k = load_or_create_frozen_k()
    city_config = _load_city_config()

    signals = generate_signals(
        market_df,
        forecasts_df,
        "track_b_flat",
        exclude_cities=EXCLUDED_CITIES,
    )
    selected = apply_selection(signals, "edge_threshold", EDGE_THRESHOLD_DEFAULT)

    result = run_extended_mcp_backtest(
        selected,
        market_df,
        calendar_dates,
        city_config,
        frozen_k,
        exit_rule=EXIT_RULE,
    )
    summary = build_extended_summary(result, calendar_dates, start_date, end_date)
    summary["edge_threshold"] = EDGE_THRESHOLD_DEFAULT
    result["summary"] = summary

    save_outputs(result, summary, args.output_dir, write_figure=not args.no_figures)
    print_extended_summary(summary)
    print(f"\nOutputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
