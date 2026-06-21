"""Run continuous MCP challenge simulation on IS + OOS + fresh validation data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
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

from backtest_utils import sharpe_stats  # noqa: E402
from build_splits import discover_city_csvs, load_city_frame  # noqa: E402
from run_trackB_grid import (  # noqa: E402
    LOW_OOS_COVERAGE_CITIES,
    _calendar_date_keys,
    apply_selection,
    generate_signals,
)
from entry_interface import filter_to_trading_window  # noqa: E402
from snapshot_stability import stability_entry  # noqa: E402
from src.data_store import TRAIN_CITIES  # noqa: E402
from src.sizing import taker_fee_cents  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FRESH_DIR = PROJECT_ROOT / "data" / "fresh_validation"
FORECASTS_OOS_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"
FORECASTS_FRESH_PATH = FRESH_DIR / "forecasts_fresh.parquet"
MARKET_FRESH_PATH = FRESH_DIR / "market_fresh.parquet"
RAW_DATA_DIR = PROJECT_ROOT / "historic_tmax_market_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "mcp_simulation_extended"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "deploy_config.json"

INITIAL_BANKROLL_CENTS = 10_000
ELIMINATION_CENTS = 7_000
FLAT_CONTRACTS_DEFAULT = 5
FLAT_CONTRACTS_REDUCED = 3
BANKROLL_REDUCTION_THRESHOLD_CENTS = 8_500
DAILY_LOSS_CAP_CENTS = 600
MAX_POSITION_PCT = 0.30

PALETTE = {"blue": "#4878CF", "orange": "#E68A2E", "grey": "#8A8A8A"}
DEDUPE_MARKET_KEYS = ["city", "event_date", "snapshot_time_local", "bucket_label"]
DEDUPE_FORECAST_KEYS = ["city", "event_date"]


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "city" not in out.columns and "source_city_folder" in out.columns:
        out["city"] = out["source_city_folder"]
    out["city"] = out["city"].map(_city_key)
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    out["snapshot_time_local"] = pd.to_datetime(out["snapshot_time_local"], utc=True)
    if out["snapshot_time_local"].dt.tz is not None:
        out["snapshot_time_local"] = out["snapshot_time_local"].dt.tz_localize(None)
    if "bucket_label" in out.columns:
        out["bucket_label"] = out["bucket_label"].astype(str)
    return out


def _dedupe_market(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame()
    combined = pd.concat(valid, ignore_index=True, sort=False)
    combined = _normalize_market_df(combined)
    return combined.drop_duplicates(subset=DEDUPE_MARKET_KEYS, keep="last")


def _load_csv_gap_market(min_exclusive_date: date | None, end_date: date | None = None) -> pd.DataFrame:
    if min_exclusive_date is None or not RAW_DATA_DIR.exists():
        return pd.DataFrame()
    city_csvs = discover_city_csvs(RAW_DATA_DIR)
    frames: list[pd.DataFrame] = []
    for city in TRAIN_CITIES:
        if city not in city_csvs:
            continue
        df = load_city_frame(city, city_csvs[city])
        mask = df["event_date"] > min_exclusive_date
        if end_date is not None:
            mask &= df["event_date"] <= end_date
        subset = df.loc[mask].copy()
        if not subset.empty:
            frames.append(subset)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_forecasts() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if FORECASTS_OOS_PATH.exists():
        frames.append(pd.read_parquet(FORECASTS_OOS_PATH))
    else:
        print(f"WARNING: missing {FORECASTS_OOS_PATH}; proceeding without OOS forecasts.")
    if FORECASTS_FRESH_PATH.exists():
        frames.append(pd.read_parquet(FORECASTS_FRESH_PATH))
    else:
        print(f"WARNING: missing {FORECASTS_FRESH_PATH}; proceeding without fresh forecasts.")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["city"] = combined["city"].map(_city_key)
    combined["event_date"] = pd.to_datetime(combined["event_date"]).dt.strftime("%Y-%m-%d")
    return combined.drop_duplicates(subset=DEDUPE_FORECAST_KEYS, keep="last")


def load_simulation_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str]:
    """Assemble market + forecast data for the full simulation window."""
    is_path = SPLIT_DIR / "threshold_opt.parquet"
    oos_path = SPLIT_DIR / "time_holdout.parquet"

    market_frames_init: list[pd.DataFrame] = []
    for split_path in [is_path, oos_path]:
        if split_path.exists():
            df = pd.read_parquet(split_path)
            market_frames_init.append(_normalize_market_df(df))
        else:
            print(f"WARNING: missing {split_path}")

    if not market_frames_init:
        raise FileNotFoundError("No split parquets found")

    base_market = pd.concat(market_frames_init, ignore_index=True)
    base_dates = pd.to_datetime(base_market["event_date"])
    start_date = base_dates.min().strftime("%Y-%m-%d")

    market_frames = [base_market]
    fresh_max_date: date | None = None

    if MARKET_FRESH_PATH.exists():
        fresh_market = pd.read_parquet(MARKET_FRESH_PATH)
        fresh_market = _normalize_market_df(fresh_market)
        split_max = base_dates.max().date()
        fresh_market = fresh_market.loc[
            pd.to_datetime(fresh_market["event_date"]).dt.date > split_max
        ].copy()
        if not fresh_market.empty:
            market_frames.append(fresh_market)
            fresh_max_date = pd.to_datetime(fresh_market["event_date"]).max().date()
    else:
        print(f"WARNING: missing {MARKET_FRESH_PATH}; proceeding without fresh market data.")

    gap_market = _load_csv_gap_market(fresh_max_date or base_dates.max().date())
    if not gap_market.empty:
        print(
            f"Loaded {len(gap_market):,} supplemental CSV rows after "
            f"{fresh_max_date or base_dates.max().date()}."
        )
        market_frames.append(gap_market)

    market_df = _dedupe_market(market_frames)
    if market_df.empty:
        raise ValueError("No market data available for simulation.")

    # IS data included intentionally: full-window MCP survival test, not OOS edge validation.
    forecasts_df = _load_forecasts()

    end_date = pd.to_datetime(market_df["event_date"]).max().strftime("%Y-%m-%d")
    calendar_dates = _calendar_date_keys(market_df)
    return market_df, forecasts_df, calendar_dates, start_date, end_date


def _fit_contracts(
    contracts: int,
    entry_price: float,
    bankroll_cents: int,
    daily_spent_cents: int,
    max_position_pct: float = MAX_POSITION_PCT,
) -> int:
    """Shrink contracts to satisfy 30% position cap and daily loss cap."""
    while contracts > 0:
        capital_at_risk = int(contracts * entry_price * 100)
        if capital_at_risk > int(max_position_pct * bankroll_cents):
            contracts -= 1
            continue
        if daily_spent_cents + capital_at_risk > DAILY_LOSS_CAP_CENTS:
            return 0
        return contracts
    return 0


def enrich_signals_with_entry_time(
    signals: pd.DataFrame,
    market_df: pd.DataFrame,
    k: int = 1,
) -> pd.DataFrame:
    """Attach stability entry snapshot time to each signal row."""
    out = signals.copy()
    out["entry_time"] = pd.NaT

    market = market_df.copy()
    market["snapshot_time_local"] = pd.to_datetime(market["snapshot_time_local"])
    day_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for idx, row in out.iterrows():
        if bool(row["no_signal"]):
            continue
        city = str(row["city"])
        event_date = str(row["event_date"])
        cache_key = (city, event_date)
        if cache_key not in day_cache:
            day_cache[cache_key] = market[
                market["city"].eq(city) & market["event_date"].eq(event_date)
            ].copy()
        day_df = filter_to_trading_window(day_cache[cache_key])
        if day_df.empty:
            continue
        stability = stability_entry(day_df, k=k)
        if not stability.no_signal:
            out.at[idx, "entry_time"] = stability.entry_snapshot_time

    return out


def find_intraday_exit(
    market_df: pd.DataFrame,
    city: str,
    event_date: str,
    bucket_label: str,
    entry_price: float,
    entry_time: pd.Timestamp,
    profit_target: float = 0.15,
) -> tuple[bool, float]:
    """Check if the profit target is hit after entry.

    Returns:
        (exited_early, exit_price)
        If exited_early is False, exit_price is not used (held to settlement).
    """
    if pd.isna(entry_time):
        return False, 0.0

    day_bucket = market_df[
        (market_df["city"] == city)
        & (market_df["event_date"] == event_date)
        & (market_df["bucket_label"].astype(str) == str(bucket_label))
    ].copy()
    day_bucket["snapshot_time_local"] = pd.to_datetime(day_bucket["snapshot_time_local"])
    after_entry = day_bucket[day_bucket["snapshot_time_local"] > entry_time]
    after_entry = after_entry.sort_values("snapshot_time_local")

    for _, snap in after_entry.iterrows():
        current_price = float(snap["yes_mid_close"])
        if current_price >= entry_price + profit_target:
            return True, current_price

    return False, 0.0


def run_mcp_backtest(
    signals: pd.DataFrame,
    calendar_dates: list[str],
    market_df: pd.DataFrame,
    profit_target: float = 0.15,
    initial_bankroll_cents: int = INITIAL_BANKROLL_CENTS,
    elimination_cents: int = ELIMINATION_CENTS,
    flat_contracts_default: int = FLAT_CONTRACTS_DEFAULT,
    flat_contracts_reduced: int = FLAT_CONTRACTS_REDUCED,
    bankroll_reduction_threshold_cents: int = BANKROLL_REDUCTION_THRESHOLD_CENTS,
    daily_loss_cap_cents: int = DAILY_LOSS_CAP_CENTS,
    max_position_pct: float = MAX_POSITION_PCT,
) -> dict[str, object]:
    """Walk day-by-day through calendar dates with MCP challenge constraints."""
    traded = signals[~signals["no_signal"]].copy()
    bankroll = float(initial_bankroll_cents)
    peak_bankroll = bankroll
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
        day_contracts: list[int] = []
        n_trades = 0

        for _, row in day_trades.iterrows():
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
            fee_cents = taker_fee_cents(contracts, entry_price)
            daily_spent += cost_cents

            exited_early, exit_price = find_intraday_exit(
                market_df,
                str(row["city"]),
                str(row["event_date"]),
                str(row["entry_bucket"]),
                entry_price,
                pd.to_datetime(row["entry_time"]),
                profit_target=profit_target,
            )

            if exited_early:
                exit_fee_cents = taker_fee_cents(contracts, exit_price)
                gross_exit_cents = int(contracts * exit_price * 100)
                net_pnl_cents = gross_exit_cents - cost_cents - fee_cents - exit_fee_cents
                exit_type = "profit_target_15c"
            else:
                payout_cents = contracts * 100 if bool(row["resolved"]) else 0
                net_pnl_cents = payout_cents - cost_cents - fee_cents
                exit_type = "settlement"
                exit_price = np.nan

            day_pnl += net_pnl_cents
            n_trades += 1
            day_contracts.append(contracts)
            trade_records.append(
                {
                    **row.to_dict(),
                    "contracts": contracts,
                    "cost_cents": cost_cents,
                    "fee_cents": fee_cents,
                    "net_pnl_cents": net_pnl_cents,
                    "exit_type": exit_type,
                    "exit_price": exit_price,
                    "opening_bankroll_cents": int(opening_bankroll),
                }
            )

        bankroll += day_pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        cumulative_pnl = bankroll - initial_bankroll_cents
        drawdown_cents = bankroll - peak_bankroll
        contracts_used = int(max(day_contracts)) if day_contracts else base_contracts
        daily_rows.append(
            {
                "date": event_date,
                "n_trades": n_trades,
                "daily_pnl_cents": day_pnl,
                "bankroll_cents": bankroll,
                "cumulative_pnl_cents": cumulative_pnl,
                "drawdown_cents": drawdown_cents,
                "contracts_used": contracts_used,
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
        "summary": {},
        "eliminated": eliminated,
        "elimination_date": elimination_date,
    }


def _daily_narrative(daily_log: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    if daily_log.empty:
        return {
            "positive_trading_days": 0,
            "n_trading_days": 0,
            "best_day": None,
            "worst_day": None,
        }

    trading_days = daily_log.loc[daily_log["n_trades"] > 0].copy()
    positive_days = int((trading_days["daily_pnl_cents"] > 0).sum())
    best = trading_days.loc[trading_days["daily_pnl_cents"].idxmax()] if not trading_days.empty else None
    worst = trading_days.loc[trading_days["daily_pnl_cents"].idxmin()] if not trading_days.empty else None

    def _day_city(date_value: str, largest: bool) -> str | None:
        if trades.empty:
            return None
        day_trades = trades.loc[trades["event_date"].eq(date_value)]
        if day_trades.empty:
            return None
        row = day_trades.loc[
            day_trades["net_pnl_cents"].idxmax() if largest else day_trades["net_pnl_cents"].idxmin()
        ]
        return str(row["city"])

    best_day = None
    worst_day = None
    if best is not None:
        best_day = {
            "date": str(best["date"]),
            "pnl_cents": float(best["daily_pnl_cents"]),
            "city": _day_city(str(best["date"]), largest=True),
        }
    if worst is not None:
        worst_day = {
            "date": str(worst["date"]),
            "pnl_cents": float(worst["daily_pnl_cents"]),
            "city": _day_city(str(worst["date"]), largest=False),
        }

    return {
        "positive_trading_days": positive_days,
        "n_trading_days": int(len(trading_days)),
        "best_day": best_day,
        "worst_day": worst_day,
    }


def build_summary(
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

    n_trades = int(len(trades))
    mean_edge = float(trades["edge"].mean()) if not trades.empty else float("nan")
    n_trading_days = int((daily_log["n_trades"] > 0).sum()) if not daily_log.empty else 0
    trades_per_day = n_trades / n_calendar_days if n_calendar_days > 0 else 0.0
    projected_60d = trades_per_day * 60.0
    meets_80_trade_min = projected_60d >= 80

    if len(daily_returns) == 0 or np.std(daily_returns, ddof=1) == 0:
        sharpe_annual = 0.0
        sharpe_ci_lo = float("nan")
        sharpe_ci_hi = float("nan")
        sortino_annual = 0.0
    else:
        n = len(daily_returns)
        std_return = float(np.std(daily_returns, ddof=1))
        sr_daily = float(np.mean(daily_returns) / std_return)
        sharpe_annual = sr_daily * np.sqrt(252)
        se_sr = np.sqrt((1 + 0.5 * sharpe_annual**2) / n)
        sharpe_ci_lo = float(sharpe_annual - 1.96 * se_sr)
        sharpe_ci_hi = float(sharpe_annual + 1.96 * se_sr)
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 1e-6
        sortino_annual = float(np.mean(daily_returns) / downside_std * np.sqrt(252))

    peak = np.maximum.accumulate(bankroll_path)
    dd = bankroll_path - peak
    max_drawdown_cents = float(dd.min()) if len(dd) else 0.0
    peak_at_max_dd = float(peak[np.argmin(dd)]) if len(dd) else float(INITIAL_BANKROLL_CENTS)
    max_drawdown_pct = max_drawdown_cents / peak_at_max_dd if peak_at_max_dd > 0 else 0.0
    min_bankroll_cents = float(bankroll_path.min()) if len(bankroll_path) else float(INITIAL_BANKROLL_CENTS)

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

    total_pnl_cents = float(daily_pnl.sum()) if len(daily_pnl) else 0.0
    final_bankroll_cents = float(bankroll_path[-1]) if len(bankroll_path) else float(INITIAL_BANKROLL_CENTS)

    if not trades.empty:
        wins = trades.loc[trades["net_pnl_cents"] > 0, "net_pnl_cents"].sum()
        losses_total = trades.loc[trades["net_pnl_cents"] < 0, "net_pnl_cents"].sum()
        win_rate = float((trades["net_pnl_cents"] > 0).mean())
        mean_pnl_per_trade_cents = float(trades["net_pnl_cents"].mean())
        profit_factor = float(wins / abs(losses_total)) if losses_total < 0 else float("inf")
        n_profit_target_exits = int((trades["exit_type"] == "profit_target_15c").sum())
        n_settlement_exits = int((trades["exit_type"] == "settlement").sum())
        city_summary = (
            trades.groupby("city")
            .agg(
                n_trades=("net_pnl_cents", "count"),
                total_pnl=("net_pnl_cents", "sum"),
                win_rate=("net_pnl_cents", lambda x: (x > 0).mean()),
                n_profit_target=("exit_type", lambda x: (x == "profit_target_15c").sum()),
                n_settlement=("exit_type", lambda x: (x == "settlement").sum()),
            )
            .to_dict(orient="index")
        )
    else:
        win_rate = 0.0
        mean_pnl_per_trade_cents = 0.0
        profit_factor = float("nan")
        n_profit_target_exits = 0
        n_settlement_exits = 0
        city_summary = {}

    if not trades.empty and len(daily_log) > 0:
        trading_pnl = daily_log.loc[daily_log["n_trades"] > 0, "daily_pnl_cents"]
        if not trading_pnl.empty and trading_pnl.sum() > 0:
            sorted_pnl = trading_pnl.sort_values(ascending=False)
            top3_pnl = sorted_pnl.head(3).sum()
            pnl_concentration_top3 = float(top3_pnl / trading_pnl.sum())
        else:
            pnl_concentration_top3 = float("nan")
    else:
        pnl_concentration_top3 = float("nan")

    psr_0 = float("nan")
    min_trl_0 = float("nan")
    if len(daily_returns) > 0:
        stats = sharpe_stats(pd.Series(daily_returns))
        psr_0 = float(stats["PSR_0"])
        min_trl_0 = float(stats["MinTRL_0"])

    narrative = _daily_narrative(daily_log, trades)
    gonogo_passes = {
        "not_eliminated": bool(not eliminated),
        "projected_trades_80": bool(meets_80_trade_min),
        "sharpe_positive": bool(sharpe_annual > 0),
        "max_dd_under_30pct": bool(max_drawdown_pct > -0.30),
        "win_rate_above_50pct": bool(win_rate > 0.5),
        "psr_above_95pct": bool(psr_0 > 0.95) if np.isfinite(psr_0) else False,
    }

    return {
        "start_date": start_date,
        "end_date": end_date,
        "n_calendar_days": n_calendar_days,
        "window_pct_of_60d": round(n_calendar_days / 60.0 * 100.0, 1),
        "sharpe_annual": round(sharpe_annual, 2),
        "sharpe_ci_lo": round(sharpe_ci_lo, 2) if np.isfinite(sharpe_ci_lo) else None,
        "sharpe_ci_hi": round(sharpe_ci_hi, 2) if np.isfinite(sharpe_ci_hi) else None,
        "sortino_annual": round(sortino_annual, 2),
        "total_pnl_cents": round(total_pnl_cents, 1),
        "total_pnl_dollars": round(total_pnl_cents / 100.0, 2),
        "final_bankroll_cents": round(final_bankroll_cents, 1),
        "max_drawdown_cents": round(max_drawdown_cents, 1),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "worst_losing_streak": worst_losing_streak,
        "eliminated": bool(eliminated),
        "elimination_date": elimination_date,
        "min_bankroll_cents": round(min_bankroll_cents, 1),
        "n_trades": n_trades,
        "n_trading_days": n_trading_days,
        "trades_per_day": round(trades_per_day, 2),
        "projected_60d_trades": round(projected_60d, 1),
        "meets_80_trade_min": bool(meets_80_trade_min),
        "win_rate": round(win_rate, 4),
        "mean_edge": round(mean_edge, 4) if np.isfinite(mean_edge) else None,
        "mean_pnl_per_trade_cents": round(mean_pnl_per_trade_cents, 2),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "n_profit_target_exits": n_profit_target_exits,
        "n_settlement_exits": n_settlement_exits,
        "pnl_concentration_top3": (
            round(pnl_concentration_top3, 4) if np.isfinite(pnl_concentration_top3) else None
        ),
        "per_city": city_summary,
        "psr_0": round(psr_0, 4) if np.isfinite(psr_0) else None,
        "min_trl_0": round(min_trl_0, 1) if np.isfinite(min_trl_0) else None,
        "gonogo": gonogo_passes,
        "gonogo_pass_count": int(sum(gonogo_passes.values())),
        **narrative,
    }


def _cumulative_pnl(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype=float)
    frame = trades_df.copy()
    frame["_date"] = pd.to_datetime(frame["event_date"])
    return frame.groupby("_date", sort=True)["net_pnl_cents"].sum().cumsum()


def plot_equity_curve(daily_log: pd.DataFrame, summary: dict[str, object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    dates = pd.to_datetime(daily_log["date"]).to_numpy()
    bankroll_dollars = (daily_log["bankroll_cents"] / 100.0).to_numpy()
    ax.plot(dates, bankroll_dollars, color=PALETTE["blue"], linewidth=1.8, label="Bankroll")
    for level, label in [(100, "Start"), (85, "Reduced sizing"), (70, "Elimination")]:
        ax.axhline(level, color=PALETTE["grey"], linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(dates[-1], level, f"  {label}", va="center", fontsize=8, color=PALETTE["grey"])
    ax.axhspan(0, 70, facecolor=PALETTE["grey"], alpha=0.12)
    ax.text(
        dates[len(dates) // 2],
        35,
        "Elimination zone",
        ha="center",
        va="center",
        fontsize=8,
        color=PALETTE["grey"],
    )
    subtitle = (
        f"{summary['start_date']} to {summary['end_date']} | "
        f"Final ${summary['final_bankroll_cents'] / 100:.2f} | "
        f"Sharpe {summary['sharpe_annual']:.2f}"
    )
    ax.set_title("MCP Challenge Simulation: Equity Curve", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll ($)")
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=9, color=PALETTE["grey"])
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_daily_pnl(daily_log: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    dates = pd.to_datetime(daily_log["date"]).to_numpy()
    pnl = daily_log["daily_pnl_cents"].to_numpy()
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in pnl]
    ax.bar(dates, pnl, color=colors, width=0.8)
    ax.axhline(0, color=PALETTE["grey"], linewidth=0.8)
    ax.set_title("Daily PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily PnL (cents)")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trade_heatmap(
    trades: pd.DataFrame,
    market_df: pd.DataFrame,
    calendar_dates: list[str],
    output_path: Path,
) -> None:
    cities = sorted(market_df["city"].dropna().unique())
    dates = calendar_dates
    market_days = set(
        zip(
            market_df["city"].astype(str),
            pd.to_datetime(market_df["event_date"]).dt.strftime("%Y-%m-%d"),
        )
    )
    trade_lookup: dict[tuple[str, str], float] = {}
    if not trades.empty:
        for _, row in trades.iterrows():
            trade_lookup[(str(row["city"]), str(row["event_date"]))] = float(row["net_pnl_cents"])

    values = np.full((len(cities), len(dates)), np.nan)
    for i, city in enumerate(cities):
        for j, day in enumerate(dates):
            if (city, day) not in market_days:
                values[i, j] = 2.0  # white / no market data
            elif (city, day) not in trade_lookup:
                values[i, j] = 0.0  # light grey / no trade
            elif trade_lookup[(city, day)] > 0:
                values[i, j] = 1.0  # green / win
            else:
                values[i, j] = -1.0  # red / loss

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#d62728", "#E6E6E6", "#ffffff", "#2ca02c"])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-1, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontsize=8)
    step = max(1, len(dates) // 12)
    xticks = list(range(0, len(dates), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([dates[i] for i in xticks], rotation=45, ha="right", fontsize=7)
    ax.set_title("Trade Outcomes by City and Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("City")
    plt.colorbar(im, ax=ax, ticks=[-1, 0, 2, 1], label="Outcome")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_vs_baselines(trades: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cumulative = _cumulative_pnl(trades)
    if not cumulative.empty:
        ax.plot(
            cumulative.index.to_numpy(),
            cumulative.to_numpy(),
            color=PALETTE["blue"],
            linewidth=1.8,
            label="Track-B MCP simulation",
        )

    baseline_specs = [
        ("Make the market", SPLIT_DIR / "oos_results" / "make_the_market_OOS.parquet"),
        ("Implied favorite", SPLIT_DIR / "oos_results" / "implied_favorite_OOS.parquet"),
    ]
    for label, path in baseline_specs:
        if not path.exists():
            continue
        baseline = pd.read_parquet(path)
        no_signal = (
            baseline["no_signal"].fillna(False).astype(bool)
            if "no_signal" in baseline.columns
            else pd.Series(False, index=baseline.index)
        )
        pnl = pd.to_numeric(baseline["net_pnl_cents"], errors="coerce").where(~no_signal, 0.0).fillna(0.0)
        daily = (
            baseline.assign(_pnl=pnl, _date=pd.to_datetime(baseline["event_date"]))
            .groupby("_date", sort=True)["_pnl"]
            .sum()
            .cumsum()
        )
        ax.plot(
            daily.index.to_numpy(),
            daily.to_numpy(),
            linewidth=1.2,
            linestyle="--",
            color=PALETTE["orange"] if "market" in label.lower() else PALETTE["grey"],
            label=label,
        )

    ax.axhline(0, color=PALETTE["grey"], linestyle=":", linewidth=0.8)
    ax.set_title("Track-B vs Market Baselines")
    ax.set_xlabel("Event date")
    ax.set_ylabel("Cumulative net PnL (cents)")
    ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: dict[str, object]) -> None:
    print("\n=== MCP CHALLENGE SIMULATION ===")
    print(
        f"Period: {summary['start_date']} to {summary['end_date']} "
        f"({summary['n_calendar_days']} calendar days)"
    )
    print("\nPERFORMANCE")
    print(f"  Total PnL:          ${summary['total_pnl_dollars']:.2f}")
    print(f"  Final bankroll:     ${summary['final_bankroll_cents'] / 100:.2f}")
    ci_lo = summary["sharpe_ci_lo"]
    ci_hi = summary["sharpe_ci_hi"]
    if ci_lo is not None and ci_hi is not None:
        print(
            f"  Sharpe (annual):    {summary['sharpe_annual']:.2f} "
            f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        )
    else:
        print(f"  Sharpe (annual):    {summary['sharpe_annual']:.2f}")
    print(f"  Sortino (annual):   {summary['sortino_annual']:.2f}")
    psr = summary["psr_0"]
    min_trl = summary["min_trl_0"]
    print(f"  PSR(SR*=0):         {psr:.1%}" if psr is not None else "  PSR(SR*=0):         n/a")
    print(f"  MinTRL(SR*=0):      {min_trl:.0f} days" if min_trl is not None else "  MinTRL(SR*=0):      n/a")

    print("\nRISK")
    print(
        f"  Max drawdown:       ${summary['max_drawdown_cents'] / 100:.2f} "
        f"({summary['max_drawdown_pct']:.1%})"
    )
    print(f"  Min bankroll:       ${summary['min_bankroll_cents'] / 100:.2f}")
    print(f"  Eliminated:         {summary['eliminated']}")
    print(f"  Worst losing streak: {summary['worst_losing_streak']} days")

    print("\nACTIVITY")
    print(f"  Total trades:       {summary['n_trades']}")
    print(
        f"  Trading days:       {summary['n_trading_days']}/{summary['n_calendar_days']}"
    )
    print(f"  Trades/day:         {summary['trades_per_day']:.1f}")
    pass_fail = "PASS" if summary["meets_80_trade_min"] else "FAIL"
    print(
        f"  Projected 60-day:   {summary['projected_60d_trades']:.0f} trades "
        f"({pass_fail} >= 80)"
    )

    print("\nQUALITY")
    print(f"  Win rate:           {summary['win_rate']:.1%}")
    mean_edge = summary["mean_edge"]
    print(f"  Mean edge:          {mean_edge:.3f}" if mean_edge is not None else "  Mean edge:          n/a")
    pf = summary["profit_factor"]
    print(f"  Profit factor:      {pf:.2f}" if pf is not None else "  Profit factor:      n/a")

    print("\nEXIT MIX")
    print(f"  Profit target 15c:  {summary['n_profit_target_exits']}")
    print(f"  Settlement:         {summary['n_settlement_exits']}")
    conc = summary.get("pnl_concentration_top3")
    if conc is not None:
        print(f"  PnL top-3 days:     {conc:.1%} of trading-day PnL")

    per_city = summary.get("per_city", {})
    if per_city:
        print("\nPER-CITY")
        for city in sorted(per_city):
            row = per_city[city]
            print(
                f"  {city:18s} trades={int(row['n_trades']):3d}  "
                f"PnL=${row['total_pnl'] / 100:.2f}  "
                f"win={row['win_rate']:.1%}  "
                f"pt={int(row['n_profit_target'])}  settle={int(row['n_settlement'])}"
            )

    gonogo = summary["gonogo"]
    print("\nMCP GO/NO-GO")
    print(f"  1. Not eliminated:            {'PASS' if gonogo['not_eliminated'] else 'FAIL'}")
    print(
        f"  2. Projected trades >= 80:    "
        f"{'PASS' if gonogo['projected_trades_80'] else 'FAIL'}"
    )
    print(f"  3. Sharpe > 0:                {'PASS' if gonogo['sharpe_positive'] else 'FAIL'}")
    print(
        f"  4. Max DD < 30%:              "
        f"{'PASS' if gonogo['max_dd_under_30pct'] else 'FAIL'}"
    )
    print(
        f"  5. Win rate > 50%:            "
        f"{'PASS' if gonogo['win_rate_above_50pct'] else 'FAIL'}"
    )
    print(
        f"  6. PSR(0) > 95%:              "
        f"{'PASS' if gonogo['psr_above_95pct'] else 'FAIL'}"
    )


def generate_figures(
    result: dict[str, object],
    market_df: pd.DataFrame,
    calendar_dates: list[str],
    summary: dict[str, object],
    figure_dir: Path,
) -> None:
    daily_log: pd.DataFrame = result["daily_log"]
    trades: pd.DataFrame = result["trades"]
    plot_equity_curve(daily_log, summary, figure_dir / "equity_curve.png")
    plot_daily_pnl(daily_log, figure_dir / "daily_pnl.png")
    plot_trade_heatmap(trades, market_df, calendar_dates, figure_dir / "trade_heatmap.png")
    plot_cumulative_vs_baselines(trades, figure_dir / "cumulative_vs_baselines.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP challenge simulation.")
    parser.add_argument("--edge-threshold", type=float, default=0.037)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config.exists():
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)
        print(f"Loaded deploy config from {args.config} (edge_threshold={config.get('edge_threshold')})")
    else:
        print(f"WARNING: config not found at {args.config}; using CLI defaults.")

    market_df, forecasts_df, calendar_dates, start_date, end_date = load_simulation_data()
    print(
        f"Simulation period: {start_date} to {end_date} "
        f"({len(calendar_dates)} calendar days)"
    )

    signals = enrich_signals_with_entry_time(
        generate_signals(
            market_df,
            forecasts_df,
            "track_b_flat",
            exclude_cities=LOW_OOS_COVERAGE_CITIES,
        ),
        market_df,
    )
    selected = apply_selection(signals, "edge_threshold", args.edge_threshold)

    result = run_mcp_backtest(selected, calendar_dates, market_df, profit_target=0.15)
    summary = build_summary(result, calendar_dates, start_date, end_date)
    result["summary"] = summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["trades"].to_parquet(args.output_dir / "trades.parquet", index=False)
    result["daily_log"].to_parquet(args.output_dir / "daily_log.parquet", index=False)
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.no_figures:
        generate_figures(result, market_df, calendar_dates, summary, args.output_dir)

    print_summary(summary)


if __name__ == "__main__":
    main()
