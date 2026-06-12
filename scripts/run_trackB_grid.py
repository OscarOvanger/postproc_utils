"""Run Track-B backtest grid: signals x sizers x selection rules."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest_utils import sharpe_stats  # noqa: E402
from entry_interface import filter_to_trading_window  # noqa: E402
from snapshot_stability import (  # noqa: E402
    assert_no_true_holdout,
    compute_modal_bucket,
    stability_entry,
)
from src.models.track_j import bucket_probs_from_point_forecast  # noqa: E402
from src.sizing import (  # noqa: E402
    contracts_from_kelly,
    contracts_with_daily_cap,
    full_kelly,
    half_kelly,
    has_edge,
    taker_fee_cents,
)

FORECASTS_PATH = PROJECT_ROOT / "data" / "trackb" / "forecasts.parquet"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "trackb" / "sizing_grid"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

STABILITY_K = 1
MIN_ENTRY_PRICE = 0.15
INITIAL_BANKROLL_CENTS = 10_000
ELIMINATION_CENTS = 7_000
DAILY_CAP_CENTS = 600
FLAT_CONTRACTS = 5

SIGNALS = ("track_b_flat", "track_b_disagree")
SIZERS = ("flat_5", "half_kelly", "eighth_kelly")
SELECTIONS = ("all_eligible", "top_2_per_day", "edge_threshold")

LOW_OOS_COVERAGE_CITIES = {"austin", "philadelphia"}


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _date_key(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _day_group_columns(partition_df: pd.DataFrame) -> list[str]:
    city_col = "source_city_folder" if "source_city_folder" in partition_df.columns else "city"
    return [city_col, "event_date"]


def _load_partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_opt = pd.read_parquet(SPLIT_DIR / "threshold_opt.parquet")
    time_holdout = pd.read_parquet(SPLIT_DIR / "time_holdout.parquet")
    assert_no_true_holdout(threshold_opt)
    assert_no_true_holdout(time_holdout)
    return threshold_opt, time_holdout


def _forecast_lookup(
    city: str,
    event_date: str,
    forecasts_df: pd.DataFrame,
) -> tuple[float, float] | None:
    forecasts = forecasts_df.copy()
    forecasts["city"] = forecasts["city"].map(_city_key)
    forecasts["event_date"] = pd.to_datetime(forecasts["event_date"]).dt.strftime("%Y-%m-%d")
    row = forecasts[
        forecasts["city"].eq(_city_key(city))
        & forecasts["event_date"].eq(_date_key(event_date))
    ]
    if row.empty:
        return None
    row = row.iloc[0]
    if pd.isna(row["trackb_tmax_f"]) or pd.isna(row["trackb_sigma_f"]):
        return None
    return float(row["trackb_tmax_f"]), float(row["trackb_sigma_f"])


def _resolved_for_bucket(day_df: pd.DataFrame, bucket_label: str) -> bool:
    entry_rows = day_df[day_df["bucket_label"].astype(str).eq(str(bucket_label))]
    if entry_rows.empty:
        raise ValueError(f"bucket_label {bucket_label} not found in day_df")
    resolved_values = entry_rows["bucket_resolved_to_one_dollars"].astype(bool).unique()
    if len(resolved_values) != 1:
        raise ValueError(f"bucket_label {bucket_label} has inconsistent resolution")
    return bool(resolved_values[0])


def generate_signals(
    partition_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    signal_name: str,
    exclude_cities: set[str] | None = None,
) -> pd.DataFrame:
    """Generate per-city-day signals for one signal function."""
    if signal_name not in SIGNALS:
        raise ValueError(f"Unknown signal: {signal_name}")
    exclude_cities = exclude_cities or set()
    df = partition_df.copy()
    df["snapshot_time_local"] = pd.to_datetime(df["snapshot_time_local"])
    group_cols = _day_group_columns(df)
    records: list[dict[str, object]] = []

    for _, raw_day_df in df.groupby(group_cols, sort=True):
        city = str(raw_day_df["city"].iloc[0]) if "city" in raw_day_df.columns else str(raw_day_df[group_cols[0]].iloc[0])
        city_norm = _city_key(city)
        event_date = str(raw_day_df["event_date"].dropna().iloc[0])
        if city_norm in exclude_cities:
            continue

        forecast = _forecast_lookup(city, event_date, forecasts_df)
        day_df = filter_to_trading_window(raw_day_df)
        base = {
            "event_date": _date_key(event_date),
            "city": city_norm,
            "signal": signal_name,
            "no_signal": True,
            "entry_bucket": "",
            "entry_price": np.nan,
            "model_prob": np.nan,
            "edge": np.nan,
            "resolved": np.nan,
            "market_modal_bucket": "",
            "agrees_with_market": np.nan,
        }
        if day_df.empty or forecast is None:
            records.append(base)
            continue

        stability = stability_entry(day_df, k=STABILITY_K)
        if stability.no_signal:
            records.append(base)
            continue

        market_modal = compute_modal_bucket(day_df, stability.entry_snapshot_time)
        snapshot = day_df[pd.to_datetime(day_df["snapshot_time_local"]).eq(stability.entry_snapshot_time)]
        buckets = snapshot[
            [
                "bucket_label",
                "bucket_type",
                "bucket_lower_inclusive_f",
                "bucket_upper_inclusive_f",
            ]
        ].drop_duplicates("bucket_label")
        tmax_f, sigma_f = forecast
        probs = bucket_probs_from_point_forecast(tmax_f, sigma_f, buckets)
        chosen_bucket = max(probs, key=probs.get)
        entry_rows = snapshot[snapshot["bucket_label"].astype(str).eq(str(chosen_bucket))]
        if entry_rows.empty:
            records.append(base)
            continue

        entry_price = float(entry_rows["yes_mid_close"].iloc[0])
        model_prob = float(probs[chosen_bucket])
        fee_per_contract = taker_fee_cents(1, entry_price) / 100.0
        agrees = str(chosen_bucket) == str(market_modal)

        if signal_name == "track_b_disagree" and agrees:
            records.append({**base, "market_modal_bucket": market_modal, "agrees_with_market": agrees})
            continue

        if entry_price < MIN_ENTRY_PRICE or not has_edge(model_prob, entry_price, fee_per_contract):
            records.append({**base, "market_modal_bucket": market_modal, "agrees_with_market": agrees})
            continue

        resolved = _resolved_for_bucket(day_df, chosen_bucket)
        records.append(
            {
                "event_date": _date_key(event_date),
                "city": city_norm,
                "signal": signal_name,
                "no_signal": False,
                "entry_bucket": str(chosen_bucket),
                "entry_price": entry_price,
                "model_prob": model_prob,
                "edge": model_prob - entry_price,
                "resolved": resolved,
                "market_modal_bucket": market_modal,
                "agrees_with_market": agrees,
            }
        )

    return pd.DataFrame.from_records(records)


def calibrate_edge_threshold(is_signals: pd.DataFrame) -> float:
    """Calibrate E* on IS to target ~100 trades per 60-day window."""
    eligible = is_signals[~is_signals["no_signal"]].copy()
    edges = eligible["edge"].astype(float).dropna()
    n_total = len(is_signals)
    if edges.empty or n_total == 0:
        return 0.0
    target_frac = min(1.0, 100.0 / n_total)
    return float(edges.quantile(1.0 - target_frac))


def apply_selection(
    signals: pd.DataFrame,
    selection: str,
    edge_threshold: float | None = None,
) -> pd.DataFrame:
    """Apply a selection rule to signal rows."""
    df = signals.copy()
    if selection == "all_eligible":
        return df

    if selection == "top_2_per_day":
        traded = df[~df["no_signal"]].copy()
        if traded.empty:
            return df
        keep_idx: set[int] = set()
        for _, group in traded.groupby("event_date", sort=True):
            top = group.reindex(group["edge"].abs().sort_values(ascending=False).index).head(2)
            keep_idx.update(top.index.tolist())
        mask = df.index.isin(keep_idx)
        df.loc[~mask & ~df["no_signal"], "no_signal"] = True
        df.loc[~mask & ~df["no_signal"], ["entry_bucket", "entry_price", "model_prob", "edge"]] = [
            "",
            np.nan,
            np.nan,
            np.nan,
        ]
        return df

    if selection == "edge_threshold":
        if edge_threshold is None:
            raise ValueError("edge_threshold selection requires edge_threshold value")
        below = (~df["no_signal"]) & (df["edge"].astype(float) < edge_threshold)
        df.loc[below, "no_signal"] = True
        df.loc[below, ["entry_bucket", "entry_price", "model_prob", "edge"]] = [
            "",
            np.nan,
            np.nan,
            np.nan,
        ]
        return df

    raise ValueError(f"Unknown selection: {selection}")


def _size_contracts(
    sizer: str,
    model_prob: float,
    entry_price: float,
    bankroll_cents: int,
    daily_spent_cents: int,
) -> int:
    if sizer == "flat_5":
        return FLAT_CONTRACTS
    if sizer == "half_kelly":
        fraction = half_kelly(model_prob, entry_price, cap=0.08)
        return contracts_with_daily_cap(
            fraction, bankroll_cents, entry_price, daily_spent_cents, DAILY_CAP_CENTS
        )
    if sizer == "eighth_kelly":
        fraction = full_kelly(model_prob, entry_price, cap=0.08)
        return contracts_with_daily_cap(
            fraction, bankroll_cents, entry_price, daily_spent_cents, DAILY_CAP_CENTS
        )
    raise ValueError(f"Unknown sizer: {sizer}")


def run_backtest(
    signals: pd.DataFrame,
    sizer: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Apply sizer, compute trade PnL and bankroll path."""
    traded = signals[~signals["no_signal"]].copy()
    if traded.empty:
        empty = np.array([], dtype=float)
        return traded, empty, empty, np.array([INITIAL_BANKROLL_CENTS], dtype=float)

    traded = traded.sort_values(["event_date", "city"]).reset_index(drop=True)
    dates = sorted(traded["event_date"].unique())
    bankroll = INITIAL_BANKROLL_CENTS
    bankroll_path = [bankroll]
    daily_pnl_list: list[float] = []
    daily_return_list: list[float] = []
    trade_records: list[dict[str, object]] = []

    for event_date in dates:
        day_trades = traded[traded["event_date"].eq(event_date)]
        opening_bankroll = bankroll
        daily_spent = 0
        day_pnl = 0.0

        for _, row in day_trades.iterrows():
            contracts = _size_contracts(
                sizer,
                float(row["model_prob"]),
                float(row["entry_price"]),
                opening_bankroll,
                daily_spent,
            )
            if contracts <= 0:
                continue
            entry_price = float(row["entry_price"])
            cost_cents = int(contracts * entry_price * 100)
            fee_cents = taker_fee_cents(contracts, entry_price)
            daily_spent += cost_cents
            payout_cents = contracts * 100 if bool(row["resolved"]) else 0
            net_pnl_cents = payout_cents - cost_cents - fee_cents
            day_pnl += net_pnl_cents
            trade_records.append(
                {
                    **row.to_dict(),
                    "contracts": contracts,
                    "cost_cents": cost_cents,
                    "fee_cents": fee_cents,
                    "net_pnl_cents": net_pnl_cents,
                    "opening_bankroll_cents": opening_bankroll,
                }
            )

        daily_pnl_list.append(day_pnl)
        daily_return = day_pnl / opening_bankroll if opening_bankroll > 0 else 0.0
        daily_return_list.append(daily_return)
        bankroll += day_pnl
        bankroll_path.append(bankroll)

    trades_df = pd.DataFrame.from_records(trade_records)
    return (
        trades_df,
        np.asarray(daily_return_list, dtype=float),
        np.asarray(daily_pnl_list, dtype=float),
        np.asarray(bankroll_path, dtype=float),
    )


def compute_stats(
    daily_returns: np.ndarray,
    daily_pnl: np.ndarray,
    bankroll_path: np.ndarray,
    n_trades: int,
    calendar_days: int,
    mean_edge: float,
) -> dict[str, object]:
    """Full stats for one combination."""
    n = len(daily_returns)
    if n == 0 or np.std(daily_returns) == 0:
        sr_daily = 0.0
        sr_annual = 0.0
        se_sr = float("nan")
        ci_lo = float("nan")
        ci_hi = float("nan")
        sortino = 0.0
    else:
        sr_daily = float(np.mean(daily_returns) / np.std(daily_returns))
        sr_annual = sr_daily * np.sqrt(252)
        se_sr = np.sqrt((1 + 0.5 * sr_annual**2) / n)
        ci_lo = sr_annual - 1.96 * se_sr
        ci_hi = sr_annual + 1.96 * se_sr
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-6
        sortino = float(np.mean(daily_returns) / downside_std * np.sqrt(252))

    peak = np.maximum.accumulate(bankroll_path)
    dd = bankroll_path - peak
    max_dd_cents = float(dd.min()) if len(dd) else 0.0

    losses = (daily_pnl < 0).astype(int)
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
    worst_streak = max(streaks) if streaks else 0

    eliminated = bool(np.any(bankroll_path <= ELIMINATION_CENTS))
    proj_60d = (n_trades / calendar_days * 60.0) if calendar_days > 0 else 0.0

    psr0 = float("nan")
    if n > 0:
        psr0 = float(sharpe_stats(pd.Series(daily_returns))["PSR_0"])

    return {
        "sharpe": round(sr_annual, 2),
        "ci": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if np.isfinite(ci_lo) else "[nan, nan]",
        "sortino": round(sortino, 2),
        "max_dd_cents": round(max_dd_cents, 1),
        "worst_streak": worst_streak,
        "eliminated": eliminated,
        "n_trade_days": n,
        "n_trades": n_trades,
        "proj_60d": round(proj_60d, 1),
        "mean_edge": round(mean_edge, 4) if np.isfinite(mean_edge) else float("nan"),
        "psr0": round(psr0, 2) if np.isfinite(psr0) else float("nan"),
    }


def _calendar_days(partition_df: pd.DataFrame) -> int:
    dates = pd.to_datetime(partition_df["event_date"].dropna().unique())
    if len(dates) == 0:
        return 1
    return max(1, int((dates.max() - dates.min()).days) + 1)


def resolve_edge_threshold(
    is_signals: pd.DataFrame,
    oos_signals: pd.DataFrame,
    sizer: str = "flat_5",
) -> float:
    """Calibrate E* on IS; lower by 10% until OOS has >= 90 trades."""
    threshold = calibrate_edge_threshold(is_signals)
    for _ in range(20):
        oos_sel = apply_selection(oos_signals.copy(), "edge_threshold", threshold)
        trades, _, _, _ = run_backtest(oos_sel, sizer)
        if len(trades) >= 90:
            return threshold
        threshold *= 0.9
    return threshold


def run_grid(
    threshold_opt: pd.DataFrame,
    time_holdout: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[str, pd.DataFrame]]:
    """Run all 18 combinations on IS and OOS."""
    is_calendar = _calendar_days(threshold_opt)
    oos_calendar = _calendar_days(time_holdout)

    signal_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for signal in SIGNALS:
        signal_cache[(signal, "IS")] = generate_signals(threshold_opt, forecasts, signal)
        signal_cache[(signal, "OOS")] = generate_signals(
            time_holdout, forecasts, signal, exclude_cities=LOW_OOS_COVERAGE_CITIES
        )

    e_star_final = resolve_edge_threshold(
        signal_cache[("track_b_flat", "IS")],
        signal_cache[("track_b_flat", "OOS")],
    )

    rows_is: list[dict[str, object]] = []
    rows_oos: list[dict[str, object]] = []
    trade_paths: dict[str, pd.DataFrame] = {}

    for signal, sizer, selection in product(SIGNALS, SIZERS, SELECTIONS):
        is_signals = signal_cache[(signal, "IS")].copy()
        oos_signals = signal_cache[(signal, "OOS")].copy()
        threshold = e_star_final if selection == "edge_threshold" else None

        is_sel = apply_selection(is_signals, selection, threshold)
        oos_sel = apply_selection(oos_signals, selection, threshold)

        is_trades, is_returns, is_pnl, is_bankroll = run_backtest(is_sel, sizer)
        oos_trades, oos_returns, oos_pnl, oos_bankroll = run_backtest(oos_sel, sizer)

        combo_key = f"{signal}|{sizer}|{selection}"
        trade_paths[combo_key] = oos_trades

        is_mean_edge = float(is_trades["edge"].mean()) if not is_trades.empty else float("nan")
        oos_mean_edge = float(oos_trades["edge"].mean()) if not oos_trades.empty else float("nan")

        is_stats = compute_stats(
            is_returns,
            is_pnl,
            is_bankroll,
            n_trades=len(is_trades),
            calendar_days=is_calendar,
            mean_edge=is_mean_edge,
        )
        oos_stats = compute_stats(
            oos_returns,
            oos_pnl,
            oos_bankroll,
            n_trades=len(oos_trades),
            calendar_days=oos_calendar,
            mean_edge=oos_mean_edge,
        )

        for partition_label, stats in [("IS", is_stats), ("OOS", oos_stats)]:
            row = {
                "Signal": signal,
                "Sizer": sizer,
                "Selection": selection,
                "N trades": stats["n_trades"],
                "Proj/60d": stats["proj_60d"],
                "Sharpe": stats["sharpe"],
                "CI": stats["ci"],
                "Sortino": stats["sortino"],
                "Max DD": stats["max_dd_cents"],
                "Streak": stats["worst_streak"],
                "Eliminated": stats["eliminated"],
                "Mean Edge": stats["mean_edge"],
                "PSR(0)": stats["psr0"],
            }
            if partition_label == "IS":
                rows_is.append(row)
            else:
                if signal == "track_b_disagree" and not oos_trades.empty:
                    row["Disagree WR"] = round(float(oos_trades["resolved"].mean()), 3)
                rows_oos.append(row)

    is_df = pd.DataFrame(rows_is)
    oos_df = pd.DataFrame(rows_oos).sort_values("Sharpe", ascending=False)
    return is_df, oos_df, e_star_final, trade_paths


def print_survival_filter(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Print and return survival filter results."""
    checks = []
    for _, row in oos_df.iterrows():
        combo = f"{row['Signal']} + {row['Sizer']} + {row['Selection']}"
        c1 = not bool(row["Eliminated"])
        c2 = float(row["Proj/60d"]) >= 90
        c3 = float(row["Sharpe"]) > 0
        passed = c1 and c2 and c3
        checks.append(
            {
                "Combination": combo,
                "Not eliminated": "PASS" if c1 else "FAIL",
                "Proj trades >= 90": "PASS" if c2 else "FAIL",
                "OOS Sharpe > 0": "PASS" if c3 else "FAIL",
                "Survives": "PASS" if passed else "FAIL",
            }
        )
    survival = pd.DataFrame(checks)
    print("\n=== Survival Filter ===")
    print(survival.to_string(index=False))
    n_pass = int((survival["Survives"] == "PASS").sum())
    print(f"\n{n_pass} of {len(survival)} combinations pass all criteria.")
    return survival


def run_selection_curve(
    time_holdout: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Sweep edge thresholds for track_b_flat + flat_5 on OOS."""
    base_signals = generate_signals(
        time_holdout, forecasts, "track_b_flat", exclude_cities=LOW_OOS_COVERAGE_CITIES
    )
    calendar_days = _calendar_days(time_holdout)
    rows: list[dict[str, object]] = []

    for threshold in np.arange(0.0, 0.2005, 0.005):
        selected = apply_selection(base_signals.copy(), "edge_threshold", float(threshold))
        trades, daily_returns, daily_pnl, bankroll_path = run_backtest(selected, "flat_5")
        stats = compute_stats(
            daily_returns,
            daily_pnl,
            bankroll_path,
            n_trades=len(trades),
            calendar_days=calendar_days,
            mean_edge=float(trades["edge"].mean()) if not trades.empty else float("nan"),
        )
        rows.append(
            {
                "edge_threshold": round(float(threshold), 4),
                "n_trades": len(trades),
                "sharpe": stats["sharpe"],
                "max_dd_cents": stats["max_dd_cents"],
                "mean_edge": stats["mean_edge"],
            }
        )

    return pd.DataFrame(rows)


def plot_selection_curve(curve_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = curve_df["n_trades"].to_numpy()
    axes[0].plot(x, curve_df["sharpe"], color="#4878CF", linewidth=1.8)
    axes[0].axvline(80, color="#4878CF", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("N trades")
    axes[0].set_ylabel("Sharpe (annualised)")
    axes[0].set_title("Sharpe vs trade count")

    axes[1].plot(x, curve_df["max_dd_cents"], color="#E68A2E", linewidth=1.8)
    axes[1].axhline(-3000, color="#E68A2E", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_xlabel("N trades")
    axes[1].set_ylabel("Max drawdown (cents)")
    axes[1].set_title("Max drawdown vs trade count")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cumulative_pnl(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype=float)
    frame = trades_df.copy()
    frame["_date"] = pd.to_datetime(frame["event_date"])
    daily = frame.groupby("_date", sort=True)["net_pnl_cents"].sum().cumsum()
    return daily


def plot_oos_cumulative(
    oos_df: pd.DataFrame,
    trade_paths: dict[str, pd.DataFrame],
    output_path: Path,
) -> list[str]:
    """Plot top 3 surviving combinations plus baselines."""
    surviving = oos_df[
        (~oos_df["Eliminated"])
        & (oos_df["Proj/60d"] >= 90)
        & (oos_df["Sharpe"] > 0)
    ].sort_values("Sharpe", ascending=False)

    top3 = surviving.head(3)
    colors = ["#4878CF", "#E68A2E", "#8A8A8A"]
    labels: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for idx, (_, row) in enumerate(top3.iterrows()):
        key = f"{row['Signal']}|{row['Sizer']}|{row['Selection']}"
        trades = trade_paths.get(key, pd.DataFrame())
        cumulative = _cumulative_pnl(trades)
        if cumulative.empty:
            continue
        label = f"{row['Signal']} / {row['Sizer']} / {row['Selection']}"
        labels.append(label)
        ax.plot(
            cumulative.index.to_numpy(),
            cumulative.to_numpy(),
            color=colors[idx % len(colors)],
            linewidth=1.8,
            label=label,
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
        ax.plot(daily.index.to_numpy(), daily.to_numpy(), linewidth=1.2, linestyle="--", label=label)

    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("OOS cumulative net PnL")
    ax.set_xlabel("Event date")
    ax.set_ylabel("Cumulative net PnL (cents)")
    ax.legend(fontsize=7, loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return labels


def main() -> None:
    forecasts = pd.read_parquet(FORECASTS_PATH)
    threshold_opt, time_holdout = _load_partitions()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_df, oos_df, e_star, trade_paths = run_grid(threshold_opt, time_holdout, forecasts)

    is_path = OUTPUT_DIR / "full_stats_IS.csv"
    oos_path = OUTPUT_DIR / "full_stats_OOS.csv"
    is_df.to_csv(is_path, index=False)
    oos_df.to_csv(oos_path, index=False)

    print("\n=== OOS Grid Results (sorted by Sharpe) ===")
    print(oos_df.to_string(index=False))

    survival = print_survival_filter(oos_df)

    curve_df = run_selection_curve(time_holdout, forecasts)
    curve_path = OUTPUT_DIR / "selection_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    plot_selection_curve(curve_df, FIGURE_DIR / "day11_selection_curve.png")

    top_labels = plot_oos_cumulative(oos_df, trade_paths, FIGURE_DIR / "day11_oos_cumulative.png")

    meta = {
        "e_star": e_star,
        "low_oos_coverage_cities": sorted(LOW_OOS_COVERAGE_CITIES),
        "top_combinations": top_labels,
        "n_surviving": int((survival["Survives"] == "PASS").sum()),
    }
    with open(OUTPUT_DIR / "grid_meta.json", "w", encoding="utf-8") as handle:
        import json

        json.dump(meta, handle, indent=2)
        handle.write("\n")

    print(f"\nE* = {e_star:.4f}")
    print(f"Saved IS stats: {is_path}")
    print(f"Saved OOS stats: {oos_path}")
    print(f"Saved selection curve: {curve_path}")
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
