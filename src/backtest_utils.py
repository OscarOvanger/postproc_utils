"""Shared backtest evaluation helpers reused across all baseline strategies."""

from __future__ import annotations

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.special import erfinv
    from scipy.stats import kurtosis, norm, skew
except ImportError:  # pragma: no cover - fallback when scipy is unavailable
    erfinv = None
    norm = None
    skew = None
    kurtosis = None


TRADING_DAYS_PER_YEAR = 252


def bucket_sort_key(bucket_label: str) -> tuple[int, float]:
    """
    Return a sort key that orders temperature buckets by their numeric bound.

    Ordering: LESS_THAN buckets first (by upper bound ascending), then RANGE
    buckets (by lower bound ascending), then GREATER_THAN buckets last. The
    numeric bound is parsed from the ``bucket_label`` string itself (e.g.
    ``<83``, ``83-84``, ``>97``) so this never falls back to a lexicographic
    sort.
    """
    label = str(bucket_label).strip()
    numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", label)]

    if label.startswith("<"):
        bound = numbers[0] if numbers else float("inf")
        return (0, bound)
    if label.startswith(">"):
        bound = numbers[0] if numbers else float("inf")
        return (2, bound)
    lower = numbers[0] if numbers else float("inf")
    return (1, lower)


def _day_index_columns(results_df: pd.DataFrame) -> list[str]:
    """Return the columns that identify one trading day in a results frame."""
    if "city" in results_df.columns and "event_date" in results_df.columns:
        return ["city", "event_date"]
    if "event_date" in results_df.columns:
        return ["event_date"]
    raise ValueError("results_df must contain an event_date column")


def daily_returns(
    results_df: pd.DataFrame,
    pnl_col: str = "net_pnl_cents",
    capital: float = 100.0,
) -> pd.Series:
    """
    Convert a per-trade results dataframe into a daily return series.

    ``capital`` is the at-risk capital per day in cents; each day's return is
    its summed net PnL divided by ``capital``. Days where ``no_signal`` is True
    contribute a return of 0 (no position). Multiple trades on the same day
    (e.g. distribution-copy or sell-longshots) are aggregated by summing their
    PnL. The returned series is indexed by event_date (a (city, event_date)
    MultiIndex when a ``city`` column is present).
    """
    if results_df.empty:
        return pd.Series(dtype=float, name="daily_return")
    if capital == 0:
        raise ValueError("capital must be non-zero")

    df = results_df.copy()
    index_cols = _day_index_columns(df)

    if "no_signal" in df.columns:
        no_signal_mask = df["no_signal"].fillna(False).astype(bool)
    else:
        no_signal_mask = pd.Series(False, index=df.index)

    pnl = pd.to_numeric(df[pnl_col], errors="coerce")
    pnl = pnl.where(~no_signal_mask, 0.0).fillna(0.0)
    df = df.assign(_pnl=pnl)

    daily_pnl = df.groupby(index_cols, sort=True)["_pnl"].sum()
    returns = daily_pnl / capital

    if len(index_cols) == 1:
        returns.index = returns.index.get_level_values(0) if isinstance(
            returns.index, pd.MultiIndex
        ) else returns.index
    returns.name = "daily_return"
    return returns


def cumulative_pnl_plot(
    results_dict: dict,
    baseline_names: list[str],
    title: str = "",
    capital: float = 100.0,
) -> plt.Figure:
    """
    Plot cumulative net PnL over time for selected baselines.

    results_dict: {baseline_name: results_df}
    Each results_df must have event_date and net_pnl_cents columns.
    Days with no_signal are treated as zero PnL.
    Returns matplotlib Figure without calling plt.show().
    """
    if capital == 0:
        raise ValueError("capital must be non-zero")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    plotted = False
    for baseline_name in baseline_names:
        if baseline_name not in results_dict:
            raise KeyError(f"Missing results for baseline: {baseline_name}")

        results_df = results_dict[baseline_name].copy()
        if results_df.empty:
            continue
        missing = {"event_date", "net_pnl_cents"}.difference(results_df.columns)
        if missing:
            raise ValueError(
                f"{baseline_name} results missing required columns: {sorted(missing)}"
            )

        index_cols = _day_index_columns(results_df)
        if "no_signal" in results_df.columns:
            no_signal_mask = results_df["no_signal"].fillna(False).astype(bool)
        else:
            no_signal_mask = pd.Series(False, index=results_df.index)

        pnl = pd.to_numeric(results_df["net_pnl_cents"], errors="coerce")
        pnl = pnl.where(~no_signal_mask, 0.0).fillna(0.0)
        daily_pnl = (
            results_df.assign(_pnl=pnl)
            .groupby(index_cols, sort=True)["_pnl"]
            .sum()
            .reset_index()
        )
        daily_pnl["event_date"] = pd.to_datetime(daily_pnl["event_date"])
        daily_by_date = daily_pnl.groupby("event_date", sort=True)["_pnl"].sum()
        cumulative = daily_by_date.cumsum() / capital
        ax.plot(
            cumulative.index.to_numpy(),
            cumulative.to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=3.0,
            label=baseline_name.replace("_", " "),
        )
        plotted = True

    ax.axhline(0, color="#8A8A8A", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Event date")
    ax.set_ylabel(f"Cumulative net PnL / ${capital:0.0f}")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def _n_trades_and_no_signal(results_df: pd.DataFrame) -> tuple[int, int]:
    """Return (n_trade_days, n_no_signal_days) from a results frame."""
    if results_df is None or results_df.empty:
        return 0, 0
    index_cols = _day_index_columns(results_df)
    if "no_signal" in results_df.columns:
        per_day = results_df.groupby(index_cols, sort=False)["no_signal"].apply(
            lambda values: bool(values.fillna(False).astype(bool).all())
        )
        n_no_signal = int(per_day.sum())
        n_trades = int((~per_day).sum())
    else:
        n_trades = int(results_df.groupby(index_cols, sort=False).ngroups)
        n_no_signal = 0
    return n_trades, n_no_signal


def _normal_cdf(x: float) -> float:
    if norm is not None:
        return float(norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _erfinv(x: float) -> float:
    if erfinv is not None:
        return float(erfinv(x))
    # Winitzki approximation for inverse error function
    a = 0.147
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), x)


def _sample_skew(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.shape[0] < 3:
        return float("nan")
    if skew is not None:
        return float(skew(arr, bias=False))
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(((arr - mean) / std) ** 3))


def _sample_excess_kurtosis(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.shape[0] < 4:
        return float("nan")
    if kurtosis is not None:
        return float(kurtosis(arr, fisher=True, bias=False))
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)


def _psr_denominator(sr_hat: float, skewness: float, excess_kurt: float) -> float:
    adjustment = 1.0 - skewness * sr_hat + ((excess_kurt - 1.0) / 4.0) * sr_hat**2
    if adjustment <= 0:
        return float("nan")
    return math.sqrt(adjustment)


def _probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_star: float,
    n_obs: int,
    skewness: float,
    excess_kurt: float,
) -> float:
    if n_obs < 2 or not math.isfinite(sr_hat):
        return float("nan")
    if sr_star == 1.0 and sr_hat <= sr_star:
        return float("nan")
    denom = _psr_denominator(sr_hat, skewness, excess_kurt)
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr_hat - sr_star) * math.sqrt(n_obs - 1) / denom
    return _normal_cdf(z)


def _min_track_record_length(
    sr_hat: float,
    sr_star: float,
    skewness: float,
    excess_kurt: float,
    z_alpha: float = 1.645,
) -> float:
    if not math.isfinite(sr_hat) or sr_hat <= sr_star:
        return float("nan")
    denom = sr_hat - sr_star
    adjustment = 1.0 - skewness * sr_hat + ((excess_kurt - 1.0) / 4.0) * sr_hat**2
    return 1.0 + adjustment * (z_alpha / denom) ** 2


def sharpe_stats(returns: pd.Series) -> dict:
    """
    Compute summary risk/return statistics for a daily return series.

    Returns a dict with: ``n_days``, ``n_trades``, ``n_no_signal``,
    ``mean_return``, ``std_return``, ``sharpe_daily``, ``sharpe_annual``
    (annualised by sqrt(252)), ``sharpe_se`` (Lo 2002:
    sqrt((1 + 0.5*SR^2) / T)), ``sharpe_ci_low``/``sharpe_ci_high`` (95% CI on
    the annualised Sharpe), ``max_drawdown``, ``sortino_annual``, and
    ``lag1_autocorr``. Any statistic that cannot be computed (e.g. zero std)
    is returned as NaN. ``n_trades`` here counts days with a non-zero return.
    """
    nan = float("nan")
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    n_days = int(values.shape[0])

    stats = {
        "n_days": n_days,
        "n_trades": int((values != 0).sum()),
        "n_no_signal": int((values == 0).sum()),
        "mean_return": nan,
        "std_return": nan,
        "sharpe_daily": nan,
        "sharpe_annual": nan,
        "sharpe_se": nan,
        "sharpe_ci_low": nan,
        "sharpe_ci_high": nan,
        "max_drawdown": nan,
        "sortino_annual": nan,
        "lag1_autocorr": nan,
        "skewness": nan,
        "kurtosis": nan,
        "PSR_0": nan,
        "PSR_1": nan,
        "MinTRL_0": nan,
    }

    if n_days == 0:
        return stats

    mean_return = float(values.mean())
    stats["mean_return"] = mean_return

    if n_days < 2:
        return stats

    std_return = float(values.std(ddof=1))
    stats["std_return"] = std_return

    annual_factor = np.sqrt(TRADING_DAYS_PER_YEAR)

    if std_return > 0:
        sharpe_daily = mean_return / std_return
        sharpe_annual = sharpe_daily * annual_factor
        sharpe_se_daily = np.sqrt((1.0 + 0.5 * sharpe_daily**2) / n_days)
        sharpe_se_annual = sharpe_se_daily * annual_factor
        stats["sharpe_daily"] = float(sharpe_daily)
        stats["sharpe_annual"] = float(sharpe_annual)
        stats["sharpe_se"] = float(sharpe_se_annual)
        stats["sharpe_ci_low"] = float(sharpe_annual - 1.96 * sharpe_se_annual)
        stats["sharpe_ci_high"] = float(sharpe_annual + 1.96 * sharpe_se_annual)

    cumulative = values.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    stats["max_drawdown"] = float(drawdown.min())

    downside = values[values < 0]
    if downside.shape[0] > 0:
        downside_std = float(downside.std(ddof=1))
        if math.isfinite(downside_std) and downside_std > 0:
            stats["sortino_annual"] = float(mean_return / downside_std * annual_factor)

    if n_days >= 3:
        lag1 = values.autocorr(lag=1)
        stats["lag1_autocorr"] = (
            float(lag1) if lag1 is not None and math.isfinite(lag1) else nan
        )

    if std_return > 0:
        skewness = _sample_skew(values)
        excess_kurt = _sample_excess_kurtosis(values)
        stats["skewness"] = skewness
        stats["kurtosis"] = excess_kurt
        if math.isfinite(skewness) and math.isfinite(excess_kurt):
            sharpe_daily = mean_return / std_return
            stats["PSR_0"] = _probabilistic_sharpe_ratio(
                sharpe_daily, 0.0, n_days, skewness, excess_kurt
            )
            stats["PSR_1"] = _probabilistic_sharpe_ratio(
                sharpe_daily, 1.0, n_days, skewness, excess_kurt
            )
            stats["MinTRL_0"] = _min_track_record_length(
                sharpe_daily, 0.0, skewness, excess_kurt
            )

    return stats


def bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 2000,
    block_size: int | None = None,
) -> dict:
    """
    Block bootstrap confidence interval on annualised Sharpe.

    ``block_size`` defaults to ``max(1, int(sqrt(len(returns))))``. Uses a
    circular block bootstrap: sample blocks with replacement, concatenate until
    at least the original length, trim to exact length, and compute annualised
    Sharpe on each bootstrap sample.
    """
    nan = float("nan")
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy(
        dtype=float
    )
    n_obs = int(values.shape[0])
    if block_size is None:
        block_size = max(1, int(math.sqrt(n_obs))) if n_obs else 1
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    result = {
        "sharpe_boot_mean": nan,
        "sharpe_boot_se": nan,
        "sharpe_boot_ci_low": nan,
        "sharpe_boot_ci_high": nan,
        "n_boot": int(n_boot),
        "block_size": block_size,
    }
    if n_obs < 2:
        return result

    rng = np.random.default_rng(0)
    boot_values: list[float] = []
    n_blocks = int(math.ceil(n_obs / block_size))
    annual_factor = math.sqrt(TRADING_DAYS_PER_YEAR)

    for _ in range(n_boot):
        starts = rng.integers(0, n_obs, size=n_blocks)
        sample_parts = [
            values[(start + np.arange(block_size)) % n_obs] for start in starts
        ]
        sample = np.concatenate(sample_parts)[:n_obs]
        sample_std = float(sample.std(ddof=1))
        if sample_std > 0:
            boot_values.append(float(sample.mean() / sample_std * annual_factor))

    if not boot_values:
        return result

    boot = np.asarray(boot_values, dtype=float)
    result["sharpe_boot_mean"] = float(np.mean(boot))
    result["sharpe_boot_se"] = float(np.std(boot, ddof=1)) if boot.shape[0] > 1 else nan
    result["sharpe_boot_ci_low"] = float(np.percentile(boot, 2.5))
    result["sharpe_boot_ci_high"] = float(np.percentile(boot, 97.5))
    return result


def deflated_sharpe(returns: pd.Series, n_variants: int) -> dict:
    """
    Compute the deflated Sharpe ratio following Bailey and Lopez de Prado (2014).
    """
    nan = float("nan")
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    n_days = int(values.shape[0])
    result = {
        "n_variants": int(n_variants),
        "e_max_sr": nan,
        "sr_deflated": nan,
        "psr_deflated": nan,
    }
    if n_days < 2 or n_variants < 1:
        return result

    mean_return = float(values.mean())
    std_return = float(values.std(ddof=1))
    if std_return == 0:
        return result

    sr_hat = mean_return / std_return
    skewness = _sample_skew(values)
    excess_kurt = _sample_excess_kurtosis(values)
    if not math.isfinite(skewness) or not math.isfinite(excess_kurt):
        return result

    variance_adj = max(
        1.0 - skewness * sr_hat + ((excess_kurt - 1.0) / 4.0) * sr_hat**2,
        0.0,
    )
    if n_variants == 1:
        e_max_sr = 0.0
    else:
        e_max_sr = (
            math.sqrt(2.0)
            * _erfinv((n_variants - 1) / n_variants)
            * (1.0 / math.sqrt(n_days))
            * math.sqrt(variance_adj)
        )
    result["e_max_sr"] = float(e_max_sr)
    result["sr_deflated"] = float(sr_hat - e_max_sr)
    result["psr_deflated"] = _probabilistic_sharpe_ratio(
        sr_hat, e_max_sr, n_days, skewness, excess_kurt
    )
    return result


def _entry_time_column(results_df: pd.DataFrame) -> str:
    if "entry_snapshot_time" in results_df.columns:
        return "entry_snapshot_time"
    if "entry_time" in results_df.columns:
        return "entry_time"
    if "crossing_snapshot_time" in results_df.columns:
        return "crossing_snapshot_time"
    raise ValueError(
        "results_df must contain entry_snapshot_time, entry_time, or crossing_snapshot_time"
    )


def _datetime_join_key(values: pd.Series) -> pd.Series:
    """Return timezone-normalized datetime keys for joining trade and market rows."""
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(None)


def bucket_decile_breakdown(
    results_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance breakdown by the entry price of the bucket traded.

    For each trade, look up ``yes_mid_close`` for the entered bucket at the
    entry snapshot. NO-side longshot trades use the sold bucket's YES price as
    the price axis.
    """
    decile_labels = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]
    empty = pd.DataFrame(
        {
            "price_decile": decile_labels,
            "n_trades": [0] * 10,
            "win_rate": [float("nan")] * 10,
            "mean_net_pnl_c": [float("nan")] * 10,
            "sharpe": [float("nan")] * 10,
        }
    )
    if results_df.empty:
        return empty

    entry_col = _entry_time_column(results_df)
    required_results = {"event_date", "city", "bucket_label", entry_col, "net_pnl_cents"}
    missing_results = required_results.difference(results_df.columns)
    if missing_results:
        raise ValueError(
            f"results_df is missing required columns: {sorted(missing_results)}"
        )
    required_market = {
        "event_date",
        "bucket_label",
        "snapshot_time_local",
        "yes_mid_close",
    }
    missing_market = required_market.difference(market_df.columns)
    if missing_market:
        raise ValueError(
            f"market_df is missing required columns: {sorted(missing_market)}"
        )

    trades = results_df.copy()
    if "no_signal" in trades.columns:
        trades = trades[~trades["no_signal"].fillna(False).astype(bool)].copy()
    trades = trades.dropna(subset=[entry_col, "bucket_label"])
    if trades.empty:
        return empty

    trades["_event_date_key"] = pd.to_datetime(trades["event_date"]).dt.date.astype(str)
    trades["_entry_time_key"] = _datetime_join_key(trades[entry_col])
    trades["_bucket_key"] = trades["bucket_label"].astype(str)
    trades["_city_key"] = trades["city"].astype(str)

    market = market_df.copy()
    market["_event_date_key"] = pd.to_datetime(market["event_date"]).dt.date.astype(str)
    market["_entry_time_key"] = _datetime_join_key(market["snapshot_time_local"])
    market["_bucket_key"] = market["bucket_label"].astype(str)
    if "city" in market.columns:
        market["_city_key"] = market["city"].astype(str)
    elif "source_city_folder" in market.columns:
        market["_city_key"] = market["source_city_folder"].astype(str)
    else:
        raise ValueError("market_df must contain city or source_city_folder")

    market_lookup = market[
        [
            "_city_key",
            "_event_date_key",
            "_entry_time_key",
            "_bucket_key",
            "yes_mid_close",
        ]
    ].drop_duplicates(
        ["_city_key", "_event_date_key", "_entry_time_key", "_bucket_key"]
    )

    merged = trades.merge(
        market_lookup,
        on=["_city_key", "_event_date_key", "_entry_time_key", "_bucket_key"],
        how="left",
    )
    merged["entry_yes_price"] = pd.to_numeric(
        merged["yes_mid_close"], errors="coerce"
    )
    merged = merged.dropna(subset=["entry_yes_price"])
    if merged.empty:
        return empty

    merged["price_decile"] = pd.cut(
        merged["entry_yes_price"].clip(lower=0.0, upper=1.0),
        bins=np.linspace(0.0, 1.0, 11),
        labels=decile_labels,
        include_lowest=True,
        right=True,
    )
    merged["net_pnl_cents"] = pd.to_numeric(
        merged["net_pnl_cents"], errors="coerce"
    )
    if "resolved_correctly" in merged.columns:
        wins = merged["resolved_correctly"].astype("boolean")
    else:
        wins = merged["net_pnl_cents"] > 0
    merged["_win"] = wins.astype(float)

    rows: list[dict[str, object]] = []
    for label in decile_labels:
        decile = merged[merged["price_decile"].astype(str).eq(label)]
        if decile.empty:
            rows.append(
                {
                    "price_decile": label,
                    "n_trades": 0,
                    "win_rate": float("nan"),
                    "mean_net_pnl_c": float("nan"),
                    "sharpe": float("nan"),
                }
            )
            continue
        pnl = decile["net_pnl_cents"].dropna()
        std = float(pnl.std(ddof=1)) if pnl.shape[0] >= 2 else float("nan")
        sharpe = (
            float(pnl.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))
            if math.isfinite(std) and std > 0
            else float("nan")
        )
        rows.append(
            {
                "price_decile": label,
                "n_trades": int(decile.shape[0]),
                "win_rate": float(decile["_win"].mean()),
                "mean_net_pnl_c": float(pnl.mean()) if not pnl.empty else float("nan"),
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows)


def is_oos_comparison(is_stats_csv: str, oos_stats_csv: str) -> pd.DataFrame:
    """
    Load IS and OOS full-stats CSVs and produce a side-by-side comparison.
    """
    is_stats = pd.read_csv(is_stats_csv)
    oos_stats = pd.read_csv(oos_stats_csv)
    required = {
        "Baseline",
        "Sharpe",
        "Sharpe_CI_low",
        "Sharpe_CI_high",
        "PSR_0",
        "N_trades",
        "NoSignal_pct",
    }
    for name, frame in [("IS", is_stats), ("OOS", oos_stats)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} stats CSV is missing columns: {sorted(missing)}")

    merged = is_stats.merge(
        oos_stats,
        on="Baseline",
        how="outer",
        suffixes=("_IS", "_OOS"),
    )
    comparison = pd.DataFrame(
        {
            "Baseline": merged["Baseline"],
            "IS_Sharpe": merged["Sharpe_IS"],
            "IS_Sharpe_CI_low": merged["Sharpe_CI_low_IS"],
            "IS_Sharpe_CI_high": merged["Sharpe_CI_high_IS"],
            "OOS_Sharpe": merged["Sharpe_OOS"],
            "OOS_Sharpe_CI_low": merged["Sharpe_CI_low_OOS"],
            "OOS_Sharpe_CI_high": merged["Sharpe_CI_high_OOS"],
            "IS_PSR0": merged["PSR_0_IS"],
            "OOS_PSR0": merged["PSR_0_OOS"],
            "IS_n_trades": merged["N_trades_IS"],
            "OOS_n_trades": merged["N_trades_OOS"],
            "IS_NoSig_pct": merged["NoSignal_pct_IS"],
            "OOS_NoSig_pct": merged["NoSignal_pct_OOS"],
        }
    )
    comparison["Sharpe_decay"] = (
        comparison["IS_Sharpe"] - comparison["OOS_Sharpe"]
    )
    return comparison


def full_stats_table(
    results_dict: dict[str, pd.DataFrame],
    n_variants: int,
    capital: float = 100.0,
) -> pd.DataFrame:
    """
    Build a wide summary table with one row per baseline.
    """
    rows: list[dict[str, object]] = []
    for baseline_name, results_df in results_dict.items():
        returns = daily_returns(results_df, capital=capital)
        stats = sharpe_stats(returns)
        deflated = deflated_sharpe(returns, n_variants=n_variants)
        n_trades, n_no_signal = _n_trades_and_no_signal(results_df)
        n_days = n_trades + n_no_signal
        no_signal_pct = (100.0 * n_no_signal / n_days) if n_days else float("nan")

        pnl = pd.to_numeric(results_df.get("net_pnl_cents"), errors="coerce")
        if "no_signal" in results_df.columns:
            pnl = pnl.where(~results_df["no_signal"].fillna(False), 0.0)
        index_cols = _day_index_columns(results_df)
        mean_net_pnl = float(
            results_df.assign(_pnl=pnl.fillna(0.0))
            .groupby(index_cols, sort=True)["_pnl"]
            .sum()
            .mean()
        )

        fee_col = (
            "fee_cents"
            if "fee_cents" in results_df.columns
            else "total_fee_cents"
            if "total_fee_cents" in results_df.columns
            else None
        )
        if fee_col is not None and "gross_pnl_cents" in results_df.columns:
            total_fees = float(pd.to_numeric(results_df[fee_col], errors="coerce").sum())
            total_gross = float(
                pd.to_numeric(results_df["gross_pnl_cents"], errors="coerce").sum()
            )
            fee_drag_pct = (
                100.0 * abs(total_fees) / abs(total_gross) if total_gross != 0 else float("nan")
            )
        else:
            fee_drag_pct = float("nan")

        rows.append(
            {
                "Baseline": baseline_name,
                "N_days": n_days,
                "N_trades": n_trades,
                "NoSignal_pct": no_signal_pct,
                "MeanNetPnL_c": mean_net_pnl,
                "Sharpe": stats["sharpe_annual"],
                "Sharpe_CI_low": stats["sharpe_ci_low"],
                "Sharpe_CI_high": stats["sharpe_ci_high"],
                "PSR_0": stats["PSR_0"],
                "PSR_1": stats["PSR_1"],
                "MinTRL_0": stats["MinTRL_0"],
                "SR_deflated": deflated["sr_deflated"],
                "MaxDrawdown": stats["max_drawdown"],
                "Sortino": stats["sortino_annual"],
                "Lag1_autocorr": stats["lag1_autocorr"],
                "Fee_drag_pct": fee_drag_pct,
            }
        )

    return pd.DataFrame(rows)


def print_summary_table(
    baseline_name: str,
    results_df: pd.DataFrame,
    capital: float = 100.0,
) -> None:
    """
    Print a clean one-block performance summary for a baseline result frame.

    Reports the baseline name, number of days, number of trade days, percent
    no-signal days, mean net PnL in cents, annualised Sharpe with its 95%
    confidence interval, max drawdown, and fee drag (total fees / total gross
    PnL). ``capital`` is the per-day at-risk capital in cents used to scale
    returns.
    """
    returns = daily_returns(results_df, capital=capital)
    stats = sharpe_stats(returns)
    n_trades, n_no_signal = _n_trades_and_no_signal(results_df)
    n_days = n_trades + n_no_signal
    no_signal_pct = (100.0 * n_no_signal / n_days) if n_days else float("nan")

    if "net_pnl_cents" in results_df.columns:
        mean_net_pnl = float(
            pd.to_numeric(results_df["net_pnl_cents"], errors="coerce").mean()
        )
    else:
        mean_net_pnl = float("nan")

    fee_col = "fee_cents" if "fee_cents" in results_df.columns else (
        "total_fee_cents" if "total_fee_cents" in results_df.columns else None
    )
    if fee_col is not None and "gross_pnl_cents" in results_df.columns:
        total_fees = float(pd.to_numeric(results_df[fee_col], errors="coerce").sum())
        total_gross = float(
            pd.to_numeric(results_df["gross_pnl_cents"], errors="coerce").sum()
        )
        fee_drag = total_fees / total_gross if total_gross != 0 else float("nan")
    else:
        fee_drag = float("nan")

    line = "=" * 60
    print(line)
    print(f"Baseline: {baseline_name}")
    print(line)
    print(f"  N days              : {n_days}")
    print(f"  N trades            : {n_trades}")
    print(f"  % no-signal         : {no_signal_pct:0.1f}%")
    print(f"  Mean net PnL (cents): {mean_net_pnl:0.4f}")
    print(
        f"  Sharpe (annual)     : {stats['sharpe_annual']:0.3f} "
        f"[95% CI {stats['sharpe_ci_low']:0.3f}, {stats['sharpe_ci_high']:0.3f}]"
    )
    print(f"  Max drawdown        : {stats['max_drawdown']:0.4f}")
    print(f"  Fee drag (fees/gross): {fee_drag:0.4f}")
    print(line)


def disagreement_diagnostic(results_df: pd.DataFrame) -> dict:
    """
    Compute Track-J vs market disagreement statistics.

    `results_df` must have: agrees_with_market, resolved_correctly,
    net_pnl_cents, no_signal columns.
    """
    required = {"agrees_with_market", "resolved_correctly", "net_pnl_cents", "no_signal"}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"results_df is missing required columns: {sorted(missing)}")

    trades = results_df[~results_df["no_signal"].fillna(False).astype(bool)].copy()
    trades["agrees_with_market"] = trades["agrees_with_market"].astype(bool)
    trades["resolved_correctly"] = trades["resolved_correctly"].astype(bool)
    trades["net_pnl_cents"] = pd.to_numeric(trades["net_pnl_cents"], errors="coerce")

    agree = trades[trades["agrees_with_market"]]
    disagree = trades[~trades["agrees_with_market"]]

    def win_rate(frame: pd.DataFrame) -> float:
        return float(frame["resolved_correctly"].mean()) if not frame.empty else float("nan")

    def mean_pnl(frame: pd.DataFrame) -> float:
        return float(frame["net_pnl_cents"].mean()) if not frame.empty else float("nan")

    def sharpe(frame: pd.DataFrame) -> float:
        if frame.empty:
            return float("nan")
        return float(sharpe_stats(daily_returns(frame))["sharpe_annual"])

    total_pnl = float(trades["net_pnl_cents"].sum()) if not trades.empty else 0.0
    disagree_pnl = float(disagree["net_pnl_cents"].sum()) if not disagree.empty else 0.0
    pct_pnl_from_disagree = (
        100.0 * disagree_pnl / total_pnl if total_pnl != 0 else float("nan")
    )

    return {
        "n_agree": int(agree.shape[0]),
        "n_disagree": int(disagree.shape[0]),
        "win_rate_agree": win_rate(agree),
        "win_rate_disagree": win_rate(disagree),
        "mean_pnl_agree": mean_pnl(agree),
        "mean_pnl_disagree": mean_pnl(disagree),
        "pct_pnl_from_disagree": pct_pnl_from_disagree,
        "sharpe_agree": sharpe(agree),
        "sharpe_disagree": sharpe(disagree),
    }
