"""Re-settle extended Kalshi backtest trades using ASOS daily max (Wunderground equivalent)."""

from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
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

from compute_wunderground_bias import (  # noqa: E402
    _load_all_asos,
    _load_city_config,
    _load_cli,
)
from run_mcp_simulation import load_simulation_data  # noqa: E402
from src.trackj.build_ngboost_features import build_asos_daily_max_map  # noqa: E402

TRADES_PATH = PROJECT_ROOT / "data" / "mcp_simulation_extended" / "trades.parquet"
BIAS_DETAIL_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_bias_detail.parquet"
OUTPUT_JSON = PROJECT_ROOT / "data" / "polymarket" / "wunderground_resettlement.json"
OUTPUT_DETAIL = PROJECT_ROOT / "data" / "polymarket" / "wunderground_resettlement_detail.parquet"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "wunderground_resettlement_histogram.png"

BUCKET_COLS = [
    "bucket_label",
    "bucket_type",
    "bucket_lower_inclusive_f",
    "bucket_upper_inclusive_f",
]


def _city_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _parse_kalshi_bucket(label: str) -> tuple[str, float | None, float | None]:
    text = str(label).strip()
    if text.startswith(">"):
        return "GREATER_THAN", float(text[1:]), None
    if text.startswith("<"):
        return "LESS_THAN", None, float(text[1:])
    if "-" in text:
        lo, hi = text.split("-", 1)
        return "RANGE", float(lo), float(hi)
    raise ValueError(f"Unable to parse bucket label: {label!r}")


def _temp_in_bucket(
    temp: float,
    bucket_label: str,
    bucket_type: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> bool:
    t = int(round(temp))
    if bucket_type is None or (bucket_type == "LESS_THAN" and upper is None) or (
        bucket_type == "GREATER_THAN" and lower is None
    ):
        bucket_type, lower, upper = _parse_kalshi_bucket(bucket_label)
    if bucket_type == "RANGE":
        return float(lower) <= t <= float(upper)
    if bucket_type == "LESS_THAN":
        return t <= float(upper)
    if bucket_type == "GREATER_THAN":
        return t >= float(lower)
    return False


def _assign_bucket(temp: float, bucket_defs: pd.DataFrame) -> str | None:
    t = int(round(temp))
    for _, row in bucket_defs.iterrows():
        if _temp_in_bucket(
            t,
            str(row["bucket_label"]),
            str(row["bucket_type"]),
            pd.to_numeric(row.get("bucket_lower_inclusive_f"), errors="coerce"),
            pd.to_numeric(row.get("bucket_upper_inclusive_f"), errors="coerce"),
        ):
            return str(row["bucket_label"])
    return None


def _load_market_bucket_defs() -> dict[tuple[str, str], pd.DataFrame]:
    market_df, _, _, _, _ = load_simulation_data()
    market_df = market_df.copy()
    market_df["city_key"] = market_df["city"].map(_city_key)
    market_df["event_date"] = pd.to_datetime(market_df["event_date"]).dt.strftime("%Y-%m-%d")
    bucket_map: dict[tuple[str, str], pd.DataFrame] = {}
    for (city_key, event_date), group in market_df.groupby(["city_key", "event_date"], sort=False):
        defs = group[BUCKET_COLS].drop_duplicates("bucket_label").copy()
        bucket_map[(str(city_key), str(event_date))] = defs
    return bucket_map


def _lookup_temps_from_sources(
    pairs: pd.DataFrame,
    bias_detail: pd.DataFrame,
    city_config: dict,
) -> pd.DataFrame:
    bias = bias_detail.copy()
    bias["event_date"] = pd.to_datetime(bias["date"]).dt.strftime("%Y-%m-%d")
    bias = bias.rename(columns={"asos_daily_max": "asos_max"})
    merged = pairs.merge(
        bias[["city", "event_date", "cli_tmax", "asos_max", "bias"]],
        on=["city", "event_date"],
        how="left",
    )

    missing = merged[merged["cli_tmax"].isna()].copy()
    if missing.empty:
        return merged

    for city, group in missing.groupby("city"):
        dates = pd.to_datetime(group["event_date"])
        start = dates.min().date()
        end = dates.max().date()
        cli = _load_cli(city, city_config[city], start, end)
        cli["event_date"] = cli["date"].astype(str)
        city_raw = PROJECT_ROOT / "data" / "trackj" / "raw" / city
        asos = _load_all_asos([city_raw, city_raw / "asos"], city_config[city]["nws_station"], start, end)
        asos_max = build_asos_daily_max_map(asos)
        cli_lookup = dict(zip(cli["event_date"], cli["tmax_f"]))
        for idx, row in group.iterrows():
            day = str(row["event_date"])
            cli_tmax = cli_lookup.get(day)
            asos_val = asos_max.get(day)
            if cli_tmax is None or asos_val is None:
                continue
            merged.loc[idx, "cli_tmax"] = float(cli_tmax)
            merged.loc[idx, "asos_max"] = float(asos_val)
            merged.loc[idx, "bias"] = float(cli_tmax) - float(asos_val)

    return merged


def _recompute_settlement_pnl(row: pd.Series) -> tuple[bool, int]:
    wunder_win = _temp_in_bucket(
        row["asos_max"],
        str(row["entry_bucket"]),
        row.get("entry_bucket_type"),
        row.get("entry_bucket_lower"),
        row.get("entry_bucket_upper"),
    )
    payout = int(row["contracts"]) * 100 if wunder_win else 0
    net_pnl = payout - int(row["cost_cents"]) - int(row["fee_cents"])
    return wunder_win, net_pnl


def _attach_entry_bucket_bounds(frame: pd.DataFrame, bucket_map: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    types: list[str | None] = []
    lowers: list[float | None] = []
    uppers: list[float | None] = []
    for _, row in frame.iterrows():
        defs = bucket_map.get((str(row["city"]), str(row["event_date"])))
        if defs is None:
            try:
                btype, lo, hi = _parse_kalshi_bucket(str(row["entry_bucket"]))
            except ValueError:
                btype, lo, hi = None, None, None
            types.append(btype)
            lowers.append(lo)
            uppers.append(hi)
            continue
        match = defs[defs["bucket_label"].astype(str) == str(row["entry_bucket"])]
        if match.empty:
            try:
                btype, lo, hi = _parse_kalshi_bucket(str(row["entry_bucket"]))
            except ValueError:
                btype, lo, hi = None, None, None
        else:
            m = match.iloc[0]
            btype = str(m["bucket_type"])
            lo = pd.to_numeric(m["bucket_lower_inclusive_f"], errors="coerce")
            hi = pd.to_numeric(m["bucket_upper_inclusive_f"], errors="coerce")
            lo = None if pd.isna(lo) else float(lo)
            hi = None if pd.isna(hi) else float(hi)
        types.append(btype)
        lowers.append(lo)
        uppers.append(hi)
    out = frame.copy()
    out["entry_bucket_type"] = types
    out["entry_bucket_lower"] = lowers
    out["entry_bucket_upper"] = uppers
    return out


def _print_pnl_table(trades: pd.DataFrame) -> None:
    cli_pnl = int(trades["net_pnl_cents"].sum())
    wunder_pnl = int(trades["wunderground_pnl_cents"].sum())
    cli_wr = float((trades["net_pnl_cents"] > 0).mean())
    wunder_wr = float((trades["wunderground_pnl_cents"] > 0).mean())
    print("\n=== PNL IMPACT ===")
    print(f"{'Scenario':<28} | {'N trades':>8} | {'Win rate':>8} | {'Total PnL (c)':>13}")
    print("-" * 70)
    print(f"{'Original (CLI settlement)':<28} | {len(trades):>8d} | {cli_wr:>7.1%} | {cli_pnl:>13d}")
    print(f"{'Wunderground settlement':<28} | {len(trades):>8d} | {wunder_wr:>7.1%} | {wunder_pnl:>13d}")
    print(f"{'Difference':<28} | {'':>8} | {wunder_wr - cli_wr:>+7.1%} | {wunder_pnl - cli_pnl:>+13d}")


def _print_city_table(settlement: pd.DataFrame) -> None:
    print("\n=== PER-CITY FLIP ANALYSIS (settlement trades) ===")
    print(
        f"{'City':<18} {'N settle':>8} {'Flips':>6} {'Flip%':>7} "
        f"{'CLI PnL':>8} {'WU PnL':>8} {'Win->Loss':>10} {'Loss->Win':>10}"
    )
    for city in sorted(settlement["city"].unique()):
        city_df = settlement[settlement["city"] == city]
        flips = city_df["bucket_flipped"].sum()
        win_loss = int(((city_df["cli_resolved"]) & (~city_df["wunderground_resolved"])).sum())
        loss_win = int((~(city_df["cli_resolved"]) & (city_df["wunderground_resolved"])).sum())
        print(
            f"{city:<18} {len(city_df):>8d} {int(flips):>6d} "
            f"{100.0 * flips / len(city_df):>6.1f}% "
            f"{int(city_df['net_pnl_cents'].sum()):>8d} "
            f"{int(city_df['wunderground_pnl_cents'].sum()):>8d} "
            f"{win_loss:>10d} {loss_win:>10d}"
        )


def _save_histogram(settlement: pd.DataFrame) -> None:
    if settlement.empty:
        return
    diff = settlement["cli_tmax"] - settlement["asos_max"]
    near_boundary = int((diff.abs() <= 1.0).sum())
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(diff, bins=30, color="#4878CF", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#8A8A8A", linestyle=":", linewidth=0.8)
    ax.axvline(1, color="#E68A2E", linestyle="--", linewidth=1.0, label="+1°F")
    ax.axvline(-1, color="#E68A2E", linestyle="--", linewidth=1.0, label="-1°F")
    ax.set_title("CLI minus ASOS daily max — settlement trades")
    ax.set_xlabel("CLI Tmax − ASOS max (°F)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.text(
        0.98,
        0.95,
        f"{near_boundary} trades within 1°F of bucket boundary",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    trades = pd.read_parquet(TRADES_PATH)
    trades["event_date"] = pd.to_datetime(trades["event_date"]).dt.strftime("%Y-%m-%d")
    bias_detail = pd.read_parquet(BIAS_DETAIL_PATH)
    city_config = _load_city_config()
    bucket_map = _load_market_bucket_defs()

    pairs = trades[["city", "event_date"]].drop_duplicates()
    temps = _lookup_temps_from_sources(pairs, bias_detail, city_config)
    detail = trades.merge(temps, on=["city", "event_date"], how="left")
    detail = _attach_entry_bucket_bounds(detail, bucket_map)

    missing_temps = detail["cli_tmax"].isna() | detail["asos_max"].isna()
    if missing_temps.any():
        print(f"WARNING: {int(missing_temps.sum())} trades missing CLI/ASOS temps and excluded.")
        detail = detail.loc[~missing_temps].copy()

    cli_buckets: list[str | None] = []
    wunder_buckets: list[str | None] = []
    for _, row in detail.iterrows():
        defs = bucket_map.get((str(row["city"]), str(row["event_date"])))
        if defs is None:
            cli_buckets.append(str(row["entry_bucket"]) if _temp_in_bucket(
                row["cli_tmax"], str(row["entry_bucket"]), row["entry_bucket_type"],
                row["entry_bucket_lower"], row["entry_bucket_upper"],
            ) else None)
            wunder_buckets.append(str(row["entry_bucket"]) if _temp_in_bucket(
                row["asos_max"], str(row["entry_bucket"]), row["entry_bucket_type"],
                row["entry_bucket_lower"], row["entry_bucket_upper"],
            ) else None)
            continue
        cli_buckets.append(_assign_bucket(row["cli_tmax"], defs))
        wunder_buckets.append(_assign_bucket(row["asos_max"], defs))

    detail["cli_bucket"] = cli_buckets
    detail["wunder_bucket"] = wunder_buckets
    detail["bucket_flipped"] = detail["cli_bucket"].astype(str) != detail["wunder_bucket"].astype(str)

    detail["cli_resolved"] = detail["resolved"].astype(bool)
    detail["wunderground_resolved"] = detail.apply(
        lambda row: _temp_in_bucket(
            row["asos_max"],
            str(row["entry_bucket"]),
            row["entry_bucket_type"],
            row["entry_bucket_lower"],
            row["entry_bucket_upper"],
        ),
        axis=1,
    )
    detail["trade_outcome_changed"] = detail["cli_resolved"] != detail["wunderground_resolved"]

    detail["wunderground_pnl_cents"] = detail["net_pnl_cents"]
    settlement_mask = detail["exit_type"] == "settlement"
    for idx, row in detail.loc[settlement_mask].iterrows():
        _, wunder_pnl = _recompute_settlement_pnl(row)
        detail.loc[idx, "wunderground_pnl_cents"] = wunder_pnl

    settlement = detail.loc[settlement_mask].copy()
    profit_target = detail.loc[detail["exit_type"] == "profit_target_15c"]

    total_trades = len(detail)
    n_settlement = len(settlement)
    n_profit_target = len(profit_target)
    bucket_flips = int(settlement["bucket_flipped"].sum())
    outcome_changes = int(settlement["trade_outcome_changed"].sum())
    cli_pnl = int(detail["net_pnl_cents"].sum())
    wunder_pnl = int(detail["wunderground_pnl_cents"].sum())

    print("\n=== BUCKET FLIP RATE ===")
    print(f"Total trades: {total_trades}")
    print(
        f"Settlement trades where ASOS max falls in a different bucket than CLI: "
        f"{bucket_flips} ({100.0 * bucket_flips / n_settlement:.1f}%)"
    )
    print(
        f"Trades where flip changes outcome (win↔loss): "
        f"{outcome_changes} ({100.0 * outcome_changes / n_settlement:.1f}% of settlement)"
    )

    _print_pnl_table(detail)
    _print_city_table(settlement)

    win_loss = int(((settlement["cli_resolved"]) & (~settlement["wunderground_resolved"])).sum())
    loss_win = int((~(settlement["cli_resolved"]) & (settlement["wunderground_resolved"])).sum())
    print("\n=== FLIP DIRECTION ANALYSIS ===")
    print(f"CLI win + Wunderground loss: {win_loss}")
    print(f"CLI loss + Wunderground win: {loss_win}")
    print(
        f"Net win-rate change (settlement): "
        f"{settlement['cli_resolved'].mean():.1%} -> {settlement['wunderground_resolved'].mean():.1%}"
    )

    _save_histogram(settlement)

    pnl_diff = wunder_pnl - cli_pnl
    print(
        f"\nOf {total_trades} total trades, {n_profit_target} exited via profit target "
        f"(unaffected by settlement source). Of the {n_settlement} settlement trades, "
        f"{bucket_flips} ({100.0 * bucket_flips / n_settlement:.1f}%) would have resolved in a "
        f"different bucket under Wunderground rules, changing the outcome for {outcome_changes} "
        f"trades. Net PnL impact: {pnl_diff / 100:+.2f} USD."
    )

    summary = {
        "total_trades": total_trades,
        "profit_target_trades": n_profit_target,
        "settlement_trades": n_settlement,
        "bucket_flips": bucket_flips,
        "bucket_flip_rate": round(100.0 * bucket_flips / n_settlement, 1) if n_settlement else 0.0,
        "outcome_changes": outcome_changes,
        "outcome_change_rate": round(100.0 * outcome_changes / n_settlement, 1) if n_settlement else 0.0,
        "cli_win_rate": round(float((detail["net_pnl_cents"] > 0).mean()), 4),
        "wunderground_win_rate": round(float((detail["wunderground_pnl_cents"] > 0).mean()), 4),
        "cli_pnl_cents": cli_pnl,
        "wunderground_pnl_cents": wunder_pnl,
        "pnl_difference_cents": pnl_diff,
        "cli_win_to_wunder_loss": win_loss,
        "cli_loss_to_wunder_win": loss_win,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    detail.to_parquet(OUTPUT_DETAIL, index=False)
    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_DETAIL}")
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
