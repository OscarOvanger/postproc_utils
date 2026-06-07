"""Run Day 6 silent-sins audit checks on the IS/OOS baseline outputs."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest_utils import _entry_time_column  # noqa: E402
from frozen_params import FROZEN_PARAMS_PATH  # noqa: E402
from snapshot_stability import SPLIT_DIR, assert_no_true_holdout  # noqa: E402

BASELINES = [
    "implied_favorite",
    "distribution_copy",
    "sell_longshots",
    "make_the_market",
    "mode_prob_threshold",
    "entropy_threshold",
    "momentum_threshold",
]
OOS_DIR = SPLIT_DIR / "oos_results"
IS_DIR = SPLIT_DIR / "smoke_test_results"
TIME_HOLDOUT_PATH = SPLIT_DIR / "time_holdout.parquet"
AUDIT_PATH = SPLIT_DIR / "audit_results.txt"
ENTRY_FLOOR = time(10, 0)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    violations: list[str]


def oos_parquet_paths() -> list[Path]:
    return [OOS_DIR / f"{baseline}_OOS.parquet" for baseline in BASELINES]


def load_time_holdout() -> pd.DataFrame:
    if "true_holdout" in TIME_HOLDOUT_PATH.as_posix():
        raise AssertionError("true_holdout.parquet must not be loaded")
    df = pd.read_parquet(TIME_HOLDOUT_PATH)
    assert_no_true_holdout(df)
    if "partition" not in df.columns:
        raise AssertionError("time_holdout must contain a partition column")
    labels = set(df["partition"].dropna().astype(str).unique())
    if labels != {"time_holdout"}:
        raise AssertionError(f"Expected only time_holdout rows, found {sorted(labels)}")
    return df


def non_signal_trades(results_df: pd.DataFrame) -> pd.DataFrame:
    if "no_signal" not in results_df.columns:
        return results_df.copy()
    return results_df[~results_df["no_signal"].fillna(False).astype(bool)].copy()


def normalize_city_key(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def city_key(frame: pd.DataFrame) -> pd.Series:
    if "source_city_folder" in frame.columns:
        return normalize_city_key(frame["source_city_folder"])
    if "city" in frame.columns:
        return normalize_city_key(frame["city"])
    raise ValueError("frame must contain source_city_folder or city")


def add_join_keys(frame: pd.DataFrame, entry_col: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["_city_key"] = city_key(out)
    out["_event_date_key"] = pd.to_datetime(out["event_date"]).dt.date.astype(str)
    if "bucket_label" in out.columns:
        out["_bucket_key"] = out["bucket_label"].astype(str)
    if entry_col is not None:
        out["_entry_time_key"] = pd.to_datetime(out[entry_col], errors="coerce").dt.tz_localize(
            None
        )
    return out


def settlement_lookup(time_holdout: pd.DataFrame) -> pd.DataFrame:
    required = {"event_date", "bucket_label", "bucket_resolved_to_one_dollars"}
    missing = required.difference(time_holdout.columns)
    if missing:
        raise ValueError(f"time_holdout missing settlement columns: {sorted(missing)}")
    market = add_join_keys(time_holdout)
    market["_settlement"] = (
        pd.to_numeric(market["bucket_resolved_to_one_dollars"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    return market[
        ["_city_key", "_event_date_key", "_bucket_key", "_settlement"]
    ].drop_duplicates(["_city_key", "_event_date_key", "_bucket_key"])


def check_settled_price_as_entry(time_holdout: pd.DataFrame) -> CheckResult:
    violations: list[str] = []
    settlements = settlement_lookup(time_holdout)
    n_violations = 0

    for path in oos_parquet_paths():
        results = pd.read_parquet(path)
        trades = non_signal_trades(results)
        if trades.empty or "entry_price" not in trades.columns:
            continue

        exact_settlement_prices = trades[
            pd.to_numeric(trades["entry_price"], errors="coerce").isin([0.0, 1.0])
        ]
        for _, row in exact_settlement_prices.iterrows():
            n_violations += 1
            violations.append(
                f"{path.name}: entry_price={row.get('entry_price')} for "
                f"{row.get('city', row.get('source_city_folder', 'unknown'))} "
                f"{row.get('event_date')} {row.get('bucket_label', 'unknown bucket')}"
            )

        if "bucket_label" not in trades.columns:
            continue
        priced = trades.dropna(subset=["entry_price", "bucket_label"]).copy()
        if priced.empty:
            continue
        priced = add_join_keys(priced)
        merged = priced.merge(
            settlements,
            on=["_city_key", "_event_date_key", "_bucket_key"],
            how="left",
            validate="many_to_one",
        )
        entry_price = pd.to_numeric(merged["entry_price"], errors="coerce")
        settlement = pd.to_numeric(merged["_settlement"], errors="coerce")
        settled_price_mask = entry_price.eq(settlement)
        for _, row in merged[settled_price_mask].iterrows():
            n_violations += 1
            violations.append(
                f"{path.name}: entry_price equals settlement "
                f"({row['entry_price']} == {row['_settlement']}) for "
                f"{row.get('city', row.get('source_city_folder', 'unknown'))} "
                f"{row.get('event_date')} {row.get('bucket_label', 'unknown bucket')}"
            )

    passed = n_violations == 0
    return CheckResult(
        "Check 1 — Settled price as entry",
        passed,
        f"Check 1 — Settled price as entry: {'PASS' if passed else 'FAIL'} "
        f"({n_violations} violations)",
        violations,
    )


def called_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def check_oos_threshold_leakage() -> CheckResult:
    violations: list[str] = []
    if not FROZEN_PARAMS_PATH.exists():
        violations.append(f"Missing frozen params: {FROZEN_PARAMS_PATH}")
    else:
        frozen_mtime = FROZEN_PARAMS_PATH.stat().st_mtime
        for path in oos_parquet_paths():
            if not path.exists():
                violations.append(f"Missing OOS result parquet: {path}")
                continue
            if frozen_mtime >= path.stat().st_mtime:
                violations.append(
                    f"{FROZEN_PARAMS_PATH.name} was not older than {path.name}"
                )

    forbidden = {"optimise_t_star", "optimise_h_star", "optimise_momentum"}
    oos_script = PROJECT_ROOT / "scripts" / "run_oos_evaluation.py"
    calls = called_function_names(oos_script)
    leaked_calls = sorted(forbidden.intersection(calls))
    if leaked_calls:
        violations.append(
            "run_oos_evaluation.py calls forbidden optimiser(s): "
            + ", ".join(leaked_calls)
        )

    passed = not violations
    return CheckResult(
        "Check 2 — OOS threshold leakage",
        passed,
        f"Check 2 — OOS threshold leakage: {'PASS' if passed else 'FAIL'}",
        violations,
    )


def check_sharpe_ci() -> CheckResult:
    violations: list[str] = []
    for label, path in [
        ("IS", IS_DIR / "full_stats_table_IS.csv"),
        ("OOS", OOS_DIR / "full_stats_table_OOS.csv"),
    ]:
        stats = pd.read_csv(path)
        missing = {"Sharpe_CI_low", "Sharpe_CI_high"}.difference(stats.columns)
        if missing:
            violations.append(f"{label} stats missing columns: {sorted(missing)}")
            continue
        null_rows = stats[["Sharpe_CI_low", "Sharpe_CI_high"]].isnull().any(axis=1)
        for _, row in stats[null_rows].iterrows():
            violations.append(f"{label} stats has null Sharpe CI for {row['Baseline']}")

    passed = not violations
    return CheckResult(
        "Check 3 — Sharpe without CI",
        passed,
        f"Check 3 — Sharpe without CI: {'PASS' if passed else 'FAIL'}",
        violations,
    )


def post_floor_snapshot_lookup(time_holdout: pd.DataFrame) -> pd.DataFrame:
    required = {"event_date", "snapshot_time_local", "bucket_label"}
    missing = required.difference(time_holdout.columns)
    if missing:
        raise ValueError(f"time_holdout missing entry-time columns: {sorted(missing)}")
    market = time_holdout.copy()
    market["snapshot_time_local"] = pd.to_datetime(
        market["snapshot_time_local"], errors="coerce"
    ).dt.tz_localize(None)
    market = market[market["snapshot_time_local"].dt.time >= ENTRY_FLOOR].copy()
    market = add_join_keys(market, entry_col="snapshot_time_local")
    return market[
        ["_city_key", "_event_date_key", "_bucket_key", "_entry_time_key"]
    ].drop_duplicates()


def check_entry_time_floor(time_holdout: pd.DataFrame) -> CheckResult:
    violations: list[str] = []
    snapshots = post_floor_snapshot_lookup(time_holdout)

    for path in oos_parquet_paths():
        results = pd.read_parquet(path)
        trades = non_signal_trades(results)
        if trades.empty:
            continue
        try:
            entry_col = _entry_time_column(trades)
        except ValueError as exc:
            violations.append(f"{path.name}: {exc}")
            continue

        trades = trades.copy()
        trades[entry_col] = pd.to_datetime(trades[entry_col], errors="coerce").dt.tz_localize(
            None
        )
        floor_violations = trades[trades[entry_col].dt.time < ENTRY_FLOOR]
        for _, row in floor_violations.iterrows():
            violations.append(
                f"{path.name}: entry before 10:00 at {row[entry_col]} for "
                f"{row.get('city', row.get('source_city_folder', 'unknown'))} "
                f"{row.get('event_date')}"
            )

        if "bucket_label" not in trades.columns:
            continue
        keyed = add_join_keys(trades.dropna(subset=[entry_col, "bucket_label"]), entry_col)
        merged = keyed.merge(
            snapshots.assign(_snapshot_exists=True),
            on=["_city_key", "_event_date_key", "_bucket_key", "_entry_time_key"],
            how="left",
            validate="many_to_one",
        )
        missing_snapshot = merged[merged["_snapshot_exists"].isna()]
        for _, row in missing_snapshot.iterrows():
            violations.append(
                f"{path.name}: entry_snapshot_time not found in post-10AM market rows "
                f"for {row.get('city', row.get('source_city_folder', 'unknown'))} "
                f"{row.get('event_date')} {row.get('bucket_label')} at {row[entry_col]}"
            )

    passed = not violations
    return CheckResult(
        "Check 4 — Post-entry data leakage / floor violation",
        passed,
        f"Check 4 — Post-entry data leakage / floor violation: "
        f"{'PASS' if passed else 'FAIL'}",
        violations,
    )


def format_results(results: list[CheckResult]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(result.message)
        if result.violations:
            lines.append("Violations:")
            lines.extend(f"  - {violation}" for violation in result.violations)
    passed = sum(result.passed for result in results)
    lines.append(f"AUDIT COMPLETE: {passed}/4 checks passed.")
    return "\n".join(lines) + "\n"


def main() -> None:
    time_holdout = load_time_holdout()
    results = [
        check_settled_price_as_entry(time_holdout),
        check_oos_threshold_leakage(),
        check_sharpe_ci(),
        check_entry_time_floor(time_holdout),
    ]
    output = format_results(results)
    AUDIT_PATH.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
