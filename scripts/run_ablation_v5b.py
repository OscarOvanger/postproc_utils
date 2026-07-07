#!/usr/bin/env python3
"""Holdout split and ablation study for v5b ngboost_flat_hold_to_settlement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest.common as bc  # noqa: E402
from backtest.step5_report import FLAT_VARIANT, compute_variant_metrics  # noqa: E402

PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
REPORT_MD = PROJECT_ROOT / "reports" / "ablation_v5b.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "ablation_v5b.json"
V5B_HISTORICAL = PROJECT_ROOT / "reports" / "backtest_results_v5b.json"
FLAT_VARIANT_KEY = "ngboost_flat_hold_to_settlement"


@dataclass
class RunSpec:
    label: str
    tag: str
    start_date: str | None = None
    end_date: str | None = None
    disable_rolling_bias: bool = False
    disable_basket: bool = False
    skip_run: bool = False


HOLDOUT_RUNS = [
    RunSpec("Full (historical v5b)", "v5b_historical", skip_run=True),
    RunSpec("Full (post-fix)", "ablation_holdout_full"),
    RunSpec("Early Feb3–Apr15", "ablation_holdout_early", "2026-02-03", "2026-04-15"),
    RunSpec("Late Apr16–Jun30", "ablation_holdout_late", "2026-04-16", "2026-06-30"),
]

ABLATION_RUNS = [
    RunSpec("v5b (full)", "ablation_full"),
    RunSpec("No bias", "ablation_no_bias", disable_rolling_bias=True),
    RunSpec("No basket", "ablation_no_basket", disable_basket=True),
    RunSpec("No both", "ablation_no_both", disable_rolling_bias=True, disable_basket=True),
]


def _run_step(script: str, tag: str, extra: list[str]) -> None:
    cmd = [
        str(PYTHON),
        str(SCRIPTS_DIR / "backtest" / script),
        "--output-tag",
        tag,
        "--flat-only",
        "--force",
        *extra,
    ]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def run_backtest(spec: RunSpec) -> None:
    if spec.skip_run:
        return
    extra: list[str] = []
    if spec.start_date:
        extra.extend(["--start-date", spec.start_date])
    if spec.end_date:
        extra.extend(["--end-date", spec.end_date])
    if spec.disable_rolling_bias:
        extra.append("--disable-rolling-bias")
    if spec.disable_basket:
        extra.append("--disable-basket")
    _run_step("step3_ngboost_kelly.py", spec.tag, extra)
    _run_step("step4_mcp_simulation.py", spec.tag, [])


def metrics_for_tag(tag: str, sample_days: int) -> dict:
    bc.configure_output_tag(tag)
    return compute_variant_metrics(
        FLAT_VARIANT,
        eligible_days=sample_days,
        sample_days=sample_days,
        n_variants=4,
    )


def load_historical_full() -> dict:
    if not V5B_HISTORICAL.exists():
        return {}
    data = json.loads(V5B_HISTORICAL.read_text(encoding="utf-8"))
    return data.get("variants", {}).get(FLAT_VARIANT_KEY, {})


def summarize_row(label: str, m: dict) -> dict:
    return {
        "label": label,
        "n_trades": m.get("n_trades", 0),
        "sharpe": m.get("sharpe"),
        "max_drawdown_usd": m.get("max_drawdown_usd"),
        "final_bankroll_usd": m.get("final_bankroll_usd"),
        "win_rate": m.get("win_rate"),
    }


def format_table(rows: list[dict], title: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| Period | N trades | Sharpe | Max DD | Final bankroll | Win% |",
        "|--------|----------|--------|--------|----------------|------|",
    ]
    for row in rows:
        sharpe = row.get("sharpe")
        sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) and sharpe == sharpe else "?"
        max_dd = row.get("max_drawdown_usd")
        max_dd_s = f"${max_dd:.2f}" if isinstance(max_dd, (int, float)) else "?"
        final_b = row.get("final_bankroll_usd")
        final_s = f"${final_b:.2f}" if isinstance(final_b, (int, float)) else "?"
        win = row.get("win_rate")
        win_s = f"{100 * win:.1f}%" if isinstance(win, (int, float)) else "?"
        lines.append(
            f"| {row['label']} | {row.get('n_trades', '?')} | {sharpe_s} | "
            f"{max_dd_s} | {final_s} | {win_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_ablation_table(rows: list[dict]) -> str:
    lines = [
        "## Ablation (full window)",
        "",
        "| Variant | Shrinkage | Budget | Flat | Bias | Basket | Sharpe |",
        "|---------|-----------|--------|------|------|--------|--------|",
    ]
    for row in rows:
        sharpe = row.get("sharpe")
        sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) and sharpe == sharpe else "?"
        lines.append(
            f"| {row['label']} | 0.6 | anti | yes | "
            f"{'yes' if row.get('bias') else 'NO'} | "
            f"{'yes' if row.get('basket') else 'NO'} | {sharpe_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="v5b holdout + ablation runner")
    parser.add_argument("--holdout-only", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--skip-runs", action="store_true", help="Only rebuild report from existing tags")
    args = parser.parse_args()

    if not bc.ELIGIBLE_DATES_CSV.exists():
        print(f"ERROR: missing {bc.ELIGIBLE_DATES_CSV}")
        sys.exit(1)

    eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
    full_days = len(eligible["date"].unique())

    specs: list[RunSpec] = []
    if not args.ablation_only:
        specs.extend(HOLDOUT_RUNS)
    if not args.holdout_only:
        specs.extend(ABLATION_RUNS)

    if not args.skip_runs:
        for spec in specs:
            print(f"\n=== Running {spec.label} (tag={spec.tag}) ===")
            run_backtest(spec)

    holdout_rows: list[dict] = []
    for spec in HOLDOUT_RUNS if not args.ablation_only else []:
        if spec.skip_run:
            hist = load_historical_full()
            if hist:
                holdout_rows.append(summarize_row(spec.label, hist))
            continue
        if spec.start_date and spec.end_date:
            n_days = len(
                bc.filter_eligible_by_date(eligible, spec.start_date, spec.end_date)["date"].unique()
            )
        else:
            n_days = full_days
        m = metrics_for_tag(spec.tag, n_days)
        holdout_rows.append(summarize_row(spec.label, m))

    ablation_rows: list[dict] = []
    for spec in ABLATION_RUNS if not args.holdout_only else []:
        m = metrics_for_tag(spec.tag, full_days)
        row = summarize_row(spec.label, m)
        row["bias"] = not spec.disable_rolling_bias
        row["basket"] = not spec.disable_basket
        ablation_rows.append(row)

    report = {
        "holdout": holdout_rows,
        "ablation": ablation_rows,
        "flat_variant": FLAT_VARIANT_KEY,
        "eligible_city_dates": len(eligible),
        "sample_days_full": full_days,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_parts = [
        "# v5b Ablation and Holdout Report",
        "",
        f"Eligible city-dates (full): {len(eligible)}",
        "",
    ]
    if holdout_rows:
        md_parts.append(format_table(holdout_rows, "Holdout split"))
        early = next((r for r in holdout_rows if "Early" in r["label"]), None)
        late = next((r for r in holdout_rows if "Late" in r["label"]), None)
        if early and late:
            e_sh = early.get("sharpe")
            l_sh = late.get("sharpe")
            if isinstance(e_sh, (int, float)) and isinstance(l_sh, (int, float)):
                robust = e_sh > 0 and l_sh > 0
                md_parts.append(
                    f"**Holdout verdict:** {'Robust (Sharpe > 0 in both halves)' if robust else 'Concentrated — review for overfit'}."
                )
                md_parts.append("")
    if ablation_rows:
        md_parts.append(format_ablation_table(ablation_rows))
        no_both = next((r for r in ablation_rows if r["label"] == "No both"), None)
        if no_both and isinstance(no_both.get("sharpe"), (int, float)):
            delta = abs(no_both["sharpe"] - 1.75)
            md_parts.append(
                f"**No-both vs v5 baseline (Sharpe 1.75):** {no_both['sharpe']:.2f} "
                f"(delta {delta:.2f})."
            )
            md_parts.append("")

    REPORT_MD.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\nWrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
