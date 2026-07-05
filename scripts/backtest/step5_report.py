#!/usr/bin/env python3
"""Step 5: full backtest results report and GO/NO-GO decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest.common as bc  # noqa: E402
from backtest_utils import bootstrap_sharpe, deflated_sharpe, sharpe_stats  # noqa: E402

VARIANTS = [
    "modal_maker_hold_to_settlement",
    "modal_maker_profit_target_15c",
    "ngboost_kelly_hold_to_settlement",
    "ngboost_kelly_profit_target_15c",
]
FLAT_VARIANT = "ngboost_flat_hold_to_settlement"
ORIGINAL_REPORT_JSON = PROJECT_ROOT / "reports" / "full_backtest_results.json"
PROJ_TRADES_MIN = 80
CI_LOWER_MIN = -3.0
MAX_DD_PCT_LIMIT = -0.30
PNL_CONCENTRATION_MAX = 0.50


def trade_log_path(variant: str) -> Path:
    if variant.startswith("modal_maker_"):
        return bc.TRADES_DIR / f"modal_maker_{variant.removeprefix('modal_maker_')}.jsonl"
    if variant.startswith("ngboost_flat_"):
        return bc.TRADES_DIR / f"ngboost_flat_{variant.removeprefix('ngboost_flat_')}.jsonl"
    return bc.TRADES_DIR / f"ngboost_kelly_{variant.removeprefix('ngboost_kelly_')}.jsonl"


def equity_path(variant: str) -> Path:
    return bc.EQUITY_DIR / f"{variant}.csv"


def trades_to_daily_returns(trades: list[dict], dates: list[str]) -> pd.Series:
    by_date: dict[str, float] = {d: 0.0 for d in dates}
    for t in trades:
        if not t.get("traded"):
            continue
        d = str(t["date"])
        by_date[d] = by_date.get(d, 0.0) + float(t.get("pnl_usd", 0.0))
    capital = bc.INITIAL_BANKROLL_USD * 100  # cents basis for sharpe_stats
    returns = pd.Series(
        {d: (by_date[d] * 100) / capital for d in dates},
        dtype=float,
    )
    return returns.sort_index()


def compute_variant_metrics(
    variant: str,
    eligible_days: int,
    sample_days: int,
    n_variants: int,
) -> dict:
    trades = bc.read_jsonl(trade_log_path(variant))
    equity_file = equity_path(variant)
    if not trades:
        return {"variant": variant, "error": "no trades"}

    traded = [t for t in trades if t.get("traded")]
    dates = sorted({str(t["date"]) for t in trades})
    daily_rets = trades_to_daily_returns(trades, dates)
    stats = sharpe_stats(daily_rets)
    boot = bootstrap_sharpe(daily_rets, n_boot=2000)
    dsr = deflated_sharpe(daily_rets, n_variants)

    equity_df = pd.read_csv(equity_file) if equity_file.exists() else pd.DataFrame()
    final_bankroll = float(equity_df["bankroll_usd"].iloc[-1]) if not equity_df.empty else bc.INITIAL_BANKROLL_USD
    eliminated = bool(equity_df["eliminated"].any()) if "eliminated" in equity_df.columns else False

    pnls = [float(t["pnl_usd"]) for t in traded]
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls) if pnls else 0.0

    daily_pnl = daily_rets * bc.INITIAL_BANKROLL_USD
    top3 = daily_pnl.nlargest(3).sum() if len(daily_pnl) else 0.0
    concentration = abs(top3 / total_pnl) if total_pnl != 0 else 0.0

    per_city: dict[str, float] = {}
    for t in traded:
        c = str(t["city"])
        per_city[c] = per_city.get(c, 0.0) + float(t["pnl_usd"])

    proj_60d = (len(traded) / sample_days * 60) if sample_days > 0 else 0

    return {
        "variant": variant,
        "n_trades": len(traded),
        "n_days": len(dates),
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "total_pnl_usd": round(total_pnl, 4),
        "avg_pnl_usd": round(avg_pnl, 4),
        "sharpe": stats["sharpe_annual"],
        "sharpe_ci_low": stats["sharpe_ci_low"],
        "sharpe_ci_high": stats["sharpe_ci_high"],
        "bootstrap_ci_low": boot["sharpe_boot_ci_low"],
        "bootstrap_ci_high": boot["sharpe_boot_ci_high"],
        "sortino": stats["sortino_annual"],
        "max_drawdown_usd": stats["max_drawdown"] * bc.INITIAL_BANKROLL_USD,
        "max_drawdown_pct": stats["max_drawdown"],
        "psr_0": stats.get("PSR_0", float("nan")),
        "deflated_sharpe": dsr.get("sr_deflated", float("nan")),
        "projected_trades_60d": round(proj_60d, 1),
        "pnl_concentration_top3": round(concentration, 4),
        "per_city_pnl": {k: round(v, 4) for k, v in sorted(per_city.items())},
        "final_bankroll_usd": round(final_bankroll, 4),
        "eliminated": eliminated,
        "eligible_days": eligible_days,
    }


def gonogo(metrics: dict, baseline_sharpe: float) -> dict[str, bool]:
    sharpe = metrics.get("sharpe", float("nan"))
    return {
        "sharpe_gt_baseline": bool(np.isfinite(sharpe) and sharpe > baseline_sharpe),
        "deflated_sharpe_positive": bool(metrics.get("deflated_sharpe", float("nan")) > 0),
        "ci_lower_gt_neg3": bool(metrics.get("bootstrap_ci_low", float("nan")) > CI_LOWER_MIN),
        "projected_trades_80": metrics["projected_trades_60d"] >= PROJ_TRADES_MIN,
        "max_dd_under_30pct": metrics["max_drawdown_pct"] > MAX_DD_PCT_LIMIT,
        "pnl_concentration_ok": metrics["pnl_concentration_top3"] < PNL_CONCENTRATION_MAX,
        "not_eliminated": not metrics["eliminated"],
    }


def load_live_trades() -> list[dict]:
    records: list[dict] = []
    paper = PROJECT_ROOT / "logs" / "poly_paper_trades.jsonl"
    if paper.exists():
        records.extend(bc.read_jsonl(paper))
    return records


def live_sanity_check(modal_trades: list[dict]) -> list[str]:
    flags: list[str] = []
    live = load_live_trades()
    if not live:
        flags.append("No live trade logs found (poly_paper_trades.jsonl) — sanity check skipped.")
        return flags

    bt_by_date: dict[str, float] = {}
    for t in modal_trades:
        if t.get("traded"):
            d = str(t["date"])
            bt_by_date[d] = bt_by_date.get(d, 0.0) + float(t.get("pnl_usd", 0.0))

    for entry in live:
        d = str(entry.get("date", ""))
        if d not in bt_by_date:
            continue
        live_pnl = sum(float(tr.get("pnl_usd", 0) or 0) for tr in entry.get("trades", []))
        bt_pnl = bt_by_date[d]
        if abs(live_pnl - bt_pnl) > 5.0:
            flags.append(
                f"LOUD FLAG: {d} modal_maker backtest PnL ${bt_pnl:.2f} vs live ${live_pnl:.2f} "
                f"(delta ${abs(live_pnl - bt_pnl):.2f})"
            )
    return flags


def recommend(results: dict[str, dict]) -> str:
    valid = {
        k: v for k, v in results.items()
        if np.isfinite(v.get("sharpe", float("nan"))) and v.get("n_trades", 0) > 0
    }
    if not valid:
        return (
            "Recommendation: continue manual trading and modal_maker cron for now — no variant "
            "produced enough trades for a reliable comparison. "
            "Biggest caveat: Telonex backfill sample is extremely thin until more city-dates "
            "have order-book, HRRR, and WU coverage aligned."
        )
    best = max(valid.items(), key=lambda kv: kv[1].get("sharpe", -999))
    name, m = best
    if m.get("sharpe", 0) <= 0 or m.get("eliminated"):
        return (
            "Recommendation: continue manual trading and modal_maker cron for now — no variant shows "
            "a convincing risk-adjusted edge under MCP constraints. "
            "Biggest caveat: thin pre-Feb-2026 sample and possible divergence between backtest "
            "assumptions and live execution."
        )
    return (
        f"Recommendation: pilot {name.replace('_', ' ')} starting Monday if GO/NO-GO gates pass, "
        f"replacing modal_maker-only cron where it dominates. "
        f"Biggest caveat: sample covers only ~{m.get('eligible_days', '?')} eligible city-days since "
        "Telonex history began (~Feb 2026); extrapolation to a full 60-day MCP cycle is uncertain."
    )


def describe_equity_curve(variant: str) -> str:
    eq = pd.read_csv(equity_path(variant))
    if eq.empty:
        return "No equity data."
    peak = float(eq["bankroll_usd"].max())
    final = float(eq["bankroll_usd"].iloc[-1])
    trough = float(eq["bankroll_usd"].min())
    peak_idx = int(eq["bankroll_usd"].idxmax())
    peak_date = str(eq.loc[peak_idx, "date"])
    if peak > bc.INITIAL_BANKROLL_USD + 5 and final < peak - 10:
        return (
            f"Peak-and-bleed pattern: bankroll peaked at ${peak:.2f} on {peak_date}, "
            f"trough ${trough:.2f}, final ${final:.2f}."
        )
    return f"Bankroll range ${trough:.2f}–${peak:.2f}, final ${final:.2f}."


def write_v5b_comparison_report(baseline: dict, v5b: dict) -> None:
    """Compare v5 flat vs v5b flat (rolling bias + boundary baskets)."""
    out = PROJECT_ROOT / "reports" / "backtest_v5b_comparison.md"
    base_flat = baseline.get("variants", {}).get(FLAT_VARIANT, {})
    new_flat = v5b.get("variants", {}).get(FLAT_VARIANT, {})
    metrics = [
        ("sharpe", "Sharpe", "{:.2f}"),
        ("win_rate", "Win rate", "{:.1%}"),
        ("max_drawdown_usd", "Max DD", "${:.2f}"),
        ("final_bankroll_usd", "Final bankroll", "${:.2f}"),
        ("n_trades", "N trades", "{:.0f}"),
    ]
    lines = [
        "# Backtest v5b Comparison\n\n",
        "v5 (flat) vs v5b (flat + rolling bias + boundary baskets).\n\n",
        "| Metric | v5 (flat) | v5b (flat+bias+basket) | Delta |\n",
        "|--------|----------:|-----------------------:|------:|\n",
    ]
    for key, label, fmt in metrics:
        ov = base_flat.get(key, float("nan"))
        nv = new_flat.get(key, float("nan"))
        if key == "win_rate" and isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            delta = nv - ov
            delta_s = f"{delta:+.1%}" if np.isfinite(delta) else "—"
        elif isinstance(ov, (int, float)) and isinstance(nv, (int, float)) and np.isfinite(ov) and np.isfinite(nv):
            delta = nv - ov
            if "$" in fmt:
                delta_s = f"${delta:+.2f}"
            else:
                delta_s = f"{delta:+.2f}"
        else:
            delta_s = "—"
        lines.append(f"| {label} | {fmt.format(ov)} | {fmt.format(nv)} | {delta_s} |\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def write_comparison_report(
    original: dict,
    v5: dict,
    flat_gonogo: dict[str, bool],
    flat_metrics: dict,
) -> None:
    out = PROJECT_ROOT / "reports" / "backtest_v5_comparison.md"
    lines = [
        "# Backtest v5 Comparison\n\n",
        "Original (10-city, pre-risk-management) vs v5 (shrinkage λ=0.6, anti-cyclic budget, top-2/day).\n\n",
        "## 1. Side-by-side by variant\n\n",
        "| Variant | Metric | Original | v5 | Delta |\n",
        "|---------|--------|--------:|---:|------:|\n",
    ]
    metrics = [
        ("n_trades", "Trades", "{:.0f}"),
        ("total_pnl_usd", "Total PnL ($)", "${:.2f}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("max_drawdown_usd", "Max DD ($)", "${:.2f}"),
        ("final_bankroll_usd", "MCP final bankroll ($)", "${:.2f}"),
        ("win_rate", "Win rate (%)", "{:.1f}%"),
    ]
    compare_variants = VARIANTS + [FLAT_VARIANT]
    orig_vars = original.get("variants", {})
    v5_vars = v5.get("variants", {})

    for variant in compare_variants:
        if variant not in v5_vars:
            continue
        o = orig_vars.get(variant)
        n = v5_vars[variant]
        for key, label, fmt in metrics:
            nv = n.get(key, float("nan"))
            ov = o.get(key, float("nan")) if o else float("nan")
            if key == "win_rate":
                ov *= 100
                nv *= 100
            delta = nv - ov if isinstance(nv, (int, float)) and isinstance(ov, (int, float)) else float("nan")
            if isinstance(delta, float) and np.isfinite(delta):
                if "$" in fmt:
                    delta_s = f"${delta:+.2f}"
                elif "%" in fmt:
                    delta_s = f"{delta:+.1f}pp"
                else:
                    delta_s = f"{delta:+.2f}"
            else:
                delta_s = "—"
            lines.append(
                f"| {variant} | {label} | {fmt.format(ov)} | {fmt.format(nv)} | {delta_s} |\n"
            )

    lines.extend(["\n## 2. Per-city PnL — ngboost_flat_hold_to_settlement (v5)\n\n"])
    for city, pnl in flat_metrics.get("per_city_pnl", {}).items():
        lines.append(f"- {city}: ${pnl:.2f}\n")

    lines.extend([
        "\n## 3. Equity curve — ngboost_flat_hold_to_settlement (v5 MCP)\n\n",
        describe_equity_curve(FLAT_VARIANT) + "\n\n",
        "## 4. GO/NO-GO matrix — ngboost_flat_hold_to_settlement\n\n",
        "| Criterion | Result |\n|-----------|--------|\n",
    ])
    for crit, passed in flat_gonogo.items():
        lines.append(f"| {crit} | {'PASS' if passed else 'FAIL'} |\n")

    flat_final = flat_metrics.get("final_bankroll_usd", bc.INITIAL_BANKROLL_USD)
    lines.extend([
        "\n## 5. MCP-constrained final bankroll (step4)\n\n",
        "| Variant | Final bankroll |\n|---------|---------------:|\n",
    ])
    for variant in compare_variants:
        if variant in v5_vars:
            fb = v5_vars[variant].get("final_bankroll_usd", float("nan"))
            lines.append(f"| {variant} | ${fb:.2f} |\n")

    lines.extend([
        "\n## Summary\n\n",
        f"Deployment candidate **ngboost_flat_hold_to_settlement** MCP final bankroll: "
        f"**${flat_final:.2f}** ({'eliminated' if flat_metrics.get('eliminated') else 'survived'}). "
        f"Risk changes reduced Kelly pro-cyclical sizing; see side-by-side table for Kelly variants.\n",
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def run_report(
    variants: list[str],
    report_md: Path,
    report_json: Path,
    compare_original: bool = False,
    compare_tag: str = "",
) -> None:
    n_variants = len(variants)
    for variant in variants:
        if not trade_log_path(variant).exists():
            print(f"ERROR: missing {trade_log_path(variant)} — run steps 2-4 first")
            sys.exit(1)
        if not equity_path(variant).exists():
            print(f"ERROR: missing {equity_path(variant)}")
            sys.exit(1)

    eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV) if bc.ELIGIBLE_DATES_CSV.exists() else pd.DataFrame()
    eligible_days = len(eligible)
    sample_days = len(eligible["date"].unique()) if not eligible.empty else 1

    results: dict[str, dict] = {}
    for variant in variants:
        results[variant] = compute_variant_metrics(
            variant, eligible_days, sample_days, n_variants
        )

    baseline_sharpe = results["modal_maker_hold_to_settlement"]["sharpe"]
    gonogo_all: dict[str, dict] = {}
    for variant, m in results.items():
        gonogo_all[variant] = gonogo(m, baseline_sharpe)

    modal_trades = bc.read_jsonl(trade_log_path("modal_maker_hold_to_settlement"))
    sanity_flags = live_sanity_check(modal_trades)

    lines = ["# Full Backtest Results\n", f"Eligible city-dates: {eligible_days}\n\n"]
    lines.append("## Variant metrics\n\n")
    lines.append(
        "| Variant | Trades | Sharpe [CI] | Sortino | Max DD | Win% | Total PnL | Proj/60d | Final $ |\n"
    )
    lines.append("|---------|-------:|-------------|--------:|-------:|-----:|----------:|---------:|--------:|\n")
    for variant, m in results.items():
        ci = f"[{m['bootstrap_ci_low']:.2f}, {m['bootstrap_ci_high']:.2f}]"
        lines.append(
            f"| {variant} | {m['n_trades']} | {m['sharpe']:.2f} {ci} | {m['sortino']:.2f} | "
            f"${m['max_drawdown_usd']:.2f} | {100*m['win_rate']:.1f}% | "
            f"${m['total_pnl_usd']:.2f} | {m['projected_trades_60d']:.0f} | "
            f"${m['final_bankroll_usd']:.2f} |\n"
        )

    lines.append("\n## GO/NO-GO matrix\n\n")
    criteria = list(next(iter(gonogo_all.values())).keys()) if gonogo_all else []
    lines.append("| Criterion | " + " | ".join(variants) + " |\n")
    lines.append("|" + "---|" * (len(variants) + 1) + "\n")
    for crit in criteria:
        row = f"| {crit} |"
        for v in variants:
            val = gonogo_all[v].get(crit, False)
            row += f" {'PASS' if val else 'FAIL'} |"
        lines.append(row + "\n")

    if sanity_flags:
        lines.append("\n## Live sanity check\n\n")
        for flag in sanity_flags:
            lines.append(f"- {flag}\n")

    lines.append("\n## Per-city PnL (hold_to_settlement variants)\n\n")
    for label in ["modal_maker_hold_to_settlement", "ngboost_kelly_hold_to_settlement"]:
        if label in results:
            m = results[label]
            lines.append(f"### {label}\n\n")
            for city, pnl in m.get("per_city_pnl", {}).items():
                lines.append(f"- {city}: ${pnl:.2f}\n")
    if FLAT_VARIANT in results:
        m = results[FLAT_VARIANT]
        lines.append(f"### {FLAT_VARIANT}\n\n")
        for city, pnl in m.get("per_city_pnl", {}).items():
            lines.append(f"- {city}: ${pnl:.2f}\n")

    rec = recommend(results)
    lines.append(f"\n## Recommendation\n\n{rec}\n")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("".join(lines), encoding="utf-8")

    payload = {
        "eligible_days": eligible_days,
        "variants": results,
        "gonogo": gonogo_all,
        "sanity_flags": sanity_flags,
        "recommendation": rec,
    }
    report_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {report_md}")
    print(f"Wrote {report_json}")
    print(f"\n{rec}")

    if compare_original and ORIGINAL_REPORT_JSON.exists():
        with open(ORIGINAL_REPORT_JSON, encoding="utf-8") as handle:
            original = json.load(handle)
        flat_metrics = results.get(FLAT_VARIANT, {})
        flat_gonogo = gonogo_all.get(FLAT_VARIANT, {})
        write_comparison_report(original, payload, flat_gonogo, flat_metrics)

    if compare_tag:
        baseline_json = PROJECT_ROOT / "reports" / f"backtest_results_{compare_tag}.json"
        if baseline_json.exists():
            with open(baseline_json, encoding="utf-8") as handle:
                baseline = json.load(handle)
            write_v5b_comparison_report(baseline, payload)
        else:
            print(f"WARNING: baseline report not found: {baseline_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest results report")
    parser.add_argument("--output-tag", default="", help="Output suffix, e.g. v5")
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="Write backtest_v5_comparison.md vs original full_backtest_results.json",
    )
    parser.add_argument(
        "--compare-tag",
        default="",
        help="Baseline output tag for comparison report (e.g. v5 vs v5b)",
    )
    args = parser.parse_args()

    if args.output_tag:
        bc.configure_output_tag(args.output_tag)

    variants = list(VARIANTS)
    if args.output_tag in ("v5", "v5b"):
        variants.append(FLAT_VARIANT)

    if args.output_tag:
        report_md = PROJECT_ROOT / "reports" / f"backtest_results_{args.output_tag}.md"
        report_json = PROJECT_ROOT / "reports" / f"backtest_results_{args.output_tag}.json"
    else:
        report_md = PROJECT_ROOT / "reports" / "full_backtest_results.md"
        report_json = PROJECT_ROOT / "reports" / "full_backtest_results.json"

    run_report(
        variants,
        report_md,
        report_json,
        compare_original=args.compare_original,
        compare_tag=args.compare_tag,
    )


if __name__ == "__main__":
    main()
