#!/usr/bin/env python3
"""Run backtest steps 2-5 on a restricted 6-city subset without touching 10-city outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INCLUDED_CITIES = [
    "houston",
    "los_angeles",
    "austin",
    "chicago",
    "new_york",
    "atlanta",
]
EXCLUDED_CITIES = ["miami", "seattle", "san_francisco", "dallas"]

SOURCE_ELIGIBLE = PROJECT_ROOT / "reports" / "backtest_eligible_dates.csv"
FILTERED_ELIGIBLE = PROJECT_ROOT / "reports" / "backtest_eligible_dates_6city.csv"
TRADES_DIR_6 = PROJECT_ROOT / "data" / "backtest_trades_6city"
EQUITY_DIR_6 = PROJECT_ROOT / "data" / "backtest_equity_6city"
REPORT_JSON_6 = PROJECT_ROOT / "reports" / "backtest_results_6city.json"
REPORT_MD_6 = PROJECT_ROOT / "reports" / "backtest_results_6city.md"
COMPARISON_MD = PROJECT_ROOT / "reports" / "backtest_6city_comparison.md"
FULL_RESULTS_JSON = PROJECT_ROOT / "reports" / "full_backtest_results.json"

VARIANTS = [
    "modal_maker_hold_to_settlement",
    "modal_maker_profit_target_15c",
    "ngboost_kelly_hold_to_settlement",
    "ngboost_kelly_profit_target_15c",
]
PROJ_TRADES_MIN = 80
CI_LOWER_MIN = -3.0
MAX_DD_PCT_LIMIT = -0.30
PNL_CONCENTRATION_MAX = 0.50
N_VARIANTS = len(VARIANTS)


def _patch_output_dirs() -> None:
    """Point step modules at 6-city output dirs without editing their source files."""
    import backtest.common as common
    import backtest.step2_modal_maker as step2
    import backtest.step3_ngboost_kelly as step3
    import backtest.step4_mcp_simulation as step4

    common.TRADES_DIR = TRADES_DIR_6
    common.EQUITY_DIR = EQUITY_DIR_6
    step2.TRADES_DIR = TRADES_DIR_6
    step3.TRADES_DIR = TRADES_DIR_6
    step4.TRADES_DIR = TRADES_DIR_6
    step4.EQUITY_DIR = EQUITY_DIR_6


def load_filtered_eligible() -> pd.DataFrame:
    if not SOURCE_ELIGIBLE.exists():
        raise FileNotFoundError(f"Missing {SOURCE_ELIGIBLE} — run step1 first")
    df = pd.read_csv(SOURCE_ELIGIBLE)
    df["city"] = df["city"].astype(str)
    filtered = df[df["city"].isin(INCLUDED_CITIES)].copy()
    FILTERED_ELIGIBLE.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(FILTERED_ELIGIBLE, index=False)
    return filtered


def trade_log_path_6(variant: str) -> Path:
    if variant.startswith("modal_maker_"):
        return TRADES_DIR_6 / f"modal_maker_{variant.removeprefix('modal_maker_')}.jsonl"
    return TRADES_DIR_6 / f"ngboost_kelly_{variant.removeprefix('ngboost_kelly_')}.jsonl"


def equity_path_6(variant: str) -> Path:
    return EQUITY_DIR_6 / f"{variant}.csv"


def run_steps_2_to_4(eligible: pd.DataFrame, force: bool) -> None:
    import backtest.step2_modal_maker as step2
    import backtest.step3_ngboost_kelly as step3
    import backtest.step4_mcp_simulation as step4

    TRADES_DIR_6.mkdir(parents=True, exist_ok=True)
    EQUITY_DIR_6.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 2: modal_maker ({len(eligible)} city-dates) ===")
    for variant in step2.EXIT_VARIANTS:
        step2.run_variant(variant, eligible, force)

    print(f"\n=== Step 3: ngboost_kelly ({len(eligible)} city-dates) ===")
    for variant in step3.EXIT_VARIANTS:
        step3.run_variant(variant, eligible, force)

    print("\n=== Step 4: MCP equity simulation ===")
    for variant in step4.VARIANTS:
        step4.simulate_variant(variant, force)


def compute_metrics(
    variant: str,
    eligible_days: int,
    sample_days: int,
) -> dict:
    from backtest.common import INITIAL_BANKROLL_USD, read_jsonl
    from backtest_utils import bootstrap_sharpe, deflated_sharpe, sharpe_stats

    trades = read_jsonl(trade_log_path_6(variant))
    if not trades:
        return {"variant": variant, "error": "no trades"}

    traded = [t for t in trades if t.get("traded")]
    dates = sorted({str(t["date"]) for t in trades})
    by_date: dict[str, float] = {d: 0.0 for d in dates}
    for t in trades:
        if not t.get("traded"):
            continue
        d = str(t["date"])
        by_date[d] = by_date.get(d, 0.0) + float(t.get("pnl_usd", 0.0))
    capital = INITIAL_BANKROLL_USD * 100
    daily_rets = pd.Series(
        {(d): (by_date[d] * 100) / capital for d in dates},
        dtype=float,
    ).sort_index()

    stats = sharpe_stats(daily_rets)
    boot = bootstrap_sharpe(daily_rets, n_boot=2000)
    dsr = deflated_sharpe(daily_rets, N_VARIANTS)

    equity_file = equity_path_6(variant)
    equity_df = pd.read_csv(equity_file) if equity_file.exists() else pd.DataFrame()
    final_bankroll = (
        float(equity_df["bankroll_usd"].iloc[-1]) if not equity_df.empty else INITIAL_BANKROLL_USD
    )
    eliminated = bool(equity_df["eliminated"].any()) if "eliminated" in equity_df.columns else False

    pnls = [float(t["pnl_usd"]) for t in traded]
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    daily_pnl = daily_rets * INITIAL_BANKROLL_USD
    top3 = daily_pnl.nlargest(3).sum() if len(daily_pnl) else 0.0
    concentration = abs(top3 / total_pnl) if total_pnl != 0 else 0.0

    per_city: dict[str, float] = {}
    for t in traded:
        c = str(t["city"])
        per_city[c] = per_city.get(c, 0.0) + float(t["pnl_usd"])

    proj_60d = (len(traded) / sample_days * 60) if sample_days > 0 else 0.0

    return {
        "variant": variant,
        "n_trades": len(traded),
        "n_days": len(dates),
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "total_pnl_usd": round(total_pnl, 4),
        "sharpe": stats["sharpe_annual"],
        "bootstrap_ci_low": boot["sharpe_boot_ci_low"],
        "bootstrap_ci_high": boot["sharpe_boot_ci_high"],
        "sortino": stats["sortino_annual"],
        "max_drawdown_usd": stats["max_drawdown"] * INITIAL_BANKROLL_USD,
        "max_drawdown_pct": stats["max_drawdown"],
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


def write_6city_report(results: dict, gonogo_all: dict, eligible_days: int) -> None:
    lines = [
        "# Backtest Results (6-city restricted)\n\n",
        f"Included: {', '.join(INCLUDED_CITIES)}\n\n",
        f"Excluded: {', '.join(EXCLUDED_CITIES)}\n\n",
        f"Eligible city-dates: {eligible_days}\n\n",
        "## Variant metrics\n\n",
        "| Variant | Trades | Sharpe [CI] | Sortino | Max DD | Win% | Total PnL | Proj/60d |\n",
        "|---------|-------:|-------------|--------:|-------:|-----:|----------:|---------:|\n",
    ]
    for variant, m in results.items():
        ci = f"[{m['bootstrap_ci_low']:.2f}, {m['bootstrap_ci_high']:.2f}]"
        lines.append(
            f"| {variant} | {m['n_trades']} | {m['sharpe']:.2f} {ci} | {m['sortino']:.2f} | "
            f"${m['max_drawdown_usd']:.2f} | {100 * m['win_rate']:.1f}% | "
            f"${m['total_pnl_usd']:.2f} | {m['projected_trades_60d']:.0f} |\n"
        )

    lines.append("\n## GO/NO-GO matrix\n\n")
    criteria = list(next(iter(gonogo_all.values())).keys()) if gonogo_all else []
    lines.append("| Criterion | " + " | ".join(VARIANTS) + " |\n")
    lines.append("|" + "---|" * (len(VARIANTS) + 1) + "\n")
    for crit in criteria:
        row = f"| {crit} |"
        for v in VARIANTS:
            val = gonogo_all[v].get(crit, False)
            row += f" {'PASS' if val else 'FAIL'} |"
        lines.append(row + "\n")

    lines.append("\n## Per-city PnL (ngboost_kelly hold_to_settlement)\n\n")
    kelly = results.get("ngboost_kelly_hold_to_settlement", {})
    for city, pnl in kelly.get("per_city_pnl", {}).items():
        lines.append(f"- {city}: ${pnl:.2f}\n")

    REPORT_MD_6.write_text("".join(lines), encoding="utf-8")
    payload = {
        "included_cities": INCLUDED_CITIES,
        "excluded_cities": EXCLUDED_CITIES,
        "eligible_days": eligible_days,
        "variants": results,
        "gonogo": gonogo_all,
    }
    REPORT_JSON_6.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {REPORT_MD_6}")
    print(f"Wrote {REPORT_JSON_6}")


def edge_threshold_for_trades(n_trades: int, sample_days: int, target: int = 80) -> float | None:
    """Minimum per-trade edge (approx) if trades scale linearly with edge filter."""
    if n_trades <= 0 or sample_days <= 0:
        return None
    rate = n_trades / sample_days
    if rate <= 0:
        return None
    return target / (60 * rate)


def write_comparison_report(
    full: dict,
    six: dict,
    eligible_6: pd.DataFrame,
    per_city_delta: dict[str, dict[str, float]],
) -> None:
    sample_days = len(eligible_6["date"].unique())
    baseline_key = "ngboost_kelly_hold_to_settlement"
    full_k = full["variants"][baseline_key]
    six_k = six["variants"][baseline_key]

    def row(metric: str, fval, sval, fmt: str = "{:.2f}") -> str:
        try:
            delta = sval - fval if isinstance(fval, (int, float)) and isinstance(sval, (int, float)) else ""
            if delta != "":
                delta_s = f"{delta:+.2f}" if isinstance(delta, float) else str(delta)
            else:
                delta_s = "—"
            fs = fmt.format(fval) if isinstance(fval, (int, float)) else str(fval)
            ss = fmt.format(sval) if isinstance(sval, (int, float)) else str(sval)
        except (TypeError, ValueError):
            fs, ss, delta_s = str(fval), str(sval), "—"
        return f"| {metric} | {fs} | {ss} | {delta_s} |\n"

    lines = [
        "# 6-City vs 10-City Backtest Comparison\n\n",
        f"**Included cities:** {', '.join(INCLUDED_CITIES)}\n\n",
        f"**Excluded cities:** {', '.join(EXCLUDED_CITIES)} "
        "(miscalibrated per NGBoost v3 calibration audit)\n\n",
        f"**Model:** same as full backtest (`reports/backtest_model_path.txt` → ngboost_v2)\n\n",
        f"**Eligible city-dates:** 10-city {full.get('eligible_days', '?')} → "
        f"6-city {six.get('eligible_days', '?')}\n\n",
        "## 1. Side-by-side metrics (ngboost_kelly hold_to_settlement)\n\n",
        "| Metric | 10-city | 6-city | Delta |\n",
        "|--------|---------|--------|-------|\n",
        row("N trades", full_k["n_trades"], six_k["n_trades"], "{:.0f}"),
        row("Win rate", 100 * full_k["win_rate"], 100 * six_k["win_rate"], "{:.1f}%"),
        row("Total PnL ($)", full_k["total_pnl_usd"], six_k["total_pnl_usd"], "{:.2f}"),
        f"| Sharpe [CI] | {full_k['sharpe']:.2f} [{full_k['bootstrap_ci_low']:.2f}, {full_k['bootstrap_ci_high']:.2f}] | "
        f"{six_k['sharpe']:.2f} [{six_k['bootstrap_ci_low']:.2f}, {six_k['bootstrap_ci_high']:.2f}] | "
        f"{six_k['sharpe'] - full_k['sharpe']:+.2f} |\n",
        row("Sortino", full_k["sortino"], six_k["sortino"]),
        row("Max DD ($)", full_k["max_drawdown_usd"], six_k["max_drawdown_usd"]),
        row("Proj trades/60d", full_k["projected_trades_60d"], six_k["projected_trades_60d"]),
        row("Final bankroll ($)", full_k["final_bankroll_usd"], six_k["final_bankroll_usd"]),
        f"| Eliminated | {'Yes' if full_k['eliminated'] else 'No'} | "
        f"{'Yes' if six_k['eliminated'] else 'No'} | — |\n",
        "\n## All variants comparison\n\n",
        "| Variant | 10-city Sharpe | 6-city Sharpe | 10-city PnL | 6-city PnL | 10-city Trades | 6-city Trades |\n",
        "|---------|---------------:|--------------:|------------:|-----------:|---------------:|--------------:|\n",
    ]
    for v in VARIANTS:
        fm = full["variants"][v]
        sm = six["variants"][v]
        lines.append(
            f"| {v} | {fm['sharpe']:.2f} | {sm['sharpe']:.2f} | "
            f"${fm['total_pnl_usd']:.2f} | ${sm['total_pnl_usd']:.2f} | "
            f"{fm['n_trades']} | {sm['n_trades']} |\n"
        )

    lines.append("\n## 2. Per-city PnL (6 included cities, ngboost_kelly hold_to_settlement)\n\n")
    lines.append(
        "| City | 10-city PnL | 6-city PnL | Delta | Note |\n"
        "|------|------------:|-----------:|------:|------|\n"
    )
    full_pc = full_k.get("per_city_pnl", {})
    six_pc = six_k.get("per_city_pnl", {})
    for city in INCLUDED_CITIES:
        f_pnl = full_pc.get(city, 0.0)
        s_pnl = six_pc.get(city, 0.0)
        delta = s_pnl - f_pnl
        note = "Kelly sizing changed" if abs(delta) > 0.01 else "unchanged"
        lines.append(f"| {city} | ${f_pnl:.2f} | ${s_pnl:.2f} | ${delta:+.2f} | {note} |\n")

    lines.append(
        "\nKelly allocation is computed **per calendar day across all eligible cities**. "
        "Removing 4 cities changes the daily bet pool and regional caps, so per-city PnL "
        "can differ even when the same city-dates trade.\n"
    )

    lines.append("\n## 3. GO/NO-GO matrix (6-city ngboost_kelly hold_to_settlement)\n\n")
    g6 = six["gonogo"]["ngboost_kelly_hold_to_settlement"]
    g10 = full["gonogo"]["ngboost_kelly_hold_to_settlement"]
    lines.append("| Criterion | 10-city | 6-city |\n|---|---|---|\n")
    for crit in g6:
        lines.append(
            f"| {crit} | {'PASS' if g10[crit] else 'FAIL'} | {'PASS' if g6[crit] else 'FAIL'} |\n"
        )

    e_star = edge_threshold_for_trades(six_k["n_trades"], sample_days, PROJ_TRADES_MIN)
    lines.extend(
        [
            "\n## 4. Kelly allocation effect\n\n",
            "- Removing miami/seattle/sf/dallas eliminates large negative PnL buckets "
            f"(${full_pc.get('miami', 0):.2f} + ${full_pc.get('seattle', 0):.2f} + "
            f"${full_pc.get('san_francisco', 0):.2f} + ${full_pc.get('dallas', 0):.2f} "
            f"= ${sum(full_pc.get(c, 0) for c in EXCLUDED_CITIES):.2f} in 10-city Kelly backtest).\n",
            "- Remaining 6 cities alone summed to roughly "
            f"${sum(full_pc.get(c, 0) for c in INCLUDED_CITIES):.2f} in the 10-city run, "
            f"but 6-city isolated run total PnL is ${six_k['total_pnl_usd']:.2f} because "
            "daily Kelly budget and regional caps are re-optimized with fewer competing bets.\n",
            "\n## 5. Trade count / MCP viability\n\n",
            f"- 6-city projected trades/60d: **{six_k['projected_trades_60d']:.0f}** "
            f"(MCP minimum gate: {PROJ_TRADES_MIN})\n",
        ]
    )
    if e_star is not None and six_k["projected_trades_60d"] < PROJ_TRADES_MIN:
        lines.append(
            f"- Rough scaling: to reach {PROJ_TRADES_MIN} trades/60d at current edge hit rate, "
            f"trade frequency would need to rise ~{PROJ_TRADES_MIN / max(six_k['projected_trades_60d'], 0.1):.1f}× "
            f"(equivalent to lowering minimum edge threshold proportionally).\n"
        )
    else:
        lines.append("- Projected trade count meets the 80-trade MCP gate.\n")

    sharpe_positive = six_k["sharpe"] > 0
    lines.append("\n## 6. Recommendation\n\n")
    if sharpe_positive and not six_k["eliminated"]:
        lines.append(
            "6-city ngboost_kelly shows improved risk-adjusted returns vs the 10-city run. "
            "Consider deploying Kelly on the 6 included cities only, pending live validation. "
            "Trade volume may still be below MCP minimum — monitor projected trades.\n"
        )
    elif six_k["total_pnl_usd"] > full_k["total_pnl_usd"]:
        lines.append(
            "6-city restriction improves total PnL but Sharpe remains non-positive or marginal. "
            "City selection helps; model edge is not yet convincing for autonomous deployment.\n"
        )
    else:
        lines.append(
            "Do not deploy 6-city Kelly yet — restriction alone does not produce a robust positive-Sharpe backtest.\n"
        )

    COMPARISON_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {COMPARISON_MD}")


def verify_10city_preserved() -> None:
    paths = [
        PROJECT_ROOT / "reports" / "full_backtest_results.json",
        PROJECT_ROOT / "data" / "backtest_trades" / "ngboost_kelly_hold_to_settlement.jsonl",
        PROJECT_ROOT / "data" / "backtest_equity" / "ngboost_kelly_hold_to_settlement.csv",
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"10-city artifact missing (should be preserved): {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restricted 6-city backtest wrapper")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    parser.add_argument("--skip-run", action="store_true", help="Only regenerate reports from existing 6city outputs")
    args = parser.parse_args()

    print("=== Restricted 6-city backtest ===")
    print(f"Included: {INCLUDED_CITIES}")
    print(f"Excluded: {EXCLUDED_CITIES}")

    verify_10city_preserved()
    _patch_output_dirs()

    eligible = load_filtered_eligible()
    print(f"\nFiltered eligible: {len(eligible)} city-dates ({len(eligible['date'].unique())} unique dates)")
    print(f"Wrote {FILTERED_ELIGIBLE}")

    if not args.skip_run:
        run_steps_2_to_4(eligible, force=args.force)

    eligible_days = len(eligible)
    sample_days = len(eligible["date"].unique()) if not eligible.empty else 1

    results_6: dict[str, dict] = {}
    for variant in VARIANTS:
        results_6[variant] = compute_metrics(variant, eligible_days, sample_days)

    baseline = results_6["modal_maker_hold_to_settlement"]["sharpe"]
    gonogo_6 = {v: gonogo(m, baseline) for v, m in results_6.items()}
    write_6city_report(results_6, gonogo_6, eligible_days)

    full_payload = json.loads(FULL_RESULTS_JSON.read_text(encoding="utf-8"))
    six_payload = json.loads(REPORT_JSON_6.read_text(encoding="utf-8"))
    write_comparison_report(full_payload, six_payload, eligible, {})

    verify_10city_preserved()
    print("\nVerified: 10-city artifacts unchanged.")


if __name__ == "__main__":
    main()
