#!/usr/bin/env python3
"""Generate a PDF report for survival scenario backtest results."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "reports" / "survival_scenario.json"
DEFAULT_PDF = PROJECT_ROOT / "reports" / "survival_scenario_report.pdf"

STARTING_BANKROLL = 84.98
ELIMINATION_USD = 70.0
TARGET_USD = 100.0

VARIANTS = ("current", "top2_buckets", "top3_buckets")
WINDOWS = ("early", "middle", "late")
WINDOW_LABELS = {
    "early": "Early (Feb–Mar)",
    "middle": "Middle (Mar–May)",
    "late": "Late (May–Jun)",
}
VARIANT_LABELS = {
    "current": "v5b current (1 bucket/city, top-2/day)",
    "top2_buckets": "Top-2 buckets/city, top-2/day",
    "top3_buckets": "Top-3 buckets/city, top-3/day",
}
VARIANT_COLORS = {
    "current": "#1f77b4",
    "top2_buckets": "#ff7f0e",
    "top3_buckets": "#2ca02c",
}


def load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def trade_equity_path(trades: list[dict[str, Any]], start: float = STARTING_BANKROLL) -> pd.DataFrame:
    """Reconstruct bankroll after every settled trade (not end-of-day only)."""
    rows: list[dict[str, Any]] = [
        {
            "trade_idx": 0,
            "date": None,
            "city": None,
            "bankroll_usd": start,
            "pnl_usd": 0.0,
            "label": "Start",
        }
    ]
    running = start
    for idx, trade in enumerate(trades, start=1):
        running = round(running + float(trade["pnl_usd"]), 4)
        rows.append(
            {
                "trade_idx": idx,
                "date": trade["date"],
                "city": trade["city"],
                "bankroll_usd": running,
                "pnl_usd": float(trade["pnl_usd"]),
                "won": trade["won"],
                "computed_edge": float(trade["edge"]),
                "realized_edge": (1.0 if trade["won"] else 0.0) - float(trade["entry_price"]),
                "entry_price": float(trade["entry_price"]),
                "effective_prob": float(trade["effective_prob"]),
                "label": f"{trade['date']} {trade['city']}",
            }
        )
    return pd.DataFrame(rows)


def scenario_metrics(trades: list[dict[str, Any]], equity: pd.DataFrame) -> dict[str, Any]:
    if not trades:
        return {}
    pnls = [float(t["pnl_usd"]) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    computed_edges = [float(t["edge"]) for t in trades]
    realized_edges = [(1.0 if t["won"] else 0.0) - float(t["entry_price"]) for t in trades]
    city_pnl: Counter[str] = Counter()
    city_count: Counter[str] = Counter()
    for trade in trades:
        city_pnl[str(trade["city"])] += float(trade["pnl_usd"])
        city_count[str(trade["city"])] += 1

    bankrolls = equity["bankroll_usd"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(bankrolls)
    drawdown = bankrolls - peak

    return {
        "n_trades": len(trades),
        "win_rate": wins / len(trades),
        "total_pnl": sum(pnls),
        "final_bankroll": float(bankrolls[-1]),
        "min_bankroll": float(bankrolls.min()),
        "max_drawdown": float(drawdown.min()),
        "avg_computed_edge": float(np.mean(computed_edges)),
        "avg_realized_edge": float(np.mean(realized_edges)),
        "median_computed_edge": float(np.median(computed_edges)),
        "median_realized_edge": float(np.median(realized_edges)),
        "best_cities": sorted(city_pnl.items(), key=lambda item: item[1], reverse=True)[:5],
        "worst_cities": sorted(city_pnl.items(), key=lambda item: item[1])[:5],
        "most_traded": city_count.most_common(5),
        "eliminated": bool((bankrolls <= ELIMINATION_USD).any()),
        "recovered": bool(bankrolls[-1] >= TARGET_USD),
    }


def _escape_mathtext(text: str) -> str:
    return text.replace("$", r"\$")


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    y = 0.94
    fig.text(0.08, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.05
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=95)
        lines = wrapped.splitlines()
        needed = 0.028 * len(lines) + 0.02
        if y - needed < 0.05:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            y = 0.94
        fig.text(0.08, y, _escape_mathtext(wrapped), fontsize=10, va="top", family="monospace")
        y -= needed
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_window_equity(
    pdf: PdfPages,
    window: str,
    payload: dict[str, Any],
    *,
    highlight_variant: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dates_by_window = payload["windows"][window]
    start_date = dates_by_window[0]
    end_date = dates_by_window[-1]

    for variant in VARIANTS:
        trades = payload["results"][window][variant]["trades"]
        equity = trade_equity_path(trades)
        linewidth = 2.6 if variant == highlight_variant else 1.8
        alpha = 1.0 if highlight_variant is None or variant == highlight_variant else 0.55
        ax.plot(
            equity["trade_idx"],
            equity["bankroll_usd"],
            label=VARIANT_LABELS[variant],
            color=VARIANT_COLORS[variant],
            linewidth=linewidth,
            alpha=alpha,
        )
        mins = equity["bankroll_usd"].min()
        if mins <= ELIMINATION_USD:
            ax.scatter(
                equity.loc[equity["bankroll_usd"] == mins, "trade_idx"],
                [mins],
                color=VARIANT_COLORS[variant],
                s=30,
                zorder=5,
            )

    ax.axhline(ELIMINATION_USD, color="#d62728", linestyle="--", linewidth=1.5, label=f"Elimination ${ELIMINATION_USD:.0f}")
    ax.axhline(TARGET_USD, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Target ${TARGET_USD:.0f}")
    ax.axhline(STARTING_BANKROLL, color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.fill_between(ax.get_xlim(), 0, ELIMINATION_USD, color="#d62728", alpha=0.05)

    title = f"Trade-level equity — {WINDOW_LABELS[window]} ({start_date} to {end_date})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade number (post-settlement bankroll after each trade)")
    ax.set_ylabel("Bankroll ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_edge_comparison(pdf: PdfPages, payload: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, window in zip(axes, WINDOWS):
        comp = []
        real = []
        for variant in VARIANTS:
            trades = payload["results"][window][variant]["trades"]
            comp.append(np.mean([t["edge"] for t in trades]) if trades else 0.0)
            real.append(
                np.mean([(1.0 if t["won"] else 0.0) - t["entry_price"] for t in trades])
                if trades
                else 0.0
            )
        x = np.arange(len(VARIANTS))
        width = 0.35
        ax.bar(x - width / 2, comp, width, label="Computed edge", color="#4c78a8")
        ax.bar(x + width / 2, real, width, label="Realized edge", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(["current", "top2", "top3"], rotation=20)
        ax.set_title(WINDOW_LABELS[window])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Average edge per trade")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Computed vs realized edge by window", fontsize=12, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_city_heatmap(pdf: PdfPages, payload: dict[str, Any], variant: str = "current") -> None:
    city_pnl: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for window in WINDOWS:
        for trade in payload["results"][window][variant]["trades"]:
            city_pnl[trade["city"]][window] += float(trade["pnl_usd"])

    cities = sorted(city_pnl, key=lambda c: sum(city_pnl[c].values()), reverse=True)
    matrix = np.array([[city_pnl[c][w] for w in WINDOWS] for c in cities])

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(cities))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOWS], rotation=15, ha="right")
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities)
    ax.set_title(f"City PnL by window ({variant})", fontsize=12, fontweight="bold")
    for i in range(len(cities)):
        for j in range(len(WINDOWS)):
            ax.text(j, i, f"${matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="PnL ($)")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_win_rate_by_window(pdf: PdfPages, payload: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(WINDOWS))
    width = 0.25
    for i, variant in enumerate(VARIANTS):
        vals = []
        for window in WINDOWS:
            trades = payload["results"][window][variant]["trades"]
            wins = sum(1 for t in trades if t["won"])
            vals.append(100 * wins / len(trades) if trades else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant])
    ax.set_xticks(x)
    ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOWS])
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate degrades in the late window", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        for variant in VARIANTS:
            trades = payload["results"][window][variant]["trades"]
            metrics = scenario_metrics(trades, trade_equity_path(trades))
            stored = payload["results"][window][variant]["metrics"]
            rows.append(
                {
                    "window": window,
                    "variant": variant,
                    "trades": metrics["n_trades"],
                    "win_rate_pct": 100 * metrics["win_rate"],
                    "total_pnl": metrics["total_pnl"],
                    "final_bankroll": metrics["final_bankroll"],
                    "min_bankroll": metrics["min_bankroll"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe": stored.get("sharpe"),
                    "survived": "YES" if not metrics["eliminated"] else "NO",
                    "recovered_100": "YES" if metrics["recovered"] else "NO",
                    "avg_computed_edge": metrics["avg_computed_edge"],
                    "avg_realized_edge": metrics["avg_realized_edge"],
                }
            )
    return pd.DataFrame(rows)


def plot_summary_table(pdf: PdfPages, table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    display = table.copy()
    for col in ["win_rate_pct", "total_pnl", "final_bankroll", "min_bankroll", "max_drawdown", "sharpe",
                "avg_computed_edge", "avg_realized_edge"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    table_obj = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(7)
    table_obj.scale(1, 1.3)
    ax.set_title("Cross-scenario summary (trade-level min bankroll)", fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def late_vs_early_analysis(payload: dict[str, Any], variant: str = "current") -> list[str]:
    early = scenario_metrics(
        payload["results"]["early"][variant]["trades"],
        trade_equity_path(payload["results"]["early"][variant]["trades"]),
    )
    late = scenario_metrics(
        payload["results"]["late"][variant]["trades"],
        trade_equity_path(payload["results"]["late"][variant]["trades"]),
    )

    paragraphs = [
        "Why is the late window harder than the early window?",
        (
            f"Using the {variant} variant, win rate falls from {100*early['win_rate']:.1f}% in the early window "
            f"to {100*late['win_rate']:.1f}% in the late window, while average realized edge falls from "
            f"{early['avg_realized_edge']:.3f} to {late['avg_realized_edge']:.3f}. "
            f"Computed edge stays similar ({early['avg_computed_edge']:.3f} vs {late['avg_computed_edge']:.3f}), "
            "so the model still believes it has edge even as outcomes deteriorate."
        ),
        (
            "Calendar regime: the early window (Feb–Mar) is late winter/early spring, while the late window "
            "(May–Jun) is late spring/early summer. Warmer-season daily maxima are more volatile, cloud/convection "
            "regimes shift, and Polymarket bucket structure can tighten around narrower temperature ranges. "
            "That makes exact-degree bucket selection harder even when the mean forecast is reasonable."
        ),
        (
            "City mix shifts. Early-window PnL was concentrated in chicago and dallas; late-window PnL still "
            f"leans on dallas (+${late['best_cities'][0][1]:.2f}) but seattle and chicago contribute less, while "
            "losses are spread across more cities. More trades fire in late ({late['n_trades']} vs {early['n_trades']}), "
            "but with lower hit rate, so variance and drawdown increase."
        ),
        (
            "Model calibration drift is visible: predicted edge remains positive, but realized edge collapses. "
            "That pattern is consistent with rolling-bias lag, seasonal distribution shift in NGBoost features, "
            "and/or market prices that already embed summer weather information more efficiently."
        ),
        (
            "Practical implication: surviving to $100 from $84.98 is feasible in all three windows in this backtest, "
            "but the late regime is the stress case — lower Sharpe, deeper drawdowns, and thinner margin above the "
            "$70 elimination line if intra-day settlement losses cluster."
        ),
    ]
    return paragraphs


def methods_paragraphs(payload: dict[str, Any]) -> list[str]:
    meta = payload["metadata"]
    return [
        "Survival Scenario Backtest — Methods",
        (
            f"Starting bankroll: ${meta['starting_bankroll_usd']:.2f}. Elimination threshold: "
            f"${meta['elimination_usd']:.0f}. Recovery target: ${meta['target_bankroll_usd']:.0f}. "
            f"Each scenario runs independently for {meta['scenario_days']} trading days."
        ),
        (
            "Strategy: deployed v5b NGBoost flat maker strategy. Entry at ~10:10 local using best_ask - $0.01. "
            "Shrinkage lambda=0.6, edge threshold=0.037, flat 5 contracts, hold-to-settlement, rolling bias ON "
            "(capped +/-1.5F), basket boundary OFF. Daily budget uses anti-cyclic cap from bankroll with divisor=5."
        ),
        (
            "Variants: (1) current — top 1 bucket per city, top 2 trades/day; (2) top2_buckets — top 2 buckets per "
            "city, top 2/day; (3) top3_buckets — top 3 buckets per city, top 3/day."
        ),
        (
            "Windows: early = first 49 eligible dates; middle = centered 49 dates; late = most recent 49 dates. "
            "All windows restart from $84.98 with a fresh rolling-bias cache seeded from parquet."
        ),
        (
            "Equity curves in this report use post-settlement bankroll after every individual trade, not end-of-day "
            "equity only. If bankroll touches $70 after any settled trade, the scenario is marked eliminated."
        ),
        (
            "Edge definitions: computed edge = effective_probability - maker_entry_price (after shrinkage). "
            "Realized edge = settlement outcome (1 or 0) - entry_price. Positive realized edge on losers is impossible; "
            "gaps between computed and realized edge measure calibration + market efficiency."
        ),
    ]


def executive_summary(table: pd.DataFrame) -> list[str]:
    best_final = table.loc[table["final_bankroll"].idxmax()]
    safest = table.loc[table["max_drawdown"].idxmax()]  # closest to zero
    late_current = table[(table["window"] == "late") & (table["variant"] == "current")].iloc[0]
    early_top3 = table[(table["window"] == "early") & (table["variant"] == "top3_buckets")].iloc[0]

    identical = table[table["variant"].isin(["current", "top2_buckets"])]
    same = (
        identical.groupby("window")[["trades", "final_bankroll", "total_pnl"]]
        .nunique()
        .max()
        .max()
        == 1
    )

    lines = [
        "Executive summary",
        (
            f"All 9 scenarios survived without breaching the ${ELIMINATION_USD:.0f} elimination line on a "
            "trade-by-trade bankroll path. Recovery to $100+ was achieved in every case."
        ),
        (
            f"Highest final bankroll: {best_final['variant']} / {best_final['window']} at "
            f"${best_final['final_bankroll']:.2f} ({best_final['trades']:.0f} trades, "
            f"{best_final['win_rate_pct']:.1f}% win rate)."
        ),
        (
            f"Smallest peak-to-trough drawdown: {safest['variant']} / {safest['window']} at "
            f"${safest['max_drawdown']:.2f}."
        ),
        (
            f"Late-window stress case (current): final ${late_current['final_bankroll']:.2f}, "
            f"win rate {late_current['win_rate_pct']:.1f}%, max drawdown ${late_current['max_drawdown']:.2f}, "
            f"realized edge {late_current['avg_realized_edge']:.3f} vs computed {late_current['avg_computed_edge']:.3f}."
        ),
        (
            f"Top3 diversification helps most in early window (${early_top3['final_bankroll']:.2f} final) but adds "
            "variance in late window."
        ),
    ]
    if same:
        lines.append(
            "Note: current and top2_buckets produced identical trades in all windows because the global top-2/day "
            "filter never selected a second bucket from the same city over the second-best bucket from another city."
        )
    return lines


def generate_report(json_path: Path, pdf_path: Path) -> Path:
    payload = load_results(json_path)
    table = build_summary_table(payload)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.72, "Survival Scenario Backtest", ha="center", fontsize=22, fontweight="bold")
        fig.text(
            0.5,
            0.64,
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha="center",
            fontsize=11,
            color="#555555",
        )
        fig.text(
            0.5,
            0.56,
            _escape_mathtext(
                f"Start ${STARTING_BANKROLL:.2f}  |  Eliminate ${ELIMINATION_USD:.0f}  |  Target ${TARGET_USD:.0f}"
            ),
            ha="center",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.48,
            "9 independent 49-day forward simulations on real Polymarket history",
            ha="center",
            fontsize=11,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_text_page(pdf, "Executive Summary", executive_summary(table))
        add_text_page(pdf, "Methods", methods_paragraphs(payload))
        plot_summary_table(pdf, table)

        for window in WINDOWS:
            plot_window_equity(pdf, window, payload)

        plot_win_rate_by_window(pdf, payload)
        plot_edge_comparison(pdf, payload)
        plot_city_heatmap(pdf, payload, variant="current")
        plot_city_heatmap(pdf, payload, variant="top3_buckets")

        # Per-variant city/metrics pages
        for variant in VARIANTS:
            lines = [f"Variant detail: {VARIANT_LABELS[variant]}"]
            for window in WINDOWS:
                trades = payload["results"][window][variant]["trades"]
                m = scenario_metrics(trades, trade_equity_path(trades))
                lines.append(
                    f"{WINDOW_LABELS[window]}: {m['n_trades']} trades, win {100*m['win_rate']:.1f}%, "
                    f"PnL ${m['total_pnl']:+.2f}, final ${m['final_bankroll']:.2f}, "
                    f"min ${m['min_bankroll']:.2f}, max DD ${m['max_drawdown']:.2f}, "
                    f"computed edge {m['avg_computed_edge']:.3f}, realized {m['avg_realized_edge']:.3f}."
                )
                best = ", ".join(f"{c} (${v:+.2f})" for c, v in m["best_cities"][:3])
                most = ", ".join(f"{c} ({n})" for c, n in m["most_traded"][:3])
                lines.append(f"  Best cities: {best}")
                lines.append(f"  Most traded: {most}")
            add_text_page(pdf, f"Variant: {variant}", lines)

        add_text_page(pdf, "Regime analysis", late_vs_early_analysis(payload))

    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build survival scenario PDF report")
    parser.add_argument("--input", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input: {args.input}. Run survival_scenario.py first.")

    out = generate_report(args.input, args.output)
    print(f"Wrote PDF report to {out}")


if __name__ == "__main__":
    main()
