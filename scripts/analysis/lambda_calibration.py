#!/usr/bin/env python3
"""Calibration analysis to choose shrinkage lambda from realized win rates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import backtest.common as bc  # noqa: E402
from sizing import effective_probability  # noqa: E402

EDGE_THRESHOLD = 0.037
WINDOW_DAYS = 49
LAMBDAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
WINDOWS = {
    "early": ("2026-02-03", "2026-03-23"),
    "middle": ("2026-03-18", "2026-05-08"),
    "late": ("2026-05-04", "2026-06-24"),
}
OUTPUT_CSV = PROJECT_ROOT / "data" / "analysis" / "lambda_calibration.csv"
TRADE_FILE = "ngboost_flat_hold_to_settlement.jsonl"


def resolve_trades_path(output_tag: str | None) -> Path:
    if output_tag:
        bc.configure_output_tag(output_tag)
    path = bc.TRADES_DIR / TRADE_FILE
    if path.exists():
        return path
    for tag in ("v5b", "v5", ""):
        if tag:
            bc.configure_output_tag(tag)
        candidate = bc.TRADES_DIR / TRADE_FILE
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {TRADE_FILE} found under data/backtest_trades*; run step3 backtest first."
    )


def load_trades(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_and_prepare(trades: list[dict], lam_used: float) -> pd.DataFrame:
    if not trades:
        raise ValueError("No trade records found")

    print("Trade record keys:", sorted(trades[0].keys()))

    rows: list[dict] = []
    for rec in trades:
        if not rec.get("traded"):
            continue
        if "won" not in rec or rec["won"] is None:
            raise AssertionError(f"Missing settlement outcome for trade on {rec.get('date')}")
        date_str = str(rec["date"])
        entry_price = float(rec["entry_price"])
        p_eff_stored = float(rec.get("effective_prob", rec.get("model_prob")))
        raw_prob = rec.get("raw_prob")
        if raw_prob is None:
            if lam_used <= 0:
                raise AssertionError("lam_used must be > 0 to recover raw probability")
            raw_prob = (p_eff_stored - (1.0 - lam_used) * entry_price) / lam_used
        raw_prob = float(raw_prob)
        if not (0.0 < raw_prob < 1.0):
            raise AssertionError(
                f"Recovered raw_prob out of range for {date_str} {rec.get('city')}: {raw_prob}"
            )
        rows.append(
            {
                "date": date_str,
                "city": rec.get("city"),
                "bucket": rec.get("bucket"),
                "entry_price": entry_price,
                "raw_prob": raw_prob,
                "won": bool(rec["won"]),
            }
        )
    return pd.DataFrame(rows)


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    return sub.reset_index(drop=True)


def evaluate_lambda(window_df: pd.DataFrame, lam: float) -> dict:
    p_eff = window_df.apply(
        lambda row: effective_probability(row["raw_prob"], row["entry_price"], lam),
        axis=1,
    )
    edge = p_eff - window_df["entry_price"]
    kept = window_df[edge >= EDGE_THRESHOLD].copy()
    kept_p_eff = p_eff[edge >= EDGE_THRESHOLD]
    n_trades = len(kept)
    if n_trades == 0:
        return {
            "lambda": lam,
            "n_trades": 0,
            "mean_p_eff": float("nan"),
            "win_rate": float("nan"),
            "calibration_gap": float("nan"),
            "trades_per_day": 0.0,
        }
    win_rate = float(kept["won"].mean())
    mean_p_eff = float(kept_p_eff.mean())
    return {
        "lambda": lam,
        "n_trades": n_trades,
        "mean_p_eff": mean_p_eff,
        "win_rate": win_rate,
        "calibration_gap": mean_p_eff - win_rate,
        "trades_per_day": n_trades / WINDOW_DAYS,
    }


def print_window_table(window: str, start: str, end: str, results: list[dict]) -> None:
    print(f"\nWindow: {window} ({start} to {end})")
    print("-" * 88)
    print(
        f"{'lambda':>6} {'n_trades':>8} {'mean_p_eff':>11} {'win_rate':>9} "
        f"{'cal_gap':>9} {'trades/day':>11}"
    )
    for row in results:
        print(
            f"{row['lambda']:6.2f} {row['n_trades']:8d} {row['mean_p_eff']:11.4f} "
            f"{row['win_rate']:9.4f} {row['calibration_gap']:+9.4f} {row['trades_per_day']:11.2f}"
        )

    eligible = [r for r in results if r["trades_per_day"] >= 1.0 and pd.notna(r["calibration_gap"])]
    if eligible:
        best = min(eligible, key=lambda r: abs(r["calibration_gap"]))
        print(
            f"Best lambda (|gap| min, trades/day >= 1.0): {best['lambda']:.2f} "
            f"(gap {best['calibration_gap']:+.4f}, {best['trades_per_day']:.2f} trades/day)"
        )
    else:
        print("Best lambda: none met trades/day >= 1.0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shrinkage lambda calibration from settled trades")
    parser.add_argument("--output-tag", default=None, help="Backtest trades tag, e.g. v5b")
    parser.add_argument("--trades-path", type=Path, default=None, help="Override JSONL trade log path")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    trades_path = args.trades_path or resolve_trades_path(args.output_tag)
    config = bc.load_trading_config()
    lam_used = float(config.get("shrinkage_lambda", 0.6))

    print(f"Loading trades from {trades_path}")
    trades = load_trades(trades_path)
    df = verify_and_prepare(trades, lam_used)
    print(f"Prepared {len(df)} settled trades with recovered raw probabilities (lam_used={lam_used})")

    all_rows: list[dict] = []
    for window, (start, end) in WINDOWS.items():
        window_df = filter_window(df, start, end)
        if window_df.empty:
            raise AssertionError(f"No trades in window {window} ({start} to {end})")
        results = [evaluate_lambda(window_df, lam) for lam in LAMBDAS]
        for row in results:
            row["window"] = window
            row["start_date"] = start
            row["end_date"] = end
            all_rows.append(row)
        print_window_table(window, start, end, results)

    out_df = pd.DataFrame(all_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
