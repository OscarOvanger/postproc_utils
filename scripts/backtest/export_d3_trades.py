#!/usr/bin/env python3
"""Export D3 pace-amendment trade-level records for report figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest.common as bc  # noqa: E402
from backtest.ngboost_inference import NgBoostBacktestModels  # noqa: E402
from backtest.survival_scenario import (  # noqa: E402
    CONFIG_D_EXCLUDED,
    STARTING_BANKROLL_USD,
    VARIANTS,
    apply_pace_variant,
    build_config_d_base,
    build_windows,
    check_prerequisites,
    filter_eligible_cities,
    load_runtime_dependencies,
    run_scenario,
)
from src.poly_trading_pipeline import load_wunderground_bias  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "pace_amendment_d3_trades"


def main() -> None:
    check_prerequisites()
    load_runtime_dependencies()
    eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
    eligible["date"] = eligible["date"].astype(str)
    eligible["city"] = eligible["city"].astype(str)
    all_dates = sorted(eligible["date"].unique().tolist())
    windows = build_windows(all_dates)
    models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
    wu = bc.load_wu_targets()
    wu_bias = load_wunderground_bias()

    base_config = build_config_d_base()
    cfg = apply_pace_variant(base_config, "D3")
    filtered_eligible = filter_eligible_cities(eligible, CONFIG_D_EXCLUDED)
    variant_config = VARIANTS["current"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting D3 trades (start=${STARTING_BANKROLL_USD:.2f}) -> {OUTPUT_DIR}")

    for window_name, dates in windows.items():
        scenario = run_scenario(
            "D3",
            variant_config,
            window_name,
            dates,
            filtered_eligible[filtered_eligible["date"].isin(dates)].copy(),
            models,
            cfg,
            wu,
            wu_bias,
        )
        out_path = OUTPUT_DIR / f"d3_{window_name}_trades.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for trade in scenario["trades"]:
                if trade.get("traded"):
                    handle.write(json.dumps(trade, default=str) + "\n")
        metrics = scenario["metrics"]
        print(
            f"  {window_name}: {metrics['trades']} trades, "
            f"final=${metrics['final_bankroll_usd']:.2f}, "
            f"min_br=${metrics['min_bankroll_usd']:.2f} -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
