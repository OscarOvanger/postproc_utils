#!/usr/bin/env python3
"""Step 3: HRRR-NGBoost + Kelly strategy on real Polymarket data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import backtest.common as bc  # noqa: E402
from backtest.ngboost_inference import (  # noqa: E402
    NgBoostBacktestModels,
    describe_two_piece_mode,
    parse_snapshot_bucket,
    predict_bucket_probs_from_mu,
    predict_mu_sigma,
    two_piece_ratio_for_date,
)
from ngboost_kelly import BetCandidate, allocate_ngboost_kelly  # noqa: E402
from rolling_bias import RollingBiasCache  # noqa: E402
from poly_trading_pipeline import basket_companion_label, load_wunderground_bias  # noqa: E402
from sizing import seasonal_shrinkage_lambda  # noqa: E402

EXIT_VARIANTS = ["hold_to_settlement", "profit_target_15c"]
PROFIT_TARGET_MIN_ENTRY = 0.55


def _evaluate_quote(
    label: str,
    qrow,
    probs: dict[str, float],
    lam: float,
) -> BetCandidate | None:
    if label.startswith("Will "):
        return None
    ask = qrow.get("best_ask")
    if ask is None:
        return None
    cost = round(float(ask) - bc.MAKER_TICK, 4)
    if cost < bc.PRICE_FLOOR:
        return None
    raw_prob = probs.get(label, 0.0)
    p_eff = bc.effective_probability(raw_prob, cost, lam)
    edge = p_eff - cost
    if edge <= 0:
        return None
    return BetCandidate(
        city="",
        bucket=label,
        prob=p_eff,
        cost=cost,
        region="other",
    )


def collect_day_bets(
    day_rows: pd.DataFrame,
    models: NgBoostBacktestModels,
    config: dict,
    bias_cache: RollingBiasCache,
    wu_bias: dict,
    *,
    disable_rolling_bias: bool = False,
    disable_basket: bool = False,
) -> tuple[list[BetCandidate], dict[str, float], dict[str, str]]:
    if day_rows.empty:
        return [], {}, {}
    date_str = str(day_rows.iloc[0]["date"])
    lam = seasonal_shrinkage_lambda(config, date_str)
    ratio_down = two_piece_ratio_for_date(config, date_str)
    margin = float(config.get("basket_boundary_margin_f", 1.0))
    if disable_basket:
        margin = 0.0
    candidates: list[BetCandidate] = []
    raw_mu_by_city: dict[str, float] = {}
    skip_reasons: dict[str, str] = {}

    for _, row in day_rows.iterrows():
        city = str(row["city"])
        date_str = str(row["date"])
        if not bc.features_eligible_cached(city, date_str):
            continue
        cloud_cover = bc.peak_cloud_cover_for_day(city, date_str)
        if bc.convective_skip(city, date_str, cloud_cover, config):
            if cloud_cover is None:
                skip_reasons[city] = "convective_skip (missing cloud cover)"
            else:
                skip_reasons[city] = f"convective_skip cloud={cloud_cover:.3f}"
            continue
        frame = bc.load_day_snapshot(city, date_str)
        if frame is None:
            continue

        snap_rows, entry_ts, _ = bc.select_entry_snapshot(frame, city, date_str)
        if snap_rows.empty:
            continue

        quotes = bc.quotes_at_entry(snap_rows)
        bucket_labels = [
            str(b) for b in quotes["bucket"].astype(str).tolist()
            if not str(b).startswith("Will ")
        ]
        if not bucket_labels:
            continue

        mu_sigma = predict_mu_sigma(models, city, date_str)
        if mu_sigma is None:
            continue
        mu, _sigma = mu_sigma
        raw_mu_by_city[city] = mu
        static_bias = float(wu_bias.get(city, {}).get("median_bias", 0.0))
        mu_wu = mu - static_bias
        bias = 0.0 if disable_rolling_bias else bias_cache.bias(city, date_str)
        mu_adj = mu_wu - bias
        probs = predict_bucket_probs_from_mu(
            models, city, date_str, bucket_labels, mu_adj, ratio_down=ratio_down
        )
        if not probs:
            continue

        city_candidates: list[BetCandidate] = []
        best: BetCandidate | None = None
        for _, qrow in quotes.iterrows():
            label = str(qrow["bucket"])
            cand = _evaluate_quote(label, qrow, probs, lam)
            if cand is None:
                continue
            cand.city = city
            cand.region = bc.CITY_TO_REGION.get(city, "other")
            city_candidates.append(cand)
            if best is None or (cand.prob - cand.cost) > (best.prob - best.cost):
                best = cand

        if best is None:
            continue

        candidates.append(best)

        range_buckets: list[tuple[str, int, int]] = []
        for label in bucket_labels:
            try:
                parsed = parse_snapshot_bucket(label)
            except ValueError:
                continue
            if parsed["type"] != "RANGE":
                continue
            range_buckets.append((label, int(parsed["lower"]), int(parsed["upper"])))

        companion_label = basket_companion_label(mu_adj, best.bucket, range_buckets, margin)
        if companion_label:
            for _, qrow in quotes.iterrows():
                if str(qrow["bucket"]) != companion_label:
                    continue
                cand = _evaluate_quote(companion_label, qrow, probs, lam)
                if cand is None:
                    break
                cand.city = city
                cand.region = bc.CITY_TO_REGION.get(city, "other")
                candidates.append(cand)
                break

    return candidates, raw_mu_by_city, skip_reasons


def record_day_residuals(
    day_rows: pd.DataFrame,
    raw_mu_by_city: dict[str, float],
    bias_cache: RollingBiasCache,
    wu: pd.DataFrame,
    wu_bias: dict,
    date_str: str,
) -> None:
    for _, row in day_rows.iterrows():
        city = str(row["city"])
        if city not in raw_mu_by_city:
            continue
        wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
        if wu_row.empty:
            continue
        actual = float(wu_row.iloc[0]["wunderground_tmax"])
        static_bias = float(wu_bias.get(city, {}).get("median_bias", 0.0))
        corrected_mu = raw_mu_by_city[city] - static_bias
        bias_cache.record(city, date_str, corrected_mu, actual)


def filter_top_bets(
    bets: list[BetCandidate],
    config: dict,
    edge_threshold: float = 0.0,
) -> list[BetCandidate]:
    filtered = [b for b in bets if (b.prob - b.cost) >= edge_threshold]
    filtered.sort(key=lambda b: b.prob - b.cost, reverse=True)
    return filtered[: bc.max_trades_per_day(config)]


def flat_contracts(bankroll: float, config: dict) -> int:  # noqa: ARG001
    n_contracts = int(config.get("n_contracts_default", 5))
    assert n_contracts == 5, f"Polymarket minimum order size is 5 contracts, got {n_contracts}"
    return n_contracts


def settle_trade(
    city: str,
    date_str: str,
    bucket: str,
    entry_price: float,
    n_contracts: int,
    exit_variant: str,
    wu: pd.DataFrame,
) -> dict:
    wu_row = wu[(wu["city"] == city) & (wu["date"] == date_str)]
    actual = float(wu_row.iloc[0]["wunderground_tmax"])
    won = bc.temp_in_bucket(actual, bucket)
    exit_type = "settlement"
    exit_price = 1.0 if won else 0.0
    pnl = bc.settlement_pnl(n_contracts=n_contracts, entry_price=entry_price, won=won)

    if exit_variant == "profit_target_15c" and entry_price >= PROFIT_TARGET_MIN_ENTRY:
        frame = bc.load_day_snapshot(city, date_str)
        if frame is not None:
            _, entry_ts, _ = bc.select_entry_snapshot(frame, city, date_str)
            intraday = bc.intraday_snapshots_after_entry(frame, city, date_str, entry_ts)
            hit, target_price = bc.check_profit_target_exit(intraday, bucket, entry_price)
            if hit and target_price is not None:
                exit_type = "profit_target_15c"
                exit_price = target_price
                pnl = bc.profit_target_pnl(n_contracts, entry_price, target_price)

    return {
        "exit_type": exit_type,
        "exit_price": exit_price,
        "pnl_usd": pnl,
        "won": won if exit_type == "settlement" else pnl > 0,
        "actual_tmax": actual,
    }


def run_kelly_variant(
    exit_variant: str,
    eligible: pd.DataFrame,
    config: dict,
    force: bool,
    wu_bias: dict,
    *,
    disable_rolling_bias: bool = False,
    disable_basket: bool = False,
) -> None:
    out_path = bc.TRADES_DIR / f"ngboost_kelly_{exit_variant}.jsonl"
    if bc.skip_if_exists(out_path, force, f"step3/kelly/{exit_variant}"):
        return

    models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
    wu = bc.load_wu_targets()
    bias_cache = RollingBiasCache(
        halflife_days=int(config.get("rolling_bias_halflife_days", 20)),
        max_correction_f=float(config.get("max_rolling_correction_f", 1.5)),
    )
    bias_cache.seed_from_parquet()
    bankroll = bc.INITIAL_BANKROLL_USD
    records: list[dict] = []
    regional_bindings = 0
    t0 = time.time()

    dates = sorted(eligible["date"].unique())
    for di, date_str in enumerate(dates):
        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [kelly/{exit_variant}] day {di + 1}/{len(dates)} bankroll=${bankroll:.2f} ({elapsed:.1f}s)")

        day_rows = eligible[eligible["date"] == date_str]
        budget = bc.daily_budget_ngboost(bankroll, config)
        day_bets, raw_mu, _skip_reasons = collect_day_bets(
            day_rows,
            models,
            config,
            bias_cache,
            wu_bias,
            disable_rolling_bias=disable_rolling_bias,
            disable_basket=disable_basket,
        )
        bets = filter_top_bets(day_bets, config)

        if not bets or budget <= 0:
            for _, row in day_rows.iterrows():
                records.append({
                    "city": row["city"],
                    "date": date_str,
                    "strategy": "ngboost_kelly",
                    "exit_variant": exit_variant,
                    "traded": False,
                    "no_trade_reason": "no_positive_edge" if bets else "zero_budget",
                    "bankroll_after": bankroll,
                })
            record_day_residuals(day_rows, raw_mu, bias_cache, wu, wu_bias, date_str)
            continue

        alloc = allocate_ngboost_kelly(
            bets,
            bankroll_usd=bankroll,
            daily_budget_usd=budget,
            regions=bc.REGIONS,
        )
        if alloc.regional_cap_bound:
            regional_bindings += 1

        day_pnl = 0.0
        traded_any = False
        for i, bet in enumerate(alloc.bets):
            n_contracts = alloc.contracts[i] if i < len(alloc.contracts) else 0
            if n_contracts <= 0:
                continue
            traded_any = True
            result = settle_trade(
                bet.city, date_str, bet.bucket, bet.cost, n_contracts, exit_variant, wu
            )
            day_pnl += result["pnl_usd"]
            records.append({
                "city": bet.city,
                "date": date_str,
                "strategy": "ngboost_kelly",
                "exit_variant": exit_variant,
                "traded": True,
                "bucket": bet.bucket,
                "model_prob": bet.prob,
                "effective_prob": bet.prob,
                "edge": bet.prob - bet.cost,
                "entry_price": bet.cost,
                "exit_price": result["exit_price"],
                "exit_type": result["exit_type"],
                "n_contracts": n_contracts,
                "won": result["won"],
                "pnl_usd": result["pnl_usd"],
                "region": bet.region,
                "regional_cap_bound": alloc.regional_cap_bound,
                "bankroll_before": bankroll,
            })

        if not traded_any:
            for _, row in day_rows.iterrows():
                records.append({
                    "city": row["city"],
                    "date": date_str,
                    "strategy": "ngboost_kelly",
                    "exit_variant": exit_variant,
                    "traded": False,
                    "no_trade_reason": "kelly_zero_size",
                    "bankroll_after": bankroll,
                })

        bankroll += day_pnl
        for rec in records[-len(alloc.bets) :]:
            if rec.get("traded"):
                rec["bankroll_after"] = bankroll

        record_day_residuals(day_rows, raw_mu, bias_cache, wu, wu_bias, date_str)

    bc.write_jsonl(out_path, records)
    traded = sum(1 for r in records if r.get("traded"))
    print(
        f"Wrote {len(records)} rows ({traded} trades) to {out_path}; "
        f"regional cap bound on {regional_bindings} days"
    )


def run_flat_variant(
    eligible: pd.DataFrame,
    config: dict,
    force: bool,
    wu_bias: dict,
    *,
    disable_rolling_bias: bool = False,
    disable_basket: bool = False,
) -> None:
    exit_variant = "hold_to_settlement"
    out_path = bc.TRADES_DIR / f"ngboost_flat_{exit_variant}.jsonl"
    if bc.skip_if_exists(out_path, force, "step3/flat"):
        return

    edge_threshold = float(config.get("edge_threshold", 0.037))
    models = NgBoostBacktestModels.from_path_file(bc.MODEL_PATH_FILE)
    wu = bc.load_wu_targets()
    bias_cache = RollingBiasCache(
        halflife_days=int(config.get("rolling_bias_halflife_days", 20)),
        max_correction_f=float(config.get("max_rolling_correction_f", 1.5)),
    )
    bias_cache.seed_from_parquet()
    bankroll = bc.INITIAL_BANKROLL_USD
    records: list[dict] = []
    t0 = time.time()

    dates = sorted(eligible["date"].unique())
    for di, date_str in enumerate(dates):
        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [flat] day {di + 1}/{len(dates)} bankroll=${bankroll:.2f} ({elapsed:.1f}s)")

        day_rows = eligible[eligible["date"] == date_str]
        budget = bc.daily_budget_ngboost(bankroll, config)
        day_bets, raw_mu, _skip_reasons = collect_day_bets(
            day_rows,
            models,
            config,
            bias_cache,
            wu_bias,
            disable_rolling_bias=disable_rolling_bias,
            disable_basket=disable_basket,
        )
        bets = filter_top_bets(
            day_bets,
            config,
            edge_threshold=edge_threshold,
        )

        if not bets or budget <= 0:
            record_day_residuals(day_rows, raw_mu, bias_cache, wu, wu_bias, date_str)
            continue

        n_contracts = flat_contracts(bankroll, config)
        assert n_contracts == 5
        day_pnl = 0.0
        day_spent = 0.0
        for bet in bets:
            trade_cost = n_contracts * bet.cost
            if day_spent + trade_cost > budget:
                continue
            result = settle_trade(
                bet.city, date_str, bet.bucket, bet.cost, n_contracts, exit_variant, wu
            )
            day_spent += trade_cost
            day_pnl += result["pnl_usd"]
            records.append({
                "city": bet.city,
                "date": date_str,
                "strategy": "ngboost_flat",
                "exit_variant": exit_variant,
                "traded": True,
                "bucket": bet.bucket,
                "model_prob": bet.prob,
                "effective_prob": bet.prob,
                "edge": bet.prob - bet.cost,
                "entry_price": bet.cost,
                "exit_price": result["exit_price"],
                "exit_type": result["exit_type"],
                "n_contracts": n_contracts,
                "won": result["won"],
                "pnl_usd": result["pnl_usd"],
                "bankroll_before": bankroll,
            })

        bankroll += day_pnl
        for rec in records[-len(bets) :]:
            if rec.get("traded"):
                rec["bankroll_after"] = bankroll

        record_day_residuals(day_rows, raw_mu, bias_cache, wu, wu_bias, date_str)

    bc.write_jsonl(out_path, records)
    traded = sum(1 for r in records if r.get("traded"))
    print(f"Wrote {len(records)} rows ({traded} trades) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NGBoost Kelly backtest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="", help="Output suffix, e.g. v5")
    parser.add_argument("--start-date", default=None, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument(
        "--disable-rolling-bias",
        action="store_true",
        help="Skip rolling EWMA bias correction",
    )
    parser.add_argument(
        "--disable-basket",
        action="store_true",
        help="Skip boundary basket companion bets",
    )
    parser.add_argument(
        "--flat-only",
        action="store_true",
        help="Run only ngboost_flat_hold_to_settlement variant",
    )
    parser.add_argument(
        "--include-flat",
        action="store_true",
        help="Also run ngboost_flat_hold_to_settlement variant",
    )
    args = parser.parse_args()

    if args.output_tag:
        bc.configure_output_tag(args.output_tag)

    if not bc.ELIGIBLE_DATES_CSV.exists():
        print(f"ERROR: run step1 first — missing {bc.ELIGIBLE_DATES_CSV}")
        sys.exit(1)
    if not bc.MODEL_PATH_FILE.exists():
        print(f"ERROR: run step0 first — missing {bc.MODEL_PATH_FILE}")
        sys.exit(1)

    eligible = pd.read_csv(bc.ELIGIBLE_DATES_CSV)
    eligible = bc.filter_eligible_by_date(eligible, args.start_date, args.end_date)
    if eligible.empty:
        print("ERROR: no eligible city-dates after date filter")
        sys.exit(1)
    if args.start_date or args.end_date:
        print(
            f"Date filter: {args.start_date or '...'} to {args.end_date or '...'} "
            f"→ {len(eligible)} city-dates"
        )

    config = bc.load_trading_config()
    print(describe_two_piece_mode(config))
    wu_bias = load_wunderground_bias()
    bias_kwargs = {
        "disable_rolling_bias": args.disable_rolling_bias,
        "disable_basket": args.disable_basket,
    }
    if not args.flat_only:
        for variant in EXIT_VARIANTS:
            run_kelly_variant(variant, eligible, config, args.force, wu_bias, **bias_kwargs)
    if args.include_flat or args.output_tag in ("v5", "v5b") or args.flat_only:
        run_flat_variant(eligible, config, args.force, wu_bias, **bias_kwargs)


if __name__ == "__main__":
    main()
