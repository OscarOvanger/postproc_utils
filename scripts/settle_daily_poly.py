"""Settle Polymarket trades, update bankroll, and append rolling bias residuals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rolling_bias import (  # noqa: E402
    RESIDUALS_PATH,
    load_residuals_df,
    save_residuals_and_snapshot,
)

LOGS_DIR = PROJECT_ROOT / "logs"
BANKROLL_FILE = LOGS_DIR / "current_bankroll.txt"
SETTLEMENT_LOG = LOGS_DIR / "poly_settlements.jsonl"

ICAO_MAP = {
    "atlanta": "KATL",
    "austin": "KAUS",
    "chicago": "KORD",
    "dallas": "KDAL",
    "houston": "KHOU",
    "los_angeles": "KLAX",
    "miami": "KMIA",
    "new_york": "KLGA",
    "san_francisco": "KSFO",
    "seattle": "KSEA",
}


def parse_date(s: str) -> str:
    if s == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    if s == "today":
        return date.today().isoformat()
    return s


def load_state(date_str: str) -> dict | None:
    path = LOGS_DIR / f"auto_trader_state_{date_str}.json"
    if not path.exists():
        print(f"No state file: {path}")
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fetch_wu_actual(icao: str, date_str: str) -> int | None:
    """Fetch max hourly METAR temperature from IEM ASOS.

    This is the Wunderground-equivalent resolution source:
    max of hourly spot readings, rounded to nearest integer F.
    """
    y, m, d = date_str.split("-")
    next_d = date.fromisoformat(date_str) + timedelta(days=1)
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={icao}&data=tmpf&tz=UTC&format=onlycomma"
        f"&year1={y}&month1={int(m)}&day1={int(d)}"
        f"&year2={next_d.year}&month2={next_d.month}&day2={next_d.day}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        temps = []
        for row in reader:
            val = row.get("tmpf", "M").strip()
            if val not in ("M", ""):
                try:
                    temps.append(float(val))
                except ValueError:
                    continue
        if not temps:
            return None
        return round(max(temps))
    except Exception as exc:
        print(f"  IEM fetch failed for {icao} on {date_str}: {exc}")
        return None


def parse_bucket(label: str) -> dict:
    """Parse bucket label like '72-73°F' into {type, lower, upper}."""
    text = label.replace("°F", "").replace("\u00b0F", "").strip()
    if "or higher" in text.lower() or text.endswith("+"):
        val = float("".join(c for c in text if c.isdigit() or c == "."))
        return {"type": "GREATER_THAN", "lower": val, "upper": None}
    if "or lower" in text.lower() or text.startswith("<"):
        val = float("".join(c for c in text if c.isdigit() or c == "."))
        return {"type": "LESS_THAN", "lower": None, "upper": val}
    if "-" in text:
        parts = text.split("-")
        return {"type": "RANGE", "lower": float(parts[0]), "upper": float(parts[1])}
    return {"type": "UNKNOWN", "lower": None, "upper": None}


def bucket_settles_yes(actual_f: int, bucket: dict) -> bool:
    btype = bucket["type"]
    if btype == "RANGE":
        return bucket["lower"] <= actual_f <= bucket["upper"]
    if btype == "LESS_THAN":
        return actual_f <= bucket["upper"]
    if btype == "GREATER_THAN":
        return actual_f >= bucket["lower"]
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Settle Polymarket trades and append rolling bias residuals"
    )
    parser.add_argument("--date", required=True, help="Event date (YYYY-MM-DD or 'yesterday')")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    args = parser.parse_args()

    date_str = parse_date(args.date)
    print(f"\n=== Settling {date_str} ===")

    state = load_state(date_str)
    if state is None:
        sys.exit(1)

    positions = [
        p
        for p in state.get("positions", [])
        if p.get("status") in ("filled", "settlement_pending", "exited")
    ]
    if not positions:
        print("No filled positions to settle.")

    bankroll = (
        float(BANKROLL_FILE.read_text().strip())
        if BANKROLL_FILE.exists()
        else state.get("bankroll", 100.0)
    )
    print(f"Bankroll before settlement: ${bankroll:.2f}")

    settlement_results = []
    total_pnl = 0.0

    for pos in positions:
        city = pos["city"]
        bucket_label = pos["bucket_label"]
        fill_price = pos.get("fill_price") or pos.get("maker_entry_price")
        n_contracts = pos["n_contracts"]

        if fill_price is None:
            print(f"  {city} {bucket_label}: no fill price, skipping")
            continue

        if pos.get("exit_reason") and pos.get("pnl") is not None:
            pnl = pos["pnl"]
            print(
                f"  {city} {bucket_label}: already exited ({pos['exit_reason']}), "
                f"PnL=${pnl:.2f}"
            )
            settlement_results.append(
                {
                    "city": city,
                    "bucket_label": bucket_label,
                    "fill_price": fill_price,
                    "n_contracts": n_contracts,
                    "actual": None,
                    "won": None,
                    "pnl": pnl,
                    "exit_reason": pos["exit_reason"],
                }
            )
            total_pnl += pnl
            continue

        icao = ICAO_MAP.get(city)
        if not icao:
            print(f"  {city}: unknown ICAO station, skipping")
            continue

        actual = fetch_wu_actual(icao, date_str)
        if actual is None:
            print(f"  {city} ({icao}): no WU data available yet, skipping")
            continue

        bucket = parse_bucket(bucket_label)
        won = bucket_settles_yes(actual, bucket)
        if won:
            pnl = n_contracts * (1.0 - fill_price)
        else:
            pnl = -n_contracts * fill_price

        result_str = "WIN" if won else "LOSS"
        print(
            f"  {city} {bucket_label} @ ${fill_price:.2f} | actual={actual}F | "
            f"{result_str} | PnL=${pnl:+.2f}"
        )

        settlement_results.append(
            {
                "city": city,
                "bucket_label": bucket_label,
                "fill_price": fill_price,
                "n_contracts": n_contracts,
                "actual": actual,
                "won": won,
                "pnl": pnl,
                "exit_reason": "settlement",
            }
        )
        total_pnl += pnl

    n_wins = sum(1 for r in settlement_results if r.get("won") is True)
    n_losses = sum(1 for r in settlement_results if r.get("won") is False)
    new_bankroll = bankroll + total_pnl
    print(f"\n  Settled: {len(settlement_results)} trades ({n_wins}W / {n_losses}L)")
    print(f"  Total PnL: ${total_pnl:+.2f}")
    print(f"  Bankroll: ${bankroll:.2f} -> ${new_bankroll:.2f}")

    wu_adj = state.get("wu_adjusted_forecasts", {})
    if not wu_adj:
        if state.get("signal") == "ngboost":
            wu_adj = state.get("raw_forecasts", {})
        else:
            fab = state.get("forecasts_after_bias", {})
            rba = state.get("rolling_bias_applied", {})
            for city in fab:
                wu_adj[city] = int(round(float(fab[city]) + float(rba.get(city, 0.0))))

    new_residuals = []
    all_forecast_cities = set(wu_adj.keys())
    for pos in positions:
        all_forecast_cities.add(pos["city"])

    actuals_cache = {}
    for result in settlement_results:
        if result.get("actual") is not None:
            actuals_cache[result["city"]] = result["actual"]

    print("\n--- Rolling bias residuals ---")
    for city in sorted(all_forecast_cities):
        if city not in wu_adj:
            print(f"  {city}: no wu_adjusted_forecast, skipping residual")
            continue
        forecast = wu_adj[city]
        actual = actuals_cache.get(city)
        if actual is None:
            icao = ICAO_MAP.get(city)
            if icao:
                actual = fetch_wu_actual(icao, date_str)
        if actual is None:
            print(f"  {city}: no WU actual available, skipping residual")
            continue
        residual = float(forecast) - float(actual)
        print(f"  {city}: forecast={forecast}F actual={actual}F residual={residual:+.1f}F")
        new_residuals.append(
            {
                "city": city,
                "date": date_str,
                "forecast": float(forecast),
                "wu_actual": float(actual),
                "residual": residual,
            }
        )

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    SETTLEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTLEMENT_LOG, "a", encoding="utf-8") as handle:
        record = {
            "date": date_str,
            "settlements": settlement_results,
            "total_pnl": round(total_pnl, 2),
            "bankroll_before": round(bankroll, 2),
            "bankroll_after": round(new_bankroll, 2),
            "n_residuals_appended": len(new_residuals),
        }
        handle.write(json.dumps(record) + "\n")
    print(f"\n  Settlement appended to {SETTLEMENT_LOG}")

    BANKROLL_FILE.write_text(f"{new_bankroll:.2f}\n", encoding="utf-8")
    print(f"  Bankroll updated: ${new_bankroll:.2f}")

    if new_residuals:
        existing = load_residuals_df()
        new_df = pd.DataFrame(new_residuals)
        if not existing.empty:
            mask = ~(
                (existing["city"].isin(new_df["city"]))
                & (existing["date"].isin([date_str]))
            )
            existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
        save_residuals_and_snapshot(combined)
        print(f"  Appended {len(new_residuals)} residuals to {RESIDUALS_PATH}")
        print("  Rolling bias snapshot updated.")
    else:
        print("  No residuals to append.")

    state_path = LOGS_DIR / f"auto_trader_state_{date_str}.json"
    for pos in state.get("positions", []):
        for result in settlement_results:
            if pos["city"] == result["city"] and pos["bucket_label"] == result["bucket_label"]:
                pos["pnl"] = result["pnl"]
                pos["exit_reason"] = result["exit_reason"]
                pos["status"] = "settled"
    state["phase"] = "settled"
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    print(f"  State file updated: {state_path}")


if __name__ == "__main__":
    main()
