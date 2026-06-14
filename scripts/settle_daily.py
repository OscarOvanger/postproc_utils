"""Settle paper trades using NWS CLI Tmax."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.data_pipeline import _load_cli_target  # noqa: E402
from src.trackj.fetch_cli_target import fetch_cli_target  # noqa: E402
from src.fees import net_pnl, taker_fee  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"
PAPER_LOG = LOG_DIR / "paper_trades.jsonl"
SETTLEMENT_LOG = LOG_DIR / "settlements.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Settle paper trades for one event date")
    parser.add_argument("--date", type=str, required=True, help="Event date YYYY-MM-DD")
    parser.add_argument(
        "--decision-file",
        type=str,
        default=str(PAPER_LOG),
        help="Path to paper_trades.jsonl",
    )
    return parser.parse_args()


def _load_decision(event_date: str, decision_file: Path) -> dict | None:
    if not decision_file.exists():
        print(f"No decision log found at {decision_file}")
        return None
    match = None
    with open(decision_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("date") == event_date:
                match = entry
    return match


def _cli_tmax(city: str, event_date: str) -> float | None:
    from src.data_pipeline import _load_city_config

    cfg = _load_city_config(city)
    target = date.fromisoformat(event_date)
    cli = _load_cli_target(city, event_date)
    if cli.empty:
        cli = fetch_cli_target(
            cfg,
            target,
            target,
            PROJECT_ROOT / "data" / "trackj" / "raw",
            PROJECT_ROOT / "data" / "trackj",
            no_fetch=False,
        )
    if cli.empty:
        return None
    cli = cli.copy()
    cli["date_key"] = pd.to_datetime(cli["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = cli[cli["date_key"] == event_date]
    if row.empty:
        return None
    val = row.iloc[0].get("tmax_f")
    return float(val) if pd.notna(val) else None


def _parse_bucket_label(label: str) -> tuple[str, float | None, float | None]:
    text = str(label).strip()
    if text.endswith("+") or text.endswith("+"):
        lower = float(text.rstrip("+").strip())
        return "GREATER_THAN", lower, None
    if text.startswith("<") or text.startswith("≤"):
        upper = float(text.lstrip("<≤ ").strip())
        return "LESS_THAN", None, upper
    if "-" in text:
        lo, hi = text.split("-", 1)
        return "RANGE", float(lo), float(hi)
    return "UNKNOWN", None, None


def _tmax_in_bucket(tmax: float, bucket_label: str, market_row: dict | None = None) -> bool:
    if market_row and market_row.get("bucket_type"):
        bucket_type = str(market_row["bucket_type"])
        lower = pd.to_numeric(market_row.get("bucket_lower_inclusive_f"), errors="coerce")
        upper = pd.to_numeric(market_row.get("bucket_upper_inclusive_f"), errors="coerce")
    else:
        bucket_type, lower, upper = _parse_bucket_label(bucket_label)
        lower = pd.to_numeric(lower, errors="coerce")
        upper = pd.to_numeric(upper, errors="coerce")

    if bucket_type == "RANGE":
        return float(lower) <= tmax <= float(upper)
    if bucket_type == "LESS_THAN":
        return tmax <= float(upper)
    if bucket_type == "GREATER_THAN":
        return tmax >= float(lower)
    return False


def _load_market_buckets(city: str, event_date: str) -> pd.DataFrame:
    from build_splits import discover_city_csvs, load_city_frame

    raw_dir = PROJECT_ROOT / "historic_tmax_market_data"
    if not raw_dir.exists():
        return pd.DataFrame()
    city_csvs = discover_city_csvs(raw_dir)
    path = city_csvs.get(city)
    if path is None:
        return pd.DataFrame()
    df = load_city_frame(city, path)
    df["date_key"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    day = df[df["date_key"] == event_date]
    if day.empty:
        return pd.DataFrame()
    return day.drop_duplicates("bucket_label")


def _trade_pnl_cents(trade: dict, won: bool) -> int:
    price = float(trade["market_price"])
    contracts = int(trade["n_contracts"])
    gross = (1.0 - price) * contracts * 100 if won else -price * contracts * 100
    return int(round(net_pnl(gross, contracts, price, order_type="taker")))


def _cumulative_pnl_cents() -> int:
    if not SETTLEMENT_LOG.exists():
        return 0
    total = 0
    with open(SETTLEMENT_LOG, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            total += int(entry.get("daily_pnl_cents", 0))
    return total


def main() -> None:
    args = _parse_args()
    event_date = args.date
    decision = _load_decision(event_date, Path(args.decision_file))
    if decision is None:
        raise SystemExit(f"No paper decision found for {event_date}")

    trades = decision.get("trades", [])
    if not trades:
        print(f"No trades to settle for {event_date}")
        return

    print(f"\n=== SETTLEMENT — {event_date} ===\n")
    settled_trades: list[dict] = []
    daily_pnl_cents = 0
    n_wins = 0

    for trade in trades:
        city = trade["city"]
        bucket_label = trade["bucket_label"]
        cli_tmax = _cli_tmax(city, event_date)
        if cli_tmax is None:
            print(f"  {city}: CLI Tmax unavailable — skip settlement")
            continue

        buckets = _load_market_buckets(city, event_date)
        bucket_row = None
        if not buckets.empty:
            match = buckets[buckets["bucket_label"].astype(str).eq(str(bucket_label))]
            if not match.empty:
                bucket_row = match.iloc[0].to_dict()

        won = _tmax_in_bucket(cli_tmax, bucket_label, bucket_row)
        pnl_cents = _trade_pnl_cents(trade, won)
        daily_pnl_cents += pnl_cents
        if won:
            n_wins += 1

        result = {
            "city": city,
            "bucket_label": bucket_label,
            "cli_tmax_f": cli_tmax,
            "won": won,
            "pnl_cents": pnl_cents,
            "n_contracts": trade["n_contracts"],
            "entry_price": trade["market_price"],
        }
        settled_trades.append(result)
        status = "WIN" if won else "LOSS"
        print(
            f"  {city} | {bucket_label} | CLI={cli_tmax:.0f}F | {status} | "
            f"PnL={pnl_cents:+d} cents"
        )

    cumulative = _cumulative_pnl_cents() + daily_pnl_cents
    settlement = {
        "date": event_date,
        "mode": decision.get("mode", "paper"),
        "n_trades": len(settled_trades),
        "n_wins": n_wins,
        "daily_pnl_cents": daily_pnl_cents,
        "cumulative_pnl_cents": cumulative,
        "trades": settled_trades,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTLEMENT_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(settlement, default=str) + "\n")

    print(f"\nDaily PnL:       {daily_pnl_cents:+d} cents")
    print(f"Cumulative PnL:  {cumulative:+d} cents (${cumulative / 100:.2f})")
    print(f"Settlement log:  {SETTLEMENT_LOG}")


if __name__ == "__main__":
    main()
