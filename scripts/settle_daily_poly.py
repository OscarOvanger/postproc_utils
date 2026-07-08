"""Settle Polymarket trades, update bankroll, and append rolling bias residuals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from poly_order_status import load_posted_orders  # noqa: E402
from src.polymarket_api import ORDER_LOG_PATH, PolymarketClient  # noqa: E402
from src.rolling_bias import (  # noqa: E402
    RESIDUALS_PATH,
    load_residuals_df,
    save_residuals_and_snapshot,
)

LOGS_DIR = PROJECT_ROOT / "logs"
BANKROLL_FILE = LOGS_DIR / "current_bankroll.txt"
SETTLEMENT_LOG = LOGS_DIR / "poly_settlements.jsonl"
GAMMA_STRUCTURE_DIR = PROJECT_ROOT / "data" / "polymarket_history"
WU_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"

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


def load_settled_dates() -> set[str]:
    dates: set[str] = set()
    if not SETTLEMENT_LOG.exists():
        return dates
    with open(SETTLEMENT_LOG, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                dates.add(str(json.loads(line)["date"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return dates


def settlement_exists(date_str: str) -> bool:
    if not SETTLEMENT_LOG.exists():
        return False
    with open(SETTLEMENT_LOG, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("date") == date_str and record.get("settlements"):
                return True
    return False


def fetch_wu_actual(icao: str, date_str: str) -> int | None:
    """Fetch max hourly METAR temperature from IEM ASOS."""
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


def _order_has_immediate_fill(record: dict[str, Any]) -> float:
    response = record.get("response") or {}
    if response.get("status") == "matched" or response.get("takingAmount"):
        try:
            return float(response.get("takingAmount") or record.get("size") or 0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _portfolio_helpers():
    from poly_portfolio_status import (
        TokenMeta,
        build_token_index,
        enrich_token_index_from_scan,
        fetch_ngboost_forecast,
        load_wu_targets,
        temp_in_bucket,
        winning_bucket_for_city,
        _NgBoostModels,
    )

    return {
        "TokenMeta": TokenMeta,
        "build_token_index": build_token_index,
        "enrich_token_index_from_scan": enrich_token_index_from_scan,
        "fetch_ngboost_forecast": fetch_ngboost_forecast,
        "load_wu_targets": load_wu_targets,
        "temp_in_bucket": temp_in_bucket,
        "winning_bucket_for_city": winning_bucket_for_city,
        "_NgBoostModels": _NgBoostModels,
    }


def load_token_index_from_gamma_cache(event_date: str) -> dict[str, Any]:
    cache_path = GAMMA_STRUCTURE_DIR / f"gamma_structure_{event_date}.json"
    if not cache_path.exists():
        return {}
    helpers = _portfolio_helpers()
    TokenMeta = helpers["TokenMeta"]
    index: dict[str, Any] = {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    for city_slug, city_entry in data.get("cities", {}).items():
        display = str(city_entry.get("title", city_slug)).split(" in ")
        city_display = (
            display[1].split(" on ")[0].title()
            if len(display) > 1
            else city_slug.replace("_", " ").title()
        )
        for market in city_entry.get("markets", []):
            token = str(market.get("yes_token_id", ""))
            if not token:
                continue
            index[token] = TokenMeta(
                token_id=token,
                city=str(market.get("city", city_slug)),
                city_display=city_display,
                bucket_label=str(market.get("bucket_label", "")),
                event_date=str(market.get("event_date", event_date)),
            )
    return index


def build_token_index_for_date(event_date: str) -> dict[str, Any]:
    helpers = _portfolio_helpers()
    build_token_index = helpers["build_token_index"]
    enrich_token_index_from_scan = helpers["enrich_token_index_from_scan"]
    index = load_token_index_from_gamma_cache(event_date)
    if index:
        return index
    index = build_token_index(event_dates={event_date}, refresh_labels=False)
    today = str(date.today())
    enrich_token_index_from_scan(
        index,
        event_date,
        include_closed=event_date < today,
    )
    return index


def reconstruct_positions_from_orders(
    event_date: str,
    client: PolymarketClient,
) -> list[dict[str, Any]]:
    """Rebuild filled positions for event_date from poly_orders.jsonl."""
    posted = load_posted_orders(ORDER_LOG_PATH)
    token_index = build_token_index_for_date(event_date)
    positions: list[dict[str, Any]] = []

    for order_id, record in posted.items():
        token = str(record.get("token_id", ""))
        meta = token_index.get(token)
        if meta is None or meta.event_date != event_date:
            continue

        matched = _order_has_immediate_fill(record)
        fill_price = _to_float(record.get("price"))
        if matched <= 0:
            status_info = client.get_order_status(order_id, token_id=token)
            matched = _to_float(status_info.get("size_matched")) or 0.0
            fill_price = _to_float(status_info.get("fill_price")) or fill_price
        if matched <= 0:
            continue

        positions.append(
            {
                "city": meta.city,
                "bucket_label": meta.bucket_label,
                "yes_token_id": token,
                "order_id": order_id,
                "fill_price": fill_price,
                "maker_entry_price": fill_price,
                "n_contracts": matched,
                "status": "filled",
            }
        )
    return positions


def settle_positions_from_wu_parquet(
    positions: list[dict[str, Any]],
    event_date: str,
    wu: pd.DataFrame,
    token_index: dict[str, Any],
) -> list[dict[str, Any]]:
    """Settle positions using wunderground_targets.parquet (portfolio logic)."""
    helpers = _portfolio_helpers()
    winning_bucket_for_city = helpers["winning_bucket_for_city"]
    temp_in_bucket = helpers["temp_in_bucket"]
    buckets_by_city: dict[str, list[str]] = {}
    for meta in token_index.values():
        if meta.event_date != event_date:
            continue
        buckets_by_city.setdefault(meta.city, [])
        if meta.bucket_label not in buckets_by_city[meta.city]:
            buckets_by_city[meta.city].append(meta.bucket_label)

    results: list[dict[str, Any]] = []
    for pos in positions:
        city = pos["city"]
        bucket_label = pos["bucket_label"]
        fill_price = pos.get("fill_price") or pos.get("maker_entry_price")
        n_contracts = pos["n_contracts"]
        if fill_price is None:
            print(f"  {city} {bucket_label}: no fill price, skipping")
            continue

        winner, actual = winning_bucket_for_city(
            wu,
            city,
            event_date,
            buckets_by_city.get(city, [bucket_label]),
        )
        if actual is not None:
            won = temp_in_bucket(actual, bucket_label)
            pnl = n_contracts * (1.0 - fill_price) if won else -n_contracts * fill_price
            result_str = "WIN" if won else "LOSS"
            print(
                f"  {city} {bucket_label} @ ${fill_price:.2f} | actual={int(round(actual))}F | "
                f"{result_str} | PnL=${pnl:+.2f}"
            )
            results.append(
                {
                    "city": city,
                    "bucket_label": bucket_label,
                    "fill_price": fill_price,
                    "n_contracts": n_contracts,
                    "actual": int(round(actual)),
                    "won": won,
                    "pnl": pnl,
                    "exit_reason": "settlement",
                    "winning_bucket": winner,
                }
            )
        else:
            print(f"  {city} {bucket_label}: no WU parquet data for {event_date}, skipping")
    return results


def settle_positions_from_iem(
    positions: list[dict[str, Any]],
    event_date: str,
) -> list[dict[str, Any]]:
    """Settle positions using IEM ASOS (normal state-file path)."""
    results: list[dict[str, Any]] = []
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
            results.append(
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
            continue

        icao = ICAO_MAP.get(city)
        if not icao:
            print(f"  {city}: unknown ICAO station, skipping")
            continue

        actual = fetch_wu_actual(icao, event_date)
        if actual is None:
            print(f"  {city} ({icao}): no WU data available yet, skipping")
            continue

        bucket = parse_bucket(bucket_label)
        won = bucket_settles_yes(actual, bucket)
        pnl = n_contracts * (1.0 - fill_price) if won else -n_contracts * fill_price
        result_str = "WIN" if won else "LOSS"
        print(
            f"  {city} {bucket_label} @ ${fill_price:.2f} | actual={actual}F | "
            f"{result_str} | PnL=${pnl:+.2f}"
        )
        results.append(
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
    return results


def wu_adjusted_from_state(state: dict[str, Any]) -> dict[str, float | int]:
    wu_adj = state.get("wu_adjusted_forecasts", {})
    if wu_adj:
        return wu_adj
    if state.get("signal") == "ngboost":
        return state.get("raw_forecasts", {})
    fab = state.get("forecasts_after_bias", {})
    rba = state.get("rolling_bias_applied", {})
    return {
        city: int(round(float(fab[city]) + float(rba.get(city, 0.0))))
        for city in fab
    }


def wu_adjusted_for_backfill(event_date: str, cities: list[str]) -> dict[str, float]:
    """Best-effort NGBoost forecast map for rolling-bias residuals on backfill dates."""
    if not cities:
        return {}
    helpers = _portfolio_helpers()
    models = helpers["_NgBoostModels"]()
    fetch_ngboost_forecast = helpers["fetch_ngboost_forecast"]
    out: dict[str, float] = {}
    missing: list[str] = []
    for city in cities:
        mu = fetch_ngboost_forecast(city, event_date, models=models)
        if mu is not None:
            out[city] = float(mu)
        else:
            missing.append(city)
    if missing:
        try:
            from src.poly_trading_pipeline import prepare_poly_trades

            _, metadata = prepare_poly_trades(
                event_date,
                bankroll=100.0,
                wait_for_open=False,
                raise_on_no_market=False,
            )
            raw = metadata.get("raw_forecasts", {})
            for city in missing:
                if city in raw:
                    out[city] = float(raw[city])
                else:
                    print(f"  {city}: no backfill forecast available for residual")
        except Exception as exc:
            for city in missing:
                print(f"  {city}: forecast fallback failed ({exc})")
    return out


def compute_residuals(
    *,
    event_date: str,
    positions: list[dict[str, Any]],
    settlement_results: list[dict[str, Any]],
    wu_adj: dict[str, float | int],
) -> list[dict[str, Any]]:
    existing = load_residuals_df()
    existing_pairs = set()
    if not existing.empty:
        existing_pairs = {
            (str(row.city), str(row.date))
            for row in existing.itertuples(index=False)
        }

    actuals_cache = {
        r["city"]: r["actual"]
        for r in settlement_results
        if r.get("actual") is not None
    }
    all_cities = set(wu_adj.keys()) | {p["city"] for p in positions}
    new_residuals: list[dict[str, Any]] = []

    print("\n--- Rolling bias residuals ---")
    for city in sorted(all_cities):
        if (city, event_date) in existing_pairs:
            print(f"  {city}: residual for {event_date} already exists, skipping")
            continue
        if city not in wu_adj:
            print(f"  {city}: no wu_adjusted_forecast, skipping residual")
            continue
        forecast = wu_adj[city]
        actual = actuals_cache.get(city)
        if actual is None:
            icao = ICAO_MAP.get(city)
            if icao:
                actual = fetch_wu_actual(icao, event_date)
        if actual is None:
            helpers = _portfolio_helpers()
            wu = helpers["load_wu_targets"]()
            _, tmax = helpers["winning_bucket_for_city"](wu, city, event_date, [])
            if tmax is not None:
                actual = int(round(tmax))
        if actual is None:
            print(f"  {city}: no WU actual available, skipping residual")
            continue
        residual = float(forecast) - float(actual)
        print(f"  {city}: forecast={forecast}F actual={actual}F residual={residual:+.1f}F")
        new_residuals.append(
            {
                "city": city,
                "date": event_date,
                "forecast": float(forecast),
                "wu_actual": float(actual),
                "residual": residual,
            }
        )
    return new_residuals


def append_residuals(new_residuals: list[dict[str, Any]], event_date: str) -> None:
    if not new_residuals:
        print("  No residuals to append.")
        return
    existing = load_residuals_df()
    new_df = pd.DataFrame(new_residuals)
    if not existing.empty:
        mask = ~(
            (existing["city"].isin(new_df["city"]))
            & (existing["date"].isin([event_date]))
        )
        existing = existing[mask]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
    save_residuals_and_snapshot(combined)
    print(f"  Appended {len(new_residuals)} residuals to {RESIDUALS_PATH}")
    print("  Rolling bias snapshot updated.")


def reconcile_bankroll() -> None:
    file_val = float(BANKROLL_FILE.read_text().strip()) if BANKROLL_FILE.exists() else 0.0
    try:
        api_val = PolymarketClient().get_balance()
    except Exception as exc:
        print(f"\nRECONCILIATION: file=${file_val:.2f} | API=ERROR ({exc})")
        return
    diff = abs(file_val - api_val)
    print(
        f"\nRECONCILIATION: file=${file_val:.2f} | API=${api_val:.2f} | "
        f"diff=${diff:.2f}"
    )
    if diff >= 0.50:
        print("*** RECONCILIATION MISMATCH — investigate before updating bankroll file ***")


def print_residual_coverage() -> None:
    df = load_residuals_df()
    print("\n--- Rolling-bias residual coverage (latest date per city) ---")
    if df.empty:
        print("  (no residuals on file)")
        return
    for city in sorted(df["city"].unique()):
        sub = df[df["city"] == city]
        latest = str(sub["date"].max())
        print(f"  {city}: latest={latest}")


def find_dates_needing_backfill() -> list[str]:
    posted = load_posted_orders(ORDER_LOG_PATH)
    settled = load_settled_dates()
    today = str(date.today())
    candidate_dates: set[str] = set()
    for record in posted.values():
        order_day = str(record.get("timestamp", ""))[:10]
        if order_day and order_day < today:
            candidate_dates.add(order_day)
    return sorted(d for d in candidate_dates if d not in settled)


def settle_from_state(date_str: str, state: dict[str, Any], dry_run: bool) -> None:
    positions = [
        p
        for p in state.get("positions", [])
        if p.get("status") in ("filled", "settlement_pending", "exited")
    ]
    if not positions:
        print("No filled positions to settle.")
        return

    bankroll = (
        float(BANKROLL_FILE.read_text().strip())
        if BANKROLL_FILE.exists()
        else state.get("bankroll", 100.0)
    )
    print(f"Bankroll before settlement: ${bankroll:.2f}")

    settlement_results = settle_positions_from_iem(positions, date_str)
    total_pnl = sum(r.get("pnl") or 0 for r in settlement_results)
    n_wins = sum(1 for r in settlement_results if r.get("won") is True)
    n_losses = sum(1 for r in settlement_results if r.get("won") is False)
    new_bankroll = bankroll + total_pnl
    print(f"\n  Settled: {len(settlement_results)} trades ({n_wins}W / {n_losses}L)")
    print(f"  Total PnL: ${total_pnl:+.2f}")
    print(f"  Bankroll: ${bankroll:.2f} -> ${new_bankroll:.2f}")

    wu_adj = wu_adjusted_from_state(state)
    new_residuals = compute_residuals(
        event_date=date_str,
        positions=positions,
        settlement_results=settlement_results,
        wu_adj=wu_adj,
    )

    if dry_run:
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

    append_residuals(new_residuals, date_str)

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


def append_missing_residuals_for_date(date_str: str, dry_run: bool) -> None:
    """Append rolling-bias residuals for a date that already has a settlement record."""
    if not SETTLEMENT_LOG.exists():
        return
    settlements: list[dict[str, Any]] = []
    with open(SETTLEMENT_LOG, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("date") == date_str:
                settlements = record.get("settlements", [])
    if not settlements:
        return
    cities = sorted({s["city"] for s in settlements})
    wu_adj = wu_adjusted_for_backfill(date_str, cities)
    positions = [{"city": c} for c in cities]
    new_residuals = compute_residuals(
        event_date=date_str,
        positions=positions,
        settlement_results=settlements,
        wu_adj=wu_adj,
    )
    if dry_run or not new_residuals:
        return
    append_residuals(new_residuals, date_str)


def settle_from_order_log(date_str: str, dry_run: bool) -> None:
    if settlement_exists(date_str):
        print(f"Settlement record already exists for {date_str}, checking residuals only.")
        append_missing_residuals_for_date(date_str, dry_run)
        return

    print(f"Backfilling {date_str} from order log (no state file).")
    client = PolymarketClient()
    positions = reconstruct_positions_from_orders(date_str, client)
    if not positions:
        print("No filled positions found in order log for this date.")
        return

    bankroll = (
        float(BANKROLL_FILE.read_text().strip()) if BANKROLL_FILE.exists() else 100.0
    )
    print(f"Bankroll (unchanged for backfill): ${bankroll:.2f}")

    wu = _portfolio_helpers()["load_wu_targets"]()
    token_index = build_token_index_for_date(date_str)
    settlement_results = settle_positions_from_wu_parquet(
        positions, date_str, wu, token_index
    )
    total_pnl = sum(r.get("pnl") or 0 for r in settlement_results)
    n_wins = sum(1 for r in settlement_results if r.get("won") is True)
    n_losses = sum(1 for r in settlement_results if r.get("won") is False)
    print(f"\n  Settled: {len(settlement_results)} trades ({n_wins}W / {n_losses}L)")
    print(f"  Total PnL (bookkeeping only, not applied): ${total_pnl:+.2f}")

    traded_cities = sorted({p["city"] for p in positions})
    wu_adj = wu_adjusted_for_backfill(date_str, traded_cities)
    new_residuals = compute_residuals(
        event_date=date_str,
        positions=positions,
        settlement_results=settlement_results,
        wu_adj=wu_adj,
    )

    if dry_run:
        print("\n[DRY RUN] No files written.")
        reconcile_bankroll()
        return

    SETTLEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTLEMENT_LOG, "a", encoding="utf-8") as handle:
        record = {
            "date": date_str,
            "source": "order_log_backfill",
            "settlements": settlement_results,
            "total_pnl": round(total_pnl, 2),
            "bankroll_before": round(bankroll, 2),
            "bankroll_after": round(bankroll, 2),
            "n_residuals_appended": len(new_residuals),
        }
        handle.write(json.dumps(record) + "\n")
    print(f"\n  Backfill settlement appended to {SETTLEMENT_LOG}")
    print("  Bankroll file NOT modified (outcomes already reflected in cash).")

    append_residuals(new_residuals, date_str)
    reconcile_bankroll()


def settle_date(date_str: str, dry_run: bool) -> None:
    print(f"\n=== Settling {date_str} ===")
    state = load_state(date_str)
    if state is not None:
        settle_from_state(date_str, state, dry_run)
    else:
        settle_from_order_log(date_str, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Settle Polymarket trades and append rolling bias residuals"
    )
    parser.add_argument("--date", help="Event date (YYYY-MM-DD, 'yesterday', or 'today')")
    parser.add_argument(
        "--backfill-all-missing",
        action="store_true",
        help="Backfill all past dates with order-log fills and no settlement record",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    args = parser.parse_args()

    if args.backfill_all_missing:
        dates = find_dates_needing_backfill()
        if not dates:
            print("No dates need backfill.")
        for date_str in dates:
            state_path = LOGS_DIR / f"auto_trader_state_{date_str}.json"
            if state_path.exists() and not settlement_exists(date_str):
                settle_date(date_str, args.dry_run)
            elif not state_path.exists():
                settle_from_order_log(date_str, args.dry_run)
        print_residual_coverage()
        return

    if not args.date:
        parser.error("--date is required unless --backfill-all-missing is set")

    date_str = parse_date(args.date)
    settle_date(date_str, args.dry_run)
    print_residual_coverage()


if __name__ == "__main__":
    main()
