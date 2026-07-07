#!/usr/bin/env python3
"""Retroactive paper-trade validation against Wunderground-equivalent actuals."""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fetch_wunderground_target import _daily_targets_from_asos  # noqa: E402
from src.polymarket_api import parse_bucket_label  # noqa: E402
from src.trackj.build_asos_features import load_cached_asos  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs"
STATE_GLOB = "auto_trader_state_*.json"
PAPER_LOG_PATH = LOGS_DIR / "poly_paper_trades.jsonl"
BANKROLL_PATH = LOGS_DIR / "current_bankroll.txt"
WU_CACHE_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_actuals.json"
WU_PARQUET_PATH = PROJECT_ROOT / "data" / "polymarket" / "wunderground_targets.parquet"
SETTLE_OUT_PATH = LOGS_DIR / "retro_settlements.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"
RAW_ROOT = PROJECT_ROOT / "data" / "trackj" / "raw"

VALID_STATUSES = frozenset({"filled", "exited", "settlement_pending"})
INTRADAY_EXIT_REASONS = frozenset({"profit_target_15c", "maker_exit_18c"})
STATUS_PRIORITY = {"exited": 3, "settlement_pending": 2, "filled": 1}
DEFAULT_N_CONTRACTS = 5
DEFAULT_BANKROLL = 86.63
TMPF_MIN = -30.0
TMPF_MAX = 140.0

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _parse_args() -> argparse.Namespace:
    today = date.today()
    default_start = today - timedelta(days=6)
    parser = argparse.ArgumentParser(
        description="Validate auto-trader paper trades against Wunderground actuals.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start.isoformat(),
        help="Inclusive start date YYYY-MM-DD (default: last 7 days)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=today.isoformat(),
        help="Inclusive end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="Append computed settlements to logs/retro_settlements.jsonl",
    )
    parser.add_argument(
        "--source",
        choices=("state", "paper-log", "both"),
        default="both",
        help="Trade source: state files, poly_paper_trades.jsonl, or both (default: both)",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _load_city_config() -> dict[str, dict[str, Any]]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _parse_state_date(path: Path) -> date | None:
    stem = path.stem
    prefix = "auto_trader_state_"
    if not stem.startswith(prefix):
        return None
    try:
        return date.fromisoformat(stem[len(prefix) :])
    except ValueError:
        return None


def _read_bankroll_file() -> float:
    if not BANKROLL_PATH.exists():
        return DEFAULT_BANKROLL
    text = BANKROLL_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return DEFAULT_BANKROLL
    try:
        return float(text)
    except ValueError:
        return DEFAULT_BANKROLL


def scan_state_files(start: date, end: date) -> list[tuple[date, Path, dict[str, Any]]]:
    rows: list[tuple[date, Path, dict[str, Any]]] = []
    for path in LOGS_DIR.glob(STATE_GLOB):
        file_date = _parse_state_date(path)
        if file_date is None or file_date < start or file_date > end:
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append((file_date, path, state))
    rows.sort(key=lambda x: x[0])
    return rows


def uses_recorded_pnl(pos: dict[str, Any]) -> bool:
    if pos.get("status") != "exited":
        return False
    if pos.get("pnl") is None:
        return False
    reason = str(pos.get("exit_reason") or "")
    if reason in INTRADAY_EXIT_REASONS:
        return True
    lower = reason.lower()
    return "settlement" in lower or "profit_target" in lower


def bucket_settles_yes(actual_tmax_f: int, bucket: dict[str, Any]) -> bool:
    btype = bucket["type"]
    if btype == "RANGE":
        return bucket["lower"] <= actual_tmax_f <= bucket["upper"]
    if btype == "LESS_THAN":
        return actual_tmax_f <= bucket["upper"]
    if btype == "GREATER_THAN":
        return actual_tmax_f >= bucket["lower"]
    return False


def settlement_pnl_usd(n_contracts: float, entry_price: float, won: bool) -> float:
    per = (1.0 - entry_price) if won else (-entry_price)
    return round(per * n_contracts, 4)


def _position_row(event_date: str, pos: dict[str, Any]) -> dict[str, Any] | None:
    status = str(pos.get("status") or "")
    if status not in VALID_STATUSES:
        return None

    entry_price = _to_float(pos.get("fill_price")) or _to_float(pos.get("maker_entry_price"))
    if entry_price is None:
        return None

    city = str(pos.get("city") or "")
    bucket_label = str(pos.get("bucket_label") or "")
    if not city or not bucket_label:
        return None

    return {
        "event_date": event_date,
        "city": city,
        "bucket_label": bucket_label,
        "maker_entry_price": _to_float(pos.get("maker_entry_price")),
        "fill_price": _to_float(pos.get("fill_price")),
        "entry_price": entry_price,
        "n_contracts": int(_to_float(pos.get("n_contracts")) or DEFAULT_N_CONTRACTS),
        "edge": _to_float(pos.get("edge")),
        "model_prob": _to_float(pos.get("model_prob")),
        "status": status,
        "exit_price": _to_float(pos.get("exit_price")),
        "exit_reason": pos.get("exit_reason"),
        "pnl": _to_float(pos.get("pnl")),
    }


def _dedupe_positions(
    rows: list[dict[str, Any]],
    start_bankroll: float | None,
) -> tuple[list[dict[str, Any]], float | None]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["event_date"], row["city"], row["bucket_label"])
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            continue
        if STATUS_PRIORITY.get(row["status"], 0) >= STATUS_PRIORITY.get(existing["status"], 0):
            by_key[key] = row
    positions = sorted(by_key.values(), key=lambda r: (r["event_date"], r["city"], r["bucket_label"]))
    return positions, start_bankroll


def extract_positions_from_state(
    state_files: list[tuple[date, Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], float | None]:
    """Return deduped positions and bankroll from state files."""
    rows: list[dict[str, Any]] = []
    start_bankroll: float | None = None

    for file_date, _path, state in state_files:
        if start_bankroll is None:
            br = _to_float(state.get("bankroll"))
            if br is not None:
                start_bankroll = br

        event_date = str(state.get("date") or file_date.isoformat())
        for pos in state.get("positions", []):
            row = _position_row(event_date, pos)
            if row is not None:
                rows.append(row)

    return _dedupe_positions(rows, start_bankroll)


def extract_positions_from_paper_log(start: date, end: date) -> tuple[list[dict[str, Any]], float | None]:
    """Return deduped auto-trader positions from poly_paper_trades.jsonl."""
    if not PAPER_LOG_PATH.exists():
        return [], None

    rows: list[dict[str, Any]] = []
    start_bankroll: float | None = None
    with open(PAPER_LOG_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("source") != "auto_trader":
                continue
            event_date = str(entry.get("date") or "")
            if not event_date:
                continue
            try:
                event_day = date.fromisoformat(event_date)
            except ValueError:
                continue
            if event_day < start or event_day > end:
                continue
            if start_bankroll is None:
                br = _to_float(entry.get("bankroll"))
                if br is not None:
                    start_bankroll = br
            for pos in entry.get("positions", []):
                row = _position_row(event_date, pos)
                if row is not None:
                    rows.append(row)

    return _dedupe_positions(rows, start_bankroll)


def extract_positions(
    *,
    start: date,
    end: date,
    source: str,
) -> tuple[list[dict[str, Any]], float | None]:
    rows: list[dict[str, Any]] = []
    start_bankroll: float | None = None

    if source in ("state", "both"):
        state_files = scan_state_files(start, end)
        state_rows, state_bankroll = extract_positions_from_state(state_files)
        rows.extend(state_rows)
        start_bankroll = state_bankroll

    if source in ("paper-log", "both"):
        log_rows, log_bankroll = extract_positions_from_paper_log(start, end)
        rows.extend(log_rows)
        if start_bankroll is None:
            start_bankroll = log_bankroll

    return _dedupe_positions(rows, start_bankroll)


@dataclass
class WuActualCache:
    city_config: dict[str, dict[str, Any]]
    data: dict[str, dict[str, int]] = field(default_factory=dict)
    parquet: pd.DataFrame | None = None
    dirty: bool = False

    @classmethod
    def load(cls, city_config: dict[str, dict[str, Any]]) -> WuActualCache:
        cache = cls(city_config=city_config)
        if WU_CACHE_PATH.exists():
            try:
                raw = json.loads(WU_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cache.data = {
                        str(d): {str(c): int(v) for c, v in cities.items()}
                        for d, cities in raw.items()
                        if isinstance(cities, dict)
                    }
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        if WU_PARQUET_PATH.exists():
            df = pd.read_parquet(WU_PARQUET_PATH)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["city"] = df["city"].astype(str)
            df["wunderground_tmax"] = pd.to_numeric(df["wunderground_tmax"], errors="coerce")
            cache.parquet = df
        return cache

    def _icao(self, city: str) -> str | None:
        cfg = self.city_config.get(city)
        if not cfg:
            return None
        return str(cfg.get("nws_station") or "")

    def _from_parquet(self, city: str, event_date: str) -> tuple[int | None, str | None]:
        if self.parquet is None or self.parquet.empty:
            return None, None
        row = self.parquet[(self.parquet["city"] == city) & (self.parquet["date"] == event_date)]
        if row.empty:
            return None, None
        val = row.iloc[0]["wunderground_tmax"]
        if pd.isna(val):
            return None, None
        return int(round(float(val))), "parquet"

    def _from_cached_asos(self, city: str, event_date: str) -> tuple[int | None, str | None]:
        icao = self._icao(city)
        if not icao:
            return None, None
        d = date.fromisoformat(event_date)
        raw_dir = RAW_ROOT / city / "asos"
        asos = load_cached_asos(raw_dir, icao, d, d)
        if asos.empty:
            return None, None
        targets = _daily_targets_from_asos(asos, city, icao, d, d)
        if targets.empty:
            return None, None
        val = targets.iloc[0]["wunderground_tmax"]
        if pd.isna(val):
            return None, None
        return int(round(float(val))), "cached_asos"

    def _from_iem(self, city: str, event_date: str) -> tuple[int | None, str | None]:
        icao = self._icao(city)
        if not icao:
            return None, None
        d = date.fromisoformat(event_date)
        params = {
            "station": icao,
            "data": "tmpf",
            "tz": "UTC",
            "format": "onlycomma",
            "year1": d.year,
            "month1": d.month,
            "day1": d.day,
            "year2": d.year,
            "month2": d.month,
            "day2": d.day,
        }
        try:
            resp = requests.get(IEM_ASOS_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            return None, None

        frame = pd.read_csv(io.StringIO(resp.text))
        if frame.empty or "tmpf" not in frame.columns:
            return None, None
        tmpf = pd.to_numeric(frame["tmpf"], errors="coerce").dropna()
        if tmpf.empty:
            return None, None
        daily_max = float(tmpf.max())
        if not (TMPF_MIN <= daily_max <= TMPF_MAX):
            return None, None
        return int(round(daily_max)), "iem_asos"

    def get(self, city: str, event_date: str) -> tuple[int | None, str | None]:
        if event_date in self.data and city in self.data[event_date]:
            return self.data[event_date][city], "json_cache"

        actual, source = self._from_parquet(city, event_date)
        if actual is None:
            actual, source = self._from_cached_asos(city, event_date)
        if actual is None:
            actual, source = self._from_iem(city, event_date)

        if actual is not None:
            self.data.setdefault(event_date, {})[city] = actual
            self.dirty = True
        return actual, source

    def save(self) -> None:
        if not self.dirty:
            return
        WU_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WU_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)


@dataclass
class ResolvedTrade:
    event_date: str
    city: str
    bucket_label: str
    entry_price: float
    edge: float | None
    actual_tmax_f: int | None
    won: bool | None
    pnl: float | None
    pending: bool
    pnl_source: str
    settle_record: dict[str, Any] | None = None


def resolve_trades(
    positions: list[dict[str, Any]],
    wu_cache: WuActualCache,
) -> list[ResolvedTrade]:
    resolved: list[ResolvedTrade] = []
    for pos in positions:
        edge = pos.get("edge")
        if uses_recorded_pnl(pos):
            won: bool | None = None
            if pos.get("pnl") is not None and pos["pnl"] > 0:
                won = True
            elif pos.get("pnl") is not None and pos["pnl"] < 0:
                won = False
            resolved.append(
                ResolvedTrade(
                    event_date=pos["event_date"],
                    city=pos["city"],
                    bucket_label=pos["bucket_label"],
                    entry_price=pos["entry_price"],
                    edge=edge,
                    actual_tmax_f=None,
                    won=won,
                    pnl=pos["pnl"],
                    pending=False,
                    pnl_source="recorded",
                )
            )
            continue

        actual, source = wu_cache.get(pos["city"], pos["event_date"])
        if actual is None:
            resolved.append(
                ResolvedTrade(
                    event_date=pos["event_date"],
                    city=pos["city"],
                    bucket_label=pos["bucket_label"],
                    entry_price=pos["entry_price"],
                    edge=edge,
                    actual_tmax_f=None,
                    won=None,
                    pnl=None,
                    pending=True,
                    pnl_source="pending",
                )
            )
            continue

        try:
            bucket = parse_bucket_label(pos["bucket_label"])
        except ValueError:
            resolved.append(
                ResolvedTrade(
                    event_date=pos["event_date"],
                    city=pos["city"],
                    bucket_label=pos["bucket_label"],
                    entry_price=pos["entry_price"],
                    edge=edge,
                    actual_tmax_f=actual,
                    won=None,
                    pnl=None,
                    pending=True,
                    pnl_source="parse_error",
                )
            )
            continue

        won = bucket_settles_yes(actual, bucket)
        pnl = settlement_pnl_usd(pos["n_contracts"], pos["entry_price"], won)
        settle_record = {
            "event_date": pos["event_date"],
            "city": pos["city"],
            "bucket_label": pos["bucket_label"],
            "actual_tmax_f": actual,
            "won": won,
            "entry_price": pos["entry_price"],
            "n_contracts": pos["n_contracts"],
            "pnl": pnl,
            "source": source or "computed",
        }
        resolved.append(
            ResolvedTrade(
                event_date=pos["event_date"],
                city=pos["city"],
                bucket_label=pos["bucket_label"],
                entry_price=pos["entry_price"],
                edge=edge,
                actual_tmax_f=actual,
                won=won,
                pnl=pnl,
                pending=False,
                pnl_source=source or "computed",
                settle_record=settle_record,
            )
        )
    return resolved


def _fmt_money(value: float, *, signed: bool = True) -> str:
    if signed and value >= 0:
        return f"+${value:.2f}"
    if signed:
        return f"-${abs(value):.2f}"
    return f"${value:.2f}"


def _fmt_edge(edge: float | None) -> str:
    if edge is None:
        return "   —  "
    sign = "+" if edge >= 0 else ""
    return f"{sign}{edge:.3f}"


def _fmt_win(won: bool | None, pending: bool) -> str:
    if pending:
        return "PND"
    if won is True:
        return "YES"
    if won is False:
        return "NO"
    return "—"


def _fmt_pnl(pnl: float | None, pending: bool) -> str:
    if pending or pnl is None:
        return "pending"
    return _fmt_money(pnl)


def _normalize_bucket_display(label: str) -> str:
    return label.replace("°F", "").replace("°", "").strip()


def print_report(
    *,
    start: date,
    end: date,
    start_bankroll: float,
    trades: list[ResolvedTrade],
    source_label: str = "",
) -> None:
    print("=== Retroactive Paper Trade Analysis ===")
    print(f"Period: {start.isoformat()} to {end.isoformat()}")
    if source_label:
        print(f"Source: {source_label}")
    print(f"Bankroll at start: ${start_bankroll:.2f}")
    print()

    if not trades:
        print("No trades found.")
        print()
        print("Summary:")
        print("  Total trades: 0")
        print("  Settled: 0 (0W / 0L)")
        print("  Pending: 0")
        print("  Win rate: —")
        print("  Total PnL: $0.00")
        print("  Mean PnL/trade: —")
        print(f"  Projected bankroll: ${start_bankroll:.2f}")
        return

    header = (
        f"{'Date':<10} | {'City':<13} | {'Bucket':<9} | {'Entry':>5} | "
        f"{'Edge':>6} | {'Actual':>6} | {'Win':>3} | {'PnL':>8}"
    )
    print(header)
    for t in trades:
        actual = f"{t.actual_tmax_f}F" if t.actual_tmax_f is not None else "—"
        print(
            f"{t.event_date:<10} | {t.city:<13} | "
            f"{_normalize_bucket_display(t.bucket_label):<9} | "
            f"${t.entry_price:.2f} | {_fmt_edge(t.edge)} | "
            f"{actual:>6} | {_fmt_win(t.won, t.pending):>3} | "
            f"{_fmt_pnl(t.pnl, t.pending):>8}"
        )

    settled = [t for t in trades if not t.pending and t.pnl is not None]
    pending = [t for t in trades if t.pending]
    wins = [t for t in settled if t.won is True]
    losses = [t for t in settled if t.won is False]
    total_pnl = sum(t.pnl for t in settled)
    win_rate = (100.0 * len(wins) / len(settled)) if settled else None
    mean_pnl = (total_pnl / len(settled)) if settled else None

    print()
    print("Summary:")
    print(f"  Total trades: {len(trades)}")
    print(f"  Settled: {len(settled)} ({len(wins)}W / {len(losses)}L)")
    print(f"  Pending: {len(pending)}")
    if win_rate is not None:
        print(f"  Win rate: {win_rate:.1f}%")
    else:
        print("  Win rate: —")
    print(f"  Total PnL: {_fmt_money(total_pnl)}")
    if mean_pnl is not None:
        print(f"  Mean PnL/trade: {_fmt_money(mean_pnl)}")
    else:
        print("  Mean PnL/trade: —")
    print(f"  Projected bankroll: ${start_bankroll + total_pnl:.2f}")


def write_settlements(trades: list[ResolvedTrade]) -> None:
    records = [t.settle_record for t in trades if t.settle_record is not None]
    if not records:
        return
    SETTLE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTLE_OUT_PATH, "a", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")


def main() -> None:
    args = _parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        start, end = end, start

    city_config = _load_city_config()
    wu_cache = WuActualCache.load(city_config)

    positions, file_bankroll = extract_positions(start=start, end=end, source=args.source)
    start_bankroll = file_bankroll if file_bankroll is not None else _read_bankroll_file()

    trades = resolve_trades(positions, wu_cache)
    wu_cache.save()

    print_report(
        start=start,
        end=end,
        start_bankroll=start_bankroll,
        trades=trades,
        source_label=args.source,
    )

    if args.settle:
        write_settlements(trades)


if __name__ == "__main__":
    main()
