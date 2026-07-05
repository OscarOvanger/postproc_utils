#!/usr/bin/env python3
"""Backfill Polymarket Tmax order-book history from Telonex into snapshot parquets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_polymarket_history import (  # noqa: E402
    DOWNSAMPLE_MINUTES,
    SNAPSHOTS_DIR,
    TARGET_CITIES,
    bucket_label_from_market,
)
from probe_polymarket_providers import (  # noqa: E402
    _city_from_question,
    _parse_slug_date,
    _slug_matches_city,
    load_telonex_verification_earliest,
    resolve_telonex_channel,
)
from src.provider_keys import load_telonex_key  # noqa: E402

SNAPSHOT_COLUMNS = [
    "timestamp",
    "bucket",
    "best_bid",
    "best_ask",
    "midpoint",
    "bid_depth",
    "ask_depth",
]
DEFAULT_START_DATE = "2026-02-03"
REQUEST_SLEEP_SECONDS = 0.15
PROGRESS_PATH = PROJECT_ROOT / "data" / "polymarket_history" / "telonex_backfill_progress.json"


def setup_logging(log_path: Path | None, *, tqdm_mode: bool) -> None:
    handlers: list[logging.Handler] = []
    if not tqdm_mode:
        handlers.append(logging.StreamHandler(sys.stdout))
    elif log_path is None:
        handlers.append(_TqdmLoggingHandler())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    if tqdm_mode:
        for name in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=sys.stderr)
            else:
                sys.stderr.write(msg + "\n")
        except Exception:
            self.handleError(record)


def progress_enabled(args: argparse.Namespace) -> bool:
    if args.no_progress:
        return False
    if args.progress:
        return True
    return sys.stderr.isatty() and tqdm is not None


def load_catalog(*, show_progress: bool) -> pd.DataFrame:
    from telonex import get_markets_dataframe

    if show_progress and tqdm is not None:
        tqdm.write("Loading Telonex markets catalog (~1–2 min)...", file=sys.stderr)
    else:
        logging.info("Loading Telonex markets catalog...")
    markets = get_markets_dataframe(exchange="polymarket")
    q = markets["question"].astype(str)
    mask = q.str.contains(r"highest temperature", case=False, na=False)
    tmax = markets.loc[mask].copy()
    tmax["city"] = tmax["question"].map(_city_from_question)
    tmax["event_date"] = tmax["slug"].map(_parse_slug_date)
    us = tmax[tmax["city"].isin(TARGET_CITIES)].copy()
    us = us[us.apply(lambda row: _slug_matches_city(str(row["slug"]), str(row["city"])), axis=1)]
    logging.info("Catalog rows for 10 US Tmax cities: %s", f"{len(us):,}")
    return us


def bucket_label_from_row(row: pd.Series) -> str:
    market = {
        "question": row.get("question"),
        "label": row.get("groupItemTitle") or row.get("question"),
        "groupItemTitle": row.get("groupItemTitle"),
    }
    return bucket_label_from_market(market)


def _timestamps(df: pd.DataFrame) -> pd.Series:
    if "timestamp_us" in df.columns:
        return pd.to_datetime(pd.to_numeric(df["timestamp_us"], errors="coerce"), unit="us", utc=True)
    if "t" in df.columns:
        return pd.to_datetime(df["t"], utc=True)
    raise ValueError("No timestamp column in Telonex frame")


def _sum_level_sizes(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.rsplit("_", 1)[-1]) if c.rsplit("_", 1)[-1].isdigit() else 0,
    )
    if not cols:
        return pd.Series(0.0, index=df.index)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(0.0).sum(axis=1)


def telonex_to_snapshot(df: pd.DataFrame, bucket: str, channel: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    out = pd.DataFrame()
    out["timestamp"] = _timestamps(df)
    if channel.startswith("book_snapshot") or "bid_price_0" in df.columns:
        out["best_bid"] = pd.to_numeric(df.get("bid_price_0"), errors="coerce")
        out["best_ask"] = pd.to_numeric(df.get("ask_price_0"), errors="coerce")
        out["bid_depth"] = _sum_level_sizes(df, "bid_size_")
        out["ask_depth"] = _sum_level_sizes(df, "ask_size_")
    else:
        out["best_bid"] = pd.to_numeric(df.get("bid_price"), errors="coerce")
        out["best_ask"] = pd.to_numeric(df.get("ask_price"), errors="coerce")
        out["bid_depth"] = pd.to_numeric(df.get("bid_size"), errors="coerce")
        out["ask_depth"] = pd.to_numeric(df.get("ask_size"), errors="coerce")
    out["midpoint"] = (out["best_bid"] + out["best_ask"]) / 2.0
    out["bucket"] = bucket
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp")
    out = out.set_index("timestamp")
    out = out.resample(f"{DOWNSAMPLE_MINUTES}min").last().dropna(how="all")
    out = out.reset_index()
    return out[SNAPSHOT_COLUMNS]


def download_bucket_day(
    api_key: str,
    *,
    slug: str,
    event_date: str,
    channel: str,
) -> pd.DataFrame:
    from telonex import get_dataframe

    channel_used, _ = resolve_telonex_channel(channel)
    to_date = (datetime.fromisoformat(event_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    return get_dataframe(
        api_key=api_key,
        exchange="polymarket",
        channel=channel_used,
        slug=slug,
        outcome="Yes",
        from_date=event_date,
        to_date=to_date,
    )


def city_floor_dates(start_date: str, cities: list[str]) -> dict[str, str]:
    earliest = load_telonex_verification_earliest()
    floors: dict[str, str] = {}
    for city in cities:
        city_earliest = earliest.get(city, start_date)
        floors[city] = max(start_date, city_earliest)
    return floors


def enumerate_city_dates(
    catalog: pd.DataFrame,
    cities: list[str],
    floors: dict[str, str],
    end_date: str | None,
) -> list[tuple[str, str, list[dict[str, str]]]]:
    tasks: list[tuple[str, str, list[dict[str, str]]]] = []
    end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for city in cities:
        floor = floors[city]
        subset = catalog[catalog["city"] == city].copy()
        subset = subset[
            (subset["event_date"].astype(str) >= floor) & (subset["event_date"].astype(str) <= end)
        ]
        if subset.empty:
            logging.warning("No catalog rows for %s on/after %s", city, floor)
            continue

        for event_date, group in subset.groupby("event_date"):
            buckets: list[dict[str, str]] = []
            for _, row in group.iterrows():
                slug = str(row["slug"])
                try:
                    bucket = bucket_label_from_row(row)
                except Exception:
                    logging.warning("Skipping unparseable bucket slug=%s", slug)
                    continue
                buckets.append({"slug": slug, "bucket": bucket})
            if buckets:
                tasks.append((city, str(event_date), buckets))
    tasks.sort(key=lambda item: (item[0], item[1]))
    return tasks


def order_tasks(
    tasks: list[tuple[str, str, list[dict[str, str]]]],
    cities: list[str],
    *,
    interleave: bool,
) -> list[tuple[str, str, list[dict[str, str]]]]:
    """Return task order: city-by-city (default) or round-robin one date per city."""
    if not interleave:
        return tasks

    by_city: dict[str, list[tuple[str, str, list[dict[str, str]]]]] = {city: [] for city in cities}
    for task in tasks:
        by_city.setdefault(task[0], []).append(task)

    ordered: list[tuple[str, str, list[dict[str, str]]]] = []
    max_dates = max((len(by_city[c]) for c in cities), default=0)
    for date_idx in range(max_dates):
        for city in cities:
            city_tasks = by_city.get(city, [])
            if date_idx < len(city_tasks):
                ordered.append(city_tasks[date_idx])
    return ordered


def load_progress() -> dict[str, Any]:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {"completed": [], "failed": [], "skipped": []}


def save_progress(progress: dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def backfill_city_date(
    api_key: str,
    city: str,
    event_date: str,
    buckets: list[dict[str, str]],
    *,
    channel: str,
    force: bool,
    show_progress: bool = False,
) -> tuple[bool, str]:
    out_path = SNAPSHOTS_DIR / city / f"{event_date}.parquet"
    if out_path.exists() and not force:
        try:
            existing = pd.read_parquet(out_path)
            if not existing.empty:
                return True, "skipped_existing"
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    bucket_iter: Iterator[dict[str, str]] = buckets
    bucket_bar = None
    if show_progress and tqdm is not None and len(buckets) > 1:
        bucket_bar = tqdm(
            buckets,
            desc=f"{city} {event_date}",
            unit="bucket",
            leave=False,
            dynamic_ncols=True,
        )
        bucket_iter = bucket_bar

    try:
        for bucket_info in bucket_iter:
            slug = bucket_info["slug"]
            bucket = bucket_info["bucket"]
            try:
                raw = download_bucket_day(api_key, slug=slug, event_date=event_date, channel=channel)
                frame = telonex_to_snapshot(raw, bucket, channel)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                logging.warning(
                    "Bucket download failed city=%s date=%s slug=%s: %s",
                    city,
                    event_date,
                    slug,
                    exc,
                )
            time.sleep(REQUEST_SLEEP_SECONDS)
    finally:
        if bucket_bar is not None:
            bucket_bar.close()

    if not frames:
        return False, "no_data"

    combined = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "bucket"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return True, f"wrote_{len(combined)}_rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Polymarket history from Telonex")
    parser.add_argument("--cities", nargs="+", default=TARGET_CITIES)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None, help="Inclusive end date (default: today UTC)")
    parser.add_argument(
        "--channel",
        default="book_snapshot_5",
        choices=["quotes", "book_snapshot_5", "book_snapshot_25", "book_snapshot_full", "order_book"],
    )
    parser.add_argument("--resume", action="store_true", help="Skip city-dates with existing parquet")
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Round-robin one date per city before advancing (balanced partial coverage)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars (default: on when stderr is a TTY)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (use for nohup/background)",
    )
    parser.add_argument("--max-city-dates", type=int, default=None, help="Stop after N city-dates (debug)")
    return parser.parse_args()


def run_backfill(args: argparse.Namespace) -> None:
    show_progress = progress_enabled(args)
    setup_logging(args.log_file, tqdm_mode=show_progress)
    api_key = load_telonex_key()
    catalog = load_catalog(show_progress=show_progress)
    floors = city_floor_dates(args.start_date, args.cities)
    tasks = enumerate_city_dates(catalog, args.cities, floors, args.end_date)
    tasks = order_tasks(tasks, args.cities, interleave=args.interleave)
    mode = "interleaved round-robin" if args.interleave else "city-by-city"
    plan_msg = (
        f"Backfill plan: {len(tasks)} city-dates across {len(args.cities)} cities "
        f"(channel={args.channel}, start={args.start_date}, order={mode})"
    )
    if show_progress and tqdm is not None:
        tqdm.write(plan_msg, file=sys.stderr)
    else:
        logging.info(plan_msg)

    progress = load_progress()
    completed = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    task_iter: Iterator[tuple[int, tuple[str, str, list[dict[str, str]]]]]
    task_iter = enumerate(tasks, start=1)
    city_bar = None
    if show_progress and tqdm is not None:
        city_bar = tqdm(
            tasks,
            total=len(tasks),
            desc="City-dates",
            unit="day",
            dynamic_ncols=True,
        )
        task_iter = enumerate(city_bar, start=1)

    for idx, (city, event_date, buckets) in task_iter:
        if city_bar is not None:
            city_bar.set_postfix(
                city=city[:8],
                date=event_date,
                wrote=completed,
                skip=skipped,
                fail=failed,
                refresh=False,
            )

        if args.max_city_dates is not None and completed + skipped >= args.max_city_dates:
            logging.info("Reached --max-city-dates=%d", args.max_city_dates)
            break

        key = f"{city}|{event_date}"
        if args.resume and not args.force:
            out_path = SNAPSHOTS_DIR / city / f"{event_date}.parquet"
            if out_path.exists():
                try:
                    if not pd.read_parquet(out_path).empty:
                        skipped += 1
                        if city_bar is None and idx % 25 == 0:
                            logging.info("Progress %d/%d (skipped existing)", idx, len(tasks))
                        continue
                except Exception:
                    pass

        ok, detail = backfill_city_date(
            api_key,
            city,
            event_date,
            buckets,
            channel=args.channel,
            force=args.force,
            show_progress=show_progress,
        )
        if ok:
            if detail == "skipped_existing":
                skipped += 1
            else:
                completed += 1
                progress["completed"].append(
                    {"key": key, "detail": detail, "at": datetime.now(timezone.utc).isoformat()}
                )
            if city_bar is None and completed > 0 and (completed % 5 == 0 or detail.startswith("wrote")):
                elapsed = time.time() - t0
                logging.info(
                    "[%d/%d] %s %s -> %s (done=%d skip=%d fail=%d, %.1fs)",
                    idx,
                    len(tasks),
                    city,
                    event_date,
                    detail,
                    completed,
                    skipped,
                    failed,
                    elapsed,
                )
        else:
            failed += 1
            progress["failed"].append({"key": key, "detail": detail, "at": datetime.now(timezone.utc).isoformat()})
            logging.warning("FAILED %s %s: %s", city, event_date, detail)

        if idx % 10 == 0:
            save_progress(progress)

    if city_bar is not None:
        city_bar.close()

    save_progress(progress)
    elapsed = time.time() - t0
    summary = (
        f"Backfill pass complete: wrote={completed} skipped={skipped} failed={failed} "
        f"elapsed={elapsed:.1f}s progress={PROGRESS_PATH}"
    )
    if show_progress and tqdm is not None:
        tqdm.write(summary, file=sys.stderr)
    else:
        logging.info(summary)


def main() -> None:
    args = parse_args()
    try:
        run_backfill(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user — progress saved; re-run with --resume")
        raise SystemExit(130)
    except Exception:
        logging.exception("Backfill crashed")
        raise


if __name__ == "__main__":
    main()
