#!/usr/bin/env python3
"""Offline extraction of live trades at freeze for report TAB-9."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from poly_order_status import load_posted_orders  # noqa: E402
from poly_portfolio_status import (  # noqa: E402
    ORDER_LOG_PATH,
    build_token_index,
    enrich_token_index_for_dates,
    load_wu_targets,
    order_dates_from_posted,
    settlement_pnl,
    temp_in_bucket,
    winning_bucket_for_city,
)

CHALLENGE_START = date(2026, 6, 26)
FREEZE = date(2026, 7, 7)
SETTLED_THROUGH = date(2026, 7, 6)
OUTPUT_CSV = PROJECT_ROOT / "data" / "analysis" / "freeze_live_trades.csv"
OUTPUT_UNSETTLED_CSV = PROJECT_ROOT / "data" / "analysis" / "freeze_live_unsettled.csv"


def main() -> None:
    posted = load_posted_orders(ORDER_LOG_PATH)
    event_dates = order_dates_from_posted(posted)
    token_index = build_token_index(event_dates=event_dates, refresh_labels=False)
    enrich_token_index_for_dates(token_index, event_dates)
    wu = load_wu_targets()

    rows: list[dict] = []
    for order_id, record in sorted(posted.items(), key=lambda x: x[1].get("timestamp", "")):
        placed = date.fromisoformat(str(record.get("timestamp", ""))[:10])
        if placed < CHALLENGE_START or placed > FREEZE:
            continue
        token = str(record.get("token_id", ""))
        meta = token_index.get(token)
        if meta is None:
            continue
        event_date = meta.event_date or placed.isoformat()
        is_taker = not bool(record.get("post_only", True))
        entry = float(record.get("price", 0))
        size = float(record.get("size", 5))
        buckets = [meta.bucket_label] if meta.bucket_label else ["?"]

        winner, actual = winning_bucket_for_city(wu, meta.city, event_date, buckets)
        won = temp_in_bucket(actual, meta.bucket_label) if actual is not None and meta.bucket_label else None
        pnl = (
            settlement_pnl(n_contracts=size, entry_price=entry, won=bool(won), is_taker=is_taker)
            if won is not None
            else None
        )
        settled = pnl is not None and date.fromisoformat(event_date) <= SETTLED_THROUGH
        rows.append(
            {
                "order_id": order_id,
                "date": event_date,
                "city": meta.city_display,
                "bucket": meta.bucket_label,
                "entry": entry,
                "pnl": pnl if settled else None,
                "won": won if settled else None,
                "status": "settled" if settled else "unsettled",
            }
        )

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    settled_df = df[df["status"] == "settled"].copy()
    unsettled_df = df[df["status"] == "unsettled"].copy()
    settled_out = settled_df.drop(columns=["status", "order_id"], errors="ignore")
    unsettled_out = unsettled_df.drop(columns=["status", "order_id", "pnl", "won"], errors="ignore")
    settled_out.to_csv(OUTPUT_CSV, index=False)
    unsettled_out.to_csv(OUTPUT_UNSETTLED_CSV, index=False)

    wins = int(settled_df["won"].sum()) if not settled_df.empty else 0
    pnl = float(settled_df["pnl"].sum()) if not settled_df.empty else 0.0
    print(f"Placed {len(df)} | Settled {len(settled_df)} | Unsettled {len(unsettled_df)}")
    print(f"Wins {wins} | Win rate {100*wins/len(settled_df) if len(settled_df) else 0:.1f}% | PnL {pnl:+.2f}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
