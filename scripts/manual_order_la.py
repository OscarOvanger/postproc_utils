# scripts/manual_order_la.py
# One-shot script: place maker buy on LA 68-69°F, then print order response and exit.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import PolymarketClient

TOKEN_ID = "42906705436731319043609701814603224071338727719862365829893099765972024511451"
PRICE = 0.18
SIZE = 6
TICK_SIZE = "0.01"
NEG_RISK = True

client = PolymarketClient()
resp = client.place_order(
    token_id=TOKEN_ID,
    side="YES",
    price=PRICE,
    size=float(SIZE),
    tick_size=TICK_SIZE,
    neg_risk=NEG_RISK,
    dry_run=False,
    post_only=True,
)
print("Order response:", resp)
order_id = resp.get("orderID") or resp.get("order_id") or resp.get("id")
print(f"Order ID: {order_id}")
print(f"Placed: LA 68-69°F, {SIZE} contracts @ ${PRICE}, maker GTC post_only")
