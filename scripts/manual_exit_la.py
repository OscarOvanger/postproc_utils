# scripts/manual_exit_la.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import PolymarketClient

TOKEN_ID = "42906705436731319043609701814603224071338727719862365829893099765972024511451"

client = PolymarketClient()
resp = client.place_order(
    token_id=TOKEN_ID,
    side="SELL",
    price=0.37,
    size=6.0,
    tick_size="0.01",
    neg_risk=True,
    dry_run=False,
    post_only=True,
)
print("Order response:", resp)
order_id = resp.get("orderID") or resp.get("order_id") or resp.get("id")
print(f"Order ID: {order_id}")
print(f"Placed: LA 68-69°F SELL, 6 contracts @ $0.37, maker GTC post_only")
print(f"If filled: profit = 6 x ($0.37 - $0.22) = $0.90")
