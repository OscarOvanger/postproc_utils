# scripts/manual_reorder_la.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import PolymarketClient

TOKEN_ID = "42906705436731319043609701814603224071338727719862365829893099765972024511451"
OLD_ORDER_ID = "0x317974d70fb39c21c3672850ed6de57027afa0098d09eefb7efc51efa92313f4"

client = PolymarketClient()

# Cancel old order
print("Cancelling old order...")
clob = client.client
if hasattr(clob, "cancel"):
    try:
        cancel_resp = clob.cancel(OLD_ORDER_ID)
    except Exception:
        cancel_resp = client.cancel_order(OLD_ORDER_ID)
elif hasattr(clob, "cancel_order"):
    from src.polymarket_api import _clob_types

    OrderPayload = _clob_types()["OrderPayload"]
    cancel_resp = clob.cancel_order(OrderPayload(orderID=OLD_ORDER_ID))
else:
    cancel_resp = client.cancel_order(OLD_ORDER_ID)
print(f"Cancel response: {cancel_resp}")

# Place new order
print("\nPlacing new order at $0.22...")
resp = client.place_order(
    token_id=TOKEN_ID,
    side="YES",
    price=0.22,
    size=6.0,
    tick_size="0.01",
    neg_risk=True,
    dry_run=False,
    post_only=True,
)
print(f"Order response: {resp}")
order_id = resp.get("orderID") or resp.get("order_id") or resp.get("id")
print(f"Order ID: {order_id}")
print(f"Placed: LA 68-69°F, 6 contracts @ $0.22, maker GTC post_only")
print(f"Capital at risk: $1.32")
