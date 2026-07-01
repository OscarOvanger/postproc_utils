# scripts/check_open_orders.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import PolymarketClient

client = PolymarketClient()
clob = client.client

if hasattr(clob, "get_orders"):
    try:
        orders = clob.get_orders()
    except Exception as exc:
        print(f"get_orders() failed: {exc}")
        print("Trying get_open_orders()...")
        orders = clob.get_open_orders()
elif hasattr(clob, "get_open_orders"):
    orders = clob.get_open_orders()
else:
    print("Neither get_orders() nor get_open_orders() available on client.")
    print("Client methods:", [m for m in dir(clob) if not m.startswith("_")])
    sys.exit(1)

if orders is None:
    orders = []
if isinstance(orders, dict):
    orders = orders.get("data", orders.get("orders", [orders]))

print(f"Open orders: {len(orders)}\n")
for o in orders:
    if not isinstance(o, dict):
        print(f"  (non-dict entry): {o!r}\n")
        continue
    oid = o.get("id") or o.get("orderID", "?")
    token = o.get("asset_id", o.get("token_id", "?"))
    side = o.get("side", "?")
    price = o.get("price", "?")
    orig_size = o.get("original_size", o.get("size", "?"))
    remaining = o.get("size_matched", "?")
    status = o.get("status", "?")
    created = o.get("created_at", o.get("timestamp", "?"))
    oid_str = str(oid)
    token_str = str(token)
    print(f"  ID: {oid_str[:20]}...")
    print(f"  Token: {token_str[:30]}...")
    print(f"  Side: {side} | Price: {price} | Size: {orig_size} | Matched: {remaining}")
    print(f"  Status: {status} | Created: {created}")
    print()
