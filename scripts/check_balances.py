# scripts/check_balances.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_api import PolymarketClient

client = PolymarketClient()

HOU_TOKEN = "4349364364702823313999155826816416707563057617613972695812754314942723210256"
LA_TOKEN = "42906705436731319043609701814603224071338727719862365829893099765972024511451"


def _print_balance(label: str, result: object) -> None:
    if isinstance(result, dict):
        raw = result.get("balance", "0")
        shares = int(raw) / 1_000_000
        print(f"{label}: {shares:.2f}")
        print(f"  raw: {result}")
    else:
        print(f"{label}: (unexpected response type)")
        print(f"  raw: {result!r}")


try:
    from py_clob_client_v2 import BalanceAllowanceParams, AssetType

    bal = client.client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    pusd = int(bal.get("balance", "0")) / 1_000_000
    print(f"pUSD balance: ${pusd:.2f}")

    hou_bal = client.client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=HOU_TOKEN)
    )
    hou_shares = int(hou_bal.get("balance", "0")) / 1_000_000
    print(f"Houston 92-93°F YES shares: {hou_shares:.2f}")

    la_bal = client.client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=LA_TOKEN)
    )
    la_shares = int(la_bal.get("balance", "0")) / 1_000_000
    print(f"LA 68-69°F YES shares: {la_shares:.2f}")
except Exception as exc:
    print(f"Direct balance fetch failed: {exc}")
    print("Trying PolymarketClient helpers...")
    try:
        print(f"pUSD balance: ${client.get_balance():.2f}")
        print(f"Houston 92-93°F YES shares: {client.get_conditional_balance(HOU_TOKEN):.2f}")
        print(f"LA 68-69°F YES shares: {client.get_conditional_balance(LA_TOKEN):.2f}")
    except Exception as helper_exc:
        print(f"Helper fetch failed: {helper_exc}")
        print("Client methods:", [m for m in dir(client.client) if not m.startswith("_")])
