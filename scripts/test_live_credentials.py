#!/usr/bin/env python3
"""Verify Polymarket API credentials by placing and cancelling a live test order."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_clob_client_v2 import (  # noqa: E402
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    ClobClient,
    OrderArgs,
    OrderPayload,
    OrderType,
    PartialCreateOrderOptions,
    Side,
)

from src.polymarket_api import (  # noqa: E402
    CHAIN_ID,
    CLOB_HOST,
    DEFAULT_CREDENTIALS_PATH,
    SIGNATURE_TYPE,
    load_credentials,
)

TERMINAL_CURSOR = "LTE="
MAX_MARKET_PAGES = 50


def _terminal_cursor(cursor: str | None) -> bool:
    return not cursor or cursor == TERMINAL_CURSOR


def _extract_order_id(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    for key in ("orderID", "order_id", "id"):
        value = response.get(key)
        if value:
            return str(value)
    nested = response.get("order")
    if isinstance(nested, dict):
        for key in ("orderID", "order_id", "id"):
            value = nested.get(key)
            if value:
                return str(value)
    return None


def _market_is_active(market: dict[str, Any]) -> bool:
    if market.get("closed"):
        return False
    if market.get("accepting_orders") is False:
        return False
    if market.get("acceptingOrders") is False:
        return False
    return True


def _token_from_market(market: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    tokens = market.get("tokens") or []
    if tokens:
        for token in tokens:
            if not isinstance(token, dict):
                continue
            token_id = token.get("token_id") or token.get("asset_id")
            if token_id:
                return str(token_id), market
        first = tokens[0]
        if isinstance(first, dict):
            token_id = first.get("token_id") or first.get("asset_id")
            if token_id:
                return str(token_id), market

    token_id = market.get("token_id") or market.get("asset_id")
    if token_id:
        return str(token_id), market
    return None, market


def find_active_token(client: ClobClient) -> tuple[str, dict[str, Any]]:
    """Paginate CLOB markets and return the first active token_id."""
    paginators = (
        ("get_sampling_simplified_markets", client.get_sampling_simplified_markets),
        ("get_markets", client.get_markets),
    )

    for label, fetch_page in paginators:
        cursor = "MA=="
        pages = 0
        while pages < MAX_MARKET_PAGES and not _terminal_cursor(cursor):
            page = fetch_page(next_cursor=cursor)
            pages += 1
            markets = page.get("data", []) if isinstance(page, dict) else []
            for market in markets:
                if not isinstance(market, dict) or not _market_is_active(market):
                    continue
                token_id, picked = _token_from_market(market)
                if token_id:
                    print(
                        f"Selected market via {label}: "
                        f"condition_id={picked.get('condition_id', 'n/a')} "
                        f"token_id={token_id[:20]}..."
                    )
                    return token_id, picked
            cursor = page.get("next_cursor") if isinstance(page, dict) else None

    raise RuntimeError(
        "No active market with an open order book found after paginating CLOB markets."
    )


def build_client(credentials: dict[str, str]) -> ClobClient:
    return ClobClient(
        host=CLOB_HOST,
        chain_id=CHAIN_ID,
        key=credentials["private_key"],
        signature_type=SIGNATURE_TYPE,
        funder=credentials["funder"],
        creds=ApiCreds(
            api_key=credentials["api_key"],
            api_secret=credentials["api_secret"],
            api_passphrase=credentials["api_passphrase"],
        ),
        retry_on_error=True,
    )


def main() -> None:
    try:
        creds_path = DEFAULT_CREDENTIALS_PATH
        print(f"Loading credentials from {creds_path}")
        credentials = load_credentials(creds_path)
        client = build_client(credentials)

        client.get_api_keys()
        print("Auth OK")

        balance_result = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw_balance = 0
        if isinstance(balance_result, dict):
            raw_balance = balance_result.get("balance", 0)
        pusd_balance = float(raw_balance) / 1_000_000
        print(f"pUSD balance: ${pusd_balance:.6f}")

        token_id, market = find_active_token(client)
        neg_risk = bool(market.get("neg_risk", market.get("negRisk", True)))
        tick_size = str(market.get("minimum_tick_size") or market.get("tick_size") or "0.01")

        confirm = input(
            "This will place a real $0.01 order on Polymarket and immediately cancel it. "
            "Proceed? (y/n): "
        ).strip()
        if confirm.lower() != "y":
            print("Aborted.")
            sys.exit(0)

        order_args = OrderArgs(
            token_id=token_id,
            price=0.01,
            size=5,
            side=Side.BUY,
        )
        options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
        signed_order = client.create_order(order_args, options)

        post_response = client.post_order(
            signed_order,
            order_type=OrderType.GTC,
            post_only=True,
        )
        print("Post response:")
        print(json.dumps(post_response, indent=2, default=str))

        order_id = _extract_order_id(post_response)
        if not order_id:
            raise RuntimeError(f"Could not extract order ID from post response: {post_response}")

        cancel_response = client.cancel_order(OrderPayload(orderID=order_id))
        print("Cancel response:")
        print(json.dumps(cancel_response, indent=2, default=str))

        print("LIVE CREDENTIALS VERIFIED")
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
