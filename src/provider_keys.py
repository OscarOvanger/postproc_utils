"""Load API keys for third-party Polymarket data providers."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLYMARKETDATA_KEY_PATH = PROJECT_ROOT / "config" / "polymarketdata_key.txt"
TELONEX_KEY_PATH = PROJECT_ROOT / "config" / "telonex_key.txt"


def _read_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def load_polymarketdata_key() -> str:
    key = os.environ.get("POLYMARKETDATA_API_KEY") or _read_key_file(POLYMARKETDATA_KEY_PATH)
    if not key:
        raise FileNotFoundError(
            f"PolymarketData API key not found. Set POLYMARKETDATA_API_KEY or create {POLYMARKETDATA_KEY_PATH}"
        )
    return key


def load_telonex_key() -> str:
    key = os.environ.get("TELONEX_API_KEY") or _read_key_file(TELONEX_KEY_PATH)
    if not key:
        raise FileNotFoundError(
            f"Telonex API key not found. Set TELONEX_API_KEY or create {TELONEX_KEY_PATH}"
        )
    return key
