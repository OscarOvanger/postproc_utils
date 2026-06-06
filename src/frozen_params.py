"""Load and persist frozen threshold parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snapshot_stability import SPLIT_DIR

FROZEN_PARAMS_PATH = SPLIT_DIR / "frozen_params.json"


def load_frozen_params(split_dir: Path = SPLIT_DIR) -> dict[str, Any]:
    """Return the frozen parameter dict, or empty dict if not yet created."""
    path = split_dir / "frozen_params.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_frozen_params(
    updates: dict[str, Any],
    split_dir: Path = SPLIT_DIR,
) -> dict[str, Any]:
    """Merge updates into frozen_params.json and return the full dict."""
    path = split_dir / "frozen_params.json"
    payload = load_frozen_params(split_dir)
    payload.update(updates)
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload
