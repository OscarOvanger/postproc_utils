"""Audit local data directory sizes for HuggingFace migration planning."""

from __future__ import annotations

import sys
from pathlib import Path

import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackj.fetch_nws_forecast import TRAIN_CITIES  # noqa: E402


def dir_size_and_count(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            count += 1
    return total, count


def fmt_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def main() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "city_config.json").read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str]] = []

    for city in TRAIN_CITIES:
        station = str(config[city]["nws_station"]).lower()
        gfs_dir = PROJECT_ROOT / "data" / "raw" / (f"gfs_{station}" if station != "kaus" else "gfs_kaus")
        size, count = dir_size_and_count(gfs_dir)
        rows.append((f"data/raw/{gfs_dir.name}/", fmt_mb(size), str(count)))

    for label, rel in [
        ("data/trackb/", "data/trackb"),
        ("data/trackj/", "data/trackj"),
        ("data/splits/", "data/splits"),
        ("models/trackj/", "models/trackj"),
        ("models/trackb/", "models/trackb"),
    ]:
        size, count = dir_size_and_count(PROJECT_ROOT / rel)
        rows.append((label, fmt_mb(size), str(count)))

    print(f"{'Directory':<35} | {'Size on disk':>12} | {'N files':>8}")
    print("-" * 62)
    for directory, size, n_files in rows:
        print(f"{directory:<35} | {size:>12} | {n_files:>8}")


if __name__ == "__main__":
    main()
