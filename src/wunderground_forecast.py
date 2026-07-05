"""Scrape live WU daily high forecast from Polymarket resolution station pages."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITY_CONFIG_PATH = PROJECT_ROOT / "config" / "city_config.json"

# Polymarket WU resolution station pages (ICAO suffix in URL).
WU_PAGE_BY_CITY: dict[str, str] = {
    "austin": "https://www.wunderground.com/weather/us/tx/austin/KAUS",
    "atlanta": "https://www.wunderground.com/weather/us/ga/atlanta/KATL",
    "chicago": "https://www.wunderground.com/weather/us/il/chicago/KORD",
    "dallas": "https://www.wunderground.com/history/daily/us/tx/dallas/KDAL",
    "houston": "https://www.wunderground.com/weather/us/tx/houston/KHOU",
    "los_angeles": "https://www.wunderground.com/weather/us/ca/los-angeles/KLAX",
    "miami": "https://www.wunderground.com/weather/us/fl/miami/KMIA",
    "new_york": "https://www.wunderground.com/weather/us/ny/new-york-city/KLGA",
    "san_francisco": "https://www.wunderground.com/weather/us/ca/san-francisco/KSFO",
    "seattle": "https://www.wunderground.com/weather/us/wa/seatac/KSEA",
}

# "iconTodayThu 07/02 High 97 °F" / "High 97F" in the daily forecast block.
FORECAST_HIGH_RE = re.compile(r"High\s+(\d+)\s*(?:&#176;|°)?F", re.IGNORECASE)
HIGH_AROUND_RE = re.compile(r"High around (\d+)F", re.IGNORECASE)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MCP_Project/1.0)"})


def wu_url_for_city(city_slug: str) -> str | None:
    return WU_PAGE_BY_CITY.get(city_slug)


def _parse_high_from_html(html: str) -> int | None:
    match = FORECAST_HIGH_RE.search(html)
    if match:
        return int(match.group(1))

    match = HIGH_AROUND_RE.search(html)
    if match:
        return int(match.group(1))

    return None


def fetch_wu_high(city_slug: str, *, timeout: float = 30.0) -> int | None:
    """Return scraped WU daily high (°F) for city, or None on failure."""
    url = wu_url_for_city(city_slug)
    if not url:
        return None
    try:
        response = SESSION.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return _parse_high_from_html(response.text)


def load_city_stations() -> dict[str, str]:
    """Map city slug -> ICAO station from city_config (for footnotes)."""
    if not CITY_CONFIG_PATH.exists():
        return {}
    config = json.loads(CITY_CONFIG_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for slug in WU_PAGE_BY_CITY:
        entry = config.get(slug) or config.get(f"{slug}_city")
        if entry and entry.get("nws_station"):
            out[slug] = str(entry["nws_station"])
    return out
