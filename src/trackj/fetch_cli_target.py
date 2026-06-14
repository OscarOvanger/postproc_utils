from __future__ import annotations

import calendar
import csv
import hashlib
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .hf_data_store import sync_city_to_hf


IEM_RETRIEVE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
WMO_RE = re.compile(
    r"^\s*(?P<ttaaii>[A-Z]{4}\d{2})\s+(?P<cccc>[A-Z]{4})\s+(?P<ddhhmm>\d{6})(?:\s+(?P<bbb>[A-Z]{3}))?\s*$",
    re.MULTILINE,
)
PRODUCT_ID_RE = re.compile(r"(?P<stamp>\d{12})-(?P<cccc>[A-Z]{4})-(?P<ttaaii>[A-Z]{4}\d{2})-(?P<pil>[A-Z0-9]{3,6})(?:-(?P<bbb>[A-Z]{3}))?")
HEADLINE_RE = re.compile(
    r"\.\.\.THE\s+(?P<station>.*?)\s+CLIMATE\s+SUMMARY\s+FOR\s+(?P<month>[A-Z]+)\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})\.\.\.",
    re.IGNORECASE | re.DOTALL,
)
MONTHS = {name: i for i, names in enumerate([
    (), ("JAN", "JANUARY"), ("FEB", "FEBRUARY"), ("MAR", "MARCH"), ("APR", "APRIL"), ("MAY",),
    ("JUN", "JUNE"), ("JUL", "JULY"), ("AUG", "AUGUST"), ("SEP", "SEPT", "SEPTEMBER"),
    ("OCT", "OCTOBER"), ("NOV", "NOVEMBER"), ("DEC", "DECEMBER")
]) for name in names}


@dataclass(frozen=True)
class RawProduct:
    product_id: str
    issue_timestamp_utc: datetime | None
    path: Path
    url: str
    sha256: str


def make_session() -> requests.Session:
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.headers.update({"User-Agent": "MCP_trading_research oscar@utexas.edu"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_text(text: str) -> str:
    return text.replace("\x01", "").replace("\x03", "").replace("\r\n", "\n").strip()


def month_ranges(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        start_dt = datetime.combine(max(cursor, start), datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(min(date(cursor.year, cursor.month, last_day), end) + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        yield start_dt, end_dt
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)


def build_product_id(issue_timestamp_utc: datetime, cccc: str, ttaaii: str, pil: str, bbb: str | None = None) -> str:
    base = f"{issue_timestamp_utc:%Y%m%d%H%M}-{cccc}-{ttaaii}-{pil}"
    return f"{base}-{bbb}" if bbb else base


def infer_issue_timestamp(product_text: str, batch_start: datetime, batch_end: datetime) -> datetime | None:
    match = WMO_RE.search(product_text)
    if not match:
        return None
    day, hour, minute = int(match["ddhhmm"][:2]), int(match["ddhhmm"][2:4]), int(match["ddhhmm"][4:6])
    candidates = []
    for offset in (-1, 0, 1):
        month = batch_start.month + offset
        year = batch_start.year
        if month < 1:
            month += 12; year -= 1
        elif month > 12:
            month -= 12; year += 1
        try:
            candidates.append(datetime(year, month, day, hour, minute, tzinfo=timezone.utc))
        except ValueError:
            pass
    if not candidates:
        return None
    window = [c for c in candidates if batch_start - timedelta(days=1) <= c < batch_end + timedelta(days=1)]
    return min(window or candidates, key=lambda c: abs(c - batch_start))


def split_products(response_text: str, pil: str) -> list[str]:
    text = response_text.replace("\r\n", "\n")
    chunks = [normalize_text(chunk) for chunk in re.findall(r"\x01(.*?)\x03", text, re.DOTALL)]
    products = [chunk for chunk in chunks if pil in chunk and "$$" in chunk]
    if products:
        return products
    starts = [m.start() for m in re.finditer(r"(?m)^\s*\d{3}\s*\n[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}", text)]
    return [normalize_text(text[start:starts[i + 1] if i + 1 < len(starts) else len(text)]) for i, start in enumerate(starts) if pil in text[start:starts[i + 1] if i + 1 < len(starts) else len(text)]]


def fetch_cli_range(city_config: dict, start_date: date, end_date: date, raw_dir: Path, overwrite: bool = False, sleep_seconds: float = 1.1) -> list[RawProduct]:
    pil, wfo = city_config["nws_pil"], city_config["wfo"]
    session = make_session()
    saved_all = []
    for batch_start, batch_end in month_ranges(start_date, end_date):
        response = session.get(IEM_RETRIEVE_URL, params={"pil": pil, "center": wfo, "sdate": batch_start.strftime("%Y-%m-%d %H:%M"), "edate": batch_end.strftime("%Y-%m-%d %H:%M"), "limit": "9999", "fmt": "text", "order": "asc"}, timeout=10)
        response.raise_for_status()
        products = split_products(response.text, pil)
        rows = save_products(products, raw_dir, batch_start, batch_end, pil, overwrite=overwrite)
        append_manifest(raw_dir, rows, batch_start, batch_end)
        saved_all.extend(rows)
        print(f"{city_config['city']} {batch_start:%Y-%m-%d} to {batch_end:%Y-%m-%d}: retrieved {len(products)} products")
        time.sleep(sleep_seconds)
    return saved_all


def save_products(products: list[str], raw_dir: Path, batch_start: datetime, batch_end: datetime, pil: str, overwrite: bool = False) -> list[RawProduct]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved, seen = [], {}
    for text in products:
        match = WMO_RE.search(text)
        issue_ts = infer_issue_timestamp(text, batch_start, batch_end)
        if match and issue_ts:
            product_id = build_product_id(issue_ts, match["cccc"], match["ttaaii"], pil, match["bbb"])
        else:
            product_id = f"UNKNOWN-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        if product_id in seen:
            product_id = f"{product_id}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        seen[product_id] = 1
        path = raw_dir / f"{product_id}.txt"
        if overwrite or not path.exists():
            path.write_text(text.strip() + "\n", encoding="utf-8")
        saved.append(RawProduct(product_id, issue_ts, path, f"https://mesonet.agron.iastate.edu/api/1/nwstext/{product_id}", hashlib.sha256(text.encode()).hexdigest()))
    return saved


def append_manifest(raw_dir: Path, rows: list[RawProduct], batch_start: datetime, batch_end: datetime) -> None:
    path = raw_dir / "fetch_manifest.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id", "issue_timestamp_utc", "path", "url", "sha256", "batch_start_utc", "batch_end_utc"])
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({**asdict(row), "issue_timestamp_utc": row.issue_timestamp_utc.isoformat().replace("+00:00", "Z") if row.issue_timestamp_utc else "", "path": row.path.as_posix(), "batch_start_utc": batch_start.isoformat(), "batch_end_utc": batch_end.isoformat()})


def parse_file(path: Path, city_config: dict) -> dict:
    text = normalize_text(path.read_text(encoding="utf-8"))
    valid_date, station_name = parse_valid_date(text)
    tmax, tmin = parse_temperature_values(text)
    issue_ts = parse_product_id_timestamp(path.stem)
    return {
        "date": valid_date,
        "tmax_f": tmax,
        "tmin_f": tmin,
        "source_product_id": path.stem,
        "station_name": station_name,
        "station_id": city_config["nws_station"],
        "source": f"NWS_{city_config['nws_pil']}",
        "report_issue_timestamp_utc": issue_ts.isoformat().replace("+00:00", "Z") if issue_ts else None,
    }


def parse_product_id_timestamp(product_id: str | None) -> datetime | None:
    match = PRODUCT_ID_RE.search(product_id or "")
    return datetime.strptime(match["stamp"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc) if match else None


def parse_valid_date(text: str) -> tuple[str | None, str | None]:
    match = HEADLINE_RE.search(text)
    if not match:
        return None, None
    return date(int(match["year"]), MONTHS[match["month"].upper()], int(match["day"])).isoformat(), " ".join(match["station"].split()).upper()


def parse_temperature_values(text: str) -> tuple[float | None, float | None]:
    section = re.search(r"TEMPERATURE \(F\)\s*(?P<body>.*?)(?:\n\s*[A-Z][A-Z /]+(?:\([A-Z]+\))?\s*\n|\n\s*\.{8,}|\Z)", text, re.I | re.S)
    body = section["body"] if section else ""
    vals = []
    for label in ("MAXIMUM", "MINIMUM"):
        m = re.search(rf"^\s*{label}\s+(?P<value>-?\d+(?:\.\d+)?|MM|M)\b", body, re.M)
        vals.append(float(m["value"]) if m and m["value"] not in {"M", "MM"} else None)
    return vals[0], vals[1]


def _correction_rank(product_id: str | None) -> int:
    suffix = (product_id or "").rsplit("-", 1)[-1]
    return 2 if suffix in {"CCA", "CCB", "CCC", "COR"} else 1 if suffix in {"AAA", "AAB", "AAC"} else 0


def fetch_cli_target(city_config: dict, start_date: date, end_date: date, raw_dir: Path, output_dir: Path, no_fetch: bool = False) -> pd.DataFrame:
    city = city_config["city"]
    city_raw_dir = Path(raw_dir) / city / "cli"
    if not no_fetch:
        fetch_cli_range(city_config, start_date, end_date, city_raw_dir)
        if os.environ.get("TRACKJ_SKIP_HF_SYNC", "0") == "1":
            print(f"HF raw sync skipped for {city} CLI data (TRACKJ_SKIP_HF_SYNC=1)")
        else:
            try:
                sync_city_to_hf(city, raw_dir)
            except Exception as exc:
                print(f"Warning: HF raw sync skipped for {city} CLI data: {exc}")
    parsed = pd.DataFrame([parse_file(path, city_config) for path in sorted(city_raw_dir.glob("*.txt"))])
    calendar_df = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")})
    if parsed.empty:
        target = calendar_df.assign(tmax_f=pd.NA, tmin_f=pd.NA, source_product_id=pd.NA)
    else:
        parsed["issue_dt"] = pd.to_datetime(parsed["report_issue_timestamp_utc"], utc=True, errors="coerce")
        parsed["correction_rank"] = parsed["source_product_id"].map(_correction_rank)
        selected = parsed[parsed["date"].notna()].sort_values(["date", "correction_rank", "issue_dt", "source_product_id"]).groupby("date", as_index=False).tail(1)
        target = calendar_df.merge(selected[["date", "tmax_f", "tmin_f", "source_product_id"]], on="date", how="left")
    city_output = Path(output_dir) / city
    city_output.mkdir(parents=True, exist_ok=True)
    target.to_parquet(city_output / "cli_target.parquet", index=False)
    return target
