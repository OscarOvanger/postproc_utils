#!/bin/bash
# Foreground HRRR 10Z fetch with per-city tqdm bars, overall ETA, and disk stats.
# Run from project root:
#   ./scripts/run_hrrr_v2_fetch_foreground.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
SCRIPT="scripts/fetch_hrrr_all_cities.py"
DATA_DIR="data/hrrr_v2"
DAY_WORKERS="${HRRR_DAY_WORKERS:-8}"
MAX_CONCURRENT="${HRRR_MAX_CONCURRENT:-32}"
export PYTHONWARNINGS=ignore

# --- disk helpers ---
human_bytes() {
  local bytes=$1
  if (( bytes >= 1073741824 )); then
    awk -v b="$bytes" 'BEGIN { printf "%.2f GB", b/1073741824 }'
  elif (( bytes >= 1048576 )); then
    awk -v b="$bytes" 'BEGIN { printf "%.1f MB", b/1048576 }'
  else
    awk -v b="$bytes" 'BEGIN { printf "%.0f KB", b/1024 }'
  fi
}

print_disk_status() {
  local label="${1:-}"
  local cache_bytes
  cache_bytes=$(du -sk "$DATA_DIR" 2>/dev/null | awk '{print $1 * 1024}' || echo 0)
  local grib_bytes
  grib_bytes=$(du -sk "$HOME/data/hrrr" 2>/dev/null | awk '{print $1 * 1024}' || echo 0)
  local avail_bytes
  avail_bytes=$(df -k "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $4 * 1024}')
  local used_pct
  used_pct=$(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $5}')
  local avail_human
  avail_human=$(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
  echo ""
  echo "── Disk ──────────────────────────────────────────────"
  [[ -n "$label" ]] && echo "  $label"
  echo "  HRRR cache ($DATA_DIR): $(human_bytes "$cache_bytes")"
  echo "  GRIB scratch (~/data/hrrr): $(human_bytes "$grib_bytes")"
  echo "  Volume free space:      $avail_human  (${used_pct} used on data volume)"
  echo "──────────────────────────────────────────────────────"
}

# Count days still needing fetch (respects monthly CSV cache)
count_pending_days() {
  local city=$1 start=$2 end=$3
  $PYTHON - <<PY
import sys
from datetime import date
from pathlib import Path
import pandas as pd

city = "$city"
start = date.fromisoformat("$start")
end = date.fromisoformat("$end")
data_dir = Path("$DATA_DIR") / city
dates = pd.date_range(start, end, freq="D").date.tolist()
pending = 0
for d in dates:
    cache = data_dir / f"hrrr_{city}_{d:%Y%m}.csv"
    if not cache.exists():
        pending += 1
        continue
    try:
        df = pd.read_csv(cache)
        if df.empty or str(d) not in df["date"].astype(str).values:
            pending += 1
    except Exception:
        pending += 1
print(pending)
PY
}

format_duration() {
  local secs=$1
  if (( secs < 0 )); then secs=0; fi
  local h=$(( secs / 3600 ))
  local m=$(( (secs % 3600) / 60 ))
  local s=$(( secs % 60 ))
  if (( h > 0 )); then
    printf "%dh %02dm %02ds" "$h" "$m" "$s"
  elif (( m > 0 )); then
    printf "%dm %02ds" "$m" "$s"
  else
    printf "%ds" "$s"
  fi
}

# Job queue: "city start end"
JOBS=(
  "houston        2025-01-01 2026-06-25"
  "los_angeles    2025-01-01 2026-06-25"
  "austin         2025-01-01 2026-06-25"
  "dallas         2025-01-01 2026-06-25"
  "chicago        2025-01-01 2026-06-25"
  "san_francisco  2025-01-01 2026-06-25"
  "seattle        2025-01-01 2026-06-25"
  "new_york       2025-01-01 2026-06-25"
  "miami          2025-01-01 2026-06-25"
  "atlanta        2025-01-01 2026-06-25"
  "houston        2021-01-01 2024-12-31"
  "los_angeles    2021-01-01 2024-12-31"
  "austin         2021-01-01 2024-12-31"
  "dallas         2021-01-01 2024-12-31"
  "chicago        2021-01-01 2024-12-31"
  "san_francisco  2021-01-01 2024-12-31"
  "seattle        2021-01-01 2024-12-31"
  "new_york       2021-01-01 2024-12-31"
  "miami          2021-01-01 2024-12-31"
  "atlanta        2021-01-01 2024-12-31"
)

echo "=== HRRR 10Z Foreground Fetch ==="
echo "Jobs: ${#JOBS[@]} city-ranges  |  Python: $PYTHON"
echo "Parallelism: ${DAY_WORKERS} days/city, ${MAX_CONCURRENT} max concurrent GRIB downloads"
print_disk_status "Before fetch"

# Pre-scan pending work
TOTAL_PENDING=0
declare -a JOB_PENDING
echo ""
echo "Scanning cache for pending days..."
for job in "${JOBS[@]}"; do
  read -r city start end <<< "$job"
  pending=$(count_pending_days "$city" "$start" "$end")
  JOB_PENDING+=("$pending")
  TOTAL_PENDING=$(( TOTAL_PENDING + pending ))
  printf "  %-16s %s → %s : %4d days to fetch\n" "$city" "$start" "$end" "$pending"
done
echo ""
echo "Total pending days across all jobs: $TOTAL_PENDING"
if (( TOTAL_PENDING == 0 )); then
  echo "Nothing to fetch — all dates cached. Building parquet..."
  $PYTHON "$SCRIPT" --start 2021-01-01 --end 2026-06-25
  print_disk_status "Complete (cache hit)"
  exit 0
fi

GLOBAL_START=$SECONDS
COMPLETED_DAYS=0
JOB_IDX=0

for job in "${JOBS[@]}"; do
  read -r city start end <<< "$job"
  pending=${JOB_PENDING[$JOB_IDX]}
  JOB_IDX=$(( JOB_IDX + 1 ))

  if (( pending == 0 )); then
    echo ""
    echo ">>> [$JOB_IDX/${#JOBS[@]}] $city ($start → $end) — fully cached, skipping"
    continue
  fi

  elapsed=$(( SECONDS - GLOBAL_START ))
  if (( COMPLETED_DAYS > 0 )); then
    rate_x100=$(( (COMPLETED_DAYS * 100) / elapsed ))   # days per 100 sec
    if (( rate_x100 > 0 )); then
      remaining=$(( TOTAL_PENDING - COMPLETED_DAYS ))
      eta_secs=$(( (remaining * elapsed * 100) / (COMPLETED_DAYS * 100) ))
      overall_eta=$(format_duration "$eta_secs")
    else
      overall_eta="calculating..."
    fi
  else
    overall_eta="calculating..."
  fi

  echo ""
  echo "══════════════════════════════════════════════════════"
  echo " Job $JOB_IDX/${#JOBS[@]}: $city  ($start → $end)"
  echo " Pending this job: $pending days  |  Overall done: $COMPLETED_DAYS / $TOTAL_PENDING"
  echo " Overall ETA: $overall_eta  |  Elapsed: $(format_duration "$elapsed")"
  print_disk_status
  echo "══════════════════════════════════════════════════════"

  CITY_START=$SECONDS
  $PYTHON "$SCRIPT" --start "$start" --end "$end" --city "$city" \
    --day-workers "$DAY_WORKERS" --max-concurrent "$MAX_CONCURRENT"
  CITY_ELAPSED=$(( SECONDS - CITY_START ))

  # Re-count how many were actually fetched this run
  still_pending=$(count_pending_days "$city" "$start" "$end")
  fetched_this=$(( pending - still_pending ))
  COMPLETED_DAYS=$(( COMPLETED_DAYS + fetched_this ))

  if (( fetched_this > 0 && CITY_ELAPSED > 0 )); then
    city_rate=$(awk -v f="$fetched_this" -v e="$CITY_ELAPSED" 'BEGIN { printf "%.1f", f / (e/3600) }')
    echo "✓ $city finished: $fetched_this fetched in $(format_duration "$CITY_ELAPSED") (~${city_rate} days/hr)"
  else
    echo "✓ $city finished (cache only)"
  fi
done

TOTAL_ELAPSED=$(( SECONDS - GLOBAL_START ))
echo ""
echo "=== ALL DONE ==="
echo "Total elapsed: $(format_duration "$TOTAL_ELAPSED")"
print_disk_status "After fetch"

# Rebuild combined parquet for all cities (no re-fetch)
echo ""
echo "Building combined parquet..."
$PYTHON - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fetch_hrrr", Path("scripts/fetch_hrrr_all_cities.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
combined = mod.build_combined_parquet(list(mod.HRRR_STATIONS))
mod.print_summary(combined)
PY

print_disk_status "Final"
