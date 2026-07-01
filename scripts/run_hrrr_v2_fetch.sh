#!/bin/bash
# scripts/run_hrrr_v2_fetch.sh
# HRRR 10Z fetch in priority order: active trading cities first, recent data first.

set -e
PYTHON=".venv/bin/python"
SCRIPT="scripts/fetch_hrrr_all_cities.py"
WORKERS="${HRRR_WORKERS:-18}"
DAY_WORKERS="${HRRR_DAY_WORKERS:-8}"
MAX_CONCURRENT="${HRRR_MAX_CONCURRENT:-32}"
export PYTHONWARNINGS=ignore

echo "=== HRRR 10Z Fetch: Phase 1 - Active trading cities, 2025-2026 ==="
for city in houston los_angeles; do
  echo ">>> $city (2025-01-01 to 2026-06-25)..."
  $PYTHON $SCRIPT --start 2025-01-01 --end 2026-06-25 --city $city \
    --day-workers $DAY_WORKERS --max-concurrent $MAX_CONCURRENT
done

echo ""
echo "=== Phase 2 - Remaining cities, 2025-2026 ==="
for city in austin dallas chicago san_francisco seattle new_york miami atlanta; do
  echo ">>> $city (2025-01-01 to 2026-06-25)..."
  $PYTHON $SCRIPT --start 2025-01-01 --end 2026-06-25 --city $city \
    --day-workers $DAY_WORKERS --max-concurrent $MAX_CONCURRENT
done

echo ""
echo "=== Phase 3 - All cities, 2021-2024 training data ==="
for city in houston los_angeles austin dallas chicago san_francisco seattle new_york miami atlanta; do
  echo ">>> $city (2021-01-01 to 2024-12-31)..."
  $PYTHON $SCRIPT --start 2021-01-01 --end 2024-12-31 --city $city \
    --day-workers $DAY_WORKERS --max-concurrent $MAX_CONCURRENT
done

echo ""
echo "=== ALL DONE ==="
