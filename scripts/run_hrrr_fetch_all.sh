#!/bin/bash
# scripts/run_hrrr_fetch_all.sh
# HRRR fetch in priority order: active trading cities first, recent data first.

set -e
PYTHON=".venv/bin/python"
SCRIPT="scripts/fetch_hrrr_all_cities.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

export PYTHONWARNINGS=ignore

run_phase() {
  local phase_name="$1"
  local start_date="$2"
  local end_date="$3"
  local append_logs=0
  if [[ "$4" == "0" || "$4" == "1" ]]; then
    append_logs="$4"
    shift 4
  else
    shift 3
  fi
  local cities=("$@")
  local queue
  queue=$(IFS=,; echo "${cities[*]}")

  for i in "${!cities[@]}"; do
    local city="${cities[$i]}"
    local tee_args=()
    if [[ "$append_logs" == "1" ]]; then
      tee_args=(-a)
    fi
    $PYTHON "$SCRIPT" \
      --start "$start_date" \
      --end "$end_date" \
      --city "$city" \
      --phase "$phase_name" \
      --queue "$queue" \
      --queue-pos "$i" \
      2>&1 | tee "${tee_args[@]}" "$LOG_DIR/hrrr_fetch_${city}.log"
  done
}

echo "=== HRRR FETCH START ==="
echo "Logs: $LOG_DIR/hrrr_fetch_<city>.log"
echo ""

run_phase "Phase 1 - Active trading cities, 2025-2026" \
  2025-01-01 2026-06-25 houston los_angeles

run_phase "Phase 2 - Remaining cities, 2025-2026" \
  2025-01-01 2026-06-25 \
  austin dallas chicago san_francisco seattle new_york miami atlanta

run_phase "Phase 3 - All cities, 2021-2024 training data" \
  2021-01-01 2024-12-31 1 \
  houston los_angeles austin dallas chicago san_francisco seattle new_york miami atlanta

echo ""
echo "=== ALL HRRR FETCHES COMPLETE ==="
