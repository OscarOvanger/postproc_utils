#!/bin/bash
# Polymarket v5 autotrader cron wrapper (paper or live).
# Usage: cron_autotrader.sh [paper|live]

# Capture all stderr so cron failures are never silent.
STDERR_LOG="/Users/oscaro/Desktop/MCP_Project/logs/cron_stderr.log"
LOCKDIR="/Users/oscaro/Desktop/MCP_Project/logs/cron_autotrader.lockdir"
STALE_LOCK_MINUTES=60
mkdir -p "$(dirname "$STDERR_LOG")"
exec 2>> "$STDERR_LOG"
echo "=== $(date) === $0 $*" >> "$STDERR_LOG"

set -euo pipefail

PROJECT_DIR="/Users/oscaro/Desktop/MCP_Project"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$PROJECT_DIR"

release_lock() {
    rmdir "$LOCKDIR" 2>/dev/null || true
}
trap release_lock EXIT

acquire_lock() {
    if mkdir "$LOCKDIR" 2>/dev/null; then
        return 0
    fi
    if [[ ! -d "$LOCKDIR" ]]; then
        echo "previous run still active"
        return 1
    fi
    # macOS: stat -f %m for mtime
    lock_mtime=$(stat -f %m "$LOCKDIR" 2>/dev/null || echo 0)
    now=$(date +%s)
    age_min=$(( (now - lock_mtime) / 60 ))
    if (( age_min > STALE_LOCK_MINUTES )); then
        echo "STALE LOCK (${age_min}m) — alerting and proceeding" >> "$STDERR_LOG"
        rmdir "$LOCKDIR" 2>/dev/null || rm -rf "$LOCKDIR"
        mkdir "$LOCKDIR" 2>/dev/null && return 0
    fi
    echo "previous run still active"
    return 1
}

if ! acquire_lock; then
    exit 0
fi

LOG="$PROJECT_DIR/logs/cron_autotrader_$(date +%Y-%m-%d).log"
mkdir -p "$PROJECT_DIR/logs"
echo "=== cron_autotrader.sh $(date) ===" >> "$LOG"

# Activate venv
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "ERROR: venv not found at $PROJECT_DIR/.venv/bin/activate" >&2
    exit 1
fi

echo "  which python: $(which python)" >> "$LOG"
echo "  python version: $(python --version 2>&1)" >> "$LOG"

export TRACKJ_SKIP_HF_SYNC=1

# Pushover credentials from .env (if present)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.env"
    set +a
fi
echo "  .env loaded: $(test -f "$PROJECT_DIR/.env" && echo yes || echo no)" >> "$LOG"

send_pushover() {
    local title="$1"
    local message="$2"
    local user_key="${PUSHOVER_USER_KEY:-${PUSHOVER_USER:-}}"
    local api_token="${PUSHOVER_API_TOKEN:-${PUSHOVER_TOKEN:-}}"
    if [[ -z "$user_key" || -z "$api_token" ]]; then
        return 0
    fi
    curl -s --max-time 15 \
        --form-string "token=${api_token}" \
        --form-string "user=${user_key}" \
        --form-string "title=${title}" \
        --form-string "message=${message}" \
        https://api.pushover.net/1/messages.json >/dev/null || true
}

MODE="${1:-paper}"
DATE=$(date +%Y-%m-%d)
LOG="$PROJECT_DIR/logs/cron_autotrader_${DATE}.log"

BANKROLL_FILE="$PROJECT_DIR/logs/current_bankroll.txt"
if [[ -f "$BANKROLL_FILE" ]]; then
    BANKROLL=$(tr -d '[:space:]' < "$BANKROLL_FILE")
else
    BANKROLL="86.63"
    echo "$BANKROLL" > "$BANKROLL_FILE"
fi

{
    echo "=== cron_autotrader.sh $(date) ==="
    echo "Mode: $MODE | Bankroll: $BANKROLL"
} >> "$LOG"

# Pre-flight: too close to MCP elimination threshold
if awk -v b="$BANKROLL" 'BEGIN { exit !(b < 72) }'; then
    msg="Bankroll \$${BANKROLL} below \$72 floor — skipping trading today."
    echo "WARNING: $msg" >> "$LOG"
    send_pushover "Autotrader skipped" "$msg"
    exit 0
fi

set +e
python "$PROJECT_DIR/scripts/auto_trader_poly.py" \
    --mode "$MODE" \
    --strategy ngboost \
    --bankroll "$BANKROLL" \
    >> "$LOG" 2>&1
EXIT_CODE=$?
set -e

echo "Exit code: $EXIT_CODE" >> "$LOG"

if [[ "$EXIT_CODE" -ne 0 ]]; then
    send_pushover "Autotrader error ($MODE)" "cron_autotrader exited with code ${EXIT_CODE}. See ${LOG}"
fi

echo "=== completed $(date) ===" >> "$LOG"

exit "$EXIT_CODE"
