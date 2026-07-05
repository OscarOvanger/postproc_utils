#!/bin/bash
# Polymarket v5 autotrader cron wrapper (paper or live).
# Usage: cron_autotrader.sh [paper|live]
set -euo pipefail

PROJECT_DIR="${HOME}/Desktop/MCP_Project"
cd "$PROJECT_DIR"

mkdir -p logs

# Activate venv
# shellcheck source=/dev/null
source "$PROJECT_DIR/.venv/bin/activate"

export TRACKJ_SKIP_HF_SYNC=1

# Pushover credentials from .env (if present)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.env"
    set +a
fi

send_pushover() {
    local title="$1"
    local message="$2"
    local user_key="${PUSHOVER_USER_KEY:-${PUSHOVER_USER:-}}"
    local api_token="${PUSHOVER_API_TOKEN:-${PUSHOVER_TOKEN:-}}"
    if [[ -z "$user_key" || -z "$api_token" ]]; then
        return 0
    fi
    curl -s \
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

# Optional VPN check for live mode (uncomment when ready)
# if [[ "$MODE" == "live" ]]; then
#     if ! curl -s --max-time 5 https://api.ipify.org | grep -qv "^$"; then
#         echo "VPN check failed" >> "$LOG"
#         send_pushover "Autotrader VPN check failed" "Live trading skipped — check VPN."
#         exit 1
#     fi
# fi

set +e
python "$PROJECT_DIR/scripts/auto_trader_poly.py" \
    --mode "$MODE" \
    --strategy trackb \
    --bankroll "$BANKROLL" \
    >> "$LOG" 2>&1
EXIT_CODE=$?
set -e

echo "Exit code: $EXIT_CODE" >> "$LOG"

if [[ "$EXIT_CODE" -ne 0 ]]; then
    send_pushover "Autotrader error ($MODE)" "cron_autotrader exited with code ${EXIT_CODE}. See ${LOG}"
fi

exit "$EXIT_CODE"
