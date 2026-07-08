#!/bin/bash
# Polymarket autotrader watchdog (settle or trader path).
# Usage: watchdog.sh settle|trader

set -euo pipefail

PROJECT_DIR="/Users/oscaro/Desktop/MCP_Project"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$PROJECT_DIR"

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
        echo "Pushover not configured: $title — $message"
        return 0
    fi
    curl -s \
        --form-string "token=${api_token}" \
        --form-string "user=${user_key}" \
        --form-string "title=${title}" \
        --form-string "message=${message}" \
        https://api.pushover.net/1/messages.json >/dev/null || true
}

MODE="${1:-}"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$PROJECT_DIR/logs"
STATE_FILE="$LOG_DIR/auto_trader_state_${DATE}.json"
SETTLE_LOG="$LOG_DIR/cron_settle.log"
CRON_LOG="$LOG_DIR/cron_stdout.log"

case "$MODE" in
    settle)
        if [[ -f "$SETTLE_LOG" ]] && grep -q "$DATE" "$SETTLE_LOG"; then
            echo "Settle watchdog OK: settlement log mentions $DATE"
            exit 0
        fi
        if [[ -f "$LOG_DIR/current_bankroll.txt" ]]; then
            mtime=$(stat -f "%Sm" -t "%Y-%m-%d" "$LOG_DIR/current_bankroll.txt" 2>/dev/null || true)
            if [[ "$mtime" == "$DATE" ]]; then
                echo "Settle watchdog OK: bankroll file updated today"
                exit 0
            fi
        fi
        msg="Settlement watchdog: no evidence of settlement for $DATE."
        echo "ALERT: $msg"
        send_pushover "Settlement watchdog" "$msg"
        exit 1
        ;;
    trader)
        if [[ -f "$STATE_FILE" ]]; then
            echo "Trader watchdog OK: state file exists ($STATE_FILE)"
            exit 0
        fi
        if [[ -f "$CRON_LOG" ]] && grep -q "AUTO-TRADER INITIALIZE: $DATE" "$CRON_LOG"; then
            echo "Trader watchdog OK: initialize logged for $DATE"
            exit 0
        fi
        msg="Trade watchdog: no autotrader run recorded for $DATE (missing state file)."
        echo "ALERT: $msg"
        send_pushover "Trade watchdog" "$msg"
        exit 1
        ;;
    *)
        echo "Usage: $0 settle|trader" >&2
        exit 2
        ;;
esac
