#!/bin/bash
# ROSClaw v1.0 Progress Monitor
# Run this every 2 hours to check all sessions

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
REPORT_FILE="/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0/docs/release/v1.0/progress_reports/progress_${TIMESTAMP// /_}.md"

mkdir -p "$(dirname "$REPORT_FILE")"

echo "# Progress Report - $TIMESTAMP" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

SESSIONS=("rosclaw" "rosclaw_qwen" "know" "how" "memory" "practice" "dashboard" "provider" "sandbox" "swarm" "hermes")

for session in "${SESSIONS[@]}"; do
    echo "## $session" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    tmux capture-pane -t "$session" -p -S -50 2>/dev/null | tail -20 >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
done

echo "Report saved to: $REPORT_FILE"
