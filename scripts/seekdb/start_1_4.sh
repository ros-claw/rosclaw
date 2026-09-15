#!/usr/bin/env bash
# Start seekdb Engine 1.4.0 in server mode on a FRESH data directory.
#
# PR-SDB-140-2.  Hard rule (outline §七/八): 1.4 must never open a 1.3
# data dir — this script refuses to start over one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

SEEKDB_BIN="${SEEKDB_BIN:-$(command -v seekdb || echo /usr/bin/seekdb)}"
if [ ! -x "$SEEKDB_BIN" ]; then
    echo "seekdb binary not found at $SEEKDB_BIN — run install_1_4.sh first" >&2
    exit 1
fi

# Refuse to start over a 1.3 data directory: the 1.3 layout carries a
# pylibseekdb marker (embedded) or an older data_version stamp.  If the
# dir exists and is non-empty but lacks our 1.4 stamp, refuse.
STAMP="$SEEKDB_DATA_DIR/.rosclaw_seekdb_1_4"
if [ -d "$SEEKDB_DATA_DIR" ] && [ -n "$(ls -A "$SEEKDB_DATA_DIR" 2>/dev/null)" ] && [ ! -f "$STAMP" ]; then
    echo "REFUSING: $SEEKDB_DATA_DIR is non-empty and has no 1.4 stamp." >&2
    echo "1.3 -> 1.4 is NOT in-place upgradeable (upstream).  Migrate instead:" >&2
    echo "  scripts/seekdb/migrate_1_3_to_1_4.sh <old-data-or-dump>" >&2
    exit 4
fi

mkdir -p "$SEEKDB_DATA_DIR" "$SEEKDB_LOG_DIR"

if [ -f "$SEEKDB_PID_FILE" ] && kill -0 "$(cat "$SEEKDB_PID_FILE")" 2>/dev/null; then
    echo "seekdb already running (pid $(cat "$SEEKDB_PID_FILE"))"
    exit 0
fi

echo "starting seekdb $SEEKDB_ENGINE_VERSION on port $SEEKDB_PORT (data: $SEEKDB_DATA_DIR)"
nohup "$SEEKDB_BIN" \
    --base-dir "$SEEKDB_HOME" \
    --data-dir "$SEEKDB_DATA_DIR" \
    --redo-dir "$SEEKDB_DATA_DIR/redo" \
    --port "$SEEKDB_PORT" \
    >>"$SEEKDB_LOG_DIR/seekdb.log" 2>&1 &
echo $! >"$SEEKDB_PID_FILE"
touch "$STAMP"

# readiness probe: obshell/agent handshake, then SQL port
for _ in $(seq 1 60); do
    if (echo >"/dev/tcp/127.0.0.1/$SEEKDB_PORT") 2>/dev/null; then
        echo "seekdb READY on 127.0.0.1:$SEEKDB_PORT (pid $(cat "$SEEKDB_PID_FILE"))"
        exit 0
    fi
    sleep 2
done
echo "seekdb failed to become ready in 120s — see $SEEKDB_LOG_DIR/seekdb.log" >&2
exit 5
