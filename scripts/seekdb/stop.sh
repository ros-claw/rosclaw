#!/usr/bin/env bash
# Stop the seekdb 1.4 server started by start_1_4.sh (graceful first).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

if [ ! -f "$SEEKDB_PID_FILE" ]; then
    echo "no pid file at $SEEKDB_PID_FILE — not running?"
    exit 0
fi
PID="$(cat "$SEEKDB_PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "stale pid file (pid $PID gone)"
    rm -f "$SEEKDB_PID_FILE"
    exit 0
fi
echo "stopping seekdb pid $PID (SIGTERM)"
kill "$PID"
for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
    echo "still alive after 30s — SIGKILL" >&2
    kill -9 "$PID"
fi
rm -f "$SEEKDB_PID_FILE"
echo "stopped"
