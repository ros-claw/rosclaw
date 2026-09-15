#!/usr/bin/env bash
# seekdb 1.4 server status: pid, port, RSS, engine version (if reachable).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

echo "seekdb $SEEKDB_ENGINE_VERSION server status"
echo "home:    $SEEKDB_HOME"
echo "data:    $SEEKDB_DATA_DIR"
echo "port:    $SEEKDB_PORT"

PID=""
if [ -f "$SEEKDB_PID_FILE" ] && kill -0 "$(cat "$SEEKDB_PID_FILE")" 2>/dev/null; then
    PID="$(cat "$SEEKDB_PID_FILE")"
else
    # the engine re-execs under its own wrapper — the pid file can point at
    # the dead launcher; fall back to the process actually holding our port
    PID="$(pgrep -f "seekdb.*--port.*$SEEKDB_PORT" | head -1 || true)"
fi
if [ -n "$PID" ]; then
    RSS_KB="$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null || echo '?')"
    THREADS="$(ls "/proc/$PID/task" 2>/dev/null | wc -l || echo '?')"
    echo "pid:     $PID (alive)"
    echo "rss:     $((RSS_KB / 1024)) MiB"
    echo "threads: $THREADS"
else
    echo "pid:     not running"
fi

if (echo >"/dev/tcp/127.0.0.1/$SEEKDB_PORT") 2>/dev/null; then
    echo "sql:     port $SEEKDB_PORT OPEN"
else
    echo "sql:     port $SEEKDB_PORT closed"
fi
