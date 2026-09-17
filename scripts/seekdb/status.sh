#!/usr/bin/env bash
# seekdb 1.4 server status with instance identity (PR-SDB-140-5, P0-4).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

PYBIN="python3"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
[ -x "$REPO_ROOT/.venv/bin/python" ] && PYBIN="$REPO_ROOT/.venv/bin/python"
SDK_VER="$("$PYBIN" -c 'import importlib.metadata as m; print(m.version("pyseekdb"))' 2>/dev/null || echo "unknown")"

echo "seekdb $SEEKDB_ENGINE_VERSION server status"
echo "mode:      server"
echo "engine:    $SEEKDB_ENGINE_VERSION (expect mark $SEEKDB_VERSION_MARK)"
echo "sdk:       pyseekdb $SDK_VER"
echo "base dir:  $SEEKDB_HOME"
echo "data dir:  $SEEKDB_DATA_DIR"
echo "port:      $SEEKDB_PORT (bind: $(seekdb_port_bind_addr "$SEEKDB_PORT"))"

COLLISIONS="$(seekdb_port_collision_count "$SEEKDB_PORT")"
echo "collisions: $COLLISIONS seekdb process(es) claim this port"

PID=""
IDENTITY="ABSENT"
if [ -f "$SEEKDB_RUNTIME_JSON" ]; then
    RP_PID="$(seekdb_runtime_read pid || true)"
    RP_PSTART="$(seekdb_runtime_read process_start_time || true)"
    if [ -n "${RP_PID:-}" ] && kill -0 "$RP_PID" 2>/dev/null; then
        if [ -n "${RP_PSTART:-}" ] && [ "$(seekdb_pid_starttime "$RP_PID")" = "$RP_PSTART" ] \
            && seekdb_pid_identity_ok "$RP_PID" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
            PID="$RP_PID"
            IDENTITY="VERIFIED"
        else
            IDENTITY="MISMATCH (pid $RP_PID failed identity checks)"
        fi
    else
        IDENTITY="STALE (runtime.json pid ${RP_PID:-?} not alive)"
    fi
fi
if [ -z "$PID" ] && [ "$COLLISIONS" = "1" ]; then
    CAND="$(seekdb_pids_for_port "$SEEKDB_PORT")"
    if seekdb_pid_identity_ok "$CAND" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
        PID="$CAND"
        IDENTITY="VERIFIED (adopted from live process; no runtime.json)"
    fi
fi
[ "$COLLISIONS" -gt 1 ] && IDENTITY="AMBIGUOUS_INSTANCE"

echo "pid:       ${PID:-none}"
echo "identity:  $IDENTITY"
if [ -n "$PID" ]; then
    RSS_KB="$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null || echo '?')"
    THREADS="$(ls "/proc/$PID/task" 2>/dev/null | wc -l || echo '?')"
    echo "rss:       $((RSS_KB / 1024)) MiB"
    echo "threads:   $THREADS"
fi
