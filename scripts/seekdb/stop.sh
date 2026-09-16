#!/usr/bin/env bash
# Stop the seekdb 1.4 server — identity-verified, fail-closed (PR-SDB-140-5,
# P0-3).  Never guesses a pid: an ambiguous or unknown instance is REFUSED,
# not killed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

# --- resolve the candidate set on our port ---------------------------------
mapfile -t CANDIDATES < <(seekdb_pids_for_port "$SEEKDB_PORT")
if [ "${#CANDIDATES[@]}" -gt 1 ]; then
    echo "AMBIGUOUS_INSTANCE: ${#CANDIDATES[@]} seekdb processes claim port $SEEKDB_PORT" >&2
    echo "  pids: ${CANDIDATES[*]} — refusing to guess; resolve manually" >&2
    exit 7
fi

PID=""
if [ -f "$SEEKDB_RUNTIME_JSON" ]; then
    RP_PID="$(seekdb_runtime_read pid || true)"
    RP_PSTART="$(seekdb_runtime_read process_start_time || true)"
    if [ -n "${RP_PID:-}" ]; then
        if ! kill -0 "$RP_PID" 2>/dev/null; then
            echo "runtime.json pid $RP_PID is dead — stale runtime state"
            rm -f "$SEEKDB_RUNTIME_JSON" "$SEEKDB_PID_FILE"
            [ "${#CANDIDATES[@]}" -eq 0 ] && exit 0
        else
            # PID-reuse + identity guard: start time AND full cmdline must match
            if [ -n "${RP_PSTART:-}" ] && [ "$(seekdb_pid_starttime "$RP_PID")" != "$RP_PSTART" ]; then
                echo "REFUSE_TO_STOP: pid $RP_PID was REUSED (start time changed);" >&2
                echo "  the recorded engine is gone and a different process holds the pid." >&2
                exit 8
            fi
            if ! seekdb_pid_identity_ok "$RP_PID" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
                echo "REFUSE_TO_STOP: pid $RP_PID cmdline no longer matches our instance" >&2
                exit 8
            fi
            PID="$RP_PID"
        fi
    fi
fi

if [ -z "$PID" ]; then
    # No usable runtime.json: exactly one identity-matching candidate may be
    # adopted (recovered identity); anything else refuses.
    if [ "${#CANDIDATES[@]}" -eq 0 ]; then
        echo "seekdb not running on port $SEEKDB_PORT"
        rm -f "$SEEKDB_PID_FILE" "$SEEKDB_RUNTIME_JSON"
        exit 0
    fi
    if seekdb_pid_identity_ok "${CANDIDATES[0]}" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
        PID="${CANDIDATES[0]}"
        echo "adopting identity-verified engine pid $PID (runtime.json was absent)"
    else
        echo "REFUSE_TO_STOP: port $SEEKDB_PORT is held by an UNKNOWN seekdb" >&2
        echo "  (pid ${CANDIDATES[0]}, cmdline does not match this SEEKDB_HOME)" >&2
        exit 7
    fi
fi

echo "stopping seekdb pid $PID (identity verified, SIGTERM)"
kill "$PID"
for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
    echo "still alive after 30s — SIGKILL (pid identity re-verified)" >&2
    if seekdb_pid_identity_ok "$PID" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
        kill -9 "$PID"
    else
        echo "REFUSED: pid $PID identity changed during shutdown — not killing" >&2
        exit 8
    fi
fi
rm -f "$SEEKDB_PID_FILE" "$SEEKDB_RUNTIME_JSON"
echo "stopped"
