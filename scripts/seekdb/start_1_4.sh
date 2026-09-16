#!/usr/bin/env bash
# Start seekdb Engine 1.4.0 in server mode on a FRESH (or stamped 1.4) data
# directory, with full instance identity verification (PR-SDB-140-5, P0-2).
#
# Hard rules:
#   * 1.4 must never open a 1.3 data dir (outline §七/八).
#   * "port answers" is NOT readiness: the engine re-execs (pid file can
#     point at a dead launcher) and seekdb binds SO_REUSEPORT (two engines
#     can share one port).  READY requires: OUR engine pid verified via
#     /proc cmdline (binary+base-dir+data-dir+port), SELECT VERSION()
#     containing the 1.4.0 mark, and a real SQL round-trip.
#   * The 1.4 stamp is the RESULT of a verified start, never part of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

if [ ! -x "$SEEKDB_BIN" ]; then
    echo "seekdb binary not found at $SEEKDB_BIN — run install_1_4.sh first" >&2
    exit 1
fi

# --- 1. data dir / no-in-place-upgrade gate --------------------------------
STAMP="$SEEKDB_DATA_DIR/.rosclaw_seekdb_1_4"
if [ -d "$SEEKDB_DATA_DIR" ] && [ -n "$(ls -A "$SEEKDB_DATA_DIR" 2>/dev/null)" ] && [ ! -f "$STAMP" ]; then
    echo "REFUSING: $SEEKDB_DATA_DIR is non-empty and has no 1.4 stamp." >&2
    echo "1.3 -> 1.4 is NOT in-place upgradeable (upstream).  Migrate instead:" >&2
    echo "  scripts/seekdb/migrate_1_3_to_1_4.sh <old-data-or-dump>" >&2
    exit 4
fi
mkdir -p "$SEEKDB_DATA_DIR" "$SEEKDB_LOG_DIR"

# --- 2. port occupancy gate -------------------------------------------------
# Whoever already claims our port must be provably OUR instance, else we do
# not start (a second engine would silently split reads/writes).
mapfile -t EXISTING < <(seekdb_pids_for_port "$SEEKDB_PORT")
if [ "${#EXISTING[@]}" -gt 1 ]; then
    echo "HARD FAIL: ${#EXISTING[@]} seekdb processes already claim port $SEEKDB_PORT" >&2
    echo "  pids: ${EXISTING[*]} — resolve the collision before starting" >&2
    exit 6
fi
if [ "${#EXISTING[@]}" -eq 1 ]; then
    if seekdb_pid_identity_ok "${EXISTING[0]}" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
        echo "seekdb already running (pid ${EXISTING[0]}, identity verified)"
        exit 0
    fi
    echo "HARD FAIL: port $SEEKDB_PORT is held by an UNKNOWN seekdb instance" >&2
    echo "  pid ${EXISTING[0]} cmdline does not match base=$SEEKDB_HOME data=$SEEKDB_DATA_DIR" >&2
    echo "  refusing to start a second engine over it" >&2
    exit 6
fi
# Non-seekdb listener on the port?
if (echo >"/dev/tcp/127.0.0.1/$SEEKDB_PORT") 2>/dev/null; then
    echo "HARD FAIL: port $SEEKDB_PORT is held by a non-seekdb process" >&2
    exit 6
fi

# --- 3. launch --------------------------------------------------------------
echo "starting seekdb $SEEKDB_ENGINE_VERSION on port $SEEKDB_PORT (data: $SEEKDB_DATA_DIR)"
nohup "$SEEKDB_BIN" \
    --base-dir "$SEEKDB_HOME" \
    --data-dir "$SEEKDB_DATA_DIR" \
    --redo-dir "$SEEKDB_DATA_DIR/redo" \
    --port "$SEEKDB_PORT" \
    >>"$SEEKDB_LOG_DIR/seekdb.log" 2>&1 &
LAUNCHER_PID=$!
echo "$LAUNCHER_PID" >"$SEEKDB_PID_FILE"

cleanup_failed_start() {
    # kill only processes PROVEN to be ours (identity-verified)
    local pid
    for pid in $(seekdb_pids_for_port "$SEEKDB_PORT") "$LAUNCHER_PID"; do
        if seekdb_pid_identity_ok "$pid" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        elif [ "$pid" = "$LAUNCHER_PID" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    rm -f "$SEEKDB_RUNTIME_JSON"
}

# --- 4. find the REAL engine pid (post re-exec) ------------------------------
# The launcher and the re-exec'd engine carry IDENTICAL cmdlines; only the
# daemonized engine has ppid 1.  Keep waiting for it specifically.
ENGINE_PID=""
for _ in $(seq 1 60); do
    ENGINE_PID="$(seekdb_find_engine_pid "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT" || true)"
    if [ -n "$ENGINE_PID" ] && [ "$(seekdb_pid_ppid "$ENGINE_PID")" = "1" ]; then
        break
    fi
    ENGINE_PID=""
    kill -0 "$LAUNCHER_PID" 2>/dev/null || break  # launcher died; see log
    sleep 2
done
if [ -z "$ENGINE_PID" ]; then
    echo "seekdb engine never appeared with our identity — see $SEEKDB_LOG_DIR/seekdb.log" >&2
    cleanup_failed_start
    exit 5
fi

# --- 5. SQL proof: version mark + round-trip --------------------------------
# Process-visible != SQL-ready: retry until the engine accepts queries.
PYBIN="python3"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
[ -x "$REPO_ROOT/.venv/bin/python" ] && PYBIN="$REPO_ROOT/.venv/bin/python"

SQL_OK=0
SQL_RC=0
for _ in $(seq 1 45); do
    "$PYBIN" - "$SEEKDB_PORT" "$SEEKDB_VERSION_MARK" <<'PY' && SQL_OK=1
import sys
port, mark = int(sys.argv[1]), sys.argv[2]
try:
    import pymysql
except ImportError:
    print("pymysql unavailable in probe interpreter", file=sys.stderr)
    sys.exit(1)
try:
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root", password="",
                           connect_timeout=5, read_timeout=5, write_timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    if mark not in version:
        print(f"version mark mismatch: {version!r} lacks {mark!r}", file=sys.stderr)
        sys.exit(2)
    cur.execute("CREATE DATABASE IF NOT EXISTS rosclaw_start_probe")
    cur.execute("USE rosclaw_start_probe")
    cur.execute("CREATE TABLE IF NOT EXISTS probe (id VARCHAR(64) PRIMARY KEY, v VARCHAR(64))")
    cur.execute("INSERT INTO probe (id, v) VALUES ('p', 'ok') ON DUPLICATE KEY UPDATE v='ok'")
    cur.execute("SELECT v FROM probe WHERE id='p'")
    assert cur.fetchone()[0] == "ok"
    cur.execute("DELETE FROM probe WHERE id='p'")
    conn.commit()
    cur.close(); conn.close()
    print(f"engine version verified: {version}")
except SystemExit:
    raise
except Exception as exc:
    print(f"SQL probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    SQL_RC=$?
    [ "$SQL_OK" = "1" ] && break
    # version-mark mismatch (exit 2) is terminal — wrong engine entirely
    [ "$SQL_RC" = "2" ] && break
    sleep 2
done
if [ "$SQL_OK" != "1" ]; then
    echo "seekdb started but failed SQL verification — NOT marking ready" >&2
    cleanup_failed_start
    exit 5
fi

# --- 6. identity record, THEN stamp, THEN READY ------------------------------
echo "$ENGINE_PID" >"$SEEKDB_PID_FILE"
PSTART="$(seekdb_pid_starttime "$ENGINE_PID")"
seekdb_runtime_write "$ENGINE_PID" "$PSTART" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$STAMP"
echo "seekdb READY on 127.0.0.1:$SEEKDB_PORT (pid $ENGINE_PID, identity verified)"
