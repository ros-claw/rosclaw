#!/usr/bin/env bash
# Doctor for the seekdb 1.4 server with identity separation (PR-SDB-140-5,
# P0-4 + P1-7).  READY requires ALL of:
#   PROCESS_OK / IDENTITY_OK / SQL_OK / VERSION_OK / DATA_DIR_OK / BIND_OK
# An ambiguous instance (two engines sharing the port) is NEVER READY.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

FAIL=0
check() {  # check <name> <ok> <detail>
    if [ "$2" = "0" ]; then echo "PASS  $1  $3"; else echo "FAIL  $1  $3"; FAIL=1; fi
}

# --- DATA_DIR_OK ------------------------------------------------------------
STAMP="$SEEKDB_DATA_DIR/.rosclaw_seekdb_1_4"
if [ -f "$STAMP" ]; then
    check "DATA_DIR_OK" 0 "$SEEKDB_DATA_DIR (1.4 stamp present)"
elif [ ! -d "$SEEKDB_DATA_DIR" ] || [ -z "$(ls -A "$SEEKDB_DATA_DIR" 2>/dev/null)" ]; then
    check "DATA_DIR_OK" 0 "$SEEKDB_DATA_DIR (empty/fresh)"
else
    check "DATA_DIR_OK" 1 "$SEEKDB_DATA_DIR non-empty WITHOUT 1.4 stamp (1.3 dir?)"
fi

# --- PROCESS_OK + IDENTITY_OK ------------------------------------------------
COLLISIONS="$(seekdb_port_collision_count "$SEEKDB_PORT")"
PID=""
if [ -f "$SEEKDB_RUNTIME_JSON" ]; then
    RP_PID="$(seekdb_runtime_read pid || true)"
    RP_PSTART="$(seekdb_runtime_read process_start_time || true)"
    if [ -n "${RP_PID:-}" ] && kill -0 "$RP_PID" 2>/dev/null \
        && [ "$(seekdb_pid_starttime "$RP_PID")" = "${RP_PSTART:-mismatch}" ] \
        && seekdb_pid_identity_ok "$RP_PID" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
        PID="$RP_PID"
    fi
fi
if [ -z "$PID" ] && [ "$COLLISIONS" = "1" ]; then
    CAND="$(seekdb_pids_for_port "$SEEKDB_PORT")"
    if seekdb_pid_identity_ok "$CAND" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT"; then
        PID="$CAND"
    fi
fi

if [ -n "$PID" ]; then
    check "PROCESS_OK" 0 "pid $PID alive"
else
    check "PROCESS_OK" 1 "no live engine process for this instance"
fi

if [ "$COLLISIONS" -gt 1 ]; then
    check "IDENTITY_OK" 1 "AMBIGUOUS_INSTANCE: $COLLISIONS seekdb processes claim port $SEEKDB_PORT"
elif [ -n "$PID" ]; then
    check "IDENTITY_OK" 0 "pid $PID cmdline matches binary+base-dir+data-dir+port"
else
    check "IDENTITY_OK" 1 "no identity-verified engine"
fi

# --- BIND_OK (P1-7: no silent wide bind) -------------------------------------
BIND="$(seekdb_port_bind_addr "$SEEKDB_PORT")"
case "$BIND" in
    closed) check "BIND_OK" 0 "port $SEEKDB_PORT closed" ;;
    127.0.0.1|::1|'::1,127.0.0.1'|'127.0.0.1,::1')
        check "BIND_OK" 0 "loopback only ($BIND)" ;;
    *)
        if [ "${ROSCLAW_SEEKDB_ALLOW_WIDE_BIND:-}" = "1" ]; then
            check "BIND_OK" 0 "wide bind ($BIND) explicitly allowed via ROSCLAW_SEEKDB_ALLOW_WIDE_BIND"
        else
            check "BIND_OK" 1 "port $SEEKDB_PORT listens on $BIND — wide bind exposes the DB; set ROSCLAW_SEEKDB_ALLOW_WIDE_BIND=1 only if intended"
        fi
        ;;
esac

# --- SQL_OK + VERSION_OK ------------------------------------------------------
PYBIN="python3"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
[ -x "$REPO_ROOT/.venv/bin/python" ] && PYBIN="$REPO_ROOT/.venv/bin/python"

"$PYBIN" - "$SEEKDB_PORT" "$SEEKDB_VERSION_MARK" <<'PY'
import sys
port, mark = int(sys.argv[1]), sys.argv[2]
try:
    from rosclaw.memory.seekdb_client import SeekDBSQLStore
except ImportError:
    SeekDBSQLStore = None
import pymysql

try:
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root", password="",
                           connect_timeout=5, read_timeout=5, write_timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    cur.close(); conn.close()
except Exception as exc:
    print(f"FAIL  SQL_OK  {type(exc).__name__}: {exc}")
    print("FAIL  VERSION_OK  (no connection)")
    sys.exit(1)

print("PASS  SQL_OK  connect + SELECT ok")
if mark in version:
    print(f"PASS  VERSION_OK  {version}")
else:
    print(f"FAIL  VERSION_OK  {version!r} lacks {mark!r}")
    sys.exit(2)

# round-trip through rosclaw's own store when available
if SeekDBSQLStore is not None:
    try:
        store = SeekDBSQLStore(f"mysql://root@127.0.0.1:{port}/rosclaw_doctor")
        store.connect()
        store.insert("memory_items", {"id": "doctor_p1", "memory_type": "episode", "robot_id": "doctor"})
        rows = store.query("memory_items", {"id": "doctor_p1"})
        store.delete("memory_items", "doctor_p1")
        store.disconnect()
        print(f"PASS  SQL round-trip (rosclaw store)  rows={len(rows)}")
    except Exception as exc:
        print(f"FAIL  SQL round-trip (rosclaw store)  {type(exc).__name__}: {exc}")
        sys.exit(1)
PY
check "SQL_OK+VERSION_OK" $? ""

echo
if [ "$FAIL" = "0" ]; then echo "DOCTOR: READY (identity verified)"; else echo "DOCTOR: NOT READY"; exit 1; fi
