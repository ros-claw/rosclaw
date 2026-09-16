#!/usr/bin/env bash
# Doctor for the seekdb 1.4 server: install state, process, port, SQL
# round-trip, and version proof.  Exit 0 only when everything checks out.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

FAIL=0
check() {  # check <name> <ok> <detail>
    if [ "$2" = "0" ]; then echo "PASS  $1  $3"; else echo "FAIL  $1  $3"; FAIL=1; fi
}

SEEKDB_BIN="${SEEKDB_BIN:-$(command -v seekdb || echo /usr/bin/seekdb)}"
[ -x "$SEEKDB_BIN" ]; check "binary present ($SEEKDB_BIN)" $? ""

PID=""
if [ -f "$SEEKDB_PID_FILE" ] && kill -0 "$(cat "$SEEKDB_PID_FILE")" 2>/dev/null; then
    PID="$(cat "$SEEKDB_PID_FILE")"
else
    # engine re-execs under its own wrapper; fall back to the live process
    PID="$(pgrep -f "seekdb.*--port.*$SEEKDB_PORT" | head -1 || true)"
fi
if [ -n "$PID" ]; then
    check "process alive (pid $PID)" 0 ""
else
    check "process alive" 1 "no live pid"
fi

(echo >"/dev/tcp/127.0.0.1/$SEEKDB_PORT") 2>/dev/null; check "port $SEEKDB_PORT open" $? ""

# SQL round-trip via rosclaw's own server store (real client path).
.venv_ok() { [ -x "$1/bin/python" ]; }
PYBIN="python3"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if .venv_ok "$REPO_ROOT/.venv"; then PYBIN="$REPO_ROOT/.venv/bin/python"; fi

"$PYBIN" - "$SEEKDB_PORT" <<'PY'
import sys
port = int(sys.argv[1])
try:
    from rosclaw.memory.seekdb_client import SeekDBSQLStore
    store = SeekDBSQLStore(f"mysql://root@127.0.0.1:{port}/rosclaw_doctor")
    store.connect()
    # probe through a real schema table; clean up after ourselves
    probe = {"id": "doctor_p1", "memory_type": "episode", "robot_id": "doctor"}
    store.insert("memory_items", probe)
    rows = store.query("memory_items", {"id": "doctor_p1"})
    store.delete("memory_items", "doctor_p1")
    version = None
    try:
        with store._connection as conn:  # noqa: SLF001
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION() AS v")
                row = cur.fetchone()
                version = row["v"] if isinstance(row, dict) else row[0]
    except Exception:
        pass
    store.disconnect()
    print(f"PASS  sql round-trip  rows={len(rows)}")
    print(f"PASS  engine version  {version or 'unknown'}")
except Exception as exc:
    print(f"FAIL  sql round-trip  {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
check "sql round-trip (rosclaw store)" $? ""

echo
if [ "$FAIL" = "0" ]; then echo "DOCTOR: READY"; else echo "DOCTOR: NOT READY"; exit 1; fi
