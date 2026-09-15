#!/usr/bin/env bash
# Logical migration: old SeekDB 1.3 store -> fresh 1.4 server (PR-SDB-140-2).
#
#   old 1.3 (any rosclaw backend)
#     -> dump (JSONL + deterministic checksums)
#     -> restore into the fresh 1.4 instance
#     -> parity validation (table set / counts / id sets / checksums)
#
# Cutover (pointing rosclaw.yaml at the new instance) is a deliberate
# operator step AFTER parity passes — this script never edits config.
#
# Usage:
#   migrate_1_3_to_1_4.sh --from sqlite:/path/to/knowledge.sqlite   # or embedded:/path | mysql://... \
#                         --to mysql://root@127.0.0.1:2881/rosclaw
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM_URL=""
TO_URL=""
WORK="$(mktemp -d /tmp/seekdb-migrate-XXXXXX)"
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM_URL="$2"; shift 2 ;;
        --to) TO_URL="$2"; shift 2 ;;
        --workdir) WORK="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$FROM_URL" ] || [ -z "$TO_URL" ]; then
    echo "usage: migrate_1_3_to_1_4.sh --from <store-url> --to <store-url> [--workdir DIR]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PYBIN="python3"
[ -x "$REPO_ROOT/.venv/bin/python" ] && PYBIN="$REPO_ROOT/.venv/bin/python"

echo "migration workdir: $WORK"
echo "from: $FROM_URL"
echo "to:   $TO_URL"

set +e
PYTHONPATH="$REPO_ROOT/src" "$PYBIN" - "$FROM_URL" "$TO_URL" "$WORK" <<'PY'
import os
import sys
from rosclaw.storage.migrate import StoreMigrator, MANIFEST
from rosclaw.storage.factory import StoreFactory
import json
from pathlib import Path

from_url, to_url, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])

def open_store(url: str):
    if url.startswith("sqlite:"):
        from rosclaw.memory.seekdb_client import SQLiteStructuredStore
        s = SQLiteStructuredStore(url[len("sqlite:"):])
    elif url.startswith("embedded:"):
        # legacy pylibseekdb embedded dir (1.3-era); database via #fragment
        from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore
        spec = url[len("embedded:"):]
        path, _, db = spec.partition("#")
        s = SeekDBEmbeddedRetrievalStore(path=path, database=db or "rosclaw")
    elif url.startswith("server:"):
        # seekdb server retrieval store (pyseekdb over MySQL protocol) —
        # the right target for retrieval-side collections (raw records
        # with embeddings, no SQL NOT NULL constraints).
        from urllib.parse import urlparse

        from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

        spec = url[len("server:"):]
        hostport, _, db = spec.partition("#")
        host, _, port = hostport.partition(":")
        s = SeekDBServerRetrievalStore(host=host, port=int(port or 2881), database=db or "rosclaw")
    else:
        from rosclaw.memory.seekdb_client import SeekDBSQLStore
        s = SeekDBSQLStore(url)
    s.connect()
    return s

src = open_store(from_url)
manifest = StoreMigrator(src).dump(work / "dump")
src.disconnect()
print(f"[1/3] dump: {sum(t['rows'] for t in manifest['tables'].values())} rows, "
      f"{len(manifest['tables'])} tables")

dst = open_store(to_url)
restored = StoreMigrator(dst).restore(work / "dump")
print(f"[2/3] restore: {sum(restored.values())} rows")

report = StoreMigrator.parity(json.loads((work / 'dump' / MANIFEST).read_text()), dst)
dst.disconnect()
ok = report["ok"]
bad = {t: r for t, r in report["tables"].items() if not r["ok"]}
print(f"[3/3] parity: {'PASS' if ok else 'FAIL'}"
      + ("" if ok else f" — mismatches: {list(bad)[:5]}"))
(work / "parity_verdict.json").write_text(json.dumps({"ok": bool(ok)}, indent=2))

# pylibseekdb's C++ atexit shutdown OVERRIDES the process exit code to 0
# (verified 2026-09-15: sys.exit(1) after an open/closed embedded store
# exits 0).  os._exit bypasses atexit — the migration verdict must survive.
verdict = 0 if ok else 1
print(f"[3/3] parity verdict={'PASS' if ok else 'FAIL'} exit={verdict}", flush=True)
os._exit(verdict)
PY
RC=$?
set -e
if [ "$RC" != "0" ]; then
    echo "MIGRATION FAILED (python exit $RC) — parity did not pass; NOT safe to cut over" >&2
    exit "$RC"
fi

echo "migration complete.  Cutover is your deliberate next step:"
echo "  set runtime.seekdb_url in rosclaw.yaml to $TO_URL"
