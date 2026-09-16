#!/usr/bin/env bash
# Release soak campaign (PR-SDB-140-5, P1-2/P1-3/P1-4), fully scripted:
#   fresh restart (identity-verified) -> sampler T0 -> idle 5min ->
#   35k medium workload (load+workload phases) -> 1w4r 60min -> 2w8r 60min
#   -> idle 10min -> idle 30min -> end.
# Writes phase markers for the sampler; all artifacts under $1 (workdir).
set -uo pipefail

WORK="${1:?workdir required}"
URL="mysql://root@127.0.0.1:2881/soak140"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="$REPO/.venv/bin/python"
PHASE_FILE="$WORK/phase.txt"
mkdir -p "$WORK"

phase() { echo "$1" > "$PHASE_FILE"; echo "[$(date -u +%H:%M:%S)] phase -> $1"; }

SEEKDB_HOME="${SEEKDB_HOME:-/tmp/seekdb140/home}"
export SEEKDB_HOME SEEKDB_PORT=2881
export SEEKDB_DATA_DIR="$SEEKDB_HOME/data" SEEKDB_PID_FILE="$SEEKDB_HOME/seekdb.pid"
export SEEKDB_LOG_DIR="$SEEKDB_HOME/log" SEEKDB_RUNTIME_JSON="$SEEKDB_HOME/runtime.json"

phase fresh_restart
bash "$REPO/scripts/seekdb/stop.sh" || true
bash "$REPO/scripts/seekdb/start_1_4.sh" || { echo "start FAILED"; exit 1; }

ENGINE_PID="$(python3 -c "import json; print(json.load(open('$SEEKDB_RUNTIME_JSON'))['pid'])")"
echo "engine pid: $ENGINE_PID"

# sampler covers the whole campaign: idle5 + load + workload + 2x60min soak
# + idle 40min  => ~10800s + margin
phase idle_5min
"$PY" "$REPO/validation/seekdb/scripts/resource_sampler.py" \
    --pid "$ENGINE_PID" --interval-s 15 --duration-s 13000 \
    --phase-file "$PHASE_FILE" --out "$WORK/sampler.jsonl" &
SAMPLER=$!

sleep 300   # T1: idle 5 min

phase load_35k
"$PY" "$REPO/validation/seekdb/scripts/medium_workload.py" \
    --seekdb-url "$URL" --report "$WORK/medium_workload.json" \
    > "$WORK/medium_workload.log" 2>&1
MW=$?
echo "medium workload exit=$MW"

phase soak_1w4r
"$PY" "$REPO/validation/seekdb/scripts/concurrency_matrix.py" \
    --seekdb-url "$URL" --lanes 1w4r --duration-s 3600 --period-s 0.05 \
    --report "$WORK/soak_1w4r.json" > "$WORK/soak_1w4r.log" 2>&1
echo "soak 1w4r exit=$?"

phase soak_2w8r
"$PY" "$REPO/validation/seekdb/scripts/concurrency_matrix.py" \
    --seekdb-url "$URL" --lanes 2w8r --duration-s 3600 --period-s 0.05 \
    --report "$WORK/soak_2w8r.json" > "$WORK/soak_2w8r.log" 2>&1
echo "soak 2w8r exit=$?"

phase idle_10min
sleep 600
phase idle_30min
sleep 1800
phase end

kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null
echo "CAMPAIGN DONE (medium=$MW)"
