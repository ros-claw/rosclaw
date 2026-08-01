#!/usr/bin/env bash
# Controlled docker fault-injection window (任务书 §二十三 + 任务D).
# Kills the embedding and cosmos containers one at a time, verifies
# fail-closed behaviour, restores, and verifies recovery. Every step is
# timestamped into the run's fault_injection log so soak availability can
# be split into normal vs fault windows (§24.2).
set -uo pipefail

REPORT_DIR="${1:?usage: inject_fault.sh <report_dir>}"
LOG="$REPORT_DIR/fault_injection/docker_faults.log"
mkdir -p "$REPORT_DIR/fault_injection"
# Operator provides the sudo credential out-of-band; never hard-code it.
SUDO_PW="${TY1200_SUDO_PW:?export TY1200_SUDO_PW for the fault window (operator credential)}"
SUDO="echo $SUDO_PW | sudo -S"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

embedding_health() {
  curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q Qwen && echo UP || echo DOWN
}
cosmos_health() {
  curl -s --max-time 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q Cosmos && echo UP || echo DOWN
}
embedding_infer() {
  curl -s --max-time 10 http://127.0.0.1:8000/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-embedding-0.6b","input":"probe"}' 2>/dev/null | head -c 60
}

log "=== fault window start ==="
log "pre: embedding=$(embedding_health) cosmos=$(cosmos_health)"

# --- fault 1: kill embedding (restart policy: no) ---
log "FAULT inject: docker kill vllm (embedding)"
eval "$SUDO docker kill vllm" >>"$LOG" 2>&1
sleep 5
state=$(embedding_health)
log "post-kill embedding health: $state (expect DOWN)"
probe=$(embedding_infer)
if [[ -z "$probe" ]]; then
  log "embedding inference during fault: refused/empty (fail closed, no fabricated success) -> PASS"
else
  log "embedding inference during fault returned: $probe -> CHECK"
fi
log "degradation check: wiki retrieval must fall back to BM25/keyword (see wiki notes)"

log "RESTORE: docker start vllm"
eval "$SUDO docker start vllm" >>"$LOG" 2>&1
for i in $(seq 1 36); do
  sleep 10
  if [[ "$(embedding_health)" == "UP" ]]; then break; fi
done
log "post-restore embedding health: $(embedding_health) after ~$((i*10))s"

# --- fault 2: kill cosmos (restart policy: unless-stopped) ---
log "FAULT inject: docker kill rosclaw-ty1200-cosmos"
eval "$SUDO docker kill rosclaw-ty1200-cosmos" >>"$LOG" 2>&1
sleep 5
log "post-kill cosmos health: $(cosmos_health) (expect DOWN)"
# unless-stopped should bring it back without manual start
for i in $(seq 1 36); do
  sleep 10
  if [[ "$(cosmos_health)" == "UP" ]]; then break; fi
done
log "auto-restart cosmos health: $(cosmos_health) after ~$((i*10))s"
if [[ "$(cosmos_health)" != "UP" ]]; then
  log "auto-restart did not recover; manual docker start"
  eval "$SUDO docker start rosclaw-ty1200-cosmos" >>"$LOG" 2>&1
  sleep 30
  log "manual-restore cosmos health: $(cosmos_health)"
fi

log "=== fault window end ==="
log "final: embedding=$(embedding_health) cosmos=$(cosmos_health)"
