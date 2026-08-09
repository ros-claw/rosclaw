#!/usr/bin/env bash
# Gate Evidence Manifest V2（五审 §11.3/PR-HF5-5）：所有 Native Agent Gate
# job 统一的机器可读 manifest。
#
# V1 缺陷：`merge_ref_sha` 保存的是 ref 名（refs/pull/N/merge）不是 commit；
# soak job 没有 manifest。V2 每字段都是真实值：
#
#   checked_out_sha   git rev-parse HEAD（实际 checkout 的 commit——PR 时是
#                     synthetic merge commit，push 时是 main 的 exact commit）
#   head_sha          PR head / push head
#   base_sha          PR base / push before
#   event_name        pull_request | push | schedule | workflow_dispatch
#   workflow_hash     本 workflow 文件内容的 sha256（证据与定义同源）
#
# 用法: write_evidence_manifest.sh <job-name> <out.json>
set -euo pipefail

job="${1:?job name required}"
out="${2:?output path required}"

checked_out_sha="$(git rev-parse HEAD)"
head_sha="${GITHUB_EVENT_HEAD_SHA:-${GITHUB_SHA}}"
base_sha="${GITHUB_EVENT_BASE_SHA:-}"
event_name="${GITHUB_EVENT_NAME:-unknown}"
workflow_hash="$(sha256sum "${GITHUB_WORKSPACE:-.}/.github/workflows/native-agent-gate.yml" | cut -d' ' -f1)"

mkdir -p "$(dirname "$out")"
cat > "$out" <<EOF
{
  "schema_version": "rosclaw.gate_evidence.v2",
  "job": "${job}",
  "checked_out_sha": "${checked_out_sha}",
  "head_sha": "${head_sha}",
  "base_sha": "${base_sha}",
  "event_name": "${event_name}",
  "workflow_hash": "sha256:${workflow_hash}",
  "run_id": "${GITHUB_RUN_ID:-}",
  "run_attempt": "${GITHUB_RUN_ATTEMPT:-}",
  "runner_os": "${RUNNER_OS:-unknown}",
  "runner_arch": "${RUNNER_ARCH:-unknown}",
  "node_version": "$(node --version 2>/dev/null || echo unavailable)",
  "python_version": "$(python --version 2>/dev/null || echo unavailable)",
  "test_selection": "${TEST_SELECTION:-unspecified}",
  "test_conclusion": "${TEST_CONCLUSION:-unknown}",
  "bundle_digest": "${BUNDLE_DIGEST:-}",
  "evidence_schema_version": "rosclaw.journey_evidence.v2",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo "manifest written: $out (checked_out_sha=$checked_out_sha)"
