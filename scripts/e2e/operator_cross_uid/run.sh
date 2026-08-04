#!/usr/bin/env bash
# T1 跨 UID operator e2e 主机封装：构建镜像（如需）并运行。
# 用法：scripts/e2e/operator_cross_uid/run.sh
# 环境：ROSCLAW_E2E_IMAGE 可覆盖镜像名。
set -euo pipefail

IMAGE="${ROSCLAW_E2E_IMAGE:-rosclaw-operator-cross-uid-e2e:local}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker unavailable" >&2
    exit 127
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "building $IMAGE ..." >&2
    docker build -t "$IMAGE" -f "$REPO_ROOT/scripts/e2e/operator_cross_uid/Dockerfile" "$REPO_ROOT"
fi

exec docker run --rm "$IMAGE"
