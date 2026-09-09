#!/usr/bin/env bash
# W11（规格 §15.1）：共同 JS staging——离线 tar 与 PyPI wheel 消费
# 同一组产物，不在两条链中分别构建不同 JS。
#
# 构建在专用工作目录（默认 dist/.js-build，可传参覆盖）：rsync
# 开发 checkout 副本后在其内 npm ci/build——绝不在 packages/<pkg>
# 开发目录里 npm ci（依赖被删状态事故源，规格 §15.1 明令）。
#
# 产出（$STAGE/<pkg>/）：
#   dist/                   tsc 构建产物（含 build-stamp/prompts/skills）
#   package.json            运行时元数据
#   package-lock.json       锁定依赖（安装侧 npm ci 可重现）
#   src/                    调试参考（tar 包既有内容，保持 parity）
#   node_modules.prod.tar.gz 生产依赖 tarball（离线 tar 安装侧免 npm）
#
# 用法: build_js_staging.sh [STAGE_DIR] [WORK_DIR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-$REPO_ROOT/dist/js-stage}"
WORK="${2:-$REPO_ROOT/dist/.js-build}"

rm -rf "$WORK"
mkdir -p "$WORK" "$STAGE"

for pkg in rosclaw-tui rosclaw-agent; do
  src_dir="$REPO_ROOT/packages/$pkg"
  [ -d "$src_dir" ] || { echo "missing packages/$pkg" >&2; exit 1; }
  # clean build——绝不用"dist 存在即跳过"（stale dist 事故源）。
  rm -rf "$WORK/$pkg" "$STAGE/$pkg"
  rsync -a --exclude node_modules --exclude dist "$src_dir/" "$WORK/$pkg/"
  # copy-assets.mjs 的 python-tree 相对路径（pkgRoot/../../src/…）
  # 在 staging 工作目录不成立——喂它设计的包内 fallback（prompts/）。
  if [ "$pkg" = "rosclaw-agent" ]; then
    mkdir -p "$WORK/$pkg/prompts"
    cp "$REPO_ROOT/src/rosclaw/agentd/context/prompts/native_agent_v2.md" \
       "$WORK/$pkg/prompts/"
  fi
  (cd "$WORK/$pkg" && npm ci --silent && npm run build --silent)
  mkdir -p "$STAGE/$pkg"
  cp -r "$WORK/$pkg/dist" "$WORK/$pkg/package.json" \
        "$WORK/$pkg/package-lock.json" "$STAGE/$pkg/"
  cp -r "$WORK/$pkg/src" "$STAGE/$pkg/"
  # 生产依赖 tarball（第二次 npm ci 发生在 WORK 副本——开发
  # checkout 的 node_modules 全程不被触碰）。
  (cd "$WORK/$pkg" && rm -rf node_modules && npm ci --omit=dev --silent)
  tar -C "$WORK/$pkg" -czf "$STAGE/$pkg/node_modules.prod.tar.gz" node_modules
  # staging 完整性硬校验——缺产物即硬错误（规格 §15.4）。
  [ -f "$STAGE/$pkg/dist/src/main.js" ] || {
    echo "FAIL: $pkg dist/src/main.js 未产出" >&2; exit 1;
  }
  [ -s "$STAGE/$pkg/node_modules.prod.tar.gz" ] || {
    echo "FAIL: $pkg 生产依赖 tarball 为空" >&2; exit 1;
  }
done

echo "js staging ready: $STAGE"
