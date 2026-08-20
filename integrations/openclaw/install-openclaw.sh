#!/usr/bin/env bash
# ROSClaw × OpenClaw 安装脚本（设计 §7/§8/§14）。
#
# 幂等：可重复运行。版本全部来自 version.lock（E2E 验证过的组合），
# 不装 latest。
#
# 用法：
#   integrations/openclaw/install-openclaw.sh
#
# 前置：Node 在支持窗口内（>=22.22.3 <23 / >=24.15 <25 / >=25.9）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=version.lock
source "$SCRIPT_DIR/version.lock"

echo "==> 目标版本: openclaw=$OPENCLAW_VERSION acpx=$OPENCLAW_ACPX_VERSION feishu=$OPENCLAW_FEISHU_VERSION node=$NODE_VERSION"

# ---- 1. Node 版本检查（不自动改用户的 Node——这属于系统级变更，报指引） ----
if ! command -v node >/dev/null 2>&1; then
  echo "FAIL: node 不在 PATH。安装 Node $NODE_VERSION（推荐 nvm: nvm install 24）。" >&2
  exit 1
fi
node_ver="$(node -v)"
node_major="${node_ver#v}"; node_major="${node_major%%.*}"
node_ok=0
case "$node_major" in
  22) [ "$(printf '%s\n' "22.22.3" "${node_ver#v}" | sort -V | head -1)" = "22.22.3" ] && node_ok=1 ;;
  24) [ "$(printf '%s\n' "24.15.0" "${node_ver#v}" | sort -V | head -1)" = "24.15.0" ] && node_ok=1 ;;
  25) [ "$(printf '%s\n' "25.9.0" "${node_ver#v}" | sort -V | head -1)" = "25.9.0" ] && node_ok=1 ;;
  *)  [ "$node_major" -gt 25 ] && node_ok=1 ;;
esac
if [ "$node_ok" != "1" ]; then
  echo "FAIL: node $node_ver 不在支持窗口（>=22.22.3 <23 / >=24.15 <25 / >=25.9）。" >&2
  exit 1
fi
echo "OK: node $node_ver"

# ---- 2. OpenClaw 本体（锁版本，不 latest） ----
current="$(openclaw --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9]+\.[0-9]+(-[0-9]+)?' | head -1 || true)"
if [ "$current" = "$OPENCLAW_VERSION" ]; then
  echo "OK: openclaw $current 已是锁定版本"
else
  echo "==> 安装 openclaw@$OPENCLAW_VERSION（当前: ${current:-未安装}）"
  npm install -g "openclaw@$OPENCLAW_VERSION"
fi

# ---- 3. 插件（锁版本） ----
for spec in "@openclaw/acpx@$OPENCLAW_ACPX_VERSION" "@openclaw/feishu@$OPENCLAW_FEISHU_VERSION"; do
  name="${spec%%@*}"
  if openclaw plugins list 2>/dev/null | grep -q "${name#@openclaw/}"; then
    echo "OK: 插件 $name 已安装（如需对齐版本请 openclaw plugins install $spec）"
  else
    echo "==> 安装插件 $spec"
    openclaw plugins install "$spec"
  fi
done

# ---- 4. Gateway systemd 服务 ----
echo "==> 安装/刷新 gateway 服务定义"
openclaw gateway install --force || true

echo
echo "安装完成。下一步："
echo "  1. integrations/openclaw/configure-openclaw.sh   # 应用安全基线 + rosclaw harness"
echo "  2. openclaw channels login --channel feishu      # 配飞书 App ID/Secret"
echo "  3. rosclaw channel doctor --require-openclaw     # 验收"
