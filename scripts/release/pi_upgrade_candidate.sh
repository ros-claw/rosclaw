#!/usr/bin/env bash
# W08（规格 §12.1）：Pi 候选升级——一轮可复现的 candidate 尝试。
#
# 不在聊天启动时更新；不改生产 pin——在专用工作目录做完整实验：
#   bump pins（deps+overrides）→ npm install 重生成 lock → 应用
#   补丁（锚点漂移=硬失败，进入报告）→ 构建 + TS 套件 → 报告。
#
# 报告写实测结果（补丁应用/构建/测试哪些过、哪些不过），不以
# fixture 代替真实模型结论（真实模型 A/B 属 W10，标 NOT_RUN）。
#
# 用法: pi_upgrade_candidate.sh <target_version> [REPORT_JSON]
# 回退: git checkout -- packages/*/package.json packages/*/package-lock.json
set -euo pipefail

TARGET="${1:?用法: pi_upgrade_candidate.sh <target_version> [REPORT_JSON]}"
REPORT="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$REPO_ROOT/dist/.pi-upgrade-$TARGET"
STAGE="/tmp"

rm -rf "$WORK"
mkdir -p "$WORK"

PI_PKGS='@earendil-works/pi-coding-agent @earendil-works/pi-agent-core @earendil-works/pi-ai @earendil-works/pi-tui'

echo "==> candidate: pi $TARGET（工作目录 $WORK）"
for pkg in rosclaw-tui rosclaw-agent; do
  rsync -a --exclude node_modules --exclude dist \
    "$REPO_ROOT/packages/$pkg/" "$WORK/$pkg/"
done

# bump 所有 pin（dependencies + overrides 两段同版本对齐——
# 不凭版本号相同推定同一模块实例，§12.1-2）。
for pkg in rosclaw-tui rosclaw-agent; do
  python3 - "$WORK/$pkg/package.json" "$TARGET" <<'PY'
import json, sys
path, target = sys.argv[1], sys.argv[2]
data = json.load(open(path))
bumped = []
for section in ("dependencies", "devDependencies", "overrides"):
    for name in list((data.get(section) or {})):
        if name.startswith("@earendil-works/pi-"):
            old = data[section][name]
            if old != target:
                data[section][name] = target
                bumped.append(f"{section}:{name} {old}->{target}")
if bumped:
    json.dump(data, open(path, "w"), indent="\t", ensure_ascii=False)
    open(path, "a").write("\n")
print("\n".join(bumped) if bumped else "(no pi pins)")
PY
done

OLD_LOCK=$(sha256sum "$REPO_ROOT/packages/rosclaw-agent/package-lock.json" | cut -d' ' -f1)

status_lock=ok
status_patch=skipped
status_build=skipped
status_tests=skipped
patch_log=""
for pkg in rosclaw-tui rosclaw-agent; do
  # --ignore-scripts：补丁应用与 lock 生成分开（postinstall 会
  # 自动跑补丁——分开才能分辨 lock 失败 vs 补丁锚点漂移）。
  echo "==> npm install --ignore-scripts（重生成 lock）packages/$pkg"
  if ! (cd "$WORK/$pkg" && npm install --ignore-scripts --silent); then
    status_lock="failed:$pkg"
    break
  fi
done

if [ "$status_lock" = "ok" ]; then
  echo "==> 应用上游补丁（锚点漂移=硬失败）"
  if patch_log=$(cd "$WORK/rosclaw-agent" && node patches/apply-upstream-patches.mjs 2>&1); then
    status_patch=ok
  else
    status_patch="failed"
  fi
  echo "$patch_log"
fi

if [ "$status_patch" = "ok" ]; then
  echo "==> 构建"
  for pkg in rosclaw-tui rosclaw-agent; do
    mkdir -p "$WORK/$pkg/prompts"
    cp "$REPO_ROOT/src/rosclaw/agentd/context/prompts/native_agent_v2.md" \
       "$WORK/$pkg/prompts/" 2>/dev/null || true
    if ! (cd "$WORK/$pkg" && npm run build --silent); then
      status_build="failed:$pkg"
      break
    fi
  done
  [ "$status_build" = skipped ] || true
  if [ "$status_build" = "skipped" ]; then status_build=ok; fi
fi

if [ "$status_build" = "ok" ]; then
  echo "==> TS 套件"
  if (cd "$WORK/rosclaw-agent" && node --test dist/test/*.test.js 2>&1 | tail -6); then
    status_tests=ok
  else
    status_tests="failed"
  fi
fi

NEW_LOCK=$(sha256sum "$WORK/rosclaw-agent/package-lock.json" | cut -d' ' -f1)

python3 - "$REPORT" <<PY
import json, sys
report = {
    "candidate": "pi $TARGET",
    "from_version": "0.83.0",
    "lock_digest_before": "sha256:$OLD_LOCK",
    "lock_digest_after": "sha256:$NEW_LOCK",
    "lock_changed": "$OLD_LOCK" != "$NEW_LOCK",
    "steps": {
        "pin_bump_and_lock": "$status_lock",
        "patch_apply": "$status_patch",
        "build": "$status_build",
        "ts_suite": "$status_tests",
    },
    "patch_log": """$(echo "$patch_log" | head -20 | sed 's/"/\\"/g')""",
    "real_model_ab": "NOT_RUN（真实模型验收属 W10——不合成冒充）",
    "rollback": "git checkout -- packages/*/package.json packages/*/package-lock.json",
}
text = json.dumps(report, ensure_ascii=False, indent=1)
print(text)
if "$REPORT":
    open("$REPORT", "w").write(text + "\n")
PY

# 诚实终态：任一步失败即非零退出（候选被证据拒绝是有效结论）。
[ "$status_tests" = "ok" ] || exit 1
echo "==> 候选 $TARGET 通过契约层（lock/补丁/构建/TS 套件）"
