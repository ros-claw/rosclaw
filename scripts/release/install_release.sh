#!/usr/bin/env bash
# PR-12 + 审计 P0-05 + 二次复核 R6：发布包安装器（verify-before-execute）。
#
# R6 信任模型：验签公钥必须来自**包外 trust anchor**——
#   --trusted-key PATH            显式指定；或
#   $ROSCLAW_RELEASE_KEY           环境变量；或
#   /etc/rosclaw/release-pub.pem  系统级锚；或
#   ~/.rosclaw/release-keys/release-pub.pem  用户预置锚。
# 包内 bundle-signing-public.pem 只作信息对照，不再用于验签
# （"自带钥匙证明自己"不是发行者信任）。--allow-untrusted-dev 是
# 显式开发者逃生门（输出大字警告，结果标记 DEV-UNTRUSTED）。
#
# 用法：./install.sh [--prefix DIR] [--offline] [--trusted-key PATH]
#       [--trusted-fingerprint SHA256] [--allow-untrusted-dev]
set -euo pipefail

PREFIX="${ROSCLAW_PREFIX:-$HOME/.local/share/rosclaw}"
OFFLINE=0
TRUSTED_KEY=""
TRUSTED_FP=""
ALLOW_UNTRUSTED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    --trusted-key) TRUSTED_KEY="$2"; shift 2 ;;
    --trusted-fingerprint) TRUSTED_FP="$2"; shift 2 ;;
    --allow-untrusted-dev) ALLOW_UNTRUSTED=1; shift ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT="$PREFIX/current"
PREVIOUS="$PREFIX/previous"
NEXT="$PREFIX/next-$RANDOM"

echo "==> [1/5] 校验信任锚、manifest 签名与全部文件 hash（先验证，后执行）"
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1 ($2)" >&2; exit 2; }; }
need openssl "signature verification"
need python3 "3.11+"
[ -f "$SRC_DIR/manifest.json" ] || { echo "manifest.json 缺失，拒绝安装。" >&2; exit 2; }
[ -f "$SRC_DIR/manifest.sig" ] || { echo "manifest.sig 缺失，拒绝安装。" >&2; exit 2; }

# --- trust anchor 解析（R6：锚在包外） -------------------------------------
if [ -z "$TRUSTED_KEY" ]; then
  for candidate in "${ROSCLAW_RELEASE_KEY:-}" /etc/rosclaw/release-pub.pem \
                   "$HOME/.rosclaw/release-keys/release-pub.pem"; do
    [ -n "$candidate" ] && [ -f "$candidate" ] && { TRUSTED_KEY="$candidate"; break; }
  done
fi
if [ -z "$TRUSTED_KEY" ]; then
  if [ "$ALLOW_UNTRUSTED" = "1" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "!! DEV-UNTRUSTED：未提供包外信任锚——仅验证包内自洽性，" >&2
    echo "!! 不证明发行者身份。仅允许在开发环境使用。             !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    [ -f "$SRC_DIR/bundle-signing-public.pem" ] || { echo "签名公钥缺失，拒绝安装。" >&2; exit 2; }
    TRUSTED_KEY="$SRC_DIR/bundle-signing-public.pem"
  else
    echo "未找到包外信任锚（release public key）。R6：包内公钥不能自证。" >&2
    echo "请提供 --trusted-key PATH，或预置 ~/.rosclaw/release-keys/release-pub.pem；" >&2
    echo "开发环境可显式 --allow-untrusted-dev（结果不受信）。" >&2
    exit 2
  fi
fi
# 可选指纹钉住：锚本身也要匹配预期指纹。
if [ -n "$TRUSTED_FP" ]; then
  actual_fp="$(openssl pkey -pubin -in "$TRUSTED_KEY" -outform DER 2>/dev/null | openssl dgst -sha256 -r | awk '{print $1}')"
  [ "$actual_fp" = "$TRUSTED_FP" ] || {
    echo "信任锚指纹不匹配：期望 $TRUSTED_FP，实际 $actual_fp——拒绝安装。" >&2; exit 2; }
fi
# 信息对照：包内公钥若与锚不一致，说明包被重打包（非致命——锚才是权威，
# 但必须告诉用户）。
if [ -f "$SRC_DIR/bundle-signing-public.pem" ] && \
   ! cmp -s "$TRUSTED_KEY" "$SRC_DIR/bundle-signing-public.pem"; then
  echo "WARN: 包内公钥与信任锚不同——以包外锚为准（包可能被重新分发）。" >&2
fi

openssl dgst -sha256 -verify "$TRUSTED_KEY" \
  -signature "$SRC_DIR/manifest.sig" "$SRC_DIR/manifest.json" \
  || { echo "manifest 签名无效——包可能被篡改，拒绝执行任何安装动作。" >&2; exit 2; }

# 全文件 hash 校验 + **额外文件拒绝**（R6：manifest 未列出的一律拒装）。
python3 - "$SRC_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
allowed_extra = {"manifest.json", "manifest.sig", "bundle-signing-public.pem"}
bad = []
for rel, expected in manifest["files"].items():
    path = root / rel
    if not path.exists():
        bad.append(f"missing: {rel}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        bad.append(f"tampered: {rel}")
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        # 只允许解析后仍在包内的相对 symlink（bundled node 的 bin  shim）；
        # 绝对链接/逃逸链接一律拒（R6 的本意是防穿越与外泄）。
        target = path.resolve()
        if not str(target).startswith(str(root.resolve())) or not target.exists():
            bad.append(f"unsafe symlink in bundle: {path.relative_to(root)}")
        continue
    if path.is_file():
        rel = str(path.relative_to(root))
        if rel not in manifest["files"] and rel not in allowed_extra:
            bad.append(f"unlisted file (extra-file rejection): {rel}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(2)
print(f"verified {len(manifest['files'])} files; no unlisted extras")
PY

echo "==> [2/5] 安装到 staging"
mkdir -p "$PREFIX"
cp -r "$SRC_DIR" "$NEXT"

# Python 组件：online（PyPI）或 offline（vendor wheels，不触网）。
python3 -m venv "$NEXT/.venv"
"$NEXT/.venv/bin/pip" install --quiet --upgrade pip ${OFFLINE:+--no-index}
if [ "$OFFLINE" = "1" ]; then
  # R6：离线模式缺 wheel 是硬失败，不是 warning。
  [ -d "$NEXT/vendor/wheels" ] && [ -n "$(ls -A "$NEXT/vendor/wheels" 2>/dev/null)" ] || {
    echo "离线安装但 vendor/wheels 为空——构建期 pip download 不完整，拒绝继续。" >&2; exit 2; }
  # PNA-10：离线只装 wheel（force-include 数据随 wheel；不从源码目录
  # 重新跑 hatch 构建——stage 里没有那些数据目录）。
  ls "$NEXT"/vendor/wheels/rosclaw-*.whl >/dev/null 2>&1 || {
    echo "离线包缺 rosclaw 自身 wheel，拒绝继续。" >&2; exit 2; }
  "$NEXT/.venv/bin/pip" install --quiet --no-index \
    --find-links "$NEXT/vendor/wheels" rosclaw
else
  "$NEXT/.venv/bin/pip" install --quiet "$NEXT"
fi

# Node 组件：优先随包 bundled runtime（规格 §27.4，目标机免装 Node）；
# 其次系统 Node >= 22.19；用随包 node_modules（离线可用），不现场 npm。
NODE_OK=0
if [ -x "$NEXT/vendor/node-runtime/bin/node" ]; then
  NODE_BIN="$NEXT/vendor/node-runtime/bin/node"
  NODE_OK=1
  echo "==> using bundled node $($NODE_BIN --version)"
fi
if [ "$NODE_OK" = "0" ]; then
for candidate in node /usr/bin/node /usr/local/bin/node; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$($candidate --version | tr -d 'v')"
    major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
    if [ "$major" -gt 22 ] || { [ "$major" -eq 22 ] && [ "$minor" -ge 19 ]; }; then
      NODE_BIN="$candidate"; NODE_OK=1; break
    fi
  fi
done
fi
if [ "$NODE_OK" = "1" ]; then
  for pkg in rosclaw-tui rosclaw-modeld rosclaw-agent; do
    if [ -f "$NEXT/vendor/node_modules_pack/$pkg.tar.gz" ]; then
      tar -C "$NEXT/packages/$pkg" -xzf "$NEXT/vendor/node_modules_pack/$pkg.tar.gz"
    else
      (cd "$NEXT/packages/$pkg" && "$NODE_BIN" "$(command -v npm)" ci --omit=dev --silent)
    fi
    # dist 已在包内（构建期产物）；不现场 TypeScript build。
    [ -f "$NEXT/packages/$pkg/dist/src/main.js" ] || {
      echo "$pkg 缺 dist/src/main.js（包不完整）。" >&2; exit 2;
    }
  done
elif [ "${ROSCLAW_REQUIRE_TUI:-0}" = "1" ]; then
  echo "FAIL: Node >= 22.19 not found and ROSCLAW_REQUIRE_TUI=1——"
  echo "      完整安装验收失败（不允许静默回退 --basic）。" >&2
  exit 2
else
  echo "WARN: Node >= 22.19 not found — rosclaw-tui/modeld 不可用；"
  echo "      Python-only 安装完成（rosclaw chat --basic 为显式救援模式）。"
fi

echo "==> [3/5] CLI 入口"
mkdir -p "$PREFIX/bin"
if [ -d "$NEXT/vendor/tool-bins" ]; then
  cp "$NEXT/vendor/tool-bins/"* "$PREFIX/bin/" 2>/dev/null || true
  chmod +x "$PREFIX/bin/fd" "$PREFIX/bin/rg" 2>/dev/null || true
fi
cat > "$PREFIX/bin/rosclaw" <<EOF2
#!/usr/bin/env bash
export PATH="$PREFIX/bin:\$PATH"
exec "$CURRENT/.venv/bin/python" -m rosclaw.entrypoint "\$@"
EOF2
chmod +x "$PREFIX/bin/rosclaw"

echo "==> [4/5] 原子切换 current（previous 保留供回滚）"
rm -rf "$PREFIX/previous.tmp"
[ -e "$CURRENT" ] && mv "$CURRENT" "$PREFIX/previous.tmp" || true
mv "$NEXT" "$CURRENT.new" && mv "$CURRENT.new" "$CURRENT"
rm -rf "$PREVIOUS"
[ -e "$PREFIX/previous.tmp" ] && mv "$PREFIX/previous.tmp" "$PREVIOUS" || true

echo "==> [5/5] 安装后健康检查（失败自动回滚）"
if ! "$CURRENT/.venv/bin/python" -c "import rosclaw, rosclaw.agentd, rosclaw.operatord" >/dev/null 2>&1; then
  echo "健康检查失败——自动回滚到 previous。" >&2
  [ -d "$PREVIOUS" ] && "$CURRENT/rollback.sh" || rm -rf "$CURRENT"
  exit 3
fi
echo "==> installed & healthy. export PATH=\"$PREFIX/bin:\$PATH\""
echo "==> 验证: rosclaw doctor；回滚: $PREFIX/current/rollback.sh"
