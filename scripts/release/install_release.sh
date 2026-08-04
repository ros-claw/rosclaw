#!/usr/bin/env bash
# PR-12 + 审计 P0-05：发布包安装器（verify-before-execute，事务化）。
# 用法：tar xzf rosclaw-<ver>-linux-arm64.tar.gz && cd rosclaw-<ver>-linux-arm64 && ./install.sh [--prefix DIR] [--offline]
set -euo pipefail

PREFIX="${ROSCLAW_PREFIX:-$HOME/.local/share/rosclaw}"
OFFLINE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT="$PREFIX/current"
PREVIOUS="$PREFIX/previous"
NEXT="$PREFIX/next-$RANDOM"

echo "==> [1/5] 校验 manifest 签名与全部文件 hash（先验证，后执行）"
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1 ($2)" >&2; exit 2; }; }
need openssl "signature verification"
need python3 "3.11+"
[ -f "$SRC_DIR/manifest.json" ] || { echo "manifest.json 缺失，拒绝安装。" >&2; exit 2; }
[ -f "$SRC_DIR/manifest.sig" ] || { echo "manifest.sig 缺失，拒绝安装。" >&2; exit 2; }
[ -f "$SRC_DIR/bundle-signing-public.pem" ] || { echo "签名公钥缺失，拒绝安装。" >&2; exit 2; }
openssl dgst -sha256 -verify "$SRC_DIR/bundle-signing-public.pem" \
  -signature "$SRC_DIR/manifest.sig" "$SRC_DIR/manifest.json" \
  || { echo "manifest 签名无效——包可能被篡改，拒绝执行任何安装动作。" >&2; exit 2; }
python3 - "$SRC_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
bad = []
for rel, expected in manifest["files"].items():
    path = root / rel
    if not path.exists():
        bad.append(f"missing: {rel}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        bad.append(f"tampered: {rel}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(2)
print(f"verified {len(manifest['files'])} files")
PY

echo "==> [2/5] 安装到 staging"
mkdir -p "$PREFIX"
cp -r "$SRC_DIR" "$NEXT"

# Python 组件：online（PyPI）或 offline（vendor wheels，不触网）。
python3 -m venv "$NEXT/.venv"
"$NEXT/.venv/bin/pip" install --quiet --upgrade pip ${OFFLINE:+--no-index}
if [ "$OFFLINE" = "1" ]; then
  "$NEXT/.venv/bin/pip" install --quiet --no-index \
    --find-links "$NEXT/vendor/wheels" "$NEXT"
else
  "$NEXT/.venv/bin/pip" install --quiet "$NEXT"
fi

# Node 组件：用随包 node_modules（离线可用）；缺失时在线 npm ci。
NODE_OK=0
for candidate in node /usr/bin/node /usr/local/bin/node; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$($candidate --version | tr -d 'v')"
    major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
    if [ "$major" -gt 22 ] || { [ "$major" -eq 22 ] && [ "$minor" -ge 19 ]; }; then
      NODE_BIN="$candidate"; NODE_OK=1; break
    fi
  fi
done
if [ "$NODE_OK" = "1" ]; then
  for pkg in rosclaw-tui rosclaw-modeld; do
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
cat > "$PREFIX/bin/rosclaw" <<EOF2
#!/usr/bin/env bash
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
