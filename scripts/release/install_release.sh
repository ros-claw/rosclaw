#!/usr/bin/env bash
# PR-12：发布包安装器（component installer）。
# 用法：tar xzf rosclaw-<ver>-linux-arm64.tar.gz && cd rosclaw-<ver>-linux-arm64 && ./install.sh [--prefix DIR]
set -euo pipefail

PREFIX="${ROSCLAW_PREFIX:-$HOME/.local/share/rosclaw}"
if [ "${1:-}" = "--prefix" ]; then PREFIX="$2"; fi
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT="$PREFIX/current"
PREVIOUS="$PREFIX/previous"
NEXT="$PREFIX/next-$RANDOM"

echo "==> installing rosclaw to $PREFIX"
mkdir -p "$PREFIX"
cp -r "$SRC_DIR" "$NEXT"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1 ($2)" >&2; exit 2; }; }
need python3 "3.11+"

# Python 组件
python3 -m venv "$NEXT/.venv"
"$NEXT/.venv/bin/pip" install --quiet --upgrade pip
"$NEXT/.venv/bin/pip" install --quiet "$NEXT"

# Node 组件（TUI/modeld）：需要 Node >= 22.19；缺失时诚实降级说明
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
    (cd "$NEXT/packages/$pkg" && "$NODE_BIN" "$(command -v npm)" ci --silent && "$NODE_BIN" "$(command -v npm)" run build --silent)
  done
else
  echo "WARN: Node >= 22.19 not found — rosclaw-tui/modeld 不可用；"
  echo "      Python-only 安装完成（rosclaw chat --basic 与 legacy backend 可用）。"
fi

# CLI 入口链接
mkdir -p "$PREFIX/bin"
cat > "$PREFIX/bin/rosclaw" <<EOF2
#!/usr/bin/env bash
exec "$CURRENT/.venv/bin/python" -m rosclaw.entrypoint "\$@"
EOF2
chmod +x "$PREFIX/bin/rosclaw"

# 原子切换 current；旧版本保留为 previous（rollback.sh 目标）。
rm -rf "$PREFIX/previous.tmp"
[ -e "$CURRENT" ] && mv "$CURRENT" "$PREFIX/previous.tmp" || true
mv "$NEXT" "$CURRENT.new" && mv "$CURRENT.new" "$CURRENT"
rm -rf "$PREVIOUS"
[ -e "$PREFIX/previous.tmp" ] && mv "$PREFIX/previous.tmp" "$PREVIOUS" || true

echo "==> installed. 加入 PATH: export PATH=\"$PREFIX/bin:\$PATH\""
echo "==> 验证: rosclaw doctor；回滚: $PREFIX/current/rollback.sh"
