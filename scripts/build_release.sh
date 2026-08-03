#!/usr/bin/env bash
# PR-12：Linux arm64 发布包构建（诚实打包：不隐藏任何构建步骤）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cd "$REPO_ROOT" && python3 -c "import re; print(re.search(r'version = \"([^\"]+)\"', open('pyproject.toml').read()).group(1))")"
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64) ARCH_NAME="arm64" ;;
  x86_64)  ARCH_NAME="x64" ;;
  *)       ARCH_NAME="$ARCH" ;;
esac
BUNDLE="rosclaw-${VERSION}-linux-${ARCH_NAME}"
DIST_DIR="${REPO_ROOT}/dist"
STAGE="${DIST_DIR}/${BUNDLE}"

echo "==> building ${BUNDLE}"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 1. Python 源码与元数据
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO_ROOT/src" "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/README.md" \
  "$REPO_ROOT/LICENSE" "$STAGE/" 2>/dev/null || \
  { cp -r "$REPO_ROOT/src" "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/README.md" "$STAGE/"; }

# 2. Node 包（预先构建 dist；lockfile 一并打包，安装侧 npm ci 可重现）
for pkg in rosclaw-tui rosclaw-modeld; do
  src_dir="$REPO_ROOT/packages/$pkg"
  [ -d "$src_dir" ] || { echo "missing packages/$pkg" >&2; exit 1; }
  if [ ! -d "$src_dir/dist/src" ]; then
    echo "==> building packages/$pkg"
    (cd "$src_dir" && npm ci --silent && npm run build --silent)
  fi
  mkdir -p "$STAGE/packages/$pkg"
  cp -r "$src_dir/dist" "$src_dir/package.json" "$src_dir/package-lock.json" \
        "$src_dir/tsconfig.json" "$STAGE/packages/$pkg/"
  cp -r "$src_dir/src" "$STAGE/packages/$pkg/"
done

# 3. 第三方声明与安装脚本
cp -r "$REPO_ROOT/third_party" "$STAGE/"
cp "$REPO_ROOT/scripts/release/install_release.sh" "$STAGE/install.sh"
cp "$REPO_ROOT/scripts/release/rollback.sh" "$STAGE/"
chmod +x "$STAGE/install.sh" "$STAGE/rollback.sh"

# 4. manifest（含各组件 hash 供回滚校验）
python3 - "$STAGE" "$VERSION" "$ARCH_NAME" <<'PY'
import hashlib, json, sys
from pathlib import Path
stage, version, arch = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
files = {}
for path in sorted(stage.rglob("*")):
    if path.is_file():
        files[str(path.relative_to(stage))] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "product": "rosclaw",
    "version": version,
    "platform": f"linux-{arch}",
    "files": files,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
PY

mkdir -p "$DIST_DIR"
tar -C "$DIST_DIR" -czf "${DIST_DIR}/${BUNDLE}.tar.gz" "$BUNDLE"
echo "==> ${DIST_DIR}/${BUNDLE}.tar.gz"
