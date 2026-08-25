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
for pkg in rosclaw-tui rosclaw-modeld rosclaw-agent; do
  src_dir="$REPO_ROOT/packages/$pkg"
  [ -d "$src_dir" ] || { echo "missing packages/$pkg" >&2; exit 1; }
  # 规格 §27.1：clean build——绝不用"dist 存在即跳过"（stale dist 是
  # 必须消除的事故源）。
  echo "==> clean-building packages/$pkg"
  (cd "$src_dir" && rm -rf dist && npm ci --silent && npm run build --silent)
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

# 4. SBOM（审计 P0-05.2 + R6）：CycloneDX 1.5 JSON（pip freeze/npm ls
# 只是依赖快照，不是 SBOM——保留快照作调试参考，正式 SBOM 用 CycloneDX）。
"$REPO_ROOT/.venv/bin/python" -m pip freeze --disable-pip-version-check 2>/dev/null   > "$STAGE/sbom-python.txt" || true
for pkg in rosclaw-tui rosclaw-modeld rosclaw-agent; do
  (cd "$REPO_ROOT/packages/$pkg" && npm ls --omit=dev --depth=2 2>/dev/null)     > "$STAGE/sbom-$pkg.txt" || true
done
python3 - "$STAGE" "$VERSION" <<'PY'
import json, re, sys
from pathlib import Path

stage, version = Path(sys.argv[1]), sys.argv[2]
components = []
for line in (stage / "sbom-python.txt").read_text().splitlines():
    match = re.match(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)$", line.strip())
    if match:
        name, ver = match.groups()
        components.append({
            "type": "library", "name": name, "version": ver,
            "purl": f"pkg:pypi/{name.lower()}@{ver}",
        })
for pkg_file in stage.glob("sbom-rosclaw-*.txt"):
    for line in pkg_file.read_text().splitlines():
        match = re.search(r"([@a-z0-9_/.-]+)@([0-9][A-Za-z0-9_.+-]*)$", line.strip().lstrip("├─└│ "))
        if match:
            name, ver = match.groups()
            components.append({
                "type": "library", "name": name, "version": ver,
                "purl": f"pkg:npm/{name}@{ver}",
            })
seen = set()
unique = []
for component in components:
    if component["purl"] not in seen:
        seen.add(component["purl"])
        unique.append(component)
sbom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "version": 1,
    "metadata": {
        "component": {
            "type": "application", "name": "rosclaw", "version": version,
            "purl": f"pkg:generic/rosclaw@{version}",
        }
    },
    "components": sorted(unique, key=lambda c: c["purl"]),
}
(stage / "sbom-cyclonedx.json").write_text(json.dumps(sbom, indent=1, ensure_ascii=False))
print(f"cyclonedx components: {len(unique)}")
PY

# 5. 离线资产（审计 P0-05.4）：目标机不再现场 TypeScript build、
# 不访问 PyPI/npm。
mkdir -p "$STAGE/vendor/wheels" "$STAGE/vendor/node_modules_pack"
# R6：离线包缺 wheel 是硬失败（不再 warning 后继续）。
# 构建后端（hatchling）也必须进 wheels——PEP 517 离线构建 rosclaw 自身
# 需要它（R6/PNA-10：缺 wheel 已在安装侧硬失败，这里补齐来源）。
# rosclaw 自身打成 wheel 进 vendor（force-include 数据随 wheel 走；
# 安装侧离线不再从源码目录跑 hatch 构建）。
"$REPO_ROOT/.venv/bin/python" -m pip wheel --no-deps --quiet \
  --wheel-dir "$STAGE/vendor/wheels" "$REPO_ROOT" || {
    echo "FAIL: rosclaw wheel 构建失败，拒绝产出。" >&2
    exit 1
  }
"$REPO_ROOT/.venv/bin/python" -m pip download   --disable-pip-version-check --quiet --dest "$STAGE/vendor/wheels"   "$REPO_ROOT" hatchling setuptools wheel || {
    echo "FAIL: pip download 不完整——离线包将缺 wheels，拒绝产出。" >&2
    exit 1
  }
[ -n "$(ls -A "$STAGE/vendor/wheels" 2>/dev/null)" ] || {
  echo "FAIL: vendor/wheels 为空——拒绝产出离线包。" >&2
  exit 1
}
for pkg in rosclaw-tui rosclaw-modeld rosclaw-agent; do
  (cd "$REPO_ROOT/packages/$pkg" && npm ci --omit=dev --silent)
  tar -C "$REPO_ROOT/packages/$pkg" -czf "$STAGE/vendor/node_modules_pack/$pkg.tar.gz" node_modules
done

# 6. build-info.json（规格 §27.2）：commit/版本/hash/Node 版本可追溯。
python3 - "$STAGE" "$VERSION" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

stage, version = Path(sys.argv[1]), sys.argv[2]
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = ""
node_version = subprocess.check_output(["node", "--version"], text=True).strip()     if subprocess.call(["bash", "-c", "command -v node >/dev/null"]) == 0 else ""
info = {
    "rosclaw_commit": commit,
    "pi_version": "0.83.0",
    "pi_commit": "588915ec71714688cee8b7153339e8bdebb3e82e",
    "node_version": node_version,
    "packages": {},
}
for pkg in ("rosclaw-tui", "rosclaw-modeld", "rosclaw-agent"):
    pkg_dir = stage / "packages" / pkg
    lock = pkg_dir / "package-lock.json"
    dist = pkg_dir / "dist"
    info["packages"][pkg] = {
        "package_lock_sha256": _sha(lock) if lock.exists() else "",
        "dist_sha256": _tree_sha(dist) if dist.exists() else "",
    }
(stage / "build-info.json").write_text(json.dumps(info, indent=1))
print("build-info written")
PY

# 7. bundled Node runtime（规格 §27.4）：目标机无需预装 Node。
# 构建机需网络下载一次；ROSCLAW_SKIP_NODE_BUNDLE=1 可跳过（降级为前置条件）。
if [ "${ROSCLAW_SKIP_NODE_BUNDLE:-0}" != "1" ]; then
  NODE_VER="22.19.0"
  case "$ARCH_NAME" in
    arm64) NODE_ARCH="arm64" ;;
    x64)   NODE_ARCH="x64" ;;
  esac
  NODE_TARBALL="node-v${NODE_VER}-linux-${NODE_ARCH}.tar.xz"
  if [ ! -f "$DIST_DIR/vendor-cache/$NODE_TARBALL" ]; then
    mkdir -p "$DIST_DIR/vendor-cache"
    for mirror in "https://registry.npmmirror.com/-/binary/node/v${NODE_VER}"                   "https://nodejs.org/dist/v${NODE_VER}"; do
      curl -fsSL "$mirror/$NODE_TARBALL" -o "$DIST_DIR/vendor-cache/$NODE_TARBALL" && break || true
    done
  fi
  if [ -s "$DIST_DIR/vendor-cache/$NODE_TARBALL" ]; then
    mkdir -p "$STAGE/vendor/node-runtime"
    tar -C "$STAGE/vendor/node-runtime" -xJf "$DIST_DIR/vendor-cache/$NODE_TARBALL" --strip-components=1
    echo "==> bundled node v${NODE_VER} (${NODE_ARCH})"
  else
    echo "FAIL: 无法获得 Node runtime（离线构建请预置 $DIST_DIR/vendor-cache/$NODE_TARBALL）" >&2
    exit 1
  fi
fi

# 9.5 fd/ripgrep 二进制（Pi InteractiveMode 初始化必需；离线目标机不能
# 现场下载）。经 ghfast 代理取 GitHub release 二进制。
if [ "${ROSCLAW_SKIP_TOOL_BINS:-0}" != "1" ]; then
  mkdir -p "$STAGE/vendor/tool-bins" "$DIST_DIR/vendor-cache"
  case "$ARCH_NAME" in
    arm64) FD_ARCH="aarch64-unknown-linux-gnu"; RG_ARCH="aarch64-unknown-linux-gnu" ;;
    # ripgrep 的 x86_64 Linux 资产是 musl（fd 是 gnu）——上游命名不对称，
    # 不能用同一个后缀（CI x86_64 构建 404 修复）。
    x64)   FD_ARCH="x86_64-unknown-linux-gnu";   RG_ARCH="x86_64-unknown-linux-musl" ;;
  esac
  FD_TARBALL="fd-v10.2.0-${FD_ARCH}.tar.gz"
  RG_TARBALL="ripgrep-14.1.1-${RG_ARCH}.tar.gz"
  for pair in "sharkdp/fd:v10.2.0:$FD_TARBALL" "BurntSushi/ripgrep:14.1.1:$RG_TARBALL"; do
    repo="${pair%%:*}"; rest="${pair#*:}"; tag="${rest%%:*}"; tarball="${rest#*:}"
    cache="$DIST_DIR/vendor-cache/$tarball"
    if [ ! -s "$cache" ]; then
      curl -fsSL "https://ghfast.top/https://github.com/$repo/releases/download/$tag/$tarball" -o "$cache" || \
        curl -fsSL "https://github.com/$repo/releases/download/$tag/$tarball" -o "$cache" || {
          echo "FAIL: 无法获得 $tarball（离线构建请预置 $cache）" >&2; exit 1; }
    fi
  done
  tar -C "$DIST_DIR/vendor-cache" -xzf "$DIST_DIR/vendor-cache/$FD_TARBALL" "fd-v10.2.0-${FD_ARCH}/fd"
  cp "$DIST_DIR/vendor-cache/fd-v10.2.0-${FD_ARCH}/fd" "$STAGE/vendor/tool-bins/fd"
  tar -C "$DIST_DIR/vendor-cache" -xzf "$DIST_DIR/vendor-cache/$RG_TARBALL" "ripgrep-14.1.1-${RG_ARCH}/rg"
  cp "$DIST_DIR/vendor-cache/ripgrep-14.1.1-${RG_ARCH}/rg" "$STAGE/vendor/tool-bins/rg"
  chmod +x "$STAGE/vendor/tool-bins/fd" "$STAGE/vendor/tool-bins/rg"
  echo "==> vendored fd/rg binaries"
fi

# 8. manifest（含各组件 hash 供回滚校验 + 签名输入——必须最后于内容）
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

# 9. 分离签名（审计 P0-05.3）：dev 签名密钥在本机（不入库），
# 公钥随包发布；安装器先验签再执行任何脚本。
SIGN_DIR="${ROSCLAW_SIGNING_HOME:-$HOME/.rosclaw/signing}"
if [ ! -f "$SIGN_DIR/dev-signing-private.pem" ]; then
  mkdir -p "$SIGN_DIR" && chmod 700 "$SIGN_DIR"
  openssl ecparam -name prime256v1 -genkey -noout -out "$SIGN_DIR/dev-signing-private.pem"
  chmod 600 "$SIGN_DIR/dev-signing-private.pem"
  openssl ec -in "$SIGN_DIR/dev-signing-private.pem" -pubout -out "$SIGN_DIR/dev-signing-public.pem"
fi
cp "$SIGN_DIR/dev-signing-public.pem" "$STAGE/bundle-signing-public.pem"
openssl dgst -sha256 -sign "$SIGN_DIR/dev-signing-private.pem"   -out "$STAGE/manifest.sig" "$STAGE/manifest.json"

mkdir -p "$DIST_DIR"
tar -C "$DIST_DIR" -czf "${DIST_DIR}/${BUNDLE}.tar.gz" "$BUNDLE"
echo "==> ${DIST_DIR}/${BUNDLE}.tar.gz"
