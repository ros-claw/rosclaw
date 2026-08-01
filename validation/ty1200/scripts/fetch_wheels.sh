#!/usr/bin/env bash
# 网络抖动时的兜底 wheel 通道：直接从 PyPI 元数据取 URL 用 curl（可续传）下载。
# setup_seekdb.sh 在 uv 安装失败时调用。
set -uo pipefail

DEST="${1:-/tmp/seekdb_wheels}"
mkdir -p "$DEST"
cd "$DEST"

fetch() { # fetch <package> <version-pin-or-empty>
  local pkg="$1" pin="${2:-}"
  local meta="${pkg}_meta.json"
  for i in 1 2 3; do
    timeout 25 curl -s "https://pypi.org/pypi/${pkg}/json" -o "$meta" && break
    sleep 2
  done
  python3 - "$meta" "$pkg" "$pin" <<'PY'
import json, subprocess, sys
meta, pkg, pin = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(meta))
ver = pin or d["info"]["version"]
urls = []
for u in d["releases"].get(ver, []):
    f = u["filename"]
    if ("cp312" in f and "x86_64" in f and "musl" not in f) or "py3-none-any" in f or "abi3" in f and "manylinux" in f:
        urls.append((u["url"], f))
if not urls:
    print(f"no wheel for {pkg}=={ver}", file=sys.stderr)
    sys.exit(1)
url, fname = urls[0]
print(f"downloading {fname}")
subprocess.run(["curl", "-sL", "-C", "-", "--retry", "5", "--retry-delay", "3", "-o", fname, url], check=True)
PY
}

fetch pyseekdb "1.3.0"
fetch seekdb ""
fetch seekdb-lib ""
fetch pylibseekdb "1.3.0.post3"
echo "wheels ready in $DEST"
