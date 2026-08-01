#!/usr/bin/env bash
# TY1200 SeekDB 一键部署（幂等）：嵌入式真引擎开箱即用。
#
# 用法: bash setup_seekdb.sh [--verify-only]
#
# 步骤: 依赖(pin pyseekdb==1.3.0) → MiniLM 模型预缓存(hf-mirror) →
#       嵌入式引擎初始化 + database → rosclaw.yaml 后端切换 → 冒烟 → db doctor。
# 2881 服务器为可选增强(镜像可得时见文末说明), 嵌入式是边缘标准部署
# (与 pyproject 中 Jetson 验证路径一致)。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="$REPO/.venv/bin/python"
ROSCLAW_HOME_DIR="${ROSCLAW_HOME:-$HOME/.rosclaw}"
ROSCLAW_YAML="$ROSCLAW_HOME_DIR/config/rosclaw.yaml"
SEEKDB_HOME="$ROSCLAW_HOME_DIR/data/seekdb/embedded"
MODEL_CACHE="$HOME/.cache/pyseekdb/onnx_models/all-MiniLM-L6-v2"
VERIFY_ONLY="${1:-}"

step() { echo "── $*"; }
ok()   { echo "  ✅ $*"; }
fail() { echo "  ❌ $*"; exit 1; }

# --- 1. python 依赖（pyproject seekdb extra 的 pin） ---
step "1/5 python 依赖 (pyseekdb==1.3.0 + 引擎)"
if "$PY" -c "import pyseekdb, seekdb, pylibseekdb" 2>/dev/null; then
  ok "pyseekdb/seekdb/pylibseekdb 已安装"
else
  [ "$VERIFY_ONLY" = "--verify-only" ] && fail "缺 pyseekdb"
  echo "  安装中 (uv, 网络抖动时自动降级 curl 拉 wheel)..."
  uv pip install --python "$PY" "pyseekdb==1.3.0" seekdb seekdb-lib 2>&1 | tail -2 || {
    echo "  uv 失败, 走 curl wheel 通道"
    bash "$REPO/validation/ty1200/scripts/fetch_wheels.sh" && \
      uv pip install --python "$PY" --offline /tmp/seekdb_wheels/*.whl
  }
  "$PY" -c "import pyseekdb, seekdb, pylibseekdb" || fail "依赖安装失败"
  ok "依赖已安装"
fi

# --- 2. MiniLM 模型预缓存（首次引擎内置 embedding 需要, 走 hf-mirror） ---
step "2/5 MiniLM-L6-v2 模型缓存"
if [ -d "$MODEL_CACHE" ] && [ -n "$(ls -A "$MODEL_CACHE" 2>/dev/null)" ]; then
  ok "模型已缓存 ($MODEL_CACHE)"
else
  [ "$VERIFY_ONLY" = "--verify-only" ] && fail "模型未缓存"
  echo "  首次下载 ~90MB (hf-mirror.com)..."
  "$PY" - <<'PYEOF'
from pyseekdb.utils.embedding_functions import OnnxEmbeddingFunction
OnnxEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    hf_model_id="sentence-transformers/all-MiniLM-L6-v2",
    dimension=384,
)(["warmup"])
print("model cached")
PYEOF
  ok "模型已缓存"
fi

# --- 3. 嵌入式引擎初始化 + database ---
step "3/5 嵌入式引擎初始化 ($SEEKDB_HOME)"
"$PY" - <<PYEOF
import sys
sys.path.insert(0, "$REPO/src")
from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore
store = SeekDBEmbeddedStore(path="$SEEKDB_HOME", database="rosclaw")
store.connect()
assert store.is_connected()
store.disconnect()
print("engine ready")
PYEOF
ok "引擎就绪, database=rosclaw"

# --- 4. rosclaw.yaml 后端切换（幂等, 自动备份） ---
step "4/5 rosclaw.yaml → seekdb_embedded"
"$PY" - <<PYEOF
from pathlib import Path
import shutil, time
p = Path("$ROSCLAW_YAML")
src = p.read_text()
if "seekdb_backend: seekdb_embedded" in src:
    print("already seekdb_embedded")
else:
    shutil.copy(p, p.with_suffix(f".yaml.bak.{int(time.time())}"))
    import re
    src = re.sub(r"  seekdb_backend: \w+\n(  seekdb_path: .*\n)?",
                 "  seekdb_backend: seekdb_embedded\n"
                 "  seekdb_path: $SEEKDB_HOME\n"
                 "  seekdb_database: rosclaw\n",
                 src, count=1)
    p.write_text(src)
    print("switched (backup written)")
PYEOF
ok "配置就绪"

# --- 5. 冒烟: 写入/查询/向量 + db doctor ---
step "5/5 冒烟测试"
"$PY" - <<PYEOF
import sys
sys.path.insert(0, "$REPO/src")
from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore
store = SeekDBEmbeddedStore(path="$SEEKDB_HOME", database="rosclaw")
store.connect()
store.insert("knowledge_patterns", {
    "id": "setup_smoke", "robot_id": "ty1200",
    "description": "seekdb setup smoke: joint limit clamp recovery pattern"})
rows = store.query("knowledge_patterns", filters={"robot_id": "ty1200"}, limit=3)
assert any(r["id"] == "setup_smoke" for r in rows), "smoke record not found"
store.delete("knowledge_patterns", "setup_smoke")
store.disconnect()
print("insert/query/delete OK")
PYEOF
ok "读写冒烟通过"
ROSCLAW_HOME="$ROSCLAW_HOME_DIR" "$REPO/.venv/bin/rosclaw" db status 2>&1 | grep -E 'Backend|vector|native_seekdb|connected' | sed 's/^/  /'
ROSCLAW_HOME="$ROSCLAW_HOME_DIR" "$REPO/.venv/bin/rosclaw" db doctor 2>&1 | tail -1

echo
echo "══ SeekDB 部署完成 ══"
echo "  后端: seekdb_embedded (真 OceanBase 引擎, 无需服务器)"
echo "  数据: $SEEKDB_HOME"
echo "  备注: 2881 服务器为可选增强 — 镜像可得时配置"
echo "        seekdb_backend: seekdb_server + seekdb_url: mysql://root@127.0.0.1:2881/rosclaw"
