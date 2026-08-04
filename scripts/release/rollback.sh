#!/usr/bin/env bash
# PR-12 + 二次复核 R6：完整回滚——把 current 切回 previous。
# R6：回滚目标必须重新通过签名 + 全量 hash 校验（被篡改的 previous
# 不得成为回滚目标）。签名公钥：优先包外锚（与 install 同一查找
# 顺序），找不到时退回目标包内公钥——previous 是当初用锚验证过
# 才安装进来的，这里以 hash 完整性为主防线并如实标记。
set -euo pipefail

PREFIX="${ROSCLAW_PREFIX:-$HOME/.local/share/rosclaw}"
CURRENT="$PREFIX/current"
PREVIOUS="$PREFIX/previous"

[ -d "$PREVIOUS" ] || { echo "没有可回滚的 previous 版本。" >&2; exit 2; }
[ -f "$PREVIOUS/manifest.json" ] || { echo "previous 缺 manifest.json，拒绝回滚。" >&2; exit 2; }
[ -f "$PREVIOUS/manifest.sig" ] || { echo "previous 缺 manifest.sig，拒绝回滚。" >&2; exit 2; }

TRUSTED_KEY=""
for candidate in "${ROSCLAW_RELEASE_KEY:-}" /etc/rosclaw/release-pub.pem \
                 "$HOME/.rosclaw/release-keys/release-pub.pem" \
                 "$PREVIOUS/bundle-signing-public.pem"; do
  [ -n "$candidate" ] && [ -f "$candidate" ] && { TRUSTED_KEY="$candidate"; break; }
done
[ -n "$TRUSTED_KEY" ] || { echo "无可用的验签公钥，拒绝回滚。" >&2; exit 2; }
case "$TRUSTED_KEY" in
  "$PREVIOUS"*) echo "WARN: 使用 previous 包内公钥验签（无包外锚）——以 hash 完整性为主防线。" >&2 ;;
esac

openssl dgst -sha256 -verify "$TRUSTED_KEY" \
  -signature "$PREVIOUS/manifest.sig" "$PREVIOUS/manifest.json" \
  || { echo "previous manifest 签名无效——回滚目标可能被篡改，拒绝回滚。" >&2; exit 2; }

python3 - "$PREVIOUS" <<'PY'
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
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        bad.append(f"tampered: {rel}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(2)
print(f"rollback target verified ({len(manifest['files'])} files)")
PY

mv "$CURRENT" "$PREFIX/failed-$(date +%Y%m%d%H%M%S)"
mv "$PREVIOUS" "$CURRENT"
echo "==> 已回滚到 previous（失败版本保留在 $PREFIX/failed-* 供排查）。"
