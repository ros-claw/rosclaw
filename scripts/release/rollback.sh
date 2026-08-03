#!/usr/bin/env bash
# PR-12：完整回滚——把 current 切回 previous（含校验）。
set -euo pipefail

PREFIX="${ROSCLAW_PREFIX:-$HOME/.local/share/rosclaw}"
CURRENT="$PREFIX/current"
PREVIOUS="$PREFIX/previous"

[ -d "$PREVIOUS" ] || { echo "没有可回滚的 previous 版本。" >&2; exit 2; }
# manifest 校验：回滚目标必须结构完整。
[ -f "$PREVIOUS/manifest.json" ] || { echo "previous 缺 manifest.json，拒绝回滚。" >&2; exit 2; }

mv "$CURRENT" "$PREFIX/failed-$(date +%Y%m%d%H%M%S)"
mv "$PREVIOUS" "$CURRENT"
echo "==> 已回滚到 previous（失败版本保留在 $PREFIX/failed-* 供排查）。"
