#!/bin/sh
# 合并守门（十五审纪律修正）：三个工作流在目标分支最新 commit 上
# 全部 completed/success 才输出 GREEN——#366/#370 的教训：只看 CI/CD
# 一行导致 Gate 红被合入两次。
# 用法：scripts/check-pr-green.sh <branch>
set -eu
BRANCH="${1:?usage: check-pr-green.sh <branch>}"
SHA=$(gh api "repos/ros-claw/rosclaw/branches/$BRANCH" --jq '.commit.sha[:7]')
echo "branch=$BRANCH head=$SHA"
FAIL=0
for WF in "CI/CD" "Native Agent Gate" "First Boot Acceptance"; do
  LINE=$(gh api "repos/ros-claw/rosclaw/actions/runs?branch=$BRANCH&per_page=10" \
    --jq "[.workflow_runs[] | select(.name==\"$WF\" and (.head_sha | startswith(\"$SHA\")))] | sort_by(.created_at) | last | \"\(.status)/\(.conclusion)\"")
  echo "  $WF: $LINE"
  [ "$LINE" = "completed/success" ] || FAIL=1
done
if [ "$FAIL" = "0" ]; then
  echo "GREEN——可以合并"
else
  echo "NOT GREEN——禁止合并" >&2
  exit 1
fi
