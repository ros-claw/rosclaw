#!/usr/bin/env python3
"""merge truth 机器生成——合并状态只信 GitHub API，禁止人工填写。

0916 三审根治：#546 误记事件（update-branch 被接受误当合并完成）
之后，一切"已合入"表述必须可由本脚本复现。

用法：
    GITHUB_TOKEN=... python3 scripts/acceptance/merge_truth.py 543 544 546
输出：
    /tmp/merge-truth.json（原始 API 事实）+ stdout 渲染 markdown 表
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

REPO = "ros-claw/rosclaw"


def _api(path: str, token: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("GITHUB_TOKEN 未设置", file=sys.stderr)
        return 2
    pr_numbers = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not pr_numbers:
        print("用法: merge_truth.py <PR号>...", file=sys.stderr)
        return 2
    rows = []
    for n in pr_numbers:
        d = _api(f"/repos/{REPO}/pulls/{n}", token)
        rows.append({
            "pr": n,
            "title": d["title"],
            "state": d["state"],
            "merged": d["merged"],
            "merged_at": d.get("merged_at") or "",
            "merge_commit": (d.get("merge_commit_sha") or "")[:10],
        })
    main = _api(f"/repos/{REPO}/branches/main", token)
    truth = {
        "generated_by": "github-api",
        "repo": REPO,
        "main_sha": main["commit"]["sha"],
        "prs": rows,
    }
    with open("/tmp/merge-truth.json", "w") as fh:
        json.dump(truth, fh, indent=1, ensure_ascii=False)
    print(f"main = {main['commit']['sha'][:10]}\n")
    print("| PR | 标题 | merged | merged_at | merge_commit |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(
            f"| #{r['pr']} | {r['title'][:50]} | "
            f"{'**true**' if r['merged'] else 'false'} | "
            f"{r['merged_at'][:10] or '—'} | {r['merge_commit'] or '—'} |"
        )
    unmerged = [r["pr"] for r in rows if not r["merged"]]
    print(f"\n未合并: {unmerged or '无'}")
    return 1 if unmerged else 0


if __name__ == "__main__":
    sys.exit(main())
