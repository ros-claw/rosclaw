#!/usr/bin/env python3
"""HarnessBench v1 operator 入口（MH11，0916 优化 §五/§六/§十）。

真实模型跑 HarnessBench：独立 workspace + 外部 oracle + 真实 LLM。
纪律：缺凭据 → 全部 NOT_RUN（不合成冒充）；不复制 OAuth 凭据，
日志绝不打印 key（输出只有 verdict/指标）。
chatgpt 档案使用 PI 内置 openai-codex + 显式 --model-id；
单一 --readonly-oauth-path 只读引用专用 OAuth 来源，不复制/登录/刷新；
会话和模型配置各 trial 隔离，缺失或过期诚实 NOT_RUN。
ChatGPT A 侧是 PI SDK InteractiveMode launcher（不是原 PI CLI），B 是 ROSClaw。

用法：
  ROSCLAW_KIMI_API_KEY=... python scripts/harnessbench_run.py \
      --legs B --tasks R02,E01,U01,H01 --runs 1 --out /tmp/harnessbench
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.harnessbench.runner import (  # noqa: E402
    aggregate,
    error_record,
    model_credential_issue,
    resolve_model_profile,
    run_leg,
    validate_readonly_oauth_namespace,
)
from benchmarks.harnessbench.tasks import TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legs", default="B", help="逗号分隔：A,B（默认 B）")
    parser.add_argument("--tasks", default="R02,E01,U01,H01")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default="/tmp/harnessbench")
    parser.add_argument("--settle-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--model",
        default="kimi-k3",
        help="模型档案（MODEL_PROFILES 键；0916 §十一 多模型资格认证）",
    )
    parser.add_argument(
        "--model-id",
        help="chatgpt 的显式 PI 内置模型 ID；不隐式选择模型或修改全局默认",
    )
    parser.add_argument(
        "--readonly-oauth-path",
        type=Path,
        help="chatgpt 专用 OAuth auth.json 的绝对路径（owned0600/无链接）；只读、不复制、不刷新",
    )
    args = parser.parse_args()

    try:
        profile = resolve_model_profile(args.model, args.model_id)
        if args.model == "chatgpt" and args.readonly_oauth_path is not None:
            validate_readonly_oauth_namespace(args.readonly_oauth_path, Path(args.out))
        issue = model_credential_issue(
            args.model,
            model_id=args.model_id,
            readonly_oauth_path=args.readonly_oauth_path,
            settle_timeout=args.settle_timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if issue is not None:
        print(
            json.dumps(
                {
                    "status": "NOT_RUN",
                    "model": args.model,
                    "provider": profile["provider"] if profile else "kimi-code",
                    "model_id": profile["model"] if profile else "k3",
                    "reason": issue,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    out_root = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    # Builtin trials must not reuse a preexisting/symlinked orchestration tree.
    out_root.mkdir(parents=True, exist_ok=args.model != "chatgpt")
    legs = [leg.strip().upper() for leg in args.legs.split(",") if leg.strip()]
    task_ids = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    for task_id in task_ids:
        if task_id not in TASKS:
            print(f"unknown task {task_id}; have {sorted(TASKS)}", file=sys.stderr)
            return 2

    records = []
    for leg in legs:
        for task_id in task_ids:
            for run_idx in range(1, args.runs + 1):
                print(f"[harnessbench] leg={leg} task={task_id} run={run_idx} ...", flush=True)
                try:
                    record = run_leg(
                        leg,
                        task_id,
                        out_root,
                        run_idx,
                        settle_timeout=args.settle_timeout,
                        model=args.model,
                        model_id=args.model_id,
                        readonly_oauth_path=args.readonly_oauth_path,
                    )
                except Exception as exc:  # noqa: BLE001 —— 单次失败不拖垮矩阵
                    # RunInfraError 携带部分记录：stall 跑的真实 wall_time
                    # 落账（live 实证：1200s stall 记 0.0 污染聚合统计）。
                    record = error_record(exc, leg, task_id, run_idx, model=args.model)
                records.append(record)
                print(f"  → {record['verdict']} ({record['wall_time_s']}s)", flush=True)
                (out_root / "results.json").write_text(
                    json.dumps({"records": records}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    summary = aggregate(records)
    report = {
        "schema_version": "rosclaw.harnessbench.v1",
        "out_root": str(out_root),
        "model": args.model,
        "model_id": args.model_id,
        "credential_mode": "EXPLICIT_READONLY_NO_REFRESH"
        if args.model == "chatgpt"
        else "EXISTING_PROFILE",
        "legs": legs,
        "tasks": task_ids,
        "runs_per_task": args.runs,
        "summary": summary,
        "records": records,
    }
    (out_root / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[harnessbench] results → {out_root / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
