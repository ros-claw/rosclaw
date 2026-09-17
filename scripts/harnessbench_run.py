#!/usr/bin/env python3
"""HarnessBench v1 operator 入口（MH11，0916 优化 §五/§六/§十）。

真实模型跑 HarnessBench：独立 workspace + 外部 oracle + 真实 LLM。
纪律：无 key → 全部 NOT_RUN（不合成冒充）；key 只读环境变量，
日志绝不打印 key（输出只有 verdict/指标）。

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

from benchmarks.harnessbench.runner import aggregate, has_model_key, run_leg  # noqa: E402
from benchmarks.harnessbench.tasks import TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legs", default="B", help="逗号分隔：A,B（默认 B）")
    parser.add_argument("--tasks", default="R02,E01,U01,H01")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default="/tmp/harnessbench")
    parser.add_argument("--settle-timeout", type=float, default=1200.0)
    args = parser.parse_args()

    if not has_model_key():
        print(
            json.dumps(
                {
                    "status": "NOT_RUN",
                    "reason": "no model key in environment (ROSCLAW_KIMI_API_KEY/KIMI_API_KEY/MOONSHOT_API_KEY)",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    out_root = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
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
                    )
                except Exception as exc:  # noqa: BLE001 —— 单次失败不拖垮矩阵
                    record = {
                        "leg": leg,
                        "task_id": task_id,
                        "run": run_idx,
                        "verdict": "ERROR",
                        "error": str(exc)[:300],
                        "oracle": {
                            "task_success": False,
                            "verified_success": False,
                            "false_success": False,
                            "reason": "runner_error",
                        },
                        "wall_time_s": 0.0,
                        "tool_calls": 0,
                        "glue_bytes": 0,
                        "python_loc": 0,
                        "xml_loc": 0,
                    }
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
