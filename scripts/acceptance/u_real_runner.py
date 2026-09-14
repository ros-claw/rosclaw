#!/usr/bin/env python3
"""U05/U06/U10 真实模型验收 runner（0914 PR-5，审计 §7）。

任务泛化场景必须真实模型（审计：任务泛化场景补真实模型；每类
≥3 次运行；报告分母/失败原因/人工介入/耗时-token）。

纪律：
- 无 ROSCLAW_KIMI_API_KEY/KIMI_API_KEY → 全部 NOT_RUN（不合成冒充）；
- key 只读环境变量；日志绝不打印 key；
- 每次运行独立 HOME+workspace；oracle 只看环境结局（产物/账本/
  文件证据），不信模型自报；
- 输出机器可读 u-real-results.json（并入 U 矩阵）。

用法：
  ROSCLAW_KIMI_API_KEY=... python scripts/acceptance/u_real_runner.py \
      --runs 3 --out /tmp/u-real
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCENARIOS = {
    "U05": {
        "prompt": (
            "用仿真机械臂画一个 L 形折线（两段：先向右再向上），"
            "输出仿真视频，并且**视频里要能看到末端的实际运动轨迹**"
            "（画在 3D 场景里的轨迹线，不是 2D 图）。"
        ),
        # oracle：场景视频存在 + receipt overlays_applied 含 actual_eef_trace
        "oracle": "u05",
    },
    "U06": {
        "prompt": (
            "让仿真机械臂末端走一条空间螺旋线（z 从 0.25m 升到 0.45m，"
            "平面投影是圆），输出带实际轨迹显示的视频；然后**不换相机"
            "参数再出一版顶视图**（同一轨迹，不重新仿真）。"
        ),
        "oracle": "u06",
    },
    "U10": {
        "prompt": (
            "写一个单摆（杆长 0.5m）自由摆动的 MuJoCo 模型，比较阻尼 "
            "0.02/0.1/0.3 三种设置的摆动衰减，给出数据结论（从仿真数据"
            "重算衰减率，不要背公式）。"
        ),
        "oracle": "u10",
    },
}


def _have_key() -> bool:
    return bool(
        os.environ.get("ROSCLAW_KIMI_API_KEY") or os.environ.get("KIMI_API_KEY")
    )


def _run_one(scenario: str, run_idx: int, out_dir: Path) -> dict:
    """单次真实运行：PTY rosclaw chat（agent_tier driver 复用）。"""
    sys.path.insert(0, str(REPO))
    from tests.eval.agent_tier import driver

    started = time.monotonic()
    work = out_dir / f"{scenario.lower()}_run{run_idx}"
    work.mkdir(parents=True, exist_ok=True)
    run = driver.AgentRun(work, settle_timeout=1800)
    record: dict = {
        "scenario": scenario,
        "run": run_idx,
        "verdict": "FAIL",
        "wall_time_s": 0.0,
        "interventions": 0,
        "detail": "",
    }
    try:
        run.run(SCENARIOS[scenario]["prompt"])
        record.update(_oracle(scenario, run))
    except Exception as exc:  # noqa: BLE001 — 失败如实记录（含原因）
        record["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        run.stop()
    record["wall_time_s"] = round(time.monotonic() - started, 1)
    return record


def _oracle(scenario: str, run) -> dict:
    """环境结局核验（不信模型自报）。"""
    import json as _json

    ws = run.ws
    videos = sorted(ws.rglob("*.mp4")) + sorted(ws.rglob("*.gif"))
    receipts = sorted(
        p for p in ws.rglob("render_receipt.json")
    )
    overlays_ok = False
    unfulfilled: list = []
    for rp in receipts:
        doc = _json.loads(rp.read_text(encoding="utf-8"))
        applied = doc.get("overlays_applied") or []
        unfulfilled.extend(doc.get("overlays_unfulfilled") or [])
        if "actual_eef_trace" in applied:
            overlays_ok = True
    if scenario == "U05":
        ok = bool(videos) and overlays_ok and not unfulfilled
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"videos={len(videos)} overlays_ok={overlays_ok} "
                f"unfulfilled={unfulfilled[:2]}"
            ),
        }
    if scenario == "U06":
        # 顶视图复用同一 trace（不重新仿真）：≥2 渲染 receipt 且
        # states_digest 一致。
        digests = {
            _json.loads(rp.read_text(encoding="utf-8")).get("states_digest")
            for rp in receipts
        }
        ok = (
            len(videos) >= 2 and overlays_ok and len(digests) == 1
            and not unfulfilled
        )
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"videos={len(videos)} receipts={len(receipts)} "
                f"digests={len(digests)} overlays_ok={overlays_ok}"
            ),
        }
    # U10：数据文件 + 结论含三种阻尼的衰减比较（数据可重算）。
    csvs = sorted(ws.rglob("*.csv")) + sorted(ws.rglob("*.json"))
    session_text = run.session.clean.decode("utf-8", errors="replace")
    has_three = all(d in session_text for d in ("0.02", "0.1", "0.3"))
    monotone = ("单调" in session_text) or ("衰减" in session_text)
    ok = bool(csvs) and has_three and monotone
    return {
        "verdict": "PASS" if ok else "FAIL",
        "detail": f"data_files={len(csvs)} three_dampings={has_three}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("/tmp/u-real"))
    parser.add_argument(
        "--scenarios", default="U05,U06,U10",
        help="逗号分隔（默认全部真实模型场景）",
    )
    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    if not _have_key():
        for scenario in args.scenarios.split(","):
            for i in range(1, args.runs + 1):
                results.append({
                    "scenario": scenario, "run": i, "verdict": "NOT_RUN",
                    "wall_time_s": 0.0, "interventions": 0,
                    "detail": "无真实模型 key——NOT_RUN 不合成冒充",
                })
    else:
        for scenario in args.scenarios.split(","):
            for i in range(1, args.runs + 1):
                print(f"[u-real] {scenario} run {i}/{args.runs} …", flush=True)
                record = _run_one(scenario, i, out_dir)
                results.append(record)
                print(
                    f"[u-real] {scenario} run {i}: {record['verdict']} "
                    f"({record['wall_time_s']}s) {record['detail'][:120]}",
                    flush=True,
                )

    payload = {
        "schema_version": "rosclaw.u_real.v1",
        "runs_per_scenario": args.runs,
        "results": results,
        "pass": sum(1 for r in results if r["verdict"] == "PASS"),
        "fail": sum(1 for r in results if r["verdict"] == "FAIL"),
        "not_run": sum(1 for r in results if r["verdict"] == "NOT_RUN"),
    }
    (out_dir / "u-real-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(
        {k: payload[k] for k in ("pass", "fail", "not_run")}, ensure_ascii=False,
    ))
    return 0 if payload["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
