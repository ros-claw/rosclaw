"""大道至简 R3：A/B 真实比较门禁（2026-09-05 方案）。

同一个模型、同一个机器、同一个陌生任务，直接比较：

| 组别 | 配置                               |
| ---- | ---------------------------------- |
| A    | 原生 Pi Coding Agent（pi CLI，无 ROSClaw 扩展） |
| B    | Pi + ROSClaw Physical Extensions（rosclaw chat）   |

ROSClaw 必须证明：成功率不低于原生 Pi、物理任务完成更快、少写
重复胶水、用户看到的内容更少。**B 成功率 < A → 阻断发布
（exit 1）**——「原生 Pi 能画 hello，接入 ROSClaw 反而画成五角星」
即负价值实证（0905）。

用法（真实模型，operator gate——与 star_canary 同级）：
    ROSCLAW_KIMI_API_KEY 走环境变量（绝不落盘）：
    python3 scripts/ab_compare.py [--runs 2] [--out /tmp/ab-report.json]

CI 不跑真实比较（无 key）——门控比较器的正确性由
tests/agentd/test_ddzj_r3_ab_gate.py 用合成夹具钉死。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------
# 任务清单（陌生物理任务——任何一侧都不得有预制 recipe）
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AbTask:
    task_id: str
    prompt: str
    # 成功判据（两组同一标准——产出物必须在工作目录/任务目录可核验）：
    expect_artifact_glob: str  # 产出的视频/gif
    expect_trace: bool  # 必须有轨迹数据（json/csv）
    forbid_star_impostor: bool  # trace 不得是五角星冒充（plan shape=star5）


TASKS: list[AbTask] = [
    AbTask(
        task_id="draw_hi_letters",
        prompt="请用仿真机械臂把字母 HI 画出来，给我仿真视频和末端轨迹数据",
        expect_artifact_glob="*.mp4",
        expect_trace=True,
        forbid_star_impostor=True,
    ),
    AbTask(
        task_id="draw_custom_polyline",
        prompt="让仿真机械臂沿一条 L 形折线走一遍并渲染轨迹动画",
        expect_artifact_glob="*.gif",
        expect_trace=True,
        forbid_star_impostor=True,
    ),
]


# ---------------------------------------------------------------------
# 单次运行（PTY 驱动——与 star_canary 同一设施）
# ---------------------------------------------------------------------


@dataclass
class RunResult:
    group: str  # "A" | "B"
    task_id: str
    ok: bool
    wall_seconds: float
    tool_calls: int
    glue_bytes: int  # 模型写的代码字节数（重复胶水指标）
    visible_lines: int  # 用户看到的输出行数
    evidence: dict = field(default_factory=dict)


def _count_session_stats(session_dir: Path) -> tuple[int, int]:
    """pi session JSONL → (工具调用数, 模型写的代码字节数)。"""
    tool_calls = 0
    glue_bytes = 0
    for f in session_dir.glob("**/*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = entry.get("message") or {}
            for block in (msg.get("content") or []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall":
                    tool_calls += 1
                    if block.get("name") in ("write", "edit"):
                        args = block.get("arguments") or block.get("input") or {}
                        glue_bytes += len(str(args.get("content", args.get("newText", ""))))
    return tool_calls, glue_bytes


def _verify_task(task: AbTask, workdir: Path) -> tuple[bool, dict]:
    """同一成功判据核验两组产出。"""
    artifacts = list(workdir.glob(f"**/{task.expect_artifact_glob}"))
    traces = list(workdir.glob("**/trace*.json")) + list(
        workdir.glob("**/trace*.csv")
    )
    evidence: dict = {"artifacts": len(artifacts), "traces": len(traces)}
    ok = bool(artifacts)
    if task.expect_trace:
        ok = ok and bool(traces)
    if task.forbid_star_impostor:
        # 五角星冒充检测：plan 记录 shape=star5 而任务要的不是五角星
        # = 假成功（0905 的 hello→五角星）。
        for plan in workdir.glob("**/plans/*.json"):
            try:
                payload = json.loads(plan.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("shape") == "star5":
                evidence["star_impostor"] = str(plan)
                ok = False
    return ok, evidence


def run_once(
    group: str,
    task: AbTask,
    base: Path,
    *,
    pi_entry: tuple[str, str] | None,
    rosclaw_bin: Path,
    timeout_s: int = 900,
) -> RunResult:
    """跑一侧一次。A=pi CLI（无扩展）；B=rosclaw chat。"""
    from tests.agentd.test_product_journey import PtySession
    from tests.agentd.test_seventeen_gate_live import _prepare_home

    workdir = base / f"{group}_{task.task_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if group == "A":
        assert pi_entry is not None, "A 组需要 pi CLI（node + pi entry）"
        node, entry = pi_entry
        session = PtySession(
            [node, entry], dict(os.environ), cwd=workdir,
            log_path=workdir / "pty.log",
        )
        session_dir = workdir / ".pi" / "sessions"
    else:
        home, env = _prepare_home(workdir)
        session = PtySession(
            [str(rosclaw_bin), "chat"], env, cwd=workdir,
            log_path=workdir / "pty.log",
        )
        session.expect(b"ROSClaw Native Agent", timeout=120)
        session_dir = home / "agent" / "sessions"
    try:
        session.send(task.prompt + "\r")
        # 任务完成的通用信号：产出物出现或明显终态文字。
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if list(workdir.glob(f"**/{task.expect_artifact_glob}")):
                break
            time.sleep(5.0)
        session.send("/quit\r")
        time.sleep(2.0)
    finally:
        session.stop()
    wall = time.monotonic() - started
    ok, evidence = _verify_task(task, workdir)
    tool_calls, glue_bytes = (
        _count_session_stats(session_dir) if session_dir.exists() else (0, 0)
    )
    visible_lines = 0
    log = workdir / "pty.log"
    if log.exists():
        visible_lines = len(
            log.read_text(encoding="utf-8", errors="replace").splitlines()
        )
    return RunResult(
        group=group, task_id=task.task_id, ok=ok, wall_seconds=round(wall, 1),
        tool_calls=tool_calls, glue_bytes=glue_bytes,
        visible_lines=visible_lines, evidence=evidence,
    )


# ---------------------------------------------------------------------
# 门禁比较器（CI 可测的纯逻辑——真实数据由 operator 跑）
# ---------------------------------------------------------------------


def gate_verdict(results: list[RunResult]) -> tuple[bool, list[str]]:
    """B 成功率 < A → 阻断（负价值）。两侧都缺数据 → 不可判定
    （诚实：不算过）。"""
    reasons: list[str] = []
    by_group: dict[str, list[RunResult]] = {"A": [], "B": []}
    for r in results:
        if r.group in by_group:
            by_group[r.group].append(r)
    a, b = by_group["A"], by_group["B"]
    if not a or not b:
        return False, ["两侧缺数据——A/B 比较未真实运行，不得判过"]
    a_rate = sum(1 for r in a if r.ok) / len(a)
    b_rate = sum(1 for r in b if r.ok) / len(b)
    if b_rate < a_rate:
        reasons.append(
            f"负价值：B 成功率 {b_rate:.0%} < A {a_rate:.0%}——"
            "接入 ROSClaw 反而更差，阻断发布"
        )
    impostors = [r for r in b if r.evidence.get("star_impostor")]
    if impostors:
        reasons.append(
            f"B 组 {len(impostors)} 次五角星冒充（star_impostor）——"
            "形状冒充 = 假成功，阻断"
        )
    b_faster = sum(r.wall_seconds for r in b) <= sum(r.wall_seconds for r in a)
    b_less_glue = sum(r.glue_bytes for r in b) <= sum(r.glue_bytes for r in a)
    if not b_faster:
        reasons.append("注意：B 总耗时未优于 A（证据强度降一级，不阻断）")
    if not b_less_glue:
        reasons.append("注意：B 胶水代码量未优于 A（证据强度降一级，不阻断）")
    return (not any(r.startswith("负价值") or r.startswith("B 组") for r in reasons)), reasons


def main() -> int:
    runs = 1
    out = Path("/tmp/ab-report.json")
    argv = sys.argv[1:]
    if "--runs" in argv:
        runs = int(argv[argv.index("--runs") + 1])
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    if not os.environ.get("ROSCLAW_KIMI_API_KEY"):
        print("ERROR: ROSCLAW_KIMI_API_KEY 不在环境（真实模型 gate）", file=sys.stderr)
        return 2
    from rosclaw.agentd.pi_entry import find_pi_agent_entry

    pi_entry = find_pi_agent_entry()
    rosclaw_bin = Path(sys.executable).parent / "rosclaw"
    if not rosclaw_bin.exists():
        rosclaw_bin = Path(sys.executable)
    base = Path(f"/tmp/ab-{int(time.time())}")
    results: list[RunResult] = []
    for i in range(runs):
        for task in TASKS:
            for group in ("A", "B"):
                r = run_once(group, task, base / f"run{i}",
                             pi_entry=pi_entry, rosclaw_bin=rosclaw_bin)
                results.append(r)
                print(f"[{group}/{task.task_id}] ok={r.ok} "
                      f"wall={r.wall_seconds}s tools={r.tool_calls} "
                      f"glue={r.glue_bytes}B lines={r.visible_lines}")
    ok, reasons = gate_verdict(results)
    report = {
        "schema_version": "rosclaw.ab_compare.v1",
        "runs": runs,
        "results": [r.__dict__ for r in results],
        "gate": {"ok": ok, "reasons": reasons},
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n门禁：{'PASS' if ok else 'BLOCKED'}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"报告：{out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
