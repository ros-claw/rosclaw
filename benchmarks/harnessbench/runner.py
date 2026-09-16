"""HarnessBench 运行器（MH11，0916 优化 §七/§十）。

一次运行 = 独立 workspace（task.md + model/ + allowed_assets）+
独立 HOME；Agent 只能看到 workspace 与工具面——看不到 tests/
oracle/expected patch/golden answer（§7.1 答案零泄漏）。

两侧条件（§十）：
- A = 原生 coding agent（pi CLI，mujoco+python+bash，无 ROSClaw
  sim 工具）；
- B = ROSClaw Physical Harness（rosclaw chat + `rosclaw sim` CLI）。

同模型/同 prompt/同任务/同 settle 判据。指标：task_success/
verified_success/false_success + wall_time/tool_calls/glue_bytes/
python LOC/XML LOC（§10.1——verified_success↑、false_success→0、
glue_code↓ 三个最关键）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.harnessbench import oracle
from benchmarks.harnessbench.tasks import TASKS

REPO = Path(__file__).resolve().parents[2]

#: B 侧提示追加（A/B 公平：双方都知道自己有什么工具）。
_B_TOOL_HINT = (
    "\n\n环境里装有 ROSClaw 仿真工具链：rosclaw sim --help 查看"
    "（capabilities/load/inspect/patch/rollout/observe/audit/"
    "branch-experiment/compare/compile-world/render，JSON 接口，"
    "默认以当前目录为任务根）。"
)
_A_TOOL_HINT = "\n\n环境里有 Python（含 mujoco/numpy）与 bash。没有专用仿真工具链——一切自己动手。"


def has_model_key() -> bool:
    return any(
        os.environ.get(v) for v in ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")
    )


def _count_session_stats(session_dir: Path) -> tuple[int, int]:
    """pi session JSONL → (工具调用数, 模型写的代码字节数)。"""
    tool_calls = 0
    glue_bytes = 0
    if not session_dir.is_dir():
        return 0, 0
    for f in session_dir.glob("**/*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = entry.get("message") or {}
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall":
                    tool_calls += 1
                    if block.get("name") in ("write", "edit"):
                        args = block.get("arguments") or block.get("input") or {}
                        glue_bytes += len(str(args.get("content", args.get("newText", ""))))
    return tool_calls, glue_bytes


def _code_loc(workspace: Path) -> dict[str, int]:
    """模型写的 Python/XML 行数（glue code 指标；staged model/ 除外）。"""
    python_loc = 0
    xml_loc = 0
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        if rel.parts[0] in ("model", "sim", "task.md", "answer.json"):
            continue
        if path.suffix == ".py":
            python_loc += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        elif path.suffix in (".xml", ".mjcf"):
            xml_loc += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return {"python_loc": python_loc, "xml_loc": xml_loc}


def stage_workspace(base: Path, task_id: str) -> Path:
    """独立 task workspace（§7.1）：task.md + staged files，别无他物。"""
    task = TASKS[task_id]
    base.mkdir(parents=True, exist_ok=True)
    (base / "task.md").write_text(task.prompt, encoding="utf-8")
    for rel, content in task.staged_files.items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return base


def _wait_settled(session, workspace: Path, settle_timeout: float) -> None:
    """等回合收束：输出静止 ≥20s 且工作区文件静止 ≥15s
    （driver.AgentRun 同款判据——两侧同标准）。"""
    deadline = time.monotonic() + settle_timeout
    started = time.monotonic()
    last_len = -1
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        with session._lock:
            current = len(session.output)
        if current != last_len:
            last_len = current
            quiet_since = time.monotonic()
        try:
            newest = max(
                (p.stat().st_mtime for p in workspace.rglob("*") if p.is_file()),
                default=0.0,
            )
        except OSError:
            newest = time.time()
        files_quiet = time.time() - newest > 15
        if time.monotonic() - quiet_since > 20 and files_quiet and time.monotonic() - started > 45:
            return
        time.sleep(1.0)
    raise AssertionError(f"回合 {settle_timeout}s 未收束（见 PTY 日志）")


def run_leg(
    leg: str,
    task_id: str,
    out_root: Path,
    run_idx: int,
    *,
    settle_timeout: float = 1200.0,
) -> dict[str, Any]:
    """跑一侧一次：A=pi CLI（无扩展）；B=rosclaw chat（sim CLI）。"""
    sys.path.insert(0, str(REPO))
    from tests.agentd.test_product_journey import PtySession

    task = TASKS[task_id]
    work = out_root / f"{leg.lower()}_{task_id.lower()}_run{run_idx}"
    stage_workspace(work, task_id)
    hint = _B_TOOL_HINT if leg == "B" else _A_TOOL_HINT
    prompt = task.prompt + hint

    started = time.monotonic()
    record: dict[str, Any] = {
        "leg": leg,
        "task_id": task_id,
        "category": task.category,
        "run": run_idx,
        "wall_time_s": 0.0,
        "tool_calls": 0,
        "glue_bytes": 0,
        "verdict": "ERROR",
    }

    session = None
    session_dir: Path | None = None
    try:
        if leg == "A":
            from scripts.ab_compare import _find_native_pi_cli, _prepare_native_pi_env

            pi_entry = _find_native_pi_cli()
            assert pi_entry is not None, "A 组需要原生 pi CLI"
            session = PtySession(
                [pi_entry],
                _prepare_native_pi_env(work),
                cwd=work,
                log_path=work / "pty.log",
            )
            session.expect(b"ctrl+o to show full startup help", timeout=120)
            session_dir = work / ".pi-agent" / "sessions"
        else:
            from tests.eval.agent_tier import driver

            run = driver.AgentRun(work, settle_timeout=settle_timeout)
            home_env = run.env
            session = PtySession(
                [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
                home_env,
                cwd=work,
                log_path=work / "pty.log",
            )
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session_dir = run.home / "agent" / "sessions"

        session.send(prompt + "\r")
        _wait_settled(session, work, settle_timeout)
    finally:
        record["wall_time_s"] = round(time.monotonic() - started, 1)
        if session is not None:
            with __import__("contextlib").suppress(Exception):
                session.stop()

    if session_dir is not None:
        record["tool_calls"], record["glue_bytes"] = _count_session_stats(session_dir)
    record.update(_code_loc(work))

    # Oracle 独立判定（环境结局，不信自报）。
    verdict = oracle.judge(task_id, work)
    record["oracle"] = verdict
    record["verdict"] = (
        "VERIFIED"
        if verdict.get("verified_success")
        else "FALSE_SUCCESS"
        if verdict.get("false_success")
        else "DONE"
        if verdict.get("task_success")
        else "FAIL"
    )
    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按侧聚合（§10.1 指标；verified_success↑/false_success→0/glue↓）。"""
    summary: dict[str, Any] = {}
    for leg in ("A", "B"):
        rows = [r for r in records if r["leg"] == leg]
        if not rows:
            continue
        total = len(rows)
        verified = sum(1 for r in rows if r["oracle"].get("verified_success"))
        done = sum(1 for r in rows if r["oracle"].get("task_success"))
        false_success = sum(1 for r in rows if r["oracle"].get("false_success"))
        summary[leg] = {
            "runs": total,
            "task_success_rate": done / total,
            "verified_success_rate": verified / total,
            "false_success_rate": false_success / total,
            "wall_time_s_median": sorted(r["wall_time_s"] for r in rows)[total // 2],
            "tool_calls_median": sorted(r["tool_calls"] for r in rows)[total // 2],
            "glue_bytes_median": sorted(r["glue_bytes"] for r in rows)[total // 2],
            "python_loc_median": sorted(r["python_loc"] for r in rows)[total // 2],
            "xml_loc_median": sorted(r["xml_loc"] for r in rows)[total // 2],
        }
    return summary
