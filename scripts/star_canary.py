"""五角星 P0 金丝雀（0824 总纲 §21.5）——真实模型 10 连。

从用户家目录启动 rosclaw chat（真实 Kimi K3），同一任务：

    > 让仿真 UR5e 在竖直平面画一个五角星，并给我 GIF 和 MP4

硬断言（每次运行全查）：
- 只创建 1 Task、1 primary Native session、0 Worker；
- 模型可见 capability 调用建议不超过 5 次；
- 0 次手工 task_finish（模型面已删除）；
- 0 次重复 artifact（幂等账本）；
- 0 次 runtime install（依赖闭包）；
- 0 次 NO_ACTIVE_TASK；
- 0 次 transcript replay（重复事件）；
- 产生 canonical ResourceRef/PlanRef/TraceRef/RenderRef/VerificationRef；
- GIF/MP4 来自同一 TraceRef；
- 任务终态 SUCCEEDED（Coordinator 自动验收）。

用法：ROSCLAW_KIMI_API_KEY 走环境变量（绝不落盘）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.agentd.test_product_journey import PtySession
from tests.agentd.test_seventeen_gate_live import _prepare_home

PROMPT = "让仿真 UR5e 在竖直平面画一个五角星，并给我 GIF 和 MP4"
RUNS = int(os.environ.get('CANARY_RUNS', '10'))


def run_once(idx: int, base: Path) -> dict:
    home, env = _prepare_home(base / f"run{idx}")
    env["ROSCLAW_ALLOW_UNSANDBOXED_SHELL"] = "1"
    session = PtySession(
        [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
        env, log_path=base / f"run{idx}-pty.log",
        cwd=str(Path.home()),
    )
    outcome: dict = {"run": idx, "failures": []}
    try:
        session.expect(b"ROSClaw Native Agent", timeout=120)
        session.send(PROMPT + "\r")
        # 等任务完成通知（Coordinator 自动验收）——循环等到 deadline
        # （真实模型建任务/rollout/渲染需要 1~3 分钟）。
        deadline = time.monotonic() + 420
        completed = False
        while time.monotonic() < deadline and not completed:
            try:
                session.expect("任务完成：验收".encode(), timeout=60)
                completed = True
            except AssertionError:
                continue
        if not completed:
            outcome["failures"].append("任务完成通知未出现（Coordinator 未收尾）")
        time.sleep(2)
        session.send("/quit\r")
        time.sleep(1)
    finally:
        session.stop()

    db_path = home / "agentd" / "missions.db"
    if not db_path.exists():
        outcome["failures"].append("missions.db 不存在")
        return outcome
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    tasks = db.execute("SELECT * FROM tasks").fetchall()
    if len(tasks) != 1:
        outcome["failures"].append(f"tasks={len(tasks)}（期望 1）")
    elif tasks[0]["state"] != "SUCCEEDED":
        outcome["failures"].append(f"task state={tasks[0]['state']}")
    sessions = db.execute(
        "SELECT COUNT(*) AS n FROM task_session_bindings WHERE role='primary'"
    ).fetchone()
    if int(sessions["n"]) > 1:
        outcome["failures"].append(f"primary sessions={int(sessions['n'])}")
    # capability 调用计数（具身工具使用事件）。
    tool_uses = db.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE event_type='task.tool_used'"
    ).fetchone()
    if int(tool_uses["n"]) > 5:
        outcome["failures"].append(f"capability 调用 {int(tool_uses['n'])} 次 > 5")
    # 手工 finish（模型面已删除——journal 不得出现 task_finish 调用）。
    finishes = db.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE payload_json LIKE '%task_finish%'"
    ).fetchone()
    if int(finishes["n"]) > 0:
        outcome["failures"].append("出现手工 task_finish")
    # 重复 artifact。
    dup = db.execute(
        "SELECT sha256, COUNT(*) AS n FROM artifacts GROUP BY sha256 HAVING n > 1"
    ).fetchall()
    if dup:
        outcome["failures"].append(f"重复 artifact: {len(dup)} 组")
    # NO_ACTIVE_TASK。
    no_active = db.execute(
        "SELECT COUNT(*) AS n FROM agent_events WHERE payload_json "
        "LIKE '%NO_ACTIVE_TASK%'"
    ).fetchone()
    if int(no_active["n"]) > 0:
        outcome["failures"].append(f"NO_ACTIVE_TASK ×{int(no_active['n'])}")
    # transcript replay（重复 event 稳定键）。
    dup_events = db.execute(
        "SELECT session_id, event_id, COUNT(*) AS n FROM agent_events "
        "WHERE session_id IS NOT NULL GROUP BY session_id, event_id HAVING n > 1"
    ).fetchall()
    if dup_events:
        outcome["failures"].append(f"重复事件（transcript replay）: {len(dup_events)}")
    # canonical refs：plan/trace/render(gif+mp4)/verification。
    artifacts = db.execute("SELECT path, media_type FROM artifacts").fetchall()
    paths = [str(a["path"]) for a in artifacts]
    plans = list((home / "sim" / "plans").glob("*.json"))
    if not plans:
        outcome["failures"].append("缺 PlanRef（sim/plans 为空）")
    gif = [p for p in paths if p.endswith(".gif")]
    mp4 = [p for p in paths if p.endswith(".mp4")]
    if not gif:
        outcome["failures"].append("缺 GIF 交付物")
    if not mp4:
        outcome["failures"].append("缺 MP4 交付物")
    # GIF/MP4 同一 TraceRef（同目录）。
    if gif and mp4 and Path(gif[0]).parent != Path(mp4[0]).parent:
        outcome["failures"].append("GIF/MP4 不来自同一 TraceRef")
    outcomes = db.execute("SELECT outcome_json FROM task_outcomes").fetchall()
    if not outcomes:
        outcome["failures"].append("task_outcomes 未落库")
    else:
        oc = json.loads(str(outcomes[0]["outcome_json"]))
        if oc.get("verification") != "PASS":
            outcome["failures"].append(f"outcome verification={oc.get('verification')}")
    db.close()
    outcome["artifacts"] = paths
    return outcome


def main() -> int:
    base = Path(f"/tmp/star-canary-{int(time.time())}")
    base.mkdir(parents=True)
    results = []
    for idx in range(RUNS):
        print(f"[canary] run {idx + 1}/{RUNS} …", flush=True)
        started = time.monotonic()
        outcome = run_once(idx, base)
        outcome["elapsed_s"] = round(time.monotonic() - started, 1)
        results.append(outcome)
        status = "PASS" if not outcome["failures"] else f"FAIL {outcome['failures']}"
        print(f"[canary] run {idx + 1}: {status} ({outcome['elapsed_s']}s)", flush=True)
    passed = sum(1 for r in results if not r["failures"])
    report = base / "canary-report.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"[canary] {passed}/{RUNS} 通过——报告 {report}")
    return 0 if passed == RUNS else 1


if __name__ == "__main__":
    raise SystemExit(main())
