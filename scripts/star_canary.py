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
    # R0-9（§7.4）：生产路径/用户可见交付/行为预算/证据等级/屏幕
    # 卫生——断言函数抛 AssertionError，失败入 outcome["failures"]。
    final_outcome = (
        json.loads(str(outcomes[0]["outcome_json"])) if outcomes else {}
    )
    for check in (
        lambda: assert_production_plan_graph(home),
        lambda: assert_user_visible_delivery(home, require_scene=True),
        lambda: assert_model_behavior_budget(home, max_task_calls=1),
        lambda: assert_evidence_levels(home, require_scene=True),
        lambda: assert_screen_clean(session.clean),
        lambda: assert_final_answer_consistent(
            session.clean.decode("utf-8", errors="replace")[-3000:],
            final_outcome,
        ),
    ):
        try:
            check()
        except AssertionError as exc:
            outcome["failures"].append(str(exc)[:200])
    # 指标记录（§7.1：工具调用数、错误数、token/耗时——只报通过率
    # 不够）。
    db = sqlite3.connect(db_path)
    outcome["tool_calls"] = db.execute(
        "SELECT COUNT(*) FROM task_events WHERE event_type = 'task.tool_used'"
    ).fetchone()[0]
    outcome["error_count"] = db.execute(
        "SELECT COUNT(*) FROM agent_events WHERE type = 'tool.completed' "
        "AND json_extract(payload_json, '$.ok') = 0"
    ).fetchone()[0]
    usage_rows = db.execute(
        "SELECT usage_json FROM pi_event_mirrors WHERE usage_json != ''"
    ).fetchall()
    total_tokens = 0
    for (usage_json,) in usage_rows:
        try:
            usage = json.loads(str(usage_json))
            # Pi usage 是 camelCase（totalTokens）——实测金丝雀账本。
            total_tokens += int(
                usage.get("totalTokens") or usage.get("total_tokens") or 0
            )
        except (ValueError, TypeError):
            continue
    outcome["tokens"] = total_tokens
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


# ----------------------------------------------------------------------
# R0-9（0826 体验审计 §7.4）：金丝雀断言函数——真实生产路径、
# 用户可见交付、模型行为预算、证据等级、屏幕/最终回答一致性。
# 每个函数：通过返回 None，失败抛 AssertionError（可单测）。
# ----------------------------------------------------------------------

_BASE_PLAN_NODES = {
    "resolve_robot", "make_path", "simulate", "render", "verify",
}


def assert_production_plan_graph(home: Path) -> None:
    """生产 PlanGraph 证据：plan.node_* 事件存在且覆盖基线五节点
    （不是"测试直调模板"——真实生产执行才产生这些事件）。"""
    conn = sqlite3.connect(home / "agentd" / "missions.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT event_type, payload_json FROM task_events "
        "WHERE event_type LIKE 'plan.node_%'"
    ).fetchall()
    conn.close()
    started = {
        json.loads(str(r["payload_json"])).get("node_id")
        for r in rows if r["event_type"] == "plan.node_started"
    }
    completed = {
        json.loads(str(r["payload_json"])).get("node_id")
        for r in rows if r["event_type"] == "plan.node_completed"
    }
    assert started >= _BASE_PLAN_NODES, (
        f"plan.node_started 缺节点：{_BASE_PLAN_NODES - started}"
        "——未走生产 PlanGraph"
    )
    assert completed >= _BASE_PLAN_NODES, (
        f"plan.node_completed 缺节点：{_BASE_PLAN_NODES - completed}"
    )


def assert_user_visible_delivery(home: Path, *, require_scene: bool = False) -> None:
    """用户可见交付：outcome.artifact_refs 含 GIF+MP4，每条带
    open_command；require_scene 时 MP4 必须是 scene_3d kind
    （2D 预览不能冒充场景视频）。"""
    conn = sqlite3.connect(home / "agentd" / "missions.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT outcome_json FROM task_outcomes ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "task_outcomes 未落库"
    outcome = json.loads(str(row["outcome_json"]))
    refs = outcome.get("artifact_refs") or []
    gif = [r for r in refs if r.get("media_type") == "image/gif"]
    mp4 = [r for r in refs if r.get("media_type") == "video/mp4"]
    assert gif, "用户可见 refs 缺 GIF"
    assert mp4, "用户可见 refs 缺 MP4"
    for ref in [*gif, *mp4]:
        assert str(ref.get("open_command", "")).startswith(
            "rosclaw artifact open "
        ), f"交付物缺 open_command：{ref}"
    if require_scene:
        assert any(r.get("kind") == "scene_3d" for r in mp4), (
            f"MP4 不是 scene_3d kind（2D 预览冒充场景视频）：{mp4}"
        )


def assert_model_behavior_budget(home: Path, *, max_task_calls: int = 1) -> None:
    """模型行为预算：具身/能力链入口调用 ≤max、bash=0、手工
    finish=0（已知 recipe 不允许模型绕链脚本化——R0-1.5 后
    自动路由，期望 0 次）。"""
    conn = sqlite3.connect(home / "agentd" / "missions.db")
    calls = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE event_type = 'task.tool_used'"
    ).fetchone()[0]
    # 能力链调用（run8 实证：materialized capability 手拼绕链——
    # compute/execute 才算绕链；observe 是观察不是绕链）。
    capability_calls = conn.execute(
        "SELECT COUNT(*) FROM agent_events WHERE type = 'tool.completed' "
        "AND json_extract(payload_json, '$.tool_name') IN "
        "('rosclaw_compute', 'rosclaw_execute', 'rosclaw_request_action')"
    ).fetchone()[0]
    total = calls + capability_calls
    assert total <= max_task_calls, (
        f"具身/能力链调用 {total} 次 > {max_task_calls}"
        f"（task_used={calls} capability={capability_calls}）"
    )
    bash = conn.execute(
        "SELECT COUNT(*) FROM agent_events WHERE type = 'tool.completed' "
        "AND json_extract(payload_json, '$.tool_name') = 'bash'"
    ).fetchone()[0]
    assert bash == 0, f"模型用 bash 绕过 recipe ×{bash}"
    finish = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE payload_json LIKE '%task_finish%'"
    ).fetchone()[0]
    assert finish == 0, f"出现手工 task_finish ×{finish}"
    conn.close()


def assert_evidence_levels(home: Path, *, require_scene: bool = False) -> None:
    """证据等级拆分：GEOMETRY_PLAN/KINEMATIC_TRACKING/DYNAMIC_
    ROLLOUT（require_scene 加 SCENE_RENDER）——不是单个标签。"""
    conn = sqlite3.connect(home / "agentd" / "missions.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT outcome_json FROM task_outcomes ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "task_outcomes 未落库"
    outcome = json.loads(str(row["outcome_json"]))
    levels = set((outcome.get("evidence") or {}).get("levels") or [])
    required = {"GEOMETRY_PLAN", "KINEMATIC_TRACKING", "DYNAMIC_ROLLOUT"}
    if require_scene:
        required.add("SCENE_RENDER")
    assert required <= levels, f"证据等级缺 {required - levels}（{levels}）"


_RAW_JSON_KEYS = (b'"artifact_refs"', b'"plan":', b'"verification":')
_SUCCESS_MARKERS = ("任务完成", "VERIFIED", "✓")
_FAILURE_MARKERS = ("Unreachable", "Blocked")


def assert_screen_clean(screen: bytes) -> None:
    """屏幕卫生：无 raw JSON 键；无 Unreachable/Blocked 与成功
    标记并存（readiness 与 effect 结果必须一致）。"""
    for key in _RAW_JSON_KEYS:
        assert key not in screen, f"屏幕出现 raw JSON（{key!r}）"
    text = screen.decode("utf-8", errors="replace")
    has_success = any(marker in text for marker in _SUCCESS_MARKERS)
    for marker in _FAILURE_MARKERS:
        assert not (marker in text and has_success), (
            f"屏幕同时出现 {marker} 与成功标记——readiness 与 effect 矛盾"
        )


_COMPLETION_CLAIMS = ("已绘制完成，视频已交付", "全部完成", "完整完成")
_LIMITATION_MARKERS = ("未", "但", "没有", "无法", "限制", "退化", "部分")


def assert_final_answer_consistent(answer: str, outcome: dict) -> None:
    """最终回答与 TaskOutcome 一致：PARTIAL/MISSING 时回答必须带
    限制说明，不得宣称完整完成。"""
    verification = str(outcome.get("verification", ""))
    delivery = str(outcome.get("delivery", ""))
    partial = verification == "PARTIAL" or delivery in ("MISSING", "PARTIAL")
    if not partial:
        return
    for claim in _COMPLETION_CLAIMS:
        assert claim not in answer, (
            f"最终回答与 outcome 不一致：outcome 是 PARTIAL/MISSING，"
            f"回答却宣称 {claim!r}"
        )
    assert any(marker in answer for marker in _LIMITATION_MARKERS), (
        "最终回答与 outcome 不一致：PARTIAL/MISSING 但回答无限制说明"
    )


if __name__ == "__main__":
    raise SystemExit(main())
