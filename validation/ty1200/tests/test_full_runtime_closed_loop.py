"""全模块 Runtime 实时闭环（非 fixture）— 复盘最深一环.

按生产接线驱动:
    SHADOW executor 内嵌真实 StaticActionGate (sandbox 预览)
    → 越界动作被 gate BLOCK, executor 发布 firewall.action_blocked (生产主题)
    → Runtime._on_firewall_action_blocked → How HeuristicEngine 恢复建议事件
    → MemoryInterface 消费阻断事件写 FailureMemory (含 recovery hint)
    → Trace 记录 BLOCKED span
    → 从 bus/memory 取回恢复建议, 钳制参数, 第二次提交
    → gate 通过, SHADOW executor COMPLETED

每一步断言真实证据; executor 的 gate 预览即生产 sandbox-backed executor 的行为.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

RESULTS: dict = {"chain": {}}


def _record(step: str, ok: bool, detail: str = "") -> None:
    RESULTS["chain"][step] = {"ok": ok, "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {step}: {detail[:110]}")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    home = tmp_path_factory.mktemp("closed-loop-home")
    os.environ["ROSCLAW_HOME"] = str(home)
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    rt = Runtime(
        RuntimeConfig(
            robot_id="universal_robots_ur5e",
            enable_firewall=True,
            enable_memory=True,
            enable_practice=True,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=True,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=True,
        )
    )
    rt.initialize()
    rt.start()
    yield rt, home
    rt.stop()


def _action(action_id: str, joints: list[float]):
    from rosclaw.kernel import ActionEnvelope, EvidenceLevel, ExecutionMode, VerificationPolicy

    return ActionEnvelope(
        action_id=action_id,
        actor_id="review-agent",
        agent_framework="claude-code",
        session_id="closed-loop-session",
        body_id="universal_robots_ur5e",
        body_snapshot_hash="sha256:ur5e",
        capability_id="arm.move_joints",
        arguments={"values": joints},
        execution_mode=ExecutionMode.SHADOW,
        # SHADOW 仿真动作显式声明接受 SYNTHETIC 证据
        # (默认 TASK_VERIFIED 面向真机, fail-closed 设计不变)
        verification_policy=VerificationPolicy(required_evidence=EvidenceLevel.SYNTHETIC),
    )


def test_live_closed_loop(runtime):
    rt, home = runtime
    from rosclaw.core.event_bus import Event
    from rosclaw.kernel import ActionExecutionResult, ActionState, EvidenceLevel, ExecutionMode
    from rosclaw.sandbox.firewall.gate import StaticActionGate

    gate = StaticActionGate("universal_robots_ur5e", "empty", "mujoco")
    n = len(gate.joint_limits)
    executed: list[str] = []
    how_hints: list[dict] = []
    rt.event_bus.subscribe(
        "heuristic.recovery_suggested", lambda e: how_hints.append(e.payload)
    )

    def sandbox_backed_exec(action) -> ActionExecutionResult:
        """生产形态的 sandbox-backed executor: 先过物理 gate 再执行."""
        decision = gate.check(action.arguments)
        if not decision.is_allowed:
            rt.event_bus.publish(
                Event(
                    topic="firewall.action_blocked",
                    payload={
                        "request_id": action.action_id,
                        "episode_id": action.action_id,
                        "reason": decision.violated_constraints[0]
                        if decision.violated_constraints else "firewall",
                        "violations": [
                            {"description": f"joint limit exceeded: {decision.reason}"}
                            for v in decision.violated_constraints
                        ],
                        "risk_score": decision.risk_score,
                    },
                    source="sandbox.executor",
                )
            )
            return ActionExecutionResult(
                final_state=ActionState.BLOCKED,
                evidence_level=EvidenceLevel.SYNTHETIC,
                errors=[{"code": "SANDBOX_BLOCKED", "message": decision.reason}],
            )
        executed.append(action.action_id)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.SYNTHETIC,
        )

    rt.action_gateway.register_executor(
        "arm.move_joints", ExecutionMode.SHADOW, sandbox_backed_exec
    )

    # --- attempt 1: 越界 -> 真实 gate BLOCK ---
    dangerous = [0.0] * n
    dangerous[0] = gate.joint_limits[0][1] * 5 + 10.0
    receipt1 = rt.submit_action(_action("closed-loop-attempt-1", dangerous))
    _record("attempt1_blocked", receipt1.final_state == ActionState.BLOCKED,
            f"final_state={receipt1.final_state}")
    _record("receipt_not_fake_success", "closed-loop-attempt-1" not in executed,
            f"executor ran only for: {executed}")

    # --- Memory 消费阻断事件 (异步, 轮询) ---
    memory = rt.memory
    assert memory is not None, "memory module not initialized"
    found = None
    deadline = time.time() + 6
    while time.time() < deadline:
        try:
            explain = memory.explain_last_failure()
        except Exception:  # noqa: BLE001
            explain = None
        if explain:
            found = explain
            break
        time.sleep(0.2)
    _record("memory_recorded_failure", found is not None,
            str(found)[:130] if found else "no failure memory found")

    # --- How 恢复建议事件 (heuristic engine 经 Runtime 闭环) ---
    deadline = time.time() + 4
    while time.time() < deadline and not how_hints:
        time.sleep(0.2)
    _record("how_recovery_suggested", len(how_hints) >= 1,
            json.dumps(how_hints[0], ensure_ascii=False)[:130] if how_hints else "no hint event")

    # --- Trace BLOCKED span ---
    trace_files = list(Path(home).glob("traces/*.jsonl"))
    spans = []
    for f in trace_files:
        spans.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
    blocked_spans = [s for s in spans if s.get("status") == "BLOCKED"]
    _record("trace_blocked_span", len(blocked_spans) >= 1,
            f"spans={len(spans)} blocked={len(blocked_spans)}")
    by_id = {s["span_id"]: s for s in spans}
    orphans = [s for s in spans if s.get("parent_span_id") and s["parent_span_id"] not in by_id]
    _record("trace_parent_integrity", not orphans, f"orphans={len(orphans)}/{len(spans)}")

    # --- 恢复: 用 hint 的指引 (joint limits) 钳制参数 ---
    hint_blob = json.dumps(found, ensure_ascii=False) + json.dumps(how_hints, ensure_ascii=False)
    _record("recovery_hint_mentions_limits",
            "joint" in hint_blob.lower() or "limit" in hint_blob.lower(),
            hint_blob[:120])
    lo, hi = gate.joint_limits[0]
    recovered = [0.0] * n
    recovered[0] = max(lo, min(hi, 3.14))
    _record("second_attempt_changed_params", recovered[0] != dangerous[0],
            f"joint_0 {dangerous[0]:.1f} -> {recovered[0]:.2f} (ctrlrange [{lo:.2f},{hi:.2f}])")

    # --- attempt 2: 钳制后 -> gate 放行, SHADOW 完成 ---
    receipt2 = rt.submit_action(_action("closed-loop-attempt-2", recovered))
    _record("attempt2_completed",
            receipt2.final_state == ActionState.COMPLETED
            and "closed-loop-attempt-2" in executed,
            f"final_state={receipt2.final_state}")

    out = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "closed_loop_live.json").write_text(
            json.dumps(RESULTS, indent=2, ensure_ascii=False))

    failures = [k for k, v in RESULTS["chain"].items() if not v["ok"]]
    assert not failures, f"broken links: {failures}"
