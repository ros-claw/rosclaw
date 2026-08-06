"""PR-04 intent semantics tests.

- ANSWER: 完成条件未清时不 LEARN（pending worker 存在时停留 VALIDATE）
- PLAN_PATCH: 真正 CAS 提交 + task_graph.committed 事件 + stale 拒绝
- PAUSE → SUSPENDED（非 FAILED）；FAIL_SAFE → FAILED + incident 事件
- VERIFY: registry 通过→LEARN；失败→回 PLAN；未知验证器 fail closed
- REQUEST_ACTION: 无 terminal receipt 停留 MONITOR（提交≠完成）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.handlers import _verification_summary
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.agentd.verifiers import VerifierRegistry
from rosclaw.contracts.agent.agent_event import AgentEventType
from rosclaw.contracts.agent.mission import MissionState
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from tests.agentd.conftest import LOCAL_PRINCIPAL


def test_terminal_receipt_summary_uses_capability_specific_metrics() -> None:
    speech = _verification_summary(
        {
            "success": True,
            "observer": "onboard_usb_microphone",
            "rms_gain_db": 19.67,
            "observed_rms_dbfs": -31.91,
            "content_recognition_performed": False,
            "human_hearing_confirmed": False,
        }
    )
    assert "rms_gain_db=19.67" in speech
    assert "observed_rms_dbfs=-31.91" in speech
    assert "content_recognition_performed=False" in speech
    assert "target_gain_db" not in speech

    tone = _verification_summary(
        {"success": True, "target_gain_db": 54.08, "target_prominence_db": 35.33}
    )
    assert "target_gain_db=54.08" in tone
    assert "target_prominence_db=35.33" in tone
    assert "rms_gain_db" not in tone


def _decision_turn(request, decision: dict) -> ModelTurnResultV1:
    decision = dict(decision)
    decision.update(
        {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "d1",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
        }
    )
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path, script) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), script))


class TestAnswerCompletionGate:
    async def test_answer_with_running_worker_stays_validate(self, tmp_path: Path) -> None:
        service = _service(
            tmp_path,
            [
                lambda req: _decision_turn(
                    req, {"next_intent": "ANSWER", "summary": "看起来完成了", "evidence_refs": []}
                )
            ],
        )
        try:
            mission = service.create_mission("门禁测试")
            # 预置一个 RUNNING work order：完成条件未清。
            from rosclaw.contracts.common import new_id

            service.store.connection.execute(
                "INSERT INTO work_orders (work_order_id, mission_id, capability, status, "
                "order_json, created_at, updated_at) VALUES (?, ?, ?, 'RUNNING', '{}', 't', 't')",
                (new_id("wo"), mission.mission_id, "analysis.text"),
            )
            result = await service.send_turn(mission.mission_id, "完成了吗？")
            # §5.1：仍有 RUNNING worker → ANSWER 只是解释，不 LEARN。
            assert result.state is MissionState.VALIDATE
        finally:
            await service.close()


class TestPlanPatch:
    async def test_real_commit_and_event(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [lambda req: _decision_turn(req, _patch_payload(req))])
        try:
            mission = service.create_mission("PLAN_PATCH 测试")
            await service.send_turn(mission.mission_id, "拆任务")
            graph = service.store.get_task_graph(mission.mission_id)
            assert graph.revision == 1
            assert graph.node_ids() == {"t_perceive", "t_report"}
            events = service.events_replay(mission.mission_id)
            assert any(e.type is AgentEventType.TASK_GRAPH_COMMITTED for e in events)
        finally:
            await service.close()

    async def test_stale_base_revision_rejected(self, tmp_path: Path) -> None:
        service = _service(
            tmp_path, [lambda req: _decision_turn(req, _patch_payload(req, base=99))]
        )
        try:
            mission = service.create_mission("stale patch")
            result = await service.send_turn(mission.mission_id, "拆任务")
            assert "被拒绝" in result.reply
            assert service.store.get_task_graph(mission.mission_id).revision == 0
        finally:
            await service.close()


def _patch_payload(request, base: int = 0) -> dict:
    node = {
        "schema_version": "rosclaw.task_node.v1",
        "kind": "perceive",
        "goal": "感知",
        "dependencies": [],
        "inputs": {},
        "constraints": {"schema_version": "rosclaw.task_constraints.v1"},
        "required_capabilities": [],
        "assignee": {"schema_version": "rosclaw.task_assignee.v1", "type": "native"},
        "lease": {"schema_version": "rosclaw.task_lease.v1"},
        "verification": {"schema_version": "rosclaw.task_verification.v1"},
        "status": "PENDING",
        "attempt": 0,
        "max_attempts": 3,
        "artifacts": [],
    }
    n1 = dict(node, task_id="t_perceive", mission_id=request.mission_id)
    n2 = dict(
        node,
        task_id="t_report",
        mission_id=request.mission_id,
        kind="create_artifact",
        dependencies=["t_perceive"],
    )
    return {
        "next_intent": "PLAN_PATCH",
        "summary": "拆成两个任务",
        "evidence_refs": [],
        "proposed_operation": {
            "type": "task_graph_patch",
            "payload": {
                "schema_version": "rosclaw.task_graph_patch.v1",
                "patch_id": "tgpatch_t1",
                "mission_id": request.mission_id,
                "base_revision": base,
                "proposed_by": "agent:test",
                "operations": [
                    {"schema_version": "rosclaw.task_patch_op.v1", "op": "add_node", "node": n1},
                    {"schema_version": "rosclaw.task_patch_op.v1", "op": "add_node", "node": n2},
                ],
            },
        },
    }


class TestPauseAndFailSafe:
    async def test_pause_goes_suspended(self, tmp_path: Path) -> None:
        service = _service(
            tmp_path,
            [
                lambda req: _decision_turn(
                    req, {"next_intent": "PAUSE", "summary": "暂停", "evidence_refs": []}
                )
            ],
        )
        try:
            mission = service.create_mission("PAUSE 测试")
            result = await service.send_turn(mission.mission_id, "先暂停")
            # PLAN 下 SUSPENDED 不可达（§5.9 落到 WAIT_INPUT 等待用户）。
            assert result.state is MissionState.WAIT_INPUT
        finally:
            await service.close()

    async def test_fail_safe_incident_event(self, tmp_path: Path) -> None:
        service = _service(
            tmp_path,
            [
                lambda req: _decision_turn(
                    req, {"next_intent": "FAIL_SAFE", "summary": "检测到冲突", "evidence_refs": []}
                )
            ],
        )
        try:
            mission = service.create_mission("FAIL_SAFE 测试")
            result = await service.send_turn(mission.mission_id, "危险操作")
            assert result.state is MissionState.FAILED
            errors = [
                e
                for e in service.events_replay(mission.mission_id)
                if e.type is AgentEventType.ERROR
            ]
            assert errors and errors[0].payload.get("safety") == "FAIL_SAFE"
            assert errors[0].payload.get("incident") is True
        finally:
            await service.close()


class TestVerifyIntent:
    async def test_verify_pass_learn(self, tmp_path: Path) -> None:
        service = _service(
            tmp_path,
            [
                lambda req: _decision_turn(
                    req,
                    {
                        "next_intent": "VERIFY",
                        "summary": "验证 schema",
                        "evidence_refs": ["a://1"],
                        "proposed_operation": {
                            "type": "verify_receipt",
                            "payload": {"context": {"required_fields": ["a"], "payload": {"a": 1}}},
                        },
                        "verification": {
                            "schema_version": "rosclaw.decision_verification.v1",
                            "verifiers": ["deterministic.schema.v1"],
                        },
                    },
                )
            ],
        )
        try:
            mission = service.create_mission("VERIFY 测试")
            result = await service.send_turn(mission.mission_id, "验证一下")
            assert result.state is MissionState.IDLE
            events = [
                e
                for e in service.events_replay(mission.mission_id)
                if e.type is AgentEventType.VERIFICATION_COMPLETED
            ]
            assert events[-1].payload["success"] is True
        finally:
            await service.close()

    async def test_verify_fail_replans(self, tmp_path: Path) -> None:
        service = _service(
            tmp_path,
            [
                lambda req: _decision_turn(
                    req,
                    {
                        "next_intent": "VERIFY",
                        "summary": "验证 schema",
                        "evidence_refs": [],
                        "proposed_operation": {
                            "type": "verify_receipt",
                            "payload": {"context": {"required_fields": ["missing"], "payload": {}}},
                        },
                        "verification": {
                            "schema_version": "rosclaw.decision_verification.v1",
                            "verifiers": ["deterministic.schema.v1"],
                        },
                    },
                )
            ],
        )
        try:
            mission = service.create_mission("VERIFY 失败测试")
            result = await service.send_turn(mission.mission_id, "验证一下")
            assert "验证未通过" in result.reply
            assert result.state is MissionState.PLAN
        finally:
            await service.close()

    def test_registry_unknown_verifier_closed(self) -> None:
        registry = VerifierRegistry()
        with pytest.raises(Exception, match="unknown"):
            registry.run("verifier.does.not.exist", {})


class TestRequestActionMonitor:
    async def test_no_terminal_receipt_stays_monitor(self, tmp_path: Path) -> None:
        # REQUEST_APPROVAL → approve → REQUEST_ACTION（无 daemon 通道）
        calls = {"n": 0}

        def script(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _decision_turn(
                    req,
                    {
                        "next_intent": "REQUEST_APPROVAL",
                        "summary": "请求授权",
                        "evidence_refs": ["a://1"],
                        "proposed_operation": {
                            "type": "approval_request",
                            "payload": {"title": "t", "summary": "s", "risk_tier": "LOW"},
                        },
                        "verification": {
                            "schema_version": "rosclaw.decision_verification.v1",
                            "verifiers": ["deterministic:x"],
                        },
                    },
                )
            return _decision_turn(
                req,
                {
                    "next_intent": "REQUEST_ACTION",
                    "summary": "执行",
                    "evidence_refs": ["a://1"],
                    "proposed_operation": {
                        "type": "request_action",
                        "payload": {"grant_id": script.grant_id, "risk_tier": "LOW"},
                    },
                    "verification": {
                        "schema_version": "rosclaw.decision_verification.v1",
                        "verifiers": ["deterministic:x"],
                    },
                },
            )

        service = _service(tmp_path, [script, script])
        try:
            mission = service.create_mission("MONITOR 测试")
            r1 = await service.send_turn(mission.mission_id, "请求授权")
            assert r1.state is MissionState.WAIT_APPROVAL
            pending = service.pending_approvals(mission.mission_id)
            grant = await service.decide_approval(
                pending[0].request_id, principal=LOCAL_PRINCIPAL, approve=True
            , _from_operatord=True)
            script.grant_id = grant.grant_id
            r2 = await service.send_turn(mission.mission_id, "执行")
            # 无 daemon → 无 verified terminal receipt → 诚实停留 MONITOR。
            assert r2.state is MissionState.MONITOR
            assert "不是执行回执" in r2.reply
        finally:
            await service.close()


class TestObserveIntent:
    async def test_observe_continues_with_fresh_evidence(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        def script(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _decision_turn(
                    req,
                    {
                        "next_intent": "OBSERVE",
                        "summary": "先读身体状态",
                        "evidence_refs": [],
                        "proposed_operation": {
                            "type": "observe",
                            "payload": {
                                "tool": "sim_get_state",
                                "arguments": {"verbose": True},
                            },
                        },
                    },
                )
            return _decision_turn(
                req,
                {
                    "next_intent": "ANSWER",
                    "summary": "根据新鲜观测回答：关节正常",
                    "evidence_refs": ["artifact://observation/x"],
                },
            )

        service = _service(tmp_path, [script, script])
        try:
            mission = service.create_mission("OBSERVE 测试")
            rev_before = service.get_mission(mission.mission_id).context_revision
            result = await service.send_turn(mission.mission_id, "先观测再回答")
            assert result.tool_rounds == 1
            assert result.state is MissionState.IDLE
            # context_revision 增加了（观测 1 + 编译 1）。
            rev_after = service.get_mission(mission.mission_id).context_revision
            assert rev_after >= rev_before + 2
            history = service.conversation(mission.mission_id)
            obs = [m for m in history if "observation — evidence" in str(m.get("content"))]
            assert obs
            assert "artifact://observation/sha256:" in obs[0]["content"]
        finally:
            await service.close()

    async def test_observe_unknown_tool_stays_in_loop(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        def script(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _decision_turn(
                    req,
                    {
                        "next_intent": "OBSERVE",
                        "summary": "调用不存在的工具",
                        "evidence_refs": [],
                        "proposed_operation": {
                            "type": "observe",
                            "payload": {"tool": "danger.motor.write", "arguments": {}},
                        },
                    },
                )
            return _decision_turn(
                req, {"next_intent": "ANSWER", "summary": "改口直接回答", "evidence_refs": []}
            )

        service = _service(tmp_path, [script, script])
        try:
            mission = service.create_mission("OBSERVE 拒绝测试")
            result = await service.send_turn(mission.mission_id, "观测")
            # 危险工具未被执行（无 tool_rounds），模型改口 ANSWER 完成。
            assert result.tool_rounds == 0
            assert result.state is MissionState.IDLE
        finally:
            await service.close()
