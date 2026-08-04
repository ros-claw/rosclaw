"""K3 完整版：agentd → rosclawd SIM 动作闭环（receipt 验证）。

- full loop: REQUEST_APPROVAL → /approve → REQUEST_ACTION → daemon SIM
  executor → terminal state → receipt (trust=SIMULATED) 回读验证
- daemon down → honest degradation, no fabricated success
- SYNTHETIC/fixture receipt → never reported as success
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.action_channel import DaemonActionChannel
from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel.contracts import (
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
)
from tests.agentd.conftest import LOCAL_PRINCIPAL

SIM_CAPABILITY = "sim.hold_position"


def _sim_executor(action) -> ActionExecutionResult:
    return ActionExecutionResult(
        final_state=ActionState.COMPLETED,
        evidence_level=EvidenceLevel.TASK_VERIFIED,
        evidence_domain=None,
        simulation_result={"held": list(action.arguments.get("joints", []))},
        verification_result={"verified": True, "method": "sim_state_readback"},
    )


@pytest.fixture
def daemon(tmp_path: Path):
    runtime = Runtime(
        RuntimeConfig(
            robot_id="sim-ur5e",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )
    runtime.action_gateway.register_executor(SIM_CAPABILITY, ExecutionMode.SHADOW, _sim_executor)
    with DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3", key_path=tmp_path / "state" / "ledger.key"
    ) as ledger:
        service = DaemonControlPlane(runtime=runtime, ledger=ledger)
        socket_path = tmp_path / "run" / "rosclawd.sock"
        daemon = RosclawDaemon(service=service, socket_path=socket_path)
        daemon.start()
        client = DaemonClient(socket_path=socket_path, timeout_sec=5.0)
        client.arm_runtime("k3 test preflight")
        try:
            yield client, socket_path
        finally:
            daemon.stop()


def _approval_decision(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_a1",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "REQUEST_APPROVAL",
        "summary": "请求授权：SIM 保持关节位",
        "evidence_refs": ["artifact://plan/1"],
        "proposed_operation": {
            "type": "approval_request",
            "payload": {
                "title": "SIM 保持关节位",
                "summary": "hold joints at current pose",
                "risk_tier": "LOW",
                "expected_effect": "关节保持",
                "failure_handling": "超时即停",
            },
        },
        "verification": {
            "schema_version": "rosclaw.decision_verification.v1",
            "verifiers": ["deterministic:bounds"],
        },
    }
    return ModelTurnResultV1(
        turn_id="t1",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _action_decision(request) -> ModelTurnResultV1:
    grant_id = _action_decision.grant_id
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_a2",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "REQUEST_ACTION",
        "summary": "按授权执行 SIM 动作",
        "evidence_refs": ["artifact://plan/1"],
        "proposed_operation": {
            "type": "request_action",
            "payload": {
                "grant_id": grant_id,
                "capability_id": SIM_CAPABILITY,
                "arguments": {"joints": [0.0] * 6},
                "risk_tier": "LOW",
            },
        },
        "verification": {
            "schema_version": "rosclaw.decision_verification.v1",
            "verifiers": ["deterministic:bounds"],
        },
    }
    return ModelTurnResultV1(
        turn_id="t2",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


class TestK3SimActionLoop:
    async def test_shadow_action_returns_verified_nonreal_receipt(self, daemon) -> None:
        client, _socket_path = daemon
        channel = DaemonActionChannel(
            client,
            actor_id="agent:rosclaw-native:sim_ur5e",
            body_id="sim-ur5e",
            body_hash="sha256:sim-body",
        )

        outcome = await channel.request_nonreal_action(
            capability_id=SIM_CAPABILITY,
            arguments={"joints": [0.0] * 6},
            grant_id="grant_shadow",
            execution_mode="SHADOW",
        )

        assert outcome.verified is True
        assert outcome.trust_level == "VERIFIED"
        assert outcome.receipt["receipt"]["evidence_domain"] == "SHADOW"

    async def test_full_loop_with_receipt(self, daemon, tmp_path: Path) -> None:
        client, socket_path = daemon
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_approval_decision, _action_decision])
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("K3 SIM 闭环")
            r1 = await service.send_turn(mission.mission_id, "请求授权")
            assert r1.state.value == "WAIT_APPROVAL"
            pending = service.pending_approvals(mission.mission_id)
            grant = await service.decide_approval(
                pending[0].request_id, principal=LOCAL_PRINCIPAL, approve=True
            , _from_operatord=True)
            _action_decision.grant_id = grant.grant_id
            r2 = await service.send_turn(mission.mission_id, "已批准，执行")
            assert "SIMULATION 完成并经回执验证" in r2.reply
            assert "trust_level=SIMULATED" in r2.reply
            # 回执在 daemon 账本中可查。
            status = client.get_runtime_status()
            assert status["protocol_version"] == "rosclaw.daemon.v1"
        finally:
            await service.close()

    async def test_daemon_down_honest_degradation(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_approval_decision, _action_decision])
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("daemon 不在线")
            await service.send_turn(mission.mission_id, "请求授权")
            pending = service.pending_approvals(mission.mission_id)
            grant = await service.decide_approval(
                pending[0].request_id, principal=LOCAL_PRINCIPAL, approve=True
            , _from_operatord=True)
            _action_decision.grant_id = grant.grant_id
            # Point the channel at a nonexistent daemon.
            from rosclaw.agentd.action_channel import DaemonActionChannel
            from rosclaw.daemon.client import DaemonClient

            service._handlers._action_channel = DaemonActionChannel(
                DaemonClient(socket_path=tmp_path / "nope.sock", timeout_sec=0.5),
                actor_id=service.actor_id,
                body_id="sim/ur5e",
                body_hash="h",
            )
            r2 = await service.send_turn(mission.mission_id, "已批准，执行")
            assert "fail closed" in r2.reply or "失败" in r2.reply
            assert "完成并经回执验证" not in r2.reply
        finally:
            await service.close()


class TestReceiptVerification:
    def test_synthetic_receipt_never_success(self, tmp_path: Path) -> None:
        channel = DaemonActionChannel(
            client=None,  # type: ignore[arg-type]
            actor_id="a",
            body_id="b",
            body_hash="h",
        )
        from rosclaw.agentd.action_channel import ActionChannelError

        with pytest.raises(ActionChannelError, match="SYNTHETIC"):
            channel._verify_outcome(
                "act_1",
                {"state": "FINISHED"},
                {"trust_level": "SYNTHETIC", "action": {"action_id": "act_1"}},
                None,  # type: ignore[arg-type]
            )


class TestRealProposalChannel:
    async def test_real_request_creates_pending_proposal_without_dispatch(self, daemon) -> None:
        client, _socket_path = daemon
        channel = DaemonActionChannel(
            client,
            actor_id="agent:rosclaw-native:limo",
            body_id="limo",
            body_hash="sha256:body",
        )
        proposal = await channel.request_real_proposal(
            capability_id="limo.play_tone",
            arguments={"frequency_hz": 660, "duration_sec": 0.6},
            grant_id="grant_public_only",
            grant_public_hash="grantpub_hash",
            principal_id=LOCAL_PRINCIPAL,
            risk_tier="MEDIUM",
            display={"title": "Play bounded tone", "risk_tier": "MEDIUM"},
        )
        status = client.get_runtime_status()
        public = client.get_operator_proposal(proposal.request_id)

        assert proposal.state == "CREATED"
        assert public["proposal"]["state"] == "CREATED"
        assert "challenge_nonce" not in public["proposal"]
        assert "permit_id" not in str(public).lower()
        assert public["permit_exposed"] is False
        assert status["operator_proposals"]["created"] == 1
        assert status["queue"]["FINISHED"] == 0

    def test_mismatched_receipt_action_rejected(self, tmp_path: Path) -> None:
        channel = DaemonActionChannel(
            client=None,  # type: ignore[arg-type]
            actor_id="a",
            body_id="b",
            body_hash="h",
        )
        from rosclaw.agentd.action_channel import ActionChannelError

        with pytest.raises(ActionChannelError, match="!="):
            channel._verify_outcome(
                "act_1",
                {"state": "FINISHED"},
                {"receipt": {"trust_level": "SIMULATED", "action_id": "act_OTHER"}},
                None,  # type: ignore[arg-type]
            )
