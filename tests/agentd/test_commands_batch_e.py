"""批次 E（第一部分）命令测试：workers/grants/body/mode/context/session/
new/retry/failover/thinking/scoped-models。"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.ui.commands import CommandRequestV1


def _answer(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), [_answer] * 50))


def _cmd(name: str, args: dict | None = None, mission_id: str | None = None, key: str = "k") -> CommandRequestV1:
    return CommandRequestV1(
        request_id=f"r_{key}",
        idempotency_key=key,
        command_name=name,
        arguments=args or {},
        mission_id=mission_id,
    )


class TestWorkerCommands:
    async def test_workers_list_and_inspect(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(_cmd("workers"))
            assert result.ok and result.data["workers"]
            ids = {w["worker_id"] for w in result.data["workers"]}
            assert "worker:native:basic" in ids
            inspect = await service.commands.execute(
                _cmd("worker", {"subcommand": "inspect", "worker_id": "worker:native:basic"}, key="k2")
            )
            assert inspect.ok and inspect.data["registry_status"]
            unknown = await service.commands.execute(
                _cmd("worker", {"subcommand": "inspect", "worker_id": "ghost"}, key="k3")
            )
            assert not unknown.ok and unknown.error_code == "unknown_worker"
        finally:
            await service.close()

    async def test_enable_disable_audit(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            disable = await service.commands.execute(
                _cmd("worker", {"subcommand": "disable", "worker_id": "worker:native:basic"}, key="k4")
            )
            assert disable.ok
            assert service._registry.status_of("worker:native:basic") == "DISABLED"
            enable = await service.commands.execute(
                _cmd("worker", {"subcommand": "enable", "worker_id": "worker:native:basic"}, key="k5")
            )
            assert enable.ok
            assert service._registry.status_of("worker:native:basic") == "ENABLED"
        finally:
            await service.close()


class TestGrantsCommands:
    async def test_grants_empty_and_revoke_requires_id(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(_cmd("grants"))
            assert result.ok and result.data["grants"] == []
            revoke = await service.commands.execute(_cmd("revoke", {}, key="k6"))
            assert not revoke.ok and revoke.error_code == "invalid_arguments"
            blob = json.dumps(result.data, ensure_ascii=False)
            assert "signature" not in blob and "permit" not in blob.lower()
        finally:
            await service.close()


class TestBodyDoctorMode:
    async def test_body_honest(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(_cmd("body"))
            assert result.ok and result.data["body_id"]
        finally:
            await service.close()

    async def test_doctor_structured(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(_cmd("doctor"))
            assert result.ok and ("status" in result.data or "ready" in result.data)
        finally:
            await service.close()

    async def test_mode_display_and_no_inplace_upgrade(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("mode 测试")
            show = await service.commands.execute(_cmd("mode", mission_id=mission.mission_id))
            assert show.ok and show.data["mode"] == "SIMULATION"
            upgrade = await service.commands.execute(
                _cmd("mode", {"mode": "REAL"}, mission_id=mission.mission_id, key="k7")
            )
            assert not upgrade.ok and upgrade.error_code == "mode_change_forbidden"
        finally:
            await service.close()


class TestContextSessionNew:
    async def test_context_before_and_after_compile(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("context 测试")
            before = await service.commands.execute(
                _cmd("context", mission_id=mission.mission_id)
            )
            assert before.ok and before.data["compiled"] is False
            await service.send_turn(mission.mission_id, "hi")
            after = await service.commands.execute(
                _cmd("context", mission_id=mission.mission_id, key="k8")
            )
            assert after.ok and after.data["compiled"] is True
            assert "constitution" in after.data["layers"]
            blob = json.dumps(after.data, ensure_ascii=False)
            assert "sk-" not in blob
        finally:
            await service.close()

    async def test_session_and_new(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            created = await service.commands.execute(_cmd("new", {"goal": "新任务"}))
            assert created.ok and created.data["mission_id"].startswith("mis_")
            session = await service.commands.execute(
                _cmd("session", mission_id=created.data["mission_id"], key="k9")
            )
            assert session.ok and session.data["goal"] == "新任务"
            assert session.data["archived"] is False
            empty_goal = await service.commands.execute(_cmd("new", {}, key="k10"))
            assert not empty_goal.ok
        finally:
            await service.close()


class TestRetryFailoverThinking:
    async def test_retry_rejects_nothing_and_resubmits(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("retry 测试")
            nothing = await service.commands.execute(
                _cmd("retry", mission_id=mission.mission_id)
            )
            assert not nothing.ok and nothing.error_code == "nothing_to_retry"
            await service.send_turn(mission.mission_id, "请回答")
            retry = await service.commands.execute(
                _cmd("retry", mission_id=mission.mission_id, key="k11")
            )
            assert retry.ok and retry.data["turn_id"].startswith("turn_")
            await service._turn_tasks[mission.mission_id]
        finally:
            await service.close()

    async def test_failover_and_thinking(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            failover = await service.commands.execute(_cmd("failover"))
            assert failover.ok
            show = await service.commands.execute(_cmd("thinking", key="k12"))
            assert show.ok
            set_effort = await service.commands.execute(
                _cmd("thinking", {"effort": "max"}, key="k13")
            )
            assert set_effort.ok
            assert service._gateway.profile.vendor_parameters["reasoning_effort"] == "max"
            scoped = await service.commands.execute(
                _cmd("scoped-models", {"subcommand": "add", "target": "kimi-code/k3"}, key="k14")
            )
            assert scoped.ok and "kimi-code/k3" in scoped.data["scoped_models"]
        finally:
            await service.close()
