"""批次 B 测试（补充实施文档 §5）：UI contracts + EventBus 完整性。

- agent.started / agent.settled 总能发出（成功与失败路径）
- turn.ended / turn.cancel.requested
- 命令注册表：capabilities、幂等、unknown/disabled、archive 后拒绝新 turn
- snapshot：权威状态、无 secret、sequence watermark
- SSE Last-Event-ID 断线恢复
- interaction：respond/幂等/过期/secret redaction/不能伪造授权
- 全事件 secret scan + sequence 无缺口
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.agent_event import AgentEventType
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


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


def _fail(request) -> ModelTurnResultV1:
    from rosclaw.agentd.models.gateway import ModelGatewayError

    raise ModelGatewayError("http_error", "HTTP 500: boom")


def _service(tmp_path: Path, script) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), script))


def _events(service: AgentService, mission_id: str):
    return service.events_replay(mission_id, after_sequence=0, limit=10_000)


class TestAgentLifecycleEvents:
    async def test_settled_emitted_on_success(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            mission = service.create_mission("生命周期")
            turn_id = await service.submit_turn_v2(mission.mission_id, "你好")
            await service._turn_tasks[mission.mission_id]
            types = [e.type for e in _events(service, mission.mission_id)]
            assert AgentEventType.TURN_ACCEPTED in types
            assert AgentEventType.AGENT_STARTED in types
            assert AgentEventType.TURN_ENDED in types
            assert AgentEventType.AGENT_SETTLED in types
            # settled 是最后一个 agent 生命周期事件。
            lifecycle = [
                e.type
                for e in _events(service, mission.mission_id)
                if e.type.value.startswith(("agent.", "turn."))
            ]
            assert lifecycle[-1] is AgentEventType.AGENT_SETTLED
            # turn_id 贯穿。
            settled = [e for e in _events(service, mission.mission_id)
                       if e.type is AgentEventType.AGENT_SETTLED]
            assert settled[0].turn_id == turn_id
        finally:
            await service.close()

    async def test_settled_emitted_on_failure(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_fail] * 5)
        try:
            mission = service.create_mission("失败路径")
            await service.submit_turn_v2(mission.mission_id, "会失败")
            await service._turn_tasks[mission.mission_id]
            events = _events(service, mission.mission_id)
            types = [e.type for e in events]
            # 模型错误被 loop 吸收为 reply（不抛异常）→ 正常 settled；
            # 但无论成败，settled 必须存在。
            assert AgentEventType.AGENT_SETTLED in types
        finally:
            await service.close()

    async def test_cancel_emits_event(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            mission = service.create_mission("取消事件")
            await service.cancel(mission.mission_id)
            types = [e.type for e in _events(service, mission.mission_id)]
            assert AgentEventType.TURN_CANCEL_REQUESTED in types
        finally:
            await service.close()

    async def test_sequence_has_no_gaps(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            mission = service.create_mission("sequence 连续")
            await service.submit_turn_v2(mission.mission_id, "hi")
            await service._turn_tasks[mission.mission_id]
            seqs = [e.sequence for e in _events(service, mission.mission_id)]
            assert seqs == list(range(1, len(seqs) + 1))
        finally:
            await service.close()


class TestCommandRegistry:
    async def test_capabilities_and_unknown_command(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path, [_answer] * 5)
        try:
            client = TestClient(create_app(service))
            mission = service.create_mission("命令测试")
            caps = client.get(f"/v1/capabilities?mission_id={mission.mission_id}").json()
            names = {c["name"] for c in caps["commands"]}
            assert {"compact", "cancel", "rename", "archive", "status", "tools"} <= names
            for spec in caps["commands"]:
                assert spec["owner"] != "SAFETY_CONTROL", "/approve /estop 不在通用注册表"
            r = client.post(
                f"/v1/missions/{mission.mission_id}/commands",
                json={
                    "request_id": "r1",
                    "idempotency_key": "k1",
                    "command_name": "/nosuch",
                    "arguments": {},
                },
            )
            body = r.json()
            assert not body["ok"] and body["error_code"] == "unknown_command"
        finally:
            await service.close()

    async def test_rename_archive_and_idempotency(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path, [_answer] * 5)
        try:
            client = TestClient(create_app(service))
            mission = service.create_mission("重命名")
            r1 = client.post(
                f"/v1/missions/{mission.mission_id}/commands",
                json={
                    "request_id": "r2",
                    "idempotency_key": "k-rename",
                    "command_name": "rename",
                    "arguments": {"name": "新名字"},
                },
            ).json()
            assert r1["ok"]
            # 同一 idempotency_key 重放 → 相同结果，不重复执行。
            r2 = client.post(
                f"/v1/missions/{mission.mission_id}/commands",
                json={
                    "request_id": "r3",
                    "idempotency_key": "k-rename",
                    "command_name": "rename",
                    "arguments": {"name": "新名字"},
                },
            ).json()
            assert r2 == r1
            assert service.store.mission_meta(mission.mission_id)["display_name"] == "新名字"
            renamed = [e for e in _events(service, mission.mission_id)
                       if e.type is AgentEventType.MISSION_RENAMED]
            assert len(renamed) == 1, "重放不得产生第二个事件"
            # archive → 只读。
            ra = client.post(
                f"/v1/missions/{mission.mission_id}/commands",
                json={
                    "request_id": "r4",
                    "idempotency_key": "k-archive",
                    "command_name": "archive",
                    "arguments": {},
                },
            ).json()
            assert ra["ok"]
            from rosclaw.contracts.common import ValidationError

            with pytest.raises(ValidationError, match="archived"):
                await service.send_turn(mission.mission_id, "归档后")
        finally:
            await service.close()

    async def test_tools_command_lists_catalog(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path, [_answer] * 5)
        try:
            client = TestClient(create_app(service))
            mission = service.create_mission("工具列表")
            r = client.post(
                f"/v1/missions/{mission.mission_id}/commands",
                json={
                    "request_id": "r5",
                    "idempotency_key": "k-tools",
                    "command_name": "tools",
                    "arguments": {},
                },
            ).json()
            assert r["ok"]
            ids = {t["tool_id"] for t in r["data"]["tools"]}
            assert "sim_get_state" in ids and "sim_body_profile" in ids
        finally:
            await service.close()


class TestSnapshot:
    async def test_snapshot_authoritative_and_clean(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path, [_answer] * 5)
        try:
            client = TestClient(create_app(service))
            mission = service.create_mission("快照测试")
            await service.send_turn(mission.mission_id, "hi")
            snap = client.get(f"/v1/missions/{mission.mission_id}/snapshot").json()
            assert snap["schema_version"] == "rosclaw.ui.mission_snapshot.v1"
            assert snap["mission_id"] == mission.mission_id
            assert snap["state"] == "IDLE"
            assert snap["mode"] == "SIMULATION"
            assert snap["last_event_sequence"] >= 1
            assert not snap["turn_in_flight"]
            # secret scan：快照文本不得含任何 secret 形态。
            blob = json.dumps(snap, ensure_ascii=False).lower()
            for pattern in ("sk-", "api_key", "secret", "permit", "password", "bearer"):
                assert pattern not in blob, f"snapshot leaks {pattern}"
        finally:
            await service.close()

    async def test_snapshot_404(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path, [_answer] * 5)
        try:
            client = TestClient(create_app(service))
            assert client.get("/v1/missions/mis_ghost/snapshot").status_code == 404
        finally:
            await service.close()


class TestSseResume:
    async def test_last_event_id_header_resumes(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path, [_answer] * 5)
        try:
            client = TestClient(create_app(service))
            mission = service.create_mission("断线恢复")
            await service.send_turn(mission.mission_id, "hi")
            events = _events(service, mission.mission_id)
            assert len(events) >= 2
            mid = events[len(events) // 2].sequence
            with client.stream(
                "GET",
                f"/v2/missions/{mission.mission_id}/events?follow=false",
                headers={"Last-Event-ID": str(mid)},
            ) as response:
                assert response.status_code == 200
                frames = [line for line in response.iter_lines() if line.startswith("data: ")]
            resumed = [json.loads(f[len("data: "):]) for f in frames]
            assert resumed, "replay after Last-Event-ID must return later events"
            assert all(e["sequence"] > mid for e in resumed)
            expected = [e.sequence for e in events if e.sequence > mid]
            assert [e["sequence"] for e in resumed] == expected
        finally:
            await service.close()


class TestInteractions:
    async def test_respond_idempotent_and_no_authority(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            req = service.interactions.create(
                "select",
                title="选择目标",
                options=[{"value": "a"}, {"value": "b"}],
            )
            r1 = service.interactions.respond(req.interaction_id, value="a", idempotency_key="i1")
            r2 = service.interactions.respond("whatever", value="a", idempotency_key="i1")
            assert r1 == r2
            # generic interaction 不产生任何授权。
            assert service.list_grants() == []
            assert service.pending_approvals() == []
        finally:
            await service.close()

    async def test_secret_value_redacted(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            req = service.interactions.create("input", title="API key", masked=True)
            record = service.interactions.respond(req.interaction_id, value="sk-should-not-store")
            assert record["value"] == "<redacted-secret>"
        finally:
            await service.close()

    async def test_invalid_option_rejected(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            from rosclaw.contracts.common import ValidationError

            req = service.interactions.create(
                "select", options=[{"value": "a"}]
            )
            with pytest.raises(ValidationError, match="not among"):
                service.interactions.respond(req.interaction_id, value="z")
        finally:
            await service.close()


class TestEventSecretScan:
    async def test_no_secret_shapes_in_any_event(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 5)
        try:
            mission = service.create_mission("事件扫描")
            await service.submit_turn_v2(mission.mission_id, "hi")
            await service._turn_tasks[mission.mission_id]
            for event in _events(service, mission.mission_id):
                blob = json.dumps(event.model_dump(mode="json"), ensure_ascii=False).lower()
                for pattern in (
                    "sk-",
                    "api_key",
                    "password",
                    "bearer ",
                    "private_key",
                    "permit_secret",
                ):
                    assert pattern not in blob, f"{event.type} leaks {pattern}"
        finally:
            await service.close()
