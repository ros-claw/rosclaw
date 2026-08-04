"""R4（二次复核 P1-1/P1-3/P1-4/P1-8）：transcript projection、
exactly-once 恢复、生命周期关闭、控制 token、projection owner 过滤。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.operator_socket import OperatorSocketServer, display_hash_for, operator_call
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _turn(text: str) -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=text,
        assistant_message={"role": "assistant", "content": text},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


async def _service(tmp_path: Path, turns: list[ModelTurnResultV1]) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), turns))


class TestTranscriptProjection:
    async def test_blocks_cover_user_assistant_and_seq(self, tmp_path: Path) -> None:
        service = await _service(tmp_path, [_turn("你好，世界"), _turn("第二回合")])
        mission = service.create_mission("transcript 测试")
        await service.send_turn(mission.mission_id, "第一条用户消息")
        await service.send_turn(mission.mission_id, "第二条用户消息")
        page = service.transcript(mission.mission_id)
        kinds = [b["kind"] for b in page["blocks"]]
        assert kinds.count("user") == 2
        assert kinds.count("assistant") == 2
        # 稳定块 ID + 单调 sequence
        ids = [b["block_id"] for b in page["blocks"]]
        assert len(set(ids)) == len(ids)
        seqs = [b["sequence"] for b in page["blocks"]]
        assert seqs == sorted(seqs)
        assert page["latest_sequence"] > 0
        users = [b for b in page["blocks"] if b["kind"] == "user"]
        assert users[0]["text"] == "第一条用户消息"
        await service.close()

    async def test_pagination_before_seq(self, tmp_path: Path) -> None:
        service = await _service(tmp_path, [_turn("x")])
        mission = service.create_mission("分页")
        for i in range(4):
            await service.send_turn(mission.mission_id, f"msg-{i}")
        full = service.transcript(mission.mission_id, limit=100)
        first_user_seq = next(b["sequence"] for b in full["blocks"] if b["kind"] == "user")
        page = service.transcript(mission.mission_id, before_seq=first_user_seq + 1, limit=2)
        assert page["latest_sequence"] == full["latest_sequence"]
        assert all(b["sequence"] <= first_user_seq for b in page["blocks"])
        await service.close()


class TestTranscriptEndpointAndToken:
    async def test_endpoint_requires_token_and_serves_projection(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = await _service(tmp_path, [_turn("hi")])
        mission = service.create_mission("endpoint")
        await service.send_turn(mission.mission_id, "hello")
        app = create_app(service)
        anon = TestClient(app)
        assert anon.get(f"/v2/missions/{mission.mission_id}/transcript").status_code == 401
        assert anon.get("/missions").status_code == 401
        assert anon.get("/health").status_code == 200
        authed = TestClient(app, headers={"x-rosclaw-token": service.control_token})
        page = authed.get(f"/v2/missions/{mission.mission_id}/transcript").json()
        assert page["latest_sequence"] > 0
        assert any(b["kind"] == "user" for b in page["blocks"])
        # SSE 从 latest_sequence 续接 → 不回放历史（exactly-once）。
        with authed.stream(
            "GET",
            f"/v2/missions/{mission.mission_id}/events",
            params={"follow": "false", "after_sequence": page["latest_sequence"]},
        ) as res:
            res.read()
            body = res.text
        assert "data:" not in body, "after latest_sequence 不应再回放任何历史事件"
        await service.close()


class TestLifecycleClose:
    async def test_open_context_manager_closes(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        async with AgentService.open(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn("x")])
        ) as service:
            mission = service.create_mission("lifecycle")
            assert mission.mission_id
        # close 后 store 已关（再访问应抛异常而非泄漏句柄）。
        with pytest.raises(Exception):  # noqa: B017
            service.list_missions()

    async def test_token_file_written_and_removed(self, tmp_path: Path) -> None:
        service = await _service(tmp_path, [_turn("x")])
        path = service.write_control_token_file()
        assert path.exists()
        assert path.read_text() == service.control_token
        assert path.stat().st_mode & 0o777 == 0o600
        await service.close()
        assert not path.exists()


class TestProjectionOwnerFilter:
    async def test_list_without_mission_filters_by_peer(self, tmp_path: Path) -> None:
        """P1-8：省略 mission_id 不得扩大范围（peer 非 owner → 空）。"""
        from tests.agentd.test_operator_socket import _approval_turn

        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_approval_turn] * 2)
        )
        mission = service.create_mission("owner filter")
        await service.send_turn(mission.mission_id, "请求授权")
        assert service.pending_approvals(mission.mission_id)
        sock = tmp_path / "run" / "operator.sock"
        server = OperatorSocketServer(service, sock)
        await server.start()
        try:
            listed = await operator_call(sock, "approvals.list")
            owner = service.principal_for_request(
                service.pending_approvals(mission.mission_id)[0].request_id
            )
            peer_principal = f"user:local:{__import__('os').getuid()}"
            if owner == peer_principal:
                assert listed["approvals"], "owner==peer 时应可见"
            else:
                assert listed["approvals"] == []
            # 显式 mission_id 不受 owner 过滤限制（socket 周界即 ACL）。
            scoped = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
            assert len(scoped["approvals"]) == 1
            assert scoped["approvals"][0]["display_hash"] == display_hash_for(
                service.pending_approvals(mission.mission_id)[0]
            )
        finally:
            await server.stop()
            await service.close()
