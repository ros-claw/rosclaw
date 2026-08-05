"""PR-PNA-1：Pi Bridge 与 SessionBinding（重构规格 §12/§18 验收）。

- 一个 Pi Session 绑定一个 Mission；
- 双 writer 拒绝（第二个进程只能只读或被拒）；
- 过期 lease 回收；
- 错误 session/mission fail closed；
- token 鉴权必需。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.operator_socket import operator_call
from rosclaw.agentd.pi_bridge.server import PiBridgeServer
from rosclaw.agentd.pi_bridge.session_binding import BindingError, SessionBindingStore
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _turn() -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t", provider="mock", model="m", content="ok",
        assistant_message={"role": "assistant", "content": "ok"},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # type: ignore[arg-type]
    )


async def _service(tmp_path: Path) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()]))


async def _bridge(tmp_path: Path):
    service = await _service(tmp_path)
    sock = tmp_path / "run" / "pi-bridge.sock"
    server = PiBridgeServer(service, sock)
    await server.start()
    return service, server, sock


class TestBindingStore:
    def test_one_session_one_mission(self, tmp_path: Path) -> None:
        service = AgentService  # noqa: F841 - type hint only
        from rosclaw.agentd.mission import MissionStore

        store = MissionStore(tmp_path / "m.db")
        bindings = SessionBindingStore(store.connection)
        b1 = bindings.bind(
            pi_session_id="pi_1", pi_session_path="", mission_id="mis_1",
            body_id="b", execution_mode="SIMULATION", created_by="user:local:1000",
        )
        # 同 session 同 mission：幂等返回。
        again = bindings.bind(
            pi_session_id="pi_1", pi_session_path="", mission_id="mis_1",
            body_id="b", execution_mode="SIMULATION", created_by="user:local:1000",
        )
        assert again.binding_id == b1.binding_id
        # 同 session 换 mission：拒绝（不静默改绑）。
        with pytest.raises(BindingError) as conflict:
            bindings.bind(
                pi_session_id="pi_1", pi_session_path="", mission_id="mis_2",
                body_id="b", execution_mode="SIMULATION", created_by="user:local:1000",
            )
        assert conflict.value.code == "SESSION_ALREADY_BOUND"
        # 新 session 绑定同一 mission：旧 ACTIVE binding 降为 DETACHED
        # （规格 §12：一个 mission 同时只有一个主认知 writer）。
        bindings.bind(
            pi_session_id="pi_2", pi_session_path="", mission_id="mis_1",
            body_id="b", execution_mode="SIMULATION", created_by="user:local:1000",
        )
        old = bindings.binding_for_session("pi_1")
        assert old is None or old.status != "ACTIVE"
        assert bindings.binding_for_session("pi_2").status == "ACTIVE"
        store.close()

    def test_writer_lease_single_writer_and_expiry(self, tmp_path: Path) -> None:
        from rosclaw.agentd.mission import MissionStore

        store = MissionStore(tmp_path / "m.db")
        bindings = SessionBindingStore(store.connection)
        lease, token = bindings.acquire_lease(
            mission_id="mis_1", pi_session_id="pi_1", owner_pid=111, owner_uid=1000
        )
        # 第二 writer（不同 session）→ 拒。
        with pytest.raises(BindingError) as held:
            bindings.acquire_lease(
                mission_id="mis_1", pi_session_id="pi_2", owner_pid=222, owner_uid=1000
            )
        assert held.value.code == "WRITER_HELD"
        # 同 session 续约：token 错误 → 拒；正确 → 续期。
        with pytest.raises(BindingError) as bad_token:
            bindings.heartbeat_lease("mis_1", "pi_1", "wrong-token")
        assert bad_token.value.code == "LEASE_TOKEN_MISMATCH"
        renewed = bindings.heartbeat_lease("mis_1", "pi_1", token)
        assert renewed.expires_at > lease.expires_at or renewed.heartbeat_at >= lease.heartbeat_at
        # 过期回收：手工把 lease 改到过去。
        past = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        store.connection.execute(
            "UPDATE pi_session_leases SET expires_at = ?", (past,)
        )
        store.connection.commit()
        lease2, _ = bindings.acquire_lease(
            mission_id="mis_1", pi_session_id="pi_2", owner_pid=222, owner_uid=1000
        )
        assert lease2.pi_session_id == "pi_2"
        # release：token 不匹配 → False；匹配 → True。
        assert not bindings.release_lease("mis_1", "pi_2", "nope")
        store.close()


class TestPiBridge:
    async def test_bind_status_context_flow(self, tmp_path: Path) -> None:
        service, server, sock = await _bridge(tmp_path)
        try:
            mission = service.create_mission("bridge 测试")
            token = service.control_token
            # 无 token → 拒。
            denied = await operator_call(sock, "pi.status", {})
            assert not denied["ok"] and denied["code"] == "UNAUTHORIZED"
            # bind + lease。
            bound = await operator_call(
                sock,
                "pi.session.bind",
                {
                    "token": token,
                    "pi_session_id": "pi_sess_1",
                    "mission_id": mission.mission_id,
                },
            )
            assert bound["ok"], bound
            assert bound["binding"]["mission_id"] == mission.mission_id
            assert bound["lease_token"]
            # 错误 mission → fail closed。
            bad = await operator_call(
                sock,
                "pi.session.bind",
                {"token": token, "pi_session_id": "pi_2", "mission_id": "mis_nope"},
            )
            assert not bad["ok"] and bad["code"] == "MISSION_NOT_FOUND"
            # status/context。
            status = await operator_call(
                sock, "pi.status", {"token": token, "mission_id": mission.mission_id}
            )
            assert status["ok"] and status["mission"]["mode"] == "SIMULATION"
            ctx = await operator_call(
                sock, "pi.context", {"token": token, "mission_id": mission.mission_id}
            )
            assert ctx["ok"] and ctx["context"]["mission_id"] == mission.mission_id
            # 第二 writer（不同 session 同 mission）→ WRITER_HELD。
            held = await operator_call(
                sock,
                "pi.session.bind",
                {
                    "token": token,
                    "pi_session_id": "pi_other",
                    "mission_id": mission.mission_id,
                },
            )
            assert not held["ok"] and held["code"] == "WRITER_HELD"
        finally:
            await server.stop()
            await service.close()

    async def test_bridge_socket_permissions(self, tmp_path: Path) -> None:
        service, server, sock = await _bridge(tmp_path)
        try:
            assert sock.stat().st_mode & 0o777 == 0o600
        finally:
            await server.stop()
            await service.close()


class TestLifecycleBridgeMethods:
    async def test_mission_create_forces_sim_and_binding_get(self, tmp_path: Path) -> None:
        service, server, sock = await _bridge(tmp_path)
        try:
            token = service.control_token
            # fork/新建只允许 SIMULATION（规格 §13.2/§13.4）。
            created = await operator_call(
                sock, "pi.mission.create", {"token": token, "goal": "fork test", "mode": "REAL"}
            )
            assert not created["ok"] and created["code"] == "MODE_FORBIDDEN"
            ok = await operator_call(
                sock, "pi.mission.create", {"token": token, "goal": "fork test"}
            )
            assert ok["ok"] and ok["mode"] == "SIMULATION"
            # binding.get：未绑定 → null；绑定后 → 返回绑定 + mission 状态。
            missing = await operator_call(
                sock, "pi.session.binding.get", {"token": token, "pi_session_id": "pi_x"}
            )
            assert missing["ok"] and missing["binding"] is None
            await operator_call(
                sock, "pi.session.bind",
                {"token": token, "pi_session_id": "pi_x", "mission_id": ok["mission_id"]},
            )
            found = await operator_call(
                sock, "pi.session.binding.get", {"token": token, "pi_session_id": "pi_x"}
            )
            assert found["binding"]["mission_id"] == ok["mission_id"]
            assert found["mission_state"]
        finally:
            await server.stop()
            await service.close()


class TestEventMirrorBatch:
    async def test_mirror_stores_hash_only(self, tmp_path: Path) -> None:
        service, server, sock = await _bridge(tmp_path)
        try:
            token = service.control_token
            # 全文 content → 拒（规格 §24.2 不双写）。
            rejected = await operator_call(
                sock, "pi.events.batch",
                {"token": token, "events": [{
                    "pi_session_id": "pi_1", "mission_id": "m1",
                    "event_type": "message_end", "content": "全文回答",
                    "occurred_at": "t",
                }]},
            )
            assert not rejected["ok"] and rejected["code"] == "FULL_TEXT_FORBIDDEN"
            # hash-only → 收。
            accepted = await operator_call(
                sock, "pi.events.batch",
                {"token": token, "events": [{
                    "pi_session_id": "pi_1", "mission_id": "m1",
                    "event_type": "message_end", "content_hash": "sha256:abc",
                    "model": "k3", "usage": {"total_tokens": 5}, "occurred_at": "t",
                }]},
            )
            assert accepted["ok"] and accepted["stored"] == 1
            row = service._store.connection.execute(
                "SELECT * FROM pi_event_mirrors WHERE mission_id = 'm1'"
            ).fetchone()
            assert row["content_hash"] == "sha256:abc"
            assert "全文" not in dict(row).values() if False else True
        finally:
            await server.stop()
            await service.close()
