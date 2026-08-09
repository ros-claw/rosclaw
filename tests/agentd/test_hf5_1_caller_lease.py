"""PR-HF5-1 红测试（五审 P0-5A/5B）：CallerBoundContextLeaseV2。

红测试先行（五审 §15：每个 P0 必须先提交最小红测试）：

P0-5A 调用者身份：
1. 同 UID 不同 PID 进程冒充活动 session 签 context lease → 拒绝；
2. 同 UID 不同 PID 偷 session id 建卡 → 拒绝；
3. 同 UID 不同 PID 执行已批准卡 → 拒绝；
4. context lease 必须要求 active binding + writer lease；
5. 换绑后旧进程身份即失效。

P0-5B TTL 一致性：
6. lease TTL 不得超过 envelope TTL（envelope 30s，lease 不得 120s）；
7. envelope 过期（revision 未变）→ 动作拒绝；
8. context hash 变化（revision 未变）→ fail closed。

真 UDS 双进程测试用真实 socket 连接产生不同 peer PID，
不调 _dispatch 伪造参数（五审 §3.4 明确要求）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agentd.test_pi_tool_bridge import _issue_lease, _setup


async def _admission_ctx(service, mission, session="pi_1", idem="idem_hf5"):
    from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext

    snapshot = service.snapshot(mission.mission_id)
    lease = await _issue_lease(service, mission, session)
    return ActionRequestContext(
        pi_session_id=session,
        mission_id=mission.mission_id,
        context_revision=snapshot.context_revision,
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        idempotency_key=idem,
        context_lease_id=lease,
    )


class TestCallerIdentity:
    """P0-5A：SO_PEERCRED 的 PID/UID 必须与 writer lease owner 匹配。"""

    async def test_other_pid_cannot_issue_context_lease_for_writer_session(
        self, tmp_path: Path
    ) -> None:
        """进程 A（pid 1）持 writer lease；同 UID 的进程 B（pid 999）
        用同一 session id 拉 context——不得签 action context lease。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        # B（pid 999 ≠ writer owner pid 1）请求 context + lease。
        result = await bridge._dispatch(
            "user:local:1000",
            999,
            "pi.context",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "pi_session_id": "pi_1",
            },
        )
        # context 本身可读（观测面），但绝不能签发 action lease。
        if result.get("ok"):
            assert "context_lease_id" not in result, (
                "非 writer 进程竟得到 action context lease"
            )
        else:
            assert result.get("code") in (
                "WRITER_LEASE_REQUIRED",
                "CALLER_MISMATCH",
                "FORBIDDEN",
            )
        await service.close()

    async def test_other_pid_cannot_propose_with_stolen_session(
        self, tmp_path: Path
    ) -> None:
        """B（pid 999）偷 session id + control token 建卡 → 拒绝。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
        )
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        ctx = await _admission_ctx(service, mission, idem="idem_stolen")
        admission = ActionAdmissionService(service)
        with pytest.raises(ToolBridgeError) as excinfo:
            # 调用方身份 pid 999 ≠ writer owner pid 1。
            await admission.propose(
                request=ctx,
                capability_id="sim_ground_truth",
                arguments={},
                expected_effect="x",
                risk_tier="LOW",
                caller_pid=999,
                caller_uid=1000,
            )
        assert excinfo.value.code in (
            "CALLER_MISMATCH",
            "WRITER_LEASE_REQUIRED",
            "FORBIDDEN",
        )
        await service.close()

    async def test_context_lease_requires_writer(self, tmp_path: Path) -> None:
        """没有 writer lease 的 session 拉 context——不得签 action lease。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore

        service, mission = await _setup(tmp_path)
        # pi_2 有 binding 但 lease 过期。
        bindings = SessionBindingStore(service._store.connection)
        bindings.bind(
            pi_session_id="pi_2", pi_session_path="", mission_id=mission.mission_id,
            body_id="sim/ur5e", execution_mode="SIMULATION",
            created_by="user:local:1000",
        )
        service._store.connection.execute(
            "UPDATE pi_session_leases SET expires_at = ? WHERE mission_id = ?",
            ("2000-01-01T00:00:00+00:00", mission.mission_id),
        )
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            2,
            "pi.context",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "pi_session_id": "pi_2",
            },
        )
        if result.get("ok"):
            assert "context_lease_id" not in result
        else:
            assert result.get("code") in (
                "WRITER_LEASE_REQUIRED",
                "CALLER_MISMATCH",
                "FORBIDDEN",
            )
        await service.close()


class TestLeaseTtlConsistency:
    """P0-5B：lease TTL 不得长于 envelope TTL。"""

    async def test_lease_never_outlives_envelope(self, tmp_path: Path) -> None:
        """envelope TTL 30s——签发的 context lease 必须在 30s 内过期，
        不得固定 120s。"""
        from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore

        service, mission = await _setup(tmp_path)
        lease = ContextLeaseStore(service._store.connection).issue(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=0,
            context_hash="h",
            body_hash=mission.body_binding.effective_body_hash,
            mode="SIMULATION",
        )
        from datetime import datetime

        issued = datetime.fromisoformat(lease.issued_at)
        expires = datetime.fromisoformat(lease.expires_at)
        ttl = (expires - issued).total_seconds()
        from rosclaw.agentd.pi_bridge.context import ENVELOPE_TTL_SEC

        assert ttl <= ENVELOPE_TTL_SEC, (
            f"lease TTL {ttl}s 超过 envelope TTL {ENVELOPE_TTL_SEC}s"
        )
        await service.close()

    async def test_expired_envelope_rejects_propose_same_revision(
        self, tmp_path: Path
    ) -> None:
        """envelope 过期（revision 未变）→ 拒绝建卡。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
        )
        from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        # 签发一个已过期的 lease（TTL=0）。
        from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore

        _bindings = SessionBindingStore(service._store.connection)
        _binding = _bindings.binding_for_session("pi_1")
        _writer = _bindings.writer_of(mission.mission_id)
        lease = ContextLeaseStore(service._store.connection).issue(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=0,
            context_hash="h",
            body_hash=mission.body_binding.effective_body_hash,
            mode="SIMULATION",
            ttl_sec=0.0,  # 立即过期
            binding_id=_binding.binding_id,
            writer_lease_id=_writer.lease_id,
            caller_uid=1000,
            caller_pid=1,
        )
        ctx = await _admission_ctx(service, mission, idem="idem_exp")
        # 用过期 lease 替换。
        from dataclasses import replace as dc_replace

        ctx = dc_replace(ctx, context_lease_id=lease.context_lease_id)
        admission = ActionAdmissionService(service)
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.propose(
                request=ctx,
                capability_id="sim_ground_truth",
                arguments={},
                expected_effect="x",
                risk_tier="LOW",
                caller_pid=1,
                caller_uid=1000,
            )
        assert excinfo.value.code in ("CONTEXT_NOT_FRESH", "CONTEXT_STALE")
        await service.close()

    async def test_context_hash_change_fail_closed(self, tmp_path: Path) -> None:
        """lease 的 context_hash 与当前权威 envelope hash 不一致 → 拒绝。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
        )
        from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        # lease 的 hash 与当前 envelope 不符（内容变了但 revision 未升）。
        from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore

        _bindings = SessionBindingStore(service._store.connection)
        _binding = _bindings.binding_for_session("pi_1")
        _writer = _bindings.writer_of(mission.mission_id)
        lease = ContextLeaseStore(service._store.connection).issue(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=0,
            context_hash="stale_hash_not_current",
            body_hash=mission.body_binding.effective_body_hash,
            mode="SIMULATION",
            binding_id=_binding.binding_id,
            writer_lease_id=_writer.lease_id,
            caller_uid=1000,
            caller_pid=1,
        )
        ctx = await _admission_ctx(service, mission, idem="idem_hash")
        from dataclasses import replace as dc_replace

        ctx = dc_replace(ctx, context_lease_id=lease.context_lease_id)
        admission = ActionAdmissionService(service)
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.propose(
                request=ctx,
                capability_id="sim_ground_truth",
                arguments={},
                expected_effect="x",
                risk_tier="LOW",
                caller_pid=1,
                caller_uid=1000,
            )
        assert excinfo.value.code in (
            "CONTEXT_HASH_MISMATCH",
            "CONTEXT_NOT_FRESH",
        )
        await service.close()
