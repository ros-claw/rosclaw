"""PR-EIGHT-4 红测试（八审 §1.5/P0-4）：JIT Context Revalidation。

红测试先行——当前 30s wall-clock 租约在模型思考期间过期即
CONTEXT_NOT_FRESH，用户被迫回复"继续"触发新回合（八审真实会话
出现 12 次）。上下文 freshness 应保护"机器人状态是否已变化"，
不应惩罚"模型想了多久"：

- SIM + 仅 TTL 过期 + 权威上下文未变（hash/revision/binding/
  writer/caller 全同）→ 透明续租，模型/用户无感；
- 过期 + revision/hash 变化 → 结构化 NEEDS_REPLAN（不是笼统
  CONTEXT_NOT_FRESH）；
- 被撤销 lease 绝不复活；REAL/SHADOW 不自动续租。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _setup


async def _propose(service, mission, lease_id: str, *, idem: str):
    from rosclaw.agentd.pi_bridge.action_admission import (
        ActionAdmissionService,
        ActionRequestContext,
    )

    snapshot = service.snapshot(mission.mission_id)
    ctx = ActionRequestContext(
        pi_session_id="pi_1",
        mission_id=mission.mission_id,
        context_revision=snapshot.context_revision,
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        idempotency_key=idem,
        context_lease_id=lease_id,
    )
    return await ActionAdmissionService(service).propose(
        caller_pid=1,
        caller_uid=1000,
        request=ctx,
        capability_id="sim_ground_truth",
        arguments={},
        expected_effect="jit probe",
        risk_tier="LOW",
    )


class TestJitRevalidation:
    async def test_expired_lease_unchanged_context_auto_renews(
        self, tmp_path: Path
    ) -> None:
        """SIM + 仅 TTL 过期 + 上下文未变 → 透明续租继续（无
        CONTEXT_NOT_FRESH）。"""
        service, mission = await _setup(tmp_path)
        lease_id = await _issue_lease(service, mission)
        service._store.connection.execute(
            "UPDATE pi_context_leases SET expires_at = '2000-01-01T00:00:00+00:00' "
            "WHERE context_lease_id = ?",
            (lease_id,),
        )
        result = await _propose(service, mission, lease_id, idem="idem_jit_1")
        assert result.get("approval_id"), f"未续租反被拒: {result}"
        # 续租产生新的有效 lease（旧的不复活）。
        rows = service._store.connection.execute(
            "SELECT context_lease_id, revoked, expires_at FROM pi_context_leases "
            "WHERE mission_id = ? AND revoked = 0",
            (mission.mission_id,),
        ).fetchall()
        assert rows, "续租后无有效 lease"
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        assert any(r[2] > now for r in rows), f"续租的 lease 仍过期: {rows}"
        await service.close()

    async def test_expired_lease_changed_revision_needs_replan(
        self, tmp_path: Path
    ) -> None:
        """过期 + context revision 已变 → NEEDS_REPLAN（结构化，不是
        笼统 CONTEXT_NOT_FRESH，更不该默默放行）。"""
        service, mission = await _setup(tmp_path)
        lease_id = await _issue_lease(service, mission)
        service._store.connection.execute(
            "UPDATE pi_context_leases SET expires_at = '2000-01-01T00:00:00+00:00' "
            "WHERE context_lease_id = ?",
            (lease_id,),
        )
        # 权威 revision 前进（关键状态变化）。
        service._store.bump_context_revision(mission.mission_id)
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        try:
            result = await _propose(service, mission, lease_id, idem="idem_jit_2")
        except ToolBridgeError as exc:
            assert exc.code == "NEEDS_REPLAN", f"错误码应为 NEEDS_REPLAN: {exc.code}"
            await service.close()
            return
        assert result.get("code") == "NEEDS_REPLAN" or result.get("replan"), (
            f"revision 变化后竟直接放行: {result}"
        )
        await service.close()

    async def test_revoked_lease_never_resurrects(self, tmp_path: Path) -> None:
        """被撤销的 lease（换绑/重建失效）绝不自动续租。"""
        service, mission = await _setup(tmp_path)
        lease_id = await _issue_lease(service, mission)
        service._store.connection.execute(
            "UPDATE pi_context_leases SET revoked = 1 WHERE context_lease_id = ?",
            (lease_id,),
        )
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        try:
            result = await _propose(service, mission, lease_id, idem="idem_jit_3")
            assert not result.get("ok", True), f"撤销 lease 竟放行: {result}"
        except ToolBridgeError as exc:
            assert exc.code in (
                "CONTEXT_NOT_FRESH",
                "CONTEXT_LEASE_MISMATCH",
                "CONTEXT_LEASE_REQUIRED",
            ), f"撤销 lease 错误码不对: {exc.code}"
        await service.close()

    async def test_fresh_lease_unaffected(self, tmp_path: Path) -> None:
        """对照：未过期 lease 走原路径（JIT 不改变正常语义）。"""
        service, mission = await _setup(tmp_path)
        lease_id = await _issue_lease(service, mission)
        result = await _propose(service, mission, lease_id, idem="idem_jit_4")
        assert result.get("approval_id"), result
        await service.close()
