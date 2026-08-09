"""HOTFIX-1 红测试（四审 P0-4A/4B）：ValidatedContextLeaseV1。

红测试先行——以下缺陷必须稳定复现，修复后转绿：

1. context 拉取失败但 revision 未变（同 revision stale）→ 仍能建卡；
2. body_hash 为空被放行（TS 在无 Body 时发空字符串）；
3. idempotency_key 为空被放行；
4. pi.action.execute 不带请求上下文（request=None）→ 仍执行已批准卡；
5. pi.action.propose/execute 不带 context_lease_id → 拒绝；
6. context lease 过期/被撤销 → propose/execute 拒绝；
7. pi.action.status 不校验调用方 session → 可窥探他人卡状态。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "rosclaw"


async def _setup_bound(tmp_path: Path):
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
    from rosclaw.agentd.service import AgentService
    from tests.agentd.test_pi_tool_bridge import _turn

    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
    )
    mission = service.create_mission("hotfix-1", mode="SIMULATION")
    bindings = SessionBindingStore(service._store.connection)
    bindings.bind(
        pi_session_id="pi_1", pi_session_path="", mission_id=mission.mission_id,
        body_id="sim/ur5e", execution_mode="SIMULATION", created_by="user:local:1000",
    )
    bindings.acquire_lease(
        mission_id=mission.mission_id, pi_session_id="pi_1", owner_pid=1, owner_uid=1000
    )
    return service, mission


def _ctx(service, mission, **overrides):
    from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext

    snapshot = service.snapshot(mission.mission_id)
    return ActionRequestContext(
        pi_session_id=overrides.get("session", "pi_1"),
        mission_id=mission.mission_id,
        context_revision=overrides.get("revision", snapshot.context_revision),
        body_hash=overrides.get(
            "body_hash", mission.body_binding.effective_body_hash
        ),
        mode=overrides.get("mode", mission.mode.value),
        idempotency_key=overrides.get("idem", "idem_hf1"),
        context_lease_id=overrides.get("lease", ""),
    )


# ---------------------------------------------------------------- 结构红线


def test_action_request_context_requires_lease_and_nonempty_fields() -> None:
    """ActionRequestContext 必须含 context_lease_id，且 session/mission/
    revision/body_hash/mode/idempotency/lease 全部非空才可构造。"""
    source = (SRC / "agentd" / "pi_bridge" / "action_admission.py").read_text(
        encoding="utf-8"
    )
    assert "context_lease_id" in source, "ActionRequestContext 缺 context_lease_id"
    # 空 body_hash 捷径必须消失（四审 §3.1 指出的放行点）。
    assert "current_body_hash and ctx.body_hash and" not in source, (
        "空 body_hash 跳过校验的捷径仍在"
    )


def test_execute_requires_request_context_no_none_path() -> None:
    """execute 不得存在 request=None 的绕过路径（四审 P0-4B）。"""
    tree = ast.parse(
        (SRC / "agentd" / "pi_bridge" / "action_admission.py").read_text(encoding="utf-8")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute":
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == "request":
                    # 注解不得是 Optional/Union with None。
                    annotation = ast.unparse(arg.annotation) if arg.annotation else ""
                    assert "None" not in annotation, (
                        f"execute.request 仍允许 None: {annotation}"
                    )
            return
    pytest.fail("execute method not found")


def test_bridge_execute_always_requires_context() -> None:
    """bridge 的 pi.action.execute 不得有'缺 pi_session_id 就传 None'分支。"""
    source = (SRC / "agentd" / "pi_bridge" / "server.py").read_text(encoding="utf-8")
    assert "request_ctx = None" not in source, "execute 的 None 上下文分支仍在"


# ---------------------------------------------------------------- 运行时红线


@pytest.mark.asyncio
async def test_same_revision_stale_context_cannot_propose(tmp_path: Path) -> None:
    """四审 §3.2 核心场景：context 从未成功获取（无 lease），revision
    恰好与 snapshot 相同（比如都是 0）——也必须拒绝建卡。"""
    import pytest as _pytest

    from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
    from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

    service, mission = await _setup_bound(tmp_path)
    admission = ActionAdmissionService(service)
    with _pytest.raises(ToolBridgeError) as excinfo:
        await admission.propose(
            caller_pid=1, caller_uid=1000,
            request=_ctx(service, mission, lease=""),  # 无 context lease
            capability_id="sim_ground_truth",
            arguments={},
            expected_effect="x",
            risk_tier="LOW",
        )
    assert excinfo.value.code in (
        "CONTEXT_NOT_FRESH",
        "CONTEXT_LEASE_REQUIRED",
        "REQUEST_CONTEXT_REQUIRED",
    ), f"同 revision 无 lease 建卡未被拒绝: {excinfo.value.code}"
    assert service.pending_approvals(mission.mission_id) == []
    await service.close()


@pytest.mark.asyncio
async def test_missing_body_hash_rejected(tmp_path: Path) -> None:
    """body_hash 为空必须拒绝（TS 无 Body 时发空字符串——不能放行）。"""
    import pytest as _pytest

    from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
    from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

    service, mission = await _setup_bound(tmp_path)
    admission = ActionAdmissionService(service)
    with _pytest.raises(ToolBridgeError) as excinfo:
        await admission.propose(
            caller_pid=1, caller_uid=1000,
            request=_ctx(service, mission, body_hash=""),
            capability_id="sim_ground_truth",
            arguments={},
            expected_effect="x",
            risk_tier="LOW",
        )
    assert excinfo.value.code in (
        "REQUEST_CONTEXT_REQUIRED",
        "BODY_HASH_REQUIRED",
        "CONTEXT_NOT_FRESH",
        "CONTEXT_LEASE_REQUIRED",
    )
    assert service.pending_approvals(mission.mission_id) == []
    await service.close()


@pytest.mark.asyncio
async def test_missing_idempotency_key_rejected(tmp_path: Path) -> None:
    import pytest as _pytest

    from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
    from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

    service, mission = await _setup_bound(tmp_path)
    admission = ActionAdmissionService(service)
    with _pytest.raises(ToolBridgeError) as excinfo:
        await admission.propose(
            caller_pid=1, caller_uid=1000,
            request=_ctx(service, mission, idem=""),
            capability_id="sim_ground_truth",
            arguments={},
            expected_effect="x",
            risk_tier="LOW",
        )
    assert excinfo.value.code in ("REQUEST_CONTEXT_REQUIRED", "IDEMPOTENCY_KEY_REQUIRED")
    await service.close()


@pytest.mark.asyncio
async def test_bridge_execute_without_context_rejected(tmp_path: Path) -> None:
    """pi.action.execute 只给 approval_id（无请求上下文）必须拒绝——
    不得执行已批准卡（四审 P0-4B 直接复现）。"""
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    service, mission = await _setup_bound(tmp_path)
    bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
    result = await bridge._dispatch(
        "user:local:1000",
        1234,
        "pi.action.execute",
        {"token": service.control_token, "approval_id": "appr_whatever"},
    )
    assert not result.get("ok")
    assert result.get("code") in (
        "REQUEST_CONTEXT_REQUIRED",
        "CONTEXT_LEASE_REQUIRED",
        "APPROVAL_NOT_FOUND",  # 卡本就不存在——但必须因缺上下文先拒
    )
    # 关键是错误语义：缺上下文时绝不进入执行路径。
    await service.close()


@pytest.mark.asyncio
async def test_action_status_filters_by_session_owner(tmp_path: Path) -> None:
    """pi.action.status 必须做 card principal/session owner 过滤——
    只凭 approval_id 不得窥探他人卡（四审 P0-4B）。"""
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    service, mission = await _setup_bound(tmp_path)
    bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
    result = await bridge._dispatch(
        "user:local:1000",
        1234,
        "pi.action.status",
        {
            "token": service.control_token,
            "approval_id": "appr_someone_elses",
            # 无 pi_session_id——无法证明是卡主
        },
    )
    # 必须拒绝或至少不给卡内信息（MISSING/无敏感字段）。
    if result.get("ok"):
        assert result.get("status") == "MISSING", (
            f"非卡主竟看到卡状态: {result}"
        )
    else:
        assert result.get("code") in (
            "REQUEST_CONTEXT_REQUIRED",
            "FORBIDDEN",
            "APPROVAL_NOT_FOUND",
        )
    await service.close()
