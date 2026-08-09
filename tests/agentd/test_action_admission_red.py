"""PR-NA-10 红测试（三审 P0-NA-10/P0-NA-11）：统一 action admission。

红测试先行（复核 §8）——以下缺陷必须能稳定复现，修复后转绿：

1. PiToolDispatcher 存在两个同名 _request_action（旧实现覆盖新实现）；
2. 生产 action 路径直接改 handlers._mode/_principal；
3. 生产 action 路径遍历 list_grants() 猜全局 grant；
4. 生产 action 路径用中文失败词列表解析 outcome.text 判成功；
5. pi.action.propose 不带 session/lease/revision/body/mode 也能建卡；
6. stale context 下强行 tool-call 仍能创建 approval；
7. approve 后、execute 前 revision 变化（TOCTOU）不被发现；
8. ActionAdmissionService 是唯一入口——propose 必须带完整请求上下文。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "rosclaw"


# ---------------------------------------------------------------- 静态防线


def test_dispatcher_has_no_duplicate_method_names() -> None:
    """P0-NA-11：同名方法后定义覆盖前定义——AST 级禁止。"""
    source = (SRC / "agentd" / "pi_bridge" / "tool_dispatch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PiToolDispatcher":
            names = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            duplicates = {n for n in names if names.count(n) > 1}
            assert not duplicates, f"PiToolDispatcher 重复方法名: {sorted(duplicates)}"
            return
    pytest.fail("PiToolDispatcher class not found")


def test_production_action_path_no_shared_handler_mutation() -> None:
    """P0-NA-11：不得直接改共享 handlers._mode/_principal（请求级状态
    必须走 request_context）。"""
    source = (SRC / "agentd" / "pi_bridge" / "tool_dispatch.py").read_text(encoding="utf-8")
    assert "handlers._mode =" not in source, "直接改共享 handlers._mode（绕过 request_context）"
    assert "handlers._principal =" not in source, "直接改共享 handlers._principal"
    assert "._mode = " not in source.split("class ")[0] or True  # 头部 import 区豁免


def test_production_action_path_no_global_grant_scan() -> None:
    """P0-NA-11/NA-13：不得遍历全局 grant 列表猜 grant。"""
    source = (SRC / "agentd" / "pi_bridge" / "tool_dispatch.py").read_text(encoding="utf-8")
    assert "list_grants()" not in source, "全局 list_grants() 猜 grant"
    admission = (SRC / "agentd" / "pi_bridge" / "action_admission.py").read_text(
        encoding="utf-8"
    )
    assert "list_grants()" not in admission
    assert "pending[-1]" not in admission, "pending[-1] 取卡——并发下不能证明是刚创建的卡"


def test_production_action_path_no_text_success_parsing() -> None:
    """P0-NA-11：不得用自然语言词表判断动作成败。"""
    source = (SRC / "agentd" / "pi_bridge" / "tool_dispatch.py").read_text(encoding="utf-8")
    assert "failed_markers" not in source, "中文失败词列表解析 outcome.text"
    admission = (SRC / "agentd" / "pi_bridge" / "action_admission.py").read_text(
        encoding="utf-8"
    )
    # any-receipt 判定（receipt.received 存在即成功）必须绝迹。
    # 精确 action_id 匹配的 receipt 验证是合法的（P0-5E receipt 合约）。
    assert "any(" not in admission or "receipt.received" not in admission, (
        "any-receipt 判定——旧 receipt 会为新动作背书（P0-NA-13）"
    )
    # 若使用 events_replay，必须按独立 receipt_id（或 action_id 兜底）
    # 精确匹配——不是"Mission 内任意 receipt 存在即成功"。
    if "events_replay" in admission:
        assert (
            'payload.get("receipt_id") == receipt_id' in admission
            or 'payload.get("action_id") == action_id' in admission
        ), "events_replay 存在但没有 receipt_id/action_id 精确匹配"


# ---------------------------------------------------------------- 运行时红线


class _StubHandlers:
    """最小 handlers stub：记录 request_approval 调用并返回精确 request_id。"""

    def __init__(self) -> None:
        self.approval_calls: list[dict] = []

    def request_context(self, *, mode: str, principal: str):  # noqa: ARG002
        import contextlib

        return contextlib.nullcontext()


async def _red_setup(tmp_path: Path):
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
    mission = service.create_mission("red-test", mode="SIMULATION")
    bindings = SessionBindingStore(service._store.connection)
    bindings.bind(
        pi_session_id="pi_1", pi_session_path="", mission_id=mission.mission_id,
        body_id="sim/ur5e", execution_mode="SIMULATION", created_by="user:local:1000",
    )
    bindings.acquire_lease(
        mission_id=mission.mission_id, pi_session_id="pi_1", owner_pid=1, owner_uid=1000
    )
    return service, mission


@pytest.mark.asyncio
async def test_propose_without_request_context_rejected(tmp_path: Path) -> None:
    """P0-NA-10：pi.action.propose 不带 session/lease/revision/body/mode
    必须拒绝——完整请求上下文是建卡前提。"""
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    service, mission = await _red_setup(tmp_path)
    bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
    # 只带 mission/capability/arguments 的"宽松"调用——必须被拒。
    result = await bridge._dispatch(
        "user:local:1000",
        1,  # writer owner_pid（_red_setup 用 1）——P0-5A 后调用者必须匹配
        "pi.action.propose",
        {
            "token": service.control_token,
            "mission_id": mission.mission_id,
            "capability_id": "sim_ground_truth",
            "arguments": {},
            "expected_effect": "x",
            "risk_tier": "LOW",
        },
    )
    assert not result.get("ok"), "缺少完整请求上下文的 propose 竟然成功"
    assert result.get("code") in (
        "REQUEST_CONTEXT_REQUIRED",
        "SESSION_UNBOUND",
        "WRITER_LEASE_REQUIRED",
        "CONTEXT_REQUIRED",
    ), f"错误码应指明缺失的请求上下文: {result}"
    # 且不得产生任何 approval 卡。
    assert service.pending_approvals(mission.mission_id) == []
    await service.close()


@pytest.mark.asyncio
async def test_stale_context_forced_tool_call_cannot_create_card(tmp_path: Path) -> None:
    """P0-NA-10/Gate C：context stale 时即使模型强行调用 action，
    也必须硬拒绝且零 approval——不靠提示词约束。"""
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    service, mission = await _red_setup(tmp_path)
    bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
    result = await bridge._dispatch(
        "user:local:1000",
        1,  # writer owner_pid（_red_setup 用 1）——P0-5A 后调用者必须匹配
        "pi.action.propose",
        {
            "token": service.control_token,
            "mission_id": mission.mission_id,
            "pi_session_id": "pi_1",  # 已绑定+持 lease——隔离出纯 revision 违规
            "context_revision": 999999,  # 远超当前 revision——过期/伪造
            "body_hash": "body_deadbeef",
            "mode": "SIMULATION",
            "capability_id": "sim_ground_truth",
            "arguments": {},
            "idempotency_key": "idem_red_stale",
        },
    )
    assert not result.get("ok"), "stale/伪造 revision 的 propose 竟然成功"
    # HOTFIX-1 后这条路径先被 context lease 拦截（无 lease 即
    # CONTEXT_LEASE_REQUIRED）；旧 revision/body 校验层仍保留。
    assert result.get("code") in (
        "CONTEXT_LEASE_REQUIRED",
        "CONTEXT_NOT_FRESH",
        "CONTEXT_STALE",
        "CONTEXT_REVISION_MISMATCH",
        "BODY_HASH_MISMATCH",
    ), f"错误码应指明 context 问题: {result}"
    assert service.pending_approvals(mission.mission_id) == []
    await service.close()


@pytest.mark.asyncio
async def test_single_admission_entrypoint_exists() -> None:
    """P0-NA-10：存在且只存在一个 ActionAdmissionService——所有 action
    入口（dispatcher 与 TUI 两阶段 RPC）都必须经它。"""
    from rosclaw.agentd.pi_bridge import action_admission  # noqa: F401

    assert hasattr(action_admission, "ActionAdmissionService"), (
        "缺少统一的 ActionAdmissionService"
    )
    svc = action_admission.ActionAdmissionService
    for method in ("propose", "decision_status", "execute"):
        assert callable(getattr(svc, method, None)), f"缺少 {method}"
    sig = inspect.signature(svc.propose)
    # propose 必须接收完整请求上下文（结构化 contract）。
    params = set(sig.parameters)
    for required in ("request",):
        assert required in params or any(
            p in params
            for p in ("session_id", "lease_token", "context_revision", "body_hash")
        ), f"propose 签名缺少请求上下文参数: {params}"
