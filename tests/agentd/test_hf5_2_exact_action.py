"""PR-HF5-2 红测试（五审 P0-5C）：ExactActionV1 + ApprovalRequestV3。

红测试先行——以下缺陷修复前必须稳定复现：

1. SIM 审批卡 capability_id 为空（只挂 daemon_capability_id）；
2. display_hash 不绑定 capability/mission/mode/context；
3. 危险 capability 可用无害 title 通过；
4. MCP 默认值未在建卡前展开（人批准 {}，执行 660Hz）；
5. executor 收到的参数与卡片展示的参数不是同一份；
6. 嵌套 schema/type/enum/range/non-finite 未完整校验；
7. 卡片与 txn 的 capability 不一致未拒绝。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agentd.test_pi_tool_bridge import (
    SIM_ACTION_CAPABILITY,
    _issue_lease,
    _setup,
)


async def _propose(service, mission, *, idem: str, arguments=None, capability=None):
    from rosclaw.agentd.pi_bridge.action_admission import (
        ActionAdmissionService,
        ActionRequestContext,
    )

    snapshot = service.snapshot(mission.mission_id)
    lease = _issue_lease(service, mission)
    ctx = ActionRequestContext(
        pi_session_id="pi_1",
        mission_id=mission.mission_id,
        context_revision=snapshot.context_revision,
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        idempotency_key=idem,
        context_lease_id=lease,
    )
    admission = ActionAdmissionService(service)
    card = await admission.propose(
        request=ctx,
        capability_id=capability or SIM_ACTION_CAPABILITY,
        arguments=arguments if arguments is not None else {},
        expected_effect="hf5-2",
        risk_tier="LOW",
    )
    return card


class TestExactActionContract:
    async def test_sim_approval_card_has_nonempty_capability_id(
        self, tmp_path: Path
    ) -> None:
        """SIM 卡也必须有一等 capability_id（不再只挂 daemon 字段）。"""
        service, mission = await _setup(tmp_path)
        card = await _propose(service, mission, idem="idem_cap")
        # 返回卡必须有 capability。
        assert card["capability_id"] == SIM_ACTION_CAPABILITY
        # DB 里的 ApprovalRequest 也必须持久化 exact_action（含 capability）。
        stored = service._broker.get_request(card["approval_id"])
        assert stored is not None
        exact_json = getattr(stored, "exact_action_json", "") or ""
        assert exact_json, "ApprovalRequest 未持久化 exact_action"
        import json

        exact = json.loads(exact_json)
        assert exact["capability_id"] == SIM_ACTION_CAPABILITY
        assert exact["action_intent_hash"], "缺 action_intent_hash"
        await service.close()

    async def test_display_hash_binds_capability_mission_mode_context(
        self, tmp_path: Path
    ) -> None:
        """display_hash 必须绑定 capability/mission/mode/context——
        任一变化 hash 必须不同。"""
        service, mission = await _setup(tmp_path)
        card = await _propose(service, mission, idem="idem_hash")
        stored = service._broker.get_request(card["approval_id"])
        assert stored is not None
        from rosclaw.agentd.operator_socket import display_hash_for

        base = display_hash_for(stored)
        # 篡改 capability → hash 必须不同。
        import json

        exact = json.loads(stored.exact_action_json)
        exact["capability_id"] = "dangerous.move"
        tampered = stored.model_copy(
            update={"exact_action_json": json.dumps(exact)}
        )
        assert display_hash_for(tampered) != base, (
            "capability 变化但 display_hash 未变——hash 未绑定 capability"
        )
        await service.close()

    async def test_dangerous_capability_cannot_use_harmless_title(
        self, tmp_path: Path
    ) -> None:
        """title 由合约派生（含 capability）——调用方传的 title 不得
        让危险 capability 伪装成无害文本。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )

        service, mission = await _setup(tmp_path)
        snapshot = service.snapshot(mission.mission_id)
        lease = _issue_lease(service, mission)
        ctx = ActionRequestContext(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=snapshot.context_revision,
            body_hash=mission.body_binding.effective_body_hash,
            mode=mission.mode.value,
            idempotency_key="idem_title",
            context_lease_id=lease,
        )
        admission = ActionAdmissionService(service)
        card = await admission.propose(
            request=ctx,
            capability_id=SIM_ACTION_CAPABILITY,
            arguments={},
            expected_effect="x",
            risk_tier="LOW",
            title="播放提示音（无害）",  # 调用方给的 title
        )
        # 卡片 title 必须含真实 capability——不是调用方给的无害文本。
        assert SIM_ACTION_CAPABILITY in card["title"], (
            f"卡片 title 未绑定真实 capability: {card['title']}"
        )
        assert card["title"] != "播放提示音（无害）"
        await service.close()

    async def test_schema_validation_nested_and_nonfinite(
        self, tmp_path: Path
    ) -> None:
        """完整 JSON Schema 校验：嵌套 type/range/enum/NaN 全部拒绝。"""
        from rosclaw.agentd.pi_bridge.action_schema import (
            validate_action_arguments,
        )
        from rosclaw.contracts.common import ValidationError

        schema = {
            "type": "object",
            "properties": {
                "frequency_hz": {"type": "integer", "minimum": 100, "maximum": 2000},
                "mode": {"type": "string", "enum": ["beep", "pulse"]},
                "nested": {
                    "type": "object",
                    "properties": {"depth": {"type": "number", "minimum": 0}},
                },
            },
            "additionalProperties": False,
        }
        # 合法通过。
        ok = validate_action_arguments(schema, {"frequency_hz": 660, "mode": "beep"})
        assert ok["frequency_hz"] == 660
        # 超范围拒绝。
        with pytest.raises(ValidationError):
            validate_action_arguments(schema, {"frequency_hz": 9999})
        # enum 拒绝。
        with pytest.raises(ValidationError):
            validate_action_arguments(schema, {"mode": "explode"})
        # 嵌套 type 拒绝。
        with pytest.raises(ValidationError):
            validate_action_arguments(schema, {"nested": {"depth": -1}})
        # NaN 拒绝。
        with pytest.raises(ValidationError):
            validate_action_arguments(schema, {"frequency_hz": float("nan")})
        # 未知字段拒绝。
        with pytest.raises(ValidationError):
            validate_action_arguments(schema, {"evil": True})

    async def test_defaults_expanded_before_approval(self, tmp_path: Path) -> None:
        """带 default 的 schema：{} 必须展开成默认值后再建卡。"""
        from rosclaw.agentd.pi_bridge.action_schema import (
            validate_action_arguments,
        )

        schema = {
            "type": "object",
            "properties": {
                "frequency_hz": {"type": "integer", "default": 660},
                "duration_sec": {"type": "number", "default": 0.25},
            },
        }
        normalized = validate_action_arguments(schema, {})
        assert normalized == {"frequency_hz": 660, "duration_sec": 0.25}, (
            f"默认值未展开: {normalized}"
        )
