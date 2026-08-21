"""PR-N5A 红测试（调整方案 §三.N5A）：CapabilityDescriptorV2 / ToolProjectionV1 合约层。

红测试先行——合约不存在时必须红。

合约要点：
1. CapabilityDescriptorV2（rosclaw.capability.v2）描述"ROSClaw 具备
   什么能力"：input/output schema + effect + compatibility +
   execution + evidence 五块；
2. EffectClassV1 正式枚举冻结为 8 值（N5C 单一 Effect Contract 的
   目标词表，先在合约层冻结）；
3. ToolProjectionV1（rosclaw.tool_projection.v1）描述能力如何投影
   给 Harness/模型：direct / propose_only / internal；
4. 版本闸与 golden schema 稳定性沿用 ContractModel 机制。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from rosclaw.contracts.common import UnsupportedVersionError

GOLDEN_DIR = Path(__file__).parent / "golden"


def _capability_payload(**overrides) -> dict:
    payload = {
        "schema_version": "rosclaw.capability.v2",
        "capability_id": "ur5e.simulate_cartesian_trajectory",
        "version": "1.0.0",
        "source": "rosclaw-builtin",
        "description": "UR5e MuJoCo 笛卡尔轨迹动力学仿真",
        "input_schema": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "effect": {
            "class": "SIMULATED_EFFECT",
            "domain": "simulation_state",
            "reversible": True,
            "risk_tier": "LOW",
            "requires_fresh_real_context": False,
        },
        "compatibility": {
            "modes": ["SIMULATION"],
            "body_types": ["manipulator"],
            "robot_ids": ["ur5e"],
            "resource_kinds": ["robot", "world"],
            "runtime_requirements": ["rosclaw-simulation"],
        },
        "execution": {
            "executor_ref": "python:rosclaw.agentd.sim_trajectory:simulate",
            "long_running": True,
            "cancellation": "cooperative",
            "concurrency": "exclusive",
            "idempotent": True,
        },
        "evidence": {
            "evidence_class": "SIM_DYN_ROLLOUT",
            "verifier_ref": "trajectory_tracking_v2",
            "resource_provenance_required": True,
        },
    }
    payload.update(overrides)
    return payload


class TestEffectClassV1:
    def test_enum_frozen_to_eight_values(self) -> None:
        """EffectClassV1 正式枚举 = N5C 目标词表 8 值，不得增减。"""
        from rosclaw.contracts.agent.capability import EffectClassV1

        assert {e.value for e in EffectClassV1} == {
            "READ_ONLY",
            "PURE_COMPUTE",
            "WORKSPACE_WRITE",
            "HOST_PROCESS",
            "NETWORK_EFFECT",
            "SIMULATED_EFFECT",
            "SHADOW_PROPOSAL",
            "PHYSICAL_EFFECT",
        }


class TestCapabilityDescriptorV2:
    def test_round_trip_and_hash_deterministic(self) -> None:
        from rosclaw.contracts.agent.capability import CapabilityDescriptorV2

        cap = CapabilityDescriptorV2.model_validate_contract(_capability_payload())
        assert cap.capability_id == "ur5e.simulate_cartesian_trajectory"
        assert cap.effect.class_.value == "SIMULATED_EFFECT"
        assert cap.compatibility.modes == ["SIMULATION"]
        assert cap.execution.idempotent is True
        assert cap.evidence.resource_provenance_required is True
        again = CapabilityDescriptorV2.model_validate_contract(_capability_payload())
        assert cap.canonical_hash() == again.canonical_hash()

    def test_golden_schema(self) -> None:
        from rosclaw.contracts.agent.capability import CapabilityDescriptorV2

        golden = GOLDEN_DIR / "rosclaw.capability.v2.json"
        assert golden.exists(), f"missing golden file {golden}"
        current = CapabilityDescriptorV2.model_json_schema()
        current["$id"] = "rosclaw://schemas/rosclaw.capability.v2"
        current["title"] = "rosclaw.capability.v2"
        assert json.loads(golden.read_text(encoding="utf-8")) == current, (
            "schema rosclaw.capability.v2 drifted from golden; if intentional, "
            "re-export via rosclaw.contracts.export and review the diff"
        )

    def test_invalid_effect_class_rejected(self) -> None:
        from rosclaw.contracts.agent.capability import CapabilityDescriptorV2

        payload = _capability_payload()
        payload["effect"]["class"] = "MAYBE_SAFE"
        with pytest.raises((PydanticValidationError, Exception)):
            CapabilityDescriptorV2.model_validate_contract(payload)

    def test_unknown_major_version_rejected(self) -> None:
        from rosclaw.contracts.agent.capability import CapabilityDescriptorV2

        payload = _capability_payload()
        payload["schema_version"] = "rosclaw.capability.v9"
        with pytest.raises(UnsupportedVersionError):
            CapabilityDescriptorV2.model_validate_contract(payload)

    def test_physical_effect_requires_provenance_by_default(self) -> None:
        """PHYSICAL_EFFECT 能力的 resource_provenance_required 默认 True
        （物理效应必须能反查资源证明——fail closed）。"""
        from rosclaw.contracts.agent.capability import CapabilityDescriptorV2

        payload = _capability_payload()
        payload["effect"]["class"] = "PHYSICAL_EFFECT"
        payload["effect"]["requires_fresh_real_context"] = True
        payload["compatibility"]["modes"] = ["REAL"]
        payload.pop("evidence")
        cap = CapabilityDescriptorV2.model_validate_contract(payload)
        assert cap.evidence.resource_provenance_required is True


class TestToolProjectionV1:
    def _payload(self, **overrides) -> dict:
        payload = {
            "schema_version": "rosclaw.tool_projection.v1",
            "tool_name": "simulate_ur5e_trajectory",
            "capability_id": "ur5e.simulate_cartesian_trajectory",
            "input_schema": {"type": "object", "additionalProperties": False},
            "output_schema": {"type": "object", "additionalProperties": False},
            "presentation_ref": "simulation_trajectory_card",
            "exposure": "direct",
        }
        payload.update(overrides)
        return payload

    def test_golden_schema(self) -> None:
        from rosclaw.contracts.agent.capability import ToolProjectionV1

        golden = GOLDEN_DIR / "rosclaw.tool_projection.v1.json"
        assert golden.exists(), f"missing golden file {golden}"
        current = ToolProjectionV1.model_json_schema()
        current["$id"] = "rosclaw://schemas/rosclaw.tool_projection.v1"
        current["title"] = "rosclaw.tool_projection.v1"
        assert json.loads(golden.read_text(encoding="utf-8")) == current, (
            "schema rosclaw.tool_projection.v1 drifted from golden; if "
            "intentional, re-export via rosclaw.contracts.export and review "
            "the diff"
        )

    def test_exposure_values(self) -> None:
        from rosclaw.contracts.agent.capability import ToolProjectionV1

        for exposure in ("direct", "propose_only", "internal"):
            proj = ToolProjectionV1.model_validate_contract(
                self._payload(exposure=exposure)
            )
            assert proj.exposure.value == exposure
        with pytest.raises((PydanticValidationError, Exception)):
            ToolProjectionV1.model_validate_contract(
                self._payload(exposure="secret_backdoor")
            )

    def test_registered_in_export(self) -> None:
        """两个新合约进入 ALL_CONTRACTS（CI 扫描/工具链可见）。"""
        from rosclaw.contracts.export import ALL_CONTRACTS

        assert "rosclaw.capability.v2" in ALL_CONTRACTS
        assert "rosclaw.tool_projection.v1" in ALL_CONTRACTS
