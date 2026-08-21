"""PR-N5B 红测试（调整方案 §三.N5B）：ToolResultEnvelopeV2 合约层。

红测试先行——合约不存在时必须红。

每个模型可调用能力必须声明并验证输出 Schema；执行结果统一为
canonical envelope：模型文本、UI 展示、CLI JSON 都是 canonical
value 的投影，executor 不得返回裸字符串冒充结构化结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from rosclaw.contracts.common import UnsupportedVersionError

GOLDEN_DIR = Path(__file__).parent / "golden"


def _envelope_payload(**overrides) -> dict:
    payload = {
        "schema_version": "rosclaw.tool_result_envelope.v2",
        "call_id": "call_1",
        "capability_id": "ur5e.simulate_cartesian_trajectory",
        "status": "SUCCEEDED",
        "value": {"run_id": "run_1"},
        "artifact_refs": [],
        "evidence_refs": [],
    }
    payload.update(overrides)
    return payload


class TestToolResultEnvelopeV2:
    def test_round_trip(self) -> None:
        from rosclaw.contracts.agent.tool_result import ToolResultEnvelopeV2

        env = ToolResultEnvelopeV2.model_validate_contract(_envelope_payload())
        assert env.status.value == "SUCCEEDED"
        assert env.value == {"run_id": "run_1"}
        assert env.error is None

    def test_golden_schema(self) -> None:
        from rosclaw.contracts.agent.tool_result import ToolResultEnvelopeV2

        golden = GOLDEN_DIR / "rosclaw.tool_result_envelope.v2.json"
        assert golden.exists(), f"missing golden file {golden}"
        current = ToolResultEnvelopeV2.model_json_schema()
        current["$id"] = "rosclaw://schemas/rosclaw.tool_result_envelope.v2"
        current["title"] = "rosclaw.tool_result_envelope.v2"
        assert json.loads(golden.read_text(encoding="utf-8")) == current, (
            "schema rosclaw.tool_result_envelope.v2 drifted from golden; if "
            "intentional, re-export via rosclaw.contracts.export and review "
            "the diff"
        )

    def test_status_enum_frozen(self) -> None:
        from rosclaw.contracts.agent.tool_result import ToolResultStatusV1

        assert {s.value for s in ToolResultStatusV1} == {
            "SUCCEEDED", "FAILED", "BLOCKED", "PENDING",
        }

    def test_error_shape(self) -> None:
        from rosclaw.contracts.agent.tool_result import ToolResultEnvelopeV2

        env = ToolResultEnvelopeV2.model_validate_contract(_envelope_payload(
            status="FAILED",
            value=None,
            error={
                "code": "INVALID_CAPABILITY_OUTPUT",
                "message": "missing required field run_id",
                "retryable": False,
                "recovery": ["检查 executor 输出与 output_schema 是否一致"],
            },
        ))
        assert env.error is not None
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"
        assert env.error.retryable is False

    def test_unknown_major_version_rejected(self) -> None:
        from rosclaw.contracts.agent.tool_result import ToolResultEnvelopeV2

        with pytest.raises(UnsupportedVersionError):
            ToolResultEnvelopeV2.model_validate_contract(
                _envelope_payload(schema_version="rosclaw.tool_result_envelope.v9")
            )

    def test_invalid_status_rejected(self) -> None:
        from rosclaw.contracts.agent.tool_result import ToolResultEnvelopeV2

        with pytest.raises((PydanticValidationError, Exception)):
            ToolResultEnvelopeV2.model_validate_contract(
                _envelope_payload(status="KINDA_WORKED")
            )

    def test_registered_in_export(self) -> None:
        from rosclaw.contracts.export import ALL_CONTRACTS

        assert "rosclaw.tool_result_envelope.v2" in ALL_CONTRACTS
