from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rosclaw.knowledge.context_adapter import build_how_context, build_memory_evidence


def test_body_software_runtime_and_memory_are_explicitly_separated():
    context = build_how_context(
        task="LIMO patrol",
        robot_model="limo",
        ros_distro="melodic",
        simulator="gazebo",
        safety_limits=["speed<=0.3"],
        memory_evidence=[
            {
                "memory_id": "mem_1",
                "summary": "Previous patrol required a wider turn.",
                "confidence": 0.8,
                "receipt_ref": "receipt_1",
                "created_at": datetime.now(UTC),
            }
        ],
    )
    assert context.body.robot_model == "limo"
    assert context.software.ros_distro == "melodic"
    assert context.runtime.task == "LIMO patrol"
    assert context.memory_evidence[0].evidence_domain == "memory"


@pytest.mark.parametrize("field", ["trajectory", "video", "sensor_data", "raw_content"])
def test_raw_memory_and_practice_content_is_rejected(field):
    with pytest.raises(ValueError, match="forbidden"):
        build_memory_evidence(
            [
                {
                    "memory_id": "mem_1",
                    "summary": "bounded summary",
                    "confidence": 0.5,
                    "created_at": datetime.now(UTC),
                    field: "must not cross",
                }
            ]
        )
