"""WakeConditionV1 — a registered wait that resumes a MissionRunner (大纲 §5.8)."""

from __future__ import annotations

from typing import Literal

from rosclaw.contracts.common import ContractModel


class WakeConditionV1(ContractModel):
    SCHEMA = "rosclaw.wake_condition.v1"

    schema_version: Literal["rosclaw.wake_condition.v1"] = "rosclaw.wake_condition.v1"
    type: Literal[
        "worker_completed",
        "approval_decided",
        "receipt_terminal",
        "sensor_condition",
        "deadline",
    ]
    reference_id: str | None = None
    deadline: str | None = None
    on_timeout: Literal["REPLAN", "WAIT_INPUT", "FAIL_SAFE"] = "WAIT_INPUT"
