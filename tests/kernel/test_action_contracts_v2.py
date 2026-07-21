"""Safety additions to the backwards-compatible ActionEnvelope v1 wire schema."""

from __future__ import annotations

from datetime import UTC, datetime
from math import inf, nan

import pytest

from rosclaw.kernel import (
    AcknowledgementStage,
    ActionEnvelope,
    ExecutionMode,
    OrphanPolicy,
    VerificationPolicy,
)


@pytest.mark.parametrize("timeout", [0.0, -1.0, inf, nan, True])
def test_verification_timeout_must_be_finite_and_bounded(timeout: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VerificationPolicy(timeout_sec=timeout)  # type: ignore[arg-type]


def test_stop_capability_is_not_coerced_from_null() -> None:
    payload = _action().to_dict()
    payload["stop_capability"] = None

    with pytest.raises(ValueError, match="stop_capability"):
        ActionEnvelope.from_dict(payload)


def _action(**overrides: object) -> ActionEnvelope:
    values = {
        "actor_id": "agent",
        "agent_framework": "codex",
        "session_id": "session",
        "body_id": "body",
        "capability_id": "body.move",
        "arguments": {},
        "execution_mode": ExecutionMode.SHADOW,
    }
    values.update(overrides)
    return ActionEnvelope(**values)  # type: ignore[arg-type]


def test_action_contract_always_serializes_finite_lease_and_stop_policy() -> None:
    before = datetime.now(UTC)
    action = _action()
    payload = action.to_dict()

    assert action.deadline_at is not None and action.deadline_at > before
    assert payload["lease_ttl_ms"] == 10_000
    assert payload["renew_interval_ms"] == 3_000
    assert payload["orphan_policy"] == "STOP_ON_CLIENT_LOSS"
    assert payload["stop_capability"] == "safety.emergency_stop"
    assert ActionEnvelope.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"lease_ttl_ms": 99}, "lease_ttl_ms"),
        ({"lease_ttl_ms": 300, "renew_interval_ms": 300}, "less than"),
        ({"stop_capability": ""}, "stop_capability"),
    ],
)
def test_action_contract_rejects_unbounded_or_unstoppable_lease(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _action(**overrides)


def test_acknowledgement_stages_are_distinct_wire_values() -> None:
    assert [stage.value for stage in AcknowledgementStage] == [
        "REQUEST_ACCEPTED",
        "COMMAND_DISPATCHED",
        "PROTOCOL_ACKNOWLEDGED",
        "DELIVERY_INFERRED",
        "EFFECT_OBSERVED",
        "TASK_VERIFIED",
    ]
    assert OrphanPolicy("STOP_ON_CLIENT_LOSS") is OrphanPolicy.STOP_ON_CLIENT_LOSS


@pytest.mark.parametrize("field", ["lease_ttl_ms", "renew_interval_ms"])
def test_action_contract_rejects_boolean_wire_lease_values(field: str) -> None:
    payload = _action().to_dict()
    payload[field] = True

    with pytest.raises(TypeError, match=field):
        ActionEnvelope.from_dict(payload)
