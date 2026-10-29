"""Synthetic channel and receipt fixtures only; no Runtime, ROS or hardware."""

import json
from copy import deepcopy
from types import SimpleNamespace as Namespace
from unittest.mock import AsyncMock

import pytest

from rosclaw.agentd.action_channel import ActionChannelError
from rosclaw.agentd.action_dispatch import request_action
from rosclaw.agentd.action_receipt_feedback import MAX_FEEDBACK_BYTES, historical_action_feedback
from tests.agentd.test_native_predispatch_replan import canonical_block


def receipt():
    return {
        "action_id": "owned",
        "capability_id": "docking.undock",
        "body_id": "fixture",
        "body_snapshot_hash": "b" * 64,
        "execution_mode": "SIMULATION",
        "evidence_domain": "SIMULATION",
        "final_state": "COMPLETED",
        "evidence_level": "PHYSICALLY_OBSERVED",
        "trust_level": "SIMULATED",
        "verified": True,
        "observations": [
            {"sim_t": 10.0, "public_odom_xy": [0.0, 0.0], "docked": True},
            {"sim_t": 11.0, "public_odom_xy": [-0.35, 0.0], "docked": False},
        ],
        "dispatch_result": {"status": 4},
        "verification_result": {"passed": True, "scope": "requested_primitive_goal_only"},
        "errors": [],
    }


def project(value, **kwargs):
    return historical_action_feedback(
        value,
        capability="docking.undock",
        action_id="owned",
        mode="SIMULATION",
        expected_body_id="fixture",
        expected_body_hash="b" * 64,
        **kwargs,
    )


def parsed(text):
    return json.loads(text[text.index("{") :])


def test_original_before_after_pose_is_visible_but_never_claimed_current():
    value = receipt()
    before = deepcopy(value)
    result = parsed(project(value))
    assert result["last_recorded_observations"] == value["observations"]
    assert result["current_state_asserted"] is False and result["evidence_promoted"] is False
    assert result["receipt_ref"] == "receipt://owned"
    assert result["verified"] is True and result["evidence_level"] == "PHYSICALLY_OBSERVED"
    assert value == before


def test_only_last_two_original_observations_with_explicit_omission_count():
    value = receipt()
    value["observations"].insert(1, {"sim_t": 10.5, "public_odom_xy": [-0.2, 0.0]})
    result = parsed(project(value))
    assert result["last_recorded_observations"] == value["observations"][-2:]
    assert result["recorded_observation_count"] == 3 and result["omitted_observation_count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_id", "other"),
        ("body_id", "other"),
        ("body_snapshot_hash", "other"),
        ("capability_id", "other"),
        ("execution_mode", "REAL"),
        ("evidence_domain", "REAL"),
        ("evidence_domain", "SYNTHETIC"),
        ("evidence_domain", "FIXTURE"),
    ],
)
def test_mismatched_real_or_fixture_receipts_are_not_projected(field, value):
    raw = receipt()
    raw[field] = value
    assert project(raw) == ""


def test_shadow_unknown_evidence_remains_unknown_and_unverified():
    raw = receipt()
    raw.update(execution_mode="SHADOW", evidence_domain="UNKNOWN", verified=False)
    result = parsed(
        historical_action_feedback(
            raw,
            capability="docking.undock",
            action_id="owned",
            mode="SHADOW",
            expected_body_id="fixture",
            expected_body_hash="b" * 64,
        )
    )
    assert result["verified"] is False and result["evidence_domain"] == "UNKNOWN"


def test_oversized_and_nonfinite_fields_are_explicitly_omitted():
    raw = receipt()
    raw["observations"] = [{"large": "一" * 5000}]
    raw["errors"] = [{"message": "x" * 5000}]
    raw["dispatch_result"] = {"number": float("nan")}
    raw["verification_result"] = {"data": "x" * 5000}
    text = project(raw)
    q = parsed(text)
    assert q["omitted_observation_count"] == 1
    assert set(q["inline_omitted_fields"]) == {
        "errors",
        "last_recorded_observations",
        "dispatch_result",
        "verification_result",
    }
    assert "NaN" not in text and len(text.encode()) <= MAX_FEEDBACK_BYTES
    assert q["verified"] is True


def test_projection_does_not_export_authorization_grant_permit_or_lease_payloads():
    raw = receipt()
    raw["authorization_decision"] = {"permit": "PRIVATE_AUTH"}
    raw["resource_lease"] = {"token": "PRIVATE_LEASE"}
    raw["grant"] = {"secret": "PRIVATE_GRANT"}
    text = project(raw)
    assert "PRIVATE_" not in text


@pytest.mark.parametrize("budget", [0, 50, 256, 512, 1024, 2048, 3072])
def test_utf8_projection_stays_inside_actual_native_summary_budget(budget):
    raw = receipt()
    raw["errors"] = [{"message": "原始错误" * 20}]
    text = project(raw, max_bytes=budget)
    assert len(text.encode()) <= budget
    if "\n{" in text:
        assert parsed(text)["current_state_asserted"] is False


def fixture(value, verified=True):
    outcome = Namespace(
        receipt={"receipt": value},
        action_id=value["action_id"],
        state=value["final_state"],
        trust_level="SIMULATED",
        verified=verified,
    )
    channel = Namespace(
        request_nonreal_action=AsyncMock(return_value=outcome),
        request_real_proposal=AsyncMock(
            return_value=Namespace(
                request_id="proposal", action_id="real-proposal", state="PENDING"
            )
        ),
    )
    grant = Namespace(grant_id="fixture-grant", public_hash="public", principal="fixture")
    service = Namespace(
        _body_id="fixture",
        _broker=Namespace(
            action_intent_for_grant=lambda _: "fixture-intent", verify=lambda *args, **kwargs: grant
        ),
        _action_channel=channel,
        _consent_channel=None,
        _tool_catalog=None,
        _current_body_hash=lambda: "b" * 64,
        _events=Namespace(append=AsyncMock()),
    )
    decision = Namespace(
        mission_id="fixture",
        summary="fixture",
        proposed_operation=Namespace(
            payload={
                "grant_id": grant.grant_id,
                "capability_id": value["capability_id"],
                "arguments": {},
            }
        ),
    )
    return (service, decision, channel)


async def test_nonreal_dispatch_formatter_delivers_original_pose_into_model_visible_text():
    raw = receipt()
    service, decision, channel = fixture(raw)
    q = await request_action(service, decision, mode="SIMULATION", principal="fixture")
    assert q.terminal_receipt is True and q.error_code is None
    assert parsed(q.text)["last_recorded_observations"] == raw["observations"]
    assert len(q.text.encode()) < 4000 and channel.request_nonreal_action.await_count == 1


async def test_failed_receipt_preserves_diagnostic_and_original_failure_class():
    raw = receipt()
    raw.update(
        final_state="FAILED",
        verified=False,
        evidence_level="REQUESTED",
        errors=[
            {
                "code": "PUBLIC_COVERAGE_FAILED",
                "message": "Original freshness guard: odom; capture_age_s=1.827904",
            }
        ],
    )
    service, decision, channel = fixture(raw, verified=False)
    q = await request_action(service, decision, mode="SIMULATION", principal="fixture")
    assert q.error_code == "ACTION_NOT_VERIFIED" and q.terminal_receipt is False
    assert parsed(q.text)["errors"] == raw["errors"] and parsed(q.text)["verified"] is False
    assert len(q.text) < 4000 and channel.request_nonreal_action.await_count == 1


async def test_known_idle_dock_precondition_stays_blocked_with_no_retry():
    raw = canonical_block()
    raw.update(body_id="fixture", body_snapshot_hash="b" * 64)
    service, decision, channel = fixture(raw, verified=False)
    q = await request_action(service, decision, mode="SIMULATION", principal="fixture")
    assert q.error_code == "ACTION_PRECONDITION_NOT_MET" and q.terminal_receipt is False
    assert parsed(q.text)["final_state"] == "BLOCKED" and parsed(q.text)["verified"] is False
    assert channel.request_nonreal_action.await_count == 1


async def test_unknown_transport_does_not_fabricate_receipt_feedback():
    service, decision, channel = fixture(receipt())
    channel.request_nonreal_action.side_effect = ActionChannelError("fixture response lost")
    q = await request_action(service, decision, mode="SIMULATION", principal="fixture")
    assert q.terminal_receipt is False and "historical_original" not in q.text
    assert channel.request_nonreal_action.await_count == 1


async def test_real_mode_retains_operator_proposal_path_and_never_uses_nonreal_projection():
    service, decision, channel = fixture(receipt())
    q = await request_action(service, decision, mode="REAL", principal="fixture")
    assert q.terminal_receipt is False and "historical_original" not in q.text
    channel.request_nonreal_action.assert_not_awaited()
    channel.request_real_proposal.assert_awaited_once()
