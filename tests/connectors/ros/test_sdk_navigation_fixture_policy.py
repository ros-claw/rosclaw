"""Anonymous SDK operator policy validation; no ROS, Body mutation or actions."""

import copy

import pytest

from integrations.ros_probe.acceptance.navigation_fixture_policy import (
    select_navigation_fixture_policy,
)
from tests.connectors.ros.test_navigation_body_workspace import source


def inputs():
    _, _, _, baseline = source(namespace="/unbranded/controls")
    proposal = {
        "schema_version": "rosclaw.sdk_navigation_policy_input.v1",
        "scope": "OPERATOR_OWNED_DISPOSABLE_SIM_NEW_BODY",
        "policy_id": "explicit-current-source-policy",
        "body_id": baseline["body_id"],
        "source_urdf_sha256": baseline["source_urdf_sha256"],
        "navigation_requirements_hash": baseline["navigation_requirements_hash"],
        "navigation_limits": copy.deepcopy(baseline["navigation_limits"]),
    }
    proposal["navigation_limits"].update(max_speed_mps=0.11, max_yaw_rate_rps=0.22)
    return baseline, proposal


def select(base, proposal):
    return select_navigation_fixture_policy(
        base, proposal, action_name="/unbranded/controls/go", frame_id="world"
    )


def test_legacy_policy_is_byte_semantically_unchanged_and_copied():
    base, _ = inputs()
    before = copy.deepcopy(base)
    result = select(base, None)
    assert result == before and result is not base
    result["navigation_limits"]["max_speed_mps"] = 999
    assert base == before


def test_explicit_new_source_policy_does_not_mutate_inputs_or_grant_new_fields():
    base, proposal = inputs()
    a, b = copy.deepcopy(base), copy.deepcopy(proposal)
    result = select(base, proposal)
    assert result["navigation_limits"]["max_speed_mps"] == 0.11
    assert result["source_snapshot_hash"] == base["source_snapshot_hash"]
    assert result["witness_declaration"] == base["witness_declaration"]
    assert set(result) == set(base) and base == a and proposal == b


@pytest.mark.parametrize("key", ["body_id", "source_urdf_sha256", "navigation_requirements_hash"])
def test_old_or_unrelated_body_source_policy_is_rejected(key):
    base, proposal = inputs()
    proposal[key] = "unrelated"
    with pytest.raises(ValueError, match="binding"):
        select(base, proposal)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "schema",
        "scope",
        "identity",
        "infinite",
        "boolean",
        "negative",
        "zero",
        "missing_limit",
        "extra_limit",
        "stop_over_speed",
        "stop_over_yaw",
        "mutate_snapshot",
        "real_execution",
        "oversized",
    ],
)
def test_invalid_or_authority_changing_policy_is_rejected(fault):
    base, proposal = inputs()
    if fault == "missing":
        proposal.pop("scope")
    elif fault == "extra":
        proposal["extra"] = "value"
    elif fault == "schema":
        proposal["schema_version"] = "other"
    elif fault == "scope":
        proposal["scope"] = "LIVE_ROBOT"
    elif fault == "identity":
        proposal["policy_id"] = "bad\nidentity"
    elif fault == "infinite":
        proposal["navigation_limits"]["max_speed_mps"] = float("inf")
    elif fault == "boolean":
        proposal["navigation_limits"]["max_speed_mps"] = True
    elif fault == "negative":
        proposal["navigation_limits"]["max_speed_mps"] = -1
    elif fault == "zero":
        proposal["navigation_limits"]["max_speed_mps"] = 0
    elif fault == "missing_limit":
        proposal["navigation_limits"].pop("max_speed_mps")
    elif fault == "extra_limit":
        proposal["navigation_limits"]["new_limit"] = 1
    elif fault == "stop_over_speed":
        proposal["navigation_limits"]["stop_speed_mps"] = 1
    elif fault == "stop_over_yaw":
        proposal["navigation_limits"]["stop_yaw_rate_rps"] = 1
    elif fault == "mutate_snapshot":
        proposal["source_snapshot_hash"] = "old"
    elif fault == "real_execution":
        proposal["usable_for_real_execution"] = True
    else:
        proposal["policy_id"] = "x" * 9000
    with pytest.raises((ValueError, TypeError)):
        select(base, proposal)
