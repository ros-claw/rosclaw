"""Product status tests for the LeRobot Bridge golden path (终稿 §8.1)."""

from __future__ import annotations

import pytest

from rosclaw.product import load_product_status


@pytest.fixture
def bridge() -> dict:
    status = load_product_status()
    return status["golden_paths"]["rh56_single_step"]


def test_lerobot_bridge_status_present(bridge) -> None:
    assert bridge["display"]["en"]
    assert bridge["capability"] == "rh56.single_step"
    assert bridge["integration"]["family"] == "lerobot"


def test_bridge_version_1_0_1(bridge) -> None:
    assert bridge["integration"]["bridge_version"] == "1.0.1"


def test_reference_policy_declared(bridge) -> None:
    policy = bridge["integration"]["reference_policy"]
    assert policy["type"] == "rosclaw_rh56_reference"
    assert policy["action_dim"] == 6
    assert policy["semantic_source"] == "explicit_policy_contract"


def test_reference_bodies_declared(bridge) -> None:
    bodies = bridge["integration"]["reference_bodies"]
    assert "inspire_rh56_left" in bodies
    assert "inspire_rh56_right" in bodies


def test_supported_modes_declared(bridge) -> None:
    modes = bridge["integration"]["supported_execution_modes"]
    assert "proposal_only" in modes
    assert "shadow" in modes
    assert "single_step_receding_horizon" in modes


def test_unsupported_modes_declared(bridge) -> None:
    unsupported = bridge["integration"]["unsupported"]
    for item in (
        "can_rh56_execution",
        "open_loop_action_chunks",
        "multiple_active_sessions",
        "unattended_execution",
    ):
        assert item in unsupported


def test_agent_discoverable_true(bridge) -> None:
    assert bridge["agent_discoverable"] is True


def test_agent_ready_false_before_blackbox(bridge) -> None:
    # agent_ready may only flip after the external Agent black-box passes.
    assert bridge["agent_ready"] is False


def test_support_tier_not_overclaimed(bridge) -> None:
    assert bridge["agent_discoverable"] is True
    assert bridge["agent_ready"] is False
    assert bridge["support_tier"] != "H5_AGENT_BLACKBOX_VERIFIED"
    assert bridge["support_tier"] == "H1_CONTRACT_VERIFIED"
    assert bridge["candidate_tier"] == "H4_HARDWARE_ACTUATION_VERIFIED"


def test_hardware_evidence_records_present(bridge) -> None:
    evidence = bridge.get("evidence", [])
    assert evidence, "real-hardware evidence records must stay attached"
    for record in evidence:
        assert record.get("id")
        assert record.get("observation_scope")
