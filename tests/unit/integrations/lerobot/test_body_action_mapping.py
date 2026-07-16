"""Unit tests for body action mapping."""

from __future__ import annotations

import math

import pytest

from rosclaw.body.action_mapping import (
    ActionSpace,
    BodyActionSpace,
    generate_action_mapping,
    map_action_proposal_to_body,
    map_action_to_body,
    resolve_body_action_space,
    validate_action_mapping,
    validate_mapped_action,
)
from rosclaw.body.action_mapping.schema import MappingCompatibility


@pytest.fixture
def exact_policy_space() -> ActionSpace:
    return ActionSpace(
        representation="joint_position",
        names=["j1", "j2", "j3"],
        units=["rad", "rad", "rad"],
    )


@pytest.fixture
def exact_body_space() -> BodyActionSpace:
    return BodyActionSpace(
        body_id="mock_rh56",
        representation="joint_position",
        joint_names=["j1", "j2", "j3"],
        units=["rad", "rad", "rad"],
    )


def test_resolve_body_action_space_from_list() -> None:
    body = type("Body", (), {"profile_id": "b", "joints": [
        {"name": "j1", "type": "revolute", "limits": {"lower": -1, "upper": 1}},
        {"name": "j2", "type": "prismatic", "limits": {"lower": 0, "upper": 0.5}},
        {"name": "fixed", "type": "fixed"},
    ]})()
    space = resolve_body_action_space(body)
    assert space.body_id == "b"
    assert space.joint_names == ["j1", "j2"]
    assert space.units == ["rad", "m"]
    assert space.joint_limits["j1"]["lower"] == -1.0


def test_exact_mapping(exact_policy_space, exact_body_space) -> None:
    mapping = generate_action_mapping(exact_policy_space, exact_body_space)
    assert mapping.compatibility == MappingCompatibility.EXACT
    assert not mapping.is_blocked()
    assert len(mapping.joints) == 3


def test_convertible_mapping_unit_conversion() -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1", "j2"],
        units=["deg", "mm"],
    )
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1", "j2"],
        units=["rad", "m"],
    )
    mapping = generate_action_mapping(policy, body)
    assert mapping.compatibility == MappingCompatibility.CONVERTIBLE
    assert not mapping.is_blocked()


def test_partial_mapping_blocked_by_default(exact_policy_space, exact_body_space) -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1", "j2"],
        units=["rad", "rad"],
    )
    mapping = generate_action_mapping(policy, exact_body_space)
    assert mapping.compatibility == MappingCompatibility.PARTIAL
    assert mapping.is_blocked()


def test_partial_mapping_allowed() -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1", "j2"],
        units=["rad", "rad"],
    )
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1", "j2", "j3"],
        units=["rad", "rad", "rad"],
    )
    mapping = generate_action_mapping(policy, body, allow_partial=True)
    assert mapping.compatibility == MappingCompatibility.PARTIAL
    assert not mapping.is_blocked()


def test_incompatible_representation(exact_policy_space, exact_body_space) -> None:
    policy = ActionSpace(
        representation="cartesian_pose",
        names=["j1", "j2", "j3"],
        units=["rad", "rad", "rad"],
    )
    mapping = generate_action_mapping(policy, exact_body_space)
    assert mapping.compatibility == MappingCompatibility.UNKNOWN
    assert mapping.is_blocked()


def test_incompatible_units() -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1"],
        units=["N"],
    )
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1"],
        units=["rad"],
    )
    mapping = generate_action_mapping(policy, body)
    assert mapping.compatibility == MappingCompatibility.INCOMPATIBLE
    assert mapping.is_blocked()


def test_map_action_identity(exact_policy_space, exact_body_space) -> None:
    mapping = generate_action_mapping(exact_policy_space, exact_body_space)
    mapped = map_action_to_body([0.1, 0.2, 0.3], mapping)
    assert not mapped.blocked
    assert mapped.body_action_values == [0.1, 0.2, 0.3]
    assert mapped.body_joint_names == ["j1", "j2", "j3"]


def test_map_action_with_conversion() -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1"],
        units=["deg"],
    )
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1"],
        units=["rad"],
    )
    mapping = generate_action_mapping(policy, body)
    mapped = map_action_to_body([180.0], mapping)
    assert not mapped.blocked
    assert abs(mapped.body_action_values[0] - math.pi) < 1e-6


def test_map_action_chunked() -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1", "j2"],
        units=["rad", "rad"],
        is_chunked=True,
        chunk_size=2,
    )
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1", "j2"],
        units=["rad", "rad"],
    )
    mapping = generate_action_mapping(policy, body)
    mapped = map_action_to_body([0.1, 0.2, 0.3, 0.4], mapping, chunk_size=2)
    assert not mapped.blocked
    assert mapped.body_action_values == [0.1, 0.2, 0.3, 0.4]
    assert mapped.chunk_size == 2


def test_map_action_length_mismatch(exact_policy_space, exact_body_space) -> None:
    mapping = generate_action_mapping(exact_policy_space, exact_body_space)
    mapped = map_action_to_body([0.1, 0.2], mapping)
    assert mapped.blocked
    assert "length mismatch" in mapped.block_reasons[0]


def test_validate_action_mapping_blocks_partial() -> None:
    policy = ActionSpace(
        representation="joint_position",
        names=["j1"],
        units=["rad"],
    )
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1", "j2"],
        units=["rad", "rad"],
    )
    mapping = generate_action_mapping(policy, body)
    report = validate_action_mapping(mapping)
    assert report["blocked"] is True
    assert report["status"] == "blocked"


def test_validate_mapped_action_detects_nan() -> None:
    mapped = map_action_to_body(
        [0.1, float("nan")],
        generate_action_mapping(
            ActionSpace(representation="joint_position", names=["j1", "j2"], units=["rad", "rad"]),
            BodyActionSpace(body_id="b", representation="joint_position", joint_names=["j1", "j2"], units=["rad", "rad"]),
        ),
    )
    report = validate_mapped_action(mapped)
    assert report["blocked"] is True
    assert any("Invalid value" in issue for issue in report["issues"])


def test_map_action_proposal_v2() -> None:
    proposal = {
        "type": "rosclaw.action_proposal.v2",
        "action": {
            "representation": "joint_position",
            "names": ["j1", "j2"],
            "units": ["rad", "rad"],
            "values": [0.5, -0.5],
            "shape": [2],
        },
    }
    body = BodyActionSpace(
        body_id="b",
        representation="joint_position",
        joint_names=["j1", "j2"],
        units=["rad", "rad"],
    )
    policy = ActionSpace.from_proposal_v2(proposal)
    mapping = generate_action_mapping(policy, body)
    mapped = map_action_proposal_to_body(proposal, mapping)
    assert not mapped.blocked
    assert mapped.body_action_values == [0.5, -0.5]
