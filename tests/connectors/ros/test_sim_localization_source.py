"""Explicit synthetic pose-prior source contracts; no localization or World."""

import math
from copy import deepcopy

import pytest

from rosclaw.connectors.ros.context.sim_localization_source import apply_frozen_localization_prior
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros import test_sim_navigation_source as navigation
from tests.connectors.ros.test_sim_navigation_source import source  # noqa: F401


def declaration():
    return {
        "schema_version": "rosclaw.sim_localization_initial_prior.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "world_name": "synthetic_world",
        "map_frame": "declared_map",
        "world_to_map_xyyaw": [1, 2, math.pi / 2],
        "source_pose_kind": "OPERATOR_FROZEN_SPAWN_PRIOR",
    }


def apply(result, policy):
    return apply_frozen_localization_prior(
        result, world_name="synthetic_world", map_frame="declared_map", declaration=policy
    )


def test_explicit_transform_initializes_only_operator_frozen_source_spawn(source):  # noqa: F811
    original = navigation.generate(source)
    prior = deepcopy(original)
    result = apply(original, declaration())
    params = result["parameters"]["/explicit_nav/localization"]["ros__parameters"]
    assert params["set_initial_pose"] is True
    assert params["initial_pose"]["x"] == pytest.approx(1.2)
    assert params["initial_pose"]["y"] == pytest.approx(2.1)
    assert params["initial_pose"]["yaw"] == pytest.approx(0.3 + math.pi / 2)
    assert original == prior  # Source inputs are immutable.
    report = result["report"]["localization_initial_prior"]
    assert report["actual_localization_verified"] is False
    assert report["live_ground_truth_correction"] is False
    assert report["authorization"] is False
    assert result["report"]["parameters_hash"] == digest(result["parameters"])
    for name in original["parameters"]:
        if name != "/explicit_nav/localization":
            assert result["parameters"][name] == original["parameters"][name]


@pytest.mark.parametrize(
    "fault",
    [
        "world",
        "frame",
        "approval",
        "real",
        "live_pose",
        "nan",
        "bool",
        "missing",
        "extra",
        "source_mutation",
    ],
)
def test_unregistered_or_live_correction_never_becomes_an_initialization_prior(source, fault):  # noqa: F811
    original, policy = navigation.generate(source), declaration()
    if fault == "world":
        policy["world_name"] = "other_world"
    elif fault == "frame":
        policy["map_frame"] = "other_map"
    elif fault == "approval":
        policy["approved"] = False
    elif fault == "real":
        policy["evidence_domain"] = "REAL"
    elif fault == "live_pose":
        policy["source_pose_kind"] = "LIVE_GROUND_TRUTH_CORRECTION"
    elif fault == "nan":
        policy["world_to_map_xyyaw"][0] = float("nan")
    elif fault == "bool":
        policy["world_to_map_xyyaw"][0] = True
    elif fault == "missing":
        policy.pop("world_to_map_xyyaw")
    elif fault == "extra":
        policy["robot_profile"] = "known_robot"
    else:
        original["parameters"]["/explicit_nav/localization"]["ros__parameters"][
            "set_initial_pose"
        ] = True
    with pytest.raises(ValueError):
        apply(original, policy)
