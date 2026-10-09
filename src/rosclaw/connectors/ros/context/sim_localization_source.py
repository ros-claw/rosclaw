"""Explicit operator frozen-spawn AMCL prior; source proposal, never pose proof."""

import math
from copy import deepcopy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def apply_frozen_localization_prior(navigation, *, world_name, map_frame, declaration):
    """Initialize at declared spawn before launch, without live GT correction."""
    if (
        type(navigation) is not dict
        or set(navigation) != {"report", "parameters", "launch_nodes"}
        or type(navigation["report"]) is not dict
        or navigation["report"].get("parameters_hash") != digest(navigation["parameters"])
        or navigation["report"].get("launch_spec_hash") != digest(navigation["launch_nodes"])
        or navigation["report"].get("authorization") is not False
        or navigation["report"].get("usable_for_real_execution") is not False
    ):
        raise ValueError("intact unlaunched generic navigation source required")
    if (
        any(
            type(v) is not str or not 0 < len(v) <= 256 or any(c.isspace() for c in v)
            for v in (world_name, map_frame)
        )
        or type(declaration) is not dict
        or set(declaration)
        != {
            "schema_version",
            "source",
            "approved",
            "evidence_domain",
            "world_name",
            "map_frame",
            "world_to_map_xyyaw",
            "source_pose_kind",
        }
        or declaration["schema_version"] != "rosclaw.sim_localization_initial_prior.v1"
        or declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
        or declaration["world_name"] != world_name
        or declaration["map_frame"] != map_frame
        or declaration["source_pose_kind"] != "OPERATOR_FROZEN_SPAWN_PRIOR"
    ):
        raise ValueError("closed approved SIM world/map frozen-spawn prior required")
    transform = declaration["world_to_map_xyyaw"]
    spawn = navigation["report"].get("spawn_xyyaw")
    for value in (transform, spawn):
        if (
            type(value) is not list
            or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or any(abs(v) > 1000 for v in value[:2])
            or abs(value[2]) > math.pi
        ):
            raise ValueError(
                "bounded finite explicit rigid transform and frozen source spawn required"
            )
    result = deepcopy(navigation)
    node = result["report"]["nodes"]["localization"]
    parameters = result["parameters"][node]["ros__parameters"]
    if parameters.get("global_frame_id") != map_frame or parameters.get("use_sim_time") is not True:
        raise ValueError("exact actual localization SIM/map parameters required")
    x, y, yaw = spawn
    tx, ty, angle = transform
    pose = {
        "x": tx + math.cos(angle) * x - math.sin(angle) * y,
        "y": ty + math.sin(angle) * x + math.cos(angle) * y,
        "z": 0.0,
        "yaw": math.atan2(math.sin(yaw + angle), math.cos(yaw + angle)),
    }
    parameters.update(set_initial_pose=True, initial_pose=pose)
    report = {
        "schema_version": "rosclaw.sim_localization_initial_prior_source.v1",
        "declaration_hash": digest(declaration),
        "parent_navigation_report_hash": digest(navigation["report"]),
        "frozen_world_spawn_xyyaw": spawn,
        "declared_world_to_map_xyyaw": transform,
        "proposed_initial_pose": pose,
        "actual_localization_verified": False,
        "live_ground_truth_correction": False,
        "requires_actual_SDK_AMCL_TF_validation": True,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    result["report"]["localization_initial_prior"] = report
    result["report"]["parameters_hash"] = digest(result["parameters"])
    return result
