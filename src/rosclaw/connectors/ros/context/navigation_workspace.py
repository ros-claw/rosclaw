"""Compile a new navigation-only SIM Body workspace, never admit or execute it.

Reuse the existing N04 sim_workspace compiler's URDF/profile/Body/lock/reopen
flow without its mandatory cleaning attachment or coverage configuration.
This operator fixture workflow opens no ROS connection and grants no authority.
"""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from rosclaw.body.compiler import compute_checksum
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyYaml, CalibrationYaml, EurdfProfile
from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.contracts.common import content_hash


def compile_navigation_body_workspace(
    destination, model, urdf_bytes, requirements, *, policy, now=None
):
    """Require fresh reviewed evidence and explicit source-bound SIM policy.

    The destination must not exist. No known robot profile, conventional frame,
    inferred safety limit, simulated cleaner or live Body installation is used.
    A compiled snapshot still needs independent source admission and guarded
    physics acceptance; policy.approved is a fixture declaration, not a Permit.
    """
    review = review_navigation_candidate(model, urdf_bytes, requirements, now=now)
    if review["status"] != "PROPOSED":
        raise ValueError(
            "complete fresh navigation evidence required: " + str(review["unknown_fields"])
        )
    keys = {
        "schema_version",
        "source",
        "approved",
        "policy_id",
        "body_id",
        "source_snapshot_hash",
        "source_urdf_sha256",
        "navigation_requirements_hash",
        "navigation_limits",
        "witness_declaration",
    }
    candidate = review["candidate"]
    if (
        type(policy) is not dict
        or set(policy) != keys
        or policy["schema_version"] != "rosclaw.navigation_sim_fixture_policy.v1"
        or policy["source"] != "simulator_operator_fixture_policy"
        or policy["approved"] is not True
        or any(type(policy[k]) is not str or not policy[k] for k in ("policy_id", "body_id"))
        or policy["source_snapshot_hash"] != model.snapshot_hash
        or policy["source_urdf_sha256"] != candidate["source_urdf_sha256"]
        or policy["navigation_requirements_hash"] != review["requirements_hash"]
    ):
        raise ValueError("explicit matching navigation-only SIM fixture policy required")
    declaration = policy["witness_declaration"]
    limits = policy["navigation_limits"]
    if (
        type(declaration) is not dict
        or set(declaration) != {"observation_topic", "lease_service", "run_id"}
        or type(limits) is not dict
        or set(limits)
        != {
            "max_goal_distance_m",
            "max_duration_sec",
            "goal_position_tolerance_m",
            "goal_yaw_tolerance_rad",
            "max_speed_mps",
            "max_yaw_rate_rps",
            "stop_speed_mps",
            "stop_yaw_rate_rps",
        }
    ):
        raise ValueError(
            "explicit independent SIM observer/lease and all navigation limits required"
        )
    navigation = candidate["interfaces"]["navigation.navigate_to_pose"][0]
    SimulationNavigationBinding(
        policy["body_id"],
        "0" * 64,
        navigation["name"],
        declaration["lease_service"],
        declaration["observation_topic"],
        candidate["frames"]["map"],
        declaration["run_id"],
        **limits,
    )
    # JSON round-trip freezes only ordinary configuration, not caller references.
    policy = json.loads(json.dumps(policy, allow_nan=False))
    declaration = policy["witness_declaration"]
    policy_hash = content_hash("navfixturepolicy", policy)
    review_hash = content_hash("navobservationreview", review)
    robot = ET.fromstring(urdf_bytes)
    joints = []
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        values = (
            {key: float(value) for key, value in limit.attrib.items()} if limit is not None else {}
        )
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("finite observed joint limits required")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "limits": values,
            }
        )
    bindings = {
        role: deepcopy(candidate["interfaces"][role][0])
        for role in (
            "navigation.navigate_to_pose",
            "state.odometry",
            "mapping.occupancy_map",
            "sensing.lidar",
        )
    }
    bindings.update(
        {
            "safety.simulation_lease": {
                "name": declaration["lease_service"],
                "srv_type": "std_srvs/srv/SetBool",
            },
            "safety.simulation_observation": {
                "name": declaration["observation_topic"],
                "msg_type": "std_msgs/msg/String",
                "evidence_domain": "GAZEBO_PHYSICS",
                "run_id": declaration["run_id"],
                "urdf_sha256": candidate["source_urdf_sha256"],
            },
        }
    )
    profile_id = "observed_sim_navigation_" + candidate["source_urdf_sha256"][:16]
    profile = EurdfProfile(
        profile_id=profile_id,
        profile_version="1.0.0",
        vendor="UNKNOWN; observed robot_description",
        model=candidate["geometry"]["model_identity"],
        display_name=candidate["geometry"]["model_identity"],
        description="Source-bound navigation-only SIM fixture; physical admission not evaluated.",
        assets={"urdf": "robot.urdf"},
        identity={"robot_class": "mobile_base"},
        frames={"root": candidate["frames"]["base"], **candidate["frames"]},
        joints=joints,
        sensors=[
            {"name": "observed_lidar", "type": "lidar", "parent_link": candidate["frames"]["lidar"]}
        ],
        actuators=[],
        provider_interfaces={
            "ros_capability_bindings": bindings,
            "sim_fixture_evidence": {
                "source_snapshot_hash": model.snapshot_hash,
                "source_urdf_sha256": candidate["source_urdf_sha256"],
                "navigation_requirements_hash": review["requirements_hash"],
                "navigation_review_hash": review_hash,
                "fixture_policy_hash": policy_hash,
                "collision_envelope": deepcopy(candidate["geometry"]),
                "evidence_domain": "SIMULATION",
                "binding_verified": False,
                "physical_acceptance_level": "NOT_RUN",
                "usable_for_real_execution": False,
            },
        },
        capability_hints={"sim_fixture_declared": ["navigation.navigate_to_pose"]},
        safety={
            "safety_level": "STRICT",
            "environment": {"real_robot_execution_allowed": False},
            "navigation_simulation": deepcopy(policy["navigation_limits"]),
        },
        sandbox={"compatible_engines": ["gazebo"], "preferred_engine": "gazebo"},
    )
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("new workspace destination must not be a symlink")
    destination = destination.resolve()
    destination.mkdir(parents=False, exist_ok=False)
    resolver = BodyResolver(workspace=destination)
    resolver.ensure_body_dir()
    resolver.eurdf_profile_path.write_text(yaml.safe_dump(profile.to_dict(), sort_keys=False))
    urdf_path = resolver.eurdf_profile_path.with_name("robot.urdf")
    urdf_path.write_bytes(urdf_bytes)
    checksum = compute_checksum(resolver.eurdf_profile_path)
    uri = f"rosclaw://eurdf/{profile_id}@1.0.0"
    resolver.eurdf_lock_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "rosclaw.eurdf_lock.v1",
                "profile_id": profile_id,
                "profile_version": "1.0.0",
                "uri": uri,
                "source": "observed_navigation_SIM_fixture",
                "checksum": checksum,
            }
        )
    )
    body = BodyYaml(
        body_instance={"id": policy["body_id"], "robot_model": profile_id},
        model_ref={"eurdf_uri": uri, "checksum": checksum},
        installed_components={
            "sensors": {"observed_lidar": {"installed": True, "status": "available"}},
            "actuators": {},
        },
        agent_policy={"direct_real_robot_execution_allowed": False},
        metadata={
            "source": "evidence_bound_navigation_fixture_compilation",
            "evidence_domain": "SIMULATION",
        },
    )
    resolver.body_yaml_path.write_text(yaml.safe_dump(body.to_dict(), sort_keys=False))
    resolver.calibration_yaml_path.write_text(yaml.safe_dump(CalibrationYaml().to_dict()))
    effective = resolver.recompile_effective_body()
    reopened = BodyResolver(workspace=destination).get_effective_body(recompile_if_stale=False)
    binding = SimulationNavigationBinding.from_body(reopened)
    if (
        effective.compute_hash() != binding.body_snapshot_hash
        or hashlib.sha256(urdf_path.read_bytes()).hexdigest() != candidate["source_urdf_sha256"]
    ):
        raise ValueError("compiled Body/URDF source changed")
    config = {
        "executor_kind": "navigation_only",
        "body_id": binding.body_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
    }
    config_path = destination / "execution_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    evidence_paths = []
    for name, value in (
        ("source-snapshot.json", model.to_dict()),
        ("navigation-requirements.json", requirements),
        ("fixture-policy.json", policy),
        ("navigation-review.json", review),
    ):
        path = destination / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        evidence_paths.append(path)
    manifest = {
        "schema_version": "rosclaw.navigation_body_workspace.v1",
        "status": "CONTRACT_COMPILED_NOT_ADMITTED",
        "body_snapshot_hash": binding.body_snapshot_hash,
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": candidate["source_urdf_sha256"],
        "navigation_review_hash": review_hash,
        "fixture_policy_hash": policy_hash,
        "requirements_hash": review["requirements_hash"],
        "evidence_domain": "SIMULATION",
        "binding_verified": False,
        "physical_acceptance_level": "NOT_RUN",
        "usable_for_real_execution": False,
        "execution_entry": "request_action",
        "direct_actions_dispatched": False,
        "file_sha256": {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                resolver.eurdf_profile_path,
                resolver.eurdf_lock_path,
                resolver.body_yaml_path,
                resolver.calibration_yaml_path,
                resolver.effective_body_path,
                urdf_path,
                config_path,
                *evidence_paths,
            )
        },
    }
    manifest["manifest_hash"] = content_hash("navbodyworkspace", manifest)
    (destination / "navigation-body-workspace.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest
