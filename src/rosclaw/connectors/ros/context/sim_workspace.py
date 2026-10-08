"""Create a new simulator-owned Body workspace from fresh explicit evidence.

No existing product Body is overwritten. No launch, ROS write, dependency
installation, navigation or physical acceptance happens in this compiler.
"""

import hashlib
import math
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from rosclaw.body.compiler import compute_checksum
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyYaml, CalibrationYaml, EurdfProfile
from rosclaw.connectors.ros.context.sim_binding import propose_sim_fixture_binding
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def compile_sim_body_workspace(destination, model, urdf_bytes, *, attachment, policy, now=None):
    proposal = propose_sim_fixture_binding(
        model, urdf_bytes, attachment=attachment, policy=policy, now=now
    )
    if proposal["status"] != "READY_FOR_SIM_WORKSPACE_COMPILATION":
        raise ValueError("SIM Body compilation requires a complete fresh approved fixture proposal")
    specification = proposal["specification"]
    robot = ET.fromstring(urdf_bytes)
    joints = []
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        limits = (
            {key: float(value) for key, value in limit.attrib.items()} if limit is not None else {}
        )
        if any(not math.isfinite(value) for value in limits.values()):
            raise ValueError("finite observed joint limits required")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "limits": limits,
            }
        )
    profile_id = "observed_sim_" + proposal["source_urdf_sha256"][:16]
    bindings = deepcopy(specification["ros_capability_bindings"])
    bindings["coverage.verify"] = {
        "name": "rosclaw.coverage_verifier.v1",
        "cleaning_polygon": specification["cleaning_polygon"],
    }
    profile = EurdfProfile(
        profile_id=profile_id,
        profile_version="1.0.0",
        vendor="UNKNOWN; observed robot_description",
        model=specification["model_identity"],
        display_name=specification["model_identity"],
        description="Fresh observed collision model; explicit simulator-only cleaning declaration.",
        assets={"urdf": "robot.urdf"},
        identity={"robot_class": "mobile_base"},
        frames={"root": specification["frames"]["base"], **specification["frames"]},
        joints=joints,
        sensors=[
            {
                "name": "observed_lidar",
                "type": "lidar",
                "parent_link": specification["frames"]["lidar"],
            }
        ],
        actuators=[],  # A joint or drive topic does not identify a physical actuator provider.
        provider_interfaces={
            "ros_capability_bindings": bindings,
            "sim_fixture_evidence": {
                "source_snapshot_hash": model.snapshot_hash,
                "source_urdf_sha256": proposal["source_urdf_sha256"],
                "fixture_policy_hash": proposal["fixture_policy_hash"],
                "attachment_hash": proposal["attachment"]["attachment_hash"],
                "collision_envelope": proposal["candidate"]["geometry"],
                "cleaner_kind": "SIMULATED_CLEANING",
                "evidence_domain": "SIMULATION",
                "usable_for_real_execution": False,
            },
        },
        capability_hints={
            "sim_fixture_declared": [
                "navigation.navigate_to_pose",
                "cleaning.enable",
                "cleaning.disable",
            ]
        },
        safety={"safety_level": "STRICT", "environment": {"real_robot_execution_allowed": False}},
        sandbox={"compatible_engines": ["gazebo"], "preferred_engine": "gazebo"},
        metadata={
            "evidence_domain": "SIMULATION",
            "urdf_sha256": proposal["source_urdf_sha256"],
            "physical_radius_m": specification["physical_radius_m"],
            "cleaning_polygon": specification["cleaning_polygon"],
            "cleaner_kind": "SIMULATED_CLEANING",
            "source_snapshot_hash": model.snapshot_hash,
            "fixture_policy_hash": proposal["fixture_policy_hash"],
            "attachment_hash": proposal["attachment"]["attachment_hash"],
            "binding_proposal_hash": proposal["proposal_hash"],
            "physical_acceptance_level": "NOT_RUN",
            "usable_for_real_execution": False,
        },
    )
    destination = Path(destination)
    destination.mkdir(parents=False, exist_ok=False)
    resolver = BodyResolver(workspace=destination)
    resolver.ensure_body_dir()
    resolver.eurdf_profile_path.write_text(yaml.safe_dump(profile.to_dict(), sort_keys=False))
    resolver.eurdf_profile_path.with_name("robot.urdf").write_bytes(urdf_bytes)
    checksum = compute_checksum(resolver.eurdf_profile_path)
    uri = f"rosclaw://eurdf/{profile_id}@1.0.0"
    resolver.eurdf_lock_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "rosclaw.eurdf_lock.v1",
                "profile_id": profile_id,
                "profile_version": "1.0.0",
                "uri": uri,
                "source": "observed_SIM_fixture",
                "checksum": checksum,
            }
        )
    )
    body = BodyYaml(
        body_instance={"id": specification["body_id"], "robot_model": profile_id},
        model_ref={"eurdf_uri": uri, "checksum": checksum},
        installed_components={
            "sensors": {"observed_lidar": {"installed": True, "status": "available"}},
            "actuators": {},
        },
        agent_policy={"direct_real_robot_execution_allowed": False},
        metadata={
            "evidence_domain": "SIMULATION",
            "source": "evidence_bound_fixture_compilation",
            "fixture_policy_hash": proposal["fixture_policy_hash"],
        },
    )
    resolver.body_yaml_path.write_text(yaml.safe_dump(body.to_dict(), sort_keys=False))
    resolver.calibration_yaml_path.write_text(yaml.safe_dump(CalibrationYaml().to_dict()))
    effective = resolver.recompile_effective_body()
    reopened = BodyResolver(workspace=destination).get_effective_body(recompile_if_stale=False)
    body_hash = effective.compute_hash()
    if body_hash != reopened.compute_hash() or body_hash != reopened.effective_body_hash:
        raise ValueError("compiled Body snapshot integrity mismatch")
    if (
        hashlib.sha256(resolver.eurdf_profile_path.with_name("robot.urdf").read_bytes()).hexdigest()
        != proposal["source_urdf_sha256"]
    ):
        raise ValueError("compiled URDF source changed")
    manifest = {
        "schema_version": "rosclaw.sim_body_workspace.v1",
        "status": "CONTRACT_COMPILED",
        "body_snapshot_hash": body_hash,
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": proposal["source_urdf_sha256"],
        "fixture_policy_hash": proposal["fixture_policy_hash"],
        "attachment_hash": proposal["attachment"]["attachment_hash"],
        "evidence_domain": "SIMULATION",
        "physical_acceptance_level": "NOT_RUN",
        "usable_for_real_execution": False,
        "execution_entry": "request_action",
        "direct_actions_dispatched": False,
    }
    manifest["manifest_hash"] = digest(manifest)
    (destination / "sim-body-workspace.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
