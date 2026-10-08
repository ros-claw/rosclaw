"""Reopen generic compiled SIM sources for Native fixture declarations only.

This is not live source admission, execution permission or physical acceptance.
No default robot profile, geometry, frame, endpoint or task region is invented.
"""

import hashlib
from copy import deepcopy
from pathlib import Path

import yaml

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def prepare_sim_native_fixture(root, admission):
    root = Path(root)
    if type(admission) is not dict:
        raise ValueError("typed generic execution proposal required")
    data = deepcopy(admission)
    saved = data.pop("artifact_hash", None)
    if (
        saved != digest(data)
        or data.get("schema_version") != "rosclaw.sim_execution_config.v1"
        or data.get("status") != "READY_FOR_SIM_DAEMON_SOURCE_ADMISSION"
        or data.get("requires_independent_source_admission") is not True
        or data.get("physical_acceptance_level") != "NOT_RUN"
        or data.get("usable_for_real_execution") is not False
        or data.get("actions_dispatched") is not False
        or any(
            type(data.get(k)) is not str or not 1 <= len(data[k]) <= 256
            for k in ("run_id", "mission_id")
        )
    ):
        raise ValueError("intact unexecuted generic SIM proposal required")
    raw = (root / "home/sim-body-workspace.yaml").read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("bounded compiled generic Body manifest required")
    manifest = yaml.safe_load(raw)
    if type(manifest) is not dict:
        raise ValueError("typed compiled generic Body manifest required")
    manifest_hash = manifest.pop("manifest_hash", None)
    if (
        manifest_hash != digest(manifest)
        or manifest_hash != data.get("workspace_manifest_hash")
        or manifest.get("schema_version") != "rosclaw.sim_body_workspace.v1"
        or manifest.get("status") != "CONTRACT_COMPILED"
        or manifest.get("usable_for_real_execution") is not False
        or manifest.get("source_snapshot_hash") != data.get("source_snapshot_hash")
    ):
        raise ValueError("generic Body manifest differs from execution proposal")
    resolver = BodyResolver(workspace=root / "home")
    effective = resolver.get_effective_body(recompile_if_stale=False)
    config = data.get("configuration")
    interface = manifest.get("execution_interface_proposal")
    if type(config) is not dict or type(interface) is not dict:
        raise ValueError("explicit generic configuration/interfaces required")
    evidence = effective.provider_interfaces.get("sim_fixture_evidence", {})
    envelope = evidence.get("collision_envelope", {})
    brush = effective.provider_interfaces.get("ros_capability_bindings", {}).get(
        "coverage.verify", {}
    )
    if (
        effective.compute_hash() != manifest["body_snapshot_hash"]
        or effective.compute_hash() != config.get("body_snapshot_hash")
        or effective.body_instance_id != config.get("body_id")
        or evidence.get("source_snapshot_hash") != data["source_snapshot_hash"]
        or evidence.get("execution_interface_proposal") != interface
        or envelope.get("physical_radius_m") != config.get("physical_radius_m")
        or config.get("endpoints") != interface.get("endpoints")
        or config.get("observation_topic") != interface.get("observation_topic")
        or evidence.get("cleaner_kind") != "SIMULATED_CLEANING"
        or evidence.get("evidence_domain") != "SIMULATION"
        or evidence.get("usable_for_real_execution") is not False
    ):
        raise ValueError("actual compiled generic Body/interfaces differ")
    for path in (root / "robot.urdf", resolver.eurdf_profile_path.with_name("robot.urdf")):
        raw = path.read_bytes()
        if (
            len(raw) > 2_000_000
            or hashlib.sha256(raw).hexdigest() != manifest["source_urdf_sha256"]
        ):
            raise ValueError("actual captured and compiled URDF must match")
    grid = config.get("grid")
    region = data.get("region_preflight")
    if (
        type(grid) is not dict
        or type(region) is not dict
        or grid != region.get("grid")
        or grid.get("frame_id") != effective.frames.get("map")
        or grid.get("cleaning_polygon") != brush.get("cleaning_polygon")
        or digest(config.get("mission_polygon")) != region.get("allowed_polygon_hash")
        or digest(grid) != region.get("grid_hash")
    ):
        raise ValueError("generic fixed region/frame/brush differs from compiled sources")
    config["generic_execution_proposal"] = {
        "run_id": data["run_id"],
        "mission_id": data["mission_id"],
        "proposal_hash": saved,
        "source_snapshot_hash": data["source_snapshot_hash"],
        "requires_independent_source_admission": True,
    }
    body = {
        "body_id": effective.body_instance_id,
        "effective_body_hash": effective.compute_hash(),
        "base_frame": effective.frames["base"],
        "map_frame": effective.frames["map"],
        "physical_radius_m": config["physical_radius_m"],
        "cleaning_polygon": deepcopy(grid["cleaning_polygon"]),
        "coverage_polygon": deepcopy(config["mission_polygon"]),
    }
    return {
        "status": "READY_FOR_NATIVE_DECLARATIONS_NOT_SOURCE_ADMITTED",
        "physical_acceptance": "NOT_RUN",
        "actions_dispatched": False,
        "usable_for_real_execution": False,
        "body": body,
        "execution_config": config,
    }
