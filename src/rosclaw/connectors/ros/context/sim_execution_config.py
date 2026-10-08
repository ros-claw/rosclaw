"""Prepare source-bound SIM daemon configuration, never register or dispatch it."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.context.sim_attachment import derive_allowed_region_grid
from rosclaw.connectors.ros.context.sim_binding import propose_sim_fixture_binding
from rosclaw.connectors.ros.context.sim_execution_interfaces import propose_sim_execution_interfaces
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.sim_endpoints import freeze_sim_spawn


def prepare_sim_execution_config(
    workspace,
    model,
    urdf_bytes,
    *,
    policy,
    attachment,
    measured_map,
    start_pose,
    allowed_polygon,
    mission_id,
    now=None,
):
    """Reopen the compiled source and bind its real captured map/spawn/task area.

    No defaults exist for unknown robot geometry, frame, endpoint or spawn.
    The returned configuration still requires independent daemon source
    admission and actual SIM receipts before it is usable as acceptance.
    """
    now = now or datetime.now(UTC)
    if type(mission_id) is not str or not 1 <= len(mission_id) <= 256:
        raise ValueError("bounded nonempty mission id required")
    workspace = Path(workspace)
    raw = (workspace / "sim-body-workspace.yaml").read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("bounded compiled workspace manifest required")
    manifest = yaml.safe_load(raw)
    if type(manifest) is not dict:
        raise ValueError("compiled SIM workspace manifest required")
    saved = manifest.pop("manifest_hash", None)
    if (
        saved != digest(manifest)
        or manifest.get("schema_version") != "rosclaw.sim_body_workspace.v1"
        or manifest.get("status") != "CONTRACT_COMPILED"
        or manifest.get("usable_for_real_execution") is not False
    ):
        raise ValueError("compiled SIM workspace integrity required")
    proposal = propose_sim_fixture_binding(
        model, urdf_bytes, attachment=attachment, policy=policy, now=now
    )
    if proposal["status"] != "READY_FOR_SIM_WORKSPACE_COMPILATION":
        raise ValueError("complete fresh SIM Body source required")
    spec = proposal["specification"]
    execution = propose_sim_execution_interfaces(model, policy, spec, now=now)
    if execution["status"] != "READY_FOR_SIM_INTERFACE_COMPILATION":
        raise ValueError("complete fresh SIM execution interfaces required")
    resolver = BodyResolver(workspace=workspace)
    effective = resolver.get_effective_body(recompile_if_stale=False)
    if (
        manifest["body_snapshot_hash"] != effective.compute_hash()
        or any(
            manifest[key] != proposal[key]
            for key in ("source_snapshot_hash", "source_urdf_sha256", "fixture_policy_hash")
        )
        or manifest.get("execution_interface_proposal") != execution
        or hashlib.sha256(
            resolver.eurdf_profile_path.with_name("robot.urdf").read_bytes()
        ).hexdigest()
        != proposal["source_urdf_sha256"]
    ):
        raise ValueError("compiled Body differs from the captured source/interface proposal")
    actual_bindings = effective.provider_interfaces.get("ros_capability_bindings", {})
    for role in (
        "navigation.navigate_to_pose",
        "cleaning.enable",
        "cleaning.disable",
        "mapping.occupancy_map",
    ):
        if actual_bindings.get(role) != spec["ros_capability_bindings"][role]:
            raise ValueError("compiled Body interface mismatch: " + role)
    if (
        actual_bindings.get("coverage.execute", {}).get("name")
        != execution["endpoints"]["navigate_complete_coverage"]
    ):
        raise ValueError("compiled coverage action differs from the captured source")
    try:
        capture = datetime.fromisoformat(measured_map["captured_at"])
        fresh = 0 <= (now - capture).total_seconds() < 0.3
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("fresh typed measured-map capture required") from exc
    if (
        not fresh
        or measured_map.get("source") != actual_bindings["mapping.occupancy_map"]["name"]
        or measured_map.get("observation_complete") is not True
    ):
        raise ValueError("fresh complete actual bound map required")
    if (
        start_pose.get("body_snapshot_hash") != effective.compute_hash()
        or type(start_pose.get("run_id")) is not str
        or not start_pose["run_id"]
    ):
        raise ValueError("independent spawn must bind the compiled Body and fresh run")
    if (
        effective.frames.get("map") != measured_map.get("frame_id")
        or effective.provider_interfaces.get("sim_fixture_evidence", {})
        .get("collision_envelope", {})
        .get("physical_radius_m")
        != spec["physical_radius_m"]
        or actual_bindings.get("coverage.verify", {}).get("cleaning_polygon")
        != spec["cleaning_polygon"]
    ):
        raise ValueError("compiled frame/geometry differs from measured region source")
    region = derive_allowed_region_grid(
        measured_map,
        allowed_polygon=allowed_polygon,
        allowed_frame_id=effective.frames["map"],
        attachment=attachment,
        physical_radius_m=spec["physical_radius_m"],
        start_pose=start_pose,
        now=now,
    )
    yaw = start_pose.get("yaw")
    spawn = freeze_sim_spawn((start_pose["x"], start_pose["y"], yaw))
    grid = region["grid"]
    if len(region["legal_center_cells"]) > 5000:
        raise ValueError("bounded daemon repair-center contract required")
    centers = [
        [
            grid["origin"][0] + (i % grid["width"] + 0.5) * grid["resolution"],
            grid["origin"][1] + (i // grid["width"] + 0.5) * grid["resolution"],
        ]
        for i in region["legal_center_cells"]
    ]
    config = {
        "body_id": spec["body_id"],
        "body_snapshot_hash": effective.compute_hash(),
        "grid": grid,
        "physical_radius_m": spec["physical_radius_m"],
        "recovery_centers": centers,
        "mission_polygon": [list(p) for p in allowed_polygon],
        "configured_spawn": list(spawn),
        "endpoints": execution["endpoints"],
        "observation_topic": execution["observation_topic"],
        "repair_strategy": "pose_aware",
    }
    admission = {
        "schema_version": "rosclaw.sim_execution_config.v1",
        "status": "READY_FOR_SIM_DAEMON_SOURCE_ADMISSION",
        "evidence_role": "compiled_body_and_captured_initial_region_not_mission_acceptance",
        "run_id": start_pose["run_id"],
        "mission_id": mission_id,
        "source_snapshot_hash": model.snapshot_hash,
        "workspace_manifest_hash": saved,
        "region_preflight": region,
        "requires_independent_source_admission": True,
        "physical_acceptance_level": "NOT_RUN",
        "usable_for_real_execution": False,
        "actions_dispatched": False,
        "configuration": config,
    }
    admission["artifact_hash"] = digest(admission)
    return admission
