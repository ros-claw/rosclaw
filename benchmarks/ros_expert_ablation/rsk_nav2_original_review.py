"""Independent original-file review of a stopped, never-admitted SDK replica.

This is a read-only reviewer, not an SDK owner, a model provider or a pilot gate.
It never renews the original case or turns observations into motion authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import SDK_IMAGE
from benchmarks.ros_expert_ablation.rsk_episode_runner import OriginalRef, _read
from integrations.ros_probe.acceptance.simulation_atomic_read import read_bounded_regular
from rosclaw.connectors.ros.cli.ros_cli import _read_discovery_urdf
from rosclaw.connectors.ros.context.collision_assets import bind_prepared_collision_originals
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.engineering.probe_planner import (
    assess_diagnostic_followup,
    plan_diagnostic_probes,
)
from rosclaw.connectors.ros.intelligence import build_system_model
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from rosclaw.contracts.common import content_hash


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _original(root, path, limit=8_000_000):
    root, path = Path(root).resolve(strict=True), Path(path).absolute()
    path.relative_to(root)
    for component in [path, *path.parents]:
        if component == root:
            break
        if component.is_symlink():
            raise ValueError("original path cannot traverse a symlink")
    raw = read_bounded_regular(path, limit)
    ref = OriginalRef(path=str(path.relative_to(root)), sha256=_sha(raw))
    return json.loads(_read(root, ref, limit=limit)), ref.model_dump(mode="json")


def _check_sink(sink, run):
    required = {
        "run_id",
        "body_snapshot_hash",
        "terminal",
        "action_id",
        "daemon_instance_id",
        "lease_expires_ns",
        "lease_updates",
        "received_commands",
        "published_velocity_messages",
    }
    if (
        type(sink) is not dict
        or not required <= set(sink)
        or sink["run_id"] != run
        or sink["body_snapshot_hash"] is not None
        or sink["terminal"] is not False
        or sink["action_id"] is not None
        or sink["daemon_instance_id"] is not None
        or any(
            type(sink[key]) is not int or sink[key] != 0
            for key in [
                "lease_expires_ns",
                "lease_updates",
                "received_commands",
                "published_velocity_messages",
            ]
        )
    ):
        raise ValueError("original source-only sink must remain never-admitted and zero-command")


def _check_pointer(pointer, native, window, phase):
    if (
        type(pointer) is not dict
        or set(pointer) != {"file", "sha256", "captured_at", "received_monotonic_ns"}
        or type(pointer["file"]) is not str
        or re.fullmatch(r"original-native-[0-9]{3,6}\.json", pointer["file"]) is None
        or pointer["sha256"] != window["native_" + phase]["sha256"]
        or pointer["file"] != window["native_original_file_" + phase]
        or pointer["captured_at"] != native["captured_at"]
        or native.get("environment", {}).get("rmw") != "rmw_cyclonedds_cpp"
        or native.get("environment", {}).get("ros_generation") != "ros2"
        or native.get("completeness", {}).get("graph") is not True
        or type(pointer["received_monotonic_ns"]) is not int
        or pointer["received_monotonic_ns"] < 0
        or type(window.get("host_observed_monotonic_ns_" + phase)) is not int
        or not 0
        <= window["host_observed_monotonic_ns_" + phase] - pointer["received_monotonic_ns"]
        <= 500_000_000
    ):
        raise ValueError("original Native/pointer identity incomplete or changed")


def _mounts(container):
    values = container["Mounts"]
    destinations = [row["Destination"] for row in values]
    if len(destinations) != len(set(destinations)):
        raise ValueError("duplicate SDK mount destinations")
    return {row["Destination"]: row for row in values}


def review_stopped_owned_nav2_window(
    world,
    *,
    root,
    worktree,
    capture_source,
    original_run_id,
    original_graph,
    plan,
    selected,
    source_urdf_sha256,
    requirements_hash,
    staging_hash,
):
    """Review originals for an existing replica, without claiming a dispatch.

    The eventual closed owner/provider must pin its actual source, policy,
    plan/recipe, budget and four original entry attestations before SDK use.
    This review alone starts zero Worlds and grants no permission.
    """
    root, worktree = Path(root).resolve(strict=True), Path(worktree).resolve(strict=True)
    world = Path(world).resolve(strict=True)
    world.relative_to(root)
    if re.fullmatch(r"[0-9a-f]{40}", capture_source) is None:
        raise ValueError("exact original capture revision required")
    expected_plan = plan_diagnostic_probes(
        original_graph,
        issue_ids=sorted({h.issue_id for h in plan.hypotheses}),
        budget_sec=plan.budget_sec,
        max_probes=plan.max_probes,
    )
    if expected_plan != plan:
        raise ValueError("original diagnostic plan differs from Core recomputation")
    references = {}
    registration, references["world_registration"] = _original(root, world / "preregistration.json")
    settings, references["settings"] = _original(root, world / "settings.json")
    window_path = world / "read-only-window/window-original-manifest.json"
    window, references["window"] = _original(root, window_path)
    run = settings["run_id"]
    if (
        window["schema_version"] != "rosclaw.rsk_owned_nav2_observation_window.v2"
        or registration["source"] != capture_source
        or window["source_revision"] != capture_source
        or window["run_id"] != run
        or run == original_run_id
        or re.fullmatch(r"[0-9a-f]{32}", run) is None
        or settings["engineering_capture"]["run_id"] != run
        or window["requested_operation"] != selected.request.operation
        or selected not in plan.probes
        or plan.graph_hash != original_graph.graph_hash
        or window["source_body_snapshot_hash"] is not None
    ):
        raise ValueError("replica source/run/original diagnostic intent mismatch")
    for name in ["admission.json", "proof.json"]:
        try:
            os.lstat(world / "execution/execution-control" / name)
        except FileNotFoundError:
            continue
        raise ValueError("replica has an admission or execution proof file")
    created, references["container_before"] = _original(root, world / "container-before-start.json")
    stopped, references["container_after"] = _original(root, world / "container-after-stop.json")
    actual = json.loads(
        subprocess.check_output(["docker", "inspect", stopped["Id"]], text=True, timeout=15)
    )[0]
    for key in ["Id", "Image", "State", "Config", "HostConfig"]:
        if actual[key] != stopped[key]:
            raise ValueError("actual stopped SDK differs from original container metadata")
    if sorted(json.dumps(m, sort_keys=True) for m in actual["Mounts"]) != sorted(
        json.dumps(m, sort_keys=True) for m in stopped["Mounts"]
    ):
        raise ValueError("actual stopped SDK mount records changed")
    if (
        created["Id"] != stopped["Id"]
        or stopped["State"]["Running"]
        or stopped["State"]["Pid"] != 0
        or stopped["State"]["ExitCode"] != 0
    ):
        raise ValueError("same successful stopped owned SDK container required")
    for container in [created, stopped]:
        host = container["HostConfig"]
        mounts = _mounts(container)
        if (
            container["Image"] != SDK_IMAGE
            or host["NetworkMode"] != "none"
            or host["ReadonlyRootfs"] is not True
            or host["Privileged"]
            or host.get("Devices")
            or host.get("DeviceRequests")
            or host.get("PidMode") == "host"
            or host.get("IpcMode") == "host"
            or host.get("CapDrop") != ["ALL"]
            or "no-new-privileges" not in host.get("SecurityOpt", [])
            or container["Config"]["User"] in {"", "0", "0:0", "root"}
            or not 0 < host["NanoCpus"] <= 6_000_000_000
            or not 0 < host["Memory"] <= 10 * 1024**3
            or not 0 < host["PidsLimit"] <= 512
            or not 0 < host["ShmSize"] <= 512 * 1024**2
            or container["Config"]["Entrypoint"] != ["bash"]
            or container["Config"]["Cmd"]
            != [
                "-c",
                "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
                "export PYTHONPATH=/repo/src:/repo/integrations/ros_probe/acceptance:"
                "$PYTHONPATH && python3 /repo/integrations/ros_probe/acceptance/"
                "source_binding_fixture.py",
            ]
        ):
            raise ValueError("fixed isolated bounded non-root SDK resources required")
        for target, source, writable in [
            ("/repo", worktree, False),
            ("/assets", world / "assets", False),
            ("/settings.json", world / "settings.json", False),
            ("/evidence", world / "execution", True),
        ]:
            mount = mounts.get(target, {})
            if mount.get("Source") != str(source) or mount.get("RW") is not writable:
                raise ValueError("SDK mounts differ from declared source/assets/evidence")
        for mount in container["Mounts"]:
            if mount.get("RW") is True and mount["Destination"] != "/evidence":
                raise ValueError("undeclared writable SDK bind mount")
        if any(
            value not in container["Config"]["Env"]
            for value in ["ROS_LOCALHOST_ONLY=1", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"]
        ):
            raise ValueError("fixed isolated original RMW required")
    for name in [
        "benchmarks/ros_expert_ablation/rsk_nav2_observation_window.py",
        "integrations/ros_probe/ros2/probe.py",
        "integrations/ros_probe/acceptance/source_binding_fixture.py",
        "integrations/ros_probe/acceptance/simulation_native_capture.py",
        "integrations/ros_probe/acceptance/simulation_atomic_read.py",
    ]:
        original_code = subprocess.check_output(
            ["git", "show", capture_source + ":" + name], cwd=worktree, timeout=15
        )
        if read_bounded_regular(worktree / name, 2_000_000) != original_code:
            raise ValueError("original SDK capture implementation changed after capture")
    urdf = read_bounded_regular(world / "assets/robot.urdf", 8_000_000)
    if _sha(urdf) != source_urdf_sha256 or original_graph.source_urdf_sha256 != source_urdf_sha256:
        raise ValueError("replica URDF differs from original diagnostic Body context")
    requirements, references["requirements"] = _original(
        root, world / "execution/engineering-original-requirements.json"
    )
    staging = {
        name: yaml.safe_load(read_bounded_regular(world / "assets" / name, 8_000_000))
        for name in ["nav2.yaml", "bringup.json"]
    }
    if (
        content_hash("navrequirements", requirements) != requirements_hash
        or original_graph.requirements_hash != requirements_hash
        or content_hash("acistaging", staging) != staging_hash
    ):
        raise ValueError("replica requirements/staging differ from original context")
    originals = {}
    for field in [
        "native_before",
        "native_after",
        "pointer_before",
        "pointer_after",
        "sink_before",
        "sink_after",
    ]:
        ref = window[field]
        name = ref["file"]
        if Path(name).name != name:
            raise ValueError("closed local original window filenames required")
        path = window_path.parent / name
        raw = read_bounded_regular(path, 8_000_000)
        if _sha(raw) != ref["sha256"] or len(raw) != ref["bytes"]:
            raise ValueError("window original SHA/size mismatch")
        originals[field] = json.loads(raw)
    for phase in ["before", "after"]:
        sink = originals["sink_" + phase]
        _check_sink(sink, run)
        pointer, native = originals["pointer_" + phase], originals["native_" + phase]
        _check_pointer(pointer, native, window, phase)
        # Resolve the producer's immutable original frame as well as its copy.
        raw = read_bounded_regular(world / "execution" / pointer["file"], 8_000_000)
        if _sha(raw) != pointer["sha256"]:
            raise ValueError("copied Native differs from immutable SDK original frame")
    before, after = originals["native_before"], originals["native_after"]
    stamps = [datetime.fromisoformat(n["captured_at"]) for n in [before, after]]
    steady_span = (
        originals["pointer_after"]["received_monotonic_ns"]
        - originals["pointer_before"]["received_monotonic_ns"]
    ) / 1_000_000_000
    if (
        any(stamp.tzinfo is None for stamp in stamps)
        or not stamps[0] < stamps[1] <= datetime.now(UTC)
        or not 5 <= (stamps[1] - stamps[0]).total_seconds() <= 10
        or not 5 <= steady_span <= 10
        or not 5.6 <= window["host_window_elapsed_sec"] <= 15
        or not 5.6
        <= (
            window["host_observed_monotonic_ns_after"] - window["host_observed_monotonic_ns_before"]
        )
        / 1_000_000_000
        <= window["host_window_elapsed_sec"]
        or window["native_capture_span_sec"] != (stamps[1] - stamps[0]).total_seconds()
        or originals["pointer_before"]["sha256"] == originals["pointer_after"]["sha256"]
        or int(originals["pointer_before"]["file"][16:-5])
        >= int(originals["pointer_after"]["file"][16:-5])
    ):
        raise ValueError("original bounded aware/steady observation interval required")
    original_assets = (
        _read_discovery_urdf(
            world / "assets/collision-asset-originals.json", "collision original assets"
        )
        if (world / "assets/collision-asset-originals.json").is_file()
        else None
    )
    graphs = []
    for native in [before, after]:
        model = build_system_model(
            RosGraphSnapshot.from_dict(
                {
                    "ros_version": "ros2",
                    "distro": "jazzy",
                    "endpoint": "fixture://owned-nav2-readonly-replica",
                    "captured_at": native["captured_at"],
                    **native["graph"],
                }
            ),
            native=native,
        )
        if original_assets is not None:
            bind_prepared_collision_originals(model.observations, original_assets)
        graphs.append(
            compile_operational_graph(
                model,
                urdf,
                requirements,
                source_ref="owned-nav2-readonly-replica:" + run,
                now=datetime.fromisoformat(native["captured_at"]),
                evidence_domain="SEALED_REPLAY",
            )
        )
    wanted = {
        h.issue_id
        for h in plan.hypotheses
        if h.id in {p.hypothesis_id for p in selected.predictions}
    }
    scopes = {(f.invariant, tuple(f.targets)) for f in original_graph.findings if f.id in wanted}
    ids = [f.id for f in graphs[0].findings if (f.invariant, tuple(f.targets)) in scopes]
    if not scopes or scopes != {
        (f.invariant, tuple(f.targets)) for f in graphs[0].findings if f.id in ids
    }:
        raise ValueError("original exact diagnostic target scope missing from SDK replica")
    rebound = plan_diagnostic_probes(
        graphs[0], issue_ids=ids, budget_sec=plan.budget_sec, max_probes=plan.max_probes
    )
    assessment = assess_diagnostic_followup(
        rebound, graphs[0], graphs[1], now=graphs[1].reviewed_at
    )
    return {
        "status": "REVIEWED_OWNED_NAV2_REPLICA_ORIGINALS_OBSERVATIONAL_ONLY",
        "capture_source_revision": capture_source,
        "probe_run_id": run,
        "original_run_id": original_run_id,
        "original_intent_experiment_hash": plan.experiment_hash,
        "replica_rebound_experiment": rebound.model_dump(mode="json"),
        "assessment": assessment.model_dump(mode="json"),
        "originals": references,
        "old_case_rebound": False,
        "Body_or_L1_admission": "NOT_GRANTED",
        "canonical_navigation_success": False,
        "causal_verification": "NOT_ESTABLISHED",
        "Memory_promotion": "NOT_GRANTED",
        "Worlds_started_by_this_file_review": 0,
        "execution_authority": "NONE",
    }
