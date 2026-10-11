"""Synthetic compiler contracts; no World, runtime admission or held-out asset."""

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.context.navigation_workspace import compile_navigation_body_workspace
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_observation_review import NOW, navigation_fixture


def source(namespace="/sample", base="chassis", sensor="laser_mount", width=0.4):
    model, raw, requirements = navigation_fixture(namespace, base, sensor)
    raw = raw.replace(b'box size="0.4', f'box size="{width}'.encode())
    for description in model.observations["urdf_descriptions"].values():
        description.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    model.seal()
    review = review_navigation_candidate(model, raw, requirements, now=NOW)
    policy = {
        "schema_version": "rosclaw.navigation_sim_fixture_policy.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "policy_id": "synthetic-policy",
        "body_id": "synthetic-observed-base",
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": hashlib.sha256(raw).hexdigest(),
        "navigation_requirements_hash": review["requirements_hash"],
        "navigation_limits": {
            "max_goal_distance_m": 0.5,
            "max_duration_sec": 30.0,
            "goal_position_tolerance_m": 0.05,
            "goal_yaw_tolerance_rad": 0.1,
            "max_speed_mps": 0.2,
            "max_yaw_rate_rps": 0.5,
            "stop_speed_mps": 0.005,
            "stop_yaw_rate_rps": 0.01,
        },
        "witness_declaration": {
            "observation_topic": namespace + "/independent_truth",
            "lease_service": namespace + "/sim_deadman",
            "run_id": "synthetic-new-run",
        },
    }
    return model, raw, requirements, policy


@pytest.mark.parametrize(
    "namespace,base,sensor,width",
    [
        ("/sample", "chassis", "laser_mount", 0.4),
        ("/renamed/robot", "platform", "range_frame", 0.6),
    ],
)
def test_compiler_reopens_source_bound_navigation_body_without_profile(
    tmp_path, namespace, base, sensor, width
):
    model, raw, requirements, policy = source(namespace, base, sensor, width)
    original = deepcopy(policy)
    destination = tmp_path / "new-fixture"
    manifest = compile_navigation_body_workspace(
        destination, model, raw, requirements, policy=policy, now=NOW
    )
    body = BodyResolver(workspace=destination).get_effective_body(recompile_if_stale=False)
    binding = SimulationNavigationBinding.from_body(body)
    assert binding.action_name == namespace + "/go"
    assert binding.lease_service == namespace + "/sim_deadman"
    assert body.frames["base"] == base and body.frames["lidar"] == sensor
    assert binding.body_snapshot_hash == manifest["body_snapshot_hash"]
    assert "navigation.navigate_to_pose" in body.capabilities["degraded"]
    assert not body.actuators
    assert not any(
        k.startswith(("cleaning.", "coverage.", "localization.set_"))
        for k in body.provider_interfaces["ros_capability_bindings"]
    )
    assert (
        body.provider_interfaces["sim_fixture_evidence"]["collision_envelope"]["physical_radius_m"]
        > width / 2
    )
    assert (
        manifest["binding_verified"] is False and manifest["physical_acceptance_level"] == "NOT_RUN"
    )
    assert (
        manifest["usable_for_real_execution"] is False
        and manifest["direct_actions_dispatched"] is False
    )
    saved = manifest.pop("manifest_hash")
    assert saved == content_hash("navbodyworkspace", manifest)
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected
    config = json.loads((destination / "execution_config.json").read_text())
    assert config == {
        "executor_kind": "navigation_only",
        "body_id": binding.body_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
    }
    assert policy == original
    policy["navigation_limits"]["max_speed_mps"] = 1000
    assert (
        SimulationNavigationBinding.from_body(
            BodyResolver(workspace=destination).get_effective_body(recompile_if_stale=False)
        ).max_speed_mps
        == 0.2
    )


@pytest.mark.parametrize(
    "fault",
    [
        "not_approved",
        "wrong_source",
        "wrong_snapshot",
        "wrong_urdf",
        "wrong_requirements",
        "missing_limit",
        "nan_limit",
        "boolean_limit",
        "stop_over_speed",
        "relative_endpoint",
        "same_endpoint",
        "missing_run",
        "extra_coverage",
        "stale_snapshot",
        "tampered_snapshot",
        "inactive",
        "unknown_geometry",
    ],
)
def test_incomplete_or_mismatched_source_never_creates_workspace(tmp_path, fault):
    model, raw, requirements, policy = source()
    if fault == "not_approved":
        policy["approved"] = False
    elif fault == "wrong_source":
        policy["source"] = "agent_proposal"
    elif fault == "wrong_snapshot":
        policy["source_snapshot_hash"] = "wrong"
    elif fault == "wrong_urdf":
        policy["source_urdf_sha256"] = "wrong"
    elif fault == "wrong_requirements":
        policy["navigation_requirements_hash"] = "wrong"
    elif fault == "missing_limit":
        policy["navigation_limits"].pop("max_duration_sec")
    elif fault == "nan_limit":
        policy["navigation_limits"]["max_speed_mps"] = float("nan")
    elif fault == "boolean_limit":
        policy["navigation_limits"]["max_speed_mps"] = True
    elif fault == "stop_over_speed":
        policy["navigation_limits"]["stop_speed_mps"] = 1
    elif fault == "relative_endpoint":
        policy["witness_declaration"]["lease_service"] = "relative"
    elif fault == "same_endpoint":
        policy["witness_declaration"]["lease_service"] = policy["witness_declaration"][
            "observation_topic"
        ]
    elif fault == "missing_run":
        policy["witness_declaration"]["run_id"] = ""
    elif fault == "extra_coverage":
        policy["grid"] = {}
    elif fault == "stale_snapshot":
        model.captured_at -= timedelta(seconds=20)
        model.seal()
    elif fault == "tampered_snapshot":
        model.environment["use_sim_time"] = False
    elif fault == "inactive":
        model.lifecycle[0].state = "INACTIVE"
        model.seal()
    elif fault == "unknown_geometry":
        raw = raw.replace(b'<box size="0.4 0.2 0.1"/>', b'<mesh filename="unknown.stl"/>')
        for description in model.observations["urdf_descriptions"].values():
            description.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        model.seal()
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        compile_navigation_body_workspace(
            destination, model, raw, requirements, policy=policy, now=NOW
        )
    assert not destination.exists()


def test_existing_body_workspace_is_never_overwritten(tmp_path):
    model, raw, requirements, policy = source()
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "operator-source.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        compile_navigation_body_workspace(
            destination, model, raw, requirements, policy=policy, now=NOW
        )
    assert list(destination.iterdir()) == [marker] and marker.read_text() == "preserve"


def test_relative_destination_preserves_reopen_and_relative_file_digests(tmp_path, monkeypatch):
    model, raw, requirements, policy = source()
    monkeypatch.chdir(tmp_path)
    manifest = compile_navigation_body_workspace(
        "relative-fixture", model, raw, requirements, policy=policy, now=NOW
    )
    for name, digest in manifest["file_sha256"].items():
        assert not Path(name).is_absolute()
        assert (
            hashlib.sha256((tmp_path / "relative-fixture" / name).read_bytes()).hexdigest()
            == digest
        )


def test_dangling_destination_symlink_is_refused_before_target_write(tmp_path):
    model, raw, requirements, policy = source()
    destination, target = tmp_path / "link", tmp_path / "target"
    destination.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        compile_navigation_body_workspace(
            destination, model, raw, requirements, policy=policy, now=NOW
        )
    assert destination.is_symlink() and not target.exists()


@pytest.mark.parametrize(
    "fault,runtime_home_exists",
    [
        (None, False),
        (None, True),
        ("policy", False),
        ("stale", False),
        ("tamper", False),
        ("policy_symlink", False),
    ],
)
def test_actual_pinned_cli_compiles_or_refuses_without_runtime(
    tmp_path, fault, runtime_home_exists
):
    model, raw, requirements, policy = source("/cli/renamed", "platform", "range_frame", 0.6)
    now = datetime.now(UTC)
    model.captured_at = now
    for row in [*model.signals, *model.transforms, *model.lifecycle]:
        row.captured_at = now
    for key in ("message_frames", "urdf_descriptions"):
        for row in model.observations[key].values():
            row["captured_at"] = now.isoformat()
    model.observations["parameter_captured_at"] = {
        key: now.isoformat() for key in model.observations["parameter_captured_at"]
    }
    for row in model.observations["localization_quality"]:
        row["captured_at"] = now.isoformat()
    if fault == "stale":
        model.captured_at -= timedelta(seconds=20)
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    if fault == "policy":
        policy["approved"] = False
    snapshot = model.to_dict()
    if fault == "tamper":
        snapshot["environment"]["use_sim_time"] = False
    for filename, value in [
        ("snapshot.json", snapshot),
        ("requirements.json", requirements),
        ("policy.json", policy),
    ]:
        (tmp_path / filename).write_text(json.dumps(value))
    (tmp_path / "robot.urdf").write_bytes(raw)
    policy_path = tmp_path / "policy.json"
    if fault == "policy_symlink":
        policy_path = tmp_path / "policy-link.json"
        policy_path.symlink_to("policy.json")
    repository = Path(__file__).resolve().parents[3]
    destination = tmp_path / "compiled"
    if runtime_home_exists:
        (tmp_path / "private-home").mkdir()
    completed = subprocess.run(
        [
            # Pin the child to pytest's interpreter. System-install CI has no
            # repository .venv; local pytest still uses the pinned launcher.
            sys.executable,
            "-m",
            "rosclaw.entrypoint",
            "ros",
            "compile-navigation-body",
            "--snapshot",
            str(tmp_path / "snapshot.json"),
            "--urdf",
            str(tmp_path / "robot.urdf"),
            "--requirements",
            str(tmp_path / "requirements.json"),
            "--policy",
            str(policy_path),
            "--workspace",
            str(destination),
            "--json",
        ],
        cwd=repository,
        env=dict(
            os.environ,
            PYTHONPATH=str(repository / "src"),
            ROSCLAW_HOME=str(tmp_path / "private-home"),
        ),
        capture_output=True,
        text=True,
        timeout=15,
    )
    result = json.loads(completed.stdout)
    assert result["ok"] is (fault is None)
    assert completed.returncode == (0 if fault is None else 1)
    assert destination.exists() is (fault is None)
    if fault is None:
        assert result["status"] == "CONTRACT_COMPILED_NOT_ADMITTED"
        assert {
            str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()
        } == set(result["file_sha256"]) | {"navigation-body-workspace.json"}
        binding = SimulationNavigationBinding.from_body(
            BodyResolver(workspace=destination).get_effective_body(recompile_if_stale=False)
        )
        assert binding.action_name == "/cli/renamed/go" and binding.run_id == "synthetic-new-run"
    assert not (tmp_path / "private-home/memory.sqlite").exists()
