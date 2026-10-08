"""Actual Body compiler/reopen integration with anonymous synthetic URDFs."""

import hashlib
from datetime import timedelta

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.context.sim_workspace import compile_sim_body_workspace
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros.test_sim_binding_proposal import NOW, example


@pytest.mark.parametrize(
    "namespace,base,sensor",
    [("/sample", "chassis", "laser_mount"), ("/isolated/robot", "renamed_body", "front_range")],
)
def test_exclusive_workspace_compiles_and_reopens_exact_body_bindings(
    tmp_path, namespace, base, sensor
):
    model, data, cleaner, policy = example(namespace, base, sensor)
    target = tmp_path / "owned_sim_workspace"
    result = compile_sim_body_workspace(
        target, model, data, attachment=cleaner, policy=policy, now=NOW
    )
    resolver = BodyResolver(workspace=target)
    effective = resolver.get_effective_body(recompile_if_stale=False)
    assert result["body_snapshot_hash"] == effective.compute_hash() == effective.effective_body_hash
    assert effective.frames["base"] == base and effective.frames["map"] == "world"
    assert (
        effective.provider_interfaces["ros_capability_bindings"]["sensing.lidar"]["name"]
        == namespace + "/scan2"
    )
    assert (
        effective.provider_interfaces["ros_capability_bindings"]["safety.collision_monitor"]["name"]
        == namespace + "/collision_monitor"
    )
    assert (
        hashlib.sha256(resolver.eurdf_profile_path.with_name("robot.urdf").read_bytes()).hexdigest()
        == result["source_urdf_sha256"]
    )
    assert (
        effective.provider_interfaces["sim_fixture_evidence"]["cleaner_kind"]
        == "SIMULATED_CLEANING"
    )
    assert (
        effective.provider_interfaces["sim_fixture_evidence"]["source_urdf_sha256"]
        == result["source_urdf_sha256"]
    )
    assert result["physical_acceptance_level"] == "NOT_RUN"
    assert not result["usable_for_real_execution"] and not result["direct_actions_dispatched"]
    manifest_hash = result.pop("manifest_hash")
    assert manifest_hash == digest(result)
    assert (target / "sim-body-workspace.yaml").exists()


def test_existing_product_workspace_is_never_overwritten(tmp_path):
    model, data, cleaner, policy = example()
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "operator_body.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        compile_sim_body_workspace(target, model, data, attachment=cleaner, policy=policy, now=NOW)
    assert marker.read_text() == "keep"
    assert list(target.iterdir()) == [marker]


def test_expired_or_rejected_fixture_proposal_creates_no_workspace(tmp_path):
    model, data, cleaner, policy = example()
    target = tmp_path / "rejected"
    with pytest.raises(ValueError):
        compile_sim_body_workspace(
            target, model, data, attachment=cleaner, policy=policy, now=NOW + timedelta(seconds=6)
        )
    assert not target.exists()
    policy["approved"] = False
    with pytest.raises(ValueError):
        compile_sim_body_workspace(target, model, data, attachment=cleaner, policy=policy, now=NOW)
    assert not target.exists()


def test_compiled_body_hash_binds_cleaner_declaration_and_policy_source(tmp_path):
    from rosclaw.connectors.ros.context.sim_attachment import validate_sim_attachment

    model, data, cleaner, policy = example()
    first = compile_sim_body_workspace(
        tmp_path / "first", model, data, attachment=cleaner, policy=policy, now=NOW
    )
    cleaner["cleaning_polygon"] = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
    policy["attachment_hash"] = validate_sim_attachment(cleaner)["attachment_hash"]
    second = compile_sim_body_workspace(
        tmp_path / "second", model, data, attachment=cleaner, policy=policy, now=NOW
    )
    assert first["body_snapshot_hash"] != second["body_snapshot_hash"]
