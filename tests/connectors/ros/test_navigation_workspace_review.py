"""Read-only artifact integrity, not live navigation or publisher authority."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from rosclaw.body.compiler import EffectiveBodyCompiler
from rosclaw.body.schema import BodyYaml, CalibrationYaml, EurdfProfile
from rosclaw.connectors.ros.context.navigation_workspace import compile_navigation_body_workspace
from rosclaw.connectors.ros.context.navigation_workspace_review import (
    _FILES,
    inspect_navigation_body_workspace,
)
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_body_workspace import NOW, source


def workspace(tmp_path, **kwargs):
    model, raw, requirements, policy = source(**kwargs)
    root = tmp_path / "compiled"
    manifest = compile_navigation_body_workspace(
        root, model, raw, requirements, policy=policy, now=NOW
    )
    return root, manifest


def reseal(root, **updates):
    """Adversarial self-consistent file hashes still must not bypass semantics."""
    path = root / "navigation-body-workspace.json"
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_hash")
    manifest.update(updates)
    manifest["file_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _FILES
    }
    manifest["manifest_hash"] = content_hash("navbodyworkspace", manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"namespace": "/other/robot", "base": "platform", "sensor": "range", "width": 0.6}],
)
def test_original_artifacts_are_inspected_without_writes_or_live_promotion(tmp_path, kwargs):
    root, manifest = workspace(tmp_path, **kwargs)
    before = {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
        for p in [root, *root.rglob("*")]
    }
    result = inspect_navigation_body_workspace(
        root,
        expected_manifest_hash=manifest["manifest_hash"],
        expected_body_hash=manifest["body_snapshot_hash"],
        now=NOW + timedelta(days=1),
    )
    assert result["status"] == "ARTIFACT_INTEGRITY_VERIFIED_NOT_ADMITTED"
    assert result["verified_output_files"] == 11 and result["snapshot_age_ms"] == 86_400_000
    assert result["current_navigation_readiness"] == "NOT_EVALUATED"
    assert result["physical_source_provenance"] == "NOT_VERIFIED"
    assert not result["binding_verified"] and not result["usable_for_real_execution"]
    assert result["physical_acceptance_level"] == "NOT_RUN"
    after = {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
        for p in [root, *root.rglob("*")]
    }
    assert after == before


@pytest.mark.parametrize("name", sorted(_FILES))
def test_every_declared_output_tamper_is_refused(tmp_path, name):
    root, _ = workspace(tmp_path)
    with (root / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="file hash mismatch"):
        inspect_navigation_body_workspace(root, now=NOW)


@pytest.mark.parametrize("pin", ["expected_manifest_hash", "expected_body_hash"])
def test_separately_pinned_hash_mismatch_is_refused(tmp_path, pin):
    root, _ = workspace(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        inspect_navigation_body_workspace(root, now=NOW, **{pin: "wrong"})


@pytest.mark.parametrize(
    "fault",
    [
        "extra_file",
        "maintenance",
        "generated_source",
        "missing_file",
        "file_symlink",
        "directory_symlink",
        "root_symlink",
        "fifo",
        "empty",
        "oversized",
    ],
)
def test_non_closed_or_non_regular_workspace_is_refused_without_blocking(tmp_path, fault):
    root, _ = workspace(tmp_path)
    target = root / "execution_config.json"
    if fault in {"extra_file", "maintenance", "generated_source"}:
        name = {
            "extra_file": "extra.json",
            "maintenance": "body/maintenance.log",
            "generated_source": "body/refs/generated/new.yaml",
        }[fault]
        (root / name).write_text("unreviewed")
    elif fault == "missing_file":
        target.unlink()
    elif fault == "file_symlink":
        outside = tmp_path / "original.json"
        target.rename(outside)
        target.symlink_to(outside)
    elif fault == "directory_symlink":
        outside = tmp_path / "original-refs"
        (root / "body/refs").rename(outside)
        (root / "body/refs").symlink_to(outside, target_is_directory=True)
    elif fault == "root_symlink":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    elif fault == "fifo":
        target.unlink()
        os.mkfifo(target)
    elif fault == "empty":
        target.write_bytes(b"")
    elif fault == "oversized":
        target.write_bytes(b"x" * 5_000_001)
    with pytest.raises((ValueError, OSError)):
        inspect_navigation_body_workspace(root, now=NOW)


@pytest.mark.parametrize("fault", ["duplicate_json", "nan_json", "duplicate_yaml", "yaml_alias"])
def test_resealed_ambiguous_serializations_are_refused(tmp_path, fault):
    root, _ = workspace(tmp_path)
    if fault == "duplicate_json":
        p = root / "fixture-policy.json"
        p.write_text(
            p.read_text().replace('"approved": true', '"approved": false, "approved": true')
        )
    elif fault == "nan_json":
        p = root / "fixture-policy.json"
        p.write_text(p.read_text().replace('"max_speed_mps": 0.2', '"max_speed_mps": NaN'))
    elif fault == "duplicate_yaml":
        p = root / "body/refs/eurdf.profile.yaml"
        p.write_text(p.read_text() + "\nmodel: duplicate\n")
    elif fault == "yaml_alias":
        p = root / "body/refs/eurdf.profile.yaml"
        p.write_text(p.read_text() + "\nextra: &value []\nother: *value\n")
    reseal(root)
    with pytest.raises(ValueError):
        inspect_navigation_body_workspace(root, now=NOW)


@pytest.mark.parametrize("fault", ["cleaning_capability", "REAL_agent_policy", "actuator_provider"])
def test_resealed_and_recompiled_extra_authority_is_still_refused(tmp_path, fault):
    root, _ = workspace(tmp_path)
    profile_path, instance_path = root / "body/refs/eurdf.profile.yaml", root / "body/body.yaml"
    profile, instance = (
        yaml.safe_load(profile_path.read_text()),
        yaml.safe_load(instance_path.read_text()),
    )
    if fault == "cleaning_capability":
        profile["capability_hints"]["unobserved"] = ["cleaning.execute"]
    elif fault == "REAL_agent_policy":
        instance["agent_policy"]["direct_real_robot_execution_allowed"] = True
    else:
        profile["provider_interfaces"]["unobserved_actuator"] = {"name": "fake_arm"}
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))
    checksum = "sha256:" + hashlib.sha256(profile_path.read_bytes()).hexdigest()
    instance["model_ref"]["checksum"] = checksum
    instance_path.write_text(yaml.safe_dump(instance))
    lock_path = root / "body/refs/eurdf.lock"
    lock = yaml.safe_load(lock_path.read_text())
    lock["checksum"] = checksum
    lock_path.write_text(yaml.safe_dump(lock))
    cached = EffectiveBodyCompiler().compile(
        EurdfProfile.from_dict(profile), BodyYaml.from_dict(instance), CalibrationYaml()
    )
    (root / "body/refs/effective_body.json").write_text(json.dumps(cached.to_dict()))
    (root / "execution_config.json").write_text(
        json.dumps(
            {
                "executor_kind": "navigation_only",
                "body_id": cached.body_instance_id,
                "body_snapshot_hash": cached.compute_hash(),
            }
        )
    )
    reseal(root, body_snapshot_hash=cached.compute_hash())
    with pytest.raises(ValueError, match="compiler contract mismatch"):
        inspect_navigation_body_workspace(root, now=NOW)


def test_file_replaced_after_capture_is_refused(tmp_path, monkeypatch):
    import rosclaw.connectors.ros.context.navigation_workspace_review as module

    root, _ = workspace(tmp_path)
    original = module._inspect

    def replace(*args):
        result = original(*args)
        path = root / "execution_config.json"
        raw = path.read_bytes()
        path.unlink()
        path.write_bytes(raw)
        return result

    monkeypatch.setattr(module, "_inspect", replace)
    with pytest.raises(ValueError, match="changed during review"):
        inspect_navigation_body_workspace(root, now=NOW)


@pytest.mark.parametrize("fault", ["sealed_model", "review_time", "requirements", "physical_claim"])
def test_resealed_source_or_admission_claim_does_not_bypass_original_review(tmp_path, fault):
    root, _ = workspace(tmp_path)
    if fault == "physical_claim":
        reseal(root, physical_acceptance_level="TASK_VERIFIED", binding_verified=True)
    else:
        name = {
            "sealed_model": "source-snapshot.json",
            "review_time": "navigation-review.json",
            "requirements": "navigation-requirements.json",
        }[fault]
        path = root / name
        value = json.loads(path.read_text())
        if fault == "sealed_model":
            value["environment"]["use_sim_time"] = False
        elif fault == "review_time":
            value["reviewed_at"] = (NOW + timedelta(days=1)).isoformat()
        else:
            value["costmap_topics"][0] = "/unobserved_grid"
        path.write_text(json.dumps(value))
        reseal(root)
    with pytest.raises(ValueError):
        inspect_navigation_body_workspace(root, now=NOW)


def test_manifest_cannot_add_an_external_member(tmp_path):
    root, _ = workspace(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("not a compiler source")
    path = root / "navigation-body-workspace.json"
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_hash")
    manifest["file_sha256"]["../outside.json"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest["manifest_hash"] = content_hash("navbodyworkspace", manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="artifact contract"):
        inspect_navigation_body_workspace(root, now=NOW)
    assert outside.read_text() == "not a compiler source"


@pytest.mark.parametrize(
    "fault,runtime_home_exists", [(None, False), (None, True), ("tamper", False), ("pin", False)]
)
def test_actual_pinned_inspection_cli_is_read_only_and_explicit_about_history(
    tmp_path, fault, runtime_home_exists
):
    root, _ = workspace(tmp_path)
    if fault == "tamper":
        (root / "execution_config.json").write_text("{}")
    if runtime_home_exists:
        (tmp_path / "private-home").mkdir()

    def capture():
        return {
            str(p.relative_to(root)): (
                p.read_bytes() if p.is_file() else None,
                p.stat().st_mtime_ns,
                p.stat().st_ctime_ns,
            )
            for p in [root, *root.rglob("*")]
        }

    before = capture()
    repository = Path(__file__).resolve().parents[3]
    argv = [
        sys.executable,
        "-m",
        "rosclaw.entrypoint",
        "ros",
        "inspect-navigation-workspace",
        "--workspace",
        str(root),
        "--json",
    ]
    if fault == "pin":
        argv += ["--expected-manifest-hash", "wrong"]
    completed = subprocess.run(
        argv,
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
    if fault is None:
        assert result["status"] == "ARTIFACT_INTEGRITY_VERIFIED_NOT_ADMITTED"
        assert result["current_navigation_readiness"] == "NOT_EVALUATED"
    assert (tmp_path / "private-home").exists() is runtime_home_exists
    assert capture() == before


def test_static_byte_cache_does_not_cache_clocks_or_hide_file_changes(tmp_path):
    from rosclaw.connectors.ros.context.navigation_workspace_review import _static_integrity_json

    _static_integrity_json.cache_clear()
    root, manifest = workspace(tmp_path)
    first = inspect_navigation_body_workspace(root, now=NOW)
    first["manifest_hash"] = "caller mutation"
    later = inspect_navigation_body_workspace(root, now=NOW + timedelta(days=2))
    assert later["manifest_hash"] == manifest["manifest_hash"]
    assert later["snapshot_age_ms"] == 172800000
    assert later["inspected_at"] == (NOW + timedelta(days=2)).isoformat()
    assert later["current_navigation_readiness"] == "NOT_EVALUATED"
    assert _static_integrity_json.cache_info().hits >= 1
    with pytest.raises(ValueError, match="Body hash mismatch"):
        inspect_navigation_body_workspace(root, expected_body_hash="0" * 64, now=NOW)
    # A valid previous key never substitutes for a newly read changed file.
    file = root / "body/calibration.yaml"
    file.write_bytes(file.read_bytes() + b"\nunknown: changed\n")
    with pytest.raises(ValueError):
        inspect_navigation_body_workspace(root, now=NOW)
    _static_integrity_json.cache_clear()


def test_static_byte_cache_cannot_hide_symlink_with_identical_bytes(tmp_path):
    root, _ = workspace(tmp_path)
    inspect_navigation_body_workspace(root, now=NOW)
    file = root / "body/refs/eurdf.profile.yaml"
    outside = tmp_path / "same-original.yaml"
    outside.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        inspect_navigation_body_workspace(root, now=NOW)
