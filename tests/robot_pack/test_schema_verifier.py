from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from rosclaw.robot_pack.schema import RobotPackManifest
from rosclaw.robot_pack.verifier import default_trust_store_path, verify_robot_pack


def test_builtin_realsense_pack_has_trusted_complete_integrity(builtin_pack_root: Path) -> None:
    result = verify_robot_pack(builtin_pack_root)

    assert result.ok
    assert result.trusted
    assert result.signature_status == "valid"
    assert result.manifest is not None
    assert result.manifest.canonical_ref == ("rosclaw://robot_pack/ros-claw/realsense-d400@1.0.0")
    assert len(result.checked_files) == 5


def test_payload_tamper_fails_checksum_and_never_stays_trusted(copied_pack: Path) -> None:
    capability = copied_pack / "capabilities" / "camera.capture_rgbd.yaml"
    capability.write_text(
        capability.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
    )

    result = verify_robot_pack(copied_pack)

    assert not result.ok
    assert any("checksum mismatch" in error for error in result.errors)


def test_manifest_tamper_invalidates_detached_signature(copied_pack: Path) -> None:
    manifest = copied_pack / "robot-pack.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "Guarded RGB-D onboarding",
            "Modified RGB-D onboarding",
        ),
        encoding="utf-8",
    )

    result = verify_robot_pack(copied_pack)

    assert not result.ok
    assert result.signature_status == "invalid"
    assert "detached Ed25519 signature is invalid" in result.errors


def test_trusted_key_scope_must_include_exact_pack_version(
    builtin_pack_root: Path,
    tmp_path: Path,
) -> None:
    trust = json.loads(default_trust_store_path().read_text(encoding="utf-8"))
    trust["keys"]["rosclaw-realsense-pack-v1"]["scopes"] = [
        "ros-claw/realsense-d400@2.*"
    ]
    trust_path = tmp_path / "keys.json"
    trust_path.write_text(json.dumps(trust), encoding="utf-8")

    result = verify_robot_pack(builtin_pack_root, trust_store_path=trust_path)

    assert not result.ok
    assert result.signature_status == "scope_mismatch"
    assert any("not scoped" in error for error in result.errors)


def test_untracked_payload_is_rejected(copied_pack: Path) -> None:
    (copied_pack / "undeclared.bin").write_bytes(b"not in checksums")

    result = verify_robot_pack(copied_pack)

    assert not result.ok
    assert "untracked payload file: undeclared.bin" in result.errors


def test_symlink_payload_is_rejected(copied_pack: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.write_text("outside", encoding="utf-8")
    (copied_pack / "escape").symlink_to(target)

    result = verify_robot_pack(copied_pack)

    assert not result.ok
    assert any("symlinks are forbidden" in error for error in result.errors)


def test_symlink_pack_root_is_rejected(builtin_pack_root: Path, tmp_path: Path) -> None:
    linked_root = tmp_path / "linked-pack"
    linked_root.symlink_to(builtin_pack_root, target_is_directory=True)

    result = verify_robot_pack(linked_root)

    assert not result.ok
    assert any("root cannot be a symbolic link" in error for error in result.errors)


def test_schema_rejects_actuation_in_perception_only_pack(builtin_pack_root: Path) -> None:
    raw = yaml.safe_load((builtin_pack_root / "robot-pack.yaml").read_text(encoding="utf-8"))
    raw["capabilities"][0]["safety_class"] = "actuation"

    try:
        RobotPackManifest.model_validate(raw)
    except ValidationError as exc:
        assert "actuation-forbidden packs may expose only read_only capabilities" in str(exc)
    else:
        raise AssertionError("unsafe Pack schema was accepted")


def test_schema_requires_trusted_signature_contract(builtin_pack_root: Path) -> None:
    raw = yaml.safe_load((builtin_pack_root / "robot-pack.yaml").read_text(encoding="utf-8"))
    raw["integrity"]["signature"]["required"] = False

    try:
        RobotPackManifest.model_validate(raw)
    except ValidationError as exc:
        assert "Input should be True" in str(exc)
    else:
        raise AssertionError("unsigned Robot Pack contract was accepted")


def test_schema_requires_hardware_and_independent_observer_for_h3(
    builtin_pack_root: Path,
) -> None:
    raw = yaml.safe_load((builtin_pack_root / "robot-pack.yaml").read_text(encoding="utf-8"))
    read_only = next(stage for stage in raw["verification"] if stage["id"] == "read-only")
    read_only["requires_hardware"] = False
    read_only["requires_independent_observer"] = False

    try:
        RobotPackManifest.model_validate(raw)
    except ValidationError as exc:
        assert "H3/H4 verification requires hardware and independent observation" in str(exc)
    else:
        raise AssertionError("H3 stage without hardware evidence was accepted")
