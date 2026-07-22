from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.robot_pack.discovery import (
    DiscoveredDevice,
    DiscoveryReport,
    StreamProfile,
)
from rosclaw.robot_pack.instance import configure_robot_instance
from rosclaw.robot_pack.verification import verify_installed_robot_pack


def _install_fake_adapter(home: Path) -> None:
    InstalledRegistry(home=home).add(
        InstalledRecord(
            server_name="librealsense-mcp",
            manifest_id="ros-claw/librealsense-mcp",
            name="librealsense-mcp",
            version="1.0.0",
            installed_at="2026-07-21T00:00:00Z",
            artifact_type="test",
            server_dir=str(home / "mcp" / "servers" / "librealsense-mcp"),
            extra={"repo_commit": "fdea4c3cfd03e7acf1adb664a9ffca5733d44b59"},
        )
    )


def _discovery(serial: str) -> DiscoveryReport:
    return DiscoveryReport(
        device_type="camera",
        attempted_backends=("pyrealsense2",),
        devices=(
            DiscoveredDevice(
                device_type="camera",
                vendor_id="8086",
                product_id="0b5b",
                model="Intel RealSense D405",
                serial=serial,
                firmware="5.16.0.1",
                usb_speed="3.2",
                stable_uri=f"realsense://{serial}",
                stable_path=None,
                backend="pyrealsense2",
                stream_profiles=(
                    StreamProfile("depth", "z16", 30, 640, 480),
                    StreamProfile("color", "rgb8", 30, 640, 480),
                ),
                body_profile="realsense_d405",
                pack_ref="rosclaw://robot_pack/ros-claw/realsense-d400@1.0.0",
                identity_complete=True,
            ),
        ),
    )


def _receipt(home: Path, instance, *, tamper_hash: bool = False) -> Path:
    artifacts = home / "artifacts" / "robot-packs" / "acceptance"
    artifacts.mkdir(parents=True)
    color = artifacts / "color.png"
    depth = artifacts / "depth.png"
    color.write_bytes(b"real-color-frame")
    depth.write_bytes(b"real-depth-frame")
    hashes = {
        "color": f"sha256:{hashlib.sha256(color.read_bytes()).hexdigest()}",
        "depth": f"sha256:{hashlib.sha256(depth.read_bytes()).hexdigest()}",
    }
    if tamper_hash:
        hashes["depth"] = "sha256:" + "0" * 64
    captured_at = datetime.now(UTC)
    started_at = captured_at - timedelta(seconds=1)
    transition_at = captured_at + timedelta(milliseconds=100)
    finished_at = captured_at + timedelta(seconds=1)

    def timestamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    path = artifacts / "receipt.json"
    payload = {
        "schema_version": "rosclaw.receipt.v1",
        "action_id": "action-rs-acceptance",
        "trace_id": "trace_rs_acceptance",
        "body_id": instance.instance_id,
        "body_snapshot_hash": instance.body_snapshot_hash,
        "capability_id": "camera.capture_rgbd",
        "execution_mode": "REAL",
        "final_state": "COMPLETED",
        "evidence_level": "PHYSICALLY_OBSERVED",
        "verified": True,
        "trust_level": "VERIFIED",
        "usable_for_real_execution": True,
        "policy_decision": {"allowed": True},
        "authorization_decision": {"authorized": True},
        "resource_lease": {
            "lease_id": "lease_rs_acceptance",
            "resource_id": instance.instance_id,
            "action_id": "action-rs-acceptance",
            "acquired_at": timestamp(started_at),
            "exclusive": True,
        },
        "dispatch_result": {"accepted": True},
        "driver_ack": {"acknowledged": True},
        "verification_result": {"success": True},
        "artifacts": [color.as_uri(), depth.as_uri(), path.as_uri()],
        "transitions": [
            {"state": "DISPATCHED", "at": timestamp(transition_at)},
            {"state": "DRIVER_ACKNOWLEDGED", "at": timestamp(transition_at)},
            {"state": "PHYSICALLY_OBSERVED", "at": timestamp(transition_at)},
            {"state": "COMPLETED", "at": timestamp(transition_at)},
        ],
        "observations": [
            {
                "kind": "rgbd_capture",
                "device_identity": {
                    "model": instance.device.model,
                    "serial": instance.device.serial,
                    "stable_uri": instance.device.stable_uri,
                },
                "captured_at": timestamp(captured_at),
                "metrics": {"width": 640, "height": 480, "aligned": True},
                "artifacts": {"color": color.as_uri(), "depth": depth.as_uri()},
                "artifact_hashes": hashes,
            }
        ],
        "started_at": timestamp(started_at),
        "finished_at": timestamp(finished_at),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tools(_server_name: str, _home: Path) -> list[dict[str, str]]:
    return [
        {"name": "list_devices"},
        {"name": "start_pipeline"},
        {"name": "capture_aligned_rgbd"},
    ]


def test_contract_verification_persists_h1_evidence(installed_pack) -> None:
    home, store, _manifest = installed_pack

    report = verify_installed_robot_pack("realsense", home=home)

    assert report.passed
    assert report.support_tier == "H1_CONTRACT_VERIFIED"
    assert report.observed_candidate_tier is None
    assert report.report_path is not None and Path(report.report_path).is_file()
    assert store.list_installed()[0].latest_verification_id == report.evidence_id


def test_full_local_read_only_run_is_candidate_not_canonical_h3(installed_pack) -> None:
    home, store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="acceptance-d405",
        serial="REAL123",
        model="D405",
        allow_offline=True,
    )

    report = verify_installed_robot_pack(
        "acceptance-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("REAL123"),
        adapter_probe=_tools,
        receipt_path=_receipt(home, instance),
    )

    assert report.passed
    assert report.support_tier == "H1_CONTRACT_VERIFIED"
    assert report.observed_candidate_tier == "H3_HARDWARE_READ_VERIFIED"
    assert not report.canonical_promotion
    assert any("independent physical observer" in item for item in report.promotion_blockers)
    assert store.list_installed()[0].support_tier == "H1_CONTRACT_VERIFIED"


def test_read_only_verification_rejects_artifact_hash_mismatch(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="hash-d405",
        serial="HASH123",
        model="D405",
        allow_offline=True,
    )

    report = verify_installed_robot_pack(
        "hash-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("HASH123"),
        adapter_probe=_tools,
        receipt_path=_receipt(home, instance, tamper_hash=True),
    )

    assert not report.passed
    receipt_check = next(
        check for check in report.checks if check.id == "read-only.rgbd-artifact-receipt"
    )
    assert receipt_check.status == "fail"
    assert "hash does not match" in receipt_check.message


def test_read_only_verification_rejects_receipt_without_driver_ack(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="ack-d405",
        serial="ACK123",
        model="D405",
        allow_offline=True,
    )
    receipt = _receipt(home, instance)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["driver_ack"]["acknowledged"] = False
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_installed_robot_pack(
        "ack-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("ACK123"),
        adapter_probe=_tools,
        receipt_path=receipt,
    )

    check = next(item for item in report.checks if item.id == "read-only.rgbd-artifact-receipt")
    assert check.status == "fail"
    assert "control-plane decisions" in check.message


def test_read_only_verification_rejects_artifacts_outside_managed_root(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="outside-d405",
        serial="OUT123",
        model="D405",
        allow_offline=True,
    )
    receipt = _receipt(home, instance)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    outside = home / "outside-color.png"
    outside.write_bytes(b"outside")
    payload["observations"][0]["artifacts"]["color"] = outside.as_uri()
    payload["artifacts"][0] = outside.as_uri()
    payload["observations"][0]["artifact_hashes"]["color"] = (
        f"sha256:{hashlib.sha256(outside.read_bytes()).hexdigest()}"
    )
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_installed_robot_pack(
        "outside-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("OUT123"),
        adapter_probe=_tools,
        receipt_path=receipt,
    )

    check = next(item for item in report.checks if item.id == "read-only.rgbd-artifact-receipt")
    assert check.status == "fail"
    assert "managed Robot Pack artifact root" in check.message


def test_read_only_verification_rejects_invalid_timestamp_and_transition_order(
    installed_pack,
) -> None:
    home, _store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="sequence-d405",
        serial="SEQ123",
        model="D405",
        allow_offline=True,
    )
    receipt = _receipt(home, instance)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["observations"][0]["captured_at"] = "not-a-timestamp"
    payload["transitions"] = list(reversed(payload["transitions"]))
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_installed_robot_pack(
        "sequence-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("SEQ123"),
        adapter_probe=_tools,
        receipt_path=receipt,
    )

    check = next(item for item in report.checks if item.id == "read-only.rgbd-artifact-receipt")
    assert check.status == "fail"
    assert "timestamp" in check.message or "transitions" in check.message


def test_read_only_verification_rejects_noncanonical_lease_and_timing(
    installed_pack,
) -> None:
    home, _store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="timing-d405",
        serial="TIME123",
        model="D405",
        allow_offline=True,
    )
    receipt = _receipt(home, instance)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.pop("resource_lease")
    payload["transitions"][0].pop("at")
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_installed_robot_pack(
        "timing-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("TIME123"),
        adapter_probe=_tools,
        receipt_path=receipt,
    )

    check = next(item for item in report.checks if item.id == "read-only.rgbd-artifact-receipt")
    assert check.status == "fail"
    assert "lease" in check.message or "timing" in check.message


def test_read_only_verification_rejects_symlinked_managed_artifact_root(
    installed_pack,
) -> None:
    home, _store, _manifest = installed_pack
    _install_fake_adapter(home)
    instance, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="symlink-d405",
        serial="LINK123",
        model="D405",
        allow_offline=True,
    )
    external = home / "external-artifacts"
    external.mkdir()
    artifacts_parent = home / "artifacts"
    artifacts_parent.mkdir()
    (artifacts_parent / "robot-packs").symlink_to(external, target_is_directory=True)
    receipt = _receipt(home, instance)

    report = verify_installed_robot_pack(
        "symlink-d405",
        stage="read-only",
        home=home,
        discovery_report=_discovery("LINK123"),
        adapter_probe=_tools,
        receipt_path=receipt,
    )

    check = next(item for item in report.checks if item.id == "read-only.rgbd-artifact-receipt")
    assert check.status == "fail"
    assert "symbolic link" in check.message
