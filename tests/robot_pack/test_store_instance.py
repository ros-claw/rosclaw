from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from rosclaw.body.resolver import BodyResolver
from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.robot_pack.discovery import DiscoveredDevice, DiscoveryReport
from rosclaw.robot_pack.instance import (
    RobotInstanceError,
    _adapter_source_matches_lock,
    configure_robot_instance,
    load_robot_instance,
    resolve_adapter_binding,
)
from rosclaw.robot_pack.store import RobotPackStore, RobotPackStoreError


def test_signed_pack_install_is_idempotent_and_locked(tmp_path: Path) -> None:
    home = tmp_path / "home"
    store = RobotPackStore(home)

    first = store.install("realsense")
    second = store.install("realsense")

    assert first == second
    assert Path(first.path).is_dir()
    assert first.signature_status == "valid"
    assert first.trusted
    assert first.support_tier == "H1_CONTRACT_VERIFIED"
    assert (home / "robots" / "robot-packs.lock.json").is_file()


def test_pack_install_rolls_back_files_when_lock_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    store = RobotPackStore(home)

    def fail_lock_write(_records) -> None:
        raise OSError("simulated lock write failure")

    monkeypatch.setattr(store, "_save_records_unlocked", fail_lock_write)

    with pytest.raises(OSError, match="simulated lock write failure"):
        store.install("realsense")

    destination = home / "robots" / "packs" / "ros-claw" / "realsense-d400" / "1.0.0"
    assert not destination.exists()
    assert not list(destination.parent.glob(".1.0.0.*"))


def test_offline_config_creates_bound_body_but_not_hardware_evidence(installed_pack) -> None:
    home, _store, _manifest = installed_pack

    config, path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="lab-d405",
        serial="OFFLINE123",
        model="D405",
        allow_offline=True,
    )

    assert path.is_file()
    assert config.device.offline_configured
    assert config.device.firmware_at_configure == "unknown"
    assert config.adapter.status == "not_installed"
    assert config.safety["actuation"] == "forbidden"
    body = BodyResolver(workspace=home, body_id="lab-d405").get_current_body_yaml()
    assert body.body_instance["serial_number"] == "OFFLINE123"
    assert body.capabilities["enabled"] == ["camera.capture_rgbd"]
    assert body.agent_policy["direct_real_robot_execution_allowed"] is False


def test_offline_config_requires_explicit_model_and_serial(installed_pack) -> None:
    home, _store, _manifest = installed_pack

    with pytest.raises(RobotInstanceError, match="both --model and --serial"):
        configure_robot_instance(
            "realsense",
            home=home,
            model="D405",
            allow_offline=True,
        )


def test_idempotent_configure_rejects_tampered_existing_contract(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    _config, path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="tampered-d405",
        serial="TAMPER123",
        model="D405",
        allow_offline=True,
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["safety"]["actuation"] = "allowed"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RobotInstanceError, match="no longer matches"):
        configure_robot_instance(
            "realsense",
            home=home,
            instance_id="tampered-d405",
            serial="TAMPER123",
            model="D405",
            allow_offline=True,
        )


def test_idempotent_configure_refreshes_adapter_binding(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    first, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="refresh-d405",
        serial="REFRESH123",
        model="D405",
        allow_offline=True,
    )
    assert first.adapter.status == "not_installed"
    InstalledRegistry(home=home).add(
        InstalledRecord(
            server_name="librealsense-mcp",
            manifest_id="librealsense",
            name="librealsense-mcp",
            version="1",
            installed_at="2026-07-21T00:00:00Z",
            artifact_type="test",
            server_dir=str(home / "mcp"),
            extra={"repo_commit": "fdea4c3cfd03e7acf1adb664a9ffca5733d44b59"},
        )
    )

    refreshed, _path = configure_robot_instance(
        "realsense",
        home=home,
        instance_id="refresh-d405",
        serial="REFRESH123",
        model="D405",
        allow_offline=True,
    )

    assert refreshed.adapter.status == "installed"
    assert refreshed.adapter.server_name == "librealsense-mcp"


def test_installed_payload_tamper_fails_closed(installed_pack) -> None:
    _home, store, _manifest = installed_pack
    record = store.list_installed()[0]
    policy = Path(record.path) / "policies" / "perception-only.yaml"
    policy.write_text(policy.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")

    with pytest.raises(RobotPackStoreError, match="failed integrity verification"):
        store.resolve_installed("realsense-d400")


def test_store_rejects_lock_record_path_outside_managed_pack_root(installed_pack) -> None:
    home, store, _manifest = installed_pack
    payload = json.loads(store.index_path.read_text(encoding="utf-8"))
    record = next(iter(payload["packs"].values()))
    record["path"] = str(home / "outside" / "realsense-d400")
    store.index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RobotPackStoreError, match="managed Pack destination"):
        store.list_installed()


def test_store_rejects_string_boolean_in_lock_record(installed_pack) -> None:
    _home, store, _manifest = installed_pack
    payload = json.loads(store.index_path.read_text(encoding="utf-8"))
    record = next(iter(payload["packs"].values()))
    record["trusted"] = "false"
    store.index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RobotPackStoreError, match="trusted must be a boolean"):
        store.list_installed()


def test_store_rejects_lock_support_tier_above_signed_pack_candidate(installed_pack) -> None:
    _home, store, _manifest = installed_pack
    payload = json.loads(store.index_path.read_text(encoding="utf-8"))
    record = next(iter(payload["packs"].values()))
    record["support_tier"] = "H6_REFERENCE_SUPPORTED"
    store.index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RobotPackStoreError, match="support tier"):
        store.resolve_installed("realsense")


def test_instance_lookup_does_not_follow_traversal_identifier(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    outside = home / "outside.yaml"
    outside.write_text("schema_version: rosclaw.robot_instance.v1\n", encoding="utf-8")

    with pytest.raises(RobotInstanceError, match="not configured"):
        load_robot_instance("../outside", home=home)


def test_live_config_requires_exact_vendor_product_serial_and_uri(installed_pack) -> None:
    home, _store, manifest = installed_pack
    incomplete = DiscoveryReport(
        device_type="camera",
        attempted_backends=("pyrealsense2",),
        devices=(
            DiscoveredDevice(
                device_type="camera",
                vendor_id="ffff",
                product_id="0b5b",
                model="Intel RealSense D405",
                serial="",
                firmware="5.16.0.1",
                usb_speed="3.2",
                stable_uri="",
                stable_path=None,
                backend="pyrealsense2",
                pack_ref=manifest.canonical_ref,
            ),
        ),
    )

    with pytest.raises(RobotInstanceError, match="exact vendor, product, model, serial"):
        configure_robot_instance(
            "realsense",
            home=home,
            discovery_report=incomplete,
        )


def test_adapter_binding_rejects_similarly_named_server(installed_pack) -> None:
    home, _store, manifest = installed_pack
    InstalledRegistry(home=home).add(
        InstalledRecord(
            server_name="evil-librealsense-mcp",
            manifest_id="evil",
            name="evil",
            version="1",
            installed_at="2026-07-21T00:00:00Z",
            artifact_type="test",
            server_dir=str(home / "mcp" / "evil"),
            extra={"repo_commit": "fdea4c3cfd03e7acf1adb664a9ffca5733d44b59"},
        )
    )

    binding = resolve_adapter_binding(manifest, home)

    assert binding.status == "not_installed"


def test_adapter_binding_rejects_git_record_without_locked_source(installed_pack) -> None:
    home, _store, manifest = installed_pack
    revision = "fdea4c3cfd03e7acf1adb664a9ffca5733d44b59"
    InstalledRegistry(home=home).add(
        InstalledRecord(
            server_name="librealsense-mcp",
            manifest_id="librealsense",
            name="librealsense-mcp",
            version="1",
            installed_at="2026-07-21T00:00:00Z",
            artifact_type="git",
            server_dir=str(home / "mcp" / "installed" / "librealsense-mcp" / "source"),
            extra={
                "source_url": "https://github.com/ros-claw/librealsense-mcp",
                "repo_commit": revision,
                "requested_revision": revision,
            },
        )
    )

    binding = resolve_adapter_binding(manifest, home)

    assert binding.status == "version_mismatch"


def test_git_adapter_lock_detects_tracked_source_drift(tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = home / "mcp" / "installed" / "librealsense-mcp" / "source"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    tracked = source / "mcp_server.py"
    tracked.write_text("print('locked')\n", encoding="utf-8")
    (source / ".gitignore").write_text("ignored_payload.py\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(source), "add", "mcp_server.py", ".gitignore"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=ROSClaw Test",
            "-c",
            "user.email=rosclaw-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "locked",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    record = InstalledRecord(
        server_name="librealsense-mcp",
        manifest_id="librealsense",
        name="librealsense-mcp",
        version="1",
        installed_at="2026-07-21T00:00:00Z",
        artifact_type="git",
        server_dir=str(source),
        extra={
            "source_url": "https://github.com/ros-claw/librealsense-mcp",
            "repo_commit": revision,
            "requested_revision": revision,
        },
    )

    assert _adapter_source_matches_lock(
        record,
        "https://github.com/ros-claw/librealsense-mcp.git",
        revision,
        home,
    )
    injected = source / "shadow_module.py"
    injected.write_text("print('injected')\n", encoding="utf-8")
    assert not _adapter_source_matches_lock(
        record,
        "https://github.com/ros-claw/librealsense-mcp",
        revision,
        home,
    )
    injected.unlink()
    ignored = source / "ignored_payload.py"
    ignored.write_text("print('hidden injection')\n", encoding="utf-8")
    assert not _adapter_source_matches_lock(
        record,
        "https://github.com/ros-claw/librealsense-mcp",
        revision,
        home,
    )
    ignored.unlink()
    tracked.write_text("print('modified')\n", encoding="utf-8")
    assert not _adapter_source_matches_lock(
        record,
        "https://github.com/ros-claw/librealsense-mcp",
        revision,
        home,
    )
