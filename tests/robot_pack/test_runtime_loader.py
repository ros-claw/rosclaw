from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.permits import ExecutionPermit, PermitAuthority, action_intent_hash
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.robot_pack.instance import configure_robot_instance
from rosclaw.robot_pack.runtime_loader import (
    RealSenseCaptureExecutor,
    RobotPackRuntimeError,
    load_daemon_robot_pack,
)
from rosclaw.robot_pack.verification import _validate_receipt


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class _Gateway:
    def __init__(self) -> None:
        self.registrations: list[tuple[str, ExecutionMode, object]] = []

    def register_executor(self, capability: str, mode: ExecutionMode, executor: object) -> None:
        self.registrations.append((capability, mode, executor))


def _instance(home: Path):
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
    return configure_robot_instance(
        "realsense",
        home=home,
        instance_id="daemon-d405",
        serial="DAEMON123",
        model="D405",
        allow_offline=True,
    )[0]


def _action(
    instance,
    *,
    approved: bool = True,
    body_snapshot_hash: str | None = None,
    **arguments,
) -> ActionEnvelope:
    return ActionEnvelope(
        action_id="action-camera-test",
        actor_id="test-agent",
        agent_framework="pytest",
        session_id="session-rs",
        body_id=instance.instance_id,
        body_snapshot_hash=body_snapshot_hash or instance.body_snapshot_hash,
        capability_id="camera.capture_rgbd",
        arguments=arguments,
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator",
            approved=approved,
            approval_id="approval",
            scopes=["camera.capture_rgbd"],
        ),
    )


def test_daemon_loader_registers_only_real_read_only_executor(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    runtime = SimpleNamespace(action_gateway=_Gateway())

    status = load_daemon_robot_pack(runtime, robot_id=instance.instance_id, home=home)

    assert status is not None
    assert status["pack_ref"].endswith("realsense-d400@1.0.0")
    assert status["safety"]["actuation"] == "forbidden"
    assert [(capability, mode) for capability, mode, _ in runtime.action_gateway.registrations] == [
        ("camera.capture_rgbd", ExecutionMode.REAL)
    ]


def test_daemon_loader_rejects_body_snapshot_tamper(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    body_path = home / "bodies" / instance.instance_id / "body.yaml"
    body = yaml.safe_load(body_path.read_text(encoding="utf-8"))
    body["body_instance"]["serial_number"] = "TAMPERED"
    body_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    runtime = SimpleNamespace(action_gateway=_Gateway())

    with pytest.raises(RobotPackRuntimeError, match="Body snapshot"):
        load_daemon_robot_pack(runtime, robot_id=instance.instance_id, home=home)


def test_daemon_loader_rejects_adapter_revision_drift(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    registry = InstalledRegistry(home=home)
    record = registry.get("librealsense-mcp")
    assert record is not None
    record.extra["repo_commit"] = "0" * 40
    registry.add(record)
    runtime = SimpleNamespace(action_gateway=_Gateway())

    with pytest.raises(RobotPackRuntimeError, match="adapter binding"):
        load_daemon_robot_pack(runtime, robot_id=instance.instance_id, home=home)


def test_daemon_loader_rejects_instance_device_contract_tamper(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    config_path = home / "robots" / "instances" / f"{instance.instance_id}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["device"]["model"] = "D435i"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RobotPackRuntimeError, match="device contract"):
        load_daemon_robot_pack(
            SimpleNamespace(action_gateway=_Gateway()),
            robot_id=instance.instance_id,
            home=home,
        )


def test_daemon_loader_rejects_instance_safety_contract_tamper(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    config_path = home / "robots" / "instances" / f"{instance.instance_id}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["safety"]["actuation"] = "allowed"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RobotPackRuntimeError, match="safety contract"):
        load_daemon_robot_pack(
            SimpleNamespace(action_gateway=_Gateway()),
            robot_id=instance.instance_id,
            home=home,
        )


def test_daemon_loader_rejects_symlinked_instance_config(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    config_path = home / "robots" / "instances" / f"{instance.instance_id}.yaml"
    external = home / "external-instance.yaml"
    config_path.replace(external)
    config_path.symlink_to(external)

    with pytest.raises(RobotPackRuntimeError, match="symbolic link"):
        load_daemon_robot_pack(
            SimpleNamespace(action_gateway=_Gateway()),
            robot_id=instance.instance_id,
            home=home,
        )


def test_executor_blocks_serial_substitution_before_adapter_call(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    executor = RealSenseCaptureExecutor(instance, home=home)

    result = executor(_action(instance, serial="OTHER"))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_DEVICE_IDENTITY_MISMATCH"


def test_executor_blocks_artifact_path_outside_workspace(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    executor = RealSenseCaptureExecutor(instance, home=home)

    result = executor(_action(instance, output_dir="/tmp/outside-rosclaw"))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_ARTIFACT_PATH_DENIED"


def test_executor_blocks_symlinked_artifact_root(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    external = home / "external-artifacts"
    external.mkdir()
    artifacts = home / "artifacts"
    artifacts.mkdir()
    (artifacts / "robot-packs").symlink_to(external, target_is_directory=True)

    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_ARTIFACT_PATH_DENIED"
    assert list(external.iterdir()) == []


def test_executor_requires_daemon_authorization_and_exact_body_snapshot(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    executor = RealSenseCaptureExecutor(instance, home=home)

    unauthorized = executor(_action(instance, approved=False))
    stale_body = executor(_action(instance, body_snapshot_hash="0" * 64))

    assert unauthorized.errors[0]["code"] == "ROBOT_PACK_AUTHORIZATION_REQUIRED"
    assert stale_body.errors[0]["code"] == "ROBOT_PACK_BODY_SNAPSHOT_MISMATCH"


def test_executor_hashes_real_artifacts_and_reports_physical_observation(
    installed_pack,
    monkeypatch,
) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)

    def fake_run(params):
        output = Path(params["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        color = output / "color.png"
        depth = output / "depth.png"
        color.write_bytes(b"color")
        depth.write_bytes(b"depth")
        return {
            "status": "success",
            "serial": "DAEMON123",
            "timestamp": _now(),
            "server_name": "librealsense-mcp",
            "tool": "capture_aligned_rgbd",
            "artifacts": {"color": str(color), "depth": str(depth)},
            "mcp_result": {"width": 640, "height": 480, "aligned": True},
        }

    import rosclaw.skill.builtins.realsense_capture_rgbd.runner as runner

    monkeypatch.setattr(runner, "run", fake_run)
    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "COMPLETED"
    assert result.evidence_level.value == "PHYSICALLY_OBSERVED"
    assert result.dispatch_result["accepted"] is True
    assert result.observations[0]["device_identity"]["serial"] == "DAEMON123"
    assert set(result.observations[0]["artifact_hashes"]) == {"color", "depth"}


def test_executor_rejects_preexisting_action_artifacts(installed_pack) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)
    output = home / "artifacts" / "robot-packs" / "action-camera-test"
    output.mkdir(parents=True)
    (output / "color.png").write_bytes(b"stale")

    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_ARTIFACT_COLLISION"


def test_executor_rejects_success_without_capture_metadata(installed_pack, monkeypatch) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)

    def fake_run(params):
        output = Path(params["output_dir"])
        (output / "color.png").write_bytes(b"color")
        (output / "depth.png").write_bytes(b"depth")
        return {
            "status": "success",
            "serial": "DAEMON123",
            "timestamp": "",
            "artifacts": {
                "color": str(output / "color.png"),
                "depth": str(output / "depth.png"),
            },
        }

    import rosclaw.skill.builtins.realsense_capture_rgbd.runner as runner

    monkeypatch.setattr(runner, "run", fake_run)
    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_CAPTURE_METADATA_INVALID"


@pytest.mark.parametrize(
    ("timestamp", "server_name"),
    [
        ("not-a-timestamp", "librealsense-mcp"),
        (None, "different-realsense-mcp"),
    ],
)
def test_executor_rejects_invalid_timestamp_or_adapter_server(
    installed_pack,
    monkeypatch,
    timestamp,
    server_name,
) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)

    def fake_run(params):
        output = Path(params["output_dir"])
        color = output / "color.png"
        depth = output / "depth.png"
        color.write_bytes(b"color")
        depth.write_bytes(b"depth")
        return {
            "status": "success",
            "serial": "DAEMON123",
            "timestamp": timestamp or _now(),
            "server_name": server_name,
            "tool": "capture_aligned_rgbd",
            "artifacts": {"color": str(color), "depth": str(depth)},
            "mcp_result": {"width": 640, "height": 480, "aligned": True},
        }

    import rosclaw.skill.builtins.realsense_capture_rgbd.runner as runner

    monkeypatch.setattr(runner, "run", fake_run)
    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_CAPTURE_METADATA_INVALID"


def test_executor_rejects_stale_capture_completion_timestamp(installed_pack, monkeypatch) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)

    def fake_run(params):
        output = Path(params["output_dir"])
        color = output / "color.png"
        depth = output / "depth.png"
        color.write_bytes(b"color")
        depth.write_bytes(b"depth")
        stale = datetime.now(UTC) - timedelta(hours=1)
        return {
            "status": "success",
            "serial": "DAEMON123",
            "timestamp": stale.isoformat().replace("+00:00", "Z"),
            "server_name": "librealsense-mcp",
            "tool": "capture_aligned_rgbd",
            "artifacts": {"color": str(color), "depth": str(depth)},
            "mcp_result": {"width": 640, "height": 480, "aligned": True},
        }

    import rosclaw.skill.builtins.realsense_capture_rgbd.runner as runner

    monkeypatch.setattr(runner, "run", fake_run)
    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_CAPTURE_METADATA_INVALID"


def test_executor_rejects_non_mapping_adapter_response(installed_pack, monkeypatch) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)

    import rosclaw.skill.builtins.realsense_capture_rgbd.runner as runner

    monkeypatch.setattr(runner, "run", lambda _params: None)
    result = RealSenseCaptureExecutor(instance, home=home)(_action(instance))

    assert result.final_state.value == "FAILED"
    assert result.errors[0]["code"] == "ROBOT_PACK_ADAPTER_PROTOCOL_ERROR"


def test_signed_pack_runs_through_daemon_permit_and_canonical_receipt(
    installed_pack,
    monkeypatch,
) -> None:
    home, _store, _manifest = installed_pack
    instance = _instance(home)

    def fake_run(params):
        output = Path(params["output_dir"])
        assert output.is_dir()
        assert not any(output.iterdir())
        color = output / "color.png"
        depth = output / "depth.png"
        color.write_bytes(b"daemon-color")
        depth.write_bytes(b"daemon-depth")
        return {
            "status": "success",
            "serial": "DAEMON123",
            "timestamp": _now(),
            "server_name": "librealsense-mcp",
            "tool": "capture_aligned_rgbd",
            "artifacts": {"color": str(color), "depth": str(depth)},
            "mcp_result": {"width": 640, "height": 480, "aligned": True},
        }

    import rosclaw.skill.builtins.realsense_capture_rgbd.runner as runner

    monkeypatch.setattr(runner, "run", fake_run)
    runtime = Runtime(
        RuntimeConfig(
            robot_id=instance.instance_id,
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )
    pack_status = load_daemon_robot_pack(runtime, robot_id=instance.instance_id, home=home)
    assert pack_status is not None

    action = ActionEnvelope(
        action_id="action-daemon-rgbd",
        actor_id="codex-agent",
        agent_framework="codex",
        session_id="session-daemon-rgbd",
        body_id=instance.instance_id,
        body_snapshot_hash=instance.body_snapshot_hash,
        capability_id="camera.capture_rgbd",
        arguments={},
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator-1",
            approved=False,
            approval_id="permit-realsense",
            scopes=[],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.PHYSICALLY_OBSERVED,
            timeout_sec=2.0,
        ),
    )
    peer = PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
    permits = PermitAuthority()
    permits.register(
        ExecutionPermit(
            permit_id="permit-realsense",
            principal_id="operator-1",
            peer_uid=peer.uid,
            body_id=instance.instance_id,
            body_snapshot_hash=instance.body_snapshot_hash,
            capabilities=("camera.capture_rgbd",),
            action_intent_hash=action_intent_hash(action),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )
    )
    service = DaemonControlPlane(runtime=runtime, permits=permits)
    service.start()
    try:
        service.arm_runtime("Robot Integration test preflight complete", peer)
        service.request_action(action, peer)
        deadline = time.monotonic() + 2.0
        while True:
            status = service.get_action_status(action.action_id, peer)
            if status["state"] == "FINISHED":
                break
            if time.monotonic() >= deadline:
                pytest.fail("daemon did not finish the Robot Pack action")
            time.sleep(0.01)
        daemon_status = service.get_runtime_status(peer)
    finally:
        service.close()

    receipt = status["receipt"]
    assert receipt["final_state"] == "COMPLETED", receipt
    assert receipt["evidence_level"] == "PHYSICALLY_OBSERVED"
    assert receipt["authorization_decision"]["authorized"] is True
    assert receipt["dispatch_result"]["accepted"] is True
    assert receipt["driver_ack"]["acknowledged"] is True
    assert set(receipt["observations"][0]["artifact_hashes"]) == {"color", "depth"}
    receipt_path = home / "artifacts" / "robot-packs" / action.action_id / "receipt.json"
    receipt_ok, receipt_message, _evidence = _validate_receipt(receipt_path, instance, home)
    assert receipt_ok, receipt_message
    assert daemon_status["hardware_actions_executed"] == 1
    assert daemon_status["permits"]["consumed_actions"] == 1
