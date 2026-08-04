"""Fail-closed daemon-side executors loaded from configured Robot Packs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from rosclaw.body.resolver import BodyResolver
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
)
from rosclaw.mcp.onboarding.installed import InstalledRegistry
from rosclaw.robot_pack.instance import RobotInstanceConfig, resolve_adapter_binding
from rosclaw.robot_pack.schema import RobotPackManifest
from rosclaw.robot_pack.store import RobotPackStore
from rosclaw.robot_pack.verifier import verify_robot_pack


class RobotPackRuntimeError(RuntimeError):
    """Raised when daemon startup encounters a configured but unsafe Pack."""


_DAEMON_EXECUTOR_CONTRACTS: dict[str, tuple[str, frozenset[str]]] = {
    "camera.capture_rgbd": ("read_only", frozenset({"REAL"})),
    "limo.navigate_to_pose": ("actuation", frozenset({"SHADOW", "REAL"})),
    "limo.play_tone": ("actuation", frozenset({"SHADOW", "REAL"})),
    "limo.speak_text": ("actuation", frozenset({"SHADOW", "REAL"})),
    "limo.set_initial_pose": ("actuation", frozenset({"SHADOW", "REAL"})),
}


def validate_daemon_loader_contract(
    manifest: RobotPackManifest,
) -> tuple[bool, tuple[str, ...]]:
    """Validate that every declared capability has a matching daemon executor contract."""

    errors: list[str] = []
    for capability in manifest.capabilities:
        expected = _DAEMON_EXECUTOR_CONTRACTS.get(capability.id)
        if expected is None:
            errors.append(f"no daemon executor is implemented for {capability.id}")
            continue
        expected_safety_class, required_modes = expected
        if capability.safety_class != expected_safety_class:
            errors.append(
                f"{capability.id} requires safety_class {expected_safety_class}, "
                f"got {capability.safety_class}"
            )
        missing_modes = required_modes.difference(capability.execution_modes)
        if missing_modes:
            errors.append(
                f"{capability.id} is missing execution modes: {', '.join(sorted(missing_modes))}"
            )
    return not errors, tuple(errors)


class _ArtifactDirectoryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RealSenseCaptureExecutor:
    """Daemon-owned adapter from ActionEnvelope to the existing MCP-only skill runner."""

    def __init__(self, instance: RobotInstanceConfig, *, home: Path) -> None:
        self.instance = instance
        self.home = home.resolve()
        self.artifacts_root = self.home / "artifacts" / "robot-packs"

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        if action.body_id != self.instance.instance_id:
            return _failed_result(
                "ROBOT_PACK_BODY_MISMATCH",
                f"Action body {action.body_id!r} is not Pack instance {self.instance.instance_id!r}",
            )
        if action.body_snapshot_hash != self.instance.body_snapshot_hash:
            return _failed_result(
                "ROBOT_PACK_BODY_SNAPSHOT_MISMATCH",
                "Action Body snapshot does not match the configured Robot Pack instance",
            )
        if action.capability_id != "camera.capture_rgbd":
            return _failed_result(
                "ROBOT_PACK_CAPABILITY_MISMATCH",
                f"Unsupported RealSense Pack capability: {action.capability_id}",
            )
        if (
            not action.authorization.approved
            or not action.authorization.approval_id
            or action.capability_id not in action.authorization.scopes
        ):
            return _failed_result(
                "ROBOT_PACK_AUTHORIZATION_REQUIRED",
                "RealSense REAL execution requires daemon-authored capability authorization",
            )
        requested_serial = action.arguments.get("serial")
        if requested_serial and str(requested_serial) != self.instance.device.serial:
            return _failed_result(
                "ROBOT_PACK_DEVICE_IDENTITY_MISMATCH",
                "Action attempted to substitute a different device serial",
            )

        try:
            output_dir = self._output_directory(action)
        except _ArtifactDirectoryError as exc:
            return _failed_result(
                exc.code,
                str(exc),
            )
        params = {
            **action.arguments,
            "workspace": str(self.home),
            "body_id": self.instance.instance_id,
            "serial": self.instance.device.serial,
            "server_name": self.instance.adapter.server_name,
            "output_dir": str(output_dir),
        }
        capture_started_at = datetime.now(UTC)
        try:
            from rosclaw.skill.builtins.realsense_capture_rgbd.runner import run

            result = run(params)
        except Exception as exc:  # noqa: BLE001 - native/MCP failures become receipts
            return _failed_result("ROBOT_PACK_ADAPTER_ERROR", str(exc), output_dir=output_dir)
        capture_finished_at = datetime.now(UTC)

        if not isinstance(result, dict):
            return _failed_result(
                "ROBOT_PACK_ADAPTER_PROTOCOL_ERROR",
                "RealSense adapter returned a non-mapping response",
                output_dir=output_dir,
            )
        if result.get("status") != "success":
            return _failed_result(
                "ROBOT_PACK_CAPTURE_FAILED",
                str(result.get("reason") or "RealSense MCP capture failed"),
                output_dir=output_dir,
            )
        captured_at = str(result.get("timestamp") or "")
        captured_timestamp = _parse_timestamp(captured_at)
        mcp_result = result.get("mcp_result")
        metadata_ok = bool(
            result.get("serial") == self.instance.device.serial
            and result.get("server_name") == self.instance.adapter.server_name
            and result.get("tool") in {"capture_aligned_rgbd", "capture_frames"}
            and captured_timestamp is not None
            and capture_started_at - timedelta(seconds=5)
            <= captured_timestamp
            <= capture_finished_at + timedelta(seconds=5)
            and isinstance(mcp_result, dict)
            and _is_positive_int(mcp_result.get("width"))
            and _is_positive_int(mcp_result.get("height"))
            and mcp_result.get("aligned") is True
        )
        if not metadata_ok:
            return _failed_result(
                "ROBOT_PACK_CAPTURE_METADATA_INVALID",
                "Capture must report the exact serial, timestamp, positive dimensions, and RGB-D alignment",
                output_dir=output_dir,
            )
        assert isinstance(mcp_result, dict)
        artifacts = _resolve_rgbd_artifacts(result, output_dir)
        missing = [name for name in ("color", "depth") if name not in artifacts]
        if missing:
            return _failed_result(
                "ROBOT_PACK_ARTIFACT_MISSING",
                f"Capture did not produce required artifacts: {', '.join(missing)}",
                output_dir=output_dir,
            )
        hashes = {name: f"sha256:{_hash_file(path)}" for name, path in artifacts.items()}
        artifact_uris = [path.as_uri() for path in artifacts.values()]
        observation = {
            "kind": "rgbd_capture",
            "device_identity": {
                "model": self.instance.device.model,
                "serial": self.instance.device.serial,
                "stable_uri": self.instance.device.stable_uri,
                "firmware": result.get("firmware") or self.instance.device.firmware_at_configure,
            },
            "captured_at": captured_at,
            "artifact_hashes": hashes,
            "artifacts": {name: path.as_uri() for name, path in artifacts.items()},
            "metrics": {
                **(result.get("metrics") if isinstance(result.get("metrics"), dict) else {}),
                "width": mcp_result["width"],
                "height": mcp_result["height"],
                "aligned": True,
            },
        }
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.PHYSICALLY_OBSERVED,
            policy_decision={
                "allowed": True,
                "policy": "robot-pack/perception-only",
                "reason": "read-only Pack capability",
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "server_name": result.get("server_name"),
                "tool": result.get("tool"),
            },
            driver_ack={"acknowledged": True, "captured_at": captured_at},
            observations=[observation],
            verification_result={
                "success": True,
                "predicate": "aligned RGB-D artifacts exist and match recorded hashes",
                "artifact_hashes": hashes,
            },
            artifacts=artifact_uris,
            artifact_directory=str(output_dir),
        )

    def _output_directory(self, action: ActionEnvelope) -> Path:
        artifacts_parent = self.artifacts_root.parent
        if artifacts_parent.is_symlink() or self.artifacts_root.is_symlink():
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_PATH_DENIED",
                "Robot Pack artifact path cannot contain a symbolic link",
            )
        artifacts_parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(exist_ok=True)
        if (
            artifacts_parent.is_symlink()
            or self.artifacts_root.is_symlink()
            or self.artifacts_root.resolve() != self.artifacts_root
        ):
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_PATH_DENIED",
                "Robot Pack artifact path cannot contain a symbolic link",
            )
        configured = action.arguments.get("output_dir")
        candidate = (
            Path(str(configured)).expanduser()
            if configured
            else self.artifacts_root / action.action_id
        )
        if not candidate.is_absolute():
            candidate = self.artifacts_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.artifacts_root)
        except ValueError:
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_PATH_DENIED",
                "RealSense artifacts must remain under ROSCLAW_HOME/artifacts/robot-packs",
            ) from None
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_COLLISION",
                "A fresh artifact directory is required for every action id",
            ) from exc
        return candidate


class LimoInitialPoseExecutor:
    """Daemon-owned adapter for one bounded ROS 1 AMCL initialization operation."""

    _CAPABILITY = "limo.set_initial_pose"
    _SCHEMA = "limo.initial-pose.v1"
    _PROTOCOL = "rosclaw.limo.worker.v1"
    _COVARIANCE = [0.25, 0.25, 0.0, 0.0, 0.0, 0.0685]
    _ARGUMENT_KEYS = {
        "schema_version",
        "target_pose",
        "route_policy_id",
        "route_policy_hash",
        "map_id",
        "map_image_hash",
        "covariance_diagonal",
        "expected_effect",
    }

    def __init__(
        self,
        instance: RobotInstanceConfig,
        *,
        adapter_source: Path,
        python_executable: str = "/usr/bin/python2",
    ) -> None:
        self.instance = instance
        self.adapter_source = adapter_source.resolve()
        self.python_executable = python_executable
        self.worker_path = self.adapter_source / "worker" / "limo_initial_pose_worker.py"

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        contract_error = self._validate_action(action, require_authorization=True)
        if contract_error is not None:
            return _failed_result(contract_error[0], contract_error[1])
        if not self._trusted_worker_path():
            return _failed_result(
                "LIMO_WORKER_INTEGRITY_ERROR",
                "Revision-locked LIMO initial-pose worker is missing or unsafe",
            )

        request = {
            "protocol": self._PROTOCOL,
            "operation": "SET_INITIAL_POSE",
            "schema_version": self._SCHEMA,
            "action_id": action.action_id,
            "body_id": action.body_id,
            "body_snapshot_hash": action.body_snapshot_hash,
            "target_pose": action.arguments["target_pose"],
            "covariance_diagonal": self._COVARIANCE,
            "subscriber_timeout_sec": 3.0,
            "verification_timeout_sec": min(12.0, action.verification_policy.timeout_sec),
        }
        try:
            completed = subprocess.run(
                [self.python_executable, str(self.worker_path)],
                input=json.dumps(request, separators=(",", ":")),
                capture_output=True,
                text=True,
                check=False,
                timeout=min(20.0, action.verification_policy.timeout_sec + 5.0),
                env=self._worker_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _failed_result("LIMO_WORKER_LAUNCH_FAILED", str(exc))
        if len(completed.stdout.encode("utf-8")) > 262_144:
            return _failed_result("LIMO_WORKER_PROTOCOL_ERROR", "Worker output exceeded byte limit")
        try:
            result = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            return _failed_result("LIMO_WORKER_PROTOCOL_ERROR", f"Invalid worker JSON: {exc}")
        if (
            completed.returncode != 0
            or not isinstance(result, dict)
            or result.get("ok") is not True
        ):
            message = result.get("error") if isinstance(result, dict) else completed.stderr
            return _failed_result("LIMO_INITIAL_POSE_FAILED", str(message or "ROS 1 worker failed"))
        verification_error = self._validate_result(action, result)
        if verification_error is not None:
            return _failed_result("LIMO_INITIAL_POSE_VERIFICATION_FAILED", verification_error)

        target = action.arguments["target_pose"]
        observed = result["observed_amcl_pose"]
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            policy_decision={
                "allowed": True,
                "policy": "robot-pack/limo-initial-pose-v1",
                "reason": "map-frame pose and fixed covariance contract passed",
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "operation": "SET_INITIAL_POSE",
                "topic": "/initialpose",
            },
            driver_ack={
                "acknowledged": True,
                "subscriber_count": result["subscriber_count"],
                "dispatched_wall_time": result["dispatched_wall_time"],
            },
            observations=[
                {
                    "kind": "amcl_localization_initialized",
                    "target_pose": target,
                    "observed_amcl_pose": observed,
                    "map_to_odom": result["map_to_odom"],
                    "completed_wall_time": result["completed_wall_time"],
                }
            ],
            verification_result={
                "success": True,
                "predicate": "post-dispatch AMCL pose and map-to-odom transform observed",
                "position_error_m": math.hypot(
                    float(observed["x"]) - float(target["x"]),
                    float(observed["y"]) - float(target["y"]),
                ),
                "yaw_error_rad": abs(
                    math.atan2(
                        math.sin(float(observed["yaw"]) - float(target["yaw"])),
                        math.cos(float(observed["yaw"]) - float(target["yaw"])),
                    )
                ),
            },
        )

    def _validate_action(
        self, action: ActionEnvelope, *, require_authorization: bool
    ) -> tuple[str, str] | None:
        if action.body_id != self.instance.instance_id:
            return "ROBOT_PACK_BODY_MISMATCH", "Action Body does not match LIMO instance"
        if action.body_snapshot_hash != self.instance.body_snapshot_hash:
            return "ROBOT_PACK_BODY_SNAPSHOT_MISMATCH", "Action Body snapshot is stale"
        if action.capability_id != self._CAPABILITY:
            return "ROBOT_PACK_CAPABILITY_MISMATCH", "Unsupported LIMO capability"
        if require_authorization and (
            not action.authorization.approved
            or not action.authorization.approval_id
            or self._CAPABILITY not in action.authorization.scopes
        ):
            return (
                "ROBOT_PACK_AUTHORIZATION_REQUIRED",
                "LIMO REAL execution requires daemon-authored exact authorization",
            )
        arguments = action.arguments
        if set(arguments) != self._ARGUMENT_KEYS or arguments.get("schema_version") != self._SCHEMA:
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Initial-pose action fields are not exact"
        pose = arguments.get("target_pose")
        if not isinstance(pose, dict) or set(pose) != {"frame_id", "x", "y", "yaw"}:
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "target_pose fields are not exact"
        if pose.get("frame_id") != "map":
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Only map-frame poses are accepted"
        values = (pose.get("x"), pose.get("y"), pose.get("yaw"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Pose values must be numbers"
        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose["yaw"])
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Pose values must be finite"
        if not (-50.0 <= x <= 50.0 and -50.0 <= y <= 50.0 and -math.pi <= yaw <= math.pi):
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Pose exceeds bounded map contract"
        if arguments.get("covariance_diagonal") != self._COVARIANCE:
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Covariance must match operator policy"
        expected = arguments.get("expected_effect")
        if expected != {
            "kind": "localization_initialized",
            "final_frame": "map",
            "map_to_odom_required": True,
        }:
            return "LIMO_INITIAL_POSE_CONTRACT_INVALID", "Expected effect is not exact"
        for name in ("route_policy_id", "route_policy_hash", "map_id", "map_image_hash"):
            if not isinstance(arguments.get(name), str) or not arguments[name].strip():
                return "LIMO_INITIAL_POSE_CONTRACT_INVALID", f"{name} is required"
        return None

    def _trusted_worker_path(self) -> bool:
        if self.adapter_source.is_symlink() or self.worker_path.is_symlink():
            return False
        try:
            self.worker_path.resolve().relative_to(self.adapter_source)
        except ValueError:
            return False
        return self.worker_path.is_file() and os.access(self.worker_path, os.R_OK)

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        allowed = {
            "PATH",
            "PYTHONPATH",
            "ROS_MASTER_URI",
            "ROS_HOSTNAME",
            "ROS_IP",
            "ROS_PACKAGE_PATH",
            "ROS_DISTRO",
            # Fixed allowlisted PulseAudio bridge used by the isolated LIMO
            # audio worker.  The worker independently rejects any other path.
            "ROSCLAW_LIMO_PULSE_SERVER",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    def _validate_result(self, action: ActionEnvelope, result: dict[str, Any]) -> str | None:
        if (
            result.get("protocol") != self._PROTOCOL
            or result.get("action_id") != action.action_id
            or result.get("operation") != "SET_INITIAL_POSE"
            or result.get("topic") != "/initialpose"
            or result.get("accepted") is not True
        ):
            return "Worker acknowledgement did not match the action"
        if not isinstance(result.get("subscriber_count"), int) or result["subscriber_count"] < 1:
            return "AMCL subscriber acknowledgement is missing"
        observed = result.get("observed_amcl_pose")
        if not isinstance(observed, dict) or observed.get("frame_id") != "map":
            return "Post-dispatch AMCL pose is missing"
        try:
            observed_values = [float(observed[name]) for name in ("x", "y", "yaw")]
        except (KeyError, TypeError, ValueError):
            return "Post-dispatch AMCL pose is malformed"
        if not all(math.isfinite(value) for value in observed_values):
            return "Post-dispatch AMCL pose is non-finite"
        transform = result.get("map_to_odom")
        if not isinstance(transform, dict):
            return "map-to-odom transform is missing"
        if not (
            isinstance(transform.get("translation"), list)
            and len(transform["translation"]) == 3
            and isinstance(transform.get("rotation"), list)
            and len(transform["rotation"]) == 4
        ):
            return "map-to-odom transform is malformed"
        target = action.arguments["target_pose"]
        position_error = math.hypot(
            observed_values[0] - float(target["x"]), observed_values[1] - float(target["y"])
        )
        yaw_error = abs(
            math.atan2(
                math.sin(observed_values[2] - float(target["yaw"])),
                math.cos(observed_values[2] - float(target["yaw"])),
            )
        )
        if position_error > 1.0 or yaw_error > 1.0:
            return "AMCL observation did not converge near the requested estimate"
        return None


class LimoInitialPoseShadowExecutor:
    """Daemon-owned contract preview that performs no ROS or hardware operation."""

    def __init__(self, instance: RobotInstanceConfig, *, adapter_source: Path) -> None:
        self.instance = instance
        self.validator = LimoInitialPoseExecutor(instance, adapter_source=adapter_source)

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        contract_error = self.validator._validate_action(action, require_authorization=False)
        if contract_error is not None:
            return _failed_result(contract_error[0], contract_error[1])
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            policy_decision={
                "allowed": True,
                "policy": "robot-pack/limo-initial-pose-v1",
                "reason": "bounded initial-pose contract is valid for SHADOW preview",
            },
            authorization_decision={"authorized": False, "required": False},
            dispatch_result={
                "accepted": True,
                "shadow": True,
                "operation": "VALIDATE_INITIAL_POSE",
                "hardware_dispatched": False,
            },
            driver_ack={"acknowledged": True, "shadow": True},
            observations=[
                {
                    "kind": "initial_pose_contract_preview",
                    "target_pose": action.arguments["target_pose"],
                    "hardware_observed": False,
                }
            ],
            verification_result={
                "success": True,
                "predicate": "exact initial-pose action contract validated without ROS dispatch",
            },
        )


class _LimoFixedWorkerExecutor:
    """Shared fail-closed launcher for revision-locked LIMO workers."""

    capability_id = ""
    schema = ""
    operation = ""
    worker_name = ""
    policy_name = ""
    evidence_level = EvidenceLevel.TASK_VERIFIED
    argument_keys: frozenset[str] = frozenset()

    def __init__(
        self,
        instance: RobotInstanceConfig,
        *,
        adapter_source: Path,
        python_executable: str = "/usr/bin/python2",
    ) -> None:
        self.instance = instance
        self.adapter_source = adapter_source.resolve()
        self.python_executable = python_executable
        self.worker_path = self.adapter_source / "worker" / self.worker_name

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        contract_error = self._validate_action(action, require_authorization=True)
        if contract_error is not None:
            return _failed_result(contract_error[0], contract_error[1])
        if not self._trusted_worker_path():
            return _failed_result(
                "LIMO_WORKER_INTEGRITY_ERROR",
                f"Revision-locked LIMO {self.operation} worker is missing or unsafe",
            )
        request = self._worker_request(action)
        try:
            completed = subprocess.run(
                [self.python_executable, str(self.worker_path)],
                input=json.dumps(request, separators=(",", ":")),
                capture_output=True,
                text=True,
                check=False,
                timeout=min(130.0, action.verification_policy.timeout_sec + 8.0),
                env=LimoInitialPoseExecutor._worker_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _failed_result("LIMO_WORKER_LAUNCH_FAILED", str(exc))
        if len(completed.stdout.encode("utf-8")) > 262_144:
            return _failed_result("LIMO_WORKER_PROTOCOL_ERROR", "Worker output exceeded byte limit")
        try:
            result = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            return _failed_result("LIMO_WORKER_PROTOCOL_ERROR", f"Invalid worker JSON: {exc}")
        if (
            completed.returncode != 0
            or not isinstance(result, dict)
            or result.get("ok") is not True
        ):
            message = result.get("error") if isinstance(result, dict) else completed.stderr
            return _failed_result(
                f"LIMO_{self.operation}_FAILED",
                str(message or "fixed-operation worker failed"),
            )
        verification_error = self._validate_result(action, result)
        if verification_error is not None:
            return self._verification_failure_result(action, result, verification_error)
        return self._success_result(action, result)

    def _validate_action(
        self, action: ActionEnvelope, *, require_authorization: bool
    ) -> tuple[str, str] | None:
        if action.body_id != self.instance.instance_id:
            return "ROBOT_PACK_BODY_MISMATCH", "Action Body does not match LIMO instance"
        if action.body_snapshot_hash != self.instance.body_snapshot_hash:
            return "ROBOT_PACK_BODY_SNAPSHOT_MISMATCH", "Action Body snapshot is stale"
        if action.capability_id != self.capability_id:
            return "ROBOT_PACK_CAPABILITY_MISMATCH", "Unsupported LIMO capability"
        if require_authorization and (
            not action.authorization.approved
            or not action.authorization.approval_id
            or self.capability_id not in action.authorization.scopes
        ):
            return (
                "ROBOT_PACK_AUTHORIZATION_REQUIRED",
                "LIMO REAL execution requires daemon-authored exact authorization",
            )
        if (
            set(action.arguments) != self.argument_keys
            or action.arguments.get("schema_version") != self.schema
        ):
            return (
                f"LIMO_{self.operation}_CONTRACT_INVALID",
                f"{self.operation} action fields are not exact",
            )
        return self._validate_arguments(action.arguments)

    def _trusted_worker_path(self) -> bool:
        if self.adapter_source.is_symlink() or self.worker_path.is_symlink():
            return False
        try:
            self.worker_path.resolve().relative_to(self.adapter_source)
        except ValueError:
            return False
        return self.worker_path.is_file() and os.access(self.worker_path, os.R_OK)

    def _validate_arguments(self, arguments: dict[str, Any]) -> tuple[str, str] | None:
        raise NotImplementedError

    def _worker_request(self, action: ActionEnvelope) -> dict[str, Any]:
        raise NotImplementedError

    def _validate_result(self, action: ActionEnvelope, result: dict[str, Any]) -> str | None:
        raise NotImplementedError

    def _success_result(
        self, action: ActionEnvelope, result: dict[str, Any]
    ) -> ActionExecutionResult:
        raise NotImplementedError

    def _verification_failure_result(
        self,
        action: ActionEnvelope,
        result: dict[str, Any],
        verification_error: str,
    ) -> ActionExecutionResult:
        del action, result
        return _failed_result(
            f"LIMO_{self.operation}_VERIFICATION_FAILED",
            verification_error,
        )


class LimoNavigationExecutor(_LimoFixedWorkerExecutor):
    """Daemon-owned move_base executor with fresh ROS preflight and stop verification."""

    capability_id = "limo.navigate_to_pose"
    schema = "limo.navigation.v2"
    operation = "NAVIGATION"
    worker_name = "limo_navigation_worker.py"
    policy_name = "robot-pack/limo-navigation-v2"
    evidence_level = EvidenceLevel.TASK_VERIFIED
    argument_keys = frozenset(
        {
            "schema_version",
            "target_pose",
            "readiness_snapshot_hash",
            "route_policy_id",
            "route_policy_hash",
            "map_id",
            "map_image_hash",
            "goal_tolerance",
            "expected_effect",
        }
    )

    def _validate_arguments(self, arguments: dict[str, Any]) -> tuple[str, str] | None:
        code = "LIMO_NAVIGATION_CONTRACT_INVALID"
        pose = arguments.get("target_pose")
        if not isinstance(pose, dict) or set(pose) != {"frame_id", "x", "y", "yaw"}:
            return code, "target_pose fields are not exact"
        if pose.get("frame_id") != "map":
            return code, "Only map-frame navigation goals are accepted"
        values = (pose.get("x"), pose.get("y"), pose.get("yaw"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return code, "Navigation pose values must be numbers"
        x, y, yaw = (float(cast(int | float, value)) for value in values)
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return code, "Navigation pose values must be finite"
        if not (-50.0 <= x <= 50.0 and -50.0 <= y <= 50.0 and -math.pi <= yaw <= math.pi):
            return code, "Navigation pose exceeds bounded map contract"
        tolerance = arguments.get("goal_tolerance")
        if not isinstance(tolerance, dict) or set(tolerance) != {"xy_m", "yaw_rad"}:
            return code, "goal_tolerance fields are not exact"
        xy_m = tolerance.get("xy_m")
        yaw_rad = tolerance.get("yaw_rad")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (xy_m, yaw_rad)
        ):
            return code, "Goal tolerances must be numbers"
        if (
            not 0.05 <= float(cast(int | float, xy_m)) <= 0.5
            or not 0.05 <= float(cast(int | float, yaw_rad)) <= 0.8
        ):
            return code, "Goal tolerances exceed daemon bounds"
        expected = arguments.get("expected_effect")
        if expected != {
            "kind": "navigate_to_pose",
            "final_frame": "map",
            "stop_required": True,
        }:
            return code, "Expected navigation effect is not exact"
        for name in (
            "readiness_snapshot_hash",
            "route_policy_id",
            "route_policy_hash",
            "map_id",
            "map_image_hash",
        ):
            if not isinstance(arguments.get(name), str) or not arguments[name].strip():
                return code, f"{name} is required"
        return None

    def _worker_request(self, action: ActionEnvelope) -> dict[str, Any]:
        return {
            "protocol": "rosclaw.limo.worker.v1",
            "operation": "NAVIGATE_TO_POSE",
            "schema_version": self.schema,
            "action_id": action.action_id,
            "body_id": action.body_id,
            "body_snapshot_hash": action.body_snapshot_hash,
            "target_pose": action.arguments["target_pose"],
            "goal_tolerance": action.arguments["goal_tolerance"],
            "server_timeout_sec": 3.0,
            "navigation_timeout_sec": min(
                100.0, max(5.0, action.verification_policy.timeout_sec - 10.0)
            ),
            "verification_timeout_sec": 5.0,
        }

    def _validate_result(self, action: ActionEnvelope, result: dict[str, Any]) -> str | None:
        if (
            result.get("protocol") != "rosclaw.limo.worker.v1"
            or result.get("action_id") != action.action_id
            or result.get("operation") != "NAVIGATE_TO_POSE"
            or result.get("action_server") != "/move_base"
            or result.get("accepted") is not True
            or result.get("terminal_state") != 3
        ):
            return "Worker acknowledgement did not match the navigation action"
        observed = result.get("observed_final_pose")
        if not isinstance(observed, dict) or set(observed) != {"frame_id", "x", "y", "yaw"}:
            return "Final AMCL pose is missing or malformed"
        if observed.get("frame_id") != "map":
            return "Final AMCL pose is not map-frame"
        tolerance = action.arguments["goal_tolerance"]
        try:
            position_error = float(result["position_error_m"])
            yaw_error = float(result["yaw_error_rad"])
        except (KeyError, TypeError, ValueError):
            return "Navigation error metrics are malformed"
        if (
            not all(math.isfinite(value) for value in (position_error, yaw_error))
            or position_error > float(tolerance["xy_m"])
            or yaw_error > float(tolerance["yaw_rad"])
        ):
            return "Final AMCL pose is outside requested tolerance"
        stopped = result.get("stopped_odometry")
        if not isinstance(stopped, dict):
            return "Stopped odometry verification is missing"
        try:
            linear = float(stopped["linear_speed_mps"])
            angular = float(stopped["angular_speed_radps"])
        except (KeyError, TypeError, ValueError):
            return "Stopped odometry verification is malformed"
        if linear > 0.03 or angular > 0.08:
            return "Base did not verify stopped after navigation"
        preflight = result.get("preflight")
        if (
            not isinstance(preflight, dict)
            or preflight.get("chassis_error_code") != 0
            or not isinstance(preflight.get("map_to_odom"), dict)
            or not isinstance(preflight.get("map_to_base"), dict)
            or not isinstance(preflight.get("initial_goal_error"), dict)
            or not isinstance(preflight.get("active_goal_tolerance"), dict)
            or not isinstance(preflight.get("goal_already_satisfied"), bool)
        ):
            return "Daemon-owned navigation preflight evidence is incomplete"
        motion = result.get("motion_evidence")
        if not isinstance(motion, dict):
            return "Navigation motion evidence is missing"
        movement_expected = motion.get("movement_expected")
        motion_observed = motion.get("motion_observed")
        if not isinstance(movement_expected, bool) or not isinstance(motion_observed, bool):
            return "Navigation motion classification is malformed"
        if preflight["goal_already_satisfied"] is movement_expected:
            return "Navigation goal and movement classifications are inconsistent"
        try:
            motion_metrics = [
                float(motion[name])
                for name in (
                    "odom_translation_m",
                    "odom_rotation_rad",
                    "map_translation_m",
                    "map_rotation_rad",
                    "translation_threshold_m",
                    "rotation_threshold_rad",
                )
            ]
        except (KeyError, TypeError, ValueError):
            return "Navigation displacement metrics are malformed"
        if not all(math.isfinite(value) and value >= 0.0 for value in motion_metrics):
            return "Navigation displacement metrics are invalid"
        if movement_expected and not motion_observed:
            return "Expected base movement was not observed in odometry"
        if motion.get("physical_motion_independently_observed") is not False:
            return "Worker cannot claim independent physical motion observation"
        return None

    def _success_result(
        self, action: ActionEnvelope, result: dict[str, Any]
    ) -> ActionExecutionResult:
        motion = result["motion_evidence"]
        movement_expected = motion["movement_expected"]
        motion_observed = motion["motion_observed"]
        observation_kind = (
            "navigation_motion_observed"
            if movement_expected
            else "navigation_goal_already_satisfied"
        )
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            policy_decision={
                "allowed": True,
                "policy": self.policy_name,
                "reason": (
                    "fresh ROS preflight, bounded goal, and odometry motion checks passed"
                    if movement_expected
                    else "goal was already inside the active planner tolerance before dispatch"
                ),
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "operation": "NAVIGATE_TO_POSE",
                "action_server": "/move_base",
                "terminal_state": result["terminal_state"],
                "terminal_text": result["terminal_text"],
                "movement_expected": movement_expected,
                "motion_observed": motion_observed,
            },
            driver_ack={
                "acknowledged": True,
                "dispatched_wall_time": result["dispatched_wall_time"],
            },
            observations=[
                {
                    "kind": observation_kind,
                    "target_pose": action.arguments["target_pose"],
                    "observed_final_pose": result["observed_final_pose"],
                    "preflight": result["preflight"],
                    "stopped_odometry": result["stopped_odometry"],
                    "map_to_base_after": result["map_to_base_after"],
                    "motion_evidence": motion,
                    "completed_wall_time": result["completed_wall_time"],
                }
            ],
            verification_result={
                "success": True,
                "predicate": (
                    "move_base SUCCEEDED, final pose reached tolerance, odometry movement was "
                    "observed, and the base stopped"
                    if movement_expected
                    else "goal was already inside active tolerance, move_base SUCCEEDED without "
                    "required motion, and the base remained stopped"
                ),
                "position_error_m": result["position_error_m"],
                "yaw_error_rad": result["yaw_error_rad"],
                "movement_expected": movement_expected,
                "motion_observed": motion_observed,
                "physical_motion_independently_observed": False,
                "odom_translation_m": motion["odom_translation_m"],
                "odom_rotation_rad": motion["odom_rotation_rad"],
            },
        )

    def _verification_failure_result(
        self,
        action: ActionEnvelope,
        result: dict[str, Any],
        verification_error: str,
    ) -> ActionExecutionResult:
        # A task-level verification failure does not erase a valid worker ACK.
        # Preserve the fact that move_base accepted and completed the hardware
        # dispatch, while still failing closed on the requested task predicate.
        if (
            result.get("protocol") != "rosclaw.limo.worker.v1"
            or result.get("action_id") != action.action_id
            or result.get("operation") != "NAVIGATE_TO_POSE"
            or result.get("action_server") != "/move_base"
            or result.get("accepted") is not True
            or not isinstance(result.get("terminal_state"), int)
            or isinstance(result.get("terminal_state"), bool)
            or not isinstance(result.get("dispatched_wall_time"), (int, float))
            or isinstance(result.get("dispatched_wall_time"), bool)
        ):
            return super()._verification_failure_result(action, result, verification_error)

        motion = result.get("motion_evidence")
        motion = motion if isinstance(motion, dict) else {}
        observation = {
            "kind": "navigation_verification_failed",
            "target_pose": action.arguments["target_pose"],
            "observed_final_pose": result.get("observed_final_pose"),
            "preflight": result.get("preflight"),
            "stopped_odometry": result.get("stopped_odometry"),
            "map_to_base_after": result.get("map_to_base_after"),
            "motion_evidence": motion,
            "completed_wall_time": result.get("completed_wall_time"),
        }
        return ActionExecutionResult(
            final_state=ActionState.FAILED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
            policy_decision={
                "allowed": True,
                "policy": self.policy_name,
                "reason": "bounded navigation dispatch was allowed; task verification failed",
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "operation": "NAVIGATE_TO_POSE",
                "action_server": "/move_base",
                "terminal_state": result["terminal_state"],
                "terminal_text": result.get("terminal_text"),
                "movement_expected": motion.get("movement_expected"),
                "motion_observed": motion.get("motion_observed"),
            },
            driver_ack={
                "acknowledged": True,
                "dispatched_wall_time": result["dispatched_wall_time"],
            },
            observations=[observation],
            verification_result={
                "success": False,
                "predicate": verification_error,
                "position_error_m": result.get("position_error_m"),
                "yaw_error_rad": result.get("yaw_error_rad"),
                "movement_expected": motion.get("movement_expected"),
                "motion_observed": motion.get("motion_observed"),
                "physical_motion_independently_observed": motion.get(
                    "physical_motion_independently_observed"
                ),
                "odom_translation_m": motion.get("odom_translation_m"),
                "odom_rotation_rad": motion.get("odom_rotation_rad"),
            },
            errors=[
                {
                    "code": "LIMO_NAVIGATION_VERIFICATION_FAILED",
                    "message": verification_error,
                }
            ],
        )


class LimoToneExecutor(_LimoFixedWorkerExecutor):
    """Daemon-owned bounded tone executor with microphone loopback verification."""

    capability_id = "limo.play_tone"
    schema = "limo.tone.v1"
    operation = "TONE"
    worker_name = "limo_tone_worker.py"
    policy_name = "robot-pack/limo-tone-v1"
    evidence_level = EvidenceLevel.TASK_VERIFIED
    argument_keys = frozenset(
        {
            "schema_version",
            "frequency_hz",
            "duration_sec",
            "volume_percent",
            "expected_effect",
        }
    )

    def _validate_arguments(self, arguments: dict[str, Any]) -> tuple[str, str] | None:
        code = "LIMO_TONE_CONTRACT_INVALID"
        frequency = arguments.get("frequency_hz")
        duration = arguments.get("duration_sec")
        volume = arguments.get("volume_percent")
        if isinstance(frequency, bool) or frequency not in {440, 660, 880}:
            return code, "frequency_hz must be one of 440, 660, or 880"
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not 0.2 <= float(duration) <= 1.0
        ):
            return code, "duration_sec must be within [0.2, 1.0]"
        if isinstance(volume, bool) or not isinstance(volume, int) or not 5 <= volume <= 25:
            return code, "volume_percent must be an integer within [5, 25]"
        if arguments.get("expected_effect") != {
            "kind": "speaker_tone",
            "playback_required": True,
            "mixer_restore_required": True,
            "microphone_loopback_required": True,
        }:
            return code, "Expected tone effect is not exact"
        return None

    def _worker_request(self, action: ActionEnvelope) -> dict[str, Any]:
        return {
            "protocol": "rosclaw.limo.worker.v1",
            "operation": "PLAY_TONE",
            "schema_version": self.schema,
            "action_id": action.action_id,
            "body_id": action.body_id,
            "body_snapshot_hash": action.body_snapshot_hash,
            "frequency_hz": action.arguments["frequency_hz"],
            "duration_sec": action.arguments["duration_sec"],
            "volume_percent": action.arguments["volume_percent"],
        }

    def _validate_result(self, action: ActionEnvelope, result: dict[str, Any]) -> str | None:
        if (
            result.get("protocol") != "rosclaw.limo.worker.v1"
            or result.get("action_id") != action.action_id
            or result.get("operation") != "PLAY_TONE"
            or result.get("accepted") is not True
            or result.get("mixer_restored") is not True
        ):
            return "Worker acknowledgement did not match the tone action"
        if (
            result.get("frequency_hz") != action.arguments["frequency_hz"]
            or result.get("duration_sec") != action.arguments["duration_sec"]
            or result.get("volume_percent") != action.arguments["volume_percent"]
        ):
            return "Tone playback acknowledgement parameters do not match"
        backend = result.get("playback_backend")
        device = result.get("device")
        if (
            backend not in {"alsa", "pulseaudio"}
            or not isinstance(device, str)
            or (backend == "alsa" and not device.startswith("plughw:"))
            or (backend == "pulseaudio" and not device.startswith("pulse:alsa_output.usb-"))
        ):
            return "Audio playback backend acknowledgement is missing or invalid"
        if not isinstance(result.get("frame_count"), int) or result["frame_count"] < 1:
            return "ALSA playback frame acknowledgement is missing"
        expected_peak = 0.9 * action.arguments["volume_percent"] / 100.0
        try:
            digital_peak = float(result["digital_peak_scale"])
        except (KeyError, TypeError, ValueError):
            return "Tone PCM peak acknowledgement is malformed"
        if (
            result.get("volume_mapping") != "pcm_linear_percent"
            or result.get("reference_output_gain_percent") != 100
            or not math.isclose(digital_peak, expected_peak, rel_tol=1e-9, abs_tol=1e-9)
            or not isinstance(result.get("original_output_state"), dict)
        ):
            return "Tone output-gain acknowledgement is missing or invalid"
        loopback = result.get("acoustic_loopback")
        if result.get("acoustic_loopback_detected") is not True or not isinstance(loopback, dict):
            return "Onboard microphone did not observe the requested speaker tone"
        if (
            loopback.get("detected") is not True
            or loopback.get("sensor") != "onboard_usb_microphone"
            or loopback.get("target_frequency_hz") != action.arguments["frequency_hz"]
            or loopback.get("audio_retained") is not False
            or loopback.get("audio_content_returned") is not False
            or not isinstance(loopback.get("baseline"), dict)
            or not isinstance(loopback.get("during_playback"), dict)
            or not isinstance(loopback.get("thresholds"), dict)
        ):
            return "Microphone loopback acknowledgement is missing or invalid"
        try:
            target_gain_db = float(loopback["target_gain_db"])
            observed_target_dbfs = float(loopback["during_playback"]["target_dbfs"])
            observed_prominence_db = float(loopback["during_playback"]["target_prominence_db"])
            minimum_target_dbfs = float(loopback["thresholds"]["minimum_target_dbfs"])
            minimum_gain_db = float(loopback["thresholds"]["minimum_gain_db"])
            minimum_prominence_db = float(loopback["thresholds"]["minimum_prominence_db"])
        except (KeyError, TypeError, ValueError):
            return "Microphone loopback metrics are malformed"
        if not all(
            math.isfinite(value)
            for value in (
                target_gain_db,
                observed_target_dbfs,
                observed_prominence_db,
                minimum_target_dbfs,
                minimum_gain_db,
                minimum_prominence_db,
            )
        ):
            return "Microphone loopback metrics are not finite"
        if (
            minimum_target_dbfs != -45.0
            or minimum_gain_db != 10.0
            or minimum_prominence_db != 8.0
            or observed_target_dbfs < minimum_target_dbfs
            or target_gain_db < minimum_gain_db
            or observed_prominence_db < minimum_prominence_db
        ):
            return "Microphone loopback evidence does not satisfy the fixed thresholds"
        return None

    def _success_result(
        self, action: ActionEnvelope, result: dict[str, Any]
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            policy_decision={
                "allowed": True,
                "policy": self.policy_name,
                "reason": "bounded synthesized tone and mixer-restore contract passed",
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "operation": "PLAY_TONE",
                "device": result["device"],
            },
            driver_ack={
                "acknowledged": True,
                "playback_backend": result["playback_backend"],
                "frame_count": result["frame_count"],
                "sample_rate_hz": result["sample_rate_hz"],
                "mixer_restored": True,
                "volume_mapping": result["volume_mapping"],
                "digital_peak_scale": result["digital_peak_scale"],
                "reference_output_gain_percent": result["reference_output_gain_percent"],
                "acoustic_loopback_detected": True,
            },
            observations=[
                {
                    "kind": "speaker_tone_driver_ack",
                    "frequency_hz": result["frequency_hz"],
                    "duration_sec": result["duration_sec"],
                    "volume_percent": result["volume_percent"],
                    "volume_mapping": result["volume_mapping"],
                    "digital_peak_scale": result["digital_peak_scale"],
                    "reference_output_gain_percent": result["reference_output_gain_percent"],
                    "started_wall_time": result["started_wall_time"],
                    "completed_wall_time": result["completed_wall_time"],
                    "human_hearing_confirmed": False,
                    "acoustic_loopback": result["acoustic_loopback"],
                }
            ],
            verification_result={
                "success": True,
                "predicate": (
                    "the fixed backend accepted single-stage PCM playback, the onboard "
                    "microphone observed the requested frequency above fixed thresholds, and "
                    "the prior output state was restored"
                ),
                "acoustic_output_independently_observed": True,
                "observer": "onboard_usb_microphone",
                "target_gain_db": result["acoustic_loopback"]["target_gain_db"],
                "target_dbfs": result["acoustic_loopback"]["during_playback"]["target_dbfs"],
                "target_prominence_db": result["acoustic_loopback"]["during_playback"][
                    "target_prominence_db"
                ],
                "human_hearing_confirmed": False,
            },
        )


class LimoSpeechExecutor(_LimoFixedWorkerExecutor):
    """Daemon-owned bounded TTS executor with microphone energy verification."""

    capability_id = "limo.speak_text"
    schema = "limo.speech.v1"
    operation = "SPEECH"
    worker_name = "limo_speech_worker.py"
    policy_name = "robot-pack/limo-speech-v1"
    evidence_level = EvidenceLevel.TASK_VERIFIED
    argument_keys = frozenset(
        {
            "schema_version",
            "text",
            "language",
            "volume_percent",
            "rate_wpm",
            "expected_effect",
        }
    )

    def _validate_arguments(self, arguments: dict[str, Any]) -> tuple[str, str] | None:
        code = "LIMO_SPEECH_CONTRACT_INVALID"
        text = arguments.get("text")
        if (
            not isinstance(text, str)
            or not 1 <= len(text) <= 80
            or text != text.strip()
            or any(unicodedata.category(character).startswith("C") for character in text)
        ):
            return code, "Speech text must contain 1-80 bounded characters without controls"
        if arguments.get("language") not in {"cmn", "en"}:
            return code, "Speech language must be cmn or en"
        volume = arguments.get("volume_percent")
        rate = arguments.get("rate_wpm")
        if isinstance(volume, bool) or not isinstance(volume, int) or not 10 <= volume <= 25:
            return code, "volume_percent must be an integer within [10, 25]"
        if isinstance(rate, bool) or not isinstance(rate, int) or not 120 <= rate <= 200:
            return code, "rate_wpm must be an integer within [120, 200]"
        if arguments.get("expected_effect") != {
            "kind": "speaker_speech",
            "playback_required": True,
            "mixer_restore_required": True,
            "microphone_loopback_required": True,
            "content_recognition_required": False,
        }:
            return code, "Expected speech effect is not exact"
        return None

    def _worker_request(self, action: ActionEnvelope) -> dict[str, Any]:
        return {
            "protocol": "rosclaw.limo.worker.v1",
            "operation": "SPEAK_TEXT",
            "schema_version": self.schema,
            "action_id": action.action_id,
            "body_id": action.body_id,
            "body_snapshot_hash": action.body_snapshot_hash,
            "text": action.arguments["text"],
            "language": action.arguments["language"],
            "volume_percent": action.arguments["volume_percent"],
            "rate_wpm": action.arguments["rate_wpm"],
        }

    def _validate_result(self, action: ActionEnvelope, result: dict[str, Any]) -> str | None:
        if (
            result.get("protocol") != "rosclaw.limo.worker.v1"
            or result.get("action_id") != action.action_id
            or result.get("operation") != "SPEAK_TEXT"
            or result.get("accepted") is not True
            or result.get("mixer_restored") is not True
        ):
            return "Worker acknowledgement did not match the speech action"
        if (
            result.get("language") != action.arguments["language"]
            or result.get("volume_percent") != action.arguments["volume_percent"]
            or result.get("rate_wpm") != action.arguments["rate_wpm"]
            or result.get("text_character_count") != len(action.arguments["text"])
            or result.get("text_sha256")
            != "sha256:" + hashlib.sha256(action.arguments["text"].encode("utf-8")).hexdigest()
        ):
            return "Speech synthesis acknowledgement parameters do not match"
        backend = result.get("playback_backend")
        device = result.get("device")
        if (
            backend not in {"alsa", "pulseaudio"}
            or not isinstance(device, str)
            or (backend == "alsa" and not device.startswith("plughw:"))
            or (backend == "pulseaudio" and not device.startswith("pulse:alsa_output.usb-"))
        ):
            return "Speech audio backend acknowledgement is missing or invalid"
        if (
            not isinstance(result.get("frame_count"), int)
            or result["frame_count"] < 1
            or not isinstance(result.get("sample_rate_hz"), int)
            or result["sample_rate_hz"] < 8000
        ):
            return "Speech PCM acknowledgement is missing"
        expected_peak = round(0.9 * action.arguments["volume_percent"] / 100.0, 4)
        try:
            digital_peak = float(result["digital_peak_scale"])
        except (KeyError, TypeError, ValueError):
            return "Speech PCM peak acknowledgement is malformed"
        if (
            result.get("volume_mapping") != "normalized_pcm_linear_percent"
            or result.get("reference_output_gain_percent") != 100
            or not math.isclose(digital_peak, expected_peak, rel_tol=1e-6, abs_tol=1e-6)
            or not isinstance(result.get("original_output_state"), dict)
        ):
            return "Speech output-gain acknowledgement is missing or invalid"
        loopback = result.get("acoustic_loopback")
        if result.get("acoustic_loopback_detected") is not True or not isinstance(loopback, dict):
            return "Onboard microphone did not observe synthesized speech"
        if (
            loopback.get("detected") is not True
            or loopback.get("sensor") != "onboard_usb_microphone"
            or loopback.get("content_recognition_performed") is not False
            or loopback.get("audio_retained") is not False
            or loopback.get("audio_content_returned") is not False
            or not isinstance(loopback.get("baseline"), dict)
            or not isinstance(loopback.get("during_playback"), dict)
            or not isinstance(loopback.get("thresholds"), dict)
        ):
            return "Speech microphone loopback acknowledgement is missing or invalid"
        try:
            rms_gain = float(loopback["rms_gain_db"])
            observed_rms = float(loopback["during_playback"]["rms_dbfs"])
            minimum_rms = float(loopback["thresholds"]["minimum_rms_dbfs"])
            minimum_gain = float(loopback["thresholds"]["minimum_gain_db"])
        except (KeyError, TypeError, ValueError):
            return "Speech microphone metrics are malformed"
        if not all(
            math.isfinite(value) for value in (rms_gain, observed_rms, minimum_rms, minimum_gain)
        ):
            return "Speech microphone metrics are not finite"
        if (
            minimum_rms != -45.0
            or minimum_gain != 8.0
            or observed_rms < minimum_rms
            or rms_gain < minimum_gain
        ):
            return "Speech microphone evidence does not satisfy the fixed thresholds"
        return None

    def _success_result(
        self, action: ActionEnvelope, result: dict[str, Any]
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            policy_decision={
                "allowed": True,
                "policy": self.policy_name,
                "reason": "bounded TTS, acoustic loopback, and mixer-restore contract passed",
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "operation": "SPEAK_TEXT",
                "device": result["device"],
            },
            driver_ack={
                "acknowledged": True,
                "playback_backend": result["playback_backend"],
                "frame_count": result["frame_count"],
                "sample_rate_hz": result["sample_rate_hz"],
                "mixer_restored": True,
                "text_sha256": result["text_sha256"],
                "text_character_count": result["text_character_count"],
                "acoustic_loopback_detected": True,
            },
            observations=[
                {
                    "kind": "speaker_speech_driver_ack",
                    "language": result["language"],
                    "rate_wpm": result["rate_wpm"],
                    "volume_percent": result["volume_percent"],
                    "text_sha256": result["text_sha256"],
                    "text_character_count": result["text_character_count"],
                    "started_wall_time": result["started_wall_time"],
                    "completed_wall_time": result["completed_wall_time"],
                    "human_hearing_confirmed": False,
                    "content_recognition_performed": False,
                    "acoustic_loopback": result["acoustic_loopback"],
                }
            ],
            verification_result={
                "success": True,
                "predicate": (
                    "eSpeak-NG synthesized the approved text, the fixed USB backend completed "
                    "playback, the onboard microphone observed bounded acoustic energy, and "
                    "the prior output state was restored; linguistic content was not recognized"
                ),
                "acoustic_output_independently_observed": True,
                "content_recognition_performed": False,
                "observer": "onboard_usb_microphone",
                "rms_gain_db": result["acoustic_loopback"]["rms_gain_db"],
                "observed_rms_dbfs": result["acoustic_loopback"]["during_playback"]["rms_dbfs"],
                "human_hearing_confirmed": False,
            },
        )


class _LimoFixedWorkerShadowExecutor:
    """Contract-only SHADOW preview for a fixed LIMO worker."""

    def __init__(self, validator: _LimoFixedWorkerExecutor) -> None:
        self.validator = validator

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        contract_error = self.validator._validate_action(action, require_authorization=False)
        if contract_error is not None:
            return _failed_result(contract_error[0], contract_error[1])
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=self.validator.evidence_level,
            policy_decision={
                "allowed": True,
                "policy": self.validator.policy_name,
                "reason": "fixed-operation contract is valid for SHADOW preview",
            },
            authorization_decision={"authorized": False, "required": False},
            dispatch_result={
                "accepted": True,
                "shadow": True,
                "operation": f"VALIDATE_{self.validator.operation}",
                "hardware_dispatched": False,
            },
            driver_ack={"acknowledged": True, "shadow": True},
            observations=[
                {
                    "kind": f"{self.validator.operation.lower()}_contract_preview",
                    "arguments": action.arguments,
                    "hardware_observed": False,
                }
            ],
            verification_result={
                "success": True,
                "predicate": "exact fixed-operation contract validated without worker dispatch",
            },
        )


class LimoNavigationShadowExecutor(_LimoFixedWorkerShadowExecutor):
    def __init__(self, instance: RobotInstanceConfig, *, adapter_source: Path) -> None:
        super().__init__(LimoNavigationExecutor(instance, adapter_source=adapter_source))


class LimoToneShadowExecutor(_LimoFixedWorkerShadowExecutor):
    def __init__(self, instance: RobotInstanceConfig, *, adapter_source: Path) -> None:
        super().__init__(LimoToneExecutor(instance, adapter_source=adapter_source))


class LimoSpeechShadowExecutor(_LimoFixedWorkerShadowExecutor):
    def __init__(self, instance: RobotInstanceConfig, *, adapter_source: Path) -> None:
        super().__init__(LimoSpeechExecutor(instance, adapter_source=adapter_source))


def load_daemon_robot_pack(
    runtime: Any,
    *,
    robot_id: str,
    home: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load exactly one configured Pack instance into a daemon Runtime."""

    resolved_home = resolve_home(str(home) if home is not None else None)
    instances_root = resolved_home / "robots" / "instances"
    config_path = instances_root / f"{robot_id}.yaml"
    if instances_root.is_symlink() or config_path.is_symlink():
        raise RobotPackRuntimeError("Configured Robot Pack instance cannot be a symbolic link")
    if not config_path.is_file():
        return None
    try:
        instance = RobotInstanceConfig.from_path(config_path)
        store = RobotPackStore(resolved_home)
        record, manifest = store.resolve_installed(instance.pack.ref)
    except Exception as exc:  # noqa: BLE001 - configured state must fail daemon startup
        raise RobotPackRuntimeError(f"Configured Robot Pack cannot be loaded: {exc}") from exc

    verification = verify_robot_pack(record.path)
    if not verification.ok or not verification.trusted:
        raise RobotPackRuntimeError(
            "Configured Robot Pack failed trusted integrity verification: "
            + "; ".join(verification.errors)
        )
    if verification.manifest_digest != instance.pack.manifest_digest:
        raise RobotPackRuntimeError("Robot instance Pack digest does not match installed content")
    _validate_instance_contract(instance, manifest, robot_id=robot_id)
    if manifest.safety.agent_southbound_access != "daemon_only":
        raise RobotPackRuntimeError("Robot Pack does not require daemon-only Agent access")
    if manifest.safety.actuation == "forbidden" and any(
        capability.safety_class != "read_only" for capability in manifest.capabilities
    ):
        raise RobotPackRuntimeError("Actuation-forbidden Pack exposes a non-read-only capability")

    try:
        resolver = BodyResolver(workspace=resolved_home, body_id=instance.instance_id)
        body = resolver.get_current_body_yaml()
        effective = resolver.get_effective_body()
    except Exception as exc:  # noqa: BLE001 - configured Body must be complete
        raise RobotPackRuntimeError(f"Configured Robot Pack Body cannot be loaded: {exc}") from exc
    body_matches = bool(
        effective.effective_body_hash == instance.body_snapshot_hash
        and body.body_instance.get("serial_number") == instance.device.serial
        and body.metadata.get("robot_pack_ref") == instance.pack.ref
        and body.metadata.get("robot_pack_manifest_digest") == instance.pack.manifest_digest
        and body.metadata.get("device_stable_uri") == instance.device.stable_uri
        and body.metadata.get("perception_only") is manifest.safety.perception_only
        and body.metadata.get("no_actuation") is (manifest.safety.actuation == "forbidden")
        and sorted(body.capabilities.get("enabled", [])) == sorted(instance.capabilities)
        and body.agent_policy.get("direct_real_robot_execution_allowed") is False
        and body.agent_policy.get("robot_pack_gateway") == "rosclawd"
    )
    if not body_matches:
        raise RobotPackRuntimeError(
            "Configured Robot Pack Body snapshot or device binding no longer matches the instance"
        )

    current_adapter = resolve_adapter_binding(manifest, resolved_home)
    if (
        instance.adapter.status != "installed"
        or current_adapter.status != "installed"
        or not current_adapter.server_name
        or current_adapter.server_name != instance.adapter.server_name
    ):
        raise RobotPackRuntimeError(
            "Configured Robot Pack adapter binding is missing or no longer matches its locked revision"
        )

    loader_contract_ok, loader_contract_errors = validate_daemon_loader_contract(manifest)
    if not loader_contract_ok:
        raise RobotPackRuntimeError("; ".join(loader_contract_errors))

    registered: list[str] = []
    limo_adapter_source: Path | None = None
    if any(capability.id.startswith("limo.") for capability in manifest.capabilities):
        record_entry = InstalledRegistry(home=resolved_home).get(current_adapter.server_name)
        if record_entry is None:
            raise RobotPackRuntimeError("LIMO adapter installation record is missing")
        limo_adapter_source = Path(record_entry.server_dir)
    for capability in manifest.capabilities:
        if capability.id == "camera.capture_rgbd" and capability.safety_class == "read_only":
            executor: Any = RealSenseCaptureExecutor(instance, home=resolved_home)
        elif capability.id == "limo.set_initial_pose" and capability.safety_class == "actuation":
            assert limo_adapter_source is not None
            executor = LimoInitialPoseExecutor(
                instance,
                adapter_source=limo_adapter_source,
            )
            runtime.action_gateway.register_executor(
                capability.id,
                ExecutionMode.SHADOW,
                LimoInitialPoseShadowExecutor(
                    instance,
                    adapter_source=limo_adapter_source,
                ),
            )
            registered.append(f"{capability.id}:SHADOW")
        elif capability.id == "limo.navigate_to_pose" and capability.safety_class == "actuation":
            assert limo_adapter_source is not None
            executor = LimoNavigationExecutor(instance, adapter_source=limo_adapter_source)
            runtime.action_gateway.register_executor(
                capability.id,
                ExecutionMode.SHADOW,
                LimoNavigationShadowExecutor(instance, adapter_source=limo_adapter_source),
            )
            registered.append(f"{capability.id}:SHADOW")
        elif capability.id == "limo.play_tone" and capability.safety_class == "actuation":
            assert limo_adapter_source is not None
            executor = LimoToneExecutor(instance, adapter_source=limo_adapter_source)
            runtime.action_gateway.register_executor(
                capability.id,
                ExecutionMode.SHADOW,
                LimoToneShadowExecutor(instance, adapter_source=limo_adapter_source),
            )
            registered.append(f"{capability.id}:SHADOW")
        elif capability.id == "limo.speak_text" and capability.safety_class == "actuation":
            assert limo_adapter_source is not None
            executor = LimoSpeechExecutor(instance, adapter_source=limo_adapter_source)
            runtime.action_gateway.register_executor(
                capability.id,
                ExecutionMode.SHADOW,
                LimoSpeechShadowExecutor(instance, adapter_source=limo_adapter_source),
            )
            registered.append(f"{capability.id}:SHADOW")
        else:
            raise RobotPackRuntimeError(
                f"No daemon-side executor is implemented for Pack capability {capability.id!r}"
            )
        runtime.action_gateway.register_executor(
            capability.id,
            ExecutionMode.REAL,
            executor,
        )
        registered.append(f"{capability.id}:REAL")

    status = {
        "loaded": True,
        "instance_id": instance.instance_id,
        "pack_ref": record.ref,
        "manifest_digest": record.manifest_digest,
        "signature_status": verification.signature_status,
        "support_tier": record.support_tier,
        "device": {
            "type": instance.device.type,
            "model": instance.device.model,
            "stable_uri": instance.device.stable_uri,
        },
        "registered_executors": registered,
        "safety": instance.safety,
    }
    runtime.robot_pack_status = status
    return status


def _resolve_rgbd_artifacts(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    raw = result.get("artifacts")
    if not isinstance(raw, dict):
        raw = {}
    candidates = {
        "color": raw.get("color") or raw.get("color_path") or raw.get("save_path"),
        "depth": raw.get("depth") or raw.get("depth_path"),
    }
    resolved: dict[str, Path] = {}
    for name, value in candidates.items():
        if not value:
            fallback = output_dir / f"{name}.png"
            value = fallback if fallback.is_file() else None
        if value:
            unresolved = Path(str(value)).expanduser()
            if unresolved.is_symlink():
                continue
            path = unresolved.resolve()
            try:
                path.relative_to(output_dir)
            except ValueError:
                continue
            if path.is_file() and path.stat().st_size > 0:
                resolved[name] = path
    return resolved


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_instance_contract(
    instance: RobotInstanceConfig,
    manifest: RobotPackManifest,
    *,
    robot_id: str,
) -> None:
    if instance.instance_id != robot_id:
        raise RobotPackRuntimeError("Robot instance id does not match its configured file name")

    variant = next(
        (
            candidate
            for candidate in manifest.device.variants
            if instance.device.product_id in candidate.product_ids
        ),
        None,
    )
    model = instance.device.model.casefold()
    parsed_uri = urlparse(instance.device.stable_uri)
    if manifest.discovery.backend == "realsense":
        stable_identity_ok = bool(
            parsed_uri.scheme == "realsense" and parsed_uri.netloc == instance.device.serial
        )
    elif manifest.discovery.backend == "manual":
        stable_identity_ok = bool(
            instance.device.offline_configured
            and instance.device.discovery_backend == "offline_operator_input"
            and parsed_uri.scheme == "ros1"
            and parsed_uri.netloc
        )
    else:
        stable_identity_ok = False
    device_ok = bool(
        instance.device.type == manifest.device.type
        and instance.device.vendor_id in manifest.device.vendor_ids
        and variant is not None
        and any(pattern.casefold() in model for pattern in variant.model_patterns)
        and variant.body_profile == instance.body_profile
        and stable_identity_ok
    )
    if not device_ok:
        raise RobotPackRuntimeError(
            "Robot instance device contract does not match the signed Robot Pack"
        )

    declared_capabilities = sorted(capability.id for capability in manifest.capabilities)
    if sorted(instance.capabilities) != declared_capabilities:
        raise RobotPackRuntimeError(
            "Robot instance capability contract does not match the signed Robot Pack"
        )

    expected_safety = {
        "perception_only": manifest.safety.perception_only,
        "actuation": manifest.safety.actuation,
        "direct_driver_access": manifest.safety.direct_driver_access,
        "agent_southbound_access": manifest.safety.agent_southbound_access,
    }
    if instance.safety != expected_safety:
        raise RobotPackRuntimeError(
            "Robot instance safety contract does not match the signed Robot Pack"
        )
    if instance.adapter.component_id != manifest.adapter.component_id:
        raise RobotPackRuntimeError(
            "Robot instance adapter contract does not match the signed Robot Pack"
        )


def _failed_result(
    code: str,
    message: str,
    *,
    output_dir: Path | None = None,
) -> ActionExecutionResult:
    return ActionExecutionResult(
        final_state=ActionState.FAILED,
        evidence_level=EvidenceLevel.REQUESTED,
        policy_decision={"allowed": False, "reason": message},
        dispatch_result={"accepted": False},
        errors=[{"code": code, "message": message}],
        artifact_directory=str(output_dir) if output_dir else None,
    )


__all__ = [
    "LimoInitialPoseExecutor",
    "LimoInitialPoseShadowExecutor",
    "RealSenseCaptureExecutor",
    "RobotPackRuntimeError",
    "load_daemon_robot_pack",
]
