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
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.robot_pack.instance import configure_robot_instance
from rosclaw.robot_pack.runtime_loader import (
    LimoInitialPoseExecutor,
    LimoInitialPoseShadowExecutor,
    LimoNavigationExecutor,
    LimoNavigationShadowExecutor,
    LimoSpeechExecutor,
    LimoSpeechShadowExecutor,
    LimoToneExecutor,
    LimoToneShadowExecutor,
    RealSenseCaptureExecutor,
    RobotPackRuntimeError,
    load_daemon_robot_pack,
)
from rosclaw.robot_pack.store import RobotPackStore
from rosclaw.robot_pack.verification import _validate_receipt


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def test_limo_worker_environment_passes_only_fixed_pulse_bridge(monkeypatch) -> None:
    monkeypatch.setenv("ROSCLAW_LIMO_PULSE_SERVER", "unix:/run/rosclaw/pulse/native")
    monkeypatch.setenv("UNTRUSTED_AUDIO_DEVICE", "/tmp/device")

    environment = LimoInitialPoseExecutor._worker_environment()

    assert environment["ROSCLAW_LIMO_PULSE_SERVER"] == "unix:/run/rosclaw/pulse/native"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "UNTRUSTED_AUDIO_DEVICE" not in environment


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


def _limo_instance(tmp_path: Path):
    source = tmp_path / "adapter"
    workers = source / "worker"
    workers.mkdir(parents=True)
    for name in (
        "limo_initial_pose_worker.py",
        "limo_navigation_worker.py",
        "limo_speech_worker.py",
        "limo_tone_worker.py",
    ):
        (workers / name).write_text("# fixed worker fixture\n", encoding="utf-8")
    instance = SimpleNamespace(
        instance_id="limo",
        body_snapshot_hash="a" * 64,
        adapter=SimpleNamespace(component_id="limo-ros-mcp"),
    )
    return instance, source


def _limo_action(instance, **argument_overrides) -> ActionEnvelope:
    arguments = {
        "schema_version": "limo.initial-pose.v1",
        "target_pose": {"frame_id": "map", "x": 0.75, "y": -1.25, "yaw": 0.35},
        "route_policy_id": "lab-default",
        "route_policy_hash": "sha256:route",
        "map_id": "lab-map",
        "map_image_hash": "sha256:map",
        "covariance_diagonal": [0.25, 0.25, 0.0, 0.0, 0.0, 0.0685],
        "expected_effect": {
            "kind": "localization_initialized",
            "final_frame": "map",
            "map_to_odom_required": True,
        },
        **argument_overrides,
    }
    return ActionEnvelope(
        action_id="action-limo-initial-pose",
        actor_id="test-agent",
        agent_framework="pytest",
        session_id="session-limo",
        body_id=instance.instance_id,
        body_snapshot_hash=instance.body_snapshot_hash,
        capability_id="limo.set_initial_pose",
        arguments=arguments,
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator",
            approved=True,
            approval_id="permit-limo",
            scopes=["limo.set_initial_pose"],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=15.0,
        ),
    )


def _limo_navigation_action(instance, **argument_overrides) -> ActionEnvelope:
    arguments = {
        "schema_version": "limo.navigation.v2",
        "target_pose": {"frame_id": "map", "x": 0.4, "y": 0.0, "yaw": 0.0},
        "readiness_snapshot_hash": "sha256:readiness",
        "route_policy_id": "lab-default",
        "route_policy_hash": "sha256:route",
        "map_id": "lab-map",
        "map_image_hash": "sha256:map",
        "goal_tolerance": {"xy_m": 0.15, "yaw_rad": 0.2},
        "expected_effect": {
            "kind": "navigate_to_pose",
            "final_frame": "map",
            "stop_required": True,
        },
        **argument_overrides,
    }
    return ActionEnvelope(
        action_id="action-limo-navigation",
        actor_id="test-agent",
        agent_framework="pytest",
        session_id="session-limo-navigation",
        body_id=instance.instance_id,
        body_snapshot_hash=instance.body_snapshot_hash,
        capability_id="limo.navigate_to_pose",
        arguments=arguments,
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator",
            approved=True,
            approval_id="permit-navigation",
            scopes=["limo.navigate_to_pose"],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=60.0,
        ),
    )


def _limo_tone_action(instance, **argument_overrides) -> ActionEnvelope:
    arguments = {
        "schema_version": "limo.tone.v1",
        "frequency_hz": 660,
        "duration_sec": 0.6,
        "volume_percent": 18,
        "expected_effect": {
            "kind": "speaker_tone",
            "playback_required": True,
            "mixer_restore_required": True,
            "microphone_loopback_required": True,
        },
        **argument_overrides,
    }
    return ActionEnvelope(
        action_id="action-limo-tone",
        actor_id="test-agent",
        agent_framework="pytest",
        session_id="session-limo-tone",
        body_id=instance.instance_id,
        body_snapshot_hash=instance.body_snapshot_hash,
        capability_id="limo.play_tone",
        arguments=arguments,
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator",
            approved=True,
            approval_id="permit-tone",
            scopes=["limo.play_tone"],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=10.0,
        ),
    )


def _limo_speech_action(instance, **argument_overrides) -> ActionEnvelope:
    arguments = {
        "schema_version": "limo.speech.v1",
        "text": "你好，我是 ROSClaw LIMO 巡检机器人。",
        "language": "cmn",
        "volume_percent": 18,
        "rate_wpm": 160,
        "expected_effect": {
            "kind": "speaker_speech",
            "playback_required": True,
            "mixer_restore_required": True,
            "microphone_loopback_required": True,
            "content_recognition_required": False,
        },
        **argument_overrides,
    }
    return ActionEnvelope(
        action_id="action-limo-speech",
        actor_id="test-agent",
        agent_framework="pytest",
        session_id="session-limo-speech",
        body_id=instance.instance_id,
        body_snapshot_hash=instance.body_snapshot_hash,
        capability_id="limo.speak_text",
        arguments=arguments,
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator",
            approved=True,
            approval_id="permit-speech",
            scopes=["limo.speak_text"],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=15.0,
        ),
    )


def _limo_tone_loopback_payload() -> dict[str, object]:
    return {
        "detected": True,
        "sensor": "onboard_usb_microphone",
        "capture_device": "plughw:2,0",
        "target_frequency_hz": 660,
        "target_gain_db": 24.0,
        "thresholds": {
            "minimum_target_dbfs": -45.0,
            "minimum_gain_db": 10.0,
            "minimum_prominence_db": 8.0,
        },
        "baseline": {"target_dbfs": -54.0},
        "during_playback": {
            "target_dbfs": -30.0,
            "target_prominence_db": 18.0,
        },
        "audio_retained": False,
        "audio_content_returned": False,
    }


def _limo_speech_loopback_payload() -> dict[str, object]:
    return {
        "detected": True,
        "sensor": "onboard_usb_microphone",
        "capture_device": "plughw:2,0",
        "rms_gain_db": 18.5,
        "thresholds": {"minimum_rms_dbfs": -45.0, "minimum_gain_db": 8.0},
        "baseline": {"sample_count": 1600, "rms_dbfs": -55.0, "peak_dbfs": -44.0},
        "during_playback": {
            "sample_count": 16000,
            "rms_dbfs": -36.5,
            "peak_dbfs": -20.0,
        },
        "content_recognition_performed": False,
        "audio_retained": False,
        "audio_content_returned": False,
    }


def _limo_navigation_worker_payload(
    action: ActionEnvelope, *, movement_expected: bool = True, motion_observed: bool = True
) -> dict[str, object]:
    return {
        "protocol": "rosclaw.limo.worker.v1",
        "ok": True,
        "accepted": True,
        "action_id": action.action_id,
        "operation": "NAVIGATE_TO_POSE",
        "action_server": "/move_base",
        "terminal_state": 3,
        "terminal_text": "Goal reached.",
        "dispatched_wall_time": 10.0,
        "completed_wall_time": 20.0,
        "observed_final_pose": {"frame_id": "map", "x": 0.39, "y": 0.01, "yaw": 0.02},
        "position_error_m": 0.014,
        "yaw_error_rad": 0.02,
        "stopped_odometry": {"linear_speed_mps": 0.0, "angular_speed_radps": 0.0},
        "preflight": {
            "chassis_error_code": 0,
            "map_to_odom": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]},
            "map_to_base": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]},
            "initial_goal_error": {"position_m": 0.4, "yaw_rad": 0.0},
            "active_goal_tolerance": {"xy_m": 0.2, "yaw_rad": 0.15},
            "goal_already_satisfied": not movement_expected,
        },
        "map_to_base_after": {
            "translation": [0.39, 0.01, 0],
            "rotation": [0, 0, 0.01, 1],
        },
        "motion_evidence": {
            "movement_expected": movement_expected,
            "motion_observed": motion_observed,
            "odom_translation_m": 0.39 if motion_observed else 0.0,
            "odom_rotation_rad": 0.02 if motion_observed else 0.0,
            "map_translation_m": 0.39 if motion_observed else 0.0,
            "map_rotation_rad": 0.02 if motion_observed else 0.0,
            "translation_threshold_m": 0.02,
            "rotation_threshold_rad": 0.03,
            "physical_motion_independently_observed": False,
        },
    }


def test_limo_executor_returns_task_verified_receipt(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_action(instance)
    result_payload = {
        "protocol": "rosclaw.limo.worker.v1",
        "ok": True,
        "accepted": True,
        "action_id": action.action_id,
        "operation": "SET_INITIAL_POSE",
        "topic": "/initialpose",
        "subscriber_count": 1,
        "dispatched_wall_time": 10.0,
        "completed_wall_time": 11.0,
        "observed_amcl_pose": {"frame_id": "map", "x": 0.8, "y": -1.2, "yaw": 0.4},
        "map_to_odom": {
            "translation": [0.8, -1.2, 0.0],
            "rotation": [0.0, 0.0, 0.2, 0.98],
        },
    }

    def fake_run(command, **kwargs):
        assert command[0] == "/usr/bin/python2"
        request = __import__("json").loads(kwargs["input"])
        assert request["operation"] == "SET_INITIAL_POSE"
        return SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(result_payload), stderr=""
        )

    monkeypatch.setattr("rosclaw.robot_pack.runtime_loader.subprocess.run", fake_run)
    result = LimoInitialPoseExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert result.dispatch_result["topic"] == "/initialpose"
    assert result.verification_result["success"] is True


def test_limo_executor_rejects_contract_drift_before_worker(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("rosclaw.robot_pack.runtime_loader.subprocess.run", fake_run)
    result = LimoInitialPoseExecutor(instance, adapter_source=source)(
        _limo_action(instance, covariance_diagonal=[1.0] * 6)
    )

    assert result.final_state is ActionState.FAILED
    assert result.errors[0]["code"] == "LIMO_INITIAL_POSE_CONTRACT_INVALID"
    assert called is False


def test_limo_shadow_executor_validates_without_worker_dispatch(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_action(instance)
    action.execution_mode = ExecutionMode.SHADOW
    action.authorization = AuthorizationContext()

    def forbidden_run(*_args, **_kwargs):
        pytest.fail("SHADOW executor must not launch the ROS worker")

    monkeypatch.setattr("rosclaw.robot_pack.runtime_loader.subprocess.run", forbidden_run)
    result = LimoInitialPoseShadowExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.dispatch_result["hardware_dispatched"] is False
    assert result.verification_result["success"] is True


def test_limo_navigation_executor_returns_task_verified_receipt(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_navigation_action(instance)
    payload = _limo_navigation_worker_payload(action)

    def fake_run(command, **kwargs):
        assert command[0] == "/usr/bin/python2"
        request = __import__("json").loads(kwargs["input"])
        assert request["operation"] == "NAVIGATE_TO_POSE"
        return SimpleNamespace(returncode=0, stdout=__import__("json").dumps(payload), stderr="")

    monkeypatch.setattr("rosclaw.robot_pack.runtime_loader.subprocess.run", fake_run)
    result = LimoNavigationExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert result.dispatch_result["terminal_state"] == 3
    assert result.dispatch_result["motion_observed"] is True
    assert result.verification_result["success"] is True


def test_limo_navigation_executor_rejects_success_without_expected_motion(
    tmp_path, monkeypatch
) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_navigation_action(instance)
    payload = _limo_navigation_worker_payload(action, motion_observed=False)
    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        ),
    )

    result = LimoNavigationExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.FAILED
    assert result.errors[0]["code"] == "LIMO_NAVIGATION_VERIFICATION_FAILED"
    assert "movement was not observed" in result.errors[0]["message"]
    assert result.evidence_level is EvidenceLevel.DRIVER_CONFIRMED
    assert result.dispatch_result["accepted"] is True
    assert result.driver_ack["acknowledged"] is True
    assert result.verification_result["success"] is False
    assert result.observations[0]["motion_evidence"]["movement_expected"] is True


def test_limo_navigation_executor_preserves_dispatch_when_pose_misses_tolerance(
    tmp_path, monkeypatch
) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_navigation_action(instance)
    payload = _limo_navigation_worker_payload(action)
    payload["yaw_error_rad"] = 0.08
    action.arguments["goal_tolerance"]["yaw_rad"] = 0.05
    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        ),
    )

    result = LimoNavigationExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.FAILED
    assert result.evidence_level is EvidenceLevel.DRIVER_CONFIRMED
    assert result.dispatch_result == {
        "accepted": True,
        "adapter": instance.adapter.component_id,
        "operation": "NAVIGATE_TO_POSE",
        "action_server": "/move_base",
        "terminal_state": 3,
        "terminal_text": "Goal reached.",
        "movement_expected": True,
        "motion_observed": True,
    }
    assert result.driver_ack == {"acknowledged": True, "dispatched_wall_time": 10.0}
    assert result.observations[0]["kind"] == "navigation_verification_failed"
    assert result.verification_result["success"] is False
    assert result.verification_result["yaw_error_rad"] == 0.08
    assert result.errors[0]["code"] == "LIMO_NAVIGATION_VERIFICATION_FAILED"


def test_limo_navigation_executor_reports_goal_already_satisfied(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_navigation_action(instance)
    payload = _limo_navigation_worker_payload(
        action, movement_expected=False, motion_observed=False
    )
    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        ),
    )

    result = LimoNavigationExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.dispatch_result["movement_expected"] is False
    assert result.observations[0]["kind"] == "navigation_goal_already_satisfied"


def test_limo_tone_executor_returns_microphone_verified_receipt(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_tone_action(instance)
    payload = {
        "protocol": "rosclaw.limo.worker.v1",
        "ok": True,
        "accepted": True,
        "action_id": action.action_id,
        "operation": "PLAY_TONE",
        "device": "pulse:alsa_output.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo",
        "playback_backend": "pulseaudio",
        "frequency_hz": 660,
        "duration_sec": 0.6,
        "volume_percent": 18,
        "sample_rate_hz": 16000,
        "frame_count": 9600,
        "volume_mapping": "pcm_linear_percent",
        "digital_peak_scale": 0.162,
        "reference_output_gain_percent": 100,
        "original_output_state": {"backend": "pulseaudio", "unmuted": True},
        "started_wall_time": 10.0,
        "completed_wall_time": 10.6,
        "mixer_restored": True,
        "acoustic_loopback": _limo_tone_loopback_payload(),
        "acoustic_loopback_detected": True,
    }

    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        ),
    )
    result = LimoToneExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert result.driver_ack["playback_backend"] == "pulseaudio"
    assert result.driver_ack["mixer_restored"] is True
    assert result.verification_result["acoustic_output_independently_observed"] is True
    assert result.verification_result["target_gain_db"] == 24.0


def test_limo_tone_executor_fails_without_microphone_loopback(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_tone_action(instance)
    payload = {
        "protocol": "rosclaw.limo.worker.v1",
        "ok": True,
        "accepted": True,
        "action_id": action.action_id,
        "operation": "PLAY_TONE",
        "device": "plughw:2,0",
        "playback_backend": "alsa",
        "frequency_hz": 660,
        "duration_sec": 0.6,
        "volume_percent": 18,
        "sample_rate_hz": 16000,
        "frame_count": 9600,
        "volume_mapping": "pcm_linear_percent",
        "digital_peak_scale": 0.162,
        "reference_output_gain_percent": 100,
        "original_output_state": {"backend": "alsa", "volume_percent": 0},
        "started_wall_time": 10.0,
        "completed_wall_time": 10.6,
        "mixer_restored": True,
        "acoustic_loopback": {
            **_limo_tone_loopback_payload(),
            "detected": False,
        },
        "acoustic_loopback_detected": False,
    }
    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        ),
    )

    result = LimoToneExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.FAILED
    assert result.errors[0]["code"] == "LIMO_TONE_VERIFICATION_FAILED"
    assert "did not observe" in result.errors[0]["message"]


def test_limo_speech_executor_returns_energy_verified_receipt(tmp_path, monkeypatch) -> None:
    instance, source = _limo_instance(tmp_path)
    action = _limo_speech_action(instance)
    text = action.arguments["text"]
    payload = {
        "protocol": "rosclaw.limo.worker.v1",
        "ok": True,
        "accepted": True,
        "action_id": action.action_id,
        "operation": "SPEAK_TEXT",
        "device": "pulse:alsa_output.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo",
        "playback_backend": "pulseaudio",
        "language": "cmn",
        "rate_wpm": 160,
        "volume_percent": 18,
        "sample_rate_hz": 22050,
        "frame_count": 24000,
        "text_character_count": len(text),
        "text_sha256": "sha256:" + __import__("hashlib").sha256(text.encode()).hexdigest(),
        "volume_mapping": "normalized_pcm_linear_percent",
        "digital_peak_scale": 0.162,
        "reference_output_gain_percent": 100,
        "original_output_state": {"backend": "pulseaudio", "unmuted": True},
        "started_wall_time": 10.0,
        "completed_wall_time": 11.4,
        "mixer_restored": True,
        "acoustic_loopback": _limo_speech_loopback_payload(),
        "acoustic_loopback_detected": True,
    }
    worker_call = {}

    def run_worker(*_args, **kwargs):
        worker_call.update(kwargs)
        return SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        )

    monkeypatch.setattr("rosclaw.robot_pack.runtime_loader.subprocess.run", run_worker)

    result = LimoSpeechExecutor(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert result.driver_ack["text_sha256"] == payload["text_sha256"]
    assert result.verification_result["acoustic_output_independently_observed"] is True
    assert result.verification_result["content_recognition_performed"] is False
    assert result.verification_result["rms_gain_db"] == 18.5
    assert worker_call["timeout"] == 90.0


@pytest.mark.parametrize(
    ("executor_type", "action_factory", "override", "code"),
    [
        (
            LimoNavigationExecutor,
            _limo_navigation_action,
            {"goal_tolerance": {"xy_m": 2.0, "yaw_rad": 0.2}},
            "LIMO_NAVIGATION_CONTRACT_INVALID",
        ),
        (
            LimoToneExecutor,
            _limo_tone_action,
            {"volume_percent": 80},
            "LIMO_TONE_CONTRACT_INVALID",
        ),
        (
            LimoSpeechExecutor,
            _limo_speech_action,
            {"volume_percent": 80},
            "LIMO_SPEECH_CONTRACT_INVALID",
        ),
    ],
)
def test_limo_fixed_workers_reject_contract_drift_before_dispatch(
    tmp_path, monkeypatch, executor_type, action_factory, override, code
) -> None:
    instance, source = _limo_instance(tmp_path)
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("rosclaw.robot_pack.runtime_loader.subprocess.run", fake_run)
    result = executor_type(instance, adapter_source=source)(action_factory(instance, **override))

    assert result.final_state is ActionState.FAILED
    assert result.errors[0]["code"] == code
    assert called is False


@pytest.mark.parametrize(
    ("shadow_type", "action_factory"),
    [
        (LimoNavigationShadowExecutor, _limo_navigation_action),
        (LimoToneShadowExecutor, _limo_tone_action),
        (LimoSpeechShadowExecutor, _limo_speech_action),
    ],
)
def test_limo_fixed_worker_shadows_never_dispatch(
    tmp_path, monkeypatch, shadow_type, action_factory
) -> None:
    instance, source = _limo_instance(tmp_path)
    action = action_factory(instance)
    action.execution_mode = ExecutionMode.SHADOW
    action.authorization = AuthorizationContext()
    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("SHADOW must not launch a worker"),
    )

    result = shadow_type(instance, adapter_source=source)(action)

    assert result.final_state is ActionState.COMPLETED
    assert result.dispatch_result["hardware_dispatched"] is False


def test_daemon_loader_registers_limo_initial_pose_executor(tmp_path) -> None:
    home = tmp_path / "limo-home"
    RobotPackStore(home).install("limo")
    InstalledRegistry(home=home).add(
        InstalledRecord(
            server_name="limo-ros-mcp",
            manifest_id="limo-ros-mcp",
            name="limo-ros-mcp",
            version="0.8.8",
            installed_at="2026-07-30T00:00:00Z",
            artifact_type="test",
            server_dir=str(home / "mcp"),
            extra={"repo_commit": "bcc1abfba2e0a450f9c4153cc6fe99b279b7bc00"},
        )
    )
    configure_robot_instance(
        "limo",
        home=home,
        instance_id="limo",
        serial="LIMO-LAB-01",
        model="LIMO",
        stable_uri="ros1://localhost/limo",
        allow_offline=True,
        switch_active=True,
    )
    runtime = SimpleNamespace(action_gateway=_Gateway())

    status = load_daemon_robot_pack(runtime, robot_id="limo", home=home)

    assert status is not None
    assert status["pack_ref"].endswith("limo-ros1@0.1.30")
    assert status["registered_executors"] == [
        "limo.set_initial_pose:SHADOW",
        "limo.set_initial_pose:REAL",
        "limo.navigate_to_pose:SHADOW",
        "limo.navigate_to_pose:REAL",
        "limo.play_tone:SHADOW",
        "limo.play_tone:REAL",
        "limo.speak_text:SHADOW",
        "limo.speak_text:REAL",
    ]
    assert isinstance(runtime.action_gateway.registrations[0][2], LimoInitialPoseShadowExecutor)
    assert isinstance(runtime.action_gateway.registrations[1][2], LimoInitialPoseExecutor)
    assert isinstance(runtime.action_gateway.registrations[2][2], LimoNavigationShadowExecutor)
    assert isinstance(runtime.action_gateway.registrations[3][2], LimoNavigationExecutor)
    assert isinstance(runtime.action_gateway.registrations[4][2], LimoToneShadowExecutor)
    assert isinstance(runtime.action_gateway.registrations[5][2], LimoToneExecutor)
    assert isinstance(runtime.action_gateway.registrations[6][2], LimoSpeechShadowExecutor)
    assert isinstance(runtime.action_gateway.registrations[7][2], LimoSpeechExecutor)


def test_signed_limo_pack_runs_tone_through_daemon_permit_and_receipt(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "limo-tone-home"
    RobotPackStore(home).install("limo")
    adapter_source = home / "mcp"
    workers = adapter_source / "worker"
    workers.mkdir(parents=True)
    for name in (
        "limo_initial_pose_worker.py",
        "limo_navigation_worker.py",
        "limo_speech_worker.py",
        "limo_tone_worker.py",
    ):
        (workers / name).write_text("# fixed worker fixture\n", encoding="utf-8")
    InstalledRegistry(home=home).add(
        InstalledRecord(
            server_name="limo-ros-mcp",
            manifest_id="limo-ros-mcp",
            name="limo-ros-mcp",
            version="0.8.8",
            installed_at="2026-07-31T00:00:00Z",
            artifact_type="test",
            server_dir=str(adapter_source),
            extra={"repo_commit": "bcc1abfba2e0a450f9c4153cc6fe99b279b7bc00"},
        )
    )
    instance = configure_robot_instance(
        "limo",
        home=home,
        instance_id="limo",
        serial="LIMO-LAB-01",
        model="LIMO",
        stable_uri="ros1://localhost/limo",
        allow_offline=True,
        switch_active=True,
    )[0]
    action = _limo_tone_action(instance)
    payload = {
        "protocol": "rosclaw.limo.worker.v1",
        "ok": True,
        "accepted": True,
        "action_id": action.action_id,
        "operation": "PLAY_TONE",
        "device": "plughw:2,0",
        "playback_backend": "alsa",
        "frequency_hz": 660,
        "duration_sec": 0.6,
        "volume_percent": 18,
        "sample_rate_hz": 16000,
        "frame_count": 9600,
        "volume_mapping": "pcm_linear_percent",
        "digital_peak_scale": 0.162,
        "reference_output_gain_percent": 100,
        "original_output_state": {"backend": "alsa", "volume_percent": 0},
        "started_wall_time": 10.0,
        "completed_wall_time": 10.6,
        "mixer_restored": True,
        "acoustic_loopback": _limo_tone_loopback_payload(),
        "acoustic_loopback_detected": True,
    }
    monkeypatch.setattr(
        "rosclaw.robot_pack.runtime_loader.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=__import__("json").dumps(payload), stderr=""
        ),
    )
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
    assert load_daemon_robot_pack(runtime, robot_id=instance.instance_id, home=home) is not None
    peer = PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
    permits = PermitAuthority()
    permits.register(
        ExecutionPermit(
            permit_id="permit-tone",
            principal_id="operator",
            peer_uid=peer.uid,
            body_id=instance.instance_id,
            body_snapshot_hash=instance.body_snapshot_hash,
            capabilities=("limo.play_tone",),
            action_intent_hash=action_intent_hash(action),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )
    )
    service = DaemonControlPlane(runtime=runtime, permits=permits)
    service.start()
    try:
        service.arm_runtime("Offline signed LIMO Pack integration test", peer)
        service.request_action(action, peer)
        deadline = time.monotonic() + 2.0
        while True:
            status = service.get_action_status(action.action_id, peer)
            if status["state"] == "FINISHED":
                break
            if time.monotonic() >= deadline:
                pytest.fail("daemon did not finish the LIMO tone action")
            time.sleep(0.01)
    finally:
        service.close()

    receipt = status["receipt"]
    assert receipt["final_state"] == "COMPLETED", receipt.get("errors")
    assert receipt["evidence_level"] == "TASK_VERIFIED"
    assert receipt["dispatch_result"]["operation"] == "PLAY_TONE"
    assert receipt["driver_ack"]["mixer_restored"] is True
    assert receipt["verification_result"]["acoustic_output_independently_observed"] is True


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
