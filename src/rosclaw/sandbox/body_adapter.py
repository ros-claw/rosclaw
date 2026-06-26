"""Sandbox body adapter — consumes EffectiveBody for simulation config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.schema import EffectiveBody


class SandboxBodyAdapter:
    """Lightweight adapter that derives sandbox configuration from EffectiveBody.

    The adapter never reads ``body.yaml`` directly; it only inspects the compiled
    Effective Body Model so that the sandbox and the rest of ROSClaw share the
    same body truth.
    """

    def __init__(self, body: EffectiveBody):
        self._body = body
        self.effective_body_hash = body.effective_body_hash
        self.eurdf_uri = body.eurdf_uri
        self.body_instance_id = body.body_instance_id

    @classmethod
    def from_effective_body(cls, body: EffectiveBody) -> SandboxBodyAdapter:
        return cls(body)

    def to_mujoco_config(self) -> dict[str, Any]:
        """Return a MuJoCo-compatible config dict derived from the body."""
        safety = self._body.safety or {}
        global_limits = safety.get("global_limits") or safety.get("safety_limits") or {}
        return {
            "effective_body_hash": self.effective_body_hash,
            "eurdf_uri": self.eurdf_uri,
            "urdf_path": self._body.source_trace.get("urdf", ""),
            "mesh_dirs": [],
            "disabled_actuators": self._disabled_actuators(),
            "joint_limits": self._joint_limits(),
            "safety": {
                "max_linear_speed_mps": global_limits.get("max_linear_speed_mps"),
                "max_angular_speed_radps": global_limits.get("max_angular_speed_radps"),
                "max_joint_speed_scale": global_limits.get("max_joint_speed_scale", 0.5),
                "max_joint_torque_scale": global_limits.get("max_joint_torque_scale", 0.5),
                "require_estop_ready": global_limits.get("require_estop_ready", True),
                "require_sandbox_validation": global_limits.get("require_sandbox_validation", True),
            },
            "collision": {
                "self_collision_check": True,
                "policy": "strict",
            },
            "calibration_offsets": self._calibration_offsets(),
        }

    def to_isaac_config(self) -> dict[str, Any]:
        """Return an Isaac Sim compatible config dict derived from the body."""
        cfg = self.to_mujoco_config()
        cfg["engine"] = "isaac"
        cfg["isaac_specific"] = {
            "stage_unit_scale": 1.0,
            "physics_dt": 0.008333,
            "render_dt": 0.033333,
        }
        return cfg

    def write_configs(self, output_dir: Path) -> dict[str, Path]:
        """Write engine config files under ``output_dir`` and return paths."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)
        mujoco_path = output_dir / "mujoco.config.json"
        isaac_path = output_dir / "isaac.config.json"
        mujoco_path.write_text(json.dumps(self.to_mujoco_config(), indent=2), encoding="utf-8")
        isaac_path.write_text(json.dumps(self.to_isaac_config(), indent=2), encoding="utf-8")
        return {"mujoco": mujoco_path, "isaac": isaac_path}

    def write_configs_yaml(self, output_dir: Path) -> dict[str, Path]:
        """Write engine config files as YAML under ``output_dir``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        mujoco_path = output_dir / "mujoco.config.yaml"
        isaac_path = output_dir / "isaac.config.yaml"
        mujoco_path.write_text(yaml.safe_dump(self.to_mujoco_config(), sort_keys=False), encoding="utf-8")
        isaac_path.write_text(yaml.safe_dump(self.to_isaac_config(), sort_keys=False), encoding="utf-8")
        return {"mujoco": mujoco_path, "isaac": isaac_path}

    def _disabled_actuators(self) -> list[str]:
        disabled: list[str] = []
        for name, actuator in (self._body.actuators or {}).items():
            if actuator.get("status") in ("unavailable", "disabled", "faulty"):
                disabled.append(name)
        # Capabilities marked blocked/disabled in the body also disable actuators.
        blocked = set(self._body.capabilities.get("blocked", []))
        for cap in blocked:
            if cap not in disabled:
                disabled.append(cap)
        return sorted(disabled)

    def _joint_limits(self) -> dict[str, dict[str, float]]:
        limits: dict[str, dict[str, float]] = {}
        for joint_name, joint in (self._body.joints or {}).items():
            if isinstance(joint, dict):
                limits[str(joint_name)] = {
                    "lower": joint.get("lower", -3.14),
                    "upper": joint.get("upper", 3.14),
                    "velocity": joint.get("velocity", 1.0),
                    "effort": joint.get("effort", 10.0),
                }
        return limits

    def _calibration_offsets(self) -> dict[str, Any]:
        """Extract calibration offsets from effective body runtime overlay."""
        overlay = self._body.runtime_state or {}
        calibration = overlay.get("calibration", {})
        offsets: dict[str, Any] = {
            "joint_offsets": calibration.get("joint_offsets", {}),
            "sensor_extrinsics": calibration.get("sensor_extrinsics", {}),
            "sensor_intrinsics": calibration.get("sensor_intrinsics", {}),
            "tool_frames": calibration.get("tool_frames", {}),
        }
        return offsets
