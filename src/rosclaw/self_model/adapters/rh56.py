"""RH56 hand self adapter + analytical forward prior (v3 §8.3/§8.4, PR-PE-4).

The RH56 forward prior predicts what a gesture command does next — NOT a
pelvis:

* 下一时刻 joint position（一阶位置响应，每关节独立时间常数）
* tracking error 与 time-to-reach（同模型的直接读出）
* temperature delta（简化热 RC 模型：动作发热 vs 环境散热）

Bounded residual 只挂 shadow（本适配器内置 analytical_only=True 直到
shadow 数据足够——v3 §8.4：Analytical Prior + Bounded Residual，
Residual 只在 Shadow 学习）.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

from ..protocols import ForwardModelProtocol, SelfPrediction, SelfStateAdapter

JOINTS = ("little", "ring", "middle", "index", "thumb", "thumb_rot")

# Per-joint first-order time constants (seconds) — calibrated from REAL
# gesture telemetry (session prac_20260730T024004Z_3260ea, 2026-07-30:
# median per-joint tau over measured motion windows; little/ring share
# the middle estimate until their own evidence exists).  Recalibrate by
# residual evidence only — never G1 parameters, never guesses (v3 §8.5).
DEFAULT_TAU_S: dict[str, float] = {
    "little": 0.52,
    "ring": 0.52,
    "middle": 0.52,
    "index": 0.28,
    "thumb": 0.49,
    "thumb_rot": 0.33,
}

# Simplified thermal RC: one gesture's holding-current heat input at
# force_set 300, cooling toward ambient with ~20 min time constant
# (measured: idle floor 45 °C, coast-recovery minutes–hour).
THERMAL_HEAT_PER_GESTURE_C = 0.06
THERMAL_COOL_TAU_S = 1200.0


@dataclass(frozen=True)
class RH56SelfState:
    """§8.3 RH56SelfState blocks."""

    identity: dict[str, Any]
    kinematics: dict[str, Any]
    interaction: dict[str, Any]
    health: dict[str, Any]
    perception: dict[str, Any]
    task: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "kinematics": self.kinematics,
            "interaction": self.interaction,
            "health": self.health,
            "perception": self.perception,
            "task": self.task,
        }


class RH56HandSelfAdapter(SelfStateAdapter):
    """One RH56 hand (left or right) as a self-model body."""

    def __init__(
        self,
        body_id: str,
        *,
        firmware: str = "unmeasured",
        calibration_hash: str = "unmeasured",
        tau_s: dict[str, float] | None = None,
    ) -> None:
        self._body_id = body_id
        self._identity = {
            "body_id": body_id,
            "firmware": firmware,
            "calibration_hash": calibration_hash,
            "transport_profile": "ftdi_serial_460800",
        }
        self._tau = dict(tau_s or DEFAULT_TAU_S)

    def body_id(self) -> str:
        return self._body_id

    def body_hash(self) -> str:
        return canonical_hash(self._identity, prefix="body")

    def current_state(self) -> dict[str, Any]:
        return self._identity

    def health_channels(self) -> dict[str, float]:
        return {}

    def self_state(
        self,
        telemetry: dict[str, Any],
        *,
        target: dict[str, float] | None = None,
        task: dict[str, Any] | None = None,
        perception: dict[str, Any] | None = None,
    ) -> RH56SelfState:
        """Build the full §8.3 state from one telemetry reading."""
        angles = telemetry.get("angle_actual") or {}
        angle_set = telemetry.get("angle_set") or target or {}
        tracking = {
            joint: float(angle_set.get(joint, 0.0)) - float(angles.get(joint, 0.0))
            for joint in JOINTS
            if joint in angles or joint in angle_set
        }
        temps = telemetry.get("temperature_c") or {}
        return RH56SelfState(
            identity=self._identity,
            kinematics={
                "joint_position": angles,
                "target_position": angle_set,
                "tracking_error": tracking,
                "joint_velocity_estimate": telemetry.get("velocity_estimate") or {},
            },
            interaction={
                "force": telemetry.get("force_act") or {},
                "current_ma": telemetry.get("current_ma") or {},
                "contact_probability": telemetry.get("contact_probability"),
            },
            health={
                "temperature": temps,
                "temperature_max": max(
                    (float(v) for v in temps.values() if isinstance(v, (int, float)) and v > 0),
                    default=None,
                ),
                "status": telemetry.get("status"),
                "transport_latency_ms": telemetry.get("transport_latency_ms"),
            },
            perception=perception or {},
            task=task or {},
        )


class RH56ForwardPrior(ForwardModelProtocol):
    """Analytical forward prior for one RH56 hand (§8.4).

    predict(state, action):
      state  = {"pos_<joint>": raw, "temp_max": °C, "ambient_c": °C?}
      action = {"target_<joint>": raw, "dt_s": seconds, "gestures": n?}

    channels: next joint positions, per-joint |tracking error| after dt,
    time-to-reach (95% settle), temperature delta."""

    def __init__(
        self,
        body_id: str,
        body_hash: str,
        *,
        tau_s: dict[str, float] | None = None,
        heat_per_gesture_c: float = THERMAL_HEAT_PER_GESTURE_C,
        cool_tau_s: float = THERMAL_COOL_TAU_S,
    ) -> None:
        self._body_id = body_id
        self._body_hash = body_hash
        self._tau = dict(tau_s or DEFAULT_TAU_S)
        self._heat = heat_per_gesture_c
        self._cool_tau = cool_tau_s

    @property
    def model_hash(self) -> str:
        return canonical_hash(
            {
                "kind": "rh56_forward_prior",
                "body_id": self._body_id,
                "body_hash": self._body_hash,
                "tau_s": self._tau,
                "heat_per_gesture_c": self._heat,
                "cool_tau_s": self._cool_tau,
            },
            prefix="fwm",
        )

    def expected_body_hash(self) -> str:
        return self._body_hash

    def predict(self, state: dict[str, float], action: dict[str, float]) -> SelfPrediction:
        dt = float(action.get("dt_s") or 0.1)
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("dt_s must be positive and finite")
        channels: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for joint in JOINTS:
            pos_key = f"pos_{joint}"
            if pos_key not in state:
                continue
            pos = float(state[pos_key])
            target = float(action.get(f"target_{joint}", pos))
            tau = self._tau.get(joint, 0.18)
            velocity = state.get(f"vel_{joint}")
            # Two-mode prior (real-telemetry driven, 2026-07-30): the RH56
            # servo approaches its command register ONLY during gesture
            # execution windows; elsewhere the register holds a STALE
            # value while the hand holds position.  Predicting toward a
            # stale register is 5× worse than predicting hold (measured:
            # RMSE 899 vs 173).  Motion evidence (measured velocity
            # toward the target) is what separates the modes — never the
            # register alone.
            moving_toward = (
                isinstance(velocity, (int, float))
                and abs(velocity) > 30.0
                and (target - pos) * velocity > 0
            )
            if moving_toward:
                residual = (pos - target) * math.exp(-dt / tau)
                channels[f"next_pos_{joint}"] = target + residual
                channels[f"tracking_error_{joint}"] = abs(residual)
                if abs(pos - target) > 1.0:
                    channels[f"time_to_reach_{joint}"] = tau * math.log(20.0)
                else:
                    channels[f"time_to_reach_{joint}"] = 0.0
            else:
                channels[f"next_pos_{joint}"] = pos
                channels[f"tracking_error_{joint}"] = abs(target - pos)
                channels[f"time_to_reach_{joint}"] = 0.0
            uncertainty[f"next_pos_{joint}"] = 15.0 if moving_toward else 8.0
        temp = state.get("temp_max")
        gestures = float(action.get("gestures", 1.0))
        if isinstance(temp, (int, float)):
            ambient = float(state.get("ambient_c", 25.0))
            heat = self._heat * gestures
            cool = (float(temp) - ambient) * (1.0 - math.exp(-dt / self._cool_tau))
            channels["temp_delta"] = heat - cool
            uncertainty["temp_delta"] = 0.5
        return SelfPrediction(
            channels=channels,
            uncertainty=uncertainty,
            model_hash=self.model_hash,
            analytical_only=True,
        )


class BodyHashMismatchError(RuntimeError):
    """Loading a model against a different body hash (v3 §8.7)."""


def bind_model_to_body(
    model: ForwardModelProtocol, adapter: SelfStateAdapter
) -> ForwardModelProtocol:
    """The only honest way to load a forward model: hash must match."""
    if model.expected_body_hash() != adapter.body_hash():
        raise BodyHashMismatchError(
            f"model bound to {model.expected_body_hash()} cannot load on "
            f"body {adapter.body_id()} ({adapter.body_hash()})"
        )
    return model
