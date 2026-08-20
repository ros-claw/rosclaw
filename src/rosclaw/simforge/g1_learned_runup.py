"""SIM-only learned G1 approach controller used by the free-kick showcase."""

from __future__ import annotations

import collections
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    hash_bytes,
    hash_json,
)

_BALANCE_NAME = "GR00T-WholeBodyControl-Balance.onnx"
_WALK_NAME = "GR00T-WholeBodyControl-Walk.onnx"


@dataclass(frozen=True)
class G1LearnedRunupConfig:
    """Frozen high-speed approach and velocity-matched plant schedule."""

    start_x_m: float = -3.484
    start_y_m: float = 0.0
    settle_duration_sec: float = 0.5
    run_duration_sec: float = 3.0
    brake_duration_sec: float = 1.2
    plant_duration_sec: float = 0.2
    forward_velocity_command_mps: float = 1.30
    lateral_velocity_command_mps: float = 0.25
    control_dt_sec: float = 0.02
    physics_dt_sec: float = 0.002
    schema_version: str = "rosclaw.simforge.g1_learned_runup_config.v2"

    def __post_init__(self) -> None:
        values = (
            self.start_x_m,
            self.start_y_m,
            self.settle_duration_sec,
            self.run_duration_sec,
            self.brake_duration_sec,
            self.plant_duration_sec,
            self.forward_velocity_command_mps,
            self.lateral_velocity_command_mps,
            self.control_dt_sec,
            self.physics_dt_sec,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("learned run-up config must be finite")
        if not -6.0 <= self.start_x_m <= -2.0:
            raise ValueError("learned run-up start must be in [-6, -2] m")
        if not -2.0 <= self.start_y_m <= 2.0:
            raise ValueError("learned run-up lateral start must be in [-2, 2] m")
        if not 0.3 <= self.settle_duration_sec <= 1.0:
            raise ValueError("learned run-up settle duration must be in [0.3, 1.0] s")
        if not 2.0 <= self.run_duration_sec <= 8.0:
            raise ValueError("learned run-up duration must be in [2.0, 8.0] s")
        if not 0.8 <= self.brake_duration_sec <= 2.0:
            raise ValueError("learned run-up plant duration must be in [0.8, 2.0] s")
        if not 0.1 <= self.plant_duration_sec < self.brake_duration_sec:
            raise ValueError("plant hold must be positive and shorter than braking")
        if not 0.8 <= self.forward_velocity_command_mps <= 1.5:
            raise ValueError("learned run-up velocity command must be in [0.8, 1.5] m/s")
        if not -0.5 <= self.lateral_velocity_command_mps <= 0.5:
            raise ValueError("learned run-up lateral command must be in [-0.5, 0.5] m/s")
        ratio = self.control_dt_sec / self.physics_dt_sec
        if not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError("control dt must be an integer multiple of physics dt")

    @property
    def total_duration_sec(self) -> float:
        return self.settle_duration_sec + self.run_duration_sec + self.brake_duration_sec

    @property
    def config_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1LearnedGaitQualification:
    eligible: bool
    balance_policy_hash: str
    walk_policy_hash: str
    input_size: int
    output_size: int
    errors: tuple[str, ...]
    source: str = "NVlabs/GR00T-WholeBodyControl"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_learned_gait_qualification.v1"

    @property
    def qualification_hash(self) -> str:
        return hash_json(asdict(self))

    def require_eligible(self) -> None:
        if not self.eligible:
            raise ValueError("learned gait assets are not eligible: " + "; ".join(self.errors))


def qualify_g1_learned_gait(policy_root: Path) -> G1LearnedGaitQualification:
    """Fail closed unless both pinned policies expose the frozen 516->15 contract."""

    import onnxruntime

    root = policy_root.expanduser().resolve()
    balance = root / _BALANCE_NAME
    walk = root / _WALK_NAME
    errors: list[str] = []
    for path in (balance, walk):
        if not path.is_file():
            errors.append(f"missing_policy={path.name}")
    input_size = 0
    output_size = 0
    if not errors:
        for path in (balance, walk):
            try:
                session = onnxruntime.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                shape_in = int(session.get_inputs()[0].shape[-1])
                shape_out = int(session.get_outputs()[0].shape[-1])
                input_size, output_size = shape_in, shape_out
                if (shape_in, shape_out) != (516, 15):
                    errors.append(
                        f"policy_shape={path.name}:{shape_in}->{shape_out},expected=516->15"
                    )
            except Exception as exc:  # noqa: BLE001 - qualification reports dependency errors
                errors.append(f"policy_load={path.name}:{type(exc).__name__}:{exc}")
    zero = "sha256:" + "0" * 64
    return G1LearnedGaitQualification(
        eligible=not errors,
        balance_policy_hash=hash_bytes(balance.read_bytes()) if balance.is_file() else zero,
        walk_policy_hash=hash_bytes(walk.read_bytes()) if walk.is_file() else zero,
        input_size=input_size,
        output_size=output_size,
        errors=tuple(errors),
    )


class G1LearnedRunupController:
    """Stateful adapter for the pinned history-based locomotion policies."""

    default_lower = np.asarray(
        (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0),
        dtype=np.float32,
    )
    lower_kp = np.asarray(
        (150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250),
        dtype=np.float64,
    )
    lower_kd = np.asarray(
        (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5),
        dtype=np.float64,
    )
    arm_kp = np.asarray((100, 100, 50, 50, 20, 20, 20) * 2, dtype=np.float64)
    arm_kd = np.asarray((2, 2, 2, 2, 1, 1, 1) * 2, dtype=np.float64)

    def __init__(self, policy_root: Path) -> None:
        import onnxruntime

        self.qualification = qualify_g1_learned_gait(policy_root)
        self.qualification.require_eligible()
        root = policy_root.expanduser().resolve()
        self._balance = onnxruntime.InferenceSession(
            str(root / _BALANCE_NAME), providers=["CPUExecutionProvider"]
        )
        self._walk = onnxruntime.InferenceSession(
            str(root / _WALK_NAME), providers=["CPUExecutionProvider"]
        )
        self._history: collections.deque[np.ndarray] = collections.deque(maxlen=6)
        # Preserve the training/deployment adapter's float32 action recurrence.
        # This history policy is sensitive to silently widening its recurrent
        # target state before it meets the float64 MuJoCo state.
        self.action = np.zeros(15, dtype=np.float32)
        self.target = self.default_lower.copy()
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        self._history.extend(np.zeros(86, dtype=np.float32) for _ in range(6))
        self.action[:] = 0.0
        self.target = self.default_lower.copy()

    def torque(self, data: Any) -> np.ndarray:
        torque = np.zeros(29, dtype=np.float64)
        torque[:15] = (self.target - data.qpos[7:22]) * self.lower_kp - data.qvel[
            6:21
        ] * self.lower_kd
        torque[15:] = -data.qpos[22:36] * self.arm_kp - data.qvel[21:35] * self.arm_kd
        limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        return np.clip(torque, -limits, limits)

    def update(self, data: Any, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("learned gait command must be a finite xyz velocity vector")
        observation = np.zeros(86, dtype=np.float32)
        observation[0:3] = command * np.asarray((2.0, 2.0, 0.5), dtype=np.float32)
        observation[3] = 0.74
        observation[7:10] = np.asarray(data.qvel[3:6], dtype=np.float32) * 0.5
        observation[10:13] = _gravity_in_body(np.asarray(data.qpos[3:7], dtype=np.float64))
        padded_default = np.zeros(29, dtype=np.float64)
        padded_default[:15] = self.default_lower
        observation[13:42] = data.qpos[7:36] - padded_default
        observation[42:71] = data.qvel[6:35] * 0.05
        observation[71:86] = self.action
        self._history.append(observation)
        stacked = np.concatenate(self._history).astype(np.float32, copy=False)[None, :]
        session = self._balance if float(np.linalg.norm(command)) <= 0.05 else self._walk
        input_name = session.get_inputs()[0].name
        self.action = np.asarray(session.run(None, {input_name: stacked})[0][0], dtype=np.float32)
        if self.action.shape != (15,) or not np.all(np.isfinite(self.action)):
            raise FloatingPointError("learned gait policy emitted an invalid action")
        self.target = np.asarray(
            self.default_lower + np.float32(0.25) * self.action,
            dtype=np.float32,
        )
        return self.target.copy()


def _gravity_in_body(quaternion_wxyz: np.ndarray) -> np.ndarray:
    import mujoco

    conjugate = quaternion_wxyz.copy()
    conjugate[1:] *= -1.0
    value = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(value, np.asarray((0.0, 0.0, -1.0)), conjugate)
    return value.astype(np.float32)


__all__ = [
    "G1LearnedGaitQualification",
    "G1LearnedRunupConfig",
    "G1LearnedRunupController",
    "qualify_g1_learned_gait",
]
