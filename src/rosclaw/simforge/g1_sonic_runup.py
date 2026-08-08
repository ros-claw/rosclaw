"""SIM-only SONIC full-body approach policy for the G1 football loop.

The planner produces a 30 Hz whole-body motion reference.  The low-latency
SONIC encoder/decoder then closes the loop from ten frames of proprioception
and emits all 29 normalized joint actions at 50 Hz.  A 500 Hz PD loop is the
only layer that turns those actions into MuJoCo torques.

This module deliberately keeps qualification, reference generation and the
closed-loop policy in one auditable boundary.  A pretty kinematic planner
rollout is never reported as a physics result.
"""

from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    hash_json,
)

_PLANNER = "planner_sonic.onnx"
_ENCODER = "low_latency/model_encoder.onnx"
_DECODER = "low_latency/model_decoder.onnx"
_OBSERVATION_CONFIG = "low_latency/observation_config.yaml"

# The mappings are copied from GEAR-SONIC's pinned policy_parameters.hpp.
# ``mujoco_to_isaaclab`` indexes MuJoCo-order arrays to produce IsaacLab order;
# ``isaaclab_to_mujoco`` does the inverse for decoder actions.
MUJOCO_TO_ISAACLAB = np.asarray(
    (
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ),
    dtype=np.int64,
)
ISAACLAB_TO_MUJOCO = np.asarray(
    (
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ),
    dtype=np.int64,
)


@dataclass(frozen=True)
class G1SonicQualification:
    """Pinned SONIC asset and tensor-contract qualification."""

    eligible: bool
    planner_hash: str
    encoder_hash: str
    decoder_hash: str
    observation_config_hash: str
    planner_input_count: int
    encoder_input_size: int
    decoder_input_size: int
    decoder_output_size: int
    errors: tuple[str, ...]
    source: str = "NVlabs/GR00T-WholeBodyControl:GEAR-SONIC"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_sonic_qualification.v1"

    @property
    def qualification_hash(self) -> str:
        return hash_json(asdict(self))

    def require_eligible(self) -> None:
        if not self.eligible:
            raise ValueError("SONIC assets are not eligible: " + "; ".join(self.errors))


@dataclass(frozen=True)
class G1SonicRunupConfig:
    """Frozen whole-body run/decelerate schedule used before the kick splice."""

    run_velocity_mps: float = 1.50
    brake_velocity_mps: float = 0.55
    execution_duration_sec: float = 3.40
    policy_dt_sec: float = 0.02
    physics_dt_sec: float = 0.002
    gain_scale: float = 1.0
    joint_gain_scales: tuple[float, ...] = (1.0,) * 29
    authority_calibration_hash: str | None = None
    planner_seed: int = 0
    schema_version: str = "rosclaw.simforge.g1_sonic_runup_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.run_velocity_mps,
            self.brake_velocity_mps,
            self.execution_duration_sec,
            self.policy_dt_sec,
            self.physics_dt_sec,
            self.gain_scale,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SONIC run-up config must be finite")
        if not 0.8 <= self.run_velocity_mps <= 1.8:
            raise ValueError("SONIC run velocity must be in [0.8, 1.8] m/s")
        if not 0.2 <= self.brake_velocity_mps <= 0.8:
            raise ValueError("SONIC brake velocity must be in [0.2, 0.8] m/s")
        if not 3.0 <= self.execution_duration_sec <= 4.5:
            raise ValueError("SONIC execution duration must be in [3.0, 4.5] s")
        if not 0.5 <= self.gain_scale <= 1.0:
            raise ValueError("SONIC gain scale must be in [0.5, 1.0]")
        if len(self.joint_gain_scales) != 29 or not all(
            math.isfinite(value) and 0.5 <= value <= 1.0
            for value in self.joint_gain_scales
        ):
            raise ValueError("SONIC joint gain scales must contain 29 values in [0.5, 1.0]")
        if self.authority_calibration_hash is not None and not self.authority_calibration_hash.startswith(
            "sha256:"
        ):
            raise ValueError("SONIC authority calibration hash must be SHA-256")
        ratio = self.policy_dt_sec / self.physics_dt_sec
        if not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError("SONIC policy dt must be an integer multiple of physics dt")
        if self.planner_seed < 0:
            raise ValueError("SONIC planner seed must be non-negative")

    @property
    def execution_frames(self) -> int:
        return int(round(self.execution_duration_sec / self.policy_dt_sec))

    @property
    def config_hash(self) -> str:
        return hash_json(asdict(self))


def qualify_g1_sonic(model_root: Path) -> G1SonicQualification:
    """Fail closed unless the public models expose the pinned tensor contract."""

    import onnxruntime

    root = model_root.expanduser().resolve()
    paths = {
        "planner": root / _PLANNER,
        "encoder": root / _ENCODER,
        "decoder": root / _DECODER,
        "observation_config": root / _OBSERVATION_CONFIG,
    }
    errors = [f"missing_asset={name}" for name, path in paths.items() if not path.is_file()]
    planner_input_count = 0
    encoder_input_size = 0
    decoder_input_size = 0
    decoder_output_size = 0
    if not errors:
        try:
            planner = onnxruntime.InferenceSession(
                str(paths["planner"]), providers=["CPUExecutionProvider"]
            )
            planner_input_count = len(planner.get_inputs())
            planner_outputs = planner.get_outputs()
            planner_shapes = tuple(tuple(item.shape) for item in planner_outputs)
            if planner_input_count != 11 or planner_shapes != ((1, 64, 36), (1,)):
                errors.append(
                    f"planner_contract={planner_input_count}:{planner_shapes},expected=11:((1,64,36),(1,))"
                )
        except Exception as exc:  # noqa: BLE001 - qualification records dependency errors
            errors.append(f"planner_load={type(exc).__name__}:{exc}")
        try:
            encoder = onnxruntime.InferenceSession(
                str(paths["encoder"]), providers=["CPUExecutionProvider"]
            )
            encoder_input_size = int(encoder.get_inputs()[0].shape[-1])
            encoder_output_size = int(encoder.get_outputs()[0].shape[-1])
            if (encoder_input_size, encoder_output_size) != (1247, 64):
                errors.append(
                    f"encoder_contract={encoder_input_size}->{encoder_output_size},expected=1247->64"
                )
        except Exception as exc:  # noqa: BLE001 - qualification records dependency errors
            errors.append(f"encoder_load={type(exc).__name__}:{exc}")
        try:
            decoder = onnxruntime.InferenceSession(
                str(paths["decoder"]), providers=["CPUExecutionProvider"]
            )
            decoder_input_size = int(decoder.get_inputs()[0].shape[-1])
            decoder_output_size = int(decoder.get_outputs()[0].shape[-1])
            if (decoder_input_size, decoder_output_size) != (994, 29):
                errors.append(
                    f"decoder_contract={decoder_input_size}->{decoder_output_size},expected=994->29"
                )
        except Exception as exc:  # noqa: BLE001 - qualification records dependency errors
            errors.append(f"decoder_load={type(exc).__name__}:{exc}")
    zero = "sha256:" + "0" * 64
    return G1SonicQualification(
        eligible=not errors,
        planner_hash=_file_hash(paths["planner"]) if paths["planner"].is_file() else zero,
        encoder_hash=_file_hash(paths["encoder"]) if paths["encoder"].is_file() else zero,
        decoder_hash=_file_hash(paths["decoder"]) if paths["decoder"].is_file() else zero,
        observation_config_hash=(
            _file_hash(paths["observation_config"])
            if paths["observation_config"].is_file()
            else zero
        ),
        planner_input_count=planner_input_count,
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
        decoder_output_size=decoder_output_size,
        errors=tuple(errors),
    )


class G1SonicRunupController:
    """Full-body SONIC planner plus proprioceptive neural policy."""

    default_angles = np.asarray(
        (
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.363,
            0.0,
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.363,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,
            0.2,
            -0.2,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,
        ),
        dtype=np.float64,
    )

    def __init__(self, model_root: Path, config: G1SonicRunupConfig | None = None) -> None:
        import onnxruntime

        self.config = config or G1SonicRunupConfig()
        self.qualification = qualify_g1_sonic(model_root)
        self.qualification.require_eligible()
        root = model_root.expanduser().resolve()
        self._planner = onnxruntime.InferenceSession(
            str(root / _PLANNER), providers=["CPUExecutionProvider"]
        )
        self._encoder = onnxruntime.InferenceSession(
            str(root / _ENCODER), providers=["CPUExecutionProvider"]
        )
        self._decoder = onnxruntime.InferenceSession(
            str(root / _DECODER), providers=["CPUExecutionProvider"]
        )
        self._history: collections.deque[tuple[np.ndarray, ...]] = collections.deque(maxlen=10)
        self.reference = np.empty((0, 36), dtype=np.float64)
        self.reference_digest = ""
        self.action = np.zeros(29, dtype=np.float32)
        self.target = self.default_angles.copy()
        self._kp, self._kd, self._action_scale = _sonic_control_parameters(
            self.config.gain_scale,
            self.config.joint_gain_scales,
        )

    @property
    def kp(self) -> np.ndarray:
        return self._kp

    @property
    def kd(self) -> np.ndarray:
        return self._kd

    def reset(self, data: Any) -> None:
        initial_qpos = np.asarray(data.qpos[:36], dtype=np.float64).copy()
        self.reference = self._generate_reference(initial_qpos)
        if len(self.reference) < self.config.execution_frames + 10:
            raise ValueError("SONIC planner reference is shorter than execution plus lookahead")
        digest = hashlib.sha256(np.ascontiguousarray(self.reference).tobytes()).hexdigest()
        self.reference_digest = "sha256:" + digest
        self.action[:] = 0.0
        self.target = self.default_angles.copy()
        self._history.clear()
        entry = self._history_entry(data, self.action)
        self._history.extend(tuple(value.copy() for value in entry) for _ in range(10))

    def update(self, data: Any, frame: int) -> np.ndarray:
        if not 0 <= frame < self.config.execution_frames:
            raise IndexError("SONIC execution frame is outside the frozen schedule")
        return self._update_from_reference(data, frame)

    @property
    def recovery_extension_frames(self) -> int:
        """Number of post-handoff frames with a complete look-ahead window."""

        return max(0, len(self.reference) - 9 - self.config.execution_frames)

    def extend_stationary_recovery(self, minimum_extension_frames: int) -> None:
        """Pad the planner's terminal standing pose for neural feedback recovery.

        Only the reference look-ahead is padded.  Policy history, measured
        proprioception, decoder actions and the simulated world remain
        continuous.  The resulting reference digest is updated so evidence
        binds the exact recovery horizon.
        """

        if minimum_extension_frames < 0:
            raise ValueError("SONIC stationary recovery frames must be non-negative")
        if len(self.reference) < 10:
            raise RuntimeError("SONIC must be reset before extending recovery")
        required_length = self.config.execution_frames + minimum_extension_frames + 9
        padding = required_length - len(self.reference)
        if padding > 0:
            terminal = np.repeat(self.reference[-1:, :], padding, axis=0)
            self.reference = np.concatenate((self.reference, terminal), axis=0)
            digest = hashlib.sha256(
                np.ascontiguousarray(self.reference).tobytes()
            ).hexdigest()
            self.reference_digest = "sha256:" + digest

    def update_recovery_extension(self, data: Any, frame: int) -> np.ndarray:
        """Continue the frozen neural motion tail without resetting policy state.

        The normal ``update`` boundary remains unchanged: shooting code cannot
        silently consume future planner motion.  A readiness-recovery caller
        must opt into this separate, bounded API after it has abstained.
        """

        if not 0 <= frame < self.recovery_extension_frames:
            raise IndexError("SONIC recovery frame is outside the frozen reference tail")
        return self._update_from_reference(data, self.config.execution_frames + frame)

    def _update_from_reference(self, data: Any, frame: int) -> np.ndarray:
        encoder_input = self._encoder_observation(data, frame)
        token = self._encoder.run(None, {self._encoder.get_inputs()[0].name: encoder_input})[0]
        decoder_input = np.zeros((1, 994), dtype=np.float32)
        decoder_input[0, :64] = token[0]
        history = list(self._history)
        offsets = (64, 94, 384, 674, 964, 994)
        for field, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            values = np.asarray([entry[field] for entry in history], dtype=np.float32)
            decoder_input[0, start:end] = values.reshape(-1)
        action = self._decoder.run(None, {self._decoder.get_inputs()[0].name: decoder_input})[0][0]
        self.action = np.asarray(action, dtype=np.float32)
        if self.action.shape != (29,) or not np.all(np.isfinite(self.action)):
            raise FloatingPointError("SONIC decoder emitted an invalid action")
        self.target = self.default_angles + self.action[ISAACLAB_TO_MUJOCO] * self._action_scale
        return self.target.copy()

    def observe(self, data: Any) -> None:
        """Commit the post-control proprioceptive state for the next policy tick."""

        self._history.append(self._history_entry(data, self.action))

    def raw_torque(self, data: Any) -> np.ndarray:
        return (self.target - data.qpos[7:36]) * self._kp - data.qvel[6:35] * self._kd

    def torque(self, data: Any) -> np.ndarray:
        limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        return np.clip(self.raw_torque(data), -limits, limits)

    def _generate_reference(self, initial_qpos: np.ndarray) -> np.ndarray:
        context = np.repeat(initial_qpos.astype(np.float32)[None, :], 4, axis=0)
        schedule = (
            (3, self.config.run_velocity_mps),
            (3, self.config.run_velocity_mps),
            (2, self.config.brake_velocity_mps),
            (0, -1.0),
        )
        segments: list[np.ndarray] = []
        for index, (mode, velocity) in enumerate(schedule):
            feed = {
                "context_mujoco_qpos": context[None, :].astype(np.float32),
                "target_vel": np.asarray((velocity,), dtype=np.float32),
                "mode": np.asarray((mode,), dtype=np.int64),
                "movement_direction": np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
                "facing_direction": np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
                "random_seed": np.asarray((self.config.planner_seed + index,), dtype=np.int64),
                "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                "allowed_pred_num_tokens": np.asarray(
                    ((1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),), dtype=np.int64
                ),
                "height": np.asarray((-1.0,), dtype=np.float32),
            }
            output, count = self._planner.run(None, feed)
            segment = np.asarray(output[0, : int(count[0])], dtype=np.float64)
            if segment.ndim != 2 or segment.shape[1] != 36 or len(segment) < 2:
                raise FloatingPointError("SONIC planner emitted an invalid motion segment")
            if not np.all(np.isfinite(segment)):
                raise FloatingPointError("SONIC planner emitted a non-finite motion segment")
            segments.append(segment)
            context = segment[-4:].astype(np.float32)
        return _resample_segments_30_to_50(segments, self.config.policy_dt_sec)

    def _encoder_observation(self, data: Any, frame: int) -> np.ndarray:
        import mujoco

        future = self.reference[frame : frame + 10]
        positions = future[:, 7:36][:, MUJOCO_TO_ISAACLAB]
        velocities = np.gradient(positions, self.config.policy_dt_sec, axis=0)
        observation = np.zeros((1, 1247), dtype=np.float32)
        # Offset 0 is encoder_mode_4.  G1 is mode 0, hence four zeros.
        observation[0, 4:294] = positions.reshape(-1)
        observation[0, 294:584] = velocities.reshape(-1)
        current_inverse = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        current_inverse[1:] *= -1.0
        orientation: list[float] = []
        for reference_quaternion in future[:, 3:7]:
            relative = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(relative, current_inverse, reference_quaternion)
            matrix = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(matrix, relative)
            orientation.extend(matrix.reshape(3, 3)[:, :2].reshape(-1))
        observation[0, 584:644] = orientation
        return observation

    def _history_entry(
        self, data: Any, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        joint_position = (np.asarray(data.qpos[7:36], dtype=np.float64) - self.default_angles)[
            MUJOCO_TO_ISAACLAB
        ]
        joint_velocity = np.asarray(data.qvel[6:35], dtype=np.float64)[MUJOCO_TO_ISAACLAB]
        return (
            np.asarray(data.qvel[3:6], dtype=np.float64).copy(),
            joint_position.copy(),
            joint_velocity.copy(),
            np.asarray(action, dtype=np.float64).copy(),
            _gravity_in_body(np.asarray(data.qpos[3:7], dtype=np.float64)),
        )


def _resample_segments_30_to_50(segments: list[np.ndarray], policy_dt_sec: float) -> np.ndarray:
    output: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        source_time = np.arange(len(segment), dtype=np.float64) / 30.0
        target_time = np.arange(0.0, source_time[-1] + 1e-12, policy_dt_sec, dtype=np.float64)
        resampled = np.empty((len(target_time), 36), dtype=np.float64)
        for column in (*range(3), *range(7, 36)):
            resampled[:, column] = np.interp(target_time, source_time, segment[:, column])
        # The public planner stays close enough between 30 Hz frames for
        # normalized linear quaternion interpolation to be unambiguous.
        for column in range(3, 7):
            resampled[:, column] = np.interp(target_time, source_time, segment[:, column])
        norm = np.linalg.norm(resampled[:, 3:7], axis=1)
        if np.any(norm < 1e-8):
            raise FloatingPointError("SONIC planner emitted a degenerate quaternion")
        resampled[:, 3:7] /= norm[:, None]
        output.append(resampled if index == 0 else resampled[1:])
    return np.concatenate(output, axis=0)


def _sonic_control_parameters(
    gain_scale: float,
    joint_gain_scales: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same motor constants and effort limits as the pinned C++ deployment.
    armature = np.asarray(
        (
            0.025101925,
            0.025101925,
            0.010177520,
            0.025101925,
            0.003609725,
            0.003609725,
            0.025101925,
            0.025101925,
            0.010177520,
            0.025101925,
            0.003609725,
            0.003609725,
            0.010177520,
            0.003609725,
            0.003609725,
            0.003609725,
            0.003609725,
            0.003609725,
            0.003609725,
            0.003609725,
            0.00425,
            0.00425,
            0.003609725,
            0.003609725,
            0.003609725,
            0.003609725,
            0.003609725,
            0.00425,
            0.00425,
        ),
        dtype=np.float64,
    )
    effort = np.asarray(
        (
            139,
            139,
            88,
            139,
            25,
            25,
            139,
            139,
            88,
            139,
            25,
            25,
            88,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            5,
            5,
            25,
            25,
            25,
            25,
            25,
            5,
            5,
        ),
        dtype=np.float64,
    )
    natural_frequency = 10.0 * 2.0 * math.pi
    stiffness = armature * natural_frequency**2
    damping = 4.0 * armature * natural_frequency
    doubled = np.asarray((4, 5, 10, 11, 13, 14), dtype=np.int64)
    stiffness[doubled] *= 2.0
    damping[doubled] *= 2.0
    action_scale = 0.25 * effort / (armature * natural_frequency**2)
    joint_scale = np.asarray(joint_gain_scales, dtype=np.float64)
    return stiffness * gain_scale * joint_scale, damping * gain_scale * joint_scale, action_scale


def _gravity_in_body(quaternion_wxyz: np.ndarray) -> np.ndarray:
    import mujoco

    conjugate = quaternion_wxyz.copy()
    conjugate[1:] *= -1.0
    value = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(value, np.asarray((0.0, 0.0, -1.0)), conjugate)
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "G1SonicQualification",
    "G1SonicRunupConfig",
    "G1SonicRunupController",
    "qualify_g1_sonic",
]
