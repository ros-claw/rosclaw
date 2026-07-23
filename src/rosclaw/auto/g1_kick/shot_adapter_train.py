"""Small uncertainty-aware MLP learned from GoalForge Practice teachers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    ShotParameters,
    hash_bytes,
    hash_json,
)

INPUT_NAMES = (
    "ball_x",
    "ball_y",
    "target_y",
    "target_z",
    "pelvis_yaw",
    "support_foot_right",
    "support_foot_friction_belief",
    "ball_mass_belief",
    "control_latency_belief_ms",
    "body_calibration_state",
    *(f"memory_{index}" for index in range(9)),
)
OUTPUT_NAMES = (
    "stance_offset_x",
    "stance_offset_y",
    "pelvis_yaw_offset",
    "com_shift_y",
    "swing_amplitude",
    "swing_speed_scale",
    "foot_yaw_offset",
    "contact_phase_offset",
    "recovery_step_length",
)
OUTPUT_BOUNDS = np.asarray(
    (
        (-0.12, 0.12),
        (-0.12, 0.12),
        (-0.20, 0.20),
        (-0.08, 0.08),
        (0.75, 1.15),
        (0.80, 1.15),
        (-0.12, 0.12),
        (-0.10, 0.10),
        (0.0, 0.15),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class ShotAdapterTeacherSample:
    context: tuple[tuple[str, float], ...]
    best_safe_patch: ShotParameters
    teacher_evaluation_hash: str

    def __post_init__(self) -> None:
        values = dict(self.context)
        if tuple(values) != INPUT_NAMES:
            raise ValueError("Shot Adapter teacher input contract is invalid")
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Shot Adapter teacher context must be finite")
        if not self.teacher_evaluation_hash.startswith("sha256:"):
            raise ValueError("Shot Adapter teacher label requires evaluation evidence")

    @classmethod
    def from_values(
        cls,
        *,
        context: dict[str, float],
        best_safe_patch: ShotParameters,
        teacher_evaluation_hash: str,
    ) -> ShotAdapterTeacherSample:
        return cls(
            context=tuple((name, float(context[name])) for name in INPUT_NAMES),
            best_safe_patch=best_safe_patch,
            teacher_evaluation_hash=teacher_evaluation_hash,
        )


@dataclass(frozen=True)
class ShotAdapterTrainingMetrics:
    sample_count: int
    final_mse: float
    residual_uncertainty: tuple[float, ...]
    epochs: int
    seed: int


@dataclass(frozen=True)
class ShotAdapterInference:
    parameters: ShotParameters
    uncertainty: float
    output_uncertainty: tuple[float, ...]
    inference_ms: float
    model_hash: str


class G1ShotAdapter:
    """One-hidden-layer MLP with bounded output projection."""

    def __init__(
        self,
        *,
        dataset_snapshot_hash: str,
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        residual_uncertainty: np.ndarray,
        metrics: ShotAdapterTrainingMetrics,
    ) -> None:
        if not dataset_snapshot_hash.startswith("sha256:"):
            raise ValueError("learned Shot Adapter requires Dataset Snapshot provenance")
        self.dataset_snapshot_hash = dataset_snapshot_hash
        self.input_mean = np.asarray(input_mean, dtype=np.float64)
        self.input_scale = np.asarray(input_scale, dtype=np.float64)
        self.w1 = np.asarray(w1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.w2 = np.asarray(w2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)
        self.residual_uncertainty = np.asarray(
            residual_uncertainty,
            dtype=np.float64,
        )
        self.metrics = metrics
        self._validate_shapes()

    @classmethod
    def train(
        cls,
        *,
        samples: tuple[ShotAdapterTeacherSample, ...],
        dataset_snapshot_hash: str,
        hidden_size: int = 24,
        epochs: int = 800,
        seed: int = 20260723,
    ) -> G1ShotAdapter:
        if len(samples) < 8:
            raise ValueError("Shot Adapter training requires at least eight teachers")
        if not 8 <= hidden_size <= 64 or not 100 <= epochs <= 5000:
            raise ValueError("Shot Adapter MLP configuration is outside safe bounds")
        x = np.asarray(
            [[dict(sample.context)[name] for name in INPUT_NAMES] for sample in samples],
            dtype=np.float64,
        )
        y = np.asarray(
            [_patch_vector(sample.best_safe_patch) for sample in samples],
            dtype=np.float64,
        )
        input_mean = x.mean(axis=0)
        input_scale = np.maximum(x.std(axis=0), 1e-4)
        xn = np.clip((x - input_mean) / input_scale, -6.0, 6.0)
        yn = _normalize_output(y)
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 0.18, size=(len(INPUT_NAMES), hidden_size))
        b1: np.ndarray = np.zeros(hidden_size, dtype=np.float64)
        w2 = rng.normal(0.0, 0.12, size=(hidden_size, len(OUTPUT_NAMES)))
        b2: np.ndarray = np.zeros(len(OUTPUT_NAMES), dtype=np.float64)
        learning_rate = 0.025
        for epoch in range(epochs):
            hidden = np.tanh(xn @ w1 + b1)
            predicted = np.tanh(hidden @ w2 + b2)
            error = predicted - yn
            grad_output = (2.0 / len(samples)) * error * (1.0 - predicted**2)
            grad_w2 = hidden.T @ grad_output + 1e-5 * w2
            grad_b2 = grad_output.sum(axis=0)
            grad_hidden = (grad_output @ w2.T) * (1.0 - hidden**2)
            grad_w1 = xn.T @ grad_hidden + 1e-5 * w1
            grad_b1 = grad_hidden.sum(axis=0)
            rate = learning_rate * (0.25 + 0.75 * (1.0 - epoch / epochs))
            w1 -= rate * np.clip(grad_w1, -2.0, 2.0)
            b1 -= rate * np.clip(grad_b1, -2.0, 2.0)
            w2 -= rate * np.clip(grad_w2, -2.0, 2.0)
            b2 -= rate * np.clip(grad_b2, -2.0, 2.0)
        prediction = np.tanh(np.tanh(xn @ w1 + b1) @ w2 + b2)
        residual = np.sqrt(np.mean((prediction - yn) ** 2, axis=0))
        metrics = ShotAdapterTrainingMetrics(
            sample_count=len(samples),
            final_mse=float(np.mean((prediction - yn) ** 2)),
            residual_uncertainty=tuple(float(value) for value in residual),
            epochs=epochs,
            seed=seed,
        )
        return cls(
            dataset_snapshot_hash=dataset_snapshot_hash,
            input_mean=input_mean,
            input_scale=input_scale,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            residual_uncertainty=np.maximum(residual, 1e-4),
            metrics=metrics,
        )

    @property
    def model_hash(self) -> str:
        arrays = (
            self.input_mean,
            self.input_scale,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.residual_uncertainty,
        )
        payload = b"".join(np.ascontiguousarray(array).astype("<f8").tobytes() for array in arrays)
        return hash_bytes(payload + self.dataset_snapshot_hash.encode())

    def infer(self, context: dict[str, float]) -> ShotAdapterInference:
        import time

        if set(context) != set(INPUT_NAMES):
            raise ValueError("Shot Adapter inference context is invalid")
        started = time.perf_counter()
        x = np.asarray([context[name] for name in INPUT_NAMES], dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("Shot Adapter inference context must be finite")
        normalized = np.clip((x - self.input_mean) / self.input_scale, -6.0, 6.0)
        hidden = np.tanh(normalized @ self.w1 + self.b1)
        raw = np.tanh(hidden @ self.w2 + self.b2)
        projected = _denormalize_output(raw)
        distance_uncertainty = max(0.0, float(np.max(np.abs(normalized))) - 2.5) * 0.05
        output_uncertainty = np.clip(
            self.residual_uncertainty + distance_uncertainty,
            0.0,
            1.0,
        )
        parameters = ShotParameters(
            **dict(zip(OUTPUT_NAMES, projected, strict=True)),
            policy_type="learned_adapter",
            dataset_snapshot_hash=self.dataset_snapshot_hash,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ShotAdapterInference(
            parameters=parameters,
            uncertainty=float(np.max(output_uncertainty)),
            output_uncertainty=tuple(float(value) for value in output_uncertainty),
            inference_ms=elapsed_ms,
            model_hash=self.model_hash,
        )

    def export(self, output_dir: Path) -> dict[str, Any]:
        root = output_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        weights = root / "shot-adapter.npz"
        np.savez_compressed(
            weights,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            residual_uncertainty=self.residual_uncertainty,
        )
        manifest = {
            "schema_version": "rosclaw.g1_shot_adapter.model.v1",
            "model_hash": self.model_hash,
            "weights_hash": hash_bytes(weights.read_bytes()),
            "dataset_snapshot_hash": self.dataset_snapshot_hash,
            "input_names": list(INPUT_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "metrics": asdict(self.metrics),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _validate_shapes(self) -> None:
        count = len(INPUT_NAMES)
        output_count = len(OUTPUT_NAMES)
        hidden = self.w1.shape[1] if self.w1.ndim == 2 else 0
        expected = (
            self.input_mean.shape == (count,)
            and self.input_scale.shape == (count,)
            and self.w1.shape == (count, hidden)
            and self.b1.shape == (hidden,)
            and self.w2.shape == (hidden, output_count)
            and self.b2.shape == (output_count,)
            and self.residual_uncertainty.shape == (output_count,)
        )
        if not expected or hidden == 0:
            raise ValueError("Shot Adapter MLP tensor shapes are invalid")
        arrays = (
            self.input_mean,
            self.input_scale,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.residual_uncertainty,
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Shot Adapter MLP contains non-finite values")


@dataclass(frozen=True)
class ShotAdapterChampion:
    body_hash: str
    kick_prior_hash: str
    model_hash: str
    dataset_snapshot_hash: str
    validation_evidence_hash: str
    active: bool = True

    @property
    def registry_hash(self) -> str:
        return hash_json(asdict(self))


class ShotAdapterRegistry:
    def __init__(self) -> None:
        self._active: ShotAdapterChampion | None = None

    def activate(
        self,
        *,
        model: G1ShotAdapter,
        body_hash: str,
        kick_prior_hash: str,
        validation_evidence_hash: str,
        fall_rate: float,
        torque_violation_rate: float,
    ) -> ShotAdapterChampion:
        if fall_rate != 0.0 or torque_violation_rate != 0.0:
            raise ValueError("unsafe Shot Adapter cannot be activated")
        if not validation_evidence_hash.startswith("sha256:"):
            raise ValueError("Shot Adapter activation requires validation evidence")
        champion = ShotAdapterChampion(
            body_hash=body_hash,
            kick_prior_hash=kick_prior_hash,
            model_hash=model.model_hash,
            dataset_snapshot_hash=model.dataset_snapshot_hash,
            validation_evidence_hash=validation_evidence_hash,
        )
        self._active = champion
        return champion

    def resolve(self, *, body_hash: str, kick_prior_hash: str) -> ShotAdapterChampion:
        if (
            self._active is None
            or self._active.body_hash != body_hash
            or self._active.kick_prior_hash != kick_prior_hash
        ):
            raise LookupError("no compatible active G1 Shot Adapter")
        return self._active


def build_shot_adapter_context(
    *,
    observed_context: dict[str, float],
    twin_context: dict[str, float],
    memory_summary: tuple[float, ...],
    pelvis_yaw: float = 0.0,
    support_foot: str = "left",
) -> dict[str, float]:
    if len(memory_summary) != 9 or support_foot not in {"left", "right"}:
        raise ValueError("invalid Shot Adapter context components")
    values = {
        "ball_x": observed_context["ball_x"],
        "ball_y": observed_context["ball_y"],
        "target_y": observed_context["target_y"],
        "target_z": observed_context["target_z"],
        "pelvis_yaw": pelvis_yaw,
        "support_foot_right": float(support_foot == "right"),
        "support_foot_friction_belief": twin_context["support_friction_belief"],
        "ball_mass_belief": twin_context["ball_mass_belief"],
        "control_latency_belief_ms": twin_context["control_latency_belief_ms"],
        "body_calibration_state": observed_context["body_calibration_state"],
        **{f"memory_{index}": float(value) for index, value in enumerate(memory_summary)},
    }
    return {name: float(values[name]) for name in INPUT_NAMES}


def _patch_vector(patch: ShotParameters) -> np.ndarray:
    return np.asarray([float(getattr(patch, name)) for name in OUTPUT_NAMES])


def _normalize_output(values: np.ndarray) -> np.ndarray:
    lower = OUTPUT_BOUNDS[:, 0]
    upper = OUTPUT_BOUNDS[:, 1]
    return np.clip(2.0 * (values - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def _denormalize_output(values: np.ndarray) -> np.ndarray:
    lower = OUTPUT_BOUNDS[:, 0]
    upper = OUTPUT_BOUNDS[:, 1]
    return np.clip(lower + (values + 1.0) * 0.5 * (upper - lower), lower, upper)


__all__ = [
    "G1ShotAdapter",
    "INPUT_NAMES",
    "OUTPUT_NAMES",
    "ShotAdapterChampion",
    "ShotAdapterInference",
    "ShotAdapterRegistry",
    "ShotAdapterTeacherSample",
    "ShotAdapterTrainingMetrics",
    "build_shot_adapter_context",
]
