"""Derive a phase-conditioned augmentation of a support-bound IQL actor.

The derivation preserves the complete base network and appends one dormant
hidden channel.  That channel becomes active only for one event-phase one-hot
feature and contributes a bounded joint-space delta.  The result is still an
unevaluated, SIM-only candidate: this module grants neither promotion nor
hardware authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.approach_strike_contracts import EVENT_PHASE_NAMES, STATE_FEATURES
from rosclaw.growth.learners.iql import NumpyIQLActor, _array_content_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_APPROACH_STRIKE_PHASE_IDS = frozenset((0, 1, 2, 3, 4))
_SILU_UNIT_INPUT = 1.2784645427610738


@dataclass(frozen=True)
class G1PhaseConditionedResidualConfig:
    """One small candidate delta bound to an approach-to-strike phase."""

    event_phase_id: int
    joint_delta_nm: tuple[float, ...]
    activation_logit: float = 5.0
    schema_version: str = "rosclaw.growth.g1_phase_conditioned_residual_config.v1"

    def __post_init__(self) -> None:
        if self.event_phase_id not in _APPROACH_STRIKE_PHASE_IDS:
            raise ValueError("phase-conditioned residual phase must be in [0, 4]")
        if len(self.joint_delta_nm) != len(G1_DDS_JOINT_NAMES):
            raise ValueError("phase-conditioned residual must contain 29 joint deltas")
        if not all(math.isfinite(value) for value in self.joint_delta_nm):
            raise ValueError("phase-conditioned joint deltas must be finite")
        if not any(abs(value) > 0.0 for value in self.joint_delta_nm):
            raise ValueError("phase-conditioned residual requires a non-zero joint delta")
        if any(abs(value) > 20.0 for value in self.joint_delta_nm):
            raise ValueError("phase-conditioned joint deltas must remain within 20 Nm")
        if not math.isfinite(self.activation_logit) or not 4.0 <= self.activation_logit <= 12.0:
            raise ValueError("phase-conditioned activation logit must be in [4, 12]")


def derive_g1_phase_conditioned_residual_candidate(
    *,
    base_candidate_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: G1PhaseConditionedResidualConfig,
) -> Path:
    """Append a phase basis to a base IQL actor without changing other phases."""

    base_path = base_candidate_path.expanduser().resolve()
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("phase-conditioned candidate must remain outside the source checkout")
    if root.exists():
        raise FileExistsError("phase-conditioned candidate output already exists")

    base_actor = NumpyIQLActor.load(base_path)
    if base_actor.task_id != "g1_approach_strike_transition":
        raise ValueError("phase-conditioned candidate requires an approach-to-strike actor")
    if base_actor.state_features != tuple(STATE_FEATURES):
        raise ValueError("phase-conditioned candidate state contract mismatch")
    base = _read_json(base_path)
    artifact = dict(base.get("artifact", {}))
    weights_path = Path(str(artifact.get("weights_path", ""))).expanduser().resolve()
    with np.load(weights_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}

    augmented = _append_phase_basis(arrays, config)
    root.mkdir(parents=True)
    output_weights = root / "actor_weights.npz"
    np.savez_compressed(output_weights, **augmented)  # type: ignore[arg-type]
    artifact.update(
        {
            "weights_path": str(output_weights),
            "weights_hash": _file_hash(output_weights),
            "weights_content_hash": _array_content_hash(augmented),
            "phase_conditioned_augmentation": asdict(config),
        }
    )
    candidate = {
        **base,
        "learner_id": "phase_conditioned_residual_augmentation",
        "source_candidate_hash": base_actor.candidate_hash,
        "artifact": artifact,
        "status": "CANDIDATE_UNEVALUATED",
        "darwin_evaluated": False,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    candidate.pop("candidate_hash", None)
    candidate["candidate_hash"] = canonical_hash(candidate)
    output = root / "candidate.json"
    output.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Reload through the public fail-closed path before returning the artifact.
    NumpyIQLActor.load(output)
    return output


def _append_phase_basis(
    arrays: dict[str, np.ndarray], config: G1PhaseConditionedResidualConfig
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value, dtype=np.float32).copy() for name, value in arrays.items()}
    first_weight = result["net__0__weight"]
    first_bias = result["net__0__bias"]
    second_weight = result["net__2__weight"]
    second_bias = result["net__2__bias"]
    output_weight = result["net__4__weight"]
    if (
        first_weight.ndim != 2
        or second_weight.shape[1] != first_weight.shape[0]
        or output_weight.shape[1] != second_weight.shape[0]
    ):
        raise ValueError("phase-conditioned base actor layer shapes are incompatible")

    first_size, state_size = first_weight.shape
    second_size = second_weight.shape[0]
    expanded_first = np.zeros((first_size + 1, state_size), dtype=np.float32)
    expanded_first[:first_size] = first_weight
    expanded_first_bias = np.zeros(first_size + 1, dtype=np.float32)
    expanded_first_bias[:first_size] = first_bias
    expanded_second = np.zeros((second_size + 1, first_size + 1), dtype=np.float32)
    expanded_second[:second_size, :first_size] = second_weight
    expanded_second_bias = np.zeros(second_size + 1, dtype=np.float32)
    expanded_second_bias[:second_size] = second_bias
    expanded_output = np.zeros((output_weight.shape[0], second_size + 1), dtype=np.float32)
    expanded_output[:, :second_size] = output_weight

    phase_name = EVENT_PHASE_NAMES[config.event_phase_id]
    phase_index = STATE_FEATURES.index(f"event_phase.{phase_name}")
    mean = float(result["state_mean"][phase_index])
    std = float(result["state_std"][phase_index])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
        raise ValueError("phase-conditioned event feature normalization is invalid")
    zero_state = -mean / std
    active_state = (1.0 - mean) / std
    scale = config.activation_logit
    input_weight = 2.0 * scale / (active_state - zero_state)
    input_bias = -scale - input_weight * zero_state
    expanded_first[first_size, phase_index] = input_weight
    expanded_first_bias[first_size] = input_bias

    negative_activation = _silu(-scale)
    positive_activation = _silu(scale)
    second_scale = _SILU_UNIT_INPUT / (positive_activation - negative_activation)
    expanded_second[second_size, first_size] = second_scale
    expanded_second_bias[second_size] = -second_scale * negative_activation
    action_std = result["action_std"]
    expanded_output[:, second_size] = (
        np.asarray(config.joint_delta_nm, dtype=np.float32) / action_std
    )

    result["net__0__weight"] = expanded_first
    result["net__0__bias"] = expanded_first_bias
    result["net__2__weight"] = expanded_second
    result["net__2__bias"] = expanded_second_bias
    result["net__4__weight"] = expanded_output
    return result


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("phase-conditioned base candidate must be a JSON object")
    return value


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "G1PhaseConditionedResidualConfig",
    "derive_g1_phase_conditioned_residual_candidate",
]
