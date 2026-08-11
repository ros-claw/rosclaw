from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.approach_strike_contracts import EVENT_PHASE_NAMES, STATE_FEATURES
from rosclaw.growth.learners.iql import NumpyIQLActor, _array_content_hash, _file_hash
from rosclaw.growth.phase_conditioned_residual import (
    G1PhaseConditionedResidualConfig,
    derive_g1_phase_conditioned_residual_candidate,
)


def _base_candidate(root: Path) -> Path:
    root.mkdir()
    hidden = 5
    arrays = {
        "net__0__weight": np.zeros((hidden, len(STATE_FEATURES)), dtype=np.float32),
        "net__0__bias": np.zeros(hidden, dtype=np.float32),
        "net__2__weight": np.zeros((hidden, hidden), dtype=np.float32),
        "net__2__bias": np.zeros(hidden, dtype=np.float32),
        "net__4__weight": np.zeros((29, hidden), dtype=np.float32),
        "net__4__bias": np.zeros(29, dtype=np.float32),
        "state_mean": np.zeros(len(STATE_FEATURES), dtype=np.float32),
        "state_std": np.ones(len(STATE_FEATURES), dtype=np.float32),
        "action_mean": np.linspace(-2.0, 2.0, 29, dtype=np.float32),
        "action_std": np.linspace(1.0, 3.0, 29, dtype=np.float32),
    }
    for phase in EVENT_PHASE_NAMES:
        index = STATE_FEATURES.index(f"event_phase.{phase}")
        arrays["state_mean"][index] = 0.2
        arrays["state_std"][index] = 0.4
    weights = root / "actor_weights.npz"
    np.savez_compressed(weights, **arrays)
    candidate = {
        "schema_version": "rosclaw.growth.iql_candidate.v1",
        "learner_id": "test",
        "task_id": "g1_approach_strike_transition",
        "status": "CANDIDATE_UNEVALUATED",
        "darwin_evaluated": False,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "artifact": {
            "format": "numpy_npz_no_pickle",
            "actor_output": "sim_teacher_residual_torque_nm",
            "learned_output_fraction": 1.0,
            "state_features": list(STATE_FEATURES),
            "weights_path": str(weights),
            "weights_hash": _file_hash(weights),
            "weights_content_hash": _array_content_hash(arrays),
        },
    }
    candidate["candidate_hash"] = canonical_hash(candidate)
    path = root / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return path


def _state(phase_id: int) -> np.ndarray:
    value = np.zeros(len(STATE_FEATURES), dtype=np.float32)
    value[STATE_FEATURES.index(f"event_phase.{EVENT_PHASE_NAMES[phase_id]}")] = 1.0
    return value


def test_derivation_preserves_base_and_adds_only_the_selected_phase(tmp_path: Path) -> None:
    base_path = _base_candidate(tmp_path / "base")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    delta = np.zeros(29, dtype=np.float64)
    delta[3] = -1.2

    output = derive_g1_phase_conditioned_residual_candidate(
        base_candidate_path=base_path,
        output_dir=tmp_path / "evidence" / "candidate",
        source_checkout=checkout,
        config=G1PhaseConditionedResidualConfig(
            event_phase_id=4,
            joint_delta_nm=tuple(float(item) for item in delta),
        ),
    )

    base = NumpyIQLActor.load(base_path)
    actor = NumpyIQLActor.load(output)
    for phase_id in range(4):
        assert np.allclose(actor.action(_state(phase_id)), base.action(_state(phase_id)), atol=1e-6)
    expected = base.action(_state(4)) + delta
    assert np.allclose(actor.action(_state(4)), expected, atol=1e-5)
    payload = json.loads(output.read_text())
    assert payload["source_candidate_hash"] == base.candidate_hash
    assert payload["activation_ceiling"] == "SIM_ONLY"
    assert not payload["promotion_authorized"]


def test_phase_conditioned_candidate_fails_closed(tmp_path: Path) -> None:
    base_path = _base_candidate(tmp_path / "base")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    delta = (1.0,) + (0.0,) * 28
    with pytest.raises(ValueError, match="outside the source checkout"):
        derive_g1_phase_conditioned_residual_candidate(
            base_candidate_path=base_path,
            output_dir=checkout / "candidate",
            source_checkout=checkout,
            config=G1PhaseConditionedResidualConfig(event_phase_id=4, joint_delta_nm=delta),
        )
    with pytest.raises(ValueError, match="phase"):
        G1PhaseConditionedResidualConfig(event_phase_id=5, joint_delta_nm=delta)
    with pytest.raises(ValueError, match="non-zero"):
        G1PhaseConditionedResidualConfig(event_phase_id=4, joint_delta_nm=(0.0,) * 29)
