from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.learners.iql import (
    NumpyIQLActor,
    _array_content_hash,
    _file_hash,
)
from rosclaw.growth.recovery_dataset import STATE_FEATURES
from rosclaw.simforge.g1_recovery_iql_evaluation import (
    evaluate_g1_recovery_iql_candidate,
)


def _candidate(tmp_path: Path, *, bad_weight_hash: bool = False) -> Path:
    hidden = 16
    arrays = {
        "net__0__weight": np.zeros((hidden, len(STATE_FEATURES)), dtype=np.float32),
        "net__0__bias": np.zeros(hidden, dtype=np.float32),
        "net__2__weight": np.zeros((hidden, hidden), dtype=np.float32),
        "net__2__bias": np.zeros(hidden, dtype=np.float32),
        "net__4__weight": np.zeros((29, hidden), dtype=np.float32),
        "net__4__bias": np.zeros(29, dtype=np.float32),
        "state_mean": np.zeros(len(STATE_FEATURES), dtype=np.float32),
        "state_std": np.ones(len(STATE_FEATURES), dtype=np.float32),
        "action_mean": np.zeros(29, dtype=np.float32),
        "action_std": np.ones(29, dtype=np.float32),
    }
    weights = tmp_path / "weights.npz"
    np.savez_compressed(weights, **arrays)
    candidate = {
        "schema_version": "rosclaw.growth.iql_candidate.v1",
        "status": "CANDIDATE_UNEVALUATED",
        "artifact": {
            "format": "numpy_npz_no_pickle",
            "weights_path": str(weights),
            "weights_hash": "sha256:" + "0" * 64 if bad_weight_hash else _file_hash(weights),
            "weights_content_hash": _array_content_hash(arrays),
        },
    }
    candidate["candidate_hash"] = canonical_hash(candidate)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return path


def test_iql_actor_loads_safe_npz_and_produces_finite_torque(tmp_path: Path) -> None:
    actor = NumpyIQLActor.load(_candidate(tmp_path))

    action = actor.action(np.zeros(len(STATE_FEATURES)))

    assert action.shape == (29,)
    assert np.all(np.isfinite(action))
    assert np.array_equal(action, np.zeros(29))


def test_iql_actor_rejects_tampered_weight_commitment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="weight hash mismatch"):
        NumpyIQLActor.load(_candidate(tmp_path, bad_weight_hash=True))


def test_iql_evaluation_output_must_be_outside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        evaluate_g1_recovery_iql_candidate(
            candidate_path=tmp_path / "candidate.json",
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )
