from __future__ import annotations

import math

import pytest

from rosclaw.continual.teacher_manifold import (
    TeacherManifoldGateContract,
    TeacherManifoldMetric,
)
from tests.continual.helpers import digest


def _gate() -> TeacherManifoldGateContract:
    return TeacherManifoldGateContract(
        gate_id="whole_body.recovery.teacher.v1",
        teacher_artifact_hash=digest("teacher-memory"),
        body_hash=digest("body"),
        metrics=(
            TeacherManifoldMetric("pelvis_height", scale=0.10),
            TeacherManifoldMetric("body_speed", scale=1.0),
        ),
        retention_core_radius=0.10,
        full_plasticity_radius=0.30,
    )


def test_teacher_manifold_gate_freezes_known_states_and_opens_smoothly() -> None:
    gate = _gate()
    reference = {"pelvis_height": 0.75, "body_speed": 0.0}

    known = gate.evaluate(reference, reference)
    transition = gate.evaluate(
        {"pelvis_height": 0.75 + 0.2 * math.sqrt(2.0) * 0.10, "body_speed": 0.0},
        reference,
    )
    novel = gate.evaluate({"pelvis_height": 0.85, "body_speed": 1.0}, reference)

    assert known.inside_retention_core is True
    assert known.plasticity_fraction == 0.0
    assert transition.normalized_distance == pytest.approx(0.2)
    assert transition.plasticity_fraction == pytest.approx(0.5)
    assert novel.fully_plastic is True
    assert novel.decision_hash.startswith("sha256:")


def test_teacher_manifold_gate_fails_closed_on_observation_drift() -> None:
    gate = _gate()
    reference = {"pelvis_height": 0.75, "body_speed": 0.0}

    with pytest.raises(ValueError, match="exactly"):
        gate.evaluate({"pelvis_height": 0.75}, reference)
    with pytest.raises(ValueError, match="finite"):
        gate.evaluate({"pelvis_height": float("nan"), "body_speed": 0.0}, reference)


def test_teacher_manifold_gate_has_no_hardware_or_promotion_authority() -> None:
    with pytest.raises(ValueError, match="SIM_ONLY"):
        TeacherManifoldGateContract(
            gate_id="unsafe.gate",
            teacher_artifact_hash=digest("teacher-memory"),
            body_hash=digest("body"),
            metrics=(TeacherManifoldMetric("state", scale=1.0),),
            retention_core_radius=0.1,
            full_plasticity_radius=0.3,
            hardware_execution_allowed=True,
        )
