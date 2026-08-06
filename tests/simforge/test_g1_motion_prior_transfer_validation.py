from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.simforge.g1_motion_prior_transfer_validation import (
    _absolute_physics_checks,
    _teacher_bc_rejection_reasons,
)
from rosclaw.simforge.phase4_cli import _recovery_validation


def test_absolute_physics_checks_reject_relative_improvement_that_remains_unsafe() -> None:
    assert _absolute_physics_checks(
        {"critical_failure_rate": 0.5, "success_rate": 0.25}
    ) == {
        "zero_transfer_critical_failures": False,
        "minimum_transfer_success_rate_50pct": False,
    }


def test_absolute_physics_checks_accept_safe_effective_transfer() -> None:
    assert all(
        _absolute_physics_checks(
            {"critical_failure_rate": 0.0, "success_rate": 0.5}
        ).values()
    )


def test_unsafe_teacher_rollout_is_rejected_from_behavior_cloning() -> None:
    result = SimpleNamespace(
        finite_state=True,
        post_kick_fall=True,
        joint_limit_violation=True,
        torque_limit_violation=False,
    )

    assert _teacher_bc_rejection_reasons(result) == (
        "post_kick_fall",
        "joint_limit_violation",
    )


def test_motion_prior_transfer_cli_wires_profile_and_preserves_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rosclaw.simforge.g1_motion_prior_transfer_validation as validation

    captured: dict[str, object] = {}

    def run_fake(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(decision="REJECTED", to_dict=lambda: {"decision": "REJECTED"})

    monkeypatch.setattr(validation, "run_g1_motion_prior_transfer_validation", run_fake)
    motion_prior = tmp_path / "prior.json"
    code = _recovery_validation(
        [
            "simforge",
            "validate",
            "g1-goalforge",
            "--profile",
            "motion-prior-transfer",
            "--motion-prior",
            str(motion_prior),
            "--output",
            str(tmp_path / "evidence"),
            "--device",
            "cuda:3",
            "--gpu-epochs",
            "7",
        ]
    )

    assert code == 2
    assert captured["motion_prior_path"] == motion_prior
    assert captured["output_dir"] == tmp_path / "evidence"
    assert captured["device"] == "cuda:3"
    assert captured["epochs"] == 7
