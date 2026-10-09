from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.feedback.contracts import ControllerSnapshot, canonical_hash
from rosclaw.feedback.ilc import ILCFeedforward
from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.simforge.g1_feedback_evolution import (
    FeedbackEvolutionDecision,
    run_g1_feedback_evolution,
)
from rosclaw.simforge.g1_ilc_validation import G1ILCFeedforwardCandidate
from rosclaw.simforge.phase4_cli import _recovery_validation


def _use_probe_clock(monkeypatch: pytest.MonkeyPatch, elapsed_ns: int) -> None:
    """Control structural probes; production deadline enforcement stays real."""
    from rosclaw.simforge import g1_feedback_evolution

    def runtime_with_clock(**kwargs):
        clock = itertools.count(step=elapsed_ns)
        return build_g1_balance_runtime(**kwargs, compute_clock_ns=lambda: next(clock))

    monkeypatch.setattr(g1_feedback_evolution, "build_g1_balance_runtime", runtime_with_clock)


@pytest.mark.parametrize("elapsed_ns", [0, 1_000_000, 4_000_000])
def test_feedback_evolution_builds_offline_candidate_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elapsed_ns: int,
) -> None:
    _use_probe_clock(monkeypatch, elapsed_ns)
    paths = _write_evidence(tmp_path)

    result = run_g1_feedback_evolution(
        **paths,
        output_path=tmp_path / "evolution.json",
        source_checkout=tmp_path / "source",
    )

    assert result.decision is FeedbackEvolutionDecision.NEED_MORE_EVIDENCE
    assert result.candidate_artifact_verified
    assert not result.activated
    assert not result.registry_mutated
    assert not result.hardware_command_sent
    checks = {check.gate: check for check in result.checks}
    assert all(checks[f"F{index}"].passed for index in range(1, 6))
    assert checks["F6"].missing
    assert checks["F15"].missing
    assert result.candidate.rollback_count == 8
    assert result.candidate.accepted_update_count == 1
    persisted = json.loads((tmp_path / "evolution.json").read_text(encoding="utf-8"))
    assert persisted["candidate"]["candidate_hash"] == result.candidate.candidate_hash
    assert persisted["activation"]["activated"] is False


def test_feedback_evolution_rejects_probe_deadline_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_probe_clock(monkeypatch, 4_000_001)
    paths = _write_evidence(tmp_path)
    result = run_g1_feedback_evolution(
        **paths,
        output_path=tmp_path / "overdue-evolution.json",
        source_checkout=tmp_path / "source",
    )
    assert result.decision is FeedbackEvolutionDecision.REJECTED
    checks = {check.gate: check for check in result.checks}
    assert all(not checks[gate].passed and not checks[gate].missing for gate in ["F3", "F4", "F5"])
    assert not result.activated
    assert not result.registry_mutated
    assert not result.hardware_command_sent


def test_feedback_evolution_rejects_tampered_candidate_artifact(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    ilc = json.loads(paths["ilc_path"].read_text(encoding="utf-8"))
    artifact = Path(ilc["candidate_feedforward"]["artifact_path"])
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_g1_feedback_evolution(
            **paths,
            output_path=tmp_path / "evolution.json",
            source_checkout=tmp_path / "source",
        )


def test_feedback_evolution_cli_keeps_missing_evidence_nonzero(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_probe_clock(monkeypatch, 1_000_000)
    paths = _write_evidence(tmp_path)
    code = _recovery_validation(
        [
            "simforge",
            "validate",
            "g1-goalforge",
            "--profile",
            "feedback-evolution",
            "--feedback-evidence",
            str(paths["feedback_path"]),
            "--holdout-evidence",
            str(paths["holdout_path"]),
            "--ilc-evidence",
            str(paths["ilc_path"]),
            "--chaos-evidence",
            str(paths["chaos_path"]),
            "--output",
            str(tmp_path / "cli-evolution.json"),
        ]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "NEED_MORE_EVIDENCE"


def _write_evidence(tmp_path: Path) -> dict[str, Path]:
    body_hash = canonical_hash({"body": "g1-test"})
    kick_prior_hash = canonical_hash({"kick": "fixed"})
    regime_hash = canonical_hash({"regime": "same"})
    runtime = build_g1_balance_runtime(body_hash=body_hash)
    snapshot = ControllerSnapshot(
        controller_id=runtime.spec.loop_id + ":controller",
        controller_type=type(runtime.controller).__name__,
        body_hash=body_hash,
        loop_spec_hash=runtime.spec.spec_hash,
        config=runtime.controller.config_dict(),
    )
    receipt = {
        "controller_snapshot_hash": snapshot.snapshot_hash,
        "controller_hash": runtime.spec.controller_hash,
        "loop_spec_hash": runtime.spec.spec_hash,
        "jitter_p99_ms": 0.0,
        "dropped_frame_count": 0,
        "deadline_miss_count": 0,
        "correction_applied": True,
        "tracking_improved": False,
    }
    feedback = {
        "schema_version": "rosclaw.g1_feedback.validation.v1",
        "body_hash": body_hash,
        "kick_prior_hash": kick_prior_hash,
        "deadline_compliance_rate": 1.0,
        "cases": [
            {
                "baseline": _metrics(),
                "feedback": _metrics(),
                "feedback_receipt": receipt,
                "trajectory_strict_replay": True,
            }
        ],
        "claims": {"evidence_domain": "SIM", "real_hardware": False},
        "passed": True,
    }
    holdout = {
        "schema_version": "rosclaw.g1_feedback.holdout.v1",
        "body_hash": body_hash,
        "kick_prior_hash": kick_prior_hash,
        "deadline_miss_count": 0,
        "holdout_cases": [
            {
                "baseline_fall": False,
                "feedback_fall": False,
                "baseline_joint_violation": False,
                "feedback_joint_violation": False,
                "baseline_torque_violation": False,
                "feedback_torque_violation": False,
                "strict_replay": True,
            }
        ],
        "holdout_passed": True,
        "historical_regression_passed": True,
        "claims": {"evidence_domain": "SIM", "real_hardware": False},
        "passed": True,
    }

    artifact = tmp_path / "selected-feedforward.npz"
    values = np.full((2, 29), 0.001, dtype=np.float64)
    np.savez_compressed(artifact, feedforward_residual=values)
    feedforward = ILCFeedforward(
        body_hash=body_hash,
        regime_hash=regime_hash,
        joint_names=tuple(f"joint-{index}" for index in range(29)),
        values=values,
        residual_limit=0.008,
        trial=1,
        source_receipt_hashes=(canonical_hash({"receipt": 1}),),
    )
    manifest = feedforward.to_manifest()
    candidate = G1ILCFeedforwardCandidate(
        trajectory_hash=feedforward.trajectory_hash,
        body_hash=body_hash,
        regime_hash=regime_hash,
        joint_names=feedforward.joint_names,
        shape=(2, 29),
        residual_limit=0.008,
        residual_peak=float(manifest["residual_peak"]),
        trial=1,
        selected_campaign_trial=10,
        source_receipt_hashes=feedforward.source_receipt_hashes,
        value_hash=str(manifest["value_hash"]),
        artifact_path=str(artifact),
        artifact_hash=_file_hash(artifact),
    )
    trials = [
        {
            "trial": index,
            "update_accepted": index == 2,
            "deadline_miss_count": 0,
            "feedforward_hash": candidate.trajectory_hash if index == 10 else None,
        }
        for index in range(1, 11)
    ]
    ilc = {
        "schema_version": "rosclaw.g1_ilc.validation.v2",
        "body_hash": body_hash,
        "kick_prior_hash": kick_prior_hash,
        "regime_hash": regime_hash,
        "trials": trials,
        "candidate_feedforward": candidate.to_dict(),
        "monotonic_error": True,
        "error_reduction": 0.1,
        "strict_replay": True,
        "wrong_regime_rejected": True,
        "claims": {"evidence_domain": "SIM", "real_hardware": False},
        "passed": True,
    }
    chaos = {
        "dds": {
            "schema_version": "rosclaw.g1.dds_loopback_receipt.v1",
            "passed": True,
            "real_hardware_opened": False,
        },
        "executor": {
            "schema_version": "rosclaw.g1.chaos.v1",
            "passed": True,
            "real_hardware_opened": False,
            "old_trigger_replay_count": 0,
            "stale_task_verified_count": 0,
        },
        "passed": True,
    }
    paths = {}
    for name, value in (
        ("feedback", feedback),
        ("holdout", holdout),
        ("ilc", ilc),
        ("chaos", chaos),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[f"{name}_path"] = path
    return paths


def _metrics() -> dict[str, bool]:
    return {
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
    }


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
