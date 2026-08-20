from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.football_outcome_model import (
    G1FootballOutcomeModel,
    load_g1_football_outcome_model,
)
from rosclaw.growth.proprioceptive_expert_router import G1StrikeHandoffFeatures


def _model() -> G1FootballOutcomeModel:
    model = G1FootballOutcomeModel(
        expert_phases=(190, 205, 214),
        feature_location=(0.0,) * 6,
        feature_scale=(1.0,) * 6,
        development_seeds=(1, 2, 3),
        development_features=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.1),
            (2.0, 0.0, 0.0, 0.0, 0.0, 0.1),
        ),
        hard_safe_by_phase=((True, True, False),) * 3,
        precision_hit_by_phase=((False, True, False),) * 3,
        stability_qualified_by_phase=((False, True, False),) * 3,
        penalized_error_by_phase=((0.4, 0.02, 2.0),) * 3,
        saturation_score_by_phase=((0.1, 0.0, 1.0),) * 3,
        kick_tilt_score_by_phase=((0.2, 0.1, 1.0),) * 3,
        final_speed_score_by_phase=((0.1, 0.05, 1.0),) * 3,
        neighbor_count=1,
        distance_power=2.0,
        hard_failure_weight=4.0,
        precision_weight=1.0,
        saturation_weight=0.1,
        maximum_retry_support_distance=2.0,
        minimum_direct_attempt_safety_probability=0.75,
        baseline_phase=190,
        baseline_hard_safe_episodes=3,
        baseline_precision_hits=0,
        baseline_stability_qualified_episodes=0,
        baseline_mean_penalized_error_m=0.4,
        cross_validation_hard_safe_episodes=3,
        cross_validation_precision_hits=3,
        cross_validation_stability_qualified_episodes=3,
        cross_validation_mean_penalized_error_m=0.02,
        cross_validation_retry_recommendations=0,
        all_experts_unsafe_states=0,
        source_evidence_hashes=("sha256:" + "1" * 64,),
        source_implementation_hashes=("sha256:" + "2" * 64,),
        source_schema_versions=("flow:v1",),
        body_hash="sha256:" + "3" * 64,
        experiment_context_hash="sha256:" + "4" * 64,
        accepted=True,
        failure_codes=(),
        model_hash="sha256:" + "0" * 64,
    )
    return replace(model, model_hash=canonical_hash(model.to_dict(include_hash=False)))


def test_football_outcome_model_selects_a_shot_and_never_terminally_abstains() -> None:
    decision = _model().decide(
        G1StrikeHandoffFeatures(0.05, 0.0, 0.0, 0.0, 0.0, 0.1)
    )

    assert decision.selected_phase_start_frame == 205
    assert decision.predicted_precision_probability == 1.0
    assert not decision.retry_recommended


def test_football_outcome_loader_rejects_recovery_as_task_success(tmp_path: Path) -> None:
    value = _model().to_dict()
    value["objective"]["recovery_only_is_task_success"] = True
    unsigned = dict(value)
    unsigned.pop("model_hash")
    value["model_hash"] = canonical_hash(unsigned)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="task objective"):
        load_g1_football_outcome_model(path)


def test_v2_loader_rejects_stability_plasticity_regression(tmp_path: Path) -> None:
    value = _model().to_dict()
    value["schema_version"] = "rosclaw.growth.g1_football_outcome_model.v2"
    value["baseline_mean_saturation_score"] = 0.05
    value["cross_validation_mean_saturation_score"] = 0.06
    value["stability_plasticity_guard_enforced"] = True
    value["objective"]["stability_plasticity_guard_enforced"] = True
    value["objective"]["cross_validation_saturation_may_not_exceed_baseline"] = True
    unsigned = dict(value)
    unsigned.pop("model_hash")
    value["model_hash"] = canonical_hash(unsigned)
    path = tmp_path / "regressed-v2.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="model contract"):
        load_g1_football_outcome_model(path)
