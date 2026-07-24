"""Choreography contract + validator + state machine tests (PR-SAFE-2, v4 §13)."""

from __future__ import annotations

import pytest

from rosclaw.how.choreography import (
    ChoreographyValidator,
    PatchState,
    PatchStateMachine,
    build_timing_model,
    efficacy_learnable,
    load_contract,
)
from rosclaw.how.choreography.timing import RoundTiming

CONTRACT_PATH = "configs/choreography/rh56_rps_v1.yaml"


def _validator():
    return ChoreographyValidator(load_contract(CONTRACT_PATH))


def _model(parameters=None):
    contract = load_contract(CONTRACT_PATH)
    rounds = [
        RoundTiming(started_at=1_000.0 + i * 8.0, ended_at=1_004.0 + i * 8.0) for i in range(10)
    ]
    return build_timing_model(contract, rounds, current_parameters=parameters or {})


def _timing_model():
    """A REAL timing model for pipeline tests (empty models are refused)."""
    return _model()


# ---------------------------------------------------------------------------
# run1 death-spiral patches: 100% blocked (v4 §11.4/§13)
# ---------------------------------------------------------------------------


RUN1_DEATH_SPIRAL_PATCHES = [
    {"servo_speed_scale": 0.6},
    {"per_phase_delay_ms": 400},
    {"reveal_delay_ms": 250},
    {"gesture_motion_duration_ms": 1200},
    {"gesture_trajectory": "slower_arc"},
    {"servo_speed_scale": 0.6, "per_phase_delay_ms": 400},
]


@pytest.mark.parametrize("patch", RUN1_DEATH_SPIRAL_PATCHES)
def test_run1_death_spiral_patches_all_blocked(patch) -> None:
    validation = _validator().validate(patch, _model())
    assert validation.allowed is False
    assert any("forbidden_parameter" in v for v in validation.violations)


def test_choreography_blocks_patch_stacking() -> None:
    validator = _validator()
    # cooldown already active + cooldown_every_n_rounds → non-stackable.
    validation = validator.validate(
        {"cooldown_every_n_rounds": 10},
        _model(parameters={"inter_round_cooldown_sec": 5}),
    )
    assert validation.allowed is False
    assert any("non_stackable" in v for v in validation.violations)
    # Both non-stackable parameters in ONE patch.
    validation = validator.validate(
        {"inter_round_cooldown_sec": 5, "cooldown_every_n_rounds": 10}, _model()
    )
    assert validation.allowed is False
    assert any("non_stackable" in v for v in validation.violations)


def test_between_round_patch_allowed() -> None:
    validation = _validator().validate({"inter_round_cooldown_sec": 5}, _model())
    assert validation.allowed is True
    assert validation.reveal_window_preserved is True
    assert validation.total_round_budget_preserved is True
    # Cooldown grew by exactly 5s; reveal offset unchanged.
    assert (
        validation.patched_phase_durations["cooldown"]
        - validation.original_phase_durations["cooldown"]
        == 5000.0
    )
    assert validation.expected_reveal_offset_ms == 3300.0


def test_unknown_and_out_of_range_parameters_blocked() -> None:
    validator = _validator()
    assert validator.validate({"no_such_param": 1}, _model()).allowed is False
    validation = validator.validate({"inter_round_cooldown_sec": 120}, _model())
    assert validation.allowed is False
    assert any("out_of_range" in v for v in validation.violations)


def test_round_budget_block() -> None:
    # 25s cooldown pushes the round past the 20s budget (observed ~8s rounds).
    validation = _validator().validate({"inter_round_cooldown_sec": 25}, _model())
    assert validation.allowed is False
    assert any("round_budget" in v for v in validation.violations)


def test_telemetry_hz_timing_neutral_allowed() -> None:
    validation = _validator().validate({"telemetry_hz": 5.0}, _model())
    assert validation.allowed is True


# ---------------------------------------------------------------------------
# State machine (v4 §9)
# ---------------------------------------------------------------------------


def test_patch_state_machine_full_path() -> None:
    machine = PatchStateMachine("patch_1")
    assert machine.state is PatchState.PROPOSED
    for expected in (
        PatchState.APPLICABILITY_VALIDATED,
        PatchState.CHOREOGRAPHY_VALIDATED,
        PatchState.SANDBOX_VALIDATED,
        PatchState.APPROVED,
        PatchState.APPLIED,
        PatchState.OBSERVED,
        PatchState.CRITIC_VALIDATED,
        PatchState.LEARNED,
    ):
        assert machine.advance() is expected
    assert efficacy_learnable(machine.state) is True
    with pytest.raises(RuntimeError):
        machine.advance()


def test_patch_state_machine_block_from_any_gate() -> None:
    machine = PatchStateMachine("patch_2")
    machine.advance()
    machine.advance()
    machine.block(reason="sandbox_rejected")
    assert machine.state is PatchState.BLOCKED
    assert efficacy_learnable(machine.state) is False
    with pytest.raises(RuntimeError):
        machine.advance()
    history = machine.record.history
    assert history[-1]["reason"] == "sandbox_rejected"


# ---------------------------------------------------------------------------
# PatchProof v4 §9 fields
# ---------------------------------------------------------------------------


def test_patchproof_carries_regime_and_choreography_fields() -> None:
    from rosclaw.how.rule_efficacy import PatchProof

    proof = PatchProof(suggested_patch={"inter_round_cooldown_sec": 5})
    proof.current_regime_id = "reg_1"
    proof.matched_envelope_id = "env_1"
    proof.applicability_score = 0.91
    proof.choreography_contract_id = "rh56_rps_v1"
    proof.choreography_validation = {"allowed": True}
    proof.retrieval_physical_collection = "memory_items__qwen3_06b_768_v1__ik__g1"
    proof.embedding_profile_id = "qwen3_06b_768_v1"
    record = proof.to_record()
    assert record["current_regime_id"] == "reg_1"
    assert record["choreography_contract_id"] == "rh56_rps_v1"
    assert record["abstention_considered"] is True
    assert record["applicability_score"] == 0.91


# ---------------------------------------------------------------------------
# Pipeline integration: APPLY with forbidden patch parameters → ABSTAIN
# ---------------------------------------------------------------------------


def test_pipeline_abstains_on_choreography_violation() -> None:
    from rosclaw.how.selective import InterventionAction
    from tests.how.test_selective import (
        _candidate,
        _pipeline,
        _regime,
        _response,
        _validated_envelope,
    )

    # Candidate memory suggests a SERVO SPEED patch (run1's harmful move).
    candidate = _candidate("mem_speed", hint="降低舵机速度")
    candidate.item.metadata["patch_parameters"] = {"servo_speed_scale": 0.6}
    validator = _validator()
    pipeline, _ = _pipeline(
        _response([candidate]),
        [_validated_envelope("mem_speed")],
        choreography=validator,
    )
    pipeline._timing_model = _timing_model()
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert any(
        reason.startswith("choreography_violation:forbidden_parameter")
        for reason in decision.reason_codes
    )


def test_pipeline_abstains_when_no_timing_model() -> None:
    """Review finding: a patch with parameters and NO real timing model must
    not APPLY (budget unprovable — never validated against an empty model)."""
    from rosclaw.how.selective import InterventionAction
    from tests.how.test_selective import (
        _candidate,
        _pipeline,
        _regime,
        _response,
        _validated_envelope,
    )

    candidate = _candidate("mem_cool", hint="增加回合间冷却")
    candidate.item.metadata["patch_parameters"] = {"inter_round_cooldown_sec": 5}
    pipeline, _ = _pipeline(
        _response([candidate]),
        [_validated_envelope("mem_cool")],
        choreography=_validator(),
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert any("no_timing_model" in reason for reason in decision.reason_codes)


def test_pipeline_apply_with_safe_cooldown_patch() -> None:
    from rosclaw.how.selective import InterventionAction
    from tests.how.test_selective import (
        _candidate,
        _pipeline,
        _regime,
        _response,
        _validated_envelope,
    )

    candidate = _candidate("mem_cool", hint="增加回合间冷却")
    candidate.item.metadata["patch_parameters"] = {"inter_round_cooldown_sec": 5}
    pipeline, _ = _pipeline(
        _response([candidate]),
        [_validated_envelope("mem_cool")],
        choreography=_validator(),
    )
    pipeline._timing_model = _timing_model()
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.APPLY
    assert decision.suggested_patch["parameters"] == {"inter_round_cooldown_sec": 5}


def test_stacking_generalized_over_all_non_stackable_params() -> None:
    """Review finding: patching cooldown_sec while cooldown_every_n_rounds
    is already active must also conflict (was hardcoded to one param)."""
    validator = _validator()
    validation = validator.validate(
        {"inter_round_cooldown_sec": 5},
        _model(parameters={"cooldown_every_n_rounds": 10}),
    )
    assert validation.allowed is False
    assert any("non_stackable" in v for v in validation.violations)


def test_budget_uses_effective_current_cooldown() -> None:
    """Review finding: re-patching while a cooldown is already active must
    not under-count the round total (10s patch + live 8s cooldown > 20s)."""
    validator = _validator()
    model = _model(parameters={"inter_round_cooldown_sec": 8})
    # 18s replaces the active 8s → effective 18s cooldown + 4.4s phases > 20s.
    validation = validator.validate({"inter_round_cooldown_sec": 18}, model)
    assert validation.allowed is False
    assert any("round_budget" in v for v in validation.violations)
    # A naive model that ignored the active 8s would have allowed this.


def test_apply_patch_refuses_unknown_timing_params() -> None:
    """Defense in depth: the model never guesses an unknown parameter's
    timing effect (the validator rejects these first anyway)."""
    import pytest as _pytest

    from rosclaw.how.choreography.timing import apply_patch

    with _pytest.raises(ValueError, match="unknown timing effect"):
        apply_patch(_model(), {"some_future_param": 1}, load_contract(CONTRACT_PATH))
