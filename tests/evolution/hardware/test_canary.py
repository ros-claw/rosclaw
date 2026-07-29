"""A/B/C canary schedule + candidate selection tests (PR-EVO-HW-4 §Phase 6)."""

from __future__ import annotations

from rosclaw.evolution.hardware.canary import (
    ARMS,
    build_canary_schedule,
    select_canary_candidate,
    select_explicit_candidate,
)


def test_schedule_is_seeded_interleaved_and_balanced() -> None:
    schedule = build_canary_schedule(blocks=3, seed=42, base_seed=1000)
    assert len(schedule) == 9
    # Every block runs each arm exactly once.
    for block in range(3):
        arms_in_block = {s.arm for s in schedule if s.block == block}
        assert arms_in_block == set(ARMS)
    # §Phase 6 相同手势 Seed: all arms within a block share the same seed.
    for block in range(3):
        seeds_in_block = {s.seed for s in schedule if s.block == block}
        assert len(seeds_in_block) == 1
    # Blocks draw independent seeds.
    block_seeds = [next(s.seed for s in schedule if s.block == b) for b in range(3)]
    assert len(set(block_seeds)) == 3
    # Deterministic for the same seed.
    again = build_canary_schedule(blocks=3, seed=42, base_seed=1000)
    assert [(s.block, s.arm, s.seed) for s in schedule] == [(s.block, s.arm, s.seed) for s in again]


def test_selection_prefers_conservative_cooldown_in_degradation() -> None:
    validated = [
        {"candidate_id": "c_pose", "changes": {"neutral_pose_between_blocks": True}},
        {"candidate_id": "c_cool4", "changes": {"inter_round_cooldown_sec": 4.0}},
        {"candidate_id": "c_cool2", "changes": {"inter_round_cooldown_sec": 2.0}},
        {"candidate_id": "c_empty", "changes": {}},
    ]
    pick, reason = select_canary_candidate(validated, baseline_regime="TRACKING_DEGRADATION")
    assert pick is not None
    assert pick["candidate_id"] == "c_cool2"  # most conservative effective cooldown
    assert "cooldown" in reason


def test_selection_skips_c0_and_handles_json_strings() -> None:
    validated = [
        {"candidate_id": "c_empty", "changes": {}},
        {"candidate_id": "c_cool", "changes": '{"inter_round_cooldown_sec": 2.0}'},
    ]
    pick, _ = select_canary_candidate(validated, baseline_regime="THERMAL_DRIFT")
    assert pick is not None
    assert pick["candidate_id"] == "c_cool"


def test_selection_none_when_no_nonempty_candidate() -> None:
    pick, reason = select_canary_candidate(
        [{"candidate_id": "c_empty", "changes": {}}], baseline_regime="COLD_HEALTHY"
    )
    assert pick is None
    assert "untried non-empty" in reason


def test_selection_pose_in_healthy_regime() -> None:
    validated = [
        {"candidate_id": "c_cool2", "changes": {"inter_round_cooldown_sec": 2.0}},
        {"candidate_id": "c_pose", "changes": {"rehome_between_blocks": True}},
    ]
    pick, reason = select_canary_candidate(validated, baseline_regime="COLD_HEALTHY")
    assert pick is not None
    assert pick["candidate_id"] == "c_pose"
    assert "pose-recovery" in reason


def test_selection_excludes_already_tried_candidates() -> None:
    """The ladder walks to the next UNTRIED candidate — re-running a tested
    one wastes hardware time."""
    validated = [
        {"candidate_id": "c_cool2", "changes": {"inter_round_cooldown_sec": 2.0}},
        {"candidate_id": "c_cool4", "changes": {"inter_round_cooldown_sec": 4.0}},
    ]
    pick, _ = select_canary_candidate(
        validated, baseline_regime="THERMAL_DRIFT", exclude_ids={"c_cool2"}
    )
    assert pick is not None
    assert pick["candidate_id"] == "c_cool4"
    pick2, reason2 = select_canary_candidate(
        validated, baseline_regime="THERMAL_DRIFT", exclude_ids={"c_cool2", "c_cool4"}
    )
    assert pick2 is None
    assert "untried" in reason2


def test_explicit_candidate_selection_discloses_operator_direction() -> None:
    """Operator-directed top-up: bypasses the ladder for a VALIDATED
    candidate and says so in the reason (evidence honesty)."""
    validated = [
        {"candidate_id": "c_cool2", "changes": {"inter_round_cooldown_sec": 2.0}},
        {"candidate_id": "c_every5", "changes": {"cooldown_every_n_rounds": 5}},
    ]
    pick, reason = select_explicit_candidate(validated, "c_every5")
    assert pick is not None
    assert pick["candidate_id"] == "c_every5"
    assert "operator-directed" in reason


def test_explicit_candidate_rejects_non_validated() -> None:
    validated = [{"candidate_id": "c_cool2", "changes": {"inter_round_cooldown_sec": 2.0}}]
    pick, reason = select_explicit_candidate(validated, "c_unknown")
    assert pick is None
    assert "not VALIDATED" in reason
