"""EVO-3 protocol + stats tests (PR-EVO-3, v4 §13)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "evo3"))

from arms import ArmA, ArmB, build_arm  # noqa: E402
from offline_validations import (  # noqa: E402
    RUN1_DEATH_SPIRAL_PATCHES,
    validate_exp3_counter_regime,
    validate_exp4_choreography_protection,
)
from protocols import EXP2_MATCHED_DEGRADATION, PROTOCOLS, crossover_triggered  # noqa: E402
from stats_analysis import (  # noqa: E402
    SessionRecord,
    aggregate_by_arm,
    kaplan_meier,
    mcnemar,
    median_first_failure,
    mixed_effects_invalid_rate,
    paired_bootstrap,
    promotion_report,
    restricted_mean_survival,
)


def _session(session_id: str, arm: str, invalid: int, rounds: int = 100, first: int | None = 30):
    return SessionRecord(
        session_id=session_id,
        arm=arm,
        rounds=rounds,
        invalid_count=invalid,
        failure_count=invalid,
        first_failure_round=first if invalid else None,
        verified_count=rounds - invalid,
        peak_temperature_c=49.0,
        temperature_slope=0.01,
    )


def test_protocols_declared_with_safety_limits() -> None:
    assert set(PROTOCOLS) == {
        "evo3_exp1_healthy_abstain",
        "evo3_exp2_matched_degradation",
        "evo3_exp3_counter_regime",
        "evo3_exp4_choreography_protection",
    }
    exp1 = PROTOCOLS["evo3_exp1_healthy_abstain"]
    assert exp1.sessions_per_arm == 12
    assert exp1.rounds_per_session == 100
    assert exp1.hand_balance is True
    assert exp1.safety_limits["max_temperature_c"] <= 56.0
    exp2 = PROTOCOLS["evo3_exp2_matched_degradation"]
    assert exp2.trigger["conditions_required"] == 2
    assert exp2.safety_limits["max_temperature_c"] <= 58.0


def test_crossover_trigger_requires_two_of_three() -> None:
    assert (
        crossover_triggered(
            temperature_slope=0.2, position_error_p95=20.0, recent_invalid_rate=0.01
        )
        is True
    )
    assert (
        crossover_triggered(temperature_slope=0.2, position_error_p95=5.0, recent_invalid_rate=0.01)
        is False
    )
    # Missing evidence never counts as holding.
    assert (
        crossover_triggered(
            temperature_slope=None, position_error_p95=20.0, recent_invalid_rate=0.08
        )
        is True
    )
    assert (
        crossover_triggered(
            temperature_slope=None, position_error_p95=None, recent_invalid_rate=0.08
        )
        is False
    )
    assert EXP2_MATCHED_DEGRADATION.trigger["conditions_required"] == 2


def test_arm_behaviors() -> None:
    a = ArmA()
    assert a.respond({"signature": "x"}, None).acted is False
    b = ArmB()
    outcome = b.respond({"signature": "x"}, None)
    assert outcome.acted is True
    assert outcome.patch == {"inter_round_cooldown_sec": 5}
    with pytest.raises(ValueError):
        build_arm("C_regime_aware", pipeline=None)


def test_exp3_counter_regime_passes() -> None:
    """v4 §11.3: the thermal memory in a healthy regime → ABSTAIN."""
    from rosclaw.how.choreography import ChoreographyValidator, load_contract
    from rosclaw.how.selective import SelectiveInterventionPipeline
    from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
    from rosclaw.memory.v2.models import MemoryItem
    from rosclaw.memory.v2.regime import (
        ApplicabilityEnvelope,
        ApplicabilityStore,
        RegimeLabel,
        empty_regime,
    )
    from rosclaw.memory.v2.runtime_retrieval import build_retrieval_facade

    client = InMemoryKnowledgeStore()
    client.connect()
    client.insert(
        "memory_items",
        MemoryItem(
            memory_id="mem_hot_slow",
            memory_type="failure",
            robot_id="r1",
            body_id="rh56_right_01",
            joint_name="middle",
            failure_type="joint_not_reached",
            title="热退化 middle 不到位",
            document="56–58°C 两小时会话 middle joint_not_reached，减速+延时",
            outcome="failure",
            evidence_refs=["run1"],
            metadata={"recovery_hint": "减速并增加延时"},
        ).to_record(),
    )
    store = ApplicabilityStore(client)
    store.upsert(
        ApplicabilityEnvelope(
            memory_id="mem_hot_slow",
            body_ids=["rh56_right_01"],
            regime_labels=[RegimeLabel.COLD_HEALTHY.value],
            envelope_type="contraindicated",
            reason="breaks_reveal_timing",
            confidence=0.9,
        )
    )
    validator = ChoreographyValidator(load_contract("configs/choreography/rh56_rps_v1.yaml"))
    pipeline = SelectiveInterventionPipeline(
        build_retrieval_facade(sqlite_store=client), store, choreography_validator=validator
    )
    healthy = empty_regime(robot_id="r1", body_id="rh56_right_01", task_id="rh56_rps")
    healthy.regime_label = RegimeLabel.COLD_HEALTHY.value
    healthy.temperature_c = 49.0
    healthy.confidence = 0.8
    result = validate_exp3_counter_regime(
        pipeline,
        healthy_regime=healthy,
        failure_signature="middle joint_not_reached",
        body_id="rh56_right_01",
        joint_name="middle",
    )
    assert result["retrieve"] is True
    assert result["applicable"] is False
    assert result["decision"] == "ABSTAIN"
    assert result["passed"] is True


def test_exp4_choreography_protection_blocks_all_run1_patches() -> None:
    """v4 §11.4: 100% BLOCK, 0 real executions."""
    result = validate_exp4_choreography_protection()
    assert result["passed"] is True
    assert result["blocked"] == len(RUN1_DEATH_SPIRAL_PATCHES)
    assert result["real_executions"] == 0
    for entry in result["results"]:
        assert entry["allowed"] is False
        assert entry["violations"]


# ---------------------------------------------------------------------------
# Stats machinery
# ---------------------------------------------------------------------------


def test_paired_bootstrap_detects_improvement() -> None:
    before = [0.08, 0.06, 0.09, 0.07, 0.08, 0.10]
    after = [0.03, 0.02, 0.04, 0.03, 0.02, 0.04]
    result = paired_bootstrap(before, after, n_bootstrap=2000)
    assert result["mean_diff"] < 0
    assert result["ci"][1] < 0  # entirely negative CI → real improvement
    assert result["p_two_sided"] < 0.05


def test_mcnemar_exact() -> None:
    result = mcnemar(9, 1)
    assert result["p_exact"] < 0.05
    assert mcnemar(0, 0)["p_exact"] is None


def test_survival_curve_and_rmst() -> None:
    times = [10, 20, 30, 100, 100]
    events = [True, True, True, False, False]
    curve = kaplan_meier(times, events)
    assert curve[-1]["survival"] < 1.0
    assert median_first_failure(times, events) == 30
    rmst = restricted_mean_survival(times, events, 100)
    assert 0 < rmst < 100
    never = restricted_mean_survival([100, 100], [False, False], 100)
    assert never == 100


def test_promotion_report_session_level() -> None:
    a_invalid = [6, 7, 8, 9, 8, 7]
    c_invalid = [1, 2, 2, 3, 2, 2]
    records = [_session(f"a{i}", "A", v, first=20) for i, v in enumerate(a_invalid)] + [
        _session(f"b{i}", "C", v, first=60) for i, v in enumerate(c_invalid)
    ]
    report = promotion_report(records, arm_a="A", arm_b="C", n_bootstrap=2000)
    assert report["invalid_rate_mean_a"] == pytest.approx(sum(a_invalid) / 600)
    assert report["invalid_rate_mean_b"] == pytest.approx(sum(c_invalid) / 600)
    assert report["cohens_d"] < 0  # C better (lower invalid)
    assert report["paired_bootstrap"]["ci"][1] < 0
    assert report["rmst_gain"] > 0
    assert len(report["session_distribution_a"]) == 6
    assert "pooled round counts are never evidence" in report["note"]


def test_aggregate_by_arm_reports_session_distribution() -> None:
    records = [_session("a1", "A", 8), _session("a2", "A", 6), _session("b1", "C", 2)]
    agg = aggregate_by_arm(records)
    assert agg["A"]["sessions"] == 2
    assert agg["A"]["invalid_rate_per_session"] == [0.08, 0.06]
    assert agg["C"]["invalid_rate_mean"] == 0.02


def test_mixed_effects_honest_fallback_or_model() -> None:
    records = [_session(f"s{i}", "A" if i % 2 else "C", i % 5) for i in range(1, 10)]
    result = mixed_effects_invalid_rate(records)
    assert "available" in result
    if not result["available"]:
        assert "fallback" in result
