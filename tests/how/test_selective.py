"""Selective intervention pipeline tests (PR-HOW-3, v4 §13)."""

from __future__ import annotations

from rosclaw.how.selective import (
    InterventionAction,
    SelectiveInterventionPipeline,
    SelectiveRiskLedger,
)
from rosclaw.how.selective.decision import (
    REASON_CONFLICTING_MEMORIES,
    REASON_CONTRAINDICATED,
    REASON_NO_PATCHPROOF,
    REASON_NO_SAME_BODY_MEMORY,
    REASON_NO_SAME_JOINT_MEMORY,
    REASON_PROVIDER_DEGRADED_HIGH_RISK,
    REASON_REGIME_SCORE_LOW,
    REASON_SUGGEST_BAND,
    REASON_VALIDATED_MATCH,
)
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.regime import (
    ApplicabilityEnvelope,
    ApplicabilityStore,
    EnvelopeType,
    RegimeLabel,
    empty_regime,
)
from rosclaw.memory.v2.runtime_retrieval import MODE_ACTIVE_BM25
from rosclaw.memory.v2.runtime_retrieval.result import (
    RetrievalCandidate,
    RetrievalPurpose,
    RetrievalResponse,
)

T0 = 1_700_000_000.0


def _candidate(
    memory_id: str,
    *,
    body: str = "rh56_right_01",
    joint: str = "middle",
    hint: str = "增加回合间冷却",
    rank: int = 1,
) -> RetrievalCandidate:
    item = MemoryItem(
        memory_id=memory_id,
        memory_type="failure",
        robot_id="r1",
        body_id=body,
        joint_name=joint,
        failure_type="joint_not_reached",
        title=f"{body} {joint} failure",
        document=f"{body} 的 {joint} 未达到目标位置",
        outcome="failure",
        evidence_refs=["evt_1"],
        metadata={"recovery_hint": hint},
    )
    return RetrievalCandidate(
        memory_id=memory_id,
        memory_type="failure",
        vector_rank=None,
        bm25_rank=None,
        fusion_rank=rank,
        rerank_score=None,
        exact_entity_match=True,
        body_match=True,
        joint_match=True,
        failure_type_match=True,
        physical_collection="memory_items__fake__ik__g1",
        embedding_profile_id="fake_8d_v1",
        score_semantics="test",
        item=item,
    )


def _response(
    candidates: list[RetrievalCandidate], *, mode: str = "fake_hybrid"
) -> RetrievalResponse:
    return RetrievalResponse(
        retrieval_mode=mode,
        logical_collection="memory_items",
        physical_collection="memory_items__fake__ik__g1",
        embedding_profile_id="fake_8d_v1",
        purpose=RetrievalPurpose.HOW_INTERVENTION,
        fallback=(mode != "fake_hybrid"),
        fallback_reason=None if mode == "fake_hybrid" else "embedding_provider_unavailable:test",
        candidates=candidates,
    )


class _StubFacade:
    def __init__(self, response: RetrievalResponse) -> None:
        self._response = response
        self.calls: list = []

    def retrieve(self, query, *, purpose):
        self.calls.append((query, purpose))
        return self._response


def _regime(**overrides):
    regime = empty_regime(robot_id="r1", body_id="rh56_right_01", task_id="rh56_rps", now=T0)
    regime.regime_label = RegimeLabel.THERMAL_TRACKING_DEGRADATION.value
    regime.temperature_c = 57.0
    regime.temperature_slope_c_per_min = 0.25
    regime.session_elapsed_sec = 5400.0
    regime.cumulative_action_count = 900
    regime.position_error_p95 = 22.0
    regime.recent_failure_rate = 0.12
    regime.recent_invalid_rate = 0.10
    regime.confidence = 0.85
    for key, value in overrides.items():
        setattr(regime, key, value)
    return regime


def _healthy_regime():
    return _regime(
        regime_label=RegimeLabel.COLD_HEALTHY.value,
        temperature_c=49.0,
        temperature_slope_c_per_min=0.01,
        session_elapsed_sec=600.0,
        cumulative_action_count=80,
        position_error_p95=5.0,
        recent_failure_rate=0.0,
        recent_invalid_rate=0.02,
    )


def _validated_envelope(memory_id: str, **overrides) -> ApplicabilityEnvelope:
    envelope = ApplicabilityEnvelope(
        memory_id=memory_id,
        body_ids=["rh56_right_01"],
        task_ids=["rh56_rps"],
        joints=["middle"],
        temperature_min=55.0,
        temperature_max=60.0,
        temperature_slope_min=0.1,
        temperature_slope_max=0.5,
        elapsed_sec_min=3600.0,
        elapsed_sec_max=7200.0,
        regime_labels=[RegimeLabel.THERMAL_TRACKING_DEGRADATION.value],
        envelope_type=EnvelopeType.VALIDATED.value,
        evidence_count=4,
        success_count=3,
        confidence=0.8,
        required_features=[],
    )
    for key, value in overrides.items():
        setattr(envelope, key, value)
    return envelope


def _pipeline(
    response: RetrievalResponse,
    envelopes: list[ApplicabilityEnvelope],
    *,
    choreography: object | None = None,
) -> tuple[SelectiveInterventionPipeline, ApplicabilityStore]:
    client = InMemoryKnowledgeStore()
    client.connect()
    store = ApplicabilityStore(client)
    for envelope in envelopes:
        store.upsert(envelope)
    pipeline = SelectiveInterventionPipeline(
        _StubFacade(response), store, choreography_validator=choreography
    )
    return pipeline, store


def test_abstain_on_low_applicability() -> None:
    """v4 §13: healthy regime + thermal memory → ABSTAIN (the PR #98 harm)."""
    pipeline, _ = _pipeline(_response([_candidate("mem_slow")]), [_validated_envelope("mem_slow")])
    decision = pipeline.decide("middle joint_not_reached", _healthy_regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_REGIME_SCORE_LOW in decision.reason_codes
    assert decision.selected_memory_id == "mem_slow"
    assert decision.applicability_score < 0.70
    assert decision.abstention_considered is True


def test_abstain_on_cross_body_only() -> None:
    """v4 §13: only the other body's memories exist → ABSTAIN, never borrow."""
    pipeline, _ = _pipeline(_response([_candidate("mem_left", body="rh56_left_01")]), [])
    decision = pipeline.decide("middle joint_not_reached", _regime(), body_id="rh56_right_01")
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_NO_SAME_BODY_MEMORY in decision.reason_codes


def test_abstain_on_missing_joint_memory() -> None:
    pipeline, _ = _pipeline(_response([_candidate("mem_thumb", joint="thumb")]), [])
    decision = pipeline.decide("middle joint_not_reached", _regime(), joint_name="middle")
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_NO_SAME_JOINT_MEMORY in decision.reason_codes


def test_abstain_on_conflicting_memories() -> None:
    """v4 §13: top-1/top-2 close but different suggestions → ABSTAIN."""
    candidates = [
        _candidate("mem_a", hint="增加冷却", rank=1),
        _candidate("mem_b", hint="降低速度", rank=2),
    ]
    # Both applicable with close scores: no regime-label constraint so both
    # match, identical envelopes → identical scores.
    envelopes = [
        _validated_envelope("mem_a", regime_labels=[], envelope_id="env_a"),
        _validated_envelope("mem_b", regime_labels=[], envelope_id="env_b"),
    ]
    pipeline, _ = _pipeline(_response(candidates), envelopes)
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_CONFLICTING_MEMORIES in decision.reason_codes


def test_apply_requires_validated_envelope() -> None:
    """v4 §13: high score without patch-proof evidence → SUGGEST, not APPLY.

    Uses the inconsistent-but-possible state VALIDATED type + zero success
    evidence (e.g. a hand-authored envelope): score reaches the APPLY band
    but the evidence gate downgrades to SUGGEST — never silent APPLY.
    (OBSERVED envelopes top out below the APPLY band by design.)
    """
    premature_validated = _validated_envelope(
        "mem_slow",
        envelope_type=EnvelopeType.VALIDATED.value,
        success_count=0,
        confidence=0.95,
        evidence_count=5,
    )
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")]), [premature_validated], choreography=object()
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.SUGGEST
    assert REASON_NO_PATCHPROOF in decision.reason_codes
    assert decision.suggested_patch is not None
    assert "sandbox_validation_required" not in decision.safety_requirements


def test_apply_with_validated_envelope_and_gates() -> None:
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")]),
        [_validated_envelope("mem_slow")],
        choreography=object(),  # validator wired (SAFE-2 provides the real one)
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.APPLY
    assert REASON_VALIDATED_MATCH in decision.reason_codes
    assert decision.suggested_patch is not None
    assert decision.suggested_patch["description"] == "增加回合间冷却"
    assert decision.safety_requirements == [
        "sandbox_validation_required",
        "choreography_validation_required",
    ]


def test_apply_blocked_when_choreography_unavailable() -> None:
    """v4 §7.3: APPLY without a choreography validator is forbidden."""
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")]),
        [_validated_envelope("mem_slow")],
        choreography=None,
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert "choreography_validator_unavailable" in decision.reason_codes


def test_abstain_on_contraindicated_hit() -> None:
    contra = ApplicabilityEnvelope(
        memory_id="mem_slow",
        body_ids=["rh56_right_01"],
        task_ids=["rh56_rps"],
        regime_labels=[RegimeLabel.THERMAL_TRACKING_DEGRADATION.value],
        envelope_type=EnvelopeType.CONTRAINDICATED.value,
        reason="breaks_reveal_timing",
        evidence_refs=["run1"],
    )
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")]),
        [_validated_envelope("mem_slow"), contra],
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_CONTRAINDICATED in decision.reason_codes
    assert decision.estimated_harm == 1.0


def test_abstain_when_provider_degraded() -> None:
    """v4 §7.3: BM25-only degradation on the high-risk path → ABSTAIN."""
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")], mode=MODE_ACTIVE_BM25),
        [_validated_envelope("mem_slow")],
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_PROVIDER_DEGRADED_HIGH_RISK in decision.reason_codes


def test_suggest_band_decision() -> None:
    """Mid-band applicability (no hard rejections, moderate score) → SUGGEST."""
    # OBSERVED type factor with solid evidence/confidence lands in the band.
    envelope = _validated_envelope(
        "mem_slow",
        envelope_type=EnvelopeType.OBSERVED.value,
        confidence=0.85,
        evidence_count=3,
        success_count=2,
    )
    pipeline, _ = _pipeline(_response([_candidate("mem_slow")]), [envelope], choreography=object())
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.SUGGEST
    assert (
        REASON_SUGGEST_BAND in decision.reason_codes
        or REASON_NO_PATCHPROOF in decision.reason_codes
    )


def test_no_candidates_abstains() -> None:
    pipeline, _ = _pipeline(_response([]), [])
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert "no_retrieval_candidate" in decision.reason_codes


# ---------------------------------------------------------------------------
# SelectiveRiskLedger
# ---------------------------------------------------------------------------


def _decide_n(pipeline, regime, n: int) -> list:
    return [pipeline.decide("middle joint_not_reached", regime) for _ in range(n)]


def test_selective_risk_metrics_and_gate() -> None:
    client = InMemoryKnowledgeStore()
    client.connect()
    ledger = SelectiveRiskLedger(client)

    from rosclaw.how.selective.decision import SelectiveInterventionDecision, new_decision_id

    def make_decision(action: InterventionAction):
        return SelectiveInterventionDecision(
            decision_id=new_decision_id(),
            action=action,
            failure_signature="f",
            selected_memory_id="mem_x",
            selected_rule_id=None,
            retrieval_confidence=0.9,
            applicability_score=0.9,
            regime_confidence=0.9,
            evidence_confidence=0.9,
            expected_benefit=0.8,
            estimated_harm=0.0,
            uncertainty=0.1,
        )

    # 90 decisions: 80 APPLY + 2 SUGGEST + 8 ABSTAIN.
    ids = []
    for action in (
        [InterventionAction.APPLY] * 80
        + [InterventionAction.SUGGEST] * 2
        + [InterventionAction.ABSTAIN] * 8
    ):
        decision = make_decision(action)
        ledger.record_decision(decision, body_id="rh56_right_01")
        ids.append(decision.decision_id)
    ledger.record_outcome(ids[0], "helpful")

    metrics = ledger.metrics()
    assert metrics["eligible"] == 90
    assert metrics["coverage"] == 80 / 90
    assert metrics["abstention_rate"] == 8 / 90
    assert metrics["apply_count"] == 80
    gate = ledger.gate_report()
    assert gate["wrong_apply_zero"] is True
    # 0/1 judged: the Wilson upper bound is huge — gate honestly fails.
    assert gate["passed"] is False

    # 80 clean applies: the Wilson upper falls below the 5% gate.
    for decision_id in ids[1:80]:
        ledger.record_outcome(decision_id, "helpful")
    gate = ledger.gate_report()
    assert gate["passed"] is True
    assert gate["metrics"]["selective_harm_risk"] == 0.0
    assert gate["metrics"]["helpful_apply_precision"] == 1.0


def test_abstain_when_sqlite_lexical_fallback_serves() -> None:
    """Review finding: the lexical fallback path previously reached APPLY
    with zero vector retrieval — any degraded mode must abstain on the
    high-risk path."""
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")], mode="sqlite_memory_v2_lexical"),
        [_validated_envelope("mem_slow")],
        choreography=object(),
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.ABSTAIN
    assert REASON_PROVIDER_DEGRADED_HIGH_RISK in decision.reason_codes


def test_success_evidence_only_from_matched_envelope() -> None:
    """Review finding: patch-proof evidence was accepted from ANY envelope;
    it must come from the matched VALIDATED envelope itself."""
    weak_validated = _validated_envelope(
        "mem_slow",
        envelope_type=EnvelopeType.VALIDATED.value,
        success_count=0,
        confidence=0.95,
        evidence_count=5,
    )
    unrelated_observed = _validated_envelope(
        "mem_slow",
        envelope_id="env_other",
        envelope_type=EnvelopeType.OBSERVED.value,
        success_count=9,
        confidence=0.6,
    )
    pipeline, _ = _pipeline(
        _response([_candidate("mem_slow")]),
        [weak_validated, unrelated_observed],
        choreography=object(),
    )
    decision = pipeline.decide("middle joint_not_reached", _regime())
    assert decision.action is InterventionAction.SUGGEST
    assert REASON_NO_PATCHPROOF in decision.reason_codes
