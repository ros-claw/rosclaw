"""Selective intervention pipeline (数据库优化v4 §7.2).

    Failure Event
    → Exact Entity Extraction
    → ACTIVE Memory Retrieval (facade, purpose=HOW_INTERVENTION)
    → Regime (built upstream, passed in)
    → Regime Matcher (applicability)
    → Verified HOW Rule Match
    → Benefit/Harm Estimate
    → APPLY / SUGGEST / ABSTAIN / ESCALATE

Decision ladder (v4 §6.4/§7.3):

* any forced-ABSTAIN condition (wrong body/joint, missing regime feature,
  contraindicated hit, conflicting candidates, provider degraded on the
  high-risk path, choreography/sandbox gate unavailable for APPLY)
  → ABSTAIN
* applicability < abstain_below → ABSTAIN
* abstain_below ≤ score < suggest_below → SUGGEST
* score ≥ suggest_below AND validated envelope AND patch-proof success
  evidence AND no conflicts → APPLY candidate (still gated downstream by
  choreography + sandbox before any real motion — never by this pipeline
  alone).  A high score WITHOUT patch-proof evidence downgrades to
  SUGGEST (operator gate), never silently to APPLY.
* safety-taxonomy blocking severity → ESCALATE
"""

from __future__ import annotations

import logging
from typing import Any

from rosclaw.memory.v2.document import extract_exact_terms
from rosclaw.memory.v2.regime import (
    ApplicabilityStore,
    EnvelopeType,
    RegimeMatcher,
)
from rosclaw.memory.v2.regime.models import OperatingRegime
from rosclaw.memory.v2.retrieval import MemoryQuery
from rosclaw.memory.v2.runtime_retrieval import (
    MemoryRetrievalFacade,
    RetrievalPurpose,
)

from .decision import (
    REASON_CHOREOGRAPHY_UNAVAILABLE,
    REASON_CONFLICTING_MEMORIES,
    REASON_CONTRAINDICATED,
    REASON_NO_CANDIDATE,
    REASON_NO_PATCHPROOF,
    REASON_NO_SAME_BODY_MEMORY,
    REASON_NO_SAME_JOINT_MEMORY,
    REASON_PROVIDER_DEGRADED_HIGH_RISK,
    REASON_REGIME_FEATURE_MISSING,
    REASON_REGIME_SCORE_LOW,
    REASON_SAFETY_ESCALATION,
    REASON_SUGGEST_BAND,
    REASON_VALIDATED_MATCH,
    InterventionAction,
    SelectiveInterventionDecision,
    new_decision_id,
)

logger = logging.getLogger("rosclaw.how.selective.pipeline")

CONFLICT_SCORE_EPSILON = 0.05


class SelectiveInterventionPipeline:
    """Memory-driven selective intervention with ABSTAIN as a first-class outcome."""

    def __init__(
        self,
        facade: MemoryRetrievalFacade,
        applicability_store: ApplicabilityStore,
        *,
        matcher: RegimeMatcher | None = None,
        engine: Any | None = None,
        choreography_validator: Any | None = None,
        timing_model: Any | None = None,
    ) -> None:
        self._facade = facade
        self._envelopes = applicability_store
        self._matcher = matcher or RegimeMatcher()
        self._engine = engine
        self._choreography = choreography_validator
        self._timing_model = timing_model

    def decide(
        self,
        failure_signature: str,
        regime: OperatingRegime,
        *,
        robot_id: str | None = None,
        body_id: str | None = None,
        joint_name: str | None = None,
        limit: int = 5,
    ) -> SelectiveInterventionDecision:
        exact = extract_exact_terms(failure_signature)
        body = body_id or regime.body_id
        hands = exact.get("hands") or []
        if body is None and len(hands) == 1:
            body = f"rh56_{hands[0]}_01"
        joint = joint_name or (exact.get("joints") or [None])[0]

        # 1) ACTIVE retrieval on the intervention path (cross-body forbidden).
        response = self._facade.retrieve(
            MemoryQuery(
                text=failure_signature,
                robot_id=robot_id or regime.robot_id,
                body_id=body,
                outcome="failure",
                limit=limit,
            ),
            purpose=RetrievalPurpose.HOW_INTERVENTION,
        )
        retrieval_confidence = self._retrieval_confidence(response)

        def _verdict(
            action: InterventionAction,
            reasons: list[str],
            *,
            memory_id: str | None = None,
            rule_id: str | None = None,
            score: float = 0.0,
            envelope_id: str | None = None,
            patch: dict[str, Any] | None = None,
            benefit: float = 0.0,
            harm: float = 0.0,
            explanation: str = "",
            evidence_confidence: float = 0.0,
        ) -> SelectiveInterventionDecision:
            return SelectiveInterventionDecision(
                decision_id=new_decision_id(),
                action=action,
                failure_signature=failure_signature,
                selected_memory_id=memory_id,
                selected_rule_id=rule_id,
                retrieval_confidence=retrieval_confidence,
                applicability_score=score,
                regime_confidence=regime.confidence,
                evidence_confidence=evidence_confidence,
                expected_benefit=benefit,
                estimated_harm=harm,
                uncertainty=round(1.0 - min(retrieval_confidence, max(regime.confidence, 0.05)), 4),
                reason_codes=reasons,
                explanation=explanation,
                suggested_patch=patch,
                safety_requirements=(
                    ["sandbox_validation_required", "choreography_validation_required"]
                    if action is InterventionAction.APPLY
                    else []
                ),
                regime_label=regime.regime_label,
                matched_envelope_id=envelope_id,
            )

        # 2) Escalate on blocking safety severity (taxonomy, if engine wired).
        if self._engine is not None:
            escalate = self._safety_escalation(failure_signature)
            if escalate is not None:
                return _verdict(
                    InterventionAction.ESCALATE,
                    [REASON_SAFETY_ESCALATION],
                    explanation=f"safety taxonomy: {escalate}",
                )

        # 3) Candidate checks.
        candidates = response.candidates
        if not candidates:
            return _verdict(
                InterventionAction.ABSTAIN,
                [REASON_NO_CANDIDATE],
                explanation="retrieval returned no candidates on the intervention path",
            )
        same_body = [c for c in candidates if body is None or (c.item and c.item.body_id == body)]
        if not same_body:
            return _verdict(
                InterventionAction.ABSTAIN,
                [REASON_NO_SAME_BODY_MEMORY],
                explanation="only cross-body memories exist; HOW never borrows (v4 §7.4)",
            )
        if joint is not None:
            same_joint = [c for c in same_body if c.item is not None and c.item.joint_name == joint]
            if not same_joint:
                return _verdict(
                    InterventionAction.ABSTAIN,
                    [REASON_NO_SAME_JOINT_MEMORY],
                    explanation=f"query names joint {joint}; no {joint} memory on {body}",
                )

        # 4) Degraded retrieval on the high-risk path → ABSTAIN (v4 §7.3).
        #    BM25-on-ACTIVE, sqlite lexical fallback, and full abstain are
        #    all degraded: none of them is the pinned embedding path.
        if response.fallback:
            return _verdict(
                InterventionAction.ABSTAIN,
                [REASON_PROVIDER_DEGRADED_HIGH_RISK],
                explanation=(
                    f"retrieval degraded ({response.retrieval_mode}: "
                    f"{response.fallback_reason}); the high-risk intervention "
                    "path requires the pinned embedding path"
                ),
            )

        # 5) Applicability per candidate.  The failure's joint IS the
        # current joint context for matching (the regime may not carry one).
        if joint is not None and regime.joint_name is None:
            regime.joint_name = joint
        scored: list[tuple[Any, Any, Any]] = []  # (candidate, match_result, envelopes)
        for candidate in same_body:
            envelopes = self._envelopes.for_memory(candidate.memory_id)
            match = self._matcher.match(candidate.memory_id, envelopes, regime)
            scored.append((candidate, match, envelopes))

        contra = next(
            (m for _, m, _ in scored if "contraindicated_envelope_hit" in m.hard_rejections),
            None,
        )
        if contra is not None:
            return _verdict(
                InterventionAction.ABSTAIN,
                [REASON_CONTRAINDICATED],
                memory_id=contra.memory_id,
                envelope_id=contra.matched_envelope_id,
                harm=1.0,
                explanation="contraindicated envelope hit in the current regime",
            )

        missing = next(
            (m for _, m, _ in scored if "missing_required_features" in m.hard_rejections),
            None,
        )
        if missing is not None:
            return _verdict(
                InterventionAction.ABSTAIN,
                [REASON_REGIME_FEATURE_MISSING],
                memory_id=missing.memory_id,
                explanation=f"regime lacks {missing.missing_required_features}",
            )

        applicable = [(c, m, e) for c, m, e in scored if m.applicable]
        if not applicable:
            best = max(scored, key=lambda item: item[1].score, default=None)
            return _verdict(
                InterventionAction.ABSTAIN,
                [REASON_REGIME_SCORE_LOW],
                memory_id=best[1].memory_id if best else None,
                score=best[1].score if best else 0.0,
                explanation=(
                    f"best applicability {best[1].score:.3f} < {self._matcher.config.abstain_below}"
                    if best
                    else "no candidates"
                ),
            )

        applicable.sort(key=lambda item: item[1].score, reverse=True)
        (top_candidate, top_match, top_envelopes) = applicable[0]

        # 6) Conflict check: top-1/top-2 close but different suggestions.
        if len(applicable) >= 2:
            second_match = applicable[1][1]
            if abs(top_match.score - second_match.score) <= CONFLICT_SCORE_EPSILON:
                hint1 = self._hint(top_candidate)
                hint2 = self._hint(applicable[1][0])
                if hint1 and hint2 and hint1 != hint2:
                    return _verdict(
                        InterventionAction.ABSTAIN,
                        [REASON_CONFLICTING_MEMORIES],
                        memory_id=top_candidate.memory_id,
                        score=top_match.score,
                        explanation=(
                            f"top-2 within {CONFLICT_SCORE_EPSILON} but suggest different "
                            f"actions ({hint1!r} vs {hint2!r})"
                        ),
                    )

        # 7) Verified HOW rule match (rules outrank memory when they conflict).
        rule = self._rule_match(failure_signature)
        patch = self._patch_for(top_candidate, rule)

        # 8) Decision ladder.
        score = top_match.score
        validated = top_match.envelope_type == EnvelopeType.VALIDATED.value
        matched_envelope = next(
            (e for e in top_envelopes if e.envelope_id == top_match.matched_envelope_id),
            None,
        )
        success_evidence = bool(matched_envelope and matched_envelope.success_count > 0)
        evidence_confidence = max((e.confidence for e in top_envelopes), default=0.0)
        benefit = round(score * max(evidence_confidence, 0.0), 4)
        harm = self._harm(top_envelopes)

        if score >= self._matcher.config.suggest_below:
            if validated and success_evidence:
                if self._choreography is None:
                    # v4 §7.3: APPLY without a choreography gate is forbidden.
                    return _verdict(
                        InterventionAction.ABSTAIN,
                        [REASON_CHOREOGRAPHY_UNAVAILABLE],
                        memory_id=top_candidate.memory_id,
                        rule_id=rule,
                        score=score,
                        envelope_id=top_match.matched_envelope_id,
                        benefit=benefit,
                        harm=harm,
                        evidence_confidence=evidence_confidence,
                        explanation="APPLY requires the choreography validator (v4 §7.3)",
                    )
                choreography = self._validate_choreography(patch)
                if choreography is not None and not choreography.allowed:
                    return _verdict(
                        InterventionAction.ABSTAIN,
                        [f"choreography_violation:{v}" for v in choreography.violations],
                        memory_id=top_candidate.memory_id,
                        rule_id=rule,
                        score=score,
                        envelope_id=top_match.matched_envelope_id,
                        patch=patch,
                        benefit=benefit,
                        harm=1.0,
                        evidence_confidence=evidence_confidence,
                        explanation="choreography validator blocked the patch",
                    )
                return _verdict(
                    InterventionAction.APPLY,
                    [REASON_VALIDATED_MATCH],
                    memory_id=top_candidate.memory_id,
                    rule_id=rule,
                    score=score,
                    envelope_id=top_match.matched_envelope_id,
                    patch=patch,
                    benefit=benefit,
                    harm=harm,
                    evidence_confidence=evidence_confidence,
                    explanation="validated envelope + patch-proof evidence + regime match",
                )
            return _verdict(
                InterventionAction.SUGGEST,
                [REASON_NO_PATCHPROOF],
                memory_id=top_candidate.memory_id,
                rule_id=rule,
                score=score,
                envelope_id=top_match.matched_envelope_id,
                patch=patch,
                benefit=benefit,
                harm=harm,
                evidence_confidence=evidence_confidence,
                explanation=(
                    "applicability is high but no validated patch-proof evidence "
                    "exists — operator gate, never silent auto-apply"
                ),
            )

        return _verdict(
            InterventionAction.SUGGEST,
            [REASON_SUGGEST_BAND],
            memory_id=top_candidate.memory_id,
            rule_id=rule,
            score=score,
            envelope_id=top_match.matched_envelope_id,
            patch=patch,
            benefit=benefit,
            harm=harm,
            evidence_confidence=evidence_confidence,
            explanation=(
                f"{self._matcher.config.abstain_below} ≤ score {score:.3f} "
                f"< {self._matcher.config.suggest_below}"
            ),
        )

    # ------------------------------------------------------------------

    def _validate_choreography(self, patch: dict[str, Any] | None) -> Any | None:
        """Run the choreography validator when the patch carries parameters.

        A parameterless patch is timing-vacuous: allowed by construction but
        recorded as such.  A patch with parameters must pass the contract
        (v4 §8.4) — violations are returned to the caller for ABSTAIN.
        """
        validator = self._choreography
        if validator is None or patch is None:
            return None
        parameters = patch.get("parameters") or {}
        if not parameters:
            return None
        if self._timing_model is None:
            # Never validate against a synthetic empty model: it fakes a
            # zero current cooldown and skips the stacking check (review
            # finding).  No real timing model → the budget is unprovable.
            from rosclaw.how.choreography.validator import ChoreographyValidation

            return ChoreographyValidation(
                allowed=False,
                violations=["choreography_unavailable:no_timing_model"],
            )
        return validator.validate(parameters, self._timing_model)

    def _retrieval_confidence(self, response: Any) -> float:
        if not response.candidates:
            return 0.0
        base = 1.0 if not response.fallback else 0.6
        top = response.candidates[0]
        if top.exact_entity_match:
            base *= 1.0
        else:
            base *= 0.8
        return round(base, 4)

    def _safety_escalation(self, failure_signature: str) -> str | None:
        try:
            from rosclaw.how.intervention.safety_router import diagnose_safety, is_blocking

            diagnosis = diagnose_safety(failure_signature)
            if diagnosis and is_blocking(diagnosis.strategy):
                return f"{diagnosis.symptom}:{diagnosis.severity}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("safety escalation check failed: %s", exc)
        return None

    def _rule_match(self, failure_signature: str) -> str | None:
        if self._engine is None:
            return None
        try:
            import asyncio

            suggestion = asyncio.run(self._engine.suggest_recovery(failure_signature, None))
            if suggestion:
                return str(suggestion.get("rule_id") or "") or None
        except Exception as exc:  # noqa: BLE001
            logger.debug("rule match failed: %s", exc)
        return None

    def _hint(self, candidate: Any) -> str:
        item = candidate.item
        if item is None:
            return ""
        return str(item.metadata.get("recovery_hint") or "")

    def _patch_for(self, candidate: Any, rule_id: str | None) -> dict[str, Any] | None:
        item = candidate.item
        if item is None:
            return None
        hint = self._hint(candidate)
        if not hint and rule_id is None:
            return None
        parameters = item.metadata.get("patch_parameters")
        return {
            "type": "memory_recovery_hint",
            "description": hint,
            "parameters": parameters if isinstance(parameters, dict) else {},
            "source_memory_id": candidate.memory_id,
            "supporting_rule_id": rule_id,
        }

    def _harm(self, envelopes: list[Any]) -> float:
        harm = 0.0
        for envelope in envelopes:
            if envelope.envelope_type == EnvelopeType.CONTRAINDICATED.value:
                harm = max(harm, 0.5 + 0.5 * envelope.confidence)
            elif envelope.evidence_count > 0:
                failure_share = envelope.failure_count / envelope.evidence_count
                harm = max(harm, failure_share * envelope.confidence)
        return round(harm, 4)
