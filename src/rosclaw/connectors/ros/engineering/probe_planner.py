"""Budgeted measurement proposals and source-bound observational falsification."""

from datetime import datetime

from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import ReadOnlyProbeRequestV1
from .diagnostic_experiment import (
    DiagnosticAssessmentV1,
    DiagnosticPredictionV1,
    DiagnosticProbeV1,
    RosDiagnosticExperimentV1,
)
from .hypothesis import checked_graph, evaluate_hypothesis, propose_hypotheses

PROBES = {
    "RECEIVE_GAP": (
        "NATIVE_DIAGNOSTICS",
        "EXISTING_NATIVE_READ_ONLY_PROBE",
        "steady receive age and bounded message window",
    ),
    "QOS_CONFLICT": (
        "ENDPOINT_QOS",
        "EXISTING_NATIVE_READ_ONLY_PROBE",
        "target endpoint requested/offered reliability durability liveliness deadline",
    ),
    "CLOCK_CONFLICT": (
        "CLOCK_TF_WINDOW",
        "EXISTING_NATIVE_READ_ONLY_PROBE",
        "per-node use_sim_time plus ROS clock and steady receive times",
    ),
    "TF_TIME_CONFLICT": (
        "CLOCK_TF_WINDOW",
        "EXISTING_NATIVE_READ_ONLY_PROBE",
        "dynamic TF stamps aligned ROS clock and declared future tolerance",
    ),
    "TF_AUTHORITY_CONFLICT": (
        "TF_AUTHORITY_WINDOW",
        "PROVIDER_EXTENSION_REQUIRED",
        "complete TF parent child publisher attribution window",
    ),
    "UNRESOLVED_PATH": (
        "NATIVE_DIAGNOSTICS",
        "PROVIDER_EXTENSION_REQUIRED",
        "consumer delivery frame extrinsic broadcast readback and executor evidence",
    ),
}

PREDICTIONS = {
    "RECEIVE_GAP": (
        "steady receive age plus capture age exceeds max_age_ms",
        "steady receive age plus capture age is within max_age_ms",
    ),
    "QOS_CONFLICT": (
        "at least one current target endpoint pair violates requested/offered policy ordering",
        "all observed target endpoint pairs resolve compatible for the four measured policies",
    ),
    "CLOCK_CONFLICT": (
        "current observed per-node use_sim_time values disagree",
        "current observed per-node use_sim_time values are all resolved and agree",
    ),
    "TF_TIME_CONFLICT": (
        "aligned dynamic TF age is outside [-declared_future_tolerance_ms, max_age_ms]",
        "all observed target dynamic TF edges are within their declared window",
    ),
    "TF_AUTHORITY_CONFLICT": (
        "a current child has more than one parent or attributed publisher authority",
        "requires complete publisher attribution unavailable in this graph version; remains UNKNOWN",
    ),
    "UNRESOLVED_PATH": (
        "requires consumer-specific or parameter/executor evidence not evaluated by this graph version",
        "absence of other explanations cannot reject this open explanation; remains UNKNOWN",
    ),
}


def plan_diagnostic_probes(graph, *, issue_ids=None, budget_sec=20, max_probes=8):
    """Recheck supported explanations first, then unknowns; no likelihoods."""
    if type(budget_sec) is not int or not 1 <= budget_sec <= 30:
        raise ValueError("budget requires 1..30 seconds")
    if type(max_probes) is not int or not 1 <= max_probes <= 8:
        raise ValueError("max_probes requires 1..8")
    graph = checked_graph(graph)
    hypotheses = propose_hypotheses(graph, issue_ids=issue_ids)
    blocked = graph.source_status == "MISMATCH" or graph.snapshot_freshness != "FRESH"
    grouped = {}
    if not blocked:
        for h in hypotheses:
            operation, provider, evidence = PROBES[h.predicate]
            if h.issue_id == "F-SOURCE-001":
                operation, provider, evidence = (
                    "SOURCE_CAPTURE",
                    "OPERATOR_SOURCE_REVIEW",
                    "independent current source run World and Body identity review",
                )
            grouped.setdefault((operation, provider), []).append((h, evidence))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][1] != "EXISTING_NATIVE_READ_ONLY_PROBE",
            -sum(h.status == "SUPPORTED_OBSERVATION" for h, _ in item[1]),
            -sum(h.status == "UNKNOWN" for h, _ in item[1]),
            -len(item[1]),
            item[0],
        ),
    )
    probes, planned = [], set()
    remaining = budget_sec
    for (operation, provider), rows in ranked:
        # A five-second measurement is never advertised as equivalent to a
        # shortened one. Unaffordable or oversized groups are explicitly deferred.
        targets = sorted({t for h, _ in rows for t in h.targets}) or [graph.source_ref]
        if remaining < 5 or len(probes) >= max_probes or len(targets) > 64:
            continue
        payload = {
            "operation": operation,
            "provider": provider,
            "targets": targets,
            "finding_ids": sorted({h.issue_id for h, _ in rows}),
            "required_evidence": sorted({e for _, e in rows}),
            "max_duration_sec": 5,
        }
        predictions = [
            DiagnosticPredictionV1(
                hypothesis_id=h.id,
                if_supported="source identity remains unknown; no admission"
                if h.issue_id == "F-SOURCE-001"
                else PREDICTIONS[h.predicate][0],
                if_contradicted="independent identity review required; never inferred by eliminating other hypotheses"
                if h.issue_id == "F-SOURCE-001"
                else PREDICTIONS[h.predicate][1],
            )
            for h, _ in rows
        ]
        request = ReadOnlyProbeRequestV1(
            id=content_hash("rosdiagnosticprobe", {"graph_hash": graph.graph_hash, **payload}),
            **payload,
        )
        probes.append(
            DiagnosticProbeV1(
                request=request,
                predictions=predictions,
                discriminated_hypotheses=len(predictions),
            )
        )
        planned.update(h.id for h, _ in rows)
        remaining -= 5
    experiment = RosDiagnosticExperimentV1(
        experiment_hash="",
        graph_hash=graph.graph_hash,
        snapshot_hash=graph.snapshot_hash,
        source_ref=graph.source_ref,
        proposed_at=graph.reviewed_at,
        evidence_domain=graph.evidence_domain,
        status="BLOCKED_SOURCE_OR_TIME"
        if blocked
        else "NEEDS_READ_ONLY_PROBE"
        if hypotheses
        else "NO_TARGET_ISSUE",
        hypotheses=hypotheses,
        probes=probes,
        deferred_hypothesis_ids=sorted(h.id for h in hypotheses if h.id not in planned),
        budget_sec=budget_sec,
        max_probes=max_probes,
        total_duration_sec=budget_sec - remaining,
    )
    data = experiment.model_dump(mode="json")
    data.pop("experiment_hash")
    return RosDiagnosticExperimentV1.from_dict(
        {**data, "experiment_hash": content_hash("rosdiagnostic", data)}
    )


def _identity(graph):
    return sorted((e.kind, e.name) for e in graph.entities if e.kind in {"run", "world", "body"})


def assess_diagnostic_followup(experiment, before, after, *, now: datetime):
    """Compare independent newer captures. This does not attest probe execution.

    Historical evidence must remain SEALED_REPLAY. A hash is an integrity check,
    not authentication or proof that a native sidecar actually ran a request.
    """
    experiment = RosDiagnosticExperimentV1.from_dict(experiment.model_dump(mode="json"))
    before, after = checked_graph(before), checked_graph(after)
    if now.tzinfo is None:
        raise ValueError("aware assessment clock required")
    if experiment.graph_hash != before.graph_hash:
        raise ValueError("experiment before graph binding mismatch")
    # Recompute the sealed proposal: rehashing a malicious prediction/target must
    # not permit it to become a planner-issued experiment.
    issue_ids = sorted({h.issue_id for h in experiment.hypotheses})
    if issue_ids:
        expected = plan_diagnostic_probes(
            before,
            issue_ids=issue_ids,
            budget_sec=experiment.budget_sec,
            max_probes=experiment.max_probes,
        )
        if expected != experiment:
            raise ValueError("experiment does not match deterministic original proposal")
    reasons = []
    if experiment.status == "BLOCKED_SOURCE_OR_TIME":
        reasons.append("blocked_original_proposal")
    if any(
        getattr(before, key) != getattr(after, key)
        for key in (
            "source_ref",
            "source_urdf_sha256",
            "requirements_hash",
            "evidence_domain",
        )
    ) or _identity(before) != _identity(after):
        reasons.append("source_run_world_body_or_requirements_changed")
    if after.source_status == "MISMATCH" or after.snapshot_freshness != "FRESH":
        reasons.append("unusable_followup_source_or_snapshot")
    if after.snapshot_hash == before.snapshot_hash or after.reviewed_at <= experiment.proposed_at:
        reasons.append("followup_is_not_a_newer_independent_capture")
    if (after.reviewed_at - experiment.proposed_at).total_seconds() > experiment.budget_sec:
        reasons.append("followup_outside_experiment_budget")
    age_ms = (now - after.reviewed_at).total_seconds() * 1000
    if not 0 <= age_ms <= after.max_age_ms:
        reasons.append("assessment_clock_outside_followup_freshness")
    planned = {p.hypothesis_id for probe in experiment.probes for p in probe.predictions}
    updated = []
    for h in experiment.hypotheses:
        if reasons or h.id not in planned:
            updated.append(
                h.model_copy(
                    update={
                        "status": "UNKNOWN",
                        "supporting_evidence": [],
                        "contradicting_evidence": [],
                    }
                )
            )
        else:
            updated.append(evaluate_hypothesis(h, after, not_before=experiment.proposed_at))
    if not reasons and not any(h.status != "UNKNOWN" for h in updated):
        reasons.append("no_current_discriminating_evidence")
    withdrawn = [
        h.id
        for h in updated
        if h.status == "CONTRADICTED_OBSERVATION"
        and next(old for old in experiment.hypotheses if old.id == h.id).status
        == "SUPPORTED_OBSERVATION"
    ]
    assessment = DiagnosticAssessmentV1(
        assessment_hash="",
        experiment_hash=experiment.experiment_hash,
        before_graph_hash=before.graph_hash,
        after_graph_hash=after.graph_hash,
        status="ABSTAIN" if reasons else "OBSERVATIONAL_UPDATE",
        reasons=reasons,
        hypotheses=updated,
        withdrawn_hypothesis_ids=withdrawn,
    )
    data = assessment.model_dump(mode="json")
    data.pop("assessment_hash")
    return DiagnosticAssessmentV1.from_dict(
        {**data, "assessment_hash": content_hash("rosdiagnosticassessment", data)}
    )
