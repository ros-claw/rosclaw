"""Competing explanations scoped to observed targets; no probabilities or repair dispatch."""

from datetime import datetime

from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import RosOperationalGraphV1
from .diagnostic_experiment import RosHypothesisV1

# A falsified predicate withdraws this narrow explanation, not every possible
# failure in that subsystem. Multiple supported explanations can coexist.
CATALOG = {
    "RECEIVE_GAP": (
        "Sensor receive window is missing messages at the observer",
        ("producer failure, bridge pause and observer scheduling remain confounded",),
    ),
    "QOS_CONFLICT": (
        "Observed requested/offered QoS policy conflict on the target path",
        ("partial policies are not full RMW compatibility or consumer delivery",),
    ),
    "CLOCK_CONFLICT": (
        "Observed nodes disagree on ROS clock domain",
        ("global observed clock disagreement need not involve this consumer",),
    ),
    "TF_TIME_CONFLICT": (
        "Observed dynamic TF is outside its declared clock window",
        ("declared future tolerance must be respected; clocks must be aligned",),
    ),
    "TF_AUTHORITY_CONFLICT": (
        "Multiple parents or attributed authorities conflict in the TF window",
        ("missing publisher attribution cannot prove authority uniqueness",),
    ),
    "UNRESOLVED_PATH": (
        "Unmeasured consumer, frame, broadcast or executor path remains possible",
        ("absence of other explanations does not establish this explanation",),
    ),
}


def checked_graph(graph):
    return RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))


def _current_refs(graph, refs, not_before):
    return bool(refs) and all(
        graph.provenance[r].freshness == "FRESH"
        and 0
        <= (graph.reviewed_at - graph.provenance[r].observed_at).total_seconds() * 1000
        <= graph.max_age_ms
        and (not_before is None or graph.provenance[r].observed_at > not_before)
        for r in refs
    )


def evaluate_hypothesis(hypothesis, graph, *, not_before: datetime | None = None):
    """Re-evaluate a narrow predicate; incomplete negative coverage stays UNKNOWN."""
    predicate = hypothesis.predicate
    positives, negatives = [], []
    usable = graph.source_status != "MISMATCH" and graph.snapshot_freshness == "FRESH"
    if usable and predicate == "RECEIVE_GAP":
        for fact in graph.time_facts:
            if (
                fact.subject not in hypothesis.targets
                or fact.receive_clock != "STEADY"
                or fact.receive_age_ms is None
                or not _current_refs(graph, fact.evidence_refs, not_before)
            ):
                continue
            # A rate alone cannot attest present delivery; use steady receive age.
            capture_age = max(
                (graph.reviewed_at - graph.provenance[r].observed_at).total_seconds() * 1000
                for r in fact.evidence_refs
            )
            bucket = (
                positives if fact.receive_age_ms + capture_age > graph.max_age_ms else negatives
            )
            bucket.extend(fact.evidence_refs)
    elif usable:
        invariant = {
            "QOS_CONFLICT": "qos.observed_requested_offered_policies",
            "CLOCK_CONFLICT": "time.observed_node_clock_domains_agree",
            "TF_TIME_CONFLICT": "tf.observed_dynamic_time_window",
            "TF_AUTHORITY_CONFLICT": "tf.single_parent_and_authority",
        }.get(predicate)
        rows = [
            f
            for f in graph.findings
            if f.invariant == invariant
            and (predicate == "CLOCK_CONFLICT" or set(f.targets) & set(hypothesis.targets))
        ]
        current = [f for f in rows if _current_refs(graph, f.evidence_refs, not_before)]
        # One witnessed incompatibility suffices; a negative requires all observed
        # pairs/edges in this target scope resolved. Still no global coverage claim.
        for finding in current:
            if finding.status == "VIOLATED":
                positives.extend(finding.evidence_refs)
        if current and len(current) == len(rows) and all(f.status == "SATISFIED" for f in current):
            negatives = [r for f in current for r in f.evidence_refs]
    # Contradictory receive rows are not resolved by choosing the last value.
    status = (
        "UNKNOWN"
        if positives and negatives
        else "SUPPORTED_OBSERVATION"
        if positives
        else "CONTRADICTED_OBSERVATION"
        if negatives
        else "UNKNOWN"
    )
    if len(set(positives)) > 512 or len(set(negatives)) > 512:
        raise ValueError("diagnostic evidence scope too large; narrow the target issue")
    return hypothesis.model_copy(
        update={
            "status": status,
            "supporting_evidence": sorted(set(positives)),
            "contradicting_evidence": sorted(set(negatives)),
        }
    )


def propose_hypotheses(graph, *, issue_ids=None):
    graph = checked_graph(graph)
    selected = None if issue_ids is None else set(issue_ids)
    if selected is not None and (not selected or not selected <= {f.id for f in graph.findings}):
        raise ValueError("explicit issue IDs must resolve in the graph")
    hypotheses = []
    for finding in graph.findings:
        if finding.status == "SATISFIED" or (selected is not None and finding.id not in selected):
            continue
        if finding.id == "F-SCAN-001":
            predicates = ("RECEIVE_GAP", "QOS_CONFLICT", "CLOCK_CONFLICT", "UNRESOLVED_PATH")
        elif finding.id == "F-TF-001":
            predicates = (
                "TF_TIME_CONFLICT",
                "CLOCK_CONFLICT",
                "TF_AUTHORITY_CONFLICT",
                "UNRESOLVED_PATH",
            )
        elif finding.invariant == "qos.observed_requested_offered_policies":
            predicates = ("QOS_CONFLICT", "CLOCK_CONFLICT", "UNRESOLVED_PATH")
        elif finding.invariant.startswith(("tf.", "time.")):
            predicates = (
                "TF_TIME_CONFLICT",
                "CLOCK_CONFLICT",
                "TF_AUTHORITY_CONFLICT",
                "UNRESOLVED_PATH",
            )
        else:
            predicates = ("UNRESOLVED_PATH",)
        # Explicit failure instead of silently dropping diagnostic issues.
        if len(hypotheses) + len(predicates) > 64 or len(finding.targets) > 64:
            raise ValueError("diagnostic scope too large; select bounded issue_ids")
        # Legacy F-TF targets a localization node, while TF predicates need frames.
        targets = list(finding.targets)
        if finding.id == "F-TF-001":
            # The legacy finding does not identify missing frame endpoints. Avoid
            # pretending node names are frames: request a whole observed TF review.
            targets = sorted({e.name for e in graph.entities if e.kind == "frame"})
            if len(targets) > 64:
                raise ValueError("TF review scope too large")
        for predicate in predicates:
            description, confounders = CATALOG[predicate]
            if finding.id == "F-SOURCE-001":
                description = "Source identity requires independent operator review"
                confounders = (
                    "caller-supplied historical bindings do not prove current continuity",
                )
            h = RosHypothesisV1(
                id=content_hash(
                    "roshypothesis",
                    {"issue": finding.id, "predicate": predicate, "targets": targets},
                ),
                issue_id=finding.id,
                description=description,
                predicate=predicate,
                targets=targets,
                status="UNKNOWN",
                supporting_evidence=[],
                contradicting_evidence=[],
                confounders=confounders,
            )
            hypotheses.append(evaluate_hypothesis(h, graph))
    return hypotheses
