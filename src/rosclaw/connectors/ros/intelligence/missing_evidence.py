"""Deterministic, bounded requests for evidence; plans have no executable commands."""

from rosclaw.contracts.common import content_hash

from .graph_contracts import MissingEvidencePlanV1, ReadOnlyProbeRequestV1, RosOperationalGraphV1


def describe_missing_evidence(graph: RosOperationalGraphV1, *, max_probes=8, budget_sec=30):
    """Propose read-only measurements, including unavailable provider extensions.

    No probe is run here. Violations are not automatically assigned a root cause.
    Source mismatch blocks ordinary diagnostic continuation until source review.
    """
    if (
        type(max_probes) is not int
        or not 1 <= max_probes <= 8
        or type(budget_sec) is not int
        or not 1 <= budget_sec <= 30
    ):
        raise ValueError("probe budget requires 1..8 requests and 1..30 seconds")
    graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
    selected = [f for f in graph.findings if f.status == "UNKNOWN"]
    grouped = {}
    for finding in selected:
        if finding.id == "F-SOURCE-001" or graph.source_status == "MISMATCH":
            op, provider = "SOURCE_CAPTURE", "OPERATOR_SOURCE_REVIEW"
        elif finding.invariant.startswith("qos."):
            op, provider = "ENDPOINT_QOS", "EXISTING_NATIVE_READ_ONLY_PROBE"
            if any(
                marker in " ".join(finding.missing_evidence)
                for marker in ("lease_duration", "complete_", "full_", "GID")
            ):
                provider = "PROVIDER_EXTENSION_REQUIRED"
        elif finding.invariant == "tf.single_parent_and_authority":
            op, provider = "TF_AUTHORITY_WINDOW", "PROVIDER_EXTENSION_REQUIRED"
        elif finding.invariant.startswith(("time.", "tf.")) or finding.id == "F-TF-001":
            op, provider = "CLOCK_TF_WINDOW", "EXISTING_NATIVE_READ_ONLY_PROBE"
        else:
            op, provider = "NATIVE_DIAGNOSTICS", "EXISTING_NATIVE_READ_ONLY_PROBE"
        if "authorit" in " ".join(finding.missing_evidence):
            provider = "PROVIDER_EXTENSION_REQUIRED"
        bucket = grouped.setdefault(
            (op, provider), {"findings": [], "targets": set(), "evidence": set()}
        )
        bucket["findings"].append(finding.id)
        bucket["targets"].update(finding.targets)
        bucket["evidence"].update(finding.missing_evidence or [finding.invariant])
    requests = []
    deferred = []
    remaining = budget_sec
    if graph.source_status == "MISMATCH":
        # Include the violated source binding itself; no ordinary continuation.
        grouped = {
            ("SOURCE_CAPTURE", "OPERATOR_SOURCE_REVIEW"): {
                "findings": ["F-SOURCE-001"],
                "targets": {graph.source_ref},
                "evidence": {"new_independent_matching_source_run_Body_snapshot"},
            }
        }
    for (op, provider), values in sorted(
        grouped.items(), key=lambda item: (item[0][0] != "SOURCE_CAPTURE", item[0])
    ):
        if remaining == 0 or len(requests) >= max_probes:
            deferred.extend(values["findings"])
            continue
        targets = sorted(values["targets"])
        evidence = sorted(values["evidence"])
        # Explicitly defer oversized groups instead of truncating target coverage.
        if len(targets) > 64 or len(evidence) > 64 or len(values["findings"]) > 128:
            deferred.extend(values["findings"])
            continue
        seconds = min(5, remaining)
        payload = {
            "operation": op,
            "targets": targets or [graph.source_ref],
            "finding_ids": sorted(values["findings"]),
            "required_evidence": evidence,
            "provider": provider,
            "max_duration_sec": seconds,
        }
        requests.append(
            ReadOnlyProbeRequestV1(
                id=content_hash("opprobe", {"graph_hash": graph.graph_hash, **payload}), **payload
            )
        )
        remaining -= seconds
    status = (
        "BLOCKED_SOURCE_MISMATCH"
        if graph.source_status == "MISMATCH"
        else "NEEDS_READ_ONLY_PROBE"
        if selected
        else "NO_MISSING_EVIDENCE"
    )
    plan = MissingEvidencePlanV1(
        plan_hash="",
        graph_hash=graph.graph_hash,
        snapshot_hash=graph.snapshot_hash,
        source_ref=graph.source_ref,
        status=status,
        requests=requests,
        deferred_finding_ids=sorted(deferred),
        total_budget_sec=budget_sec - remaining,
    )
    data = plan.model_dump(mode="json")
    data.pop("plan_hash")
    return plan.model_copy(update={"plan_hash": content_hash("opprobeplan", data)})
