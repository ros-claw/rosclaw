"""Synthetic causal-diagnostic protocol tests, never independent physics claims."""

from datetime import timedelta

import pytest

from rosclaw.connectors.ros.engineering.diagnostic_experiment import (
    DiagnosticAssessmentV1,
    RosDiagnosticExperimentV1,
)
from rosclaw.connectors.ros.engineering.probe_planner import (
    assess_diagnostic_followup,
    plan_diagnostic_probes,
)
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_operational_graph_contracts import NOW, graph, native_fixture


def capture(*, stamp=NOW, qos=True, gap=True, mixed=False):
    m, u, r = native_fixture()
    m.captured_at = stamp
    for node in m.observations["node_use_sim_time"]:
        m.observations["parameter_captured_at"][node] = stamp.isoformat()
    m.completeness["time"] = True
    m.observations["node_use_sim_time"]["/sample/costmap"] = not mixed
    m.observations["node_parameters"]["/sample/costmap"]["use_sim_time"] = not mixed
    m.qos["endpoints"][0]["reliability"] = "BEST_EFFORT" if qos else "RELIABLE"
    for s in m.signals:
        s.captured_at = stamp
        s.source = "native:monotonic"
        if s.topic == "/sample/scan2":
            s.last_message_age_ms = 900 if gap else 10
    for edge in m.transforms:
        edge.captured_at = stamp
    for state in m.lifecycle:
        state.captured_at = stamp
    return graph(m, u, r, now=stamp)


def statuses(hypotheses):
    return {h.predicate: h.status for h in hypotheses}


def reseal_graph(data):
    from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1

    obj = RosOperationalGraphV1.model_validate({**data, "graph_hash": ""})
    payload = obj.model_dump(mode="json")
    payload.pop("graph_hash")
    return RosOperationalGraphV1.from_dict(
        {**payload, "graph_hash": content_hash("rosopgraph", payload)}
    )


def test_same_scan_symptom_competing_observations_and_contradiction():
    before = capture()
    plan = plan_diagnostic_probes(before, issue_ids=["F-SCAN-001"], budget_sec=30)
    assert len(plan.hypotheses) == 4
    assert statuses(plan.hypotheses)["QOS_CONFLICT"] == "SUPPORTED_OBSERVATION"
    assert statuses(plan.hypotheses)["RECEIVE_GAP"] == "SUPPORTED_OBSERVATION"
    later = NOW + timedelta(seconds=1)
    # Same missing-receive symptom, but the independent followup resolves QoS
    # negative and now witnesses a clock mismatch. Do not keep the old QoS story.
    after = capture(stamp=later, qos=False, mixed=True)
    result = assess_diagnostic_followup(plan, before, after, now=later)
    values = statuses(result.hypotheses)
    assert result.status == "OBSERVATIONAL_UPDATE"
    assert values["QOS_CONFLICT"] == "CONTRADICTED_OBSERVATION"
    assert values["CLOCK_CONFLICT"] == "SUPPORTED_OBSERVATION"
    assert values["UNRESOLVED_PATH"] == "UNKNOWN"
    assert (
        next(h.id for h in plan.hypotheses if h.predicate == "QOS_CONFLICT")
        in result.withdrawn_hypothesis_ids
    )
    assert result.probe_execution == "NOT_ATTESTED_BY_GRAPH"
    assert result.causal_verification == "NOT_ESTABLISHED"
    assert DiagnosticAssessmentV1.from_dict(result.model_dump(mode="json")) == result


def test_fresh_stream_does_not_prove_consumer_delivery():
    before = capture()
    plan = plan_diagnostic_probes(before, issue_ids=["F-SCAN-001"], budget_sec=30)
    later = NOW + timedelta(seconds=1)
    after = capture(stamp=later, qos=True, gap=False)
    result = assess_diagnostic_followup(plan, before, after, now=later)
    assert statuses(result.hypotheses)["RECEIVE_GAP"] == "CONTRADICTED_OBSERVATION"
    assert statuses(result.hypotheses)["QOS_CONFLICT"] == "SUPPORTED_OBSERVATION"
    assert statuses(result.hypotheses)["UNRESOLVED_PATH"] == "UNKNOWN"


@pytest.mark.parametrize(
    "qos,mixed,operation",
    [
        (True, False, "ENDPOINT_QOS"),
        (False, False, "NATIVE_DIAGNOSTICS"),
        (False, True, "CLOCK_TF_WINDOW"),
    ],
)
def test_same_receive_gap_selects_different_measurements_with_one_window(qos, mixed, operation):
    p = plan_diagnostic_probes(
        capture(qos=qos, mixed=mixed), issue_ids=["F-SCAN-001"], budget_sec=5
    )
    assert len(p.probes) == 1 and p.probes[0].request.operation == operation
    assert p.total_duration_sec == 5 and p.deferred_hypothesis_ids


@pytest.mark.parametrize("budget,limit", [(1, 8), (4, 8), (5, 1), (9, 8), (30, 1), (30, 8)])
def test_budget_ranking_and_explicit_deferral(budget, limit):
    g = capture()
    p = plan_diagnostic_probes(g, issue_ids=["F-SCAN-001"], budget_sec=budget, max_probes=limit)
    assert p.total_duration_sec <= budget and len(p.probes) <= limit
    assert p == RosDiagnosticExperimentV1.from_dict(p.model_dump(mode="json"))
    planned = {row.hypothesis_id for probe in p.probes for row in probe.predictions}
    assert planned | set(p.deferred_hypothesis_ids) == {h.id for h in p.hypotheses}
    if budget < 5:
        assert not p.probes and len(p.deferred_hypothesis_ids) == 4
    assert all(probe.request.execution_authority == "NONE" for probe in p.probes)


@pytest.mark.parametrize("change", ["source", "body", "requirements", "clock", "same", "budget"])
def test_followup_fail_closed(change):
    before = capture()
    p = plan_diagnostic_probes(before, issue_ids=["F-SCAN-001"], budget_sec=10)
    later = NOW + timedelta(seconds=1)
    after = capture(stamp=later)
    data = after.model_dump(mode="json")
    if change == "source":
        data["source_ref"] = "fixture:wrong-source"
    elif change == "body":
        next(e for e in data["entities"] if e["kind"] == "urdf")["name"] = "other"
        data["source_urdf_sha256"] = "other"
    elif change == "requirements":
        data["requirements_hash"] = "other"
    elif change == "same":
        data = before.model_dump(mode="json")
    elif change == "budget":
        data["reviewed_at"] = (NOW + timedelta(seconds=11)).isoformat()
    data.pop("graph_hash")
    after = reseal_graph(data)
    result = assess_diagnostic_followup(
        p, before, after, now=later + timedelta(seconds=1) if change == "clock" else later
    )
    assert result.status == "ABSTAIN" and result.reasons
    assert all(h.status == "UNKNOWN" for h in result.hypotheses)


def test_reusing_old_fact_inside_new_snapshot_does_not_count_as_probe_result():
    before = capture()
    p = plan_diagnostic_probes(before, issue_ids=["F-SCAN-001"], budget_sec=30)
    data = before.model_dump(mode="json")
    data["snapshot_hash"] = "new-envelope-but-old-observations"
    data["reviewed_at"] = (NOW + timedelta(milliseconds=100)).isoformat()
    data.pop("graph_hash")
    after = reseal_graph(data)
    result = assess_diagnostic_followup(p, before, after, now=NOW + timedelta(milliseconds=100))
    assert result.status == "ABSTAIN"
    assert result.reasons == ["no_current_discriminating_evidence"]


def test_sealed_prediction_mutation_rejected_even_after_rehash():
    before = capture()
    p = plan_diagnostic_probes(before, issue_ids=["F-SCAN-001"])
    data = p.model_dump(mode="json")
    data["probes"][0]["predictions"][0]["if_supported"] = "fabricated"
    with pytest.raises(ValueError, match="hash mismatch"):
        RosDiagnosticExperimentV1.from_dict(data)
    data.pop("experiment_hash")
    forged = RosDiagnosticExperimentV1.from_dict(
        {**data, "experiment_hash": content_hash("rosdiagnostic", data)}
    )
    with pytest.raises(ValueError, match="deterministic original"):
        assess_diagnostic_followup(
            forged,
            before,
            capture(stamp=NOW + timedelta(seconds=1)),
            now=NOW + timedelta(seconds=1),
        )


def test_tf_explanations_do_not_turn_missing_authority_into_uniqueness():
    m, u, r = native_fixture()
    m.transforms = [e for e in m.transforms if e.parent != "world"]
    g = graph(m, u, r)
    p = plan_diagnostic_probes(g, issue_ids=["F-TF-001"])
    assert len(p.hypotheses) == 4
    assert statuses(p.hypotheses)["TF_AUTHORITY_CONFLICT"] == "UNKNOWN"
    assert any(probe.request.provider == "PROVIDER_EXTENSION_REQUIRED" for probe in p.probes)
    assert all(h.causal_verification == "NOT_ESTABLISHED" for h in p.hypotheses)


def test_same_tf_symptom_time_conflict_then_attributed_authority_conflict():
    from copy import deepcopy

    def tf_capture(stamp, *, duplicate=False):
        m, u, r = native_fixture()
        m.captured_at = stamp
        for node in m.observations["node_use_sim_time"]:
            m.observations["parameter_captured_at"][node] = stamp.isoformat()
        m.completeness["time"] = True
        for e in m.transforms:
            e.captured_at = stamp
            e.age_ms = 10 if duplicate else 900
        edge = next(e for e in m.transforms if not e.static)
        edge.authority = "publisher-one"
        if duplicate:
            other = deepcopy(edge)
            other.authority = "publisher-two"
            m.transforms.append(other)
        return graph(m, u, r, now=stamp)

    before = tf_capture(NOW)
    p = plan_diagnostic_probes(before, issue_ids=["F-TF-001"], budget_sec=30)
    assert statuses(p.hypotheses)["TF_TIME_CONFLICT"] == "SUPPORTED_OBSERVATION"
    later = NOW + timedelta(seconds=1)
    after = tf_capture(later, duplicate=True)
    result = assess_diagnostic_followup(p, before, after, now=later)
    assert result.status == "OBSERVATIONAL_UPDATE"
    assert statuses(result.hypotheses)["TF_TIME_CONFLICT"] == "CONTRADICTED_OBSERVATION"
    assert statuses(result.hypotheses)["TF_AUTHORITY_CONFLICT"] == "SUPPORTED_OBSERVATION"


def test_pure_planner_never_opens_transport_or_files(monkeypatch):
    import builtins
    import os
    import socket
    import subprocess

    before = capture()

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic planning must be pure")

    with monkeypatch.context() as ctx:
        ctx.setattr(builtins, "open", forbidden)
        ctx.setattr(os, "open", forbidden)
        ctx.setattr(socket, "socket", forbidden)
        ctx.setattr(subprocess, "Popen", forbidden)
        p = plan_diagnostic_probes(before, issue_ids=["F-SCAN-001"])
        assert p.execution_authority == "NONE"


def test_unknown_issue_does_not_disappear_as_no_target_issue():
    g = capture()
    p = plan_diagnostic_probes(g, issue_ids=["F-SOURCE-001"])
    assert p.status == "NEEDS_READ_ONLY_PROBE"
    assert len(p.hypotheses) == 1 and p.hypotheses[0].status == "UNKNOWN"
    assert p.probes[0].request.provider == "OPERATOR_SOURCE_REVIEW"
    assert p.probes[0].request.operation == "SOURCE_CAPTURE"


@pytest.mark.parametrize(
    "kwargs",
    [{"budget_sec": True}, {"budget_sec": 31}, {"max_probes": 0}, {"issue_ids": ["invented"]}],
)
def test_invalid_protocol_parameters(kwargs):
    with pytest.raises(ValueError):
        plan_diagnostic_probes(capture(), **kwargs)
