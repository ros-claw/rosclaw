"""Native observations do not need fabricated navigation roles, URDF or Body."""

from datetime import timedelta

import pytest

from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from tests.connectors.ros.test_operational_graph_contracts import NOW, native_fixture


def native_graph(m, **kwargs):
    return compile_operational_graph(
        m.seal(), source_ref="fixture://native-only", now=NOW, **kwargs
    )


def test_native_only_graph_never_calls_navigation_compiler_or_invents_context(monkeypatch):
    from rosclaw.connectors.ros.intelligence import operational_graph

    m, _, _ = native_fixture()
    monkeypatch.setattr(
        operational_graph,
        "compile_engineering_case",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no Nav2 context requested")),
    )
    g = native_graph(m)
    assert g.case_hash is g.source_urdf_sha256 is g.requirements_hash is None
    assert not any(e.kind in {"urdf", "body", "run", "world"} for e in g.entities)
    assert g.source_status == "UNKNOWN" and g.current_authority == "NONE"
    assert g.dataflows[0].observed_policy_status == "VIOLATED"
    assert not any(
        f.id in {"F-TF-001", "F-LIFE-001", "F-GEOM-001", "F-ROUTE-001"} for f in g.findings
    )
    assert RosOperationalGraphV1.from_dict(g.model_dump(mode="json")) == g


@pytest.mark.parametrize("age,status", [(10, "SATISFIED"), (900, "VIOLATED"), (None, "UNKNOWN")])
def test_native_receive_observer_window_and_bounded_hypotheses(age, status):
    m, _, _ = native_fixture()
    for signal in m.signals:
        signal.source = "native:monotonic_receive"
        if signal.topic == "/sample/scan2":
            signal.last_message_age_ms = age
    g = native_graph(m)
    f = next(
        f
        for f in g.findings
        if f.invariant == "signal.observed_steady_receive_window" and f.targets == ["/sample/scan2"]
    )
    assert f.status == status
    p = plan_diagnostic_probes(g, issue_ids=[f.id], budget_sec=5)
    assert p.execution_authority == "NONE"
    assert len(p.hypotheses) == (0 if status == "SATISFIED" else 4)


def test_latched_native_age_is_not_a_stream_gap_explanation():
    m, _, _ = native_fixture()
    s = next(s for s in m.signals if s.topic == "/sample/scan2")
    s.source = "native:monotonic_receive"
    s.freshness_policy = "latched"
    s.last_message_age_ms = 900
    g = native_graph(m)
    f = next(
        f
        for f in g.findings
        if f.invariant == "signal.observed_steady_receive_window" and f.targets == [s.topic]
    )
    assert f.status == "UNKNOWN"
    p = plan_diagnostic_probes(g, issue_ids=[f.id])
    assert next(h.status for h in p.hypotheses if h.predicate == "RECEIVE_GAP") == "UNKNOWN"


@pytest.mark.parametrize(
    "context", ["urdf_only", "requirements_only", "review_without_context", "run_without_context"]
)
def test_native_context_cannot_be_partially_supplied_or_promoted(context):
    m, u, r = native_fixture()
    kwargs = (
        {"urdf_bytes": u}
        if context == "urdf_only"
        else {"requirements": r}
        if context == "requirements_only"
        else {"source_review": {"result": "MATCHED"}}
        if context == "review_without_context"
        else {"expected_run_id": "claimed-run"}
    )
    with pytest.raises(ValueError):
        native_graph(m, **kwargs)


def test_native_graph_rejects_an_added_body_context_without_context_hashes():
    m, _, _ = native_fixture()
    data = native_graph(m).model_dump(mode="json")
    data["source_status"] = "MATCHED"
    with pytest.raises(ValueError, match="cannot claim Body"):
        RosOperationalGraphV1.model_validate(data)


def test_native_stale_envelope_cannot_witness_current_qos_or_receive_window():
    m, _, _ = native_fixture()
    m.captured_at = NOW - timedelta(seconds=1)
    g = native_graph(m)
    assert g.snapshot_freshness == "STALE" and all(
        flow.observed_policy_status == "UNKNOWN" for flow in g.dataflows
    )
    assert all(
        f.status == "UNKNOWN"
        for f in g.findings
        if f.invariant == "signal.observed_steady_receive_window"
    )
