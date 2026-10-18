"""Fresh envelopes cannot renew cached per-node parameter or ROS clock samples."""

from datetime import timedelta

import pytest

from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from tests.connectors.ros.test_operational_graph_contracts import NOW, graph, native_fixture


@pytest.mark.parametrize(
    "stamp",
    [
        None,
        "not-a-time",
        "2026-10-18T05:00:00",
        (NOW - timedelta(seconds=3)).isoformat(),
        (NOW + timedelta(milliseconds=1)).isoformat(),
    ],
)
def test_clock_parameter_readback_must_have_its_own_aware_current_time(stamp):
    m, u, r = native_fixture()
    m.observations["parameter_captured_at"]["/sample/lidar"] = stamp
    g = graph(m, u, r)
    assert g.snapshot_freshness == "FRESH"
    assert next(f.status for f in g.findings if f.id == "OP-TIME-DOMAINS") == "UNKNOWN"
    assert all(
        f.status == "UNKNOWN"
        for f in g.findings
        if f.invariant == "tf.observed_dynamic_time_window"
    )
    assert all(t.clock_domain == "UNKNOWN" for t in g.transforms)
    plan = plan_diagnostic_probes(g, issue_ids=["OP-TIME-DOMAINS"])
    assert next(h.status for h in plan.hypotheses if h.predicate == "CLOCK_CONFLICT") == "UNKNOWN"


@pytest.mark.parametrize("age", [None, -1, 501, 3000])
def test_parameter_agreement_does_not_make_stale_sim_clock_current(age):
    m, u, r = native_fixture()
    m.observations["clock_last_receive_age_ms"] = age
    g = graph(m, u, r)
    assert next(f.status for f in g.findings if f.id == "OP-TIME-DOMAINS") == "SATISFIED"
    assert next(f.status for f in g.findings if f.id == "OP-CLOCK-PROGRESS") == "UNKNOWN"
    assert all(
        f.status == "UNKNOWN"
        for f in g.findings
        if f.invariant == "tf.observed_dynamic_time_window"
    )


def test_clock_finding_and_dynamic_age_link_actual_parameter_values_and_read_times():
    m, u, r = native_fixture()
    g = graph(m, u, r)
    finding = next(f for f in g.findings if f.id == "OP-TIME-DOMAINS")
    pointers = [g.provenance[ref].pointer for ref in finding.evidence_refs]
    assert "/observations/node_use_sim_time/~1sample~1lidar" in pointers
    assert "/observations/parameter_captured_at" in pointers
    ref = next(
        g.provenance[ref]
        for ref in finding.evidence_refs
        if g.provenance[ref].pointer.endswith("~1sample~1lidar")
    )
    assert ref.observed_at == NOW
    assert finding.status == "SATISFIED"


def test_wall_clock_does_not_require_a_sim_clock_stream():
    m, u, r = native_fixture()
    m.observations["node_use_sim_time"] = dict.fromkeys(m.observations["node_use_sim_time"], False)
    for node in m.observations["node_use_sim_time"]:
        m.observations["node_parameters"][node]["use_sim_time"] = False
    m.observations.pop("clock_last_receive_age_ms")
    g = graph(m, u, r)
    assert next(f.status for f in g.findings if f.id == "OP-TIME-DOMAINS") == "SATISFIED"
    assert all(t.clock_domain == "ROS_SYSTEM" for t in g.transforms)


def test_sim_clock_age_includes_time_since_native_capture():
    m, u, r = native_fixture()
    m.observations["clock_last_receive_age_ms"] = 450
    g = graph(m, u, r, now=NOW + timedelta(milliseconds=100))
    assert next(f.status for f in g.findings if f.id == "OP-TIME-DOMAINS") == "SATISFIED"
    assert next(f.status for f in g.findings if f.id == "OP-CLOCK-PROGRESS") == "UNKNOWN"


def test_conflicting_parameter_and_projected_clock_values_abstain():
    m, u, r = native_fixture()
    m.observations["node_parameters"]["/sample/lidar"]["use_sim_time"] = False
    g = graph(m, u, r)
    assert next(f.status for f in g.findings if f.id == "OP-TIME-DOMAINS") == "UNKNOWN"


def test_old_v1_envelope_only_clock_provenance_is_not_upgraded_by_hypothesis_engine():
    from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1
    from rosclaw.contracts.common import content_hash

    m, u, r = native_fixture()
    m.observations["node_use_sim_time"]["/sample/lidar"] = False
    m.observations["node_parameters"]["/sample/lidar"]["use_sim_time"] = False
    g = graph(m, u, r)
    data = g.model_dump(mode="json")
    finding = next(f for f in data["findings"] if f["id"] == "OP-TIME-DOMAINS")
    assert finding["status"] == "VIOLATED"
    # Simulate the older sealed v1 contract: only one envelope-level value ref.
    finding["evidence_refs"] = finding["evidence_refs"][:1]
    data.pop("graph_hash")
    old = RosOperationalGraphV1.from_dict({**data, "graph_hash": content_hash("rosopgraph", data)})
    p = plan_diagnostic_probes(old, issue_ids=["OP-TIME-DOMAINS"])
    assert next(h.status for h in p.hypotheses if h.predicate == "CLOCK_CONFLICT") == "UNKNOWN"
