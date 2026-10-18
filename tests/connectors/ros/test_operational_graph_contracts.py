"""Native-shaped synthetic cases and pure source replay, not physical fault trials."""

import json
import subprocess
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1
from rosclaw.connectors.ros.intelligence.missing_evidence import describe_missing_evidence
from rosclaw.connectors.ros.intelligence.operational_graph import (
    compile_operational_graph,
    summarize_operational_graph,
)
from rosclaw.connectors.ros.intelligence.provenance_edges import pointer_value
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_engineering_case_compiler import NOW, setup


def native_fixture():
    m, u, r = setup()
    m.observations["node_use_sim_time"] = {"/sample/lidar": True, "/sample/costmap": True}
    for node in m.observations["node_use_sim_time"]:
        m.observations["parameter_captured_at"][node] = NOW.isoformat()
        m.observations["node_parameters"][node] = {"use_sim_time": True}
    m.qos["endpoints"] = [
        {
            "topic": "/sample/scan2",
            "node": "/sample/lidar",
            "kind": "publisher",
            "gid": "publisher-gid",
            "reliability": "BEST_EFFORT",
            "durability": "VOLATILE",
            "liveliness": "AUTOMATIC",
            "deadline_ns": 100000000,
        },
        {
            "topic": "/sample/scan2",
            "node": "/sample/costmap",
            "kind": "subscriber",
            "gid": "subscriber-gid",
            "reliability": "RELIABLE",
            "durability": "VOLATILE",
            "liveliness": "AUTOMATIC",
            "deadline_ns": 200000000,
        },
    ]
    m.completeness["tf"] = True
    return m.seal(), u, r


def graph(m, u, r, **kwargs):
    return compile_operational_graph(
        m.seal(), u, r, source_ref="fixture:sealed-original", now=kwargs.pop("now", NOW), **kwargs
    )


def test_graph_hash_original_pointers_no_mutation_and_exact_old_five_verdicts():
    m, u, r = native_fixture()
    before = deepcopy((m.to_dict(), r))
    g = graph(m, u, r)
    c = compile_engineering_case(
        m, u, r, source_ref="fixture:sealed-original", now=NOW, evidence_domain="SEALED_REPLAY"
    )
    assert [(f.id, f.status) for f in g.findings[: len(c.constraints)]] == [
        (f.id, f.status) for f in c.constraints
    ]
    assert graph(m, u, r) == g
    assert g.current_authority == "NONE" and g.current_l1 == "NOT_EVALUATED"
    assert (m.to_dict(), r) == before
    for ref in g.provenance.values():
        doc = c.model_dump(mode="json") if ref.document_hash == c.case_hash else m.to_dict()
        assert ref.value_hash == content_hash("opvalue", pointer_value(doc, ref.pointer))
    assert RosOperationalGraphV1.from_dict(g.model_dump(mode="json")) == g
    data = g.model_dump(mode="json")
    data["findings"][0]["status"] = "SATISFIED"
    with pytest.raises(ValueError, match="hash mismatch"):
        RosOperationalGraphV1.from_dict(data)


def test_qos_same_sensor_rate_does_not_imply_consumer_delivery_or_full_compatibility():
    m, u, r = native_fixture()
    g = graph(m, u, r)
    assert g.dataflows[0].observed_policy_status == "VIOLATED"
    assert g.dataflows[0].full_compatibility == "UNKNOWN"
    assert g.dataflows[0].delivery_status == "NOT_MEASURED"
    assert (
        next(
            e for e in g.entities if e.name == "/sample/scan2" and e.kind == "topic"
        ).interface_type
        == "sensor_msgs/msg/LaserScan"
    )
    plan = describe_missing_evidence(g)
    assert any(p.operation == "ENDPOINT_QOS" for p in plan.requests)
    assert all(p.execution_status == "PROPOSED_NOT_EXECUTED" for p in plan.requests)


@pytest.mark.parametrize(
    "offered,requested,expected",
    [
        ("BEST_EFFORT", "RELIABLE", "VIOLATED"),
        ("RELIABLE", "BEST_EFFORT", "SATISFIED"),
        ("SYSTEM_DEFAULT", "RELIABLE", "UNKNOWN"),
        ("UNKNOWN", "RELIABLE", "UNKNOWN"),
    ],
)
def test_directional_qos_reliability_and_unresolved_defaults(offered, requested, expected):
    m, u, r = native_fixture()
    m.qos["endpoints"][0]["reliability"] = offered
    m.qos["endpoints"][1]["reliability"] = requested
    assert graph(m, u, r).dataflows[0].observed_policy_status == expected


@pytest.mark.parametrize(
    "field,pub,sub",
    [
        ("durability", "VOLATILE", "TRANSIENT_LOCAL"),
        ("liveliness", "AUTOMATIC", "MANUAL_BY_TOPIC"),
        ("deadline_ns", 300000000, 200000000),
    ],
)
def test_each_observed_qos_incompatibility_can_disprove_pair(field, pub, sub):
    m, u, r = native_fixture()
    m.qos["endpoints"][0]["reliability"] = "RELIABLE"
    m.qos["endpoints"][0][field] = pub
    m.qos["endpoints"][1][field] = sub
    assert graph(m, u, r).dataflows[0].observed_policy_status == "VIOLATED"


def test_namespace_remap_and_duplicate_node_only_endpoints_never_merge():
    m, u, r = native_fixture()
    m.qos["endpoints"][1]["topic"] = "/other/scan2"
    assert not graph(m, u, r).dataflows
    m, u, r = native_fixture()
    for e in m.qos["endpoints"]:
        e.pop("gid")
    m.qos["endpoints"].append(deepcopy(m.qos["endpoints"][0]))
    g = graph(m, u, r)
    assert len(g.endpoints) == 3 and len({e.id for e in g.endpoints}) == 3
    assert all(f.observed_policy_status == "UNKNOWN" for f in g.dataflows)


@pytest.mark.parametrize("age,status", [(-6000, "VIOLATED"), (700, "VIOLATED"), (20, "SATISFIED")])
def test_tf_time_window_is_not_tf_connectivity(age, status):
    m, u, r = native_fixture()
    m.transforms[0].age_ms = age
    g = graph(m, u, r)
    assert (
        next(f for f in g.findings if f.invariant == "tf.observed_dynamic_time_window").status
        == status
    )


def test_mixed_missing_stale_clocks_cannot_become_good_tf_time():
    m, u, r = native_fixture()
    m.observations["node_use_sim_time"]["/sample/costmap"] = False
    m.observations["node_parameters"]["/sample/costmap"]["use_sim_time"] = False
    g = graph(m, u, r)
    assert next(f for f in g.findings if f.id == "OP-TIME-DOMAINS").status == "VIOLATED"
    assert all(
        f.status == "UNKNOWN"
        for f in g.findings
        if f.invariant == "tf.observed_dynamic_time_window"
    )
    m.observations["node_use_sim_time"] = {}
    assert all(t.clock_domain == "UNKNOWN" for t in graph(m, u, r).transforms)
    for delta in (-1, 1):
        g = graph(m, u, r, now=NOW + timedelta(seconds=delta))
        assert all(f.status == "UNKNOWN" for f in g.findings)


def test_duplicate_tf_authority_witness_violates_but_missing_identity_stays_unknown():
    m, u, r = native_fixture()
    m.transforms[0] = m.transforms[0].model_copy(update={"authority": "/sample/localization_a"})
    m.transforms.append(m.transforms[0].model_copy(update={"authority": "/sample/localization_b"}))
    g = graph(m, u, r)
    assert any(
        f.invariant == "tf.single_parent_and_authority" and f.status == "VIOLATED"
        for f in g.findings
    )
    m, u, r = native_fixture()
    g = graph(m, u, r)
    assert all(
        f.status == "UNKNOWN" for f in g.findings if f.invariant == "tf.single_parent_and_authority"
    )
    assert any(
        p.operation == "TF_AUTHORITY_WINDOW" and p.provider == "PROVIDER_EXTENSION_REQUIRED"
        for p in describe_missing_evidence(g).requests
    )


def test_source_mismatch_probe_budgets_and_deferred_targets():
    m, u, r = native_fixture()
    g = graph(m, u, r, source_review={"result": "MISMATCH"})
    assert next(f for f in g.findings if f.id == "F-SOURCE-001").status == "VIOLATED"
    p = describe_missing_evidence(g)
    assert p.status == "BLOCKED_SOURCE_MISMATCH" and len(p.requests) == 1
    assert p.requests[0].operation == "SOURCE_CAPTURE"
    g = graph(m, u, r)
    p = describe_missing_evidence(g, max_probes=1, budget_sec=2)
    assert len(p.requests) == 1 and p.total_budget_sec == 2 and p.deferred_finding_ids
    assert p == describe_missing_evidence(g, max_probes=1, budget_sec=2)
    for invalid in (0, 9, True):
        with pytest.raises(ValueError):
            describe_missing_evidence(g, max_probes=invalid)


def test_bounds_content_integrity_and_explicit_partial_agent_projection():
    m, u, r = native_fixture()
    g = graph(m, u, r)
    view = summarize_operational_graph(g, max_characters=1200)
    assert len(json.dumps(view, ensure_ascii=False, separators=(",", ":"))) <= 1200
    assert view["omitted_findings"] > 0
    assert view["omitted_findings"] + len(view["findings"]) == len(g.findings)
    m.qos["endpoints"] *= 600
    with pytest.raises(ValueError, match="bounded native endpoint"):
        graph(m, u, r)
    m, u, r = native_fixture()
    m.snapshot_hash = "tampered"
    with pytest.raises(ValueError, match="snapshot hash"):
        compile_operational_graph(m, u, r, source_ref="x", now=NOW)
    with pytest.raises(ValueError, match="aware"):
        graph(*native_fixture(), now=NOW.replace(tzinfo=None))


def test_core_and_graph_imports_need_no_ros_and_cannot_open_transport():
    root = Path(__file__).resolve().parents[3]
    code = """import builtins,sys
original=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in {'rclpy','rospy','rosbag2_py'}: raise AssertionError(name)
 return original(name,*a,**k)
builtins.__import__=guarded
import rosclaw
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from rosclaw.connectors.ros.intelligence.missing_evidence import describe_missing_evidence
assert not any(x in sys.modules for x in ('rclpy','rospy','rosbag2_py'))
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_static_tf_is_not_rejected_as_dynamic_age_and_clock_facts_are_separate():
    m, u, r = native_fixture()
    m.transforms[0].static = True
    m.transforms[0].age_ms = None
    g = graph(m, u, r)
    dynamic_ids = {
        f.evidence_refs[0] for f in g.findings if f.invariant == "tf.observed_dynamic_time_window"
    }
    assert g.transforms[0].evidence_refs[0] not in dynamic_ids
    assert any(t.message_age_ms is not None for t in g.time_facts)
    assert all(t.domain == "UNKNOWN" for t in g.time_facts if t.receive_age_ms is not None)


def test_invalid_native_names_schema_and_graph_cross_links_fail_closed():
    m, u, r = native_fixture()
    m.qos["endpoints"][0]["topic"] = "scan"
    with pytest.raises(ValueError, match="fully-qualified"):
        graph(m, u, r)
    g = graph(*native_fixture())
    d = g.model_dump(mode="json")
    d["schema_version"] = "rosclaw.ros_operational_graph.v2"
    with pytest.raises(ValueError):
        RosOperationalGraphV1.from_dict(d)
    d = g.model_dump(mode="json")
    d["dataflows"][0]["subscriber_id"] = d["dataflows"][0]["publisher_id"]
    with pytest.raises(ValueError, match="publisher to subscriber"):
        RosOperationalGraphV1.from_dict(d)
    d = g.model_dump(mode="json")
    d["entities"][0]["evidence_refs"] = ["missing"]
    with pytest.raises(ValueError, match="unresolved provenance"):
        RosOperationalGraphV1.from_dict(d)


def test_probe_groups_do_not_claim_uncollected_targets_complete():
    g = graph(*native_fixture())
    plan = describe_missing_evidence(g, max_probes=1, budget_sec=1)
    covered = {f for p in plan.requests for f in p.finding_ids}
    assert covered | set(plan.deferred_finding_ids) == {
        f.id for f in g.findings if f.status == "UNKNOWN"
    }
    assert not covered & set(plan.deferred_finding_ids)
    assert all(p.execution_authority == "NONE" and p.side_effects == "NONE" for p in plan.requests)


def test_qos_zero_default_deadline_never_claims_full_or_partial_compatibility():
    m, u, r = native_fixture()
    m.qos["endpoints"][0]["reliability"] = "RELIABLE"
    m.qos["endpoints"][0]["deadline_ns"] = 0
    flow = graph(m, u, r).dataflows[0]
    assert flow.observed_policy_status == "UNKNOWN" and flow.full_compatibility == "UNKNOWN"


def test_pure_compilation_and_probe_planning_cannot_touch_files_network_or_processes(monkeypatch):
    import builtins
    import socket

    m, u, r = native_fixture()

    def forbidden(*args, **kwargs):
        pytest.fail("read-only graph compiler must not perform I/O or execute probes")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "write_text", forbidden)
        patch.setattr(Path, "write_bytes", forbidden)
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(subprocess, "Popen", forbidden)
        g = graph(m, u, r)
        p = describe_missing_evidence(g)
        assert p.requests and p.execution_authority == "NONE"


def test_observed_world_run_body_are_refs_and_never_new_verified_authority():
    m, u, r = native_fixture()
    source = {
        "schema_version": "rosclaw.sim_source_binding_evidence.v1",
        "result": "MATCHED",
        "ros_runtime_snapshot_hash": m.snapshot_hash,
        "sim_run_id": "historical-run",
        "body_hash": "historical-body",
        "world_id": "historical-world",
        "observed_at": NOW.isoformat(),
        "reviewed_at": NOW.isoformat(),
    }
    g = graph(m, u, r, source_review=source)
    assert {(e.kind, e.name) for e in g.entities} >= {
        ("run", "historical-run"),
        ("body", "historical-body"),
        ("world", "historical-world"),
    }
    assert g.current_l1 == "NOT_EVALUATED" and g.current_authority == "NONE"


def test_long_identities_and_cartesian_pair_explosion_are_rejected_not_truncated():
    m, u, r = native_fixture()
    m.qos["endpoints"][0]["topic"] = "/" + "x" * 513
    with pytest.raises(ValueError, match="bounded explicit"):
        graph(m, u, r)
    m, u, r = native_fixture()
    pub, sub = deepcopy(m.qos["endpoints"])
    m.qos["endpoints"] = [dict(pub, gid=f"p-{i}", node=f"/node/p{i}") for i in range(65)] + [
        dict(sub, gid=f"s-{i}", node=f"/node/s{i}") for i in range(65)
    ]
    with pytest.raises(ValueError, match="pair budget"):
        graph(m, u, r)


@pytest.mark.parametrize(
    "fault",
    [
        "oversized_URDF",
        "oversized_requirements",
        "oversized_source_review",
        "nonobject_requirements",
    ],
)
def test_direct_python_api_enforces_input_bounds_before_old_case_compilation(fault):
    m, u, r = native_fixture()
    kwargs = {}
    if fault == "oversized_URDF":
        u = b"x" * 5_000_001
    elif fault == "oversized_requirements":
        r = {"large": "x" * 5_000_001}
    elif fault == "oversized_source_review":
        kwargs["source_review"] = {"large": "x" * 5_000_001}
    else:
        r = []
    with pytest.raises(ValueError, match="bounded|exceeds 5MB"):
        graph(m, u, r, **kwargs)
