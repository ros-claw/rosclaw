"""Actual-origin fields are not topic publisher counts or uniqueness proofs."""

import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.connectors.ros.test_operational_graph_contracts import NOW, graph, native_fixture


def helper():
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/tf_window.py"
    spec = importlib.util.spec_from_file_location("native_tf_window", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_tf_window


def window(*, count=2, captured=NOW, recorded=NOW):
    return {
        "schema_version": "rosclaw.tf_attribution_window.v1",
        "captured_at": captured.isoformat(),
        "received_steady_ns": 1000,
        "window_ms": 500,
        "max_edges": 1024,
        "dropped_records": 0,
        "invalid_records": 0,
        "discovery_errors": 0,
        "discovery_current": True,
        "received_transform_callbacks": count,
        "completeness": "NOT_ESTABLISHED",
        "execution_authority": "NONE",
        "records": [
            {
                "topic": "/remapped/frames",
                "parent": "world",
                "child": "odom",
                "publisher_gid": ("01" if i == 0 else "02") * 16,
                "received_at": recorded.isoformat(),
                "received_steady_ns": 999,
                "stamp_sec": 1,
                "stamp_nanosec": 0,
            }
            for i in range(count)
        ],
    }


def test_native_message_info_window_can_witness_duplicate_authority_hidden_by_old_cache():
    m, u, r = native_fixture()
    data = helper()(json.dumps(window()), now=NOW, received_at=NOW)
    assert data["source_authentication"] == "NOT_ESTABLISHED_BY_PAYLOAD"
    m.observations["tf_attribution_window"] = data
    g = graph(m, u, r)
    assert any(
        f.status == "VIOLATED"
        for f in g.findings
        if f.invariant == "tf.observed_message_authorities_per_topic"
    )
    assert any("tf_attribution_window/records" in p.pointer for p in g.provenance.values())
    assert g.current_authority == "NONE"


def test_one_attributed_source_does_not_establish_uniqueness_or_complete_coverage():
    m, u, r = native_fixture()
    m.observations["tf_attribution_window"] = helper()(
        json.dumps(window(count=1)), now=NOW, received_at=NOW
    )
    g = graph(m, u, r)
    assert all(
        f.status == "UNKNOWN"
        for f in g.findings
        if f.invariant == "tf.observed_message_authorities_per_topic"
    )


@pytest.mark.parametrize("kind", ["old_envelope", "old_record", "future_envelope"])
def test_attribution_age_belongs_to_original_observation_not_new_envelope(kind):
    m, u, r = native_fixture()
    old = NOW - timedelta(seconds=3)
    payload = window(
        captured=old
        if kind == "old_envelope"
        else NOW + timedelta(seconds=1)
        if kind == "future_envelope"
        else NOW,
        recorded=old,
    )
    data = helper()(json.dumps(payload), now=NOW, received_at=NOW)
    # A malicious or obsolete wrapper cannot relabel original age as FRESH.
    data["status"] = "FRESH"
    m.observations["tf_attribution_window"] = data
    g = graph(m, u, r)
    assert not any(
        f.status == "VIOLATED"
        for f in g.findings
        if f.invariant == "tf.observed_message_authorities_per_topic"
    )


@pytest.mark.parametrize(
    "mutation", ["zero_gid", "malformed_gid", "invalid_stamp", "oversized", "authority", "complete"]
)
def test_invalid_or_authority_claiming_observer_payloads_fail_closed(mutation):
    d = window()
    if mutation == "zero_gid":
        d["records"][0]["publisher_gid"] = "00" * 16
    elif mutation == "malformed_gid":
        d["records"][0]["publisher_gid"] = "claimed-node-name"
    elif mutation == "invalid_stamp":
        d["records"][0]["stamp_nanosec"] = 10**9
    elif mutation == "oversized":
        d["records"] *= 1025
    elif mutation == "authority":
        d["execution_authority"] = "REAL"
    elif mutation == "complete":
        d["completeness"] = "COMPLETE"
    with pytest.raises(ValueError):
        helper()(json.dumps(d), now=NOW, received_at=NOW)


def test_separate_tf_topics_are_not_assumed_to_share_a_consumer_buffer():
    m, u, r = native_fixture()
    payload = window()
    payload["records"][1]["topic"] = "/another_robot/frames"
    m.observations["tf_attribution_window"] = helper()(
        json.dumps(payload), now=NOW, received_at=NOW
    )
    g = graph(m, u, r)
    assert all(
        f.status == "UNKNOWN"
        for f in g.findings
        if f.invariant == "tf.observed_message_authorities_per_topic"
    )
