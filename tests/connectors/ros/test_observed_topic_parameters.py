"""Relative names need captured expansion evidence and actual graph matching."""

import ast
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from rosclaw.connectors.ros.context.sim_binding import propose_sim_fixture_binding
from rosclaw.connectors.ros.intelligence.topic_parameters import observed_topic_parameter
from tests.connectors.ros.test_sim_binding_proposal import NOW, example


def add_expansion(model):
    node = "/sample/collision_monitor"
    params = model.observations["node_parameters"][node]
    params.update(
        {"ranger.topic": "scan2", "cmd_vel_in_topic": "smoothed", "cmd_vel_out_topic": "guarded"}
    )
    model.observations["resolved_topic_parameters"] = {
        node: {
            key: {
                "raw_value": raw,
                "expanded_name": "/sample/" + raw,
                "node_name": node,
                "source": node + "/get_parameters",
                "captured_at": NOW.isoformat(),
                "complete": True,
                "method": "rclpy.expand_topic_name_and_validate_full_topic_name",
                "remapping_applied": False,
            }
            for key, raw in params.items()
            if key in ("ranger.topic", "cmd_vel_in_topic", "cmd_vel_out_topic")
        }
    }
    return node


def test_relative_topic_expansion_with_same_parameter_source_and_graph_connections_is_usable():
    model, data, cleaner, policy = example()
    add_expansion(model)
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "READY_FOR_SIM_WORKSPACE_COMPILATION"
    from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities

    model.body = {
        "effective_body_hash": "synthetic_compiled_binding",
        "frames": result["specification"]["frames"],
        "ros_capability_bindings": result["specification"]["ros_capability_bindings"],
    }
    model.seal()
    monitor = next(
        r
        for r in resolve_capabilities(model, now=NOW)
        if r["semantic_id"] == "safety.collision_monitor"
    )
    assert monitor["status"] == "AVAILABLE" and not monitor["usable_for_real_execution"]
    assert (
        result["specification"]["ros_capability_bindings"]["safety.collision_monitor"][
            "sensor_topic"
        ]
        == "/sample/scan2"
    )


@pytest.mark.parametrize(
    "fault",
    ["missing", "wrong_source", "different_raw", "stale", "incomplete", "remapped", "wrong_node"],
)
def test_missing_or_mismatched_expansion_is_unknown_instead_of_namespace_guess(fault):
    model, data, cleaner, policy = example()
    node = add_expansion(model)
    record = model.observations["resolved_topic_parameters"][node]["ranger.topic"]
    if fault == "missing":
        model.observations.pop("resolved_topic_parameters")
    elif fault == "wrong_source":
        record["source"] = "/other/get_parameters"
    elif fault == "different_raw":
        record["raw_value"] = "other"
    elif fault == "stale":
        record["captured_at"] = (NOW - timedelta(seconds=6)).isoformat()
    elif fault == "incomplete":
        record["complete"] = False
    elif fault == "remapped":
        record["remapping_applied"] = True
    else:
        record["node_name"] = "/other"
    assert observed_topic_parameter(model, node, "ranger.topic", now=NOW) is None
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "UNKNOWN" and "specification" not in result


def test_expansion_cannot_replace_missing_actual_monitor_connection():
    model, data, cleaner, policy = example()
    add_expansion(model)
    next(t for t in model.graph["topics"] if t["name"] == "/sample/guarded")["publishers"] = [
        "/different"
    ]
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    assert (
        propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)[
            "status"
        ]
        == "UNKNOWN"
    )


def test_ros_host_delegates_expansion_and_validation_using_actual_server_node_identity():
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_observed_topic_parameters"
    )
    calls = []

    def expand(raw, name, namespace):
        calls.append((raw, name, namespace))
        return "/sample/scan2"

    def validate(name):
        calls.append(("validate", name))

    scope = {"expand_topic_name": expand, "validate_full_topic_name": validate}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    raw = {"ranger.topic": "scan2", "secret_token": "ignored"}
    before = deepcopy(raw)
    result = scope["resolve_observed_topic_parameters"](
        "/sample/collision_monitor", raw, NOW.isoformat()
    )
    assert calls == [("scan2", "collision_monitor", "/sample"), ("validate", "/sample/scan2")]
    assert raw == before and set(result) == {"ranger.topic"}
    assert result["ranger.topic"]["complete"] is True
    assert result["ranger.topic"]["remapping_applied"] is False
