"""Synthetic real-method graph failure contracts; never physical acceptance."""

import ast
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def reads(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2")
    )
    return importlib.import_module("graph_reads")


def test_disappearing_node_keeps_partial_truth_and_reports_incomplete_graph(reads):
    calls = []

    def actions(name, namespace):
        calls.append((name, namespace))
        if name == "vanished":
            raise RuntimeError("Failed to get action server names and type: error not set")
        return [("/robot/go", ["nav2_msgs/action/NavigateToPose"])]

    nodes, observed, errors = reads.collect_node_actions(
        lambda: [("vanished", "/robot"), ("remaining", "/robot")], actions
    )
    assert len(calls) == len(nodes) == 2
    assert observed == [{"name": "/robot/go", "action_type": "nav2_msgs/action/NavigateToPose"}]
    assert (
        errors[0]["node"] == "/robot/vanished"
        and errors[0]["operation"] == "get_action_server_names_and_types_by_node"
    )
    assert "error not set" in errors[0]["message"]


def test_failed_node_enumeration_is_explicit_unknown_without_action_facts(reads):
    def fail():
        raise RuntimeError("discovery read failed")

    nodes, actions, errors = reads.collect_node_actions(
        fail, lambda *_: pytest.fail("no node input")
    )
    assert nodes == actions == [] and len(errors) == 1
    assert errors[0]["operation"] == "get_node_names_and_namespaces"


def test_unrelated_programming_errors_are_not_silently_replaced(reads):
    with pytest.raises(TypeError):
        reads.collect_node_actions(
            lambda: [("node", "/scope")],
            lambda *_: (_ for _ in ()).throw(TypeError("invalid implementation")),
        )


def test_actual_publisher_method_outputs_unknown_on_read_failure_and_can_observe_again(reads):
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    tree = ast.parse(path.read_text())
    method = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "ReadOnlyProbe"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "publish_snapshot"
    )
    namespace = {
        "json": json,
        "os": os,
        "unknown_graph_snapshot": reads.unknown_graph_snapshot,
        "utc_now": lambda: "2026-10-15T00:00:00+00:00",
        "get_rmw_implementation_identifier": lambda: "synthetic-rmw",
        "String": lambda **kw: SimpleNamespace(**kw),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    outputs = []

    def fail():
        raise RuntimeError("original ROS graph read failed")

    node = SimpleNamespace(
        discover_reads=fail,
        snapshot=lambda: pytest.fail("failed discovery must not return old cached health"),
        publisher=SimpleNamespace(publish=lambda message: outputs.append(json.loads(message.data))),
        status_publisher=SimpleNamespace(publish=lambda _: None),
    )
    namespace["publish_snapshot"](node)
    first = outputs[0]
    assert not any(first["completeness"].values()) and first["graph"]["actions"] == []
    assert (
        first["observations"]["graph_read_errors"][0]["message"] == "original ROS graph read failed"
    )
    node.discover_reads = lambda: None
    node.snapshot = lambda: {"captured_at": "new-original-read", "completeness": {"graph": True}}
    namespace["publish_snapshot"](node)
    assert outputs[1]["captured_at"] == "new-original-read"
    assert first["completeness"]["graph"] is False
