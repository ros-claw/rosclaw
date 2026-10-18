"""Exercise real sidecar callback logic without starting ROS in the test agent."""

import ast
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as Obj

import pytest


def observer():
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    tree = ast.parse(path.read_text())
    helper = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "observed_gid"
    )
    method = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "observe"
    )
    ns = {
        "TF_ATTRIBUTION_TOPIC": "/rosclaw_probe/tf_attribution",
        "deque": deque,
        "time": time,
        "utc_now": lambda: "2026-10-18T02:00:00+00:00",
    }
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"), ns)
    obj = Obj(
        samples={},
        edges={},
        tf_cache_drops=0,
        tf_message_window=deque(maxlen=1024),
        tf_window_drops=0,
    )
    obj.observe = lambda message, info=None: ns["observe"](obj, "/remapped/tf", message, info)
    return obj, ns


def message(child="odom"):
    vector = Obj(x=0.0, y=0.0, z=0.0, w=1.0)
    return Obj(
        transforms=[
            Obj(
                header=Obj(frame_id="world", stamp=Obj(sec=1, nanosec=0)),
                child_frame_id=child,
                transform=Obj(translation=vector, rotation=vector),
            )
        ]
    )


def test_same_tf_edge_distinct_actual_publishers_are_not_collapsed():
    obj, _ = observer()
    obj.observe(message(), {"publisher_gid": bytes([1] * 16)})
    obj.observe(message(), {"publisher_gid": bytes([2] * 16)})
    assert len(obj.edges) == 2
    assert {v["authority"] for v in obj.edges.values()} == {
        bytes([1] * 16).hex(),
        bytes([2] * 16).hex(),
    }
    assert len(obj.tf_message_window) == 2


def test_jazzy_metadata_without_gid_keeps_raw_events_but_authority_unknown():
    obj, _ = observer()
    obj.observe(message(), {"publication_sequence_number": 1})
    obj.observe(message(), {"publication_sequence_number": 2})
    assert len(obj.edges) == 1 and next(iter(obj.edges.values()))["authority"] is None
    assert [r["publication_sequence_number"] for r in obj.tf_message_window] == [1, 2]
    assert all(r["publisher_gid"] is None for r in obj.tf_message_window)


def test_high_cardinality_window_has_explicit_drop_accounting():
    obj, _ = observer()
    for i in range(1029):
        obj.observe(message(f"frame-{i}"))
    assert len(obj.edges) == len(obj.tf_message_window) == 1024
    assert obj.tf_cache_drops == obj.tf_window_drops == 5


@pytest.mark.parametrize("value", [None, [], [0] * 24, [256], [-1], [True], "made-up", bytes(65)])
def test_missing_or_invalid_rmw_identity_is_not_fabricated(value):
    _, ns = observer()
    assert ns["observed_gid"](value) is None


def test_discovery_callback_requires_message_info_to_avoid_silent_one_arg_dispatch():
    import inspect

    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    tree = ast.parse(path.read_text())
    callback = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Lambda)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "observe"
            for c in ast.walk(n)
        )
    )
    fn = eval(
        compile(ast.Expression(body=callback), str(path), "eval"),
        {"topic": "/renamed/tf", "self": Obj(observe=lambda *args: args)},
    )
    with pytest.raises(TypeError):
        inspect.signature(fn).bind(object())
    inspect.signature(fn).bind(object(), object())
    assert fn("message", "metadata") == ("/renamed/tf", "message", "metadata")
