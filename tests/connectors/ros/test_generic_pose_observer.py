"""Passive observer source binding/geometry contracts; no live DDS or physics."""

import importlib.util
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.connectors.ros import test_sim_runtime_policy as policy_tests

runtime = policy_tests.runtime
sources = policy_tests.sources
frozen_runtime = policy_tests.frozen_runtime


@pytest.fixture
def observer(monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("generic_pose_test", root / "pose_observer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def message(model="fresh_model", yaw=0.7):
    return SimpleNamespace(
        transforms=[
            SimpleNamespace(
                child_frame_id=model,
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=1.2, y=-0.4, z=0.1),
                    rotation=SimpleNamespace(
                        x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
                    ),
                ),
                header=SimpleNamespace(stamp=SimpleNamespace(sec=120, nanosec=42)),
            )
        ]
    )


def test_passive_observer_selects_exact_model_and_retains_source_identity(observer):
    binding = {
        "model": "fresh_model",
        "topic": "/robot/world_pose",
        "run_id": "run",
        "runtime_policy_hash": "frozen",
    }
    sample = observer.pose_sample(message(), binding)
    assert sample["x"] == 1.2 and sample["yaw"] == pytest.approx(0.7)
    assert sample["source_topic"] == binding["topic"]
    assert sample["run_id"] == "run" and sample["runtime_policy_hash"] == "frozen"
    assert observer.pose_sample(message("foreign"), binding) is None


@pytest.mark.parametrize("fault", ["duplicate", "nan", "boolean", "quaternion", "sec", "ns"])
def test_bad_actual_pose_is_refused_without_synthetic_sample(observer, fault):
    value = message()
    transform = value.transforms[0]
    if fault == "duplicate":
        value.transforms.append(deepcopy(transform))
    elif fault == "nan":
        transform.transform.translation.x = float("nan")
    elif fault == "boolean":
        transform.transform.translation.y = True
    elif fault == "quaternion":
        transform.transform.rotation.w = 0
    elif fault == "sec":
        transform.header.stamp.sec = -1
    else:
        transform.header.stamp.nanosec = 1_000_000_000
    with pytest.raises(ValueError):
        observer.pose_sample(value, {"model": "fresh_model", "topic": "/pose"})


def test_generic_pose_requires_reopened_policy_and_never_uses_profiles(
    observer, frozen_runtime, monkeypatch
):
    import sys

    root, policy = frozen_runtime
    monkeypatch.setitem(sys.modules, "profiles", None)
    binding = observer.observer_binding(root)
    assert binding["topic"] == "/different/robot/measured_world_pose"
    assert binding["model"] == "unselected_synthetic_model"
    assert binding["runtime_policy_hash"] == policy["artifact_hash"]
    (root / "sim_runtime_policy.json").unlink()
    with pytest.raises(ValueError):
        observer.observer_binding(root)
    assert not (root / "fixture_profile.json").exists()


@pytest.mark.parametrize("fault", ["missing", "wrong_type", "duplicate", "stale"])
def test_generic_policy_requires_unique_fresh_typed_pose_stream(runtime, fault):
    from rosclaw.connectors.ros.context.sim_runtime_policy import prepare_sim_runtime_policy
    from tests.connectors.ros.test_sim_execution_interfaces import NOW

    root, admission, model, declaration = runtime
    name = declaration["topics"]["independent_pose"]
    row = next(t for t in model.graph["topics"] if t["name"] == name)
    signal = next(t for t in model.signals if t.topic == name)
    if fault == "missing":
        declaration["topics"].pop("independent_pose")
    elif fault == "wrong_type":
        row["msg_type"] = "geometry_msgs/msg/Pose"
    elif fault == "duplicate":
        model.graph["topics"].append(deepcopy(row))
    else:
        signal.last_message_age_ms = 300
    model.seal()
    admission["source_snapshot_hash"] = model.snapshot_hash
    with pytest.raises(ValueError):
        prepare_sim_runtime_policy(root, admission, model, declaration, now=NOW)


@pytest.mark.parametrize("config", [None, {}, {"body_id": "legacy"}])
def test_legacy_policy_import_needs_no_core_dependencies(observer, tmp_path, monkeypatch, config):
    import json
    import sys

    if config is not None:
        (tmp_path / "execution_config.json").write_text(json.dumps(config))
    monkeypatch.setitem(sys.modules, "rosclaw", None)
    assert observer.load_frozen_sim_runtime_policy(tmp_path) is None


@pytest.mark.parametrize(
    "config", ["[]", "{", " " * 2_000_001, '{"generic_execution_proposal":{}}']
)
def test_malformed_or_generic_missing_policy_refuses_legacy_fallback(observer, tmp_path, config):
    (tmp_path / "execution_config.json").write_text(config)
    with pytest.raises(ValueError):
        observer.load_frozen_sim_runtime_policy(tmp_path)
