"""Synthetic disposable-capture boundaries; no ROS, Docker or node launch."""

import importlib
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(directory))
    config = importlib.import_module("simulation_counterfactual_config")
    capture = importlib.import_module("simulation_engineering_capture")
    original = Path.is_file
    monkeypatch.setattr(
        capture.Path,
        "is_file",
        lambda path: True if str(path) == "/.dockerenv" else original(path),
    )
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    return config, capture


def settings(role="readonly_localized_observation", phase="before_repair"):
    return {
        "generic_source_observer": True,
        "fault": "none",
        "run_id": "a" * 32,
        "governed_sink": {"schema_version": "rosclaw.governed_sink_bootstrap.v1"},
        "engineering_capture": {
            "schema_version": "rosclaw.engineering_failure_capture.v1",
            "case": role,
            "phase": phase,
            "run_id": "a" * 32,
        },
    }


@pytest.mark.parametrize(
    "role,phase",
    [
        ("rsk_node_clock_mismatch", "before_repair"),
        ("rsk_node_clock_mismatch", "rollback"),
        ("readonly_preinitialization_observation", "before_repair"),
        ("readonly_localized_observation", "before_repair"),
    ],
)
def test_original_counterfactual_and_unadmitted_observation_roles(modules, role, phase):
    config, capture = modules
    value = settings(role, phase)
    assert config.validate_counterfactual_capture_context(value) is None
    assert capture.validate_engineering_capture(value) == value["engineering_capture"]
    assert role == "rsk_node_clock_mismatch" or role in capture.OBSERVATION_CASES


@pytest.mark.parametrize(
    "mutation",
    [
        "generic-false",
        "generic-numeric",
        "fault-injection",
        "rollback-observation",
        "after-repair",
        "run-mismatch",
        "boolean-run",
        "short-run",
        "sink-missing",
        "sink-schema",
        "sink-type",
        "capture-schema",
        "capture-extra",
        "capture-missing",
        "capture-type",
        "role-type",
        "phase-type",
        "positive-role",
        "driver-role",
        "other-fault-role",
    ],
)
def test_wrong_context_cannot_authorize_clock_overrides(modules, mutation):
    config, _ = modules
    value = settings()
    capture = value["engineering_capture"]
    if mutation.startswith("generic-"):
        value["generic_source_observer"] = False if mutation == "generic-false" else 1
    elif mutation == "fault-injection":
        value["fault"] = "world_reset"
    elif mutation == "rollback-observation":
        capture["phase"] = "rollback"
    elif mutation == "after-repair":
        capture["phase"] = "after_repair"
    elif mutation == "run-mismatch":
        capture["run_id"] = "b" * 32
    elif mutation in {"boolean-run", "short-run"}:
        value["run_id"] = capture["run_id"] = True if mutation == "boolean-run" else "short"
    elif mutation == "sink-missing":
        del value["governed_sink"]
    elif mutation == "sink-schema":
        value["governed_sink"]["schema_version"] = "unregistered"
    elif mutation == "sink-type":
        value["governed_sink"] = []
    elif mutation == "capture-schema":
        capture["schema_version"] = "unregistered"
    elif mutation == "capture-extra":
        capture["allow_goals"] = True
    elif mutation == "capture-missing":
        del value["engineering_capture"]
    elif mutation == "capture-type":
        value["engineering_capture"] = []
    elif mutation == "role-type":
        capture["case"] = {}
    elif mutation == "phase-type":
        capture["phase"] = []
    else:
        capture["case"] = {
            "positive-role": "positive",
            "driver-role": "driver_onboarding",
            "other-fault-role": "map_to_odom_missing",
        }[mutation]
    with pytest.raises(ValueError):
        config.validate_counterfactual_capture_context(value)


@pytest.mark.parametrize("boundary", ["host", "nonlocal"])
def test_isolated_local_sdk_is_still_required(modules, monkeypatch, boundary):
    config, capture = modules
    if boundary == "host":
        monkeypatch.setattr(capture.Path, "is_file", lambda _: False)
    else:
        monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0")
    with pytest.raises(ValueError):
        config.validate_counterfactual_capture_context(settings())


def test_default_clock_config_never_imports_capture_or_launches_anything(modules):
    config, _ = modules
    script = """import runpy,sys
module=runpy.run_path(sys.argv[1])
assert module['counterfactual_clock_parameters'](None,'controller_server') == {'use_sim_time':True}
assert 'simulation_engineering_capture' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", script, config.__file__], check=True, timeout=10)


def clock_originals(namespace="/example"):
    now = datetime(2026, 10, 19, tzinfo=UTC)
    descriptor = {
        "namespace": namespace,
        "counterfactual_sim": {
            "schema_version": "rosclaw.sim_counterfactual_launch.v1",
            "node_clock_overrides": {"controller_server": False, "planner_server": True},
        },
    }
    prefix = "" if namespace == "/" else namespace
    native = {
        "captured_at": now.isoformat(),
        "observations": {
            "node_parameters": {
                prefix + "/controller_server": {"use_sim_time": False},
                prefix + "/planner_server": {"use_sim_time": True},
            },
            "parameter_captured_at": {
                prefix + "/controller_server": (now - timedelta(milliseconds=500)).isoformat(),
                prefix + "/planner_server": now.isoformat(),
            },
        },
    }
    return native, descriptor, now


@pytest.mark.parametrize("namespace", ["/example", "", "/"])
def test_current_boolean_readbacks_are_required_for_all_declared_clocks(modules, namespace):
    config, _ = modules
    native, descriptor, _ = clock_originals(namespace)
    assert config.configured_clock_readback_available(native, descriptor)
    assert config.configured_clock_readback_available({}, {})


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-value",
        "numeric-value",
        "missing-node",
        "missing-second-node",
        "stale",
        "future",
        "naive-stamp",
        "naive-capture",
        "missing-stamp",
        "bad-stamp",
        "different-namespace",
        "missing-observations",
    ],
)
def test_stale_missing_or_incorrect_reads_cannot_be_replaced_by_launch_intent(modules, mutation):
    config, _ = modules
    native, descriptor, now = clock_originals()
    observations = native["observations"]
    node = "/example/controller_server"
    if mutation in {"wrong-value", "numeric-value"}:
        observations["node_parameters"][node]["use_sim_time"] = (
            True if mutation == "wrong-value" else 0
        )
    elif mutation == "missing-node":
        del observations["node_parameters"][node]
    elif mutation == "missing-second-node":
        del observations["node_parameters"]["/example/planner_server"]
    elif mutation in {"stale", "future"}:
        observations["parameter_captured_at"][node] = (
            now + timedelta(milliseconds=-501 if mutation == "stale" else 1)
        ).isoformat()
    elif mutation == "naive-stamp":
        observations["parameter_captured_at"][node] = now.replace(tzinfo=None).isoformat()
    elif mutation == "naive-capture":
        native["captured_at"] = now.replace(tzinfo=None).isoformat()
    elif mutation == "missing-stamp":
        del observations["parameter_captured_at"][node]
    elif mutation == "bad-stamp":
        observations["parameter_captured_at"][node] = "unverified"
    elif mutation == "different-namespace":
        descriptor["namespace"] = "/another"
    else:
        del native["observations"]
    assert config.configured_clock_readback_available(native, descriptor) is False


def test_invalid_typed_configuration_is_rejected_before_readback(modules):
    config, _ = modules
    native, descriptor, _ = clock_originals()
    descriptor["counterfactual_sim"]["node_clock_overrides"]["arbitrary_driver"] = False
    with pytest.raises(ValueError):
        config.configured_clock_readback_available(native, descriptor)
