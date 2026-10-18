"""Protect explicit preactivation observation and actual launch precedence."""

import ast
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_counterfactual_config")


def descriptor(overrides):
    return {
        "counterfactual_sim": {
            "schema_version": "rosclaw.sim_counterfactual_launch.v1",
            "node_clock_overrides": overrides,
        }
    }


def test_explicit_clock_override_survives_actual_launch_parameter_precedence(module, monkeypatch):
    path = Path(module.__file__).with_name("nav2_launch.py")
    tree = ast.parse(path.read_text())
    expr = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.keyword)
        and n.arg == "parameters"
        and isinstance(n.value, ast.List)
        and any(
            isinstance(x, ast.Call)
            and isinstance(x.func, ast.Name)
            and x.func.id == "counterfactual_clock_parameters"
            for x in n.value.elts
        )
    )
    values = eval(
        compile(ast.Expression(expr), str(path), "eval"),
        {
            "params": "/pinned/params.yaml",
            "descriptor": descriptor({"controller_server": False}),
            "name": "controller_server",
            "counterfactual_clock_parameters": module.counterfactual_clock_parameters,
        },
    )
    merged = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    assert merged == {"use_sim_time": False, "yaml_filename": "/evidence/map.yaml"}
    assert module.counterfactual_clock_parameters(None, "controller_server") == {
        "use_sim_time": True
    }
    assert module.counterfactual_clock_parameters(
        descriptor({"controller_server": False}), "amcl"
    ) == {"use_sim_time": True}


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"driver": False},
        {"controller_server": 0},
        {"controller_server": "false"},
        {"amcl": False, "controller_server": False, "planner_server": False},
        [],
    ],
)
def test_unbounded_or_nonboolean_override_rejected(module, overrides):
    with pytest.raises(ValueError, match="explicit SDK"):
        module.counterfactual_clock_parameters(descriptor(overrides), "controller_server")


@pytest.mark.parametrize(
    "case",
    [
        "rsk_node_clock_mismatch",
        "rsk_controller_plugin_unavailable",
        "rsk_sensor_frame_unresolvable",
    ],
)
def test_new_configuration_capture_requires_same_isolated_role(module, monkeypatch, case):
    capture = importlib.import_module("simulation_engineering_capture")
    monkeypatch.setattr(capture.Path, "is_file", lambda _: True)
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    settings = {
        "run_id": "synthetic",
        "generic_source_observer": True,
        "governed_sink": {},
        "engineering_capture": {
            "schema_version": "rosclaw.engineering_failure_capture.v1",
            "case": case,
            "phase": "before_repair",
            "run_id": "synthetic",
        },
    }
    assert capture.validate_engineering_capture(settings)["case"] == case
    settings["engineering_capture"]["phase"] = "after_repair"
    with pytest.raises(ValueError):
        capture.validate_engineering_capture(settings)
