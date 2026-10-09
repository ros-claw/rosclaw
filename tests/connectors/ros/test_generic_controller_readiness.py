"""Source-bound inactive controller and hardware reply refusal boundaries."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from tests.connectors.ros.test_generic_stack_source import synthetic_stack_inputs


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("generic_controller_readiness")
    directory = tmp_path / "source"
    importlib.import_module("generic_stack_source").prepare_generic_stack_source(
        directory, **synthetic_stack_inputs()
    )
    expected = module.controller_expectations(directory)
    rows = [
        {"name": name, "type": kind, "state": "inactive", "claimed_interfaces": []}
        for name, kind in expected["controllers"].items()
    ]
    hardware = {
        "name": expected["hardware_name"],
        "type": expected["hardware_type"],
        "plugin_name": expected["hardware_plugin"],
        "state": {"id": 3, "label": "active"},
    }
    for kind, names in expected["interfaces"].items():
        hardware[kind + "_interfaces"] = [
            {
                "name": name,
                "data_type": expected["interface_types"][kind][name],
                "is_available": True,
                "is_claimed": False,
            }
            for name in names
        ]
    observations = {
        "controllers": {
            "service": expected["controller_manager"] + "/list_controllers",
            "received_monotonic_sec": 100.0,
            "response": {"controller": rows},
        },
        "hardware": {
            "service": expected["controller_manager"] + "/list_hardware_components",
            "received_monotonic_sec": 100.0,
            "response": {"component": [hardware]},
        },
    }
    return module, directory, expected, observations


def test_exact_inactive_replies_only_allow_next_stage_without_body_or_action(fixture):
    module, _, expected, observations = fixture
    report = module.inactive_controller_readiness(expected, observations, now=100.1)
    assert report["ready_for_next_stage"] is True
    assert not report["authorization"] and not report["live_body_admitted"]
    assert not report["controller_activation"]
    assert report["physical_acceptance"] == "NOT_EVALUATED"
    assert expected["controller_manager"] == "/declared_scope/custom_manager"
    assert expected["interfaces"]["command"] == ["left_axis/velocity", "right_axis/velocity"]


@pytest.mark.parametrize(
    "change", ["active", "wrong_type", "duplicate", "missing", "claimed", "malformed_name"]
)
def test_active_foreign_missing_duplicate_or_claimed_controller_refused(fixture, change):
    module, _, expected, original = fixture
    observations = deepcopy(original)
    rows = observations["controllers"]["response"]["controller"]
    if change == "active":
        rows[0]["state"] = "active"
    elif change == "wrong_type":
        rows[0]["type"] = "foreign/Driver"
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "missing":
        rows.pop()
    elif change == "claimed":
        rows[0]["claimed_interfaces"] = ["left_axis/velocity"]
    else:
        rows[0]["name"] = []
    assert (
        module.inactive_controller_readiness(expected, observations, now=100.1)[
            "ready_for_next_stage"
        ]
        is False
    )


@pytest.mark.parametrize(
    "change",
    [
        "plugin",
        "name",
        "inactive",
        "bool_id",
        "missing_command",
        "duplicate_state",
        "unavailable",
        "claimed",
        "data_type",
        "malformed_name",
    ],
)
def test_hardware_identity_state_resources_and_types_refuse(fixture, change):
    module, _, expected, original = fixture
    observations = deepcopy(original)
    hardware = observations["hardware"]["response"]["component"][0]
    if change == "plugin":
        hardware["plugin_name"] = "foreign/Plugin"
    elif change == "name":
        hardware["name"] = "other"
    elif change == "inactive":
        hardware["state"] = {"id": 2, "label": "inactive"}
    elif change == "bool_id":
        hardware["state"]["id"] = True
    elif change == "missing_command":
        hardware["command_interfaces"].pop()
    elif change == "duplicate_state":
        hardware["state_interfaces"].append(deepcopy(hardware["state_interfaces"][0]))
    elif change == "unavailable":
        hardware["command_interfaces"][0]["is_available"] = False
    elif change == "claimed":
        hardware["command_interfaces"][0]["is_claimed"] = True
    elif change == "data_type":
        hardware["command_interfaces"][0]["data_type"] = "bool"
    else:
        hardware["command_interfaces"][0]["name"] = []
    assert (
        module.inactive_controller_readiness(expected, observations, now=100.1)[
            "ready_for_next_stage"
        ]
        is False
    )


@pytest.mark.parametrize("role", ["controllers", "hardware"])
@pytest.mark.parametrize("change", ["missing", "stale", "future", "service", "nan"])
def test_original_service_identity_and_reply_freshness_required(fixture, role, change):
    module, _, expected, observations = fixture
    if change == "missing":
        observations.pop(role)
    elif change == "service":
        observations[role]["service"] = "/other/list_controllers"
    else:
        observations[role]["received_monotonic_sec"] = {
            "stale": 90,
            "future": 101,
            "nan": float("nan"),
        }[change]
    assert (
        module.inactive_controller_readiness(expected, observations, now=100.1)[
            "ready_for_next_stage"
        ]
        is False
    )


def test_changed_sealed_source_refused_before_ros_import(fixture):
    module, directory, _, _ = fixture
    with (directory / "robot.urdf").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        module.controller_expectations(directory)
