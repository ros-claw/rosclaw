"""Instrument source rendering contracts; no robot asset or live simulator."""

import importlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("backend_probe_fixture")


def declaration():
    return {
        "schema_version": "rosclaw.backend_probe_fixture_declaration.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "run_id": "synthetic_run",
        "world_name": "synthetic_world",
        "robot_model_name": "active_robot",
        "probe_model_name": "isolated_instrument",
        "probe_xy": [6, 0],
        "sphere_radius_m": 0.05,
        "ground_z_m": 0,
        "lift_z_m": 10,
        "ground_collision_name": "ground::link::collision",
        "cleaning_polygon": [[-1.4, -1.4], [1.4, -1.4], [1.4, 1.4], [-1.4, 1.4]],
        "maximum_robot_radius_m": 0.3,
        "pose_topic": "/instrument/pose",
        "component_topic": "/instrument/components",
        "contact_topic": "/instrument/contact",
    }


def test_source_isolated_primitive_has_typed_bridges_and_no_authority(module, tmp_path):
    output = tmp_path / "probe"
    result = module.prepare_probe_fixture(output, declaration())
    assert result["evidence_role"] == "SIM_INSTRUMENT_NOT_ROBOT_BODY_OR_AUTHORITY"
    assert (
        result["physical_acceptance"] == "NOT_RUN"
        and result["world_integration"] == "NOT_IMPLEMENTED"
    )
    model = ET.parse(output / "robot.sdf").find("model")
    assert model.get("name") == "isolated_instrument"
    assert model.findtext("pose") == "6 0 0.05 0 0 0"
    assert model.findtext("link/collision/geometry/sphere/radius") == "0.05"
    assert model.findtext("link/sensor/contact/topic") == "/rosclaw_sim/backend_probe_contact"
    assert model.findtext("link/gravity") == "true"
    assert model.findtext("plugin/update_frequency") == "20"
    assert json.loads((output / "physics_binding.json").read_bytes()) == result["binding"]
    source = importlib.import_module("native_contact_evidence")
    plugin = output / "synthetic-fixture.so"
    plugin.write_bytes(b"synthetic source hash fixture, not loadable plugin")
    policy = source.prepare_native_policy(
        output,
        plugin_path=plugin,
        support_topics=["/instrument/contact"],
        ground_collisions=["ground::link::collision"],
        pose_topic="/instrument/pose",
        component_topic="/instrument/components",
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    assert policy["contact_policy"]["model_name"] == "isolated_instrument"
    assert source.reopen_native_policy(output, policy, plugin_path=plugin) == policy
    with pytest.raises(FileExistsError):
        module.prepare_probe_fixture(output, declaration())


@pytest.mark.parametrize(
    "fault",
    [
        "robot_alias",
        "inside",
        "boundary_clearance",
        "real",
        "unapproved",
        "topic_alias",
        "bad_model",
        "radius",
        "nan",
        "unbounded_height",
        "body_radius",
        "bad_polygon",
        "extra",
    ],
)
def test_unbounded_or_robot_alias_fixture_refuses_before_output(module, tmp_path, fault):
    source = declaration()
    if fault == "robot_alias":
        source["probe_model_name"] = source["robot_model_name"]
    elif fault == "inside":
        source["probe_xy"] = [0, 0]
    elif fault == "boundary_clearance":
        source["probe_xy"] = [1.6, 1.6]
    elif fault == "real":
        source["evidence_domain"] = "REAL"
    elif fault == "unapproved":
        source["approved"] = False
    elif fault == "topic_alias":
        source["component_topic"] = source["pose_topic"]
    elif fault == "bad_model":
        source["probe_model_name"] = '"; control'
    elif fault == "radius":
        source["sphere_radius_m"] = 0
    elif fault == "nan":
        source["probe_xy"] = [float("nan"), 0]
    elif fault == "unbounded_height":
        source["lift_z_m"] = 1000
    elif fault == "body_radius":
        source["maximum_robot_radius_m"] = True
    elif fault == "bad_polygon":
        source["cleaning_polygon"] = [[0, 0]]
    else:
        source["extra"] = "unknown"
    with pytest.raises(ValueError):
        module.prepare_probe_fixture(tmp_path / "probe", source)
    assert not (tmp_path / "probe").exists()
