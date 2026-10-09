"""Explicit World-source changes cannot create sensor evidence or authority."""

import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from tests.connectors.ros.test_generic_stack_source import synthetic_stack_inputs

POLICY = {
    "source": "simulator_operator_fixture_policy",
    "approved": True,
    "evidence_domain": "SIMULATION",
    "world_name": "explicit_world",
    "render_engine": "ogre2",
}


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    import generic_world_system_source

    return generic_world_system_source


def test_explicit_systems_preserve_original_models_and_do_not_claim_messages(module):
    data = synthetic_stack_inputs()
    policy = deepcopy(POLICY)
    result = module.prepare_world_system_source(data["world_bytes"], data["sdf_bytes"], policy)
    before, after = ET.fromstring(data["world_bytes"]), ET.fromstring(result["world_bytes"])
    assert ET.tostring(before.find("world/model")) == ET.tostring(after.find("world/model"))
    assert policy == POLICY
    assert len(after.findall("world/plugin")) == 4
    assert result["report"]["added_standard_roles"] == ["physics", "commands", "scene", "sensors"]
    assert result["report"]["actual_sensor_messages"] == "REQUIRED_NOT_OBSERVED"
    assert result["report"]["actual_system_loader_validation"] == "REQUIRED_NOT_RUN"
    assert result["report"]["authorization"] is False
    assert result["report"]["World_started"] is False
    assert result["report"]["physical_acceptance"] == "NOT_RUN"


def test_contact_dependency_comes_from_materialized_instrumented_robot(module):
    data = synthetic_stack_inputs()
    robot = ET.fromstring(data["sdf_bytes"])
    ET.SubElement(robot.find("model/link"), "sensor", name="actual_contact", type="contact")
    result = module.prepare_world_system_source(data["world_bytes"], ET.tostring(robot), POLICY)
    assert result["report"]["required_standard_roles"][-1] == "contact"
    assert len(ET.fromstring(result["world_bytes"]).findall("world/plugin")) == 5


def test_repeated_preparation_does_not_duplicate_or_rewrite_resolved_systems(module):
    data = synthetic_stack_inputs()
    first = module.prepare_world_system_source(data["world_bytes"], data["sdf_bytes"], POLICY)
    second = module.prepare_world_system_source(first["world_bytes"], data["sdf_bytes"], POLICY)
    assert second["world_bytes"] == first["world_bytes"]
    assert second["report"]["added_standard_roles"] == []
    assert second["report"]["configured_standard_parameters"] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("approved", False),
        ("approved", 1),
        ("evidence_domain", "REAL"),
        ("world_name", "other_world"),
        ("render_engine", "ogre1"),
        ("extra", True),
    ],
)
def test_missing_or_changed_operator_policy_refuses(module, key, value):
    data = synthetic_stack_inputs()
    policy = {**POLICY, key: value}
    with pytest.raises(ValueError):
        module.prepare_world_system_source(data["world_bytes"], data["sdf_bytes"], policy)


@pytest.mark.parametrize("fault", ["duplicate", "rebound_library", "rebound_class", "wrong_engine"])
def test_existing_conflicting_systems_are_not_silently_overwritten(module, fault):
    data = synthetic_stack_inputs()
    root = ET.fromstring(data["world_bytes"])
    world = root.find("world")
    plugin = ET.SubElement(
        world, "plugin", filename="gz-sim-sensors-system", name="gz::sim::systems::Sensors"
    )
    ET.SubElement(plugin, "render_engine").text = "ogre2"
    if fault == "duplicate":
        world.append(deepcopy(plugin))
    elif fault == "rebound_library":
        plugin.set("filename", "foreign.so")
    elif fault == "rebound_class":
        plugin.set("name", "foreign::Class")
    else:
        plugin.find("render_engine").text = "ogre1"
    with pytest.raises(ValueError):
        module.prepare_world_system_source(ET.tostring(root), data["sdf_bytes"], POLICY)


def test_optional_world_systems_are_preserved_and_rechecked_in_sealed_handoff(module, tmp_path):
    from generic_stack_source import prepare_generic_stack_source, read_prepared_generic_stack

    data = synthetic_stack_inputs()
    out = tmp_path / "prepared"
    prepare_generic_stack_source(out, **data, world_systems_declaration=POLICY)
    captured = read_prepared_generic_stack(out)["captured_files"]
    assert captured["original-sources/world.original.sdf"] == data["world_bytes"]
    assert json.loads(captured["source-declarations.json"])["world_systems"] == POLICY
    report = json.loads(captured["world-system-source-report.json"])
    assert report["added_standard_roles"] == ["physics", "commands", "scene", "sensors", "contact"]
    assert report["actual_sensor_messages"] == "REQUIRED_NOT_OBSERVED"
    assert len(ET.fromstring(captured["world.sdf"]).findall("world/plugin")) == 5


def test_other_sensor_types_remain_explicitly_unvalidated(module):
    data = synthetic_stack_inputs()
    robot = ET.fromstring(data["sdf_bytes"])
    ET.SubElement(robot.find("model/link"), "sensor", name="other", type="unimplemented_sensor")
    result = module.prepare_world_system_source(data["world_bytes"], ET.tostring(robot), POLICY)
    assert result["report"]["other_sensor_types_require_separate_runtime_evidence"] == [
        "unimplemented_sensor"
    ]
    assert result["report"]["physical_acceptance"] == "NOT_RUN"


@pytest.mark.parametrize("fault", ["remove_system", "claim_messages", "omit_report_inventory"])
def test_rehashed_handoff_still_must_match_original_system_proposal(module, tmp_path, fault):
    import hashlib

    from generic_stack_source import prepare_generic_stack_source, read_prepared_generic_stack

    from rosclaw.connectors.ros.diagnosis.coverage_audit import digest

    out = tmp_path / "prepared"
    prepare_generic_stack_source(out, **synthetic_stack_inputs(), world_systems_declaration=POLICY)
    manifest = json.loads((out / "generic-stack-source-manifest.json").read_text())
    if fault == "remove_system":
        path = out / "world.sdf"
        root = ET.fromstring(path.read_bytes())
        world = root.find("world")
        world.remove(world.find("plugin[@name='gz::sim::systems::Sensors']"))
        path.write_bytes(ET.tostring(root))
    elif fault == "claim_messages":
        path = out / "world-system-source-report.json"
        report = json.loads(path.read_text())
        report["actual_sensor_messages"] = "OBSERVED"
        path.write_text(json.dumps(report))
    else:
        manifest["output_hashes"].pop("world-system-source-report.json")
        path = None
    if path is not None:
        manifest["output_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["artifact_hash"] = digest({k: v for k, v in manifest.items() if k != "artifact_hash"})
    (out / "generic-stack-source-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="World systems differ"):
        read_prepared_generic_stack(out)
