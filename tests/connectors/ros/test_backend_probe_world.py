"""World-source preparation contracts; synthetic ELF bytes are never loaded."""

import importlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import yaml

from tests.connectors.ros.test_backend_probe_fixture import declaration


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("backend_probe_world")
    instrument = tmp_path / "instrument"
    d = declaration()
    importlib.import_module("backend_probe_fixture").prepare_probe_fixture(instrument, d)
    plugin = instrument / "synthetic-lib.so"
    plugin.write_bytes(b"\x7fELF_SYNTHETIC_NOT_LOADABLE_TEST_SOURCE")
    native = importlib.import_module("native_contact_evidence").prepare_native_policy(
        instrument,
        plugin_path=plugin,
        support_topics=["/instrument/contact"],
        ground_collisions=["ground::link::collision"],
        pose_topic="/instrument/pose",
        component_topic="/instrument/components",
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    policy = {
        "schema_version": "rosclaw.backend_cache_probe_policy.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "native_policy": native,
        "probe_xy": [6, 0],
        "sphere_radius_m": 0.05,
        "ground_z_m": 0,
        "lift_z_m": 10,
        "stable_samples": 10,
        "stable_sim_span_sec": 0.4,
        "refresh_sim_sec": 10,
        "refresh_wall_sec": 20,
    }
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "world.sdf").write_text("""<sdf version="1.9"><world name="synthetic_world">
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="rosclaw::PassivePhysics" filename="synthetic-not-loaded"><run_id>synthetic_run</run_id><body_model_name>active_robot</body_model_name><body_snapshot_hash>synthetic_robot_hash</body_snapshot_hash><attachment_hash>synthetic_brush_hash</attachment_hash><static_model>ground</static_model><obstacle_model>scene_obstacle</obstacle_model></plugin>
    <model name="ground"><static>true</static><link name="link"><collision name="collision"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision></link></model>
    <model name="scene_obstacle"><static>true</static><pose>5 0 .1 0 0 0</pose><link name="link"><collision name="collision"><geometry><box><size>.1 .1 .2</size></box></geometry></collision></link></model>
    </world></sdf>""")
    (scene / "bridge.yaml").write_text(
        yaml.safe_dump([{"ros_topic_name": "/robot/pose", "gz_topic_name": "/robot/pose"}])
    )
    binding = {
        "run_id": "synthetic_run",
        "world_name": "synthetic_world",
        "body_model_name": "active_robot",
        "body_snapshot_hash": "synthetic_robot_hash",
        "world_to_map_xyyaw": [0, 0, 0],
        "frame_transform_source": "simulator_operator_fixture_policy",
        "map_world_identity_approved": True,
        "attachment_hash": "synthetic_brush_hash",
        "grid": {"cleaning_polygon": d["cleaning_polygon"]},
        "scene_model_names": ["ground", "scene_obstacle", "active_robot"],
        "obstacle_names": ["scene_obstacle"],
    }
    (scene / "physics_binding.json").write_text(json.dumps(binding))
    return module, scene, instrument, policy, plugin, tmp_path / "output"


def prepare(fixture):
    module, scene, instrument, policy, plugin, out = fixture
    return module.prepare_probe_world(
        out,
        scene_directory=scene,
        instrument_directory=instrument,
        policy=policy,
        plugin_path=plugin,
    )


def test_prepares_exclusive_world_with_dynamic_observed_probe_and_preserves_inputs(fixture):
    module, scene, instrument, policy, plugin, out = fixture
    before = {p.name: p.read_bytes() for p in scene.iterdir()}
    result = prepare(fixture)
    assert result["physical_acceptance"] == "NOT_RUN" and result["backend_health_admitted"] is False
    assert result["authorization"] is False and result["actual_body_clearance_admitted"] is False
    assert {p.name: p.read_bytes() for p in scene.iterdir()} == before
    world = ET.parse(out / "world.sdf").find("world")
    probe = next(m for m in world.findall("model") if m.get("name") == "isolated_instrument")
    assert probe.findtext("static") == "false"
    occupancy = next(
        p for p in world.findall("plugin") if p.get("name") == "rosclaw::PassivePhysics"
    )
    assert [e.text for e in occupancy.findall("obstacle_model")] == [
        "scene_obstacle",
        "isolated_instrument",
    ]
    assert "isolated_instrument" not in [e.text for e in occupancy.findall("static_model")]
    native = next(p for p in world.findall("plugin") if p.get("name") == "rosclaw::PassiveContacts")
    assert native.findtext("topic") == "/rosclaw_sim/backend_probe_components"
    assert native.findtext("body_model_name") == "isolated_instrument"
    with pytest.raises(FileExistsError):
        prepare(fixture)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_run",
        "wrong_world",
        "robot_alias",
        "wrong_region",
        "missing_model",
        "unsupported_support",
        "no_support",
        "support_too_small",
        "support_rotated",
        "probe_wrong_world",
        "bridge_alias",
        "library_changed",
        "instrument_changed",
        "unapproved_declaration",
        "inside_region",
        "unresolved_include",
        "missing_contact_system",
        "wrong_passive_robot",
        "wrong_passive_run",
        "unknown_declared_model",
        "duplicate_physics",
        "nearby_collision",
        "unresolved_collision",
        "unapproved_frame_identity",
        "transformed_map_region",
    ],
)
def test_unresolved_or_mismatched_world_sources_are_refused_before_output(fixture, fault):
    module, scene, instrument, policy, plugin, out = fixture
    binding = json.loads((scene / "physics_binding.json").read_bytes())
    world = ET.parse(scene / "world.sdf")
    actual = world.find("world")
    if fault == "wrong_run":
        binding["run_id"] = "wrong"
    elif fault == "unapproved_frame_identity":
        binding["map_world_identity_approved"] = False
    elif fault == "transformed_map_region":
        binding["world_to_map_xyyaw"] = [1, 0, 0]
    elif fault == "wrong_world":
        binding["world_name"] = "wrong"
    elif fault == "robot_alias":
        binding["body_model_name"] = "isolated_instrument"
    elif fault == "wrong_region":
        binding["grid"]["cleaning_polygon"] = [[0, 0], [1, 0], [1, 1]]
    elif fault == "missing_model":
        binding["scene_model_names"].append("missing")
    elif fault == "unsupported_support":
        actual.find("model/link/collision/geometry/plane/normal").text = "0 1 0"
    elif fault == "no_support":
        actual.remove(actual.findall("model")[0])
    elif fault == "support_too_small":
        actual.find("model/link/collision/geometry/plane/size").text = "10 10"
    elif fault == "support_rotated":
        ET.SubElement(actual.findall("model")[0], "pose").text = "0 0 0 0 .1 0"
    elif fault == "probe_wrong_world":
        policy["native_policy"]["contact_policy"]["world_name"] = "wrong"
    elif fault == "bridge_alias":
        (scene / "bridge.yaml").write_text(
            yaml.safe_dump(
                [{"ros_topic_name": "/instrument/pose", "gz_topic_name": "/different_source"}]
            )
        )
    elif fault == "library_changed":
        plugin.write_bytes(b"\x7fELF_CHANGED_SYNTHETIC")
    elif fault == "instrument_changed":
        p = instrument / "robot.sdf"
        p.write_bytes(p.read_bytes().replace(b"isolated_instrument", b"wrong_instrument"))
    elif fault in {"unapproved_declaration", "inside_region"}:
        p = instrument / "probe-fixture.json"
        record = json.loads(p.read_bytes())
        if fault == "unapproved_declaration":
            record["declaration"]["approved"] = False
        else:
            record["declaration"]["probe_xy"] = [0, 0]
        p.write_text(json.dumps(record))
    elif fault == "unresolved_include":
        ET.SubElement(actual, "include")
    elif fault == "missing_contact_system":
        actual.remove(
            next(
                p for p in actual.findall("plugin") if p.get("name") == "gz::sim::systems::Contact"
            )
        )
    elif fault == "wrong_passive_robot":
        next(
            p for p in actual.findall("plugin") if p.get("name") == "rosclaw::PassivePhysics"
        ).find("body_model_name").text = "wrong_robot"
    elif fault == "wrong_passive_run":
        next(
            p for p in actual.findall("plugin") if p.get("name") == "rosclaw::PassivePhysics"
        ).find("run_id").text = "wrong_run"
    elif fault == "unknown_declared_model":
        ET.SubElement(
            next(p for p in actual.findall("plugin") if p.get("name") == "rosclaw::PassivePhysics"),
            "static_model",
        ).text = "hidden_model"
    elif fault == "duplicate_physics":
        ET.SubElement(
            actual, "plugin", name="gz::sim::systems::Physics", filename="gz-sim-physics-system"
        )
    elif fault == "nearby_collision":
        actual.findall("model")[1].find("pose").text = "6 0 .1 0 0 0"
    elif fault == "unresolved_collision":
        actual.findall("model")[1].find("link/collision/geometry/box").tag = "mesh"
    (scene / "physics_binding.json").write_text(json.dumps(binding))
    world.write(scene / "world.sdf")
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not out.exists()
