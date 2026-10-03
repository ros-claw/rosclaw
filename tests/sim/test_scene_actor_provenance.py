"""Actual compiled two-actor provenance and immutable, fail-closed counterexamples."""

import copy
import xml.etree.ElementTree as ET

import mujoco
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.world.actors import build_scene_actor_manifest, verify_scene_actor_manifest
from rosclaw.sim.world.compiler import compile_world
from tests.sim.test_worldspec import _t1_world

# Deliberately generic one-joint fixtures, not Unitree robots or physical controllers.
SOURCE = """<mujoco><worldbody><body name="base"><inertial pos="0 0 0"
 mass="1" diaginertia="0.1 0.1 0.1"/><joint name="axis" type="hinge" axis="0 0 1"/>
 <geom name="shell" type="sphere" size="0.1" friction="0.7 0.01 0.001"/>
 </body></worldbody><actuator><motor name="drive" joint="axis" ctrllimited="true"
 ctrlrange="-1 1"/></actuator></mujoco>"""


@pytest.fixture
def scene(tmp_path, monkeypatch):
    (tmp_path / "source.xml").write_text(SOURCE)
    backend = MujocoBackend(tmp_path)
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("provenance must never step"))
    world = compile_world(
        backend,
        _t1_world(
            body_refs=[
                {
                    "id": name,
                    "kind": "task",
                    "ref": "source.xml",
                    "pose": {"pos": [i, 0, 0], "quat": [1, 0, 0, 0]},
                }
                for i, name in enumerate(["G1", "M20"])
            ]
        ),
        name="generic_actor_binding",
    )
    return backend, world


def test_actual_compiled_indices_trnid_and_public_inspection_do_not_grant_authority(scene):
    backend, world = scene
    binding = verify_scene_actor_manifest(backend, world["actor_manifest_ref"])
    inspected = backend.inspect_model(world["model_ref"]).detail["simulation_scene"]
    assert inspected == binding
    model = mujoco.MjModel.from_xml_string(backend.store.get(world["model_ref"])["mjcf_xml"])
    assert binding["physics_steps"] == 0 and binding["usable_for_real_execution"] is False
    assert binding["mission_body_binding_changed"] is False
    for actor in binding["actors"]:
        assert actor["catalog_status"] == "LOCAL_UNREGISTERED"
        assert actor["source_dimensions"]["nv"] == 1
        assert actor["capability_qualification"] == "NOT_EVALUATED"
        assert actor["physical_authority"] == "NONE"
        motor = actor["actuators"][0]
        joint = actor["joints"][0]
        assert motor["scene_index"] == mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actor["prefix"] + "drive"
        )
        assert model.actuator_trnid[motor["scene_index"], 0] == joint["scene_index"]
        assert joint["scene_index"] == motor["scene_joint_index"]
    assert (
        binding["actors"][0]["actuators"][0]["scene_index"]
        != binding["actors"][1]["actuators"][0]["scene_index"]
    )


@pytest.mark.parametrize("mutation", ["scene_digest", "source_digest", "actuator_owner"])
def test_immutable_actor_manifest_tampering_rejected(scene, mutation):
    backend, world = scene
    altered = copy.deepcopy(backend.store.get(world["actor_manifest_ref"]))
    if mutation == "scene_digest":
        altered["model_digest"] = "sha256:" + "0" * 64
    elif mutation == "source_digest":
        altered["actors"][0]["source_model_digest"] = "sha256:" + "0" * 64
    else:
        altered["actors"][0]["actuators"][0]["scene_joint_index"] = altered["actors"][1]["joints"][
            0
        ]["scene_index"]
    ref = backend.store.put("experiments", altered)
    with pytest.raises(ValueError, match="SCENE_ACTOR_MANIFEST_BINDING_MISMATCH"):
        verify_scene_actor_manifest(backend, ref)


@pytest.mark.parametrize("mutation", ["source_digest", "source_ref", "catalog_spoof"])
def test_actual_model_source_binding_tampering_rejected(scene, mutation):
    backend, world = scene
    altered = copy.deepcopy(backend.store.get(world["model_ref"]))
    seed = altered["scene_sources"][0]
    if mutation == "source_digest":
        seed["source_model_digest"] = "sha256:" + "0" * 64
    elif mutation == "source_ref":
        seed["source_model_ref"] = world["model_ref"]
    else:
        altered["worldspec"]["body_refs"][0]["kind"] = "eurdf"
        seed["body_ref"]["kind"] = "eurdf"
    ref = backend.store.put("models", altered)
    with pytest.raises(ValueError, match="SCENE_ACTOR_SOURCE"):
        build_scene_actor_manifest(backend, ref)


@pytest.mark.parametrize("mutation", ["inertia", "mass", "axis", "friction", "caps", "motor_owner"])
def test_actual_recompiled_scene_physics_or_motor_owner_change_rejected(scene, mutation):
    backend, world = scene
    altered = copy.deepcopy(backend.store.get(world["model_ref"]))
    xml = ET.fromstring(altered["mjcf_xml"])
    if mutation in {"inertia", "mass"}:
        element = xml.find('.//body[@name="G1_base"]/inertial')
        element.set(
            "diaginertia" if mutation == "inertia" else "mass",
            "0.11 0.1 0.1" if mutation == "inertia" else "1.1",
        )
    elif mutation == "axis":
        xml.find('.//joint[@name="G1_axis"]').set("axis", "1 0 0")
    elif mutation == "friction":
        xml.find('.//geom[@name="G1_shell"]').set("friction", "0.8 0.01 0.001")
    elif mutation == "caps":
        xml.find('.//actuator/*[@name="G1_drive"]').set("ctrlrange", "-2 2")
    else:
        xml.find('.//actuator/*[@name="G1_drive"]').set("joint", "M20_axis")
    altered["mjcf_xml"] = ET.tostring(xml, encoding="unicode")
    ref = backend.store.put("models", altered)
    with pytest.raises(ValueError, match="SCENE_ACTOR_(PHYSICS_MISMATCH|ACTUATOR_OWNER_INVALID)"):
        build_scene_actor_manifest(backend, ref)


def test_catalog_source_preserves_identity_and_declares_only_measured_inertia_rounding(tmp_path):
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    if not (_default_zoo_path() / "ur5e/robot.mjcf.xml").exists():
        pytest.skip("UR5 catalog source unavailable")
    backend = MujocoBackend(tmp_path)
    world = compile_world(
        backend,
        _t1_world(body_refs=[{"id": "arm", "kind": "eurdf", "ref": "ur5e"}]),
        name="actual_catalog_source",
    )
    actor = verify_scene_actor_manifest(backend, world["actor_manifest_ref"])["actors"][0]
    assert actor["catalog_status"] == "CATALOG_SOURCE"
    assert actor["catalog_profile_ref"] == "ur5e"
    assert actor["physical_authority"] == "NONE"
    assert actor["source_binding_semantics"] == "source-binding-with-declared-rounding"
    assert set(actor["rounding_differences"]) <= {"body_inertia"}
    assert actor["rounding_differences"]["body_inertia"]["max_relative_error"] < 1e-6


def test_forged_catalog_identity_on_local_source_cannot_promote_actor(scene):
    backend, world = scene
    altered = copy.deepcopy(backend.store.get(world["model_ref"]))
    seed = altered["scene_sources"][0]
    source = copy.deepcopy(backend.store.get(seed["source_model_ref"]))
    # Forging all mutable metadata consistently still cannot replace actual catalog bytes.
    source["source"] = {"kind": "eurdf", "ref": "ur5e"}
    forged_ref = backend.store.put("models", source)
    seed["source_model_ref"] = forged_ref
    seed["source_model_digest"] = backend._model_digest(source)
    seed["body_ref"].update(kind="eurdf", ref="ur5e")
    altered["worldspec"]["body_refs"][0].update(kind="eurdf", ref="ur5e")
    ref = backend.store.put("models", altered)
    with pytest.raises(ValueError, match="SCENE_ACTOR_CATALOG_SOURCE_MISMATCH"):
        build_scene_actor_manifest(backend, ref)


def test_loading_scene_and_registering_provenance_preserves_actual_mission_and_tool_authority(
    scene, tmp_path
):
    import asyncio

    from rosclaw.agentd.pi_bridge.context import build_embodied_context
    from rosclaw.sim.world.actors import SCENE_ACTOR_MEDIA_TYPE
    from tests.agentd.test_pi_tool_bridge import _setup_ur5e

    async def check():
        service, mission = await _setup_ur5e(tmp_path / "agent")
        try:
            before = build_embodied_context(service, mission.mission_id)
            backend, world = scene
            kernel = service._task_kernel
            task = kernel.bind_message(
                mission_id=mission.mission_id,
                session_ref="pi_1",
                backend_native_id="isolated",
                message_id="generic_scene",
                text="inspect generic source provenance",
                cwd=str(tmp_path),
            )
            artifact = tmp_path / "sim/experiments" / (world["actor_manifest_ref"] + ".json")
            kernel.register_artifact(
                task_id=task["task_id"], path=str(artifact), media_type=SCENE_ACTOR_MEDIA_TYPE
            )
            assert all(
                actor["physical_authority"] == "NONE"
                for actor in backend.inspect_model(world["model_ref"]).detail["simulation_scene"][
                    "actors"
                ]
            )
            after = build_embodied_context(service, mission.mission_id)
            assert before.body == after.body
            assert after.body["binding_scope"] == "mission_body"
            assert after.body["body_id"] == "sim/ur5e"
            assert before.capabilities == after.capabilities
            assert before.tool_policy == after.tool_policy
            assert (
                before.self_state["authorization_profile"]
                == after.self_state["authorization_profile"]
            )
        finally:
            await service.close()

    asyncio.run(check())
