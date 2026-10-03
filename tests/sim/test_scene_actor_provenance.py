"""Actual compiled two-actor provenance and immutable, fail-closed counterexamples."""

import copy
import xml.etree.ElementTree as ET

import mujoco
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.world.actors import (
    _compile,
    build_scene_actor_manifest,
    compiled_model_signature,
    verify_scene_actor_manifest,
)
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
    # Even a forged fresh initial signature cannot legitimize changed source physics/ownership.
    altered["initial_compiled_signature"] = compiled_model_signature(_compile(backend, altered))
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


@pytest.mark.parametrize("transmission", ["site", "tendon"])
def test_previously_valid_nonjoint_worldspec_compilation_retains_explicit_unqualified_binding(
    tmp_path, transmission
):
    if transmission == "site":
        xml = SOURCE.replace("</body>", '<site name="force" pos="0 0 0"/></body>').replace(
            '<motor name="drive" joint="axis" ctrllimited="true"\n ctrlrange="-1 1"/>',
            '<general name="drive" site="force" gear="0 0 1 0 0 0"/>',
        )
    else:
        xml = SOURCE.replace(
            "<actuator>",
            '<tendon><fixed name="coupling"><joint joint="axis" coef="1"/></fixed></tendon><actuator>',
        ).replace('joint="axis" ctrllimited', 'tendon="coupling" ctrllimited')
    # Actual MuJoCo accepts the unchanged XML before provenance is introduced.
    source_model = mujoco.MjModel.from_xml_string(xml)
    assert source_model.nu == 1
    (tmp_path / "nonjoint.xml").write_text(xml)
    backend = MujocoBackend(tmp_path)
    world = compile_world(
        backend,
        _t1_world(body_refs=[{"id": "local", "kind": "task", "ref": "nonjoint.xml"}]),
        name="existing_nonjoint",
    )
    proof = verify_scene_actor_manifest(backend, world["actor_manifest_ref"])
    actor = proof["actors"][0]
    assert actor["physical_authority"] == "NONE"
    assert actor["actuator_binding_qualification"] == "NOT_QUALIFIED"
    assert actor["source_binding_verified"] is False
    assert actor["actuators"][0]["ownership_status"] == "NOT_QUALIFIED"
    assert "scene_joint_index" not in actor["actuators"][0]
    assert backend.inspect_model(world["model_ref"]).nu == 1


MESH = b"""v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _simultaneously_mutated_mesh_models(tmp_path):
    xml = SOURCE.replace(
        "<worldbody>", '<asset><mesh name="shape" file="shape.obj"/></asset><worldbody>'
    ).replace('type="sphere" size="0.1"', 'type="mesh" mesh="shape"')
    (tmp_path / "source.xml").write_text(xml)
    (tmp_path / "shape.obj").write_bytes(MESH)
    backend = MujocoBackend(tmp_path)
    world = compile_world(
        backend,
        _t1_world(body_refs=[{"id": "local", "kind": "task", "ref": "source.xml"}]),
        name="mesh_bound_source",
    )
    altered = copy.deepcopy(backend.store.get(world["model_ref"]))
    seed = altered["scene_sources"][0]
    source = copy.deepcopy(backend.store.get(seed["source_model_ref"]))
    modified = backend.store.put("models", MESH.replace(b"v 1 0 0", b"v 1.2 0 0"))
    source["assets"] = dict.fromkeys(source["assets"], modified)
    source_ref = backend.store.put("models", source)
    seed["source_model_ref"] = source_ref
    seed["source_model_digest"] = backend._model_digest(source)
    altered["assets"] = dict.fromkeys(altered["assets"], modified)
    return backend, world, altered


@pytest.mark.parametrize("signature_guard", ["scene", "source"])
def test_both_current_source_and_scene_mesh_changes_cannot_replace_initial_compiled_binding(
    tmp_path, signature_guard
):
    backend, world, altered = _simultaneously_mutated_mesh_models(tmp_path)
    original = backend.store.get(world["actor_manifest_ref"])
    assert original["initial_compiled_signature"]["format"] == "mjb"
    if signature_guard == "source":
        # Independently exercise source's original compiled signature after a forged scene pin.
        altered["initial_compiled_signature"] = compiled_model_signature(_compile(backend, altered))
    ref = backend.store.put("models", altered)
    expected = "INITIAL_COMPILED" if signature_guard == "scene" else "SOURCE_COMPILED"
    with pytest.raises(ValueError, match="SCENE_ACTOR_" + expected + "_SIGNATURE_MISMATCH"):
        build_scene_actor_manifest(backend, ref)
    # Existing registered proof retains its original refs and immutable captured asset blobs.
    assert verify_scene_actor_manifest(backend, world["actor_manifest_ref"]) == original


def test_corrupted_captured_asset_rejected_by_content_addressing_before_signature_comparison(
    tmp_path,
):
    backend, world, _ = _simultaneously_mutated_mesh_models(tmp_path)
    original = backend.store.get(world["model_ref"])
    original_asset = next(iter(original["assets"].values()))
    backend.store.resolve(original_asset).write_bytes(MESH.replace(b"v 1 0 0", b"v 1.2 0 0"))
    with pytest.raises(ValueError, match="STORE_DIGEST_MISMATCH"):
        verify_scene_actor_manifest(backend, world["actor_manifest_ref"])
