"""Immutable SIM scene actor provenance; this never changes mission-body authority."""

from __future__ import annotations

import hashlib
from typing import Any

import mujoco
import numpy as np

from rosclaw.contracts.common import canonical_json
from rosclaw.sim.refs import make_ref

SCENE_ACTOR_MEDIA_TYPE = "application/vnd.rosclaw.sim.scene-actors+json"
SCHEMA_VERSION = "rosclaw.sim.scene_actors.v1"


def _compile(backend, manifest):
    from rosclaw.sim.backends.mujoco.backend import _spec_from_xml_assets

    assets = {name: backend.store.get(ref) for name, ref in manifest["assets"].items()}
    return _spec_from_xml_assets(manifest["mjcf_xml"], assets).compile()


def _rows_match(source, scene, source_ids, scene_ids, fields):
    differences = {}
    for field in fields:
        original = getattr(source, field)[source_ids]
        attached = getattr(scene, field)[scene_ids]
        if np.array_equal(original, attached):
            continue
        if field != "body_inertia" or not np.allclose(original, attached, rtol=1e-6, atol=1e-12):
            raise ValueError(f"SCENE_ACTOR_PHYSICS_MISMATCH: {field}")
        differences[field] = {
            "max_absolute_error": float(np.max(np.abs(original - attached))),
            "max_relative_error": float(
                np.max(np.abs(original - attached) / np.maximum(np.abs(original), 1e-30))
            ),
        }
    return differences


def _index_map(source, scene, kind, source_ids, prefix, offset):
    result = []
    for index in source_ids:
        source_name = mujoco.mj_id2name(source, kind, index)
        expected_name = prefix + source_name if source_name else None
        scene_index = (
            mujoco.mj_name2id(scene, kind, expected_name) if expected_name else offset + index
        )
        counts = {
            mujoco.mjtObj.mjOBJ_BODY: scene.nbody,
            mujoco.mjtObj.mjOBJ_JOINT: scene.njnt,
            mujoco.mjtObj.mjOBJ_GEOM: scene.ngeom,
            mujoco.mjtObj.mjOBJ_ACTUATOR: scene.nu,
        }
        if (
            not 0 <= scene_index < counts[kind]
            or mujoco.mj_id2name(scene, kind, scene_index) != expected_name
        ):
            raise ValueError("SCENE_ACTOR_INDEX_BINDING_INVALID")
        result.append(
            {
                "source_index": int(index),
                "scene_index": int(scene_index),
                "source_name": source_name,
                "scene_name": expected_name,
            }
        )
    return result


def build_scene_actor_manifest(backend, model_ref: str) -> dict[str, Any]:
    """Recompile real source/scene models and verify namespace, topology and motors."""
    manifest = backend.store.get(model_ref)
    if not isinstance(manifest, dict) or manifest.get("kind") != "model_manifest":
        raise ValueError("SCENE_ACTOR_MODEL_INVALID")
    seeds = manifest.get("scene_sources")
    world = manifest.get("worldspec")
    if (
        not isinstance(seeds, list)
        or not isinstance(world, dict)
        or len(seeds) != len(world["body_refs"])
    ):
        raise ValueError("SCENE_ACTOR_SOURCES_MISSING")
    scene = _compile(backend, manifest)
    actors = []
    claimed_body_ids, claimed_joint_ids, claimed_motor_ids = set(), set(), set()
    for seed, body_ref in zip(seeds, world["body_refs"], strict=True):
        if seed["body_ref"] != body_ref or seed["prefix"] != body_ref["id"] + "_":
            raise ValueError("SCENE_ACTOR_SOURCE_BINDING_INVALID")
        source_manifest = backend.store.get(seed["source_model_ref"])
        if (
            source_manifest.get("kind") != "model_manifest"
            or backend._model_digest(source_manifest) != seed["source_model_digest"]
            or source_manifest["source"] != {"kind": body_ref["kind"], "ref": body_ref["ref"]}
        ):
            raise ValueError("SCENE_ACTOR_SOURCE_IDENTITY_INVALID")
        if body_ref["kind"] == "eurdf":
            from rosclaw.sim.resolve import resolve_mjcf_source, source_kind_for

            path = resolve_mjcf_source(body_ref["ref"], task_root=backend._task_root)
            if (
                source_kind_for(path, task_root=backend._task_root) != "eurdf"
                or path.read_text() != source_manifest["mjcf_xml"]
                or backend._assets_from_file(path)
                != {name: backend.store.get(ref) for name, ref in source_manifest["assets"].items()}
            ):
                raise ValueError("SCENE_ACTOR_CATALOG_SOURCE_MISMATCH")
        source = _compile(backend, source_manifest)
        prefix, offsets = seed["prefix"], seed["offsets"]
        bodies = _index_map(
            source,
            scene,
            mujoco.mjtObj.mjOBJ_BODY,
            range(1, source.nbody),
            prefix,
            offsets["bodies"] - 1,
        )
        joints = _index_map(
            source, scene, mujoco.mjtObj.mjOBJ_JOINT, range(source.njnt), prefix, offsets["joints"]
        )
        geoms = _index_map(
            source, scene, mujoco.mjtObj.mjOBJ_GEOM, range(source.ngeom), prefix, offsets["geoms"]
        )
        motors = _index_map(
            source,
            scene,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            range(source.nu),
            prefix,
            offsets["actuators"],
        )
        body_map = {0: 0, **{row["source_index"]: row["scene_index"] for row in bodies}}
        joint_map = {row["source_index"]: row["scene_index"] for row in joints}
        for row in bodies:
            if (
                scene.body_parentid[row["scene_index"]]
                != body_map[int(source.body_parentid[row["source_index"]])]
            ):
                raise ValueError("SCENE_ACTOR_SUBTREE_BINDING_INVALID")
        for row in joints:
            if (
                scene.jnt_bodyid[row["scene_index"]]
                != body_map[int(source.jnt_bodyid[row["source_index"]])]
            ):
                raise ValueError("SCENE_ACTOR_JOINT_OWNER_INVALID")
        for row in geoms:
            if (
                scene.geom_bodyid[row["scene_index"]]
                != body_map[int(source.geom_bodyid[row["source_index"]])]
            ):
                raise ValueError("SCENE_ACTOR_GEOM_OWNER_INVALID")
        for row in motors:
            original_id, attached_id = row["source_index"], row["scene_index"]
            transmission = int(source.actuator_trntype[original_id])
            if transmission not in {
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            }:
                raise ValueError("SCENE_ACTOR_TRANSMISSION_UNSUPPORTED")
            original_joint = int(source.actuator_trnid[original_id, 0])
            attached_joint = int(scene.actuator_trnid[attached_id, 0])
            if (
                attached_joint != joint_map[original_joint]
                or scene.actuator_trnid[attached_id, 1] != source.actuator_trnid[original_id, 1]
            ):
                raise ValueError("SCENE_ACTOR_ACTUATOR_OWNER_INVALID")
            row.update(
                source_joint_index=original_joint,
                scene_joint_index=attached_joint,
                transmission_type=transmission,
            )
        mappings = [
            (bodies, ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_gravcomp")),
            (
                joints,
                (
                    "jnt_type",
                    "jnt_pos",
                    "jnt_axis",
                    "jnt_range",
                    "jnt_limited",
                    "jnt_stiffness",
                    "jnt_solref",
                    "jnt_solimp",
                ),
            ),
            (
                geoms,
                (
                    "geom_type",
                    "geom_size",
                    "geom_pos",
                    "geom_quat",
                    "geom_friction",
                    "geom_contype",
                    "geom_conaffinity",
                    "geom_condim",
                    "geom_solref",
                    "geom_solimp",
                    "geom_margin",
                    "geom_gap",
                ),
            ),
            (
                motors,
                (
                    "actuator_trntype",
                    "actuator_gaintype",
                    "actuator_biastype",
                    "actuator_dyntype",
                    "actuator_gainprm",
                    "actuator_biasprm",
                    "actuator_dynprm",
                    "actuator_gear",
                    "actuator_ctrlrange",
                    "actuator_forcerange",
                    "actuator_ctrllimited",
                    "actuator_forcelimited",
                ),
            ),
        ]
        rounding_differences = {}
        for rows, fields in mappings:
            rounding_differences.update(
                _rows_match(
                    source,
                    scene,
                    [r["source_index"] for r in rows],
                    [r["scene_index"] for r in rows],
                    fields,
                )
            )
        dofs = []
        for row in joints:
            original_id, attached_id = row["source_index"], row["scene_index"]
            count = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}.get(
                int(source.jnt_type[original_id]), 1
            )
            for offset in range(count):
                dofs.append(
                    {
                        "source_index": int(source.jnt_dofadr[original_id]) + offset,
                        "scene_index": int(scene.jnt_dofadr[attached_id]) + offset,
                    }
                )
        _rows_match(
            source,
            scene,
            [r["source_index"] for r in dofs],
            [r["scene_index"] for r in dofs],
            ("dof_damping", "dof_armature", "dof_frictionloss"),
        )
        for rows, claimed in (
            (bodies, claimed_body_ids),
            (joints, claimed_joint_ids),
            (motors, claimed_motor_ids),
        ):
            indices = {row["scene_index"] for row in rows}
            if claimed.intersection(indices):
                raise ValueError("SCENE_ACTOR_OVERLAPPING_OWNERSHIP")
            claimed.update(indices)
        actors.append(
            {
                "actor_id": body_ref["id"],
                "prefix": prefix,
                "source": source_manifest["source"],
                "source_model_ref": seed["source_model_ref"],
                "source_model_digest": seed["source_model_digest"],
                "source_dimensions": {
                    "nq": source.nq,
                    "nv": source.nv,
                    "nu": source.nu,
                    "nbody": source.nbody,
                    "njnt": source.njnt,
                },
                "catalog_status": "CATALOG_SOURCE"
                if body_ref["kind"] == "eurdf"
                else "LOCAL_UNREGISTERED",
                "catalog_profile_ref": body_ref["ref"] if body_ref["kind"] == "eurdf" else None,
                "bodies": bodies,
                "joints": joints,
                "geoms": geoms,
                "actuators": motors,
                "dofs": dofs,
                "source_binding_verified": True,
                "source_binding_semantics": "source-binding-with-declared-rounding",
                "numeric_tolerances": {
                    "body_inertia": {"rtol": 1e-6, "atol": 1e-12},
                    "all_other_fields": "exact",
                },
                "rounding_differences": rounding_differences,
                "capability_qualification": "NOT_EVALUATED",
                "physical_authority": "NONE",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "scene_actor_manifest",
        "binding_scope": "simulation_scene",
        "model_ref": model_ref,
        "model_digest": backend._model_digest(manifest),
        "backend": "mujoco",
        "backend_version": str(mujoco.__version__),
        "actors": actors,
        "trust_level": "SIMULATED",
        "usable_for_real_execution": False,
        "mission_body_binding_changed": False,
        "physics_steps": 0,
    }


def persist_scene_actor_manifest(backend, model_ref: str) -> dict[str, Any]:
    manifest = build_scene_actor_manifest(backend, model_ref)
    ref = make_ref("simart", hashlib.sha256(canonical_json(manifest).encode()).hexdigest())
    backend.store.put("experiments", manifest, ref=ref)
    return {"actor_manifest_ref": ref, "manifest": manifest}


def verify_scene_actor_manifest(backend, ref: str) -> dict[str, Any]:
    declared = backend.store.get(ref)
    if not isinstance(declared, dict) or declared.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("SCENE_ACTOR_MANIFEST_INVALID")
    expected = build_scene_actor_manifest(backend, declared["model_ref"])
    if declared != expected:
        raise ValueError("SCENE_ACTOR_MANIFEST_BINDING_MISMATCH")
    return declared
