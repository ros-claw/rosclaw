"""Read-only compiled reset and saved rollout binding; no oracle physics steps."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from benchmarks.harnessbench.dynamic_oracle import (
    _valid_stats,
    replay_verified,
    runtime_validation_ok,
    same_compiled_body,
    trace_stats,
)


def keyframe_only_patch(original_xml, candidate_xml, name):
    """Only the selected keyframe qpos may change; retain body/physics/other keys."""
    trees = [ET.fromstring(xml) for xml in (original_xml, candidate_xml)]
    models = [mujoco.MjModel.from_xml_string(xml) for xml in (original_xml, candidate_xml)]
    indices = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, name) for m in models]
    if indices[0] < 0 or indices != [indices[0]] * 2 or models[0].nkey != models[1].nkey:
        return False
    if (
        models[0].opt.timestep != models[1].opt.timestep
        or models[0].opt.integrator != models[1].opt.integrator
        or [mujoco.mj_id2name(models[0], mujoco.mjtObj.mjOBJ_KEY, i) for i in range(models[0].nkey)]
        != [mujoco.mj_id2name(models[1], mujoco.mjtObj.mjOBJ_KEY, i) for i in range(models[1].nkey)]
    ):
        return False
    for field in ("time", "qvel", "act", "ctrl", "mpos", "mquat"):
        if not np.array_equal(
            getattr(models[0], "key_" + field), getattr(models[1], "key_" + field)
        ):
            return False
    unchanged = np.arange(models[0].nkey) != indices[0]
    if not np.array_equal(models[0].key_qpos[unchanged], models[1].key_qpos[unchanged]):
        return False
    for tree in trees:
        for keyframes in tree.findall("keyframe"):
            tree.remove(keyframes)
        for size in tree.findall("size"):
            size.attrib.pop("nkey", None)
    return same_compiled_body(*(ET.tostring(t, encoding="unicode") for t in trees))


def keyframe_rollout_evidence(backend, root, model_ref, name):
    """Actual initializer vector and named simkey must match the saved native trace."""
    from benchmarks.harnessbench.oracle import _experiment_receipts

    xml = backend.store.get(model_ref)["mjcf_xml"]
    model = mujoco.MjModel.from_xml_string(xml)
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if index < 0:
        return None
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, index)
    mujoco.mj_forward(model, data)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    expected = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, expected, spec)
    for receipt in _experiment_receipts(backend, {model_ref}):
        try:
            initial = backend.store.get(receipt["initial_state_ref"])
            trace = backend.store.get(receipt["trace_ref"])
            initialization = initial["initialization"]
            key = backend.store.get(initialization["keyframe_ref"])
            blob = backend.store.get(initial["state_vector_ref"])
            if (
                initial.get("kind") != "state_snapshot_v2"
                or initial.get("fidelity") != "FULL_INTEGRATION"
                or initial.get("model_ref") != model_ref
                or initial.get("model_digest") != receipt.get("model_digest")
                or initial.get("state_spec_value") != int(spec)
                or initial.get("state_size") != len(expected)
                or not isinstance(blob, bytes)
                or blob != expected.tobytes()
                or initial.get("state_digest") != "sha256:" + hashlib.sha256(blob).hexdigest()
                or initialization.get("kind") != "keyframe"
                or initialization.get("keyframe_name") != name
                or initialization.get("keyframe_index") != index
                or initialization.get("model_ref") != model_ref
                or initialization.get("model_digest") != receipt.get("model_digest")
                or trace.get("initial_state_ref") != receipt["initial_state_ref"]
                or trace.get("initialization") != initialization
                or key.get("kind") != "keyframe_initializer"
                or key.get("model_ref") != model_ref
                or key.get("model_digest") != receipt.get("model_digest")
                or key.get("name") != name
                or key.get("index") != index
                or key.get("time") != float(model.key_time[index])
                or any(
                    key.get(field) != getattr(model, "key_" + field)[index].tolist()
                    for field in ("qpos", "qvel", "act", "ctrl", "mpos", "mquat")
                )
            ):
                continue
            stats = trace_stats(trace, receipt)
            states = trace["states"]
            if (
                not _valid_stats(stats)
                or stats["end_time"] - stats["start_time"] < 0.5 - 1e-9
                or states[0]["qpos"] != data.qpos.tolist()
                or states[0]["qvel"] != data.qvel.tolist()
                or states[0]["t"] != float(data.time)
                or not runtime_validation_ok(trace, receipt)
                or max((abs(v) for v in states[-1]["qvel"]), default=0) >= 0.05
                or np.max(np.abs(np.asarray(states[-1]["qpos"]) - data.qpos), initial=0) >= 0.05
                or not replay_verified(backend, root, receipt["_ref"], receipt)
            ):
                continue
            return {
                "receipt_ref": receipt["_ref"],
                "initial_state_ref": receipt["initial_state_ref"],
                "keyframe_ref": initialization["keyframe_ref"],
                "model_ref": model_ref,
                "elapsed_s": stats["end_time"] - stats["start_time"],
                "settle_max_qvel": max((abs(v) for v in states[-1]["qvel"]), default=0),
                "settle_drift": float(
                    np.max(np.abs(np.asarray(states[-1]["qpos"]) - data.qpos), initial=0)
                ),
                "physics_steps_by_oracle": 0,
            }
        except (KeyError, TypeError, ValueError, FileNotFoundError):
            continue
    return None
