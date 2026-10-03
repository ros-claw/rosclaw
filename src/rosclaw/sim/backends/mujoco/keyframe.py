"""Explicit model-bound keyframe identity and actual MuJoCo initialization."""

from __future__ import annotations

import hashlib
from typing import Any

from rosclaw.contracts.common import canonical_json
from rosclaw.sim.refs import make_ref


def reset_keyframe(
    model, data, store, *, model_ref: str, model_digest: str, name: str | None, ref: str | None
) -> dict[str, Any]:  # noqa: ANN001
    import mujoco

    record = None
    if ref is not None:
        if not isinstance(ref, str) or not ref.startswith("simkey_"):
            raise ValueError("KEYFRAME_REF_INVALID: expected a model-bound simkey reference")
        record = store.get(ref)
        if not isinstance(record, dict) or record.get("kind") != "keyframe_initializer":
            raise ValueError("KEYFRAME_REF_INVALID: reference is not a keyframe initializer")
        if record.get("model_ref") != model_ref or record.get("model_digest") != model_digest:
            raise ValueError("KEYFRAME_REF_MISMATCH: keyframe belongs to another model")
        name = record.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("KEYFRAME_NAME_INVALID: provide a nonempty exact keyframe name")
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if index < 0:
        raise ValueError(f"KEYFRAME_NOT_FOUND: {name!r}; default initialization was not used")
    expected = {
        "schema_version": "rosclaw.sim.keyframe_initializer.v1",
        "kind": "keyframe_initializer",
        "model_ref": model_ref,
        "model_digest": model_digest,
        "name": name,
        "index": index,
        "time": float(model.key_time[index]),
        **{
            field: [float(value) for value in getattr(model, "key_" + field)[index]]
            for field in ("qpos", "qvel", "act", "ctrl", "mpos", "mquat")
        },
    }
    if record is not None and record != expected:
        raise ValueError("KEYFRAME_REF_MISMATCH: reference differs from compiled keyframe")
    actual_ref = make_ref("simkey", hashlib.sha256(canonical_json(expected).encode()).hexdigest())
    store.put("states", expected, ref=actual_ref)
    mujoco.mj_resetDataKeyframe(model, data, index)
    mujoco.mj_forward(model, data)
    return {
        "kind": "keyframe",
        "keyframe_name": name,
        "keyframe_index": index,
        "keyframe_ref": actual_ref,
        "model_ref": model_ref,
        "model_digest": model_digest,
    }
