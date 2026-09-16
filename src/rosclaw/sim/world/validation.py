"""WorldSpec 校验（PR-MH7，规格 §22/§23/§39）。

fail closed：未知字段形状、非法几何、非法四元数、前向 depends_on、
未知 affordance/target/predicate 形式全部拒绝；绝对路径与外部
URI 一律拒绝。
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "rosclaw.sim.worldspec.v1"

OBJECT_SHAPES = ("box", "cylinder", "sphere", "capsule")
TARGET_TYPES = ("body", "geom", "joint", "actuator", "site", "camera")
BODY_REF_KINDS = ("eurdf", "task")  # menagerie 属 P1（规格 §34）
AFFORDANCES = ("press", "pull", "grasp", "place", "push", "inspect", "move")
PREDICATE_FORMS = ("inside", "near", "contact", "joint_in_range", "upright", "speed_below")


def _fail(message: str) -> None:
    raise ValueError(f"WORLDSPEC_INVALID: {message}")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{field} must be a finite number, got {value!r}")
    return float(value)


def _vec(value: Any, field: str, length: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        _fail(f"{field} must be a list of {length} numbers, got {value!r}")
    vec = [_finite(v, field) for v in value]
    if positive and any(v <= 0 for v in vec):
        _fail(f"{field} must be > 0, got {vec}")
    return vec


def _quat(value: Any, field: str) -> list[float]:
    vec = _vec(value, field, 4)
    norm = math.sqrt(sum(v * v for v in vec))
    if abs(norm - 1.0) > 1e-6:
        _fail(f"{field} quat norm must be 1, got |q|={norm}")
    return vec


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    if "://" in value or value.startswith("/"):
        _fail(f"{field} must not be a URI or absolute path: {value!r}")
    return value


def _validate_action_schema(schema: Any, field: str) -> dict[str, Any]:
    """action_schema 只开放 JSON-Schema 子集（Text2Mujoco 同款约束）。"""
    if not isinstance(schema, dict):
        _fail(f"{field} must be a mapping")
    if schema.get("type") != "object":
        _fail(f"{field}.type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        _fail(f"{field}.properties must be a mapping")
    for prop_name, prop in properties.items():
        if not isinstance(prop, dict):
            _fail(f"{field}.properties.{prop_name} must be a mapping")
        prop_type = prop.get("type")
        if prop_type not in ("number", "integer", "boolean", "string", "array"):
            _fail(f"{field}.properties.{prop_name}.type unsupported: {prop_type!r}")
        for bound in ("minimum", "maximum"):
            if bound in prop:
                _finite(prop[bound], f"{field}.{prop_name}.{bound}")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(r, str) for r in required):
        _fail(f"{field}.required must be a list of strings")
    unknown = [r for r in required if r not in properties]
    if unknown:
        _fail(f"{field}.required not in properties: {unknown}")
    if len(set(required)) != len(required):
        _fail(f"{field}.required has duplicates")
    return schema


def _validate_predicate(predicate: Any, field: str) -> dict[str, Any]:
    """success/failure 机器谓词：inside / near / contact /
    joint_in_range / upright / speed_below——全部机器可执行，
    不允许自然语言条件（0916 §十五）。"""
    if not isinstance(predicate, dict):
        _fail(f"{field} must be a mapping")
    form = next((f for f in PREDICATE_FORMS if f in predicate), None)
    if form is None:
        _fail(f"{field} must have exactly one of {PREDICATE_FORMS}")
    if form in ("inside", "near"):
        channel = predicate.get("channel")
        if not isinstance(channel, str) or ":" not in channel:
            _fail(f"{field}.channel must be '<channel>:<name>', got {channel!r}")
        if predicate.get("field") not in ("pos", "quat", "value"):
            _fail(f"{field}.field unsupported: {predicate.get('field')!r}")
    if form == "inside":
        box = predicate["inside"]
        if not isinstance(box, dict):
            _fail(f"{field}.inside must be a mapping")
        _vec(box.get("min"), f"{field}.inside.min", 3)
        _vec(box.get("max"), f"{field}.inside.max", 3)
        if any(lo > hi for lo, hi in zip(box["min"], box["max"], strict=True)):
            _fail(f"{field}.inside min > max")
    elif form == "near":
        near = predicate["near"]
        if not isinstance(near, dict):
            _fail(f"{field}.near must be a mapping")
        _vec(near.get("target"), f"{field}.near.target", 3)
        tolerance = _finite(near.get("tolerance"), f"{field}.near.tolerance")
        if tolerance <= 0:
            _fail(f"{field}.near.tolerance must be > 0")
    elif form == "contact":
        spec = predicate["contact"]
        if not isinstance(spec, dict):
            _fail(f"{field}.contact must be a mapping")
        if "max_dist" in spec:
            _finite(spec["max_dist"], f"{field}.contact.max_dist")
    elif form == "joint_in_range":
        spec = predicate["joint_in_range"]
        if not isinstance(spec, dict):
            _fail(f"{field}.joint_in_range must be a mapping")
        lo = _finite(spec.get("min"), f"{field}.joint_in_range.min")
        hi = _finite(spec.get("max"), f"{field}.joint_in_range.max")
        if lo > hi:
            _fail(f"{field}.joint_in_range min > max")
    elif form == "upright":
        spec = predicate["upright"]
        if not isinstance(spec, dict):
            _fail(f"{field}.upright must be a mapping")
        tilt = _finite(spec.get("max_tilt_deg", 15.0), f"{field}.upright.max_tilt_deg")
        if tilt <= 0 or tilt > 180:
            _fail(f"{field}.upright.max_tilt_deg out of (0, 180]")
    elif form == "speed_below":
        spec = predicate["speed_below"]
        if not isinstance(spec, dict):
            _fail(f"{field}.speed_below must be a mapping")
        limit = _finite(spec.get("max"), f"{field}.speed_below.max")
        if limit < 0:
            _fail(f"{field}.speed_below.max must be >= 0")
    return predicate


def validate_worldspec(spec: Any) -> dict[str, Any]:
    """校验并规范化 WorldSpec（填充默认值），非法即 WORLDSPEC_INVALID。"""
    if not isinstance(spec, dict):
        _fail("worldspec must be a mapping")
    if spec.get("schema_version") != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION!r}, got {spec.get('schema_version')!r}")

    world = spec.get("world", {})
    if not isinstance(world, dict):
        _fail("world must be a mapping")
    gravity = _vec(world.get("gravity", [0.0, 0.0, -9.81]), "world.gravity", 3)
    seed = world.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail(f"world.seed must be a non-negative int, got {seed!r}")
    timestep = _finite(world.get("timestep_s", 0.002), "world.timestep_s")
    if timestep <= 0:
        _fail("world.timestep_s must be > 0")
    ground = bool(world.get("ground", True))

    body_refs = []
    for index, body_ref in enumerate(spec.get("body_refs", [])):
        if not isinstance(body_ref, dict):
            _fail(f"body_refs[{index}] must be a mapping")
        kind = body_ref.get("kind")
        if kind not in BODY_REF_KINDS:
            _fail(f"body_refs[{index}].kind unsupported: {kind!r}")
        entry = {
            "id": _text(body_ref.get("id"), f"body_refs[{index}].id"),
            "kind": kind,
            "ref": _text(body_ref.get("ref"), f"body_refs[{index}].ref"),
            "pose": {
                "pos": _vec(
                    body_ref.get("pose", {}).get("pos", [0, 0, 0]),
                    f"body_refs[{index}].pose.pos",
                    3,
                ),
                "quat": _quat(
                    body_ref.get("pose", {}).get("quat", [1, 0, 0, 0]),
                    f"body_refs[{index}].pose.quat",
                ),
            },
        }
        body_refs.append(entry)
    if len({b["id"] for b in body_refs}) != len(body_refs):
        _fail("body_refs id duplicates")

    objects = []
    for index, obj in enumerate(spec.get("objects", [])):
        if not isinstance(obj, dict):
            _fail(f"objects[{index}] must be a mapping")
        shape = obj.get("shape")
        if shape not in OBJECT_SHAPES:
            _fail(f"objects[{index}].shape unsupported: {shape!r}")
        dynamic = bool(obj.get("dynamic", True))
        entry = {
            "id": _text(obj.get("id"), f"objects[{index}].id"),
            "shape": shape,
            "size": _vec(obj.get("size"), f"objects[{index}].size", 3, positive=True),
            "pos": _vec(obj.get("pos"), f"objects[{index}].pos", 3),
            "quat": _quat(obj.get("quat", [1, 0, 0, 0]), f"objects[{index}].quat"),
            "dynamic": dynamic,
            "mass": _finite(obj.get("mass", 1.0), f"objects[{index}].mass"),
            "rgba": _vec(obj.get("rgba", [0.7, 0.7, 0.7, 1.0]), f"objects[{index}].rgba", 4),
            "friction": _vec(
                obj.get("friction", [1.0, 0.01, 0.001]), f"objects[{index}].friction", 3
            ),
        }
        if any(v < 0 or v > 1 for v in entry["rgba"]):
            _fail(f"objects[{index}].rgba out of [0,1]")
        if entry["mass"] <= 0:
            _fail(f"objects[{index}].mass must be > 0")
        objects.append(entry)
    if len({o["id"] for o in objects}) != len(objects):
        _fail("objects id duplicates")

    sensors = []
    for index, sensor in enumerate(spec.get("sensors", [])):
        if not isinstance(sensor, dict) or sensor.get("type") != "camera":
            _fail(f"sensors[{index}] only supports type=camera")
        sensors.append(
            {
                "type": "camera",
                "name": _text(sensor.get("name"), f"sensors[{index}].name"),
                "pos": _vec(sensor.get("pos"), f"sensors[{index}].pos", 3),
                "quat": _quat(sensor.get("quat", [1, 0, 0, 0]), f"sensors[{index}].quat"),
            }
        )

    interactions = []
    seen_ids: set[str] = set()
    for index, point in enumerate(spec.get("interaction_points", [])):
        if not isinstance(point, dict):
            _fail(f"interaction_points[{index}] must be a mapping")
        point_id = _text(point.get("id"), f"interaction_points[{index}].id")
        affordance = point.get("affordance")
        if affordance not in AFFORDANCES:
            _fail(f"interaction_points[{index}].affordance unsupported: {affordance!r}")
        target = point.get("target")
        if not isinstance(target, dict) or target.get("type") not in TARGET_TYPES:
            _fail(f"interaction_points[{index}].target.type unsupported")
        target_name = _text(target.get("name"), f"interaction_points[{index}].target.name")
        depends_on = point.get("depends_on", [])
        if not isinstance(depends_on, list):
            _fail(f"interaction_points[{index}].depends_on must be a list")
        for dep in depends_on:
            if dep == point_id:
                _fail(f"interaction_points[{index}] self dependency")
            if dep not in seen_ids:
                _fail(
                    f"interaction_points[{index}].depends_on {dep!r} must reference an earlier point"
                )
        entry = {
            "id": point_id,
            "affordance": affordance,
            "target": {"type": target["type"], "name": target_name},
            "action_schema": _validate_action_schema(
                point.get("action_schema", {"type": "object", "properties": {}}),
                f"interaction_points[{index}].action_schema",
            ),
            "preconditions": [
                _validate_predicate(p, f"interaction_points[{index}].preconditions")
                for p in point.get("preconditions", [])
            ],
            "success": [
                _validate_predicate(p, f"interaction_points[{index}].success")
                for p in point.get("success", [])
            ],
            "effects": list(point.get("effects", [])),
            "depends_on": list(depends_on),
        }
        interactions.append(entry)
        seen_ids.add(point_id)

    task = spec.get("task", {})
    if not isinstance(task, dict):
        _fail("task must be a mapping")
    task_spec = {
        "goal": str(task.get("goal", "")),
        "success": [_validate_predicate(p, "task.success") for p in task.get("success", [])],
        "failure": [_validate_predicate(p, "task.failure") for p in task.get("failure", [])],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "world": {
            "gravity": gravity,
            "ground": ground,
            "seed": seed,
            "timestep_s": timestep,
        },
        "body_refs": body_refs,
        "objects": objects,
        "sensors": sensors,
        "interaction_points": interactions,
        "task": task_spec,
        "randomization": dict(spec.get("randomization", {})),
        "evidence_requirements": list(spec.get("evidence_requirements", [])),
    }
