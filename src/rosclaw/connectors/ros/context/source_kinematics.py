"""Compare actual ECM links/joints with declared source DOF, never free poses."""

import hashlib
import math

from .geometry import IDENTITY, ZERO, _apply, _compose, _inverse
from .sim_source_binding import _numbers, _pose_matches, _transform, sdf_inventory


def _axis(vector):
    values = _numbers(vector, 3) if type(vector) is str else _numbers(" ".join(map(str, vector)), 3)
    length = math.hypot(*values)
    if not math.isfinite(length) or length <= 0:
        raise ValueError("finite nonzero observed and original joint axis required")
    return tuple(v / length for v in values)


def compile_source_kinematics(raw, *, robot_entity, world_name=None, urdf_bytes=None):
    inventory = sdf_inventory(
        raw, robot_entity=robot_entity, world_name=world_name, with_kinematics=True
    )
    source = inventory["kinematic_source"]
    from .sim_source_binding import _xml

    urdf_kinds = {}
    if urdf_bytes is not None:
        urdf = _xml(urdf_bytes)
        if urdf.tag != "robot" or urdf.get("name") != robot_entity:
            raise ValueError("matching original URDF for continuous joint semantics required")
        for joint in urdf.findall("joint"):
            if joint.get("name") in urdf_kinds:
                raise ValueError("unique original URDF joint names required")
            urdf_kinds[joint.get("name")] = joint.get("type")
    links, specs = source["link_transforms"], {}
    children = set()
    for row in source["joints"]:
        name, parent, child, kind = (row[k] for k in ("name", "parent", "child", "kind"))
        if (
            name in specs
            or parent not in links
            or child not in links
            or child in children
            or parent == child
        ):
            raise ValueError("one original supported link-joint tree required")
        if kind not in {"fixed", "revolute", "prismatic"}:
            raise ValueError("source joint requires unsupported kinematic adapter")
        if (
            urdf_bytes is not None
            and {
                "continuous": "revolute",
                "revolute": "revolute",
                "prismatic": "prismatic",
                "fixed": "fixed",
            }.get(urdf_kinds.get(name))
            != kind
        ):
            raise ValueError("source SDF joint differs from original URDF joint semantics")
        children.add(child)
        axis = None
        bounds = None
        if kind != "fixed":
            if row["axis"] is None or row["lower"] is None or row["upper"] is None:
                raise ValueError("explicit original joint axis and bounds required")
            axis = _axis(row["axis"])
            if row["axis_frame"]:
                if row["axis_frame"] not in source["frame_transforms"]:
                    raise ValueError("original axis expression frame unresolved")
                rotation = _compose(
                    _inverse(row["joint_transform"]), source["frame_transforms"][row["axis_frame"]]
                )[0]
                axis = _axis(_apply(rotation, axis))
            bounds = (float(row["lower"]), float(row["upper"]))
            if not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError("finite ordered original joint bounds required")
        specs[name] = {
            **row,
            "continuous": urdf_kinds.get(name) == "continuous",
            "normalized_axis": axis,
            "bounds": bounds,
            "parent_to_joint": _compose(_inverse(links[parent]), row["joint_transform"]),
            "joint_to_child": _compose(_inverse(row["joint_transform"]), links[child]),
        }
    if urdf_bytes is not None and {
        name for name, kind in urdf_kinds.items() if kind != "fixed"
    } != {name for name, row in specs.items() if row["kind"] != "fixed"}:
        raise ValueError("original movable URDF joints lost or changed in conversion")
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError("single original root link required")
    visited, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in visited:
            raise ValueError("cyclic original source joint tree")
        visited.add(name)
        pending.extend(row["child"] for row in specs.values() if row["parent"] == name)
    if visited != set(links):
        raise ValueError("disconnected original source joint tree")
    local_members = {
        key: _compose(_inverse(links[key[0]]), value)
        for key, value in inventory["member_transforms"].items()
    }
    return {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "robot_entity": robot_entity,
        "source": source,
        "joints": specs,
        "roots": roots,
        "local_members": local_members,
    }


def verify_source_kinematics(plan, sample, entity, *, position_tolerance, rotation_tolerance):
    """Check every loaded DOF and member pose at the same original ECM sample.

    No zero-joint assumption and no shape-symmetry exemption. Revolute links may
    rotate only around the original axis; prismatic links may translate only
    along it. Original topology, source bounds and existing precision remain.
    """
    observed = sample.get("joint_kinematics")
    if (
        type(observed) is not dict
        or set(observed) != {"schema_version", "complete", "links", "joints"}
        or observed.get("schema_version") != "rosclaw.ecm_joint_kinematics.v1"
        or observed.get("complete") is not True
    ):
        raise ValueError("current complete readonly ECM link and joint observations required")
    models = [m for m in sample["models"] if m["entity"] == entity]
    if len(models) != 1 or models[0]["name"] != plan["robot_entity"]:
        raise ValueError("kinematic sample model identity differs")
    model_inverse = _inverse(_transform(_numbers(" ".join(map(str, models[0]["world_pose"])), 6)))

    def selected(role):
        rows = observed.get(role)
        if type(rows) is not list or len(rows) > 512:
            raise ValueError("bounded original kinematic observations required")
        result, identities = {}, set()
        for row in rows:
            if (
                type(row) is not dict
                or type(row.get("entity")) is not int
                or row["entity"] <= 0
                or type(row.get("name")) is not str
                or not 1 <= len(row["name"]) <= 512
                or type(row.get("ancestry")) is not list
                or not 1 <= len(row["ancestry"]) <= 64
            ):
                raise ValueError("bounded concrete original kinematic entity required")
            if not any(x["entity"] == entity for x in row["ancestry"]):
                continue
            if row["name"] in result or row["entity"] in identities:
                raise ValueError("unique original link/joint entities required")
            if sum(x.get("model") is True for x in row["ancestry"]) != 1:
                raise ValueError("nonnested original kinematic ancestry required")
            result[row["name"]] = row
            identities.add(row["entity"])
        return result

    links, joints = selected("links"), selected("joints")
    if set(links) != set(plan["source"]["link_transforms"]) or set(joints) != set(plan["joints"]):
        raise ValueError("loaded link/joint inventory differs from original source")
    if {row["entity"] for row in links.values()} & {row["entity"] for row in joints.values()}:
        raise ValueError("link and joint identities overlap")
    transforms = {
        name: _compose(
            model_inverse, _transform(_numbers(" ".join(map(str, row["world_pose"])), 6))
        )
        for name, row in links.items()
    }
    for name in plan["roots"]:
        if not _pose_matches(
            transforms[name],
            plan["source"]["link_transforms"][name],
            position_tolerance,
            rotation_tolerance,
        ):
            raise ValueError("loaded root link differs from original model frame")
    positions = {}
    for name, spec in plan["joints"].items():
        row = joints[name]
        if any(row.get(k) != spec[k] for k in ("kind", "parent", "child")):
            raise ValueError("loaded joint type or topology differs from original source")
        actual_anchor = _transform(_numbers(" ".join(map(str, row.get("pose", ()))), 6))
        expected_anchor = _inverse(spec["joint_to_child"])
        if not _pose_matches(
            actual_anchor, expected_anchor, position_tolerance, rotation_tolerance
        ):
            raise ValueError("loaded joint child anchor differs from original source")
        relative = _compose(_inverse(transforms[spec["parent"]]), transforms[spec["child"]])
        delta = _compose(
            _compose(_inverse(spec["parent_to_joint"]), relative), _inverse(spec["joint_to_child"])
        )
        rotation, translation = delta
        if spec["kind"] == "fixed":
            if not _pose_matches(delta, (IDENTITY, ZERO), position_tolerance, rotation_tolerance):
                raise ValueError("loaded fixed joint moved")
            positions[name] = 0.0
            continue
        axis = _axis(row.get("axis", ()))
        expressed = row.get("axis_frame", "")
        if expressed:
            frames = plan["source"]["frame_transforms"]
            if expressed not in frames:
                raise ValueError("loaded joint axis frame unresolved")
            axis = _axis(
                _apply(_compose(_inverse(spec["joint_transform"]), frames[expressed])[0], axis)
            )
        if max(abs(a - b) for a, b in zip(axis, spec["normalized_axis"], strict=True)) > 1e-9:
            raise ValueError("loaded joint axis differs from original source")
        bounds = row.get("bounds")
        if (
            type(bounds) is not list
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or tuple(bounds) != spec["bounds"]
        ):
            raise ValueError("loaded joint limits differ from original source")
        if spec["kind"] == "revolute":
            if (
                max(abs(v) for v in translation) > position_tolerance
                or max(abs(a - b) for a, b in zip(_apply(rotation, axis), axis, strict=True))
                > rotation_tolerance
            ):
                raise ValueError("loaded revolute link violates original joint DOF")
            sine = sum(
                axis[i] * x
                for i, x in enumerate(
                    (
                        (rotation[2][1] - rotation[1][2]) / 2,
                        (rotation[0][2] - rotation[2][0]) / 2,
                        (rotation[1][0] - rotation[0][1]) / 2,
                    )
                )
            )
            q = math.atan2(sine, (sum(rotation[i][i] for i in range(3)) - 1) / 2)
            if not spec["continuous"]:
                direct = row.get("position")
                if (
                    type(direct) is not list
                    or len(direct) != 1
                    or type(direct[0]) not in (int, float)
                    or not math.isfinite(direct[0])
                ):
                    raise ValueError(
                        "bounded rotary joint requires original unwrapped ECM position"
                    )
                if (
                    not spec["bounds"][0] <= direct[0] <= spec["bounds"][1]
                    or abs(math.remainder(direct[0] - q, 2 * math.pi)) > rotation_tolerance
                ):
                    raise ValueError(
                        "observed unwrapped joint position or bounds differ from link pose"
                    )
                q = direct[0]
        else:
            q = sum(a * b for a, b in zip(translation, axis, strict=True))
            if (
                max(abs(a - q * b) for a, b in zip(translation, axis, strict=True))
                > position_tolerance
                or max(abs(rotation[i][j] - IDENTITY[i][j]) for i in range(3) for j in range(3))
                > rotation_tolerance
            ):
                raise ValueError("loaded prismatic link violates original joint DOF")
            if not spec["bounds"][0] <= q <= spec["bounds"][1]:
                raise ValueError("loaded translation outside original joint bounds")
        positions[name] = q
    members = {
        key: _compose(transforms[key[0]], local) for key, local in plan["local_members"].items()
    }
    return members, {
        "status": "MATCHED_DECLARED_SOURCE_JOINT_DOF",
        "source_sha256": plan["source_sha256"],
        "sample": sample["sample"],
        "joint_coordinates_from_checked_ECM_poses": positions,
        "execution_authority": "NONE",
    }
