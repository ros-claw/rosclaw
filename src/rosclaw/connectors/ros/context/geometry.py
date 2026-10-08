"""Read-only conservative collision envelope; never grants a Body binding.

Expanded URDF primitives and joint transforms are evidence, not a drive or
cleaner declaration. Unsupported meshes, frames or articulation stay UNKNOWN.
No vendor name, ROS topic, room dimensions or guessed radius is used.
"""

import hashlib
import itertools
import math
import xml.etree.ElementTree as ET

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO = (0.0, 0.0, 0.0)


def _vector(text, default=ZERO):
    values = tuple(float(v) for v in text.split()) if text is not None else default
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("URDF coordinates must be finite triples")
    return values


def _matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _apply(r, p):
    return _finite(tuple(sum(row[k] * p[k] for k in range(3)) for row in r))


def _finite(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError("collision transform overflow or nonfinite bound")
    return values


def _compose(parent, child):
    pr, pp = parent
    cr, cp = child
    translated = _apply(pr, cp)
    return _matmul(pr, cr), _finite(tuple(a + b for a, b in zip(pp, translated, strict=True)))


def _inverse(transform):
    r, p = transform
    rt = tuple(zip(*r, strict=True))
    return rt, tuple(-v for v in _apply(rt, p))


def _origin(element):
    o = element.find("origin")
    if o is None:
        return IDENTITY, ZERO
    roll, pitch, yaw = _vector(o.get("rpy"))
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    r = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return r, _vector(o.get("xyz"))


def _primitive(collision):
    geometry = collision.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise ValueError("collision requires one geometry")
    shape = geometry[0]
    if shape.tag == "box":
        size = _vector(shape.get("size"), ())
        if min(size) <= 0:
            raise ValueError("box sizes must be positive")
        half = tuple(v / 2 for v in size)
    elif shape.tag in ("cylinder", "sphere"):
        radius = float(shape.get("radius", "nan"))
        length = float(shape.get("length", "nan")) if shape.tag == "cylinder" else radius * 2
        if not all(math.isfinite(v) and v > 0 for v in [radius, length]):
            raise ValueError("radial primitive dimensions must be positive and finite")
        # A containing local box remains conservative under arbitrary rotation.
        half = (radius, radius, length / 2)
    else:
        raise ValueError(f"unsupported collision geometry: {shape.tag}")
    r, p = _origin(collision)
    return tuple(
        _finite(tuple(a + b for a, b in zip(_apply(r, point), p, strict=True)))
        for point in itertools.product(*[(-v, v) for v in half])
    )


def derive_collision_envelope(urdf_bytes, *, base_frame):
    """Fixed transforms are exact; movable subtrees use a containing sphere.

    Every collision geometry must resolve. Full-joint rotational sweeps and
    bounded prismatic motion are included. Mesh dimensions never fall back to
    a default. Output remains offline candidate evidence, even when complete.
    """
    if type(urdf_bytes) is not bytes:
        raise TypeError("expanded URDF bytes required")
    report = {
        "schema_version": "rosclaw.urdf_collision_envelope.v1",
        "evidence_domain": "OFFLINE_URDF",
        "evidence_role": "candidate_geometry_not_verified_binding",
        "source_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "base_frame": base_frame,
        "model_identity": None,
        "complete": False,
        "physical_radius_m": None,
        "unsupported_reasons": [],
        "collision_count": 0,
        "resolved_collision_count": 0,
        "capabilities_granted": [],
        "cleaning_attachment_inferred": False,
    }
    try:
        if not urdf_bytes or len(urdf_bytes) > 5_000_000:
            raise ValueError("URDF input size outside bounded limit")
        urdf_bytes.decode("utf-8", errors="strict")
        if b"\x00" in urdf_bytes:
            raise ValueError("UTF-8 expanded URDF required")
        if b"<!DOCTYPE" in urdf_bytes or b"<!ENTITY" in urdf_bytes:
            raise ValueError("URDF entity declarations forbidden")
        if any(marker in urdf_bytes for marker in (b"${", b"$(", b"<xacro:")):
            raise ValueError("unexpanded URDF expressions forbidden")
        root = ET.fromstring(urdf_bytes)
        if any(e.tag.startswith("{") for e in root.iter()):
            raise ValueError("unresolved namespaced URDF elements")
        if root.tag != "robot" or not root.get("name"):
            raise ValueError("named expanded URDF robot required")
        report["model_identity"] = root.get("name")
        elements = root.findall("link")
        links = {e.get("name"): e for e in elements}
        if (
            not 1 <= len(links) <= 512
            or len(links) != len(elements)
            or None in links
            or "" in links
        ):
            raise ValueError("URDF links must be unique, named and bounded")
        if base_frame not in links:
            raise ValueError("requested base frame absent from URDF")
        children = {name: [] for name in links}
        parents = {}
        joint_names = set()
        for j in root.findall("joint"):
            name = j.get("name")
            parent = j.find("parent")
            child = j.find("child")
            if not name or name in joint_names or parent is None or child is None:
                raise ValueError("joint identity/parent/child invalid")
            joint_names.add(name)
            a, b = parent.get("link"), child.get("link")
            if a not in links or b not in links or b in parents:
                raise ValueError("joint topology missing link or multiple parents")
            kind = j.get("type")
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"unsupported articulation: {kind}")
            travel = 0.0
            if kind != "fixed":
                axis = j.find("axis")
                vec = _vector(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0))
                if sum(v * v for v in vec) <= 0:
                    raise ValueError("joint axis cannot be zero")
            if kind == "prismatic":
                limit = j.find("limit")
                if limit is None:
                    raise ValueError("prismatic sweep requires bounded limits")
                low, high = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
                if not all(math.isfinite(v) for v in [low, high]) or low > high:
                    raise ValueError("prismatic limits invalid")
                travel = max(abs(low), abs(high))
            item = {"parent": a, "child": b, "kind": kind, "origin": _origin(j), "travel": travel}
            parents[b] = item
            children[a].append(item)
        roots = set(links) - set(parents)
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one connected root")
        root_name = next(iter(roots))
        visited = set()

        def topology(name):
            if name in visited:
                raise ValueError("joint cycle or duplicate traversal")
            visited.add(name)
            for j in children[name]:
                topology(j["child"])

        topology(root_name)
        if visited != set(links):
            raise ValueError("disconnected or cyclic collision links")
        base_chain = []
        cursor = base_frame
        while cursor in parents:
            j = parents[cursor]
            if j["kind"] != "fixed":
                raise ValueError("base frame has an unresolved moving ancestor")
            base_chain.append(j["origin"])
            cursor = j["parent"]
        root_to_base = (IDENTITY, ZERO)
        for t in reversed(base_chain):
            root_to_base = _compose(root_to_base, t)
        base_to_root = _inverse(root_to_base)
        points = {}
        for name, link in links.items():
            points[name] = []
            for collision in link.findall("collision"):
                report["collision_count"] += 1
                try:
                    points[name].extend(_primitive(collision))
                    report["resolved_collision_count"] += 1
                except (ValueError, TypeError) as exc:
                    report["unsupported_reasons"].append(f"{name}: {exc}")
        if not report["collision_count"]:
            raise ValueError("no collision evidence in URDF")
        if report["unsupported_reasons"]:
            return report

        def ball(name):
            radius = max((math.hypot(*p) for p in points[name]), default=0.0)
            for j in children[name]:
                offset = j["origin"][1]
                radius = max(radius, math.hypot(*offset) + j["travel"] + ball(j["child"]))
            return radius

        radius = 0.0

        def collect(name, transform):
            nonlocal radius
            r, p = transform
            for point in points[name]:
                q = _finite(tuple(a + b for a, b in zip(_apply(r, point), p, strict=True)))
                radius = max(radius, math.hypot(q[0], q[1]))
            for j in children[name]:
                t = _compose(transform, j["origin"])
                if j["kind"] == "fixed":
                    collect(j["child"], t)
                else:
                    radius = max(
                        radius, math.hypot(t[1][0], t[1][1]) + j["travel"] + ball(j["child"])
                    )

        collect(root_name, base_to_root)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("no positive conservative collision envelope")
        report.update(
            complete=True,
            physical_radius_m=radius,
            derivation="fixed RPY/XYZ transforms; containing primitive boxes; full movable-subtree sphere bounds",
        )
    except (ET.ParseError, ValueError, TypeError, RecursionError) as exc:
        report["unsupported_reasons"].append(str(exc))
    return report
