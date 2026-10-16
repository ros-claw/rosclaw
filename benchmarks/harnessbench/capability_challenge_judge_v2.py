"""AUTHOR reference candidate; Root-only, never stage to evaluation subjects.

No old judge, broad oracle, engine or execution imports. Independent Root review
is mandatory: these author-owned calculations are not an independent oracle.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from benchmarks.harnessbench.capability_challenges_v2 import (
    ABS_TOL,
    REL_TOL,
    SCOPE,
    VERSION,
    Instance,
    make_instance,
    sha,
)

MAX_BYTES = 65536


class Invalid(ValueError):  # noqa: N818 - Preserve the public Invalid exception API.
    pass


def require(ok: bool, code: str) -> None:
    if not ok:
        raise Invalid(code)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("DUPLICATE_KEY")
        out[key] = value
    return out


def _bad_constant(token: str) -> Any:
    raise ValueError("NONFINITE_JSON:" + token)


def parse(raw: bytes) -> Any:
    require(len(raw) <= MAX_BYTES, "SIZE_LIMIT")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant)


def finite(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return True


def equal(a: Any, b: Any) -> bool:
    if type(b) in (int, float):
        return (type(a) in (int, float) and math.isfinite(a)
                and math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL))
    if type(b) is list:
        return type(a) is list and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=False))
    if type(b) is dict:
        return type(a) is dict and a.keys() == b.keys() and all(equal(a[k], v) for k, v in b.items())
    return type(a) is type(b) and a == b


def _rotate(q: list[float], p: list[float]) -> list[float]:
    x, y, z, w = q
    return [(1-2*y*y-2*z*z)*p[0]+(2*x*y-2*z*w)*p[1]+(2*x*z+2*y*w)*p[2],
            (2*x*y+2*z*w)*p[0]+(1-2*x*x-2*z*z)*p[1]+(2*y*z-2*x*w)*p[2],
            (2*x*z-2*y*w)*p[0]+(2*y*z+2*x*w)*p[1]+(1-2*x*x-2*y*y)*p[2]]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _solve(f: str, p: dict[str, Any], aux: dict[str, bytes]) -> dict[str, Any]:
    """Candidate reference implementation, deliberately not part of public stage."""
    require(finite(p), "NONFINITE_INPUT")
    if f == "H01":
        links = []
        rr = {"best_effort": 0, "reliable": 1}
        dr = {"volatile": 0, "transient_local": 1}
        for e in sorted(p["edges"], key=lambda x: x["id"]):
            require(all(type(e.get(k)) is str and bool(e[k]) for k in ("type_requested", "type_offered")), "MISSING_TYPE")
            ok = e["type_requested"] == e["type_offered"] and e["domain_requested"] == e["domain_offered"]
            for key, rank in (("reliability", rr), ("durability", dr)):
                require(e[key+"_requested"] in rank and e[key+"_offered"] in rank, "INVALID_QOS")
                ok = ok and rank[e[key+"_offered"]] >= rank[e[key+"_requested"]]
            links.append({"id": e["id"], "compatible": ok})
        state = p["state"]
        transitions = {("unconfigured", "configure"): "inactive", ("inactive", "activate"): "active", ("active", "deactivate"): "inactive"}
        for event in p["events"]:
            require((state, event) in transitions, "INVALID_TRANSITION")
            state = transitions[state, event]
        topic = p["topic"]
        if topic.startswith("~"):
            topic = p["namespace"].rstrip("/") + "/" + p["node"] + "/" + topic[1:].lstrip("/")
        elif not topic.startswith("/"):
            topic = p["namespace"].rstrip("/") + "/" + topic
        return {"links": links, "topic": topic, "state": state}
    if f == "H02":
        q = p["q"]
        require(len(q) == 4 and abs(sum(v*v for v in q)-1) <= 1e-12, "INVALID_ROTATION")
        require(p["source"] == p["point_frame"], "FRAME_MISMATCH")
        rp = _rotate(q, p["point"])
        point = [a+b for a, b in zip(rp, p["t"], strict=False)]
        back = _rotate([-q[0], -q[1], -q[2], q[3]], [a-b for a, b in zip(point, p["t"], strict=False)])
        gravity = _rotate(q, p["accel"])
        gravity[2] -= 9.81
        normal = p["plane"][:3]
        norm = math.sqrt(sum(v*v for v in normal))
        require(norm > 0 and p["max_age"] >= 0, "INVALID_GEOMETRY")
        return {"point": point, "roundtrip": back, "gravity_free": gravity,
                "distance": (sum(a*b for a, b in zip(normal, point, strict=False))+p["plane"][3])/norm,
                "torque": [a+b for a, b in zip(_cross(p["t"], p["force"]), p["torque"], strict=False)],
                "frame": p["target"], "fresh": 0 <= p["now"]-p["stamp"] <= p["max_age"]}
    if f == "H03":
        stamps = p["anchors"] + p["clock"] + [v for stream in p["streams"] for v in stream]
        stamps += [p["action_stamp"], p["now"], p["max_age"], p["window"]] + p["seq"]
        require(all(type(v) is int for v in stamps) and p["window"] >= 0 and p["max_age"] >= 0, "INVALID_STAMP")
        used: list[set[int]] = [set() for _ in p["streams"]]
        rows = []
        for anchor in p["anchors"]:
            row = []
            for stream, consumed in zip(p["streams"], used, strict=False):
                candidates = [(abs(t-anchor), t, j) for j, t in enumerate(stream)
                              if j not in consumed and abs(t-anchor) <= p["window"]]
                j = min(candidates)[2] if candidates else -1
                if j >= 0:
                    consumed.add(j)
                row.append(j)
            rows.append(row)
        return {"matches": rows, "rewinds": [i for i in range(1, len(p["clock"])) if p["clock"][i] < p["clock"][i-1]],
                "lost": sum(max(0, b-a-1) for a, b in zip(p["seq"], p["seq"][1:], strict=False)),
                "fresh": 0 <= p["now"]-p["action_stamp"] <= p["max_age"]}
    if f == "H04":
        w, h, stride = p["width"], p["height"], p["stride"]
        require(all(type(v) is int and v > 0 for v in (w, h, stride)) and stride >= 6*w, "BUFFER_SHAPE")
        raw = bytes.fromhex(p["buffer"])
        require(len(raw) == h*stride, "BUFFER_LENGTH")
        require(sorted(p["channel_order"]) == ["B", "G", "R"] and p["endian"] in ("little", "big"), "BUFFER_TYPE")
        require(len(p["depth"]) == w*h and min(p["fx"], p["fy"], p["scale"]) > 0, "BUFFER_SHAPE")
        rgb, valid, xyz = [], [], []
        for i in range(w*h):
            u, v = i % w, i // w
            stored = struct.unpack_from(("<" if p["endian"] == "little" else ">")+"3H", raw, v*stride+u*6)
            rgb.append([stored[p["channel_order"].index(c)] for c in "RGB"])
            d = p["depth"][i]
            ok = type(d) in (int, float) and math.isfinite(d) and d > 0
            valid.append(ok)
            z = d*p["scale"] if ok else 0
            xyz.append([(u-p["cx"])*z/p["fx"], (v-p["cy"])*z/p["fy"], z] if ok else None)
        return {"rgb": rgb, "valid_depth": valid, "xyz": xyz}
    if f == "H05":
        require(p["frame"] == p["required_frame"], "FRAME_MISMATCH")
        grid, radius, res = p["grid"], p["radius"], p["resolution"]
        require(res > 0 and type(radius) is int and radius >= 0 and bool(grid) and bool(grid[0]), "INVALID_GRID")
        w, h = len(grid[0]), len(grid)
        require(all(len(row) == w and all(type(v) is int and v in (-1, 0, 100) for v in row) for row in grid), "INVALID_GRID")
        blocked = set()
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                if value != 0:
                    for yy in range(max(0, y-radius), min(h, y+radius+1)):
                        for xx in range(max(0, x-radius), min(w, x+radius+1)):
                            blocked.add((xx, yy))
        return {"voxels": [list(v) for v in sorted({tuple(math.floor(c/res) for c in point) for point in p["points"]})],
                "blocked": [list(v) for v in sorted(blocked)]}
    if f == "H06":
        nodes = p["nodes"]
        require(len(set(nodes)) == len(nodes) and p["start"] in nodes and p["goal"] in nodes, "INVALID_GRAPH")
        adj: dict[str, list[tuple[str, float]]] = {n: [] for n in nodes}
        forbidden = {frozenset(e) for e in p["forbidden"]}
        forbidden |= {frozenset(d["edge"]) for d in p["diagonals"] if not all(d["clear"])}
        for a, b, weight in p["edges"]:
            require(a in adj and b in adj and a != b and type(weight) in (int, float) and weight > 0, "INVALID_GRAPH")
            if frozenset((a, b)) not in forbidden:
                adj[a].append((b, weight))
                adj[b].append((a, weight))
        # Exhaustive small-graph search; independent Root optimum still required.
        candidates: list[tuple[float, list[str]]] = []
        def visit(path: list[str], cost: float) -> None:
            if path[-1] == p["goal"]:
                candidates.append((cost, path))
                return
            for node, weight in adj[path[-1]]:
                if node not in path:
                    visit(path+[node], cost+weight)
        require(len(nodes) <= 8, "INVALID_GRAPH")
        visit([p["start"]], 0)
        require(p["deceleration"] > 0 and p["speed"] >= 0 and p["clearance"] >= 0, "INVALID_GRAPH")
        cost, path = min(candidates) if candidates else (None, [])
        return {"reachable": bool(candidates), "cost": cost, "path": path,
                "braking_safe": p["speed"]**2/(2*p["deceleration"]) <= p["clearance"],
                "time_valid": all(b > a for a, b in zip(p["times"], p["times"][1:], strict=False))}
    if f == "H07":
        require(p["unit"] in ("m", "mm"), "INVALID_KINEMATICS")
        s = 1 if p["unit"] == "m" else 0.001
        require(0 <= p["min_reach"] <= p["max_reach"] and min(p["width"], p["max_width"]) >= 0, "INVALID_KINEMATICS")
        x = y = theta = 0.0
        for v, omega, dt in p["steps"]:
            require(dt > 0, "INVALID_KINEMATICS")
            x += v*s*dt*math.cos(theta)
            y += v*s*dt*math.sin(theta)
            theta += omega*dt
        r = math.hypot(*p["point"])
        return {"reachable": p["min_reach"] <= r <= p["max_reach"],
                "graspable": p["width"] <= p["max_width"], "odometry": [x, y, theta]}
    if f == "H08":
        require(p["P"] >= 0 and p["Q"] >= 0 and p["R"] > 0, "INVALID_UNCERTAINTY")
        require(type(p["window"]) is int and p["window"] > 0 and 0 <= p["alpha"] <= 1 and p["gate"] >= 0 and bool(p["samples"]), "INVALID_FILTER")
        ma, ema, posterior, variance = [], [], [], []
        x, P = p["initial"], p["P"]  # noqa: N806 - Preserve the established Kalman P/K notation.
        for i, z in enumerate(p["samples"]):
            subset = p["samples"][max(0, i-p["window"]+1):i+1]
            ma.append(sum(subset)/len(subset))
            ema.append(z if i == 0 else p["alpha"]*z+(1-p["alpha"])*ema[-1])
            P += p["Q"]  # noqa: N806 - Preserve the established Kalman P/K notation.
            K = P/(P+p["R"])  # noqa: N806 - Preserve the established Kalman P/K notation.
            x += K*(z-x)
            P *= 1-K  # noqa: N806 - Preserve the established Kalman P/K notation.
            posterior.append(x)
            variance.append(P)
        return {"ma": ma, "ema": ema, "posterior": posterior, "variance": variance,
                "accepted": [i for i, z in enumerate(p["samples"]) if abs(z-p["center"]) <= p["gate"]]}
    if f == "H09":
        rows = p["channels"]
        names, addresses = [r["name"] for r in rows], [r["address"] for r in rows]
        require(len(set(names)) == len(names) and len(set(addresses)) == len(addresses)
                and sorted(p["requested"]) == sorted(names) and all(type(a) is int and a >= 0 for a in addresses), "INVALID_JOINT_MAP")
        out = []
        for r in sorted(rows, key=lambda r: r["name"]):
            require(r["lo"] <= r["hi"] and r["lo"] <= r["previous"] <= r["hi"] and r["gear"] != 0 and r["period"] > 0 and r["dt"] > 0 and r["rate"] >= 0 and -1 <= r["a"] <= 1, "INVALID_LIMITS")
            pd = r["kp"]*(r["target"]-r["position"])-r["kd"]*r["velocity"]
            cap = max(r["lo"], min(r["hi"], pd))
            slew = r["rate"]*r["dt"]
            command = max(r["previous"]-slew, min(r["previous"]+slew, cap))
            out.append({"name": r["name"], "address": r["address"],
                        "delta": (r["new"]-r["old"]+r["period"]/2)%r["period"]-r["period"]/2,
                        "command": command, "force": command*r["gear"],
                        "action": r["lo"]+(r["a"]+1)*(r["hi"]-r["lo"])/2})
        return {"joints": out}
    if f == "H10":
        try:
            batch, cp = parse(aux["batch.json"]), parse(aux["checkpoint.json"])
        except (ValueError, UnicodeError, RecursionError):
            raise Invalid("INVALID_JSON") from None
        require(cp["revision"] == batch["revision"] == p["revision"] and cp["batch_sha256"] == sha(aux["batch.json"]), "STALE_CHECKPOINT")
        history = cp["history"]
        require(type(history) is list and all(type(v) is str for v in history) and len(set(history)) == len(history), "INVALID_HISTORY")
        required = len(batch["files"]) + (0 if "result.json" in history else 1)
        require(type(p["budget"]) is int and p["budget"] >= 0, "INVALID_BUDGET")
        if p["budget"] < required:
            return {"status": "BLOCKED", "code": "BUDGET_EXHAUSTED", "values": {"history": history, "required_calls": required}}
        matches = [m["id"] for m in p["landmarks"] if m["label"] == p["label"]]
        decision = "AMBIGUOUS" if len(matches) > 1 else "NOT_FOUND" if not matches else "WAIT"
        if len(matches) == 1 and "open" in p["events"] and "enter" in p["events"] and p["events"].index("open") < p["events"].index("enter"):
            decision = "ACT"
        return {"decision": decision, "target": matches[0] if len(matches) == 1 else None,
                "history": history if "result.json" in history else history+["result.json"], "required_calls": required}
    raise Invalid("UNKNOWN_FAMILY")


def reference_result(instance: Instance) -> dict[str, Any]:
    """Private candidate; only registered finite instances accepted."""
    require(instance == make_instance(instance.family, instance.role), "UNREGISTERED_INSTANCE")
    files = instance.staged()
    p = parse(files["input.json"])["data"]
    try:
        values = _solve(instance.family, p, files)
        if values.get("status") == "BLOCKED":
            return values
        return {"status": "OK", "code": "OK", "values": values}
    except Invalid as exc:
        return {"status": "INVALID", "code": str(exc), "values": None}
    # Unexpected bugs are not converted into reference answers.


def judge_bytes(task_id: str, files: dict[str, bytes], answer_bytes: bytes) -> dict[str, Any]:
    """Trusted caller supplies bytes; immutable generated plan is the authority."""
    checks: dict[str, bool] = {}
    try:
        family, suffix = task_id.split("-")
        role = {"S": "success", "I": "invalid", "M": "companion"}[suffix]
        instance = make_instance(family, role)
        original = instance.staged()
        checks["immutable_inputs"] = files == original
        answer = parse(answer_bytes)
        require(type(answer) is dict, "ANSWER_OBJECT")
        checks["keys"] = set(answer) == {"scope", "result", "input_sha256", "reason"}
        checks["scope"] = answer.get("scope") == SCOPE
        checks["input_binding"] = answer.get("input_sha256") == sha(original["input.json"])
        checks["reason"] = type(answer.get("reason")) is str and bool(answer["reason"].strip())
        # Do not evaluate changed inputs through any reference.
        checks["semantic_result"] = checks["immutable_inputs"] and equal(answer.get("result"), reference_result(instance))
    except (ValueError, KeyError, TypeError, OverflowError, UnicodeError, RecursionError):
        checks["parse_or_reference"] = False
    passed = bool(checks) and all(checks.values())
    return {"task_success": passed, "verified_success": passed, "checks": checks,
            "reason": "fixture_verified" if passed else "fixture_contract_failed",
            "version": VERSION, "evidence_scope": SCOPE, "live_integration": "NOT_RUN",
            "independent_oracle_review": "REQUIRED", "operational_recovery_verified": False}


def judge_challenge(task_id: str, root: Path) -> dict[str, Any]:
    """Read only bounded regular staged files. No writes or runtime execution."""
    try:
        family, suffix = task_id.split("-")
        instance = make_instance(family, {"S": "success", "I": "invalid", "M": "companion"}[suffix])
        def read(name: str) -> bytes:
            path = root / name
            require(not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_BYTES, "FILE_TYPE_OR_SIZE")
            with path.open("rb") as stream:
                data = stream.read(MAX_BYTES+1)
            require(len(data) <= MAX_BYTES, "SIZE_LIMIT")
            return data
        return judge_bytes(task_id, {name: read(name) for name, _ in instance.files}, read("answer.json"))
    except (OSError, ValueError, KeyError, TypeError):
        return {"task_success": False, "verified_success": False, "reason": "file_contract_failed",
                "evidence_scope": SCOPE, "live_integration": "NOT_RUN"}
