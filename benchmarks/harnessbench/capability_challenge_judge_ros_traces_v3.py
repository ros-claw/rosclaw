"""Private author candidate, not an independent Root oracle. Never stage to subjects.

Pure byte judging only; no providers, dynamic libraries, ROS, engines or old judge.
"""
from __future__ import annotations

import json
import math
import re
import struct
from typing import Any

from benchmarks.harnessbench.capability_challenges_ros_traces_v3 import (
    ABS_TOL,
    EXPONENT_ABS,
    FLOAT_ABS,
    H17_CALLBACKS,
    H17_EDGES,
    H17_EVENTS,
    H17_GROUPS,
    H18_OUTPUT_EVENTS,
    H18_PUBLICATIONS,
    H18_TIME,
    INPUT_CONTAINER,
    INPUT_FLOAT_ABS,
    INPUT_INTEGER_ABS,
    INPUT_NODES,
    INPUT_STRING_BYTES,
    INPUT_TOTAL_STRING_BYTES,
    INTEGER_DIGITS,
    JSON_DEPTH,
    JSON_TOKENS,
    MAX_BYTES,
    NUMBER_CHARS,
    REL_TOL,
    VERSION,
    Instance,
    sha,
)


class Invalid(ValueError):  # noqa: N818 - Preserve the public Invalid exception API.
    pass


def need(ok: bool, code: str = "INVALID_INPUT") -> None:
    if not ok:
        raise Invalid(code)


def keys(x: Any, names: str, code: str = "INVALID_INPUT") -> None:
    need(type(x) is dict and set(x) == set(names.split()), code)


def integer(x: Any) -> bool:
    return type(x) is int


def number(x: Any) -> bool:
    return type(x) in (int, float) and (type(x) is int or math.isfinite(x))


def text(x: Any) -> bool:
    return type(x) is str and bool(x)


def strings(x: Any) -> bool:
    return type(x) is list and all(text(v) for v in x)


def finite(x: Any) -> bool:
    if type(x) is float:
        return math.isfinite(x)
    if type(x) is list:
        return all(finite(v) for v in x)
    if type(x) is dict:
        return all(finite(v) for v in x.values())
    return True


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate key")
        d[k] = v
    return d


def _constant(value: str) -> Any:
    raise ValueError(value)


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def _json_number(token: str) -> int | float:
    """All checks precede potentially costly numeric conversion."""
    need(len(token) <= NUMBER_CHARS, "RESOURCE_LIMIT")
    is_int = not any(c in token for c in ".eE")
    if is_int:
        need(len(token.removeprefix("-")) <= INTEGER_DIGITS, "RESOURCE_LIMIT")
    need(_NUMBER.fullmatch(token) is not None, "INVALID_JSON")
    if is_int:
        return int(token)
    parts = re.split("[eE]", token)
    if len(parts) == 2:
        # Token length is already bounded, including zero-padded exponents.
        need(abs(int(parts[1])) <= EXPONENT_ABS, "RESOURCE_LIMIT")
    value = float(token)
    need(math.isfinite(value) and abs(value) <= FLOAT_ABS, "RESOURCE_LIMIT")
    return value


def _preflight(source: str) -> None:
    """Linear lexical admission before recursive JSON decoding/allocation.

    This is not a replacement JSON parser. Malformed syntax within bounded
    lexemes is rejected by json.loads; resource precedence is public.
    """
    i = depth = tokens = 0
    separators = '[]{}:," \t\r\n'
    while i < len(source):
        c = source[i]
        if c in " \t\r\n:,":
            i += 1
            continue
        if c in "]}":
            depth -= 1
            need(depth >= 0, "INVALID_JSON")
            i += 1
            continue
        tokens += 1
        need(tokens <= JSON_TOKENS, "RESOURCE_LIMIT")
        if c in "[{":
            depth += 1
            need(depth <= JSON_DEPTH, "RESOURCE_LIMIT")
            i += 1
        elif c == '"':
            i += 1
            while i < len(source):
                if source[i] == '"':
                    i += 1
                    break
                if source[i] == "\\":
                    i += 2
                else:
                    i += 1
            # Unterminated/invalid escaped strings are decoder errors.
        else:
            start = i
            while i < len(source) and source[i] not in separators:
                i += 1
            token = source[start:i]
            if c == "-" or c in "0123456789":
                _json_number(token)
            else:
                need(token in ("true", "false", "null"), "INVALID_JSON")


def parse(raw: bytes) -> Any:
    need(type(raw) is bytes and len(raw) <= MAX_BYTES, "SIZE_LIMIT")
    try:
        source = raw.decode("utf-8", errors="strict")
        _preflight(source)
        value = json.loads(source, object_pairs_hook=_pairs,
                           parse_int=_json_number, parse_float=_json_number,
                           parse_constant=_constant)
    except Invalid:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise Invalid("INVALID_JSON") from exc
    need(finite(value))
    return value


def _input_domain(value: Any) -> None:
    """Bound all later family work, before family shape/semantic checks."""
    nodes = string_bytes = 0
    stack = [value]
    while stack:
        item = stack.pop()
        nodes += 1
        need(nodes <= INPUT_NODES, "RESOURCE_LIMIT")
        if type(item) is str:
            size = len(item.encode("utf-8", errors="surrogatepass"))
            string_bytes += size
            need(size <= INPUT_STRING_BYTES and string_bytes <= INPUT_TOTAL_STRING_BYTES,
                 "RESOURCE_LIMIT")
        elif type(item) is int:
            need(abs(item) <= INPUT_INTEGER_ABS, "RESOURCE_LIMIT")
        elif type(item) is float:
            need(abs(item) <= INPUT_FLOAT_ABS, "RESOURCE_LIMIT")
        elif type(item) in (dict, list):
            need(len(item) <= INPUT_CONTAINER, "RESOURCE_LIMIT")
            if type(item) is dict:
                # Reverse push preserves depth-first insertion/key-value order.
                for k, v in reversed(list(item.items())):
                    stack.extend((v, k))
            else:
                stack.extend(reversed(item))


def _wire(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "mode byteorder payload wire_names")
    need(p["mode"] in ("encode", "decode") and strings(p["wire_names"]) and len(p["wire_names"]) == 3 and len(set(p["wire_names"])) == 3)
    need(p["byteorder"] == "little", "BYTEORDER")
    if p["mode"] == "encode":
        d = p["payload"]
        keys(d, "id label groups")
        need(integer(d["id"]) and type(d["label"]) is str and type(d["groups"]) is list)
        for g in d["groups"]:
            keys(g, "tag samples")
            need(integer(g["tag"]) and type(g["samples"]) is list and all(integer(v) for v in g["samples"]))
        need(0 <= d["id"] < 2**32 and len(d["groups"]) < 2**32 and all(0 <= g["tag"] < 256 and len(g["samples"]) < 2**32 and all(-32768 <= v <= 32767 for v in g["samples"]) for g in d["groups"]), "WIDTH")
        try:
            label = d["label"].encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise Invalid("UTF8") from exc
        need(len(label) < 2**32, "WIDTH")
        raw = struct.pack("<II", d["id"], len(label)) + label + struct.pack("<I", len(d["groups"]))
        for g in d["groups"]:
            raw += struct.pack("<BI", g["tag"], len(g["samples"]))
            raw += b"".join(struct.pack("<h", v) for v in g["samples"])
    else:
        need(type(p["payload"]) is str)
        need(re.fullmatch(r"(?:[0-9a-f]{2})*", p["payload"]) is not None, "HEX")
        raw = bytes.fromhex(p["payload"])
        offset = 0

        def read(n: int) -> bytes:
            nonlocal offset
            need(offset + n <= len(raw), "TRUNCATED")
            out = raw[offset:offset+n]
            offset += n
            return out

        def u32() -> int:
            return struct.unpack("<I", read(4))[0]

        ident = u32()
        label_bytes = read(u32())
        try:
            label_text = label_bytes.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise Invalid("UTF8") from exc
        count = u32()
        groups = []
        # Each iteration must consume >=5 bytes; huge claimed lengths fail
        # on the first unavailable read, not allocation of claimed array size.
        for _ in range(count):
            tag = read(1)[0]
            n = u32()
            sample_bytes = read(2*n)
            samples = [v[0] for v in struct.iter_unpack("<h", sample_bytes)]
            groups.append({"tag": tag, "samples": samples})
        need(offset == len(raw), "TRAILING")
        d = {"id": ident, "label": label_text, "groups": groups}
    return {"encoded_hex": raw.hex(), "decoded": d, "byte_count": len(raw)}


def _path(s: Any, absolute: bool = False) -> bool:
    if not text(s):
        return False
    if s == "/":
        return True
    if absolute and not s.startswith("/"):
        return False
    parts = s[1:].split("/") if s.startswith("/") else s.split("/")
    return all(v not in ("", ".", "..") and "~" not in v for v in parts)


def _parameter(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "namespace key parameters remaps expect")
    need(type(p["parameters"]) is dict and type(p["remaps"]) is dict and p["expect"] in ("any", "null", "boolean", "integer", "number", "string", "array", "object"))
    need(_path(p["namespace"], True) and _path(p["key"]) and all(_path(v, True) for v in p["parameters"]) and all(_path(k) and _path(v) for k, v in p["remaps"].items()), "PATH")
    for start in p["remaps"]:
        seen: set[str] = set()
        at = start
        while at in p["remaps"]:
            need(at not in seen, "REMAP_CYCLE")
            seen.add(at)
            at = p["remaps"][at]
    key = p["remaps"].get(p["key"], p["key"])
    if key.startswith("/"):
        candidates = [key]
    else:
        parts = p["namespace"].strip("/").split("/") if p["namespace"] != "/" else []
        candidates = ["/" + "/".join(parts[:i] + [key]) for i in range(len(parts), -1, -1)]
    visited = []
    for candidate in candidates:
        visited.append(candidate)
        if candidate in p["parameters"]:
            value = p["parameters"][candidate]
            kind = {type(None): "null", bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object"}[type(value)]
            need(p["expect"] == "any" or p["expect"] == kind or (p["expect"] == "number" and kind == "integer"), "TYPE_MISMATCH")
            return {"resolved_path": candidate, "found": True, "value": value, "visited": visited}
    return {"resolved_path": None, "found": False, "value": None, "visited": visited}


def _epochs(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "records clock anchor_topic streams maxgap")
    need(type(p["records"]) is list and type(p["clock"]) is list and text(p["anchor_topic"]) and strings(p["streams"]))
    for r in p["records"]:
        keys(r, "topic stamp_ns sequence")
        need(text(r["topic"]))
    need(len(p["records"]) == len(p["clock"]), "LENGTH")
    times = p["clock"] + [r[k] for r in p["records"] for k in ("stamp_ns", "sequence")] + [p["maxgap"]]
    need(all(integer(v) and v >= 0 for v in times), "TIME")
    need(len(set(p["streams"])) == len(p["streams"]) and p["anchor_topic"] not in p["streams"], "IDENTITY")
    epochs, rewinds, seen = [], [], set()
    epoch = 0
    for i, r in enumerate(p["records"]):
        if i and p["clock"][i] < p["clock"][i-1]:
            epoch += 1
            rewinds.append(i)
        epochs.append(epoch)
        identity = (epoch, r["topic"], r["sequence"])
        need(identity not in seen, "IDENTITY")
        seen.add(identity)
    matches = []
    for i, a in enumerate(p["records"]):
        if a["topic"] != p["anchor_topic"]:
            continue
        row = []
        for topic in p["streams"]:
            options = [(abs(r["stamp_ns"]-a["stamp_ns"]), r["stamp_ns"], j) for j, r in enumerate(p["records"]) if epochs[j] == epochs[i] and r["topic"] == topic and abs(r["stamp_ns"]-a["stamp_ns"]) <= p["maxgap"]]
            row.append(min(options)[2] if options else -1)
        matches.append(row)
    return {"epochs": epochs, "matches": matches, "rewinds": rewinds}


def _transpose(r: list[Any]) -> list[Any]:
    return [r[j*3+i] for i in range(3) for j in range(3)]


def _mul(a: list[Any], b: list[Any]) -> list[Any]:
    return [sum(a[i*3+k]*b[k*3+j] for k in range(3)) for i in range(3) for j in range(3)]


def _mv(r: list[Any], t: list[Any]) -> list[Any]:
    return [sum(r[i*3+j]*t[j] for j in range(3)) for i in range(3)]


def _tf(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "edges source target")
    need(type(p["edges"]) is list and text(p["source"]) and text(p["target"]))
    ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    for e in p["edges"]:
        keys(e, "parent child rotation translation")
        need(text(e["parent"]) and text(e["child"]) and type(e["rotation"]) is list and len(e["rotation"]) == 9 and type(e["translation"]) is list and len(e["translation"]) == 3 and all(number(v) for v in e["rotation"] + e["translation"]))
    for e in p["edges"]:
        r = e["rotation"]
        det = r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6])
        need(abs(det-1) <= ABS_TOL and all(abs(x-y) <= ABS_TOL for x, y in zip(_mul(_transpose(r), r), ident, strict=False)), "BAD_ROTATION")
    parents: dict[str, str] = {}
    for e in p["edges"]:
        need(e["parent"] != e["child"] and e["child"] not in parents, "MULTIPLE_PARENTS")
        parents[e["child"]] = e["parent"]
    for start in parents:
        seen: set[str] = set()
        at = start
        while at in parents:
            need(at not in seen, "CYCLE")
            seen.add(at)
            at = parents[at]
    graph: dict[str, list[Any]] = {}
    for e in p["edges"]:
        r, t = e["rotation"], e["translation"]
        inv = _transpose(r)
        graph.setdefault(e["child"], []).append((e["parent"], r, t))
        graph.setdefault(e["parent"], []).append((e["child"], inv, [-v for v in _mv(inv, t)]))
    queue = [(p["source"], ident, [0, 0, 0], [p["source"]])]
    seen = set()
    for at, rotation, translation, path in queue:
        if at == p["target"]:
            return {"reachable": True, "rotation": rotation, "translation": translation, "path": path}
        seen.add(at)
        for other, r, t in graph.get(at, []):
            if other not in seen:
                queue.append((other, _mul(r, rotation), [x+y for x, y in zip(_mv(r, translation), t, strict=False)], path + [other]))
    return {"reachable": False, "rotation": None, "translation": None, "path": []}


def _bridge(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "header duration samples bound allow_seq_drop")
    keys(p["header"], "seq stamp frame_id")
    keys(p["header"]["stamp"], "sec nsec")
    keys(p["duration"], "sec nsec")
    h = p["header"]
    need(integer(h["seq"]) and type(h["frame_id"]) is str and all(integer(x[k]) for x in (h["stamp"], p["duration"]) for k in ("sec", "nsec")) and type(p["samples"]) is list and all(number(x) for x in p["samples"]) and integer(p["bound"]) and p["bound"] >= 0 and type(p["allow_seq_drop"]) is bool)
    need(0 <= h["seq"] < 2**32, "OVERFLOW")
    st = h["stamp"]["sec"]*10**9+h["stamp"]["nsec"]
    dt = p["duration"]["sec"]*10**9+p["duration"]["nsec"]
    need(st >= 0, "NEGATIVE_STAMP")
    s, ns = divmod(st, 10**9)
    d, nd = divmod(dt, 10**9)
    need(0 <= s < 2**31 and -2**31 <= d < 2**31, "OVERFLOW")
    need(len(p["samples"]) <= p["bound"], "ARRAY_BOUND")
    need(h["seq"] == 0 or p["allow_seq_drop"], "LOSSY")
    return {"header": {"stamp": {"sec": s, "nanosec": ns}, "frame_id": h["frame_id"]}, "duration": {"sec": d, "nanosec": nd}, "samples": p["samples"], "dropped_fields": ["header.seq"] if h["seq"] else []}


def _abi(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "required offered", "MANIFEST")
    fields = ("architecture", "abi", "type_name", "type_hash")
    for m in p.values():
        keys(m, "architecture abi type_name type_hash symbols", "MANIFEST")
        need(all(text(m[k]) for k in fields) and strings(m["symbols"]) and len(set(m["symbols"])) == len(m["symbols"]), "MANIFEST")
    r, o = p["required"], p["offered"]
    mismatches = [k for k in fields if r[k] != o[k]]
    missing = sorted(set(r["symbols"])-set(o["symbols"]))
    return {"compatible": not (mismatches or missing), "missing_symbols": missing, "mismatches": mismatches}


def _executor(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "groups callbacks events")
    need(type(p["groups"]) is dict and type(p["callbacks"]) is dict and type(p["events"]) is list)
    need(len(p["groups"]) <= H17_GROUPS and len(p["callbacks"]) <= H17_CALLBACKS
         and len(p["events"]) <= H17_EVENTS, "RESOURCE_LIMIT")
    need(all(text(k) and v in ("mutuallyexclusive", "reentrant") for k, v in p["groups"].items()) and all(text(k) and text(v) and v in p["groups"] for k, v in p["callbacks"].items()), "DECLARATION")
    active: set[str] = set()
    waits: set[tuple[str, str]] = set()
    for e in p["events"]:
        need(type(e) is dict and type(e.get("op")) is str, "EVENT")
        op = e["op"]
        need(op in ("start", "finish", "wait", "unwait"), "EVENT")
        keys(e, "op callback target" if op in ("wait", "unwait") else "op callback", "EVENT")
        c = e["callback"]
        need(text(c) and c in p["callbacks"], "EVENT")
        if op == "start":
            group = p["callbacks"][c]
            need(c not in active and (p["groups"][group] == "reentrant" or all(p["callbacks"][a] != group for a in active)), "EVENT")
            active.add(c)
        elif op == "finish":
            need(c in active, "EVENT")
            active.remove(c)
            waits = {(a, b) for a, b in waits if c not in (a, b)}
        else:
            t = e["target"]
            need(text(t) and t != c and c in active and t in active, "EVENT")
            if op == "wait":
                need((c, t) in waits or len(waits) < H17_EDGES, "RESOURCE_LIMIT")
                waits.add((c, t))
            else:
                need((c, t) in waits, "EVENT")
                waits.remove((c, t))
    cycles: list[list[str]] = []
    # Only start from the minimum vertex of a simple cycle, fixing rotation.
    def walk(path: list[str]) -> None:
        for a, b in sorted(waits):
            if a != path[-1]:
                continue
            if b == path[0]:
                cycles.append(path + [b])
            elif b not in path and b > path[0]:
                walk(path + [b])
    for c in sorted(active):
        walk([c])
    return {"active": sorted(active), "waiting": [list(v) for v in sorted(waits)], "deadlocked": bool(cycles), "cycle": min(cycles) if cycles else []}


def _qos(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "publications horizon deadline lease")
    need(type(p["publications"]) is list)
    need(len(p["publications"]) <= H18_PUBLICATIONS, "RESOURCE_LIMIT")
    need(all(integer(t) and t >= 0 for t in p["publications"] + [p["horizon"]]) and all(integer(p[k]) and p[k] > 0 for k in ("deadline", "lease")), "TIME")
    need(all(t <= H18_TIME for t in p["publications"] + [p[k] for k in ("horizon", "deadline", "lease")]), "RESOURCE_LIMIT")
    pubs = p["publications"]
    need(all(a <= b for a, b in zip(pubs, pubs[1:], strict=False)) and (not pubs or p["horizon"] >= pubs[-1]), "ORDER")
    # Count with bounded integers before any range expansion or output allocation.
    count = 0
    for i, t in enumerate(pubs):
        final = i == len(pubs)-1
        end = p["horizon"] if final else pubs[i+1]-1
        count += max(0, (end-t)//p["deadline"])
        if t + p["lease"] <= end:
            count += 1 if final else 2
        need(count <= H18_OUTPUT_EVENTS, "RESOURCE_LIMIT")
    misses: list[int] = []
    losses: list[int] = []
    recovered: list[int] = []
    for i, t in enumerate(pubs):
        final = i == len(pubs)-1
        end = p["horizon"] if final else pubs[i+1]-1
        misses.extend(range(t+p["deadline"], end+1, p["deadline"]))
        if t + p["lease"] <= end:
            losses.append(t+p["lease"])
            if not final:
                recovered.append(pubs[i+1])
    alive = bool(pubs) and p["horizon"] < pubs[-1] + p["lease"]
    return {"deadline_misses": misses, "liveliness_lost": losses, "recovered": recovered, "alive": alive}


def _transaction(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "parameters updates expected_revision revision constraints")
    need(type(p["parameters"]) is dict and type(p["updates"]) is dict and type(p["constraints"]) is list and all(integer(p[k]) and p[k] >= 0 for k in ("expected_revision", "revision")))
    need(all(text(k) and type(v) in (str, bool, int, float, type(None)) for m in (p["parameters"], p["updates"]) for k, v in m.items()))
    merged = {**p["parameters"], **p["updates"]}
    ids = set()
    for c in p["constraints"]:
        keys(c, "id left op right", "PREDICATE")
        need(all(text(c[k]) for k in ("id", "left", "right", "op")) and c["id"] not in ids and c["left"] in merged and c["right"] in merged and c["op"] in ("eq", "ne", "lt", "le", "gt", "ge"), "PREDICATE")
        ids.add(c["id"])
    base = {"committed": False, "revision": p["revision"], "parameters": p["parameters"], "failed_constraints": [], "rejection": "STALE_REVISION"}
    if p["expected_revision"] != p["revision"]:
        return base
    for c in p["constraints"]:
        a, b = merged[c["left"]], merged[c["right"]]
        need((number(a) and number(b)) or (c["op"] in ("eq", "ne") and type(a) is type(b)), "PREDICATE")
    failed = []
    for c in p["constraints"]:
        a, b = merged[c["left"]], merged[c["right"]]
        op = c["op"]
        ok = (a == b if op == "eq" else a != b if op == "ne" else a < b if op == "lt" else a <= b if op == "le" else a > b if op == "gt" else a >= b)
        if not ok:
            failed.append(c["id"])
    if failed:
        return {**base, "failed_constraints": failed, "rejection": "CONSTRAINT"}
    return {"committed": True, "revision": p["revision"]+1, "parameters": merged, "failed_constraints": [], "rejection": "NONE"}


def _actions(p: dict[str, Any]) -> dict[str, Any]:
    keys(p, "events")
    need(type(p["events"]) is list)
    goals: dict[str, Any] = {}
    ignored = []
    terminals = ("SUCCEEDED", "CANCELLED", "FAILED")
    ops = ("submit", "accept", "start", "cancel_request", "cancel_ack", "result_success", "result_cancelled", "result_failure")
    for i, e in enumerate(p["events"]):
        keys(e, "op id", "EVENT")
        need(text(e["id"]) and type(e["op"]) is str and e["op"] in ops, "EVENT")
        ident, op = e["id"], e["op"]
        if op == "submit":
            need(ident not in goals, "EVENT")
            goals[ident] = {"id": ident, "state": "SUBMITTED", "cancel_requested": False, "terminal_event": None}
            continue
        need(ident in goals, "EVENT")
        g = goals[ident]
        state = g["state"]
        if state in terminals:
            ignored.append(i)
            continue
        duplicate = (op == "accept" and state == "ACCEPTED") or (op == "start" and state == "RUNNING") or (op == "cancel_request" and state in ("CANCEL_PENDING", "CANCEL_ACKED")) or (op == "cancel_ack" and state == "CANCEL_ACKED")
        if duplicate:
            ignored.append(i)
            continue
        if op.startswith("result_"):
            need(state == "CANCEL_ACKED" if op == "result_cancelled" else state in ("ACCEPTED", "RUNNING", "CANCEL_PENDING", "CANCEL_ACKED"), "EVENT")
            g["state"] = {"result_success": "SUCCEEDED", "result_cancelled": "CANCELLED", "result_failure": "FAILED"}[op]
            g["terminal_event"] = i
        else:
            allowed = {"accept": ("SUBMITTED",), "start": ("ACCEPTED",), "cancel_request": ("ACCEPTED", "RUNNING"), "cancel_ack": ("CANCEL_PENDING",)}
            need(state in allowed[op], "EVENT")
            g["state"] = {"accept": "ACCEPTED", "start": "RUNNING", "cancel_request": "CANCEL_PENDING", "cancel_ack": "CANCEL_ACKED"}[op]
            if op == "cancel_request":
                g["cancel_requested"] = True
    return {"goals": [goals[k] for k in sorted(goals)], "ignored": ignored}


SOLVERS = dict(zip((f"H{i}" for i in range(11, 21)), (_wire, _parameter, _epochs, _tf, _bridge, _abi, _executor, _qos, _transaction, _actions), strict=False))


def reference_result(instance: Instance) -> dict[str, Any]:
    """Author candidate expectation; independent review required, never public."""
    try:
        data = parse(instance.staged()["input.json"])
        _input_domain(data)
        keys(data, "version family data")
        need(data["version"] == VERSION and data["family"] == instance.family and type(data["data"]) is dict)
        values = SOLVERS[instance.family](data["data"])
        need(finite(values))
        return {"status": "OK", "code": "OK", "values": values}
    except Invalid as exc:
        return {"status": "INVALID", "code": str(exc), "values": None}


def equal(actual: Any, expected: Any, approximate: bool = False) -> bool:
    # Integer expected values require integer tokens even on approximate paths.
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return number(actual) and (math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL) if approximate else actual == expected)
    if type(expected) is list:
        return type(actual) is list and len(actual) == len(expected) and all(equal(a, b, approximate) for a, b in zip(actual, expected, strict=False))
    if type(expected) is dict:
        return type(actual) is dict and actual.keys() == expected.keys() and all(equal(actual[k], v, approximate) for k, v in expected.items())
    return type(actual) is type(expected) and actual == expected


def _values_equal(family: str, a: Any, b: Any) -> bool:
    if type(a) is not dict or type(b) is not dict or a.keys() != b.keys():
        return False
    for k in b:
        approx = (family == "H14" and k in ("rotation", "translation")) or (family == "H15" and k == "samples")
        if approx and type(b[k]) is list and type(a[k]) is list:
            # Public real-valued vector slots admit either numeric token type.
            if len(a[k]) != len(b[k]) or not all(number(x) and math.isclose(x, y, abs_tol=ABS_TOL, rel_tol=REL_TOL) for x, y in zip(a[k], b[k], strict=False)):
                return False
        elif not equal(a[k], b[k]):
            return False
    return True


def judge_bytes(instance: Instance, answer_bytes: bytes) -> dict[str, Any]:
    """Return a verdict, never execute anything. Judge defects are not input errors."""
    try:
        answer = parse(answer_bytes)
        keys(answer, "scope result input_sha256 reason")
        need(answer["scope"] == "FIXTURE_ONLY" and answer["input_sha256"] == sha(instance.staged()["input.json"]) and type(answer["reason"]) is str and bool(answer["reason"].strip()))
        keys(answer["result"], "status code values")
    except Invalid as exc:
        return {"passed": False, "code": "ANSWER_ENVELOPE", "detail": str(exc)}
    expected = reference_result(instance)
    got = answer["result"]
    good = type(got["status"]) is str and type(got["code"]) is str and got["status"] == expected["status"] and got["code"] == expected["code"]
    good = good and (_values_equal(instance.family, got["values"], expected["values"]) if expected["status"] == "OK" else got["values"] is None)
    return {"passed": bool(good), "code": "ACCEPT" if good else "WRONG_RESULT", "detail": "author candidate; independent Root review required"}
