"""Authored, NOT EXECUTED in source-only admission. No external providers.

These are author regression tests, not the withheld independent Root reference.
"""
from __future__ import annotations

import json
import struct
from copy import deepcopy
from dataclasses import replace

import pytest

from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import (
    judge_bytes,
    reference_result,
)
from benchmarks.harnessbench.capability_challenges_ros_traces_v3 import (
    CONTRACTS,
    FAMILIES,
    PLANNED_INVALID,
    RESULT_VALUE_SCHEMAS,
    ROLES,
    VERSION,
    encode,
    make_instance,
    make_plan,
    sha,
)


def data(instance):
    return json.loads(instance.staged()["input.json"])["data"]


def modified(instance, p):
    return replace(instance, files=(("input.json", encode({"version": VERSION, "family": instance.family, "data": p})),))


def result(family, p):
    return reference_result(modified(make_instance(family, "subject", 7), p))


def values(instance):
    r = reference_result(instance)
    assert r["status"] == "OK", r
    return r["values"]


def answer(instance, r=None):
    return encode({"scope": "FIXTURE_ONLY", "result": reference_result(instance) if r is None else r, "input_sha256": sha(instance.staged()["input.json"]), "reason": "Symbolic fixture computation only; no live ROS evidence."})


def invalid(family, p, code):
    assert result(family, p) == {"status": "INVALID", "code": code, "values": None}


@pytest.mark.parametrize("seed", [0, 7, -9, 2**60])
def test_plan_construction(seed):
    plan = make_plan(seed)
    assert len(plan) == 30
    assert len({i.task_id for i in plan}) == 30
    assert [i.family for i in plan] == [f for f in FAMILIES for _ in ROLES]
    for i in plan:
        assert i.staged() == make_instance(i.family, i.role, seed).staged()
        assert i.manifest()["input_hashes"]["input.json"] == sha(i.staged()["input.json"])
        assert i.manifest()["scope"] == "FIXTURE_ONLY"
        assert CONTRACTS[i.family] in i.prompt()
        assert '"$schema": "https://json-schema.org/draft/2020-12/schema"' in i.prompt()
        assert set(RESULT_VALUE_SCHEMAS[i.family]["required"]) == set(RESULT_VALUE_SCHEMAS[i.family]["properties"])
        assert RESULT_VALUE_SCHEMAS[i.family]["additionalProperties"] is False


@pytest.mark.parametrize("f", FAMILIES)
def test_planned_invalid_and_deliberate_wrong_answers(f):
    for role in ROLES:
        i = make_instance(f, role, 7)
        expected = reference_result(i)
        assert judge_bytes(i, answer(i))["passed"]
        if role == "invalid":
            assert expected == {"status": "INVALID", "code": PLANNED_INVALID[f], "values": None}
            wrong = {"status": "OK", "code": "OK", "values": {}}
        else:
            assert expected["status"] == "OK"
            wrong = deepcopy(expected)
            # Keep top-level result, corrupt a required value; not just envelope.
            k = next(iter(wrong["values"]))
            wrong["values"][k] = {"deliberately_wrong": True}
        assert not judge_bytes(i, answer(i, wrong))["passed"]
        a = json.loads(answer(i))
        a["input_sha256"] = "0" * 64
        assert not judge_bytes(i, encode(a))["passed"]
        a = json.loads(answer(i))
        a["result"]["extra"] = 0
        assert not judge_bytes(i, encode(a))["passed"]
        assert not judge_bytes(i, b'{"scope":"FIXTURE_ONLY","scope":"FIXTURE_ONLY"}')["passed"]
        assert not judge_bytes(i, b'{"x":NaN}')["passed"]


@pytest.mark.parametrize("f", FAMILIES)
def test_metamorphic_relation(f):
    a = values(make_instance(f, "subject", 7))
    b = values(make_instance(f, "companion", 7))
    if f in ("H11", "H13", "H15", "H16", "H19"):
        assert b == a
    elif f == "H12":
        assert b == {**a, "resolved_path": "/m" + a["resolved_path"], "visited": ["/m" + v for v in a["visited"]]}
    elif f == "H14":
        signs = [-1, -1, 1]
        assert b["reachable"] == a["reachable"]
        assert b["path"] == ["r_" + v for v in a["path"]]
        assert b["rotation"] == [v*signs[j//3]*signs[j%3] for j, v in enumerate(a["rotation"])]
        assert b["translation"] == [v*signs[j] for j, v in enumerate(a["translation"])]
    elif f == "H17":
        assert b == {"active": ["r_" + v for v in a["active"]], "waiting": [["r_" + v for v in pair] for pair in a["waiting"]], "deadlocked": a["deadlocked"], "cycle": ["r_" + v for v in a["cycle"]]}
    elif f == "H18":
        assert b == {k: [v + 1000 for v in a[k]] if k != "alive" else a[k] for k in a}
    else:
        assert b == {"goals": [{**g, "id": "r_" + g["id"]} for g in a["goals"]], "ignored": a["ignored"]}


def test_h11_wire_boundaries_and_roundtrip():
    i = make_instance("H11", "subject", 7)
    p = data(i)
    p["payload"] = {"id": 2**32-1, "label": "µ", "groups": [{"tag": 255, "samples": [-32768, 32767]}]}
    r = result("H11", p)["values"]
    expected = struct.pack("<II", 2**32-1, 2) + b"\xc2\xb5" + struct.pack("<IBIhh", 1, 255, 2, -32768, 32767)
    assert r["encoded_hex"] == expected.hex()
    assert r["byte_count"] == len(expected)
    p.update(mode="decode", payload=expected.hex())
    assert result("H11", p)["values"] == r
    for wire, code in [(expected[:-1].hex(), "TRUNCATED"), ((expected+b"\0").hex(), "TRAILING"), ("0z", "HEX"), ("AA", "HEX"), ((struct.pack("<II", 0, 1)+b"\xff").hex(), "UTF8")]:
        invalid("H11", {**p, "payload": wire}, code)
    invalid("H11", {**p, "byteorder": "big"}, "BYTEORDER")
    p.update(mode="encode", payload={"id": 2**32, "label": "", "groups": []})
    invalid("H11", p, "WIDTH")
    p["payload"]["id"] = True
    invalid("H11", p, "INVALID_INPUT")


def test_h12_search_missing_path_remap_and_types():
    p = data(make_instance("H12", "subject", 7))
    p["parameters"] = {"/a/gain": 3, "/gain": 4}
    assert result("H12", p)["values"] == {"resolved_path": "/a/gain", "found": True, "value": 3, "visited": ["/a/b/gain", "/a/gain"]}
    p["parameters"] = {}
    assert result("H12", p)["values"] == {"resolved_path": None, "found": False, "value": None, "visited": ["/a/b/gain", "/a/gain", "/gain"]}
    invalid("H12", {**p, "namespace": "/a//b"}, "PATH")
    invalid("H12", {**p, "remaps": {"a": "b", "b": "a"}}, "REMAP_CYCLE")
    p.update(key="/gain", parameters={"/gain": True})
    invalid("H12", p, "TYPE_MISMATCH")
    p["expect"] = "boolean"
    assert result("H12", p)["values"]["visited"] == ["/gain"]


def test_h13_epoch_ties_gap_identity_and_equal_clock():
    p = data(make_instance("H13", "subject", 7))
    assert result("H13", p)["values"] == {"epochs": [0, 0, 0, 1, 1, 1], "matches": [[0, -1], [3, 5]], "rewinds": [3]}
    # Equal stamps: original index wins, even if sequence differs.
    p["records"][2]["stamp_ns"] = 10
    assert result("H13", p)["values"]["matches"][0] == [0, -1]
    p["maxgap"] = 0
    assert result("H13", p)["values"]["matches"][0] == [-1, -1]
    p["records"][2]["sequence"] = 1
    invalid("H13", p, "IDENTITY")
    p["records"][2]["sequence"] = 2
    p["records"][0]["stamp_ns"] = 10.0
    invalid("H13", p, "TIME")


def test_h14_inverse_disconnected_cycle_rotation():
    p = data(make_instance("H14", "subject", 7))
    p["source"], p["target"] = "a", "b"
    v = result("H14", p)["values"]
    assert v == {"reachable": True, "rotation": [0, 1, 0, -1, 0, 0, 0, 0, 1], "translation": [-1, 0, -3], "path": ["a", "b"]}
    assert result("H14", {**p, "target": "unknown"})["values"] == {"reachable": False, "rotation": None, "translation": None, "path": []}
    q = deepcopy(p)
    q["edges"][0]["rotation"] = [0]*9
    invalid("H14", q, "BAD_ROTATION")
    q = deepcopy(p)
    q["edges"][0]["parent"] = "b"
    invalid("H14", q, "CYCLE")
    p.update(source="new", target="new")
    assert result("H14", p)["values"]["path"] == ["new"]


def test_h15_time_normalization_loss_bounds_and_roundtrip():
    p = data(make_instance("H15", "subject", 7))
    v = result("H15", p)["values"]
    assert v["duration"] == {"sec": -1, "nanosec": 999999999}
    s = p["header"]["stamp"]["sec"]
    assert v["header"]["stamp"] == {"sec": s+1, "nanosec": 1}
    # Reconstruct the lossless ROS1 subset from output normalized fields.
    q = deepcopy(p)
    q["header"]["stamp"] = {"sec": v["header"]["stamp"]["sec"], "nsec": 1}
    q["duration"] = {"sec": -1, "nsec": 999999999}
    assert result("H15", q)["values"] == v
    invalid("H15", {**p, "bound": 2}, "ARRAY_BOUND")
    q["header"]["stamp"] = {"sec": 0, "nsec": -1}
    invalid("H15", q, "NEGATIVE_STAMP")
    q = deepcopy(p)
    q["header"]["seq"] = 1
    q["allow_seq_drop"] = True
    assert result("H15", q)["values"]["dropped_fields"] == ["header.seq"]
    q["header"]["stamp"] = {"sec": 2**31, "nsec": 0}
    invalid("H15", q, "OVERFLOW")


def test_h16_admission_not_loading():
    p = data(make_instance("H16", "subject", 7))
    assert result("H16", p)["values"] == {"compatible": False, "missing_symbols": ["destroy"], "mismatches": ["abi"]}
    p["offered"] = deepcopy(p["required"])
    assert result("H16", p)["values"]["compatible"] is True
    for field in ("architecture", "abi", "type_name", "type_hash"):
        p["offered"][field] += "!"
    assert result("H16", p)["values"]["mismatches"] == ["architecture", "abi", "type_name", "type_hash"]
    p["offered"]["symbols"] = [""]
    invalid("H16", p, "MANIFEST")


def test_h17_wait_cycle_ownership_finish_and_canonical_cycle():
    p = data(make_instance("H17", "subject", 7))
    assert result("H17", p)["values"] == {"active": ["a", "b"], "waiting": [["a", "b"], ["b", "a"]], "deadlocked": True, "cycle": ["a", "b", "a"]}
    p["events"].append({"op": "start", "callback": "c"})
    assert result("H17", p)["values"]["cycle"] == ["a", "b", "a"]
    p["events"].append({"op": "finish", "callback": "b"})
    assert result("H17", p)["values"]["waiting"] == []
    p["callbacks"]["d"] = "solo"
    p["events"].append({"op": "start", "callback": "d"})
    invalid("H17", p, "EVENT")
    p = {"groups": {"g": "reentrant"}, "callbacks": dict.fromkeys("abc", "g"), "events": [{"op": "start", "callback": c} for c in "abc"] + [{"op": "wait", "callback": a, "target": b} for a, b in [("a", "c"), ("c", "a"), ("a", "b"), ("b", "c")]]}
    # Lexicographic, not shortest: [a,b,c,a] precedes [a,c,a].
    assert result("H17", p)["values"]["cycle"] == ["a", "b", "c", "a"]


def test_h18_inclusive_expiration_recovery_and_no_publication():
    p = {"publications": [0, 5], "horizon": 10, "deadline": 5, "lease": 5}
    assert result("H18", p)["values"] == {"deadline_misses": [10], "liveliness_lost": [10], "recovered": [], "alive": False}
    p["publications"] = [0, 6, 6]
    p["horizon"] = 16
    assert result("H18", p)["values"] == {"deadline_misses": [5, 11, 16], "liveliness_lost": [5, 11], "recovered": [6], "alive": False}
    p["publications"] = []
    assert result("H18", p)["values"] == {"deadline_misses": [], "liveliness_lost": [], "recovered": [], "alive": False}
    invalid("H18", {**p, "lease": 0}, "TIME")


def test_h19_rollback_stale_types_and_declaration_order():
    p = data(make_instance("H19", "subject", 7))
    p["updates"] = {"lo": 9999, "hi": -1}
    v = result("H19", p)["values"]
    assert v == {"committed": False, "revision": 2, "parameters": p["parameters"], "failed_constraints": ["ordered"], "rejection": "CONSTRAINT"}
    p["updates"]["lo"] = True
    invalid("H19", p, "PREDICATE")
    p["expected_revision"] = 1
    v = result("H19", p)["values"]
    assert v["rejection"] == "STALE_REVISION" and v["failed_constraints"] == []
    assert v["parameters"] == p["parameters"]
    p["constraints"][0]["op"] = "eval"
    invalid("H19", p, "PREDICATE")
    p = data(make_instance("H19", "subject", 7))
    p["updates"] = {}
    assert result("H19", p)["values"]["revision"] == 3


def test_h20_first_terminal_cancel_not_terminal_idempotence():
    p = data(make_instance("H20", "subject", 7))
    v = result("H20", p)["values"]
    assert v == {"goals": [{"id": "g", "state": "SUCCEEDED", "cancel_requested": True, "terminal_event": 5}], "ignored": [6]}
    p["events"] = p["events"][:5]
    assert result("H20", p)["values"]["goals"][0]["state"] == "CANCEL_ACKED"
    p["events"].append({"op": "cancel_ack", "id": "g"})
    assert result("H20", p)["values"]["ignored"] == [5]
    p["events"].append({"op": "result_cancelled", "id": "g"})
    assert result("H20", p)["values"]["goals"][0]["terminal_event"] == 6
    p["events"].append({"op": "result_success", "id": "foreign"})
    invalid("H20", p, "EVENT")
    p["events"] = [{"op": "submit", "id": "g"}, {"op": "accept", "id": "g"}, {"op": "result_cancelled", "id": "g"}]
    invalid("H20", p, "EVENT")


@pytest.mark.parametrize("f", FAMILIES)
def test_strict_output_integer_and_boolean_tokens(f):
    i = make_instance(f, "subject", 7)
    a = json.loads(answer(i))
    a["reason"] = "   "
    assert not judge_bytes(i, encode(a))["passed"]
    a = json.loads(answer(i))
    a["scope"] = "REAL"
    assert not judge_bytes(i, encode(a))["passed"]
    # Reject float-for-int and number-for-bool at any result nesting depth.
    r = reference_result(i)
    def corrupt(x):
        if type(x) is dict:
            for k, v in x.items():
                if type(v) is bool:
                    x[k] = int(v)
                    return True
                if type(v) is int:
                    x[k] = float(v)
                    return True
                if corrupt(v):
                    return True
        elif type(x) is list:
            for j, v in enumerate(x):
                if type(v) is bool:
                    x[j] = int(v)
                    return True
                if type(v) is int:
                    x[j] = float(v)
                    return True
                if corrupt(v):
                    return True
        return False
    assert corrupt(r["values"])
    assert not judge_bytes(i, answer(i, r))["passed"]


# v3.1 resource regressions appended below the unchanged original 14 functions.
# No execution in the source-author session. Largest byte fixture is 1 MiB + 1.
def raw_instance(raw, family="H18"):
    return replace(make_instance(family, "subject", 7), files=(("input.json", raw),))


def parse_code(raw, code):
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import Invalid, parse
    with pytest.raises(Invalid) as caught:
        parse(raw)
    assert str(caught.value) == code


def test_resource_public_pins_and_prompt_schemas():
    from benchmarks.harnessbench import capability_challenges_ros_traces_v3 as public
    assert VERSION == "rosclaw.capability_challenges.ros_traces.v3.1"
    assert (public.MAX_BYTES, public.JSON_DEPTH, public.JSON_TOKENS) == (1048576, 32, 20000)
    assert (public.INTEGER_DIGITS, public.NUMBER_CHARS, public.EXPONENT_ABS) == (20, 64, 100)
    assert (public.INPUT_NODES, public.INPUT_CONTAINER) == (2048, 128)
    assert (public.INPUT_STRING_BYTES, public.INPUT_TOTAL_STRING_BYTES) == (256, 16384)
    assert (public.H17_GROUPS, public.H17_CALLBACKS, public.H17_EVENTS, public.H17_EDGES) == (8, 8, 128, 16)
    assert (public.H18_PUBLICATIONS, public.H18_TIME, public.H18_OUTPUT_EVENTS) == (128, 10**12, 4096)
    for f in FAMILIES:
        prompt = make_instance(f, "subject", 7).prompt()
        assert "RESOURCE_LIMIT" in prompt
        assert "Admission order" in prompt
        output_schema = json.loads(prompt.split("Answer Draft202012 schema:\n")[1])
        codes = output_schema["properties"]["result"]["oneOf"][1]["properties"]["code"]["enum"]
        assert "RESOURCE_LIMIT" in codes
    assert public.INPUT_DATA_SCHEMAS["H17"]["properties"]["callbacks"]["maxProperties"] == 8
    assert public.INPUT_DATA_SCHEMAS["H18"]["properties"]["horizon"]["maximum"] == 10**12
    assert RESULT_VALUE_SCHEMAS["H17"]["properties"]["waiting"]["maxItems"] == 16
    assert RESULT_VALUE_SCHEMAS["H18"]["properties"]["deadline_misses"]["maxItems"] == 4096


def test_raw_byte_cap_exact_and_one_over_for_input_and_answer():
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import MAX_BYTES, parse
    base = make_instance("H18", "subject", 7)
    raw = base.staged()["input.json"]
    padded = raw + b" " * (MAX_BYTES - len(raw))
    assert len(padded) == MAX_BYTES
    exact = raw_instance(padded)
    assert reference_result(exact) == reference_result(base)
    assert judge_bytes(exact, answer(exact))["passed"]
    over = raw_instance(padded + b" ")
    assert reference_result(over) == {"status": "INVALID", "code": "SIZE_LIMIT", "values": None}
    assert judge_bytes(over, answer(over))["passed"]
    assert parse(b"0" + b" " * (MAX_BYTES-1)) == 0
    parse_code(b"0" + b" " * MAX_BYTES, "SIZE_LIMIT")
    a = json.loads(answer(base))
    a["reason"] += "x" * (MAX_BYTES - len(encode(a)))
    assert len(encode(a)) == MAX_BYTES
    assert judge_bytes(base, encode(a))["passed"]
    verdict = judge_bytes(base, encode(a) + b" ")
    assert verdict["code"] == "ANSWER_ENVELOPE" and verdict["detail"] == "SIZE_LIMIT"


def test_json_depth_cap_and_escaped_delimiters():
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import parse
    assert parse(b"[" * 32 + b"0" + b"]" * 32) is not None
    parse_code(b"[" * 33 + b"0" + b"]" * 33, "RESOURCE_LIMIT")
    parse_code(b"[" * 5000 + b"0" + b"]" * 5000, "RESOURCE_LIMIT")
    s = '[{\\"' * 40
    assert parse(encode(s)) == s  # Quoted brackets/escaped quotes do not count depth.
    parse_code(b"[}", "INVALID_JSON")
    parse_code(b'"unterminated', "INVALID_JSON")
    parse_code(b'"bad\\q"', "INVALID_JSON")


def test_json_number_caps_before_conversion():
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import parse
    assert parse(b"9" * 20) == 10**20 - 1
    assert parse(b"-" + b"9" * 20) == -(10**20 - 1)
    parse_code(b"9" * 21, "RESOURCE_LIMIT")
    parse_code(b"9" * 5000, "RESOURCE_LIMIT")
    exact_float = b"1." + b"0" * 62
    assert len(exact_float) == 64 and parse(exact_float) == 1.0
    parse_code(exact_float + b"0", "RESOURCE_LIMIT")
    for token in (b"1e100", b"1e-100", b"-1e+100"):
        assert type(parse(token)) is float
    for token in (b"1e101", b"1e-101", b"9e100", b"1e9999", b"1e" + b"9" * 1000):
        parse_code(token, "RESOURCE_LIMIT")
    for token in (b"01", b"1e+", b"-.1", b"NaN", b"Infinity", b"-Infinity"):
        parse_code(token, "INVALID_JSON")
    assert parse(b"true") is True
    assert parse(b"1.0") == 1.0 and type(parse(b"1.0")) is float


def test_json_token_budget_exact_and_one_over():
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import parse
    # One array token + 19999 numeric tokens, only 39999 bytes.
    exact = b"[" + b",".join([b"0"] * 19999) + b"]"
    assert len(parse(exact)) == 19999
    parse_code(exact[:-1] + b",0]", "RESOURCE_LIMIT")
    # Object keys count as tokens, while punctuation inside strings does not.
    assert parse(b'{"[{}]":0}') == {"[{}]": 0}


def test_decoded_input_domain_caps_and_precedence():
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import (
        Invalid,
        _input_domain,
    )
    node_exact = [[0] * 128 for _ in range(15)] + [[0] * 111]
    _input_domain(node_exact)  # 1 root + 16 lists + 2031 integers = 2048.
    node_over = deepcopy(node_exact)
    node_over[-1].append(0)
    for good, bad in [
        (node_exact, node_over),
        ([0] * 128, [0] * 129),
        ("a" * 256, "a" * 257),
        (["a" * 256] * 64, ["a" * 256] * 64 + ["a"]),
        ("µ" * 128, "µ" * 129),
        ({"a" * 256: 0}, {"a" * 257: 0}),
    ]:
        _input_domain(good)
        with pytest.raises(Invalid) as caught:
            _input_domain(bad)
        assert str(caught.value) == "RESOURCE_LIMIT"
        # Domain checks precede the deliberately wrong envelope shape.
        assert reference_result(raw_instance(encode(good)))["code"] == "INVALID_INPUT"
        assert reference_result(raw_instance(encode(bad)))["code"] == "RESOURCE_LIMIT"
    p = data(make_instance("H12", "subject", 7))
    p.update(parameters={"/gain": "a" * 256}, key="/gain", expect="string")
    assert result("H12", p)["values"]["value"] == "a" * 256
    p["parameters"]["/gain"] += "a"
    invalid("H12", p, "RESOURCE_LIMIT")
    # Resource admission before a later shape error, but byte/UTF8/JSON first.
    parse_code(b"\xff", "INVALID_JSON")
    parse_code(b'{"a":0,"a":1}', "INVALID_JSON")
    parse_code(b"[" * 33 + b"broken", "RESOURCE_LIMIT")


def bounded_wait_fixture():
    names = list("abcdefgh")
    pairs = [(a, b) for a in names for b in names if a < b]
    p = {"groups": {"g": "reentrant"}, "callbacks": dict.fromkeys(names, "g"),
         "events": [{"op": "start", "callback": c} for c in names]}
    return p, pairs


def wait_event(pair, op="wait"):
    return {"op": op, "callback": pair[0], "target": pair[1]}


def test_h17_callback_group_event_caps_and_cycle_depth():
    p, _ = bounded_wait_fixture()
    assert len(result("H17", p)["values"]["active"]) == 8
    invalid("H17", {**p, "callbacks": {**p["callbacks"], "i": "g"}}, "RESOURCE_LIMIT")
    p["groups"] = {"g": "reentrant", **{f"g{i}": "reentrant" for i in range(7)}}
    assert result("H17", p)["status"] == "OK"
    invalid("H17", {**p, "groups": {**p["groups"], "ninth": "reentrant"}}, "RESOURCE_LIMIT")
    p["events"] += [wait_event(("a", "b")) for _ in range(120)]
    assert len(p["events"]) == 128
    assert result("H17", p)["values"]["waiting"] == [["a", "b"]]
    invalid("H17", {**p, "events": p["events"] + [wait_event(("a", "b"))]}, "RESOURCE_LIMIT")
    p, _ = bounded_wait_fixture()
    p["events"] += [wait_event((a, b)) for a, b in zip("abcdefgh", "bcdefgha", strict=False)]
    assert result("H17", p)["values"]["cycle"] == list("abcdefgha")
    # A compact 20-vertex complete graph is excluded before cycle traversal.
    names = [f"c{i}" for i in range(20)]
    invalid("H17", {"groups": {"g": "reentrant"}, "callbacks": dict.fromkeys(names, "g"), "events": []}, "RESOURCE_LIMIT")


def test_h17_live_edge_cap_duplicates_release_and_event_precedence():
    p, pairs = bounded_wait_fixture()
    p["events"] += [wait_event(pair) for pair in pairs[:16]]
    v = result("H17", p)["values"]
    assert len(v["waiting"]) == 16 and not v["deadlocked"]
    p["events"].append(wait_event(pairs[0]))  # Duplicate at capacity is legal.
    assert result("H17", p)["values"] == v
    invalid("H17", {**p, "events": p["events"] + [wait_event(pairs[16])]}, "RESOURCE_LIMIT")
    # Capacity applies at every prefix, not just the final graph.
    invalid("H17", {**p, "events": p["events"] + [wait_event(pairs[16]), {"op": "finish", "callback": "a"}]}, "RESOURCE_LIMIT")
    released = p["events"] + [wait_event(pairs[0], "unwait"), wait_event(pairs[16])]
    assert len(result("H17", {**p, "events": released})["values"]["waiting"]) == 16
    for e in ({"op": "wait", "callback": "a", "target": []},
              {"op": "wait", "callback": "a", "target": "a"},
              {"op": "wait", "callback": "a", "target": "b", "extra": 0}):
        invalid("H17", {**p, "events": p["events"] + [e]}, "EVENT")
    invalid("H17", {**p, "callbacks": []}, "INVALID_INPUT")
    invalid("H17", {**p, "callbacks": {"a": []}}, "DECLARATION")


def test_h18_aggregate_output_exact_and_over_without_expansion():
    p = {"publications": [0], "horizon": 4096, "deadline": 1, "lease": 10**12}
    v = result("H18", p)["values"]
    assert v["deadline_misses"] == list(range(1, 4097))
    assert v["liveliness_lost"] == [] and v["alive"]
    i = modified(make_instance("H18", "subject", 7), p)
    assert len(answer(i)) < 1048576 and judge_bytes(i, answer(i))["passed"]
    invalid("H18", {**p, "horizon": 4097}, "RESOURCE_LIMIT")
    # Losses count too, so reducing misses by one admits exactly one final loss.
    p.update(horizon=4095, lease=1)
    v = result("H18", p)["values"]
    assert sum(len(v[k]) for k in ("deadline_misses", "liveliness_lost", "recovered")) == 4096
    invalid("H18", {**p, "horizon": 4096}, "RESOURCE_LIMIT")
    # Nonfinal losses AND recoveries both count in the combined budget.
    p.update(publications=[0, 4095])
    v = result("H18", p)["values"]
    assert v["liveliness_lost"] == [1] and v["recovered"] == [4095]
    assert len(v["deadline_misses"]) == 4094
    invalid("H18", {**p, "publications": [0, 4096], "horizon": 4096}, "RESOURCE_LIMIT")
    # Tiny input, huge implied output: bounded rejection, no million-entry list.
    invalid("H18", {"publications": [0], "horizon": 10**12, "deadline": 1, "lease": 10**12}, "RESOURCE_LIMIT")


def test_h18_publication_time_caps_and_shape_order():
    p = {"publications": [0] * 128, "horizon": 0, "deadline": 1, "lease": 1}
    assert result("H18", p)["values"] == {"deadline_misses": [], "liveliness_lost": [], "recovered": [], "alive": True}
    invalid("H18", {**p, "publications": [0] * 129}, "RESOURCE_LIMIT")
    p = {"publications": [10**12], "horizon": 10**12, "deadline": 10**12, "lease": 10**12}
    assert result("H18", p)["status"] == "OK"
    for field in ("horizon", "deadline", "lease"):
        invalid("H18", {**p, field: 10**12 + 1}, "RESOURCE_LIMIT")
    invalid("H18", {**p, "publications": [10**12 + 1]}, "RESOURCE_LIMIT")
    for bad in (True, 1.0, "1", None, [], -1):
        invalid("H18", {**p, "horizon": bad}, "TIME")
    invalid("H18", {**p, "publications": {}}, "INVALID_INPUT")
    invalid("H18", {**p, "deadline": 0, "horizon": 10**12 + 1}, "TIME")
    invalid("H18", {**p, "publications": [1, 0], "horizon": 10**12 + 1}, "RESOURCE_LIMIT")
    invalid("H18", {**p, "publications": [1, 0]}, "ORDER")


def test_resource_errors_are_public_results_not_success_or_silent_coercion():
    raw = b'{"version":"' + VERSION.encode() + b'","family":"H18","data":{"publications":[0],"horizon":1e9999,"deadline":1,"lease":1}}'
    i = raw_instance(raw)
    expected = {"status": "INVALID", "code": "RESOURCE_LIMIT", "values": None}
    assert reference_result(i) == expected
    assert judge_bytes(i, answer(i, expected))["passed"]
    assert not judge_bytes(i, answer(i, {**expected, "code": "INVALID_INPUT"}))["passed"]
    base = make_instance("H18", "subject", 7)
    for raw, detail in [(b"[" * 33 + b"0" + b"]" * 33, "RESOURCE_LIMIT"),
                        (b"1e9999", "RESOURCE_LIMIT"), (b"NaN", "INVALID_JSON"),
                        (b"9" * 5000, "RESOURCE_LIMIT"), (b"\xff", "INVALID_JSON")]:
        verdict = judge_bytes(base, raw)
        assert not verdict["passed"] and verdict["code"] == "ANSWER_ENVELOPE"
        assert verdict["detail"] == detail
    old = json.loads(base.staged()["input.json"])
    old["version"] = "rosclaw.capability_challenges.ros_traces.v3.0"
    assert reference_result(raw_instance(encode(old)))["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("seed", [0, 7, -9, 2**60])
def test_h17_h18_small_fixture_data_preserved_for_all_roles(seed):
    # Explicit pre-repair symbolic data, not private expected results.
    k = int.from_bytes(bytes.fromhex(sha(str(seed).encode()))[:4], "little") % 100 + 1
    h17 = {"groups": {"g": "reentrant", "solo": "mutuallyexclusive"},
           "callbacks": {"a": "g", "b": "g", "c": "solo"}, "events": [
               {"op": "start", "callback": "a"}, {"op": "start", "callback": "b"},
               {"op": "wait", "callback": "a", "target": "b"},
               {"op": "wait", "callback": "b", "target": "a"}]}
    h18 = {"publications": [k, k+5, k+16], "horizon": k+30, "deadline": 5, "lease": 7}
    for role in ROLES:
        a, b = deepcopy(h17), deepcopy(h18)
        if role == "invalid":
            a["events"].insert(0, {"op": "finish", "callback": "a"})
            b["publications"] = [k+1, k]
        elif role == "companion":
            a["callbacks"] = {"r_" + c: g for c, g in a["callbacks"].items()}
            for e in a["events"]:
                e["callback"] = "r_" + e["callback"]
                if "target" in e:
                    e["target"] = "r_" + e["target"]
            b["publications"] = [t+1000 for t in b["publications"]]
            b["horizon"] += 1000
        assert data(make_instance("H17", role, seed)) == a
        assert data(make_instance("H18", role, seed)) == b


def test_numeric_input_headroom_and_representable_outputs():
    from benchmarks.harnessbench.capability_challenges_ros_traces_v3 import (
        INPUT_FLOAT_ABS,
        INPUT_INTEGER_ABS,
    )
    assert INPUT_INTEGER_ABS == 10**15 and INPUT_FLOAT_ABS == 1e50
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import (
        Invalid,
        _input_domain,
    )
    for good, bad in [(10**15, 10**15+1), (-10**15, -10**15-1),
                      (1e50, 1.1e50), (-1e50, -1.1e50)]:
        _input_domain(good)
        with pytest.raises(Invalid) as caught:
            _input_domain(bad)
        assert str(caught.value) == "RESOURCE_LIMIT"
        assert reference_result(raw_instance(encode(bad)))["code"] == "RESOURCE_LIMIT"
    _input_domain(True)  # Bool remains bool, never an integer conversion.
    p = {"parameters": {}, "updates": {}, "expected_revision": 10**15,
         "revision": 10**15, "constraints": []}
    i = modified(make_instance("H19", "subject", 7), p)
    assert reference_result(i)["values"]["revision"] == 10**15+1
    assert judge_bytes(i, answer(i))["passed"]
    p = data(make_instance("H14", "subject", 7))
    for e in p["edges"]:
        e["rotation"] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        e["translation"] = [1e50, 1e50, 1e50]
    i = modified(make_instance("H14", "subject", 7), p)
    assert reference_result(i)["values"]["translation"] == [2e50, 2e50, 2e50]
    assert judge_bytes(i, answer(i))["passed"]


def test_supported_large_outputs_fit_shared_answer_admission():
    # Bound the H13 cross-product output, not merely the smaller staged fixtures.
    p = {"records": [{"topic": "anchor", "stamp_ns": i, "sequence": i} for i in range(128)],
         "clock": [0] * 128, "anchor_topic": "anchor",
         "streams": [f"stream{i}" for i in range(128)], "maxgap": 0}
    i = modified(make_instance("H13", "subject", 7), p)
    v = reference_result(i)["values"]
    assert len(v["matches"]) == 128 and all(row == [-1] * 128 for row in v["matches"])
    assert len(answer(i)) < 1048576 and judge_bytes(i, answer(i))["passed"]
    # Maximum event cardinality and maximum decimal event width together.
    p = {"publications": [10**12-4096], "horizon": 10**12, "deadline": 1, "lease": 10**12}
    i = modified(make_instance("H18", "subject", 7), p)
    v = reference_result(i)["values"]
    assert len(v["deadline_misses"]) == 4096 and v["deadline_misses"][-1] == 10**12
    assert len(encode([v[k] for k in ("deadline_misses", "liveliness_lost", "recovered")])) <= 57356
    assert len(answer(i)) < 1048576 and judge_bytes(i, answer(i))["passed"]
