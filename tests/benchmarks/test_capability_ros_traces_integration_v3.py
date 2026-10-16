"""Unexecuted SOURCE candidate: pure qualification, never a model/SDK test.

Host tests use the genuine accepted judge, not a mock or recorded Root answers.
Old-module reverse checks require the independently preserved SOURCE baselines
at ROS_TRACE_BASELINE_DIR (default history_inputs/source_seam_baseline). Missing
baselines fail qualification, never skip. They contain source only, not answers.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.harnessbench.capability_challenges_ros_traces_v3 import (
    FAMILIES,
    ROLES,
    VERSION,
    Instance,
    encode,
    make_instance,
    make_plan,
)
from benchmarks.harnessbench.capability_ros_traces_host_judge_v3 import (
    judge_ros_trace_workspace,
)
from benchmarks.harnessbench.capability_ros_traces_integration_v3 import (
    DEFAULT_SEED,
    REGISTRY_VERSION,
    ROS_TRACE_TASKS,
    make_ros_trace_tasks,
    registry_id,
    staged_text,
    task_from_instance,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = "benchmarks.harnessbench.capability_ros_traces_integration_v3"
HOST = "benchmarks.harnessbench.capability_ros_traces_host_judge_v3"
SUITE = ROOT / "benchmarks/harnessbench"
TASK_APPEND = (
    '\nfrom benchmarks.harnessbench.capability_ros_traces_integration_v3 import (  # noqa: E402\n'
    '    ROS_TRACE_TASKS,  # noqa: E402 - Intentional EOF registry seam preserves existing TASKS initialization.\n'
    ')\n'
    '\nassert not (TASKS.keys() & ROS_TRACE_TASKS.keys()), "H11_V31_TASK_ID_COLLISION"\n'
    'TASKS.update(ROS_TRACE_TASKS)\n'
)
ORACLE_INSERT = (
    '    if kind == "offline_ros_traces_v3":\n'
    '        from benchmarks.harnessbench.capability_ros_traces_host_judge_v3 import (\n'
    '            judge_ros_trace_workspace,\n'
    '        )\n'
    '\n        return judge_ros_trace_workspace(task, workspace)\n'
)


def dump(node):
    return ast.dump(node, include_attributes=False)


def stage(task, workspace):
    """Same text-write interface as frozen runner, without importing runner."""
    for name, text in task.staged_files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (workspace / "task.md").write_text(task.prompt, encoding="utf-8")


def answer(raw, values=None, *, code="OK"):
    return {
        "scope": "FIXTURE_ONLY",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "reason": "independently authored small fixture calculation",
        "result": {"status": "OK" if code == "OK" else "INVALID",
                   "code": code, "values": values},
    }


def genuine(instance, envelope):
    # Host-only test import. No reference/expected-answer API is used.
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import judge_bytes
    return judge_bytes(instance, encode(envelope))


def synthetic(family, data):
    raw = encode({"version": VERSION, "family": family, "data": data})
    return Instance(f"{family}-subject-7", family, "subject", 7, (("input.json", raw),))


def test_all_30_discoverable_identities():
    assert DEFAULT_SEED == 7
    assert REGISTRY_VERSION == "rosclaw.capability_challenges.ros_traces.v3.1"
    assert len(ROS_TRACE_TASKS) == 30
    assert set(ROS_TRACE_TASKS) == {
        f"{family}-{role}-7".upper() for family in FAMILIES for role in ROLES
    }
    for instance in make_plan(7):
        task = ROS_TRACE_TASKS[instance.task_id.upper()]
        assert task.task_id == task.task_id.upper() == registry_id(instance.task_id)
        assert task.oracle == {
            "kind": "offline_ros_traces_v3", "version": VERSION,
            "scope": "FIXTURE_ONLY", "canonical_task_id": instance.task_id,
            "registry_task_id": instance.task_id.upper(), "family": instance.family,
            "role": instance.role, "seed": 7,
            "input_sha256": hashlib.sha256(instance.staged()["input.json"]).hexdigest(),
        }
        assert task.prompt == instance.prompt()
        assert "Answer Draft202012 schema:" in task.prompt
        assert "Supported input Draft202012 schema" in task.prompt
        assert set(task.staged_files) == {"input.json"}


@pytest.mark.parametrize("seed", [0, -1, 8, 10**30])
def test_exact_seed_factory(seed):
    tasks = make_ros_trace_tasks(seed)
    assert len(tasks) == 30
    for instance in make_plan(seed):
        task = tasks[instance.task_id.upper()]
        assert task.oracle["seed"] == seed
        assert task.oracle["canonical_task_id"] == instance.task_id
        assert task.staged_files["input.json"].encode() == instance.staged()["input.json"]


@pytest.mark.parametrize("seed", [True, False, 7.0, "7", None])
def test_bad_seed_refused(seed):
    with pytest.raises(ValueError, match="EXACT_INTEGER_SEED"):
        make_ros_trace_tasks(seed)


@pytest.mark.parametrize("keys", [
    ["H11-SUBJECT-7"], ["h11-subject-7"], ["unused", "UNUSED"],
])
def test_alias_collision_not_overwritten(keys):
    with pytest.raises(ValueError, match="TASK_ID_COLLISION"):
        make_ros_trace_tasks(7, occupied=keys)
    assert len(make_ros_trace_tasks(7, occupied=["U01", "H11"])) == 30


def test_collision_assert_is_effective():
    tree = ast.parse((SUITE / "tasks.py").read_bytes())
    assertion = tree.body[-2]
    assert dump(assertion) == dump(ast.parse(
        'assert not (TASKS.keys() & ROS_TRACE_TASKS.keys()), "H11_V31_TASK_ID_COLLISION"'
    ).body[0])
    # Execute the actual isolated assertion, not a substitute collision helper.
    module = ast.fix_missing_locations(ast.Module(body=[assertion], type_ignores=[]))
    with pytest.raises(AssertionError, match="H11_V31_TASK_ID_COLLISION"):
        exec(compile(module, "<actual collision assertion>", "exec"),
             {"TASKS": {"H11-SUBJECT-7": object()}, "ROS_TRACE_TASKS": ROS_TRACE_TASKS})


@pytest.mark.parametrize("instance", make_plan(7), ids=lambda i: i.task_id)
def test_actual_disk_bytes_digest_and_prompt(instance, tmp_path):
    task = task_from_instance(instance)
    stage(task, tmp_path)
    raw = (tmp_path / "input.json").read_bytes()
    assert raw == instance.staged()["input.json"]
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == task.oracle["input_sha256"]
    assert (tmp_path / "task.md").read_text(encoding="utf-8") == instance.prompt()
    assert {p.name for p in tmp_path.iterdir()} == {"input.json", "task.md"}
    if instance.family == "H11" and instance.role != "invalid":
        assert "µ机器人".encode() in raw
    if instance.role == "companion" and instance.family in ("H15", "H19"):
        data = json.loads(raw)["data"]
        if instance.family == "H15":
            assert list(data["header"]) == ["frame_id", "stamp", "seq"]
        else:
            assert list(data["updates"]) == ["hi", "lo"]
        # Canonical re-encoding is forbidden here, even if JSON values compare equal.
        assert encode(json.loads(raw)) != raw


def test_utf8_and_leaf_boundaries():
    original = make_instance("H11", "subject", 7)
    with pytest.raises(UnicodeDecodeError):
        staged_text(replace(original, files=(("input.json", b"\xff\n"),)))
    for files in [(), (("secret.py", b"x"),),
                  (("input.json", b"a"), ("input.json", b"b")),
                  (("input.json", b"a"), ("answer.json", b"{}"))]:
        with pytest.raises(ValueError, match="PUBLIC_STAGE_ONLY_INPUT"):
            staged_text(replace(original, files=files))
    with pytest.raises(ValueError, match="INSTANCE_IDENTITY"):
        task_from_instance(replace(original, task_id="H11-SUBJECT-7"))


def test_public_import_graph_is_public_only():
    allowed = {
        PUBLIC: {"__future__", "hashlib", "collections.abc",
                 "benchmarks.harnessbench.capability_challenges_ros_traces_v3",
                 "benchmarks.harnessbench.task_common"},
        "benchmarks.harnessbench.capability_challenges_ros_traces_v3": {
            "__future__", "copy", "dataclasses", "hashlib", "json", "typing"},
        "benchmarks.harnessbench.task_common": {"__future__", "dataclasses"},
    }
    for name, imports in allowed.items():
        path = ROOT / (name.replace(".", "/") + ".py")
        tree = ast.parse(path.read_bytes())
        actual = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0
                actual.add(node.module)
            elif isinstance(node, ast.Import):
                actual.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__", "open"}
        assert actual <= imports
    assert not any("judge" in module for module in allowed)


def test_standalone_adapter_imports_add_no_private_or_sdk_modules():
    before = set(sys.modules)
    for stem in ("capability_ros_traces_integration_v3", "capability_ros_traces_host_judge_v3"):
        spec = importlib.util.spec_from_file_location("_pure_probe_" + stem, SUITE / (stem + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    additions = set(sys.modules) - before
    assert not any(any(marker in name.lower() for marker in (
        "judge", "mujoco", "torch", "rclpy", "rosclaw.sim", "runner", "oracle",
        "openai", "anthropic", "serial", "pybullet",
    )) for name in additions)


def h20_values(goal="g"):
    # First terminal wins at event 5; recognized later cancelled event ignored.
    return {"goals": [{"id": goal, "state": "SUCCEEDED", "cancel_requested": True,
                       "terminal_event": 5}], "ignored": [6]}


def test_genuine_host_accept_and_nontrivial_mutation(tmp_path):
    task = ROS_TRACE_TASKS["H20-SUBJECT-7"]
    stage(task, tmp_path)
    envelope = answer((tmp_path / "input.json").read_bytes(), h20_values())
    (tmp_path / "answer.json").write_bytes(encode(envelope))
    verdict = judge_ros_trace_workspace(task, tmp_path)
    assert verdict["passed"] and verdict["code"] == "ACCEPT"
    assert verdict["task_success"] and verdict["verified_success"]
    assert verdict["scope"] == verdict["evidence_class"] == "FIXTURE_ONLY"
    assert verdict["physical_verified"] is False
    for field, wrong in [("state", "CANCELLED"), ("terminal_event", 6),
                         ("cancel_requested", False)]:
        mutated = deepcopy(envelope)
        mutated["result"]["values"]["goals"][0][field] = wrong
        (tmp_path / "answer.json").write_bytes(encode(mutated))
        verdict = judge_ros_trace_workspace(task, tmp_path)
        assert not verdict["passed"] and verdict["code"] == "WRONG_RESULT"


def test_actual_oracle_branch_dispatch_without_legacy_import(tmp_path):
    tree = ast.parse((SUITE / "oracle.py").read_bytes())
    judge = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "judge")
    assignment_index = next(i for i, n in enumerate(judge.body)
                            if isinstance(n, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == "kind" for t in n.targets))
    branch = judge.body[assignment_index + 1]
    assert dump(branch) == dump(ast.parse(ORACLE_INSERT.lstrip().replace(
        "\n        ", "\n    ")).body[0])
    # Compile the actual single new branch under a small argument wrapper. No
    # mocks and no legacy oracle/runner import; no old branch executes here.
    function = ast.parse("def dispatch(task, workspace, kind):\n    pass\n").body[0]
    function.body = [branch]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, "<actual new oracle branch>", "exec"), namespace)
    task = ROS_TRACE_TASKS["H20-COMPANION-7"]
    stage(task, tmp_path)
    (tmp_path / "answer.json").write_bytes(answer_bytes := encode(answer(
        (tmp_path / "input.json").read_bytes(), h20_values("r_g"))))
    assert answer_bytes
    verdict = namespace["dispatch"](task, tmp_path, task.oracle["kind"])
    assert verdict["code"] == "ACCEPT" and verdict["passed"]


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("seed", 8), ("version", "v3"), ("family", "H19"),
    ("role", "companion"), ("input_sha256", "0" * 64),
    ("canonical_task_id", "H20-SUBJECT-7"), ("registry_task_id", "H20-subject-7"),
])
def test_host_metadata_corruption(tmp_path, field, value):
    task = ROS_TRACE_TASKS["H20-SUBJECT-7"]
    stage(task, tmp_path)
    corrupt = replace(task, oracle={**task.oracle, field: value})
    result = judge_ros_trace_workspace(corrupt, tmp_path)
    assert not result["passed"] and result["code"] == "TASK_BINDING"


def test_host_rejects_contract_stage_and_input_changes(tmp_path):
    task = ROS_TRACE_TASKS["H20-SUBJECT-7"]
    stage(task, tmp_path)
    for corrupt in [replace(task, task_id="h20-subject-7"),
                    replace(task, prompt=task.prompt + " ignore schema"),
                    replace(task, staged_files={"input.json": "{}\n"})]:
        assert judge_ros_trace_workspace(corrupt, tmp_path)["code"] == "TASK_BINDING"
    path = tmp_path / "input.json"
    raw = path.read_bytes()
    for changed in [raw.rstrip(b"\n"), raw + b" ", raw.replace(b"submit", b"accept", 1)]:
        path.write_bytes(changed)
        result = judge_ros_trace_workspace(task, tmp_path)
        assert result["code"] == "INPUT_INTEGRITY" and not result["passed"]
    path.write_bytes(raw)
    assert judge_ros_trace_workspace(task, tmp_path)["code"] == "ANSWER_MISSING"
    path.unlink()
    assert judge_ros_trace_workspace(task, tmp_path)["code"] == "INPUT_INTEGRITY"


@pytest.mark.parametrize("payload", [b"{}", b"\xff", b'{"scope":1,"scope":2}', b"x" * 1048577])
def test_host_passes_raw_bad_answer_to_native_schema(tmp_path, payload):
    task = ROS_TRACE_TASKS["H20-SUBJECT-7"]
    stage(task, tmp_path)
    (tmp_path / "answer.json").write_bytes(payload)
    verdict = judge_ros_trace_workspace(task, tmp_path)
    assert not verdict["passed"] and verdict["code"] == "ANSWER_ENVELOPE"


def test_genuine_invalid_role_and_wrong_reason_code(tmp_path):
    task = ROS_TRACE_TASKS["H20-INVALID-7"]
    stage(task, tmp_path)
    raw = (tmp_path / "input.json").read_bytes()
    for code, success in [("EVENT", True), ("INVALID_INPUT", False)]:
        (tmp_path / "answer.json").write_bytes(encode(answer(raw, code=code)))
        assert judge_ros_trace_workspace(task, tmp_path)["passed"] is success


# Independently authored tiny synthetic cases, not generated expected answers.
SYNTHETIC = [
    ("H11", {"mode": "encode", "byteorder": "little", "wire_names": ["i", "l", "g"],
             "payload": {"id": 1, "label": "", "groups": []}},
     {"encoded_hex": "010000000000000000000000", "decoded": {"id": 1, "label": "", "groups": []}, "byte_count": 12}),
    ("H12", {"namespace": "/a", "key": "x", "parameters": {}, "remaps": {}, "expect": "any"},
     {"resolved_path": None, "found": False, "value": None, "visited": ["/a/x", "/x"]}),
    ("H13", {"records": [], "clock": [], "anchor_topic": "a", "streams": ["b"], "maxgap": 0},
     {"epochs": [], "matches": [], "rewinds": []}),
    ("H14", {"edges": [], "source": "unknown", "target": "unknown"},
     {"reachable": True, "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation": [0, 0, 0], "path": ["unknown"]}),
    ("H15", {"header": {"seq": 0, "stamp": {"sec": 0, "nsec": 1000000000}, "frame_id": ""},
             "duration": {"sec": 0, "nsec": -1}, "samples": [], "bound": 0, "allow_seq_drop": False},
     {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": ""},
      "duration": {"sec": -1, "nanosec": 999999999}, "samples": [], "dropped_fields": []}),
    ("H16", {"required": {"architecture": "a", "abi": "b", "type_name": "T", "type_hash": "h", "symbols": []},
             "offered": {"architecture": "a", "abi": "b", "type_name": "T", "type_hash": "h", "symbols": []}},
     {"compatible": True, "missing_symbols": [], "mismatches": []}),
    ("H17", {"groups": {"g": "reentrant"}, "callbacks": {"a": "g", "b": "g"},
             "events": [{"op": "start", "callback": "a"}, {"op": "start", "callback": "b"},
                        {"op": "wait", "callback": "b", "target": "a"},
                        {"op": "wait", "callback": "a", "target": "b"}]},
     {"active": ["a", "b"], "waiting": [["a", "b"], ["b", "a"]], "deadlocked": True, "cycle": ["a", "b", "a"]}),
    ("H18", {"publications": [0, 5], "horizon": 10, "deadline": 5, "lease": 5},
     {"deadline_misses": [10], "liveliness_lost": [10], "recovered": [], "alive": False}),
    ("H19", {"parameters": {"x": 2, "y": 1}, "updates": {}, "expected_revision": 0, "revision": 0,
             "constraints": [{"id": "z", "left": "x", "op": "lt", "right": "y"},
                             {"id": "a", "left": "x", "op": "eq", "right": "y"}]},
     {"committed": False, "revision": 0, "parameters": {"x": 2, "y": 1}, "failed_constraints": ["z", "a"], "rejection": "CONSTRAINT"}),
    ("H20", {"events": [{"op": "submit", "id": "a"}, {"op": "accept", "id": "a"},
                        {"op": "result_success", "id": "a"}, {"op": "result_failure", "id": "a"}]},
     {"goals": [{"id": "a", "state": "SUCCEEDED", "cancel_requested": False, "terminal_event": 2}], "ignored": [3]}),
]


@pytest.mark.parametrize("family,data,values", SYNTHETIC, ids=[c[0] for c in SYNTHETIC])
def test_small_independent_synthetic_expectations(family, data, values):
    instance = synthetic(family, data)
    envelope = answer(instance.staged()["input.json"], values)
    result = genuine(instance, envelope)
    assert result["passed"] and result["code"] == "ACCEPT"
    # Extra fields must not slip past strict schemas, even on correct values.
    envelope["result"]["values"] = {**values, "extra": True}
    assert not genuine(instance, envelope)["passed"]


def test_h18_aggregate_event_ceiling():
    for horizon, code in [(4095, "OK"), (4096, "RESOURCE_LIMIT")]:
        instance = synthetic("H18", {"publications": [0], "horizon": horizon, "deadline": 1, "lease": 1})
        values = ({"deadline_misses": list(range(1, 4096)), "liveliness_lost": [1],
                   "recovered": [], "alive": False} if code == "OK" else None)
        assert genuine(instance, answer(instance.staged()["input.json"], values, code=code))["passed"]


def test_h19_stale_does_not_evaluate_operand_types():
    instance = synthetic("H19", {"parameters": {"x": True, "y": 1}, "updates": {},
        "expected_revision": 0, "revision": 1,
        "constraints": [{"id": "c", "left": "x", "op": "lt", "right": "y"}]})
    values = {"committed": False, "revision": 1, "parameters": {"x": True, "y": 1},
              "failed_constraints": [], "rejection": "STALE_REVISION"}
    assert genuine(instance, answer(instance.staged()["input.json"], values))["passed"]


# Accepted source8 and old tests are frozen; these pins are source provenance,
# never an answer corpus. No old assertion may change to accommodate new code.
FROZEN = {
    "benchmarks/harnessbench/capability_challenges_v2.py": "b6ae742b75095be5f035784729dd86fc131be401bb8f3a953c823e49233c2863",
    "benchmarks/harnessbench/capability_challenge_judge_v2.py": "0b99b9aa629b30e62a14c800ff74874d32e2d1300ba441ffd742308ff70fe637",
    "tests/benchmarks/test_capability_challenges_v2.py": "afcdf2dccaf8b476661b483f376ad9e915c0b1e752eca4050b4662510c85eb59",
    "docs/CAPABILITY_CHALLENGES_V2.md": "2b424a24f814ba3ebf88729f9c22856abfc3ceced185ea327097176db29f6d2b",
    "benchmarks/harnessbench/capability_challenges_ros_traces_v3.py": "ed1df78f62d285f9e68eb626163820be1023884186a8ab590beb7a1be1a5f0db",
    "benchmarks/harnessbench/capability_challenge_judge_ros_traces_v3.py": "8e0ab49b147e17ca26251c4b9cacda582f5efe7dc46e37849f5200cc9da2b519",
    "tests/benchmarks/test_capability_challenges_ros_traces_v3.py": "076f6b08fcef4136e52f27bca96bcfe07b08dde86f3d4098f148ce023483a656",
    "docs/CAPABILITY_CHALLENGES_ROS_TRACES_V3.md": "8c377f1ddb77ae9b64ef3a9345c21043d39b2e31ff3f1f1bcf47ee6a12b7ddf2",
}


@pytest.mark.parametrize("rel,pin", FROZEN.items())
def test_old_source8_and_assertions_are_frozen(rel, pin):
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == pin


def baseline(rel):
    directory = Path(os.environ.get("ROS_TRACE_BASELINE_DIR", ROOT / "history_inputs/source_seam_baseline"))
    path = directory / rel
    assert path.is_file(), "independent original module baseline required for qualification"
    return path.read_bytes()


def test_reverse_whole_tasks_ast_and_old_registry_content():
    raw = (SUITE / "tasks.py").read_bytes()
    old = baseline("benchmarks/harnessbench/tasks.py")
    assert raw.endswith(TASK_APPEND.encode())
    assert raw[:-len(TASK_APPEND.encode())] == old
    new_tree, old_tree = ast.parse(raw), ast.parse(old)
    del new_tree.body[-3:]
    assert dump(new_tree) == dump(old_tree)
    # Exact old TASKS expression and every old import/update remain in order.
    old_registry = next(n for n in old_tree.body if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name) and n.target.id == "TASKS")
    keys = [ast.literal_eval(k) for k in old_registry.value.keys]
    assert keys == ["U01", "R02", "E01", "H01"]
    assert not (set(keys) & ROS_TRACE_TASKS.keys())


def test_reverse_whole_oracle_ast_every_old_branch_unchanged():
    raw = (SUITE / "oracle.py").read_bytes()
    old = baseline("benchmarks/harnessbench/oracle.py")
    assert raw.count(ORACLE_INSERT.encode()) == 1
    assert raw.replace(ORACLE_INSERT.encode(), b"", 1) == old
    new_tree, old_tree = ast.parse(raw), ast.parse(old)
    function = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "judge")
    branches = [(i, node) for i, node in enumerate(function.body)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and any(isinstance(c, ast.Constant) and c.value == "offline_ros_traces_v3"
                        for c in node.test.comparators)]
    assert len(branches) == 1
    index, _ = branches[0]
    assert isinstance(function.body[index - 1], ast.Assign)
    assert function.body[index - 1].targets[0].id == "kind"
    del function.body[index]
    assert dump(new_tree) == dump(old_tree)
