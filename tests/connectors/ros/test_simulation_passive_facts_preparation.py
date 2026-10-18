"""Execute only the pure preparation method; no ROS node or Runtime imports."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def method(name, globals):
    path = (
        Path(__file__).resolve().parents[3]
        / "integrations/ros_probe/acceptance/simulation_governed_sink.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GovernedSimulationSink"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"Path": Path, "json": json, "hashlib": hashlib}
    namespace.update(globals)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def fixture(tmp_path, *, wrong_body=False):
    control = tmp_path / "execution-control"
    control.mkdir()
    raw = b'{"original_enrollment":true}\n'
    (tmp_path / "execution-facts-enrollment.json").write_bytes(raw)
    calls = []
    facts = SimpleNamespace(
        enrollment_sha256=hashlib.sha256(raw).hexdigest(),
        enrollment={
            "request": {
                "body_snapshot_hash": "wrong" if wrong_body else "current-body",
                "sim_run_id": "current-run",
            }
        },
        reader=SimpleNamespace(poll=lambda: calls.append("current_source_poll")),
        close=lambda: calls.append("closed"),
    )
    sink = SimpleNamespace(
        root=tmp_path,
        control=control,
        facts=None,
        passive_preparation=None,
        binding=SimpleNamespace(body_snapshot_hash="current-body", run_id="current-run"),
        admission=None,
        gate=None,
        monitor=None,
        client=None,
        velocity=None,
        _event=lambda *a, **k: calls.append("preparation_event"),
    )

    manifest = tmp_path / "mesh-manifest-present"
    manifest.touch()

    class Preparation:
        def __init__(self, *a, **k):
            assert k["enrollment_sha256"] == facts.enrollment_sha256

        def start(self):
            calls.append("constructor")
            assert not (control / "admission.json").exists()

        def take_if_ready(self):
            return facts

    prepare = method(
        "_prepare_passive_facts",
        {
            "_read": lambda p, n: p.read_bytes(),
            "SdkPassiveFactsPreparation": Preparation,
            "Path": lambda p: (
                manifest if str(p) == "/assets/collision-asset-originals.json" else Path(p)
            ),
        },
    )
    return sink, facts, calls, prepare


def test_expensive_replay_occurs_without_admission_or_control_and_is_not_repeated(tmp_path):
    sink, facts, calls, prepare = fixture(tmp_path)
    prepare(sink)
    assert sink.facts is None
    prepare(sink)
    prepare(sink)
    assert calls.count("constructor") == 1 and calls.count("current_source_poll") == 2
    assert sink.facts is facts
    assert all(
        getattr(sink, field) is None
        for field in ["admission", "gate", "monitor", "client", "velocity"]
    )
    marker = json.loads((tmp_path / "passive-facts-prepared.json").read_text())
    assert marker["enrollment_sha256"] == facts.enrollment_sha256
    assert marker["execution_authority"] == "NONE" and marker["admission"] == "NOT_GRANTED"


def test_different_body_cannot_prepare_or_write_ready_marker(tmp_path):
    sink, _, calls, prepare = fixture(tmp_path, wrong_body=True)
    prepare(sink)
    with pytest.raises(ValueError, match="current unadmitted Body"):
        prepare(sink)
    assert sink.facts is None and "closed" in calls
    assert not (tmp_path / "passive-facts-prepared.json").exists()


def test_unprepared_admission_cannot_start_slow_preparation_in_callback(tmp_path):
    sink, _, calls, prepare = fixture(tmp_path)
    (sink.control / "admission.json").write_text("{}")
    prepare(sink)
    assert calls == [] and sink.facts is None
    assert not (tmp_path / "passive-facts-prepared.json").exists()
