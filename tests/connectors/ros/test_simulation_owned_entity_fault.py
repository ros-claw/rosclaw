"""Synthetic Gazebo command/identity contracts; never real replacement proof."""

import hashlib
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_owned_entity_fault")
    assets = tmp_path / "assets"
    assets.mkdir()
    root = tmp_path / "evidence"
    root.mkdir()
    world = b'<sdf version="1.9"><world name="unconventional_world"><model name="declared_robot"><pose>0 0 .1 0 0 0</pose><link name="official_link"><inertial><mass>17</mass></inertial></link></model></world></sdf>'
    (assets / "world.sdf").write_bytes(world)
    (assets / "robot.urdf").write_bytes(b"synthetic-original-URDF")
    (assets / "observer-sources.json").write_text("{}")
    source = SimpleNamespace(robot_entity="declared_robot", run_id="original-run")
    monkeypatch.setattr(module.SimulationWitnessSources, "from_bytes", lambda *a, **k: source)
    records = iter(
        [
            {"sample": {"models": [{"name": "declared_robot", "entity": 10}]}},
            {"sample": {"models": []}},
            {"sample": {"models": [{"name": "declared_robot", "entity": 20}]}},
        ]
    )
    monkeypatch.setattr(module, "latest_owned_ecm", lambda *a: next(records))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "/create" in argv[3]:
            path = json.loads(
                argv[-1].split("sdf_filename: ", 1)[1].split(", allow_renaming", 1)[0]
            )
            assert (
                Path(path).read_bytes()
                == (root / "governed-replacement-model-original.sdf").read_bytes()
            )
        return SimpleNamespace(returncode=0, stdout=b"data: true\n", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    policy = {"world_sha256": hashlib.sha256(world).hexdigest(), "run_id": "original-run"}
    return module, root, assets, policy, calls


def test_only_declared_world_model_is_removed_and_original_xml_recreated(fixture):
    module, root, assets, policy, calls = fixture
    result = module.replace_owned_model(root, assets, policy)
    assert result["original_entity_id"] == 10 and result["replacement_entity_id"] == 20
    assert (
        result["original_body_removed"] is True and result["replacement_body_reenrolled"] is False
    )
    assert (
        calls[0][3] == "/world/unconventional_world/remove"
        and calls[0][-1] == 'name: "declared_robot", type: MODEL'
    )
    assert (
        calls[1][3] == "/world/unconventional_world/create"
        and "allow_renaming: false" in calls[1][-1]
    )
    model = (root / "governed-replacement-model-original.sdf").read_bytes()
    assert b"<mass>17</mass>" in model and b"official_link" in model
    assert result["replacement_model_sha256"] == hashlib.sha256(model).hexdigest()


def test_changed_world_never_sends_a_gazebo_operation(fixture):
    module, root, assets, policy, calls = fixture
    (assets / "world.sdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="original pinned World"):
        module.replace_owned_model(root, assets, policy)
    assert not calls


def test_unacknowledged_removal_never_attempts_recreation(fixture, monkeypatch):
    module, root, assets, policy, calls = fixture

    def fail(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=b"data: false\n", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(ValueError, match="not acknowledged"):
        module.replace_owned_model(root, assets, policy)
    assert len(calls) == 1


def test_expired_loading_observation_is_unknown_until_a_fresh_new_id(fixture, monkeypatch):
    module, root, assets, policy, _ = fixture
    records = iter(
        [
            {"sample": {"models": [{"name": "declared_robot", "entity": 10}]}},
            {"sample": {"models": []}},
            module.ExpiredOwnedEcmObservationError("synthetic source gap"),
            {"sample": {"models": [{"name": "declared_robot", "entity": 20}]}},
        ]
    )

    def latest(*args):
        row = next(records)
        if isinstance(row, Exception):
            raise row
        return row

    monkeypatch.setattr(module, "latest_owned_ecm", latest)
    result = module.replace_owned_model(root, assets, policy)
    assert (
        result["replacement_entity_id"] == 20 and len(result["loading_observation_unknowns"]) == 1
    )


@pytest.mark.parametrize("fault", [None, "run", "observer_pid", "stale", "future", "symlink"])
def test_latest_source_requires_fresh_original_run_and_observer(tmp_path, monkeypatch, fault):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_owned_entity_fault")
    pins = {"observer_epoch": "a" * 32, "observer_pid": 101}
    (tmp_path / "execution-facts-enrollment.json").write_text(json.dumps({"source_pins": pins}))
    sample = {
        "observer_run_label": "run",
        "observer_monotonic_ns": time.monotonic_ns(),
        "source_stream": dict(pins),
        "models": [],
    }
    if fault == "run":
        sample["observer_run_label"] = "other"
    if fault == "observer_pid":
        sample["source_stream"]["observer_pid"] = 102
    if fault == "stale":
        sample["observer_monotonic_ns"] -= 600_000_000
    if fault == "future":
        sample["observer_monotonic_ns"] += 1_000_000_000
    raw = (json.dumps(sample) + "\n").encode()
    path = tmp_path / "ecm-original.jsonl"
    if fault == "symlink":
        target = tmp_path / "not-an-original-source"
        target.write_bytes(raw)
        path.symlink_to(target)
    else:
        path.write_bytes(raw + b'{"incomplete_append":')
    if fault:
        with pytest.raises((ValueError, OSError)):
            module.latest_owned_ecm(tmp_path, {"run_id": "run"})
    else:
        result = module.latest_owned_ecm(tmp_path, {"run_id": "run"})
        assert (
            result["sample"] == sample
            and result["original_record_sha256"] == hashlib.sha256(raw).hexdigest()
        )
