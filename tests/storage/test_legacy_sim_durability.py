"""Legacy model import integrity with filesystem-only fixtures; no dynamics."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from rosclaw.sim import api


def _source(root: Path, mass: float) -> Path:
    root.mkdir()
    source = root / "robot.xml"
    source.write_text(
        f'<mujoco model="fixture"><worldbody><body name="body"><joint name="joint" type="hinge"/><geom name="geom" type="sphere" size="0.1" mass="{mass}"/></body></worldbody></mujoco>'
    )
    return source


def test_same_basename_import_cannot_change_original_reference_body(tmp_path):
    first = _source(tmp_path / "first", 1)
    second = _source(tmp_path / "second", 2)
    task = tmp_path / "task"
    ref, _ = api.load_model(first, task_root=task)
    old = api._load_model_record(ref, task)
    staged = Path(old["path"])
    original = staged.read_bytes()
    next_ref, _ = api.load_model(second, task_root=task)
    assert next_ref != ref
    assert staged.read_bytes() == original
    assert api._load_model_record(ref, task) == old
    assert Path(api._load_model_record(next_ref, task)["path"]) != staged


def test_corrupt_legacy_metadata_reference_is_not_reconstructed(tmp_path):
    source = _source(tmp_path / "source", 1)
    task = tmp_path / "task"
    ref, _ = api.load_model(source, task_root=task)
    record = task / "models" / f"{ref}.json"
    record.write_bytes(b"")
    with pytest.raises(ValueError, match="LEGACY_STORE_(CORRUPT|IMMUTABLE)"):
        api.load_model(source, task_root=task)
    assert record.read_bytes() == b""


def test_legacy_model_and_registry_are_acknowledged_after_file_and_dir_barriers(
    tmp_path, monkeypatch
):
    source = _source(tmp_path / "source", 1)
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append(("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file", None))
        real_sync(fd)

    def replace(source, target):
        events.append(("replace", str(target)))
        real_replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    task = tmp_path / "task"
    ref, _ = api.load_model(source, task_root=task)
    record = api._load_model_record(ref, task)
    for path in (record["path"], str(task / "models" / f"{ref}.json")):
        index = events.index(("replace", path))
        assert events[index - 1][0] == "file"
        assert events[index + 1][0] == "dir"


def test_tampered_staged_xml_is_rejected_without_repair(tmp_path):
    source = _source(tmp_path / "source", 1)
    task = tmp_path / "task"
    ref, _ = api.load_model(source, task_root=task)
    record = api._load_model_record(ref, task)
    staged = Path(record["path"])
    staged.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="LEGACY_STORE_CORRUPT"):
        api.load_model(source, task_root=task)
    assert staged.read_bytes() == b"tampered"


def test_identical_body_from_new_source_preserves_first_provenance_and_bytes(tmp_path):
    first = _source(tmp_path / "first", 1)
    second = _source(tmp_path / "second", 1)
    task = tmp_path / "task"
    ref, description = api.load_model(first, task_root=task)
    registry = task / "models" / f"{ref}.json"
    original = registry.read_bytes()
    repeated_ref, repeated_description = api.load_model(second, task_root=task)
    assert (repeated_ref, repeated_description) == (ref, description)
    assert registry.read_bytes() == original
    assert api._load_model_record(ref, task)["source_path"] == str(first)


def test_legacy_wall_time_retry_keeps_original_metadata_and_ref_bound_fields(tmp_path):
    target = tmp_path / "models" / "op_fixture.json"
    payload = {"states_digest": "fixture", "model_ref": "fixture", "wall_ms": 1}
    api._write_legacy_json(target, payload, tmp_path, retain_wall_time=True)
    original = target.read_bytes()
    api._write_legacy_json(target, {**payload, "wall_ms": 2}, tmp_path, retain_wall_time=True)
    assert target.read_bytes() == original
    with pytest.raises(ValueError, match="LEGACY_STORE_IMMUTABLE"):
        api._write_legacy_json(
            target, {**payload, "model_ref": "different"}, tmp_path, retain_wall_time=True
        )
    assert target.read_bytes() == original


def test_legacy_registry_failed_directory_ack_retries_without_reconstruction(tmp_path, monkeypatch):
    import hashlib

    source = _source(tmp_path / "source", 1)
    task = tmp_path / "task"
    ref = "model_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    registry = task / "models" / f"{ref}.json"
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail(fd):
            if (
                stat.S_ISDIR(os.fstat(fd).st_mode)
                and os.readlink(f"/proc/self/fd/{fd}") == str(registry.parent)
                and registry.exists()
            ):
                raise OSError("legacy registry directory sync failure")
            real(fd)

        scoped.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="legacy registry directory sync failure"):
            api.load_model(source, task_root=task)
    original = registry.read_bytes()
    assert api.load_model(source, task_root=task)[0] == ref
    assert registry.read_bytes() == original


@pytest.mark.parametrize("damage", [b"", b"[", b'"not an object"'])
def test_legacy_corrupt_state_json_fails_closed(tmp_path, damage):
    models = tmp_path / "models"
    models.mkdir()
    target = models / "op_fixture.json"
    target.write_bytes(damage)
    with pytest.raises(ValueError, match="LEGACY_STORE_CORRUPT"):
        api._load_state_record("op_fixture", tmp_path)
    assert target.read_bytes() == damage


def test_source_mutation_during_inspection_never_acknowledges_stale_description(
    tmp_path, monkeypatch
):
    source = _source(tmp_path / "source", 1)
    real = api.inspect_mjcf

    def inspect(path):
        information = real(path)
        path.write_bytes(path.read_bytes().replace(b'mass="1"', b'mass="2"'))
        return information

    monkeypatch.setattr(api, "inspect_mjcf", inspect)
    task = tmp_path / "task"
    with pytest.raises(ValueError, match="LEGACY_INPUT_CHANGED"):
        api.load_model(source, task_root=task)
    assert not list((task / "models").glob("model_*.json"))
