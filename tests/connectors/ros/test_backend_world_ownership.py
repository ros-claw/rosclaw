"""Linux procfs source parsing and frozen-source negatives, no World launch."""

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("backend_world_ownership")


def test_real_procfs_zero_size_does_not_discard_original_map_bytes(module):
    path = Path(f"/proc/{os.getpid()}/maps")
    assert path.stat().st_size == 0
    rows = module.mapped_file_rows(module.bounded_process_file(path, 16000000))
    assert rows
    for path, identity in rows.items():
        p = Path(path)
        if p.exists():
            assert identity == (p.stat().st_dev, p.stat().st_ino)


def test_unlinked_shared_memory_cannot_alias_required_mapped_library(module):
    raw = b"100-200 rw-s 0000 00:01 123 /dev/shm/unlinked (deleted)\n200-300 r-xp 0000 00:02 456 /owned/contact.so (deleted)\n300-400 r-xp 0000 00:03 789 /sdk/libgz-sim8.so.8.15.0\n"
    assert module.mapped_file_rows(raw) == {"/sdk/libgz-sim8.so.8.15.0": (os.makedev(0, 3), 789)}


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff",
        b"100-200 r-xp 0 invalid 3 /owned.so\n",
        b"100-200 r-xp 0 00:01 3 /owned.so\n200-300 r-xp 0 00:01 4 /owned.so\n",
    ],
)
def test_malformed_or_ambiguous_original_process_map_is_refused(module, raw):
    with pytest.raises((ValueError, UnicodeError)):
        module.mapped_file_rows(raw)


@pytest.mark.parametrize("fault", ["hash", "escape", "missing_world", "wrong_partition"])
def test_world_source_cannot_be_admitted_from_wrong_manifest_or_partition(module, tmp_path, fault):
    source = tmp_path / "world-source"
    source.mkdir()
    world = source / "world.sdf"
    world.write_text('<sdf version="1.9"><world name="synthetic_not_started"/></sdf>')
    lib = source / "librosclaw_passive_contacts.so"
    lib.write_bytes(b"\x7fELFsynthetic_not_loadable")
    hashes = {
        str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (world, lib)
    }
    if fault == "hash":
        hashes["world-source/world.sdf"] = "0" * 64
    elif fault == "escape":
        hashes["../outside"] = "a" * 64
    elif fault == "missing_world":
        hashes.pop("world-source/world.sdf")
    m = {
        "schema_version": "rosclaw.backend_world_source_bundle.v1",
        "source_contact_library_sha256": hashes["world-source/librosclaw_passive_contacts.so"],
        "output_hashes": hashes,
    }
    (tmp_path / "backend-world-bundle.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        module.WorldSourceOwner(
            tmp_path,
            world_pid=os.getpid(),
            world_uid=os.getuid(),
            partition="wrong_partition"
            if fault == "wrong_partition"
            else "rosclaw_backend_" + "b" * 32,
            seed=101001,
            physics_plugin=lib,
            physics_plugin_sha256=hashes["world-source/librosclaw_passive_contacts.so"],
            contact_plugin_sha256=hashes["world-source/librosclaw_passive_contacts.so"],
        )
