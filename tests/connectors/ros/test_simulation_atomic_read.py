"""Atomic snapshot unlink races are distinct from in-place original mutation."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_atomic_read")


@pytest.mark.parametrize("allow_atomic", [False, True])
def test_atomic_replacement_keeps_original_bytes_but_strict_assets_refuse_unlink(
    module, monkeypatch, tmp_path, allow_atomic
):
    path = tmp_path / "current.json"
    path.write_bytes(b"original")
    fresh = tmp_path / "new.json"
    fresh.write_bytes(b"new-data")
    original_read = module.os.read

    def replace_during_read(fd, limit):
        raw = original_read(fd, limit)
        fresh.replace(path)
        return raw

    monkeypatch.setattr(module.os, "read", replace_during_read)
    if allow_atomic:
        assert module.read_bounded_regular(path, 128, allow_atomic_unlink=True) == b"original"
        assert path.read_bytes() == b"new-data"
    else:
        with pytest.raises(ValueError, match="changed during read"):
            module.read_bounded_regular(path, 128)


def test_in_place_change_is_rejected_even_for_mutable_snapshot(module, monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    path.write_bytes(b"original")
    original_read = module.os.read

    def mutate(fd, limit):
        raw = original_read(fd, limit)
        path.write_bytes(b"changed!")
        return raw

    monkeypatch.setattr(module.os, "read", mutate)
    with pytest.raises(ValueError, match="changed during read"):
        module.read_bounded_regular(path, 128, allow_atomic_unlink=True)


def test_symlink_and_oversize_are_always_refused(module, tmp_path):
    path = tmp_path / "original.json"
    path.write_bytes(b"original")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        module.read_bounded_regular(link, 128, allow_atomic_unlink=True)
    with pytest.raises(ValueError, match="bounded original"):
        module.read_bounded_regular(path, 2, allow_atomic_unlink=True)


@pytest.mark.parametrize("link_count", [0, 1])
def test_atomic_snapshot_verifies_bytes_when_rename_ctime_arrives_late(
    module, monkeypatch, tmp_path, link_count
):
    from types import SimpleNamespace

    path = tmp_path / "current.json"
    path.write_bytes(b"original")
    original_stat = module.os.fstat
    calls = 0

    def delayed_ctime(fd):
        nonlocal calls
        info = original_stat(fd)
        calls += 1
        fields = {
            key: getattr(info, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_uid", "st_gid")
        }
        return SimpleNamespace(
            **fields, st_nlink=link_count, st_ctime_ns=info.st_ctime_ns + (calls > 1)
        )

    monkeypatch.setattr(module.os, "fstat", delayed_ctime)
    assert module.read_bounded_regular(path, 128, allow_atomic_unlink=True) == b"original"
    assert calls == 3


def test_same_size_write_with_restored_mtime_is_rejected_by_second_byte_read(
    module, monkeypatch, tmp_path
):
    path = tmp_path / "current.json"
    path.write_bytes(b"original")
    original_read = module.os.read
    before = path.stat()

    def mutate_and_restore_mtime(fd, limit):
        raw = original_read(fd, limit)
        path.write_bytes(b"changed!")
        module.os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return raw

    monkeypatch.setattr(module.os, "read", mutate_and_restore_mtime)
    with pytest.raises(ValueError, match="changed during read"):
        module.read_bounded_regular(path, 128, allow_atomic_unlink=True)


def test_atomic_snapshot_rejects_permissions_change(module, monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    path.write_bytes(b"original")
    path.chmod(0o600)
    original_read = module.os.read

    def change_permissions(fd, limit):
        raw = original_read(fd, limit)
        path.chmod(0o400)
        return raw

    monkeypatch.setattr(module.os, "read", change_permissions)
    with pytest.raises(ValueError, match="changed during read"):
        module.read_bounded_regular(path, 128, allow_atomic_unlink=True)
