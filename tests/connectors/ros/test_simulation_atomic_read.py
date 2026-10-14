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
