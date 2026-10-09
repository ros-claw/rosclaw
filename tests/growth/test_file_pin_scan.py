"""Temporary ordinary files only; no experiment input is modified."""

import hashlib
import json
import os
from dataclasses import FrozenInstanceError

import pytest

import rosclaw.growth.file_pin_scan as scan


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / "first"
    path.write_bytes(b"immutable fixture")
    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return path, expected


def test_hardlinks_verified_but_read_once(fixture):
    path, expected = fixture
    alias = path.with_name("alias")
    os.link(path, alias)
    result = scan.verify_file_pins({str(path): expected, str(alias): expected})
    assert result.verified_paths == 2
    assert result.unique_content_reads == 1
    assert result.bytes_read == path.stat().st_size
    assert result.logical_bytes == result.bytes_read * 2
    with pytest.raises(FrozenInstanceError):
        result.bytes_read = 0


def test_equal_bytes_in_distinct_files_not_cached(fixture):
    path, expected = fixture
    other = path.with_name("other")
    other.write_bytes(path.read_bytes())
    result = scan.verify_file_pins({str(path): expected, str(other): expected})
    assert result.unique_content_reads == 2


def test_empty_file(fixture):
    path, _ = fixture
    path.write_bytes(b"")
    result = scan.verify_file_pins({str(path): "sha256:" + hashlib.sha256(b"").hexdigest()})
    assert result.bytes_read == result.logical_bytes == 0
    assert result.unique_content_reads == 1


def test_conflicting_alias_hash_rejected(fixture):
    path, expected = fixture
    alias = path.with_name("alias")
    os.link(path, alias)
    with pytest.raises(ValueError, match="content mismatch"):
        scan.verify_file_pins({str(path): expected, str(alias): "sha256:" + "0" * 64})


def test_changed_content_rejected_across_calls(fixture):
    path, expected = fixture
    pins = {str(path): expected}
    scan.verify_file_pins(pins)
    path.write_bytes(b"changed content!!")
    with pytest.raises(ValueError, match="content mismatch"):
        scan.verify_file_pins(pins)


@pytest.mark.parametrize("which", ["leaf", "parent"])
def test_any_symlink_component_rejected(fixture, which):
    path, expected = fixture
    if which == "leaf":
        alias = path.with_name("alias")
        alias.symlink_to(path)
    else:
        parent = path.parent / "linked-directory"
        parent.symlink_to(path.parent, target_is_directory=True)
        alias = parent / path.name
    with pytest.raises(OSError):
        scan.verify_file_pins({str(alias): expected})


@pytest.mark.parametrize("kind", ["directory", "fifo", "missing"])
def test_nonordinary_files_rejected_without_blocking(fixture, kind):
    path, expected = fixture
    target = path.with_name(kind)
    if kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        os.mkfifo(target)
    with pytest.raises((OSError, ValueError)):
        scan.verify_file_pins({str(target): expected})


@pytest.mark.parametrize("mode", ["in_place", "replace", "remove", "symlink_parent"])
def test_changes_during_scan_rejected(fixture, monkeypatch, mode):
    path, expected = fixture
    original = scan._digest_exact_size

    def changed(stream, algorithm):
        result = original(stream, algorithm)
        if mode == "in_place":
            path.write_bytes(b"changed in place")
        elif mode == "replace":
            other = path.with_name("replacement")
            other.write_bytes(path.read_bytes())
            os.replace(other, path)
        elif mode == "remove":
            path.unlink()
        else:
            parent = path.parent
            destination = parent.with_name(parent.name + "-moved")
            parent.rename(destination)
            parent.symlink_to(destination, target_is_directory=True)
        return result

    monkeypatch.setattr(scan, "_digest_exact_size", changed)
    with pytest.raises((ValueError, OSError)):
        scan.verify_file_pins({str(path): expected})


def test_caller_mutation_does_not_change_owned_scope(fixture, monkeypatch):
    path, expected = fixture
    pins = {str(path): expected}
    original = scan._digest_exact_size
    expected_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(pins, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )

    def mutate_caller(stream, algorithm):
        pins.clear()
        pins["not a valid path"] = "forged"
        return original(stream, algorithm)

    monkeypatch.setattr(scan, "_digest_exact_size", mutate_caller)
    result = scan.verify_file_pins(pins)
    assert result.verified_paths == 1
    assert result.declaration_hash == expected_hash


@pytest.mark.parametrize("limit", [0, -1, True, 0.5, None])
def test_invalid_capacity(fixture, limit):
    path, expected = fixture
    with pytest.raises(ValueError, match="capacity"):
        scan.verify_file_pins({str(path): expected}, max_bytes=limit)


def test_capacity_counts_unique_content_not_aliases(fixture):
    path, expected = fixture
    alias = path.with_name("alias")
    os.link(path, alias)
    scan.verify_file_pins(
        {str(path): expected, str(alias): expected}, max_bytes=path.stat().st_size
    )
    with pytest.raises(ValueError, match="capacity"):
        scan.verify_file_pins({str(path): expected}, max_bytes=path.stat().st_size - 1)


@pytest.mark.parametrize(
    "pins",
    [
        {},
        [],
        {"relative": "sha256:" + "0" * 64},
        {"/a/../b": "sha256:" + "0" * 64},
        {"//a/b": "sha256:" + "0" * 64},
        {"/a//b": "sha256:" + "0" * 64},
        {"/a/./b": "sha256:" + "0" * 64},
        {"/": "sha256:" + "0" * 64},
        {"/a": "sha256:" + "z" * 64},
        {"/a": "sha256:" + "A" * 64},
        {"/a": None},
        {1: "sha256:" + "0" * 64},
    ],
)
def test_invalid_declarations(pins):
    with pytest.raises(ValueError):
        scan.verify_file_pins(pins)


def test_byte_limit_applies_across_distinct_files(fixture):
    path, expected = fixture
    other = path.with_name("other")
    other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="capacity"):
        scan.verify_file_pins(
            {str(path): expected, str(other): expected}, max_bytes=path.stat().st_size
        )


def test_open_descriptors_closed_on_hash_failure(fixture, monkeypatch):
    path, expected = fixture
    opened = []

    def fail(stream, algorithm):
        opened.append(stream)
        raise RuntimeError("injected hash failure")

    monkeypatch.setattr(scan, "_digest_exact_size", fail)
    with pytest.raises(RuntimeError, match="injected"):
        scan.verify_file_pins({str(path): expected})
    assert opened and all(stream.closed for stream in opened)


@pytest.mark.parametrize("mode", ["grow", "truncate"])
def test_length_change_before_read_is_bounded_and_rejected(fixture, monkeypatch, mode):
    path, expected = fixture
    original = scan._digest_exact_size

    def resize(stream, size):
        path.write_bytes(b"" if mode == "truncate" else path.read_bytes() + b"extra")
        return original(stream, size)

    monkeypatch.setattr(scan, "_digest_exact_size", resize)
    with pytest.raises(ValueError, match="length changed"):
        scan.verify_file_pins({str(path): expected})
