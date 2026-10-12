"""Original contact retention, bounded capacity and immutable file contracts."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def capture_class(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_source_capture").BoundedSourceJsonlCapture


def test_exact_original_lines_survive_rotation_with_valid_sha_chain(tmp_path, monkeypatch):
    cls = capture_class(monkeypatch)
    p = tmp_path / "contacts.jsonl"
    writer = cls(p, max_segments=3, max_bytes=20, max_rows=2, max_line_bytes=20)
    lines = ['{"i":0}\n', '{"i":1}\n', '{"i":2}\n', '{"i":3}\n', '{"i":4}\n']
    for line in lines:
        writer.write(line)
    writer.close()
    chain = [
        json.loads(row) for row in (tmp_path / "contacts.chain.jsonl").read_text().splitlines()
    ]
    assert len(chain) == 3
    originals = b""
    previous = ""
    for index, link in enumerate(chain):
        raw = (tmp_path / link["file"]).read_bytes()
        assert link["sha256"] == hashlib.sha256(raw).hexdigest()
        assert link["index"] == index and link["previous_sha256"] == previous
        assert link["bytes"] == len(raw) and link["rows"] == len(raw.splitlines())
        originals += raw
        previous = link["sha256"]
    assert originals == "".join(lines).encode()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(lines[0])


def test_budget_exhaustion_is_terminal_and_retains_original_segments(tmp_path, monkeypatch):
    cls = capture_class(monkeypatch)
    writer = cls(tmp_path / "c.jsonl", max_segments=2, max_rows=1)
    writer.write("{}\n")
    writer.write('{"x":1}\n')
    with pytest.raises(RuntimeError, match="exhausted"):
        writer.write('{"x":2}\n')
    assert writer.closed
    assert (tmp_path / "c.jsonl").read_bytes() == b"{}\n"
    assert (tmp_path / "c.segment-000001.jsonl").read_bytes() == b'{"x":1}\n'
    assert not (tmp_path / "c.segment-000002.jsonl").exists()
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write("{}\n")


@pytest.mark.parametrize("line", ["missing-newline", "{}\n{}\n", "x" * 10 + "\n"])
def test_bad_or_oversized_lines_close_capture(tmp_path, monkeypatch, line):
    writer = capture_class(monkeypatch)(tmp_path / "c.jsonl", max_line_bytes=10)
    with pytest.raises(RuntimeError):
        writer.write(line)
    assert writer.closed and (tmp_path / "c.jsonl").read_bytes() == b""


@pytest.mark.parametrize("kind", ["existing", "symlink", "rotation-existing"])
def test_never_overwrite_or_follow_a_source_file(tmp_path, monkeypatch, kind):
    cls = capture_class(monkeypatch)
    target = tmp_path / "original"
    target.write_bytes(b"retain")
    p = tmp_path / "c.jsonl"
    if kind == "existing":
        p.write_bytes(b"retain")
    elif kind == "symlink":
        p.symlink_to(target)
    if kind in {"existing", "symlink"}:
        with pytest.raises(OSError):
            cls(p)
    else:
        writer = cls(p, max_segments=2, max_rows=1)
        writer.write("{}\n")
        (tmp_path / "c.segment-000001.jsonl").write_bytes(b"retain")
        with pytest.raises(OSError):
            writer.write("{}\n")
        assert writer.closed
        assert (tmp_path / "c.segment-000001.jsonl").read_bytes() == b"retain"
    assert target.read_bytes() == b"retain"


def test_default_preserves_single_file_and_closed_capacity(tmp_path, monkeypatch):
    cls = capture_class(monkeypatch)
    writer = cls(tmp_path / "c.jsonl", max_rows=1)
    writer.write("{}\n")
    with pytest.raises(RuntimeError):
        writer.write("{}\n")
    assert not (tmp_path / "c.chain.jsonl").exists()
    with pytest.raises(ValueError):
        cls(tmp_path / "invalid.jsonl", max_segments=5)
