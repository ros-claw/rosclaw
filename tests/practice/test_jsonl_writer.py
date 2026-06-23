"""Tests for JsonlWriter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rosclaw.practice.writers.jsonl_writer import JsonlWriter


def test_write_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        writer = JsonlWriter(path)
        writer.write({"event": 1})
        writer.close()
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == 1


def test_multiple_records():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        with JsonlWriter(path) as writer:
            for i in range(3):
                writer.write({"i": i})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["i"] for line in lines] == [0, 1, 2]


def test_bad_record_written_as_string():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        writer = JsonlWriter(path)
        writer.write({"ok": True})
        writer.write(object())  # type: ignore[arg-type]
        writer.write({"ok": False})
        writer.close()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["ok"] is True
        assert json.loads(lines[2])["ok"] is False
        # middle line is the string representation of the object
        assert "object object" in lines[1]


def test_rotation():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        writer = JsonlWriter(path, rotate_mb=0.001)  # ~1KB
        for _ in range(100):
            writer.write({"payload": "x" * 100})
        writer.close()
        assert path.exists()
        rotated = list(Path(tmp).glob("events.jsonl.*"))
        assert len(rotated) >= 1


def test_write_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "meta.jsonl"
        writer = JsonlWriter(path)
        writer.write_atomic({"meta": True})
        writer.close()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["meta"] is True
