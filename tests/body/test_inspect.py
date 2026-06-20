"""Tests for rosclaw body inspect."""

import json
import sys
from unittest.mock import patch

import pytest

from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_inspect_text_output(linked_body, capsys):
    with patch.object(sys, "argv", ["rosclaw", "body", "inspect"]):
        rc = rosclaw_main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "ROSClaw Body Inspect" in out
    assert "unitree-g1" in out


def test_inspect_json_output(linked_body, capsys):
    with patch.object(sys, "argv", ["rosclaw", "body", "inspect", "--json"]):
        rc = rosclaw_main()
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["body_instance_id"]
    assert data["effective_body_hash"]
    assert "capabilities" in data


def test_inspect_without_body():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, patch.dict(
        "os.environ", {"HOME": tmp}
    ), patch.object(sys, "argv", ["rosclaw", "body", "inspect"]):
        rc = rosclaw_main()
        assert rc == 1
