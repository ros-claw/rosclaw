"""Tests for rosclaw body list."""

import json
import sys
from unittest.mock import patch

import pytest

from rosclaw.cli import main as rosclaw_main


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path


def test_body_list_empty():
    """Listing with no registered bodies should print a friendly message."""
    with patch.object(sys, "argv", ["rosclaw", "body", "list"]):
        rc = rosclaw_main()
    assert rc == 0


def test_body_list_text_after_link(capsys):
    """After linking a profile, list should show the body."""
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0

    capsys.readouterr()  # drain link-eurdf output
    with patch.object(sys, "argv", ["rosclaw", "body", "list"]):
        rc = rosclaw_main()
    assert rc == 0

    out = capsys.readouterr().out
    assert "ROSClaw Bodies" in out
    assert "default" in out
    assert "unitree-g1" in out
    assert "*" in out  # current body marker


def test_body_list_json_after_link(capsys, isolated_workspace):
    """JSON list output should include current body metadata."""
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0

    capsys.readouterr()  # drain link-eurdf output
    with patch.object(sys, "argv", ["rosclaw", "body", "list", "--json"]):
        rc = rosclaw_main()
    assert rc == 0

    data = json.loads(capsys.readouterr().out)
    assert data["total"] == 1
    assert data["current"] == "default"
    assert len(data["bodies"]) == 1
    body = data["bodies"][0]
    assert body["body_id"] == "default"
    assert body["profile_id"] == "unitree-g1"
    assert body["is_current"] is True
    assert (isolated_workspace / ".rosclaw" / body["path"]).exists()
