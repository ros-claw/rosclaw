"""Help must never enroll identities or start an operator service."""

import pytest

from rosclaw.operatord.cli import dispatch_operatord_argv


@pytest.mark.parametrize("command", ["enroll", "start", "register-daemon", "revoke-daemon"])
def test_help_is_read_only(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    assert dispatch_operatord_argv(["operatord", command, "--help"]) == 0
    assert "用法" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
