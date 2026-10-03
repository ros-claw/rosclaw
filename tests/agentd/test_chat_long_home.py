"""Long socket paths must fail before starting a background kernel."""

import argparse
from pathlib import Path

import pytest

from rosclaw.agentd import cli


@pytest.mark.parametrize("name", ["x" * 120, "长" * 40])
def test_long_home_has_actionable_diagnostic_and_no_kernel(name, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("kernel must not be created")

    monkeypatch.setattr(cli, "_find_pi_agent_entry", forbidden)
    assert cli._chat_pi(Path("/tmp") / name, argparse.Namespace()) == 2
    err = capsys.readouterr().err
    assert "ROSCLAW_HOME" in err
    assert "Unix socket" in err
    assert "Traceback" not in err
