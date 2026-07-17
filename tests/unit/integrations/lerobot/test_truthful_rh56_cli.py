"""Truthfulness boundaries for the RH56 rollout CLI."""

from __future__ import annotations

import argparse

import pytest

from rosclaw.integrations.lerobot.cli import (
    cmd_lerobot_rollout_arm,
    cmd_lerobot_rollout_execute,
    cmd_lerobot_rollout_rh56_shadow,
)


def test_rh56_shadow_requires_explicit_fixture_without_live_transport(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cmd_lerobot_rollout_rh56_shadow(argparse.Namespace(fixture=False))

    assert result == 1
    output = capsys.readouterr().out
    assert "shadow refused" in output
    assert "--fixture" in output


@pytest.mark.parametrize(
    "command",
    [cmd_lerobot_rollout_arm, cmd_lerobot_rollout_execute],
)
def test_rh56_real_execution_requires_runtime_gateway(
    command,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = command(argparse.Namespace(fixture=False))

    assert result == 1
    output = capsys.readouterr().out
    assert "Runtime.submit_action()" in output
    assert "--fixture" in output
