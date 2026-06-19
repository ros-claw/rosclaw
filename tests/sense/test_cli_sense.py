"""Tests for rosclaw sense CLI commands."""

import argparse

from rosclaw.sense.cli import (
    cmd_sense_explain,
    cmd_sense_now,
    cmd_sense_readiness,
    cmd_sense_state,
)


class TestSenseCli:
    def _args(self, **kwargs):
        return argparse.Namespace(**kwargs)

    def test_cmd_sense_now(self, capsys):
        args = self._args(mock="normal", robot_id="g1", json=True)
        assert cmd_sense_now(args) == 0
        out = capsys.readouterr().out
        assert '"overall_status"' in out

    def test_cmd_sense_state(self, capsys):
        args = self._args(mock="normal", robot_id="g1", json=True)
        assert cmd_sense_state(args) == 0
        out = capsys.readouterr().out
        assert '"robot_id"' in out

    def test_cmd_sense_readiness_hot_knee(self, capsys):
        args = self._args(
            task="kick_ball", mock="hot_knee", robot_id="g1", json=True
        )
        assert cmd_sense_readiness(args) == 0
        out = capsys.readouterr().out
        assert "not_ready" in out

    def test_cmd_sense_readiness_missing_task(self, capsys):
        args = self._args(task=None, mock="normal", robot_id="g1", json=False)
        assert cmd_sense_readiness(args) == 1

    def test_cmd_sense_explain(self, capsys):
        args = self._args(task="kick_ball", mock="hot_knee", robot_id="g1")
        assert cmd_sense_explain(args) == 0
        out = capsys.readouterr().out
        assert "kick_ball" in out
