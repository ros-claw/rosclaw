"""Unrelated CLI routes must not resolve optional RH56 assets or transport."""

import sys
from pathlib import Path

import pytest

from rosclaw import cli
from rosclaw.body.rh56 import resources


def test_version_route_does_not_lookup_optional_hardware_resources(monkeypatch, tmp_path):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", ["rosclaw", "version"])

    def missing(*args):
        raise FileNotFoundError("optional RH56 assets intentionally unavailable")

    monkeypatch.setattr(resources, "rh56_reference_policy_path", missing)
    monkeypatch.setattr(resources, "rh56_config_path", missing)
    assert cli.main() == 0


def test_selected_rh56_route_resolves_same_defaults_only_at_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", ["rosclaw", "lerobot", "rollout", "rh56-shadow"])
    calls, dispatched = [], []

    def policy():
        calls.append("policy")
        return Path("/synthetic/rh56/policy")

    def config(name):
        calls.append(name)
        return Path("/synthetic/rh56") / name

    monkeypatch.setattr(resources, "rh56_reference_policy_path", policy)
    monkeypatch.setattr(resources, "rh56_config_path", config)
    monkeypatch.setattr(
        cli, "_dispatch_lerobot_cli", lambda name, args: dispatched.append((name, args)) or 0
    )
    assert cli.main() == 0
    assert calls == ["policy", "rh56_right_rs485_v1.yaml", "rh56_right_01_calibration.yaml"]
    name, args = dispatched[0]
    assert name == "cmd_lerobot_rollout_rh56_shadow"
    assert args.policy_path == "/synthetic/rh56/policy"
    assert args.transport_profile == "/synthetic/rh56/rh56_right_rs485_v1.yaml"
    assert args.calibration == "/synthetic/rh56/rh56_right_01_calibration.yaml"


def test_explicit_rh56_paths_do_not_depend_on_bundled_resource_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rosclaw",
            "lerobot",
            "rollout",
            "rh56-shadow",
            "--policy.path",
            "/operator/policy",
            "--transport-profile",
            "/operator/transport.yaml",
            "--calibration",
            "/operator/calibration.yaml",
        ],
    )

    def missing(*args):
        pytest.fail("explicit RH56 paths must avoid default resource lookup")

    monkeypatch.setattr(resources, "rh56_reference_policy_path", missing)
    monkeypatch.setattr(resources, "rh56_config_path", missing)
    monkeypatch.setattr(cli, "_dispatch_lerobot_cli", lambda name, args: 0)
    assert cli.main() == 0
