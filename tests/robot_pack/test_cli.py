from __future__ import annotations

import json
from pathlib import Path

from rosclaw.robot_pack.cli import dispatch_robot_pack_argv


def test_fast_path_add_and_contract_verify(tmp_path: Path, capsys) -> None:
    home = tmp_path / "home"

    add_code = dispatch_robot_pack_argv(
        ["robot", "add", "realsense", "--home", str(home), "--json"]
    )
    add_payload = json.loads(capsys.readouterr().out)
    verify_code = dispatch_robot_pack_argv(
        [
            "robot",
            "verify",
            "realsense",
            "--stage",
            "contract",
            "--home",
            str(home),
            "--json",
        ]
    )
    verify_payload = json.loads(capsys.readouterr().out)

    assert add_code == 0
    assert add_payload["signature_status"] == "valid"
    assert verify_code == 0
    assert verify_payload["passed"] is True
    assert verify_payload["support_tier"] == "H1_CONTRACT_VERIFIED"


def test_offline_configuration_does_not_pass_read_only_verification(
    tmp_path: Path,
    capsys,
) -> None:
    home = tmp_path / "home"
    assert (
        dispatch_robot_pack_argv(["robot", "add", "realsense", "--home", str(home), "--json"]) == 0
    )
    capsys.readouterr()
    assert (
        dispatch_robot_pack_argv(
            [
                "robot",
                "configure",
                "realsense",
                "--home",
                str(home),
                "--instance",
                "offline-camera",
                "--model",
                "D405",
                "--serial",
                "OFFLINE",
                "--allow-offline",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    code = dispatch_robot_pack_argv(
        [
            "robot",
            "verify",
            "offline-camera",
            "--stage",
            "read-only",
            "--home",
            str(home),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 3
    assert payload["passed"] is False
    assert payload["support_tier"] == "H1_CONTRACT_VERIFIED"
    assert payload["observed_candidate_tier"] is None
