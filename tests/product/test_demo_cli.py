"""Black-box product CLI tests for the first verified receipt journey."""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlparse

import pytest

from rosclaw.product.cli import dispatch_product_argv
from rosclaw.product.demo import DemoConfigurationError, run_demo


def _artifact_paths(receipt: dict[str, object]) -> list[Path]:
    artifacts = receipt.get("artifacts")
    assert isinstance(artifacts, list)
    return [
        Path(urlparse(uri).path)
        for uri in artifacts
        if isinstance(uri, str) and urlparse(uri).scheme == "file"
    ]


def test_demo_catalog_and_capability_status_are_machine_readable(capsys) -> None:
    assert dispatch_product_argv(["demo", "list", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["demos"][0]["id"] == "ur5e-reach"
    assert catalog["demos"][0]["mode"] == "SIMULATION"

    assert dispatch_product_argv(["status", "capabilities", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["release"]["version"] == "1.0.1"
    assert status["golden_paths"]["ur5e_reach"]["dimensions"]["simulation"] == "verified"
    assert status["golden_paths"]["rh56_single_step"]["modes"]["real"] == "developer_observed"


def test_first_verified_receipt_and_explain_latest(tmp_path, capsys) -> None:
    home = str(tmp_path / "home")

    assert dispatch_product_argv(["demo", "run", "ur5e-reach", "--home", home, "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema_version"] == "rosclaw.receipt.v1"
    assert receipt["execution_mode"] == "SIMULATION"
    assert receipt["final_state"] == "COMPLETED"
    assert receipt["evidence_level"] == "TASK_VERIFIED"
    assert receipt["verified"] is True
    assert receipt["simulation_result"]["has_physics"] is True
    assert receipt["simulation_result"]["steps"] > 0
    assert receipt["verification_result"]["final_error_m"] <= 0.008

    artifact_paths = _artifact_paths(receipt)
    assert {path.name for path in artifact_paths} >= {
        "live.jsonl",
        "receipt.json",
        "trajectory.json",
    }
    assert all(path.is_file() for path in artifact_paths)

    assert dispatch_product_argv(["explain", "latest", "--home", home, "--json"]) == 0
    explanation = json.loads(capsys.readouterr().out)
    assert explanation["run_id"] == receipt["action_id"]
    assert explanation["policy"]["allowed"] is True
    assert explanation["execution"]["physics_executed"] is True
    assert explanation["observation"]["collision_free"] is True
    assert explanation["verification"]["task_verified"] is True
    assert len(explanation["rosclaw_contribution"]) >= 4


def test_out_of_bounds_demo_is_blocked_before_physics_and_explainable(
    tmp_path,
    capsys,
) -> None:
    home = str(tmp_path / "home")

    assert (
        dispatch_product_argv(
            [
                "demo",
                "run",
                "ur5e-reach",
                "--home",
                home,
                "--target",
                "2.0",
                "0.0",
                "0.5",
                "--json",
            ]
        )
        == 1
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["final_state"] == "BLOCKED"
    assert receipt["verified"] is False
    assert receipt["simulation_result"] is None
    assert receipt["policy_decision"]["reason"] == "target_outside_workspace"

    assert dispatch_product_argv(["explain", "latest", "--home", home, "--json"]) == 0
    explanation = json.loads(capsys.readouterr().out)
    assert explanation["policy"]["allowed"] is False
    assert explanation["execution"]["physics_executed"] is False
    assert explanation["verification"]["task_verified"] is False


def test_explain_missing_run_is_clear_and_nonzero(tmp_path, capsys) -> None:
    assert dispatch_product_argv(["explain", "latest", "--home", str(tmp_path), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "NOT_FOUND"
    assert "demo run ur5e-reach" in result["error"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_steps": 0},
        {"max_steps": 5001},
        {"tolerance_m": 0.0},
        {"tolerance_m": math.nan},
        {"target": (math.nan, 0.0, 0.5)},
    ],
)
def test_invalid_demo_configuration_creates_no_run(tmp_path, parameters) -> None:
    with pytest.raises(DemoConfigurationError):
        run_demo("ur5e-reach", home=tmp_path, **parameters)

    assert not (tmp_path / "runs").exists()
