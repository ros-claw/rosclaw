"""Malformed proposals must not bypass numeric motion constraints (no transport)."""

import math

import pytest

from rosclaw.connectors.ros.compiler.safety_contract import (
    SafetyContract,
    SafetyContractCompiler,
    SafetyLevel,
    SafetyRule,
)


def contract(*, level=SafetyLevel.HIGH_RISK, stop=True):
    return SafetyContract(
        rules={
            "motion": SafetyRule(
                capability_id="motion",
                level=level,
                read_only=False,
                destructive=False,
                requires_sandbox=True,
                requires_runtime_guard=True,
                requires_stop_guard=stop,
                max_duration_sec=1.0,
                constraints={"linear.x": [-0.2, 0.2]},
            )
        }
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"linear.x": 999},
        {"linear": {"x": 10**1000}},
        {"linear": 999},
        {"linear": None},
        {"linear": {"x": None}},
        {"linear": {"x": "999"}},
        {"linear": {"x": True}},
        {"linear": {"x": math.nan}},
        {"linear": {"x": math.inf}},
        {"linear": {"x": -math.inf}},
    ],
)
def test_malformed_constraint_arguments_fail_closed(bad):
    args = {**bad, "duration_sec": 0.1}
    decision = SafetyContractCompiler().evaluate(contract(), "motion", args)
    assert decision.decision == "BLOCK"
    assert decision.violated_constraints
    assert decision.reason.startswith("Invalid argument shape")


@pytest.mark.parametrize("duration", [True, None, "0.1", -0.1, math.nan, math.inf])
def test_invalid_duration_cannot_be_allow(duration):
    decision = SafetyContractCompiler().evaluate(
        contract(), "motion", {"linear": {"x": 0.1}, "duration_sec": duration}
    )
    assert decision.decision == "BLOCK"


def test_shadowed_invalid_duration_and_conflicting_aliases_block():
    compiler = SafetyContractCompiler()
    for args in [
        {"duration": 0.1, "duration_sec": math.nan},
        {"duration": 0.1, "duration_sec": 100},
    ]:
        assert compiler.evaluate(contract(), "motion", args).decision == "BLOCK"


@pytest.mark.parametrize(
    "args",
    [
        {"linear": {"x": 0.1}, "duration_sec": 0.1},
        {"linear": {"x": 0}, "duration_sec": 0},
        {"duration_sec": 0.1},  # omitted fields remain default, not malformed fields
    ],
)
def test_legitimate_nested_partial_and_stop_contracts_remain_allowed(args):
    assert SafetyContractCompiler().evaluate(contract(), "motion", args).decision == "ALLOW"


def test_custom_service_constraint_and_low_risk_clamping_remain_valid():
    service = contract(level=SafetyLevel.MEDIUM_RISK, stop=False)
    service.rules["motion"].constraints = {"custom_speed": [0, 1]}
    compiler = SafetyContractCompiler()
    assert compiler.evaluate(service, "motion", {"custom_speed": 0.5}).decision == "ALLOW"
    result = compiler.evaluate(service, "motion", {"custom_speed": 3})
    assert result.decision == "MODIFY"
    assert result.modified_args == {"custom_speed": 1}
    assert compiler.evaluate(service, "motion", {"custom_speed": "3"}).decision == "BLOCK"


def test_actual_cli_saved_manifest_flat_proposal_blocked(tmp_path):
    import json
    import os
    import subprocess
    import sys

    from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
    from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot, RosTopicInfo

    graph = RosGraphSnapshot(
        ros_version="ros2",
        distro="jazzy",
        endpoint="offline://fixture",
        services=[],
        actions=[],
        nodes=[],
        params=[],
        captured_at="fixture",
        topics=[
            RosTopicInfo(
                name="/fleet/robot_b/cmd_vel",
                msg_type="geometry_msgs/msg/Twist",
                is_command=True,
                risk_hint="high",
            )
        ],
    )
    manifest = CapabilityManifestCompiler(robot_id="fixture").compile(graph)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest.to_dict()))
    cap = manifest.capabilities[0]
    for proposal, expected in [
        ({"linear.x": 999, "duration_sec": 0.1}, "BLOCK"),
        ({"linear": {"x": 0}, "duration_sec": 0}, "ALLOW"),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rosclaw.entrypoint",
                "ros",
                "validate-capability",
                "--manifest",
                str(manifest_file),
                "--json",
                cap.id,
                "--args",
                json.dumps(proposal),
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(
                    __import__("pathlib").Path(__file__).resolve().parents[3] / "src"
                ),
            },
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == (1 if expected == "BLOCK" else 0), result.stderr
        body = json.loads(result.stdout)
        assert body["decision"] == expected
