"""Native CLI integration probes; saved data only, with external artifact checks.

These call real ROSClaw graph/compiler/Practice interfaces. No network transport,
ROS node, VLA inference, sensor hardware, or physical-control implementation.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path
from typing import Any

from benchmarks.harnessbench.capability_matrix import _equal
from benchmarks.harnessbench.task_common import BenchTask

REPO = Path(__file__).resolve().parents[2]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _graph(**parts: Any) -> dict:
    return {
        "ros_version": "ros2",
        "distro": "jazzy",
        "endpoint": "fixture://offline",
        "topics": [],
        "services": [],
        "actions": [],
        "nodes": [],
        "params": [],
        "captured_at": "2026-10-03T00:00:00Z",
        **parts,
    }


# Private semantic projections are never staged in native workspaces.
GRAPH_CASES = [
    (
        "command_guard_compilation",
        _graph(
            topics=[
                {"name": "/g1/cmd_vel", "msg_type": "geometry_msgs/msg/Twist", "is_command": True}
            ]
        ),
        {
            "matrix.base.velocity_command": {
                "name": "/g1/cmd_vel",
                "kind": "actuation",
                "risk": "high",
                "read_only": False,
                "requires_runtime_guard": True,
                "requires_stop_guard": True,
            }
        },
    ),
    (
        "sensor_artifact_contract",
        _graph(
            topics=[
                {"name": "/camera/image", "msg_type": "sensor_msgs/msg/Image", "is_sensor": True},
                {"name": "/laser/scan", "msg_type": "sensor_msgs/msg/LaserScan", "is_sensor": True},
            ]
        ),
        {
            "matrix.observe.camera.rgb": {
                "name": "/camera/image",
                "kind": "observation",
                "risk": "low",
                "read_only": True,
                "artifact_type": "image",
            },
            "matrix.observe.laser_scan": {
                "name": "/laser/scan",
                "kind": "observation",
                "risk": "low",
                "read_only": True,
                "artifact_type": "message",
            },
        },
    ),
    (
        "service_sandbox_contract",
        _graph(
            services=[
                {
                    "name": "/planner/clear",
                    "srv_type": "std_srvs/srv/Empty",
                    "risk_hint": "high",
                    "request_schema": {"type": "object"},
                    "response_schema": {"type": "object"},
                }
            ]
        ),
        {
            "matrix.planner.clear": {
                "name": "/planner/clear",
                "kind": "actuation",
                "risk": "high",
                "read_only": False,
                "requires_runtime_guard": True,
            }
        },
    ),
    (
        "action_guard_compilation",
        _graph(
            actions=[
                {
                    "name": "/navigate",
                    "action_type": "nav2_msgs/action/NavigateToPose",
                    "risk_hint": "medium",
                    "goal_schema": {"type": "object"},
                }
            ]
        ),
        {
            "matrix.action.navigate": {
                "name": "/navigate",
                "kind": "actuation",
                "risk": "medium",
                "read_only": False,
                "requires_runtime_guard": True,
            }
        },
    ),
    (
        "namespace_collision_disambiguation",
        _graph(
            topics=[
                {"name": "/a/cmd_vel", "msg_type": "geometry_msgs/msg/Twist", "is_command": True},
                {"name": "/b/cmd_vel", "msg_type": "geometry_msgs/msg/Twist", "is_command": True},
            ]
        ),
        {
            "matrix.base.velocity_command.a": {
                "name": "/a/cmd_vel",
                "kind": "actuation",
                "risk": "high",
                "read_only": False,
            },
            "matrix.base.velocity_command.b": {
                "name": "/b/cmd_vel",
                "kind": "actuation",
                "risk": "high",
                "read_only": False,
            },
        },
    ),
    (
        "ros1_saved_graph_preservation",
        _graph(
            ros_version="ros1",
            distro="noetic",
            topics=[
                {
                    "name": "/robot/joint_states",
                    "msg_type": "sensor_msgs/JointState",
                    "is_sensor": True,
                }
            ],
        ),
        {
            "matrix.observe.joint_states": {
                "name": "/robot/joint_states",
                "kind": "observation",
                "risk": "low",
                "read_only": True,
            }
        },
    ),
]
INTEGRATION_TASKS: dict[str, BenchTask] = {}
PRIVATE: dict[str, dict] = {}
_COMMON = (
    "This is offline native ROSClaw interface integration; evidence scope FIXTURE_ONLY. "
    "Use the full command entrypoint 'rosclaw ros compile' (not 'ros compile'); the installed "
    "ROSClaw launcher is available on PATH. A pinned .venv/bin/python -m rosclaw.entrypoint "
    "is equivalent. Use CLI help if needed. "
    "Do not construct Runtime, ROS/DDS transport, publish topics, register drivers, install packages, "
    "or access hardware. Preserve the supplied input file. Save full CLI JSON stdout to the named "
    "artifacts without fabricating or rewriting fields. Then write answer.json with "
    '{"scope":"FIXTURE_ONLY","reason":"<explain actual interface calls and limits>"}. '
    "The external grader independently inspects CLI artifacts and native session tool calls."
)
for _index, (_direction, _input, _expected) in enumerate(GRAPH_CASES, 1):
    _id = f"CI{_index:02d}"
    INTEGRATION_TASKS[_id] = BenchTask(
        task_id=_id,
        category="offline_integration",
        staged_files={"graph.json": _json(_input)},
        oracle={"kind": "offline_cli_integration", "direction": _direction},
        prompt=_COMMON + "\nUse 'ros compile' on graph.json, robot-id matrix, write manifest.json "
        "and save compile stdout to compile.json. Use 'ros list-capabilities' on that saved manifest "
        "and save stdout to list.json. Inspect each compiled capability using 'ros inspect-capability' "
        "against the saved manifest; save stdout files under inspections/<capability-id>.json. "
        "For command capabilities, also validate an offline proposal with linear.x=999 and duration_sec=0.1 using "
        "'ros validate-capability', save validate.json; report its actual decision without executing. "
        "Do not infer live ROS1/ROS2 discovery from a saved graph.",
    )
    PRIVATE[_id] = {"kind": "graph", "expected": _expected}

for _index, (_direction, _files, _command, _output) in enumerate(
    [
        (
            "invalid_graph_no_discovery",
            {"graph.json": "{}\n"},
            "ros compile --graph graph.json --robot-id matrix --json",
            "failure.json",
        ),
        (
            "missing_manifest_no_discovery",
            {"input.json": '{"case":"missing offline manifest"}\n'},
            "ros list-capabilities --manifest missing.json --json",
            "failure.json",
        ),
    ],
    7,
):
    _id = f"CI{_index:02d}"
    INTEGRATION_TASKS[_id] = BenchTask(
        task_id=_id,
        category="offline_integration",
        staged_files=_files,
        oracle={"kind": "offline_cli_integration", "direction": _direction},
        prompt=_COMMON
        + f"\nRun the actual CLI command '{_command}', save JSON stdout to {_output}, "
        "and save its numeric exit status to exitcode.txt. A missing/invalid offline file must not "
        "trigger live discovery. Do not create missing.json or repair graph.json.",
    )
    PRIVATE[_id] = {"kind": "negative"}

_BASE_FIXTURE = json.loads((REPO / "tests/fixtures/practice/rh56_minimal_loop.json").read_text())
for _index, _direction in enumerate(
    [
        "practice_record_strict_evidence",
        "practice_ingest_missing_envelope",
        "practice_raw_envelope_tamper",
        "practice_failed_outcome_preservation",
    ],
    9,
):
    _id = f"CI{_index:02d}"
    _fixture = copy.deepcopy(_BASE_FIXTURE)
    if _index == 10:
        del _fixture["events"][0]["event_id"]
    if _index == 12:
        _fixture["outcome"] = "FAILURE"
        _fixture["reward"] = -1.0
        _fixture["failure_labels"] = ["fixture_task_failed"]
    _prompt = (
        _COMMON
        + "\nUse 'practice record --fixture fixture.json --out practice-data --json', save stdout record.json and exit status record_exitcode.txt. "
    )
    if _index == 10:
        _prompt += "The supplied envelope is invalid; retain the failure without repairing the input or claiming recording success."
    else:
        if _index == 11:
            _prompt += "After recording, remove timestamp_ns from the FIRST physical_feedback_event top-level event envelope in raw/events.jsonl (timestamp_ns is outside its nested payload object) (only this deliberate corruption is authorized), preserving all other recorded files. "
        _prompt += "Use 'practice verify practice_rh56_minimal_loop --data-root practice-data --strict --json', save stdout verify.json and exit status verify_exitcode.txt. Preserve the original fixture."
    INTEGRATION_TASKS[_id] = BenchTask(
        task_id=_id,
        category="offline_integration",
        staged_files={"fixture.json": _json(_fixture)},
        oracle={"kind": "offline_cli_integration", "direction": _direction},
        prompt=_prompt,
    )
    PRIVATE[_id] = {
        "kind": "practice",
        "expected_pass": _index not in {10, 11},
        "fixture": _fixture,
    }


def _native_commands(root: Path) -> list[str]:
    commands = []
    # B leg has durable native session transcripts. A leg has no native harness
    # integration, so these probes intentionally remain unsupported on A.
    for session in (root / "rh/agent/sessions").glob("**/*.jsonl"):
        for line in session.read_text().splitlines():
            entry = json.loads(line)
            for part in entry.get("message", {}).get("content", []):
                if part.get("type") == "toolCall" and part.get("name") == "bash":
                    command = part.get("arguments", {}).get("command")
                    if isinstance(command, str):
                        commands.append(command)
    return commands


def _has_cli(commands: list[str], operation: str) -> bool:
    """Recognize a CLI argv, including plain shell path-variable assignments.

    This is evidence of a native tool call, not a tamper-proof execution receipt.
    Never execute shell text or accept echo/print/write mentioning a CLI.
    """
    wanted = shlex.split(operation)
    for command in commands:
        variables: dict[str, str] = {}
        for statement in re.split(r"[;\n]|&&|\|\|", command):
            try:
                words = shlex.split(statement.strip())
            except ValueError:
                continue
            if not words:
                continue
            while words and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", words[0]):
                key, value = words.pop(0).split("=", 1)
                variables[key] = value
            if not words:
                continue
            program = words[0]
            for _ in range(4):
                expanded = re.sub(
                    r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)",
                    lambda m, bound=variables: bound.get(m.group(1) or m.group(2), m.group(0)),
                    program,
                )
                if expanded == program:
                    break
                program = expanded
            # Simulation CLI's --root precedes its subcommand.
            if len(words) >= 4 and words[1:3] == ["sim", "--root"]:
                words = words[:2] + words[4:]
            if len(words) >= 6 and words[1:5] == ["-m", "rosclaw.entrypoint", "sim", "--root"]:
                words = words[:4] + words[6:]
            basename = Path(program).name
            if basename == "rosclaw" and words[1 : 1 + len(wanted)] == wanted:
                return True
            if (
                re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)?)?", basename)
                and words[1:3] == ["-m", "rosclaw.entrypoint"]
                and words[3 : 3 + len(wanted)] == wanted
            ):
                return True
    return False


def judge_integration(task_id: str, root: Path) -> dict:
    task, private = INTEGRATION_TASKS[task_id], PRIVATE[task_id]
    checks: dict[str, bool] = {}
    try:
        checks["immutable_inputs"] = all(
            (root / name).read_text() == value for name, value in task.staged_files.items()
        )
        answer = json.loads((root / "answer.json").read_text())
        checks["scope"] = answer.get("scope") == "FIXTURE_ONLY"
        checks["reason"] = isinstance(answer.get("reason"), str) and bool(answer["reason"].strip())
        commands = _native_commands(root)
        if private["kind"] == "graph":
            checks["native_compile"] = _has_cli(commands, "ros compile")
            checks["native_list"] = _has_cli(commands, "ros list-capabilities")
            checks["native_inspect"] = _has_cli(commands, "ros inspect-capability")
            manifest = json.loads((root / "manifest.json").read_text())
            compile_result = json.loads((root / "compile.json").read_text())
            list_result = json.loads((root / "list.json").read_text())
            checks["compile_output_binding"] = (
                compile_result.get("ok") is True and compile_result.get("manifest") == manifest
            )
            input_graph = json.loads(task.staged_files["graph.json"])
            checks["ros_scope"] = (
                manifest.get("ros", {}).get("version") == input_graph["ros_version"]
                and manifest.get("robot_id") == "matrix"
            )
            caps = {cap["id"]: cap for cap in manifest["capabilities"]}
            expected = private["expected"]
            checks["unique_exact_capabilities"] = len(caps) == len(
                manifest["capabilities"]
            ) and set(caps) == set(expected)
            checks["list_binding"] = list_result.get("ok") is True and {
                cap["id"] for cap in list_result["capabilities"]
            } == set(caps)
            for cap_id, projection in expected.items():
                cap = caps[cap_id]
                actual = {
                    "name": cap["interface"]["name"],
                    "kind": cap["kind"],
                    "risk": cap["risk"]["level"],
                    "read_only": cap["risk"]["read_only"],
                }
                for key in ("requires_runtime_guard", "requires_stop_guard"):
                    if key in projection:
                        actual[key] = cap["risk"][key]
                if "artifact_type" in projection:
                    actual["artifact_type"] = cap["practice"]["artifact_type"]
                checks[f"semantic_{cap_id}"] = _equal(actual, projection)
                inspection = json.loads((root / "inspections" / f"{cap_id}.json").read_text())
                checks[f"inspect_{cap_id}"] = (
                    inspection.get("ok") is True and inspection.get("capability") == cap
                )
            if any("velocity_command" in cap_id for cap_id in caps):
                validation = json.loads((root / "validate.json").read_text())
                checks["native_validate"] = _has_cli(commands, "ros validate-capability")
                checks["unsafe_proposal_not_allowed"] = (
                    validation.get("decision") in {"MODIFY", "BLOCK"}
                    and validation.get("decision") != "ALLOW"
                )
                checks["velocity_limit_detected"] = any(
                    "linear.x" in item for item in validation.get("violated_constraints", [])
                )
        elif private["kind"] == "negative":
            failure = json.loads((root / "failure.json").read_text())
            checks["negative_cli"] = _has_cli(
                commands, "ros compile" if task_id == "CI07" else "ros list-capabilities"
            )
            checks["negative_exit"] = (root / "exitcode.txt").read_text().strip() == "1"
            checks["negative_semantics"] = failure.get("ok") is False and bool(failure.get("error"))
        else:
            checks["native_record"] = _has_cli(commands, "practice record")
            rc = (root / "record_exitcode.txt").read_text().strip()
            if task_id == "CI10":
                checks["ingest_rejected"] = rc == "1"
                checks["no_recorded_success"] = not (
                    root / "practice-data/sessions/practice_rh56_minimal_loop/raw/events.jsonl"
                ).exists()
            else:
                checks["recorded"] = rc == "0"
                checks["native_verify"] = _has_cli(commands, "practice verify")
                verify = json.loads((root / "verify.json").read_text())
                checks["verified_verdict"] = (
                    verify.get("passed") is private["expected_pass"]
                    and verify.get("strict") is True
                )
                checks["verify_exit"] = (root / "verify_exitcode.txt").read_text().strip() == (
                    "0" if private["expected_pass"] else "1"
                )
                events_path = (
                    root / "practice-data/sessions/practice_rh56_minimal_loop/raw/events.jsonl"
                )
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                # Inspect the actual durable event envelopes rather than self-reported counts.
                fixture = private["fixture"]
                expected_events = copy.deepcopy(fixture["events"])
                if task_id == "CI11":
                    for expected_event in expected_events:
                        if expected_event["event_type"] == "physical_feedback_event":
                            del expected_event["timestamp_ns"]
                            break
                checks["raw_record_count"] = len(events) == len(expected_events)
                checks["raw_original_event_semantics"] = len(events) == len(
                    expected_events
                ) and all(
                    _equal({key: actual_event.get(key) for key in expected_event}, expected_event)
                    for actual_event, expected_event in zip(events, expected_events, strict=True)
                )
                checks["raw_sequence_order"] = [e.get("sequence_id") for e in events] == list(
                    range(1, len(events) + 1)
                )
                episode = json.loads((events_path.parent.parent / "episode.json").read_text())
                import yaml

                manifest = yaml.safe_load((events_path.parent.parent / "manifest.yaml").read_text())
                summary_path = (
                    root
                    / "practice-data/sessions"
                    / fixture["session_id"]
                    / "episodes"
                    / fixture["episode_id"]
                    / "artifacts/summary"
                    / f"summary_{fixture['episode_id']}.yaml"
                )
                summary = yaml.safe_load(summary_path.read_text())
                for label, metadata in (
                    ("episode", episode),
                    ("manifest", manifest["status"]),
                    ("summary", summary),
                ):
                    checks[f"{label}_outcome_preserved"] = (
                        str(metadata.get("outcome", "")).upper() == fixture["outcome"]
                    )
                    reward = (
                        metadata.get("reward")
                        if label != "summary"
                        else metadata.get("metrics", {}).get("reward")
                    )
                    checks[f"{label}_reward_preserved"] = _equal(reward, fixture["reward"])
                    checks[f"{label}_failure_labels_preserved"] = (
                        metadata.get("failure_labels") == fixture["failure_labels"]
                    )
                checks["body_task_session_preserved"] = (
                    manifest.get("body_id") == fixture["body_id"]
                    and manifest.get("session_id") == fixture["session_id"]
                    and manifest.get("task", {}).get("task_id") == fixture["task_id"]
                    and episode.get("event_count") == len(expected_events)
                )
                if task_id == "CI11":
                    checks["corruption_retained"] = any(
                        e.get("event_type") == "physical_feedback_event" and "timestamp_ns" not in e
                        for e in events
                    )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        checks["parse_or_evidence"] = False
    passed = bool(checks) and all(checks.values())
    return {
        "task_success": passed,
        "verified_success": passed,
        "false_success": False,
        "reason": "native_offline_interface_verified"
        if passed
        else "native_interface_evidence_failed",
        "checks": checks,
        "evidence_scope": "FIXTURE_ONLY",
        "live_integration": "NOT_RUN",
        "direction": task.oracle["direction"],
    }
