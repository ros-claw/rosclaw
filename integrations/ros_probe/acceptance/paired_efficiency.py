"""Isolated seeded SIM pairs through the existing MCP/daemon journey."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lifecycle_readiness import readiness
from observations import latest_completed_observation
from profiles import PROFILES

from experiments import (
    BOUNDARY_TRACKING_INSET_PRESETS,
    BOUNDARY_TRACKING_PRESETS,
    CONTINUOUS_BOUNDARY_PRESETS,
    INNER_RING_PROFILES,
    REPAIR_TRACKING_STRATEGY,
    continuous_boundary_strategy,
    planning_parameters,
    valid_boundary_tracking_sha256,
    validate_boundary_segment_registration,
    validate_repair_tracking_candidate_registration,
    validate_seed,
)

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]


def validate_precise_waypoint_candidate(preset, repair_strategy, enabled):
    # The same Nav2 through-poses BT serves the new boundary route, even when
    # repair uses single NavigateToPose goals. Older candidates retain their gate.
    if (
        enabled
        and repair_strategy != "pose_aware_robust_sequence"
        and preset not in CONTINUOUS_BOUNDARY_PRESETS
    ):
        raise ValueError(
            "precise waypoint BT requires continuous repair or registered continuous boundary"
        )


def validate_continuous_boundary_registration(protocol, preset, precise):
    validate_boundary_segment_registration(protocol, preset, prefix="candidate_")
    tracking = isinstance(preset, str) and preset in BOUNDARY_TRACKING_PRESETS
    inset = isinstance(preset, str) and preset in BOUNDARY_TRACKING_INSET_PRESETS
    if "candidate_boundary_corner_inset_cells" in protocol and not inset:
        raise ValueError("corner inset metadata requires its registered candidate")
    if inset and (
        type(protocol.get("candidate_boundary_corner_inset_cells")) is not int
        or protocol["candidate_boundary_corner_inset_cells"] != 1
    ):
        raise ValueError("corner inset requires exactly one existing legal grid cell")
    tracking_keys = (
        "candidate_boundary_tracking_prune_radius_m",
        "candidate_boundary_tracking_bt_sha256",
    )
    if any(k in protocol for k in tracking_keys) and not tracking:
        raise ValueError("boundary tracking metadata requires its registered candidate")
    if not isinstance(preset, str) or preset not in CONTINUOUS_BOUNDARY_PRESETS:
        return
    if (
        precise is not True
        or protocol.get("candidate_boundary_strategy") != continuous_boundary_strategy(preset)
        or type(protocol.get("candidate_boundary_stage_budget_sec")) is not int
        or protocol["candidate_boundary_stage_budget_sec"] != 180
        or type(protocol.get("candidate_boundary_waypoint_count")) is not int
        or protocol["candidate_boundary_waypoint_count"] != 9
    ):
        raise ValueError("continuous boundary protocol requires nine precise bounded waypoints")
    if tracking and (
        type(protocol.get("candidate_boundary_tracking_prune_radius_m")) not in (int, float)
        or protocol["candidate_boundary_tracking_prune_radius_m"] != 0.1
        or not valid_boundary_tracking_sha256(protocol.get("candidate_boundary_tracking_bt_sha256"))
    ):
        raise ValueError(
            "boundary tracking requires registered 100mm checkpoints and source SHA256"
        )


def validate_precise_repair_registration(protocol, enabled):
    registered = protocol.get("precise_repair_waypoints", False)
    if type(registered) is not bool or registered != enabled:
        raise ValueError("precise waypoint BT differs from preregistered protocol")


def validate_inner_ring_registration(protocol, candidate):
    if candidate not in INNER_RING_PROFILES:
        return
    expected = {
        "candidate_boundary_stage_budget_sec": 360,
        "candidate_inner_boundary_inset_cells": 1,
    }
    if any(
        type(protocol.get(key)) is not int or protocol[key] != value
        for key, value in expected.items()
    ):
        raise ValueError(
            "inner ring requires preregistered stage budget and inset before World launch"
        )


def repair_request_counts(directory, *, repair_tracking=False):
    """Retain requested goals and waypoints separately; neither proves arrival."""
    from rosclaw.connectors.ros.diagnosis.coverage_audit import read_audit

    if type(repair_tracking) is not bool:
        raise ValueError("repair tracking count mode must be a boolean")
    goals, targets, identities = 0, 0, set()
    for path in sorted((directory / "actions").glob("coverage-audit-*.jsonl")):
        for row in read_audit(path):
            if row["kind"] != "goal_started" or row["payload"].get("stage") != "REPAIR":
                continue
            payload = row["payload"]
            identity = (row["run_id"], payload["nav_goal_id"])
            if identity in identities:
                raise ValueError("duplicate original repair goal identity")
            identities.add(identity)
            args = payload["goal"]
            if set(args) == {"pose"}:
                count = 1
            elif (
                set(args) == {"poses"}
                and type(args["poses"]) is list
                and 1 <= len(args["poses"]) <= 2
            ):
                count = len(args["poses"])
            elif (
                repair_tracking
                and set(args) == {"poses", "behavior_tree"}
                and type(args["poses"]) is list
                and len(args["poses"]) == 2
                and args["behavior_tree"] == "/evidence/repair-tracking-through-poses.xml"
            ):
                count = 2
            else:
                raise ValueError("closed one/two waypoint repair request required")
            goals += 1
            targets += count
    return {"repair_requested_goal_count": goals, "repair_requested_waypoint_count": targets}


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def wait_ready(directory, container, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if command(["docker", "inspect", "--format", "{{.State.Running}}", container]) != "true":
            raise RuntimeError("owned SIM stack exited during startup")
        try:
            sample = latest_completed_observation(directory / "witness.jsonl")
            age = (
                datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])
            ).total_seconds()
            if (
                0 <= age < 0.5
                and sample["observation_complete"]
                and sample["collision_count"] == 0
                and not sample["cleaning_enabled"]
                and (directory / "measured_map.json").exists()
                and readiness(json.loads((directory / "lifecycle-readiness.json").read_text()))
                and all(
                    "Managed nodes are active" in (directory / name).read_text()
                    for name in ["nav2.log", "coverage_lifecycle.log"]
                )
            ):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.5)
    from startup_gate_failure import retain_startup_failure

    failure = retain_startup_failure(directory, require_live_lifecycle=True)
    raise TimeoutError(
        "SIM startup constraints not satisfied: " + ", ".join(failure["missing_requirements"])
    )


def journey(directory, port, profile, timeout, repair_strategy="greedy"):
    env = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY / "src") + os.pathsep + os.getenv("PYTHONPATH", ""),
    }
    with (directory / "acceptance-run.log").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--directory",
                str(directory),
                "--endpoint",
                f"ws://127.0.0.1:{port}",
                "--profile",
                profile,
                "--mission-timeout",
                str(timeout),
                "--repair-strategy",
                repair_strategy,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout + 120)
            if code:
                raise RuntimeError(f"MCP/daemon journey exited {code}; see acceptance-run.log")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    with (directory / "supplement.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "cleaning_acceptance.py"),
                "--root",
                str(directory),
                "--fixture",
                str(directory),
                "--output",
                str(directory / "accepted"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
            timeout=60,
        )


def run_arm(pair, arm, args, ordinal, image_id, commit):
    directory = pair / arm
    directory.mkdir()
    (directory / "protocol.json").write_bytes((pair / "protocol.json").read_bytes())
    name = f"reh-n02-{args.profile}-{args.seed}-{arm}-{time.time_ns()}"
    port, domain = args.port_base + ordinal, args.domain_base + ordinal
    launched = False
    row = {
        "profile": args.profile,
        "seed": args.seed,
        "arm": arm,
        "preset": "baseline" if arm == "baseline" else args.candidate,
        "repair_strategy": "greedy"
        if arm == "baseline"
        else getattr(args, "candidate_repair_strategy", "greedy"),
        "precise_repair_waypoints": arm == "candidate"
        and getattr(args, "precise_repair_waypoints", False),
        "source_commit": commit,
        "mission_timeout_sec": args.mission_timeout,
        "image_id": image_id,
        "directory": str(directory),
        "status": "RUNNING",
        "port": port,
        "ros_domain_id": domain,
    }
    (directory / "run-result.json").write_text(json.dumps(row, indent=2) + "\n")
    try:
        stack = (
            "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
            "python3 /workspace/integrations/ros_probe/acceptance/stack.py "
            f"--controller-watchdog --profile {args.profile} --coverage-preset {row['preset']} "
            f"--seed {args.seed}"
        )
        if row["precise_repair_waypoints"]:
            stack += " --precise-repair-waypoints"
        if row["repair_strategy"] == REPAIR_TRACKING_STRATEGY:
            stack += " --repair-tracking-sequence"
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--gpus",
                "all",
                "-e",
                "NVIDIA_DRIVER_CAPABILITIES=all",
                "-e",
                f"ROS_DOMAIN_ID={domain}",
                "-e",
                "ROS_LOCALHOST_ONLY=1",
                "-p",
                f"127.0.0.1:{port}:9090",
                "-v",
                f"{REPOSITORY}:/workspace:ro",
                "-v",
                f"{directory}:/evidence",
                args.image,
                "bash",
                "-c",
                stack,
            ]
        )
        launched = True
        row["container"] = name
        wait_ready(directory, name)
        journey(directory, port, args.profile, args.mission_timeout, row["repair_strategy"])
    except KeyboardInterrupt:
        row.update(status="INTERRUPTED", failure="Interrupted before paired arm completed")
        raise
    except Exception as exc:
        row.update(status="FAIL", failure=f"{type(exc).__name__}: {exc}")
    finally:
        if launched:
            # Owned fixture shutdown invokes its established zero/flush cleanup.
            # Preserve the container and all failed evidence; never remove it.
            try:
                command(["docker", "stop", "-t", "45", name], timeout=60)
            except Exception as exc:
                row.update(status="FAIL", shutdown_failure=str(exc))
        if row["status"] == "RUNNING":
            try:
                with (directory / "audit.log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "coverage_audit.py"),
                            "--directory",
                            str(directory),
                            "--output",
                            str(directory / "audit"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
                    )
                audit = json.loads((directory / "audit/plan-versus-execution.json").read_text())
                accepted = json.loads((directory / "accepted/acceptance.json").read_text())
                segments = [
                    json.loads(line)
                    for line in (directory / "audit/coverage-segment-metrics.jsonl")
                    .read_text()
                    .splitlines()
                ]
                if (
                    not audit["audit_complete"]
                    or audit["canonical_verifier_replay_equal"] is not True
                ):
                    raise RuntimeError("diagnostic integrity or canonical replay gate failed")
                request_counts = repair_request_counts(
                    directory, repair_tracking=row["repair_strategy"] == REPAIR_TRACKING_STRATEGY
                )
                if request_counts["repair_requested_goal_count"] != sum(
                    s["stage"] == "REPAIR" for s in segments
                ):
                    raise RuntimeError("original repair requests and measured segments disagree")
                row.update(
                    status="PASS",
                    measured_distance_m=audit["total_metrics"]["observed_distance_m"],
                    sim_duration_sec=audit["total_metrics"]["sim_duration_sec"],
                    main_coverage_ratio=audit["main_observed_coverage_ratio"],
                    primary_coverage_ratio=audit.get("primary_before_repair_coverage_ratio"),
                    boundary_nav_goal_result=audit.get("boundary_nav_goal_result"),
                    coverage_ratio=accepted["coverage_ratio"],
                    canonical_duration_sec=accepted["duration_sec"],
                    collision_count=accepted["collision_count"],
                    trace_gaps=accepted["trace_gaps"],
                    post_cleanup_displacement_m=accepted["post_cleanup_displacement_m"],
                    post_cleanup_yaw_change_rad=accepted["post_cleanup_yaw_change_rad"],
                    run_id=audit["run_id"],
                    audit_complete=True,
                    complete_independent_observations=accepted["complete_independent_observations"],
                    repair_goal_count=sum(s["stage"] == "REPAIR" for s in segments),
                    main_nav_goal_result=audit.get("main_nav_goal_result"),
                    optimization_status="PILOT_OBSERVATION_ONLY",
                    **request_counts,
                )
            except Exception as exc:
                row.update(status="FAIL", failure=f"{type(exc).__name__}: {exc}")
        (directory / "run-result.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main():
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument(
        "--candidate",
        choices=[
            *CONTINUOUS_BOUNDARY_PRESETS,
            "baseline",
            "diagonal",
            "headland",
            "perimeter",
            "perimeter_sequential",
            "perimeter_stateless",
            "perimeter_stateless_headland",
            "perimeter_stateless_clearance",
            "perimeter_stateless_clearance_inner_ring",
            "perimeter_stateless_inner_ring",
            "perimeter_stateless_overlap",
        ],
        required=True,
    )
    parser.add_argument(
        "--candidate-repair-strategy",
        choices=[
            "greedy",
            "pose_aware",
            "pose_aware_robust",
            "pose_aware_robust_sequence",
            REPAIR_TRACKING_STRATEGY,
        ],
        default="greedy",
    )
    parser.add_argument("--precise-repair-waypoints", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=["pilot", "evaluation"], default="pilot")
    parser.add_argument("--image", default="rosclaw/ros-expert-rebuilt:dad31022")
    parser.add_argument("--port-base", type=int, default=20191)
    parser.add_argument("--domain-base", type=int, default=81)
    parser.add_argument("--mission-timeout", type=int, default=900)
    args = parser.parse_args()
    from fixture_network import validate_ros_domain

    validate_ros_domain(args.domain_base)
    validate_ros_domain(args.domain_base + 1)
    try:
        validate_precise_waypoint_candidate(
            args.candidate, args.candidate_repair_strategy, args.precise_repair_waypoints
        )
    except ValueError as exc:
        parser.error(str(exc))
    validate_seed(args.seed)
    planning_parameters(PROFILES[args.profile], args.candidate)
    if not 1024 <= args.port_base < 65535 or not 0 <= args.domain_base < 232:
        parser.error("pair requires two valid ports and two ROS domains")
    if not 60 <= args.mission_timeout <= 1800:
        parser.error("mission timeout must be between 60 and 1800 seconds")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_precise_repair_registration(protocol, args.precise_repair_waypoints)
    validate_repair_tracking_candidate_registration(
        protocol, args.candidate, args.candidate_repair_strategy, args.precise_repair_waypoints
    )
    validate_continuous_boundary_registration(
        protocol, args.candidate, args.precise_repair_waypoints
    )
    validate_inner_ring_registration(protocol, args.candidate)
    if args.seed not in protocol[f"{args.phase}_seeds"]:
        parser.error("seed is not preregistered for this phase")
    if args.phase == "evaluation" and not protocol.get("evaluation_freeze"):
        parser.error("evaluation requires a recorded post-pilot configuration freeze")
    if command(["git", "status", "--porcelain"], cwd=REPOSITORY):
        parser.error("physical paired runs require a clean committed worktree")
    commit = command(["git", "rev-parse", "HEAD"], cwd=REPOSITORY)
    if args.phase == "evaluation":
        freeze = protocol["evaluation_freeze"]
        if (
            freeze.get("precise_repair_waypoints", False) != args.precise_repair_waypoints
            or freeze["source_commit"] != commit
            or freeze["selected_presets"].get(args.profile) != args.candidate
            or freeze.get("selected_repair_strategies", {}).get(args.profile, "greedy")
            != args.candidate_repair_strategy
        ):
            parser.error("source or candidate differs from preregistered evaluation freeze")
    image_id = command(["docker", "image", "inspect", "--format", "{{.Id}}", args.image])
    if args.phase == "evaluation" and (
        freeze.get("image_id") != image_id
        or freeze.get("mission_timeout_sec", {}).get(args.profile) != args.mission_timeout
    ):
        parser.error("image or mission deadline differs from the evaluation freeze")
    pair = args.directory.resolve()
    pair.mkdir(parents=True, exist_ok=False)
    (pair / "protocol.json").write_bytes(protocol_bytes)
    manifest = {
        "schema_version": "rosclaw.paired_efficiency_run.v1",
        "phase": args.phase,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_commit": commit,
        "image_id": image_id,
        "profile": args.profile,
        "seed": args.seed,
        "candidate": args.candidate,
        "candidate_repair_strategy": args.candidate_repair_strategy,
        "precise_repair_waypoints": args.precise_repair_waypoints,
        "mission_timeout_sec": args.mission_timeout,
        "v1_done": False,
    }
    (pair / "pair-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    arms = ["baseline", "candidate"] if args.seed % 2 else ["candidate", "baseline"]
    try:
        results = [
            run_arm(pair, arm, args, ordinal, image_id, commit) for ordinal, arm in enumerate(arms)
        ]
    except KeyboardInterrupt:
        manifest.update(
            status="INTERRUPTED",
            runs=[json.loads(p.read_text()) for p in pair.glob("*/run-result.json")],
        )
        (pair / "paired-efficiency-results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(130) from None
    manifest.update(
        status="PASS" if all(r["status"] == "PASS" for r in results) else "FAIL", runs=results
    )
    if manifest["status"] == "PASS":
        baseline = next(r for r in results if r["arm"] == "baseline")
        candidate = next(r for r in results if r["arm"] == "candidate")
        manifest["paired_measured_reduction"] = {
            key: 1 - candidate[key] / baseline[key]
            for key in ["measured_distance_m", "sim_duration_sec"]
        }
        manifest["statistical_note"] = (
            "One pair is a pilot observation, not a median improvement or P0 acceptance."
        )
    (pair / "paired-efficiency-results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
