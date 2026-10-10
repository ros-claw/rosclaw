"""Agent-side black-box journey through the existing MCP request_action tool."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fixture_body import configure_fixture_body
from profiles import PROFILES, profile_for_urdf

from rosclaw.connectors.ros.expert import inspect_system
from rosclaw.connectors.ros.mission import compile_mission
from rosclaw.connectors.ros.resolver import resolve_task
from rosclaw.connectors.ros.verification.reachable import cleanable_cells
from rosclaw.daemon.client import DaemonClient
from rosclaw.mcp import tools
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    parser.add_argument("--navigation-only", action="store_true")
    parser.add_argument("--reject-smaller-scope", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model-urdf", type=Path)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--mission-timeout", type=int, default=900)
    parser.add_argument(
        "--repair-strategy",
        choices=[
            "greedy",
            "pose_aware",
            "pose_aware_robust",
            "pose_aware_robust_sequence",
            "pose_aware_robust_tracking_sequence",
        ],
        default="greedy",
    )
    args = parser.parse_args()
    if args.navigation_only and args.reject_smaller_scope:
        parser.error("navigation and scope-rejection cases are mutually exclusive")
    if not 60 <= args.mission_timeout <= 1800:
        parser.error("mission timeout must be between 60 and 1800 seconds")
    root = args.directory.resolve()
    urdf_path = args.model_urdf or root / "robot.urdf"
    profile = profile_for_urdf(urdf_path, args.profile)
    measured = json.loads((root / "measured_map.json").read_text())
    width, height, res = measured["width"], measured["height"], measured["resolution"]
    origin = measured["origin"]
    start = int(-origin[1] / res) * width + int(-origin[0] / res)
    cells = cleanable_cells(
        width=width,
        height=height,
        resolution=res,
        occupancy=measured["occupancy"],
        start_cell=start,
        robot_radius=profile.physical_radius_m,
        cleaning_radius=profile.cleaner_half_width_m,
    )
    body = {
        "body_id": profile.body_id,
        "base_frame": "base_footprint",
        "map_frame": "map",
        "physical_radius_m": profile.physical_radius_m,
        "cleaning_polygon": profile.cleaning_polygon,
    }
    # This acceptance world is rectangular. Use the measured cleanable room
    # boundary, not a second robot-center inset before Fields2Cover headlands.
    left = origin[0] + min(i % width for i in cells) * res
    right = origin[0] + (max(i % width for i in cells) + 1) * res
    bottom = origin[1] + min(i // width for i in cells) * res
    top = origin[1] + (max(i // width for i in cells) + 1) * res
    body["coverage_polygon"] = [[left, bottom], [right, bottom], [right, top], [left, top]]
    body_hash = configure_fixture_body(root / "home", body, urdf_path)
    body["effective_body_hash"] = body_hash
    (root / "body.json").write_text(json.dumps(body, indent=2) + "\n")
    config = {
        "body_id": body["body_id"],
        "body_snapshot_hash": body_hash,
        "repair_strategy": args.repair_strategy,
        "grid": {
            "width": width,
            "height": height,
            "resolution": res,
            "origin": origin,
            "accessible_cells": cells,
            "cleaning_polygon": body["cleaning_polygon"],
            "frame_id": "map",
        },
    }
    centers = cleanable_cells(
        width=width,
        height=height,
        resolution=res,
        occupancy=measured["occupancy"],
        start_cell=start,
        robot_radius=profile.recovery_radius_m,
        cleaning_radius=0.001,
    )
    config["recovery_centers"] = [
        [origin[0] + (i % width + 0.5) * res, origin[1] + (i // width + 0.5) * res] for i in centers
    ]
    # Boundary targets get an additional 50mm circle clearance screen.
    # The original recovery centers/algorithm and denominator remain unchanged.
    boundary_cells = cleanable_cells(
        width=width,
        height=height,
        resolution=res,
        occupancy=measured["occupancy"],
        start_cell=start,
        robot_radius=max(profile.recovery_radius_m, profile.physical_radius_m + 0.05),
        cleaning_radius=0.001,
    )
    config["boundary_centers"] = [
        [origin[0] + (i % width + 0.5) * res, origin[1] + (i // width + 0.5) * res]
        for i in boundary_cells
    ]
    if (root / "experiment.json").exists():
        config["experiment"] = json.loads((root / "experiment.json").read_text())
    (root / "execution_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.prepare_only:
        print(
            json.dumps(
                {"status": "PREPARED", "body_id": body["body_id"], "body_snapshot_hash": body_hash}
            )
        )
        return
    os.environ["ROSCLAW_ROS_EXPERT"] = "1"
    os.environ["ROSCLAW_HOME"] = str(root / "home")
    ready = root / "daemon_ready.json"
    ready.unlink(missing_ok=True)
    log = (root / "daemon.log").open("w")
    daemon = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("daemon.py")),
            "--directory",
            str(root),
            "--endpoint",
            args.endpoint,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            if daemon.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("daemon failed to start; inspect daemon.log")
            await asyncio.sleep(0.2)
        client = DaemonClient(socket_path=root / "run/rosclawd.sock")
        client.arm_runtime("isolated Gazebo fixture preflight")
        runtime_client = RuntimeClient(
            project_root=Path.cwd(),
            robot_id=body["body_id"],
            runtime_profile={},
            daemon_client=client,
        )
        # Exercise the real tool wrapper, retaining its REAL interaction gate.
        tools._client = lambda: runtime_client
        model = inspect_system(
            endpoint=args.endpoint, robot_id=body["body_id"], deep=True, body=body
        )
        (root / "snapshot.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
        (root / "solution.json").write_text(
            json.dumps(resolve_task(model, "完成整个房间清扫。"), indent=2) + "\n"
        )
        mission_id = "gazebo-room-cleaning"
        (root / "task_graph.json").write_text(
            json.dumps(
                compile_mission(model, "完成整个房间清扫。", mission_id=mission_id).model_dump(
                    mode="json"
                ),
                indent=2,
            )
            + "\n"
        )

        async def request(capability, arguments, action_id, timeout, expected_state="COMPLETED"):
            ticket = await tools._request_action(
                capability_id=capability,
                arguments=arguments,
                execution_mode="SIMULATION",
                body_snapshot_hash=body_hash,
                body_id=body["body_id"],
                action_id=action_id,
                timeout_sec=timeout,
                wait_timeout_sec=0,
            )
            (root / (action_id + ".ticket.json")).write_text(json.dumps(ticket, indent=2) + "\n")
            deadline = time.monotonic() + timeout + 15
            while time.monotonic() < deadline:
                status = client.get_action_status(action_id)
                if status.get("state") in {"FINISHED", "CANCELLED"}:
                    receipt = client.get_execution_receipt(action_id)
                    (root / (action_id + ".receipt.json")).write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
                    print(json.dumps(receipt), flush=True)
                    if receipt.get("receipt", {}).get("final_state") != expected_state:
                        raise RuntimeError(f"action failed: {action_id}")
                    return receipt
                await asyncio.sleep(0.25)
            raise TimeoutError(action_id)

        await request("localization.set_initial_pose", {}, "golden-localize", 10)
        if args.reject_smaller_scope:
            receipt = await request(
                "coverage.execute",
                {
                    "mission_id": mission_id,
                    "frame_id": "map",
                    "polygons": [
                        {
                            "points": [
                                {"x": x / 2, "y": y / 2, "z": 0.0}
                                for x, y in body["coverage_polygon"]
                            ]
                        }
                    ],
                },
                "golden-scope-rejection",
                15,
                expected_state="BLOCKED",
            )
            errors = receipt.get("receipt", {}).get("errors", [])
            if not any(e.get("code") == "COVERAGE_SCOPE_REJECTED" for e in errors):
                raise RuntimeError("coverage scope was not rejected by the executor boundary")
        elif args.navigation_only:
            await request(
                "navigation.navigate_to_pose",
                {
                    "pose": {
                        "header": {"frame_id": "map"},
                        "pose": {
                            "position": {"x": 0.5, "y": -0.5, "z": 0.0},
                            "orientation": {"w": 1.0},
                        },
                    }
                },
                "golden-navigation",
                90,
            )
        else:
            await request(
                "coverage.execute",
                {
                    "mission_id": mission_id,
                    "polygons": [
                        {
                            "points": [
                                {"x": x, "y": y, "z": 0.0} for x, y in body["coverage_polygon"]
                            ]
                        }
                    ],
                },
                "golden-coverage",
                args.mission_timeout,
            )
            await request(
                "ros.expert.remember",
                {"coverage_action_id": "golden-coverage"},
                "golden-remember",
                30,
            )
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait()
        log.close()


if __name__ == "__main__":
    asyncio.run(main())
