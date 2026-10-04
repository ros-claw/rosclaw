"""Fixture capability declarations and read-only evidence for Native Agent.

Physical tool bodies deliberately cannot execute: the existing Agentd action
channel dispatches their IDs to the configured rosclawd. This stdio worker is
only a catalog and observer, never a second motion authority.
"""

import argparse
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from rosclaw.connectors.ros.context.body import configured_ros_body
from rosclaw.connectors.ros.context.compiler import compile_agent_summary
from rosclaw.connectors.ros.context.probe_client import read_snapshot
from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.resolver import resolve_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    args = parser.parse_args()
    specification = json.loads((args.directory / "body.json").read_text())
    server = FastMCP("ros-expert-gazebo-acceptance")

    @server.tool(name="ros.observe_system")
    def observe_system() -> dict:
        """Observe current ROS readiness, diagnosis, measured task area and available solutions."""
        body = configured_ros_body(specification["body_id"], specification["effective_body_hash"])
        model = read_snapshot(endpoint=args.endpoint, robot_id=specification["body_id"], body=body)
        return {
            "snapshot_id": model.snapshot_id,
            "snapshot_hash": model.snapshot_hash,
            "captured_at": model.captured_at.isoformat(),
            "system": compile_agent_summary(model),
            "diagnosis": diagnose(model),
            "solution": {
                **resolve_task(model, "完成整个房间清扫。"),
                "completion_requirements": [
                    "Canonical coverage receipt: >=98%, zero contacts, complete observations.",
                    "Call ros.expert.remember with the exact coverage action ID to persist the independent verification artifact, Practice and Memory before declaring the mission complete.",
                    "The existing TaskKernel must register the verified artifact and close the task.",
                ],
            },
            "task_area": {
                "frame_id": "map",
                "polygons": [
                    {
                        "points": [
                            {"x": x, "y": y, "z": 0.0} for x, y in specification["coverage_polygon"]
                        ]
                    }
                ],
                "source": "measured_map_fixed_cleanable_cells",
            },
            "evidence_domain": "SIMULATION",
            "authorization": False,
        }

    @server.tool(name="localization.set_initial_pose")
    def initialize_localization() -> dict:
        """Initialize AMCL at the independently verified stationary fixture spawn, through rosclawd."""
        raise RuntimeError("physical actions require the Agentd rosclawd action channel")

    @server.tool(name="coverage.execute")
    def execute_coverage(polygons: list[dict], mission_id: str = "gazebo-room-cleaning") -> dict:
        """Execute official coverage and bounded measured missed-cell repair through rosclawd.

        Use the observed task area's polygons. The daemon independently verifies
        >=98% enabled-cleaning coverage, zero contacts and complete observations.
        After a successful canonical receipt, complete the mission with
        ros.expert.remember using its exact coverage action ID; that commits
        independent verification, Practice/Memory and the registered deliverable.
        """
        raise RuntimeError("physical actions require the Agentd rosclawd action channel")

    @server.tool(name="ros.expert.remember")
    def remember(coverage_action_id: str) -> dict:
        """Persist a verified coverage receipt into existing Practice and Memory through rosclawd."""
        raise RuntimeError("mission persistence requires the Agentd rosclawd action channel")

    # Same explicit strict parameter boundary as the existing first-party
    # simulation MCP server; physical action admission rejects open objects.
    for tool in server._tool_manager._tools.values():
        tool.parameters["additionalProperties"] = False
    server.run()


if __name__ == "__main__":
    main()
