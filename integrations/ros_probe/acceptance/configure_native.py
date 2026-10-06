"""Generate explicit SIM-only Native declarations from the compiled fixture Body.

Model settings and credentials are provisioned separately in the owned home.
This writes no Robot Kit index entry and grants no REAL authority.
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml
from profiles import profile_for_urdf

from rosclaw.body.resolver import BodyResolver

OBSERVATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        name: {"type": kind}
        for name, kind in {
            "snapshot_id": "string",
            "snapshot_hash": "string",
            "captured_at": "string",
            "system": "string",
            "diagnosis": "object",
            "solution": "object",
            "task_area": "object",
            "evidence_domain": "string",
            "authorization": "boolean",
        }.items()
    },
    "required": [
        "snapshot_id",
        "snapshot_hash",
        "captured_at",
        "system",
        "diagnosis",
        "solution",
        "task_area",
        "evidence_domain",
        "authorization",
    ],
}


def configure(directory, endpoint, *, overwrite=False):
    root = directory.resolve()
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "ws"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username
        or parsed.password
    ):
        raise ValueError("an owned loopback SIM fixture endpoint is required")
    profile = profile_for_urdf(root / "robot.urdf")
    body = json.loads((root / "body.json").read_text())
    execution = json.loads((root / "execution_config.json").read_text())
    resolver = BodyResolver(workspace=root / "home")
    if not resolver.effective_body_path.is_file():
        raise ValueError("prepare the compiled fixture Body first")
    effective = resolver.get_effective_body(recompile_if_stale=False)
    body_hash = effective.compute_hash()
    if (
        effective.body_instance_id != profile.body_id
        or body["body_id"] != effective.body_instance_id
        or execution["body_id"] != effective.body_instance_id
        or body["effective_body_hash"] != body_hash
        or execution["body_snapshot_hash"] != body_hash
    ):
        raise ValueError("fixture declarations differ from the compiled Body")
    config = {
        "agent": {
            "enabled": True,
            "body_id": effective.body_instance_id,
            "default_mode": "SIMULATION",
            "default_profile": "embodied_default",
            "max_tool_rounds": 20,
            "ros_expert": {"enabled": True, "endpoint": endpoint},
        },
        "mcp_servers": [
            {
                "name": "ros-expert-fixture",
                "command": sys.executable,
                "args": [
                    str(Path(__file__).with_name("native_tools.py")),
                    "--directory",
                    str(root),
                    "--endpoint",
                    endpoint,
                ],
                "env_refs": ["ROSCLAW_HOME"],
                "observation_tools": ["ros.observe_system"],
                "action_tools": [
                    "localization.set_initial_pose",
                    "coverage.execute",
                    "ros.expert.remember",
                ],
                "supported_modes": ["SIMULATION"],
                "required_body_types": [effective.body_instance_id],
                "effect_domain": "ROS",
                "timeout_ms": 1800000 if profile.name == "burger" else 900000,
                "output_schemas": {"ros.observe_system": OBSERVATION_SCHEMA},
            }
        ],
    }
    path = root / "home/config.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(
            "fixture Native configuration already exists; use --overwrite explicitly"
        )
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return {
        "status": "CONFIGURED",
        "body_id": effective.body_instance_id,
        "body_snapshot_hash": body_hash,
        "supported_modes": ["SIMULATION"],
        "endpoint": endpoint,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(configure(args.directory, args.endpoint, overwrite=args.overwrite)))
