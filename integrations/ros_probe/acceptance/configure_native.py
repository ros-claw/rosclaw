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


def configure(directory, endpoint, *, overwrite=False, generic_proposal=None, mission_timeout=900):
    root = directory.resolve()
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "ws"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username
        or parsed.password
    ):
        raise ValueError("an owned loopback SIM fixture endpoint is required")
    if type(mission_timeout) is not int or not 60 <= mission_timeout <= 1800:
        raise ValueError("bounded SIM mission timeout required")
    generic = None
    if generic_proposal is not None:
        from rosclaw.connectors.ros.context.sim_native_fixture import prepare_sim_native_fixture

        raw = Path(generic_proposal).read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("bounded generic execution proposal required")
        generic = prepare_sim_native_fixture(root, json.loads(raw))
        body, execution = generic["body"], generic["execution_config"]
        profile = None
    else:
        profile = profile_for_urdf(root / "robot.urdf")
        body = json.loads((root / "body.json").read_text())
        execution = json.loads((root / "execution_config.json").read_text())
    resolver = BodyResolver(workspace=root / "home")
    if not resolver.effective_body_path.is_file():
        raise ValueError("prepare the compiled fixture Body first")
    effective = resolver.get_effective_body(recompile_if_stale=False)
    body_hash = effective.compute_hash()
    if (
        (profile is not None and effective.body_instance_id != profile.body_id)
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
                "timeout_ms": mission_timeout * 1000
                if generic is not None
                else (1800000 if profile.name == "burger" else 900000),
                "output_schemas": {"ros.observe_system": OBSERVATION_SCHEMA},
            }
        ],
    }
    path = root / "home/config.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(
            "fixture Native configuration already exists; use --overwrite explicitly"
        )
    if generic is not None:
        targets = {"body.json": body, "execution_config.json": execution}
        if any((root / name).exists() for name in targets):
            raise FileExistsError("generic fixture declarations require fresh exclusive outputs")
        for name, value in targets.items():
            with (root / name).open("x") as stream:
                stream.write(json.dumps(value, indent=2) + "\n")
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
    parser.add_argument("--generic-proposal", type=Path)
    parser.add_argument("--mission-timeout", type=int, default=900)
    args = parser.parse_args()
    print(
        json.dumps(
            configure(
                args.directory,
                args.endpoint,
                overwrite=args.overwrite,
                generic_proposal=args.generic_proposal,
                mission_timeout=args.mission_timeout,
            )
        )
    )
