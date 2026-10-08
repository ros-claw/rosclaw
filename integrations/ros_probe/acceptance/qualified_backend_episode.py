"""Qualified known-SIM episode contracts and closed source-only replay.

The fixture launcher owns instrumentation; robot commands require MCP/rosclawd.
This module opens no ROS/DDS/World and cannot confer robot authorization.
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

from backend_probe_world import bounded_source
from closed_backend_observation import closed_backend_observation
from probe_scene_geometry import decode_scene_json
from profiles import PROFILES

BACKEND_KEYS = {"contact_plugin_sha256", "instrument_service_binary_sha256", "source_mode"}


def validate_qualified_spec(value, base_validator):
    if (
        type(value) is not dict
        or type(value.get("schema_version")) is not str
        or value.get("schema_version")
        not in {"rosclaw.dynamic_native_episode.v2", "rosclaw.dynamic_native_episode.v3"}
    ):
        raise ValueError("closed qualified Native episode v2 protocol required")
    source = value.get("backend_source")
    if type(source) is not dict or set(source) != BACKEND_KEYS:
        raise ValueError("complete explicit original backend source hashes required")
    if source["source_mode"] != "ALL_STEP_SPATIAL_ORIGINAL_SERVICE_WIRE_REQUIRED":
        raise ValueError("qualified original source mode cannot be downgraded")
    for key in ("contact_plugin_sha256", "instrument_service_binary_sha256"):
        if type(source[key]) is not str or not re.fullmatch(r"[a-f0-9]{64}", source[key]):
            raise ValueError("exact original compiled backend ELF hash required")
    is_d3 = value["schema_version"] == "rosclaw.dynamic_native_episode.v3"
    scenario = value.get("scenario_source")
    if is_d3:
        if (
            value.get("case") != "D3"
            or type(scenario) is not dict
            or set(scenario) != {"second_target_xy", "second_dwell_sim_sec", "gap_sim_sec"}
        ):
            raise ValueError("closed qualified two-blocker D3 scenario required")
        target = scenario["second_target_xy"]
        if (
            type(target) is not list
            or len(target) != 2
            or any(type(v) not in (int, float) or not -1.2 <= v <= 1.2 for v in target)
            or target == value.get("target_xy")
        ):
            raise ValueError("distinct explicit second blocker target required")
        for key, low, high in (("second_dwell_sim_sec", 10, 30), ("gap_sim_sec", 2, 30)):
            if type(scenario[key]) is not int or not low <= scenario[key] <= high:
                raise ValueError("bounded frozen nonconcurrent D3 timings required")
    excluded = {"backend_source"}
    if is_d3:
        excluded.add("scenario_source")
    base = {key: item for key, item in value.items() if key not in excluded}
    base["schema_version"] = "rosclaw.dynamic_native_episode.v1"
    if is_d3:
        base["case"] = "D2"
    base = base_validator(base)
    if is_d3:
        base["case"], base["scenario_source"] = "D3", scenario
    return base, source


def frozen_backend_files(source, contact_plugin, instrument_service_binary):
    inputs = {}
    for key, path in (
        ("contact_plugin_sha256", contact_plugin),
        ("instrument_service_binary_sha256", instrument_service_binary),
    ):
        if path is None:
            raise ValueError("both qualified original backend ELF paths required")
        path = Path(path)
        raw = bounded_source(path, 20_000_000)
        if not raw.startswith(b"\x7fELF") or hashlib.sha256(raw).hexdigest() != source[key]:
            raise ValueError("qualified original backend ELF hash differs")
        inputs[path] = raw
    return inputs


def probe_declaration(binding, profile_name):
    profile = PROFILES[profile_name]
    return {
        "schema_version": "rosclaw.backend_probe_fixture_declaration.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "run_id": binding["run_id"],
        "world_name": binding["world_name"],
        "robot_model_name": profile.simulation_model,
        "probe_model_name": "owned_backend_instrument",
        "probe_xy": [6, 0],
        "sphere_radius_m": 0.05,
        "ground_z_m": 0,
        "lift_z_m": 10,
        "ground_collision_name": "floor::ground::ground",
        "cleaning_polygon": profile.cleaning_polygon,
        "maximum_robot_radius_m": 0.3,
        "pose_topic": "/instrument/pose",
        "component_topic": "/instrument/components",
        "contact_topic": "/instrument/contact",
    }


def qualified_projection(directory):
    if (directory / "backend-stack-fault.json").exists():
        raise ValueError("owned backend fault latched; original fault artifact retained")
    plan = decode_scene_json(bounded_source(directory / "backend-stack-source-plan.json"))
    owner = decode_scene_json(bounded_source(directory / "backend-world-source-process.json"))
    projection = decode_scene_json(
        bounded_source(directory / "backend-observer/backend-observation-latest.json")
    )
    snapshot = projection["snapshot"]
    age = time.monotonic() - snapshot["sampled_monotonic_sec"]
    if not (
        type(age) in (int, float)
        and math.isfinite(age)
        and 0 <= age < 0.15
        and owner.get("loaded_source_correspondence") is True
        and owner.get("bundle_manifest_sha256") == plan["bundle_manifest_sha256"]
        and snapshot.get("constraint_policy_hash") == plan["constraint_policy_hash"]
        and snapshot.get("source_fault") is None
        and snapshot.get("live_source_constraint_satisfied") is True
        and snapshot.get("probe_completed_cache_cycles", 0) >= 1
        and snapshot.get("robot_collision_count") == 0
        and projection["actor_envelope"].get("live_source_constraint_satisfied") is True
    ):
        raise ValueError(
            "actual mapped World and fresh qualified original source constraint required"
        )
    return projection


async def request_mcp_stop(directory, reason):
    """Canonical stdio MCP stop, never direct Robot/ROS/driver access."""
    import os

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    repo = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "ROSCLAW_HOME": str(directory / "home"),
        "ROSCLAW_DAEMON_SOCKET": str(directory / "run/rosclawd.sock"),
        "PYTHONPATH": str(repo / "src"),
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "rosclaw.entrypoint",
            "mcp",
            "serve",
            "--transport",
            "stdio",
            "--project-root",
            str(repo),
        ],
        env=env,
    )
    async with asyncio.timeout(8):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("emergency_stop", {"reason": reason})
                return result.model_dump(mode="json")


def replay_closed_qualified(directory):
    bundle = directory / "backend-bundle"
    plan = decode_scene_json(bounded_source(directory / "backend-stack-source-plan.json"))
    closed = decode_scene_json(bounded_source(directory / "backend-source-closed.json"))
    if (
        closed.get("source_closed_for_replay") is not True
        or closed.get("authorization") is not False
    ):
        raise ValueError("original observation writers must close before qualified replay")
    if closed.get("constraint_policy_hash") != plan["constraint_policy_hash"]:
        raise ValueError("closed original source policy differs")
    result = closed_backend_observation(
        directory / "backend-observer/backend-observation-events.jsonl",
        decode_scene_json(bounded_source(bundle / "robot-source/native-policy.json")),
        decode_scene_json(bounded_source(bundle / "instrument-source/probe-policy.json")),
        robot_directory=bundle / "robot-source",
        probe_directory=bundle / "instrument-source",
        robot_plugin=bundle / "robot-source/librosclaw_passive_contacts.so",
        probe_plugin=bundle / "instrument-source/librosclaw_passive_contacts.so",
        robot_pose_frame=plan["world_name"],
        probe_pose_frame=plan["world_name"],
        scene_binding=decode_scene_json(
            bounded_source(bundle / "world-source/physics_binding.json")
        ),
        probe_declaration=decode_scene_json(
            bounded_source(bundle / "world-source/probe-declaration.json")
        ),
        instrument_service_binary=bundle / "instrument-source/owned_instrument_service",
        instrument_service_binary_sha256=plan["instrument_service_binary_sha256"],
    )
    if (
        result["original_service_wire_required"] is not True
        or result["original_service_SDK_wire_replays"] < 1
        or result["spatial_source_join_required"] is not True
        or result["completed_exact_scene_joins"] < 1
        or result["robot_collision_count"] != 0
    ):
        raise ValueError("complete actual original service/spatial/all-step sources required")
    bounds = decode_scene_json(
        bounded_source(directory / "native-task-observation-boundaries.json")
    )
    first = last = None
    with (directory / "backend-observer/backend-observation-events.jsonl").open("rb") as stream:
        for line in stream:
            row = decode_scene_json(line)
            if row["kind"] == "backend_observation_sample":
                wall = row["payload"]["received_monotonic_sec"]
                first = wall if first is None else first
                last = wall
    if (
        first is None
        or not first
        <= bounds["native_started_monotonic_sec"]
        < bounds["native_ended_monotonic_sec"]
        <= last
    ):
        raise ValueError(
            "closed independent original source must span entire genuine Native invocation"
        )
    result.update(
        original_source_spans_native_invocation=True,
        original_source_first_monotonic_sec=first,
        original_source_last_monotonic_sec=last,
    )
    (directory / "closed-qualified-backend-source.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_closed_qualified(args.directory)))


if __name__ == "__main__":
    main()
