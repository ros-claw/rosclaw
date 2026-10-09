"""Actual MCP stdio worker retrieval refusal; no model, ROS Node or World."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(directory, *, mode=None, isolated_output_root=False):
    if mode not in (None, "M0", "M1", "M2"):
        raise ValueError("one explicit registered memory mode required")
    if isolated_output_root:
        if (
            mode is None
            or (directory.parent / "group-identity-marker.txt").read_text() != mode + "\n"
        ):
            raise ValueError("exact separately mounted group output identity required")
        if {p.name for p in directory.parent.iterdir()} != {"group-identity-marker.txt"}:
            raise ValueError("separate group output must contain no other worker data")
    directory.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    original = directory / "original-executed-source"
    original.mkdir()
    sources = {}
    for index, path in enumerate(
        [
            Path(__file__),
            repo / "src/rosclaw/mcp/adapters/runtime_client.py",
            repo / "src/rosclaw/connectors/ros/context/memory_worker_projection.py",
            repo / "src/rosclaw/connectors/ros/context/memory_intervention.py",
            repo / "src/rosclaw/mcp/server.py",
            repo / "src/rosclaw/agent/detectors.py",
            repo / "src/rosclaw/core/runtime.py",
            repo / "src/rosclaw/cli.py",
        ]
    ):
        raw = path.read_bytes()
        (original / (str(index) + "_" + path.name)).write_bytes(raw)
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
    (directory / "before-execution-source-hashes.json").write_text(json.dumps(sources, indent=2))
    (directory / "installed-sdk-versions.json").write_text(
        json.dumps(
            {"python": sys.version, "mcp": version("mcp"), "pydantic": version("pydantic")},
            indent=2,
        )
    )
    boundary = {
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "full_per_group_tool_isolation": "NOT_VERIFIED",
        "single_group_output_root_checked": isolated_output_root,
        "selected_mode": mode,
    }
    if Path("/proc/self/status").exists():
        raw = Path("/proc/self/status").read_bytes()
        (directory / "original-proc-self-status.txt").write_bytes(raw)
        parsed = dict(line.split(":", 1) for line in raw.decode().splitlines() if ":" in line)
        boundary.update(
            cap_eff=parsed.get("CapEff", "").strip(),
            no_new_privileges=parsed.get("NoNewPrivs", "").strip(),
        )
    if Path("/proc/net/dev").exists():
        raw = Path("/proc/net/dev").read_bytes()
        (directory / "original-proc-net-dev.txt").write_bytes(raw)
        boundary["network_interfaces"] = [
            line.split(":", 1)[0].strip() for line in raw.decode().splitlines() if ":" in line
        ]
    (directory / "actual-process-boundary.json").write_text(json.dumps(boundary, indent=2))
    mounts = Path("/proc/self/mountinfo")
    if mounts.exists():
        (directory / "original-proc-self-mountinfo.txt").write_bytes(mounts.read_bytes())
    cases = []
    modes = (mode,) if mode is not None else ("M0", "M1", "M2")
    for mode in modes:
        worker = directory / mode
        worker.mkdir(mode=0o700)
        profile = worker / "runtime.yaml"
        profile.write_text(
            yaml.safe_dump(
                {
                    "ros_expert_memory_experiment": {
                        "schema_version": "rosclaw.memory_worker_runtime.v1",
                        "mode": mode,
                        "retrieval_source": "worker_projection_only",
                    }
                }
            )
        )
        environment = dict(os.environ, ROSCLAW_HOME=str(worker / "home"), ROSCLAW_ROS_EXPERT="0")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "rosclaw.entrypoint",
                "mcp",
                "serve",
                "--transport",
                "stdio",
                "--project-root",
                str(worker),
                "--profile",
                str(profile),
            ],
            env=environment,
        )
        with (worker / "original-server-stderr.txt").open("w") as stderr:
            async with (
                asyncio.timeout(30),
                stdio_client(parameters, errlog=stderr) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "query_memory", {"instruction": "retrieve past repair", "limit": 5}
                )
                (worker / "original-tool-response.json").write_text(
                    result.model_dump_json(indent=2)
                )
                texts = [row.text for row in result.content if row.type == "text"]
                if len(texts) != 1:
                    raise ValueError("one original structured Memory response required")
                response = json.loads(texts[0])
                if (
                    response.get("ok") is not False
                    or response.get("error", {}).get("code") != "MEMORY_RETRIEVAL_DISABLED"
                ):
                    raise ValueError("actual MCP tool bypassed worker-only historical retrieval")
                cases.append(
                    {
                        "mode": mode,
                        "canonical_MCP_refusal": "MEMORY_RETRIEVAL_DISABLED",
                        "result": "PASS",
                    }
                )
    for path, sha in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("executed Memory source changed")
    report = {
        "status": "PASS_ACTUAL_MCP_STDIO_GLOBAL_MEMORY_REFUSAL",
        "cases": cases,
        "sources_unchanged": True,
        "model_called": False,
        "World_or_robot_action_started": False,
        "causal_memory_benefit_verified": False,
        "complete_OS_tool_isolation_verified": False,
    }
    (directory / "source-contract-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("M0", "M1", "M2"))
    parser.add_argument("--isolated-output-root", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.directory, mode=args.mode, isolated_output_root=args.isolated_output_root))
