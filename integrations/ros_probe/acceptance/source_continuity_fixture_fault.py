"""Explicit operator faults on one owned disposable SIM; never an Agent tool."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "reset",
            "observer_kill",
            "stale",
            "resume",
            "entity_replaced",
            "sensor_drift",
            "clock_freeze",
        ],
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--inventory-sha256", required=True)
    args = p.parse_args()
    out = Path("/evidence")
    assets = Path("/assets")
    inventory = (assets / "prepared-assets.json").read_bytes()
    if hashlib.sha256(inventory).hexdigest() != args.inventory_sha256:
        raise ValueError("independent owned staging inventory pin mismatch")
    for name, digest in json.loads(inventory)["file_sha256"].items():
        if (
            name.startswith("/")
            or ".." in Path(name).parts
            or hashlib.sha256((assets / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("immutable operator staging asset mismatch")
    world = ET.parse(assets / "world.sdf").getroot().find("world")
    plugins = [v for v in world.findall("plugin") if v.get("name") == "rosclaw::SourceWitness"]
    if (
        world.get("name") != "ros_expert"
        or len(plugins) != 1
        or plugins[0].findtext("run_id") != args.run_id
    ):
        raise ValueError("fault actor requires its explicit owned World/run")
    facts = json.loads((out / "observer-process-facts.json").read_text())
    if len(facts["observer_loaded_servers"]) != 1:
        raise ValueError("one original observer process required")
    proc = facts["observer_loaded_servers"][0]
    pid = proc["pid"]
    if (
        Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
        != proc["stat"].split(")", 1)[1].split()[19]
        or "/observer/librosclaw_source_witness.so" not in Path(f"/proc/{pid}/maps").read_text()
    ):
        raise ValueError("original observer process identity changed")
    before = time.monotonic_ns()
    commands = []

    def gz(service, reqtype, request):
        argv = [
            "gz",
            "service",
            "-s",
            "/world/ros_expert/" + service,
            "--reqtype",
            reqtype,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "5000",
            "--req",
            request,
        ]
        r = subprocess.run(argv, capture_output=True, timeout=8)
        commands.append(
            {
                "argv": argv,
                "returncode": r.returncode,
                "stdout": r.stdout.decode(),
                "stderr": r.stderr.decode(),
            }
        )
        if r.returncode or "data: true" not in r.stdout.decode():
            raise RuntimeError("owned Gazebo fault operation was not acknowledged")

    if args.mode in {"observer_kill", "stale", "resume"}:
        os.kill(
            pid,
            {"observer_kill": signal.SIGKILL, "stale": signal.SIGSTOP, "resume": signal.SIGCONT}[
                args.mode
            ],
        )
    elif args.mode in {"reset", "clock_freeze"}:
        gz(
            "control",
            "gz.msgs.WorldControl",
            "reset: {all: true}" if args.mode == "reset" else "pause: true",
        )
    elif args.mode == "sensor_drift":
        latest = json.loads((out / "continuity-latest-original-sample.json").read_text())
        robot = next(m for m in latest["models"] if m["name"] == "turtlebot3_waffle")
        lidar = next(
            s
            for s in latest["sensors"]
            if s["kind"] in {"gpu_lidar", "lidar"}
            and any(a["entity"] == robot["entity"] for a in s["ancestry"])
        )
        temporary = out / "source-inventory-fault-trigger.pending"
        with temporary.open("x") as f:
            json.dump(
                {
                    "run_id": args.run_id,
                    "mode": "sensor_kind_drift",
                    "model_entity": robot["entity"],
                    "sensor_entity": lidar["entity"],
                },
                f,
            )
        os.link(temporary, out / "source-inventory-fault-trigger.json")
        temporary.unlink()
    elif args.mode == "entity_replaced":
        model = world.find("model[@name='turtlebot3_waffle']")
        if model is None:
            raise ValueError("original owned robot model required")
        sdf = ET.Element("sdf", version="1.9")
        sdf.append(model)
        Path("/tmp/continuity-replacement-model.sdf").write_bytes(ET.tostring(sdf))
        gz("remove", "gz.msgs.Entity", 'name: "turtlebot3_waffle", type: MODEL')
        gz(
            "create",
            "gz.msgs.EntityFactory",
            'sdf_filename: "/tmp/continuity-replacement-model.sdf", allow_renaming: false',
        )
    with (out / f"continuity-fault-{args.mode}-original.json").open("x") as f:
        json.dump(
            {
                "mode": args.mode,
                "run_id": args.run_id,
                "original_process": proc,
                "before_monotonic_ns": before,
                "after_monotonic_ns": time.monotonic_ns(),
                "operations": commands,
                "collateral": "observer shares Gazebo server; kill/stale also stop World"
                if args.mode in {"observer_kill", "stale"}
                else "see raw snapshots",
                "execution_authority": "NONE",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
