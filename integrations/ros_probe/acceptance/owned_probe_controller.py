"""Owned SIM instrument controller; never a robot actuator or permission.

The launcher must pin one live Gazebo process and its actual source mappings.
Only the disjoint frozen instrument can be lifted. Robot/scene/ground sources
must be fresh before BEGIN. Service success never substitutes for measured
lift, clear cache and recontact. No Node or process starts in prepare-only mode.
"""

import argparse
import base64
import hashlib
import json
import os
import selectors
import signal
import socket
import subprocess
import time
import uuid
from contextlib import suppress
from pathlib import Path

from backend_probe_evidence import probe_policy
from backend_probe_world import bounded_source
from backend_world_ownership import WorldSourceOwner
from instrument_service_evidence import parse_original_service_record
from probe_controller_ipc import process_identity
from probe_lift_evidence import lift_request
from probe_scene_geometry import decode_scene_json

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog


def worker_arguments(policy, *, binary, robot_model, partition):
    probe_policy(policy)
    native = policy["native_policy"]
    if native["contact_policy"]["model_name"] == robot_model:
        raise ValueError("instrument worker cannot target the robot model")
    return [
        str(binary),
        native["world_name"],
        native["contact_policy"]["model_name"],
        robot_model,
        *[f"{v:.17g}" for v in (*policy["probe_xy"], policy["lift_z_m"])],
        partition,
        "--owned-runtime",
    ]


def exchange(directory, packet):
    raw = json.dumps(packet, allow_nan=False).encode()
    if len(raw) > 16384:
        raise ValueError("bounded original instrument IPC request required")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
        client.settimeout(0.08)
        client.connect(str(Path(directory) / "probe-controller.sock"))
        client.sendall(raw)
        reply = decode_scene_json(client.recv(4097))
    if (
        type(reply) is not dict
        or reply.get("accepted_original_source_event") is not True
        or reply.get("transaction_id") != packet["transaction_id"]
        or reply.get("event") != packet["kind"]
        or reply.get("authorization") is not False
        or reply.get("world_source_ownership_admitted") is not False
    ):
        raise ValueError("independent observer did not accept original instrument source event")
    return raw, reply


class BoundedWorkerOutput:
    def __init__(self, stream):
        self.stream = stream
        self.selector = selectors.DefaultSelector()
        self.selector.register(stream, selectors.EVENT_READ)
        self.buffer = b""

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise ValueError("owned instrument worker original reply deadline expired")
            chunk = os.read(self.stream.fileno(), 4096)
            if not chunk or len(self.buffer) + len(chunk) > 65536:
                raise ValueError("owned instrument worker output incomplete or oversized")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line

    def close(self):
        self.selector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "bundle-directory", "observer-directory", "physics-plugin"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--world-pid", required=True, type=int)
    parser.add_argument("--world-uid", required=True, type=int)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--duration", required=True, type=int)
    parser.add_argument("--physics-plugin-sha256", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        not 60 <= args.duration <= 1920
        or args.directory.is_symlink()
        or not args.directory.is_dir()
    ):
        raise ValueError("private owned controller output and immutable deadline required")
    manifest = decode_scene_json(
        bounded_source(args.bundle_directory / "backend-world-bundle.json")
    )
    policy = decode_scene_json(
        bounded_source(args.bundle_directory / "instrument-source/probe-policy.json")
    )
    probe_policy(policy)
    binding = decode_scene_json(
        bounded_source(args.bundle_directory / "robot-source/brush_binding.json")
    )
    robot = decode_scene_json(
        bounded_source(args.bundle_directory / "robot-source/native-policy.json")
    )
    binary = args.bundle_directory / "instrument-source/owned_instrument_service"
    binary_sha = manifest.get("instrument_service_binary_sha256")
    source = bounded_source(binary, 100_000_000)
    if not source.startswith(b"\x7fELF") or hashlib.sha256(source).hexdigest() != binary_sha:
        raise ValueError("frozen actual instrument worker executable required")
    configuration = decode_scene_json(
        bounded_source(args.bundle_directory / "observer-source/backend_actor_constraint.json")
    )
    if any(configuration[k] != binding[k] for k in ("run_id", "body_snapshot_hash")):
        raise ValueError("controller run/Body differs from frozen observer/robot")
    argv = worker_arguments(
        policy,
        binary=binary,
        robot_model=robot["contact_policy"]["model_name"],
        partition=args.partition,
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "role": "UNEXECUTED_OWNED_SIM_INSTRUMENT_PLAN",
                    "argv": argv,
                    "service_binary_sha256": binary_sha,
                    "world_source_admission": False,
                    "authorization": False,
                }
            )
        )
        return
    if os.getenv("GZ_PARTITION") != args.partition:
        raise ValueError("actual controller partition differs from owned World")
    owner = WorldSourceOwner(
        args.bundle_directory,
        world_pid=args.world_pid,
        world_uid=args.world_uid,
        partition=args.partition,
        seed=args.seed,
        physics_plugin=args.physics_plugin,
        physics_plugin_sha256=args.physics_plugin_sha256,
        contact_plugin_sha256=manifest["source_contact_library_sha256"],
    )
    ownership = owner.check()
    audit = CoverageAuditLog(
        args.directory / "owned-probe-controller-events.jsonl",
        capacity=512,
        context={
            **binding,
            "source": "owned_SIM_instrument_RPC_controller_NOT_ROBOT_ACTUATOR",
            "evidence_domain": "SIMULATION",
            "instrument_service_binary_sha256": binary_sha,
        },
    )
    child, output, child_log = None, None, None
    stopped = [False]
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    deadline = time.monotonic() + args.duration
    next_cycle = None
    next_wall_cycle = None
    try:
        audit.emit("owned_world_source_mappings", ownership)
        child_log = (args.directory / "instrument-worker-stderr.log").open("x")
        child = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=child_log
        )
        output = BoundedWorkerOutput(child.stdout)
        ready_raw = output.read(6)
        ready = decode_scene_json(ready_raw)
        if ready != {
            "status": "READY_INSTRUMENT_SOURCE_PROCESS",
            "transport_node_started": True,
            "authorization": False,
        }:
            raise ValueError("owned instrument transport worker did not initialize exactly")
        worker_identity = process_identity(child.pid, os.getuid())
        audit.emit(
            "owned_instrument_worker_ready",
            {
                "pid": child.pid,
                "process_starttime": worker_identity[2],
                "binary_sha256": binary_sha,
                "original_stdout_base64": base64.b64encode(ready_raw).decode(),
            },
        )
        while not stopped[0] and time.monotonic() < deadline:
            owner.check()
            if (
                child.poll() is not None
                or process_identity(child.pid, os.getuid()) != worker_identity
            ):
                raise ValueError("owned instrument transport worker disappeared or changed")
            if audit.dropped or audit.error:
                raise ValueError("owned instrument original controller audit incomplete")
            latest = args.observer_directory / "backend-observation-latest.json"
            if not latest.exists():
                time.sleep(0.01)
                continue
            snapshot = decode_scene_json(bounded_source(latest))["snapshot"]
            wall = time.monotonic()
            if snapshot.get("source_fault"):
                raise ValueError("independent robot/instrument source fault remains latched")
            if (
                not 0 <= wall - snapshot["sampled_monotonic_sec"] < 0.1
                or snapshot["constraint_policy_hash"] != configuration["constraint_policy_hash"]
            ):
                raise ValueError("independent original observer source stale or policy differs")
            sim = snapshot["probe_sim_time_sec"]
            geometry = snapshot.get("scene_geometry_constraint", {})
            if (
                snapshot.get("probe_phase") != "READY_FOR_LIFT"
                or snapshot.get("probe_lift_transaction_pending")
                or not geometry.get("scene_geometry_constraint_satisfied")
                or snapshot["robot_collision_count"] != 0
                or sim is None
                or (next_cycle is not None and sim < next_cycle and wall < next_wall_cycle)
            ):
                time.sleep(0.01)
                continue
            transaction = uuid.uuid4().hex + uuid.uuid4().hex
            packet = {
                "schema_version": "rosclaw.probe_controller_ipc.v1",
                "kind": "backend_probe_lift_begin",
                "run_id": binding["run_id"],
                "constraint_policy_hash": configuration["constraint_policy_hash"],
                "transaction_id": transaction,
                "request_base64": base64.b64encode(lift_request(policy)).decode(),
            }
            original, reply = exchange(args.observer_directory, packet)
            audit.emit(
                "owned_instrument_original_begin",
                {
                    "original_ipc_base64": base64.b64encode(original).decode(),
                    "observer_reply": reply,
                    "received_monotonic_sec": time.monotonic(),
                },
            )
            child.stdin.write(lift_request(policy).hex().encode() + b"\n")
            child.stdin.flush()
            raw_reply = output.read(0.12)
            audit.emit(
                "owned_instrument_original_service_reply",
                {
                    "original_service_record_base64": base64.b64encode(raw_reply).decode(),
                    "original_service_record_sha256": hashlib.sha256(raw_reply).hexdigest(),
                    "received_monotonic_sec": time.monotonic(),
                },
            )
            record, original_sources, ack = parse_original_service_record(raw_reply, policy)
            packet.update(
                kind="backend_probe_lift_ack",
                returncode=0,
                response_base64=base64.b64encode(original_sources["response_text_hex"]).decode(),
                acknowledged_at_unix_ns=ack["acknowledged_at_unix_ns"],
                service_record_base64=base64.b64encode(raw_reply).decode(),
                service_binary_sha256=binary_sha,
            )
            original, reply = exchange(args.observer_directory, packet)
            audit.emit(
                "owned_instrument_original_RPC_and_ack",
                {
                    "original_service_record_base64": base64.b64encode(raw_reply).decode(),
                    "original_ipc_base64": base64.b64encode(original).decode(),
                    "observer_reply": reply,
                    "received_monotonic_sec": time.monotonic(),
                },
            )
            next_cycle = sim + min(policy["refresh_sim_sec"] / 2, 5)
            next_wall_cycle = time.monotonic() + min(policy["refresh_wall_sec"] / 2, 10)
            time.sleep(0.01)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        audit.emit(
            "owned_instrument_controller_failed",
            {
                "error": str(exc)[:512],
                "received_monotonic_sec": time.monotonic(),
                "unpaired_original_worker_stdout_base64": base64.b64encode(output.buffer).decode()
                if output is not None
                else None,
            },
        )
        with suppress(ValueError, OSError):
            exchange(
                args.observer_directory,
                {"kind": "backend_probe_controller_fault", "error": str(exc)[:256]},
            )
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
        if output is not None:
            output.close()
        if child_log is not None:
            child_log.close()
        summary = audit.close()
        with Path(str(audit.path) + ".summary.json").open("x") as stream:
            stream.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
