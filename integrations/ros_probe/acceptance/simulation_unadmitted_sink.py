"""Strict no-admission preconditions for SDK-only negative lease probes."""

import json
import os

from simulation_current_facts import _read


def require_unadmitted_sink(root, *, run_id, body_snapshot_hash):
    sink = json.loads(_read(root / "governed-sink-status.json", 4096, allow_atomic_unlink=True))
    if (
        sink.get("run_id") != run_id
        or sink.get("body_snapshot_hash") != body_snapshot_hash
        or sink.get("terminal") is not False
        or sink.get("action_id") is not None
        or sink.get("daemon_instance_id") is not None
        or any(
            type(sink.get(k)) is not int or sink[k] != 0
            for k in [
                "lease_expires_ns",
                "lease_updates",
                "received_commands",
                "published_velocity_messages",
            ]
        )
    ):
        raise ValueError("negative probe requires original never-admitted zero-command sink")
    for name in ["admission.json", "proof.json"]:
        try:
            os.lstat(root / "execution-control" / name)
        except FileNotFoundError:
            continue
        raise ValueError("never probe a sink with any operator admission or execution proof")
    return sink
