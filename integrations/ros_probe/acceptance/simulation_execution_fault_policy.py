"""Closed SDK operator fault policy; not an Agent tool or execution permission."""

import hashlib
import json
import re


def _closed_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate operator fault field")
        result[key] = value
    return result


def parse_daemon_sigkill_policy(
    raw, *, expected_sha256, run_id, body_snapshot_hash, world_sha256, action_id
):
    if (
        type(raw) is not bytes
        or not 0 < len(raw) <= 4096
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("pinned bounded original operator fault bytes required")
    pins = {
        "run_id": run_id,
        "body_snapshot_hash": body_snapshot_hash,
        "world_sha256": world_sha256,
        "action_id": action_id,
    }
    if any(type(v) is not str or not 0 < len(v) <= 512 for v in pins.values()) or any(
        re.fullmatch("[0-9a-f]{64}", pins[k]) is None
        for k in ["body_snapshot_hash", "world_sha256"]
    ):
        raise ValueError("typed immutable original operator fault pins required")
    policy = json.loads(raw, object_pairs_hook=_closed_object)
    if (
        type(policy) is not dict
        or set(policy) != {"schema_version", "case", "phase", *pins}
        or policy["schema_version"] != "rosclaw.governed_sim_fault.v1"
        or policy["case"] != "daemon_sigkill"
        or policy["phase"] != "during_motion"
        or any(policy[k] != v for k, v in pins.items())
    ):
        raise ValueError("closed Body/run/World/action-bound during-motion SIGKILL policy required")
    return policy
