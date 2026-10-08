"""Additional SIM actor interlock. Source correspondence is not DDS authentication.

The frozen P0 launcher does not enable this interlock. New owned launchers must
pin the binding and producer policy, provide the independent observer stream,
and verify actual Body/world admission and closed original-source replay.
"""

import json
import math

CONFIG_KEYS = {"run_id", "body_snapshot_hash", "constraint_policy_hash"}
WIRE_KEYS = CONFIG_KEYS | {
    "schema_version",
    "sequence",
    "sampled_monotonic_sec",
    "live_source_constraint_satisfied",
    "robot_sim_time_sec",
    "probe_sim_time_sec",
    "robot_collision_count",
    "probe_completed_cache_cycles",
    "source_fault",
    "authorization",
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate interlock JSON key")
        result[key] = value
    return result


def nonfinite_constant(value):
    raise ValueError("nonfinite JSON constant in actor interlock")


def decode(raw):
    if type(raw) is not str or len(raw.encode("utf-8")) > 4096:
        raise ValueError("bounded original interlock JSON required")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=nonfinite_constant)


def configuration(config, binding):
    if type(config) is not dict or set(config) != CONFIG_KEYS:
        raise ValueError("exact frozen actor interlock configuration required")
    if any(type(v) is not str or not v or len(v) > 256 for v in config.values()):
        raise ValueError("bounded nonempty actor interlock identities required")
    if any(config[k] != binding[k] for k in ("run_id", "body_snapshot_hash")):
        raise ValueError("actor interlock and brush Body/run differ")
    value = config["constraint_policy_hash"]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("pinned SHA256 observation policy required")
    return config.copy()


class ActuatorObservationConstraint:
    def __init__(self, config, binding):
        self.config = configuration(config, binding)
        self.last = None
        self.opened = False
        self.fault = None

    def receive(self, raw, received_monotonic_sec):
        try:
            row = decode(raw)
            if type(row) is not dict or set(row) != WIRE_KEYS:
                raise ValueError("exact actor interlock envelope required")
            if any(row[k] != v for k, v in self.config.items()):
                raise ValueError("actor interlock source/Body/run mismatch")
            if row["schema_version"] != "rosclaw.actor_observation_interlock.v1":
                raise ValueError("unsupported actor interlock envelope")
            if row["authorization"] is not False:
                raise ValueError("observation cannot authorize robot actions")
            if type(row["live_source_constraint_satisfied"]) is not bool:
                raise ValueError("explicit observation constraint boolean required")
            seq = row["sequence"]
            if (
                type(seq) is not int
                or seq < 0
                or (seq != self.last["sequence"] + 1 if self.last else seq != 0)
            ):
                raise ValueError("contiguous actor observation sequence required")
            stamp = row["sampled_monotonic_sec"]
            if not finite(stamp) or not finite(received_monotonic_sec):
                raise ValueError("finite original monotonic clock required")
            if not 0 <= received_monotonic_sec - stamp <= 0.15:
                raise ValueError("stale or future actor observation")
            if self.last and stamp <= self.last["sampled_monotonic_sec"]:
                raise ValueError("actor observation clock did not advance")
            for key in ("robot_collision_count", "probe_completed_cache_cycles"):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError("integer original observation counters required")
            fault = row["source_fault"]
            if fault is not None and (type(fault) is not str or not fault or len(fault) > 512):
                raise ValueError("bounded explicit observation fault required")
            ready = row["live_source_constraint_satisfied"]
            if ready and (
                fault is not None
                or row["robot_collision_count"] != 0
                or row["probe_completed_cache_cycles"] < 1
                or not finite(row["robot_sim_time_sec"])
                or not finite(row["probe_sim_time_sec"])
                or abs(row["robot_sim_time_sec"] - row["probe_sim_time_sec"]) > 0.15
            ):
                raise ValueError("observation admission contradicts its source counters/clocks")
            if fault or row["robot_collision_count"] or (self.opened and not ready):
                raise ValueError(fault or "robot contact or observation lost after admission")
            self.last = row
        except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
            self.fault = self.fault or str(exc)
            return False
        return self.fault is None

    def ready(self, wall_time, actor_sim_time):
        if not finite(wall_time) or not finite(actor_sim_time):
            self.fault = self.fault or "invalid actor observation sample clock"
        ready = False
        if self.last and self.fault is None:
            row = self.last
            ready = bool(
                row["live_source_constraint_satisfied"]
                and 0 <= wall_time - row["sampled_monotonic_sec"] <= 0.15
                and abs(actor_sim_time - row["robot_sim_time_sec"]) <= 0.15
            )
            if self.opened and not ready:
                self.fault = "actor observation stale or SIM clock disagrees after admission"
        if ready and self.fault is None:
            self.opened = True
            return True
        return False


def observation_envelope(snapshot, binding, sequence):
    """Transport projection only; producer must retain original source audit."""
    row = {
        "schema_version": "rosclaw.actor_observation_interlock.v1",
        "run_id": binding["run_id"],
        "body_snapshot_hash": binding["body_snapshot_hash"],
        "constraint_policy_hash": snapshot["constraint_policy_hash"],
        "sequence": sequence,
        "authorization": False,
    }
    for key in WIRE_KEYS - set(row):
        row[key] = snapshot[key]
    return row
