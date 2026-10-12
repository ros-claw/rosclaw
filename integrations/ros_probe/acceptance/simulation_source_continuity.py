"""Pure passive continuity checks, never admission or permission to move.

The SDK operator must first obtain independent G1/L1 acceptance and pin its
original ECM sample. This helper detects later continuity loss; it does not
validate the initial physical binding or replace current ROS readiness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from rosclaw.connectors.ros.context.navigation_workspace_review import _json
from rosclaw.connectors.ros.context.sim_source_binding import _loaded_inventory

_COUNTERS = ("sim_time_ns", "iterations", "observer_monotonic_ns", "sample")


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer counter or entity identity required")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(sample: dict, world_id: str, robot_entity: str) -> bytes:
    if (
        sample["schema_version"] != "rosclaw.gazebo_ecm_source_sample.v1"
        or sample["inventory_complete"] is not True
        or sample["paused"] is not False
        or type(sample["reset_epoch"]) is not int
        or sample["reset_epoch"] != 0
    ):
        raise ValueError("incomplete, paused or reset source")
    for key in _COUNTERS:
        _integer(sample[key])
    worlds = sample["worlds"]
    if len(worlds) != 1 or worlds[0]["name"] != world_id:
        raise ValueError("World identity changed")
    _integer(worlds[0]["entity"])
    models = sample["models"]
    if not models or len(models) > 256 or len({m["name"] for m in models}) != len(models):
        raise ValueError("unique bounded model inventory required")
    if len({m["entity"] for m in models}) != len(models):
        raise ValueError("model entity identities are not unique")
    robots = [m for m in models if m["name"] == robot_entity]
    if len(robots) != 1 or robots[0]["static"] is not False:
        raise ValueError("unique dynamic robot required")
    facts = []
    for model in models:
        entity = _integer(model["entity"])
        if model["name"] != robot_entity and model["static"] is not True:
            raise ValueError("additional dynamic model")
        # Real navigation changes the robot root pose; static world poses and
        # every member's local geometry/type/pose must remain unchanged.
        pose = model["world_pose"]
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError("complete numeric model pose required")
        import math

        if any(type(v) not in (int, float) or not math.isfinite(v) for v in pose):
            raise ValueError("finite model pose required")
        inventory = _loaded_inventory(sample, entity, allow_plane=model["static"])
        members = []
        for role in ("collisions", "sensors"):
            if type(sample[role]) is not list or len(sample[role]) > 4096:
                raise ValueError("bounded complete member inventory required")
            for row in sample[role]:
                if any(a["entity"] == entity for a in row["ancestry"]):
                    members.append(
                        (role, row["ancestry"][0]["name"], row["name"], _integer(row["entity"]))
                    )
        if len({x[3] for x in members}) != len(members):
            raise ValueError("member entity identities are not unique")
        facts.append(
            {
                "name": model["name"],
                "entity": entity,
                "static": model["static"],
                "world_pose": None if model["name"] == robot_entity else pose,
                "inventory": inventory,
                "member_entities": sorted(members),
            }
        )
    all_members = sample["collisions"] + sample["sensors"]
    if len({_integer(row["entity"]) for row in all_members}) != len(all_members):
        raise ValueError("member entity identities are not globally unique")
    model_entities = {m["entity"] for m in models}
    if any(sum(a["entity"] in model_entities for a in row["ancestry"]) != 1 for row in all_members):
        raise ValueError("every source member needs one known model ancestor")
    return _canonical({"world": worlds[0], "models": sorted(facts, key=lambda x: x["name"])})


@dataclass(frozen=True)
class SimulationSourceContinuityGuard:
    """A terminal passive guard tied to one externally pinned original sample.

    It permits repeated reads of the exact same fresh sample but requires strict
    clock/counter advancement for each new sample. Any failed read permanently
    trips this instance. Recovery needs fresh source/L1 review and a new guard.
    An OK result supplies continuity evidence only, never action authority.
    """

    run_id: str
    world_id: str
    robot_entity: str
    max_age_ns: int
    reference_identity: bytes = field(repr=False)
    last_counters: tuple[int, ...] = field(repr=False)
    last_bytes: bytes = field(repr=False)
    failed_reason: str | None = None

    @classmethod
    def from_pinned_sample(
        cls,
        raw: bytes,
        *,
        expected_sha256: str,
        run_id: str,
        world_id: str,
        robot_entity: str,
        expected_world_entity: int,
        expected_robot_entity: int,
        max_age_ms: int,
        now_monotonic_ns: int,
    ) -> SimulationSourceContinuityGuard:
        if (
            type(raw) is not bytes
            or not 0 < len(raw) <= 500_000
            or hashlib.sha256(raw).hexdigest() != expected_sha256
            or type(max_age_ms) is not int
            or not 0 < max_age_ms <= 500
            or type(run_id) is not str
            or len(run_id) != 32
            or any(c not in "0123456789abcdef" for c in run_id)
            or any(type(v) is not str or not 0 < len(v) <= 128 for v in (world_id, robot_entity))
        ):
            raise ValueError("separate source pin, bounded identities and freshness required")
        sample = _json(raw)
        if sample["observer_run_label"] != run_id:
            raise ValueError("original run mismatch")
        identity = _identity(sample, world_id, robot_entity)
        if sample["worlds"][0]["entity"] != _integer(expected_world_entity) or next(
            m["entity"] for m in sample["models"] if m["name"] == robot_entity
        ) != _integer(expected_robot_entity):
            raise ValueError("independently pinned World/robot entity mismatch")
        guard = cls(
            run_id,
            world_id,
            robot_entity,
            max_age_ms * 1_000_000,
            identity,
            tuple(sample[k] for k in _COUNTERS),
            _canonical(sample),
        )
        if not guard.observe(raw, now_monotonic_ns=now_monotonic_ns)["continuous"]:
            raise ValueError("original source is not fresh")
        return guard

    def observe(self, raw: bytes, *, now_monotonic_ns: int) -> dict:
        if self.failed_reason is None:
            try:
                if type(raw) is not bytes or not 0 < len(raw) <= 500_000:
                    raise ValueError("bounded complete original sample required")
                sample = _json(raw)
                if sample["observer_run_label"] != self.run_id:
                    raise ValueError("run identity changed")
                if (
                    not 0
                    <= _integer(now_monotonic_ns) - _integer(sample["observer_monotonic_ns"])
                    <= self.max_age_ns
                ):
                    raise ValueError("source stopped, stale or future")
                if _identity(sample, self.world_id, self.robot_entity) != self.reference_identity:
                    raise ValueError("World/entity/geometry/sensor inventory changed")
                canonical = _canonical(sample)
                if sample["sample"] == self.last_counters[-1]:
                    if canonical != self.last_bytes:
                        raise ValueError("same counter was reused with different source data")
                elif any(sample[k] <= self.last_counters[i] for i, k in enumerate(_COUNTERS)):
                    raise ValueError("source clock/capture sequence regressed or froze")
                else:
                    object.__setattr__(self, "last_counters", tuple(sample[k] for k in _COUNTERS))
                    object.__setattr__(self, "last_bytes", canonical)
            except (ValueError, TypeError, KeyError, IndexError, StopIteration) as exc:
                object.__setattr__(self, "failed_reason", str(exc))
        return {
            "schema_version": "rosclaw.sim_source_continuity.v1",
            "continuous": self.failed_reason is None,
            "terminal": self.failed_reason is not None,
            "reason": self.failed_reason,
            "run_id": self.run_id,
            "last_sample": self.last_counters[-1],
            "execution_authority": "NONE",
            "L1": "NOT_EVALUATED",
            "L2": "NOT_RUN",
        }
