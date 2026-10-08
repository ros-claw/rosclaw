"""Bounded exact-step join of original scene and robot/instrument packets.

Transport ordering cannot turn a later or interpolated observation into a
same-step counterpart. This adds a spatial constraint, without admitting world
ownership, executing scene services or providing robot authority.
"""

import math

from native_contact_evidence import decode_native_packet
from probe_scene_geometry import decode_scene_json

from rosclaw.connectors.ros.verification.occupancy_geometry import parse_physics_packet


class ProbeSceneJoin:
    def __init__(self, geometry):
        self.geometry = geometry
        self.buffers = {kind: {} for kind in ("scene", "robot", "probe")}
        self.last_source = {}
        self.last_receipt = None
        self.last_join_receipts = None
        self.fault = None
        self.completed = 0

    def receive(self, kind, raw, *, received_monotonic_sec, received_unix_ns):
        try:
            if self.fault:
                raise ValueError("original spatial join rejection remains latched")
            wall, unix = received_monotonic_sec, received_unix_ns
            if (
                kind not in self.buffers
                or type(raw) is not bytes
                or not 0 < len(raw) <= 262144
                or type(wall) not in (int, float)
                or not math.isfinite(wall)
                or wall < 0
                or type(unix) is not int
                or unix <= 0
                or (
                    self.last_receipt is not None
                    and (wall < self.last_receipt[0] or unix < self.last_receipt[1])
                )
            ):
                raise ValueError("bounded original spatial source and ordered receipt required")
            self.last_receipt = wall, unix
            if kind == "scene":
                decode_scene_json(raw)
                b = self.geometry.binding
                packet = parse_physics_packet(
                    raw,
                    **{
                        k: b[k]
                        for k in (
                            "run_id",
                            "body_snapshot_hash",
                            "attachment_hash",
                            "world_name",
                            "body_model_name",
                        )
                    },
                    obstacle_names=tuple(b["obstacle_names"]),
                    scene_model_names=frozenset(b["scene_model_names"]),
                    received_at_unix_ns=unix,
                    maximum_body_planar_radius_m=self.geometry.declaration[
                        "maximum_robot_radius_m"
                    ],
                )["packet"]
            else:
                gate = self.geometry.gate
                policy = (
                    gate.robot.policy if kind == "robot" else gate.probe.policy["native_policy"]
                )
                packet, _ = decode_native_packet(raw, policy)
            stamp = packet["sim_time_sec"]
            # Every packet keeps its original bytes; rounding only indexes an
            # integer-nanosecond physics clock. Geometry still checks originals.
            key = round(stamp * 1_000_000_000)
            sequence = packet["sequence"]
            previous = self.last_source.get(kind)
            if previous is not None and (key <= previous[0] or sequence != previous[1] + 1):
                raise ValueError("original spatial source step/sequence missing or regressed")
            if not 0 <= unix - packet["captured_at_unix_ns"] < 300_000_000:
                raise ValueError("original spatial source stale or future at receipt")
            self.last_source[kind] = key, sequence
            self.buffers[kind][key] = (raw, wall, unix)
            self._expire(wall)
            results = []
            while self.buffers["scene"]:
                step = next(iter(self.buffers["scene"]))
                if any(step not in self.buffers[k] for k in ("robot", "probe")):
                    break
                sources = {k: self.buffers[k][step] for k in self.buffers}
                result = self.geometry.observe(
                    sources["scene"][0],
                    received_monotonic_sec=sources["scene"][1],
                    received_unix_ns=unix,
                    robot_component_bytes=sources["robot"][0],
                    probe_component_bytes=sources["probe"][0],
                )
                self.last_join_receipts = {k: value[1] for k, value in sources.items()}
                self.completed += 1
                results.append(
                    {
                        "original_sim_step_ns": step,
                        "original_source_receipts": {
                            k: {"monotonic_sec": value[1], "unix_ns": value[2]}
                            for k, value in sources.items()
                        },
                        "verified_monotonic_sec": wall,
                        "projection": result,
                    }
                )
                for k in self.buffers:
                    del self.buffers[k][step]
            if any(len(buf) > (8 if k == "scene" else 64) for k, buf in self.buffers.items()):
                raise ValueError("bounded original spatial join backlog exceeded")
            return results
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.fault = self.fault or str(exc)
            self.geometry.fault = self.geometry.fault or self.fault
            raise ValueError(f"original spatial join rejected: {exc}") from exc

    def _expire(self, wall):
        scenes = self.buffers["scene"]
        if any(wall - row[1] >= 0.3 for row in scenes.values()):
            raise ValueError("original scene lacks timely exact robot/probe counterparts")
        # All-step robot observations between the 20 Hz scene observations
        # belong to the independent contact gate, not to this sampled join.
        for kind in ("robot", "probe"):
            for key, row in tuple(self.buffers[kind].items()):
                if wall - row[1] >= 0.3:
                    if key in scenes:
                        raise ValueError("original spatial counterpart expired before join")
                    del self.buffers[kind][key]

    def snapshot(self, wall):
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            self.fault = self.fault or "invalid spatial join sample clock"
        elif self.last_receipt is not None and wall < self.last_receipt[0]:
            self.fault = self.fault or "spatial join sample precedes original receipt"
        else:
            try:
                self._expire(wall)
            except ValueError as exc:
                self.fault = self.fault or str(exc)
            if self.last_join_receipts and any(
                wall - receipt >= 0.3 for receipt in self.last_join_receipts.values()
            ):
                self.fault = self.fault or "oldest original joined spatial source expired"
        if self.fault:
            self.geometry.fault = self.geometry.fault or self.fault
        result = self.geometry.snapshot(wall)
        return {
            **result,
            "completed_exact_scene_joins": self.completed,
            "pending_scene_steps": len(self.buffers["scene"]),
            "join_source_fault": self.fault,
            "sampling_semantics": "SCENE_20HZ_EXACT_STEP_NOT_ALL_STEP_CONTACT_PROOF",
        }
