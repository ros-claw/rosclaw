"""Daemon-owned simulation adapter using the existing ROS action client.

It is deliberately unavailable in REAL/SHADOW. Simulation evidence must come
from an independently configured witness, never from agent-supplied poses.
"""

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from rosclaw.connectors.ros.action_client import STATUS_SUCCEEDED
from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog, digest
from rosclaw.connectors.ros.mission.boundary_pass import rectangular_boundary_targets
from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery
from rosclaw.connectors.ros.mission.repair_optimizer import rank_repair_poses
from rosclaw.connectors.ros.mission.sim_endpoints import (
    absolute_endpoint,
    freeze_sim_endpoints,
    freeze_sim_spawn,
)
from rosclaw.connectors.ros.mission.temporal_recovery import (
    TimePairedRecovery,
    paired_route_candidates,
)
from rosclaw.connectors.ros.verification.brush_timeline import validate_brush_pair
from rosclaw.connectors.ros.verification.coverage import (
    CleaningPose,
    CoverageVerifier,
    point_in_polygon,
)
from rosclaw.connectors.ros.verification.mission import replay_coverage
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting
from rosclaw.contracts.common import content_hash
from rosclaw.kernel import (
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
)

logger = logging.getLogger(__name__)


class SimulationWitness:
    """Read-only receiver on a separate connection from the action client."""

    def __init__(
        self, transport, *, brush_binding=None, observation_topic="/rosclaw_sim/observation"
    ):
        if brush_binding is not None and (
            type(brush_binding) is not dict
            or set(brush_binding)
            != {"run_id", "body_snapshot_hash", "attachment_hash", "producer_id"}
            or any(type(v) is not str or not v for v in brush_binding.values())
        ):
            raise ValueError("frozen independent brush source binding required")
        self.brush_binding = dict(brush_binding) if brush_binding is not None else None
        self.brush_pair_chain = None
        self.observation_topic = absolute_endpoint(observation_topic)
        self.transport = transport
        self.latest = None
        self.samples = []
        self.tracking = False
        self.action_fault = None
        self.receiver_faults = []
        self.lock = threading.Lock()
        self.closed = threading.Event()
        result = transport.send(
            {
                "op": "subscribe",
                "id": "ros-expert-witness",
                "topic": self.observation_topic,
                "type": "std_msgs/msg/String",
            }
        )
        if not result.ok:
            raise ConnectionError(result.error)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _receiver_fault(self, error, raw):
        encoded = (
            raw[:2_000_000].encode("utf-8", errors="backslashreplace") if type(raw) is str else None
        )
        with self.lock:
            self.receiver_faults.append(
                {
                    "error": str(error)[:512],
                    "payload_sha256": hashlib.sha256(encoded[:2_000_000]).hexdigest()
                    if encoded is not None
                    else None,
                    "payload_hash_scope": "bounded_utf8_prefix_max_2MB",
                    "captured_at": datetime.now(UTC).isoformat(),
                }
            )
            del self.receiver_faults[:-128]
            if self.tracking and self.action_fault is None:
                self.action_fault = "independent simulation observation malformed during action"

    def _read(self):
        while not self.closed.is_set():
            result = self.transport.receive(timeout_sec=0.2)
            if not result.ok:
                continue
            data = result.data if result.data is not None else {}
            if type(data) is not dict:
                self._receiver_fault("malformed observer transport envelope", None)
                continue
            if data.get("topic") != self.observation_topic:
                continue
            raw = None
            try:
                raw = data["msg"]["data"]
                if (
                    type(raw) is not str
                    or len(raw) > 2_000_000
                    or len(raw.encode("utf-8")) > 2_000_000
                ):
                    raise ValueError("bounded observation String payload required")
                sample = json.loads(raw)
                if (
                    not isinstance(sample, dict)
                    or sample.get("evidence_domain") != "GAZEBO_PHYSICS"
                ):
                    raise ValueError("independent observation domain missing or different")
                captured = datetime.fromisoformat(sample["captured_at"])
                if (
                    captured.tzinfo is None
                    or not -0.1 <= (datetime.now(UTC) - captured).total_seconds() <= 1
                ):
                    raise ValueError("independent observation capture stale or future-dated")
                if not all(
                    type(sample[k]) in (int, float) and math.isfinite(sample[k])
                    for k in ["x", "y", "yaw", "time_sec"]
                ):
                    raise ValueError("finite observation pose/clock required")
                if sample["time_sec"] < 0:
                    raise ValueError("nonnegative observation SIM clock required")
                if (
                    type(sample.get("cleaning_enabled")) is not bool
                    or type(sample.get("observation_complete")) is not bool
                    or type(sample.get("collision_count")) is not int
                    or sample["collision_count"] < 0
                ):
                    raise ValueError("typed complete collision/brush observation required")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self._receiver_fault(exc, raw)
                continue
            self._record(sample)

    def _record(self, sample):
        with self.lock:
            if getattr(self, "brush_binding", None) is not None:
                try:
                    self.brush_pair_chain = validate_brush_pair(
                        sample,
                        self.brush_binding,
                        previous_chain=getattr(self, "brush_pair_chain", None),
                    )
                except (ValueError, TypeError, KeyError):
                    sample = {
                        **sample,
                        "observation_complete": False,
                        "brush_source_fault": "unbound or unpaired independent brush state",
                    }
            self.latest = (time.monotonic(), sample)
            self.samples.append(sample)
            if self.tracking and self.action_fault is None:
                if sample["observation_complete"] is not True:
                    self.action_fault = "independent simulation witness is incomplete during action"
                elif sample["collision_count"]:
                    self.action_fault = (
                        "independent simulation witness detected collision during action"
                    )

    def fresh(self):
        with self.lock:
            if self.action_fault:
                # A short bad sample must not disappear between 100ms action
                # polls, permitting motion until the next repair batch.
                raise RuntimeError(self.action_fault)
            if not self.latest or time.monotonic() - self.latest[0] > 1:
                raise RuntimeError("independent simulation witness is stale")
            sample = dict(self.latest[1])
            if sample["observation_complete"] is not True:
                raise RuntimeError("independent simulation witness is incomplete")
            if sample["collision_count"]:
                raise RuntimeError("independent simulation witness detected collision")
            return sample

    def mark(self):
        with self.lock:
            self.tracking = True
            self.action_fault = None
            return len(self.samples)

    def since(self, offset):
        with self.lock:
            return [dict(x) for x in self.samples[offset:]]

    def close(self):
        self.closed.set()
        self.transport.close()
        self.thread.join(timeout=1)


class RosCoverageSimulationExecutor:
    """Configured inside rosclawd; action intents cannot select ROS endpoints."""

    def __init__(
        self,
        *,
        owner,
        client,
        control,
        witness,
        output,
        body_id,
        body_snapshot_hash,
        grid,
        recovery_centers=(),
        lease_control=None,
        audit_metadata=None,
        boundary_pass=False,
        boundary_strategy="through_poses",
        boundary_centers=None,
        repair_strategy="greedy",
        repair_swath_yaw=0.0,
        repair_budget_ms=500.0,
        occupancy_binding=None,
        physical_radius_m=None,
        endpoints=None,
        configured_spawn=(0.0, 0.0, 0.0),
        mission_polygon=None,
    ):
        if not owner.startswith("daemon_"):
            raise ValueError("daemon ownership is required")
        if repair_strategy not in {"greedy", "pose_aware", "pose_aware_robust"}:
            raise ValueError("unknown configured SIM repair strategy")
        if not math.isfinite(repair_swath_yaw) or not 0 < repair_budget_ms <= 1000:
            raise ValueError("configured repair yaw/budget must be finite and bounded")
        self.endpoints = freeze_sim_endpoints(endpoints)
        if occupancy_binding is not None and endpoints is not None and "hold" not in endpoints:
            raise ValueError("dynamic configured endpoints require an explicit hold service")
        self.configured_spawn = freeze_sim_spawn(configured_spawn)
        self.repair_strategy = repair_strategy
        self.repair_swath_yaw = repair_swath_yaw
        self.repair_budget_ms = repair_budget_ms
        self.owner, self.client, self.control, self.witness = owner, client, control, witness
        self.output, self.body_id = Path(output), body_id
        self.grid = {**grid, "frame_id": grid.get("frame_id", "map")}
        self.mission_polygon = None
        if mission_polygon is not None:
            if type(mission_polygon) not in (list, tuple) or not 3 <= len(mission_polygon) <= 64:
                raise ValueError("bounded explicitly configured mission polygon required")
            if any(
                type(p) not in (list, tuple)
                or len(p) != 2
                or any(type(v) not in (int, float) or not -1_000_000 <= v <= 1_000_000 for v in p)
                for p in mission_polygon
            ):
                raise ValueError("bounded finite mission polygon coordinates required")
            polygon = [tuple(float(v) for v in p) for p in mission_polygon]
            if polygon[0] == polygon[-1]:
                polygon.pop()
            CoverageVerifier(
                width=1, height=1, resolution=1, accessible_cells=[0], cleaning_polygon=polygon
            )
            configured_grid = CoverageVerifier(**self.grid)
            for i in configured_grid.accessible:
                x = (
                    configured_grid.origin[0]
                    + (i % configured_grid.width + 0.5) * configured_grid.resolution
                )
                y = (
                    configured_grid.origin[1]
                    + (i // configured_grid.width + 0.5) * configured_grid.resolution
                )
                if not point_in_polygon(x, y, polygon):
                    raise ValueError("fixed denominator extends outside approved mission polygon")
            if len(recovery_centers) > 5000 or any(
                type(p) not in (list, tuple)
                or len(p) != 2
                or any(type(v) not in (int, float) or not -1_000_000 <= v <= 1_000_000 for v in p)
                or not point_in_polygon(p[0], p[1], polygon)
                for p in recovery_centers
            ):
                raise ValueError("repair centers must remain within the approved mission polygon")
            if boundary_pass:
                raise ValueError(
                    "rectangular boundary pass is unavailable for an arbitrary mission polygon"
                )
            self.mission_polygon = tuple(polygon)
        self.body_snapshot_hash = body_snapshot_hash
        if occupancy_binding is not None and (
            type(occupancy_binding) is not dict
            or set(occupancy_binding) != {"run_id", "geometry_hash"}
            or any(type(v) is not str or not v for v in occupancy_binding.values())
        ):
            raise ValueError("daemon-configured immutable occupancy binding required")
        self.occupancy_binding = (
            MappingProxyType(dict(occupancy_binding)) if occupancy_binding is not None else None
        )
        if physical_radius_m is not None and (
            type(physical_radius_m) not in (int, float) or not 0 < physical_radius_m <= 10
        ):
            raise ValueError("configured physical Body radius must be finite and bounded")
        if occupancy_binding is not None and recovery_centers and physical_radius_m is None:
            raise ValueError("dynamic repair requires an explicit physical Body radius")
        self.physical_radius_m = physical_radius_m
        self.waiting_for_obstacle = threading.Event()
        self.temporal_recovery_state = None
        self.temporal_mission_deadlines = {}
        self.stopping = threading.Event()
        self.goal_id = None
        self.lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.recovery_centers = tuple(recovery_centers)
        self.lease_control = lease_control or control
        self.lease_lock = self.control_lock if self.lease_control is control else threading.Lock()
        self.audit = None
        self.audit_start = 0
        self.audit_metadata = dict(audit_metadata or {})
        if type(boundary_pass) is not bool:
            raise ValueError("boundary pass must be a configured boolean")
        if boundary_strategy not in ("through_poses", "sequential"):
            raise ValueError("unknown configured boundary strategy")
        self.boundary_pass = boundary_pass
        self.boundary_strategy = boundary_strategy
        self.boundary_centers = tuple(
            recovery_centers if boundary_centers is None else boundary_centers
        )

    def _audit_event(self, kind, payload):
        if self.audit is not None:
            latest = getattr(self.witness, "latest", None)
            sample = latest[1] if isinstance(latest, tuple) else latest or {}
            self.audit.emit(
                kind,
                {
                    **payload,
                    "sample_offset": len(getattr(self.witness, "samples", ())) - self.audit_start,
                },
                sim_time=sample.get("time_sec"),
            )

    def _service(self, name, arguments):
        # One response consumer per connection: concurrent RPC loops would
        # otherwise consume and discard each other's request IDs.
        with self.control_lock:
            return self.control.call_service(
                name, arguments, service_type="std_srvs/srv/SetBool", timeout_sec=0.5
            )

    def _coverage_goal(self, arguments):
        """Bind the rectangular fixture's approved area to its fixed denominator.

        Repair covers the configured whole room. A caller must not authorize a
        smaller polygon and then cause repairs outside that approved area, nor
        shrink the verification denominator to claim whole-room completion.
        """
        frame = self.grid["frame_id"]
        if arguments.get("frame_id", frame) != frame:
            raise ValueError("coverage frame differs from the configured mission frame")
        if self.mission_polygon is not None:
            polygons = arguments.get("polygons", [])
            if type(polygons) is not list or len(polygons) != 1 or type(polygons[0]) is not dict:
                raise ValueError("the configured mission requires exactly its approved polygon")
            points = polygons[0].get("points")
            if type(points) is not list or not 3 <= len(points) <= 65:
                raise ValueError("bounded approved mission points required")
            if any(
                type(p) is not dict
                or any(
                    type(p.get(k)) not in (int, float) or not -1_000_000 <= p[k] <= 1_000_000
                    for k in ("x", "y", "z")
                )
                or p["z"] != 0
                for p in points
            ):
                raise ValueError("bounded finite planar mission points required")
            requested = [(p["x"], p["y"]) for p in points]
            if requested[0] == requested[-1]:
                requested.pop()
            expected = self.mission_polygon
            if len(requested) != len(expected) or not any(
                all(
                    math.isclose(a[0], b[0], rel_tol=0, abs_tol=1e-6)
                    and math.isclose(a[1], b[1], rel_tol=0, abs_tol=1e-6)
                    for a, b in zip(requested, ordered[offset:] + ordered[:offset], strict=True)
                )
                for ordered in (expected, expected[::-1])
                for offset in range(len(expected))
            ):
                raise ValueError("coverage polygon differs from the whole approved mission area")
            canonical = [{"x": x, "y": y, "z": 0.0} for x, y in expected]
            return {"polygons": [{"points": canonical + [dict(canonical[0])]}], "frame_id": frame}
        cells, width, resolution = (
            self.grid["accessible_cells"],
            self.grid["width"],
            self.grid["resolution"],
        )
        left = self.grid["origin"][0] + min(i % width for i in cells) * resolution
        right = self.grid["origin"][0] + (max(i % width for i in cells) + 1) * resolution
        bottom = self.grid["origin"][1] + min(i // width for i in cells) * resolution
        top = self.grid["origin"][1] + (max(i // width for i in cells) + 1) * resolution
        expected = {(left, bottom), (right, bottom), (right, top), (left, top)}
        polygons = arguments.get("polygons", [])
        if len(polygons) != 1:
            raise ValueError("the configured rectangular mission requires one coverage polygon")
        points = [dict(p) for p in polygons[0]["points"]]
        if points and points[0] == points[-1]:
            points.pop()
        if (
            len(points) != 4
            or not all(
                type(p.get(k)) in (int, float) and math.isfinite(p[k])
                for p in points
                for k in ("x", "y", "z")
            )
            or any(p["z"] != 0 for p in points)
        ):
            raise ValueError("coverage requires four finite planar mission corners")
        matched = []
        for p in points:
            corner = next(
                (
                    c
                    for c in expected
                    if math.isclose(p["x"], c[0], rel_tol=0, abs_tol=1e-6)
                    and math.isclose(p["y"], c[1], rel_tol=0, abs_tol=1e-6)
                ),
                None,
            )
            if corner is None:
                raise ValueError("coverage polygon differs from the configured whole-room area")
            matched.append(corner)
        if len(set(matched)) != 4 or any(
            a[0] != b[0] and a[1] != b[1]
            for a, b in zip(matched, matched[1:] + matched[:1], strict=True)
        ):
            raise ValueError("coverage corners are repeated or self-intersecting")
        return {"polygons": [{"points": points + [dict(points[0])]}], "frame_id": frame}

    def _run_goal(
        self, name, action_type, args, goal_id, deadline, *, goal_timeout_sec=45, stage="REPAIR"
    ):
        if self.stopping.is_set() or time.monotonic() >= deadline:
            raise RuntimeError("simulation recovery stopped before dispatch")
        observed = self.witness.fresh()
        remaining = observed.get("lease_remaining_sec")
        # The active mission heartbeat already maintains these states. Require
        # actual fresh actuator/lease evidence rather than repeatedly toggling
        # the same services at every repair goal (and racing the heartbeat).
        enabled_and_leased = (
            observed.get("cleaning_enabled") is True
            and type(remaining) in (int, float)
            and math.isfinite(remaining)
            and remaining > 0.5
        )
        if not enabled_and_leased:
            lease = self._service(self.endpoints["lease"], {"data": True})
            enabled = self._service(self.endpoints["cleaning"], {"data": True})
            if not all(
                r.ok and r.data.get("values", {}).get("success") is True for r in (lease, enabled)
            ):
                raise RuntimeError("simulation recovery actuator did not acknowledge enable")
        done, result = threading.Event(), {}
        goal_deadline = min(deadline, time.monotonic() + goal_timeout_sec)
        with self.lock:
            self.goal_id = goal_id
        self._audit_event("goal_started", {"nav_goal_id": goal_id, "stage": stage, "goal": args})
        self.client.send_goal(
            action=name,
            action_type=action_type,
            args=args,
            goal_id=goal_id,
            on_feedback=lambda data: self._audit_event(
                "nav_feedback", {"nav_goal_id": goal_id, "stage": stage, "values": data}
            ),
            on_result=lambda status, values: (
                result.update(status=status, result=values),
                done.set(),
            ),
        )
        while not done.wait(0.1):
            self.witness.fresh()
            if self.stopping.is_set() or time.monotonic() > deadline:
                self.client.cancel_goal(goal_id)
                done.wait(3)
                raise RuntimeError("simulation action lost lease, interrupted or timed out")
            if time.monotonic() > goal_deadline:
                self.client.cancel_goal(goal_id)
                if not done.wait(3):
                    raise RuntimeError("timed-out repair goal cancellation was not acknowledged")
                result.update(timed_out=True, goal_timeout_sec=goal_timeout_sec)
                self._audit_event("goal_ended", {"nav_goal_id": goal_id, "result": result})
                return result
        self._audit_event("goal_ended", {"nav_goal_id": goal_id, "result": result})
        return result

    def _boundary(self, initial_result, action_id, deadline):
        """Optional legal targets; Nav2 owns paths and the witness owns credit."""
        if not self.boundary_pass:
            return {"status": "DISABLED"}
        if initial_result.get("status") != STATUS_SUCCEEDED:
            result = {"status": "SKIPPED", "reason": "upstream main goal did not succeed"}
            self._audit_event("boundary_decision", result)
            return result
        targets = rectangular_boundary_targets(
            self.boundary_centers,
            self.witness.fresh(),
            edge_midpoints=self.boundary_strategy == "sequential",
        )
        if not targets:
            result = {"status": "SKIPPED", "reason": "legal rectangular corners unavailable"}
            self._audit_event("boundary_decision", result)
            return result
        self._audit_event(
            "boundary_decision",
            {
                "status": "DISPATCH",
                "targets": targets,
                "waypoint_count": len(targets),
                "strategy": self.boundary_strategy,
                "evidence_role": "Nav2 targets, no predicted coverage credit",
            },
        )
        poses = [
            {
                "header": {"frame_id": self.grid["frame_id"]},
                "pose": {
                    "position": {"x": p["x"], "y": p["y"], "z": 0.0},
                    "orientation": {"z": math.sin(p["yaw"] / 2), "w": math.cos(p["yaw"] / 2)},
                },
            }
            for p in targets
        ]
        if self.boundary_strategy == "sequential":
            until = time.monotonic() + 180
            results = []
            for index, pose in enumerate(poses):
                remaining = until - time.monotonic()
                if remaining <= 0:
                    return {
                        "status": "FAILED",
                        "reason": "boundary stage budget exhausted",
                        "waypoint_count": len(targets),
                        "nav_goal_results": results,
                    }
                result = self._run_goal(
                    self.endpoints["navigate_to_pose"],
                    "nav2_msgs/action/NavigateToPose",
                    {"pose": pose},
                    f"{action_id}:boundary:{index}",
                    deadline,
                    goal_timeout_sec=min(45, remaining),
                    stage="BOUNDARY_PASS",
                )
                results.append(result)
                if result.get("status") != STATUS_SUCCEEDED or result.get("timed_out"):
                    return {
                        "status": "FAILED",
                        "waypoint_count": len(targets),
                        "nav_goal_results": results,
                    }
            return {
                "status": "SUCCEEDED",
                "waypoint_count": len(targets),
                "nav_goal_results": results,
            }
        result = self._run_goal(
            self.endpoints["navigate_through_poses"],
            "nav2_msgs/action/NavigateThroughPoses",
            {"poses": poses},
            f"{action_id}:boundary",
            deadline,
            goal_timeout_sec=180,
            stage="BOUNDARY_PASS",
        )
        return {
            "status": "SUCCEEDED" if result.get("status") == STATUS_SUCCEEDED else "FAILED",
            "waypoint_count": len(targets),
            "nav_goal_result": result,
        }

    def _set_obstacle_wait(self, waiting):
        # Set the flag before acquiring the same lock as heartbeat re-enable.
        # An in-flight enable completes before the acknowledged hold is set;
        # every later enable observes the hold flag inside that lock.
        if waiting:
            self.waiting_for_obstacle.set()
        with self.lease_lock:
            response = self.lease_control.call_service(
                self.endpoints["hold"],
                {"data": waiting},
                service_type="std_srvs/srv/SetBool",
                timeout_sec=0.5,
            )
            if not response.ok or response.data.get("values", {}).get("success") is not True:
                raise RuntimeError("SIM drive/brush hold was not acknowledged")
            if not waiting:
                self.waiting_for_obstacle.clear()

    def _repair(
        self,
        verifier,
        started,
        action_id,
        deadline,
        *,
        mission_id=None,
        admission_sim_time=None,
        deadline_sim_time=None,
    ):
        """Bounded missed-cell goals; every connecting path is planned by Nav2."""
        recovery = MissedRegionRecovery(verifier)
        accounting = (
            OccupancyAccounting(verifier, mission_id=mission_id, **self.occupancy_binding)
            if self.occupancy_binding is not None
            else None
        )
        temporal = accounting is not None and self.physical_radius_m is not None
        if temporal:
            recovery = TimePairedRecovery(
                accounting,
                admission_sim_time=admission_sim_time,
                deadline_sim_time=deadline_sim_time,
            )
        records, consumed = [], started
        width, resolution = verifier.width, verifier.resolution
        radius = math.ceil(verifier.radius / resolution)
        cosine, sine = math.cos(math.pi / 4), math.sin(math.pi / 4)
        polygon = [
            (px * cosine - py * sine, px * sine + py * cosine) for px, py in verifier.polygon
        ]
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if point_in_polygon(dx * resolution, dy * resolution, polygon)
        ]
        # Keep the existing minimum for wide cleaners, but allow two footprint
        # passes for smaller configured cleaners. The immutable action deadline
        # and per-cell retry limit still bound every mission.
        goal_budget = min(
            240, max(60, 2 * math.ceil(len(verifier.accessible) / max(1, len(offsets))))
        )
        index = 0
        while index < goal_budget:
            samples = self.witness.since(consumed)
            consumed += len(samples)
            for sample in samples:
                if not sample["observation_complete"] or sample["collision_count"]:
                    raise RuntimeError(
                        "recovery independent observer is incomplete or saw collision"
                    )
                if accounting is not None:
                    accounting.observe_sample(sample)
                else:
                    verifier.observe(
                        CleaningPose(
                            **{
                                k: sample[k]
                                for k in ["x", "y", "yaw", "time_sec", "cleaning_enabled"]
                            }
                        ),
                        frame_id=self.grid["frame_id"],
                    )
            if verifier.result()["coverage_ratio"] >= 0.98:
                break
            self._audit_event(
                "coverage_progress",
                {
                    "stage": "REPAIR",
                    "next_goal_index": index,
                    "coverage_ratio": verifier.result()["coverage_ratio"],
                    "covered_cells": len(verifier.visits),
                    "consumed_samples": consumed - started,
                },
            )
            centers = self.recovery_centers
            if temporal:
                if self.stopping.is_set() or time.monotonic() >= deadline:
                    self.temporal_recovery_state = {
                        "status": "BLOCKED",
                        "reason": "original_wall_deadline_or_stop",
                    }
                    if not self.waiting_for_obstacle.is_set():
                        self._set_obstacle_wait(True)
                    break
                current = samples[-1] if samples else self.witness.fresh()
                if current["time_sec"] != accounting.previous_time:
                    # Consume this newer packet through the accounting loop
                    # before using its position for a route proposal.
                    continue
                centers, reachable, route = paired_route_candidates(
                    accounting,
                    centers,
                    current,
                    physical_radius_m=self.physical_radius_m,
                    budget_ms=self.repair_budget_ms,
                )
                proposals = recovery.propose_at(
                    sim_time_sec=accounting.previous_time,
                    snapshot_sequence=accounting.previous_sequence,
                    reachable_cells=reachable,
                )
                self.temporal_recovery_state = proposals
                self._audit_event(
                    "temporal_recovery_proposal", {**proposals, "route_evidence": route}
                )
                if proposals["status"] == "WAITING_FOR_OBSTACLE":
                    if not self.waiting_for_obstacle.is_set():
                        self._set_obstacle_wait(True)
                    # Freshness/lease/clock are checked again on every poll.
                    # Waiting consumes neither an attempt nor a goal index.
                    self.stopping.wait(0.1)
                    self.witness.fresh()
                    continue
                if not proposals["ready"]:
                    if not self.waiting_for_obstacle.is_set():
                        self._set_obstacle_wait(True)
                    break
                if self.waiting_for_obstacle.is_set():
                    self._set_obstacle_wait(False)
            else:
                proposals = recovery.propose()
            if not proposals["ready"] or not centers:
                break
            # Repair the largest measured hole before spending the bounded
            # budget on isolated boundary cells. Nav2 still owns every path.
            region = max(proposals["ready"], key=lambda row: len(row["cells"]))
            missed = region["cells"]
            if not missed:
                break
            cell = min(missed, key=lambda i: (recovery.attempts.get(i, 0), i))
            x = self.grid["origin"][0] + (cell % self.grid["width"] + 0.5) * self.grid["resolution"]
            y = (
                self.grid["origin"][1]
                + (cell // self.grid["width"] + 0.5) * self.grid["resolution"]
            )
            center = min(centers, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            # Select a stopped cleaning footprint with the highest remaining
            # utility. This selects one repair goal, never a coverage path;
            # predicted cells do not enter the measured coverage accumulator.
            remaining = set(missed)
            if not temporal:
                current = self.witness.fresh()
            best = (-1, -math.inf)
            for candidate in centers:
                col = round((candidate[0] - verifier.origin[0]) / resolution - 0.5)
                row = round((candidate[1] - verifier.origin[1]) / resolution - 0.5)
                score = sum(
                    3 - recovery.attempts.get((row + dy) * width + col + dx, 0)
                    for dx, dy in offsets
                    if 0 <= col + dx < width
                    and 0 <= row + dy < verifier.height
                    and (row + dy) * width + col + dx in remaining
                )
                ranking = (
                    score,
                    -math.hypot(candidate[0] - current["x"], candidate[1] - current["y"]),
                )
                if ranking > best:
                    best, center = ranking, candidate
            heading = math.pi / 4
            if self.repair_strategy in {"pose_aware", "pose_aware_robust"}:
                ready_cells = {c for proposal in proposals["ready"] for c in proposal["cells"]}
                selection = rank_repair_poses(
                    self.grid,
                    centers,
                    ready_cells,
                    current,
                    attempts=recovery.attempts,
                    swath_yaw=self.repair_swath_yaw,
                    budget_ms=self.repair_budget_ms,
                    robust_footprint=self.repair_strategy == "pose_aware_robust",
                )
                self._audit_event(
                    "repair_candidate_selection",
                    {
                        "strategy": self.repair_strategy,
                        "status": selection.status,
                        "elapsed_ms": selection.elapsed_ms,
                        "evaluated_poses": selection.evaluated_poses,
                        "cost_model": selection.cost_model,
                        "reward_model": selection.reward_model,
                        "predicted_sequence": [asdict(pose) for pose in selection.poses],
                        "credit_role": "prediction_only_never_measured_credit",
                        "fallback": selection.status != "READY",
                    },
                )
                if selection.status == "READY":
                    selected = selection.poses[0]
                    center, heading = (selected.x, selected.y), selected.yaw
                    missed = sorted(ready_cells)
            cosine_goal, sine_goal = math.cos(heading), math.sin(heading)
            goal_polygon = [
                (px * cosine_goal - py * sine_goal, px * sine_goal + py * cosine_goal)
                for px, py in verifier.polygon
            ]
            goal_id = f"{action_id}:repair:{index}"
            result = self._run_goal(
                self.endpoints["navigate_to_pose"],
                "nav2_msgs/action/NavigateToPose",
                {
                    "pose": {
                        "header": {"frame_id": self.grid["frame_id"]},
                        "pose": {
                            "position": {"x": center[0], "y": center[1], "z": 0.0},
                            "orientation": {"z": math.sin(heading / 2), "w": math.cos(heading / 2)},
                        },
                    }
                },
                goal_id,
                deadline,
            )
            nearby = [
                i
                for i in missed
                if point_in_polygon(
                    self.grid["origin"][0]
                    + (i % self.grid["width"] + 0.5) * self.grid["resolution"]
                    - center[0],
                    self.grid["origin"][1]
                    + (i // self.grid["width"] + 0.5) * self.grid["resolution"]
                    - center[1],
                    goal_polygon,
                )
            ]
            recovery.record_attempt(nearby or [cell], action_id=goal_id)
            records.append(
                {
                    "goal_id": goal_id,
                    "target": center,
                    "result": result,
                    "coverage_before": verifier.result()["coverage_ratio"],
                }
            )
            logger.warning(
                "Measured missed-cell repair %s: coverage %.4f",
                goal_id,
                verifier.result()["coverage_ratio"],
            )
            index += 1
        return records

    def emergency_stop(self):
        self.stopping.set()
        with self.lock:
            if self.goal_id:
                self.client.cancel_goal(self.goal_id)
        result = self._service(self.endpoints["cleaning"], {"data": False})
        lease = self._service(self.endpoints["lease"], {"data": False})
        return {"acknowledged": result.ok and lease.ok, "physical_stop_verified": False}

    def __call__(self, action):
        if (
            action.execution_mode is not ExecutionMode.SIMULATION
            or action.body_id != self.body_id
            or action.body_snapshot_hash != self.body_snapshot_hash
            or action.capability_id
            not in {
                "coverage.execute",
                "navigation.navigate_to_pose",
                "localization.set_initial_pose",
            }
        ):
            return self._result(ActionState.BLOCKED, errors=[{"code": "SIMULATION_ONLY"}])
        if not self.execution_lock.acquire(blocking=False):
            return self._result(ActionState.BLOCKED, errors=[{"code": "ROS_SIMULATION_BUSY"}])
        try:
            try:
                self.audit = CoverageAuditLog(
                    self.output
                    / f"coverage-audit-{digest(action.action_id)}-{time.time_ns()}.jsonl",
                    context={
                        **self.audit_metadata,
                        "action_id": action.action_id,
                        "body_id": action.body_id,
                        "body_snapshot_hash": self.body_snapshot_hash,
                        "denominator_hash": digest(self.grid),
                    },
                )
                self._audit_event("action_admitted", {"capability_id": action.capability_id})
            except Exception:
                logger.exception(
                    "Passive audit unavailable; canonical verification remains authoritative"
                )
            if action.capability_id == "localization.set_initial_pose":
                return self._localize(action)
            return self._execute(action)
        finally:
            if self.audit is not None:
                try:
                    summary = self.audit.close()
                    Path(summary["path"] + ".summary.json").write_text(json.dumps(summary) + "\n")
                except Exception:
                    logger.exception("Passive audit summary failed")
                finally:
                    self.audit = None
            self.execution_lock.release()

    def _localize(self, action):
        """Initialize only the fixture's configured spawn through rosclawd.

        This is a known startup pose, never a ground-truth feedback controller.
        Refuse initialization after the robot has left the configured spawn.
        """
        try:
            spawn_x, spawn_y, spawn_yaw = self.configured_spawn
            observed = self.witness.fresh()
            if action.arguments:
                raise ValueError("fixture localization accepts only the configured spawn")
            if (
                observed["cleaning_enabled"]
                or observed.get("lease_remaining_sec", 0) > 0
                or math.hypot(observed["x"] - spawn_x, observed["y"] - spawn_y) > 0.02
                or abs((observed["yaw"] - spawn_yaw + math.pi) % (2 * math.pi) - math.pi) > 0.02
            ):
                raise RuntimeError("initial localization requires the stationary configured spawn")
            time.sleep(0.2)
            stationary = self.witness.fresh()
            if (
                math.hypot(stationary["x"] - observed["x"], stationary["y"] - observed["y"]) > 0.001
                or abs(stationary["yaw"] - observed["yaw"]) > 0.001
            ):
                raise RuntimeError("fixture moved during initial localization precheck")
            covariance = [0.0] * 36
            for index in (0, 7, 35):
                covariance[index] = 0.0001
            requested_at = datetime.now(UTC)
            with self.control_lock:
                response = self.control.call_service(
                    self.endpoints["set_initial_pose"],
                    {
                        "pose": {
                            "header": {"frame_id": self.grid["frame_id"]},
                            "pose": {
                                "pose": {
                                    "position": {"x": spawn_x, "y": spawn_y, "z": 0.0},
                                    "orientation": {
                                        "z": math.sin(spawn_yaw / 2),
                                        "w": math.cos(spawn_yaw / 2),
                                    },
                                },
                                "covariance": covariance,
                            },
                        }
                    },
                    service_type="nav2_msgs/srv/SetInitialPose",
                    timeout_sec=2,
                )
            if not response.ok:
                raise RuntimeError("Nav2 initial localization service did not acknowledge")
            deadline = time.monotonic() + min(8, action.verification_policy.timeout_sec)
            while time.monotonic() < deadline:
                observed = self.witness.fresh()
                if self.stopping.is_set():
                    raise RuntimeError("initial localization interrupted")
                localized = observed.get("localization") or {}
                captured = datetime.fromisoformat(
                    localized.get("captured_at", "1970-01-01T00:00:00+00:00")
                )
                if captured >= requested_at and localized.get("frame_id") == self.grid["frame_id"]:
                    distance = math.hypot(localized["x"] - spawn_x, localized["y"] - spawn_y)
                    if distance <= 0.03 and all(
                        localized["covariance"][i] <= 0.002 for i in (0, 7, 35)
                    ):
                        return self._result(
                            ActionState.COMPLETED,
                            accepted=True,
                            verification={
                                "initial_pose_source": "configured_fixture_spawn",
                                "configured_spawn": list(self.configured_spawn),
                                "localization_error_m": distance,
                            },
                            observations=[observed],
                        )
                time.sleep(0.05)
            raise RuntimeError("initial localization was not independently observed")
        except Exception as exc:
            return self._result(
                ActionState.FAILED,
                errors=[{"code": "INITIAL_LOCALIZATION_FAILED", "message": str(exc)}],
            )

    def _temporal_admission(self, action, admission, wall_deadline):
        """A repeated action cannot restart the same frozen mission's clocks."""
        mission_id = action.arguments.get("mission_id")
        duration = action.verification_policy.timeout_sec
        if type(mission_id) is not str or not mission_id or not 0 < duration <= 1800:
            raise ValueError("dynamic mission requires identity and original bounded duration")
        # Verify the actual source's complete same-tick packet before any ON
        # service, using a fresh scratch verifier rather than crediting a pose
        # outside the final retained trajectory.
        accounting = OccupancyAccounting(
            CoverageVerifier(**self.grid), mission_id=mission_id, **self.occupancy_binding
        )
        accounting.observe_sample(admission)
        now_sim = admission["time_sec"]
        if mission_id not in self.temporal_mission_deadlines:
            if len(self.temporal_mission_deadlines) >= 64:
                raise ValueError("dynamic mission admission registry exceeds bound")
            identity = {
                **self.occupancy_binding,
                "mission_id": mission_id,
                "body_snapshot_hash": self.body_snapshot_hash,
            }
            self.output.mkdir(parents=True, exist_ok=True)
            path = self.output / (content_hash("rostemporaladmission", identity) + ".json")
            # Never translate a previous process's monotonic clock or silently
            # restart its budget. A crash/partial file also refuses a new ON.
            try:
                with path.open("x") as admission_record:
                    admission_record.write(
                        json.dumps(
                            {
                                "schema_version": "rosclaw.temporal_mission_admission.v1",
                                "identity": identity,
                                "admission_sim_time": now_sim,
                                "deadline_sim_time": now_sim + duration,
                                "deadline_monotonic": wall_deadline,
                                "captured_at": datetime.now(UTC).isoformat(),
                                "restart_policy": "refuse_same_source_mission_admission_after_process_restart",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except FileExistsError as exc:
                raise ValueError(
                    "previous dynamic mission admission exists; no restarted budget"
                ) from exc
            self.temporal_mission_deadlines[mission_id] = (
                now_sim,
                now_sim + duration,
                wall_deadline,
            )
        original_admission, sim_deadline, original_wall_deadline = self.temporal_mission_deadlines[
            mission_id
        ]
        if (
            now_sim < original_admission
            or now_sim >= sim_deadline
            or time.monotonic() >= original_wall_deadline
        ):
            raise ValueError("original dynamic mission deadline expired or SIM clock reversed")
        return (
            original_admission,
            min(sim_deadline, now_sim + duration),
            min(wall_deadline, original_wall_deadline),
        )

    def _execute(self, action):
        coverage = action.capability_id == "coverage.execute"
        try:
            coverage_goal = self._coverage_goal(action.arguments) if coverage else None
        except (KeyError, TypeError, ValueError) as exc:
            return self._result(
                ActionState.BLOCKED,
                errors=[{"code": "COVERAGE_SCOPE_REJECTED", "message": str(exc)}],
            )
        self.stopping.clear()
        started = self.witness.mark()
        self.audit_start = started
        completed, box = threading.Event(), {}
        goal_id = action.action_id
        dispatched = False
        heartbeat_stop = threading.Event()
        deadline = time.monotonic() + action.verification_policy.timeout_sec
        admission_sim_time = deadline_sim_time = None
        self.temporal_recovery_state = None

        def maintain_lease():
            last_log = time.monotonic()
            count = 0
            while not heartbeat_stop.wait(0.2):
                try:
                    observed = self.witness.fresh()
                    with self.lease_lock:
                        response = self.lease_control.call_service(
                            self.endpoints["lease"],
                            {"data": True},
                            service_type="std_srvs/srv/SetBool",
                            timeout_sec=0.5,
                        )
                    count += 1
                    if time.monotonic() - last_log >= 5:
                        logger.info("Simulation daemon lease renewed %s times", count)
                        last_log = time.monotonic()
                    if (
                        not response.ok
                        and not self.stopping.is_set()
                        and time.monotonic() < deadline
                    ):
                        remaining = self.witness.fresh().get("lease_remaining_sec", 0)
                        if remaining > 0.8:
                            logger.warning(
                                "Transient lease RPC timeout; independent lease remains valid %.3fs",
                                remaining,
                            )
                            continue
                    if (
                        not response.ok
                        or time.monotonic() > deadline
                        or self.stopping.is_set()
                        or deadline_sim_time is not None
                        and observed["time_sec"] >= deadline_sim_time
                    ):
                        logger.error(
                            "Simulation heartbeat stopped: ok=%s error=%s deadline=%s stopping=%s",
                            response.ok,
                            response.error,
                            time.monotonic() > deadline,
                            self.stopping.is_set(),
                        )
                        self.stopping.set()
                        break
                    if (
                        coverage
                        and not observed["cleaning_enabled"]
                        and not heartbeat_stop.is_set()
                    ):
                        with self.lease_lock:
                            if self.waiting_for_obstacle.is_set():
                                continue
                            enabled = self.lease_control.call_service(
                                self.endpoints["cleaning"],
                                {"data": True},
                                service_type="std_srvs/srv/SetBool",
                                timeout_sec=0.3,
                            )
                        if not enabled.ok or not enabled.data.get("values", {}).get("success"):
                            raise RuntimeError("active simulation cleaning re-enable rejected")
                except Exception:
                    logger.exception("Simulation heartbeat observer/service failure")
                    self.stopping.set()
                    break

        heartbeat_thread = threading.Thread(target=maintain_lease, daemon=True)
        try:
            admission = self.witness.fresh()
            if (
                coverage
                and self.occupancy_binding is not None
                and self.physical_radius_m is not None
            ):
                admission_sim_time, deadline_sim_time, deadline = self._temporal_admission(
                    action, admission, deadline
                )
            self._service(self.endpoints["lease"], {"data": True})
            if deadline_sim_time is not None:
                self._set_obstacle_wait(False)
            heartbeat_thread.start()
            if coverage:
                response = self._service(self.endpoints["cleaning"], {"data": True})
                if not response.ok or not response.data.get("values", {}).get("success"):
                    raise RuntimeError("simulated cleaning actuator rejected enable")
                name, action_type = (
                    self.endpoints["navigate_complete_coverage"],
                    "opennav_coverage_msgs/action/NavigateCompleteCoverage",
                )
                args = coverage_goal
            else:
                name, action_type = (
                    self.endpoints["navigate_to_pose"],
                    "nav2_msgs/action/NavigateToPose",
                )
                args = {"pose": action.arguments["pose"]}
            with self.lock:
                self.goal_id = goal_id
            self._audit_event(
                "goal_started",
                {
                    "nav_goal_id": goal_id,
                    "stage": "MAIN_COVERAGE" if coverage else "NAVIGATION",
                    "goal": args,
                },
            )
            self.client.send_goal(
                action=name,
                action_type=action_type,
                args=args,
                goal_id=goal_id,
                on_feedback=lambda data: self._audit_event(
                    "nav_feedback",
                    {
                        "nav_goal_id": goal_id,
                        "stage": "MAIN_COVERAGE" if coverage else "NAVIGATION",
                        "values": data,
                    },
                ),
                on_result=lambda status, result: (
                    box.update(status=status, result=result),
                    completed.set(),
                ),
            )
            dispatched = True
            while not completed.wait(0.1):
                self.witness.fresh()
                if self.stopping.is_set() or time.monotonic() > deadline:
                    self.client.cancel_goal(goal_id)
                    completed.wait(3)
                    raise RuntimeError("simulation action interrupted or timed out")
            time.sleep(0.2)
            samples = self.witness.since(started)
            final = self.witness.fresh()
            if box.get("status") != STATUS_SUCCEEDED and not coverage:
                raise RuntimeError(f"ROS action did not succeed: {box}")
            if not samples or not all(s.get("observation_complete") is True for s in samples):
                raise RuntimeError("independent observation is incomplete")
            collision_count = max(s["collision_count"] for s in samples)
            if collision_count != 0:
                raise RuntimeError("independent observer detected a collision")
            if coverage:
                self._audit_event(
                    "goal_ended",
                    {"nav_goal_id": goal_id, "result": box, "consumed_samples": len(samples)},
                )
                boundary_result = self._boundary(box, action.action_id, deadline)
                if self.boundary_pass:
                    self._audit_event(
                        "primary_completed",
                        {
                            "consumed_samples": len(self.witness.since(started)),
                            "boundary_result": boundary_result,
                        },
                    )
                verifier = CoverageVerifier(**self.grid)
                if self.occupancy_binding is not None:
                    repairs = self._repair(
                        verifier,
                        started,
                        action.action_id,
                        deadline,
                        mission_id=action.arguments["mission_id"],
                        admission_sim_time=admission_sim_time,
                        deadline_sim_time=deadline_sim_time,
                    )
                else:
                    repairs = self._repair(verifier, started, action.action_id, deadline)
                samples = self.witness.since(started)
                final = self.witness.fresh()
                collision_count = max(s["collision_count"] for s in samples)
                if collision_count or not all(s["observation_complete"] for s in samples):
                    raise RuntimeError(
                        "final independent evidence is incomplete or contains collision"
                    )
                evidence = {
                    "mission_id": action.arguments["mission_id"],
                    "body_id": action.body_id,
                    "action_ids": [action.action_id],
                    "grid": self.grid,
                    "frame_id": self.grid["frame_id"],
                    "trajectory": [
                        {k: s[k] for k in ["x", "y", "yaw", "time_sec", "cleaning_enabled"]}
                        for s in samples
                    ],
                    "collision": {
                        "collision_count": collision_count,
                        "observation_complete": True,
                        "source": final.get("collision_source", "gazebo_ground_truth_geometry"),
                    },
                }
                if self.occupancy_binding is not None:
                    evidence.update(
                        schema_version="rosclaw.time_paired_mission_evidence.v1",
                        occupancy_binding=dict(self.occupancy_binding),
                        occupancy_samples=[
                            {k: s[k] for k in ("occupancy", "occupancy_hash")} for s in samples
                        ],
                    )
                brush_binding = getattr(self.witness, "brush_binding", None)
                if brush_binding is not None:
                    evidence["brush_evidence_binding"] = dict(brush_binding)
                    evidence["brush_evidence_samples"] = [
                        {
                            k: s[k]
                            for k in (
                                "brush_state_pair",
                                "brush_source_binding",
                                "brush_source_fault",
                            )
                        }
                        for s in samples
                    ]
                calculated, temporal = replay_coverage(evidence)
                ratio = calculated["coverage_ratio"]
                self._audit_event(
                    "coverage_final", {"coverage_ratio": ratio, "sample_count": len(samples)}
                )
                if calculated["trace_gaps"]:
                    raise RuntimeError("independent cleaning trajectory contains unverified gaps")
                self.output.mkdir(parents=True, exist_ok=True)
                filename = content_hash("rosevidence", action.action_id) + ".json"
                path = self.output / filename
                path.write_text(json.dumps(evidence, indent=2) + "\n")
                verification = {
                    "mission_id": evidence["mission_id"],
                    "coverage_ratio": ratio,
                    "recovery_attempts": repairs,
                    "initial_action_result": box,
                    "boundary_action_result": boundary_result,
                    "independent_evidence_hashes": [content_hash("rosmissionevidence", evidence)],
                    "evidence_artifact": {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                }
                if temporal is not None:
                    verification["time_paired_accounting"] = temporal
                if self.temporal_recovery_state is not None:
                    verification["temporal_recovery"] = self.temporal_recovery_state
            else:
                target = args["pose"]["pose"]["position"]
                distance = math.hypot(final["x"] - target["x"], final["y"] - target["y"])
                if distance > 0.15:
                    raise RuntimeError("ground truth did not reach navigation goal")
                verification = {"goal_distance_m": distance, "collision_count": collision_count}
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            disabled = self._service(self.endpoints["cleaning"], {"data": False})
            if not disabled.ok or not disabled.data.get("values", {}).get("success"):
                raise RuntimeError("simulated cleaning disable was not acknowledged")
            return self._result(
                ActionState.BLOCKED
                if coverage and ratio < 0.98 and self.temporal_recovery_state is not None
                else ActionState.DEGRADED
                if coverage and ratio < 0.98
                else ActionState.COMPLETED,
                verification=verification,
                observations=[final],
                accepted=dispatched,
            )
        except Exception as exc:
            with self.lock:
                if self.goal_id:
                    self.client.cancel_goal(self.goal_id)
            # Preserve failed physical observations for diagnosis; this artifact
            # carries no verification claim and cannot authorize memory success.
            failure_artifact = None
            try:
                self.output.mkdir(parents=True, exist_ok=True)
                failure_path = self.output / (
                    content_hash("rosevidence", action.action_id) + ".failed.json"
                )
                failure_path.write_text(
                    json.dumps(
                        {
                            "action_id": action.action_id,
                            "body_id": action.body_id,
                            "verification_status": "NOT_VERIFIED",
                            "error": str(exc),
                            "observations": self.witness.since(started),
                            "receiver_faults": list(getattr(self.witness, "receiver_faults", ())),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                failure_artifact = {
                    "path": str(failure_path),
                    "sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
                }
            except Exception:
                logger.exception("Could not persist failed simulation observations")
            return self._result(
                ActionState.FAILED,
                accepted=dispatched,
                verification={"failure_artifact": failure_artifact},
                errors=[{"code": "ROS_SIMULATION_FAILED", "message": str(exc)}],
            )
        finally:
            heartbeat_stop.set()
            if heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=2)
            for service in [self.endpoints["cleaning"], self.endpoints["lease"]]:
                try:
                    self._service(service, {"data": False})
                except Exception:
                    logger.exception(
                        "Simulation cleanup failed; deadman expires within 1.5 seconds"
                    )
            with self.lock:
                self.goal_id = None
            self._audit_event("cleanup_attempted", {"stage": "CLEANUP"})

    def _result(self, state, *, verification=None, observations=None, errors=None, accepted=False):
        return ActionExecutionResult(
            final_state=state,
            evidence_level=EvidenceLevel.TASK_VERIFIED
            if state is ActionState.COMPLETED
            else EvidenceLevel.REQUESTED,
            evidence_domain=EvidenceDomain.SIMULATION,
            policy_decision={
                "allowed": state is not ActionState.BLOCKED,
                "reason": "configured_gazebo_only",
            },
            simulation_result={"has_physics": True, "engine": "gazebo"},
            dispatch_result={"accepted": accepted, "owner": self.owner},
            observations=observations or [],
            verification_result=verification or {},
            errors=errors or [],
        )
