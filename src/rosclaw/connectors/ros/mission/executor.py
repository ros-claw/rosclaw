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
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.connectors.ros.action_client import STATUS_SUCCEEDED
from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog, digest
from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery
from rosclaw.connectors.ros.verification.coverage import (
    CleaningPose,
    CoverageVerifier,
    point_in_polygon,
)
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

    def __init__(self, transport):
        self.transport = transport
        self.latest = None
        self.samples = []
        self.tracking = False
        self.action_fault = None
        self.lock = threading.Lock()
        self.closed = threading.Event()
        result = transport.send(
            {
                "op": "subscribe",
                "id": "ros-expert-witness",
                "topic": "/rosclaw_sim/observation",
                "type": "std_msgs/msg/String",
            }
        )
        if not result.ok:
            raise ConnectionError(result.error)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.closed.is_set():
            result = self.transport.receive(timeout_sec=0.2)
            if not result.ok:
                continue
            data = result.data or {}
            if data.get("topic") != "/rosclaw_sim/observation":
                continue
            try:
                sample = json.loads(data["msg"]["data"])
                if (
                    not isinstance(sample, dict)
                    or sample.get("evidence_domain") != "GAZEBO_PHYSICS"
                ):
                    continue
                captured = datetime.fromisoformat(sample["captured_at"])
                if (
                    captured.tzinfo is None
                    or not -0.1 <= (datetime.now(UTC) - captured).total_seconds() <= 1
                ):
                    continue
                if not all(
                    type(sample[k]) in (int, float) and math.isfinite(sample[k])
                    for k in ["x", "y", "yaw", "time_sec"]
                ):
                    continue
                if (
                    type(sample.get("cleaning_enabled")) is not bool
                    or type(sample.get("observation_complete")) is not bool
                    or type(sample.get("collision_count")) is not int
                    or sample["collision_count"] < 0
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            self._record(sample)

    def _record(self, sample):
        with self.lock:
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
    ):
        if not owner.startswith("daemon_"):
            raise ValueError("daemon ownership is required")
        self.owner, self.client, self.control, self.witness = owner, client, control, witness
        self.output, self.body_id, self.grid = Path(output), body_id, grid
        self.body_snapshot_hash = body_snapshot_hash
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

    def _run_goal(self, name, action_type, args, goal_id, deadline, *, goal_timeout_sec=45):
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
            lease = self._service("/rosclaw_sim/lease", {"data": True})
            enabled = self._service("/rosclaw_sim/cleaning", {"data": True})
            if not all(
                r.ok and r.data.get("values", {}).get("success") is True for r in (lease, enabled)
            ):
                raise RuntimeError("simulation recovery actuator did not acknowledge enable")
        done, result = threading.Event(), {}
        goal_deadline = min(deadline, time.monotonic() + goal_timeout_sec)
        with self.lock:
            self.goal_id = goal_id
        self._audit_event("goal_started", {"nav_goal_id": goal_id, "stage": "REPAIR", "goal": args})
        self.client.send_goal(
            action=name,
            action_type=action_type,
            args=args,
            goal_id=goal_id,
            on_feedback=lambda data: self._audit_event(
                "nav_feedback", {"nav_goal_id": goal_id, "stage": "REPAIR", "values": data}
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

    def _repair(self, verifier, started, action_id, deadline):
        """Bounded missed-cell goals; every connecting path is planned by Nav2."""
        recovery = MissedRegionRecovery(verifier)
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
        for index in range(goal_budget):
            samples = self.witness.since(consumed)
            consumed += len(samples)
            for sample in samples:
                if not sample["observation_complete"] or sample["collision_count"]:
                    raise RuntimeError(
                        "recovery independent observer is incomplete or saw collision"
                    )
                verifier.observe(
                    CleaningPose(
                        **{k: sample[k] for k in ["x", "y", "yaw", "time_sec", "cleaning_enabled"]}
                    ),
                    frame_id="map",
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
            proposals = recovery.propose()
            if not proposals["ready"] or not self.recovery_centers:
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
            center = min(self.recovery_centers, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            # Select a stopped cleaning footprint with the highest remaining
            # utility. This selects one repair goal, never a coverage path;
            # predicted cells do not enter the measured coverage accumulator.
            remaining = set(missed)
            current = self.witness.fresh()
            best = (-1, -math.inf)
            for candidate in self.recovery_centers:
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
            goal_id = f"{action_id}:repair:{index}"
            result = self._run_goal(
                "/navigate_to_pose",
                "nav2_msgs/action/NavigateToPose",
                {
                    "pose": {
                        "header": {"frame_id": "map"},
                        "pose": {
                            "position": {"x": center[0], "y": center[1], "z": 0.0},
                            "orientation": {"z": math.sin(math.pi / 8), "w": math.cos(math.pi / 8)},
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
                    polygon,
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
        return records

    def emergency_stop(self):
        self.stopping.set()
        with self.lock:
            if self.goal_id:
                self.client.cancel_goal(self.goal_id)
        result = self._service("/rosclaw_sim/cleaning", {"data": False})
        lease = self._service("/rosclaw_sim/lease", {"data": False})
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
                    self.output / f"coverage-audit-{action.action_id}-{time.time_ns()}.jsonl",
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
            observed = self.witness.fresh()
            if action.arguments:
                raise ValueError("fixture localization accepts only the configured spawn")
            if (
                observed["cleaning_enabled"]
                or observed.get("lease_remaining_sec", 0) > 0
                or math.hypot(observed["x"], observed["y"]) > 0.02
                or abs((observed["yaw"] + math.pi) % (2 * math.pi) - math.pi) > 0.02
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
                    "/set_initial_pose",
                    {
                        "pose": {
                            "header": {"frame_id": "map"},
                            "pose": {
                                "pose": {
                                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                                    "orientation": {"w": 1.0},
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
                if captured >= requested_at and localized.get("frame_id") == "map":
                    distance = math.hypot(localized["x"], localized["y"])
                    if distance <= 0.03 and all(
                        localized["covariance"][i] <= 0.002 for i in (0, 7, 35)
                    ):
                        return self._result(
                            ActionState.COMPLETED,
                            accepted=True,
                            verification={
                                "initial_pose_source": "configured_fixture_spawn",
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

        def maintain_lease():
            last_log = time.monotonic()
            count = 0
            while not heartbeat_stop.wait(0.2):
                try:
                    observed = self.witness.fresh()
                    with self.lease_lock:
                        response = self.lease_control.call_service(
                            "/rosclaw_sim/lease",
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
                    if not response.ok or time.monotonic() > deadline or self.stopping.is_set():
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
                            enabled = self.lease_control.call_service(
                                "/rosclaw_sim/cleaning",
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
            self.witness.fresh()
            self._service("/rosclaw_sim/lease", {"data": True})
            heartbeat_thread.start()
            if coverage:
                response = self._service("/rosclaw_sim/cleaning", {"data": True})
                if not response.ok or not response.data.get("values", {}).get("success"):
                    raise RuntimeError("simulated cleaning actuator rejected enable")
                name, action_type = (
                    "/navigate_complete_coverage",
                    "opennav_coverage_msgs/action/NavigateCompleteCoverage",
                )
                args = coverage_goal
            else:
                name, action_type = "/navigate_to_pose", "nav2_msgs/action/NavigateToPose"
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
                verifier = CoverageVerifier(**self.grid)
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
                    "frame_id": "map",
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
                verifier = CoverageVerifier(**self.grid)
                for pose in evidence["trajectory"]:
                    verifier.observe(CleaningPose(**pose), frame_id="map")
                ratio = verifier.result()["coverage_ratio"]
                self._audit_event(
                    "coverage_final", {"coverage_ratio": ratio, "sample_count": len(samples)}
                )
                if verifier.result()["trace_gaps"]:
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
                    "independent_evidence_hashes": [content_hash("rosmissionevidence", evidence)],
                    "evidence_artifact": {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                }
            else:
                target = args["pose"]["pose"]["position"]
                distance = math.hypot(final["x"] - target["x"], final["y"] - target["y"])
                if distance > 0.15:
                    raise RuntimeError("ground truth did not reach navigation goal")
                verification = {"goal_distance_m": distance, "collision_count": collision_count}
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            disabled = self._service("/rosclaw_sim/cleaning", {"data": False})
            if not disabled.ok or not disabled.data.get("values", {}).get("success"):
                raise RuntimeError("simulated cleaning disable was not acknowledged")
            return self._result(
                ActionState.DEGRADED if coverage and ratio < 0.98 else ActionState.COMPLETED,
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
            for service in ["/rosclaw_sim/cleaning", "/rosclaw_sim/lease"]:
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
