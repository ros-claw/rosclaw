"""One daemon-owned, bounded public coverage slice.

Register this executor and its stop method inside the trusted Driver process.
Its port executes primitives internally under the OUTER canonical action's
existing resource lease. It must not submit nested daemon actions. There is
no Agent ROS transport, new task kernel, permit issuer or whole-house judge.
FIXTURE and SIMULATION are supported; this is not a verified REAL executor.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from rosclaw.body.schema import EffectiveBody
from rosclaw.connectors.ros.mission.finite_coverage_evidence import persist_archive, snapshot_record
from rosclaw.connectors.ros.public_coverage.coverage_paths import prepare_work_path
from rosclaw.connectors.ros.public_coverage.graph_ledger import IDENTITY, GraphAnchoredCoverage
from rosclaw.connectors.ros.public_coverage.grid import Grid
from rosclaw.connectors.ros.public_coverage.online import propose_transfer
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
    OrphanPolicy,
)


@dataclass(frozen=True)
class CoverageSource:
    role: str
    topic: str
    message_type: str
    frame_id: str | None
    max_receive_age_s: float
    max_header_age_s: float | None
    child_frame_id: str | None = None
    publisher_gid: str | None = None
    max_header_future_s: float = 0.1

    def __post_init__(self):
        if not all(type(v) is str and v for v in (self.role, self.topic, self.message_type)):
            raise ValueError("Explicit typed public observation source required")
        if not self.topic.startswith("/") or "/msg/" not in self.message_type:
            raise ValueError("Absolute ROS topic and message interface required")
        if (
            type(self.publisher_gid) is not str
            or not re.fullmatch(r"(?:[a-f0-9]{32}|[a-f0-9]{48})", self.publisher_gid)
            or not any(c != "0" for c in self.publisher_gid)
        ):
            raise ValueError("Pinned public publisher endpoint GID required")
        if self.message_type == "std_msgs/msg/Bool":
            if self.frame_id is not None or self.max_header_age_s is not None:
                raise ValueError("Headerless brush state must not invent a frame or header clock")
        elif self.message_type == "visualization_msgs/msg/MarkerArray":
            if type(self.frame_id) is not str or not self.frame_id:
                raise ValueError("Public graph event requires its common declared frame")
        elif (
            type(self.frame_id) is not str
            or (
                not self.frame_id
                and self.message_type
                not in {"sensor_msgs/msg/BatteryState", "irobot_create_msgs/msg/DockStatus"}
            )
            or self.max_header_age_s is None
        ):
            raise ValueError("Header-bearing observation requires its original frame and age")
        if self.max_receive_age_s is None:
            raise ValueError("Every source requires an actual receive age bound")
        future_limit = (
            1.0
            if self.role in {"map_tf", "base_tf"} and self.message_type == "tf2_msgs/msg/TFMessage"
            else 0.1
        )
        if (
            type(self.max_header_future_s) not in (int, float)
            or not math.isfinite(self.max_header_future_s)
            or not 0 <= self.max_header_future_s <= future_limit
        ):
            raise ValueError("Compiled header future tolerance exceeds native source contract")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (self.max_receive_age_s, self.max_header_age_s)
            if v is not None
        ):
            raise ValueError("Finite positive original observation ages required")


@dataclass(frozen=True)
class PublicCoverageBinding:
    body_id: str
    body_snapshot_hash: str
    driver_sha256: str
    odom_epoch: str
    graph_epoch: str
    map_frame: str
    execution_mode: ExecutionMode
    sources: tuple[CoverageSource, ...]
    robot_radius_m: float
    planning_margin_m: float
    brush_radius_m: float
    maximum_work: int
    max_duration_sec: float
    minimum_work_soc: float
    durable_task_evidence: bool = False

    def __post_init__(self):
        if type(self.durable_task_evidence) is not bool:
            raise ValueError("Explicit boolean durable evidence contract required")
        if not isinstance(self.execution_mode, ExecutionMode) or self.execution_mode not in (
            ExecutionMode.FIXTURE,
            ExecutionMode.SIMULATION,
        ):
            raise ValueError("No hardware or SHADOW coverage executor qualification")
        if not all(
            type(v) is str and v
            for v in (self.body_id, self.odom_epoch, self.graph_epoch, self.map_frame)
        ):
            raise ValueError("Explicit Body, frame and continuous epochs required")
        if any(
            not re.fullmatch(r"[a-f0-9]{64}", v)
            for v in (self.body_snapshot_hash, self.driver_sha256)
        ):
            raise ValueError("Exact compiled Body and Driver hashes required")
        expected = {
            "odom": "nav_msgs/msg/Odometry",
            "cleaner": "std_msgs/msg/Bool",
            "occupancy": "nav_msgs/msg/OccupancyGrid",
            "global_costmap": "nav_msgs/msg/OccupancyGrid",
            "local_costmap": "nav_msgs/msg/OccupancyGrid",
            "graph": "visualization_msgs/msg/MarkerArray",
            "map_tf": "tf2_msgs/msg/TFMessage",
            "base_tf": "tf2_msgs/msg/TFMessage",
            "scan": "sensor_msgs/msg/LaserScan",
            "depth": "sensor_msgs/msg/PointCloud2",
            "battery": "sensor_msgs/msg/BatteryState",
            "dock": "irobot_create_msgs/msg/DockStatus",
        }
        roles = {source.role: source for source in self.sources}
        if (
            len(roles) != len(self.sources)
            or not expected.keys() <= roles.keys()
            or any(roles[k].message_type != v for k, v in expected.items())
        ):
            raise ValueError("Distinct typed public coverage roles required")
        if any(not roles[k].child_frame_id for k in ("map_tf", "base_tf")):
            raise ValueError("Explicit public TF parent/child contracts required")
        if type(self.minimum_work_soc) not in (int, float) or not 0 < self.minimum_work_soc < 1:
            raise ValueError("Compiled finite work energy reserve required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (self.robot_radius_m, self.brush_radius_m, self.max_duration_sec)
        ):
            raise ValueError("Positive finite Body geometry and time budget required")
        if (
            type(self.planning_margin_m) not in (int, float)
            or not math.isfinite(self.planning_margin_m)
            or self.planning_margin_m < 0
            or type(self.maximum_work) is not int
            or not 1 <= self.maximum_work <= 1000
        ):
            raise ValueError("Finite planning margin and work limit required")

    @classmethod
    def from_body(cls, body: EffectiveBody):
        if body.effective_body_hash != body.compute_hash():
            raise ValueError("Compiled Body hash differs")
        available = body.capabilities.get("enabled", []) + body.capabilities.get("degraded", [])
        blocked = body.capabilities.get("blocked", []) + body.capabilities.get("disabled", [])
        required = {
            "coverage.execute_slice",
            "navigation.follow_path",
            "cleaning.enable",
            "cleaning.disable",
        }
        if not required.issubset(available) or required.intersection(blocked):
            raise ValueError("Unblocked coverage and declared internal primitives required")
        if body.safety.get("environment", {}).get("real_robot_execution_allowed") is not False:
            raise ValueError("Explicit non-hardware Body required")
        raw = body.provider_interfaces.get("public_coverage", {})
        if raw.get("schema_version") != "rosclaw.public_coverage.v1":
            raise ValueError("Compiled public coverage binding required")
        mode = ExecutionMode(raw["execution_mode"])
        if mode not in (ExecutionMode.FIXTURE, ExecutionMode.SIMULATION):
            raise ValueError("Public coverage has no REAL/SHADOW qualification")
        native = body.provider_interfaces.get("ros_capability_bindings", {})
        if (
            native.get("navigation.follow_path", {}).get("action_type")
            != "nav2_msgs/action/FollowPath"
        ):
            raise ValueError("Typed native FollowPath binding required")
        for capability in ("cleaning.enable", "cleaning.disable"):
            if native.get(capability, {}).get("state_type") != "std_msgs/msg/Bool":
                raise ValueError("Typed independent brush observation required")
        geometry = raw["geometry"]
        result = cls(
            body.body_instance_id,
            body.effective_body_hash,
            raw["driver_sha256"],
            raw["odom_epoch"],
            raw["graph_epoch"],
            body.frames["map"],
            mode,
            tuple(CoverageSource(**item) for item in raw["sources"]),
            geometry["robot_radius_m"],
            geometry["planning_tracking_margin_m"],
            geometry["brush_radius_m"],
            raw["maximum_work"],
            raw["max_duration_sec"],
            raw["minimum_work_soc"],
            raw.get("durable_task_evidence", False),
        )
        if not re.fullmatch(r"[a-f0-9]{64}", result.driver_sha256):
            raise ValueError("Actual registered Driver observation digest required")
        if not result.odom_epoch or not result.graph_epoch:
            raise ValueError("Explicit current observation epochs required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (result.robot_radius_m, result.brush_radius_m, result.max_duration_sec)
        ):
            raise ValueError("Finite positive compiled geometry and duration required")
        if (
            type(result.planning_margin_m) not in (int, float)
            or not math.isfinite(result.planning_margin_m)
            or result.planning_margin_m < 0
            or type(result.maximum_work) is not int
            or not 1 <= result.maximum_work <= 1000
        ):
            raise ValueError("Finite margin and work count required")
        roles = [source.role for source in result.sources]
        if len(set(roles)) != len(roles) or not {
            "odom",
            "cleaner",
            "occupancy",
            "global_costmap",
            "local_costmap",
            "graph",
            "map_tf",
            "base_tf",
            "scan",
            "depth",
        }.issubset(roles):
            raise ValueError("Distinct complete public work sources required")
        return result


@dataclass(frozen=True)
class CoverageSourceFact:
    role: str
    topic: str
    message_type: str
    frame_id: str | None
    header_native_s: float | None
    received_monotonic_s: float
    child_frame_id: str | None = None
    publisher_gid: str | None = None


@dataclass(frozen=True)
class PublicCoverageSnapshot:
    body_id: str
    body_snapshot_hash: str
    driver_sha256: str
    odom_epoch: str
    graph_epoch: str
    native_s: float
    captured_monotonic_s: float
    map_xy: tuple[float, float] | None
    odom_xy: tuple[float, float]
    cleaner: bool
    linear_speed: float
    yaw_rate: float
    native_motion_settled: bool
    soc: float | None
    docked: bool | None
    graph_nodes: tuple[tuple[int, tuple[float, float]], ...]
    sources: tuple[CoverageSourceFact, ...]
    grid: Grid | None
    work_source_fault: str | None = None


class PublicCoveragePort(Protocol):
    """Implemented ONLY by the trusted, Body-bound native Driver.

    snapshot/drain return actual public observations, never commanded poses.
    execute must retain the original native goal through cancel/result and
    honor the given stop/deadline and live path guards. It grants no new
    authority: admission and the exclusive Body lease belong to the outer job.
    """

    binding: PublicCoverageBinding

    def snapshot(self, *, work: bool = True) -> PublicCoverageSnapshot: ...
    def drain(self) -> list[PublicCoverageSnapshot]: ...
    def execute(
        self,
        capability: str,
        arguments: dict,
        *,
        parent: ActionEnvelope,
        primitive_id: str,
        stop_requested: threading.Event,
        deadline_monotonic: float,
    ) -> ActionExecutionResult: ...
    def emergency_stop(self) -> dict: ...


class _KnownLivePathCancellationError(RuntimeError):
    """Typed original terminal from the same owned primitive, never an ACK alone."""

    def __init__(self, primitive_id, dispatch):
        self.primitive_id, self.dispatch = primitive_id, dict(dispatch)
        super().__init__("Original native live-path cancellation is terminal")


class _PostEntryGeometryChangeError(ValueError):
    """A freshly observed path rejection before brush enable, never a source failure."""


class PublicCoverageExecutor:
    def __init__(
        self,
        *,
        binding: PublicCoverageBinding,
        port: PublicCoveragePort,
        ledger: GraphAnchoredCoverage,
        evidence_store: Path | None = None,
    ):
        if port.binding != binding or (ledger.epoch, ledger.graph_epoch) != (
            binding.odom_epoch,
            binding.graph_epoch,
        ):
            raise ValueError("Same immutable Driver/Body and current ledger epochs required")
        if ledger.anchor.brush_radius != binding.brush_radius_m:
            raise ValueError("Ledger brush geometry differs from compiled Body")
        if (evidence_store is not None) != binding.durable_task_evidence:
            raise ValueError("Compiled durable evidence contract and trusted store must agree")
        self.binding, self.port, self.ledger = binding, port, ledger
        self.stopping = threading.Event()
        self.execution_lock = threading.Lock()
        self._parent: ActionEnvelope | None = None
        self._deadline = 0.0
        self._sequence = 0
        self._recoveries = []
        self._entry_recoveries = []
        self.evidence_store = None if evidence_store is None else Path(evidence_store)
        self._evidence_started = None
        self._evidence_samples = []
        self._primitive_evidence = []
        bind_observations = getattr(port, "bind_observation_service", None)
        if callable(bind_observations):
            bind_observations(self._service_observations)

    def _service_observations(self):
        # Called by a trusted concrete port during a blocking native wait,
        # never from inside planning iteration and never without an owner.
        if self._parent is not None and not self.stopping.is_set():
            self._record()

    def _validate(self, snapshot, *, work=True, after=None, historical=False, required_roles=None):
        b = self.binding
        registered_pose = work or required_roles is not None
        if (
            snapshot.body_id,
            snapshot.body_snapshot_hash,
            snapshot.driver_sha256,
            snapshot.odom_epoch,
        ) != (b.body_id, b.body_snapshot_hash, b.driver_sha256, b.odom_epoch):
            raise ValueError("Live public observation Body/source/epoch differs")
        if registered_pose and (snapshot.graph_epoch != b.graph_epoch or snapshot.map_xy is None):
            raise ValueError("Current public graph epoch and map pose required")
        numbers = (
            snapshot.native_s,
            snapshot.captured_monotonic_s,
            *(snapshot.map_xy if registered_pose else ()),
            *snapshot.odom_xy,
            snapshot.linear_speed,
            snapshot.yaw_rate,
        )
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or type(snapshot.cleaner) is not bool
        ):
            raise ValueError("Finite actual public state required")
        facts = {fact.role: fact for fact in snapshot.sources}
        if len(facts) != len(snapshot.sources):
            raise ValueError("Ambiguous public observation source")
        if snapshot.captured_monotonic_s > time.monotonic():
            raise ValueError("Future public observation capture")
        now = snapshot.captured_monotonic_s if historical else time.monotonic()
        for source in b.sources:
            needed = (
                required_roles
                if required_roles is not None
                else ({s.role for s in b.sources} if work else {"odom", "cleaner"})
            )
            if source.role not in needed:
                continue
            fact = facts.get(source.role)
            if fact is None or (
                fact.topic,
                fact.message_type,
                fact.frame_id,
                fact.child_frame_id,
                fact.publisher_gid,
            ) != (
                source.topic,
                source.message_type,
                source.frame_id,
                source.child_frame_id,
                source.publisher_gid,
            ):
                raise ValueError("Missing or mismatched public source: " + source.role)
            header_ok = (
                fact.header_native_s is None
                if source.max_header_age_s is None
                else (
                    fact.header_native_s is not None
                    and math.isfinite(fact.header_native_s)
                    and -source.max_header_future_s
                    <= snapshot.native_s - fact.header_native_s
                    <= source.max_header_age_s
                )
            )
            if (
                not header_ok
                or not math.isfinite(fact.received_monotonic_s)
                or not 0 <= now - fact.received_monotonic_s < source.max_receive_age_s
                or (
                    after is not None
                    and source.role in {"odom", "cleaner"}
                    and fact.received_monotonic_s < after
                )
            ):
                raise ValueError("Original freshness guard: " + source.role)
        if work:
            if (
                type(snapshot.soc) not in (int, float)
                or not math.isfinite(snapshot.soc)
                or not 0 <= snapshot.soc <= 1
                or type(snapshot.docked) is not bool
            ):
                raise ValueError("Actual public battery and dock state required")
            if snapshot.grid is None or snapshot.grid._global_intent_inputs[2:] != (
                b.robot_radius_m,
                b.planning_margin_m,
            ):
                detail = (
                    "" if snapshot.work_source_fault is None else ": " + snapshot.work_source_fault
                )
                raise ValueError("Current public grid or exact compiled footprint missing" + detail)
            if (
                snapshot.grid.navigation_space is None
                or snapshot.grid.local_navigation_space is None
            ):
                raise ValueError("Both global and local public navigation constraints required")
            if not snapshot.grid.physical_pose_safe(snapshot.map_xy):
                raise ValueError("Actual complete footprint is not publicly safe")
        return snapshot

    def _guard(self, *, work=True):
        if self.stopping.is_set():
            raise RuntimeError("Owning coverage action stopped")
        if time.monotonic() >= self._deadline:
            raise TimeoutError("Original coverage action deadline")
        sample = self._validate(self.port.snapshot(work=work), work=work)
        if work and sample.native_motion_settled is not True:
            raise RuntimeError("Unknown old native motion cannot enter new work")
        if work and (sample.docked or sample.soc < self.binding.minimum_work_soc):
            raise RuntimeError(
                "Work requires undocked public state and the original energy reserve"
            )
        return sample

    def _planning_service(self):
        # A held public grid is proposal input, never permission to dispatch.
        # Service the original deadline and ALL live source/energy guards
        # without rebuilding the entire grid inside every planning callback.
        sample = self._guard(work=False)
        self._validate(sample, work=False, required_roles={s.role for s in self.binding.sources})
        if (
            sample.native_motion_settled is not True
            or type(sample.docked) is not bool
            or sample.docked
            or type(sample.soc) not in (int, float)
            or not math.isfinite(sample.soc)
            or not self.binding.minimum_work_soc <= sample.soc <= 1
        ):
            raise RuntimeError("Original native state or work energy reserve unavailable")
        return sample

    def _record(self):
        for sample in self.port.drain():
            # Credit uses captured odometry/brush/registration facts. It does
            # not relabel an old geometry raster with a newer receipt time.
            self._validate(
                sample,
                work=False,
                historical=True,
                required_roles={"odom", "cleaner", "graph", "map_tf"},
            )
            if (
                self.evidence_store is not None
                and self._evidence_started is not None
                and sample.captured_monotonic_s >= self._evidence_started
            ):
                self._evidence_samples.append(
                    snapshot_record(sample, {"odom", "cleaner", "graph", "map_tf"})
                )
            stamp = next(f.header_native_s for f in sample.sources if f.role == "odom")
            assert stamp is not None
            if self.ledger.evidence_samples and stamp <= self.ledger.evidence_samples[-1]["stamp"]:
                raise ValueError("Replayed or out-of-order coverage observation")
            self.ledger.update_graph(dict(sample.graph_nodes), graph_epoch=sample.graph_epoch)
            if len(sample.graph_nodes) < 2:
                if sample.cleaner:
                    raise ValueError("No current public graph registration for brush credit")
                for patch in self.ledger.patches.values():
                    patch.last_sample = None
                continue
            self.ledger.sample_map(
                stamp=stamp,
                map_xy=sample.map_xy,
                odom_xy=sample.odom_xy,
                cleaner=sample.cleaner,
                odom_epoch=sample.odom_epoch,
                odom_fresh=True,
                cleaner_fresh=True,
            )
        self.ledger.sync()

    def _primitive(self, capability, arguments):
        assert self._parent is not None
        cleanup = capability == "cleaning.disable"
        sample = self._guard(work=not cleanup)
        if capability == "navigation.follow_path":
            points = np.asarray(arguments["path_xy"], float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not 2 <= len(points) <= 1024
                or not np.isfinite(points).all()
                or np.linalg.norm(points[0] - sample.map_xy) > 0.10
                or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 3.0 + 1e-9
                or not sample.grid.motion_path_safe(points)
            ):
                raise ValueError("Current complete footprint rejects bounded work path")
        started = time.monotonic()
        self._sequence += 1
        primitive_id = (
            "pcov_"
            + hashlib.sha256(f"{self._parent.action_id}:{self._sequence}".encode()).hexdigest()
        )
        result = self.port.execute(
            capability,
            arguments,
            parent=self._parent,
            primitive_id=primitive_id,
            stop_requested=self.stopping,
            deadline_monotonic=self._deadline,
        )
        if self.stopping.is_set():
            raise RuntimeError("Primitive result cannot revive a stopped coverage action")
        if (
            result.dispatch_result.get("parent_action_id") != self._parent.action_id
            or result.dispatch_result.get("owned_primitive_id") != primitive_id
        ):
            raise RuntimeError("Primitive outcome belongs to another action or dispatch")
        expected_domain = (
            EvidenceDomain.FIXTURE
            if self.binding.execution_mode is ExecutionMode.FIXTURE
            else EvidenceDomain.SIMULATION
        )
        if result.evidence_domain is not expected_domain:
            raise RuntimeError("Original primitive evidence domain differs from bound execution")

        def record_primitive(after, roles):
            if self.evidence_store is None:
                return
            self._primitive_evidence.append(
                {
                    "parent_action_id": self._parent.action_id,
                    "primitive_id": primitive_id,
                    "capability_id": capability,
                    "final_state": result.final_state.value,
                    "evidence_domain": result.evidence_domain.value,
                    "arguments": {"path_xy": np.asarray(arguments["path_xy"], float).tolist()}
                    if capability == "navigation.follow_path"
                    else dict(arguments),
                    "dispatch": dict(result.dispatch_result),
                    "verification": dict(result.verification_result),
                    "before": snapshot_record(
                        sample,
                        {"odom", "cleaner"} if cleanup else {s.role for s in self.binding.sources},
                    ),
                    "after": snapshot_record(after, roles),
                }
            )

        if (
            result.final_state is not ActionState.COMPLETED
            or (result.verification_result or {}).get("passed") is not True
        ):
            dispatch = result.dispatch_result
            if (
                capability == "navigation.follow_path"
                and result.final_state is ActionState.FAILED
                and result.evidence_domain
                is (
                    EvidenceDomain.FIXTURE
                    if self.binding.execution_mode is ExecutionMode.FIXTURE
                    else EvidenceDomain.SIMULATION
                )
                and dispatch.get("action") == "/follow_path"
                and dispatch.get("action_type") == "nav2_msgs/action/FollowPath"
                and type(dispatch.get("status")) is int
                and dispatch["status"] in {5, 6}
                and dispatch.get("cancel_result_received") is True
                and isinstance(dispatch.get("live_path_invalidation"), str)
                and dispatch["live_path_invalidation"]
            ):
                after = self._validate(self.port.snapshot(work=False), work=False, after=started)
                if after.native_motion_settled is True:
                    record_primitive(after, {"odom", "cleaner"})
                    raise _KnownLivePathCancellationError(primitive_id, dispatch)
            raise RuntimeError("Internal primitive did not produce a verified original terminal")
        self._record()
        after = self._validate(
            self.port.snapshot(work=not cleanup), work=not cleanup, after=started
        )
        if not cleanup and after.native_motion_settled is not True:
            raise RuntimeError("Original native goal terminal remains unknown")
        if capability.startswith("cleaning.") and after.cleaner is not (
            capability == "cleaning.enable"
        ):
            raise RuntimeError("Actual brush state differs from requested phase")
        if (
            capability == "navigation.follow_path"
            and np.linalg.norm(np.asarray(after.map_xy) - points[-1]) > 0.05
        ):
            raise RuntimeError("Actual work endpoint exceeds original tracking limit")
        record_primitive(
            after, {"odom", "cleaner"} if cleanup else {s.role for s in self.binding.sources}
        )
        return after

    def _execute_work(self, work):
        path = np.asarray(work["path_xy"], float)
        stalled = 0
        for _ in range(60):
            current = self._guard()
            if np.linalg.norm(np.asarray(current.map_xy) - path[0]) <= 0.05:
                break
            proposal = propose_transfer(current.grid, current.map_xy, path[0])
            moved = self._primitive("navigation.follow_path", {"path_xy": list(proposal.prefix_xy)})
            progress = np.linalg.norm(np.asarray(current.map_xy) - path[0]) - np.linalg.norm(
                np.asarray(moved.map_xy) - path[0]
            )
            stalled = 0 if progress >= 0.05 else stalled + 1
            if stalled >= 3:
                raise RuntimeError("Three work-entry prefixes without observed target progress")
        else:
            raise RuntimeError("Finite public work-entry transfer exhausted")
        current = self._guard()
        if not current.grid.motion_path_safe(path):
            raise _PostEntryGeometryChangeError("Fresh post-entry geometry rejects selected work")
        before = len(self.ledger.anchor.cleaned)
        self._primitive("cleaning.enable", {})
        if len(path) > 1:
            self._primitive("navigation.follow_path", {"path_xy": path.tolist()})
        self._primitive("cleaning.disable", {})
        if len(self.ledger.anchor.cleaned) <= before:
            raise RuntimeError("No actual public brush evidence for completed work")
        return len(path) > 1

    def _observe_replanning_boundary(self, work):
        # This is not an E-stop reset. Owner loss/unknown native results never
        # reach this path; the original lease, deadline and attempt count run.
        stop_deadline = min(self._deadline, time.monotonic() + 5.0)
        stopped = self._primitive("cleaning.disable", {})
        while not self._stopped(stopped) or stopped.native_motion_settled is not True:
            if time.monotonic() >= stop_deadline:
                raise RuntimeError(
                    "No fresh stopped/off original-terminal proof after live cancellation"
                )
            time.sleep(0.02)
            stopped = self._guard(work=False)
        if time.monotonic() >= stop_deadline:
            raise RuntimeError("Original five-second stopped/off observation window expired")
        self.ledger.failed(
            np.asarray(work["path_xy"], float), stopped.map_xy, IDENTITY, now=stopped.native_s
        )
        return stopped

    def _recover_post_entry_geometry(self, work):
        stopped = self._observe_replanning_boundary(work)
        self._entry_recoveries.append(
            {
                "reason": "fresh_post_entry_geometry_rejected_before_brush_enable",
                "fresh_stopped_off": True,
                "original_native_terminal_known": True,
                "original_deadline_preserved": True,
                "replayed_native_goal": False,
                "observed_native_s": stopped.native_s,
                "execution_credit": "no completion credit for rejected work",
            }
        )

    def _recover_observed_live_path(self, work, cancelled):
        self._observe_replanning_boundary(work)
        self._recoveries.append(
            {
                "original_primitive_id": cancelled.primitive_id,
                "original_native_status": cancelled.dispatch["status"],
                "original_live_path_evidence": cancelled.dispatch.get("live_path_evidence"),
                "fresh_stopped_off": True,
                "original_native_terminal_known": True,
                "original_deadline_preserved": True,
                "replayed_native_goal": False,
                "execution_credit": "actual public brush samples only; incomplete work not counted",
            }
        )

    def _stopped(self, sample):
        return (
            not sample.cleaner and abs(sample.linear_speed) < 0.02 and abs(sample.yaw_rate) < 0.03
        )

    def emergency_stop(self):
        self.stopping.set()  # Terminal until the operator installs a new executor generation.
        requested = time.monotonic()
        try:
            response = self.port.emergency_stop()
        except Exception as exc:
            response = {"acknowledged": False, "error": str(exc)}
        try:
            sample = self._validate(self.port.snapshot(work=False), work=False, after=requested)
            observed = self._stopped(sample)
            terminal = sample.native_motion_settled
        except Exception:
            observed = False
            terminal = None
        return {
            "acknowledged": response.get("acknowledged") is True,
            "physical_stop_observed": observed,
            "original_native_terminal_known": terminal,
            "evidence_domain": "FIXTURE"
            if self.binding.execution_mode is ExecutionMode.FIXTURE
            else "SIMULATION",
            "source": "new bound public odometry and brush observations",
            "driver_result": response,
        }

    def __call__(self, action):
        b = self.binding
        if (
            action.capability_id != "coverage.execute_slice"
            or action.body_id != b.body_id
            or action.body_snapshot_hash != b.body_snapshot_hash
            or action.execution_mode is not b.execution_mode
            or action.orphan_policy is not OrphanPolicy.STOP_ON_CLIENT_LOSS
            or not action.session_id
            or self.stopping.is_set()
        ):
            return self._result(
                ActionState.BLOCKED, error="BODY_MODE_SESSION_OR_GENERATION_MISMATCH"
            )
        args = action.arguments
        maximum = args.get("maximum_work", b.maximum_work)
        minimum = args.get("minimum_completed_work", 1)
        minimum_strokes = args.get("minimum_completed_strokes", 1)
        if (
            set(args) - {"maximum_work", "minimum_completed_work", "minimum_completed_strokes"}
            or type(maximum) is not int
            or not 1 <= maximum <= b.maximum_work
            or type(minimum) is not int
            or not 1 <= minimum <= maximum
            or type(minimum_strokes) is not int
            or not 0 <= minimum_strokes <= maximum
        ):
            return self._result(ActionState.BLOCKED, error="FINITE_COVERAGE_ARGUMENTS_REQUIRED")
        if not self.execution_lock.acquire(blocking=False):
            return self._result(ActionState.BLOCKED, error="COVERAGE_BODY_BUSY")
        completed = 0
        strokes = 0
        try:
            remaining = (
                (action.deadline_at - datetime.now(UTC)).total_seconds()
                if action.deadline_at
                else b.max_duration_sec
            )
            self._deadline = time.monotonic() + min(remaining, b.max_duration_sec)
            self._parent = action
            self._sequence = 0
            self._recoveries = []
            self._entry_recoveries = []
            self._evidence_started = time.monotonic()
            self._evidence_samples = []
            self._primitive_evidence = []
            initial = self._guard()
            self._evidence_started = initial.captured_monotonic_s
            self._primitive("cleaning.disable", {})
            for _ in range(maximum):
                current = self._guard()
                self._record()
                work = self.ledger.next_work(
                    current.grid,
                    current.map_xy,
                    IDENTITY,
                    now=current.native_s,
                    prepare_path=prepare_work_path,
                    service=self._planning_service,
                )
                if work["kind"] == "NO_CURRENT_OBSERVED_WORK":
                    break
                try:
                    stroke = self._execute_work(work)
                except _PostEntryGeometryChangeError:
                    self._recover_post_entry_geometry(work)
                    continue
                except _KnownLivePathCancellationError as cancelled:
                    self._recover_observed_live_path(work, cancelled)
                    continue
                completed += 1
                if stroke:
                    strokes += 1
                if completed >= minimum and strokes >= minimum_strokes:
                    break
            final = self._guard(work=False)
            if (
                not self._stopped(final)
                or final.native_motion_settled is not True
                or completed < minimum
                or strokes < minimum_strokes
            ):
                raise RuntimeError("Finite work target or fresh stopped/off postcondition unmet")
            task_evidence = None
            if self.evidence_store is not None:
                binding_data = asdict(b)
                binding_data["execution_mode"] = b.execution_mode.value
                payload = {
                    "schema": "rosclaw.finite_coverage_evidence.v1",
                    "scope": "finite_observed_coverage_slice_only",
                    "whole_house_success": None,
                    "execution_mode": b.execution_mode.value,
                    "action": {
                        "action_id": action.action_id,
                        "session_id": action.session_id,
                        "body_id": action.body_id,
                        "body_snapshot_hash": action.body_snapshot_hash,
                        "capability_id": action.capability_id,
                    },
                    "binding": binding_data,
                    "initial": snapshot_record(initial, {source.role for source in b.sources}),
                    "final": snapshot_record(final, {"odom", "cleaner"}),
                    "goals": {
                        "minimum_completed_work": minimum,
                        "minimum_completed_strokes": minimum_strokes,
                    },
                    "completed_work": completed,
                    "completed_strokes": strokes,
                    "samples": self._evidence_samples,
                    "primitives": self._primitive_evidence,
                    "live_path_recoveries": self._recoveries,
                    "post_entry_recoveries": self._entry_recoveries,
                    "judge_read": False,
                }
                task_evidence = persist_archive(self.evidence_store, payload)
                # Serialization/fsync is inside the unchanged deadline. Neither
                # old evidence nor a persisted file replaces a fresh final check.
                fresh = self._guard(work=False)
                if not self._stopped(fresh) or fresh.native_motion_settled is not True:
                    raise RuntimeError("Fresh final state changed during evidence persistence")
            return self._result(
                ActionState.COMPLETED,
                task_evidence=task_evidence,
                completed=completed,
                strokes=strokes,
                recoveries=self._recoveries,
                entry_recoveries=self._entry_recoveries,
            )
        except Exception as exc:
            stop = self.emergency_stop()
            return self._result(
                ActionState.FAILED,
                completed=completed,
                strokes=strokes,
                error=str(exc),
                stop=stop,
                recoveries=self._recoveries,
                entry_recoveries=self._entry_recoveries,
            )
        finally:
            self._parent = None
            self.execution_lock.release()

    def _result(
        self,
        state,
        *,
        completed=0,
        strokes=0,
        error=None,
        stop=None,
        recoveries=(),
        entry_recoveries=(),
        task_evidence=None,
    ):
        fixture = self.binding.execution_mode is ExecutionMode.FIXTURE
        return ActionExecutionResult(
            final_state=state,
            evidence_level=(
                EvidenceLevel.TASK_VERIFIED
                if task_evidence is not None
                else EvidenceLevel.DRIVER_CONFIRMED
            )
            if state is ActionState.COMPLETED
            else EvidenceLevel.REQUESTED,
            evidence_domain=EvidenceDomain.FIXTURE if fixture else EvidenceDomain.SIMULATION,
            simulation_result=None
            if fixture
            else {"has_physics": True, "engine": "registered_public_ros_driver"},
            dispatch_result={
                "completed_work": completed,
                "completed_continuous_strokes": strokes,
                "source": "public_observations",
                "nested_daemon_actions": False,
                "observed_live_path_recoveries": list(recoveries),
                "observed_post_entry_recoveries": list(entry_recoveries),
            },
            verification_result={
                "passed": state is ActionState.COMPLETED,
                "scope": "finite_observed_coverage_slice_only",
                "whole_house_success": None,
                "judge_read": False,
                "stop": stop,
                "task_evidence": task_evidence,
            },
            artifacts=[] if task_evidence is None else [task_evidence["uri"]],
            errors=[] if error is None else [{"code": "PUBLIC_COVERAGE_FAILED", "message": error}],
        )
