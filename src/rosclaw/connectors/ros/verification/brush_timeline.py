"""Causal brush-state pairing for future passive-observer/actuator separation.

A complete ordered actuator watermark must reach a pose before brush-on credit
can be considered. Pending state is not silently treated as on or off. State
changes at exactly the pose timestamp are conservatively uncredited. There is
no transport, service, publication or motion permission in this module.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


@dataclass(frozen=True)
class BrushStateEvent:
    run_id: str
    body_snapshot_hash: str
    attachment_hash: str
    producer_id: str
    sequence: int
    sim_time_sec: float
    kind: str
    enabled: bool
    captured_at: str
    complete: bool
    evidence_domain: str = "SIMULATION"

    def artifact_hash(self):
        return digest(self.__dict__)


class BrushStateTimeline:
    def __init__(
        self, *, run_id, body_snapshot_hash, attachment_hash, producer_id, max_buffer=2048
    ):
        binding = (run_id, body_snapshot_hash, attachment_hash, producer_id)
        if any(type(s) is not str or not s for s in binding):
            raise ValueError("frozen run/Body/attachment/producer binding required")
        if type(max_buffer) is not int or not 2 <= max_buffer <= 2048:
            raise ValueError("bounded brush event buffer required")
        self.binding, self.max_buffer = binding, max_buffer
        self.events = deque()
        self.sequence = self.latest_time = self.watermark = self.received_monotonic = None
        self.watermark_event_hash = None
        self.pair_count = 0
        self.pair_chain_hash = "GENESIS"
        self.current_state = False
        self.previous_pose_time = None
        self.fault = None

    def append(self, event, *, artifact_hash, now=None, received_monotonic=None):
        if self.fault:
            raise ValueError("brush timeline fault is latched: " + self.fault)
        try:
            if not isinstance(event, BrushStateEvent) or event.artifact_hash() != artifact_hash:
                raise ValueError("brush event integrity/type mismatch")
            if (
                event.run_id,
                event.body_snapshot_hash,
                event.attachment_hash,
                event.producer_id,
            ) != self.binding:
                raise ValueError("brush event source binding mismatch")
            if (
                event.complete is not True
                or event.evidence_domain != "SIMULATION"
                or type(event.enabled) is not bool
            ):
                raise ValueError("complete SIM brush event required")
            if (
                type(event.sequence) is not int
                or event.sequence < 0
                or (self.sequence is not None and event.sequence != self.sequence + 1)
            ):
                raise ValueError("brush sequence gap/reorder")
            if (
                type(event.sim_time_sec) not in (int, float)
                or not math.isfinite(event.sim_time_sec)
                or event.sim_time_sec < 0
            ):
                raise ValueError("finite nonnegative brush SIM time required")
            if (
                event.kind == "TRANSITION"
                and self.watermark is not None
                and event.sim_time_sec < self.watermark
            ):
                raise ValueError("brush transition retroactively crosses a closed watermark")
            if self.latest_time is not None and event.sim_time_sec < self.latest_time:
                raise ValueError("brush time reversed")
            if event.kind not in ("TRANSITION", "WATERMARK"):
                raise ValueError("explicit transition or watermark required")
            capture = datetime.fromisoformat(event.captured_at)
            if capture.tzinfo is None:
                raise ValueError("timezone-aware brush capture required")
            age = ((now or datetime.now(UTC)) - capture).total_seconds()
            if not 0 <= age < 0.3:
                raise ValueError("brush capture stale or future-dated")
            received = time.monotonic() if received_monotonic is None else received_monotonic
            if type(received) not in (int, float) or not math.isfinite(received) or received < 0:
                raise ValueError("finite receiver monotonic timestamp required")
            if self.received_monotonic is not None and received < self.received_monotonic:
                raise ValueError("receiver monotonic timestamp reversed")
            if self.sequence is None:
                if event.kind != "WATERMARK" or event.enabled:
                    raise ValueError("initial complete brush-off watermark required")
                self.events.append(event)
            elif event.kind == "TRANSITION":
                self.events.append(event)
            elif event.enabled != self.current_state:
                raise ValueError("watermark state changed without an ordered transition")
            if len(self.events) > self.max_buffer:
                raise ValueError("unconsumed brush history exceeds bounded buffer")
            if event.kind == "WATERMARK":
                self.watermark = event.sim_time_sec
                self.watermark_event_hash = artifact_hash
            self.current_state = event.enabled
            self.sequence, self.latest_time, self.received_monotonic = (
                event.sequence,
                event.sim_time_sec,
                received,
            )
        except (ValueError, TypeError) as exc:
            self.fault = str(exc)
            raise

    def state_at(self, sim_time_sec, *, now_monotonic=None):
        if self.fault:
            raise ValueError("brush timeline fault is latched: " + self.fault)
        try:
            if (
                type(sim_time_sec) not in (int, float)
                or not math.isfinite(sim_time_sec)
                or sim_time_sec < 0
            ):
                raise ValueError("finite pose SIM time required")
            if self.previous_pose_time is not None and sim_time_sec <= self.previous_pose_time:
                raise ValueError("brush pose query reordered or reused")
            now_mono = time.monotonic() if now_monotonic is None else now_monotonic
            if type(now_mono) not in (int, float) or not math.isfinite(now_mono):
                raise ValueError("finite receiver query monotonic timestamp required")
            if self.received_monotonic is None or not 0 <= now_mono - self.received_monotonic < 0.3:
                raise ValueError("brush source missing or stale")
            if not self.events or sim_time_sec < self.events[0].sim_time_sec:
                raise ValueError("pose predates known brush state")
            if self.watermark is None or self.watermark <= sim_time_sec:
                return {
                    "status": "PENDING",
                    "enabled": None,
                    "reason": "actuator watermark has not reached this pose",
                }
            while len(self.events) > 1 and self.events[1].sim_time_sec <= sim_time_sec:
                self.events.popleft()
            event = self.events[0]
            enabled = event.enabled and event.sim_time_sec != sim_time_sec
            self.previous_pose_time = sim_time_sec
            self.pair_chain_hash = digest(
                {
                    "previous": self.pair_chain_hash,
                    "pose_sim_time_sec": sim_time_sec,
                    "state_event_hash": event.artifact_hash(),
                    "watermark_event_hash": self.watermark_event_hash,
                    "enabled": enabled,
                }
            )
            self.pair_count += 1
            return {
                "status": "PAIRED",
                "enabled": enabled,
                "state_event_hash": event.artifact_hash(),
                "watermark_sim_time_sec": self.watermark,
                "last_sequence": self.sequence,
                "watermark_event_hash": self.watermark_event_hash,
                "pair_chain_hash": self.pair_chain_hash,
            }
        except (ValueError, TypeError) as exc:
            self.fault = str(exc)
            raise

    def result(self):
        return {
            "schema_version": "rosclaw.brush_state_timeline.v1",
            "run_id": self.binding[0],
            "body_snapshot_hash": self.binding[1],
            "attachment_hash": self.binding[2],
            "producer_id": self.binding[3],
            "complete": self.fault is None and self.pair_count > 0,
            "fault": self.fault,
            "paired_pose_count": self.pair_count,
            "pair_chain_hash": self.pair_chain_hash,
            "watermark_semantics": "exclusive; same-time state transitions receive no brush-on credit",
        }
