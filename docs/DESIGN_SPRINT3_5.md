# DESIGN_SPRINT3_5: FirewallValidator + UnifiedTimeline + SeekDB

> **Author**: rosclaw_qwen (Chief Architecture Reviewer)
> **Date**: 2026-05-27
> **Status**: DRAFT — Pending executor review
> **Depends on**: ARCHITECTURE_REVIEW.md, RFC-0001 (PraxisEvent), PraxisEvent commit (66db8a8), EventBus integration (04d5b1c)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Sprint 3: FirewallValidator](#2-sprint-3-firewallvalidator)
3. [Sprint 4: UnifiedTimeline](#3-sprint-4-unifiedtimeline)
4. [Sprint 5: SeekDB Client](#4-sprint-5-seekdb-client)
5. [Cross-Sprint Integration](#5-cross-sprint-integration)
6. [EventBus Topic Registry](#6-eventbus-topic-registry)
7. [Initialization Ordering](#7-initialization-ordering)
8. [Test Strategy](#8-test-strategy)
9. [File Change Manifest](#9-file-change-manifest)
10. [Acceptance Criteria](#10-acceptance-criteria)

---

## 1. Overview

### 1.1 Goal

Bridge the gap between the **vision documents** (rosclaw_v1.0见解一/二.md, 哲学思辨.md) and the **current implementation** by providing production-ready specifications for three critical grounding engines:

| Sprint | Module | Grounding Type | Purpose |
|--------|--------|----------------|---------|
| Sprint 3 | `firewall/validator.py` | Action + Physical | e-URDF-aware trajectory validation with MuJoCo collision prediction |
| Sprint 4 | `practice/timeline.py` | Timeline | Unified multi-channel timeline binding LLM CoT with sensorimotor data |
| Sprint 5 | `memory/seekdb_client.py` | Experience | SeekDB Knowledge Plane interface replacing in-memory list |

### 1.2 Acknowledged Executor Progress

The executor (rosclaw) has already completed:
- ✅ PraxisEvent unified event structure (`core/types.py`, commit 66db8a8)
- ✅ EventBus integration into Memory, Practice, Swarm modules (commit 04d5b1c)
- ✅ MCPDrivers base + 3 implementations (MuJoCo, ROS2, Serial)
- ✅ SkillManager registry + executor + loader
- ✅ 102 tests passing (up from 77)

This design document builds on those changes. Where executor implementations already exist (e.g., `PracticeRecorder` already accepts `event_bus`), the designs integrate cleanly rather than replace.

### 1.3 Design Principles

1. **EventBus-only communication**: No module-to-module direct calls
2. **LifecycleMixin everywhere**: All new modules follow the 8-state lifecycle
3. **Graceful degradation**: Optional dependencies (MuJoCo, SeekDB) fall back to mock/in-memory
4. **PraxisEvent as spine**: All modules contribute to or consume PraxisEvent
5. **No publish during initialization**: Modules subscribe in `_do_initialize()`, publish only after `start()`

---

## 2. Sprint 3: FirewallValidator

### 2.1 Problem Statement

The current `DigitalTwinFirewall` in `firewall/decorator.py` (438 LOC) provides:
- MuJoCo-based collision detection
- 3 safety levels (STRICT/MODERATE/LENIENT)
- Decorator pattern for function-level validation

**What's missing:**
- No integration with e-URDF soft limits (currently hardcoded)
- No EventBus integration (cannot intercept `agent.command` events)
- No request-response pattern (fire-and-forget commands bypass validation)
- No SafetyEnvelope abstraction (limits are scattered across code)

### 2.2 New File: `firewall/validator.py`

```python
"""FirewallValidator - EventBus-integrated trajectory validation.

Subscribes to agent.command events, validates against e-URDF soft limits
and MuJoCo collision model, publishes firewall.response.{request_id}.

Sprint 3 of DESIGN_SPRINT3_5.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin


class ValidationLayer(Enum):
    """3-layer validation pipeline."""
    EURDF_SOFT_LIMITS = "eurdf_soft_limits"    # Joint limits from e-URDF × 0.95
    MUJOCO_COLLISION = "mujoco_collision"       # MuJoCo physics simulation
    SEMANTIC_SAFETY = "semantic_safety"          # safety.yaml rules


@dataclass
class SafetyEnvelope:
    """Extracted safety boundaries from e-URDF + safety.yaml."""
    joint_soft_limits: list[tuple[float, float]]  # (min, max) per joint, radians
    max_velocity: list[float]                       # rad/s per joint
    max_torque: list[float]                         # Nm per joint
    keepout_zones: list[dict] = field(default_factory=list)  # {name, center, radius}
    allowed_contacts: list[str] = field(default_factory=list)  # link names
    safety_level: str = "MODERATE"  # STRICT | MODERATE | LENIENT

    @classmethod
    def from_robot_model(cls, robot_model, safety_level: str = "MODERATE") -> "SafetyEnvelope":
        """Extract envelope from RobotModel (e-URDF parsed data)."""
        soft_factor = {"STRICT": 0.90, "MODERATE": 0.95, "LENIENT": 0.99}
        factor = soft_factor.get(safety_level, 0.95)

        joint_limits = []
        max_velocities = []
        max_torques = []

        for joint in robot_model.joints:
            lo = joint.limits[0] * factor if joint.limits[0] < 0 else joint.limits[0]
            hi = joint.limits[1] * factor if joint.limits[1] > 0 else joint.limits[1]
            joint_limits.append((lo, hi))
            max_velocities.append(joint.max_velocity * factor)
            max_torques.append(joint.max_torque * factor)

        return cls(
            joint_soft_limits=joint_limits,
            max_velocity=max_velocities,
            max_torque=max_torques,
            safety_level=safety_level,
        )


@dataclass
class ValidationRequest:
    """Request to validate a trajectory before execution."""
    request_id: str
    robot_id: str
    trajectory: list[list[float]]        # waypoints, each shape (dof,)
    duration_per_waypoint: list[float]   # seconds per segment
    source: str                           # "agent_runtime" | "skill_manager"
    metadata: dict = field(default_factory=dict)


@dataclass
class ViolationDetail:
    """Single safety violation found during validation."""
    layer: ValidationLayer
    severity: str           # "warning" | "error" | "critical"
    joint_index: Optional[int]
    description: str
    actual_value: Optional[float] = None
    limit_value: Optional[float] = None


@dataclass
class ValidationResponse:
    """Result of trajectory validation."""
    request_id: str
    is_safe: bool
    layers_checked: list[ValidationLayer]
    violations: list[ViolationDetail] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    simulation_duration_ms: float = 0.0

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class FirewallValidator(LifecycleMixin):
    """
    EventBus-integrated firewall with e-URDF awareness.

    Lifecycle:
        initialize() → loads SafetyEnvelope from RobotModel, subscribes to EventBus
        start() → begins processing validation requests
        stop() → unsubscribes, releases MuJoCo resources

    EventBus:
        Subscribes: agent.command (to intercept movement commands)
        Publishes:  firewall.response.{request_id} (validation result)
                    safety.violation (when violations found)
    """

    def __init__(
        self,
        robot_model,  # RobotModel from e_urdf
        event_bus: EventBus,
        mujoco_model_path: Optional[str] = None,
        safety_level: str = "MODERATE",
    ):
        super().__init__()
        self._robot_model = robot_model
        self._event_bus = event_bus
        self._mujoco_model_path = mujoco_model_path
        self._safety_level = safety_level
        self._envelope: Optional[SafetyEnvelope] = None
        self._mj_model = None
        self._mj_data = None

    def _do_initialize(self) -> None:
        """Build SafetyEnvelope from e-URDF, optionally load MuJoCo."""
        # Extract safety envelope from robot model
        self._envelope = SafetyEnvelope.from_robot_model(
            self._robot_model, self._safety_level
        )

        # Try to load MuJoCo for collision layer
        if self._mujoco_model_path:
            try:
                import mujoco
                self._mj_model = mujoco.MjModel.from_xml_path(self._mujoco_model_path)
                self._mj_data = mujoco.MjData(self._mj_model)
            except (ImportError, FileNotFoundError) as e:
                print(f"[FirewallValidator] MuJoCo unavailable: {e}")

        # Subscribe to EventBus
        self._event_bus.subscribe("agent.command", self._on_agent_command)
        print(f"[FirewallValidator] Initialized with {self._safety_level} safety, "
              f"{len(self._envelope.joint_soft_limits)} joints, "
              f"MuJoCo={'yes' if self._mj_model else 'no'}")

    def _do_start(self) -> None:
        """FirewallValidator is ready to validate."""
        self._event_bus.publish(Event(
            topic="firewall.status",
            payload={"state": "running", "safety_level": self._safety_level},
            source="firewall_validator",
        ))

    def _do_stop(self) -> None:
        """Release MuJoCo resources."""
        self._mj_model = None
        self._mj_data = None

    def _on_agent_command(self, event: Event) -> None:
        """Intercept agent commands, validate, publish response."""
        payload = event.payload
        action = payload.get("action", "")

        # Only validate movement commands
        if action not in ("move_joints", "execute_trajectory"):
            return

        request = ValidationRequest(
            request_id=payload.get("request_id", "unknown"),
            robot_id=payload.get("robot_id", "default"),
            trajectory=payload.get("trajectory", []),
            duration_per_waypoint=payload.get("durations", []),
            source=payload.get("source", "agent_runtime"),
        )

        response = self.validate(request)

        # Publish validation response
        self._event_bus.publish(Event(
            topic=f"firewall.response.{request.request_id}",
            payload={
                "is_safe": response.is_safe,
                "violations": [
                    {
                        "layer": v.layer.value,
                        "severity": v.severity,
                        "description": v.description,
                    }
                    for v in response.violations
                ],
                "warnings": response.warnings,
            },
            source="firewall_validator",
            priority=EventPriority.HIGH if response.is_safe else EventPriority.CRITICAL,
        ))

        # Publish safety violation if unsafe
        if not response.is_safe:
            self._event_bus.publish(Event(
                topic="safety.violation",
                payload={
                    "request_id": request.request_id,
                    "violations": [v.description for v in response.violations],
                    "action": "BLOCKED",
                },
                source="firewall_validator",
                priority=EventPriority.CRITICAL,
            ))

    def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Run 3-layer validation pipeline."""
        violations = []
        warnings = []
        layers_checked = []

        # Layer 1: e-URDF soft limits
        layer1_violations = self._check_eurdf_limits(request)
        violations.extend(layer1_violations)
        layers_checked.append(ValidationLayer.EURDF_SOFT_LIMITS)

        # Layer 2: MuJoCo collision (if available)
        if self._mj_model is not None:
            import time
            t0 = time.monotonic()
            layer2_violations = self._check_mujoco_collision(request)
            violations.extend(layer2_violations)
            layers_checked.append(ValidationLayer.MUJOCO_COLLISION)
            sim_ms = (time.monotonic() - t0) * 1000
        else:
            sim_ms = 0.0

        # Layer 3: Semantic safety rules
        layer3_violations, layer3_warnings = self._check_semantic_safety(request)
        violations.extend(layer3_violations)
        warnings.extend(layer3_warnings)
        layers_checked.append(ValidationLayer.SEMANTIC_SAFETY)

        is_safe = all(v.severity != "critical" for v in violations)

        return ValidationResponse(
            request_id=request.request_id,
            is_safe=is_safe,
            layers_checked=layers_checked,
            violations=violations,
            warnings=warnings,
            simulation_duration_ms=sim_ms,
        )

    def _check_eurdf_limits(self, request: ValidationRequest) -> list[ViolationDetail]:
        """Layer 1: Check trajectory against e-URDF soft limits."""
        violations = []
        if self._envelope is None:
            return violations

        for wp_idx, waypoint in enumerate(request.trajectory):
            for j_idx, value in enumerate(waypoint):
                if j_idx >= len(self._envelope.joint_soft_limits):
                    break
                lo, hi = self._envelope.joint_soft_limits[j_idx]
                if value < lo or value > hi:
                    violations.append(ViolationDetail(
                        layer=ValidationLayer.EURDF_SOFT_LIMITS,
                        severity="critical",
                        joint_index=j_idx,
                        description=f"Joint {j_idx} value {value:.3f} outside "
                                    f"soft limit [{lo:.3f}, {hi:.3f}] at waypoint {wp_idx}",
                        actual_value=value,
                        limit_value=hi if value > hi else lo,
                    ))
        return violations

    def _check_mujoco_collision(self, request: ValidationRequest) -> list[ViolationDetail]:
        """Layer 2: Simulate trajectory in MuJoCo, check for collisions."""
        violations = []
        if self._mj_data is None:
            return violations

        import mujoco

        for waypoint in request.trajectory:
            # Set joint positions
            dof = min(len(waypoint), self._mj_model.nq)
            self._mj_data.qpos[:dof] = waypoint[:dof]
            mujoco.mj_forward(self._mj_model, self._mj_data)

            # Check contacts
            for i in range(self._mj_data.ncon):
                contact = self._mj_data.contact[i]
                geom1_name = mujoco.mj_id2name(self._mj_model, 6, contact.geom1) or f"geom{contact.geom1}"
                geom2_name = mujoco.mj_id2name(self._mj_model, 6, contact.geom2) or f"geom{contact.geom2}"

                # Skip allowed contacts
                if geom1_name in self._envelope.allowed_contacts or \
                   geom2_name in self._envelope.allowed_contacts:
                    continue

                violations.append(ViolationDetail(
                    layer=ValidationLayer.MUJOCO_COLLISION,
                    severity="critical",
                    joint_index=None,
                    description=f"Collision detected: {geom1_name} <-> {geom2_name}",
                ))
                break  # One collision per waypoint is enough

        return violations

    def _check_semantic_safety(self, request: ValidationRequest) -> tuple[list[ViolationDetail], list[str]]:
        """Layer 3: Check semantic rules from safety.yaml."""
        violations = []
        warnings = []

        # Check keepout zones (if defined in e-URDF)
        if self._envelope and self._envelope.keepout_zones:
            for zone in self._envelope.keepout_zones:
                # Simplified: check if any waypoint end-effector position
                # is within keepout zone radius
                warnings.append(f"Keepout zone '{zone.get('name', 'unknown')}' defined "
                                f"but FK not computed — skipped")

        # Velocity check
        if request.duration_per_waypoint and self._envelope:
            for i, duration in enumerate(request.duration_per_waypoint):
                if duration > 0 and i > 0:
                    prev = np.array(request.trajectory[i - 1])
                    curr = np.array(request.trajectory[i])
                    velocities = np.abs(curr - prev) / duration
                    for j_idx, vel in enumerate(velocities):
                        if j_idx < len(self._envelope.max_velocity):
                            if vel > self._envelope.max_velocity[j_idx]:
                                violations.append(ViolationDetail(
                                    layer=ValidationLayer.SEMANTIC_SAFETY,
                                    severity="error",
                                    joint_index=j_idx,
                                    description=f"Joint {j_idx} velocity {vel:.2f} rad/s "
                                                f"exceeds limit {self._envelope.max_velocity[j_idx]:.2f}",
                                    actual_value=vel,
                                    limit_value=self._envelope.max_velocity[j_idx],
                                ))

        return violations, warnings
```

### 2.3 Data Flow: LLM Command → Validated Execution

```
LLM Instruction
    │
    ▼
MCPHub.handle_tool_call("move_joints", {...})
    │
    ├── publishes: agent.command {action: "move_joints", trajectory: [...], request_id: "abc123"}
    │
    ▼
FirewallValidator._on_agent_command()
    │
    ├── Layer 1: e-URDF soft limits check
    ├── Layer 2: MuJoCo collision simulation
    ├── Layer 3: Semantic safety (velocity, keepout zones)
    │
    ├── publishes: firewall.response.abc123 {is_safe: true/false, violations: [...]}
    ├── publishes: safety.violation (if unsafe, CRITICAL priority)
    │
    ▼
MCPHub awaits firewall.response.abc123 (with timeout)
    │
    ├── SAFE → forward to registered MCPDriver
    └── UNSAFE → return error to LLM, do NOT execute
```

### 2.4 Changes to Existing Files

**`firewall/decorator.py`** — No changes required. `FirewallValidator` is a new module that wraps/delegates to the existing `DigitalTwinFirewall` for MuJoCo simulation.

**`core/runtime.py`** — Add FirewallValidator initialization:
```python
# In _do_initialize(), after e-URDF parsing:
if self.config.enable_firewall and self._e_urdf is not None:
    from rosclaw.firewall.validator import FirewallValidator
    self._firewall_validator = FirewallValidator(
        robot_model=self._e_urdf.robot_model,
        event_bus=self.event_bus,
        mujoco_model_path=self.config.robot_model_path,
        safety_level=self.config.safety_level,  # New config field
    )
    self._modules.append(self._firewall_validator)
```

**`agent_runtime/mcp_hub.py`** — Replace fire-and-forget with request-response:
```python
# In handle_tool_call(), after publishing agent.command:
response_event = await self.event_bus.await_event(
    topic=f"firewall.response.{request_id}",
    timeout_sec=5.0,
)
if response_event and not response_event.payload["is_safe"]:
    return {"status": "blocked", "violations": response_event.payload["violations"]}
```

---

## 3. Sprint 4: UnifiedTimeline

### 3.1 Problem Statement

The current `PracticeRecorder` (77 LOC) wraps `DataFlywheel` and accepts `event_bus`, but:
- No unified timeline binding LLM reasoning with physical execution
- No MCAP recording integration
- No multi-channel correlation (LLM CoT at ~1Hz, commands at ~50Hz, sensorimotor at ~1000Hz)
- No PraxisEvent assembly from timeline data

### 3.2 New File: `practice/timeline.py`

```python
"""UnifiedTimeline - Multi-channel timeline binding LLM CoT with sensorimotor data.

Records all events on a single timeline with nanosecond precision.
Assembles PraxisEvent from timeline entries on praxis.completed.

Sprint 4 of DESIGN_SPRINT3_5.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import json
import time

import numpy as np

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.core.types import RobotState, PraxisEvent


class TimelineChannel(Enum):
    """Channels on the unified timeline."""
    LLM_REASONING = "llm_reasoning"         # CoT traces (~1Hz)
    AGENT_COMMAND = "agent_command"          # Movement commands (~50Hz)
    FIREWALL_RESULT = "firewall_result"      # Validation results (~50Hz)
    DRIVER_STATE = "driver_state"            # Robot state snapshots (~100Hz)
    SENSORIMOTOR = "sensorimotor"            # Raw joint data (~1000Hz)
    PRAXIS_EVENT = "praxis_event"            # Assembled practice events (async)
    SKILL_EXECUTION = "skill_execution"      # Skill start/complete (async)
    SWARM_MESSAGE = "swarm_message"          # Multi-robot coordination (async)


@dataclass
class TimelineEntry:
    """Single entry on the unified timeline."""
    timestamp_ns: int            # Nanosecond-precision timestamp
    channel: TimelineChannel
    sequence: int                # Monotonic sequence within channel
    data: dict                   # Channel-specific payload
    correlation_id: Optional[str] = None  # Links entries across channels

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "channel": self.channel.value,
            "sequence": self.sequence,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }


class UnifiedTimeline(LifecycleMixin):
    """
    Multi-channel timeline for practice recording.

    Subscribes to ALL EventBus channels and records entries on a
    single timeline. Assembles PraxisEvent when practice session ends.

    Architecture:
        Layer 1: LLM Reasoning   (~1 Hz)   — CoT traces, agent instructions
        Layer 2: Agent Commands   (~50 Hz)  — Movement commands, firewall results
        Layer 3: Sensorimotor    (~1000 Hz) — Joint positions/velocities/torques
        Layer 4: Events          (async)   — PraxisEvent, skill completion, swarm

    EventBus:
        Subscribes: agent.command, firewall.response.*, praxis.completed,
                    skill.execution.*, swarm.message.*
        Publishes:  timeline.recorded, praxis.recorded

    Note: Sensorimotor data (1kHz) bypasses EventBus via record_sensorimotor()
    to avoid EventBus overhead.
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: EventBus,
        output_dir: str = "./practice_data",
        enable_mcap: bool = False,
        buffer_size: int = 100_000,
    ):
        super().__init__()
        self._robot_id = robot_id
        self._event_bus = event_bus
        self._output_dir = Path(output_dir)
        self._enable_mcap = enable_mcap
        self._buffer_size = buffer_size

        # Timeline storage
        self._entries: list[TimelineEntry] = []
        self._sequence_counters: dict[TimelineChannel, int] = {
            ch: 0 for ch in TimelineChannel
        }

        # PraxisEvent assembly state
        self._pending_praxis: dict[str, dict] = {}  # correlation_id → partial state

        # MCAP writer (optional)
        self._mcap_writer: Optional[Any] = None

        # Sensorimotor ring buffer (bypasses EventBus)
        self._sensorimotor_buffer: list[TimelineEntry] = []
        self._sensorimotor_max = 10_000  # ~10 seconds at 1kHz

    def _do_initialize(self) -> None:
        """Subscribe to all relevant EventBus channels."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Subscribe to EventBus channels
        self._event_bus.subscribe("agent.command", self._on_agent_command)
        self._event_bus.subscribe("firewall.response.*", self._on_firewall_result)
        self._event_bus.subscribe("praxis.completed", self._on_praxis_completed)
        self._event_bus.subscribe("praxis.failed", self._on_praxis_failed)
        self._event_bus.subscribe("skill.execution.started", self._on_skill_event)
        self._event_bus.subscribe("skill.execution.completed", self._on_skill_event)
        self._event_bus.subscribe("swarm.message", self._on_swarm_message)

        # Initialize MCAP if enabled
        if self._enable_mcap:
            try:
                from mcap.writer import Writer
                # MCAP writer will be created per-session
                self._mcap_writer = True  # placeholder
            except ImportError:
                print("[UnifiedTimeline] mcap not available, using JSONL only")
                self._enable_mcap = False

        print(f"[UnifiedTimeline] Initialized for {self._robot_id}, "
              f"MCAP={'enabled' if self._enable_mcap else 'disabled'}, "
              f"buffer_size={self._buffer_size}")

    def _do_start(self) -> None:
        self._event_bus.publish(Event(
            topic="timeline.status",
            payload={"state": "running", "robot_id": self._robot_id},
            source="unified_timeline",
        ))

    def _do_stop(self) -> None:
        """Flush buffers and close MCAP writer."""
        self._flush_pending_praxis()
        if self._mcap_writer:
            self._mcap_writer = None

    # ── EventBus Handlers ──

    def _on_agent_command(self, event: Event) -> None:
        self._record(TimelineChannel.AGENT_COMMAND, event.payload,
                     correlation_id=event.payload.get("request_id"))

    def _on_firewall_result(self, event: Event) -> None:
        # Extract request_id from topic: firewall.response.{request_id}
        request_id = event.topic.split(".")[-1] if "." in event.topic else None
        self._record(TimelineChannel.FIREWALL_RESULT, event.payload,
                     correlation_id=request_id)

    def _on_praxis_completed(self, event: Event) -> None:
        """Assemble full PraxisEvent from timeline entries."""
        payload = event.payload
        correlation_id = payload.get("correlation_id", "unknown")

        # Gather all entries with this correlation_id
        related_entries = [
            e for e in self._entries
            if e.correlation_id == correlation_id
        ]

        # Extract components
        llm_entries = [e for e in related_entries if e.channel == TimelineChannel.LLM_REASONING]
        cmd_entries = [e for e in related_entries if e.channel == TimelineChannel.AGENT_COMMAND]
        fw_entries = [e for e in related_entries if e.channel == TimelineChannel.FIREWALL_RESULT]
        sensor_entries = [e for e in self._sensorimotor_buffer
                         if e.correlation_id == correlation_id]

        # Assemble PraxisEvent
        cot_trace = [e.data.get("reasoning", "") for e in llm_entries]
        trajectory = [e.data.get("waypoint", []) for e in cmd_entries]

        praxis_event = PraxisEvent(
            event_id=payload.get("event_id", correlation_id),
            event_type="success",
            timestamp=time.time(),
            robot_id=self._robot_id,
            agent_instruction=payload.get("instruction", ""),
            cot_trace=cot_trace,
            initial_state=payload.get("initial_state"),  # RobotState
            final_state=payload.get("final_state"),      # RobotState
            trajectory=trajectory,
            mcap_path=payload.get("mcap_path"),
            error_details=None,
            duration_sec=payload.get("duration_sec", 0.0),
            metadata={
                "firewall_results": [e.data for e in fw_entries],
                "sensorimotor_count": len(sensor_entries),
                "timeline_entries": len(related_entries),
            },
        )

        # Record the PraxisEvent on the timeline
        self._record(TimelineChannel.PRAXIS_EVENT, {
            "event_id": praxis_event.event_id,
            "event_type": praxis_event.event_type,
            "trajectory_waypoints": len(praxis_event.trajectory),
            "cot_steps": len(praxis_event.cot_trace),
        }, correlation_id=correlation_id)

        # Publish assembled PraxisEvent
        self._event_bus.publish(Event(
            topic="praxis.recorded",
            payload={
                "event_id": praxis_event.event_id,
                "event_type": praxis_event.event_type,
                "robot_id": praxis_event.robot_id,
                "duration_sec": praxis_event.duration_sec,
                "trajectory_waypoints": len(praxis_event.trajectory),
                "cot_steps": len(praxis_event.cot_trace),
                "sensorimotor_samples": len(sensor_entries),
            },
            source="unified_timeline",
            priority=EventPriority.NORMAL,
        ))

        # Export timeline
        self._export_timeline(correlation_id, related_entries, sensor_entries)

    def _on_praxis_failed(self, event: Event) -> None:
        correlation_id = event.payload.get("correlation_id", "unknown")
        self._record(TimelineChannel.PRAXIS_EVENT, {
            "event_type": "failure",
            "error": event.payload.get("error", "unknown"),
        }, correlation_id=correlation_id)

    def _on_skill_event(self, event: Event) -> None:
        self._record(TimelineChannel.SKILL_EXECUTION, event.payload,
                     correlation_id=event.payload.get("skill_id"))

    def _on_swarm_message(self, event: Event) -> None:
        self._record(TimelineChannel.SWARM_MESSAGE, event.payload)

    # ── Direct Recording (bypasses EventBus for high-frequency data) ──

    def record_sensorimotor(
        self,
        joint_positions: list[float],
        joint_velocities: list[float],
        joint_torques: list[float],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Record sensorimotor data directly (1kHz, bypasses EventBus)."""
        entry = TimelineEntry(
            timestamp_ns=time.time_ns(),
            channel=TimelineChannel.SENSORIMOTOR,
            sequence=self._sequence_counters[TimelineChannel.SENSORIMOTOR],
            data={
                "positions": joint_positions,
                "velocities": joint_velocities,
                "torques": joint_torques,
            },
            correlation_id=correlation_id,
        )
        self._sequence_counters[TimelineChannel.SENSORIMOTOR] += 1
        self._sensorimotor_buffer.append(entry)

        # Ring buffer eviction
        if len(self._sensorimotor_buffer) > self._sensorimotor_max:
            self._sensorimotor_buffer = self._sensorimotor_buffer[-self._sensorimotor_max:]

    def record_llm_reasoning(
        self,
        instruction: str,
        reasoning_steps: list[str],
        correlation_id: str,
    ) -> None:
        """Record LLM Chain-of-Thought on the timeline."""
        self._record(TimelineChannel.LLM_REASONING, {
            "instruction": instruction,
            "reasoning_steps": reasoning_steps,
        }, correlation_id=correlation_id)

    # ── Internal ──

    def _record(
        self,
        channel: TimelineChannel,
        data: dict,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Record a timeline entry."""
        entry = TimelineEntry(
            timestamp_ns=time.time_ns(),
            channel=channel,
            sequence=self._sequence_counters[channel],
            data=data,
            correlation_id=correlation_id,
        )
        self._sequence_counters[channel] += 1
        self._entries.append(entry)

        # Buffer eviction
        if len(self._entries) > self._buffer_size:
            self._entries = self._entries[-self._buffer_size:]

    def _export_timeline(
        self,
        correlation_id: str,
        entries: list[TimelineEntry],
        sensor_entries: list[TimelineEntry],
    ) -> None:
        """Export timeline to JSONL (and optionally MCAP)."""
        session_dir = self._output_dir / f"session_{correlation_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Export event-level entries as JSONL
        jsonl_path = session_dir / "timeline.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

        # Export sensorimotor data as binary (NumPy)
        if sensor_entries:
            positions = np.array([e.data["positions"] for e in sensor_entries])
            velocities = np.array([e.data["velocities"] for e in sensor_entries])
            torques = np.array([e.data["torques"] for e in sensor_entries])
            timestamps = np.array([e.timestamp_ns for e in sensor_entries])

            np.savez_compressed(
                session_dir / "sensorimotor.npz",
                positions=positions,
                velocities=velocities,
                torques=torques,
                timestamps=timestamps,
            )

        print(f"[UnifiedTimeline] Exported {len(entries)} events + "
              f"{len(sensor_entries)} sensorimotor samples to {session_dir}")

    def _flush_pending_praxis(self) -> None:
        """Flush any incomplete PraxisEvent assemblies."""
        for cid, state in self._pending_praxis.items():
            print(f"[UnifiedTimeline] Flushing incomplete praxis: {cid}")
        self._pending_praxis.clear()

    # ── Query API ──

    def get_entries(
        self,
        channel: Optional[TimelineChannel] = None,
        correlation_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> list[TimelineEntry]:
        """Query timeline entries with optional filters."""
        result = self._entries
        if channel:
            result = [e for e in result if e.channel == channel]
        if correlation_id:
            result = [e for e in result if e.correlation_id == correlation_id]
        if start_ns:
            result = [e for e in result if e.timestamp_ns >= start_ns]
        if end_ns:
            result = [e for e in result if e.timestamp_ns <= end_ns]
        return result

    def get_summary(self) -> dict:
        """Get timeline summary."""
        channel_counts = {}
        for ch in TimelineChannel:
            count = sum(1 for e in self._entries if e.channel == ch)
            if count > 0:
                channel_counts[ch.value] = count
        return {
            "total_entries": len(self._entries),
            "sensorimotor_samples": len(self._sensorimotor_buffer),
            "channels": channel_counts,
            "buffer_size": self._buffer_size,
        }
```

### 3.3 Unified Timeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UnifiedTimeline                              │
│                                                                   │
│  Layer 1: LLM Reasoning (~1 Hz)                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │ CoT Step 1: "Robot needs to reach position A"   │            │
│  │ CoT Step 2: "Avoid obstacle at position B"       │            │
│  │ CoT Step 3: "Use joint-space interpolation"      │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                   │
│  Layer 2: Agent Commands (~50 Hz)                               │
│  ┌─────────────────────────────────────────────────┐            │
│  │ cmd: move_joints [0.1, -0.3, 0.5, ...]          │            │
│  │ fw_result: SAFE (3 layers checked)              │            │
│  │ cmd: move_joints [0.2, -0.2, 0.6, ...]          │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                   │
│  Layer 3: Sensorimotor (~1000 Hz, direct recording)             │
│  ┌─────────────────────────────────────────────────┐            │
│  │ t=0.001: pos=[0.1, ...] vel=[0.0, ...] τ=[0.0] │            │
│  │ t=0.002: pos=[0.1, ...] vel=[0.01, ...] τ=[0.1]│            │
│  │ t=0.003: pos=[0.1, ...] vel=[0.02, ...] τ=[0.1]│            │
│  │ ... (1000 entries/second)                        │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                   │
│  Layer 4: Events (async)                                        │
│  ┌─────────────────────────────────────────────────┐            │
│  │ praxis.recorded: {event_id, duration, ...}       │            │
│  │ skill.completed: {skill_id, success}             │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                   │
│  Output: session_{id}/timeline.jsonl + sensorimotor.npz         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 MCAP Integration Strategy

MCAP (https://mcap.dev) is the preferred format for robotics data:

```python
# Future: MCAP writer integration
if self._enable_mcap and self._mcap_writer:
    from mcap.writer import Writer
    import io

    mcap_path = session_dir / "timeline.mcap"
    with open(mcap_path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="rosclaw-timeline")

        # Schema for each channel
        for ch in TimelineChannel:
            schema_id = writer.register_schema(
                name=ch.value,
                encoding="jsonschema",
                data=json.dumps({"type": "object"}).encode(),
            )
            channel_id = writer.register_channel(
                schema_id=schema_id,
                topic=f"/rosclaw/{ch.value}",
                message_encoding="json",
            )

        # Write all entries
        for entry in sorted(all_entries, key=lambda e: e.timestamp_ns):
            writer.add_message(
                channel_id=channel_map[entry.channel],
                log_time=entry.timestamp_ns,
                data=json.dumps(entry.data).encode(),
                publish_time=entry.timestamp_ns,
            )

        writer.finish()
```

**For now**: JSONL + NumPy `.npz` is the default. MCAP is opt-in via `enable_mcap=True`.

---

## 4. Sprint 5: SeekDB Client

### 4.1 Problem Statement

The current `MemoryInterface` in `memory/interface.py` (54 LOC):
- Uses an in-memory Python list (`self._experiences = []`)
- No persistence across restarts
- No structured queries (similarity search, filtering)
- No integration with the broader SeekDB Knowledge Plane

### 4.2 SeekDB Knowledge Plane Architecture

SeekDB is the shared database infrastructure for ALL ROSClaw modules:

```
┌──────────────────────────────────────────────────────────┐
│                    SeekDB Knowledge Plane                  │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ experience   │  │ skill        │  │ knowledge    │   │
│  │ _graph       │  │ _metadata    │  │ _graph       │   │
│  │              │  │              │  │              │   │
│  │ Memory       │  │ SkillManager │  │ Know module  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ heuristic    │  │ eval         │                     │
│  │ _rules       │  │ _results     │                     │
│  │              │  │              │                     │
│  │ How module   │  │ Darwin       │                     │
│  └──────────────┘  └──────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

### 4.3 New File: `memory/seekdb_client.py`

```python
"""SeekDB Client - Knowledge Plane interface for ROSClaw.

Provides abstract SeekDBClient and concrete implementations:
- SeekDBMemoryClient: In-memory for testing
- SeekDBSQLiteClient: SQLite for single-machine deployment
- (Future) SeekDBPostgresClient: PostgreSQL for production

Sprint 5 of DESIGN_SPRINT3_5.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json
import time


# ── SeekDB Schema Definitions ──

SEEKDB_SCHEMAS = {
    "experience_graph": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "event_type": "TEXT NOT NULL",
            "robot_id": "TEXT NOT NULL",
            "timestamp": "REAL NOT NULL",
            "instruction": "TEXT",
            "cot_trace": "TEXT",          # JSON array
            "trajectory": "TEXT",          # JSON array
            "outcome": "TEXT",             # "success" | "failure" | "emergency"
            "duration_sec": "REAL",
            "error_details": "TEXT",
            "tags": "TEXT",                # JSON array of strings
            "metadata": "TEXT",            # JSON object
        },
        "indices": ["robot_id", "event_type", "outcome", "timestamp"],
    },
    "skill_metadata": {
        "columns": {
            "skill_id": "TEXT PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "description": "TEXT",
            "category": "TEXT",
            "source": "TEXT",              # "json" | "demonstration" | "learned"
            "success_count": "INTEGER DEFAULT 0",
            "failure_count": "INTEGER DEFAULT 0",
            "avg_duration_sec": "REAL",
            "last_used": "REAL",
            "prerequisites": "TEXT",       # JSON array of skill_ids
            "metadata": "TEXT",            # JSON object
        },
        "indices": ["name", "category", "source"],
    },
    "knowledge_graph": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "subject": "TEXT NOT NULL",
            "predicate": "TEXT NOT NULL",
            "object": "TEXT NOT NULL",
            "confidence": "REAL DEFAULT 1.0",
            "source": "TEXT",              # Where this triple came from
            "timestamp": "REAL NOT NULL",
        },
        "indices": ["subject", "predicate", "object"],
    },
    "heuristic_rules": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "condition": "TEXT NOT NULL",   # JSON condition
            "action": "TEXT NOT NULL",       # JSON action
            "priority": "INTEGER DEFAULT 0",
            "success_count": "INTEGER DEFAULT 0",
            "failure_count": "INTEGER DEFAULT 0",
            "last_triggered": "REAL",
        },
        "indices": ["priority"],
    },
}


# ── Abstract Client ──

class SeekDBClient(ABC):
    """Abstract interface to SeekDB Knowledge Plane."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to SeekDB."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        ...

    @abstractmethod
    def insert(self, table: str, record: dict) -> str:
        """Insert a record, return its ID."""
        ...

    @abstractmethod
    def query(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query records with optional filtering."""
        ...

    @abstractmethod
    def update(self, table: str, record_id: str, updates: dict) -> bool:
        """Update a record by ID."""
        ...

    @abstractmethod
    def count(self, table: str, filters: Optional[dict] = None) -> int:
        """Count records matching filters."""
        ...


# ── In-Memory Implementation (Testing) ──

class SeekDBMemoryClient(SeekDBClient):
    """In-memory SeekDB client for testing."""

    def __init__(self):
        self._tables: dict[str, dict[str, dict]] = {}

    def connect(self) -> None:
        for table_name in SEEKDB_SCHEMAS:
            self._tables[table_name] = {}

    def disconnect(self) -> None:
        pass

    def insert(self, table: str, record: dict) -> str:
        if table not in self._tables:
            self._tables[table] = {}
        record_id = record.get("id", str(len(self._tables[table])))
        self._tables[table][record_id] = dict(record)
        return record_id

    def query(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        if table not in self._tables:
            return []

        records = list(self._tables[table].values())

        # Apply filters
        if filters:
            filtered = []
            for r in records:
                match = all(
                    r.get(k) == v for k, v in filters.items()
                )
                if match:
                    filtered.append(r)
            records = filtered

        # Sort
        if order_by:
            reverse = order_by.startswith("-")
            key = order_by.lstrip("-")
            records.sort(key=lambda r: r.get(key, 0), reverse=reverse)

        return records[:limit]

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        if table not in self._tables or record_id not in self._tables[table]:
            return False
        self._tables[table][record_id].update(updates)
        return True

    def count(self, table: str, filters: Optional[dict] = None) -> int:
        return len(self.query(table, filters, limit=1_000_000))


# ── SQLite Implementation (Single-Machine) ──

class SeekDBSQLiteClient(SeekDBClient):
    """SQLite-backed SeekDB client for single-machine deployment."""

    def __init__(self, db_path: str = "./seekdb.sqlite"):
        self._db_path = db_path
        self._conn = None

    def connect(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        for table_name, schema in SEEKDB_SCHEMAS.items():
            cols = ", ".join(f"{k} {v}" for k, v in schema["columns"].items())
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})")
            for idx_col in schema.get("indices", []):
                idx_name = f"idx_{table_name}_{idx_col}"
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({idx_col})"
                )
        self._conn.commit()

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def insert(self, table: str, record: dict) -> str:
        if table not in SEEKDB_SCHEMAS:
            raise ValueError(f"Unknown table: {table}")

        # Serialize complex fields
        serialized = {}
        for k, v in record.items():
            if isinstance(v, (list, dict)):
                serialized[k] = json.dumps(v)
            else:
                serialized[k] = v

        cols = ", ".join(serialized.keys())
        placeholders = ", ".join("?" for _ in serialized)
        values = list(serialized.values())

        record_id = serialized.get("id", str(int(time.time() * 1000)))
        if "id" not in serialized:
            serialized["id"] = record_id

        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        return record_id

    def query(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        sql = f"SELECT * FROM {table}"
        params = []

        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(conditions)

        if order_by:
            direction = "DESC" if order_by.startswith("-") else "ASC"
            key = order_by.lstrip("-")
            sql += f" ORDER BY {key} {direction}"

        sql += f" LIMIT {limit}"

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            record = dict(row)
            # Deserialize JSON fields
            for k, v in record.items():
                if isinstance(v, str) and (v.startswith("[") or v.startswith("{")):
                    try:
                        record[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            results.append(record)

        return results

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        if not updates:
            return False

        serialized = {}
        for k, v in updates.items():
            if isinstance(v, (list, dict)):
                serialized[k] = json.dumps(v)
            else:
                serialized[k] = v

        set_clause = ", ".join(f"{k} = ?" for k in serialized)
        values = list(serialized.values()) + [record_id]

        cursor = self._conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE id = ?",
            values,
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self, table: str, filters: Optional[dict] = None) -> int:
        sql = f"SELECT COUNT(*) FROM {table}"
        params = []

        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()[0]
```

### 4.4 Rewritten `memory/interface.py`

```python
"""MemoryInterface - Experience Grounding backed by SeekDB.

Replaces the in-memory list with SeekDB persistence.
Subscribes to praxis.recorded events to auto-ingest experiences.

Sprint 5 of DESIGN_SPRINT3_5.
"""

from typing import Optional
import json
import time

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.memory.seekdb_client import SeekDBClient, SeekDBMemoryClient


class MemoryInterface(LifecycleMixin):
    """
    Experience Grounding engine backed by SeekDB.

    Stores PraxisEvents as experiences in the experience_graph table.
    Provides similarity search for finding relevant past experiences.

    EventBus:
        Subscribes: praxis.recorded (to auto-ingest new experiences)
        Publishes:  memory.experience.stored
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: Optional[EventBus] = None,
        seekdb_client: Optional[SeekDBClient] = None,
    ):
        super().__init__()
        self._robot_id = robot_id
        self._event_bus = event_bus
        self._client = seekdb_client or SeekDBMemoryClient()

    def _do_initialize(self) -> None:
        self._client.connect()

        if self._event_bus:
            self._event_bus.subscribe("praxis.recorded", self._on_praxis_recorded)

        print(f"[MemoryInterface] Initialized for {self._robot_id}, "
              f"backend={type(self._client).__name__}")

    def _do_start(self) -> None:
        if self._event_bus:
            self._event_bus.publish(Event(
                topic="memory.status",
                payload={
                    "state": "running",
                    "robot_id": self._robot_id,
                    "experience_count": self._client.count("experience_graph"),
                },
                source="memory_interface",
            ))

    def _do_stop(self) -> None:
        self._client.disconnect()

    def _on_praxis_recorded(self, event: Event) -> None:
        """Auto-ingest PraxisEvent as an experience."""
        payload = event.payload
        self.store_experience(
            event_id=payload.get("event_id", ""),
            event_type=payload.get("event_type", "unknown"),
            instruction=payload.get("instruction", ""),
            duration_sec=payload.get("duration_sec", 0.0),
            metadata=payload,
        )

    def store_experience(
        self,
        event_id: str,
        event_type: str,
        instruction: str,
        cot_trace: Optional[list[str]] = None,
        trajectory: Optional[list[list[float]]] = None,
        outcome: str = "success",
        duration_sec: float = 0.0,
        error_details: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a new experience in SeekDB."""
        record = {
            "id": event_id,
            "event_type": event_type,
            "robot_id": self._robot_id,
            "timestamp": time.time(),
            "instruction": instruction,
            "cot_trace": cot_trace or [],
            "trajectory": trajectory or [],
            "outcome": outcome,
            "duration_sec": duration_sec,
            "error_details": error_details,
            "tags": tags or [],
            "metadata": metadata or {},
        }

        record_id = self._client.insert("experience_graph", record)

        if self._event_bus:
            self._event_bus.publish(Event(
                topic="memory.experience.stored",
                payload={"experience_id": record_id, "event_type": event_type},
                source="memory_interface",
            ))

        return record_id

    def find_similar_experiences(
        self,
        instruction: str,
        limit: int = 5,
        outcome_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Find past experiences similar to the given instruction.

        Current implementation: keyword matching.
        Future: vector embeddings for semantic similarity.
        """
        filters = {"robot_id": self._robot_id}
        if outcome_filter:
            filters["outcome"] = outcome_filter

        all_experiences = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="-timestamp",
            limit=100,
        )

        # Simple keyword-based scoring
        keywords = set(instruction.lower().split())
        scored = []
        for exp in all_experiences:
            exp_text = (exp.get("instruction", "") + " " +
                       " ".join(exp.get("tags", []))).lower()
            exp_words = set(exp_text.split())
            overlap = len(keywords & exp_words)
            if overlap > 0:
                scored.append((overlap, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

    def get_experience(self, experience_id: str) -> Optional[dict]:
        """Retrieve a single experience by ID."""
        results = self._client.query(
            "experience_graph",
            filters={"id": experience_id},
            limit=1,
        )
        return results[0] if results else None

    def get_statistics(self) -> dict:
        """Get experience statistics."""
        total = self._client.count("experience_graph")
        successes = self._client.count("experience_graph", {"outcome": "success"})
        failures = self._client.count("experience_graph", {"outcome": "failure"})
        emergencies = self._client.count("experience_graph", {"outcome": "emergency"})

        return {
            "total_experiences": total,
            "success_count": successes,
            "failure_count": failures,
            "emergency_count": emergencies,
            "success_rate": successes / total if total > 0 else 0.0,
        }
```

### 4.5 SeekDB Entity-Relationship Diagram

```
┌─────────────────────┐     ┌─────────────────────┐
│  experience_graph    │     │  skill_metadata      │
│                      │     │                      │
│  id (PK)             │     │  skill_id (PK)       │
│  event_type          │     │  name                │
│  robot_id            │     │  description         │
│  timestamp           │     │  category            │
│  instruction         │     │  source              │
│  cot_trace (JSON)    │     │  success_count       │
│  trajectory (JSON)   │     │  failure_count       │
│  outcome             │     │  avg_duration_sec    │
│  duration_sec        │     │  last_used           │
│  error_details       │     │  prerequisites (JSON)│
│  tags (JSON)         │     │  metadata (JSON)     │
│  metadata (JSON)     │     └─────────────────────┘
└─────────────────────┘
         │                         │
         │  experience_id          │  skill_id
         │  referenced in          │  referenced in
         ▼                         ▼
┌─────────────────────┐     ┌─────────────────────┐
│  knowledge_graph     │     │  heuristic_rules     │
│                      │     │                      │
│  id (PK)             │     │  id (PK)             │
│  subject             │     │  condition (JSON)    │
│  predicate           │     │  action (JSON)       │
│  object              │     │  priority            │
│  confidence          │     │  success_count       │
│  source              │     │  failure_count       │
│  timestamp           │     │  last_triggered      │
└─────────────────────┘     └─────────────────────┘
```

---

## 5. Cross-Sprint Integration

### 5.1 End-to-End Pipeline

```
LLM says: "Pick up the red block and place it on the table"
    │
    ▼
MCPHub.handle_tool_call("execute_task", {instruction: "..."})
    │
    ├── UnifiedTimeline.record_llm_reasoning(instruction, cot_steps, correlation_id)
    │
    ├── publishes: agent.command {action: "move_joints", trajectory: [...]}
    │
    ▼
FirewallValidator._on_agent_command()
    ├── Layer 1: e-URDF soft limits ✓
    ├── Layer 2: MuJoCo collision ✓
    ├── Layer 3: Semantic safety ✓
    ├── publishes: firewall.response.{id} {is_safe: true}
    │
    ▼
MCPDriver.execute_trajectory()
    │
    ├── UnifiedTimeline.record_sensorimotor(positions, velocities, torques)
    │   (1000 samples/second, direct recording, bypasses EventBus)
    │
    ▼
Execution complete
    │
    ├── publishes: praxis.completed {correlation_id, event_id, ...}
    │
    ▼
UnifiedTimeline._on_praxis_completed()
    ├── Assembles PraxisEvent from timeline entries
    ├── Exports: session_{id}/timeline.jsonl + sensorimotor.npz
    ├── publishes: praxis.recorded {event_id, event_type, ...}
    │
    ▼
MemoryInterface._on_praxis_recorded()
    ├── Stores experience in SeekDB experience_graph
    ├── publishes: memory.experience.stored
    │
    ▼
SkillRegistry (if applicable)
    ├── Updates skill success/failure count in SeekDB skill_metadata
```

### 5.2 Module Interaction Matrix

| Publisher → Subscriber | FirewallValidator | UnifiedTimeline | MemoryInterface | SkillExecutor |
|------------------------|:-:|:-:|:-:|:-:|
| `agent.command` | **sub** | **sub** | — | — |
| `firewall.response.*` | **pub** | **sub** | — | — |
| `safety.violation` | **pub** | — | — | — |
| `praxis.completed` | — | **sub** | — | — |
| `praxis.recorded` | — | **pub** | **sub** | **sub** |
| `memory.experience.stored` | — | — | **pub** | — |
| `skill.execution.*` | — | **sub** | — | **pub** |

---

## 6. EventBus Topic Registry

Complete registry of all EventBus topics used across Sprints 3-5:

| Topic | Publisher | Subscriber(s) | Priority | Description |
|-------|-----------|---------------|----------|-------------|
| `agent.command` | MCPHub | FirewallValidator, UnifiedTimeline | NORMAL | Movement command from agent |
| `firewall.response.{id}` | FirewallValidator | MCPHub, UnifiedTimeline | HIGH/CRITICAL | Validation result |
| `safety.violation` | FirewallValidator | Runtime | CRITICAL | Safety violation detected |
| `firewall.status` | FirewallValidator | (monitoring) | NORMAL | Firewall lifecycle status |
| `praxis.completed` | PracticeRecorder / Driver | UnifiedTimeline | NORMAL | Physical execution completed |
| `praxis.failed` | PracticeRecorder / Driver | UnifiedTimeline | HIGH | Physical execution failed |
| `praxis.recorded` | UnifiedTimeline | MemoryInterface, SkillExecutor | NORMAL | PraxisEvent assembled & recorded |
| `timeline.status` | UnifiedTimeline | (monitoring) | NORMAL | Timeline lifecycle status |
| `timeline.recorded` | UnifiedTimeline | (future consumers) | NORMAL | Individual timeline entry |
| `memory.status` | MemoryInterface | (monitoring) | NORMAL | Memory lifecycle status |
| `memory.experience.stored` | MemoryInterface | SkillExecutor | NORMAL | New experience stored |
| `skill.execution.started` | SkillExecutor | UnifiedTimeline | NORMAL | Skill execution begun |
| `skill.execution.completed` | SkillExecutor | UnifiedTimeline | NORMAL | Skill execution finished |
| `runtime.status` | Runtime | (monitoring) | HIGH | Runtime lifecycle status |
| `robot.emergency_stop` | Runtime | All drivers | CRITICAL | Emergency stop all robots |

---

## 7. Initialization Ordering

### 7.1 The Question

> "Runtime先初始化EventBus，但模块需要Bus来发布自己的ready事件？"
> (Runtime initializes EventBus first, but modules need the Bus to publish ready events?)

### 7.2 The Answer

**Modules should NOT publish during initialization.** Here is the correct ordering:

```
Runtime.__init__()
    │
    ├── self.event_bus = EventBus()          # 1. Create EventBus FIRST
    │
Runtime._do_initialize()
    │
    ├── self._memory = MemoryInterface(       # 2. Create modules, PASS event_bus
    │       robot_id, event_bus=self.event_bus)
    │
    ├── self._memory.initialize()             # 3. Call initialize() on each module
    │       └── _do_initialize():
    │           └── event_bus.subscribe(...)  #    → Subscribes only, NO publishing
    │
    ├── self._practice = PracticeRecorder(    # 4. Same pattern for all modules
    │       robot_id, event_bus=self.event_bus)
    │   self._practice.initialize()
    │       └── subscribes to topics
    │
    └── ... (all modules initialized)
    │
Runtime._do_start()
    │
    ├── for module in self._modules:
    │       module.start()                    # 5. Start each module
    │           └── _do_start():
    │               └── MAY publish status     #    → OK to publish AFTER start
    │
    └── event_bus.publish(                    # 6. Runtime publishes aggregate status
            topic="runtime.status",
            payload={"state": "running"})
```

### 7.3 Rules

1. **`__init__`**: Create EventBus, store references — no subscriptions, no publishing
2. **`_do_initialize()`**: Subscribe to topics, load configs — **NO publishing**
3. **`_do_start()`**: May publish status events — **publishing is allowed**
4. **`_do_stop()`**: May publish shutdown events — publishing is allowed

### 7.4 Why This Works

- EventBus exists before any module needs it ✓
- No race condition: subscriptions happen before any publishing ✓
- "Ready" signaling: Runtime publishes `runtime.status` after ALL modules are started ✓
- Individual modules publish their own `*.status` in `_do_start()` if needed ✓

---

## 8. Test Strategy

### 8.1 Sprint 3 Tests: `tests/test_firewall_validator.py`

```python
# Test 1: SafetyEnvelope extraction from RobotModel
def test_safety_envelope_from_robot_model():
    """Verify soft limits are 95% of hard limits for MODERATE."""

# Test 2: e-URDF limit violation detection
def test_eurdf_limit_violation():
    """Trajectory exceeding joint soft limits → critical violation."""

# Test 3: EventBus integration — SAFE command passes through
def test_safe_command_passes():
    """Valid trajectory → firewall.response with is_safe=True."""

# Test 4: EventBus integration — UNSAFE command blocked
def test_unsafe_command_blocked():
    """Out-of-limits trajectory → safety.violation published."""
```

### 8.2 Sprint 4 Tests: `tests/test_timeline.py`

```python
# Test 1: Multi-channel recording
def test_multi_channel_recording():
    """Record entries on all channels, verify ordering."""

# Test 2: Sensorimotor direct recording (1kHz)
def test_sensorimotor_direct_recording():
    """record_sensorimotor() bypasses EventBus, stores at 1kHz."""

# Test 3: PraxisEvent assembly
def test_praxis_event_assembly():
    """praxis.completed → assembled PraxisEvent with all channels."""

# Test 4: JSONL + NPZ export
def test_timeline_export():
    """Verify exported files contain correct data."""
```

### 8.3 Sprint 5 Tests: `tests/test_seekdb.py`

```python
# Test 1: SeekDBMemoryClient CRUD
def test_memory_client_crud():
    """Insert, query, update, count on in-memory backend."""

# Test 2: SeekDBSQLiteClient CRUD
def test_sqlite_client_crud():
    """Same operations on SQLite backend."""

# Test 3: MemoryInterface experience storage
def test_experience_storage():
    """store_experience() → findable in experience_graph."""

# Test 4: Similarity search
def test_similarity_search():
    """find_similar_experiences() returns keyword-matched results."""

# Test 5: PraxisEvent auto-ingestion
def test_praxis_auto_ingestion():
    """praxis.recorded event → experience auto-stored."""
```

---

## 9. File Change Manifest

### New Files (8)

| File | Sprint | LOC (est.) | Description |
|------|--------|-----------|-------------|
| `src/rosclaw/firewall/validator.py` | 3 | ~280 | FirewallValidator with e-URDF + MuJoCo |
| `src/rosclaw/practice/timeline.py` | 4 | ~320 | UnifiedTimeline multi-channel recorder |
| `src/rosclaw/memory/seekdb_client.py` | 5 | ~250 | SeekDBClient ABC + Memory + SQLite impls |
| `tests/test_firewall_validator.py` | 3 | ~150 | 4 tests for Sprint 3 |
| `tests/test_timeline.py` | 4 | ~150 | 4 tests for Sprint 4 |
| `tests/test_seekdb.py` | 5 | ~180 | 5 tests for Sprint 5 |
| `src/rosclaw/firewall/__init__.py` | 3 | ~15 | Export FirewallValidator |
| `src/rosclaw/practice/__init__.py` | 4 | ~10 | Export UnifiedTimeline |

### Modified Files (6)

| File | Sprint | Change |
|------|--------|--------|
| `src/rosclaw/memory/interface.py` | 5 | Rewrite to use SeekDBClient instead of in-memory list |
| `src/rosclaw/core/runtime.py` | 3-5 | Add FirewallValidator, UnifiedTimeline init; add safety_level config |
| `src/rosclaw/core/runtime.py` | 7 | Document initialization ordering rules |
| `src/rosclaw/agent_runtime/mcp_hub.py` | 3 | Add request-response pattern (await firewall.response) |
| `src/rosclaw/__init__.py` | 3-5 | Export new classes: FirewallValidator, UnifiedTimeline, SeekDBClient |
| `pyproject.toml` | 5 | Add optional deps: mcap, sqlite3 (stdlib) |

---

## 10. Acceptance Criteria

### Sprint 3 Acceptance

```bash
# All existing tests still pass
pytest tests/ -v  # 102+ tests

# New firewall validator tests pass
pytest tests/test_firewall_validator.py -v  # 4 tests

# Demonstration: unsafe trajectory blocked via EventBus
python -c "
from rosclaw.firewall.validator import FirewallValidator, ValidationRequest
from rosclaw.core.event_bus import EventBus

bus = EventBus()
# ... setup and validate
"
```

### Sprint 4 Acceptance

```bash
# Timeline tests pass
pytest tests/test_timeline.py -v  # 4 tests

# Demonstration: record and export timeline
python -c "
from rosclaw.practice.timeline import UnifiedTimeline
from rosclaw.core.event_bus import EventBus

bus = EventBus()
timeline = UnifiedTimeline('test_robot', bus, output_dir='/tmp/test_timeline')
timeline.initialize()
timeline.start()

# Record some data
timeline.record_sensorimotor([0.1]*6, [0.0]*6, [0.0]*6, correlation_id='test1')
timeline.record_llm_reasoning('pick up block', ['step1', 'step2'], 'test1')

print(timeline.get_summary())
"
```

### Sprint 5 Acceptance

```bash
# SeekDB tests pass
pytest tests/test_seekdb.py -v  # 5 tests

# Demonstration: store and query experiences
python -c "
from rosclaw.memory.seekdb_client import SeekDBSQLiteClient
from rosclaw.memory.interface import MemoryInterface

client = SeekDBSQLiteClient('/tmp/test_seekdb.sqlite')
memory = MemoryInterface('test_robot', seekdb_client=client)
memory.initialize()
memory.start()

memory.store_experience('exp1', 'success', 'pick up red block', duration_sec=3.2)
memory.store_experience('exp2', 'failure', 'pick up blue block', outcome='failure')

results = memory.find_similar_experiences('pick up block')
print(f'Found {len(results)} similar experiences')
print(memory.get_statistics())
"
```

### Cross-Sprint Acceptance

```bash
# ALL tests pass (existing + new)
pytest tests/ -v  # 102 + 4 + 4 + 5 = 115+ tests

# End-to-end pipeline demonstration
python -c "
from rosclaw.core.runtime import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id='test_robot',
    enable_firewall=True,
    enable_memory=True,
    enable_practice=True,
    safety_level='MODERATE',
)
runtime = Runtime(config)
runtime.initialize()
runtime.start()

status = runtime.get_status()
print(f'Runtime state: {status[\"runtime_state\"]}')
print(f'Modules active: {sum(1 for v in status[\"modules\"].values() if v)}')

runtime.stop()
"
```

---

## Appendix A: Configuration Extensions

New fields for `RuntimeConfig`:

```python
@dataclass
class RuntimeConfig:
    # ... existing fields ...
    safety_level: str = "MODERATE"          # STRICT | MODERATE | LENIENT
    enable_unified_timeline: bool = True
    timeline_output_dir: str = "./practice_data"
    enable_mcap: bool = False
    seekdb_backend: str = "memory"          # "memory" | "sqlite" | "postgres"
    seekdb_path: str = "./seekdb.sqlite"
```

## Appendix B: Future Work (Out of Scope)

- **Sprint 6+**: Swarm DDS integration for multi-robot coordination
- **Sprint 7+**: Darwin evaluation engine for automated skill improvement
- **Sprint 8+**: Know module for knowledge graph construction
- **Sprint 9+**: How module for heuristic rule learning
- **Vector embeddings**: Replace keyword matching in `find_similar_experiences()` with embedding-based semantic search
- **MCAP production**: Full MCAP writer integration replacing JSONL+NPZ
- **SeekDB PostgreSQL**: Production-grade SeekDB backend with connection pooling
