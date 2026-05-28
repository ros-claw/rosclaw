# Swarm v1.1 Integration Checklist

> **Owner**: Swarm Domain (rosclaw-swarm)
> **Date**: 2026-05-28
> **Status**: PLANNED — items to execute after v1.0 ships
> **Prerequisite**: v1.0 release frozen (see [audit-swarm.md](audits/audit-swarm.md))

---

## Summary

Swarm is architecturally ready for v1.0. This checklist tracks the integration work needed for v1.1 to connect Swarm with the rest of the ROSClaw ecosystem (e-URDF, EventBus, DDS, Practice, Memory).

| # | Item | Effort | Dependency | Priority |
|---|------|--------|-----------|----------|
| 1 | e-URDF capabilities.yaml reader | ~50 lines | e-urdf-zoo | P1 |
| 2 | Swap local event bus for rosclaw-event-bus | ~5 lines | rosclaw-event-bus | P1 |
| 3 | Add dds_bridge.py with real ROS 2 DDS | ~200 lines | ROS 2 Humble | P2 |
| 4 | Add spatial_sync.py with tf2_ros | ~150 lines | ROS 2 tf2_ros | P2 |
| 5 | Connect SwarmReflexMessage to Practice MCAP | ~80 lines | practice/timeline | P2 |

---

## Item 1: AgentCapabilities.from_eurdf()

**Goal**: Read `capabilities.yaml` from e-URDF Zoo instead of manual agent setup.

**Current state**: `AgentCapabilities` model exists and matches the RFC schema perfectly. Agents are registered manually or via `agents_example.json`.

**Implementation**:

```python
# rosclaw_swarm/eurdf_bridge.py (new file, ~50 lines)
import yaml
from pathlib import Path
from rosclaw_swarm.models import AgentCapabilities, Capability

def load_capabilities_from_eurdf(robot_name: str, zoo_path: str) -> AgentCapabilities:
    """Read capabilities.yaml from e-URDF Zoo."""
    cap_file = Path(zoo_path) / robot_name / "capabilities.yaml"
    with open(cap_file) as f:
        data = yaml.safe_load(f)

    capabilities = [
        Capability(
            name=c["name"],
            skill=c.get("skill"),
            success_rate=c.get("success_rate", 0.5),
            latency_ms=c.get("latency_ms"),
        )
        for c in data.get("capabilities", [])
    ]

    return AgentCapabilities(
        agent_id=data.get("agent_id", robot_name),
        hardware_type=data.get("hardware_type", robot_name),
        dof=data.get("dof"),
        payload_limit_kg=data.get("payload_limit_kg"),
        capabilities=capabilities,
        metadata={"eurdf.zoo_path": str(cap_file)},
    )
```

**Acceptance criteria**:
- [ ] `load_capabilities_from_eurdf("ur5e", "/path/to/zoo")` returns valid `AgentCapabilities`
- [ ] Works with existing `e-urdf-zoo` YAML format
- [ ] Unit test covers missing file, malformed YAML, and valid input
- [ ] Added to `pyproject.toml` as optional `eurdf` dependency: `pyyaml>=6.0`

**Dependencies**: e-urdf-zoo must publish `capabilities.yaml` schema.

---

## Item 2: Swap local event bus for rosclaw-event-bus

**Goal**: Replace the internal `_event_handlers` dict in `SwarmRuntimeManager` with the unified `rosclaw-event-bus`.

**Current state**: Manager has a working local pub/sub (`subscribe()` / `publish()`). Event payloads already carry `agent_id`, `task_id`, and full `SwarmContext`.

**Implementation** (~5 lines changed in `manager.py`):

```python
# BEFORE (current):
self._event_handlers: Dict[str, List[Callable]] = {}

def subscribe(self, event_type: str, handler: Callable) -> None:
    self._event_handlers.setdefault(event_type, []).append(handler)

def publish(self, event_type: str, payload: Any) -> None:
    for handler in self._event_handlers.get(event_type, []):
        handler(payload)

# AFTER (v1.1):
from rosclaw.core.event_bus import EventBus

self._bus = EventBus()  # or inject from Runtime

def subscribe(self, event_type: str, handler: Callable) -> None:
    self._bus.subscribe(f"swarm.{event_type}", handler)

def publish(self, event_type: str, payload: Any) -> None:
    self._bus.publish(f"swarm.{event_type}", payload)
```

**Acceptance criteria**:
- [ ] `SwarmSessionCreatedEvent` visible on unified EventBus as `swarm.SwarmSessionCreatedEvent`
- [ ] Other modules can subscribe to swarm events via `EventBus.subscribe()`
- [ ] All existing tests pass without modification
- [ ] No direct cross-module imports introduced

**Dependencies**: `rosclaw-event-bus` must be available as a package.

**Risk**: Low — interface is already abstracted behind subscribe/publish.

---

## Item 3: Add dds_bridge.py with real ROS 2 DDS

**Goal**: Implement real DDS communication for the Micro Reflex Plane (1000Hz).

**Current state**: `SwarmReflexMessage` model is spec-complete (matches the RFC `.msg` definition). `SwarmSimulator` provides synthetic reflex messages for testing. No real DDS transport exists.

**Implementation** (~200 lines):

```python
# rosclaw_swarm/dds_bridge.py (new file)
# Requires: rclpy, rosclaw_msgs (optional ros2 dependency)

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

def get_reflex_qos_profile() -> QoSProfile:
    """<5ms physical handshake QoS policy."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

class DDSReflexBridge:
    """Manages DDS pub/sub for SwarmReflexMessage."""
    def __init__(self, agent_id: str, domain_id: int = 42):
        ...
    def publish_reflex(self, msg: SwarmReflexMessage) -> None:
        ...
    def subscribe_peer(self, peer_id: str, callback) -> None:
        ...
```

**Acceptance criteria**:
- [ ] Two ROS 2 nodes exchange `SwarmReflexMessage` at >= 200Hz
- [ ] End-to-end latency < 5ms on loopback
- [ ] Graceful fallback to simulation when `rclpy` not available
- [ ] QoS profile matches RFC spec (BEST_EFFORT, VOLATILE, KEEP_LAST depth=1)

**Dependencies**: ROS 2 Humble, `rclpy`, `rosclaw_msgs` package with `.msg` definition.

---

## Item 4: Add spatial_sync.py with tf2_ros

**Goal**: Fuse multiple robot TF trees under a unified `swarm_world` frame.

**Current state**: `SwarmContext.shared_world_frame` field exists (default: `"swarm_world_001"`). `SwarmSimulator` tracks agent poses in a flat dict. No real TF fusion.

**Implementation** (~150 lines):

```python
# rosclaw_swarm/spatial_sync.py (new file)
import tf2_ros
import geometry_msgs.msg

class SwarmSpatialSynchronizer:
    """Merges robot TF trees under a shared world frame."""
    def __init__(self, session_id: str, agent_ids: list):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(node)
        self.shared_frame = f"swarm_world_{session_id[:8]}"

    def broadcast_handoff_point(self, name: str, x: float, y: float, z: float):
        """Publish a 3D handoff point in the shared frame."""
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = self.shared_frame
        t.child_frame_id = f"handoff_{name}"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        self.tf_broadcaster.sendTransform(t)

    def register_agent_frame(self, agent_id: str, base_frame: str):
        """Mount agent's base frame under the shared world frame."""
        ...
```

**Acceptance criteria**:
- [ ] Two robot base frames visible under `swarm_world_*` in RViz
- [ ] Handoff point coordinates consistent across all agent frames
- [ ] Graceful fallback when `tf2_ros` not available

**Dependencies**: ROS 2 Humble, `tf2_ros`, `geometry_msgs`.

---

## Item 5: Connect SwarmReflexMessage to Practice MCAP timeline

**Goal**: Record swarm reflex messages into the MCAP black box for post-hoc analysis.

**Current state**: `SwarmReflexMessage` has `stamp_ns` (nanosecond timestamp) — ready for MCAP binding. `practice/timeline.py` already subscribes to `swarm.message` in the monorepo. No MCAP channel for reflex data exists.

**Implementation** (~80 lines):

```python
# In practice/timeline.py or a new swarm_recorder.py:
def on_swarm_reflex(self, msg: SwarmReflexMessage):
    """Record reflex message to MCAP swarm_reflex channel."""
    self.timeline.add_entry(
        channel="swarm.reflex",
        stamp_ns=msg.stamp_ns,
        data={
            "sender": msg.sender_agent_id,
            "tcp_pose": msg.current_tcp_pose,
            "wrench": msg.actual_wrench,
            "torques": msg.joint_torques,
            "intent": msg.intent_phase,
        },
    )
```

**Acceptance criteria**:
- [ ] MCAP file contains `swarm.reflex` channel with timestamped entries
- [ ] Entries align on the unified nanosecond timeline with `praxis.*` channels
- [ ] SeekDB can query reflex data by session_id and time range
- [ ] No performance degradation at 200Hz reflex rate

**Dependencies**: `practice/timeline.py` UnifiedTimeline, MCAP writer.

---

## Monitoring Checklist (v1.0 to v1.1 transition)

These items are checked continuously during the v1.0 integration phase:

| Check | Frequency | Status |
|-------|-----------|--------|
| All 29 swarm tests pass | After each integration merge | Baseline: 29/29 |
| No `swarm.*` topic conflicts | After each new EventBus topic added | Verified: no conflicts |
| Pydantic models unchanged | After each PR touching models.py | Frozen |
| No direct imports of `rosclaw_swarm.*` from other modules | Weekly grep | Clean |
| `metadata` fields not mutated by external code | Code review | Clean |

### Regression test schedule

| Integration Phase | Test Command | Status |
|-------------------|-------------|--------|
| After KNOW integration | `pytest tests/ -v` | Pending |
| After HOW integration | `pytest tests/ -v` | Pending |
| After DASHBOARD integration | `pytest tests/ -v` | Pending |
| After PROVIDER fix | `pytest tests/ -v` | Pending |
| After SANDBOX integration | `pytest tests/ -v` | Pending |

---

## Related Documents

- [Audit Report: Swarm](audits/audit-swarm.md)
- [Swarm Integration Seams](swarm_integration_seams.md)
- [Event Flow Map](integration/event-flow-map.md)
- [Dependency Map](integration/dependency-map.md)
- [RFC-0001: Architecture Freeze](RFC-0001-architecture-freeze.md)
- [ROADMAP v1.1](../../ROADMAP_v1.1.md)
