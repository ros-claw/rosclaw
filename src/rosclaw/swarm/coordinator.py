"""SwarmCoordinator — Multi-robot task decomposition and consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority


@dataclass
class TaskAllocation:
    """Result of allocating a task to agents."""
    task_id: str
    assignments: list[dict[str, Any]] = field(default_factory=list)
    feasible: bool = True
    reason: str = ""


@dataclass
class AgentBid:
    """Bid from an agent for a task."""
    agent_id: str
    task_id: str
    cost: float  # lower is better
    capabilities: list[str] = field(default_factory=list)
    estimated_duration_sec: float = 0.0


class SwarmCoordinator:
    """Coordinates multi-robot task decomposition and consensus.

    - Breaks complex tasks into subtasks
    - Runs auction-based allocation with bids
    - Achieves consensus on shared world state
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self._agents: dict[str, dict] = {}
        self._tasks: dict[str, dict] = {}
        self._bids: dict[str, list[AgentBid]] = {}  # task_id -> bids
        self._consensus_state: dict[str, Any] = {}

    # ── Agent management ──

    def register_agent(self, agent_id: str, capabilities: list[str], position: Optional[tuple[float, ...]] = None) -> None:
        """Register an agent with the coordinator."""
        self._agents[agent_id] = {
            "id": agent_id,
            "capabilities": capabilities,
            "position": position,
            "status": "idle",
            "current_task": None,
        }

    def deregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the swarm."""
        self._agents.pop(agent_id, None)

    # ── Task decomposition ──

    def decompose_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Decompose a complex task into ordered subtasks."""
        task_type = task.get("type", "single")

        if task_type == "parallel_pick":
            objects = task.get("objects", [])
            return [
                {
                    "id": f"{task.get('id', 'task')}_sub_{i}",
                    "type": "pick_and_place",
                    "object": obj,
                    "location": task.get("target_location", "table_center"),
                    "required_capabilities": ["skill.pick_and_place"],
                }
                for i, obj in enumerate(objects)
            ]

        if task_type == "sequential_assembly":
            steps = task.get("steps", [])
            return [
                {
                    "id": f"{task.get('id', 'task')}_sub_{i}",
                    "type": "assembly_step",
                    "step": step,
                    "required_capabilities": step.get("capabilities", ["skill.assembly"]),
                }
                for i, step in enumerate(steps)
            ]

        # Default: single task
        return [task]

    # ── Auction-based allocation ──

    def request_bids(self, task: dict[str, Any]) -> list[AgentBid]:
        """Request bids from all capable agents for a task."""
        required = task.get("required_capabilities", [])
        bids = []

        for agent_id, agent in self._agents.items():
            if agent["status"] != "idle":
                continue
            if not all(c in agent["capabilities"] for c in required):
                continue

            # Simple cost heuristic: 1.0 base + distance penalty
            cost = 1.0
            if agent.get("position") and task.get("target_position"):
                import math
                pos = agent["position"]
                target = task["target_position"]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, target)))
                cost += dist

            bids.append(AgentBid(
                agent_id=agent_id,
                task_id=task.get("id", "unknown"),
                cost=cost,
                capabilities=agent["capabilities"],
                estimated_duration_sec=task.get("estimated_duration_sec", 10.0),
            ))

        return bids

    def allocate_task(self, task: dict[str, Any]) -> TaskAllocation:
        """Allocate a task to the best bidder using auction."""
        task_id = task.get("id", f"task_{len(self._tasks)}")
        subtasks = self.decompose_task(task)

        allocation = TaskAllocation(task_id=task_id)

        for subtask in subtasks:
            bids = self.request_bids(subtask)
            if not bids:
                allocation.feasible = False
                allocation.reason = f"No capable agent for subtask {subtask.get('id')}"
                return allocation

            # Winner = lowest cost
            winner = min(bids, key=lambda b: b.cost)
            agent = self._agents[winner.agent_id]
            agent["status"] = "busy"
            agent["current_task"] = subtask.get("id")

            allocation.assignments.append({
                "subtask_id": subtask.get("id"),
                "agent_id": winner.agent_id,
                "cost": winner.cost,
                "estimated_duration_sec": winner.estimated_duration_sec,
            })

            if self.event_bus:
                self.event_bus.publish(Event(
                    topic="swarm.task_allocated",
                    payload={
                        "task_id": task_id,
                        "subtask_id": subtask.get("id"),
                        "agent_id": winner.agent_id,
                        "cost": winner.cost,
                    },
                    source="coordinator",
                    priority=EventPriority.HIGH,
                ))

        self._tasks[task_id] = {
            "id": task_id,
            "assignments": allocation.assignments,
            "status": "allocated",
        }
        return allocation

    # ── Consensus ──

    def propose_state(self, agent_id: str, key: str, value: Any, timestamp: float) -> None:
        """Propose a shared state update from an agent."""
        if key not in self._consensus_state:
            self._consensus_state[key] = {}

        proposals = self._consensus_state[key].get("proposals", [])
        proposals.append({"agent_id": agent_id, "value": value, "timestamp": timestamp})
        self._consensus_state[key]["proposals"] = proposals

        # Simple majority consensus
        if len(proposals) >= len(self._agents) // 2 + 1:
            # Use most recent value
            latest = max(proposals, key=lambda p: p["timestamp"])
            self._consensus_state[key]["agreed_value"] = latest["value"]
            self._consensus_state[key]["agreed_by"] = latest["agent_id"]
            self._consensus_state[key]["agreed_at"] = timestamp

            if self.event_bus:
                self.event_bus.publish(Event(
                    topic="swarm.consensus_reached",
                    payload={"key": key, "value": latest["value"]},
                    source="coordinator",
                    priority=EventPriority.HIGH,
                ))

    def get_consensus(self, key: str) -> Optional[Any]:
        """Get agreed consensus value for a key."""
        return self._consensus_state.get(key, {}).get("agreed_value")

    # ── Status ──

    def get_swarm_status(self) -> dict[str, Any]:
        """Return overall swarm status."""
        return {
            "agent_count": len(self._agents),
            "active_tasks": len(self._tasks),
            "agents": [
                {"id": a["id"], "status": a["status"], "capabilities": a["capabilities"]}
                for a in self._agents.values()
            ],
            "tasks": [
                {"id": t["id"], "status": t["status"], "assignments": len(t["assignments"])}
                for t in self._tasks.values()
            ],
            "consensus_keys": list(self._consensus_state.keys()),
        }
