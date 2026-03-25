"""Assembly manifest definitions for multi-agent coordination.

Assemblies represent coordinated groups of robots working together
as a single embodied intelligence unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from rosclaw_core.definitions.robot import RobotManifest


class AgentRole(Enum):
    """Role of a robot within an assembly."""
    LEADER = auto()      # Master/supervisory control
    FOLLOWER = auto()    # Executes commands
    OBSERVER = auto()    # Sensing/monitoring only
    PEER = auto()        # Equal coordination


class CoordinationMode(Enum):
    """Coordination mode for multi-agent assemblies."""
    HIERARCHICAL = auto()    # Leader-follower chain
    DISTRIBUTED = auto()     # Peer-to-peer coordination
    CENTRALIZED = auto()     # Central coordinator


@dataclass
class AgentBinding:
    """Binding of a robot to an assembly with a specific role.

    Attributes:
        robot_id: Unique identifier for this agent instance
        manifest: Robot hardware specification
        role: Role within the assembly
        ros_namespace: ROS namespace for this agent
        transform: 6DOF pose relative to assembly origin [x,y,z,qx,qy,qz,qw]
        parameters: Role-specific configuration
    """

    robot_id: str
    manifest: RobotManifest
    role: AgentRole = AgentRole.FOLLOWER
    ros_namespace: str = "/agent"
    transform: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynchronizationConfig:
    """Configuration for multi-agent synchronization."""

    # Time synchronization
    use_ntp: bool = True
    time_tolerance_ms: float = 10.0

    # State synchronization
    sync_frequency_hz: float = 100.0
    sync_topics: list[str] = field(default_factory=list)

    # Coordination
    coordination_mode: CoordinationMode = CoordinationMode.HIERARCHICAL
    leader_id: str | None = None


@dataclass
class AssemblyManifest:
    """Multi-agent assembly specification.

    An assembly represents a coordinated group of robots working
    together as a unified embodied intelligence system.

    Attributes:
        name: Unique assembly identifier
        version: Manifest version
        agents: List of robot bindings
        workspace_bounds: 3D workspace boundaries [[min], [max]]
        sync_config: Multi-agent synchronization settings
        shared_topics: Topics shared across all agents
        collision_pairs: Robot pairs that must avoid collision
    """

    name: str
    version: str = "1.0.0"
    agents: list[AgentBinding] = field(default_factory=list)

    # Workspace configuration
    workspace_bounds: list[list[float]] = field(
        default_factory=lambda: [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]]
    )

    # Synchronization
    sync_config: SynchronizationConfig = field(default_factory=SynchronizationConfig)

    # Shared resources
    shared_topics: dict[str, str] = field(default_factory=dict)
    collision_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def agent_count(self) -> int:
        """Total number of agents in assembly."""
        return len(self.agents)

    @property
    def leader_agents(self) -> list[AgentBinding]:
        """Get all leader agents."""
        return [a for a in self.agents if a.role == AgentRole.LEADER]

    @property
    def follower_agents(self) -> list[AgentBinding]:
        """Get all follower agents."""
        return [a for a in self.agents if a.role == AgentRole.FOLLOWER]

    def get_agent(self, robot_id: str) -> AgentBinding | None:
        """Get agent binding by robot ID."""
        for agent in self.agents:
            if agent.robot_id == robot_id:
                return agent
        return None

    def validate_assembly(self) -> list[str]:
        """Validate assembly configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for unique robot IDs
        ids = [a.robot_id for a in self.agents]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate robot IDs in assembly")

        # Check for at least one leader in hierarchical mode
        if self.sync_config.coordination_mode == CoordinationMode.HIERARCHICAL:
            if not self.leader_agents:
                errors.append("Hierarchical mode requires at least one leader")
            if self.sync_config.leader_id:
                if not self.get_agent(self.sync_config.leader_id):
                    errors.append(f"Specified leader '{self.sync_config.leader_id}' not found")

        # Check collision pairs exist
        for pair in self.collision_pairs:
            if len(pair) != 2:
                errors.append(f"Invalid collision pair: {pair}")
                continue
            if not self.get_agent(pair[0]):
                errors.append(f"Collision pair references unknown agent: {pair[0]}")
            if not self.get_agent(pair[1]):
                errors.append(f"Collision pair references unknown agent: {pair[1]}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "agents": [
                {
                    "robot_id": a.robot_id,
                    "manifest": a.manifest.to_dict(),
                    "role": a.role.name,
                    "ros_namespace": a.ros_namespace,
                    "transform": a.transform,
                    "parameters": a.parameters,
                }
                for a in self.agents
            ],
            "workspace_bounds": self.workspace_bounds,
            "sync_config": {
                "use_ntp": self.sync_config.use_ntp,
                "time_tolerance_ms": self.sync_config.time_tolerance_ms,
                "sync_frequency_hz": self.sync_config.sync_frequency_hz,
                "sync_topics": self.sync_config.sync_topics,
                "coordination_mode": self.sync_config.coordination_mode.name,
                "leader_id": self.sync_config.leader_id,
            },
            "shared_topics": self.shared_topics,
            "collision_pairs": list(self.collision_pairs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyManifest:
        """Create manifest from dictionary."""
        agents = [
            AgentBinding(
                robot_id=a["robot_id"],
                manifest=RobotManifest.from_dict(a["manifest"]),
                role=AgentRole[a["role"]],
                ros_namespace=a.get("ros_namespace", "/agent"),
                transform=a.get("transform", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                parameters=a.get("parameters", {}),
            )
            for a in data.get("agents", [])
        ]

        sync_data = data.get("sync_config", {})
        sync_config = SynchronizationConfig(
            use_ntp=sync_data.get("use_ntp", True),
            time_tolerance_ms=sync_data.get("time_tolerance_ms", 10.0),
            sync_frequency_hz=sync_data.get("sync_frequency_hz", 100.0),
            sync_topics=sync_data.get("sync_topics", []),
            coordination_mode=CoordinationMode[sync_data.get("coordination_mode", "HIERARCHICAL")],
            leader_id=sync_data.get("leader_id"),
        )

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            agents=agents,
            workspace_bounds=data.get("workspace_bounds", [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]]),
            sync_config=sync_config,
            shared_topics=data.get("shared_topics", {}),
            collision_pairs=[tuple(p) for p in data.get("collision_pairs", [])],
        )
