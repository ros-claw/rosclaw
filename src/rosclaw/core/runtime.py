"""ROSClaw Runtime - Central Orchestrator

The Runtime is the OS kernel of ROSClaw. It:
1. Owns the EventBus (all communication flows through it)
2. Manages all grounding engines (firewall, memory, practice, swarm, skill)
3. Provides the AgentRuntime with a unified interface to the physical world
4. Handles lifecycle coordination across all modules

Architecture:
    Agent Runtime (LLM/MCP)
         |
         v
    ROSClaw Runtime (this file)
    |-- EventBus
    |-- FirewallValidator (Action Grounding)
    |-- MemoryInterface (Experience Grounding)
    |-- UnifiedTimeline (Timeline Grounding)
    |-- SwarmRuntimeManager (Collaboration Grounding)
    |-- SkillManager (Skill Grounding)
    |-- EURDFParser (Physical Grounding)
    |-- MCPDrivers (Hardware abstraction)
         |
         v
    Physical World (Robot)
"""

import contextlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.core.runtime")

# Provider layer imports (lazy to avoid hard deps)
ProviderRegistry = None
CapabilityRouter = None
ProviderRequest = None
GuardPipeline = None
SchemaGuard = None
ActionGuard = None

try:
    from rosclaw.provider.core.registry import ProviderRegistry
    from rosclaw.provider.core.router import CapabilityRouter
    from rosclaw.provider.guard.action_guard import ActionGuard
    from rosclaw.provider.guard.pipeline import GuardPipeline
    from rosclaw.provider.guard.schema_guard import SchemaGuard
except ImportError:
    pass


@dataclass
class RuntimeConfig:
    """Configuration for ROSClaw Runtime."""

    robot_id: str = "rosclaw_default"
    robot_model_path: str | None = None
    robot_zoo_path: str | None = None
    default_eurdf_robot: str = "ur5e"
    enable_firewall: bool = True
    enable_memory: bool = True
    enable_practice: bool = True
    enable_swarm: bool = False
    enable_skill_manager: bool = True
    enable_knowledge: bool = True  # KnowledgeInterface (KNOW module)
    enable_how: bool = True  # HeuristicEngine (HOW module)
    enable_auto: bool = True  # Self-Evolution Control Plane (AUTO module)
    enable_provider: bool = True
    joint_dof: int = 6
    sampling_rate_hz: int = 1000
    safety_level: str = "MODERATE"  # STRICT | MODERATE | LENIENT
    timeline_output_dir: str = "./practice_data"
    enable_mcap: bool = False
    seekdb_backend: str = "memory"  # "memory" | "sqlite"
    seekdb_path: str = "./seekdb.sqlite"
    # Optional SeekDB / rosclaw_practice integration for practice event persistence
    seekdb_url: str | None = field(default_factory=lambda: os.environ.get("ROSCLAW_SEEKDB_URL"))
    seekdb_fallback_dir: str = field(
        default_factory=lambda: os.environ.get(
            "ROSCLAW_SEEKDB_FALLBACK_DIR", "/data/rosclaw/fallback"
        )
    )
    embodied_memory: Any | None = None  # powermem.EmbodiedMemory instance
    providers_dir: str | None = None  # Directory to scan for provider.yaml files
    # GPU model microservice endpoints (auto-registered as HTTP providers)
    gpu_sam3_endpoint: str | None = None  # e.g. "http://localhost:8001"
    gpu_vggt_endpoint: str | None = None  # e.g. "http://localhost:8002"
    gpu_minicpm_endpoint: str | None = None  # e.g. "http://localhost:8003"
    gpu_cosmos_endpoint: str | None = None  # e.g. "http://localhost:8004"
    event_bus: Any | None = None  # External EventBus instance
    # HOW service client (optional). When set, Runtime talks to a local/private
    # rosclaw-how HTTP service instead of the built-in HeuristicEngine.
    how_url: str | None = field(default_factory=lambda: os.environ.get("ROSCLAW_HOW_URL"))
    how_api_key: str | None = field(default_factory=lambda: os.environ.get("ROSCLAW_HOW_API_KEY"))
    how_timeout: float = 10.0
    how_fallback_to_local: bool = True
    # KNOW optional integration with private rosclaw_know curated registry.
    know_curated_registry_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("ROSCLAW_KNOW_CURATED_REGISTRY_ENABLED", "").lower()
            in ("1", "true", "yes", "on")
        )
    )
    # Sense / BodySense module configuration
    enable_sense: bool = True
    sense_collector: str = "mock"
    sense_update_hz: float = 1.0
    sense_robot_profile: str | None = None
    sense_thresholds_path: str | None = None
    sense_replay_path: str | None = None
    # Persist all EventBus events to ~/.rosclaw/events/live.jsonl for dashboard tail
    enable_event_persistence: bool = True


class Runtime(LifecycleMixin):
    """
    Central runtime orchestrator for ROSClaw.

    This is the main entry point for the ROSClaw OS. It coordinates
    all grounding engines and provides a unified interface for
    agent runtimes to interact with the physical world.
    """

    def __init__(self, config: RuntimeConfig | None = None, event_bus: EventBus | None = None):
        super().__init__()
        self.config = config or RuntimeConfig()

        # Core infrastructure: accept external EventBus or create one
        self.event_bus = event_bus or self.config.event_bus or EventBus()
        self._event_sink: Any | None = None

        # Grounding engines (initialized on demand)
        self._firewall: Any | None = None
        self._memory: Any | None = None
        self._practice: Any | None = None
        self._swarm: Any | None = None
        self._skill_manager: Any | None = None
        self._knowledge: Any | None = None
        self._knowledge_batch: Any | None = None
        self._knowledge_assets: Any | None = None
        self._how: Any | None = None
        self._auto: Any | None = None
        self._sense: Any | None = None
        self._e_urdf: Any | None = None
        self._robot_profile: Any | None = None
        self._sandbox: Any | None = None
        self._episode_recorder: Any | None = None
        self._mcp_drivers: dict[str, Any] = {}

        # Provider layer
        self._provider_registry: Any | None = None
        self._capability_router: Any | None = None
        self._guard_pipeline: Any | None = None

        # Internal state
        self._agent_runtime: Any | None = None
        self._modules: list[LifecycleMixin] = []

        # Module lifecycle lock
        import threading

        self._module_lock = threading.RLock()

    def _do_initialize(self) -> None:
        """Initialize all enabled grounding engines."""
        logger.info(f"Initializing ROSClaw Runtime for {self.config.robot_id}")

        # Shared SeekDB client reused by Memory, Knowledge, HOW, Auto, and SkillManager.
        seekdb: Any | None = None

        # Initialize EventBus subscriptions for internal coordination
        self._setup_internal_subscriptions()

        # Attach persistent JSONL sink so dashboards can tail live events
        if self.config.enable_event_persistence:
            from rosclaw.core.event_sink import JsonlEventSink

            self._event_sink = JsonlEventSink()
            self._event_sink.attach(self.event_bus)

        # Initialize Physical Grounding (e-URDF)
        self._load_e_urdf()

        # Initialize Action Grounding (FirewallValidator)
        if self.config.enable_firewall and self._e_urdf is not None:
            try:
                from rosclaw.firewall.validator import FirewallValidator

                self._firewall = FirewallValidator(
                    robot_model=self._e_urdf.get_model(),
                    event_bus=self.event_bus,
                    mujoco_model_path=self.config.robot_model_path,
                    safety_level=self.config.safety_level,
                )
                self._modules.append(self._firewall)
                logger.info("Action Grounding (FirewallValidator) initialized")
            except ImportError as e:
                logger.info(f"FirewallValidator not available: {e}")

        # Initialize Sandbox (Digital Twin + Physics Simulation)
        try:
            from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

            # Map short robot_id to canonical full name via e-URDF Zoo
            canonical_robot_id = self.config.robot_id
            try:
                from rosclaw.runtime.eurdf_loader import RobotRegistry

                reg = RobotRegistry()
                profile = reg.get(canonical_robot_id)
                if profile is not None:
                    canonical_robot_id = profile.robot_id
            except Exception:
                pass
            sandbox_config = {
                "engine": "mujoco",
                "world_id": "empty",
                "robot_id": canonical_robot_id,
            }
            self._sandbox = SandboxRuntimeAdapter(
                config=sandbox_config,
                event_bus=self.event_bus,
                e_urdf_model=self._e_urdf,
                runtime=self,
            )
            self._modules.append(self._sandbox)
            logger.info("Sandbox (Digital Twin + Physics) initialized")
        except ImportError as e:
            logger.info(f"Sandbox not available: {e}")

        # Initialize Body Sense (after e-URDF/Sandbox, before SkillManager)
        if self.config.enable_sense:
            try:
                from rosclaw.sense.config import SenseConfig
                from rosclaw.sense.runtime import SenseRuntime

                self._sense = SenseRuntime(
                    config=SenseConfig(
                        robot_id=self.config.robot_id,
                        collector=self.config.sense_collector,
                        robot_profile=self.config.sense_robot_profile,
                        thresholds_path=self.config.sense_thresholds_path,
                        replay_path=self.config.sense_replay_path,
                        update_hz=self.config.sense_update_hz,
                    ),
                    event_bus=self.event_bus,
                )
                self._modules.append(self._sense)
                logger.info("Body Sense (SenseRuntime) initialized")
            except ImportError as e:
                logger.info(f"Sense module not available: {e}")

        # Initialize Experience Grounding (Memory)
        if self.config.enable_memory:
            try:
                from rosclaw.memory.interface import MemoryInterface
                from rosclaw.memory.seekdb_client import SeekDBMemoryClient, SeekDBSQLiteClient

                if self.config.seekdb_backend == "sqlite":
                    seekdb = SeekDBSQLiteClient(self.config.seekdb_path)
                else:
                    seekdb = SeekDBMemoryClient()
                self._memory = MemoryInterface(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                    seekdb_client=seekdb,
                    embodied_memory=self.config.embodied_memory,
                )
                self._modules.append(self._memory)
                if self._sense is not None:
                    self._memory.set_sense_runtime(self._sense)
                em_label = "+EmbodiedMemory" if self.config.embodied_memory else "SeekDB-only"
                logger.info(f"Experience Grounding (Memory) initialized [{em_label}]")
            except ImportError as e:
                logger.info(f"Memory module not available: {e}")

        # Initialize Timeline Grounding (UnifiedTimeline)
        if self.config.enable_practice:
            try:
                from rosclaw.practice.timeline import UnifiedTimeline

                self._practice = UnifiedTimeline(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                    output_dir=self.config.timeline_output_dir,
                    enable_mcap=self.config.enable_mcap,
                )
                self._modules.append(self._practice)
                logger.info("Timeline Grounding (UnifiedTimeline) initialized")
            except ImportError as e:
                logger.info(f"UnifiedTimeline not available: {e}")

            # Optional SeekDB bridge for practice event persistence
            seekdb_bridge = None
            if self.config.seekdb_url:
                try:
                    from rosclaw.practice.seekdb_bridge import SeekDBBridge

                    seekdb_bridge = SeekDBBridge(
                        seekdb_url=self.config.seekdb_url,
                        fallback_dir=self.config.seekdb_fallback_dir,
                    )
                    logger.info("SeekDBBridge initialized")
                except ImportError as e:
                    logger.info(f"rosclaw_practice not installed, SeekDB integration disabled: {e}")
                except Exception as e:
                    logger.warning(f"SeekDBBridge initialization failed: {e}")

            # Initialize EpisodeRecorder for artifact management
            try:
                from rosclaw.practice.episode_recorder import EpisodeRecorder

                self._episode_recorder = EpisodeRecorder(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                    seekdb_bridge=seekdb_bridge,
                    sense_runtime=self._sense,
                )
                self._modules.append(self._episode_recorder)
                logger.info("EpisodeRecorder initialized")
            except ImportError as e:
                logger.info(f"EpisodeRecorder not available: {e}")

            # Initialize Critic for automatic success detection
            try:
                from rosclaw.critic.basic_critic import BasicCritic

                self._critic = BasicCritic(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                )
                self._modules.append(self._critic)
                logger.info("Critic (BasicCritic) initialized")
            except ImportError as e:
                logger.info(f"Critic not available: {e}")

        # Initialize Collaboration Grounding (Swarm)
        if self.config.enable_swarm:
            try:
                from rosclaw.swarm.manager import SwarmRuntimeManager

                self._swarm = SwarmRuntimeManager(event_bus=self.event_bus)
                self._modules.append(self._swarm)
                logger.info("Collaboration Grounding (Swarm) initialized")
            except ImportError as e:
                logger.info(f"Swarm module not available: {e}")

        # Initialize Skill Grounding (SkillManager)
        if self.config.enable_skill_manager:
            try:
                from rosclaw.skill_manager.executor import SkillExecutor
                from rosclaw.skill_manager.registry import SkillRegistry

                registry = SkillRegistry(event_bus=self.event_bus)
                self._skill_manager = SkillExecutor(
                    self.event_bus,
                    registry,
                    seekdb_client=seekdb,
                    sense_interface=self._sense,
                )
                self._modules.append(registry)
                self._modules.append(self._skill_manager)
                logger.info("Skill Grounding (SkillManager) initialized")
            except ImportError as e:
                logger.info(f"SkillManager module not available: {e}")

        # Initialize Knowledge Grounding (KnowledgeInterface) - MUST come before HOW
        if self.config.enable_knowledge:
            try:
                from rosclaw.know.interface import KnowledgeInterface
                from rosclaw.know.storage import seed_knowledge_graph

                # Reuse Memory's SeekDB client if available
                if self._memory is not None:
                    seekdb = getattr(self._memory, "seekdb_client", None)
                self._knowledge = KnowledgeInterface(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                    seekdb_client=seekdb,
                    use_rosclaw_know_registry=self.config.know_curated_registry_enabled,
                )
                self._modules.append(self._knowledge)
                logger.info("Knowledge Grounding (KnowledgeInterface) initialized")
                # Seed knowledge_graph with baseline data
                if seekdb is not None:
                    seed_knowledge_graph(seekdb)

                # Spawn KnowledgeBatchEngine + AssetsLoader.  Both
                # are inert when rosclaw-know is not installed; failures
                # here MUST NOT kill the runtime — the query side keeps
                # working with its hard-coded fallback patterns.
                try:
                    from rosclaw.know.batch_engine import KnowledgeBatchEngine

                    self._knowledge_batch = KnowledgeBatchEngine(
                        runtime=self,
                        assets_path="data/knowledge_assets",
                    )
                    self._modules.append(self._knowledge_batch)
                    logger.info(
                        "Knowledge Batch Engine initialized (active=%s)",
                        self._knowledge_batch.is_active,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.info(f"KnowledgeBatchEngine not initialized: {e}")

                try:
                    from rosclaw.know.assets_loader import AssetsLoader

                    self._knowledge_assets = AssetsLoader(
                        runtime=self,
                        assets_path="data/knowledge_assets",
                    )
                    self._modules.append(self._knowledge_assets)
                    logger.info("Knowledge AssetsLoader initialized")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"AssetsLoader not initialized: {e}")
            except ImportError as e:
                logger.info(f"Knowledge module not available: {e}")

        # Initialize Heuristic Grounding (HeuristicEngine / HowClient) - depends on KNOW
        if self.config.enable_how:
            try:
                # Reuse Memory's SeekDB client if available
                if self._memory is not None:
                    seekdb = getattr(self._memory, "seekdb_client", None)
                self._how = self._create_how_engine(seekdb)
                if self._how is not None:
                    self._modules.append(self._how)
            except Exception as e:  # noqa: BLE001
                logger.info(f"Heuristic grounding not available: {e}")

        # Initialize Auto Self-Evolution Control Plane
        if self.config.enable_auto:
            try:
                from rosclaw.auto.plugin import AutoPlugin

                # Reuse Memory's SeekDB client if available
                if self._memory is not None:
                    seekdb = getattr(self._memory, "seekdb_client", None)
                self._auto = AutoPlugin(
                    config={},
                    event_bus=self.event_bus,
                    seekdb_client=seekdb,
                    skill_registry=getattr(self._skill_manager, "registry", None)
                    if self._skill_manager
                    else None,
                    sense_runtime=self._sense,
                )
                self._modules.append(self._auto)
                logger.info("Self-Evolution Control Plane (Auto) initialized")
            except ImportError as e:
                logger.info(f"Auto module not available: {e}")

        # Initialize Provider Layer (Capability Router + Guard)
        if self.config.enable_provider and ProviderRegistry is not None:
            try:
                self._provider_registry = ProviderRegistry(event_bus=self.event_bus)
                self._guard_pipeline = GuardPipeline()
                self._guard_pipeline.add(SchemaGuard())
                self._guard_pipeline.add(ActionGuard())
                # Create router early so it's always available even if
                # some provider registrations fail downstream.
                self._capability_router = CapabilityRouter(self._provider_registry)
                # Register providers — non-fatal if individual steps fail
                try:
                    self._register_builtin_providers()
                except Exception as e:
                    logger.info(f"Built-in provider registration warning: {e}")
                try:
                    self._register_robot_capabilities()
                except Exception as e:
                    logger.info(f"Robot capability registration warning: {e}")
                try:
                    self._load_external_providers()
                except Exception as e:
                    logger.info(f"External provider loading warning: {e}")
                logger.info("Provider Layer (Registry + Router + Guard) initialized")
            except Exception as e:
                logger.info(f"Provider layer not available: {e}")

        # Initialize all module lifecycles
        for module in self._modules:
            if isinstance(module, LifecycleMixin):
                module.initialize()

        logger.info("Initialization complete")

    def _create_how_engine(self, seekdb: Any | None) -> Any | None:
        """Return a HOW engine: HowClient when configured, else HeuristicEngine."""
        if self.config.how_url:
            try:
                from rosclaw.how.client import HowClient

                client = HowClient(
                    self.config.how_url,
                    api_key=self.config.how_api_key,
                    timeout=self.config.how_timeout,
                )
                self._run_async(client.initialize())
                logger.info(
                    "Heuristic Grounding (HowClient) initialized at %s", self.config.how_url
                )
                return client
            except Exception as exc:  # noqa: BLE001
                logger.warning("HowClient initialization failed: %s", exc)
                if not self.config.how_fallback_to_local:
                    return None
                logger.info("Falling back to local HeuristicEngine")

        if seekdb is None:
            logger.info("HeuristicEngine skipped: no SeekDB client (memory not enabled)")
            return None

        from rosclaw.how.engine import HeuristicEngine

        engine = HeuristicEngine(
            seekdb_client=seekdb,
            knowledge_interface=self._knowledge,
            event_bus=self.event_bus,
            sense_runtime=self._sense,
        )
        self._run_async(engine.initialize())
        self._run_async(engine.seed_defaults())
        logger.info("Heuristic Grounding (HeuristicEngine) initialized")
        return engine

    def _do_start(self) -> None:
        """Start all modules."""
        logger.info("Starting all modules...")
        with self._module_lock:
            for module in self._modules:
                if isinstance(module, LifecycleMixin) and module.is_ready:
                    module.start()
        self.event_bus.publish(
            Event(
                topic="runtime.status",
                payload={"state": "running", "robot_id": self.config.robot_id},
                source="runtime",
                priority=EventPriority.HIGH,
            )
        )
        logger.info("All modules started")

    def _do_stop(self) -> None:
        """Stop all modules gracefully."""
        logger.info("Shutting down...")
        if self._event_sink is not None:
            self._event_sink.close()
            self._event_sink = None
        self.event_bus.publish(
            Event(
                topic="runtime.status",
                payload={"state": "shutting_down", "robot_id": self.config.robot_id},
                source="runtime",
                priority=EventPriority.CRITICAL,
            )
        )
        # Stop in reverse order
        with self._module_lock:
            for module in reversed(self._modules):
                if isinstance(module, LifecycleMixin):
                    module.stop()
        logger.info("Shutdown complete")

    def _setup_internal_subscriptions(self) -> None:
        """Set up internal EventBus subscriptions for runtime coordination."""
        self.event_bus.subscribe("safety.violation", self._on_safety_violation)
        self.event_bus.subscribe("agent.command", self._on_agent_command)
        self.event_bus.subscribe("robot.emergency_stop", self._on_emergency_stop)
        self.event_bus.subscribe("firewall.action_blocked", self._on_firewall_action_blocked)
        self.event_bus.subscribe("rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed)
        self.event_bus.subscribe("rosclaw.sandbox.action.blocked", self._on_sandbox_action_blocked)
        self.event_bus.subscribe(
            "rosclaw.runtime.execution.failed", self._on_runtime_execution_failed
        )
        # Auto-sync critic judgments to Memory for closed-loop experience retention
        self.event_bus.subscribe("rosclaw.critic.judgment", self._on_critic_judgment)
        # Route capability requests from MCPHub through Provider layer via EventBus
        self.event_bus.subscribe("agent.capability.request", self._on_capability_request)

    def _on_safety_violation(self, event: Event) -> None:
        """Handle safety violation events."""
        logger.info(f"SAFETY VIOLATION: {event.payload}")
        self.event_bus.publish(
            Event(
                topic="robot.emergency_stop",
                payload={"reason": event.payload},
                source="runtime",
                priority=EventPriority.CRITICAL,
            )
        )

    def _on_firewall_action_blocked(self, event: Event) -> None:
        """Handle firewall action blocked: query heuristic recovery.

        When Firewall blocks an agent command, this handler queries the
        HeuristicEngine for a recovery suggestion and publishes it on
        the EventBus so the agent can attempt recovery before escalating.
        """
        if self._how is None:
            return
        try:
            request_id = event.payload.get("request_id", "")
            violations = event.payload.get("violations", [])
            error_log = "; ".join(v.get("description", "") for v in violations)
            recovery = self._run_async(
                self._how.suggest_recovery(error_log, context={"request_id": request_id})
            )
            if recovery:
                from rosclaw.how.recovery import RecoveryFormatter

                payload = RecoveryFormatter.to_event_payload(
                    recovery, request_id=request_id, source="heuristic_engine"
                )
                self.event_bus.publish(
                    Event(
                        topic="heuristic.recovery_suggested",
                        payload=payload,
                        source="runtime",
                        priority=EventPriority.HIGH,
                    )
                )
                logger.info(f"Heuristic recovery suggested for {request_id}: {recovery['action']}")
        except Exception as e:
            logger.info(f"Heuristic recovery failed: {e}")

    def _run_async(self, coro):
        """Run an async coroutine from a sync context.

        Delegates to ``rosclaw.core.async_utils.run_sync`` so we never call
        ``asyncio.run`` from inside an already-running event loop.
        """
        from rosclaw.core.async_utils import run_sync

        return run_sync(coro)

    def _on_sandbox_episode_failed(self, event: Event) -> None:
        """Handle sandbox episode failure: generate recovery hint."""
        if self._how is None:
            return
        try:
            from rosclaw.how.recovery import RecoveryEngine

            failure_type = event.payload.get("failure_type", "")
            request_id = event.payload.get("request_id", "")
            re = RecoveryEngine(self._how)
            hint = self._run_async(
                re.generate_recovery_hint(
                    failure_type,
                    context={"request_id": request_id, "source": "sandbox"},
                    sources=["sandbox_episode"],
                    request_id=request_id,
                    previous_scores=event.payload.get("previous_scores"),
                    current_iteration=event.payload.get("current_iteration"),
                )
            )
            if hint:
                payload = re.format_for_eventbus(hint, request_id=request_id)
                self.event_bus.publish(
                    Event(
                        topic="rosclaw.how.recovery_hint.generated",
                        payload=payload,
                        source="runtime",
                        priority=EventPriority.HIGH,
                    )
                )
                logger.info(
                    f"RecoveryHint generated for sandbox failure {request_id}: {hint['hint']}"
                )
        except Exception as e:
            logger.info(f"RecoveryHint generation failed: {e}")

    def _on_sandbox_action_blocked(self, event: Event) -> None:
        """Handle sandbox action blocked: generate recovery hint."""
        if self._how is None:
            return
        try:
            from rosclaw.how.recovery import RecoveryEngine

            failure_type = event.payload.get("reason", "")
            request_id = event.payload.get("request_id", "")
            re = RecoveryEngine(self._how)
            hint = self._run_async(
                re.generate_recovery_hint(
                    failure_type,
                    context={"request_id": request_id, "source": "sandbox"},
                    sources=["sandbox_action"],
                    request_id=request_id,
                    previous_scores=event.payload.get("previous_scores"),
                    current_iteration=event.payload.get("current_iteration"),
                )
            )
            if hint:
                payload = re.format_for_eventbus(hint, request_id=request_id)
                self.event_bus.publish(
                    Event(
                        topic="rosclaw.how.recovery_hint.generated",
                        payload=payload,
                        source="runtime",
                        priority=EventPriority.HIGH,
                    )
                )
                logger.info(
                    f"RecoveryHint generated for blocked action {request_id}: {hint['hint']}"
                )
        except Exception as e:
            logger.info(f"RecoveryHint generation failed: {e}")

    def _on_runtime_execution_failed(self, event: Event) -> None:
        """Handle runtime execution failure: generate recovery hint."""
        if self._how is None:
            return
        try:
            from rosclaw.how.recovery import RecoveryEngine

            failure_type = event.payload.get("error_type", "")
            request_id = event.payload.get("request_id", "")
            re = RecoveryEngine(self._how)
            hint = self._run_async(
                re.generate_recovery_hint(
                    failure_type,
                    context={"request_id": request_id, "source": "runtime"},
                    sources=["runtime_execution"],
                    request_id=request_id,
                    previous_scores=event.payload.get("previous_scores"),
                    current_iteration=event.payload.get("current_iteration"),
                )
            )
            if hint:
                payload = re.format_for_eventbus(hint, request_id=request_id)
                self.event_bus.publish(
                    Event(
                        topic="rosclaw.how.recovery_hint.generated",
                        payload=payload,
                        source="runtime",
                        priority=EventPriority.HIGH,
                    )
                )
                logger.info(
                    f"RecoveryHint generated for execution failure {request_id}: {hint['hint']}"
                )
        except Exception as e:
            logger.info(f"RecoveryHint generation failed: {e}")

    def _on_critic_judgment(self, event: Event) -> None:
        """Auto-sync critic judgment to Memory for experience retention.

        This closes the loop: Critic evaluates → Memory remembers →
        Future queries can recall "what happened" for similar tasks.
        """
        if self._memory is None:
            return
        try:
            payload = event.payload if isinstance(event.payload, dict) else {}
            episode_id = payload.get("episode_id", "unknown")
            status = payload.get("status", "UNKNOWN")
            reward = payload.get("reward", 0.0)
            reason = payload.get("reason", "")
            context = payload.get("context", {})

            # Extract skill_name from outcome inside context (BasicCritic passes full event payload as context)
            outcome = context.get("outcome", {}) if isinstance(context, dict) else {}
            skill_name = outcome.get("skill_name", "unknown")
            instruction = context.get("instruction", "") if isinstance(context, dict) else ""

            # Build tags for semantic search
            tags = [skill_name, self.config.robot_id, status.lower()]
            if reason:
                tags.append(reason.replace(" ", "_"))

            self._memory.store_experience(
                event_id=episode_id,
                event_type="praxis",
                instruction=instruction or f"{skill_name} task",
                outcome=status.lower(),
                duration_sec=0.0,
                error_details=reason if status != "SUCCESS" else None,
                tags=tags,
                metadata={
                    "critic_reward": reward,
                    "critic_status": status,
                    "skill_name": skill_name,
                    "auto_synced": True,
                    "source": "critic_judgment",
                },
            )
            logger.info(
                f"Critic judgment auto-synced to Memory: {episode_id} ({status}, r={reward})"
            )
        except Exception as e:
            logger.info(f"Critic judgment Memory sync failed (non-fatal): {e}")

    def _on_agent_command(self, event: Event) -> None:
        """Handle agent commands - route to appropriate module."""
        command = event.payload.get("action", "")
        logger.info(f"Agent command received: {command}")

    def _on_emergency_stop(self, event: Event) -> None:
        """Handle emergency stop - stop all drivers."""
        logger.info("EMERGENCY STOP triggered")
        for driver in self._mcp_drivers.values():
            if hasattr(driver, "emergency_stop"):
                driver.emergency_stop()

    def _on_capability_request(self, event: Event) -> None:
        """Handle capability requests from MCPHub via EventBus.

        Routes through CapabilityRouter and publishes response back
        to EventBus.  This removes direct MCPHub -> Runtime coupling.
        """
        if self._capability_router is None:
            return
        payload = event.payload or {}
        request_id = payload.get("request_id", "")
        capability = payload.get("capability", "")
        inputs = payload.get("inputs", {})
        context = payload.get("context", {})

        try:
            from rosclaw.provider.core.request import ProviderRequest

            req = ProviderRequest(
                request_id=request_id,
                capability=capability,
                inputs=inputs,
                context=context,
                constraints=payload.get("constraints", {}),
            )
            result = self._run_async(self._capability_router.invoke(req))
            self.event_bus.publish(
                Event(
                    topic="agent.capability.response",
                    payload={
                        "request_id": request_id,
                        "result": {
                            "status": "ok" if result.is_ok else "failed",
                            "capability": capability,
                            "provider": getattr(result, "provider", ""),
                            "result": getattr(result, "result", {}),
                        },
                    },
                    source="runtime",
                )
            )
        except Exception as e:
            self.event_bus.publish(
                Event(
                    topic="agent.capability.response",
                    payload={
                        "request_id": request_id,
                        "result": {"status": "error", "error": str(e)},
                    },
                    source="runtime",
                )
            )

    def register_driver(self, name: str, driver: Any) -> None:
        """Register an MCP driver with the runtime."""
        self._mcp_drivers[name] = driver
        logger.info(f"Driver registered: {name}")

    def get_driver(self, name: str) -> Any | None:
        """Get a registered driver by name."""
        return self._mcp_drivers.get(name)

    @property
    def firewall(self) -> Any | None:
        return self._firewall

    @property
    def memory(self) -> Any | None:
        if self._memory is None:
            return None
        return _MemoryProxy(self._memory, event_bus=self.event_bus)

    @property
    def practice(self) -> Any | None:
        return self._practice

    @property
    def swarm(self) -> Any | None:
        return self._swarm

    @property
    def skill_manager(self) -> Any | None:
        return self._skill_manager

    @property
    def knowledge(self) -> Any | None:
        return self._knowledge

    @property
    def how(self) -> Any | None:
        if self._how is None:
            return None
        return _HowProxy(self._how, self._run_async, event_bus=self.event_bus)

    @property
    def auto(self) -> Any | None:
        return self._auto

    @property
    def sense(self) -> Any | None:
        return self._sense

    @property
    def e_urdf(self) -> Any | None:
        return self._e_urdf

    @property
    def status(self) -> dict:
        """Get comprehensive runtime status (alias for get_status)."""
        return self.get_status()

    @property
    def provider_registry(self) -> Any | None:
        return self._provider_registry

    @property
    def capability_router(self) -> Any | None:
        return self._capability_router

    @property
    def guard_pipeline(self) -> Any | None:
        return self._guard_pipeline

    @property
    def sandbox(self) -> Any | None:
        """Return the physics sandbox / digital-twin adapter."""
        return self._sandbox

    @property
    def episode_recorder(self) -> Any | None:
        """Return the practice episode recorder."""
        return self._episode_recorder

    def _load_external_providers(self) -> None:
        """Scan providers_dir for provider.yaml files and load them."""
        if not self.config.providers_dir:
            return
        try:
            from rosclaw.provider.loader import ProviderLoader

            loader = ProviderLoader(self._provider_registry)
            loaded = loader.scan_directory(self.config.providers_dir)
            if loaded:
                logger.info(f"Loaded external providers: {loaded}")
        except Exception as e:
            logger.info(f"Failed to load external providers: {e}")

    def _register_builtin_providers(self) -> None:
        """Register built-in mock providers for out-of-box capability support."""
        from rosclaw.provider.builtins import (
            DeepSeekProvider,
            MockCriticProvider,
            MockSkillProvider,
            MockVLMProvider,
        )
        from rosclaw.provider.core.manifest import ProviderManifest

        # Subscribe to provider lifecycle events before registering builtins
        self.event_bus.subscribe("provider_registered", self._on_provider_event)
        self.event_bus.subscribe("provider_unregistered", self._on_provider_event)
        self.event_bus.subscribe("provider_health_changed", self._on_provider_health_changed)

        self._provider_registry.register(
            ProviderManifest.from_dict(
                {
                    "name": "mock_vlm",
                    "version": "0.1.0",
                    "type": "vlm",
                    "capabilities": ["vlm.object_grounding", "vlm.scene_understanding"],
                    "modalities": {"input": ["image", "text"], "output": ["object_list"]},
                    "safety": {"executable": False, "requires_guard": False},
                }
            ),
            lambda m: MockVLMProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("mock_vlm", ok=True)

        self._provider_registry.register(
            ProviderManifest.from_dict(
                {
                    "name": "mock_skill",
                    "version": "0.1.0",
                    "type": "skill",
                    "capabilities": ["skill.grasp", "skill.place", "skill.pick_and_place"],
                    "embodiment": {"supported_robots": []},
                    "safety": {"executable": True, "requires_guard": True},
                }
            ),
            lambda m: MockSkillProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("mock_skill", ok=True)

        self._provider_registry.register(
            ProviderManifest.from_dict(
                {
                    "name": "mock_critic",
                    "version": "0.1.0",
                    "type": "critic",
                    "capabilities": ["critic.success_detection", "critic.retry_advice"],
                    "safety": {"executable": False, "requires_guard": False},
                }
            ),
            lambda m: MockCriticProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("mock_critic", ok=True)

        # Register DeepSeek LLM provider (requires DEEPSEEK_API_KEY)
        self._provider_registry.register(
            ProviderManifest.from_dict(
                {
                    "name": "deepseek",
                    "version": "1.0.0",
                    "type": "llm",
                    "capabilities": ["llm.task_planning", "llm.summary", "llm.chat"],
                    "safety": {"executable": False, "requires_guard": False},
                }
            ),
            lambda m: DeepSeekProvider(m),
            auto_load=False,
        )
        # Health check: verify API key AND ping endpoint
        import os

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        healthy = False
        if api_key:
            try:
                import urllib.request

                req = urllib.request.Request(
                    f"{os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    healthy = resp.status == 200
            except Exception:
                healthy = False
        self._provider_registry.set_provider_health("deepseek", ok=healthy)

        # ─── GPU Model Microservice Providers ───────────────────────────────
        self._register_gpu_providers()

    def _register_gpu_providers(self) -> None:
        """Register GPU model microservices as HTTP-backed GenericProviders.

        Each service is a FastAPI container exposing model-specific endpoints.
        Endpoint URLs come from RuntimeConfig or environment variable fallbacks.
        """
        import os

        from rosclaw.provider.adapters.generic import GenericProvider
        from rosclaw.provider.core.manifest import ProviderManifest

        gpu_configs = [
            {
                "name": "gpu_sam3",
                "endpoint": self.config.gpu_sam3_endpoint
                or os.getenv("SAM3_ENDPOINT", "http://localhost:8001"),
                "capabilities": ["segmentation.mask", "segmentation.track"],
                "type": "segmentation",
                "modalities": {"input": ["image"], "output": ["mask", "bbox"]},
            },
            {
                "name": "gpu_vggt",
                "endpoint": self.config.gpu_vggt_endpoint
                or os.getenv("VGGT_ENDPOINT", "http://localhost:8002"),
                "capabilities": ["geometry.depth", "geometry.pose", "geometry.point_cloud"],
                "type": "geometry",
                "modalities": {"input": ["image"], "output": ["depth", "pointcloud", "pose"]},
            },
            {
                "name": "gpu_minicpm",
                "endpoint": self.config.gpu_minicpm_endpoint
                or os.getenv("MINICPM_ENDPOINT", "http://localhost:8003"),
                "capabilities": ["vlm.vqa", "vlm.scene_understanding", "vlm.object_grounding"],
                "type": "vlm",
                "modalities": {"input": ["image", "text"], "output": ["text", "bbox"]},
            },
            {
                "name": "gpu_cosmos",
                "endpoint": self.config.gpu_cosmos_endpoint
                or os.getenv("COSMOS_ENDPOINT", "http://localhost:8004"),
                "capabilities": [
                    "reasoning.physical",
                    "reasoning.risk_explain",
                    "critic.risk",
                    "world.risk",
                ],
                "type": "reasoning",
                "modalities": {"input": ["image", "text"], "output": ["text", "risk_score"]},
            },
        ]

        for cfg in gpu_configs:
            endpoint = cfg["endpoint"]
            if not endpoint:
                continue
            try:
                manifest = ProviderManifest.from_dict(
                    {
                        "name": cfg["name"],
                        "version": "1.0.0",
                        "type": cfg["type"],
                        "capabilities": cfg["capabilities"],
                        "modalities": cfg["modalities"],
                        "runtime": {
                            "backend": "http",
                            "protocol": "http",
                            "endpoint": endpoint,
                            "device": "cuda",
                        },
                        "safety": {"executable": False, "requires_guard": True},
                    }
                )
                self._provider_registry.register(
                    manifest,
                    lambda m: GenericProvider(m),
                    auto_load=False,
                )
                # Health check: ping /health endpoint
                healthy = False
                try:
                    import urllib.request

                    req = urllib.request.Request(f"{endpoint}/health", method="GET")
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        healthy = resp.status == 200
                except Exception:
                    healthy = False
                self._provider_registry.set_provider_health(cfg["name"], ok=healthy)
                status = "healthy" if healthy else "unreachable"
                logger.info(f"GPU provider '{cfg['name']}' registered ({status}) @ {endpoint}")
            except Exception as e:
                logger.info(f"Failed to register GPU provider '{cfg['name']}': {e}")

    def _load_e_urdf(self) -> None:
        """Load robot e-URDF from file path or e-URDF Zoo."""
        if self.config.robot_model_path:
            from rosclaw.e_urdf.parser import EURDFParser

            self._e_urdf = EURDFParser(self.config.robot_model_path)
            logger.info(f"Physical Grounding (e-URDF) loaded: {self.config.robot_model_path}")
            return
        try:
            from rosclaw.runtime.eurdf_loader import EURDFLoader

            loader = EURDFLoader(self.config.robot_zoo_path)
            robot_id = self.config.default_eurdf_robot
            self._robot_profile = loader.load(robot_id)
            robot_dir = loader.zoo_path / robot_id
            # EURDFParser expects XML (URDF/MJCF), not YAML
            xml_path = robot_dir / "robot.urdf"
            mjcf_path = robot_dir / "robot.mjcf.xml"
            from rosclaw.e_urdf.parser import EURDFParser

            if xml_path.exists():
                self._e_urdf = EURDFParser(str(xml_path))
            elif mjcf_path.exists():
                self._e_urdf = EURDFParser(str(mjcf_path))
            self.event_bus.publish(
                Event(
                    topic="rosclaw.runtime.robot_loaded",
                    payload={
                        "robot_id": robot_id,
                        "profile": self._robot_profile.to_dict(),
                    },
                    source="runtime",
                    priority=EventPriority.HIGH,
                )
            )
            logger.info(f"Robot '{robot_id}' loaded from e-URDF Zoo")
        except Exception as e:
            logger.info(f"Failed to load robot from zoo: {e}")

    def _register_robot_capabilities(self) -> None:
        """Register e-URDF capabilities as a provider in the registry."""
        if self._robot_profile is None or self._provider_registry is None:
            return
        cap_profile = self._robot_profile.capability
        if not cap_profile or not cap_profile.capabilities:
            return
        from rosclaw.provider.core.manifest import ProviderManifest
        from rosclaw.provider.core.provider import Provider
        from rosclaw.provider.core.request import ProviderRequest
        from rosclaw.provider.core.response import ProviderResponse

        robot_id = self.config.robot_id
        cap_names = []
        for c in cap_profile.capabilities:
            name = c.get("name") or c.get("id", "")
            if name:
                cap_names.append(name)

        class RobotCapabilityProvider(Provider):
            name = "robot_capabilities"
            capabilities = cap_names

            async def infer(self, request: ProviderRequest) -> ProviderResponse:
                return ProviderResponse(
                    request_id=request.request_id,
                    provider=self.name,
                    capability=request.capability,
                    result={"available": True, "robot_id": robot_id},
                )

            async def health(self):
                return {"ok": True}

        self._provider_registry.register(
            ProviderManifest.from_dict(
                {
                    "name": "robot_capabilities",
                    "version": "1.0",
                    "type": "robot",
                    "capabilities": cap_names,
                    "safety": {"executable": True, "requires_guard": True},
                }
            ),
            lambda m: RobotCapabilityProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("robot_capabilities", ok=True)
        logger.info(f"Registered {len(cap_names)} robot capabilities from e-URDF")

    def _on_provider_event(self, event: Event) -> None:
        """Handle provider_registered / provider_unregistered events."""
        logger.info(f"Provider event: {event.topic} — {event.payload}")

    def _on_provider_health_changed(self, event: Event) -> None:
        """Handle provider_health_changed events."""
        payload = event.payload
        name = payload.get("provider", "unknown")
        ok = payload.get("ok", False)
        reason = payload.get("reason", "")
        status = "healthy" if ok else "unhealthy"
        logger.info(f"Provider '{name}' is now {status} ({reason})")

    # ------------------------------------------------------------------
    # Physical World APIs (delegate to MemoryInterface / EmbodiedMemory)
    # ------------------------------------------------------------------

    def add_world_object(self, obj: Any) -> str | None:
        """Add a world object. Requires EmbodiedMemory."""
        if self._memory is None:
            return None
        return self._memory.add_world_object(obj)

    def get_world_object(self, obj_id: str) -> Any | None:
        """Get a world object by ID."""
        if self._memory is None:
            return None
        return self._memory.get_world_object(obj_id)

    def update_world_object_pose(self, obj_id: str, pose: Any, state: str | None = None) -> bool:
        """Update world object pose and optional state."""
        if self._memory is None:
            return False
        return self._memory.update_world_object_pose(obj_id, pose, state)

    def search_world_objects(
        self, center: Any, radius: float, scene_id: str | None = None
    ) -> list[Any]:
        """Search world objects within spatial radius."""
        if self._memory is None:
            return []
        return self._memory.search_world_objects(center, radius, scene_id)

    def get_scene_graph(self, scene_id: str) -> tuple[list[Any], list[Any]]:
        """Get scene graph: (objects, relations)."""
        if self._memory is None:
            return [], []
        return self._memory.get_scene_graph(scene_id)

    def sync_scene_objects(
        self,
        scene_id: str,
        detections: list[Any],
        timestamp_sec: float,
        occlusion_radius: float = 0.5,
    ) -> Any | None:
        """Sync sensor detections with world model (Object Permanence)."""
        if self._memory is None:
            return None
        return self._memory.sync_scene_objects(
            scene_id, detections, timestamp_sec, occlusion_radius
        )

    def cognitive_search(
        self,
        query: str,
        spatial_center: Any | None = None,
        spatial_radius: float = 2.0,
        temporal_interval: Any | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Cognitive search: semantic + spatial + temporal."""
        if self._memory is None:
            return []
        return self._memory.cognitive_search(
            query, spatial_center, spatial_radius, temporal_interval, limit
        )

    def record_trajectory(self, content: str, waypoints: list[tuple[Any, float]]) -> int | None:
        """Record a trajectory. Returns memory_id or None."""
        if self._memory is None:
            return None
        return self._memory.record_trajectory(content, waypoints)

    def search_similar_trajectories(
        self,
        query_waypoints: list[tuple[Any, float]],
        top_k: int = 5,
        max_dtw_distance: float | None = None,
    ) -> list[tuple[Any, float]]:
        """Search for similar trajectories using DTW."""
        if self._memory is None:
            return []
        return self._memory.search_similar_trajectories(query_waypoints, top_k, max_dtw_distance)

    # ------------------------------------------------------------------
    # Integration APIs — v1.0 Minimum Closed-Loop
    # ------------------------------------------------------------------

    def capability_invoke(self, capability_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Invoke a provider capability and return the result.

        Auto-initializes the provider layer if available but not yet set up.
        Falls back to mock responses for known capability patterns when
        the provider layer is unavailable.

        v1.0 KNOW integration: queries KnowledgeInterface before routing
        to check robot capability match and safety limits.

        Example:
            result = rt.capability_invoke(
                "vlm.object_grounding", {"image": "red_cup.jpg"}
            )
        """
        # --- KNOW pre-check (v1.0) ---
        know_result: dict[str, Any] | None = None
        if self._knowledge is not None:
            try:
                know_result = self._knowledge.query_for_provider_selection(
                    capability_name, self.config.robot_id
                )
                logger.info(
                    f"KNOW pre-check for {capability_name}: "
                    f"has_capability={know_result.get('has_capability', False)}"
                )
            except Exception as e:
                logger.info(f"KNOW pre-check failed (non-fatal): {e}")

        # Publish pre-check event for Practice/Memory tracking
        if self.event_bus is not None and know_result is not None:
            self.event_bus.publish(
                Event(
                    topic="rosclaw.provider.inference.requested",
                    payload={
                        "capability": capability_name,
                        "robot_id": self.config.robot_id,
                        "know_result": know_result,
                    },
                    source="runtime",
                    priority=EventPriority.NORMAL,
                )
            )

        # Lazy init: if provider layer is importable but not initialized, set it up now
        if (
            self._capability_router is None
            and ProviderRegistry is not None
            and CapabilityRouter is not None
        ):
            try:
                self._provider_registry = ProviderRegistry(event_bus=self.event_bus)
                self._capability_router = CapabilityRouter(self._provider_registry)
                self._register_builtin_providers()
                logger.info("Provider layer auto-initialized on first capability_invoke")
            except Exception as e:
                logger.info(f"Provider auto-init failed: {e}")

        if self._capability_router is None:
            # Graceful fallback: return mock responses for known capability families
            if capability_name.startswith("vlm."):
                return {
                    "capability": capability_name,
                    "status": "ok",
                    "result": {
                        "mock": True,
                        "objects": [{"label": "mock_object", "confidence": 0.95}],
                    },
                    "provider": "mock_fallback",
                    "note": "CapabilityRouter not initialized — mock fallback used",
                }
            elif capability_name.startswith("skill."):
                return {
                    "capability": capability_name,
                    "status": "ok",
                    "result": {"mock": True, "action": capability_name.replace("skill.", "")},
                    "provider": "mock_fallback",
                    "note": "CapabilityRouter not initialized — mock fallback used",
                }
            return {
                "error": "CapabilityRouter not initialized",
                "capability": capability_name,
                "status": "error",
                "hint": "Set enable_provider=True in RuntimeConfig, or ensure rosclaw.provider is installed",
            }

        try:
            from rosclaw.provider.core.request import ProviderRequest

            request = ProviderRequest(
                capability=capability_name,
                inputs=inputs,
                request_id=f"cap_{uuid.uuid4().hex[:8]}",
            )
            response = self._run_async(self._capability_router.invoke(request))
            return {
                "capability": capability_name,
                "status": "ok" if response.is_ok else "error",
                "result": response.result if response.is_ok else {"error": response.error},
                "provider": response.provider,
            }
        except Exception as e:
            return {"error": str(e), "capability": capability_name, "status": "error"}

    def plan_action(self, instruction: str, perception_result: dict[str, Any]) -> dict[str, Any]:
        """Plan a robot action from natural language instruction + perception.

        Returns a structured action dict ready for sandbox_check().
        """
        try:
            # Try SkillManager plan if available
            if self._skill_manager is not None and hasattr(self._skill_manager, "plan"):
                plan = self._run_async(self._skill_manager.plan(instruction, perception_result))
                return {
                    "status": "ok",
                    "instruction": instruction,
                    "action": plan,
                }
            # Fallback: heuristic action composition
            objects = perception_result.get("result", {}).get("objects", [])
            target = objects[0] if objects else {"label": "unknown"}
            action = {
                "type": "pick_and_place",
                "target": target.get("label", "unknown"),
                "trajectory": [
                    [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
                    [0.1, -1.4, 1.4, 0.0, 0.1, 0.0],
                ],
                "grip_force": 20.0,
            }
            return {
                "status": "ok",
                "instruction": instruction,
                "action": action,
            }
        except Exception as e:
            return {"error": str(e), "instruction": instruction, "status": "error"}

    def sandbox_check(self, action: dict[str, Any]) -> dict[str, Any]:
        """Check an action against the sandbox firewall.

        Returns:
            {"decision": "ALLOW" | "BLOCK", "reason": str, "violations": list}
        """
        if self._firewall is None:
            return {"decision": "ALLOW", "reason": "firewall_disabled", "violations": []}
        try:
            from rosclaw.firewall.validator import ValidationRequest

            trajectory = action.get("trajectory", [])
            request = ValidationRequest(
                request_id=action.get("request_id", "sandbox_check_001"),
                robot_id=self.config.robot_id,
                trajectory=trajectory,
                source="runtime",
                metadata=action,
            )
            response = self._firewall.validate(request)
            if response.is_safe:
                return {"decision": "ALLOW", "reason": "safe", "violations": []}
            return {
                "decision": "BLOCK",
                "reason": "; ".join(v.description for v in response.violations),
                "violations": [
                    {"layer": str(v.layer), "severity": v.severity, "description": v.description}
                    for v in response.violations
                ],
            }
        except Exception as e:
            return {"decision": "BLOCK", "reason": str(e), "violations": []}

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute an action through the **full closed loop**.

        1. Publish ``skill.execution.start``
        2. Query provider for action plan (capability_invoke)
        3. Sandbox firewall check
        4. Publish ``provider.inference.completed``
        5. Generate real joint trajectory via sandbox physics
        6. Publish ``sandbox.episode.started`` + ``sandbox.action.allowed``
        7. Publish ``skill.execution.complete``
        8. Critic evaluation
        9. Publish ``praxis.completed`` / ``praxis.failed``
        10. Memory auto-ingest via ``praxis.recorded``
        11. Publish ``dashboard.trace.updated``

        Returns execution result dict.
        """
        import time
        import uuid

        request_id = action.get("request_id", str(uuid.uuid4())[:8])
        instruction = action.get("instruction", "")
        skill_name = action.get("skill_name", "unknown")
        t0 = time.time()
        episode_id = f"ep_{int(t0)}_{uuid.uuid4().hex[:6]}"

        # 1. skill.execution.start
        self.event_bus.publish(
            Event(
                topic="skill.execution.start",
                payload={
                    "episode_id": episode_id,
                    "request_id": request_id,
                    "skill_name": skill_name,
                    "instruction": instruction,
                    "parameters": action.get("parameters", {}),
                    "robot_id": self.config.robot_id,
                    "agent_request": action,
                },
                source="runtime",
                priority=EventPriority.HIGH,
                trace_id=request_id,
            )
        )

        # 2. Provider inference (action plan)
        provider_result: dict[str, Any] = {}
        try:
            capability = action.get("capability", f"skill.{skill_name}")
            provider_inputs = action.get("parameters", {})
            provider_result = self.capability_invoke(capability, provider_inputs)
        except Exception as e:
            provider_result = {"status": "error", "error": str(e)}

        # 3. Sandbox firewall check
        sandbox_result = self.sandbox_check(action)
        is_blocked = sandbox_result.get("decision") == "BLOCK"

        # 4. provider.inference.completed
        self.event_bus.publish(
            Event(
                topic="rosclaw.provider.inference.completed",
                payload={
                    "episode_id": episode_id,
                    "request_id": request_id,
                    "capability": action.get("capability", ""),
                    "provider": provider_result.get("provider", ""),
                    "status": provider_result.get("status", "unknown"),
                    "latency_ms": int((time.time() - t0) * 1000),
                },
                source="runtime",
                priority=EventPriority.NORMAL,
            )
        )

        # 5. Generate real joint trajectory via sandbox physics
        trajectory_data: list[dict] = []
        if is_blocked:
            result = {
                "status": "blocked",
                "decision": "BLOCK",
                "reason": sandbox_result.get("reason", "firewall"),
                "violations": sandbox_result.get("violations", []),
                "source": "firewall",
            }
        else:
            if self._sandbox is not None and hasattr(self._sandbox, "validate_trajectory"):
                try:
                    raw_trajectory = action.get("trajectory", [])
                    if not raw_trajectory:
                        raw_trajectory = self._generate_trajectory(action)

                    validation = self._sandbox.validate_trajectory(raw_trajectory)

                    # Use real physics stepping if sandbox has MuJoCo model
                    has_real_physics = (
                        self._sandbox is not None and getattr(self._sandbox, "has_physics", False)  # noqa: W503
                    )
                    dt = 0.01
                    for i, waypoint in enumerate(raw_trajectory):
                        if has_real_physics:
                            state = self._sandbox.simulate_step(waypoint)
                            if state:
                                joint_positions = state.get("qpos", waypoint)[: len(waypoint)]
                                timestamp = state.get("time", i * dt)
                            else:
                                joint_positions = waypoint
                                timestamp = i * dt
                        else:
                            joint_positions = waypoint
                            timestamp = i * dt
                        trajectory_data.append(
                            {
                                "timestamp": timestamp,
                                "joint_positions": joint_positions,
                                "phase": "approach"
                                if i < len(raw_trajectory) * 0.3
                                else ("grasp" if i < len(raw_trajectory) * 0.6 else "retract"),
                            }
                        )

                    if validation.get("is_safe", True):
                        result = {
                            "status": "ok",
                            "trajectory": raw_trajectory,
                            "trajectory_data": trajectory_data,
                            "validation": validation,
                            "final_position": raw_trajectory[-1] if raw_trajectory else [],
                            "source": "sandbox",
                        }
                    else:
                        result = {
                            "status": "blocked",
                            "decision": "BLOCK",
                            "reason": validation.get("reason", "unsafe"),
                            "violations": validation.get("violations", []),
                            "source": "sandbox",
                        }
                        is_blocked = True
                except Exception as e:
                    result = {"status": "error", "error": str(e), "source": "sandbox"}
            else:
                raw_trajectory = self._generate_trajectory(action)
                for i, waypoint in enumerate(raw_trajectory):
                    trajectory_data.append(
                        {
                            "timestamp": i * 0.01,
                            "joint_positions": waypoint,
                            "phase": "mock",
                        }
                    )
                result = {
                    "status": "ok",
                    "trajectory": raw_trajectory,
                    "trajectory_data": trajectory_data,
                    "source": "fallback",
                    "note": "No sandbox physics -- mock trajectory used",
                }

        duration = time.time() - t0

        # 6. sandbox events
        if is_blocked:
            failure_text = " ".join(
                [
                    str(sandbox_result.get("reason", "")),
                    *[
                        str(violation.get("description", ""))
                        for violation in sandbox_result.get("violations", [])
                        if isinstance(violation, dict)
                    ],
                ]
            ).lower()
            if "joint" in failure_text and "limit" in failure_text:
                sandbox_failure_type = "joint_limit_exceeded"
            elif "collision" in failure_text:
                sandbox_failure_type = "collision_detected"
            elif "workspace" in failure_text or "out of bounds" in failure_text:
                sandbox_failure_type = "workspace_boundary_exceeded"
            else:
                sandbox_failure_type = "firewall_blocked"

            self.event_bus.publish(
                Event(
                    topic="firewall.action_blocked",
                    payload={
                        "episode_id": episode_id,
                        "request_id": request_id,
                        "task_id": action.get("task_id", request_id),
                        "robot_id": self.config.robot_id,
                        "skill_id": skill_name,
                        "failure_type": sandbox_failure_type,
                        "action": action,
                        "violations": sandbox_result.get("violations", []),
                        "reason": sandbox_result.get("reason", ""),
                    },
                    source="sandbox",
                    priority=EventPriority.HIGH,
                    trace_id=request_id,
                )
            )
            # Memory auto-ingests firewall.action_blocked via EventBus subscription
            # (see MemoryInterface._on_firewall_action_blocked)
            # How recovery hint generation (P0-6)
            if self._how is not None:
                try:
                    hint = self._run_async(
                        self._how.generate_recovery_hint(
                            sandbox_failure_type,
                            context={
                                "skill_name": skill_name,
                                "instruction": instruction,
                                "violations": sandbox_result.get("violations", []),
                                "reason": sandbox_result.get("reason", ""),
                            },
                        )
                    )
                    if hint:
                        self.event_bus.publish(
                            Event(
                                topic="rosclaw.how.recovery_hint.generated",
                                payload={
                                    "episode_id": episode_id,
                                    "failure_id": episode_id,
                                    "request_id": request_id,
                                    "task_id": action.get("task_id", request_id),
                                    "robot_id": self.config.robot_id,
                                    "skill_id": skill_name,
                                    "hint": hint.get("hint", ""),
                                    "rule_id": hint.get("rule_id", ""),
                                    "failure_type": sandbox_failure_type,
                                },
                                source="how",
                                priority=EventPriority.HIGH,
                            )
                        )
                except Exception as e:
                    logger.info(f"How recovery hint failed (non-fatal): {e}")
        else:
            self.event_bus.publish(
                Event(
                    topic="rosclaw.sandbox.episode.started",
                    payload={
                        "episode_id": episode_id,
                        "request_id": request_id,
                        "world_id": getattr(self._sandbox, "_world_id", "empty")
                        if self._sandbox
                        else "empty",
                        "robot_id": self.config.robot_id,
                    },
                    source="sandbox",
                    priority=EventPriority.NORMAL,
                )
            )
            self.event_bus.publish(
                Event(
                    topic="rosclaw.sandbox.action.allowed",
                    payload={
                        "episode_id": episode_id,
                        "request_id": request_id,
                        "risk_score": sandbox_result.get("risk_score", 0.0),
                    },
                    source="sandbox",
                    priority=EventPriority.NORMAL,
                )
            )

        # 7. skill.execution.complete
        self.event_bus.publish(
            Event(
                topic="skill.execution.complete",
                payload={
                    "episode_id": episode_id,
                    "request_id": request_id,
                    "skill_name": skill_name,
                    "result": result,
                    "duration_sec": duration,
                    "trajectory_waypoints": len(trajectory_data),
                },
                source="runtime",
                priority=EventPriority.HIGH,
            )
        )

        # 8. Critic evaluation
        critic_reward = 0.0
        critic_status = "UNKNOWN"
        if result.get("status") == "ok":
            critic_reward = 1.0
            critic_status = "SUCCESS"
            if len(trajectory_data) > 2:
                max_diff = 0.0
                for i in range(1, len(trajectory_data)):
                    prev = trajectory_data[i - 1].get("joint_positions", [])
                    curr = trajectory_data[i].get("joint_positions", [])
                    if prev and curr and len(prev) == len(curr):
                        diff = sum(abs(a - b) for a, b in zip(prev, curr, strict=False))
                        max_diff = max(max_diff, diff)
                if max_diff > 0.5:
                    critic_reward -= 0.2
        elif is_blocked:
            critic_reward = -1.0
            critic_status = "BLOCKED"
        else:
            critic_reward = -1.0
            critic_status = "FAILED"

        self.event_bus.publish(
            Event(
                topic="rosclaw.critic.success.detected",
                payload={
                    "episode_id": episode_id,
                    "request_id": request_id,
                    "reward": critic_reward,
                    "status": critic_status,
                    "skill_name": skill_name,
                },
                source="critic",
                priority=EventPriority.NORMAL,
            )
        )

        # 9. praxis.completed / praxis.failed
        final_event_topic = "praxis.completed" if result.get("status") == "ok" else "praxis.failed"
        self.event_bus.publish(
            Event(
                topic=final_event_topic,
                payload={
                    "episode_id": episode_id,
                    "event_type": "success" if result.get("status") == "ok" else "failure",
                    "correlation_id": request_id,
                    "request_id": request_id,
                    "task_id": action.get("task_id", request_id),
                    "robot_id": self.config.robot_id,
                    "skill_id": skill_name,
                    "skill_name": skill_name,
                    "instruction": instruction,
                    "initial_state": action.get("initial_state"),
                    "final_state": result,
                    "duration_sec": duration,
                    "outcome": {"reward": critic_reward, "status": critic_status},
                    "trajectory": trajectory_data,
                },
                source="runtime",
                priority=EventPriority.NORMAL,
            )
        )

        # 10. Memory auto-ingest
        if self._memory is not None:
            try:
                tags = [skill_name, self.config.robot_id]
                if is_blocked:
                    tags.append("blocked")
                if result.get("status") == "ok":
                    tags.append("success")
                else:
                    tags.append("failure")
                self._memory.store_experience(
                    event_id=episode_id,
                    event_type="praxis",
                    instruction=instruction,
                    outcome="success" if result.get("status") == "ok" else "failure",
                    duration_sec=duration,
                    error_details=result.get("reason") if is_blocked else result.get("error"),
                    tags=tags,
                    metadata={
                        "request_id": request_id,
                        "skill_name": skill_name,
                        "critic_reward": critic_reward,
                        "critic_status": critic_status,
                        "trajectory_waypoints": len(trajectory_data),
                        "sandbox_blocked": is_blocked,
                        "provider": provider_result.get("provider", ""),
                        "validation": sandbox_result,
                    },
                )
            except Exception as e:
                logger.info(f"Memory auto-ingest failed (non-fatal): {e}")

        # 11. dashboard.trace.updated
        self.event_bus.publish(
            Event(
                topic="rosclaw.dashboard.trace.updated",
                payload={
                    "episode_id": episode_id,
                    "request_id": request_id,
                    "robot_id": self.config.robot_id,
                    "skill_name": skill_name,
                    "status": result.get("status", "unknown"),
                    "critic_reward": critic_reward,
                    "duration_sec": duration,
                },
                source="dashboard",
                priority=EventPriority.LOW,
            )
        )

        # KNOW post-execution recording
        if self._knowledge is not None:
            try:
                self._knowledge.record_knowledge_usage(
                    {
                        "episode_id": episode_id,
                        "robot_id": self.config.robot_id,
                        "action": action,
                        "result": result,
                        "duration_sec": duration,
                        "knowledge_queried": True,
                    }
                )
            except Exception as e:
                logger.info(f"KNOW post-execution recording failed (non-fatal): {e}")

        return result

    def _generate_trajectory(self, action: dict[str, Any]) -> list[list[float]]:
        """Generate a joint trajectory from action parameters."""
        target = (
            action.get("target_pose")
            or action.get("target")
            or action.get("parameters", {}).get("target_pose")
        )
        if target and isinstance(target, list):
            steps = 10
            start = [0.0] * len(target)
            trajectory = []
            for i in range(steps + 1):
                t = i / steps
                waypoint = [start[j] + (target[j] - start[j]) * t for j in range(len(target))]
                trajectory.append(waypoint)
            return trajectory
        return [
            [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
            [0.1, -1.4, 1.4, 0.0, 0.1, 0.0],
            [0.2, -1.2, 1.2, 0.0, 0.2, 0.0],
            [0.3, -1.0, 1.0, 0.0, 0.3, 0.0],
            [0.4, -0.8, 0.8, 0.0, 0.4, 0.0],
        ]

    def get_status(self) -> dict:
        """Get comprehensive runtime status."""
        sense_status: dict[str, Any] = {
            "enabled": self.config.enable_sense,
            "available": self._sense is not None,
        }
        if self._sense is not None:
            try:
                latest = self._sense.get_body_sense()
                if latest is not None:
                    sense_status["overall_status"] = latest.overall_status
                    sense_status["blocked_capabilities"] = latest.blocked_capabilities
                    sense_status["degraded_capabilities"] = latest.degraded_capabilities
            except Exception as exc:  # noqa: BLE001
                sense_status["error"] = str(exc)

        return {
            "robot_id": self.config.robot_id,
            "runtime_state": self.state.name,
            "event_bus": {
                "topics": self.event_bus.topics,
                "history_size": len(self.event_bus._event_history),
            },
            "modules": {
                "firewall": self._firewall is not None,
                "memory": self._memory is not None,
                "practice": self._practice is not None,
                "swarm": self._swarm is not None,
                "skill_manager": self._skill_manager is not None,
                "e_urdf": self._e_urdf is not None,
                "provider_layer": self._provider_registry is not None,
                "auto": self._auto is not None,
                "sense": self._sense is not None,
            },
            "body_sense": sense_status,
            "embodied_memory": {
                "attached": self.config.embodied_memory is not None,
                "has_world_objects": (self._memory.has_embodied_memory if self._memory else False),
            },
            "drivers": list(self._mcp_drivers.keys()),
        }


class _HowProxy:
    """Sync wrapper for HeuristicEngine with EventBus transparency.

    All calls are proxied through the EventBus so that other modules
    (Dashboard, Memory, etc.) can observe How activity without direct coupling.
    """

    def __init__(self, engine, run_async, event_bus: Any | None = None):
        self._engine = engine
        self._run_async = run_async
        self._event_bus = event_bus

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._engine, name)
        if not callable(attr):
            return attr

        def _wrapper(*args, **kwargs):
            import asyncio

            # Publish request event for observability / decoupling
            if self._event_bus is not None:
                with contextlib.suppress(Exception):
                    self._event_bus.publish(
                        Event(
                            topic=f"how.{name}.requested",
                            payload={"method": name, "args": str(args), "kwargs": str(kwargs)},
                            source="runtime_proxy",
                        )
                    )

            # Execute the real call (async → sync via _run_async)
            try:
                if asyncio.iscoroutinefunction(attr):
                    result = self._run_async(attr(*args, **kwargs))
                else:
                    result = attr(*args, **kwargs)
            except Exception as exc:
                if self._event_bus is not None:
                    with contextlib.suppress(Exception):
                        self._event_bus.publish(
                            Event(
                                topic=f"how.{name}.failed",
                                payload={"method": name, "error": str(exc)},
                                source="runtime_proxy",
                            )
                        )
                raise

            # Publish completion event
            if self._event_bus is not None:
                with contextlib.suppress(Exception):
                    self._event_bus.publish(
                        Event(
                            topic=f"how.{name}.completed",
                            payload={"method": name, "result_type": type(result).__name__},
                            source="runtime_proxy",
                        )
                    )
            return result

        return _wrapper


class _MemoryProxy:
    """Transparent proxy for MemoryInterface with EventBus publishing.

    Every Memory call is mirrored as an EventBus event so that the
    rest of the system (Dashboard, Practice, etc.) can observe
    Memory activity without direct coupling to Runtime.
    """

    def __init__(self, memory, event_bus: Any | None = None):
        self._memory = memory
        self._event_bus = event_bus

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._memory, name)
        if not callable(attr):
            return attr

        def _wrapper(*args, **kwargs):
            if self._event_bus is not None:
                with contextlib.suppress(Exception):
                    self._event_bus.publish(
                        Event(
                            topic=f"memory.{name}.requested",
                            payload={"method": name, "args": str(args), "kwargs": str(kwargs)},
                            source="runtime_proxy",
                        )
                    )

            try:
                result = attr(*args, **kwargs)
            except Exception as exc:
                if self._event_bus is not None:
                    with contextlib.suppress(Exception):
                        self._event_bus.publish(
                            Event(
                                topic=f"memory.{name}.failed",
                                payload={"method": name, "error": str(exc)},
                                source="runtime_proxy",
                            )
                        )
                raise

            if self._event_bus is not None:
                with contextlib.suppress(Exception):
                    self._event_bus.publish(
                        Event(
                            topic=f"memory.{name}.completed",
                            payload={"method": name, "result_type": type(result).__name__},
                            source="runtime_proxy",
                        )
                    )
            return result

        return _wrapper
