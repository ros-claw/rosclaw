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

from dataclasses import dataclass, field
from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState

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
    from rosclaw.provider.core.request import ProviderRequest
    from rosclaw.provider.guard.pipeline import GuardPipeline
    from rosclaw.provider.guard.schema_guard import SchemaGuard
    from rosclaw.provider.guard.action_guard import ActionGuard
except ImportError:
    pass


@dataclass
class RuntimeConfig:
    """Configuration for ROSClaw Runtime."""
    robot_id: str = "rosclaw_default"
    robot_model_path: Optional[str] = None
    robot_zoo_path: Optional[str] = None
    default_eurdf_robot: str = "ur5e"
    enable_firewall: bool = True
    enable_memory: bool = True
    enable_practice: bool = True
    enable_swarm: bool = False
    enable_skill_manager: bool = True
    enable_knowledge: bool = False          # KnowledgeInterface (KNOW module)
    enable_how: bool = False                # HeuristicEngine (HOW module)
    enable_provider: bool = True
    joint_dof: int = 6
    sampling_rate_hz: int = 1000
    safety_level: str = "MODERATE"          # STRICT | MODERATE | LENIENT
    timeline_output_dir: str = "./practice_data"
    enable_mcap: bool = False
    seekdb_backend: str = "memory"          # "memory" | "sqlite"
    seekdb_path: str = "./seekdb.sqlite"
    embodied_memory: Optional[Any] = None   # powermem.EmbodiedMemory instance
    providers_dir: Optional[str] = None     # Directory to scan for provider.yaml files


class Runtime(LifecycleMixin):
    """
    Central runtime orchestrator for ROSClaw.

    This is the main entry point for the ROSClaw OS. It coordinates
    all grounding engines and provides a unified interface for
    agent runtimes to interact with the physical world.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None):
        super().__init__()
        self.config = config or RuntimeConfig()

        # Core infrastructure
        self.event_bus = EventBus()

        # Grounding engines (initialized on demand)
        self._firewall: Optional[Any] = None
        self._memory: Optional[Any] = None
        self._practice: Optional[Any] = None
        self._swarm: Optional[Any] = None
        self._skill_manager: Optional[Any] = None
        self._knowledge: Optional[Any] = None
        self._how: Optional[Any] = None
        self._e_urdf: Optional[Any] = None
        self._robot_profile: Optional[Any] = None
        self._sandbox: Optional[Any] = None
        self._episode_recorder: Optional[Any] = None
        self._mcp_drivers: dict[str, Any] = {}

        # Provider layer
        self._provider_registry: Optional[Any] = None
        self._capability_router: Optional[Any] = None
        self._guard_pipeline: Optional[Any] = None

        # Internal state
        self._agent_runtime: Optional[Any] = None
        self._modules: list[LifecycleMixin] = []

    def _do_initialize(self) -> None:
        """Initialize all enabled grounding engines."""
        print(f"[Runtime] Initializing ROSClaw Runtime for {self.config.robot_id}")

        # Initialize EventBus subscriptions for internal coordination
        self._setup_internal_subscriptions()

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
                print("[Runtime] Action Grounding (FirewallValidator) initialized")
            except ImportError as e:
                print(f"[Runtime] FirewallValidator not available: {e}")

        # Initialize Sandbox (Digital Twin + Physics Simulation)
        try:
            from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter
            sandbox_config = {
                "engine": "mujoco",
                "world_id": "empty",
                "robot_id": self.config.robot_id,
            }
            self._sandbox = SandboxRuntimeAdapter(
                config=sandbox_config,
                event_bus=self.event_bus,
                e_urdf_model=self._e_urdf,
            )
            self._modules.append(self._sandbox)
            print("[Runtime] Sandbox (Digital Twin + Physics) initialized")
        except ImportError as e:
            print(f"[Runtime] Sandbox not available: {e}")

        # Initialize Experience Grounding (Memory)
        if self.config.enable_memory:
            try:
                from rosclaw.memory.interface import MemoryInterface
                from rosclaw.memory.seekdb_client import SeekDBSQLiteClient, SeekDBMemoryClient
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
                em_label = "+EmbodiedMemory" if self.config.embodied_memory else "SeekDB-only"
                print(f"[Runtime] Experience Grounding (Memory) initialized [{em_label}]")
            except ImportError as e:
                print(f"[Runtime] Memory module not available: {e}")

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
                print("[Runtime] Timeline Grounding (UnifiedTimeline) initialized")
            except ImportError as e:
                print(f"[Runtime] UnifiedTimeline not available: {e}")

            # Initialize EpisodeRecorder for artifact management
            try:
                from rosclaw.practice.episode_recorder import EpisodeRecorder
                self._episode_recorder = EpisodeRecorder(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                )
                self._modules.append(self._episode_recorder)
                print("[Runtime] EpisodeRecorder initialized")
            except ImportError as e:
                print(f"[Runtime] EpisodeRecorder not available: {e}")

        # Initialize Collaboration Grounding (Swarm)
        if self.config.enable_swarm:
            try:
                from rosclaw.swarm.manager import SwarmRuntimeManager
                self._swarm = SwarmRuntimeManager(event_bus=self.event_bus)
                self._modules.append(self._swarm)
                print("[Runtime] Collaboration Grounding (Swarm) initialized")
            except ImportError as e:
                print(f"[Runtime] Swarm module not available: {e}")

        # Initialize Skill Grounding (SkillManager)
        if self.config.enable_skill_manager:
            try:
                from rosclaw.skill_manager.registry import SkillRegistry
                from rosclaw.skill_manager.executor import SkillExecutor
                registry = SkillRegistry(event_bus=self.event_bus)
                self._skill_manager = SkillExecutor(self.event_bus, registry)
                self._modules.append(registry)
                self._modules.append(self._skill_manager)
                print("[Runtime] Skill Grounding (SkillManager) initialized")
            except ImportError as e:
                print(f"[Runtime] SkillManager module not available: {e}")

        # Initialize Knowledge Grounding (KnowledgeInterface) - MUST come before HOW
        if self.config.enable_knowledge:
            try:
                from rosclaw.know.interface import KnowledgeInterface
                from rosclaw.know.storage import seed_knowledge_graph
                # Reuse Memory's SeekDB client if available
                seekdb = None
                if self._memory is not None:
                    seekdb = getattr(self._memory, "seekdb_client", None)
                self._knowledge = KnowledgeInterface(
                    robot_id=self.config.robot_id,
                    event_bus=self.event_bus,
                    seekdb_client=seekdb,
                )
                self._modules.append(self._knowledge)
                print("[Runtime] Knowledge Grounding (KnowledgeInterface) initialized")
                # Seed knowledge_graph with baseline data
                if seekdb is not None:
                    seed_knowledge_graph(seekdb)
            except ImportError as e:
                print(f"[Runtime] Knowledge module not available: {e}")

        # Initialize Heuristic Grounding (HeuristicEngine) - depends on KNOW
        if self.config.enable_how:
            try:
                from rosclaw.how.engine import HeuristicEngine
                # Reuse Memory's SeekDB client if available
                seekdb = None
                if self._memory is not None:
                    seekdb = getattr(self._memory, "seekdb_client", None)
                if seekdb is not None:
                    self._how = HeuristicEngine(
                        seekdb_client=seekdb,
                        knowledge_interface=self._knowledge,
                    )
                    self._modules.append(self._how)
                    # HeuristicEngine is not a LifecycleMixin;
                    # seed defaults explicitly here.
                    self._run_async(self._how.seed_defaults())
                    print("[Runtime] Heuristic Grounding (HeuristicEngine) initialized")
                else:
                    print("[Runtime] HeuristicEngine skipped: no SeekDB client (memory not enabled)")
            except ImportError as e:
                print(f"[Runtime] HeuristicEngine not available: {e}")

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
                    print(f"[Runtime] Built-in provider registration warning: {e}")
                try:
                    self._register_robot_capabilities()
                except Exception as e:
                    print(f"[Runtime] Robot capability registration warning: {e}")
                try:
                    self._load_external_providers()
                except Exception as e:
                    print(f"[Runtime] External provider loading warning: {e}")
                print("[Runtime] Provider Layer (Registry + Router + Guard) initialized")
            except Exception as e:
                print(f"[Runtime] Provider layer not available: {e}")

        # Initialize all module lifecycles
        for module in self._modules:
            if isinstance(module, LifecycleMixin):
                module.initialize()

        print("[Runtime] Initialization complete")

    def _do_start(self) -> None:
        """Start all modules."""
        print("[Runtime] Starting all modules...")
        for module in self._modules:
            if isinstance(module, LifecycleMixin) and module.is_ready:
                module.start()
        self.event_bus.publish(Event(
            topic="runtime.status",
            payload={"state": "running", "robot_id": self.config.robot_id},
            source="runtime",
            priority=EventPriority.HIGH,
        ))
        print("[Runtime] All modules started")

    def _do_stop(self) -> None:
        """Stop all modules gracefully."""
        print("[Runtime] Shutting down...")
        self.event_bus.publish(Event(
            topic="runtime.status",
            payload={"state": "shutting_down", "robot_id": self.config.robot_id},
            source="runtime",
            priority=EventPriority.CRITICAL,
        ))
        # Stop in reverse order
        for module in reversed(self._modules):
            if isinstance(module, LifecycleMixin):
                module.stop()
        print("[Runtime] Shutdown complete")

    def _setup_internal_subscriptions(self) -> None:
        """Set up internal EventBus subscriptions for runtime coordination."""
        self.event_bus.subscribe("safety.violation", self._on_safety_violation)
        self.event_bus.subscribe("agent.command", self._on_agent_command)
        self.event_bus.subscribe("robot.emergency_stop", self._on_emergency_stop)
        self.event_bus.subscribe("firewall.action_blocked", self._on_firewall_action_blocked)
        self.event_bus.subscribe("rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed)
        self.event_bus.subscribe("rosclaw.sandbox.action.blocked", self._on_sandbox_action_blocked)
        self.event_bus.subscribe("rosclaw.runtime.execution.failed", self._on_runtime_execution_failed)

    def _on_safety_violation(self, event: Event) -> None:
        """Handle safety violation events."""
        print(f"[Runtime] SAFETY VIOLATION: {event.payload}")
        self.event_bus.publish(Event(
            topic="robot.emergency_stop",
            payload={"reason": event.payload},
            source="runtime",
            priority=EventPriority.CRITICAL,
        ))

    def _on_firewall_action_blocked(self, event: Event) -> None:
        """Handle firewall action blocked: query heuristic recovery.

        When Firewall blocks an agent command, this handler queries the
        HeuristicEngine for a recovery suggestion and publishes it on
        the EventBus so the agent can attempt recovery before escalating.
        """
        if self._how is None:
            return
        try:
            import asyncio
            request_id = event.payload.get("request_id", "")
            violations = event.payload.get("violations", [])
            error_log = "; ".join(v.get("description", "") for v in violations)
            recovery = asyncio.run(self._how.suggest_recovery(error_log, context={"request_id": request_id}))
            if recovery:
                from rosclaw.how.recovery import RecoveryFormatter
                payload = RecoveryFormatter.to_event_payload(
                    recovery, request_id=request_id, source="heuristic_engine"
                )
                self.event_bus.publish(Event(
                    topic="heuristic.recovery_suggested",
                    payload=payload,
                    source="runtime",
                    priority=EventPriority.HIGH,
                ))
                print(f"[Runtime] Heuristic recovery suggested for {request_id}: {recovery['action']}")
        except Exception as e:
            print(f"[Runtime] Heuristic recovery failed: {e}")

    def _run_async(self, coro):
        """Run an async coroutine from a sync context.

        Handles both production (no running loop) and test (pytest-asyncio
        with running loop) contexts gracefully.
        """
        import asyncio
        import concurrent.futures
        try:
            loop = asyncio.get_running_loop()
            # Already inside an event loop — run in a thread to avoid
            # "cannot run nested event loop" errors during tests.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(coro)

    def _on_sandbox_episode_failed(self, event: Event) -> None:
        """Handle sandbox episode failure: generate recovery hint."""
        if self._how is None:
            return
        try:
            from rosclaw.how.recovery import RecoveryEngine

            failure_type = event.payload.get("failure_type", "")
            request_id = event.payload.get("request_id", "")
            re = RecoveryEngine(self._how)
            hint = self._run_async(re.generate_recovery_hint(
                failure_type,
                context={"request_id": request_id, "source": "sandbox"},
                sources=["sandbox_episode"],
            ))
            if hint:
                payload = re.format_for_eventbus(hint, request_id=request_id)
                self.event_bus.publish(Event(
                    topic="rosclaw.how.recovery_hint.generated",
                    payload=payload,
                    source="runtime",
                    priority=EventPriority.HIGH,
                ))
                print(f"[Runtime] RecoveryHint generated for sandbox failure {request_id}: {hint['hint']}")
        except Exception as e:
            print(f"[Runtime] RecoveryHint generation failed: {e}")

    def _on_sandbox_action_blocked(self, event: Event) -> None:
        """Handle sandbox action blocked: generate recovery hint."""
        if self._how is None:
            return
        try:
            from rosclaw.how.recovery import RecoveryEngine

            failure_type = event.payload.get("reason", "")
            request_id = event.payload.get("request_id", "")
            re = RecoveryEngine(self._how)
            hint = self._run_async(re.generate_recovery_hint(
                failure_type,
                context={"request_id": request_id, "source": "sandbox"},
                sources=["sandbox_action"],
            ))
            if hint:
                payload = re.format_for_eventbus(hint, request_id=request_id)
                self.event_bus.publish(Event(
                    topic="rosclaw.how.recovery_hint.generated",
                    payload=payload,
                    source="runtime",
                    priority=EventPriority.HIGH,
                ))
                print(f"[Runtime] RecoveryHint generated for blocked action {request_id}: {hint['hint']}")
        except Exception as e:
            print(f"[Runtime] RecoveryHint generation failed: {e}")

    def _on_runtime_execution_failed(self, event: Event) -> None:
        """Handle runtime execution failure: generate recovery hint."""
        if self._how is None:
            return
        try:
            from rosclaw.how.recovery import RecoveryEngine

            failure_type = event.payload.get("error_type", "")
            request_id = event.payload.get("request_id", "")
            re = RecoveryEngine(self._how)
            hint = self._run_async(re.generate_recovery_hint(
                failure_type,
                context={"request_id": request_id, "source": "runtime"},
                sources=["runtime_execution"],
            ))
            if hint:
                payload = re.format_for_eventbus(hint, request_id=request_id)
                self.event_bus.publish(Event(
                    topic="rosclaw.how.recovery_hint.generated",
                    payload=payload,
                    source="runtime",
                    priority=EventPriority.HIGH,
                ))
                print(f"[Runtime] RecoveryHint generated for execution failure {request_id}: {hint['hint']}")
        except Exception as e:
            print(f"[Runtime] RecoveryHint generation failed: {e}")

    def _on_agent_command(self, event: Event) -> None:
        """Handle agent commands - route to appropriate module."""
        command = event.payload.get("action", "")
        print(f"[Runtime] Agent command received: {command}")

    def _on_emergency_stop(self, event: Event) -> None:
        """Handle emergency stop - stop all drivers."""
        print("[Runtime] EMERGENCY STOP triggered")
        for driver in self._mcp_drivers.values():
            if hasattr(driver, "emergency_stop"):
                driver.emergency_stop()

    def register_driver(self, name: str, driver: Any) -> None:
        """Register an MCP driver with the runtime."""
        self._mcp_drivers[name] = driver
        print(f"[Runtime] Driver registered: {name}")

    def get_driver(self, name: str) -> Optional[Any]:
        """Get a registered driver by name."""
        return self._mcp_drivers.get(name)

    @property
    def firewall(self) -> Optional[Any]:
        return self._firewall

    @property
    def memory(self) -> Optional[Any]:
        return self._memory

    @property
    def practice(self) -> Optional[Any]:
        return self._practice

    @property
    def swarm(self) -> Optional[Any]:
        return self._swarm

    @property
    def skill_manager(self) -> Optional[Any]:
        return self._skill_manager

    @property
    def knowledge(self) -> Optional[Any]:
        return self._knowledge

    @property
    def how(self) -> Optional[Any]:
        if self._how is None:
            return None
        return _HowProxy(self._how, self._run_async)

    @property
    def e_urdf(self) -> Optional[Any]:
        return self._e_urdf

    @property
    def status(self) -> dict:
        """Get comprehensive runtime status (alias for get_status)."""
        return self.get_status()

    @property
    def provider_registry(self) -> Optional[Any]:
        return self._provider_registry

    @property
    def capability_router(self) -> Optional[Any]:
        return self._capability_router

    @property
    def guard_pipeline(self) -> Optional[Any]:
        return self._guard_pipeline

    def _load_external_providers(self) -> None:
        """Scan providers_dir for provider.yaml files and load them."""
        if not self.config.providers_dir:
            return
        try:
            from rosclaw.provider.loader import ProviderLoader
            loader = ProviderLoader(self._provider_registry)
            loaded = loader.scan_directory(self.config.providers_dir)
            if loaded:
                print(f"[Runtime] Loaded external providers: {loaded}")
        except Exception as e:
            print(f"[Runtime] Failed to load external providers: {e}")

    def _register_builtin_providers(self) -> None:
        """Register built-in mock providers for out-of-box capability support."""
        from rosclaw.provider.builtins import (
            MockVLMProvider,
            MockSkillProvider,
            MockCriticProvider,
        )
        from rosclaw.provider.core.manifest import ProviderManifest

        # Subscribe to provider lifecycle events before registering builtins
        self.event_bus.subscribe("provider_registered", self._on_provider_event)
        self.event_bus.subscribe("provider_unregistered", self._on_provider_event)
        self.event_bus.subscribe("provider_health_changed", self._on_provider_health_changed)

        self._provider_registry.register(
            ProviderManifest.from_dict({
                "name": "mock_vlm", "version": "0.1.0", "type": "vlm",
                "capabilities": ["vlm.object_grounding", "vlm.scene_understanding"],
                "modalities": {"input": ["image", "text"], "output": ["object_list"]},
                "safety": {"executable": False, "requires_guard": False},
            }),
            lambda m: MockVLMProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("mock_vlm", ok=True)

        self._provider_registry.register(
            ProviderManifest.from_dict({
                "name": "mock_skill", "version": "0.1.0", "type": "skill",
                "capabilities": ["skill.grasp", "skill.place", "skill.pick_and_place"],
                "embodiment": {"supported_robots": []},
                "safety": {"executable": True, "requires_guard": True},
            }),
            lambda m: MockSkillProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("mock_skill", ok=True)

        self._provider_registry.register(
            ProviderManifest.from_dict({
                "name": "mock_critic", "version": "0.1.0", "type": "critic",
                "capabilities": ["critic.success_detection", "critic.retry_advice"],
                "safety": {"executable": False, "requires_guard": False},
            }),
            lambda m: MockCriticProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("mock_critic", ok=True)

    def _load_e_urdf(self) -> None:
        """Load robot e-URDF from file path or e-URDF Zoo."""
        if self.config.robot_model_path:
            from rosclaw.e_urdf.parser import EURDFParser
            self._e_urdf = EURDFParser(self.config.robot_model_path)
            print(f"[Runtime] Physical Grounding (e-URDF) loaded: {self.config.robot_model_path}")
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
            self.event_bus.publish(Event(
                topic="rosclaw.runtime.robot_loaded",
                payload={
                    "robot_id": robot_id,
                    "profile": self._robot_profile.to_dict(),
                },
                source="runtime",
                priority=EventPriority.HIGH,
            ))
            print(f"[Runtime] Robot '{robot_id}' loaded from e-URDF Zoo")
        except Exception as e:
            print(f"[Runtime] Failed to load robot from zoo: {e}")

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
            ProviderManifest.from_dict({
                "name": "robot_capabilities",
                "version": "1.0",
                "type": "robot",
                "capabilities": cap_names,
                "safety": {"executable": True, "requires_guard": True},
            }),
            lambda m: RobotCapabilityProvider(m),
            auto_load=False,
        )
        self._provider_registry.set_provider_health("robot_capabilities", ok=True)
        print(f"[Runtime] Registered {len(cap_names)} robot capabilities from e-URDF")

    def _on_provider_event(self, event: Event) -> None:
        """Handle provider_registered / provider_unregistered events."""
        print(f"[Runtime] Provider event: {event.topic} — {event.payload}")

    def _on_provider_health_changed(self, event: Event) -> None:
        """Handle provider_health_changed events."""
        payload = event.payload
        name = payload.get("provider", "unknown")
        ok = payload.get("ok", False)
        reason = payload.get("reason", "")
        status = "healthy" if ok else "unhealthy"
        print(f"[Runtime] Provider '{name}' is now {status} ({reason})")

    def _on_sandbox_episode_failed(self, event: Event) -> None:
        """Handle sandbox episode failure events."""
        print(f"[Runtime] Sandbox episode failed: {event.payload}")

    def _on_sandbox_action_blocked(self, event: Event) -> None:
        """Handle sandbox action blocked events."""
        print(f"[Runtime] Sandbox action blocked: {event.payload}")

    def _on_runtime_execution_failed(self, event: Event) -> None:
        """Handle runtime execution failure events."""
        print(f"[Runtime] Execution failed: {event.payload}")

    # ------------------------------------------------------------------
    # Physical World APIs (delegate to MemoryInterface / EmbodiedMemory)
    # ------------------------------------------------------------------

    def add_world_object(self, obj: Any) -> Optional[str]:
        """Add a world object. Requires EmbodiedMemory."""
        if self._memory is None:
            return None
        return self._memory.add_world_object(obj)

    def get_world_object(self, obj_id: str) -> Optional[Any]:
        """Get a world object by ID."""
        if self._memory is None:
            return None
        return self._memory.get_world_object(obj_id)

    def update_world_object_pose(self, obj_id: str, pose: Any, state: Optional[str] = None) -> bool:
        """Update world object pose and optional state."""
        if self._memory is None:
            return False
        return self._memory.update_world_object_pose(obj_id, pose, state)

    def search_world_objects(self, center: Any, radius: float, scene_id: Optional[str] = None) -> list[Any]:
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
    ) -> Optional[Any]:
        """Sync sensor detections with world model (Object Permanence)."""
        if self._memory is None:
            return None
        return self._memory.sync_scene_objects(scene_id, detections, timestamp_sec, occlusion_radius)

    def cognitive_search(
        self,
        query: str,
        spatial_center: Optional[Any] = None,
        spatial_radius: float = 2.0,
        temporal_interval: Optional[Any] = None,
        limit: int = 10,
    ) -> list[Any]:
        """Cognitive search: semantic + spatial + temporal."""
        if self._memory is None:
            return []
        return self._memory.cognitive_search(query, spatial_center, spatial_radius, temporal_interval, limit)

    def record_trajectory(self, content: str, waypoints: list[tuple[Any, float]]) -> Optional[int]:
        """Record a trajectory. Returns memory_id or None."""
        if self._memory is None:
            return None
        return self._memory.record_trajectory(content, waypoints)

    def search_similar_trajectories(
        self,
        query_waypoints: list[tuple[Any, float]],
        top_k: int = 5,
        max_dtw_distance: Optional[float] = None,
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

        Example:
            result = rt.capability_invoke(
                "vlm.object_grounding", {"image": "red_cup.jpg"}
            )
        """
        # Lazy init: if provider layer is importable but not initialized, set it up now
        if self._capability_router is None and ProviderRegistry is not None and CapabilityRouter is not None:
            try:
                self._provider_registry = ProviderRegistry(event_bus=self.event_bus)
                self._capability_router = CapabilityRouter(self._provider_registry)
                self._register_builtin_providers()
                print("[Runtime] Provider layer auto-initialized on first capability_invoke")
            except Exception as e:
                print(f"[Runtime] Provider auto-init failed: {e}")

        if self._capability_router is None:
            # Graceful fallback: return mock responses for known capability families
            if capability_name.startswith("vlm."):
                return {
                    "capability": capability_name,
                    "status": "ok",
                    "result": {"mock": True, "objects": [{"label": "mock_object", "confidence": 0.95}]},
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
                request_id=f"cap_{len(self.event_bus._event_history)}" if hasattr(self.event_bus, '_event_history') else "cap_0",
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
        """Execute an action through the sandbox or drivers.

        Publishes ``praxis.completed`` on success so Practice auto-records
        and Memory auto-ingests the experience.

        Returns execution result dict.
        """
        import time
        t0 = time.time()
        result: dict[str, Any]
        if self._sandbox is not None and hasattr(self._sandbox, "validate_trajectory"):
            try:
                trajectory = action.get("trajectory", [])
                validation = self._sandbox.validate_trajectory(trajectory)
                result = {"status": "ok", "result": validation, "source": "sandbox"}
            except Exception as e:
                result = {"status": "error", "error": str(e), "source": "sandbox"}
        else:
            # Fallback: try MCP drivers
            result = {"status": "error", "error": "No execution backend available"}
            for name, driver in self._mcp_drivers.items():
                if hasattr(driver, "execute"):
                    try:
                        drv_result = driver.execute(action)
                        result = {"status": "ok", "result": drv_result, "source": f"driver:{name}"}
                        break
                    except Exception as e:
                        result = {"status": "error", "error": str(e), "source": f"driver:{name}"}

        # Publish praxis event for Practice/Memory auto-ingest
        duration = time.time() - t0
        if result.get("status") == "ok":
            self.event_bus.publish(Event(
                topic="praxis.completed",
                payload={
                    "event_id": f"praxis_{int(time.time_ns())}",
                    "event_type": "success",
                    "correlation_id": action.get("request_id"),
                    "instruction": action.get("instruction", ""),
                    "initial_state": action.get("initial_state"),
                    "final_state": result,
                    "duration_sec": duration,
                },
                source="runtime",
                priority=EventPriority.NORMAL,
            ))
        else:
            self.event_bus.publish(Event(
                topic="rosclaw.runtime.execution.failed",
                payload={
                    "request_id": action.get("request_id"),
                    "error_type": result.get("error", "execution_failed"),
                    "action": action,
                    "duration_sec": duration,
                },
                source="runtime",
                priority=EventPriority.CRITICAL,
            ))
        return result

    def get_status(self) -> dict:
        """Get comprehensive runtime status."""
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
            },
            "embodied_memory": {
                "attached": self.config.embodied_memory is not None,
                "has_world_objects": (
                    self._memory.has_embodied_memory if self._memory else False
                ),
            },
            "drivers": list(self._mcp_drivers.keys()),
        }


class _HowProxy:
    """Sync wrapper for HeuristicEngine so async methods work from sync contexts."""

    def __init__(self, engine, run_async):
        self._engine = engine
        self._run_async = run_async

    def generate_recovery_hint(self, failure_type: str, context: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        """Sync wrapper around HeuristicEngine.generate_recovery_hint."""
        return self._run_async(self._engine.generate_recovery_hint(failure_type, context))

    def suggest_recovery(self, error_log: str, context: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        """Sync wrapper around HeuristicEngine.suggest_recovery."""
        return self._run_async(self._engine.suggest_recovery(error_log, context))

    def record_outcome(self, rule_id: str, success: bool) -> None:
        """Sync wrapper around HeuristicEngine.record_outcome."""
        return self._run_async(self._engine.record_outcome(rule_id, success))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)
