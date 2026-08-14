"""AgentService — the rosclaw-agentd application object (PR-NA-040).

Assembles MissionStore + ContextCompiler + ModelGateway + AgentLoop from
``AgentConfig``. Runs unprivileged; holds no hardware authority. Exposes a
small local HTTP/WebSocket-free JSON API for the CLI and the console.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Request as _Request
from pydantic import BaseModel as _BaseModel

from rosclaw.agentd.config import AgentConfig
from rosclaw.agentd.context.compiler import ContextCompiler
from rosclaw.agentd.context.prompt_registry import load_prompt
from rosclaw.agentd.context.sources import SourceBundle
from rosclaw.agentd.loop import AgentLoop, LoopTurnResult
from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelProbeResult,
    OpenAICompatGateway,
)
from rosclaw.agentd.runtime_sources import (
    CatalogCapabilitySource,
    ConfigConsentSource,
    DaemonSelfSource,
    EmptyMemorySource,
    ResolverBodySource,
    SimBodySource,
    SimSelfSource,
)
from rosclaw.agentd.tools import BuiltinToolRegistry
from rosclaw.agentd.usage import UsageRecorder
from rosclaw.contracts.agent.mission import (
    BodyBinding,
    Budgets,
    ExecutionMode,
    Goal,
    MissionSessionV1,
)
from rosclaw.contracts.common import ValidationError

AGENTD_DIR = "agentd"


class RegistryOrgSource:
    """L6 organization layer backed by the WorkerRegistry."""

    def __init__(self, registry) -> None:
        self._registry = registry

    def get_org(self):
        from rosclaw.agentd.context.sources import OrgFacts

        lines = []
        for card in self._registry.list():
            status = self._registry.status_of(card.worker_id) or "UNKNOWN"
            caps = ", ".join(c.name for c in card.capabilities)
            lines.append(f"- {card.worker_id} [{card.kind.value}/{status}] capabilities: {caps}")
        return OrgFacts(workers_summary="Registered workers:\n" + "\n".join(lines) if lines else "")


class BrokerConsentSource:
    """L7 consent layer bound to real grants (§5.5: grant 生效/撤销/过期
    触发重编译——public hash 变化即触发）。"""

    def __init__(self, base, conn) -> None:
        self._base = base
        self._conn = conn

    @property
    def policy_hash(self) -> str:
        return self._base.policy_hash

    def get_consent(self, mission_id: str):
        import json as _json
        from datetime import UTC, datetime

        from rosclaw.agentd.context.sources import ConsentFacts

        facts = self._base.get_consent(mission_id)
        row = self._conn.execute(
            "SELECT g.public_json FROM mission_grants AS g "
            "JOIN operator_requests AS r ON r.request_id = g.request_id "
            "WHERE g.revoked = 0 AND g.consumed = 0 AND g.expires_at > ? "
            "AND r.mission_id = ? ORDER BY g.created_at DESC LIMIT 1",
            (datetime.now(UTC).isoformat(), mission_id),
        ).fetchone()
        grant_hash = None
        scope_summary = facts.public_scope_summary
        if row is not None:
            public = _json.loads(row["public_json"])
            grant_hash = public.get("public_hash")
            scope = public.get("scope") if isinstance(public.get("scope"), dict) else {}
            scope_summary = (
                f"{scope_summary}\nActive mission grant: grant_id={public.get('grant_id')}; "
                f"tier={scope.get('tier')}; risk_ceiling={public.get('risk_ceiling')}; "
                f"public_hash={grant_hash}. Reference this exact public grant_id in "
                "REQUEST_ACTION. No private signature or permit is exposed."
            ).strip()
        return ConsentFacts(
            policy_hash=facts.policy_hash,
            mission_grant_public_hash=grant_hash,
            public_scope_summary=scope_summary,
            allowed_risk_tiers=facts.allowed_risk_tiers,
        )


class AgentService:
    def __init__(
        self,
        config: AgentConfig,
        rosclaw_home: Path,
        *,
        gateway: ModelGateway | None = None,
    ) -> None:
        self._config = config
        self._home = rosclaw_home
        db_dir = rosclaw_home / AGENTD_DIR
        db_dir.mkdir(parents=True, exist_ok=True)
        self._store = MissionStore(db_dir / "missions.db")
        # Worker registry first: the context compiler's org layer reads it.
        from rosclaw.agentd.handlers import ServiceIntentHandlers
        from rosclaw.agentd.workers import NativeWorkerAdapter, WorkerManager, WorkerRegistry

        self._body_id = config.active_body_id
        self._daemon_client = None
        daemon_socket = os.environ.get("ROSCLAW_DAEMON_SOCKET") or str(
            rosclaw_home / "run" / "rosclawd.sock"
        )
        if Path(daemon_socket).exists():
            from rosclaw.daemon.client import DaemonClient

            self._daemon_client = DaemonClient(socket_path=daemon_socket)
        self._registry = WorkerRegistry(self._store.connection)
        self._registry.register_builtins(actor_id=self.actor_id)
        self._simulation_body = self._body_id.startswith("sim/")
        if self._simulation_body:
            self._body_source = SimBodySource(self._body_id)
            self_source = SimSelfSource()
        else:
            self._body_source = ResolverBodySource(
                workspace=rosclaw_home,
                body_id=self._body_id,
            )
            self_source = DaemonSelfSource(self._daemon_client)
        body = self._body_source.get_body(self._body_id)
        self._tools = BuiltinToolRegistry(
            body_id=self._body_id,
            body_summary=body.summary if body else "configured body is unavailable",
        )
        # PR-05: Tool/Capability Catalog — descriptors, resolver, evidence.
        from rosclaw.agentd.tooling.artifact_result import ArtifactResultStore
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.catalog_registry import CatalogToolRegistry
        from rosclaw.agentd.tooling.mcp_adapter import McpCapabilityAdapter, McpServerConfig
        from rosclaw.agentd.tooling.native_tools import register_native_tools
        from rosclaw.agentd.tooling.resolver import ToolResolver

        self._tool_catalog = ToolCatalog()
        register_native_tools(self._tool_catalog, self._tools, simulation=self._simulation_body)
        self._artifact_store = ArtifactResultStore(db_dir)
        self._tool_resolver = ToolResolver(self._tool_catalog)
        self._tool_registry = CatalogToolRegistry(
            self._tool_catalog,
            self._tool_resolver,
            artifact_store=self._artifact_store,
            body_type=self._body_id,
        )
        for _srv in self._config.mcp_servers:
            if "env" in _srv or "api_key" in _srv or "token" in _srv:
                raise ValidationError(
                    "mcp_servers 不允许内联 env/api_key/token 值——"
                    "凭据只用 env_refs（环境变量名引用）（与 config.yaml 禁令一致）"
                )
        # 七审 PR-SEVEN-1：第一方 Robot Kit 原子激活——活跃 body 匹配
        # kit 的 body_instance_template 且用户未配置同名 server/未禁用
        # 时自动注入（package-relative 模块入口；不再有"有 body identity
        # 但动作能力为 0"的假就绪）。
        from rosclaw.sim.robot_kit import kit_for_body, kit_server_spec

        mcp_servers = list(self._config.mcp_servers)
        self._active_kit = None
        disabled_kits = set(self._config.raw.get("kits", {}).get("disabled", []) or [])
        kit = kit_for_body(self._body_id)
        if kit is not None and kit.kit_id not in disabled_kits:
            server_name = kit.executor_identity.removeprefix("mcp:")
            if not any(s.get("name") == server_name for s in mcp_servers):
                mcp_servers.append(kit_server_spec(kit))
            self._active_kit = kit
        self._mcp_adapters = [
            McpCapabilityAdapter(
                McpServerConfig(
                    name=str(s.get("name", "")),
                    command=str(s.get("command", "")),
                    args=tuple(str(a) for a in s.get("args", []) or []),
                    env_refs=tuple(str(r) for r in s.get("env_refs", []) or []),
                    observation_tools=tuple(s.get("observation_tools", []) or ()),
                    action_tools=tuple(s.get("action_tools", []) or ()),
                    compute_tools=tuple(s.get("compute_tools", []) or ()),
                    supported_modes=tuple(s.get("supported_modes", ("SIMULATION",)) or ()),
                    required_body_types=tuple(s.get("required_body_types", []) or ()),
                    effect_domain=str(s.get("effect_domain", "") or ""),
                    timeout_ms=int(s.get("timeout_ms", 5000)),
                ),
                self._tool_catalog,
            )
            for s in mcp_servers
            if s.get("name") and s.get("command")
        ]
        self._mcp_discovered = False
        # 验收轮根因修复：discovery 互斥锁——flag-first 模式（先置
        # True 再发现）会让并发调用方在发现进行中提前返回，拿到空
        # 能力目录（context hash 随后翻转 → CONTEXT_HASH_MISMATCH）。
        self._discovery_lock = asyncio.Lock()
        # 批次 B：命令注册表（命令永不进入模型上下文）。
        from rosclaw.agentd.ui.command_service import CommandService
        from rosclaw.agentd.ui.interaction_service import InteractionService

        self._commands = CommandService(self)
        self._interactions = InteractionService()
        #: /scoped-models 快捷切换集合（内存态）。
        self._scoped_models: set[str] = set()
        # 批次 E：导出/导入/设置。
        from rosclaw.agentd.ui.export_service import ExportService
        from rosclaw.agentd.ui.import_service import ImportService
        from rosclaw.agentd.ui.settings_service import SettingsService

        self._exporter = ExportService(self)
        self._importer = ImportService(self)
        self._settings = SettingsService(self._home / "config.yaml")
        from rosclaw.agentd.ui.branch_service import BranchService

        self._branches = BranchService(self)
        consent_source = ConfigConsentSource()
        from rosclaw.operator import OperatorBroker

        self._broker = OperatorBroker(
            self._store.connection, policy_hash=consent_source.policy_hash
        )
        self._compiler = ContextCompiler(
            SourceBundle(
                constitution_text=load_prompt("native_agent_v1.md").text,
                body=self._body_source,
                self_source=self_source,
                capabilities=CatalogCapabilitySource(
                    self._tool_catalog,
                    home=rosclaw_home,
                    body_id=self._body_id,
                ),
                memory=EmptyMemorySource(),
                organization=RegistryOrgSource(self._registry),
                consent=BrokerConsentSource(consent_source, self._store.connection),
                runtime_status_summary=(
                    "agentd local; simulated body; rosclawd optional"
                    if self._simulation_body
                    else "agentd bound to a real body through rosclawd; physical evidence is daemon-owned"
                ),
            ),
            max_input_tokens=config.max_input_tokens,
            dynamic_tool_limit=config.dynamic_tool_limit,
        )
        if gateway is not None:
            self._gateway: ModelGateway = gateway
        else:
            from rosclaw.agentd.models.failover import FailoverGateway

            policy = config.to_policy()
            chain = policy.fallback_chain()
            if config.model_backend == "modeld":
                # 批次 D：AgentLoop 不再直接接触 OpenAI-compatible 协议细节。
                from rosclaw.agentd.models.modeld_gateway import ModeldGateway

                candidates = [(p, ModeldGateway(p, home=self._home)) for p in chain]
            else:
                candidates = [(p, OpenAICompatGateway(p)) for p in chain]
            # 单 profile 也走 FailoverGateway：统一的 cooldown/RPM 语义。
            self._gateway = FailoverGateway(candidates)
        self._prompt = load_prompt("native_agent_v1.md")
        self._loops: dict[str, AgentLoop] = {}
        self._lock = asyncio.Lock()
        self._usage = UsageRecorder(self._store.connection)
        # AgentEventV2 journal + live bus (PR-02).
        from rosclaw.agentd.events import AgentEventStore

        self._events = AgentEventStore(self._store.connection)
        self._turn_tasks: dict[str, asyncio.Task] = {}
        # 十审 W0：WorkOrder 后台驱动任务——delegate 立即返回后由它们
        # 驱动 run_to_completion；close() 时统一取消（DB 终态权威不变：
        # 未完成的单由 sweeper/重启对账处理，绝不永久假装 RUNNING 健康）。
        self._worker_bg_tasks: dict[str, asyncio.Task] = {}
        from rosclaw.agentd.runner import MissionRunner

        self._runner = MissionRunner(self)
        # External harness packs (PR-WF-054): register cards by probe result
        # (missing binary → DISABLED with T0 note, never fake readiness).
        # 同步探活（init 可能在 async 上下文中被构造，不能 run_until_complete）。
        from rosclaw.agentd.pi_entry import find_pi_agent_entry
        from rosclaw.agentd.workers.external import ExternalHarnessAdapter
        from rosclaw.agentd.workers.packs import ALL_PACKS, card_for_pack
        from rosclaw.agentd.workers.pi_managed import PiManagedAdapter

        external_adapter = ExternalHarnessAdapter(cwd=rosclaw_home)
        for pack in ALL_PACKS:
            card = card_for_pack(pack)
            self._registry.register(card, actor_id=self.actor_id)
            ready, detail = self._probe_pack_sync(
                pack.executable, pack.min_version, pack.install_hint
            )
            if not ready:
                self._registry.set_status(
                    pack.worker_id, "DISABLED", actor_id=self.actor_id, reason=detail
                )
        self._worker_manager = WorkerManager(
            self._store.connection,
            adapters={
                "native_inproc": NativeWorkerAdapter(self._gateway),
                "external_cli": external_adapter,
                # 十审 W1：内置 Pi headless Worker（与主 Agent 同一模型配置）。
                "pi_managed": PiManagedAdapter(
                    rosclaw_home=rosclaw_home, conn=self._store.connection
                ),
            },
            actor_id=self.actor_id,
            event_recorder=self._record_worker_event,
        )
        # 十一审 PR-E：pi_managed 的 WAITING_INPUT 状态迁移需要 manager。
        self._worker_manager._adapters["pi_managed"]._manager_ref = self._worker_manager
        # 十四审 PR-14.2：RetryCoordinator 是唯一重试决策者——自动/手动
        # retry 同一 CAS 仲裁，一个 root job 一张卡、一个活跃 attempt。
        from rosclaw.agentd.workers.retry import RetryCoordinator

        def _candidates(worker_hint: str, capability: str):
            from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

            return PiToolDispatcher(self)._candidates_for(worker_hint, capability)

        self._retry_coordinator = RetryCoordinator(
            self._store.connection,
            manager=self._worker_manager,
            candidates_fn=_candidates,
            spawn_fn=self.spawn_worker_driver,
        )
        # 内置 Pi Worker 的就绪性取决于 node+dist——不可用时诚实 DISABLED
        # （绝不"看起来装了就 ENABLED"）。
        if find_pi_agent_entry() is None:
            self._registry.set_status(
                "worker:rosclaw:pi",
                "DISABLED",
                actor_id=self.actor_id,
                reason="rosclaw-agent dist 或 Node ≥22.19 不可用",
            )
        self._handlers = ServiceIntentHandlers(
            registry=self._registry,
            manager=self._worker_manager,
            actor_id=self.actor_id,
            broker=self._broker,
            body_id=self._body_id,
            body_hash=body.effective_body_hash if body else "",
            mode=config.default_mode,
        )
        # 批次 B/PR-12：handlers 的事件（grant.consumed、receipt.received 等）
        # 接入 AgentEventV2 journal。
        self._handlers.set_event_sink(self._event_sink_for)
        # Daemon action channel (K3) + consent channel (ADR-0007): only when
        # a rosclawd client is actually available — otherwise both degrade
        # honestly.
        self._action_channel = None
        self._consent_channel = None
        if self._daemon_client is not None:
            from rosclaw.agentd.action_channel import DaemonActionChannel
            from rosclaw.agentd.consent_channel import DaemonConsentChannel

            self._action_channel = DaemonActionChannel(
                self._daemon_client,
                actor_id=self.actor_id,
                body_id=self._body_id,
                body_hash=body.effective_body_hash if body else "",
            )
            self._handlers._action_channel = self._action_channel
            self._consent_channel = DaemonConsentChannel(
                self._daemon_client,
                actor_id=self.actor_id,
                body_id=self._body_id,
                body_hash=body.effective_body_hash if body else "",
            )
            self._handlers._consent_channel = self._consent_channel
        # PR-12：SIM 物理权威（无 daemon 时）。mcp_servers[] 带
        # sim_executor: true 的 server 同时提供 SIM actuation。
        # 六审 §6.2.4：executor 按 (body, capability source) 路由——
        # 每个 sim_executor server 一个通道，按 source 名索引；不再有
        # 执行任意物理动作的全局通道。
        self._sim_executors: dict[str, object] = {}
        for sim_server in (s for s in mcp_servers if s.get("sim_executor")):
            if self._daemon_client is not None:
                break
            from rosclaw.agentd.sim_executor import SimActionChannel
            from rosclaw.agentd.tooling.persistent_client import PersistentMcpClient

            # 观测与 SIM 执行共享同一 server 进程（有状态身体）。
            shared_client = PersistentMcpClient(
                command=str(sim_server.get("command", "")),
                args=tuple(str(a) for a in sim_server.get("args", []) or []),
            )
            self._shared_mcp_client = shared_client
            server_name = str(sim_server.get("name", "sim"))
            self._sim_executors[f"mcp:{server_name}"] = SimActionChannel(
                command=str(sim_server.get("command", "")),
                args=tuple(str(a) for a in sim_server.get("args", []) or []),
                name=server_name,
                client=shared_client,
            )
            for adapter in self._mcp_adapters:
                if adapter.source == f"mcp:{server_name}":
                    adapter._client = shared_client
        self._handlers._sim_executors = self._sim_executors
        # Team Fabric: enabled via config `team.enabled`. Local coordinator
        # in P0 (local_sim); ROS 2/Zenoh transports are later PRs.
        team_cfg = (config.raw.get("team") or {}) if config.raw else {}
        if team_cfg.get("enabled"):
            from rosclaw.team import TeamCoordinator

            self._team_coordinator = TeamCoordinator(
                self._store.connection,
                team_id=str(team_cfg.get("team_id", "default_team")),
                actor_id=self.actor_id,
                policy_hash=consent_source.policy_hash,
            )
            self._handlers._team_coordinator = self._team_coordinator
        else:
            self._team_coordinator = None

    # ------------------------------------------------------------------
    @staticmethod
    def _probe_pack_sync(executable: str, min_version: str, install_hint: str) -> tuple[bool, str]:
        """Synchronous pack probe (used at init; adapter.probe is the async path)."""
        import shutil
        import subprocess

        from rosclaw.agentd.workers.packs import version_ok

        exe = shutil.which(executable)
        if exe is None:
            return False, f"二进制 {executable!r} 未找到（T0 Discovered）。{install_hint}"
        try:
            out = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                timeout=15,
                text=True,
            )
            version_text = (out.stdout or out.stderr).strip().split()[0]
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"version probe failed: {exc}"
        if not version_ok(version_text, min_version):
            return False, f"{version_text} < 最小兼容版本 {min_version}，请升级。"
        return True, version_text

    # ------------------------------------------------------------------
    @property
    def store(self) -> MissionStore:
        return self._store

    @property
    def actor_id(self) -> str:
        safe = self._body_id.replace("/", "_")
        return f"agent:rosclaw-native:{safe}"

    def _loop_for(self, mission_id: str) -> AgentLoop:
        loop = self._loops.get(mission_id)
        if loop is None:
            loop = AgentLoop(
                store=self._store,
                compiler=self._compiler,
                gateway=self._gateway,
                prompt=self._prompt,
                tools=self._tool_registry,
                handlers=self._handlers,
                actor_id=self.actor_id,
                max_tool_rounds=self._config.max_tool_rounds,
                usage_recorder=self._usage,
                event_sink=self._event_sink_for(mission_id),
                decision_protocol=self._config.decision_protocol,
                legacy_fenced_json_fallback=self._config.legacy_fenced_json_fallback,
            )
            self._loops[mission_id] = loop
        return loop

    def _event_sink_for(self, mission_id: str):
        """Per-mission event sink closure (PR-02)."""
        from rosclaw.contracts.agent.agent_event import Visibility

        async def sink(type, payload, *, visibility=None, task_id=None) -> None:
            await self._events.append(
                mission_id,
                type,
                payload,
                visibility=visibility or Visibility.USER,
                task_id=task_id,
            )

        return sink

    # ------------------------------------------------------------------
    # /v2 event-streaming surface (PR-02): turn submit decoupled from SSE.
    # ------------------------------------------------------------------
    async def submit_turn_v2(self, mission_id: str, text: str) -> str:
        """202-style submit via MissionRunner (wake-tracked, PR-03)."""
        return await self._runner.submit_turn(mission_id, text)

    def events_replay(self, mission_id: str, *, after_sequence: int = 0, limit: int = 1000):
        return self._events.replay(mission_id, after_sequence=after_sequence, limit=limit)

    def events_subscribe(self, mission_id: str) -> asyncio.Queue:
        return self._events.bus.subscribe(mission_id)

    def events_unsubscribe(self, mission_id: str, queue: asyncio.Queue) -> None:
        self._events.bus.unsubscribe(mission_id, queue)

    async def cancel_turn_v2(self, mission_id: str) -> None:
        await self.cancel(mission_id)
        task = self._turn_tasks.get(mission_id)
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    def create_mission(
        self,
        goal_text: str,
        *,
        mode: str | None = None,
        owner_principal: str | None = None,
    ) -> MissionSessionV1:
        if owner_principal is None:
            # 默认主体来自真实本地 uid——与 operator.sock 的 SO_PEERCRED
            # 身份一致（CI 等非 1000 uid 环境下 EXACT_ACTION verify 才能
            # 通过 principal 校验）。
            owner_principal = f"user:local:{os.getuid()}"
        requested_mode = ExecutionMode(mode or self._config.default_mode)
        body = self._body_source.get_body(self._body_id)
        if body is None:
            raise ValidationError(
                f"body {self._body_id!r} is not linked, hash-valid, and available"
            )
        if requested_mode is not ExecutionMode.SIMULATION:
            gaps: list[str] = []
            if requested_mode is ExecutionMode.REAL and self._config.physical_action_count <= 0:
                gaps.append("agent.budgets.physical_action_count must be greater than zero")
            if self._simulation_body:
                gaps.append("configured body is simulated, not a live BodyResolver body")
            if self._daemon_client is None:
                gaps.append("rosclawd socket is unavailable")
            else:
                try:
                    status = self._daemon_client.get_runtime_status()
                except Exception as exc:  # noqa: BLE001 - report an honest prerequisite gap
                    gaps.append(f"rosclawd status unavailable: {exc}")
                else:
                    if not status.get("running"):
                        gaps.append("rosclawd is not running")
                    if status.get("robot_id") != self._body_id:
                        gaps.append(
                            f"rosclawd robot_id {status.get('robot_id')!r} != {self._body_id!r}"
                        )
                    pack = status.get("robot_pack") or {}
                    if not pack.get("loaded") or pack.get("signature_status") != "valid":
                        gaps.append("verified Robot Pack is not loaded")
                    suffix = f":{requested_mode.value}"
                    if not any(
                        str(item).endswith(suffix)
                        for item in status.get("registered_executors") or []
                    ):
                        gaps.append(f"no {requested_mode.value} executor is registered")
            if gaps:
                raise ValidationError(
                    f"mode {requested_mode.value} requested but prerequisites are missing: "
                    + "; ".join(gaps)
                )
        return self._store.create_mission(
            owner_principal=owner_principal,
            goal=Goal(text=goal_text, language=self._config.language),
            body_binding=BodyBinding(
                body_id=body.body_id, effective_body_hash=body.effective_body_hash
            ),
            mode=requested_mode,
            budgets=Budgets(physical_action_count=self._config.physical_action_count),
            actor_id=self.actor_id,
        )

    def list_missions(self) -> list[MissionSessionV1]:
        return self._store.list_missions()

    def get_mission(self, mission_id: str) -> MissionSessionV1 | None:
        return self._store.get_mission(mission_id)

    async def send_turn(self, mission_id: str, text: str, on_text_delta=None) -> LoopTurnResult:
        mission = self._store.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        if self.mission_archived(mission_id):
            raise ValidationError(
                f"mission {mission_id!r} is archived (read-only); create a new mission"
            )
        await self._ensure_mcp_discovered()
        async with self._runner.lock_for(mission_id):
            if self._handlers is not None:
                self._handlers._mode = mission.mode.value
                self._handlers._principal = mission.owner_principal
            loop = self._loop_for(mission_id)
            # R4：同步路径也写用户可见事件——transcript projection 才能
            # 覆盖用户消息与助手回复（与 v2 runner 同一事件词汇）。
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._events.append(
                mission_id, AgentEventType.TURN_ACCEPTED, {"text": text[:500]}
            )
            result = await loop.run_user_turn(
                mission, text, now=datetime.now(UTC), on_text_delta=on_text_delta
            )
            reply = getattr(result, "reply", "") or ""
            if reply:
                await self._events.append(
                    mission_id, AgentEventType.MODEL_TEXT_DELTA, {"text": reply}
                )
                await self._events.append(mission_id, AgentEventType.MESSAGE_ENDED, {})
            return result

    def mission_usage(self, mission_id: str) -> dict:
        return self._usage.mission_totals(mission_id)

    def usage_report(self, mission_id: str) -> dict:
        """八审 §4 P0-8：面向 /tokens 的用量报告——provider 请求数、
        token 分项、工具调用计数、provider 延迟与端到端跨度分离。
        聚合实时计算，不存可变计数器。"""
        totals = self._usage.mission_totals(mission_id)
        rows = self._usage.rows(mission_id)
        latencies = sorted(int(r.get("latency_ms") or 0) for r in rows)
        latency: dict[str, int | None] = {"p50": None, "p95": None}
        if latencies:
            import math

            latency["p50"] = latencies[
                max(0, min(len(latencies) - 1, math.ceil(0.50 * len(latencies)) - 1))
            ]
            latency["p95"] = latencies[
                max(0, min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1))
            ]
        wall_span_ms: int | None = None
        if len(rows) >= 2:
            from datetime import datetime

            first = datetime.fromisoformat(rows[0]["recorded_at"])
            last = datetime.fromisoformat(rows[-1]["recorded_at"])
            wall_span_ms = int((last - first).total_seconds() * 1000)
        tool_events = self._store.connection.execute(
            "SELECT type, COUNT(*) AS n FROM agent_events "
            "WHERE mission_id = ? AND type IN ('tool.proposed', 'tool.completed') "
            "GROUP BY type",
            (mission_id,),
        ).fetchall()
        tool_counts = {t: int(n) for t, n in tool_events}
        return {
            **totals,
            "provider_latency_ms": latency,
            "wall_span_ms": wall_span_ms,
            "tool_calls": {
                "proposed": tool_counts.get("tool.proposed", 0),
                "completed": tool_counts.get("tool.completed", 0),
            },
        }

    async def robot_kit_status(self) -> dict:
        """七审 PR-SEVEN-1.4：kit 完整性状态——identity/capabilities/
        executor/policy/probes 要么全 READY 要么 BROKEN（不再"有
        identity 没动作"的假就绪）。"""
        await self._ensure_mcp_discovered()
        kit = getattr(self, "_active_kit", None)
        if kit is None:
            from rosclaw.sim.robot_kit import kit_for_body

            candidate = kit_for_body(self._body_id)
            return {
                "kit_id": "",
                "state": "BROKEN",
                "reason": f"no first-party kit for body {self._body_id}",
                "action_capability_count": 0,
                "executor": "MISSING",
                # 七审 PR-SEVEN-5：结构化 remediation——模型/TUI 不再只能
                # 说"请重新绑定 profile"。
                "remediation": (
                    {
                        "kind": "enable_robot_kit",
                        "kit_id": candidate.kit_id,
                        "idempotent": True,
                        "cancellable": True,
                        "real_authorization": False,
                        "command": f"/robot repair {candidate.kit_id}",
                    }
                    if candidate
                    else None
                ),
            }
        server_name = kit.executor_identity.removeprefix("mcp:")
        action_count = sum(
            1
            for tool_id in kit.action_tools
            if self._tool_catalog.get(tool_id) is not None
            and self._tool_catalog.quarantine_reason(tool_id) is None
        )
        executor_ready = self._sim_executors.get(kit.executor_identity) is not None
        ready = action_count == len(kit.action_tools) and executor_ready
        result = {
            "kit_id": kit.kit_id,
            "display_name": kit.display_name,
            "state": "READY" if ready else "BROKEN",
            "action_capability_count": action_count,
            "expected_action_count": len(kit.action_tools),
            "observation_capability_count": sum(
                1
                for tool_id in kit.observation_tools
                if self._tool_catalog.get(tool_id) is not None
            ),
            "executor": "READY" if executor_ready else "MISSING",
            "executor_identity": kit.executor_identity,
            "approval_policy": kit.approval_policy,
            "server": server_name,
        }
        if not ready:
            result["remediation"] = {
                "kind": "retry_kit_activation",
                "kit_id": kit.kit_id,
                "idempotent": True,
                "cancellable": True,
                "real_authorization": False,
                "command": f"/robot repair {kit.kit_id}",
            }
        return result

    async def action_blockers(self) -> list[dict[str, str]]:
        """七审 §2.1/PR-SEVEN-2.4：动作受阻原因全量聚合（按可操作性
        排序）——Header 显示最可操作的 1-2 个，/status 显示全量。"""
        blockers: list[dict[str, str]] = []
        await self._ensure_mcp_discovered()
        kit_status = await self.robot_kit_status()
        if kit_status["state"] != "READY":
            blockers.append(
                {
                    "code": "ROBOT_KIT_INCOMPLETE",
                    "detail": kit_status.get("reason")
                    or f"动作能力 {kit_status.get('action_capability_count', 0)}"
                    f"/executor {kit_status.get('executor')}",
                }
            )
        return blockers

    # -- 七审 PR-SEVEN-5：Robot-first UX 与自修复 ----------------------------

    @staticmethod
    def _make_mcp_adapter(spec: dict, catalog) -> object:
        """server spec → McpCapabilityAdapter（__init__ 与 repair 共用）。"""
        from rosclaw.agentd.tooling.mcp_adapter import (
            McpCapabilityAdapter,
            McpServerConfig,
        )

        return McpCapabilityAdapter(
            McpServerConfig(
                name=str(spec.get("name", "")),
                command=str(spec.get("command", "")),
                args=tuple(str(a) for a in spec.get("args", []) or []),
                env_refs=tuple(str(r) for r in spec.get("env_refs", []) or ()),
                observation_tools=tuple(spec.get("observation_tools", []) or ()),
                action_tools=tuple(spec.get("action_tools", []) or ()),
                compute_tools=tuple(spec.get("compute_tools", []) or ()),
                supported_modes=tuple(spec.get("supported_modes", ("SIMULATION",)) or ()),
                required_body_types=tuple(spec.get("required_body_types", []) or ()),
                effect_domain=str(spec.get("effect_domain", "") or ""),
                timeout_ms=int(spec.get("timeout_ms", 5000)),
            ),
            catalog,
        )

    async def _activate_kit_runtime(self, kit) -> None:
        """激活/重激活 kit：adapter + SIM executor + 重发现。幂等——
        已存在的 adapter/executor 不重复创建。"""
        from rosclaw.sim.robot_kit import kit_server_spec

        spec = kit_server_spec(kit)
        server_name = str(spec["name"])
        if not any(a.source == kit.executor_identity for a in self._mcp_adapters):
            self._mcp_adapters.append(self._make_mcp_adapter(spec, self._tool_catalog))
        if self._daemon_client is None and kit.executor_identity not in self._sim_executors:
            from rosclaw.agentd.sim_executor import SimActionChannel
            from rosclaw.agentd.tooling.persistent_client import PersistentMcpClient

            shared = PersistentMcpClient(
                command=str(spec["command"]), args=tuple(spec["args"])
            )
            self._shared_mcp_client = shared
            self._sim_executors[kit.executor_identity] = SimActionChannel(
                command=str(spec["command"]),
                args=tuple(spec["args"]),
                name=server_name,
                client=shared,
            )
            for adapter in self._mcp_adapters:
                if adapter.source == kit.executor_identity:
                    adapter._client = shared
        self._active_kit = kit
        self._mcp_discovered = False
        await self._ensure_mcp_discovered()

    async def robot_list(self) -> dict:
        """PR-SEVEN-5：第一方 kit 清单 + 活跃/状态标记（/robots）。"""
        from rosclaw.sim.robot_kit import load_first_party_kits

        disabled = set(self._config.raw.get("kits", {}).get("disabled", []) or [])
        active_status = await self.robot_kit_status()
        kits = []
        for kit in load_first_party_kits():
            matches_body = kit.body_instance_template == self._body_id
            is_active = matches_body and self._active_kit is not None
            if is_active:
                state = active_status.get("state", "BROKEN")
            elif kit.kit_id in disabled:
                state = "DISABLED"
            else:
                state = "AVAILABLE"
            kits.append(
                {
                    "kit_id": kit.kit_id,
                    "display_name": kit.display_name,
                    "robot_type": kit.robot_type,
                    "body_instance_template": kit.body_instance_template,
                    "mode": kit.mode,
                    "active": is_active,
                    "state": state,
                }
            )
        return {"ok": True, "kits": kits, "active_body_id": self._body_id}

    def robot_resolve(self, query: str) -> dict:
        """PR-SEVEN-5：自然语言 → kit 候选。唯一候选自动选；多候选
        由调用方交互选；无匹配诚实空。"""
        from rosclaw.sim.robot_kit import match_kits

        def _card(kit) -> dict:
            return {
                "kit_id": kit.kit_id,
                "display_name": kit.display_name,
                "robot_type": kit.robot_type,
                "body_instance_template": kit.body_instance_template,
                "mode": kit.mode,
            }

        candidates = match_kits(query)
        return {
            "ok": True,
            "candidates": [_card(k) for k in candidates],
            "selected": _card(candidates[0]) if len(candidates) == 1 else None,
        }

    async def doctor_task(self, goal: str) -> dict:
        """PR-SEVEN-5：task readiness——"画五角星"需要 trajectory +
        executor + verifier。MISSING 时给结构化 remediation（幂等、
        可取消、绝不自动完成 REAL 授权）。"""
        from rosclaw.sim.robot_kit import kit_for_body, required_groups_for_goal

        required = required_groups_for_goal(goal)
        await self._ensure_mcp_discovered()
        kit = self._active_kit
        status = await self.robot_kit_status()
        missing: list[str] = []
        if kit is None or status.get("state") != "READY":
            missing = list(required)
        else:

            def _usable(tool_id: str) -> bool:
                return (
                    self._tool_catalog.get(tool_id) is not None
                    and self._tool_catalog.quarantine_reason(tool_id) is None
                )

            checks = {
                "trajectory": any(
                    "plan" in t for t in kit.compute_tools if _usable(t)
                ),
                "verifier": any(
                    "verify" in t for t in (*kit.compute_tools, *kit.observation_tools)
                    if _usable(t)
                ),
                "executor": status.get("executor") == "READY",
            }
            missing = [name for name in required if not checks.get(name, False)]
        remediation = None
        if missing:
            target = kit or kit_for_body(self._body_id)
            remediation = {
                "kind": "enable_robot_kit",
                "kit_id": target.kit_id if target else "",
                "idempotent": True,
                "cancellable": True,
                "real_authorization": False,
                "command": (
                    f"/robot repair {target.kit_id}" if target else ""
                ),
            }
        return {
            "ok": True,
            "goal": goal,
            "required": required,
            "missing": missing,
            "state": "READY" if not missing else "MISSING",
            "remediation": remediation,
        }

    async def robot_repair(self, kit_id: str = "") -> dict:
        """PR-SEVEN-5：一键修复——取消禁用（持久化）+ 清隔离 + 重建
        adapter/executor + 重发现。幂等；不触碰任何 REAL 授权。"""
        import logging

        from rosclaw.sim.robot_kit import kit_for_body, load_first_party_kits

        kits = {k.kit_id: k for k in load_first_party_kits()}
        kit = kits.get(kit_id) if kit_id else (self._active_kit or kit_for_body(self._body_id))
        if kit is None:
            return {
                "ok": False,
                "error": f"unknown robot kit {kit_id!r}",
                "code": "KIT_UNKNOWN",
            }
        if kit.body_instance_template != self._body_id:
            return {
                "ok": False,
                "error": f"kit {kit.kit_id} binds {kit.body_instance_template}, "
                f"active body is {self._body_id}",
                "code": "BODY_MISMATCH",
            }
        disabled = list(self._config.raw.get("kits", {}).get("disabled", []) or [])
        persisted = True
        if kit.kit_id in disabled:
            disabled.remove(kit.kit_id)
            self._config.raw.setdefault("kits", {})["disabled"] = disabled
            try:
                self._settings.set_key("kits.disabled", disabled)
            except Exception:  # noqa: BLE001 — 内存态仍生效，持久化失败诚实上报
                persisted = False
        self._tool_catalog.lift_source_quarantine(kit.executor_identity)
        await self._activate_kit_runtime(kit)
        status = await self.robot_kit_status()
        logging.getLogger("rosclaw.agentd.robot").info(
            "robot.repair kit=%s state=%s persisted=%s",
            kit.kit_id,
            status.get("state"),
            persisted,
        )
        return {
            "ok": status.get("state") == "READY",
            "robot_kit": status,
            "persisted": persisted,
        }

    async def robot_use(self, body_id: str) -> dict:
        """PR-SEVEN-5：切换活跃机器人。同 body 幂等；跨 body 仅允许
        有第一方 SIM kit 的目标且 developer 剖面——v1 不做热切换
        （context/lease 语义 fail-closed），持久化后重启生效；无 kit
        的 body（含 REAL 真机）一律拒绝，绝不自动完成真机授权。"""
        from rosclaw.sim.robot_kit import kit_for_body

        body_id = body_id.strip()
        if not body_id:
            return {"ok": False, "error": "body_id required", "code": "INVALID_ARGUMENT"}
        if body_id == self._body_id:
            return {"ok": True, "body_id": body_id, "changed": False}
        kit = kit_for_body(body_id)
        if kit is None:
            return {
                "ok": False,
                "error": f"no first-party kit for body {body_id!r} — REAL/未知本体"
                "需要完整的安装/授权工作流，robot use 绝不自动完成",
                "code": "BODY_UNKNOWN",
            }
        if kit.mode != "SIMULATION" or self.authorization_profile() != "DEV_SIM_ONLY":
            return {
                "ok": False,
                "error": "robot use 仅允许 SIM kit + developer 剖面",
                "code": "MODE_FORBIDDEN",
            }
        self._settings.set_key("agent.body_id", body_id)
        return {
            "ok": True,
            "body_id": body_id,
            "changed": True,
            "restart_required": True,
        }

    def sim_executor_identity_for(self, source: str) -> str:
        """六审 §6.2.4：按 capability source 解析 SIM 执行通道身份。
        身份即路由目标（mcp:<server>/native:agentd）；通道缺失时由
        execute fail closed（EXECUTOR_FOR_BODY_UNAVAILABLE）。"""
        return source

    async def _ensure_mcp_discovered(self) -> None:
        """Discover configured MCP servers once (PR-05); failures quarantine
        the source honestly and never block the turn.

        验收轮根因：并发调用方不得在发现进行中提前返回（否则拿到空
        目录的 context hash 会在发现完成后翻转）——互斥锁 + 完成后
        才置 flag。"""
        if self._mcp_discovered:
            return
        async with self._discovery_lock:
            if self._mcp_discovered:
                return
            for adapter in self._mcp_adapters:
                try:
                    await adapter.discover()
                except Exception:  # noqa: BLE001 - discovery must never break chat
                    self._tool_catalog.quarantine_source(
                        adapter.source, "discovery_crashed"
                    )
            self._mcp_discovered = True

    @property
    def tool_catalog(self):
        return self._tool_catalog

    @property
    def tool_resolver(self):
        return self._tool_resolver

    # -- modeld 管理面（批次 D：/providers /model /login /logout） ---------------
    def _modeld_mgmt(self):
        """共享的 modeld 管理通道（懒启动；runtime 缺失 → None）。"""
        if getattr(self, "_modeld_mgmt_instance", None) is not None:
            return self._modeld_mgmt_instance
        from rosclaw.agentd.models.modeld_gateway import ModeldGateway, _find_modeld_runtime

        if _find_modeld_runtime() is None:
            return None
        profile = (
            self._config.to_policy().default
            if self._config.profiles
            else getattr(self._gateway, "profile", None)
        )
        if profile is None:
            return None
        self._modeld_mgmt_instance = ModeldGateway(profile, home=self._home)
        return self._modeld_mgmt_instance

    async def modeld_providers(self) -> dict:
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"available": False, "providers": [], "error": "modeld runtime unavailable"}
        try:
            data = await mgmt.manage("GET", "/v1/providers")
            data["available"] = True
            return data
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "providers": [], "error": str(exc)}

    async def modeld_models(self, provider: str) -> dict:
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"models": [], "error": "modeld runtime unavailable"}
        return await mgmt.manage("GET", f"/v1/models?provider={provider}")

    async def modeld_login(self, provider: str, api_key: str) -> dict:
        """API key 登录：secret 只经内存进 modeld，不落 mission journal。"""
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"ok": False, "error": "modeld runtime unavailable"}
        return await mgmt.manage(
            "POST", f"/v1/auth/{provider}/login", {"mode": "api_key", "api_key": api_key}
        )

    async def modeld_logout(self, provider: str) -> dict:
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"ok": False, "error": "modeld runtime unavailable"}
        return await mgmt.manage("POST", f"/v1/auth/{provider}/logout", {})

    def current_model_label(self) -> str:
        profile = self._gateway.profile
        return f"{profile.provider}/{profile.model}（profile: {profile.name}）"

    def switch_model(self, provider: str, model: str) -> dict:
        """切换默认 profile 的 provider/model（内存态；持久化走 /settings）。

        切换永不改变工具权限、Mission mode 或 grant（§8.5）。
        """
        if not self._config.profiles:
            return {"ok": False, "error_code": "no_profiles", "message": "未配置模型"}
        profile = self._config.to_policy().default
        old = f"{profile.provider}/{profile.model}"
        if self._config.model_backend != "modeld":
            return {
                "ok": False,
                "error_code": "legacy_backend",
                "message": (
                    "legacy backend 的运行时在启动时绑定 endpoint，/model 暂不支持热切换；"
                    "请设置 models.backend: modeld 后重试（Kimi 现有配置无需改动）"
                ),
            }
        # modeld provider 名映射与 ModeldGateway 一致。
        from rosclaw.agentd.models.modeld_gateway import _PROVIDER_MAP

        mapped = _PROVIDER_MAP.get(provider, provider)
        # profile 对象被 config / FailoverGateway candidates / 既有 AgentLoop
        # 共享；frozen dataclass 的就地字段替换让"下一 turn 生效"在所有
        # 引用点同时成立（当前 turn 不中断）。
        object.__setattr__(profile, "provider", mapped)
        object.__setattr__(profile, "model", model)
        return {
            "ok": True,
            "message": (
                f"模型已从 {old} 切换为 {mapped}/{model}（下一 turn 生效；"
                "当前 turn 不中断；持久化配置修改将在 /settings 提供）"
            ),
        }

    def _record_worker_event(self, mission_id: str, to_status: str, payload: dict) -> None:
        """Sync bridge: WorkerManager transitions → AgentEventV2 (批次 B)."""
        import asyncio as _asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        event_type = {
            "CLAIMED": AgentEventType.WORKER_CLAIMED,
            "RUNNING": AgentEventType.WORKER_STARTED,
            "SUBMITTED": AgentEventType.WORKER_SUBMITTED,
            "VERIFYING": AgentEventType.WORKER_VERIFYING,
            "ACCEPTED": AgentEventType.WORKER_ACCEPTED,
            "FAILED": AgentEventType.WORKER_FAILED,
            "EXPIRED": AgentEventType.WORKER_EXPIRED,
        }.get(to_status)
        if event_type is None:
            return
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (e.g. sync CLI path) — worker_events table is the record
        loop.create_task(self._events.append(mission_id, event_type, payload))

    @property
    def commands(self):
        return self._commands

    @property
    def interactions(self):
        return self._interactions

    @property
    def scoped_models(self) -> set[str]:
        return self._scoped_models

    @property
    def exporter(self):
        return self._exporter

    @property
    def importer(self):
        return self._importer

    @property
    def settings(self):
        return self._settings

    @property
    def branches(self):
        return self._branches

    def reload_domains(self, domains: list[str]) -> dict:
        """/reload（§8.15）：分域原子重载；安全域永远拒绝。

        可 reload：prompts（prompt registry）、workers（pack 重新探活注册）、
        models（modeld provider catalog refresh 提示）。
        不可 reload：rosclawd Policy、Robot Pack 签名、Body 安全边界、
        Permit、设备权限、REAL 风险上限。
        """
        results: dict[str, dict] = {}
        for domain in domains:
            if domain == "prompts":
                from rosclaw.agentd.context.prompt_registry import load_prompt

                try:
                    self._prompt = load_prompt("native_agent_v1.md")
                    results[domain] = {
                        "ok": True,
                        "detail": f"prompt v{self._prompt.version} hash={self._prompt.content_hash[:16]}（活跃 turn 不换 prompt，下一 turn 生效）",
                    }
                except Exception as exc:  # noqa: BLE001
                    results[domain] = {"ok": False, "detail": f"{exc}（保持旧配置）"}
            elif domain == "workers":
                from rosclaw.agentd.workers.packs import ALL_PACKS, card_for_pack

                refreshed = []
                for pack in ALL_PACKS:
                    try:
                        self._registry.register(card_for_pack(pack), actor_id=self.actor_id)
                        refreshed.append(pack.worker_id)
                    except Exception as exc:  # noqa: BLE001
                        results.setdefault(domain, {"ok": False, "detail": str(exc)})
                else:
                    results[domain] = {"ok": True, "detail": f"re-registered {len(refreshed)} packs"}
            elif domain == "models":
                results[domain] = {
                    "ok": True,
                    "detail": "modeld provider catalog 为启动时构建；/model 可切换，重启后重建。",
                }
            elif domain in ("policy", "robot_pack", "body", "permits", "permissions", "safety"):
                results[domain] = {
                    "ok": False,
                    "detail": f"{domain} 属安全域，/reload 永不修改（走专用管理面）",
                }
            else:
                results[domain] = {"ok": False, "detail": f"未知 reload 域 {domain!r}"}
        return results

    def conversation(self, mission_id: str) -> list[dict]:
        return self._store.conversation(mission_id)

    # ------------------------------------------------------------------
    async def compact(
        self,
        mission_id: str,
        *,
        instructions: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """`/compact` 的服务端实现（PR-07）。"""
        mission = self._store.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        async with self._runner.lock_for(mission_id):
            loop = self._loop_for(mission_id)
            return await loop.compact_conversation(
                mission, reason="manual", focus=instructions, dry_run=dry_run
            )

    def compaction_status(self, mission_id: str) -> dict:
        from rosclaw.agentd.context.compaction import (
            CompactionStore,
            estimate_messages_tokens,
        )

        store = CompactionStore(self._store.connection)
        entries = store.list(mission_id)
        return {
            "compactions": len(entries),
            "last": entries[-1].model_dump(mode="json") if entries else None,
            "current_view_tokens": estimate_messages_tokens(self._store.conversation(mission_id)),
            "journal_events": len(self._store.events(mission_id)),
        }

    async def cancel(self, mission_id: str) -> None:
        from rosclaw.contracts.agent.agent_event import AgentEventType

        await self._events.append(mission_id, AgentEventType.TURN_CANCEL_REQUESTED, {})
        loop = self._loops.get(mission_id)
        if loop is not None:
            loop.request_cancel()

    # ------------------------------------------------------------------
    # 批次 B：UI 控制面（命令/快照/归档/重命名）
    # ------------------------------------------------------------------
    def turn_in_flight(self, mission_id: str) -> bool:
        task = self._turn_tasks.get(mission_id)
        return task is not None and not task.done()

    def rename_mission(self, mission_id: str, name: str) -> None:
        if self._store.get_mission(mission_id) is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        self._store.set_mission_meta(mission_id, display_name=name)
        import asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._events.append(
                    mission_id, AgentEventType.MISSION_RENAMED, {"name": name[:120]}
                )
            )
        except RuntimeError:
            pass

    def archive_mission(self, mission_id: str) -> None:
        if self._store.get_mission(mission_id) is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        self._store.set_mission_meta(mission_id, archived=True)
        import asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._events.append(mission_id, AgentEventType.MISSION_ARCHIVED, {})
            )
        except RuntimeError:
            pass

    def mission_archived(self, mission_id: str) -> bool:
        return bool(self._store.mission_meta(mission_id)["archived"])

    def status_snapshot(self, mission_id: str | None = None) -> dict:
        data: dict = {
            "agent": "rosclaw-agentd",
            "body_id": self._body_id,
            "daemon_connected": self._daemon_client is not None,
            "mcp_servers": [a.source for a in self._mcp_adapters],
            "model_profile": self._gateway.profile.name,
            "model": self._gateway.profile.model,
            "tools_registered": len(self._tool_catalog.list()),
        }
        if mission_id:
            mission = self._store.get_mission(mission_id)
            if mission is not None:
                data["mission"] = {
                    "state": mission.state.value,
                    "mode": mission.mode.value,
                    "pending_approvals": len(self.pending_approvals(mission_id)),
                    "turn_in_flight": self.turn_in_flight(mission_id),
                    "archived": self.mission_archived(mission_id),
                }
        return data

    def snapshot(self, mission_id: str):
        """MissionSnapshotV1 — 重连校准用权威快照（批次 B §5.3）。"""
        from rosclaw.contracts.ui.snapshots import MissionSnapshotV1

        mission = self._store.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        from rosclaw.agentd.context.compaction import CompactionStore
        from rosclaw.agentd.mission.store import _utcnow

        meta = self._store.mission_meta(mission_id)
        grants = [
            {
                "grant_id": g.get("grant_id"),
                "tier": g.get("tier"),
                "risk_ceiling": g.get("risk_ceiling"),
                "expires_at": g.get("expires_at"),
            }
            for g in self.list_grants()
            if not g.get("revoked") and not g.get("consumed")
        ]
        orders = [
            {
                "work_order_id": o.work_order_id,
                "status": o.status,
                "assigned_to": o.assigned_to,
            }
            for o in self._worker_manager.orders_for_mission(mission_id)
            if o.status not in ("ACCEPTED", "REJECTED", "EXPIRED", "CANCELLED")
        ]
        return MissionSnapshotV1(
            mission_id=mission_id,
            name=meta["display_name"] or mission.goal.text[:60],
            goal_text=mission.goal.text,
            state=mission.state.value,
            mode=mission.mode.value,
            body_id=mission.body_binding.body_id,
            context_id=f"ctx_{mission_id}",
            context_revision=mission.context_revision,
            task_graph_revision=mission.task_graph_revision,
            last_event_sequence=self._events.latest_sequence(mission_id),
            turn_in_flight=self.turn_in_flight(mission_id),
            pending_approvals=[
                {
                    "request_id": r.request_id,
                    "title": r.action_display.title,
                    "risk_tier": r.action_display.risk_tier,
                    "expires_at": r.expires_at,
                }
                for r in self.pending_approvals(mission_id)
            ],
            active_grants=grants,
            open_work_orders=orders,
            usage=self._usage.mission_totals(mission_id),
            budgets=mission.budgets.model_dump(mode="json"),
            compaction_count=CompactionStore(self._store.connection).count(mission_id),
            tool_count=len(self._tool_catalog.list()),
            captured_at=_utcnow(),
        )

    # ------------------------------------------------------------------
    # approvals (Operator Broker surface for CLI/console)
    # ------------------------------------------------------------------
    def pending_approvals(self, mission_id: str | None = None):
        return self._broker.pending_requests(mission_id)

    async def decide_approval(
        self,
        request_id: str,
        *,
        principal: str,
        approve: bool,
        _from_operatord: bool = False,
    ):
        """认知层裁决（审计 P0-01：仅 operatord 路径可调用）。

        - agentd 只创建 proposal、读取公开结果——daemon proposal 的
          decision 属 rosclaw-operatord + rosclawd ACL，agentd 代码里
          没有任何 proposal.decide 路径；
        - REAL/daemon 卡必须已由 operatord 完成 daemon 侧决定
          （apply_decision 已先行校验）。
        """
        if not _from_operatord:
            raise ValidationError(
                "approvals are decided by rosclaw-operatord only (P0-01); "
                "agentd does not serve a decision path"
            )
        grant = self._broker.decide(request_id, principal=principal, approve=approve)
        approval_req = self._broker.get_request(request_id)
        if approval_req is not None:
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._events.append(
                approval_req.mission_id,
                AgentEventType.APPROVAL_DECIDED,
                {
                    "request_id": request_id,
                    "approved": approve,
                    "grant_id": grant.grant_id if grant else None,
                },
            )
            self._runner.notify_approval_decided(
                approval_req.mission_id,
                request_id,
                approved=approve,
                grant_id=grant.grant_id if grant else None,
            )
        # 审计 P0-01：agentd 不再裁决 daemon proposal（该路径已迁往
        # rosclaw-operatord）；daemon 卡的物理决定由 operatord 直接经
        # rosclawd ACL 完成。
        return grant

    def authorization_profile(self) -> str:
        """当前授权剖面（审计 P0-01.6）：同 UID 一体运行只能 DEV_SIM_ONLY。"""
        if self._daemon_client is not None:
            return "OPERATORD_BACKED"
        from rosclaw.operatord import DEV_SIM_ONLY_LABEL

        return DEV_SIM_ONLY_LABEL

    async def daemon_proposal_is_decided(self, request_id: str) -> bool:
        """REAL 卡确认：该 broker 请求关联的 daemon proposal 已被决定
        （由 operatord 经 rosclawd ACL 完成）。无 daemon/无关联 → False。"""
        if self._consent_channel is None:
            return False
        request = self._broker.get_request(request_id)
        proposal_id = getattr(request, "daemon_proposal_id", None) or (
            request.model_dump(mode="json").get("daemon_proposal_id") if request else None
        )
        if not proposal_id:
            return False
        try:
            proposal = await self._consent_channel.proposal(proposal_id)
        except Exception:  # noqa: BLE001 - 读不到即未决定（fail closed）
            return False
        return proposal.get("state") in ("SUBMITTED", "TERMINAL", "DECLINED")

    async def daemon_identity(self) -> dict | None:
        """daemon 签名公钥（验证 DecisionReceiptV1）；进程内缓存。

        信任锚：daemon socket 的本机 UID/组隔离；公钥本身无需保密。
        """
        if self._daemon_client is None:
            return None
        cached = getattr(self, "_daemon_identity_cache", None)
        if cached is not None:
            return cached
        try:
            identity = await asyncio.to_thread(self._daemon_client.daemon_identity)
        except Exception:  # noqa: BLE001 - 取不到即无法验证（fail closed）
            return None
        self._daemon_identity_cache = identity
        return identity

    async def verify_decision_receipt(self, receipt_data: dict, card) -> tuple[bool, str]:
        """R3/P0-6：只接受 daemon 签名有效、decision=ACCEPT、所有字段与
        本地卡片精确相等、未过期、未重放的 DecisionReceiptV1。

        返回 (ok, error)；ok=False 时 error 为人类可读原因。
        DECLINE receipt 用 ``verify_decline_receipt``（同样验证签名与
        字段，但只用于关闭请求，绝不生成 grant）。
        """
        from rosclaw.agentd.operator_socket import display_hash_for
        from rosclaw.contracts.operator.decision import DecisionReceiptV1

        try:
            receipt = DecisionReceiptV1.from_dict(receipt_data)
        except (ValueError, KeyError, TypeError) as exc:
            return False, f"invalid receipt: {exc}"
        identity = await self.daemon_identity()
        if identity is None:
            return False, "daemon identity unavailable — cannot verify receipt"
        if receipt.daemon_key_id != str(identity.get("daemon_key_id", "")):
            return False, "receipt signed by an unknown daemon key"
        if receipt.daemon_instance_id != str(identity.get("daemon_instance_id", "")):
            return False, "receipt belongs to a different daemon generation"
        if not receipt.verify_signature(str(identity.get("public_key_pem", ""))):
            return False, "receipt signature invalid"
        mismatches: list[str] = []
        if receipt.proposal_id != (card.daemon_proposal_id or ""):
            mismatches.append("proposal_id")
        if receipt.agent_request_id != card.request_id:
            mismatches.append("agent_request_id")
        if receipt.mission_id != card.mission_id:
            mismatches.append("mission_id")
        if receipt.execution_mode != card.mode:
            mismatches.append("execution_mode")
        if receipt.capability_id != (card.daemon_capability_id or ""):
            mismatches.append("capability_id")
        if receipt.canonical_args_hash != (card.daemon_action_intent_hash or ""):
            mismatches.append("canonical_args_hash")
        if receipt.display_hash != display_hash_for(card):
            mismatches.append("display_hash")
        if mismatches:
            return False, "receipt fields do not match the local card: " + ", ".join(mismatches)
        try:
            expires = datetime.fromisoformat(receipt.expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False, "receipt expires_at is not ISO-8601"
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        from rosclaw.kernel.contracts import utc_now as _utc_now

        if _utc_now() >= expires:
            return False, "receipt expired"
        if not self._record_decision_receipt(receipt):
            return False, "receipt replay — this challenge nonce was already applied"
        return True, ""

    def _record_decision_receipt(self, receipt) -> bool:
        """持久化 receipt（challenge_nonce UNIQUE = 跨重启重放防线）。"""
        import json as _json
        import sqlite3 as _sqlite3
        from datetime import UTC as _UTC
        from datetime import datetime as _datetime

        try:
            self._store.connection.execute(
                "INSERT INTO decision_receipts "
                "(receipt_id, proposal_id, challenge_nonce, decision, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_id,
                    receipt.proposal_id,
                    receipt.challenge_nonce,
                    receipt.decision,
                    _json.dumps(receipt.to_dict(), ensure_ascii=False),
                    _datetime.now(_UTC).isoformat(),
                ),
            )
            self._store.connection.commit()
        except _sqlite3.IntegrityError:
            return False
        return True

    def verify_operatord_signature(
        self,
        *,
        enrollment_id: str,
        public_key_pem: str,
        payload: bytes,
        signature_b64: str,
    ) -> tuple[bool, str]:
        """SIM（DEV_SIM_ONLY）剖面：TOFU 钉住 operatord 公钥并验签。

        首次见到 enrollment_id 时钉住公钥；此后公钥变化即拒绝
        （可能的冒充/重装）。信任边界如实标注：同机同 UID 场景这只是
        完整性检查，不是独立身份（DEV_SIM_ONLY）。
        """
        from datetime import UTC as _UTC
        from datetime import datetime as _datetime

        from rosclaw.contracts.operator.decision import verify_b64

        row = self._store.connection.execute(
            "SELECT public_key_pem FROM operatord_keys WHERE enrollment_id = ?",
            (enrollment_id,),
        ).fetchone()
        if row is None:
            self._store.connection.execute(
                "INSERT INTO operatord_keys (enrollment_id, public_key_pem, first_seen_at) "
                "VALUES (?, ?, ?)",
                (enrollment_id, public_key_pem, _datetime.now(_UTC).isoformat()),
            )
            self._store.connection.commit()
            pinned = public_key_pem
        else:
            pinned = str(row["public_key_pem"])
            if pinned != public_key_pem:
                return False, (
                    "operatord public key changed for this enrollment — "
                    "possible impersonation or reinstall; refusing"
                )
        if not verify_b64(pinned, payload, signature_b64):
            return False, "operatord signature invalid"
        return True, ""

    def list_grants(self):
        rows = self._store.connection.execute(
            "SELECT public_json, revoked, consumed, expires_at FROM mission_grants "
            "ORDER BY created_at DESC"
        ).fetchall()
        import json as _json

        from rosclaw.contracts.operator.grant import MissionGrantV1

        grants = []
        for row in rows:
            grant = MissionGrantV1(**_json.loads(row["public_json"]))
            grants.append(
                {
                    "grant_id": grant.grant_id,
                    "principal": grant.principal,
                    "mode": grant.mode,
                    "tier": grant.scope.tier,
                    "risk_ceiling": grant.risk_ceiling,
                    "revoked": bool(row["revoked"]),
                    "consumed": bool(row["consumed"]),
                    "expires_at": row["expires_at"],
                    "public_hash": grant.public_hash,
                }
            )
        return grants

    def principal_for_request(self, request_id: str) -> str:
        """approval request 所属 mission 的 owner principal。"""
        req = self._broker.get_request(request_id)
        if req is not None:
            mission = self._store.get_mission(req.mission_id)
            if mission is not None:
                return mission.owner_principal
        return f"user:local:{os.getuid()}"

    def revoke_grant(self, grant_id: str, *, principal: str) -> None:
        self._broker.revoke(grant_id, principal=principal)
        import asyncio as _asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._append_global_event(AgentEventType.GRANT_REVOKED, {"grant_id": grant_id})
        )

    async def _append_global_event(self, event_type, payload: dict) -> None:
        """Grant events span missions; journal against the owning mission when
        resolvable, otherwise against the first mission that referenced it."""
        row = self._store.connection.execute(
            "SELECT r.mission_id FROM mission_grants g "
            "JOIN operator_requests r ON r.request_id = g.request_id "
            "WHERE g.grant_id = ?",
            (payload.get("grant_id"),),
        ).fetchone()
        mission_id = row["mission_id"] if row else ""
        if mission_id:
            await self._events.append(mission_id, event_type, payload)

    # ------------------------------------------------------------------
    async def probe(self) -> ModelProbeResult:
        try:
            return await self._gateway.probe()
        except ModelGatewayError as exc:
            return ModelProbeResult(reachable=False, error=f"{exc.kind}: {exc}")

    def status(self) -> dict:
        profile = self._gateway.profile
        return {
            "agent_enabled": self._config.enabled,
            "default_mode": self._config.default_mode,
            "body_id": self._body_id,
            "daemon_connected": self._daemon_client is not None,
            "profile": profile.name,
            "provider": profile.provider,
            "model": profile.model,
            "base_url": profile.base_url,
            "api_key_ref": profile.api_key_ref,
            "missions": len(self._store.list_missions()),
            "maturity": "experimental",
        }

    async def estop(self, reason: str, *, principal: str) -> dict:
        """PR-11：直达 rosclawd 的急停路径，绕过模型。

        无 daemon 时诚实报不可用——绝不假装已停（§12.5/§14.2）。
        """
        if self._daemon_client is None:
            raise ValidationError(
                "estop unavailable: rosclawd not connected; nothing was stopped (honest)"
            )
        return await asyncio.to_thread(
            self._daemon_client.emergency_stop,
            reason,
            source=f"operator:{principal}",
        )

    async def start_operator_socket(self, socket_path: Path | None = None) -> Path:
        """PR-11：operator.sock（peer identity + display hash + estop）。幂等。"""
        from rosclaw.agentd.operator_socket import OperatorSocketServer

        existing = getattr(self, "_operator_socket", None)
        if existing is not None:
            return existing._path
        path = socket_path or (self._home / "run" / "operator.sock")
        self._operator_socket = OperatorSocketServer(self, path)
        await self._operator_socket.start()
        return path

    async def start_pi_bridge(self, socket_path: Path | None = None) -> Path:
        """PR-PNA-1：pi-bridge.sock（SessionBinding + writer lease + 状态投影）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        existing = getattr(self, "_pi_bridge", None)
        if existing is not None:
            return existing._path
        path = socket_path or (self._home / "run" / "pi-bridge.sock")
        self._pi_bridge = PiBridgeServer(self, path)
        await self._pi_bridge.start()
        return path

    async def reconcile_workers_on_start(self) -> list[str]:
        """十审 W4：agentd 重启对账——所有非终态 WorkOrder 都是孤儿
        （内存 run 注册表已随进程消失）：

        - 有 child.pid 的 pi worker 子进程：整组杀掉（不留孤儿继续计费）；
        - RUNNING → FAILED（agentd_restart）；OFFERED/CLAIMED → CANCELLED；
        - 绝不假装它们仍在健康运行（resume 后 /jobs 显示真实状态）。
        """
        import signal as _signal

        reconciled: list[str] = []
        conn = self._store.connection
        rows = conn.execute(
            "SELECT work_order_id, status FROM work_orders "
            "WHERE status NOT IN ('ACCEPTED', 'FAILED', 'EXPIRED', 'CANCELLED')"
        ).fetchall()
        for row in rows:
            wo_id = row["work_order_id"]
            pid_file = self._home / "work" / wo_id / "child.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    import contextlib as _cl

                    with _cl.suppress(ProcessLookupError, PermissionError):
                        os.killpg(os.getpgid(pid), _signal.SIGTERM)
                        await asyncio.sleep(0.5)
                        os.killpg(os.getpgid(pid), _signal.SIGKILL)
                except (ValueError, OSError):
                    pass
                pid_file.unlink(missing_ok=True)
            try:
                if row["status"] == "RUNNING":
                    self._worker_manager._transition(wo_id, "FAILED", "agentd_restart")
                else:
                    self._worker_manager._transition(wo_id, "CANCELLED", "agentd_restart")
                reconciled.append(wo_id)
            except Exception:  # noqa: BLE001 - 对账继续
                pass
        return reconciled

    async def _drive_worker(self, order) -> None:
        """基础设施错误自动重试至多一次（复用 worktree/workspace，不从零
        再花 token）。十四审 PR-14.2：重试只能有一个所有者——
        RetryCoordinator（总纲 §3.5）：
        - 只认结构化可重试 cause（PROVIDER_TRANSIENT/WORKER_CRASH/
          EVENT_PIPE_BROKEN）；"worker exited" 进程表象永不是依据；
        - 自动/手动 retry 同一 CAS——绝不裂变成三张任务卡；
        - USER_CANCELLED/USER_PAUSED/语义失败不自动重试。"""
        result, _report = await self._worker_manager.run_to_completion(order)
        if result.status != "FAILED" or order.inputs.get("_auto_retried"):
            return
        from rosclaw.agentd.workers.retry import parse_cause

        cause = parse_cause(result.summary)
        if cause is None:
            return
        await self._retry_coordinator.request_retry(
            order, cause=cause, actor="auto", note=result.summary[:120]
        )

    def spawn_worker_driver(self, order) -> None:
        """十审 W0：WorkOrder 后台驱动——pi 工具请求栈立即返回后由本
        任务驱动 run_to_completion 到终态；请求断开不影响权威状态。"""
        task = asyncio.create_task(self._drive_worker(order))
        self._worker_bg_tasks[order.work_order_id] = task

        def _done(t: asyncio.Task, wo_id: str = order.work_order_id) -> None:
            self._worker_bg_tasks.pop(wo_id, None)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:  # 防御：manager 已兜底——这里只记录。
                import logging

                logging.getLogger("rosclaw.agentd.workers").warning(
                    "worker driver for %s raised: %s", wo_id, exc
                )

        task.add_done_callback(_done)

    async def close(self) -> None:
        # 十审 W0：先杀活动 Worker 的底层进程树（adapter 级），再取消
        # 后台驱动任务——顺序反过来会让驱动先死、进程泄漏。
        import contextlib as _cl

        with _cl.suppress(Exception):
            await self._worker_manager.shutdown()
        for task in list(getattr(self, "_worker_bg_tasks", {}).values()):
            task.cancel()
        if getattr(self, "_worker_bg_tasks", None):
            with _cl.suppress(Exception):
                await asyncio.gather(*self._worker_bg_tasks.values(), return_exceptions=True)
            self._worker_bg_tasks.clear()
        # 六审 §7：产品 supervisor 管理的 operatord 随 service 终止。
        managed = getattr(self, "_managed_operator", None)
        if managed is not None:
            managed.terminate()
            self._managed_operator = None
        if getattr(self, "_shared_mcp_client", None) is not None:
            await self._shared_mcp_client.close()
            self._shared_mcp_client = None
        if getattr(self, "_operator_socket", None) is not None:
            await self._operator_socket.stop()
            self._operator_socket = None
        if getattr(self, "_pi_bridge", None) is not None:
            await self._pi_bridge.stop()
            self._pi_bridge = None
        await self._gateway.close()
        self._store.close()
        # P1-4：control token 文件随服务关闭删除（不残留可重用的令牌）。
        token_path = self._home / "run" / "agentd-control.token"
        import contextlib as _contextlib

        with _contextlib.suppress(OSError):
            token_path.unlink(missing_ok=True)

    # -- 生命周期与控制台鉴权（二次复核 P1-3/P1-4） -----------------------------

    @classmethod
    def open(cls, config: AgentConfig, rosclaw_home: Path, **kwargs):
        """统一生命周期（P1-3）：``async with AgentService.open(...) as svc``
        ——正常退出、异常、Ctrl-C 都保证 close（子进程/socket/句柄不残留）。"""
        import contextlib as _contextlib

        @_contextlib.asynccontextmanager
        async def _ctx():
            service = cls(config, rosclaw_home, **kwargs)
            try:
                yield service
            finally:
                await service.close()

        return _ctx()

    @property
    def control_token(self) -> str:
        """每次启动生成的高熵 ephemeral 控制 token（P1-4）。"""
        token = getattr(self, "_control_token", None)
        if token is None:
            import secrets as _secrets

            token = _secrets.token_urlsafe(32)
            self._control_token = token
        return token

    def write_control_token_file(self) -> Path:
        """把 control token 写入 0600 临时文件（TUI/CLI 同 UID 读取；
        不写 journal、不进命令行参数）。"""
        run_dir = self._home / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(run_dir, 0o700)
        path = run_dir / "agentd-control.token"
        tmp = run_dir / ".agentd-control.token.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, self.control_token.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp, path)
        return path

    # -- transcript projection（二次复核 R4/P1-1） -------------------------------

    def transcript(
        self,
        mission_id: str,
        *,
        before_seq: int | None = None,
        limit: int = 500,
    ) -> dict:
        """journal → transcript 块（稳定 ID + before_seq 分页）。

        TUI 不再从底层事件猜聊天记录；journal 仍是唯一权威。
        """
        from rosclaw.agentd.transcript import project_transcript

        limit = max(1, min(int(limit), 2000))
        events = self._events.replay(
            mission_id,
            before_sequence=before_seq,
            limit=limit + 1,
        )
        has_more = len(events) > limit
        events = events[:limit]
        return {
            "mission_id": mission_id,
            "blocks": project_transcript(events),
            "latest_sequence": self._events.latest_sequence(mission_id),
            "oldest_sequence": events[0].sequence if events else 0,
            "has_more": has_more,
        }


# ----------------------------------------------------------------------
# Local HTTP API (console + CLI clients). No domain state lives in the
# HTTP layer; everything delegates to AgentService.
# ----------------------------------------------------------------------
class MissionCreate(_BaseModel):
    goal: str
    mode: str | None = None


class TurnCreate(_BaseModel):
    text: str


class DecisionCreate(_BaseModel):
    approve: bool
    principal: str | None = None  # 缺省按 mission owner 解析（不再硬编码 uid）


class CommandRequestCreate(_BaseModel):
    request_id: str
    idempotency_key: str
    command_name: str
    arguments: dict = {}


class InteractionRespond(_BaseModel):
    value: object = None
    idempotency_key: str = ""


def _turn_payload(result) -> dict:
    return {
        "mission_id": result.mission_id,
        "reply": result.reply,
        "state": result.state.value,
        "tool_rounds": result.tool_rounds,
        "model_turns": result.model_turns,
        "tokens_used": result.tokens_used,
        "degraded": result.degraded,
    }


def create_app(service: AgentService):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    @asynccontextmanager
    async def lifespan(_app):
        # PR-11：operator.sock 随 HTTP 服务启动（同一事件循环）。
        # P1-3：HTTP 服务停止时必须关闭 service（子进程/socket/句柄）。
        # P1-4：ephemeral control token 落 0600 文件供同机 TUI/CLI。
        # 十审 W4：先做 Worker 崩溃对账（孤儿进程清理 + 诚实终态）。
        await service.reconcile_workers_on_start()
        await service.start_operator_socket()
        await service.start_pi_bridge()
        service.write_control_token_file()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="rosclaw-agentd", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def csrf_origin_guard(request: _Request, call_next):
        """PR-11 §14.1：浏览器 Console 的 CSRF/Origin 防线。

        - 携带 Origin 的变更请求只允许本机来源（localhost/127.0.0.1/
          [::1]）——任意网页不得向 localhost approval API 发请求（§19.6）。
        - 设置 ROSCLAW_CONSOLE_TOKEN 时，变更请求必须带 X-Rosclaw-Token
          （一次性 pairing token 语义）。
        """
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if origin:
                from urllib.parse import urlparse

                host = urlparse(origin).hostname or ""
                if host not in ("localhost", "127.0.0.1", "::1"):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": f"origin {host!r} rejected (CSRF guard)"},
                    )
            pairing = os.environ.get("ROSCLAW_CONSOLE_TOKEN")
            if pairing and request.headers.get("x-rosclaw-token") != pairing:
                return JSONResponse(
                    status_code=403, content={"detail": "pairing token required"}
                )
        # P1-4：ephemeral control token——除 /health 与 /console（HTML
        # 壳）外全部端点（含敏感 GET）都必须携带。token 经 0600 文件
        # 交给同机 TUI/CLI，不写 journal、不进 ps。
        public_paths = ("/health", "/console")
        if (
            request.url.path not in public_paths
            and not request.url.path.startswith("/static/")
            and request.headers.get("x-rosclaw-token") != service.control_token
        ):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "control token required (P1-4): read "
                        "run/agentd-control.token (0600) on this host"
                    )
                },
            )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "rosclaw-agentd"}

    @app.get("/status")
    async def status() -> dict:
        return service.status()

    @app.get("/probe")
    async def probe() -> dict:
        result = await service.probe()
        return {
            "reachable": result.reachable,
            "models_visible": list(result.models_visible),
            "expected_model_present": result.expected_model_present,
            "chat_ok": result.chat_ok,
            "tool_call_ok": result.tool_call_ok,
            "error": result.error,
        }

    @app.post("/missions", status_code=201)
    async def create_mission(payload: MissionCreate) -> dict:
        try:
            mission = service.create_mission(payload.goal, mode=payload.mode)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return mission.model_dump(mode="json")

    @app.get("/missions")
    async def list_missions() -> list[dict]:
        return [m.model_dump(mode="json") for m in service.list_missions()]

    @app.get("/missions/{mission_id}")
    async def get_mission(mission_id: str) -> dict:
        mission = service.get_mission(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return mission.model_dump(mode="json")

    @app.post("/missions/{mission_id}/turns")
    async def send_turn(mission_id: str, payload: TurnCreate) -> dict:
        try:
            result = await service.send_turn(mission_id, payload.text)
        except ValidationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _turn_payload(result)

    # ------------------------------------------------------------------
    # /v2 (PR-02): submit decoupled from event stream; TUI may disconnect
    # freely — the mission keeps running server-side.
    # ------------------------------------------------------------------
    @app.post("/v2/missions/{mission_id}/turns", status_code=202)
    async def v2_submit_turn(mission_id: str, payload: TurnCreate) -> dict:
        try:
            turn_id = await service.submit_turn_v2(mission_id, payload.text)
        except ValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"turn_id": turn_id, "mission_id": mission_id, "accepted": True}

    @app.get("/v2/missions/{mission_id}/transcript")
    async def v2_transcript(
        mission_id: str,
        before_seq: int | None = None,
        limit: int = 500,
    ) -> dict:
        """R4/P1-1：transcript projection（稳定块 ID + before_seq 分页）。
        响应含 latest_sequence——SSE 应从其续接（after_sequence），
        恢复 exactly-once。"""
        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return service.transcript(mission_id, before_seq=before_seq, limit=limit)

    @app.get("/v2/missions/{mission_id}/events")
    async def v2_events(
        mission_id: str,
        request: _Request,
        after_sequence: int = 0,
        visibility: str | None = None,
        follow: bool = True,
    ):
        """Long-lived SSE: replay journal from after_sequence, then live.
        ``follow=false`` returns after replay (bounded read).
        批次 B：支持标准 Last-Event-ID 头（断线恢复优先于 query 参数）。"""
        import json as _json

        from fastapi.responses import StreamingResponse

        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")

        last_event_id = request.headers.get("last-event-id")
        if last_event_id and last_event_id.isdigit():
            after_sequence = max(after_sequence, int(last_event_id))

        def frame(event) -> str:
            return f"id: {event.sequence}\ndata: {_json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"

        async def stream():
            for event in service.events_replay(mission_id, after_sequence=after_sequence):
                if visibility and event.visibility.value != visibility:
                    continue
                yield frame(event)
            if not follow:
                return
            queue = service.events_subscribe(mission_id)
            try:
                while True:
                    event = await queue.get()
                    if visibility and event.visibility.value != visibility:
                        continue
                    yield frame(event)
            finally:
                service.events_unsubscribe(mission_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v2/missions/{mission_id}/cancel")
    async def v2_cancel(mission_id: str) -> dict:
        await service.cancel_turn_v2(mission_id)
        return {"cancelled": True}

    # ------------------------------------------------------------------
    # 批次 B：capabilities / commands / snapshot / interactions
    # ------------------------------------------------------------------
    @app.get("/v1/capabilities")
    async def v1_capabilities(mission_id: str | None = None) -> dict:
        """服务端命令注册表（含 disabled_reason）。"""
        state = None
        in_flight = False
        if mission_id:
            mission = service.get_mission(mission_id)
            if mission is None:
                raise HTTPException(status_code=404, detail="mission not found")
            state = mission.state.value
            in_flight = service.turn_in_flight(mission_id)
        specs = service.commands.specs(mission_state=state, turn_in_flight=in_flight)
        return {
            "commands": [s.model_dump(mode="json") for s in specs],
            "event_stream": {"url": "/v2/missions/{mission_id}/events", "resume": "Last-Event-ID"},
            "snapshot_url": "/v1/missions/{mission_id}/snapshot",
        }

    @app.post("/v1/missions/{mission_id}/commands")
    async def v1_command(mission_id: str, payload: CommandRequestCreate) -> dict:
        from rosclaw.contracts.ui.commands import CommandRequestV1

        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        request = CommandRequestV1(
            request_id=payload.request_id,
            idempotency_key=payload.idempotency_key,
            command_name=payload.command_name.lstrip("/"),
            arguments=payload.arguments,
            mission_id=mission_id,
        )
        result = await service.commands.execute(request)
        return result.model_dump(mode="json")

    @app.get("/v1/missions/{mission_id}/snapshot")
    async def v1_snapshot(mission_id: str) -> dict:
        try:
            return service.snapshot(mission_id).model_dump(mode="json")
        except ValidationError:
            raise HTTPException(status_code=404, detail="mission not found") from None

    @app.post("/v1/interactions/{interaction_id}/respond")
    async def v1_interaction_respond(interaction_id: str, payload: InteractionRespond) -> dict:
        """通用 select/confirm/input/editor 响应。物理/授权决定永远走
        /approvals/{request_id}/decide——本端点绝不能改变授权状态。"""
        try:
            return service.interactions.respond(
                interaction_id, value=payload.value, idempotency_key=payload.idempotency_key
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/missions/{mission_id}/turns/stream")
    async def send_turn_stream(mission_id: str, payload: TurnCreate):
        """SSE: text deltas as they arrive, then one final result event."""
        import json as _json

        from fastapi.responses import StreamingResponse

        queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_delta(piece: str) -> None:
            queue.put_nowait({"type": "delta", "text": piece})

        async def run() -> None:
            try:
                result = await service.send_turn(mission_id, payload.text, on_delta)
                queue.put_nowait({"type": "final", **_turn_payload(result)})
            except Exception as exc:  # noqa: BLE001 - surfaced as SSE data
                queue.put_nowait({"type": "error", "detail": str(exc)})
            finally:
                queue.put_nowait({"type": "eof"})

        async def events():
            task = asyncio.create_task(run())
            try:
                while True:
                    event = await queue.get()
                    if event["type"] == "eof":
                        break
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/missions/{mission_id}/usage")
    async def mission_usage(mission_id: str) -> dict:
        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return service.mission_usage(mission_id)

    @app.get("/missions/{mission_id}/conversation")
    async def mission_conversation(mission_id: str) -> list[dict]:
        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return service.conversation(mission_id)

    @app.post("/missions/{mission_id}/cancel")
    async def cancel(mission_id: str) -> dict:
        await service.cancel(mission_id)
        return {"cancelled": True}

    @app.get("/approvals/pending")
    async def approvals_pending(mission_id: str | None = None) -> list[dict]:
        # 审计 P0-01：list 也要按 owner 过滤——必须给出 mission_id，
        # 不做全局枚举（防低权限本地进程收集操作计划）。
        if not mission_id:
            raise HTTPException(
                status_code=400,
                detail="mission_id required (global pending enumeration is not served)",
            )
        return [r.model_dump(mode="json") for r in service.pending_approvals(mission_id)]

    @app.post("/approvals/{request_id}/decide")
    async def approvals_decide(request_id: str, payload: DecisionCreate) -> dict:
        # 审计 P0-01/B3：HTTP 决定旁路默认关闭——决定只在
        # rosclaw-operatord（enrollment proof + human presence + daemon ACL）。
        # 仅开发剖面（DEV_SIM_ONLY）可显式打开。
        if os.environ.get("ROSCLAW_DEV_HTTP_DECIDE") != "1":
            raise HTTPException(
                status_code=403,
                detail=(
                    "HTTP approval decisions are disabled (P0-01): decisions "
                    "belong to rosclaw-operatord. Set ROSCLAW_DEV_HTTP_DECIDE=1 "
                    "only in a DEV_SIM_ONLY dev profile."
                ),
            )
        principal = service.principal_for_request(request_id)
        try:
            grant = await service.decide_approval(
                request_id, principal=principal, approve=payload.approve,
                _from_operatord=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "approved": payload.approve,
            "grant_id": grant.grant_id if grant else None,
            "public_hash": grant.public_hash if grant else None,
            "profile": "DEV_SIM_ONLY",
        }

    @app.get("/grants")
    async def grants_list() -> list[dict]:
        return service.list_grants()

    @app.post("/grants/{grant_id}/revoke")
    async def grants_revoke(grant_id: str) -> dict:
        # 审计 P0-01/B3：HTTP 撤销旁路默认关闭（operatord 专属）。
        if os.environ.get("ROSCLAW_DEV_HTTP_DECIDE") != "1":
            raise HTTPException(
                status_code=403,
                detail="HTTP grant revocation is disabled (P0-01): use rosclaw-operatord",
            )
        try:
            service.revoke_grant(grant_id, principal=f"user:local:{os.getuid()}")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"revoked": True, "profile": "DEV_SIM_ONLY"}

    @app.get("/console", response_class=HTMLResponse)
    async def console() -> str:
        return _CONSOLE_HTML

    return app


_CONSOLE_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ROSClaw Console</title>
<style>
body{font-family:system-ui,sans-serif;max-width:840px;margin:2rem auto;padding:0 1rem}
#log{border:1px solid #ccc;border-radius:8px;padding:1rem;height:55vh;overflow:auto;white-space:pre-wrap}
.row{display:flex;gap:.5rem;margin-top:1rem}
input{flex:1;padding:.5rem}button{padding:.5rem 1rem}
.meta{color:#666;font-size:.85em}
.turn{margin:.4rem 0}.who{font-weight:600}
.tool{color:#795e26;font-size:.85em}
</style></head><body>
<h2>ROSClaw Console <span class="meta" id="status"></span></h2>
<div class="row"><input id="goal" placeholder="新 Mission 目标（SIMULATION）">
<button onclick="createMission()">创建 Mission</button></div>
<div id="log"></div>
<div class="row"><input id="text" placeholder="对当前 Mission 说话…"
 onkeydown="if(event.key==='Enter')send()"><button onclick="send()">发送</button></div>
<script>
let missionId = null;
const logEl = () => document.getElementById('log');
function add(cls, who, text){ const d=document.createElement('div');
  d.className='turn'; d.innerHTML=`<span class="who ${cls}">${who}</span> `;
  const s=document.createElement('span'); s.textContent=text; d.appendChild(s);
  logEl().appendChild(d); logEl().scrollTop=logEl().scrollHeight; return s; }
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const AUTH = TOKEN ? {'x-rosclaw-token': TOKEN} : {};
async function status(){ const r = await fetch('/status',{headers:AUTH}); const s = await r.json();
  document.getElementById('status').textContent =
    `profile=${s.profile} model=${s.model} mode=${s.default_mode}`; }
async function createMission(){
  const goal = document.getElementById('goal').value; if(!goal) return;
  const r = await fetch('/missions',{method:'POST',headers:{'Content-Type':'application/json',...AUTH},
    body:JSON.stringify({goal})});
  const m = await r.json();
  if(r.status!==201){ add('meta','系统','创建失败: '+(m.detail||'')); return; }
  missionId = m.mission_id; add('meta','系统',`mission ${missionId} 已创建（${m.mode}）`); }
async function send(){
  if(!missionId){ add('meta','系统','请先创建 Mission'); return; }
  const t = document.getElementById('text'); const text = t.value; if(!text) return;
  t.value=''; add('','你',text);
  const span = add('','ROSClaw','');
  const r = await fetch(`/missions/${missionId}/turns/stream`,{method:'POST',
    headers:{'Content-Type':'application/json',...AUTH},body:JSON.stringify({text})});
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf='';
  while(true){ const {done, value} = await reader.read(); if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split('\\n\\n'); buf = parts.pop();
    for(const p of parts){ const line = p.split('\\n').find(l=>l.startsWith('data: '));
      if(!line) continue; const ev = JSON.parse(line.slice(6));
      if(ev.type==='delta'){ span.textContent += ev.text;
        logEl().scrollTop=logEl().scrollHeight; }
      else if(ev.type==='final'){
        if(!span.textContent && ev.reply) span.textContent = ev.reply;
        add('meta','状态',`state=${ev.state} 工具轮次=${ev.tool_rounds} tokens=${ev.tokens_used}`
          +(ev.degraded?` degraded=${ev.degraded}`:''));
        const u = await (await fetch(`/missions/${missionId}/usage`,{headers:AUTH})).json();
        add('meta','用量',`累计 tokens=${u.total_tokens} 轮次=${u.model_turns} 成本(微单位)=${u.cost_microunits}`);}
      else if(ev.type==='error'){ span.textContent = '错误: '+ev.detail; } } } }
status();
</script></body></html>"""
