# Changelog

All notable changes to ROSClaw will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-09-17

Internal Alpha 发布节点：0914 审计修复轨道 + MuJoCo Harness 轨道 +
三审 backlog（B-1~B-6）全部闭环（18 PR，merge truth 机器复核）。

### Added

- **真实 Agent Gate**（`scripts/acceptance/agent_gate_runner.py`）：
  干净 wheel 安装、10 个未见任务 ×3 真实 K3 验收（29/30）；
  环境结局 oracle（per-render 证据绑定 / 相机差异像素级判定 /
  物理结论从原始序列重算——不信终端文字不信模型自报）。
- **Harness 资格认证器**（`scripts/acceptance/harness_qualification.py`）：
  EGL/OSMesa/Xvfb 逐后端真实单帧渲染 + rollout 动力学闭环，
  机器生成资格矩阵（Jetson aarch64 实证 QUALIFIED）。
- **MuJoCo Harness**（MH 轨道）：SimulationRuntime + MujocoBackend
  原生仿真内核（状态真权归 MuJoCo、Control Schema、多模态观测、
  并行批次实验、HarnessBench v1 + `rosclaw sim` CLI）。
- **取消传播闭环**：Esc/Ctrl-C/自然语言停止 → operation + 同步渲染
  子进程（注册表 + killpg）+ 模型回合全停；迟到成功不得翻转
  CANCELLED（账本防护）。
- **chat 自动登录流**：未配置时 TTY 确认写默认配置直接进 chat
  （/login 会话内完成）。
- **setup status `--probe`**：显式联网模型探测（默认本地-only）。

### Fixed

- **wheel JS 运行时闭包**（G-1a）：wheel 内嵌 js_stage 无
  node_modules，干净安装 `rosclaw chat` 即死——首跑 npm ci
  bootstrap（lock digest 幂等、filelock 防并发、
  JS_RUNTIME_BOOTSTRAP_FAILED 诚实报错带手动命令）。
- **验收 oracle 假绿根治**（X-2）：U10 提示词回显可过 / U06
  None digest 冒充一致性 / U05 无视频↔trace 绑定——全部根治；
  per-render receipt（同 trace 多渲染不再互覆盖销毁 RenderRef
  证据）。
- **渲染取消传播**：同步渲染子进程不再在取消后孤儿跑完全程。
- **轨迹 overlay 三年假可见**：mjv_connector 线宽单位误解
  （米≠像素）——1.6m 胶囊自遮蔽相机；5mm + 帧像素证据。
- **取消传播进程组**：killpg 灭整个进程组（孙子渲染/仿真进程
  不再孤儿）。
- Kimi 403 分类三义区分：配额耗尽（QUOTA_EXHAUSTED）/并发限流
  （RATE_LIMITED）/真凭据失败（AUTH_FAILED）——关键词优先于
  状态码。
- rosclaw-tui Ctrl-C 取消死链（/v2/missions/:id/cancel 404 →
  真实路由）。

### Changed

- **单一仿真栈**（ADR-0015）：SimulationRuntime 为唯一仿真内核；
  `src/rosclaw/agentd/sim_render.py` 降级 deprecated 适配层（公开函数集合
  机器冻结——防分叉门禁）。
- setup status 默认本地-only——看状态不再烧模型 API 调用。

### Governance

- **merge truth 机器生成**（`scripts/acceptance/merge_truth.py`）：
  合并状态只信 GitHub API——#546 误记事件根治（update-branch
  被接受≠合并完成）；有未合并 PR 时 rc=1 可作报告定稿门禁。

## [Unreleased] - 2026-06-21

### Added

- **Body Module (P0 three-layer body system)**
  - `src/rosclaw/body/` self-contained module: schema, resolver, compiler, renderer, diff, notes, compatibility, validators, and CLI.
  - e-URDF / `~/.rosclaw/body/body.yaml` / `~/.rosclaw/body/EMBODIMENT.md` three-layer body model.
  - `EffectiveBodyCompiler` merges Physical DNA, instance state, calibration, and maintenance events into a single Effective Body Model.
  - `BodyResolver` API and `rosclaw://` URI scheme for cross-module body access.
  - `BodyDiffer` with impact-aware change categorization (`structural`, `installed_component`, `actuator_status`, `sensor_status`, `calibration`, `safety`, `capability`, `incident`).
  - `SkillCompatibilityChecker` with `compatible` / `degraded` / `blocked` / `unknown` statuses and incremental recheck.
  - CLI commands: `rosclaw body link-eurdf`, `inspect`, `diff`, `update-state`, `note`.
  - Generated artifacts: `~/.rosclaw/body/EMBODIMENT.md`, `~/.rosclaw/body/skill_compatibility.yaml`, `generated/*.json` summaries, and historical snapshots.

- **Body Module (P2 multi-body observability and fleet operations)**
  - `BodyRegistryManager` with multi-body registry, legacy migration, and archive-on-remove.
  - `BodyResolver` routing by `body_id` with `--body` CLI selector and legacy fallback.
  - `FleetCompatibilityAggregator` in `src/rosclaw/body/fleet.py` for cross-body skill compatibility aggregation.
  - New CLI commands: `rosclaw body fleet-compat`, `rosclaw fleet status`, `rosclaw fleet stop`.
  - New MCP tools: `list_bodies`, `get_body`, `switch_body`, `list_body_history`, `check_skill_compatibility`, `fleet_skill_compatibility`.
  - Dashboard `/api/body` endpoint, `/body` HTML page, and WebSocket body section.
- Provider / Sandbox / Memory body adapters: `ProviderBodyBinder.diagnose()`, `SandboxBodyAdapter.to_mujoco_config()` / `to_isaac_config()`, and `BodyMemoryEventWriter.write_body_change()`.
- New CLI commands: `rosclaw provider diagnose`, `rosclaw sandbox generate-config`, and `rosclaw body update-state --from-provider-health`.
- Fleet compatibility cache (`FleetCompatibilityCache`) and switch-body runtime hooks (`BodySwitchHooks`).
- Dashboard body sub-routes: `/api/body/effective`, `/api/body/skills`, `/api/body/history`, `/api/body/provider-health`.
- Architecture contract test `tests/architecture/test_no_direct_body_yaml_access.py` forbidding non-body modules from reading `~/.rosclaw/body/body.yaml` directly.

- **e-URDF-Zoo integration and dexterous-hand safety (PR #47)**
  - `rosclaw.eurdf.zoo_client.EurdfZooClient`: discover, resolve, pull, validate,
    and convert manifest-driven e-URDF-Zoo assets into `RobotCompleteProfile` /
    `EurdfProfile`.
  - `rosclaw eurdf` CLI: `info`, `search`, `validate`, `pull`, and `cache list`.
  - Dexterous-hand safety hardening in `EffectiveBodyCompiler`,
    `SafetyInvariantEngine`, and `BodyQueryEngine`.
  - Sample manifest-driven assets under `e-urdf-zoo/robots/` for
    `dexhands/inspire_hand/right`, `dexhands/ability_hand/left`, and
    `grippers/panda/default`.
  - New tests: `tests/eurdf/`, `tests/body/test_body_init_from_zoo.py`, and
    `tests/body/test_dexhand_agent_safety.py`.

- **RealSense D405 perception-only MVP (PR #48)**
  - `rosclaw bench realsense` command with device discovery, snapshot capture,
    and `RSDataCollector` / `CameraEvidence` parsing.
  - `realsense_capture_rgbd` builtin skill that routes to an installed
    RealSense MCP server (`capture_aligned_rgbd`, `capture_color_image`,
    `capture_frames`) and persists artifacts to an output directory.
  - `scene_risk_scan` perception-only builtin skill placeholder.
  - `rosclaw mcp` onboarding helpers: `source_installer`, `hub_client`, and
    `stdio_client` health/tool-call utilities.
  - Perception-only e-URDF-Zoo fixtures under `e-urdf-zoo/realsense_d405/`,
    `realsense_d435i/`, and `realsense_dual/` (later superseded by builtin
    Python profiles in PR #49).
  - Dashboard practice, memory, and provider-image APIs wired to RealSense
    evidence paths.
  - Extensive unit-test coverage under `tests/test_*realsense*.py` and
    `tests/test_bench_realsense_rs_data_collect_parser.py`.
  - Memory CLI (`rosclaw memory ingest/query`) now persists real practice
    evidence and returns actual SeekDB results by default; `--demo` re-enables
    the mock fallback.

- **Builtin RealSense e-URDF profiles (PR #49)**
  - Added `rosclaw.eurdf_zoo.profiles.realsense_d405`,
    `realsense_d435i`, and `realsense_dual` as builtin Python profiles.
  - `RobotRegistry` falls back to builtin Python profiles when a profile is
    not found in the e-URDF-Zoo directory layout.
  - Wired hyphenated aliases (`realsense-d405`, `realsense-d435i`,
    `realsense-dual`) through `src/rosclaw/body/cli.py`,
    `src/rosclaw/body/registry.py`, and `src/rosclaw/body/service.py`.
  - Added `tests/integration/test_realsense_profiles.py` covering embodiment,
    safety, capability, simulation, semantic, and benchmark profiles.
  - Kept PR #48 RealSense skills compatible by aligning builtin profile
    capabilities with the `realsense_capture_rgbd` skill manifest.

- **RealSense practice event pipeline + offline hub catalog (PR #48 follow-up)**
  - Rewrote `rosclaw practice run` to emit the full event sequence
    (`runtime.start`, `skill.start/result`, `camera.rgbd_frame`,
    `provider.request/result`, `sandbox.decision`, `runtime.stop`) and copy
    skill artifacts into `artifacts/frames/`.
  - Deferred event emission until the skill succeeds, so zero events now
    correctly yield `FAILED`.
  - Added PNG-IHDR fallback for image dimensions when PIL is missing.
  - Added built-in offline Hub catalog so `rosclaw hub search realsense`
    works without login.
  - Added `tests/test_hub_search_offline.py`.

- **RealSense doctor/bench reporting + workspace isolation (PR #48 follow-up)**
  - `rosclaw bench realsense` report now includes camera info, serial,
    firmware, USB speed, profile, and status fields.
  - `rosclaw doctor` default path now includes RealSense SDK, ROS2, USB speed,
    MCP, and profile checks.
  - Refactored `_run_doctor_realsense` to share `_collect_realsense_checks`
    with the default doctor path.
  - `hub.lockfile.DEFAULT_LOCKFILE_PATH` is now lazy/respects `ROSCLAW_HOME`.
  - Workspace isolation helpers in `src/rosclaw/firstboot/workspace.py` so tests can use
    a custom home directory.
  - New tests: `tests/test_doctor_default_realsense.py`,
    `tests/test_dashboard_realsense_api.py`,
    `tests/test_practice_records_events.py`,
    `tests/test_workspace_isolation.py`.

- **RealSense practice events, dashboard streams, bench semantics (PR #48 follow-up)**
  - `PracticeCoordinator` now emits `runtime.start`/`runtime.stop` and tracks
    source vs lifecycle events.
  - Empty source list disables all sources; runtime-only sessions succeed.
  - Skill failures record failure labels and force `FAILED` outcome.
  - `practice start` defaults the camera skill from the body profile ID.
  - Dashboard registers `/api/realsense/*` before the greedy artifact route.
  - `bench realsense` exposes per-stream FPS and aggregate metrics.
  - `practice validate/show` print episode summaries with frame/provider/decision
    counts and timeline/runtime/source checks.
  - New tests: `tests/test_practice_runtime_source.py`,
    `tests/test_practice_validate_show.py`,
    `tests/test_bench_realsense_metric_semantics.py`.

- **Practice provider resolution + dashboard glob fix (PR #48 follow-up)**
  - Restored `Robot:` line in `rosclaw practice show` output.
  - `rosclaw practice start` no longer hard-codes provider mapping; it uses
    `ROSCLAW_PRACTICE_DEFAULT_PROVIDER` or scans `ProviderRegistry` for a
    provider advertising the requested capability.
  - Dashboard provider route globs `provider_result_*.json` and serves the
    latest, matching the new result filename convention.

### Changed

- Repository ownership is now explicit: reviewed product evidence lives under
  `docs/evidence/`, raw `reports/` output is ignored, the RH56 deterministic
  reference policy is co-located with its LeRobot worker plugin, and website
  Supabase migrations are owned only by `ros-claw/rosclaw-website`.
- `SkillExecutor._check_body_compatibility()` is now **fail-closed**: resolver
  errors and `unknown` compatibility statuses block execution instead of
  allowing it.

### Fixed

- `rosclaw agent install` now replaces ROSClaw-owned legacy MCP entries while
  preserving unrelated MCP servers, and validation rejects malformed stdio
  command ordering instead of accepting a configuration that cannot start.
- Checked-in Agent guidance and LeRobot fixtures no longer contain
  developer-machine absolute paths.

### Tests

- Added `tests/body/` suite covering schema, link-eurdf, inspect, effective body,
  diff, update-state, notes, skill compatibility, and cross-module references.
- Added `TestSkillExecutorBodyCheck` to verify fail-closed behavior on resolver
  errors and unknown compatibility.
- Added `tests/body/test_multi_body_registry.py`, `tests/body/test_fleet_compatibility.py`,
  `tests/mcp/test_body_tools.py`, and `tests/dashboard/test_body_page.py`.

## [1.0.0] - 2026-05-28

### Added

- **Core Architecture**
  - EventBus with publish/subscribe, priority queuing, and history
  - LifecycleMixin with 8-state lifecycle management
  - Runtime orchestrating all six grounding engines
  - Command-Response pattern for MCPHub via asyncio Futures

- **Six Grounding Engines**
  - **Physical (e-URDF)**: Robot model parser with JointSpec/LinkSpec
  - **Action (Firewall)**: DigitalTwinFirewall with 3-layer validation (joint limits, collision, torque), SafetyLevel enum (STRICT/MODERATE/PERMISSIVE)
  - **Timeline (Practice)**: UnifiedTimeline with 8-channel sensorimotor recording at 1kHz, PracticeRecorder with PraxisEvent schema
  - **Experience (Memory)**: SeekDB with Memory/SQLite backends, experience storage and similarity search
  - **Skill (SkillManager)**: SkillRegistry with stats, SkillExecutor with precondition checking, SkillLoader for JSON/programmed skills
  - **Collaboration (Swarm)**: SwarmRuntimeManager for multi-agent task allocation

- **LLM Provider Abstraction**
  - LLMProvider ABC with plan_task, analyze_failure, generate_skill_description, health_check
  - DeepSeekProvider, OpenAIProvider, QwenProvider implementations
  - Factory pattern: get_provider(), list_providers(), register_provider()
  - Backward-compatible aliases (DeepSeekClient, DeepSeekConfig)

- **MCP Drivers**
  - BaseDriver abstract base with DriverState and TrajectoryCommand
  - MuJoCoSimDriver with mock mode fallback
  - ROS2Driver and SerialDriver stubs

- **CLI Tool**
  - `rosclaw --version`
  - `rosclaw init [DIR]` — workspace initialization with config file
  - `rosclaw run` / `rosclaw start` — runtime launcher
  - `rosclaw status` — status check

- **Docker Support**
  - Dockerfile based on python:3.11-slim
  - docker-compose.yml with volume mounts and health checks

- **Developer Experience**
  - Makefile with install, test, lint, format, clean, all targets
  - hello_robot.py example demonstrating full workflow
  - CONTRIBUTING.md with dev standards and PR process
  - MIT LICENSE

- **Documentation**
  - API_REFERENCE.md with full public API
  - BENCHMARK.md with performance metrics
  - SECURITY_AUDIT.md
  - ROLE_SWAP_REVIEW.md (architecture review)
  - COLLABORATION_LOG.md

### Fixed

- LifecycleMixin `_state` collision with BaseDriver `_driver_state` (renamed to `_lifecycle_state`)
- EventBus accepting non-callable handlers (added TypeError guard)
- MuJoCoSimDriver empty path causing ParseXML error (added whitespace guard)
- SkillExecutor registry parameter inconsistency (reverted to required)
- Driver `move_joints` allowed before initialization (added `_ensure_ready` guard)
- Joint position validation accepting unsafe values (tightened bound to 1e5)
- EventBus `clear_history` missing type annotation

### Security

- Input validation for joint positions (type, finiteness, bounds)
- SkillRegistry validation (SkillEntry type check, empty name rejection)
- Double-initialization guard in LifecycleMixin
