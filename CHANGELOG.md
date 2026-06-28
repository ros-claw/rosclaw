# Changelog

All notable changes to ROSClaw will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
    `realsense-dual`) through `body/cli.py`, `body/registry.py`, and
    `body/service.py`.
  - Added `tests/integration/test_realsense_profiles.py` covering embodiment,
    safety, capability, simulation, semantic, and benchmark profiles.
  - Kept PR #48 RealSense skills compatible by aligning builtin profile
    capabilities with the `realsense_capture_rgbd` skill manifest.

### Changed

- `SkillExecutor._check_body_compatibility()` is now **fail-closed**: resolver
  errors and `unknown` compatibility statuses block execution instead of
  allowing it.

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
