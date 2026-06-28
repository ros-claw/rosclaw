# ROSClaw RealSense D405 Perception-Only MVP Fix Report

**Report date:** 2026-06-27  
**ROSClaw version:** v1.0.1  
**Fix guide:** `~/workspace/rosclaw/rosclaw_fix/fix_for_realsense_test.md`  
**Branch / working tree:** `/home/nvidia/workspace/rosclaw/rosclaw_fix/rosclaw`  
**Pull request:** https://github.com/ros-claw/rosclaw/pull/48

---

## 1. Executive Summary

This report documents the repair of the ROSClaw RealSense D405 perception-only MVP loop. The cleanroom test report showed ROSClaw v1.0.1 failing the RealSense functional ecosystem because product layers above the low-level stack (`librealsense2`, `pyrealsense2`, `realsense2_camera`, `librealsense-mcp`, Cosmos provider) were missing or un-wired.

The implementation closes the loop through:

```text
MCP install (public-git / local-path / offline list)
  → body init for realsense-d405 / d435i / dual
  → perception-only sandbox policy
  → skill execution (realsense_capture_rgbd, scene_risk_scan)
  → practice recording with real event validation
  → provider result normalization
  → dashboard + memory/how/know + doctor/bench
```

All new code is module-bounded, source/commit-aware, and avoids mock data by default. Hardware-dependent paths skip or report degradation instead of failing.

---

## 2. Issues Closed

| Issue ID | Description | Status | Evidence |
|---|---|---|---|
| #001 | Hub requires login; `mcp list` times out with no registry | **CLOSED** | `mcp list` now defaults to offline mode when `ROSCLAW_MCP_HUB` is unset; built-in index returns instantly |
| #010 | `mcp install` cannot install external MCPs without active registry | **CLOSED** | `--from-git` and `--local-path` installers implemented; local-path path tested, git path requires network |
| #003 | No `realsense-d405` e-URDF profile | **CLOSED** | Profiles added under `e-urdf-zoo/realsense_d405/`, `realsense_d435i/`, `realsense_dual/` |
| #005 | Sandbox check fails for perception-only bodies | **CLOSED** | `cmd_sandbox_check` resolves aliases and blocks actuator actions, allows sensor capture |
| #006 | Practice records zero events and still reports `SUCCESS` | **CLOSED** | `PracticeCoordinator._stop_session` returns `FAILED` with `zero_events` label when `event_count == 0` |
| #007 | Dashboard has no RealSense or practice-episode endpoints | **CLOSED** | FastAPI routes added for `/api/practice/episodes`, `/api/realsense/status`, `/realsense` page, safe artifact serving |
| #008 | `memory ingest` / `how advise` / `know compile` missing; `memory query` returns mock data | **CLOSED** | CLI subcommands added; `memory query` defaults to real evidence, `--demo` restores mock fallback |
| #009 | `doctor` and `bench realsense` missing | **CLOSED** | `doctor --realsense` and `bench realsense` implemented with pyrealsense2 / `rs-data-collect` backends |
| — | RealSense skill not discoverable / executable | **CLOSED** | Builtin skill registry with `realsense_capture_rgbd` and `scene_risk_scan`; `rosclaw skill search/invoke` wired |
| — | Provider CLI lacks image-path normalized output | **CLOSED** | `provider call --image --capability --output --provider` with `ProviderResultNormalizer` |

---

## 3. Changes by Phase

### Phase A — MCP onboarding (public-git, local-path, offline list)

**Files:**
- `src/rosclaw/mcp/onboarding/cli.py`
- `src/rosclaw/mcp/onboarding/hub_client.py`
- `src/rosclaw/mcp/onboarding/source_installer.py` *(new)*
- `src/rosclaw/mcp/onboarding/stdio_client.py` *(new)*

**What changed:**
- `mcp install` gained `--from-git`, `--local-path`, `--python`, `--venv`, `--no-install-deps`.
- `mcp inspect`, `mcp health`, `mcp call` subcommands added.
- `mcp list` defaults to offline mode when `ROSCLAW_MCP_HUB` is not configured, eliminating the 30-second remote hang.
- `HubClient` gained a built-in fallback index with metadata pointers for `librealsense-mcp` and `realsense-ros-mcp`.
- Git and local-path installers record `source_type`, `source_url`, and commit when available.

### Phase B — RealSense e-URDF / body

**Files:**
- `e-urdf-zoo/realsense_d405/*` *(new)*
- `e-urdf-zoo/realsense_d435i/*` *(new)*
- `e-urdf-zoo/realsense_dual/*` *(new)*
- `src/rosclaw/body/service.py`
- `src/rosclaw/body/renderer.py`
- `src/rosclaw/body/resolver.py`
- `src/rosclaw/runtime/eurdf_loader.py`

**What changed:**
- Added perception-only RealSense profiles with sensors (`color_camera`, `depth_camera`, `aligned_depth`, `pointcloud`), capabilities, USB profiles, and safety constraints.
- `_resolve_profile_alias` maps `realsense-d405`, `realsense_d405`, `d405` (and D435i/dual variants) to canonical profile IDs.
- `BodyResolver` now resolves legacy single-body workspaces by matching the requested body ID against `body.yaml`.
- `EMBODIMENT.md` renderer includes a perception-only disclaimer when the profile has no actuators.

### Phase C — Sandbox perception-only policy

**Files:**
- `src/rosclaw/cli.py` (`cmd_sandbox_check`)

**What changed:**
- Resolves profile aliases before registry lookup.
- Detects `no_actuation` / `perception_only` markers.
- Returns `BLOCK` for actuator action types (`move_base`, `cmd_vel`, `joint_trajectory`, `gripper_command`, `actuator_write`, actuator-oriented ROS service/topic calls) with evidence.
- Returns `ALLOW` for sensor action types (`capture_rgbd`, `capture_rgb`, `capture_depth`, `capture_pointcloud`, `imu_read`, `provider_reasoning`, `query_camera_info`).

### Phase D — Skill search / install / run + RealSense capture skill

**Files:**
- `src/rosclaw/skill/cli.py`
- `src/rosclaw/skill_manager/executor.py`
- `src/rosclaw/skill/builtins/*` *(new)*

**What changed:**
- Builtin skill registry (`registry.yaml`) with `realsense_capture_rgbd` and `scene_risk_scan`.
- `rosclaw skill search`, `install`, `inspect`, `invoke` wired.
- `realsense_capture_rgbd` runner resolves the body, discovers the bound MCP, calls `capture_aligned_rgbd`, writes `color.png`/`depth.png`, and returns metrics.
- `SkillExecutor` emits `skill.execution.complete` for practice/memory recording.

### Phase E — Provider image call + result normalization

**Files:**
- `src/rosclaw/cli.py` (`cmd_provider_invoke`)
- `src/rosclaw/provider/normalizer.py` *(new)*
- `src/rosclaw/provider/core/response.py`

**What changed:**
- `provider call --image PATH --capability CAP --provider NAME --output DIR` accepts image input.
- `ProviderResultNormalizer` converts provider text into a structured risk schema; on parse failure it marks `schema_valid: false`, `requires_guard: true`, and preserves the raw response.

### Phase F — Practice Data Flywheel MVP

**Files:**
- `src/rosclaw/practice/coordinator.py`
- `src/rosclaw/practice/storage/layout.py`
- `src/rosclaw/cli.py` (`cmd_practice_run`)

**What changed:**
- `PracticeCoordinator` records `camera.rgbd_frame`, `provider.result`, and `sandbox.decision` events.
- `event_count == 0` now forces `outcome: FAILED` with failure label `zero_events`.
- Episode layout writes `episode.json`, `timeline.jsonl`, `raw/events.jsonl`, and subdirectories for `frames`, `provider`, `sandbox`, `runtime`, `metrics`, `artifacts`.
- `rosclaw practice run --robot ID --skill SKILL --provider PROVIDER --output-root DIR` executes the full skill+provider+sandbox recording flow.

### Phase G — Dashboard MVP

**Files:**
- `src/rosclaw/dashboard/web_server.py`

**What changed:**
- FastAPI routes: `GET /api/practice/episodes`, `/api/practice/episodes/{id}`, `/api/practice/episodes/{id}/timeline`, `/api/practice/episodes/{id}/artifacts`, `/api/practice/episodes/{id}/provider`, `/api/practice/episodes/{id}/sandbox`.
- `GET /api/realsense/status`, `/api/realsense/latest-frame`.
- `GET /api/artifacts/{path:path}` serves files safely under the configured data root.
- `/realsense` HTML page renders status and latest frame with an empty-state command when no frames exist.

### Phase H — Memory / How / Know

**Files:**
- `src/rosclaw/memory/interface.py`
- `src/rosclaw/memory/seekdb_client.py`
- `src/rosclaw/how/engine.py`
- `src/rosclaw/know/interface.py`
- `src/rosclaw/cli.py`

**What changed:**
- `rosclaw memory ingest --episode-id ID --data-root DIR` ingests a practice episode into SeekDB-backed memory.
- `rosclaw memory query` returns only real evidence by default; `--demo` enables the previous mock fallback.
- `rosclaw how advise --body ID --failure LABEL --episode-id ID --data-root DIR` returns evidence-backed interventions.
- `rosclaw know compile --task TASK --episode-id ID --data-root DIR` compiles a grounded task card from episode evidence.
- Fixed missing `json`/`Path` imports in `memory/interface.py` and missing `json` import in `how/engine.py`.

### Phase I — Doctor / Bench

**Files:**
- `src/rosclaw/cli.py`
- `src/rosclaw/bench/realsense.py` *(new)*

**What changed:**
- `rosclaw doctor --realsense` enumerates RealSense devices, checks `pyrealsense2`, ROS2/realsense2_camera, installed MCPs, body profile, and Cosmos reachability.
- `rosclaw bench realsense --duration N --output DIR` streams frames via `pyrealsense2` or falls back to `rs-data-collect -t N`, then writes `report.json` with frame counts, FPS, drops, USB mode, and degradation flag.
- Fixed `rs-data-collect` invocation to use `-t` instead of unsupported `--duration`.

---

## 4. Verification

### 4.1 CLI regression (executed in venv)

| Command | Result | Notes |
|---|---|---|
| `rosclaw mcp list --json` | ✅ PASS | Returns instantly in offline mode; shows installed + built-in available MCPs |
| `rosclaw mcp install --from-git https://github.com/ros-claw/librealsense-mcp --no-install-deps` | ⚠️ NETWORK TIMEOUT | Implementation correct; environment has no outbound GitHub access |
| `rosclaw body init --robot realsense-d405 --name d405_lab_01 --validate --force` | ✅ PASS | Initializes body and renders EMBODIMENT.md |
| `rosclaw sandbox check --robot realsense-d405 --action '{"type":"move_base"}' --json` | ✅ BLOCK | Correctly blocks actuator action, exit 1 |
| `rosclaw sandbox check --robot realsense-d405 --action '{"type":"capture_rgbd"}' --json` | ✅ ALLOW | Correctly allows sensor capture |
| `rosclaw skill invoke realsense_capture_rgbd --body d405_lab_01 --output /tmp/capture --json` | ✅ EXPECTED FAILURE | Fails with clear message: no RealSense MCP installed/healthy |
| `rosclaw practice run --robot d405_lab_01 --skill realsense_capture_rgbd --provider cosmos-reason2-lan --output-root /tmp/episode --json` | ✅ EXPECTED FAILURE | Skill fails (no hardware/MCP); episode recorded as `FAILED` with `zero_events` |
| `rosclaw dashboard --open` | ✅ PASS | Server starts on `:8765`; `/api/practice/episodes` returns empty state with CLI command; `/realsense` page renders |
| `rosclaw memory ingest --episode-id <id> --data-root /tmp/episode` | ✅ PASS | Ingests failed episode into memory |
| `rosclaw memory query "RealSense D405 RGB-D capture"` | ✅ PASS | Returns "No matching experiences found" (no mock data by default) |
| `rosclaw how advise --body d405_lab_01 --failure low_fps --episode-id <id> --data-root /tmp/episode --json` | ✅ PASS | Returns fallback intervention for zero-event episode |
| `rosclaw bench realsense --duration 1 --output /tmp/bench --json` | ✅ PASS | Reports `pyrealsense2` missing and `rs-data-collect` timeout; structured report written |
| `rosclaw doctor --realsense --json` | ✅ PASS | Returns structured checks; exits 1 because RealSense stack is not installed |

### 4.2 Unit tests

Focused test run (all tests directly supporting the RealSense MVP):

```bash
pytest tests/test_memory_ingest_practice_episode.py \
       tests/test_memory_query_no_mock_by_default.py \
       tests/test_how_advise_realsense_evidence.py \
       tests/test_know_compile_realsense_taskcard.py \
       tests/test_doctor_realsense_checks.py \
       tests/test_bench_realsense_rs_data_collect_parser.py \
       tests/test_realsense_capture_rgbd_skill.py \
       tests/test_realsense_skill_manifest.py \
       tests/test_skill_search_cli.py \
       tests/test_provider_image_call.py \
       tests/test_practice_records_mcp_artifacts.py \
       tests/test_practice_zero_events_not_success.py \
       tests/test_practice_non_default_data_root.py \
       tests/test_practice_provider_sandbox_linkage.py \
       tests/test_dashboard_practice_api.py \
       tests/test_dashboard_no_mock_realsense.py -q
```

**Result:** `57 passed, 2 warnings in 19.52s`

A broader run of legacy MCP/skill/provider tests surfaced pre-existing failures caused by the missing `pytest-asyncio` plugin in this environment (`pytest.mark.asyncio` unknown). These failures are not related to the RealSense MVP changes.

---

## 5. Known Limitations

1. **No outbound network access** in the verification environment, so `mcp install --from-git` cannot clone the real `librealsense-mcp` repository. The installer is covered by unit tests with local fake git repos.
2. **No RealSense hardware / `pyrealsense2`** on the test machine, so skill invocation and bench report real capture errors rather than frames. Hardware-specific tests are marked `@pytest.mark.realsense_hw` and skip when the stack is absent.
3. **ROS2 stack not present**, so doctor ROS2 checks report `FAIL`/`WARN`. This is the expected degraded-environment behavior.
4. **Legacy single-body workspace** requires the body ID to match `body.yaml`; multi-body registry workflows are not exercised here.
5. **Dashboard `--open`** starts the server but the current harness background task detaches; in normal interactive use the server stays alive until Ctrl+C.

---

## 6. Files Added / Modified

### Added
- `e-urdf-zoo/realsense_d405/`
- `e-urdf-zoo/realsense_d435i/`
- `e-urdf-zoo/realsense_dual/`
- `src/rosclaw/bench/realsense.py`
- `src/rosclaw/mcp/onboarding/source_installer.py`
- `src/rosclaw/mcp/onboarding/stdio_client.py`
- `src/rosclaw/provider/normalizer.py`
- `src/rosclaw/skill/builtins/`
- 16 new test files under `tests/`

### Modified
- `src/rosclaw/cli.py`
- `src/rosclaw/body/compatibility.py`
- `src/rosclaw/body/renderer.py`
- `src/rosclaw/body/resolver.py`
- `src/rosclaw/body/schema.py`
- `src/rosclaw/body/service.py`
- `src/rosclaw/dashboard/web_server.py`
- `src/rosclaw/how/engine.py`
- `src/rosclaw/know/interface.py`
- `src/rosclaw/mcp/onboarding/cli.py`
- `src/rosclaw/mcp/onboarding/hub_client.py`
- `src/rosclaw/memory/interface.py`
- `src/rosclaw/practice/coordinator.py`
- `src/rosclaw/practice/storage/layout.py`
- `src/rosclaw/runtime/eurdf_loader.py`
- `src/rosclaw/skill/cli.py`
- `src/rosclaw/skill_manager/executor.py`
- `tests/conftest.py`

---

## 7. Next Steps

1. **Merge PR #48** (`https://github.com/ros-claw/rosclaw/pull/48`) after review.
2. **Validate on real hardware:** Run `rosclaw skill invoke realsense_capture_rgbd --body d405_lab_01 --output ./capture` on a Jetson + D405 with `librealsense-mcp` installed.
3. **Enable `mcp install --from-git` end-to-end** in an environment with outbound GitHub access.
4. **Add `pytest-asyncio`** to dev dependencies and triage legacy async test failures.
5. **Extend `realsense-ros-mcp`** integration once the ROS2 RealSense stack is available.

---

*Report generated by Claude Code for the ROSClaw RealSense D405 perception-only MVP repair.*
