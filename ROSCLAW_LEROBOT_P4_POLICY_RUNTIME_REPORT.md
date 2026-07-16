# ROSClaw × LeRobot Bridge P4 — Persistent Policy Runtime Report

**Date:** 2026-07-16  
**Branch:** `main`  
**Status:** P4-A through P4-E implementation complete; P4.1 Acceptance Hardening complete. Full LeRobot unit + integration test suite passes (284 passed, 3 skipped). Ready for P5 real-hardware execution gating.

---

## 1. Executive Summary

P4 upgrades the ROSClaw ↔ LeRobot bridge from a **one-shot subprocess-per-inference** model to a **persistent policy runtime** that keeps a LeRobot policy resident across many inference steps, preserves episode-scoped temporal state, enforces explicit observation/action contracts, maps actions to the current effective body, and runs sandbox-preflighted proposal-only / shadow loops without ever executing hardware commands.

**P4.1 Acceptance Hardening** closes the safety gaps between "engineering prototype" and "real-robot前置层":

- Action semantics are no longer guessed from `policy_type`; only explicit policy metadata is authoritative. Missing metadata produces a fail-closed `unknown_action_semantics` block.
- `WARMUP` runs a real end-to-end inference (`preprocess → select_action → postprocess`) and validates shape / NaN / Inf before the live loop starts.
- The runtime enforces a **single active session**; a second `CREATE_SESSION` returns `session_capacity_exceeded`.
- `INFER` enforces strict **step-index monotonicity** (`expected = last + 1`); mismatches return `non_monotonic_step_index`.
- Worker death immediately fails pending requests with `worker_died` instead of waiting for the 120 s timeout.
- Action batch dimensions are normalized (`[1, D] → [D]`); undeclared `[N>1, D]` tensors are rejected as `batch_action_not_supported`, while declared `[chunk_size, D]` chunks remain valid.
- The rollout loop uses a **fixed-deadline scheduler** and records `deadline_miss_ms`, `overrun_count`, and `effective_control_hz`.
- Shadow/proposal rollouts can finalize into the full **Practice lifecycle** (session/episode/manifest/timeline/catalog) via `finalize_rollout_practice_session`.

All acceptance gates from §19 of the P4 specification and the P4.1 hardening checklist pass:

| Gate | Result |
|------|--------|
| Policy loads once | ✅ `LOAD_POLICY` succeeds and `policy_loaded` stays true |
| 100 continuous inferences on same worker | ✅ 100 `INFER` calls on stable worker PID |
| `policy.reset()` per session | ✅ called in `CREATE_SESSION` / `RESET_SESSION` |
| Sessions do not leak | ✅ `active_sessions` tracked and closed |
| Single active session enforced | ✅ second session returns `session_capacity_exceeded` |
| Step monotonicity enforced | ✅ mismatch returns `non_monotonic_step_index` |
| Worker death wakes pending requests | ✅ `worker_died` returned immediately |
| Pre/post processors invoked | ✅ `make_pre_post_processors` built and `_move_processor_to_device` applied |
| State ordering uses names | ✅ `observation_adapter` enforces flat named keys |
| No `action_dim=7` fallback for real inference | ✅ shape taken from `output_features.action.shape` |
| No policy_type → representation/unit guessing | ✅ explicit metadata only; unknown semantics block |
| WARMUP runs real inference | ✅ preprocess/model/postprocess/shape/NaN-Inf validated |
| Action batch `[1,D]` squeezed | ✅ normalized to `[D]` |
| Undeclared `[N>1,D]` blocked | ✅ `batch_action_not_supported` |
| Declared chunks `[T,D]` preserved | ✅ `chunk_size` metadata distinguishes chunk from batch |
| Proposals carry representation/names/units/timing/source/authoritative | ✅ `rosclaw.action_proposal.v2` populated |
| One mock body achieves `exact` mapping | ✅ RH56 6-DoF mock test |
| Incompatible body mapping blocked | ✅ fail-closed, `INCOMPATIBLE_MAPPING` stop reason |
| Sandbox preflight end-to-end | ✅ `sandbox_run` invoked per step in shadow mode |
| Proposal-only loop 100 steps | ✅ CLI path exercised in integration tests |
| Shadow mode no hardware execution | ✅ `hardware_actions_executed == 0` always |
| Fixed-deadline scheduler | ✅ `next_deadline = loop_start + (step+1)/control_hz` |
| Deadline miss / overrun metrics | ✅ recorded per step |
| Practice full lifecycle | ✅ session/episode/manifest/timeline/catalog |
| Python 3.11 core never imports torch/lerobot | ✅ subprocess boundary test passes |

---

## 2. Baseline: One-Shot Runtime (P3)

Before P4 every `provider.infer()` call:

1. Spawned a fresh Python 3.12 subprocess via `subprocess.run`.
2. Loaded the policy and pre/post processors from disk.
3. Ran one inference.
4. Exited.

Problems:

- High latency: model + processor load dominates every step.
- No episode state: `policy.reset()` was effectively per-inference.
- No temporal chunking or action-chunk continuity.
- Observation/action validation was implicit and fail-open.
- Body mapping and sandbox preflight were not wired into inference.

---

## 3. Architecture

### 3.1 Dual-runtime isolation

| Runtime | Python | Role | torch/lerobot |
|---------|--------|------|---------------|
| ROSClaw core | 3.11 | CLI, provider, contracts, body mapping, sandbox | **forbidden** |
| LeRobot worker | 3.12 | Policy load, preprocessor, `select_action`, postprocessor | **allowed** |

Communication: JSONL over stdin/stdout with protocol `rosclaw.lerobot.policy_runtime.v1`.

### 3.2 Core components

```text
rosclaw.integrations.lerobot
├── policy_runtime
│   ├── protocol.py       # JSONL framing + protocol version
│   ├── manager.py        # PersistentRuntimeManager (subprocess Popen + reader threads)
│   ├── client.py         # ROSClaw-side protocol client
│   ├── state.py          # RuntimeState state machine
│   └── session.py        # Episode session bookkeeping
├── policy_worker_runtime.py   # Worker entry point (LeRobot interpreter)
├── policy_worker_service.py   # Long-lived worker service (loads policy once)
├── contracts
│   ├── observation.py    # rosclaw.observation_contract.v1 / snapshot.v1
│   ├── action.py         # rosclaw.action_proposal.v2
│   ├── hashes.py         # deterministic hashing helpers
│   └── validator.py      # observation validation
├── rollout
│   ├── loop.py           # proposal-only / shadow loop orchestration
│   ├── state.py          # RolloutMode, RolloutStopReason
│   ├── observation_source.py
│   ├── recorder.py       # Practice-compatible JSONL trace recorder
│   ├── metrics.py
│   ├── sandbox_preflight.py
│   └── practice_bridge.py # rollout trace → full Practice lifecycle
├── action_adapter.py     # processed action → rosclaw.action_proposal.v2
├── observation_adapter.py# ROSClaw obs → worker flat tensor keys
└── cli.py                # `rosclaw lerobot policy ...`, `mapping ...`, `rollout ...`

rosclaw.body.action_mapping
├── schema.py
├── resolver.py
├── mapper.py
├── units.py
├── chunk.py
├── validator.py
└── report.py
```

### 3.3 Worker protocol methods

- `HELLO` — protocol handshake
- `PROBE` — runtime capability discovery
- `LOAD_POLICY` — materialize policy, build processors, extract metadata
- `WARMUP` — run one real end-to-end inference (`preprocess → select_action → postprocess`) and validate shape / NaN / Inf
- `CREATE_SESSION` — new episode, call `policy.reset()`
- `RESET_SESSION` — reset episode, call `policy.reset()`
- `INFER` — preprocess → `select_action` → postprocess → return raw + processed action
- `HEALTH` — loaded state, active sessions
- `CLOSE_SESSION` — drop session
- `UNLOAD_POLICY` — free policy
- `SHUTDOWN` — graceful exit

### 3.4 State machines

**Runtime state:** `idle → starting → ready → error → stopped`  
**Rollout state:** `idle → running → completed | failed | incompatible_mapping | sandbox_block | stale_observation | nan_action | timeout`  
**Session state:** created per episode; `last_step_index` enforces monotonicity.

---

## 4. Real Policy Results

### 4.1 Test setup

- Policy: `lerobot/act_aloha_sim_transfer_cube_human`
- Runtime: LeRobot 0.6.x (Python 3.12) with one-time processor migration
- Device: CPU (`device=cpu`)
- Fixture: `examples/lerobot/sample_observation_aloha_act.json`
- Test: `tests/integrations/test_lerobot_persistent_runtime_smoke.py::test_persistent_runtime_100_inferences`

### 4.2 Latency / stability

Measured on the same persistent worker:

| Metric | Value |
|--------|-------|
| `LOAD_POLICY` latency | **7.7 s** (includes model + processor load) |
| Mean inference latency | **88.0 ms** |
| Min inference latency | **28.9 ms** |
| p95 inference latency | **77.1 ms** |
| Max inference latency | **3.66 s** (first inference warm-up / JIT / import side effects) |
| Worker PID across 100 steps | **stable (745985)** |
| `HEALTH` after 100 inferences | `policy_loaded: true, active_sessions: 1` |
| NaN/Inf actions observed | **0** |

The first inference is slower; after warm-up the loop is stable.

### 4.3 Action shape and semantics

- `output_features.action.shape` = `[14]`
- Processed action returned shape = `[1, 14]` (postprocessor preserves a batch dimension); the contract layer squeezes `[1, D]` → `[D]`.
- **P4.1 fail-closed semantics:** `policy_worker_service._extract_metadata` no longer infers `action_representation` / `action_unit` from `policy_type`. Only explicit config tags are propagated. The generic ALOHA smoke policy therefore yields `representation="unknown"`, `units="unknown"`, and `names=[]`, which `build_action_proposal_v2` blocks as `unknown_action_semantics`. A real-robot run must supply explicit metadata or an observation contract that carries authoritative action semantics.
- `chunk_size` from `policy.config` is propagated in `metadata.extra.chunk_size` so that `[chunk_size, D]` action chunks are distinguished from unsupported batched actions.

---

## 5. Contract Examples

### 5.1 Observation snapshot (v1)

```json
{
  "schema_version": "rosclaw.observation_snapshot.v1",
  "observation_id": "obs_0001",
  "source_time": 1721011200.0,
  "receive_time": 1721011200.05,
  "fields": {
    "observation.state": {
      "shape": [14],
      "dtype": "float32",
      "names": ["joint_0", "joint_1", "..."],
      "values_hash": "sha256:..."
    },
    "observation.images.top": {
      "shape": [1, 3, 480, 640],
      "dtype": "float32",
      "source": "file:/path/to/top.jpg"
    }
  },
  "task": "Put the cube in the bin."
}
```

### 5.2 Action proposal (v2)

```json
{
  "schema_version": "rosclaw.action_proposal.v2",
  "proposal_id": "p4_semantics_0",
  "session_id": "p4_semantics_session",
  "step_index": 0,
  "representation": "joint_position",
  "reference_frame": "world",
  "semantic_source": "explicit_policy_contract",
  "authoritative": true,
  "action": {
    "names": ["left_hip_pitch", "left_knee", "..."],
    "units": "radian",
    "shape": [14],
    "values": [...],
    "raw_values": [...]
  },
  "timing": {
    "infer_ms": 88.0,
    "preprocess_ms": 12.0,
    "postprocess_ms": 5.0
  },
  "safety": {
    "executable": false,
    "requires_sandbox": true,
    "sandbox_decision": null
  }
}
```

---

## 6. Processor Evidence

`policy_worker_service._handle_load_policy`:

1. Reads `type`/`policy_type` from `config.json`.
2. Calls `get_policy_class(policy_type)` and `policy_cls.from_pretrained(local_path)`.
3. Builds `preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=local_path)`.
4. Forces every `DeviceProcessorStep` to the requested device via `_move_processor_to_device` (LeRobot 0.6.x otherwise defaults to CUDA on a mixed machine).

`INFER` flow:

```
observation (flat dict)
  → _observation_to_tensor (state [1, D], images [1, C, H, W])
  → preprocessor
  → policy.select_action(model_input)
  → postprocessor
  → raw_action + processed_action returned
```

Smoke test verifies:

- `LOAD_POLICY` returns `policy_metadata.output_features.action`.
- 100 `INFER` calls succeed with stable PID.
- No NaN/Inf values.
- `RESET_SESSION` and `HEALTH` keep worker alive.

---

## 7. Body Action Mapping

### 7.1 Compatibility states

| State | Meaning | Default behavior |
|-------|---------|------------------|
| `exact` | Names, order, units, limits match | Allow |
| `convertible` | Names match, unit/scale/sign adjust | Allow with conversion |
| `partial` | Only subset of body joints covered | **Block** unless `--allow-partial` |
| `incompatible` | No overlap or unsafe limits | **Block** |
| `unknown` | Cannot resolve representation/names | **Block** |

### 7.2 Mock RH56 smoke

Test: `tests/integrations/test_lerobot_rollout_rh56_smoke.py`

- 6-DoF `joint_position` policy with names exactly matching the mock RH56 legs.
- Shadow loop runs 3 steps.
- Sandbox preflight returns `ALLOW`.
- Trace records `rollout.action.mapped` and `rollout.sandbox.decision` events.
- `hardware_actions_executed == 0`.

Incompatible mapping test:

- Policy returns `unknown_joint_*` names.
- Body mapper reports no overlap.
- Loop stops immediately with `RolloutStopReason.INCOMPATIBLE_MAPPING`.

---

## 8. Sandbox Preflight

The shadow loop builds a body action candidate and routes it through `sandbox_run` / `validate_trajectory`:

- Runs before the action would be sent to actuators.
- Decision values: `ALLOW`, `BLOCK`, `DEGRADED`.
- `BLOCK` stops the rollout (fail-closed).
- Only preflight; no hardware command is ever issued.

---

## 9. Proposal-Only / Shadow Rollout Results

### 9.1 Proposal-only

- Runs over a historical JSON observation fixture.
- Produces `rosclaw.action_proposal.v2` at each step.
- Records Practice events but does not touch hardware.

### 9.2 Shadow

- Uses live body observations from a `SenseInterface` / collector.
- Maps policy action → body action.
- Runs sandbox preflight.
- Records trace.
- `hardware_actions_executed` stays `0`.

### 9.3 Stop reasons

The rollout state machine is fail-closed for:

- Runtime failure / worker death
- Stale or missing observation
- NaN/Inf action
- Incompatible or unknown body mapping
- Sandbox `BLOCK`
- Deadline miss threshold
- `Ctrl+C` / SIGINT
- `--execute` flag (explicitly rejected)

---

## 10. Practice Events

The trace recorder emits JSONL events such as:

- `runtime.loaded`
- `session.created`
- `observation.validated`
- `inference.completed`
- `rollout.action.mapped`
- `rollout.sandbox.decision`
- `rollout.step`
- `episode.summary`

Each event includes `event_time`, `task_id`, `session_id`, `step_index`, and evidence payloads for later replay and diagnosis.

---

## 11. Dual-Runtime Test Matrix

| Test | Runtime | torch/lerobot | Result |
|------|---------|---------------|--------|
| `test_lerobot_runtime_boundary.py::test_core_module_does_not_import_torch_or_lerobot` | Python 3.11 core | absent | ✅ 13 modules pass |
| `test_lerobot_runtime_boundary.py::test_worker_module_can_import_torch_and_lerobot` | Python 3.12 worker | present | ✅ no core guard error |
| `test_lerobot_persistent_runtime_smoke.py::test_persistent_runtime_100_inferences` | Python 3.12 worker | present | ✅ pass |
| `test_lerobot_persistent_runtime_smoke.py::test_persistent_runtime_action_proposal_has_semantics` | Python 3.12 worker | present | ✅ pass (fail-closed unknown semantics) |
| `test_lerobot_rollout_rh56_smoke.py` | Python 3.11 core (mock runtime) | absent | ✅ pass |
| Unit tests under `tests/unit/integrations/lerobot/` | Python 3.11 core | absent | ✅ 63 pass |
| `tests/integrations/test_lerobot_cli.py` + other integration tests | Python 3.11 core / worker | mixed | ✅ 221 passed, 3 skipped |

Total LeRobot tests: **284 passed, 3 skipped**.

```bash
python -m pytest tests/integrations/test_lerobot_*.py tests/unit/integrations/lerobot/ -q  # 284 passed, 3 skipped
```

---

## 12. Known Limitations

1. **LeRobot 0.6.x processor migration** — older pretrained policies require a one-time migrated copy (`<policy>_migrated`). The smoke test skips if it is missing and tells the operator to migrate.
2. **CPU-only metrics** — GPU memory measurement was not available in the test environment; only CPU latency is reported.
3. **Postprocessor batch dimension** — processed action retains `[1, D]` shape because LeRobot's postprocessor is batch-preserving. The contract layer normalizes this to `[D]`; consumers see the squeezed shape.
4. **First-inference warm-up** — the very first `INFER` can be several seconds due to JIT/import side effects. `WARMUP` now absorbs this overhead before the live loop starts.
5. **Action semantics require explicit metadata** — the bridge no longer guesses representation/units/names from `policy_type`. Real-robot policies must include explicit action metadata in their config or be paired with an authoritative observation/action contract.
6. **Real robot shadow** — tested with a mock body and mock collector. A real-body shadow gate is the first P5 milestone and requires a linked robot and safe lab environment.

---

## 13. P5 Gate Status

P5 real-hardware execution may begin only after the following hard gates are closed:

| Gate | Status | Notes |
|------|--------|-------|
| Real-body shadow 1000 steps, 0 hardware actions | ⏳ pending | Requires linked ROSClaw body and lab safety procedure |
| Real policy with explicit action metadata (representation, units, names) | ⏳ pending | Current smoke policy lacks explicit tags; use body-specific policy or manifest |
| `rosclaw practice verify --strict` on shadow output | ✅ ready | `finalize_rollout_practice_session` writes full lifecycle |
| Operator emergency-stop and human oversight | ⏳ pending | Lab procedure, not code |

### P5 follow-up work

1. **Real-robot shadow gate** — run shadow mode on a linked ROSClaw body with a human-held emergency stop, verifying timing, body mapping, and 0 hardware actions.
2. **GPU memory telemetry** — add `torch.cuda.memory_summary` to `PROBE`/`HEALTH` when CUDA is available.
3. **Action chunk execution** — extend the runtime to return full action chunks and a chunk selector for temporal consistency.
4. **Online policy update** — support `UNLOAD_POLICY` + `LOAD_POLICY` hot-swap for A/B policy evaluation.
5. **Distributed worker** — add a TCP/Unix-socket variant of `PersistentRuntimeManager` so the worker can run on a GPU node separate from the ROSClaw core.
6. **Observability dashboard** — surface runtime state, per-step latency, and sandbox decisions to the practice dashboard.

---

## 14. Files Delivered

### New tests

- `tests/integrations/test_lerobot_persistent_runtime_smoke.py`
- `tests/integrations/test_lerobot_runtime_boundary.py`
- `tests/integrations/test_lerobot_rollout_rh56_smoke.py`

### Core implementation (selected)

- `src/rosclaw/integrations/lerobot/policy_runtime/manager.py`
- `src/rosclaw/integrations/lerobot/policy_runtime/protocol.py`
- `src/rosclaw/integrations/lerobot/policy_runtime/client.py`
- `src/rosclaw/integrations/lerobot/policy_runtime/state.py`
- `src/rosclaw/integrations/lerobot/policy_runtime/session.py`
- `src/rosclaw/integrations/lerobot/policy_worker_runtime.py`
- `src/rosclaw/integrations/lerobot/policy_worker_service.py`
- `src/rosclaw/integrations/lerobot/action_adapter.py`
- `src/rosclaw/integrations/lerobot/observation_adapter.py`
- `src/rosclaw/integrations/lerobot/contracts/*.py`
- `src/rosclaw/integrations/lerobot/rollout/*.py` (incl. new `practice_bridge.py`)
- `src/rosclaw/integrations/lerobot/cli.py`
- `src/rosclaw/body/action_mapping/*.py`
- `src/rosclaw/cli.py`

### Report

- `ROSCLAW_LEROBOT_P4_POLICY_RUNTIME_REPORT.md`

---

## 15. Post-Merge Verification (2026-07-16)

After the P4.1 hardening pass, the following verification was run on local `main`:

- Removed `policy_type → joint_position/radian` semantic guessing in `_extract_metadata`.
- Added `SemanticSource` and `authoritative` flags to `ActionProposalV2`; blocked proposals when representation is unknown or names are missing.
- Made `WARMUP` run a real end-to-end inference with NaN/Inf/shape validation.
- Enforced single active session (`session_capacity_exceeded`) and strict step-index monotonicity (`non_monotonic_step_index`).
- Added `worker_generation` and immediate `worker_died` propagation on worker death.
- Normalized action batches (`[1,D] → [D]`) and rejected undeclared `[N>1,D]` while preserving declared `[chunk_size,D]` chunks.
- Switched rollout loop to fixed-deadline scheduling with `deadline_miss_ms`, `overrun_count`, and `effective_control_hz`.
- Added `finalize_rollout_practice_session` to import rollout traces into the full Practice lifecycle.
- Updated CLI flags (`--strict-deadline`, `--max-deadline-misses`, `--practice-root`, `--observation` for warmup).

Verification results:

```bash
# LeRobot unit + integration regression
python -m pytest tests/integrations/test_lerobot_*.py tests/unit/integrations/lerobot/ -q  # 284 passed, 3 skipped

# Runtime boundary: core Python never imports torch/lerobot
env -u PYTHONPATH python -m pytest tests/integrations/test_lerobot_runtime_boundary.py -q  # 19 passed

# End-to-end proposal-only rollout → Practice lifecycle → strict verify
python -m rosclaw.cli lerobot rollout proposal-only \
  --policy.path /home/dell/.rosclaw/cache/lerobot/policies/lerobot_act_aloha_sim_transfer_cube_human_migrated \
  --observation-fixture examples/lerobot/sample_observation_aloha_act.json \
  --steps 100 --control-hz 10 --practice-root /tmp/rosclaw_lerobot_practice --json

python -m rosclaw.cli practice verify --strict --data-root /tmp/rosclaw_lerobot_practice <practice_id>
# Passed: True (catalog_exists, practice_record, manifest_exists, events_jsonl_exists,
#              session_record, episode_records, event_types, event_count, artifact_records)
```

Push status:

- The P4.1 changes were committed on local `main`, merged with the latest upstream
  commits, and pushed successfully to `ros-claw/rosclaw` `main` (`bd8adfd`).
- Full LeRobot unit + integration regression was re-run after the merge and remains
  green (`284 passed, 3 skipped`).

---

*Generated as part of ROSClaw × LeRobot Bridge P4-E Acceptance Hardening.*  
*Co-Authored-By: Claude <noreply@anthropic.com>*
