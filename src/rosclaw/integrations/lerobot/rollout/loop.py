"""Proposal-only and shadow rollout loops for the persistent policy runtime."""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.body.action_mapping import (
    ActionSpace,
    BodyActionSpace,
    generate_action_mapping,
    map_action_proposal_to_body,
    resolve_body_action_space,
    validate_action_mapping,
    validate_mapped_action,
)
from rosclaw.body.resolver import BodyResolver
from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.contracts import (
    ObservationContract,
    ObservationFeatureSnapshot,
    ObservationSnapshot,
    validate_observation_snapshot,
)
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.rollout.metrics import RolloutMetrics, StepTimer
from rosclaw.integrations.lerobot.rollout.observation_source import (
    BodyObservationSource,
    FixtureObservationSource,
    ObservationSource,
)
from rosclaw.integrations.lerobot.rollout.practice_bridge import finalize_rollout_practice_session
from rosclaw.integrations.lerobot.rollout.recorder import RolloutRecorder
from rosclaw.integrations.lerobot.rollout.sandbox_preflight import run_sandbox_preflight
from rosclaw.integrations.lerobot.rollout.state import (
    RolloutMode,
    RolloutResult,
    RolloutStopReason,
)


@dataclass
class RolloutConfig:
    """Configuration for a proposal-only or shadow rollout."""

    mode: RolloutMode
    policy_path: str
    robot_id: str = "rosclaw_default"
    body_id: str | None = None
    python_executable: str | None = None
    device: str = "cpu"
    dtype: str = "auto"
    allow_network: bool = False
    runtime_timeout_sec: float = 120.0
    startup_timeout_sec: float = 60.0
    revision: str = "main"

    # Observation source
    observation_fixture: str | Path | None = None
    observation_contract: ObservationContract | dict[str, Any] | None = None
    use_live_body: bool = False
    collector: str = "mock"
    scenario: str = "normal"

    # Loop control
    steps: int | None = None
    duration_sec: float | None = None
    control_hz: float = 10.0
    strict_deadline: bool = False
    max_deadline_misses: int | None = None

    # Safety / execution
    execute: bool = False
    require_exact_mapping: bool = True
    allow_partial_mapping: bool = False
    run_sandbox_preflight: bool = True
    # Optional pre-resolved body for shadow/execute loops (avoids resolver
    # lookups and monkey-patching in RH56 mock gates).
    body_override: Any | None = None
    # Optional RH56 sandbox context: {"profile": TransportProfile,
    # "calibration": RH56Calibration|None, "current_positions": list|None,
    # "max_step_delta_raw": float|None}.  When set, the RH56 range checker
    # replaces the generic humanoid MuJoCo firewall.
    rh56_context: dict[str, Any] | None = None

    # Output
    trace_path: str | Path | None = None
    task_id: str | None = None
    practice_data_root: str | Path | None = None

    tags: list[str] = field(default_factory=list)

    def is_time_limited(self) -> bool:
        return self.duration_sec is not None and self.duration_sec > 0

    def is_step_limited(self) -> bool:
        return self.steps is not None and self.steps > 0


def run_proposal_only_loop(config: RolloutConfig) -> RolloutResult:
    """Run a proposal-only rollout over a fixture observation sequence."""
    if config.observation_fixture is None:
        return _failed_result(
            config.mode,
            RolloutStopReason.OBSERVATION_REQUIRED_MISSING,
            "proposal-only mode requires --observation-fixture",
        )
    source: ObservationSource = FixtureObservationSource(config.observation_fixture)
    return _run_loop(config, source)


def run_shadow_loop(config: RolloutConfig) -> RolloutResult:
    """Run a shadow rollout using live body observations without executing."""
    source: ObservationSource = BodyObservationSource(
        robot_id=config.robot_id,
        collector=config.collector,
        scenario=config.scenario,
        contract=config.observation_contract,
    )
    try:
        return _run_loop(config, source)
    finally:
        source.close()


def _failed_result(mode: RolloutMode, reason: RolloutStopReason, message: str) -> RolloutResult:
    return RolloutResult(
        mode=mode,
        stop_reason=reason,
        errors=[message],
    )


def _run_loop(config: RolloutConfig, source: ObservationSource) -> RolloutResult:
    if config.execute:
        return _failed_result(
            config.mode,
            RolloutStopReason.RUNTIME_FAILURE,
            "P4 rollouts are proposal-only / shadow; --execute is not allowed",
        )

    result = RolloutResult(mode=config.mode, stop_reason=RolloutStopReason.COMPLETED)
    metrics = RolloutMetrics()
    trace_path = Path(
        config.trace_path
        or f"/tmp/rosclaw_lerobot_rollout_{config.mode.value}_{uuid.uuid4().hex[:8]}.jsonl"
    )

    body = None
    body_space = None
    mapping = None
    mapping_report: dict[str, Any] = {"blocked": False, "block_reasons": []}
    if config.mode == RolloutMode.SHADOW:
        body = config.body_override if config.body_override is not None else _load_body(config.body_id)
        body_space = resolve_body_action_space(body)

    contract = _resolve_contract(config.observation_contract)

    recorder = RolloutRecorder(
        trace_path=trace_path,
        robot_id=config.robot_id,
        body_id=getattr(body, "body_instance_id", None),
        policy_id=config.policy_path,
        task_id=config.task_id,
    )

    runtime = PersistentRuntimeManager(
        python_executable=config.python_executable or _default_python(),
        policy_path=config.policy_path,
        device=config.device,
        dtype=config.dtype,
        allow_network=config.allow_network,
        timeout_sec=config.runtime_timeout_sec,
        startup_timeout_sec=config.startup_timeout_sec,
    )

    try:
        state = runtime.start()
        if state.state != "ready":
            result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
            result.errors.append(f"Runtime failed to start: {state.error}")
            return result

        recorder.record_runtime_started(
            {
                "python_executable": str(runtime.python_executable),
                "policy_path": config.policy_path,
                "device": config.device,
            }
        )

        load_response = runtime.call(
            "LOAD_POLICY",
            {
                "policy_path": config.policy_path,
                "revision": config.revision,
                "device": config.device,
                "allow_network": config.allow_network,
            },
            timeout_sec=config.runtime_timeout_sec,
        )
        if load_response.get("status") != "ok":
            result.stop_reason = RolloutStopReason.POLICY_LOAD_FAILED
            result.errors.append(f"LOAD_POLICY failed: {load_response}")
            return result

        policy_metadata = load_response.get("policy_metadata", {})
        initial_worker_generation = load_response.get("worker_generation")
        policy_space = _policy_space_from_metadata(policy_metadata)
        if body_space is not None:
            mapping = generate_action_mapping(
                policy_space,
                body_space,
                allow_partial=config.allow_partial_mapping,
            )
            mapping_report = validate_action_mapping(mapping)
            if mapping_report["blocked"] and config.require_exact_mapping:
                result.stop_reason = RolloutStopReason.INCOMPATIBLE_MAPPING
                result.errors.extend(mapping_report["block_reasons"])
                return result

        session_id = f"session_{uuid.uuid4().hex[:12]}"
        create_response = runtime.call(
            "CREATE_SESSION",
            {"session_id": session_id, "body_id": getattr(body, "body_instance_id", None)},
        )
        if create_response.get("status") != "ok":
            result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
            result.errors.append(f"CREATE_SESSION failed: {create_response}")
            return result

        recorder.record_session_created(
            {"session_id": session_id, "policy_path": config.policy_path}
        )

        deadline = time.monotonic() + config.duration_sec if config.is_time_limited() else None
        step_interval = 1.0 / config.control_hz if config.control_hz > 0 else 0.0
        step_index = 0
        parent_event_id: str | None = None
        loop_start = time.monotonic()
        next_deadline = loop_start + step_interval if step_interval > 0 else None

        while True:
            if config.is_step_limited() and step_index >= config.steps:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            if source.is_exhausted(step_index):
                break

            with StepTimer() as step_timer:
                step_result = _run_step(
                    runtime=runtime,
                    source=source,
                    config=config,
                    contract=contract,
                    policy_metadata=policy_metadata,
                    mapping=mapping,
                    body_space=body_space,
                    recorder=recorder,
                    metrics=metrics,
                    session_id=session_id,
                    step_index=step_index,
                    parent_event_id=parent_event_id,
                )
            metrics.record_step(step_timer.elapsed_ms)

            if step_result is None:
                # Source exhausted
                break
            if step_result.get("stop"):
                reason = step_result.get("reason", RolloutStopReason.RUNTIME_FAILURE)
                result.stop_reason = reason
                result.errors.extend(step_result.get("errors", []))
                break

            result.proposals.append(step_result.get("proposal", {}))
            result.mapped_actions.append(step_result.get("mapping_report", {}))
            result.sandbox_decisions.append(step_result.get("sandbox", {}))
            parent_event_id = step_result.get("step_event_id")
            step_index += 1

            # Fixed-deadline scheduler.
            if next_deadline is not None:
                now = time.monotonic()
                if now > next_deadline:
                    miss_ms = (now - next_deadline) * 1000.0
                    overrun = step_timer.elapsed_ms > step_interval * 1000.0
                    metrics.record_deadline_miss(miss_ms, overrun=overrun)
                    if config.strict_deadline and (
                        config.max_deadline_misses is None
                        or metrics.deadline_misses > config.max_deadline_misses
                    ):
                        result.stop_reason = RolloutStopReason.DEADLINE_MISS
                        result.errors.append(
                            f"Deadline miss at step {step_index}: {miss_ms:.1f} ms"
                        )
                        break
                sleep_time = max(0.0, next_deadline - time.monotonic())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_deadline += step_interval

        elapsed_total = time.monotonic() - loop_start
        result.steps_completed = step_index
        if elapsed_total > 0 and result.steps_completed > 0:
            metrics.effective_control_hz = result.steps_completed / elapsed_total
        else:
            metrics.effective_control_hz = 0.0
    except KeyboardInterrupt:
        result.stop_reason = RolloutStopReason.INTERRUPTED
        result.errors.append("Interrupted by operator")
    except Exception as exc:  # noqa: BLE001
        result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
        result.errors.append(f"Rollout exception: {exc}")
    finally:
        # P5: track worker restarts across the rollout (gate requires 0).
        with contextlib.suppress(Exception):
            health = runtime.call("HEALTH", {}, timeout_sec=5.0)
            final_generation = health.get("worker_generation")
            initial_generation = locals().get("initial_worker_generation")
            if initial_generation is not None and final_generation is not None:
                metrics.worker_restart_count = max(
                    0, int(final_generation) - int(initial_generation)
                )
        with contextlib.suppress(Exception):
            runtime.call("CLOSE_SESSION", {"session_id": session_id}, timeout_sec=10.0)
        runtime.stop()
        result.metrics = metrics.to_dict()
        result.trace_path = str(trace_path)
        result.hardware_actions_executed = 0
        if config.practice_data_root is not None:
            try:
                result.practice_id = finalize_rollout_practice_session(
                    trace_path,
                    result.to_dict(),
                    data_root=config.practice_data_root,
                )
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"Practice finalize failed: {exc}")
        recorder.record_summary(result.to_dict())

    return result


def _run_step(
    runtime: PersistentRuntimeManager,
    source: ObservationSource,
    config: RolloutConfig,
    contract: ObservationContract | None,
    policy_metadata: dict[str, Any],
    mapping: Any,
    body_space: BodyActionSpace,
    recorder: RolloutRecorder,
    metrics: RolloutMetrics,
    session_id: str,
    step_index: int,
    parent_event_id: str | None,
) -> dict[str, Any] | None:
    raw_observation = source.get_observation(step_index)
    if raw_observation is None:
        return None

    snapshot = _build_snapshot(raw_observation, step_index, session_id)
    validation_report = {"status": "ok", "errors": []}
    if contract is not None:
        validation = validate_observation_snapshot(contract, snapshot)
        validation_report = validation.to_dict()
        if validation.status == "blocked":
            metrics.observation_validation_failures += 1
            recorder.record_observation_failed(
                snapshot.to_dict(),
                validation_report,
                parent_event_id=parent_event_id,
                frame_id=str(step_index),
            )
            return {
                "stop": True,
                "reason": RolloutStopReason.OBSERVATION_REQUIRED_MISSING,
                "errors": [e["message"] for e in validation_report["errors"]],
            }

    obs_event_id = recorder.record_observation_validated(
        snapshot.to_dict(),
        validation_report,
        parent_event_id=parent_event_id,
        frame_id=str(step_index),
    )

    try:
        worker_obs = adapt_observation_for_worker(raw_observation, contract=contract)
    except Exception as exc:  # noqa: BLE001
        return {
            "stop": True,
            "reason": RolloutStopReason.OBSERVATION_REQUIRED_MISSING,
            "errors": [f"Observation adaptation failed: {exc}"],
        }

    with StepTimer() as infer_timer:
        infer_response = runtime.call(
            "INFER",
            {
                "session_id": session_id,
                "observation": worker_obs,
                "step_index": step_index,
            },
            timeout_sec=config.runtime_timeout_sec,
        )
    metrics.record_inference(infer_timer.elapsed_ms)

    if infer_response.get("status") != "ok":
        return {
            "stop": True,
            "reason": RolloutStopReason.RUNTIME_FAILURE,
            "errors": [f"INFER failed: {infer_response}"],
        }

    processed_action = infer_response.get("processed_action", {})
    proposal = adapt_action_to_proposal(
        processed_action,
        policy_path=config.policy_path,
        policy_metadata=policy_metadata,
        session_id=session_id,
        step_index=step_index,
        proposal_id=f"proposal_{session_id}_{step_index}",
        runtime_id=infer_response.get("runtime_id"),
        timing=infer_response.get("timing", {}),
    )
    recorder.record_inference(
        proposal,
        infer_timer.elapsed_ms,
        parent_event_id=obs_event_id,
        frame_id=str(step_index),
    )

    with StepTimer() as map_timer:
        if mapping is not None:
            mapped = map_action_proposal_to_body(proposal, mapping)
            mapping_report = validate_mapped_action(mapped)
        else:
            mapped = None
            mapping_report = {
                "status": "skipped",
                "blocked": False,
                "issues": [],
                "mapped_values": [],
            }
    metrics.record_mapping(map_timer.elapsed_ms)

    recorder.record_action_mapping(
        proposal,
        mapping_report=validate_action_mapping(mapping) if mapping is not None else mapping_report,
        mapped_action=mapping_report,
        latency_ms=map_timer.elapsed_ms,
        parent_event_id=obs_event_id,
        frame_id=str(step_index),
    )

    if mapped is not None and (mapped.blocked or mapping_report["blocked"]):
        metrics.mapping_blocks += 1
        if any("NaN" in issue or "Inf" in issue or "Invalid value" in issue for issue in mapping_report["issues"]):
            metrics.nan_inf_blocks += 1
            return {
                "stop": True,
                "reason": RolloutStopReason.NAN_INF,
                "errors": mapping_report["issues"],
            }
        return {
            "stop": True,
            "reason": RolloutStopReason.INCOMPATIBLE_MAPPING,
            "errors": mapping_report["issues"],
        }

    sandbox: dict[str, Any] = {}
    if config.run_sandbox_preflight and mapping is not None and mapped is not None:
        with StepTimer() as sandbox_timer:
            sandbox = run_sandbox_preflight(
                mapped,
                body_space,
                robot_id=config.robot_id,
                rh56_context=config.rh56_context,
            )
        metrics.record_sandbox(sandbox_timer.elapsed_ms)
        recorder.record_sandbox_decision(
            mapping_report,
            sandbox,
            sandbox_timer.elapsed_ms,
            parent_event_id=obs_event_id,
            frame_id=str(step_index),
        )
        if not sandbox.get("is_safe"):
            metrics.sandbox_blocks += 1
            return {
                "stop": True,
                "reason": RolloutStopReason.SANDBOX_BLOCK,
                "errors": [f"Sandbox blocked: {sandbox.get('reason', '')}"],
            }
    elif config.run_sandbox_preflight:
        sandbox = {
            "is_safe": True,
            "decision": "ALLOW",
            "reason": "proposal-only mode: no body mapping",
        }
        recorder.record_sandbox_decision(
            mapping_report,
            sandbox,
            0.0,
            parent_event_id=obs_event_id,
            frame_id=str(step_index),
        )

    step_event_id = recorder.record_step(
        step_index,
        {
            "mode": config.mode.value,
            "session_id": session_id,
            "proposal_id": proposal.get("proposal_id"),
            "sandbox_decision": sandbox.get("decision"),
            "hardware_executed": False,
        },
        parent_event_id=obs_event_id,
    )

    return {
        "stop": False,
        "proposal": proposal,
        "mapping_report": mapping_report,
        "sandbox": sandbox,
        "step_event_id": step_event_id,
    }


def _load_body(body_id: str | None) -> Any:
    resolver = BodyResolver()
    return resolver.get_effective_body()


def _resolve_contract(contract: ObservationContract | dict[str, Any] | None) -> ObservationContract | None:
    if contract is None:
        return None
    if isinstance(contract, dict):
        return ObservationContract.from_dict(contract)
    return contract


def _build_snapshot(
    observation: dict[str, Any],
    step_index: int,
    session_id: str,
) -> ObservationSnapshot:
    values = observation.get("observation.state", [])
    names = observation.get("state_names")
    if names is None and "state" in observation and isinstance(observation["state"], dict):
        names = list(observation["state"].keys())
    features = {
        "observation.state": ObservationFeatureSnapshot(
            valid=True,
            values=list(values),
            names=list(names) if names else None,
        )
    }
    # P5: record RH56 feedback channels as observation features so the
    # Practice trace carries force/current/temperature/status evidence.
    for key in (
        "observation.force",
        "observation.current",
        "observation.temperature",
        "observation.status",
    ):
        channel_values = observation.get(key)
        if isinstance(channel_values, list) and channel_values:
            features[key] = ObservationFeatureSnapshot(
                valid=True,
                values=list(channel_values),
                names=list(names) if names else None,
            )
    return ObservationSnapshot(
        snapshot_id=f"obs_{session_id}_{step_index}",
        session_id=session_id,
        captured_at_monotonic_ns=time.monotonic_ns(),
        features=features,
    )


def _policy_space_from_metadata(metadata: dict[str, Any]) -> ActionSpace:
    """Build an ActionSpace from policy output_features metadata."""
    output_features = metadata.get("output_features", {})
    action_feature = output_features.get("action", {})
    shape = list(action_feature.get("shape", []))
    names = action_feature.get("names") or metadata.get("extra", {}).get("action_names") or []
    # P4.1: do not synthesize generic action_* names; unknown semantics must stay unknown.
    units = action_feature.get("unit") or metadata.get("extra", {}).get("action_unit") or "unknown"
    representation = (
        action_feature.get("representation")
        or metadata.get("extra", {}).get("action_representation")
        or "unknown"
    )
    return ActionSpace(
        representation=str(representation),
        names=[str(n) for n in names],
        units=[str(units)] * len(names) if isinstance(units, str) else [str(u) for u in units],
        is_chunked=len(shape) == 2 and shape[0] > 1,
        chunk_size=shape[0] if len(shape) == 2 else None,
    )


def _default_python() -> str:
    import sys

    from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
    from rosclaw.integrations.lerobot.runtime import find_python312

    # Prefer the configured LeRobot runtime (has torch/lerobot installed);
    # a bare python3.12 from PATH would crash the worker on import.
    configured = get_configured_lerobot_runtime()
    if configured and configured.get("python_executable"):
        return str(configured["python_executable"])
    # Fallback: a conventional worker venv next to the repo, then any py3.12.
    repo_venv = Path(__file__).resolve().parents[5] / ".venv-lerobot" / "bin" / "python"
    if repo_venv.exists():
        return str(repo_venv)
    found = find_python312()
    return str(found) if found else sys.executable
