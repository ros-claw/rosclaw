"""RH56 execute loop (P5-C): receding-horizon single-step execution.

Mirrors :func:`run_rh56_shadow` but every step goes through the full
execution pipeline::

    observe → infer → map → sandbox → permit → execute one step
      → feedback verify → Practice events

Hardware execution only happens through the injected ``BodyExecutor``; with
:class:`MockModbusTransport` this validates the complete loop without any
physical device.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from rosclaw.body.action_mapping import (
    generate_action_mapping,
    map_action_proposal_to_body,
    resolve_body_action_space,
    validate_action_mapping,
    validate_mapped_action,
)
from rosclaw.body.execution.rh56_executor import RH56Executor
from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.mock_body import build_mock_rh56_body
from rosclaw.body.rh56.sandbox import run_rh56_sandbox_preflight
from rosclaw.body.rh56.transport import MockModbusTransport, RH56Transport, TransportIOError
from rosclaw.body.rh56.transport_profile import (
    load_transport_profile,
    validate_transport_binding,
)
from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.execution import (
    ArmingController,
    ExecutionReport,
    ExecutionState,
    FeedbackVerifier,
    PermitManager,
    SingleStepExecutor,
)
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.rollout.loop import _build_snapshot
from rosclaw.integrations.lerobot.rollout.practice_bridge import (
    finalize_rollout_practice_session,
)
from rosclaw.integrations.lerobot.rollout.recorder import RolloutRecorder
from rosclaw.integrations.lerobot.rollout.rh56_observation_source import (
    RH56ObservationSource,
)
from rosclaw.integrations.lerobot.rollout.state import RolloutMode, RolloutResult, RolloutStopReason


def run_rh56_execute(
    *,
    policy_path: str,
    transport_profile_path: str | Path,
    permit_id: str,
    permit_manager: PermitManager,
    arming: ArmingController,
    transport: RH56Transport | None = None,
    calibration_path: str | Path | None = None,
    body: Any | None = None,
    task: str = "hold_current",
    steps: int = 3,
    control_hz: float = 2.0,
    speed: int = 100,
    force_limit_g: float = 100.0,
    settle_ms: float = 0.0,
    practice_data_root: str | Path | None = None,
    trace_path: str | Path | None = None,
    python_executable: str | None = None,
    robot_id: str = "rh56_mock",
) -> tuple[RolloutResult, ExecutionReport]:
    """Run a receding-horizon single-step execution loop."""
    profile = load_transport_profile(transport_profile_path)
    calibration = (
        load_rh56_calibration(calibration_path) if calibration_path is not None else None
    )
    if body is None:
        body = build_mock_rh56_body(profile)
    validate_transport_binding(
        profile,
        action_dim=len(profile.action_order),
        action_names=list(profile.action_order),
    )

    if transport is None:
        transport = MockModbusTransport(profile)
    if not transport.is_connected():
        transport.connect()

    result = RolloutResult(mode=RolloutMode.EXECUTE, stop_reason=RolloutStopReason.COMPLETED)
    report = ExecutionReport(body_id=getattr(body, "body_instance_id", robot_id), task=task)
    trace = Path(trace_path or f"/tmp/rosclaw_rh56_execute_{uuid.uuid4().hex[:8]}.jsonl")
    recorder = RolloutRecorder(
        trace_path=trace,
        robot_id=robot_id,
        body_id=getattr(body, "body_instance_id", None),
        policy_id=policy_path,
        task_id=task,
    )

    def _event_sink(event_type: str, payload: dict[str, Any]) -> None:
        report.record_event(event_type, payload)
        recorder._emit(
            "runtime" if not event_type.startswith("execution.command") else "provider",
            event_type,
            payload,
        )

    executor = RH56Executor(transport, profile)
    step_executor = SingleStepExecutor(
        executor=executor,
        profile=profile,
        permit_manager=permit_manager,
        arming=arming,
        verifier=FeedbackVerifier(profile, calibration),
        event_sink=_event_sink,
    )

    runtime = PersistentRuntimeManager(
        python_executable=python_executable or _default_python(),
        policy_path=policy_path,
        device="cpu",
    )
    body_space = resolve_body_action_space(body)
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    source = RH56ObservationSource(transport, profile, task=task)

    try:
        permit = permit_manager.get(permit_id)
        if permit is None:
            result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
            result.errors.append(f"permit {permit_id} not active or revoked")
            return result, report
        _event_sink("execution.armed", {"permit_id": permit_id, "task": task})

        hashes = {
            "policy_contract_hash": permit.policy_contract_hash,
            "body_hash": permit.body_hash,
            "calibration_hash": permit.calibration_hash,
            "mapping_hash": permit.mapping_hash,
            "transport_profile_hash": permit.transport_profile_hash,
        }

        state = runtime.start()
        if state.state != "ready":
            result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
            result.errors.append(f"runtime failed: {state.error}")
            return result, report

        load_response = runtime.call(
            "LOAD_POLICY",
            {"policy_path": policy_path, "device": "cpu", "allow_network": False},
        )
        if load_response.get("status") != "ok":
            result.stop_reason = RolloutStopReason.POLICY_LOAD_FAILED
            result.errors.append(f"LOAD_POLICY failed: {load_response}")
            return result, report
        policy_metadata = load_response.get("policy_metadata", {})

        mapping = generate_action_mapping(_policy_space(policy_metadata), body_space)
        mapping_report = validate_action_mapping(mapping)
        if mapping_report["blocked"]:
            result.stop_reason = RolloutStopReason.INCOMPATIBLE_MAPPING
            result.errors.extend(mapping_report["block_reasons"])
            return result, report

        runtime.call("CREATE_SESSION", {"session_id": session_id})
        recorder.record_session_created({"session_id": session_id, "mode": "execute"})

        step_interval = 1.0 / control_hz if control_hz > 0 else 0.0
        loop_start = time.monotonic()

        for step_index in range(steps):
            obs_ns = time.monotonic_ns()
            try:
                raw_observation = source.get_observation(step_index)
            except TransportIOError as exc:
                # Observation channel failed mid-execution: estop, fault, stop.
                step_executor.emergency_stop()
                _event_sink(
                    "execution.communication_lost",
                    {"reason": f"observation read failed: {exc}", "step": step_index},
                )
                arming.fault(ExecutionState.COMMUNICATION_LOST, str(exc))
                permit_manager.revoke(permit_id, "communication_lost")
                result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
                result.errors.append(f"communication_lost: {exc}")
                break
            if raw_observation is None:
                break
            snapshot = _build_snapshot(raw_observation, step_index, session_id)
            recorder.record_observation_validated(
                snapshot.to_dict(),
                {"status": "ok", "errors": []},
                frame_id=str(step_index),
            )

            worker_obs = adapt_observation_for_worker(raw_observation)
            infer_response = runtime.call(
                "INFER",
                {"session_id": session_id, "observation": worker_obs, "step_index": step_index},
            )
            if infer_response.get("status") != "ok":
                result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
                result.errors.append(f"INFER failed: {infer_response}")
                break

            proposal = adapt_action_to_proposal(
                infer_response.get("processed_action", {}),
                policy_path=policy_path,
                policy_metadata=policy_metadata,
                session_id=session_id,
                step_index=step_index,
                proposal_id=f"proposal_{session_id}_{step_index}",
            )
            recorder.record_inference(proposal, 0.0, frame_id=str(step_index))
            result.proposals.append(proposal)

            mapped = map_action_proposal_to_body(proposal, mapping)
            mapped_report = validate_mapped_action(mapped)
            recorder.record_action_mapping(
                proposal,
                mapping_report=mapping_report,
                mapped_action=mapped_report,
                latency_ms=0.0,
                frame_id=str(step_index),
            )
            if mapped.blocked or mapped_report["blocked"]:
                result.stop_reason = RolloutStopReason.INCOMPATIBLE_MAPPING
                result.errors.extend(mapped_report["issues"])
                break

            sandbox = run_rh56_sandbox_preflight(
                mapped,
                body_space,
                profile=profile,
                calibration=calibration,
                current_positions=[float(v) for v in raw_observation["observation.state"]],
                max_step_delta_raw=permit.max_step_delta_raw,
            )
            recorder.record_sandbox_decision(
                mapped_report, sandbox, 0.0, frame_id=str(step_index)
            )
            result.sandbox_decisions.append(sandbox)
            if not sandbox.get("is_safe"):
                _event_sink(
                    "execution.step.blocked",
                    {"proposal_id": proposal["proposal_id"], "sandbox": sandbox},
                )
                result.stop_reason = RolloutStopReason.SANDBOX_BLOCK
                result.errors.append(f"Sandbox blocked: {sandbox.get('reason')}")
                break

            # Execute exactly one step.
            exec_result = step_executor.execute_candidate(
                permit_id=permit_id,
                proposal_id=proposal["proposal_id"],
                names=list(proposal["action"]["names"]),
                values=list(proposal["action"]["values"]),
                representation=proposal["representation"],
                units=proposal["action"]["units"],
                hashes=hashes,
                speed=speed,
                force_limit_g=force_limit_g,
                observation_timestamp_ns=obs_ns,
                settle_ms=settle_ms,
            )
            report.record_result(exec_result)
            if exec_result.status in {"fault", "aborted"}:
                result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
                result.errors.append(exec_result.message or exec_result.status)
                break

            result.steps_completed = step_index + 1

            # Fixed-deadline scheduling.
            if step_interval > 0:
                deadline = loop_start + (step_index + 1) * step_interval
                sleep_time = deadline - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        report.final_state = arming.machine.state.value
    except KeyboardInterrupt:
        step_executor.emergency_stop()
        result.stop_reason = RolloutStopReason.INTERRUPTED
        result.errors.append("operator abort (Ctrl+C)")
    except Exception as exc:  # noqa: BLE001
        result.stop_reason = RolloutStopReason.RUNTIME_FAILURE
        result.errors.append(f"execute loop exception: {exc}")
    finally:
        # Evidence must survive every exit path, including mid-loop faults:
        # commands already sent count as executed hardware actions.
        result.hardware_actions_executed = step_executor.hardware_actions_executed
        report.final_state = arming.machine.state.value
        try:
            runtime.call("CLOSE_SESSION", {"session_id": session_id}, timeout_sec=10.0)
        except Exception:  # noqa: BLE001
            pass
        runtime.stop()
        source.close()
        result.trace_path = str(trace)
        recorder.record_summary(result.to_dict())
        if practice_data_root is not None:
            try:
                result.practice_id = finalize_rollout_practice_session(
                    trace,
                    result.to_dict(),
                    data_root=practice_data_root,
                )
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"Practice finalize failed: {exc}")

    return result, report


def _policy_space(metadata: dict[str, Any]):
    from rosclaw.body.action_mapping import ActionSpace

    action_feature = metadata.get("output_features", {}).get("action", {})
    names = action_feature.get("names") or []
    units = action_feature.get("unit") or "unknown"
    representation = action_feature.get("representation") or "unknown"
    return ActionSpace(
        representation=str(representation),
        names=[str(n) for n in names],
        units=[str(units)] * len(names),
        is_chunked=False,
    )


def _default_python() -> str:
    import sys

    from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
    from rosclaw.integrations.lerobot.runtime import find_python312

    configured = get_configured_lerobot_runtime()
    if configured and configured.get("python_executable"):
        return str(configured["python_executable"])
    repo_venv = Path(__file__).resolve().parents[5] / ".venv-lerobot" / "bin" / "python"
    if repo_venv.exists():
        return str(repo_venv)
    found = find_python312()
    return str(found) if found else sys.executable
