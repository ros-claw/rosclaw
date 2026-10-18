"""New D: bounded current five-mechanism/application compilers, zero model calls.

No legacy G5 episode or scripted replay is changed. Original pilot entry and
common facts gate every dispatch; only the injected existing SDK can validate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from benchmarks.ros_expert_ablation.rsk_application_aci import ApplicationPhysicalACI
from benchmarks.ros_expert_ablation.rsk_episode_runner import (
    FrozenProviderV1,
    _provider_pin,
    _read,
)
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import review_pilot_entry
from benchmarks.ros_expert_ablation.rsk_registered_inputs import registered_episode_inputs
from rosclaw.contracts.common import content_hash


def current_deterministic_request(aci):
    """Only current observed signatures/options; never read a gold case label."""
    base = aci.base
    candidate = base.case.derived_documents["navigation_observation_review"]["candidate"]
    if isinstance(aci, ApplicationPhysicalACI) and not candidate["interfaces"].get(
        "navigation.navigate_to_pose"
    ):
        return {
            "operation": "ros_engineer.propose_application_bootstrap",
            "arguments": {"case_hash": base.case.case_hash},
        }
    compiled = base.conditional_staging_options()
    options = compiled["options"]
    if options:
        if len(options) > 5 or len({option.issue_id for option in options}) != len(options):
            raise ValueError("ambiguous or over-budget deterministic options; abstain")
        return {
            "operation": "ros_engineer.select_staging_options",
            "arguments": {
                "case_hash": base.case.case_hash,
                "option_refs": [option.option_ref for option in options],
            },
        }
    if any(c.status != "SATISFIED" for c in base.case.constraints):
        return None
    names = {}
    for role, key in [
        ("mapping.occupancy_map", "mapping"),
        ("state.odometry", "odometry"),
        ("sensing.lidar", "lidar"),
        ("navigation.navigate_to_pose", "navigate_to_pose"),
    ]:
        observations = candidate["interfaces"].get(role, [])
        if len(observations) != 1:
            return None
        names[key] = observations[0]["name"]
    return {
        "operation": "ros_engineer.propose_onboarding",
        "arguments": {"case_hash": base.case.case_hash, "interfaces": names},
    }


def run_rsk_deterministic_episode(
    task,
    aci,
    output,
    *,
    protocol,
    pilot_evidence,
    evidence_root,
    ordinal,
    facts_manifest,
    evaluation_episode,
    provider_settings,
    memory_policy,
):
    """One registered D allocation, zero model calls/probes, at most one validation."""
    if not isinstance(aci, RskPhysicalACI) or aci.arm != "B" or type(ordinal) is not int:
        raise ValueError("D requires a fresh no-Memory/no-probe host compiler interface")
    base = aci.base
    if aci.memory_query is not None or base.probe_evaluator is not None:
        raise ValueError("D cannot use Memory or active probe callbacks")
    root, output = Path(evidence_root).resolve(strict=True), Path(output).resolve()
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = {
        "schema_version": "rosclaw.rsk_deterministic_episode.v1",
        "status": "NO_GO",
        "protocol_hash": protocol.protocol_hash,
        "harness_source": protocol.harness_source,
        "actual_model_calls": 0,
        "actual_model_client_invocations": 0,
        "model_decisions": 0,
        "read_only_probe_Worlds": 0,
        "new_Worlds_requested": 0,
        "verified_success": False,
        "physical_validation_phase": "NOT_REQUESTED",
        "current_authority": "NONE",
        "old_G5": "UNCHANGED",
        "causal_Memory_benefit": "NOT_ESTABLISHED",
    }
    calls = []
    provider = None

    def bound():
        entry = review_pilot_entry(protocol, pilot_evidence, root=root)
        if entry["status"] != "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY":
            raise ValueError("original pilot entry is NO_GO; no compiler or SDK dispatch")
        provider = FrozenProviderV1.model_validate_json(_read(root, provider_settings))
        actual = provider.execution_kind == "ACTUAL_MODEL_CLIENT"
        allocation, _ = registered_episode_inputs(
            task,
            aci,
            protocol=protocol,
            root=root,
            ordinal=ordinal,
            arm="D",
            facts_manifest=facts_manifest,
            evaluation_episode=evaluation_episode,
            provider_settings=provider_settings,
            memory_policy=memory_policy,
            require_complete_originals=actual,
        )
        if not actual and (base.evaluator is not None or base.probe_evaluator is not None):
            raise ValueError("synthetic D contracts cannot dispatch SDK callbacks")
        if actual:
            # Verify pinned provider/source provenance without invoking it.
            _provider_pin(provider, protocol, contract_client=None)
            if base.clock is not time.monotonic:
                raise ValueError("actual D validation requires the real host monotonic clock")
        return entry, provider, allocation

    try:
        entry, provider, allocation = bound()
        if (
            base.proposals
            or base.results
            or base.validation_count
            or base.probe_count
            or base.finished
            or aci.trace
        ):
            raise ValueError("fresh episode-scoped deterministic interface required")
        result.update(
            entry_review=entry,
            allocation=allocation,
            execution_kind=provider.execution_kind,
            common_observation_hash=base.common_evidence.catalog()["common_observation_hash"],
        )
        request = current_deterministic_request(aci)
        if request is None:
            result["status"] = "ABSTAIN_NO_SUPPORTED_CURRENT_DETERMINISTIC_OPTION"
        else:
            bound()
            proposed = aci.call(
                request, remaining_sec=max(0.001, 300 - (time.monotonic() - started))
            )
            calls.append({"request": request, "result": proposed})
            if "proposal" not in proposed:
                result["status"] = "DETERMINISTIC_COMPILER_ABSTAINED"
            else:
                result.update(
                    status="CONDITIONAL_DETERMINISTIC_PROPOSAL_NOT_EXECUTED",
                    first_valid_proposal={
                        "model_decisions": 0,
                        "proposal_id": proposed["proposal"]["proposal_id"],
                    },
                )
                application = proposed["proposal"]["proposal_id"] in getattr(
                    aci, "application_proposals", {}
                )
                callback = aci.application_validator if application else base.evaluator
                if callback is not None:
                    bound()
                    remaining = 300 - (time.monotonic() - started)
                    if remaining <= 0:
                        result["status"] = "WALL_TIME_BUDGET_EXHAUSTED"
                    else:
                        request = {
                            "operation": "ros_engineer.validate_proposal",
                            "arguments": {
                                "case_hash": base.case.case_hash,
                                "proposal_id": proposed["proposal"]["proposal_id"],
                            },
                        }
                        calls.append({"request": request, "status": "HOST_DISPATCH_STARTED"})
                        verified = aci.call(request, remaining_sec=remaining)
                        calls[-1].update(status="HOST_DISPATCH_COMPLETED", result=verified)
                        result["physical_validation_phase"] = (
                            "FAILED_OR_INCOMPLETE_OR_OUT_OF_BUDGET"
                        )
                        if (
                            provider.execution_kind == "ACTUAL_MODEL_CLIENT"
                            and any(
                                v.get("accepted_success") is True for v in base.results.values()
                            )
                            and time.monotonic() - started <= 300
                        ):
                            result.update(
                                status="VERIFIED_CANONICAL_SIM_SUCCESS",
                                verified_success=True,
                                physical_validation_phase="VERIFIED_SUCCESS",
                            )
                        else:
                            result["status"] = "SDK_RESULT_UNACCEPTED_OR_INCOMPLETE"
        if base.validation_count > 1 or base.probe_count:
            raise ValueError("D independent World budget violated")
    except Exception as error:
        if calls and calls[-1].get("status") == "HOST_DISPATCH_STARTED":
            calls[-1].update(
                status="HOST_TOOL_CALLBACK_EXCEPTION_NO_AUTOMATIC_RETRY",
                error=str(error),
                error_type=type(error).__name__,
            )
        result.update(
            status="NO_GO" if not calls else "INVALID_OR_CHANGED_DETERMINISTIC_INPUT",
            error=str(error),
            error_type=type(error).__name__,
        )
    result.update(
        actual_elapsed_sec=time.monotonic() - started,
        tool_calls=calls,
        new_Worlds_requested=base.validation_count,
        verification_records=base.results,
        artifact_scope="SYNTHETIC_CONTRACT_ONLY_NOT_MODEL_OR_PHYSICAL_EVIDENCE"
        if provider and provider.execution_kind == "SYNTHETIC_CONTRACT_CLIENT"
        else "REGISTERED_DETERMINISTIC_EPISODE_OR_NO_GO",
    )
    if base.validation_count and not result["verified_success"]:
        result["physical_validation_phase"] = "FAILED_OR_INCOMPLETE_OR_OUT_OF_BUDGET"
    result["result_hash"] = content_hash("rskdepisode", result)
    with (output / "original-rsk-D-episode-result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result
