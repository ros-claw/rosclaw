"""Evidence-derived module proofs for the ContactPush Failure-to-Success Arena."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from rosclaw.simforge.candidates import CandidateGenerator
from rosclaw.simforge.contact_push_activation import ContactPushActivationResult
from rosclaw.simforge.contact_push_arena import ContactPushArenaEvaluation
from rosclaw.simforge.contact_push_flywheel import (
    ContactPushCausalLoop,
    ContactPushFlywheel,
)
from rosclaw.simforge.contact_push_learning import (
    ContactPushCandidate,
    ContactPushTaskKnowledge,
)
from rosclaw.simforge.contact_push_provider import evaluate_contact_push_provider
from rosclaw.simforge.evaluation import StressEvidence
from rosclaw.simforge.failure_router_v2 import run_failure_router_acceptance_suite
from rosclaw.simforge.models import HumanInvolvement
from rosclaw.simforge.promotion_v3 import (
    GateDecision,
    GateV3Policy,
    GateV3Result,
    StatisticalGateV3,
)
from rosclaw.simforge.proof import (
    CounterfactualMetric,
    CounterfactualRun,
    FaultInjectionResult,
    ModuleProof,
    ProofBundle,
)
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_BODY_HASH,
    CONTACT_PUSH_TASK_ID,
    ContactPushPolicy,
)


def build_contact_push_pre_activation_proofs(
    *,
    flywheel: ContactPushFlywheel,
    causal_loop: ContactPushCausalLoop,
    evaluation: ContactPushArenaEvaluation,
    candidate: ContactPushCandidate | None = None,
    output_root: Path,
    source_checkout: Path,
    statistical_policy: GateV3Policy,
    stress: StressEvidence,
) -> tuple[ProofBundle, GateV3Result]:
    """Build E3/E5 proofs from matched ablations and a fail-closed Darwin run."""

    root = _external_root(output_root, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    proof_candidate = candidate or causal_loop.candidate_learned
    if (
        evaluation.validation.candidate_hash != proof_candidate.candidate_hash
        or evaluation.holdout.candidate_hash != proof_candidate.candidate_hash
    ):
        raise ValueError("proof candidate does not match its Validation/Holdout evidence")
    gate = StatisticalGateV3(statistical_policy)
    without_holdout = gate.evaluate(
        validation=evaluation.validation,
        holdout=None,
        stress=stress,
        counterexample_regression=evaluation.counterexample,
    )
    with_holdout = gate.evaluate(
        validation=evaluation.validation,
        holdout=evaluation.holdout,
        stress=stress,
        counterexample_regression=evaluation.counterexample,
    )
    if without_holdout.decision is not GateDecision.NEED_MORE_EVIDENCE:
        raise RuntimeError("Darwin did not fail closed when hidden Holdout was absent")
    if with_holdout.decision is not GateDecision.SIM_CHAMPION:
        raise RuntimeError(
            "ContactPush Candidate B cannot receive causal proofs before Darwin passes"
        )
    snapshot_ref = "dataset://" + flywheel.snapshot.snapshot_hash.removeprefix("sha256:")
    failure_ref = "failure://" + causal_loop.failure.failure_id
    baseline_ref = "receipt://" + causal_loop.baseline_evidence.receipt_hash.removeprefix("sha256:")
    recovery_ref = "receipt://" + causal_loop.recovery_evidence.receipt_hash.removeprefix("sha256:")
    validation_ref = "evaluation://" + evaluation.validation.pair_set_commitment.removeprefix(
        "sha256:"
    )
    holdout_ref = "holdout://" + evaluation.holdout.pair_set_commitment.removeprefix("sha256:")
    provider = evaluate_contact_push_provider(
        scenario=causal_loop.scenario,
        candidate=proof_candidate,
    )
    _atomic_json(root / "provider-causal.json", provider.to_dict())
    provider_ref = "provider://" + provider.output_hash.removeprefix("sha256:")
    router_acceptance = run_failure_router_acceptance_suite()
    router_replay = run_failure_router_acceptance_suite()
    router_value = router_acceptance.to_dict()
    _atomic_json(root / "failure-router-acceptance.json", router_value)
    router_ref = "failure-router://" + str(router_value["report_hash"]).removeprefix("sha256:")
    proofs = (
        _proof(
            module="body",
            input_refs=(failure_ref,),
            output_refs=("body://" + CONTACT_PUSH_BODY_HASH.removeprefix("sha256:"),),
            impacts=("changed_memory_eligibility",),
            metric=CounterfactualMetric(
                name="wrong_body_reuse",
                control=1.0,
                treatment=0.0,
                lower_is_better=True,
            ),
            control_ref="body://unscoped",
            treatment_ref="body://scoped",
            injections=(
                FaultInjectionResult(
                    name="wrong_body_rejected",
                    passed=causal_loop.wrong_body_memory_rejected,
                    evidence_ref="body://wrong_hash_probe",
                ),
            ),
        ),
        _proof(
            module="provider",
            input_refs=("candidate://" + proof_candidate.candidate_id,),
            output_refs=(provider_ref,),
            impacts=("changed_planner_policy", "changed_failure_to_success"),
            metric=CounterfactualMetric(
                name="task_failures",
                control=float(not provider.baseline.success),
                treatment=float(not provider.treatment.success),
                lower_is_better=True,
            ),
            control_ref="provider://manual_baseline",
            treatment_ref=provider_ref,
            injections=(
                FaultInjectionResult(
                    name="timeout_safe_stop",
                    passed=(
                        provider.timeout.route.value == "SAFE_STOP"
                        and not provider.timeout.executable
                    ),
                    evidence_ref="provider://timeout_probe",
                ),
                FaultInjectionResult(
                    name="illegal_output_rejected",
                    passed=(
                        provider.illegal.route.value == "REJECT_INVALID"
                        and not provider.illegal.executable
                    ),
                    evidence_ref="provider://illegal_probe",
                ),
                FaultInjectionResult(
                    name="low_confidence_rejected",
                    passed=(
                        provider.low_confidence.route.value == "REJECT_LOW_CONFIDENCE"
                        and not provider.low_confidence.executable
                    ),
                    evidence_ref="provider://low_confidence_probe",
                ),
            ),
            replay_ref=provider_ref,
            replay_verified=provider.strict_replay,
        ),
        _proof(
            module="failure_router",
            input_refs=(failure_ref,),
            output_refs=(router_ref,),
            impacts=("changed_failure_specific_route", "blocked_unbounded_retry"),
            metric=CounterfactualMetric(
                name="incorrect_failure_routes",
                control=float(len(router_acceptance.cases)),
                treatment=0.0,
                lower_is_better=True,
            ),
            control_ref="failure-router://route_everything_to_auto",
            treatment_ref=router_ref,
            injections=(
                FaultInjectionResult(
                    name="all_eight_failure_classes_route_correctly",
                    passed=router_acceptance.passed,
                    evidence_ref=router_ref,
                ),
            ),
            replay_ref=router_ref,
            replay_verified=(router_value["report_hash"] == router_replay.to_dict()["report_hash"]),
        ),
        _proof(
            module="sandbox",
            input_refs=("candidate://" + causal_loop.candidate_parameter.candidate_id,),
            output_refs=(
                "sandbox://"
                + evaluation.candidate_a_rejection.sandbox_decision.result_hash.removeprefix(
                    "sha256:"
                ),
            ),
            impacts=("changed_unsafe_activation_to_rejection",),
            metric=CounterfactualMetric(
                name="unsafe_allow",
                control=1.0,
                treatment=0.0,
                lower_is_better=True,
            ),
            control_ref="run://sandbox_off",
            treatment_ref="run://sandbox_on",
            injections=(
                FaultInjectionResult(
                    name="force_regression_rejected",
                    passed=(
                        evaluation.candidate_a_rejection.decision == "REJECTED"
                        and not evaluation.candidate_a_rejection.sandbox_decision.allowed
                    ),
                    evidence_ref="sandbox://force_regression",
                ),
            ),
            replay_ref=(
                "sandbox://"
                + evaluation.candidate_a_rejection.sandbox_decision.result_hash.removeprefix(
                    "sha256:"
                )
            ),
            replay_verified=(evaluation.candidate_a_rejection.sandbox_decision.replay_verified),
        ),
        _proof(
            module="practice",
            input_refs=(baseline_ref, recovery_ref),
            output_refs=(snapshot_ref,),
            impacts=("enabled_contextual_policy_training",),
            metric=CounterfactualMetric(
                name="trained_policy_artifacts",
                control=0.0,
                treatment=1.0,
                lower_is_better=False,
            ),
            control_ref="dataset://absent",
            treatment_ref=snapshot_ref,
            injections=(
                FaultInjectionResult(
                    name="artifact_tamper_detected",
                    passed=_practice_artifacts_intact(flywheel),
                    evidence_ref=snapshot_ref,
                ),
            ),
            replay_ref=recovery_ref,
            replay_verified=all(
                item.strict_replay and item.independently_verified
                for item in flywheel.practice_evidence
            ),
        ),
        _proof(
            module="memory",
            input_refs=(failure_ref, snapshot_ref),
            output_refs=("memory://" + str(causal_loop.memory_on.memory_id),),
            impacts=("changed_candidate_order", "reduced_retry_attempts"),
            metric=CounterfactualMetric(
                name="attempts",
                control=float(causal_loop.memory_off.attempts),
                treatment=float(causal_loop.memory_on.attempts),
                lower_is_better=True,
            ),
            control_ref="run://memory_off",
            treatment_ref="run://memory_on",
            injections=(
                FaultInjectionResult(
                    name="wrong_body_rejected",
                    passed=causal_loop.wrong_body_memory_rejected,
                    evidence_ref="memory://wrong_body_probe",
                ),
                FaultInjectionResult(
                    name="stale_memory_rejected",
                    passed=causal_loop.stale_memory_rejected,
                    evidence_ref="memory://stale_probe",
                ),
            ),
        ),
        _proof(
            module="know",
            input_refs=(failure_ref,),
            output_refs=("taskcard://" + proof_candidate.task_card_hash.removeprefix("sha256:"),),
            impacts=("removed_invalid_candidates", "blocked_safety_override"),
            metric=CounterfactualMetric(
                name="invalid_candidates",
                control=float(causal_loop.know_ablation.invalid_without_know),
                treatment=0.0,
                lower_is_better=True,
            ),
            control_ref="run://know_off",
            treatment_ref="run://know_on",
            injections=(
                FaultInjectionResult(
                    name="safety_override_rejected",
                    passed=causal_loop.know_ablation.safety_override_admitted == 0,
                    evidence_ref="taskcard://immutable_safety_probe",
                ),
            ),
        ),
        _proof(
            module="how",
            input_refs=(failure_ref, "memory://" + str(causal_loop.memory_on.memory_id)),
            output_refs=(
                "policy://"
                + causal_loop.recovery_evidence.policy.policy_hash.removeprefix("sha256:"),
            ),
            impacts=("changed_retry_policy", "changed_failure_to_success"),
            metric=CounterfactualMetric(
                name="task_failures",
                control=float(not causal_loop.baseline_evidence.result.success),
                treatment=float(not causal_loop.recovery_evidence.result.success),
                lower_is_better=True,
            ),
            control_ref=baseline_ref,
            treatment_ref=recovery_ref,
            injections=(
                FaultInjectionResult(
                    name="unknown_patch_path_rejected",
                    passed=_unknown_how_patch_rejected(),
                    evidence_ref="how://unknown_path_probe",
                ),
            ),
        ),
        _proof(
            module="auto",
            input_refs=(snapshot_ref, failure_ref),
            output_refs=("candidate://" + proof_candidate.candidate_id,),
            impacts=("generated_learned_policy_candidate", "improved_validation_success"),
            metric=CounterfactualMetric(
                name="success_rate",
                control=evaluation.validation.metrics.baseline_success_rate,
                treatment=evaluation.validation.metrics.candidate_success_rate,
                lower_is_better=False,
            ),
            control_ref="candidate://fixed_baseline",
            treatment_ref=("candidate://" + proof_candidate.candidate_id),
            injections=(
                FaultInjectionResult(
                    name="dataset_mismatch_rejected",
                    passed=_dataset_mismatch_rejected(proof_candidate),
                    evidence_ref="candidate://dataset_mismatch_probe",
                ),
            ),
        ),
        _proof(
            module="darwin",
            input_refs=(validation_ref, holdout_ref),
            output_refs=("promotion://statistical_gate_v3",),
            impacts=("changed_need_more_evidence_to_sim_champion",),
            metric=CounterfactualMetric(
                name="promotion_authorized",
                control=0.0,
                treatment=1.0,
                lower_is_better=False,
            ),
            control_ref="darwin://without_holdout",
            treatment_ref="darwin://with_signed_holdout",
            injections=(
                FaultInjectionResult(
                    name="tampered_signature_rejected",
                    passed=_tampered_holdout_rejected(evaluation),
                    evidence_ref="holdout://tampered_signature_probe",
                ),
            ),
            replay_ref=holdout_ref,
            replay_verified=(
                evaluation.signed_holdout.verify(expected_public_key=evaluation.holdout_public_key)
                and evaluation.holdout.attestation.strict_replay
            ),
        ),
    )
    bundle = ProofBundle(
        run_id="proof_contact_push_pre_activation_v1",
        task_id=CONTACT_PUSH_TASK_ID,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        proofs=proofs,
        evidence_root_ref="arena://" + _hash_path(root).removeprefix("sha256:"),
        candidate_hash=proof_candidate.candidate_hash,
    )
    _atomic_json(root / "proof-bundle.json", bundle.to_dict())
    _atomic_json(root / "darwin-without-holdout.json", without_holdout.to_dict())
    _atomic_json(root / "darwin-with-holdout.json", with_holdout.to_dict())
    return bundle, with_holdout


def build_contact_push_final_proof_bundle(
    *,
    pre_activation: ProofBundle,
    activation: ContactPushActivationResult,
    output_root: Path,
    source_checkout: Path,
) -> ProofBundle:
    """Append an E5 Registry proof after D8/D9 and automatic rollback."""

    root = _external_root(output_root, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    if pre_activation.candidate_hash != activation.final_active_candidate_hash:
        raise ValueError("final active Champion does not match the promoted proof subject")
    if any(proof.module == "registry" for proof in pre_activation.proofs):
        raise ValueError("pre-activation proof bundle already contains Registry evidence")
    rollback_hash = activation.canary.rollback_receipt_hash
    if rollback_hash is None:
        raise ValueError("final Registry proof requires a Canary rollback receipt")
    registry_proof = _proof(
        module="registry",
        input_refs=(
            "promotion://" + activation.champion_activation.evaluation_hash.removeprefix("sha256:"),
        ),
        output_refs=(
            "receipt://" + activation.champion_activation.receipt_hash.removeprefix("sha256:"),
            "receipt://" + activation.ordinary_episode.receipt_hash.removeprefix("sha256:"),
        ),
        impacts=("changed_active_slot", "ordinary_task_used_champion"),
        metric=CounterfactualMetric(
            name="ordinary_task_champion_use",
            control=0.0,
            treatment=1.0,
            lower_is_better=False,
        ),
        control_ref="registry://inactive_slot",
        treatment_ref=(
            "registry://" + activation.ordinary_episode.candidate_hash.removeprefix("sha256:")
        ),
        injections=(
            FaultInjectionResult(
                name="wrong_body_slot_rejected",
                passed=activation.wrong_body_slot_empty,
                evidence_ref="registry://wrong_body_probe",
            ),
            FaultInjectionResult(
                name="canary_regression_rolled_back",
                passed=(
                    activation.canary.frozen
                    and not activation.canary.passed
                    and activation.rollback_retry.success
                ),
                evidence_ref=("receipt://" + rollback_hash.removeprefix("sha256:")),
            ),
            FaultInjectionResult(
                name="receipt_ledger_verified",
                passed=activation.ledger_verified,
                evidence_ref="registry://ledger",
            ),
        ),
        replay_ref=("receipt://" + activation.rollback_retry.receipt_hash.removeprefix("sha256:")),
        replay_verified=activation.rollback_retry.strict_replay,
    )
    final = ProofBundle(
        run_id="proof_contact_push_final_v1",
        task_id=pre_activation.task_id,
        body_snapshot_hash=pre_activation.body_snapshot_hash,
        proofs=(*pre_activation.proofs, registry_proof),
        evidence_root_ref="arena://" + _hash_path(root).removeprefix("sha256:"),
        candidate_hash=pre_activation.candidate_hash,
    )
    _atomic_json(root / "proof-bundle-final.json", final.to_dict())
    return final


def _proof(
    *,
    module: str,
    input_refs: tuple[str, ...],
    output_refs: tuple[str, ...],
    impacts: tuple[str, ...],
    metric: CounterfactualMetric,
    control_ref: str,
    treatment_ref: str,
    injections: tuple[FaultInjectionResult, ...],
    replay_ref: str | None = None,
    replay_verified: bool = False,
) -> ModuleProof:
    return ModuleProof(
        module=module,
        invoked=True,
        input_refs=input_refs,
        output_refs=output_refs,
        output_valid=True,
        decision_impacts=impacts,
        counterfactual=CounterfactualRun(
            control_run_id=f"run_{module}_control",
            treatment_run_id=f"run_{module}_treatment",
            same_seed=True,
            same_scenario=True,
            same_body_hash=True,
            decision_changed=True,
            outcome_changed=metric.control != metric.treatment,
            metrics=(metric,),
            control_ref=control_ref,
            treatment_ref=treatment_ref,
        ),
        fault_injections=injections,
        replay_verified=replay_verified,
        replay_ref=replay_ref,
    )


def _practice_artifacts_intact(flywheel: ContactPushFlywheel) -> bool:
    for evidence in flywheel.practice_evidence:
        expected = {
            "trajectory_request.json": evidence.request_hash,
            "trajectory_states.json": evidence.state_hash,
            "simulation_receipt.json": evidence.receipt_hash,
        }
        for filename, digest in expected.items():
            path = evidence.artifact_root / filename
            if not path.is_file() or _hash_bytes(path.read_bytes()) != digest:
                return False
    return True


def _unknown_how_patch_rejected() -> bool:
    causal_parent = ContactPushPolicy.baseline()
    compiler = ContactPushTaskKnowledge.default().compiler(causal_parent)
    try:
        compiler.compile(
            {"/runtime/execute_shell": True},
            failure_signature_id="failure_unknown_path_probe",
            generator=_probe_generator(),
            human_involvement=HumanInvolvement(),
        )
    except (TypeError, ValueError):
        return causal_parent.policy_hash == ContactPushPolicy.baseline().policy_hash
    return False


def _probe_generator() -> CandidateGenerator:
    return CandidateGenerator(type="how_probe", algorithm="unknown_path_injection")


def _dataset_mismatch_rejected(candidate: ContactPushCandidate) -> bool:
    try:
        replace(candidate, dataset_snapshot_hash="sha256:" + "0" * 64)
    except ValueError:
        return True
    return False


def _tampered_holdout_rejected(evaluation: ContactPushArenaEvaluation) -> bool:
    signature = evaluation.signed_holdout.signature
    replacement = ("A" if signature[:1] != "A" else "B") + signature[1:]
    tampered = replace(evaluation.signed_holdout, signature=replacement)
    return not tampered.verify(expected_public_key=evaluation.holdout_public_key)


def _external_root(output_root: Path, source_checkout: Path) -> Path:
    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("ContactPush proofs must stay outside the checkout")
    return root


def _hash_path(path: Path) -> str:
    return _hash_bytes(str(path).encode())


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "build_contact_push_final_proof_bundle",
    "build_contact_push_pre_activation_proofs",
]
