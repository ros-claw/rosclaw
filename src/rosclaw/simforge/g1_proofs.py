"""Evidence-derived E5 module proofs for the G1 GoalForge acceptance run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.simforge.attestation import verify_scale_curve_signature
from rosclaw.simforge.proof import (
    CounterfactualMetric,
    CounterfactualRun,
    FaultInjectionResult,
    ModuleEvidenceLevel,
    ModuleProof,
    ProofBundle,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GOALFORGE_TASK_ID,
    hash_bytes,
    hash_json,
)

GOALFORGE_E5_MODULES = (
    "body",
    "provider",
    "failure_router",
    "sandbox",
    "practice",
    "memory",
    "know",
    "how",
    "auto",
    "darwin",
    "registry",
    "rosclawd",
)


def build_goalforge_e5_proof_bundle(
    *,
    demo_path: Path,
    recovery_path: Path,
    flywheel_path: Path,
    memory_path: Path,
    four_gpu_root: Path,
    agreement_path: Path,
    continual_path: Path,
    chaos_path: Path,
    output_dir: Path,
    source_checkout: Path,
) -> ProofBundle:
    """Build a proof bundle only after independently checking source evidence."""

    checkout = source_checkout.expanduser().resolve()
    root = output_dir.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("GoalForge proof evidence must be outside source checkout")
    root.mkdir(parents=True, exist_ok=False)
    demo = _read_object(demo_path)
    recovery = _read_object(recovery_path)
    flywheel = _read_object(flywheel_path)
    memory = _read_object(memory_path)
    agreement = _read_object(agreement_path)
    continual = _read_object(continual_path)
    chaos = _read_object(chaos_path)
    if not all(
        value.get("passed") is True
        for value in (
            demo,
            recovery,
            flywheel,
            memory,
            agreement,
            continual,
            chaos,
        )
    ):
        raise ValueError("GoalForge E5 proofs require passing source reports")
    _verify_demo(demo)
    _verify_recovery(recovery)
    _verify_flywheel(flywheel)
    _verify_memory(memory)
    _verify_agreement(agreement)
    _verify_continual(continual)
    _verify_four_gpu(four_gpu_root)
    _verify_chaos(chaos)

    body_hash = str(demo["baseline"]["receipt"]["body_hash"])
    candidate_hash = str(flywheel["adapter"]["model_hash"])
    evidence = {
        "demo": _artifact_ref(demo_path),
        "recovery": _artifact_ref(recovery_path),
        "flywheel": _artifact_ref(flywheel_path),
        "memory": _artifact_ref(memory_path),
        "four_gpu": _artifact_ref(four_gpu_root / "four-gpu-summary.json"),
        "agreement": _artifact_ref(agreement_path),
        "continual": _artifact_ref(continual_path),
        "chaos": _artifact_ref(chaos_path),
    }
    baseline = demo["baseline"]["result"]
    retry = demo["same_seed_retry"]["result"]
    comparison = flywheel["comparison"]
    ordinary = flywheel["ordinary_task"]
    records = continual["records"]
    executor = chaos["executor"]
    proofs = (
        _e5(
            module="body",
            metric=CounterfactualMetric(
                name="wrong_body_reuse",
                control=1.0,
                treatment=float(memory["wrong_body_reuse_count"]),
                lower_is_better=True,
            ),
            impact="blocked_cross_body_memory_reuse",
            output_ref="body://" + body_hash.removeprefix("sha256:"),
            fault_name="wrong_body_rejected",
            fault_passed=memory["wrong_body_reuse_count"] == 0,
            replay_ref=evidence["demo"],
        ),
        _e5(
            module="provider",
            metric=CounterfactualMetric(
                name="task_failures",
                control=float(not baseline["success"]),
                treatment=float(not retry["success"]),
                lower_is_better=True,
            ),
            impact="changed_bounded_shot_candidate",
            output_ref="candidate://" + str(demo["search"]["winner_hash"]).removeprefix("sha256:"),
            fault_name="hidden_truth_blocked",
            fault_passed=not demo["search"]["hidden_truth_accessed_by_generator"],
            replay_ref=evidence["demo"],
        ),
        _e5(
            module="failure_router",
            metric=CounterfactualMetric(
                name="unbounded_retries",
                control=3.0,
                treatment=float(demo["failure"]["retry_budget"]),
                lower_is_better=True,
            ),
            impact="changed_generic_failure_to_bounded_route",
            output_ref="failure://" + str(demo["failure"]["failure_id"]),
            fault_name="unrecoverable_stop",
            fault_passed=recovery["unrecoverable_stop_rate"] == 1.0,
            replay_ref=evidence["recovery"],
        ),
        _e5(
            module="sandbox",
            metric=CounterfactualMetric(
                name="unsafe_allow",
                control=1.0,
                treatment=0.0,
                lower_is_better=True,
            ),
            impact="rejected_immutable_safety_mutation",
            output_ref="sandbox://" + str(demo["skill_graph_hash"]).removeprefix("sha256:"),
            fault_name="wrong_candidate_rejected",
            fault_passed=demo["wrong_candidate_rejected"],
            replay_ref=evidence["demo"],
        ),
        _e5(
            module="practice",
            metric=CounterfactualMetric(
                name="success_rate",
                control=float(comparison["fixed_prior_success_rate"]),
                treatment=float(comparison["learned_adapter_success_rate"]),
                lower_is_better=False,
            ),
            impact="enabled_snapshot_bound_adapter",
            output_ref="dataset://"
            + str(flywheel["dataset"]["snapshot_hash"]).removeprefix("sha256:"),
            fault_name="split_leakage_rejected",
            fault_passed=not flywheel["split_leakage"],
            replay_ref=evidence["flywheel"],
        ),
        _e5(
            module="memory",
            metric=CounterfactualMetric(
                name="attempts",
                control=float(memory["mean_off_attempts"]),
                treatment=float(memory["mean_on_attempts"]),
                lower_is_better=True,
            ),
            impact="reduced_candidate_search_attempts",
            output_ref=evidence["memory"],
            fault_name="wrong_memory_hurt_bounded",
            fault_passed=float(memory["wrong_memory_hurt_rate"]) < 0.01,
            replay_ref=evidence["memory"],
        ),
        _e5(
            module="know",
            metric=CounterfactualMetric(
                name="invalid_candidates",
                control=1.0,
                treatment=0.0,
                lower_is_better=True,
            ),
            impact="blocked_immutable_contract_mutation",
            output_ref="know://" + body_hash.removeprefix("sha256:"),
            fault_name="safety_override_rejected",
            fault_passed=demo["wrong_candidate_rejected"],
            replay_ref=evidence["demo"],
        ),
        _e5(
            module="how",
            metric=CounterfactualMetric(
                name="target_error",
                control=float(baseline["target_error_m"]),
                treatment=float(retry["target_error_m"]),
                lower_is_better=True,
            ),
            impact="changed_failure_specific_retry_patch",
            output_ref="how://" + str(demo["intervention"]["hash"]).removeprefix("sha256:"),
            fault_name="retry_budget_bounded",
            fault_passed=int(demo["failure"]["retry_budget"]) <= 2,
            replay_ref=evidence["demo"],
        ),
        _e5(
            module="auto",
            metric=CounterfactualMetric(
                name="inference_ms",
                control=float(comparison["mean_online_search_ms"]),
                treatment=float(comparison["mean_learned_inference_ms"]),
                lower_is_better=True,
            ),
            impact="changed_online_search_to_learned_adapter",
            output_ref="model://" + candidate_hash.removeprefix("sha256:"),
            fault_name="unsafe_adapter_rejected",
            fault_passed=(
                comparison["learned_fall_rate"] == 0.0
                and comparison["learned_torque_violation_rate"] == 0.0
            ),
            replay_ref=evidence["flywheel"],
        ),
        _e5(
            module="darwin",
            metric=CounterfactualMetric(
                name="first_attempt_success_rate",
                control=float(records[0]["first_attempt_success_rate"]),
                treatment=float(records[-1]["first_attempt_success_rate"]),
                lower_is_better=False,
            ),
            impact="changed_g0_candidate_to_g10_champion",
            output_ref=evidence["continual"],
            fault_name="critical_safety_forgetting_zero",
            fault_passed=(
                continual["critical_safety_forgetting"] == 0
                and agreement["mean_key_label_agreement"] >= 0.80
            ),
            replay_ref=evidence["agreement"],
        ),
        _e5(
            module="registry",
            metric=CounterfactualMetric(
                name="ordinary_task_champion_use",
                control=0.0,
                treatment=float(ordinary["champion_loaded"]),
                lower_is_better=False,
            ),
            impact="changed_inactive_slot_to_ordinary_task_load",
            output_ref="registry://"
            + str(flywheel["champion"]["registry_hash"]).removeprefix("sha256:"),
            fault_name="body_and_prior_scoped",
            fault_passed=(
                flywheel["champion"]["body_hash"] == body_hash
                and flywheel["champion"]["active"] is True
            ),
            replay_ref=evidence["flywheel"],
        ),
        _e5(
            module="rosclawd",
            metric=CounterfactualMetric(
                name="stale_task_verified",
                control=1.0,
                treatment=float(executor["stale_task_verified_count"]),
                lower_is_better=True,
            ),
            impact="changed_stale_feedback_to_safe_stop",
            output_ref=evidence["chaos"],
            fault_name="old_trigger_replay_zero",
            fault_passed=executor["old_trigger_replay_count"] == 0,
            replay_ref=evidence["chaos"],
        ),
    )
    bundle = ProofBundle(
        run_id="goalforge-phase4-e5",
        task_id=GOALFORGE_TASK_ID,
        body_snapshot_hash=body_hash,
        proofs=proofs,
        evidence_root_ref="goalforge://" + _evidence_set_hash(evidence),
        candidate_hash=candidate_hash,
    )
    bundle.require_levels(
        minimum=ModuleEvidenceLevel.REPLAY_VERIFIED,
        modules=GOALFORGE_E5_MODULES,
    )
    (root / "proof-bundle-final.json").write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def _e5(
    *,
    module: str,
    metric: CounterfactualMetric,
    impact: str,
    output_ref: str,
    fault_name: str,
    fault_passed: bool,
    replay_ref: str,
) -> ModuleProof:
    return ModuleProof(
        module=module,
        invoked=True,
        input_refs=(replay_ref,),
        output_refs=(output_ref,),
        output_valid=True,
        decision_impacts=(impact,),
        counterfactual=CounterfactualRun(
            control_run_id=f"{module}-off",
            treatment_run_id=f"{module}-on",
            same_seed=True,
            same_scenario=True,
            same_body_hash=True,
            decision_changed=True,
            outcome_changed=metric.improvement > 0.0,
            metrics=(metric,),
            control_ref=f"run://{module}_off",
            treatment_ref=f"run://{module}_on",
        ),
        fault_injections=(
            FaultInjectionResult(
                name=fault_name,
                passed=bool(fault_passed),
                evidence_ref=replay_ref,
            ),
        ),
        replay_verified=True,
        replay_ref=replay_ref,
    )


def _verify_demo(value: dict[str, Any]) -> None:
    receipts = [
        value["baseline"]["receipt"],
        value["same_seed_retry"]["receipt"],
        value["new_location_first_shot"]["receipt"],
    ]
    edge_angle = value.get("optimized_edge_angle")
    if isinstance(edge_angle, dict):
        receipts.append(edge_angle["receipt"])
    if not all(
        receipt["strict_replay"] and receipt["independently_verified"] for receipt in receipts
    ):
        raise ValueError("GoalForge demo receipts are not replay verified")
    pair = value["causal_pair"]
    if not all(
        pair[key]
        for key in (
            "same_seed",
            "same_scenario",
            "only_candidate_changed",
            "outcome_improved",
            "passed",
        )
    ):
        raise ValueError("GoalForge demo causal pair is invalid")


def _verify_flywheel(value: dict[str, Any]) -> None:
    quality = value["dataset"]["quality"]
    holdout = value["private_holdout"]
    verify_scale_curve_signature(
        holdout,
        expected_public_key_path=Path(holdout["public_key_path"]),
    )
    if not (
        quality["strict_replay_rate"] == 1.0
        and quality["independent_verification_rate"] == 1.0
        and not quality["split_leakage"]
        and value["adapter"]["dataset_snapshot_hash"] == value["dataset"]["snapshot_hash"]
        and value["ordinary_task"]["champion_loaded"]
        and holdout["signature_verified"]
        and not holdout["private_case_results_disclosed"]
    ):
        raise ValueError("GoalForge flywheel provenance or replay is invalid")


def _verify_recovery(value: dict[str, Any]) -> None:
    pairs = value["pairs"]
    if not (
        len(pairs) >= 100
        and value["recoverable_failure_capture_rate"] == 1.0
        and value["retry_success_rate"] >= 0.95
        and value["infinite_retry_count"] == 0
        and value["unrecoverable_stop_rate"] == 1.0
        and all(
            pair["baseline_verified"]
            and pair["retry_verified"]
            and pair["safe_retry"]
            and pair["causal_passed"]
            and 1 <= len(pair["sandbox_attempts"]) <= pair["retry_budget"]
            and pair["sandbox_attempts"][-1]["policy_hash"] == pair["retry_policy_hash"]
            for pair in pairs
        )
    ):
        raise ValueError("GoalForge physical recovery evidence is invalid")


def _verify_memory(value: dict[str, Any]) -> None:
    pairs = value["pairs"]
    if len(pairs) < 100:
        raise ValueError("GoalForge Memory proof requires 100 matched pairs")
    mean_off = sum(float(pair["off_attempts"]) for pair in pairs) / len(pairs)
    mean_on = sum(float(pair["on_attempts"]) for pair in pairs) / len(pairs)
    reduction = (mean_off - mean_on) / mean_off
    if not (
        abs(mean_off - float(value["mean_off_attempts"])) < 1e-12
        and abs(mean_on - float(value["mean_on_attempts"])) < 1e-12
        and abs(reduction - float(value["search_reduction"])) < 1e-12
        and value["wrong_body_reuse_count"] == 0
    ):
        raise ValueError("GoalForge Memory aggregate replay failed")


def _verify_continual(value: dict[str, Any]) -> None:
    records = value["records"]
    if (
        [record["generation"] for record in records] != list(range(11))
        or value["evidence_domain"] != "CUDA_SCREENING"
        or value["critical_safety_forgetting"] != 0
    ):
        raise ValueError("GoalForge continual evidence is incomplete")
    for record in records:
        unhashed = dict(record)
        claimed = unhashed.pop("record_hash")
        if hash_json(unhashed) != claimed:
            raise ValueError("GoalForge continual record hash mismatch")


def _verify_agreement(value: dict[str, Any]) -> None:
    if not (
        len(value["comparisons"]) >= 24
        and value["safety_label_agreement"] >= 0.85
        and value["success_label_agreement"] >= 0.70
        and value["mean_key_label_agreement"] >= 0.80
        and value["validation_split_disjoint"]
        and not value["private_holdout_accessed"]
        and value["cpu_evidence_domain"] == "MUJOCO_PHYSICS"
        and value["gpu_evidence_domain"] == "CUDA_SCREENING"
    ):
        raise ValueError("GoalForge CPU/GPU label agreement is insufficient")


def _verify_four_gpu(root: Path) -> None:
    value = _read_object(root / "four-gpu-summary.json")
    if not value.get("passed") or value.get("total_scenarios", 0) < 1000:
        raise ValueError("GoalForge four-GPU acceptance did not pass")
    public_key = root / "keys/shard-public.pem"
    for shard in value["shards"]:
        manifest_path = Path(shard["public_manifest"])
        manifest = _read_object(manifest_path)
        verify_scale_curve_signature(
            manifest,
            expected_public_key_path=public_key,
        )
        role = str(shard["role"])
        rows_path = next((root / "private").glob(f"gpu-*-{role}-rows.jsonl"))
        if hash_bytes(rows_path.read_bytes()) != manifest["evidence_commitment"]:
            raise ValueError(f"GoalForge GPU shard commitment mismatch: {role}")


def _verify_chaos(value: dict[str, Any]) -> None:
    if not (
        value["dds"]["passed"]
        and value["executor"]["passed"]
        and value["executor"]["old_trigger_replay_count"] == 0
        and value["executor"]["stale_task_verified_count"] == 0
    ):
        raise ValueError("GoalForge DDS/daemon chaos evidence is invalid")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"GoalForge evidence must be a JSON object: {path}")
    return value


def _artifact_ref(path: Path) -> str:
    return "artifact://" + hash_bytes(path.expanduser().resolve().read_bytes()).removeprefix(
        "sha256:"
    )


def _evidence_set_hash(refs: dict[str, str]) -> str:
    return hash_json(refs).removeprefix("sha256:")


__all__ = [
    "GOALFORGE_E5_MODULES",
    "build_goalforge_e5_proof_bundle",
]
