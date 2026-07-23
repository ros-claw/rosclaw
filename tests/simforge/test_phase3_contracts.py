from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from rosclaw.simforge.attestation import create_simforge_signing_key_pair
from rosclaw.simforge.contact_push_activation import activate_canary_and_rollback
from rosclaw.simforge.contact_push_arena import evaluate_contact_push_arena
from rosclaw.simforge.contact_push_evaluation import (
    HiddenContactPushEvaluator,
    create_contact_push_private_holdout,
    create_contact_push_signing_key,
)
from rosclaw.simforge.contact_push_flywheel import (
    build_canary_regression_candidate,
    build_contact_push_flywheel,
    run_contact_push_causal_loop,
)
from rosclaw.simforge.contact_push_learning import (
    CausalContactPushSearch,
    ContactPushCandidate,
    ContactPushCandidateType,
    ContactPushExpertSearch,
    ContactPushMemory,
    ContactPushTaskKnowledge,
    ContextualPolicyTrainer,
)
from rosclaw.simforge.contact_push_proofs import (
    build_contact_push_final_proof_bundle,
    build_contact_push_pre_activation_proofs,
)
from rosclaw.simforge.contact_push_stress import (
    ContactPushStressAttestation,
    sign_contact_push_stress,
)
from rosclaw.simforge.dataset_snapshot import (
    PracticeDatasetBuilder,
    PracticeEpisodeRecord,
    load_private_holdout,
    load_public_partition,
)
from rosclaw.simforge.evaluation import StressEvidence, _attest_stress_evidence
from rosclaw.simforge.failure_router_v2 import (
    FailureClass,
    FailureObservation,
    FailureRoute,
    FailureRouterV2,
    Recoverability,
    run_failure_router_acceptance_suite,
)
from rosclaw.simforge.models import HumanInvolvement, Partition
from rosclaw.simforge.phase3_gate import ContactPushPhase3Gate, Phase3Decision
from rosclaw.simforge.phase3_run import seal_phase3_evidence
from rosclaw.simforge.promotion_v3 import GateV3Policy
from rosclaw.simforge.proof import (
    CounterfactualMetric,
    CounterfactualRun,
    FaultInjectionResult,
    ModuleEvidenceLevel,
    ModuleProof,
    ProofBundle,
)
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_BODY_HASH,
    CONTACT_PUSH_TASK_ID,
    ContactPushPhysics,
    ContactPushPolicy,
    ContactPushStatus,
    generate_contact_push_scenarios,
)

_CHECKOUT = Path(__file__).resolve().parents[2]


def _verified_stress(candidate_hash: str, *, worlds: int) -> StressEvidence:
    return _attest_stress_evidence(
        StressEvidence(
            task_id=CONTACT_PUSH_TASK_ID,
            candidate_hash=candidate_hash,
            worlds=worlds,
            complete=True,
            critical_backend_disagreements=0,
            scale_curve_commitment="sha256:"
            + hashlib.sha256(f"{candidate_hash}:{worlds}".encode()).hexdigest(),
        )
    )


def _counterfactual(*, changed: bool = True) -> CounterfactualRun:
    return CounterfactualRun(
        control_run_id="run_control",
        treatment_run_id="run_treatment",
        same_seed=True,
        same_scenario=True,
        same_body_hash=True,
        decision_changed=changed,
        outcome_changed=changed,
        metrics=(
            CounterfactualMetric(
                name="attempts",
                control=3.0,
                treatment=1.0,
                lower_is_better=True,
            ),
        ),
        control_ref="run://control",
        treatment_ref="run://treatment",
    )


def _e5_proof(module: str) -> ModuleProof:
    return ModuleProof(
        module=module,
        invoked=True,
        input_refs=("failure://one",),
        output_refs=(f"{module}://output",),
        output_valid=True,
        decision_impacts=("changed_retry_action",),
        counterfactual=_counterfactual(),
        fault_injections=(
            FaultInjectionResult(
                name="tamper_rejected",
                passed=True,
                evidence_ref=f"{module}://fault",
            ),
        ),
        replay_verified=True,
        replay_ref=f"{module}://replay",
    )


def test_module_proof_level_is_derived_and_monotonic() -> None:
    assert ModuleProof(module="memory", invoked=False).level is ModuleEvidenceLevel.EXISTS
    assert ModuleProof(module="memory", invoked=True).level is ModuleEvidenceLevel.INVOKED
    causal = _e5_proof("memory")
    assert causal.level is ModuleEvidenceLevel.REPLAY_VERIFIED
    assert causal.to_dict()["level"] == "E5"

    with pytest.raises(ValueError, match="counterfactual"):
        ModuleProof(
            module="memory",
            invoked=True,
            output_refs=("memory://one",),
            output_valid=True,
            decision_impacts=("claimed_without_ablation",),
        )


def test_proof_bundle_enforces_required_module_levels() -> None:
    bundle = ProofBundle(
        run_id="run_phase3",
        task_id=CONTACT_PUSH_TASK_ID,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        proofs=(_e5_proof("sandbox"), _e5_proof("practice")),
        evidence_root_ref="artifact://phase3",
    )
    bundle.require_levels(
        minimum=ModuleEvidenceLevel.REPLAY_VERIFIED,
        modules=("sandbox", "practice"),
    )
    with pytest.raises(ValueError, match="missing=memory"):
        bundle.require_levels(
            minimum=ModuleEvidenceLevel.DECISION_IMPACT,
            modules=("memory",),
        )


def _observation(**updates: object) -> FailureObservation:
    value: dict[str, object] = {
        "task_id": CONTACT_PUSH_TASK_ID,
        "body_id": "sim_contact_pusher",
        "expected_body_hash": CONTACT_PUSH_BODY_HASH,
        "observed_body_hash": CONTACT_PUSH_BODY_HASH,
        "action_id": "action_failure",
        "evidence_refs": ("receipt://one", "artifact://two"),
        "task_success": False,
        "target_error_m": -0.12,
        "target_tolerance_m": 0.035,
        "object_overshot": True,
        "estimated_friction": 0.2,
        "peak_force_n": 12.0,
        "force_limit_n": 30.0,
    }
    value.update(updates)
    return FailureObservation(**value)  # type: ignore[arg-type]


def test_failure_router_prioritizes_runtime_body_and_impossible_causes() -> None:
    router = FailureRouterV2()
    overshoot = router.route(_observation())
    assert overshoot.primary_class is FailureClass.ENVIRONMENT_SHIFT
    assert overshoot.retry_budget == 3
    assert FailureRoute.AUTO_PARAMETER_PATCH in overshoot.recommended_route

    runtime = router.route(_observation(observation_fresh=False))
    assert runtime.primary_class is FailureClass.RUNTIME_FAULT
    assert FailureRoute.AUTO_PARAMETER_PATCH not in runtime.recommended_route

    impossible = router.route(_observation(reference_policy_solvable=False))
    assert impossible.primary_class is FailureClass.IMPOSSIBLE_TASK
    assert impossible.recoverability is Recoverability.UNRECOVERABLE
    assert impossible.retry_budget == 0
    assert impossible.recommended_route == (FailureRoute.STOP, FailureRoute.HUMAN)


def test_failure_router_acceptance_suite_covers_all_eight_routes() -> None:
    report = run_failure_router_acceptance_suite()

    assert report.passed
    assert report.failure_capture_rate == 1.0
    assert report.routing_accuracy == 1.0
    assert report.infinite_retry_count == 0
    assert report.unrecoverable_stop_rate == 1.0
    assert {case.expected_class for case in report.cases} == set(FailureClass)
    assert report.to_dict()["report_hash"].startswith("sha256:")


def test_phase3_raw_evidence_manifest_is_complete_and_repeatable(tmp_path: Path) -> None:
    root = tmp_path / "raw-evidence"
    root.mkdir()
    (root / "phase3-run.json").write_text('{"complete":true}\n', encoding="utf-8")
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"phase3-evidence")
    showcase = root / "showcase"
    showcase.mkdir()
    (showcase / "presentation.bin").write_bytes(b"sealed-separately")

    manifest_path = seal_phase3_evidence(output_root=root, source_checkout=_CHECKOUT)
    first = manifest_path.read_bytes()
    value = json.loads(first)
    paths = {item["path"] for item in value["artifacts"]}

    assert paths == {"artifact.bin", "phase3-run.json"}
    assert value["artifact_count"] == 2
    assert value["showcase_sealed_separately"] is True
    artifact_entry = next(item for item in value["artifacts"] if item["path"] == "artifact.bin")
    assert artifact_entry["sha256"] == "sha256:" + hashlib.sha256(b"phase3-evidence").hexdigest()

    seal_phase3_evidence(output_root=root, source_checkout=_CHECKOUT)
    assert manifest_path.read_bytes() == first


def _record(index: int, *, scenario_id: str | None = None) -> PracticeEpisodeRecord:
    suffix = f"{index:064x}"[-64:]
    return PracticeEpisodeRecord(
        episode_id=f"episode_{index}",
        practice_id="practice_dataset",
        scenario_id=scenario_id or f"scenario_{index}",
        seed_commitment=f"sha256:{suffix}",
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        task_id=CONTACT_PUSH_TASK_ID,
        features=(
            ("control_delay_sec", 0.01),
            ("estimated_friction", 0.3),
            ("initial_offset_y_m", 0.0),
            ("object_mass_kg", 0.4),
            ("target_distance_m", 0.2),
        ),
        policy=(
            ("contact_duration_sec", 1.0),
            ("contact_offset_y_m", 0.0),
            ("deceleration_fraction", 0.72),
            ("micro_push", True),
            ("policy_type", "parameter"),
            ("push_velocity_mps", 0.3),
        ),
        labels=(
            ("failure_type", "SUCCESS"),
            ("final_error_m", 0.0),
            ("peak_contact_force_n", 10.0),
            ("robustness", 0.03),
            ("success", True),
        ),
        artifact_hashes=(f"sha256:{suffix}",),
        complete=True,
        independently_verified=True,
        strict_replay=True,
    )


def test_dataset_snapshot_is_grouped_private_and_leak_free(tmp_path: Path) -> None:
    records = tuple(_record(index) for index in range(20))
    snapshot, files = PracticeDatasetBuilder(
        source_checkout=_CHECKOUT,
        split_secret=b"dataset-split-secret-v3",
    ).build(
        records=records,
        output_dir=tmp_path / "dataset",
        dataset_id="dataset_contact_push_contract",
        label_provenance={
            "task_success": "independent_task_verifier",
            "failure_signature": "failure_router_v2",
            "contact_force": "mujoco_contact_force",
        },
    )
    public = json.loads(files.manifest.read_text())
    assert snapshot.quality.passes
    assert "file_ref" not in public["partitions"]["holdout"]
    assert stat.S_IMODE(files.private_holdout.stat().st_mode) == 0o600

    development = load_public_partition(files.development)
    validation = load_public_partition(files.validation)
    holdout = load_private_holdout(files.private_holdout)
    scenario_sets = [
        {record.scenario_id for record in partition}
        for partition in (development, validation, holdout)
    ]
    assert not scenario_sets[0] & scenario_sets[1]
    assert not scenario_sets[0] & scenario_sets[2]
    assert not scenario_sets[1] & scenario_sets[2]


def test_dataset_snapshot_rejects_group_leakage_by_construction(tmp_path: Path) -> None:
    records = tuple(_record(index, scenario_id=f"scenario_{index // 2}") for index in range(20))
    _snapshot, files = PracticeDatasetBuilder(
        source_checkout=_CHECKOUT,
        split_secret=b"dataset-split-secret-v3",
    ).build(
        records=records,
        output_dir=tmp_path / "dataset",
        dataset_id="dataset_contact_push_grouped",
        label_provenance={"task_success": "independent_task_verifier"},
    )
    partitions = (
        load_public_partition(files.development),
        load_public_partition(files.validation),
        load_private_holdout(files.private_holdout),
    )
    owners: dict[str, int] = {}
    for partition_index, partition in enumerate(partitions):
        for record in partition:
            previous = owners.setdefault(record.scenario_id, partition_index)
            assert previous == partition_index


def test_contact_push_memory_changes_matched_search_order(tmp_path: Path) -> None:
    physics = ContactPushPhysics(trace_stride=20)
    expert = ContactPushExpertSearch(physics)
    memory = ContactPushMemory()
    training_ledger = SeedLedger(task_id=CONTACT_PUSH_TASK_ID, secret=b"memory-training-v3-secret")
    training = generate_contact_push_scenarios(
        ledger=training_ledger,
        partition=Partition.DEVELOPMENT,
        count=8,
        root_seed=17,
    )
    for scenario in training:
        result = expert.optimize(scenario)
        evidence = physics.run_and_record(
            scenario=scenario,
            policy=result.policy,
            artifact_root=tmp_path / "memory-practice",
            source_checkout=_CHECKOUT,
            practice_id="practice_memory_training",
        )
        memory.ingest_recovery(evidence)

    assert memory.repository.count() == len(training)
    target_ledger = SeedLedger(task_id=CONTACT_PUSH_TASK_ID, secret=b"x" * 32)
    target = generate_contact_push_scenarios(
        ledger=target_ledger,
        partition=Partition.DISCOVERY,
        count=1,
        root_seed=20260723,
    )[0]
    baseline = physics.run(target, ContactPushPolicy.baseline())
    search = CausalContactPushSearch(
        physics=physics,
        knowledge=ContactPushTaskKnowledge.default(),
    )
    memory_off = search.run(target, memory=None)
    memory_on = search.run(target, memory=memory)

    assert baseline.status is ContactPushStatus.OVERSHOOT
    assert memory_off.success and memory_on.success
    assert memory_on.attempts < memory_off.attempts
    assert memory_on.attempts == 1
    assert memory_on.memory_id is not None

    assert memory.retrieve(target, body_hash="sha256:" + "0" * 64) is None


@pytest.mark.parametrize(
    ("candidate_type", "policy_type"),
    [
        (ContactPushCandidateType.PARAMETER, "parameter"),
        (ContactPushCandidateType.TRAJECTORY, "trajectory"),
        (ContactPushCandidateType.SKILL_GRAPH, "skill_graph"),
    ],
)
def test_contact_push_static_candidate_types_are_executable(
    candidate_type: ContactPushCandidateType,
    policy_type: str,
) -> None:
    policy = ContactPushPolicy(
        push_velocity_mps=0.30,
        contact_duration_sec=1.0,
        contact_offset_y_m=0.0,
        deceleration_fraction=0.72,
        micro_push=candidate_type is ContactPushCandidateType.SKILL_GRAPH,
        policy_type=policy_type,
    )
    candidate = ContactPushCandidate.static(
        candidate_type=candidate_type,
        policy=policy,
        parent=ContactPushPolicy.baseline(),
        failure_signature_id="failure_candidate_type_contract",
        task_card_hash=ContactPushTaskKnowledge.default().card_hash,
        dataset_snapshot_hash="sha256:" + "1" * 64,
        lineage_refs=("failure://candidate_type_contract",),
    )
    scenario = generate_contact_push_scenarios(
        ledger=SeedLedger(
            task_id=CONTACT_PUSH_TASK_ID,
            secret=b"candidate-type-contract",
        ),
        partition=Partition.DISCOVERY,
        count=1,
        root_seed=1,
    )[0]
    round_trip = ContactPushCandidate.from_dict(candidate.to_dict())
    assert round_trip.candidate_hash == candidate.candidate_hash
    assert round_trip.policy_for(scenario).policy_hash == policy.policy_hash


def _expert_records(
    *,
    count: int,
    physics: ContactPushPhysics,
) -> tuple[PracticeEpisodeRecord, ...]:
    ledger = SeedLedger(task_id=CONTACT_PUSH_TASK_ID, secret=b"policy-training-v3-secret")
    scenarios = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.DEVELOPMENT,
        count=count,
        root_seed=123,
    )
    expert = ContactPushExpertSearch(physics)
    records = []
    for index, scenario in enumerate(scenarios):
        result = expert.optimize(scenario)
        digest = f"{index + 1:064x}"[-64:]
        records.append(
            PracticeEpisodeRecord(
                episode_id=f"episode_expert_{index}",
                practice_id="practice_policy_training",
                scenario_id=scenario.scenario_id,
                seed_commitment=scenario.seed_commitment,
                body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
                task_id=CONTACT_PUSH_TASK_ID,
                features=tuple(sorted(scenario.feature_dict().items())),
                policy=tuple(sorted(result.policy.to_dict().items())),
                labels=tuple(
                    sorted(
                        {
                            "failure_type": result.outcome.status.value,
                            "final_error_m": result.outcome.final_error_m,
                            "peak_contact_force_n": result.outcome.peak_contact_force_n,
                            "robustness": result.outcome.robustness,
                            "success": result.outcome.success,
                        }.items()
                    )
                ),
                artifact_hashes=(f"sha256:{digest}",),
                complete=True,
                independently_verified=True,
                strict_replay=True,
            )
        )
    return tuple(records)


def test_practice_dataset_trains_policy_used_on_unseen_physics(tmp_path: Path) -> None:
    physics = ContactPushPhysics(trace_stride=50)
    snapshot, files = PracticeDatasetBuilder(
        source_checkout=_CHECKOUT,
        split_secret=b"contextual-policy-split",
    ).build(
        records=_expert_records(count=30, physics=physics),
        output_dir=tmp_path / "dataset",
        dataset_id="dataset_contact_push_policy",
        label_provenance={
            "task_success": "independent_task_verifier",
            "failure_signature": "failure_router_v2",
            "contact_force": "mujoco_contact_force",
        },
    )
    model = ContextualPolicyTrainer().train(
        development=load_public_partition(files.development),
        validation=load_public_partition(files.validation),
        dataset_snapshot_hash=snapshot.snapshot_hash,
    )
    holdout_ledger = SeedLedger(
        task_id=CONTACT_PUSH_TASK_ID,
        secret=b"unseen-policy-holdout-secret",
    )
    unseen = generate_contact_push_scenarios(
        ledger=holdout_ledger,
        partition=Partition.HOLDOUT,
        count=20,
        root_seed=987,
    )
    baseline_success = sum(
        physics.run(scenario, ContactPushPolicy.baseline()).success for scenario in unseen
    )
    learned_success = sum(
        physics.run(scenario, model.predict(scenario)).success for scenario in unseen
    )
    assert model.dataset_snapshot_hash == snapshot.snapshot_hash
    assert learned_success >= 16
    assert learned_success > baseline_success


def test_hidden_contact_push_returns_only_signed_aggregate(tmp_path: Path) -> None:
    dataset_hash = "sha256:" + "a" * 64
    ledger = SeedLedger(task_id=CONTACT_PUSH_TASK_ID, secret=b"hidden-eval-ledger-secret")
    scenarios = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.HOLDOUT,
        count=3,
        root_seed=91,
    )
    private_bundle = tmp_path / "holdout-private.json"
    create_contact_push_private_holdout(
        path=private_bundle,
        scenarios=scenarios,
        artifact_root=tmp_path / "holdout-artifacts",
        source_checkout=_CHECKOUT,
        dataset_snapshot_hash=dataset_hash,
        seed_ledger_manifest_hash=ledger.public_manifest()["manifest_hash"],
        bootstrap_seed=7,
    )
    signing_key = tmp_path / "holdout-signing.key"
    public_key = create_contact_push_signing_key(signing_key)
    policy = ContactPushPolicy(
        push_velocity_mps=0.30,
        contact_duration_sec=1.0,
        contact_offset_y_m=0.0,
        deceleration_fraction=0.72,
        micro_push=True,
        policy_type="parameter",
    )
    candidate = ContactPushCandidate(
        candidate_id="candidate_hidden_contract",
        candidate_type=ContactPushCandidateType.PARAMETER,
        parent_policy_hash=ContactPushPolicy.baseline().policy_hash,
        failure_signature_id="failure_hidden_contract",
        task_card_hash=ContactPushTaskKnowledge.default().card_hash,
        dataset_snapshot_hash=dataset_hash,
        static_policy=policy,
        learned_policy=None,
        lineage_refs=("dataset://contact_push",),
        human_involvement=HumanInvolvement(),
    )
    signed = HiddenContactPushEvaluator(
        private_bundle_path=private_bundle,
        signing_key_path=signing_key,
        source_checkout=_CHECKOUT,
        timeout_sec=60,
    ).evaluate(candidate)

    assert signed.verify(expected_public_key=public_key)
    assert signed.paired_episodes == 3
    public = signed.to_dict()
    assert "scenarios" not in public
    assert "seeds" not in public
    assert stat.S_IMODE(private_bundle.stat().st_mode) == 0o600
    assert stat.S_IMODE(signing_key.stat().st_mode) == 0o600


def test_contact_push_arena_executes_failure_to_hidden_holdout(tmp_path: Path) -> None:
    flywheel = build_contact_push_flywheel(
        output_root=tmp_path / "flywheel",
        source_checkout=_CHECKOUT,
        practice_episodes=24,
        root_seed=20260723,
    )
    causal = run_contact_push_causal_loop(
        flywheel=flywheel,
        output_root=tmp_path / "causal",
        source_checkout=_CHECKOUT,
        root_seed=20260723,
    )
    evaluation = evaluate_contact_push_arena(
        flywheel=flywheel,
        causal_loop=causal,
        output_root=tmp_path / "evaluation",
        source_checkout=_CHECKOUT,
        validation_pairs=5,
        holdout_pairs=5,
        counterexample_pairs=2,
        root_seed=20260723,
    )
    assert causal.same_seed_retry_passed
    assert causal.memory_attempts_saved > 0
    assert causal.know_ablation.safety_override_admitted == 0
    assert evaluation.validation.paired_episodes == 5
    assert evaluation.holdout.paired_episodes == 5
    assert evaluation.signed_holdout.verify(expected_public_key=evaluation.holdout_public_key)
    assert evaluation.candidate_a_rejection.decision == "REJECTED"
    assert not evaluation.candidate_a_rejection.sandbox_decision.allowed
    assert stat.S_IMODE((tmp_path / "evaluation" / "holdout-private.json").stat().st_mode) == 0o600
    statistical_policy = GateV3Policy(
        min_validation_pairs=5,
        min_holdout_pairs=5,
        min_stress_worlds=20,
    )
    stress = _verified_stress(causal.candidate_learned.candidate_hash, worlds=20)
    proof_bundle, statistical = build_contact_push_pre_activation_proofs(
        flywheel=flywheel,
        causal_loop=causal,
        evaluation=evaluation,
        output_root=tmp_path / "proofs",
        source_checkout=_CHECKOUT,
        statistical_policy=statistical_policy,
        stress=stress,
    )
    promotion = ContactPushPhase3Gate(statistical_policy).evaluate(
        candidate=causal.candidate_learned,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        dataset_snapshot_hash=flywheel.snapshot.snapshot_hash,
        validation=evaluation.validation,
        holdout=evaluation.holdout,
        proof_bundle=proof_bundle,
        stress=stress,
        stress_attestation_hash="sha256:" + "2" * 64,
        counterexample_regression=evaluation.counterexample,
        same_seed_retry_passed=causal.same_seed_retry_passed,
        memory_attempts_saved=causal.memory_attempts_saved,
        know_invalid_candidates_reduced=causal.know_ablation.invalid_candidates_reduced,
        know_safety_override_count=causal.know_ablation.safety_override_admitted,
    )
    assert statistical.passed
    assert promotion.decision is Phase3Decision.SIM_CHAMPION

    regression = build_canary_regression_candidate(causal)
    regression_evaluation = evaluate_contact_push_arena(
        flywheel=flywheel,
        causal_loop=causal,
        candidate=regression,
        output_root=tmp_path / "regression-evaluation",
        source_checkout=_CHECKOUT,
        validation_pairs=5,
        holdout_pairs=5,
        counterexample_pairs=2,
        root_seed=20260723,
    )
    regression_stress = _verified_stress(regression.candidate_hash, worlds=20)
    regression_proofs, _regression_statistical = build_contact_push_pre_activation_proofs(
        flywheel=flywheel,
        causal_loop=causal,
        evaluation=regression_evaluation,
        candidate=regression,
        output_root=tmp_path / "regression-proofs",
        source_checkout=_CHECKOUT,
        statistical_policy=statistical_policy,
        stress=regression_stress,
    )
    regression_promotion = ContactPushPhase3Gate(statistical_policy).evaluate(
        candidate=regression,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        dataset_snapshot_hash=flywheel.snapshot.snapshot_hash,
        validation=regression_evaluation.validation,
        holdout=regression_evaluation.holdout,
        proof_bundle=regression_proofs,
        stress=regression_stress,
        stress_attestation_hash="sha256:" + "3" * 64,
        counterexample_regression=regression_evaluation.counterexample,
        same_seed_retry_passed=causal.same_seed_retry_passed,
        memory_attempts_saved=causal.memory_attempts_saved,
        know_invalid_candidates_reduced=causal.know_ablation.invalid_candidates_reduced,
        know_safety_override_count=causal.know_ablation.safety_override_admitted,
    )
    activation = activate_canary_and_rollback(
        champion=causal.candidate_learned,
        champion_promotion=promotion,
        regression_candidate=regression,
        regression_promotion=regression_promotion,
        registry_root=tmp_path / "registry",
        output_root=tmp_path / "activation",
        source_checkout=_CHECKOUT,
        root_seed=20260723,
    )
    assert regression_promotion.decision is Phase3Decision.SIM_CHAMPION
    assert activation.ordinary_episode.candidate_hash == causal.candidate_learned.candidate_hash
    assert activation.canary.frozen
    assert not activation.canary.passed
    assert activation.rollback_retry.success
    assert activation.final_active_candidate_hash == causal.candidate_learned.candidate_hash
    final_proofs = build_contact_push_final_proof_bundle(
        pre_activation=proof_bundle,
        activation=activation,
        output_root=tmp_path / "final-proofs",
        source_checkout=_CHECKOUT,
    )
    final_proofs.require_levels(
        minimum=ModuleEvidenceLevel.DECISION_IMPACT,
        modules=(
            "auto",
            "body",
            "darwin",
            "failure_router",
            "how",
            "know",
            "memory",
            "practice",
            "provider",
            "registry",
            "sandbox",
        ),
    )
    final_proofs.require_levels(
        minimum=ModuleEvidenceLevel.REPLAY_VERIFIED,
        modules=("sandbox", "practice", "darwin", "registry"),
    )


def test_contact_push_stress_requires_pinned_signature(tmp_path: Path) -> None:
    private_key = tmp_path / "stress-private.pem"
    public_key = tmp_path / "stress-public.pem"
    create_simforge_signing_key_pair(
        private_key_path=private_key,
        public_key_path=public_key,
        source_checkout=_CHECKOUT,
    )
    candidate_hash = "sha256:" + "a" * 64
    shards = [
        {
            "schema_version": "rosclaw.contact_push_mjwarp_shard.v1",
            "backend": "mujoco_warp",
            "candidate_hash": candidate_hash,
            "physical_gpu": str(index),
            "visible_devices": str(index),
            "worlds": 5,
            "unique_scenarios": 5,
            "steps_per_world": 1250,
            "world_steps": 6250,
            "finite_state": True,
            "critical_backend_disagreements": 0,
            "cpu_force_violations": 0,
            "exact_label_agreement_rate": 1.0,
            "world_set_commitment": "sha256:" + f"{index + 1:064x}",
        }
        for index in range(4)
    ]
    summary = {
        "schema_version": "rosclaw.contact_push_mjwarp_four_gpu.v1",
        "candidate_hash": candidate_hash,
        "requested_gpus": ["0", "1", "2", "3"],
        "successful_gpus": ["0", "1", "2", "3"],
        "failures": [],
        "worlds": 20,
        "unique_scenarios": 20,
        "world_steps": 25_000,
        "critical_backend_disagreements": 0,
        "cpu_force_violations": 0,
        "minimum_exact_label_agreement_rate": 1.0,
        "finite_state": True,
        "complete": True,
        "shards": shards,
    }
    summary["attestation"] = sign_contact_push_stress(
        summary,
        private_key_path=private_key,
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    verified = ContactPushStressAttestation.load(
        summary_path=summary_path,
        expected_candidate_hash=candidate_hash,
        expected_public_key_path=public_key,
        minimum_worlds=20,
    )
    assert verified.gate_evidence.candidate_hash == candidate_hash
    assert verified.gate_evidence.worlds == 20

    summary["worlds"] = 21
    tampered_path = tmp_path / "tampered-summary.json"
    tampered_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="signature verification failed"):
        ContactPushStressAttestation.load(
            summary_path=tampered_path,
            expected_candidate_hash=candidate_hash,
            expected_public_key_path=public_key,
            minimum_worlds=20,
        )
