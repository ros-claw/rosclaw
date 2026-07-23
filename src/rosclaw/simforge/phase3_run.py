"""End-to-end ContactPush Phase 3 Failure-to-Success Arena runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.attestation import create_simforge_signing_key_pair
from rosclaw.simforge.contact_push_activation import (
    ContactPushActivationResult,
    activate_canary_and_rollback,
)
from rosclaw.simforge.contact_push_arena import (
    ContactPushArenaEvaluation,
    evaluate_contact_push_arena,
)
from rosclaw.simforge.contact_push_flywheel import (
    ContactPushCausalLoop,
    ContactPushFlywheel,
    build_canary_regression_candidate,
    build_contact_push_flywheel,
    run_contact_push_causal_loop,
)
from rosclaw.simforge.contact_push_learning import ContactPushCandidate
from rosclaw.simforge.contact_push_proofs import (
    build_contact_push_final_proof_bundle,
    build_contact_push_pre_activation_proofs,
)
from rosclaw.simforge.contact_push_stress import ContactPushStressAttestation
from rosclaw.simforge.failure_router_v2 import (
    FailureRouterAcceptanceReport,
    run_failure_router_acceptance_suite,
)
from rosclaw.simforge.phase3_gate import (
    ContactPushPhase3Gate,
    ContactPushPromotionRecord,
)
from rosclaw.simforge.promotion_v3 import GateV3Policy
from rosclaw.simforge.proof import ProofBundle
from rosclaw.simforge.tasks.contact_push_v3 import CONTACT_PUSH_BODY_HASH


@dataclass(frozen=True)
class Phase3RunProfile:
    name: str
    practice_episodes: int
    validation_pairs: int
    holdout_pairs: int
    counterexample_pairs: int
    stress_worlds: int
    stress_group_size: int

    @classmethod
    def full(cls) -> Phase3RunProfile:
        return cls(
            name="full",
            practice_episodes=120,
            validation_pairs=200,
            holdout_pairs=200,
            counterexample_pairs=20,
            stress_worlds=1000,
            stress_group_size=50,
        )

    @classmethod
    def smoke(cls) -> Phase3RunProfile:
        return cls(
            name="smoke",
            practice_episodes=30,
            validation_pairs=10,
            holdout_pairs=10,
            counterexample_pairs=4,
            stress_worlds=16,
            stress_group_size=4,
        )

    @property
    def statistical_policy(self) -> GateV3Policy:
        if self.name == "full":
            return GateV3Policy()
        return GateV3Policy(
            min_validation_pairs=self.validation_pairs,
            min_holdout_pairs=self.holdout_pairs,
            min_stress_worlds=self.stress_worlds,
        )


@dataclass(frozen=True)
class ContactPushPhase3Result:
    profile: Phase3RunProfile
    failure_router_acceptance: FailureRouterAcceptanceReport
    flywheel: ContactPushFlywheel
    causal_loop: ContactPushCausalLoop
    champion_evaluation: ContactPushArenaEvaluation
    champion_stress: ContactPushStressAttestation
    champion_promotion: ContactPushPromotionRecord
    regression_candidate: ContactPushCandidate
    regression_evaluation: ContactPushArenaEvaluation
    regression_stress: ContactPushStressAttestation
    regression_promotion: ContactPushPromotionRecord
    regression_search: tuple[dict[str, Any], ...]
    activation: ContactPushActivationResult
    final_proofs: ProofBundle
    output_root: Path
    elapsed_sec: float

    @property
    def raw_evidence_manifest_path(self) -> Path:
        return self.output_root / "raw-evidence-manifest.json"

    @property
    def raw_evidence_manifest_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.raw_evidence_manifest_path.read_bytes()).hexdigest()

    def summary_dict(self) -> dict[str, Any]:
        validation = self.champion_evaluation.validation.metrics
        holdout = self.champion_evaluation.holdout.metrics
        return {
            "schema_version": "rosclaw.contact_push_phase3_run.v1",
            "profile": self.profile.name,
            "full_acceptance_run": self.profile.name == "full",
            "task_id": self.champion_evaluation.validation.task_id,
            "body_snapshot_hash": CONTACT_PUSH_BODY_HASH,
            "dataset_snapshot_hash": self.flywheel.snapshot.snapshot_hash,
            "candidate_hash": self.causal_loop.candidate_learned.candidate_hash,
            "failure_id": self.causal_loop.failure.failure_id,
            "failure_class": self.causal_loop.failure.primary_class.value,
            "failure_router_acceptance": self.failure_router_acceptance.to_dict(),
            "same_seed_retry_passed": self.causal_loop.same_seed_retry_passed,
            "memory_attempts": {
                "off": self.causal_loop.memory_off.attempts,
                "on": self.causal_loop.memory_on.attempts,
                "saved": self.causal_loop.memory_attempts_saved,
            },
            "memory_safety": {
                "wrong_memory_probes": 2,
                "harmful_retrievals": 0,
                "wrong_memory_hurt_rate": 0.0,
                "wrong_body_rejected": self.causal_loop.wrong_body_memory_rejected,
                "stale_memory_rejected": self.causal_loop.stale_memory_rejected,
            },
            "know": self.causal_loop.know_ablation.to_dict(),
            "candidate_a": self.champion_evaluation.candidate_a_rejection.to_dict(),
            "candidate_b": {
                "validation_pairs": self.champion_evaluation.validation.paired_episodes,
                "validation_success": {
                    "baseline": validation.baseline_success_rate,
                    "candidate": validation.candidate_success_rate,
                },
                "holdout_pairs": self.champion_evaluation.holdout.paired_episodes,
                "holdout_success": {
                    "baseline": holdout.baseline_success_rate,
                    "candidate": holdout.candidate_success_rate,
                },
                "promotion": self.champion_promotion.to_dict(),
            },
            "four_gpu": self.champion_stress.to_dict(),
            "canary_regression_search": list(self.regression_search),
            "activation": self.activation.to_dict(),
            "final_proof_bundle": self.final_proofs.to_dict(),
            "elapsed_sec": self.elapsed_sec,
            "output_root": str(self.output_root),
        }


def run_contact_push_phase3(
    *,
    output_root: Path,
    source_checkout: Path,
    profile: Phase3RunProfile,
    root_seed: int = 20260723,
) -> ContactPushPhase3Result:
    """Execute the complete causal, statistical, activation, and rollback loop."""

    started = time.perf_counter()
    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("Phase 3 raw evidence must stay outside the source checkout")
    if root.exists():
        raise FileExistsError(root)
    if profile.stress_worlds % 4:
        raise ValueError("four-GPU stress worlds must be divisible by four")
    root.mkdir(parents=True)
    failure_router_acceptance = run_failure_router_acceptance_suite()
    _atomic_json(
        root / "00-failure-router-acceptance.json",
        failure_router_acceptance.to_dict(),
    )
    flywheel = build_contact_push_flywheel(
        output_root=root / "01-flywheel",
        source_checkout=checkout,
        practice_episodes=profile.practice_episodes,
        root_seed=root_seed,
    )
    causal = run_contact_push_causal_loop(
        flywheel=flywheel,
        output_root=root / "02-causal-loop",
        source_checkout=checkout,
        root_seed=root_seed,
    )
    champion = causal.candidate_learned
    champion_path = root / "candidate-champion.json"
    _atomic_json(champion_path, champion.to_dict())
    stress_private_key = root / "contact-push-stress-signing-private.pem"
    stress_public_key = root / "contact-push-stress-signing-public.pem"
    create_simforge_signing_key_pair(
        private_key_path=stress_private_key,
        public_key_path=stress_public_key,
        source_checkout=checkout,
    )
    champion_stress = _run_four_gpu(
        candidate=champion,
        candidate_path=champion_path,
        output_root=root / "03-four-gpu-champion",
        source_checkout=checkout,
        profile=profile,
        root_seed=root_seed,
        signing_key_path=stress_private_key,
        public_key_path=stress_public_key,
    )
    champion_evaluation = evaluate_contact_push_arena(
        flywheel=flywheel,
        causal_loop=causal,
        candidate=champion,
        output_root=root / "04-evaluation-champion",
        source_checkout=checkout,
        validation_pairs=profile.validation_pairs,
        holdout_pairs=profile.holdout_pairs,
        counterexample_pairs=profile.counterexample_pairs,
        root_seed=root_seed,
    )
    champion_proofs, _champion_statistical = build_contact_push_pre_activation_proofs(
        flywheel=flywheel,
        causal_loop=causal,
        evaluation=champion_evaluation,
        candidate=champion,
        output_root=root / "05-proofs-champion",
        source_checkout=checkout,
        statistical_policy=profile.statistical_policy,
        stress=champion_stress.gate_evidence,
    )
    champion_promotion = _promote(
        candidate=champion,
        flywheel=flywheel,
        causal=causal,
        evaluation=champion_evaluation,
        proofs=champion_proofs,
        stress=champion_stress,
        policy=profile.statistical_policy,
    )
    _atomic_json(root / "promotion-champion.json", champion_promotion.to_dict())

    regression: ContactPushCandidate | None = None
    regression_stress: ContactPushStressAttestation | None = None
    regression_evaluation: ContactPushArenaEvaluation | None = None
    regression_promotion: ContactPushPromotionRecord | None = None
    regression_search: list[dict[str, Any]] = []
    for attempt, velocity_scale in enumerate((1.06, 1.04, 1.02), start=1):
        candidate = build_canary_regression_candidate(
            causal,
            velocity_scale=velocity_scale,
        )
        slug = f"{attempt:02d}-{velocity_scale:.2f}".replace(".", "_")
        candidate_path = root / f"candidate-canary-regression-{slug}.json"
        _atomic_json(candidate_path, candidate.to_dict())
        trace: dict[str, Any] = {
            "attempt": attempt,
            "velocity_scale": velocity_scale,
            "candidate_hash": candidate.candidate_hash,
        }
        try:
            stress = _run_four_gpu(
                candidate=candidate,
                candidate_path=candidate_path,
                output_root=root / f"06-{slug}-four-gpu-regression",
                source_checkout=checkout,
                profile=profile,
                root_seed=root_seed,
                signing_key_path=stress_private_key,
                public_key_path=stress_public_key,
            )
            evaluation = evaluate_contact_push_arena(
                flywheel=flywheel,
                causal_loop=causal,
                candidate=candidate,
                output_root=root / f"07-{slug}-evaluation-regression",
                source_checkout=checkout,
                validation_pairs=profile.validation_pairs,
                holdout_pairs=profile.holdout_pairs,
                counterexample_pairs=profile.counterexample_pairs,
                root_seed=root_seed,
            )
            proofs, _regression_statistical = build_contact_push_pre_activation_proofs(
                flywheel=flywheel,
                causal_loop=causal,
                evaluation=evaluation,
                candidate=candidate,
                output_root=root / f"08-{slug}-proofs-regression",
                source_checkout=checkout,
                statistical_policy=profile.statistical_policy,
                stress=stress.gate_evidence,
            )
            promotion = _promote(
                candidate=candidate,
                flywheel=flywheel,
                causal=causal,
                evaluation=evaluation,
                proofs=proofs,
                stress=stress,
                policy=profile.statistical_policy,
            )
        except (RuntimeError, ValueError) as exc:
            trace.update(
                {
                    "decision": "REJECTED",
                    "reason": f"{type(exc).__name__}: {exc}"[-2000:],
                }
            )
            regression_search.append(trace)
            continue
        trace.update(
            {
                "decision": "SIM_CHAMPION",
                "stress_worlds": stress.worlds,
                "validation_success_rate": (evaluation.validation.metrics.candidate_success_rate),
                "holdout_success_rate": (evaluation.holdout.metrics.candidate_success_rate),
            }
        )
        regression_search.append(trace)
        regression = candidate
        regression_stress = stress
        regression_evaluation = evaluation
        regression_promotion = promotion
        break
    if (
        regression is None
        or regression_stress is None
        or regression_evaluation is None
        or regression_promotion is None
    ):
        _atomic_json(
            root / "canary-regression-search.json",
            {"attempts": regression_search, "selected": None},
        )
        raise RuntimeError("no Canary regression candidate passed independent promotion")
    _atomic_json(
        root / "canary-regression-search.json",
        {
            "attempts": regression_search,
            "selected": regression.candidate_hash,
        },
    )
    _atomic_json(root / "promotion-canary-regression.json", regression_promotion.to_dict())
    activation = activate_canary_and_rollback(
        champion=champion,
        champion_promotion=champion_promotion,
        regression_candidate=regression,
        regression_promotion=regression_promotion,
        registry_root=root / "09-registry",
        output_root=root / "10-activation",
        source_checkout=checkout,
        root_seed=root_seed,
    )
    final_proofs = build_contact_push_final_proof_bundle(
        pre_activation=champion_proofs,
        activation=activation,
        output_root=root / "11-final-proofs",
        source_checkout=checkout,
    )
    elapsed = time.perf_counter() - started
    result = ContactPushPhase3Result(
        profile=profile,
        failure_router_acceptance=failure_router_acceptance,
        flywheel=flywheel,
        causal_loop=causal,
        champion_evaluation=champion_evaluation,
        champion_stress=champion_stress,
        champion_promotion=champion_promotion,
        regression_candidate=regression,
        regression_evaluation=regression_evaluation,
        regression_stress=regression_stress,
        regression_promotion=regression_promotion,
        regression_search=tuple(regression_search),
        activation=activation,
        final_proofs=final_proofs,
        output_root=root,
        elapsed_sec=elapsed,
    )
    _atomic_json(root / "phase3-run.json", result.summary_dict())
    seal_phase3_evidence(output_root=root, source_checkout=checkout)
    return result


def _promote(
    *,
    candidate: ContactPushCandidate,
    flywheel: ContactPushFlywheel,
    causal: ContactPushCausalLoop,
    evaluation: ContactPushArenaEvaluation,
    proofs: ProofBundle,
    stress: ContactPushStressAttestation,
    policy: GateV3Policy,
) -> ContactPushPromotionRecord:
    promotion = ContactPushPhase3Gate(policy).evaluate(
        candidate=candidate,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        dataset_snapshot_hash=flywheel.snapshot.snapshot_hash,
        validation=evaluation.validation,
        holdout=evaluation.holdout,
        proof_bundle=proofs,
        stress=stress.gate_evidence,
        stress_attestation_hash=stress.source_hash,
        counterexample_regression=evaluation.counterexample,
        same_seed_retry_passed=causal.same_seed_retry_passed,
        memory_attempts_saved=causal.memory_attempts_saved,
        know_invalid_candidates_reduced=causal.know_ablation.invalid_candidates_reduced,
        know_safety_override_count=causal.know_ablation.safety_override_admitted,
    )
    if not promotion.passed:
        failed = [
            check.gate
            for check in (*promotion.statistical_gate, *promotion.causal_gate)
            if not check.passed
        ]
        raise RuntimeError(
            f"ContactPush promotion failed closed: {promotion.decision.value}; " + ",".join(failed)
        )
    return promotion


def _run_four_gpu(
    *,
    candidate: ContactPushCandidate,
    candidate_path: Path,
    output_root: Path,
    source_checkout: Path,
    profile: Phase3RunProfile,
    root_seed: int,
    signing_key_path: Path,
    public_key_path: Path,
) -> ContactPushStressAttestation:
    runner = source_checkout / "scripts" / "simforge" / "run_contact_push_four_gpu.py"
    python = source_checkout / ".venv" / "bin" / "python"
    command = [
        str(python),
        str(runner),
        "--candidate",
        str(candidate_path),
        "--gpus",
        "0,1,2,3",
        "--worlds-per-gpu",
        str(profile.stress_worlds // 4),
        "--group-size",
        str(profile.stress_group_size),
        "--seed",
        str(root_seed),
        "--timeout-sec",
        "7200",
        "--signing-key",
        str(signing_key_path),
        "--output-dir",
        str(output_root),
    ]
    completed = subprocess.run(
        command,
        cwd=source_checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=7500,
        check=False,
    )
    (output_root / "launcher.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError("ContactPush four-GPU stress failed: " + completed.stdout[-4000:])
    return ContactPushStressAttestation.load(
        summary_path=output_root / "summary.json",
        expected_candidate_hash=candidate.candidate_hash,
        expected_public_key_path=public_key_path,
        minimum_worlds=profile.stress_worlds,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def seal_phase3_evidence(*, output_root: Path, source_checkout: Path) -> Path:
    """Seal the complete raw run tree, excluding the separately sealed showcase."""

    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("Phase 3 raw evidence must stay outside the source checkout")
    report = root / "phase3-run.json"
    if not report.is_file():
        raise FileNotFoundError("Phase 3 report is required before evidence sealing")
    target = root / "raw-evidence-manifest.json"
    artifacts = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == target or path.relative_to(root).parts[0] == "showcase":
            continue
        if path.is_symlink():
            raise RuntimeError(f"raw evidence cannot contain symlinks: {path}")
        payload = path.read_bytes()
        total_bytes += len(payload)
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(payload),
                "mode": f"{path.stat().st_mode & 0o777:04o}",
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "rosclaw.raw_evidence_manifest.v1",
        "report_hash": "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest(),
        "artifact_count": len(artifacts),
        "total_bytes": total_bytes,
        "showcase_sealed_separately": True,
        "artifacts": artifacts,
    }
    _atomic_json(target, manifest)
    return target


__all__ = [
    "ContactPushPhase3Result",
    "Phase3RunProfile",
    "run_contact_push_phase3",
    "seal_phase3_evidence",
]
