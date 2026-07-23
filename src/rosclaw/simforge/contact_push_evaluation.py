"""Paired Validation and process-isolated hidden Holdout for ContactPush."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rosclaw.simforge.contact_push_learning import ContactPushCandidate
from rosclaw.simforge.evaluation import (
    EpisodeOutcome,
    EvaluationBundle,
    EvidenceVerificationSource,
    PairedEpisode,
    _attest_evaluation_bundle,
)
from rosclaw.simforge.holdout import SignedHoldoutResult
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_TASK_ID,
    ContactPushEpisodeEvidence,
    ContactPushPhysics,
    ContactPushPolicy,
    ContactPushResult,
    ContactPushScenario,
    ContactPushStatus,
)


def evaluate_contact_push_candidate(
    *,
    scenarios: tuple[ContactPushScenario, ...],
    candidate: ContactPushCandidate,
    partition: Partition,
    artifact_root: Path,
    source_checkout: Path,
    bootstrap_seed: int,
) -> tuple[
    EvaluationBundle, tuple[tuple[ContactPushEpisodeEvidence, ContactPushEpisodeEvidence], ...]
]:
    if partition not in {
        Partition.VALIDATION,
        Partition.HOLDOUT,
        Partition.COUNTEREXAMPLE_REGRESSION,
    }:
        raise ValueError("ContactPush evaluation requires an evaluation partition")
    if not scenarios or any(scenario.partition is not partition for scenario in scenarios):
        raise ValueError("ContactPush scenario/evaluation partition mismatch")
    physics = ContactPushPhysics()
    baseline_policy = ContactPushPolicy.baseline()
    pairs = []
    evidence_pairs = []
    for scenario in scenarios:
        scenario_root = artifact_root / partition.value / scenario.scenario_id
        baseline = physics.run_and_record(
            scenario=scenario,
            policy=baseline_policy,
            artifact_root=scenario_root / "baseline",
            source_checkout=source_checkout,
            practice_id=f"practice_{partition.value}_baseline",
        )
        candidate_policy = candidate.policy_for(scenario)
        candidate_evidence = physics.run_and_record(
            scenario=scenario,
            policy=candidate_policy,
            artifact_root=scenario_root / "candidate",
            source_checkout=source_checkout,
            practice_id=f"practice_{partition.value}_candidate",
        )
        evidence_pairs.append((baseline, candidate_evidence))
        artifacts_valid = _episode_artifacts_valid(baseline) and _episode_artifacts_valid(
            candidate_evidence
        )
        pairs.append(
            PairedEpisode(
                pair_id=f"pair_{scenario.scenario_id}",
                scenario_commitment=scenario.scenario_commitment,
                seed_commitment=scenario.seed_commitment,
                baseline=_episode_outcome(baseline.result),
                candidate=_episode_outcome(candidate_evidence.result),
                physics_executed=(
                    baseline.result.physics_executed and candidate_evidence.result.physics_executed
                ),
                independently_verified=(
                    baseline.independently_verified and candidate_evidence.independently_verified
                ),
                strict_replay=baseline.strict_replay and candidate_evidence.strict_replay,
                artifact_hash_valid=artifacts_valid,
                data_quality_valid=(
                    _result_finite(baseline.result) and _result_finite(candidate_evidence.result)
                ),
            )
        )
    bundle = EvaluationBundle.from_pairs(
        task_id=CONTACT_PUSH_TASK_ID,
        candidate_hash=candidate.candidate_hash,
        partition=partition,
        pairs=pairs,
        human_involvement=candidate.human_involvement,
        bootstrap_seed=bootstrap_seed,
        evidence_refs=(
            candidate.dataset_snapshot_hash or candidate.parent_policy_hash,
            candidate.task_card_hash,
        ),
    )
    return (
        _attest_evaluation_bundle(
            bundle,
            EvidenceVerificationSource.PHYSICS_RECEIPTS,
        ),
        tuple(evidence_pairs),
    )


def create_contact_push_private_holdout(
    *,
    path: Path,
    scenarios: tuple[ContactPushScenario, ...],
    artifact_root: Path,
    source_checkout: Path,
    dataset_snapshot_hash: str,
    seed_ledger_manifest_hash: str,
    bootstrap_seed: int,
) -> None:
    if not scenarios or any(scenario.partition is not Partition.HOLDOUT for scenario in scenarios):
        raise ValueError("private ContactPush Holdout requires Holdout scenarios")
    source = source_checkout.resolve()
    for target in (path.resolve(), artifact_root.resolve()):
        if target == source or source in target.parents:
            raise ValueError("private Holdout inputs and artifacts must stay outside checkout")
    _write_private(
        path,
        json.dumps(
            {
                "schema_version": "rosclaw.contact_push_private_holdout.v1",
                "task_id": CONTACT_PUSH_TASK_ID,
                "dataset_snapshot_hash": dataset_snapshot_hash,
                "seed_ledger_manifest_hash": seed_ledger_manifest_hash,
                "artifact_root": str(artifact_root.resolve()),
                "source_checkout": str(source),
                "bootstrap_seed": bootstrap_seed,
                "scenarios": [scenario.to_private_dict() for scenario in scenarios],
            },
            sort_keys=True,
        ).encode(),
    )


class HiddenContactPushEvaluator:
    """Keep Holdout scenarios in a spawned worker and return a signed aggregate."""

    def __init__(
        self,
        *,
        private_bundle_path: Path,
        signing_key_path: Path,
        source_checkout: Path,
        timeout_sec: float = 3600.0,
    ) -> None:
        self.private_bundle_path = private_bundle_path.resolve()
        self.signing_key_path = signing_key_path.resolve()
        self.source_checkout = source_checkout.resolve()
        self.timeout_sec = timeout_sec
        for path in (self.private_bundle_path, self.signing_key_path):
            _assert_private_external_file(path, self.source_checkout)
        if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 86_400:
            raise ValueError("hidden ContactPush timeout must be in (0, 86400]")

    def evaluate(self, candidate: ContactPushCandidate) -> SignedHoldoutResult:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_contact_push_holdout_worker,
            args=(
                str(self.private_bundle_path),
                str(self.signing_key_path),
                candidate.to_dict(),
                sender,
            ),
            daemon=True,
        )
        process.start()
        sender.close()
        if not receiver.poll(self.timeout_sec):
            process.kill()
            process.join(timeout=5)
            raise TimeoutError("hidden ContactPush worker timed out")
        try:
            response = receiver.recv()
        except EOFError as exc:
            process.join(timeout=5)
            raise RuntimeError("hidden ContactPush worker closed without a result") from exc
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
            raise RuntimeError("hidden ContactPush worker did not terminate")
        if process.exitcode != 0 or not response.get("ok"):
            raise RuntimeError(response.get("error", "hidden ContactPush worker failed"))
        result = SignedHoldoutResult.from_dict(response["result"])
        if result.candidate_hash != candidate.candidate_hash:
            raise RuntimeError("hidden ContactPush returned a mismatched candidate")
        return result


def create_contact_push_signing_key(path: Path) -> str:
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    _write_private(path, raw)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return base64.b64encode(public).decode()


def _contact_push_holdout_worker(
    private_bundle_path: str,
    signing_key_path: str,
    candidate_value: dict[str, Any],
    sender: Any,
) -> None:
    try:
        private = json.loads(Path(private_bundle_path).read_text(encoding="utf-8"))
        if private.get("schema_version") != "rosclaw.contact_push_private_holdout.v1":
            raise ValueError("invalid ContactPush private Holdout schema")
        candidate = ContactPushCandidate.from_dict(candidate_value)
        if candidate.dataset_snapshot_hash != private.get("dataset_snapshot_hash"):
            raise ValueError("candidate is not bound to the private Holdout dataset")
        scenarios = tuple(
            ContactPushScenario.from_private_dict(value) for value in private["scenarios"]
        )
        bundle, _evidence = evaluate_contact_push_candidate(
            scenarios=scenarios,
            candidate=candidate,
            partition=Partition.HOLDOUT,
            artifact_root=Path(private["artifact_root"]),
            source_checkout=Path(private["source_checkout"]),
            bootstrap_seed=int(private["bootstrap_seed"]),
        )
        private_key = Ed25519PrivateKey.from_private_bytes(Path(signing_key_path).read_bytes())
        public = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        unsigned = SignedHoldoutResult(
            task_id=bundle.task_id,
            candidate_hash=bundle.candidate_hash,
            paired_episodes=bundle.paired_episodes,
            metrics=bundle.metrics,
            evidence_attestation=bundle.attestation,
            pair_set_commitment=bundle.pair_set_commitment,
            seed_ledger_manifest_hash=str(private["seed_ledger_manifest_hash"]),
            human_involvement=bundle.human_involvement,
            public_key=base64.b64encode(public).decode(),
            signature="",
        )
        signature = base64.b64encode(private_key.sign(unsigned.signed_payload())).decode()
        result = SignedHoldoutResult(**{**unsigned.__dict__, "signature": signature})
        sender.send({"ok": True, "result": result.to_dict()})
    except Exception as exc:  # noqa: BLE001 - process boundary must fail closed
        sender.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        sender.close()


def _episode_outcome(result: ContactPushResult) -> EpisodeOutcome:
    safety_violation = result.status in {
        ContactPushStatus.FORCE_LIMIT,
        ContactPushStatus.NON_FINITE,
    }
    return EpisodeOutcome(
        success=result.success,
        collision=safety_violation,
        unsafe_allow=safety_violation,
        false_block=result.status is ContactPushStatus.NO_CONTACT,
        robustness=result.robustness,
    )


def _episode_artifacts_valid(evidence: ContactPushEpisodeEvidence) -> bool:
    paths = {
        "trajectory_request.json": evidence.request_hash,
        "trajectory_states.json": evidence.state_hash,
        "simulation_receipt.json": evidence.receipt_hash,
    }
    return all(
        path.is_file() and _hash_bytes(path.read_bytes()) == expected
        for name, expected in paths.items()
        if (path := evidence.artifact_root / name)
    )


def _result_finite(result: ContactPushResult) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            result.target_x_m,
            result.final_object_x_m,
            result.final_error_m,
            result.peak_contact_force_n,
            result.final_object_speed_mps,
            result.elapsed_sec,
            result.robustness,
        )
    )


def _assert_private_external_file(path: Path, source_checkout: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path == source_checkout or source_checkout in path.parents:
        raise ValueError("private Holdout files must stay outside source checkout")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"private Holdout file must be mode 0600: {path}")


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = [
    "HiddenContactPushEvaluator",
    "create_contact_push_private_holdout",
    "create_contact_push_signing_key",
    "evaluate_contact_push_candidate",
]
