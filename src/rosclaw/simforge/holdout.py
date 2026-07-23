"""Process-isolated hidden holdout evaluation with signed aggregate-only results."""

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from rosclaw.simforge.candidates import CandidatePatch
from rosclaw.simforge.evaluation import (
    AggregateMetrics,
    EpisodeOutcome,
    EvaluationBundle,
    EvidenceAttestation,
    PairedEpisode,
)
from rosclaw.simforge.models import HumanInvolvement, Partition


@dataclass(frozen=True)
class SignedHoldoutResult:
    task_id: str
    candidate_hash: str
    paired_episodes: int
    metrics: AggregateMetrics
    evidence_attestation: EvidenceAttestation
    pair_set_commitment: str
    seed_ledger_manifest_hash: str
    human_involvement: HumanInvolvement
    public_key: str
    signature: str
    schema_version: str = "rosclaw.simforge.holdout_result.v1"

    def signed_payload(self) -> bytes:
        value = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "candidate_hash": self.candidate_hash,
            "paired_episodes": self.paired_episodes,
            "metrics": asdict(self.metrics),
            "evidence_attestation": asdict(self.evidence_attestation),
            "pair_set_commitment": self.pair_set_commitment,
            "seed_ledger_manifest_hash": self.seed_ledger_manifest_hash,
            "human_involvement": asdict(self.human_involvement),
            "public_key": self.public_key,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def verify(self, *, expected_public_key: str | None = None) -> bool:
        if expected_public_key is not None and self.public_key != expected_public_key:
            return False
        try:
            key = Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key))
            key.verify(base64.b64decode(self.signature), self.signed_payload())
        except (InvalidSignature, ValueError, TypeError):
            return False
        return True

    def to_evaluation_bundle(self, *, expected_public_key: str) -> EvaluationBundle:
        if not self.verify(expected_public_key=expected_public_key):
            raise ValueError("holdout aggregate signature verification failed")
        return EvaluationBundle(
            task_id=self.task_id,
            candidate_hash=self.candidate_hash,
            partition=Partition.HOLDOUT,
            paired_episodes=self.paired_episodes,
            metrics=self.metrics,
            attestation=self.evidence_attestation,
            pair_set_commitment=self.pair_set_commitment,
            human_involvement=self.human_involvement,
            evidence_refs=(self.seed_ledger_manifest_hash,),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "candidate_hash": self.candidate_hash,
            "paired_episodes": self.paired_episodes,
            "metrics": asdict(self.metrics),
            "evidence_attestation": asdict(self.evidence_attestation),
            "pair_set_commitment": self.pair_set_commitment,
            "seed_ledger_manifest_hash": self.seed_ledger_manifest_hash,
            "human_involvement": asdict(self.human_involvement),
            "public_key": self.public_key,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SignedHoldoutResult:
        metrics = dict(value["metrics"])
        metrics["success_delta_ci95"] = tuple(metrics["success_delta_ci95"])
        return cls(
            task_id=str(value["task_id"]),
            candidate_hash=str(value["candidate_hash"]),
            paired_episodes=int(value["paired_episodes"]),
            metrics=AggregateMetrics(**metrics),
            evidence_attestation=EvidenceAttestation(**value["evidence_attestation"]),
            pair_set_commitment=str(value["pair_set_commitment"]),
            seed_ledger_manifest_hash=str(value["seed_ledger_manifest_hash"]),
            human_involvement=HumanInvolvement(**value["human_involvement"]),
            public_key=str(value["public_key"]),
            signature=str(value["signature"]),
            schema_version=str(value["schema_version"]),
        )


class HiddenHoldoutService:
    """Read private cases only in a spawned worker and return one signed aggregate."""

    def __init__(
        self,
        *,
        private_bundle_path: Path,
        signing_key_path: Path,
        source_checkout: Path,
        timeout_sec: float = 300.0,
    ) -> None:
        self.private_bundle_path = private_bundle_path.resolve()
        self.signing_key_path = signing_key_path.resolve()
        self.source_checkout = source_checkout.resolve()
        self.timeout_sec = timeout_sec
        for path in (self.private_bundle_path, self.signing_key_path):
            _assert_private_external_file(path, self.source_checkout)
        if timeout_sec <= 0 or timeout_sec > 86_400:
            raise ValueError("holdout timeout must be in (0, 86400]")

    def evaluate(self, candidate: CandidatePatch) -> SignedHoldoutResult:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_holdout_worker,
            args=(
                str(self.private_bundle_path),
                str(self.signing_key_path),
                candidate.to_dict(),
                candidate.candidate_hash,
                sender,
            ),
            daemon=True,
        )
        process.start()
        sender.close()
        if not receiver.poll(self.timeout_sec):
            process.kill()
            process.join(timeout=5)
            raise TimeoutError("hidden holdout worker timed out")
        try:
            response = receiver.recv()
        except EOFError as exc:
            process.join(timeout=5)
            raise RuntimeError("hidden holdout worker closed without a result") from exc
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
            raise RuntimeError("hidden holdout worker did not terminate after returning")
        if process.exitcode != 0 or not response.get("ok"):
            raise RuntimeError(response.get("error", "hidden holdout worker failed"))
        result = SignedHoldoutResult.from_dict(response["result"])
        if result.candidate_hash != candidate.candidate_hash:
            raise RuntimeError("hidden holdout worker returned a mismatched candidate")
        return result


def create_holdout_signing_key(path: Path) -> str:
    """Create a mode-0600 Ed25519 key outside the source checkout."""

    key = Ed25519PrivateKey.generate()
    raw_private = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, raw_private)
    finally:
        os.close(descriptor)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.b64encode(public).decode()


def _holdout_worker(
    private_bundle_path: str,
    signing_key_path: str,
    candidate: dict[str, Any],
    candidate_hash: str,
    sender: Any,
) -> None:
    try:
        private_bundle = json.loads(Path(private_bundle_path).read_text())
        bundle = _run_private_bundle(private_bundle, candidate, candidate_hash)
        private_key = Ed25519PrivateKey.from_private_bytes(Path(signing_key_path).read_bytes())
        public_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        unsigned = SignedHoldoutResult(
            task_id=bundle.task_id,
            candidate_hash=bundle.candidate_hash,
            paired_episodes=bundle.paired_episodes,
            metrics=bundle.metrics,
            evidence_attestation=bundle.attestation,
            pair_set_commitment=bundle.pair_set_commitment,
            seed_ledger_manifest_hash=str(private_bundle["seed_ledger_manifest_hash"]),
            human_involvement=bundle.human_involvement,
            public_key=base64.b64encode(public_bytes).decode(),
            signature="",
        )
        signature = base64.b64encode(private_key.sign(unsigned.signed_payload())).decode()
        result = SignedHoldoutResult(**{**unsigned.__dict__, "signature": signature})
        sender.send({"ok": True, "result": result.to_dict()})
    except Exception as exc:  # noqa: BLE001 - worker boundary must return fail-closed result
        sender.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        sender.close()


def _run_private_bundle(
    private_bundle: dict[str, Any], candidate: dict[str, Any], candidate_hash: str
) -> EvaluationBundle:
    runner = private_bundle.get("runner")
    if runner == "shield_reach_mujoco_v1":
        from rosclaw.simforge.tasks.shield_reach import run_hidden_holdout_bundle

        return run_hidden_holdout_bundle(private_bundle, candidate, candidate_hash)
    if runner != "threshold_shield_test_v1":
        raise ValueError(f"unsupported built-in holdout runner: {runner}")
    changes = {str(item["path"]): item["new"] for item in candidate.get("changes", [])}
    threshold_path = str(private_bundle.get("threshold_path", "/shield/risk_threshold"))
    threshold = float(changes[threshold_path])
    baseline_threshold = float(private_bundle["baseline_threshold"])
    pairs: list[PairedEpisode] = []
    for case in private_bundle.get("cases", []):
        risk = float(case["risk"])
        should_allow = bool(case["should_allow"])
        baseline_allow = risk <= baseline_threshold
        candidate_allow = risk <= threshold

        def outcome(allowed: bool, *, expected_allow: bool, case_risk: float) -> EpisodeOutcome:
            unsafe_allow = allowed and not expected_allow
            false_block = not allowed and expected_allow
            return EpisodeOutcome(
                success=allowed == expected_allow,
                collision=unsafe_allow,
                unsafe_allow=unsafe_allow,
                false_block=false_block,
                robustness=(threshold - case_risk) if expected_allow else (case_risk - threshold),
            )

        pairs.append(
            PairedEpisode(
                pair_id=str(case["case_id"]),
                scenario_commitment=str(case["scenario_commitment"]),
                seed_commitment=str(case["seed_commitment"]),
                baseline=outcome(baseline_allow, expected_allow=should_allow, case_risk=risk),
                candidate=outcome(candidate_allow, expected_allow=should_allow, case_risk=risk),
                physics_executed=bool(case.get("physics_executed", False)),
                independently_verified=bool(case.get("independently_verified", False)),
                strict_replay=bool(case.get("strict_replay", False)),
                artifact_hash_valid=bool(case.get("artifact_hash_valid", False)),
                data_quality_valid=bool(case.get("data_quality_valid", False)),
            )
        )
    involvement = HumanInvolvement(**candidate.get("human_involvement", {}))
    return EvaluationBundle.from_pairs(
        task_id=str(private_bundle["task_id"]),
        candidate_hash=candidate_hash,
        partition=Partition.HOLDOUT,
        pairs=pairs,
        human_involvement=involvement,
        bootstrap_seed=int(private_bundle.get("bootstrap_seed", 0)),
    )


def _assert_private_external_file(path: Path, source_checkout: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path == source_checkout or source_checkout in path.parents:
        raise ValueError("holdout private files must be outside the source checkout")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"holdout private file must be mode 0600 or stricter: {path}")


__all__ = [
    "HiddenHoldoutService",
    "SignedHoldoutResult",
    "create_holdout_signing_key",
]
