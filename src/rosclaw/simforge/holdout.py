"""Process-isolated hidden holdout evaluation with signed aggregate-only results."""

from __future__ import annotations

import base64
import json
import math
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
    EvidenceVerificationSource,
    PairedEpisode,
    _attest_evaluation_bundle,
)
from rosclaw.simforge.models import HumanInvolvement, Partition

_MAX_HOLDOUT_BUNDLE_BYTES = 32 * 1024 * 1024


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

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not 1 <= len(self.task_id) <= 128:
            raise ValueError("holdout task_id must contain 1..128 characters")
        for name in (
            "candidate_hash",
            "pair_set_commitment",
            "seed_ledger_manifest_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"holdout {name} must be a sha256 digest")
        if (
            isinstance(self.paired_episodes, bool)
            or not isinstance(self.paired_episodes, int)
            or not 1 <= self.paired_episodes <= 10_000
        ):
            raise ValueError("holdout paired_episodes must be in [1, 10000]")
        if not isinstance(self.metrics, AggregateMetrics):
            raise ValueError("holdout metrics must be AggregateMetrics")
        if not isinstance(self.evidence_attestation, EvidenceAttestation):
            raise ValueError("holdout evidence_attestation must be EvidenceAttestation")
        if not isinstance(self.human_involvement, HumanInvolvement):
            raise ValueError("holdout human_involvement must be HumanInvolvement")
        if not isinstance(self.public_key, str) or len(self.public_key) > 64:
            raise ValueError("holdout public key is malformed")
        if not isinstance(self.signature, str) or len(self.signature) > 128:
            raise ValueError("holdout signature is malformed")
        if self.schema_version != "rosclaw.simforge.holdout_result.v1":
            raise ValueError("unsupported holdout result schema")

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
            public_bytes = base64.b64decode(self.public_key, validate=True)
            signature = base64.b64decode(self.signature, validate=True)
            if len(public_bytes) != 32 or len(signature) != 64:
                return False
            key = Ed25519PublicKey.from_public_bytes(public_bytes)
            key.verify(signature, self.signed_payload())
        except (InvalidSignature, ValueError, TypeError):
            return False
        return True

    def to_evaluation_bundle(self, *, expected_public_key: str) -> EvaluationBundle:
        if not self.verify(expected_public_key=expected_public_key):
            raise ValueError("holdout aggregate signature verification failed")
        bundle = EvaluationBundle(
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
        return _attest_evaluation_bundle(
            bundle,
            EvidenceVerificationSource.SIGNED_HOLDOUT,
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
            task_id=value["task_id"],
            candidate_hash=value["candidate_hash"],
            paired_episodes=value["paired_episodes"],
            metrics=AggregateMetrics(**metrics),
            evidence_attestation=EvidenceAttestation(**value["evidence_attestation"]),
            pair_set_commitment=value["pair_set_commitment"],
            seed_ledger_manifest_hash=value["seed_ledger_manifest_hash"],
            human_involvement=HumanInvolvement(**value["human_involvement"]),
            public_key=value["public_key"],
            signature=value["signature"],
            schema_version=value["schema_version"],
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
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(float(timeout_sec))
            or timeout_sec <= 0
            or timeout_sec > 86_400
        ):
            raise ValueError("holdout timeout must be in (0, 86400]")

    def evaluate(self, candidate: CandidatePatch) -> SignedHoldoutResult:
        if not isinstance(candidate, CandidatePatch):
            raise ValueError("hidden holdout candidate must be CandidatePatch")
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
        try:
            process.start()
        except Exception:
            receiver.close()
            sender.close()
            raise
        sender.close()
        try:
            if not receiver.poll(self.timeout_sec):
                raise TimeoutError("hidden holdout worker timed out")
            try:
                response = receiver.recv()
            except EOFError as exc:
                raise RuntimeError("hidden holdout worker closed without a result") from exc
            process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError("hidden holdout worker did not terminate after returning")
            if not isinstance(response, dict):
                raise RuntimeError("hidden holdout worker returned an invalid response")
            if process.exitcode != 0 or response.get("ok") is not True:
                raise RuntimeError(response.get("error", "hidden holdout worker failed"))
            result = SignedHoldoutResult.from_dict(response["result"])
            if result.candidate_hash != candidate.candidate_hash:
                raise RuntimeError("hidden holdout worker returned a mismatched candidate")
            return result
        finally:
            receiver.close()
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def create_holdout_signing_key(path: Path) -> str:
    """Create a non-overwriting mode-0600 Ed25519 key."""

    key = Ed25519PrivateKey.generate()
    raw_private = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw_private)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating holdout signing key")
            view = view[written:]
        os.fsync(descriptor)
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
        bundle_path = Path(private_bundle_path)
        private_bundle = json.loads(
            _read_bounded(bundle_path, _MAX_HOLDOUT_BUNDLE_BYTES).decode("utf-8")
        )
        bundle = _run_private_bundle(private_bundle, candidate, candidate_hash)
        key_payload = _read_bounded(Path(signing_key_path), 32)
        if len(key_payload) != 32:
            raise ValueError("holdout signing key must contain 32 bytes")
        private_key = Ed25519PrivateKey.from_private_bytes(key_payload)
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
    raw_changes = candidate.get("changes")
    if (
        not isinstance(raw_changes, list)
        or not 1 <= len(raw_changes) <= 128
        or any(not isinstance(item, dict) for item in raw_changes)
    ):
        raise ValueError("holdout candidate changes must contain 1..128 mappings")
    changes = {
        item["path"]: item["new"] for item in raw_changes if isinstance(item.get("path"), str)
    }
    if len(changes) != len(raw_changes):
        raise ValueError("holdout candidate change paths must be unique strings")
    if not 1 <= len(changes) <= 128:
        raise ValueError("holdout candidate changes must contain 1..128 entries")
    threshold_path = private_bundle.get("threshold_path", "/shield/risk_threshold")
    if not isinstance(threshold_path, str) or not 1 <= len(threshold_path) <= 256:
        raise ValueError("holdout threshold path must be a bounded string")
    threshold = _finite_threshold(changes[threshold_path], "candidate threshold")
    baseline_threshold = _finite_threshold(
        private_bundle["baseline_threshold"],
        "baseline threshold",
    )
    pairs: list[PairedEpisode] = []
    cases = private_bundle.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10_000:
        raise ValueError("private holdout cases must contain 1..10000 entries")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("private holdout cases must be mappings")
        risk = _finite_probability(case["risk"], "holdout case risk")
        should_allow = _strict_bool(case["should_allow"], "should_allow")
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
                physics_executed=_strict_bool(
                    case.get("physics_executed", False), "physics_executed"
                ),
                independently_verified=_strict_bool(
                    case.get("independently_verified", False), "independently_verified"
                ),
                strict_replay=_strict_bool(case.get("strict_replay", False), "strict_replay"),
                artifact_hash_valid=_strict_bool(
                    case.get("artifact_hash_valid", False), "artifact_hash_valid"
                ),
                data_quality_valid=_strict_bool(
                    case.get("data_quality_valid", False), "data_quality_valid"
                ),
            )
        )
    involvement_value = candidate.get("human_involvement", {})
    if not isinstance(involvement_value, dict):
        raise ValueError("holdout human_involvement must be a mapping")
    involvement = HumanInvolvement(**involvement_value)
    task_id = private_bundle.get("task_id")
    if not isinstance(task_id, str):
        raise ValueError("holdout task_id must be a string")
    bootstrap_seed = private_bundle.get("bootstrap_seed", 0)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("holdout bootstrap_seed must be an integer")
    return EvaluationBundle.from_pairs(
        task_id=task_id,
        candidate_hash=candidate_hash,
        partition=Partition.HOLDOUT,
        pairs=pairs,
        human_involvement=involvement,
        bootstrap_seed=bootstrap_seed,
    )


def _assert_private_external_file(path: Path, source_checkout: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path == source_checkout or source_checkout in path.parents:
        raise ValueError("holdout private files must be outside the source checkout")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"holdout private file must be mode 0600 or stricter: {path}")


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _finite_threshold(value: Any, name: str) -> float:
    normalized = _finite_probability(value, name)
    if not 0.1 <= normalized <= 0.9:
        raise ValueError(f"{name} must be in [0.1, 0.9]")
    return normalized


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _read_bounded(path: Path, maximum: int) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 1 <= path.stat().st_size <= maximum:
        raise ValueError(f"private holdout file has an invalid size: {path}")
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    if not 1 <= len(payload) <= maximum:
        raise ValueError(f"private holdout file has an invalid size: {path}")
    return payload


__all__ = [
    "HiddenHoldoutService",
    "SignedHoldoutResult",
    "create_holdout_signing_key",
]
