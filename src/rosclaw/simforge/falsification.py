"""Failure-directed search, deterministic minimization, and append-only regression storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.distribution import ScenarioSampler
from rosclaw.simforge.models import Partition, SamplingStrategy, ScenarioSample


@dataclass(frozen=True)
class Counterexample:
    counterexample_id: str
    task_id: str
    candidate_hash: str
    failure_signature: str
    robustness: float
    scenario: ScenarioSample
    replay_evidence_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.counterexample_id, str) or not (
            self.counterexample_id.startswith("counterexample_")
            and len(self.counterexample_id) <= 128
        ):
            raise ValueError("counterexample_id is invalid")
        if not isinstance(self.task_id, str) or not 1 <= len(self.task_id) <= 128:
            raise ValueError("counterexample task_id must contain 1..128 characters")
        if not _is_sha256(self.candidate_hash) or not _is_sha256(self.replay_evidence_ref):
            raise ValueError("counterexample hashes must be lowercase sha256 identifiers")
        if not isinstance(self.failure_signature, str) or not (
            1 <= len(self.failure_signature) <= 512
        ):
            raise ValueError("counterexample failure signature must contain 1..512 characters")
        if (
            isinstance(self.robustness, bool)
            or not isinstance(self.robustness, (int, float))
            or not math.isfinite(float(self.robustness))
            or self.robustness >= 0
        ):
            raise ValueError("counterexample robustness must be finite and negative")
        if not isinstance(self.scenario, ScenarioSample):
            raise ValueError("counterexample scenario must be ScenarioSample")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scenario": self.scenario.to_dict(reveal_seed=True),
        }


class Falsifier:
    def __init__(self, sampler: ScenarioSampler) -> None:
        self.sampler = sampler

    def search(
        self,
        *,
        task_id: str,
        candidate_hash: str,
        seed: int,
        budget: int,
        evaluator: Callable[[ScenarioSample], tuple[float, str, str]],
    ) -> tuple[Counterexample, ...]:
        if not isinstance(task_id, str) or not 1 <= len(task_id) <= 128:
            raise ValueError("falsification task_id must contain 1..128 characters")
        if not _is_sha256(candidate_hash):
            raise ValueError("falsification candidate_hash must be a sha256 identifier")
        if not callable(evaluator):
            raise ValueError("falsification evaluator must be callable")
        samples = self.sampler.sample(
            count=budget,
            seed=seed,
            partition=Partition.STRESS,
            strategy=SamplingStrategy.BOUNDARY,
        )
        failures: list[Counterexample] = []
        for scenario in samples:
            robustness, signature, evidence_ref = evaluator(scenario)
            if (
                isinstance(robustness, bool)
                or not isinstance(robustness, (int, float))
                or not math.isfinite(float(robustness))
            ):
                raise ValueError("falsification evaluator returned non-finite robustness")
            if not isinstance(signature, str) or not 1 <= len(signature) <= 512:
                raise ValueError("falsification evaluator returned an empty failure signature")
            if not _is_sha256(evidence_ref):
                raise ValueError("falsification evidence reference must be a sha256 identifier")
            if robustness >= 0:
                continue
            identity = json.dumps(
                [task_id, candidate_hash, scenario.scenario_id, signature], separators=(",", ":")
            )
            failures.append(
                Counterexample(
                    counterexample_id="counterexample_"
                    + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    task_id=task_id,
                    candidate_hash=candidate_hash,
                    failure_signature=signature,
                    robustness=float(robustness),
                    scenario=scenario,
                    replay_evidence_ref=evidence_ref,
                )
            )
        return tuple(sorted(failures, key=lambda item: item.robustness))

    @staticmethod
    def minimize(
        counterexample: Counterexample,
        *,
        nominal: Mapping[str, Any],
        still_fails: Callable[[Mapping[str, Any]], bool],
        rounds: int = 12,
    ) -> dict[str, Any]:
        if not isinstance(counterexample, Counterexample):
            raise ValueError("counterexample must be Counterexample")
        if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 100:
            raise ValueError("minimization rounds must be in [1, 100]")
        if len(nominal) > 10_000:
            raise ValueError("nominal scenario is too large")
        current = dict(counterexample.scenario.values)
        for _ in range(rounds):
            changed = False
            for name in sorted(current):
                if name not in nominal or not isinstance(current[name], (int, float)):
                    continue
                proposal = dict(current)
                proposal[name] = (float(current[name]) + float(nominal[name])) / 2
                if still_fails(proposal):
                    current = proposal
                    changed = True
            if not changed:
                break
        return current


class CounterexampleStore:
    """Append-only content-addressed storage, forbidden inside the source checkout."""

    def __init__(self, *, root: Path, source_checkout: Path) -> None:
        self.root = root.resolve()
        checkout = source_checkout.resolve()
        if self.root == checkout or checkout in self.root.parents:
            raise ValueError("counterexample artifacts must be outside the source checkout")
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, counterexample: Counterexample) -> Path:
        payload = json.dumps(
            counterexample.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if len(payload) > 1_048_576:
            raise ValueError("counterexample artifact exceeds 1 MiB")
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / "sha256" / digest[:2] / digest[2:4] / f"{digest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size > 1_048_576:
                raise RuntimeError("existing counterexample artifact exceeds 1 MiB")
            with target.open("rb") as handle:
                existing = handle.read(1_048_577)
            if len(existing) > 1_048_576 or hashlib.sha256(existing).hexdigest() != digest:
                raise RuntimeError("counterexample content-address collision")
            return target
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while storing counterexample")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        return target


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


__all__ = ["Counterexample", "CounterexampleStore", "Falsifier"]
