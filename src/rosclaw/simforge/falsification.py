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
        samples = self.sampler.sample(
            count=budget,
            seed=seed,
            partition=Partition.STRESS,
            strategy=SamplingStrategy.BOUNDARY,
        )
        failures: list[Counterexample] = []
        for scenario in samples:
            robustness, signature, evidence_ref = evaluator(scenario)
            if not math.isfinite(robustness):
                raise ValueError("falsification evaluator returned non-finite robustness")
            if not signature:
                raise ValueError("falsification evaluator returned an empty failure signature")
            if not evidence_ref.startswith("sha256:") or len(evidence_ref) != 71:
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
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / "sha256" / digest[:2] / digest[2:4] / f"{digest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("counterexample content-address collision")
            return target
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        return target


__all__ = ["Counterexample", "CounterexampleStore", "Falsifier"]
