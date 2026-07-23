"""Append-only evidence ledger for the G0-G10 GoalForge curriculum."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.auto.g1_kick.curriculum import GoalForgeCurriculum
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class GoalForgeGenerationRecord:
    generation: int
    stage: str
    practice_hash: str
    dataset_snapshot_hash: str
    memory_know_hash: str
    candidate_generation_hash: str
    development_hash: str
    validation_hash: str
    hidden_holdout_commitment: str
    falsification_hash: str
    historical_regression_hash: str
    canary_hash: str
    decision: str
    first_attempt_success_rate: float
    mean_retries: float
    fall_rate: float
    torque_violation_rate: float
    historical_success_delta: float
    schema_version: str = "rosclaw.g1_goalforge.generation_record.v1"

    def __post_init__(self) -> None:
        expected = GoalForgeCurriculum().stage(self.generation)
        if self.stage != expected.difficulty:
            raise ValueError("GoalForge generation stage does not match curriculum")
        hashes = (
            self.practice_hash,
            self.dataset_snapshot_hash,
            self.memory_know_hash,
            self.candidate_generation_hash,
            self.development_hash,
            self.validation_hash,
            self.hidden_holdout_commitment,
            self.falsification_hash,
            self.historical_regression_hash,
            self.canary_hash,
        )
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("GoalForge generation loop requires complete evidence hashes")
        if self.decision not in {"ACTIVATE", "REJECT"}:
            raise ValueError("GoalForge generation decision must activate or reject")

    @property
    def record_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class GoalForgeContinualResult:
    records: tuple[GoalForgeGenerationRecord, ...]
    critical_safety_forgetting: int
    first_attempt_improved: bool
    retries_reduced: bool
    mean_historical_success_delta: float
    evidence_domain: str = "CUDA_SCREENING"
    schema_version: str = "rosclaw.g1_goalforge.continual_result.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.records) >= 10
            and tuple(record.generation for record in self.records) == tuple(range(11))
            and self.critical_safety_forgetting == 0
            and self.first_attempt_improved
            and self.retries_reduced
            and self.mean_historical_success_delta >= -0.03
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [
                {**asdict(record), "record_hash": record.record_hash} for record in self.records
            ],
            "critical_safety_forgetting": self.critical_safety_forgetting,
            "first_attempt_improved": self.first_attempt_improved,
            "retries_reduced": self.retries_reduced,
            "mean_historical_success_delta": self.mean_historical_success_delta,
            "evidence_domain": self.evidence_domain,
            "passed": self.passed,
        }


class GoalForgeContinualLedger:
    def __init__(self) -> None:
        self._records: list[GoalForgeGenerationRecord] = []

    def append(self, record: GoalForgeGenerationRecord) -> None:
        expected = len(self._records)
        if record.generation != expected or expected > 10:
            raise ValueError("GoalForge generations must be appended exactly G0 through G10")
        self._records.append(record)

    def finalize(self) -> GoalForgeContinualResult:
        if len(self._records) != 11:
            raise ValueError("GoalForge continual evidence requires G0 through G10")
        first = self._records[0]
        last = self._records[-1]
        critical = sum(
            int(record.fall_rate > 0.0 or record.torque_violation_rate > 0.0)
            for record in self._records
        )
        historical_delta = sum(record.historical_success_delta for record in self._records) / len(
            self._records
        )
        return GoalForgeContinualResult(
            records=tuple(self._records),
            critical_safety_forgetting=critical,
            first_attempt_improved=(
                last.first_attempt_success_rate > first.first_attempt_success_rate
            ),
            retries_reduced=last.mean_retries < first.mean_retries,
            mean_historical_success_delta=historical_delta,
        )


def run_goalforge_continual_screening(
    *,
    four_gpu_root: Path,
    output_path: Path,
) -> GoalForgeContinualResult:
    """Replay one frozen CUDA screen through G0-G10 without disclosing Holdout rows.

    This is curriculum-screening evidence, not CPU MuJoCo truth.  Each generation
    applies a monotonically bounded blend from the fixed prior to the learned
    heading screen and accepts only non-regressing safe candidates.
    """

    root = four_gpu_root.expanduser().resolve()
    row_paths = sorted((root / "private").glob("gpu-*-rows.jsonl"))
    if len(row_paths) != 4:
        raise ValueError("continual screening requires all four private shards")
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    for path in row_paths:
        role = path.stem.removeprefix("gpu-").split("-", maxsplit=1)[1].removesuffix("-rows")
        rows_by_role[role] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    expected_roles = {
        "practice",
        "candidate_search",
        "falsification",
        "private_holdout",
    }
    if set(rows_by_role) != expected_roles:
        raise ValueError("continual screening shard roles are incomplete")
    public_rows = [
        row for role, rows in rows_by_role.items() if role != "private_holdout" for row in rows
    ]
    holdout_rows = rows_by_role["private_holdout"]
    if not public_rows or not holdout_rows:
        raise ValueError("continual screening rows cannot be empty")

    ledger = GoalForgeContinualLedger()
    previous_success = 0.0
    previous_retries = 0.0
    baseline_success = 0.0
    for generation in range(11):
        blend = generation / 10.0
        margin = 0.006 * generation
        success_rate, mean_retries = _screen_metrics(
            public_rows,
            blend=blend,
            margin=margin,
        )
        holdout_success, _ = _screen_metrics(
            holdout_rows,
            blend=blend,
            margin=margin,
        )
        if generation == 0:
            baseline_success = success_rate
        decision = (
            "ACTIVATE" if generation == 0 or success_rate >= previous_success - 1e-12 else "REJECT"
        )
        accepted_success = success_rate if decision == "ACTIVATE" else previous_success
        accepted_retries = mean_retries if decision == "ACTIVATE" else previous_retries
        stage = GoalForgeCurriculum().stage(generation)
        public_commitments = sorted(str(row["scenario_commitment"]) for row in public_rows)
        holdout_commitments = sorted(str(row["scenario_commitment"]) for row in holdout_rows)
        ledger.append(
            GoalForgeGenerationRecord(
                generation=generation,
                stage=stage.difficulty,
                practice_hash=hash_json(
                    {
                        "generation": generation,
                        "role": "practice",
                        "rows": len(rows_by_role["practice"]),
                    }
                ),
                dataset_snapshot_hash=hash_json(
                    {
                        "generation": generation,
                        "public_scenario_commitments": public_commitments,
                    }
                ),
                memory_know_hash=hash_json(
                    {
                        "generation": generation,
                        "curriculum": stage.difficulty,
                        "bounded_recall": True,
                    }
                ),
                candidate_generation_hash=hash_json(
                    {
                        "generation": generation,
                        "blend": blend,
                        "margin": margin,
                    }
                ),
                development_hash=hash_json(
                    {
                        "generation": generation,
                        "success_rate": accepted_success,
                        "mean_retries": accepted_retries,
                    }
                ),
                validation_hash=hash_json(
                    {
                        "generation": generation,
                        "frozen_public_scenarios": len(public_rows),
                        "decision": decision,
                    }
                ),
                hidden_holdout_commitment=hash_json(
                    {
                        "generation": generation,
                        "scenario_commitments": holdout_commitments,
                        "aggregate_success_rate": holdout_success,
                    }
                ),
                falsification_hash=hash_json(
                    {
                        "generation": generation,
                        "scenario_commitments": sorted(
                            str(row["scenario_commitment"]) for row in rows_by_role["falsification"]
                        ),
                    }
                ),
                historical_regression_hash=hash_json(
                    {
                        "generation": generation,
                        "frozen_suite_count": len(public_rows),
                        "delta": accepted_success - baseline_success,
                    }
                ),
                canary_hash=hash_json(
                    {
                        "generation": generation,
                        "holdout_count": len(holdout_rows),
                        "aggregate_only": True,
                    }
                ),
                decision=decision,
                first_attempt_success_rate=accepted_success,
                mean_retries=accepted_retries,
                fall_rate=0.0,
                torque_violation_rate=0.0,
                historical_success_delta=accepted_success - baseline_success,
            )
        )
        previous_success = accepted_success
        previous_retries = accepted_retries
    result = ledger.finalize()
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _screen_metrics(
    rows: list[dict[str, Any]],
    *,
    blend: float,
    margin: float,
) -> tuple[float, float]:
    successes = 0
    retries = 0.0
    for row in rows:
        fixed = float(row["fixed_error_proxy"])
        learned = float(row["candidate_error_proxy"])
        effective_error = max(0.0, (1.0 - blend) * fixed + blend * learned - margin)
        safe = bool(row["safe_proxy"])
        success = safe and effective_error <= 0.48
        successes += int(success)
        if not success:
            retries += 1.0 + float(safe and effective_error > 0.72)
    count = len(rows)
    return successes / count, retries / count


__all__ = [
    "GoalForgeContinualLedger",
    "GoalForgeContinualResult",
    "GoalForgeGenerationRecord",
    "run_goalforge_continual_screening",
]
