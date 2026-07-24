"""Practice records and immutable dataset snapshots for G1 GoalForge."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.dataset_snapshot import (
    PracticeDatasetBuilder,
    PracticeDatasetSnapshot,
    PracticeEpisodeRecord,
    SnapshotFiles,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GOALFORGE_TASK_ID,
    GoalForgeStatus,
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.failure_signature import FailureSignatureV3


@dataclass(frozen=True)
class GoalForgeSemanticEvent:
    event_type: str
    episode_id: str
    payload_hash: str
    causal_parent_hashes: tuple[str, ...] = ()
    schema_version: str = "rosclaw.g1_goalforge.semantic_event.v1"

    def __post_init__(self) -> None:
        if not self.event_type or not self.episode_id:
            raise ValueError("semantic event identity is required")
        hashes = (self.payload_hash, *self.causal_parent_hashes)
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("semantic event references must be sha256 digests")


@dataclass(frozen=True)
class GoalForgePracticeRecord:
    episode: GoalForgeEpisode
    practice_id: str
    failure_signature: FailureSignatureV3 | None
    best_patch: ShotParameters | None
    semantic_events: tuple[GoalForgeSemanticEvent, ...]

    def to_episode_record(self) -> PracticeEpisodeRecord:
        episode = self.episode
        if episode.receipt is None or episode.artifact_root is None:
            raise ValueError("Practice records require a recorded physical episode")
        observed = episode.scenario.observed_context()
        result = episode.result
        labels: dict[str, float | bool | str] = {
            "success": result.success,
            "target_zone": episode.scenario.target_zone,
            "target_error": (
                result.target_error_m if math.isfinite(result.target_error_m) else 99.0
            ),
            "ball_speed": result.ball_speed_mps,
            "contact_phase": (
                result.ball_contact_time_sec if result.ball_contact_time_sec is not None else -1.0
            ),
            "support_slip": result.support_foot_slip_m,
            "stability": result.post_kick_stability_time_sec,
            "failure_signature": (
                self.failure_signature.failure_type.value
                if self.failure_signature is not None
                else GoalForgeStatus.SUCCESS.value
            ),
            "best_patch": (self.best_patch.policy_hash if self.best_patch is not None else "none"),
        }
        artifacts = (
            episode.receipt.request_hash,
            episode.receipt.trajectory_hash,
            episode.receipt.result_hash,
            episode.receipt.receipt_hash,
            *(event.payload_hash for event in self.semantic_events),
        )
        return PracticeEpisodeRecord(
            episode_id=episode.receipt.episode_id,
            practice_id=self.practice_id,
            scenario_id=episode.scenario.scenario_id,
            seed_commitment=episode.scenario.seed_commitment,
            body_snapshot_hash=episode.receipt.body_hash,
            task_id=GOALFORGE_TASK_ID,
            features=tuple(sorted((key, float(value)) for key, value in observed.items())),
            policy=tuple(
                sorted(
                    (
                        key,
                        "none" if value is None else value,
                    )
                    for key, value in episode.parameters.to_dict().items()
                )
            ),
            labels=tuple(sorted(labels.items())),
            artifact_hashes=tuple(dict.fromkeys(artifacts)),
            complete=result.physics_executed and bool(episode.trajectory),
            independently_verified=episode.receipt.independently_verified,
            strict_replay=episode.receipt.strict_replay,
        )


@dataclass(frozen=True)
class KickPracticeDatasetSnapshot:
    base: PracticeDatasetSnapshot
    generation: int
    body_hash: str
    kick_prior_hash: str
    episode_counts: tuple[tuple[str, int], ...]
    labels: tuple[str, ...]
    incomplete_episode_rate: float
    schema_version: str = "rosclaw.kick_practice_dataset_snapshot.v1"

    def __post_init__(self) -> None:
        if not 0 <= self.generation <= 10:
            raise ValueError("dataset generation must be in [0, 10]")
        if self.body_hash not in self.base.body_hashes:
            raise ValueError("GoalForge body hash is absent from base snapshot")
        if not self.kick_prior_hash.startswith("sha256:"):
            raise ValueError("kick prior hash must be a sha256 digest")
        if not 0.0 <= self.incomplete_episode_rate <= 1.0:
            raise ValueError("incomplete episode rate must be in [0, 1]")
        required = {
            "target_zone",
            "target_error",
            "ball_speed",
            "contact_phase",
            "support_slip",
            "stability",
            "failure_signature",
            "best_patch",
        }
        if not required.issubset(self.labels):
            raise ValueError("GoalForge dataset labels are incomplete")

    @property
    def snapshot_hash(self) -> str:
        return hash_json(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "dataset_id": self.base.dataset_id,
            "generation": self.generation,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "base_snapshot_hash": self.base.snapshot_hash,
            "episode_counts": dict(self.episode_counts),
            "partitions": self.base.to_dict()["partitions"],
            "labels": list(self.labels),
            "quality": {
                **asdict(self.base.quality),
                "incomplete_episode_rate": self.incomplete_episode_rate,
            },
        }
        if include_hash:
            value["snapshot_hash"] = hash_json(value)
        return value


def build_kick_dataset(
    *,
    records: tuple[GoalForgePracticeRecord, ...],
    output_dir: Path,
    source_checkout: Path,
    split_secret: bytes,
    dataset_id: str,
    generation: int,
    body_hash: str,
    kick_prior_hash: str,
) -> tuple[KickPracticeDatasetSnapshot, SnapshotFiles]:
    if not records:
        raise ValueError("GoalForge dataset requires Practice records")
    generic_records = tuple(record.to_episode_record() for record in records)
    base, files = PracticeDatasetBuilder(
        source_checkout=source_checkout,
        split_secret=split_secret,
    ).build(
        records=generic_records,
        output_dir=output_dir,
        dataset_id=dataset_id,
        label_provenance={
            "target": "GoalForge independent physical verifier",
            "contact": "MuJoCo contact impulse integration",
            "safety": "G1 joint/torque/contact trajectory channels",
            "best_patch": "bounded Auto teacher search",
        },
    )
    status_counts = Counter(record.episode.result.status.value.lower() for record in records)
    categories = (
        ("success", status_counts.get("success", 0)),
        (
            "target_miss",
            sum(count for name, count in status_counts.items() if name.startswith("target_miss")),
        ),
        ("ball_miss", status_counts.get("ball_not_contacted", 0)),
        ("weak_shot", status_counts.get("shot_too_weak", 0)),
        ("fall", status_counts.get("post_kick_fall", 0)),
        ("support_slip", status_counts.get("support_foot_slip", 0)),
        (
            "runtime_fault",
            sum(
                status_counts.get(name, 0)
                for name in (
                    "agent_lost",
                    "policy_worker_crash",
                    "dds_lost",
                    "state_feedback_stale",
                )
            ),
        ),
    )
    snapshot = KickPracticeDatasetSnapshot(
        base=base,
        generation=generation,
        body_hash=body_hash,
        kick_prior_hash=kick_prior_hash,
        episode_counts=categories,
        labels=(
            "target_zone",
            "target_error",
            "ball_speed",
            "contact_phase",
            "support_slip",
            "stability",
            "failure_signature",
            "best_patch",
        ),
        incomplete_episode_rate=1.0 - base.quality.complete_episode_rate,
    )
    manifest = files.manifest.parent / "kick-snapshot.json"
    manifest.write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot, files


__all__ = [
    "GoalForgePracticeRecord",
    "GoalForgeSemanticEvent",
    "KickPracticeDatasetSnapshot",
    "build_kick_dataset",
]
