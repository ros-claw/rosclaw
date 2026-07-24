"""Immutable, leakage-checked Practice dataset snapshots."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.models import Partition

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class PracticeEpisodeRecord:
    episode_id: str
    practice_id: str
    scenario_id: str
    seed_commitment: str
    body_snapshot_hash: str
    task_id: str
    features: tuple[tuple[str, float], ...]
    policy: tuple[tuple[str, float | bool | str], ...]
    labels: tuple[tuple[str, float | bool | str], ...]
    artifact_hashes: tuple[str, ...]
    complete: bool
    independently_verified: bool
    strict_replay: bool

    def __post_init__(self) -> None:
        if not all((self.episode_id, self.practice_id, self.scenario_id, self.task_id)):
            raise ValueError("practice episode identity is required")
        if not _SHA256_RE.fullmatch(self.seed_commitment):
            raise ValueError("seed commitment must be a sha256 identifier")
        if not _SHA256_RE.fullmatch(self.body_snapshot_hash):
            raise ValueError("body snapshot hash must be a sha256 identifier")
        if not self.features or not self.policy or not self.labels:
            raise ValueError("practice episodes require features, policy, and verifier labels")
        if len({key for key, _ in self.features}) != len(self.features):
            raise ValueError("practice episode feature names must be unique")
        if len({key for key, _ in self.policy}) != len(self.policy):
            raise ValueError("practice episode policy names must be unique")
        if len({key for key, _ in self.labels}) != len(self.labels):
            raise ValueError("practice episode label names must be unique")
        for _key, value in self.features:
            if not math.isfinite(value):
                raise ValueError("practice episode features must be finite")
        for _key, value in self.policy:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("practice episode policy values must be finite")
        if not self.artifact_hashes or any(
            not _SHA256_RE.fullmatch(value) for value in self.artifact_hashes
        ):
            raise ValueError("practice episodes require sha256 artifact hashes")

    @property
    def record_hash(self) -> str:
        return _hash_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "practice_id": self.practice_id,
            "scenario_id": self.scenario_id,
            "seed_commitment": self.seed_commitment,
            "body_snapshot_hash": self.body_snapshot_hash,
            "task_id": self.task_id,
            "features": dict(self.features),
            "policy": dict(self.policy),
            "labels": dict(self.labels),
            "artifact_hashes": list(self.artifact_hashes),
            "complete": self.complete,
            "independently_verified": self.independently_verified,
            "strict_replay": self.strict_replay,
        }


@dataclass(frozen=True)
class DatasetPartition:
    partition: Partition
    count: int
    content_hash: str
    scenario_set_commitment: str
    file_ref: str | None

    def __post_init__(self) -> None:
        if self.partition not in {
            Partition.DEVELOPMENT,
            Partition.VALIDATION,
            Partition.HOLDOUT,
        }:
            raise ValueError("dataset snapshot partition is not train/validation/holdout")
        if self.count < 1:
            raise ValueError("dataset partitions cannot be empty")
        if not _SHA256_RE.fullmatch(self.content_hash) or not _SHA256_RE.fullmatch(
            self.scenario_set_commitment
        ):
            raise ValueError("dataset partition hashes must be sha256 identifiers")
        if self.partition is Partition.HOLDOUT and self.file_ref is not None:
            raise ValueError("public snapshots cannot disclose the holdout file")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "partition": self.partition.value,
            "count": self.count,
            "hash": self.content_hash,
            "scenario_set_commitment": self.scenario_set_commitment,
        }
        if self.file_ref is not None:
            value["file_ref"] = self.file_ref
        return value


@dataclass(frozen=True)
class DatasetQuality:
    complete_episode_rate: float
    artifact_hash_pass_rate: float
    independent_verification_rate: float
    strict_replay_rate: float
    split_leakage: bool

    def __post_init__(self) -> None:
        rates = (
            self.complete_episode_rate,
            self.artifact_hash_pass_rate,
            self.independent_verification_rate,
            self.strict_replay_rate,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in rates):
            raise ValueError("dataset quality rates must be in [0, 1]")

    @property
    def passes(self) -> bool:
        return (
            self.complete_episode_rate == 1.0
            and self.artifact_hash_pass_rate == 1.0
            and self.independent_verification_rate == 1.0
            and self.strict_replay_rate == 1.0
            and not self.split_leakage
        )


@dataclass(frozen=True)
class PracticeDatasetSnapshot:
    dataset_id: str
    task_id: str
    source_domain: str
    body_hashes: tuple[str, ...]
    practice_ids: tuple[str, ...]
    label_provenance: tuple[tuple[str, str], ...]
    partitions: tuple[DatasetPartition, ...]
    quality: DatasetQuality
    schema_version: str = "rosclaw.practice_dataset_snapshot.v1"

    def __post_init__(self) -> None:
        if not self.dataset_id.startswith("dataset_") or not self.task_id:
            raise ValueError("dataset snapshot identity is invalid")
        if self.source_domain != "SIMULATION":
            raise ValueError("SimForge dataset snapshots must remain in SIMULATION")
        if not self.body_hashes or any(
            not _SHA256_RE.fullmatch(value) for value in self.body_hashes
        ):
            raise ValueError("dataset snapshot requires sha256 body hashes")
        if not self.practice_ids or not self.label_provenance:
            raise ValueError("dataset snapshot requires source practices and label provenance")
        if {item.partition for item in self.partitions} != {
            Partition.DEVELOPMENT,
            Partition.VALIDATION,
            Partition.HOLDOUT,
        }:
            raise ValueError("dataset snapshot requires train, validation, and holdout")
        if not self.quality.passes:
            raise ValueError("dataset quality gate failed")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "created_from": {"practice_ids": list(self.practice_ids)},
            "source_domain": self.source_domain,
            "body_hashes": list(self.body_hashes),
            "label_provenance": dict(self.label_provenance),
            "partitions": {
                item.partition.value: item.to_dict()
                for item in sorted(self.partitions, key=lambda value: value.partition.value)
            },
            "quality": {**asdict(self.quality), "passes": self.quality.passes},
        }
        value["snapshot_hash"] = _hash_json(value)
        return value

    @property
    def snapshot_hash(self) -> str:
        return str(self.to_dict()["snapshot_hash"])


@dataclass(frozen=True)
class SnapshotFiles:
    manifest: Path
    development: Path
    validation: Path
    private_holdout: Path


class PracticeDatasetBuilder:
    """Build an immutable grouped split; Holdout rows never enter the manifest."""

    def __init__(self, *, source_checkout: Path, split_secret: bytes) -> None:
        self.source_checkout = source_checkout.resolve()
        if len(split_secret) < 16:
            raise ValueError("dataset split secret must contain at least 16 bytes")
        self._secret = bytes(split_secret)

    def build(
        self,
        *,
        records: tuple[PracticeEpisodeRecord, ...],
        output_dir: Path,
        dataset_id: str,
        label_provenance: dict[str, str],
    ) -> tuple[PracticeDatasetSnapshot, SnapshotFiles]:
        root = output_dir.expanduser().resolve()
        if root == self.source_checkout or self.source_checkout in root.parents:
            raise ValueError("Practice dataset output must be outside the source checkout")
        if len(records) < 12:
            raise ValueError("dataset snapshots require at least 12 episodes")
        if len({record.episode_id for record in records}) != len(records):
            raise ValueError("dataset episode ids must be unique")
        task_ids = {record.task_id for record in records}
        if len(task_ids) != 1:
            raise ValueError("dataset snapshot cannot mix tasks")
        groups: dict[str, list[PracticeEpisodeRecord]] = {}
        for record in records:
            groups.setdefault(record.scenario_id, []).append(record)
        if len(groups) < 6:
            raise ValueError("dataset snapshots require at least six distinct scenarios")
        assignments = self._partition_groups(groups)
        if self._has_leakage(assignments):
            raise RuntimeError("dataset split leakage detected")
        all_records = list(records)
        quality = DatasetQuality(
            complete_episode_rate=_rate(record.complete for record in all_records),
            artifact_hash_pass_rate=_rate(
                bool(record.artifact_hashes)
                and all(_SHA256_RE.fullmatch(value) for value in record.artifact_hashes)
                for record in all_records
            ),
            independent_verification_rate=_rate(
                record.independently_verified for record in all_records
            ),
            strict_replay_rate=_rate(record.strict_replay for record in all_records),
            split_leakage=False,
        )
        root.mkdir(parents=True, exist_ok=False)
        development_path = root / "development.jsonl"
        validation_path = root / "validation.jsonl"
        private_holdout_path = root / "holdout-private.json"
        manifest_path = root / "snapshot.json"
        partition_files = {
            Partition.DEVELOPMENT: development_path,
            Partition.VALIDATION: validation_path,
            Partition.HOLDOUT: private_holdout_path,
        }
        partition_contracts = []
        for partition in (
            Partition.DEVELOPMENT,
            Partition.VALIDATION,
            Partition.HOLDOUT,
        ):
            rows = sorted(assignments[partition], key=lambda item: item.episode_id)
            payload = [row.to_private_dict() for row in rows]
            serialized = "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) for row in payload
            )
            if partition is Partition.HOLDOUT:
                _write_private(
                    private_holdout_path,
                    json.dumps(
                        {
                            "schema_version": "rosclaw.practice_holdout_private.v1",
                            "dataset_id": dataset_id,
                            "episodes": payload,
                        },
                        sort_keys=True,
                    ).encode(),
                )
            else:
                partition_files[partition].write_text(serialized + "\n", encoding="utf-8")
            scenarios = sorted({row.scenario_id for row in rows})
            partition_contracts.append(
                DatasetPartition(
                    partition=partition,
                    count=len(rows),
                    content_hash=(
                        _hash_bytes(private_holdout_path.read_bytes())
                        if partition is Partition.HOLDOUT
                        else _hash_bytes((serialized + "\n").encode())
                    ),
                    scenario_set_commitment=_hash_json({"scenario_ids": scenarios}),
                    file_ref=(
                        partition_files[partition].name
                        if partition is not Partition.HOLDOUT
                        else None
                    ),
                )
            )
        snapshot = PracticeDatasetSnapshot(
            dataset_id=dataset_id,
            task_id=next(iter(task_ids)),
            source_domain="SIMULATION",
            body_hashes=tuple(sorted({record.body_snapshot_hash for record in records})),
            practice_ids=tuple(sorted({record.practice_id for record in records})),
            label_provenance=tuple(sorted(label_provenance.items())),
            partitions=tuple(partition_contracts),
            quality=quality,
        )
        _atomic_json(manifest_path, snapshot.to_dict())
        return snapshot, SnapshotFiles(
            manifest=manifest_path,
            development=development_path,
            validation=validation_path,
            private_holdout=private_holdout_path,
        )

    def _partition_groups(
        self,
        groups: dict[str, list[PracticeEpisodeRecord]],
    ) -> dict[Partition, list[PracticeEpisodeRecord]]:
        ranked = sorted(
            groups,
            key=lambda scenario_id: hmac.new(
                self._secret,
                f"rosclaw.practice.split.v1\0{scenario_id}".encode(),
                hashlib.sha256,
            ).digest(),
        )
        count = len(ranked)
        development_end = max(1, math.floor(count * 0.70))
        validation_end = max(development_end + 1, math.floor(count * 0.85))
        validation_end = min(validation_end, count - 1)
        group_partitions = {
            Partition.DEVELOPMENT: ranked[:development_end],
            Partition.VALIDATION: ranked[development_end:validation_end],
            Partition.HOLDOUT: ranked[validation_end:],
        }
        result: dict[Partition, list[PracticeEpisodeRecord]] = {}
        for partition, scenario_ids in group_partitions.items():
            result[partition] = [
                record for scenario_id in scenario_ids for record in groups[scenario_id]
            ]
            if not result[partition]:
                raise RuntimeError(f"dataset split produced an empty {partition.value} partition")
        return result

    @staticmethod
    def _has_leakage(
        assignments: dict[Partition, list[PracticeEpisodeRecord]],
    ) -> bool:
        episode_sets = [
            {record.episode_id for record in records} for records in assignments.values()
        ]
        scenario_sets = [
            {record.scenario_id for record in records} for records in assignments.values()
        ]
        return any(
            left & right
            for index, left in enumerate(episode_sets)
            for right in episode_sets[index + 1 :]
        ) or any(
            left & right
            for index, left in enumerate(scenario_sets)
            for right in scenario_sets[index + 1 :]
        )


def load_public_partition(path: Path) -> tuple[PracticeEpisodeRecord, ...]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(_record_from_dict(json.loads(line)))
    return tuple(records)


def load_private_holdout(path: Path) -> tuple[PracticeEpisodeRecord, ...]:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError("private holdout must be mode 0600 or stricter")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "rosclaw.practice_holdout_private.v1":
        raise ValueError("invalid private holdout schema")
    return tuple(_record_from_dict(row) for row in value.get("episodes", []))


def _record_from_dict(value: dict[str, Any]) -> PracticeEpisodeRecord:
    return PracticeEpisodeRecord(
        episode_id=str(value["episode_id"]),
        practice_id=str(value["practice_id"]),
        scenario_id=str(value["scenario_id"]),
        seed_commitment=str(value["seed_commitment"]),
        body_snapshot_hash=str(value["body_snapshot_hash"]),
        task_id=str(value["task_id"]),
        features=tuple(sorted((str(key), float(item)) for key, item in value["features"].items())),
        policy=tuple(sorted((str(key), item) for key, item in value["policy"].items())),
        labels=tuple(sorted((str(key), item) for key, item in value["labels"].items())),
        artifact_hashes=tuple(map(str, value["artifact_hashes"])),
        complete=bool(value["complete"]),
        independently_verified=bool(value["independently_verified"]),
        strict_replay=bool(value["strict_replay"]),
    )


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _rate(values: Any) -> float:
    normalized = list(values)
    return sum(map(bool, normalized)) / len(normalized)


def _hash_json(value: dict[str, Any]) -> str:
    return _hash_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = [
    "DatasetPartition",
    "DatasetQuality",
    "PracticeDatasetBuilder",
    "PracticeDatasetSnapshot",
    "PracticeEpisodeRecord",
    "SnapshotFiles",
    "load_private_holdout",
    "load_public_partition",
]
