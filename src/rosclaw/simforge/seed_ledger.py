"""Partition-aware deterministic seed ledger with holdout-safe exports."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Any

from rosclaw.simforge.models import Partition


@dataclass(frozen=True)
class SeedRecord:
    partition: Partition
    index: int
    seed: int
    commitment: str


class SeedLedger:
    """Derive disjoint partition seeds and reveal only cryptographic commitments."""

    def __init__(self, *, task_id: str, secret: bytes | None = None) -> None:
        if not task_id:
            raise ValueError("task_id is required")
        self.task_id = task_id
        self._secret = secret or secrets.token_bytes(32)
        if len(self._secret) < 16:
            raise ValueError("seed ledger secret must contain at least 16 bytes")
        self._records: dict[tuple[Partition, int], SeedRecord] = {}
        self._owners: dict[int, tuple[Partition, int]] = {}

    def derive(self, partition: Partition, index: int) -> SeedRecord:
        if index < 0 or index > 10_000_000:
            raise ValueError("seed index must be in [0, 10000000]")
        key = (partition, index)
        existing = self._records.get(key)
        if existing is not None:
            return existing
        message = f"rosclaw.simforge.seed.v1\0{self.task_id}\0{partition.value}\0{index}"
        digest = hmac.new(self._secret, message.encode(), hashlib.sha256).digest()
        seed = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
        owner = self._owners.get(seed)
        if owner is not None and owner != key:
            raise RuntimeError("seed collision detected across evaluation partitions")
        commitment = "sha256:" + hashlib.sha256(digest).hexdigest()
        record = SeedRecord(partition=partition, index=index, seed=seed, commitment=commitment)
        self._records[key] = record
        self._owners[seed] = key
        return record

    def allocate(self, partition: Partition, count: int) -> tuple[SeedRecord, ...]:
        if count < 1 or count > 1_000_000:
            raise ValueError("seed count must be in [1, 1000000]")
        return tuple(self.derive(partition, index) for index in range(count))

    def public_manifest(self) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in sorted(
            self._records.values(), key=lambda item: (item.partition.value, item.index)
        ):
            entry: dict[str, Any] = {
                "index": record.index,
                "commitment": record.commitment,
            }
            if record.partition.candidate_may_view_cases:
                entry["seed"] = record.seed
            grouped.setdefault(record.partition.value, []).append(entry)
        result = {
            "schema_version": "rosclaw.simforge.seed_ledger.v1",
            "task_id": self.task_id,
            "partitions": grouped,
        }
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
        result["manifest_hash"] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        return result

    def assert_disjoint(self) -> None:
        if len(self._owners) != len(self._records):
            raise RuntimeError("seed reuse detected across evaluation partitions")


__all__ = ["SeedLedger", "SeedRecord"]
