"""Crash-recoverable one-use bookkeeping, NOT sealed-data access isolation.

An external evaluator must authenticate the stage receipts, hold the dataset
outside the learner's access, and consume before revealing data or scores.
Crashing after consume intentionally burns the bank; there is no reset API.
This journal never opens a dataset or authorizes training/promotion.
"""

from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path

from rosclaw.continual.services.persistence import DurableEventLog, require_external_service_root
from rosclaw.growth.research import ResearchCampaign, _hash


class SealedUseLedger:
    def __init__(self, root: Path, *, source_checkout: Path) -> None:
        self.root = require_external_service_root(root, source_checkout)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.Lock()
        self._descriptor = os.open(
            self.root / "writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        self._closed = False
        self._failed = False
        self._consumed: set[str] = set()
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._log = DurableEventLog(self.root, service="research.sealed_use")
            for event in self._log.events:
                if event.kind != "SEALED_CONSUMED" or set(event.payload) != {
                    "campaign_hash",
                    "sealed_commitment",
                    "candidate_hash",
                    "evaluation_hash",
                    "development_receipt_hash",
                    "retention_receipt_hash",
                }:
                    raise ValueError("unrecognized sealed-use journal event")
                for value in event.payload.values():
                    _hash(value)
                bank = str(event.payload["sealed_commitment"])
                if bank in self._consumed:
                    raise ValueError("duplicate sealed consumption in journal")
                self._consumed.add(bank)
        except BaseException:
            os.close(self._descriptor)
            self._closed = True
            raise

    def consume(
        self,
        campaign: ResearchCampaign,
        *,
        candidate_hash: str,
        evaluation_hash: str,
        development_receipt_hash: str,
        retention_receipt_hash: str,
    ) -> str:
        """Burn the committed bank durably before the external evaluator opens it."""
        with self._mutex:
            if self._closed or self._failed:
                raise RuntimeError("sealed-use ledger is closed or failed")
            campaign.__post_init__()
            for value in (
                candidate_hash,
                evaluation_hash,
                development_receipt_hash,
                retention_receipt_hash,
            ):
                _hash(value)
            if evaluation_hash != campaign.hypothesis.evaluation_contract_hash:
                raise ValueError("evaluation changed after campaign commitment")
            if development_receipt_hash == retention_receipt_hash:
                raise ValueError("development and retention require distinct stage receipts")
            if campaign.sealed_commitment in self._consumed:
                raise ValueError("sealed bank already consumed; retries cannot reopen it")
            payload = {
                "campaign_hash": campaign.campaign_hash,
                "sealed_commitment": campaign.sealed_commitment,
                "candidate_hash": candidate_hash,
                "evaluation_hash": evaluation_hash,
                "development_receipt_hash": development_receipt_hash,
                "retention_receipt_hash": retention_receipt_hash,
            }
            try:
                event = self._log.append("SEALED_CONSUMED", payload)
            except BaseException:
                self._failed = True  # write/fsync ambiguity must never allow an in-process retry
                raise
            self._consumed.add(campaign.sealed_commitment)
            return event.event_hash

    def close(self) -> None:
        with self._mutex:
            if not self._closed:
                os.close(self._descriptor)
                self._closed = True

    def __enter__(self) -> SealedUseLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
