"""Memory 2.0 write gate — what is allowed into long-term memory (§5.5, §5.8).

Default exclusions (never stored):

* per-frame RGB-D and per-sample IMU/telemetry;
* heartbeats and repeated health checks;
* complete model-internal CoT traces (§5.8 — only decision summaries);
* API keys, tokens, credentials;
* LLM inferences without evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from rosclaw.memory.v2.models import GateDecision, MemoryItem, MemoryType

logger = logging.getLogger("rosclaw.memory.v2.gate")

# Event types that are pure liveness/high-frequency signals, never memories.
_NOISE_EVENT_TYPES = {
    "heartbeat",
    "health_check",
    "imu_event",
    "frame_event",
    "rps.telemetry",
    "telemetry.tick",
}

# Secret/credential patterns — memory must never become a secret store.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|credential)\s*[:=]\s*\S+"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ak_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

# Memory types that require evidence before they may be stored.
_EVIDENCE_REQUIRED_TYPES = {
    MemoryType.FAILURE.value,
    MemoryType.INTERVENTION.value,
    MemoryType.BODY.value,
    MemoryType.SIM2REAL.value,
    MemoryType.HUMAN_FEEDBACK.value,
}

_SAFETY_KEYWORDS = re.compile(
    r"(?i)(unsafe|collision|emergency|e-?stop|overcurrent|overheat|thermal|限位|过流|过温|碰撞|急停)"
)


@dataclass
class MemoryDecision:
    """Outcome of :meth:`MemoryWriteGate.evaluate`."""

    decision: str
    reason: str
    target_memory_id: str | None = None
    redacted_fields: list[str] = field(default_factory=list)


class MemoryWriteGate:
    """Evaluates memory candidates and decides STORE/MERGE/UPDATE/IGNORE/QUARANTINE."""

    def __init__(
        self,
        repository: Any | None = None,
        *,
        store_full_cot: bool = False,
        duplicate_title_threshold: float = 0.85,
    ):
        self._repo = repository
        # §5.8: full CoT is off by default; enabling it quarantines the memory
        # with a mandatory TTL instead of storing it as a normal record.
        self._store_full_cot = store_full_cot
        self._dup_threshold = duplicate_title_threshold

    def evaluate(self, candidate: MemoryItem) -> MemoryDecision:
        """Decide the fate of a memory candidate."""
        # 1. Noise rejection — high-frequency signals are not memories.
        source_type = (candidate.metadata or {}).get("source_event_type", "")
        if source_type in _NOISE_EVENT_TYPES:
            return MemoryDecision(GateDecision.IGNORE.value, f"noise event type: {source_type}")
        if not candidate.document.strip() and not candidate.title.strip():
            return MemoryDecision(GateDecision.IGNORE.value, "empty document and title")

        # 2. Secret/credential rejection.
        leaked = self._find_secrets(candidate)
        if leaked:
            return MemoryDecision(
                GateDecision.IGNORE.value,
                f"contains secret-like content: {', '.join(leaked)}",
            )

        # 3. CoT policy (§5.8).
        redacted: list[str] = []
        if not self._store_full_cot:
            redacted = self._strip_cot(candidate)
        else:
            return MemoryDecision(
                GateDecision.QUARANTINE.value,
                "full CoT requires quarantine + TTL + debug tag",
            )

        # 4. Evidence requirements.
        if candidate.memory_type in _EVIDENCE_REQUIRED_TYPES and not candidate.evidence_refs:
            if candidate.memory_type in {MemoryType.FAILURE.value, MemoryType.INTERVENTION.value}:
                return MemoryDecision(
                    GateDecision.QUARANTINE.value,
                    f"{candidate.memory_type} memory without evidence",
                    redacted,
                )
            return MemoryDecision(
                GateDecision.IGNORE.value,
                f"{candidate.memory_type} memory requires evidence_refs",
                redacted,
            )

        # 5. Safety content must carry evidence; otherwise quarantine for review.
        if _SAFETY_KEYWORDS.search(candidate.document) and not candidate.evidence_refs:
            return MemoryDecision(
                GateDecision.QUARANTINE.value,
                "safety-related content without evidence",
                redacted,
            )

        # 6. Dedup: exact content hash → UPDATE only when new evidence arrives,
        #    otherwise IGNORE; near-dup → MERGE.
        if self._repo is not None:
            existing = self._repo.find_by_content_hash(
                candidate.content_hash, robot_id=candidate.robot_id
            )
            if existing is not None:
                new_evidence = set(candidate.evidence_refs) - set(existing.evidence_refs)
                if new_evidence:
                    return MemoryDecision(
                        GateDecision.UPDATE.value,
                        f"same content with {len(new_evidence)} new evidence refs",
                        target_memory_id=existing.memory_id,
                        redacted_fields=redacted,
                    )
                return MemoryDecision(
                    GateDecision.IGNORE.value,
                    "exact duplicate of existing memory",
                    target_memory_id=existing.memory_id,
                    redacted_fields=redacted,
                )
            near = self._find_near_duplicate(candidate)
            if near is not None:
                return MemoryDecision(
                    GateDecision.MERGE.value,
                    f"near-duplicate of {near.memory_id} (title overlap)",
                    target_memory_id=near.memory_id,
                    redacted_fields=redacted,
                )

        return MemoryDecision(
            GateDecision.STORE.value, "novel evidence-backed memory", redacted_fields=redacted
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_secrets(self, candidate: MemoryItem) -> list[str]:
        haystack = f"{candidate.title}\n{candidate.document}\n{candidate.metadata}"
        hits = []
        for pattern in _SECRET_PATTERNS:
            if pattern.search(haystack):
                hits.append(pattern.pattern[:40])
        return hits

    def _strip_cot(self, candidate: MemoryItem) -> list[str]:
        """Remove CoT-like fields from metadata; keep decision summaries."""
        removed: list[str] = []
        metadata = candidate.metadata or {}
        for key in list(metadata.keys()):
            lowered = key.lower()
            if any(
                token in lowered for token in ("cot", "chain_of_thought", "raw_trace", "full_trace")
            ):
                removed.append(key)
                metadata.pop(key)
        candidate.metadata = metadata
        return removed

    def _find_near_duplicate(self, candidate: MemoryItem) -> MemoryItem | None:
        """Token-overlap near-duplicate detection within same robot+type."""
        if self._repo is None or not candidate.title.strip():
            return None
        try:
            siblings = self._repo.query(
                {
                    "robot_id": candidate.robot_id,
                    "memory_type": candidate.memory_type,
                },
                limit=200,
            )
        except Exception:  # noqa: BLE001
            return None
        candidate_tokens = _token_set(candidate.title)
        if not candidate_tokens:
            return None
        best: MemoryItem | None = None
        best_score = 0.0
        for sibling in siblings:
            if sibling.memory_id == candidate.memory_id:
                continue
            score = _jaccard(candidate_tokens, _token_set(sibling.title))
            if score > best_score:
                best_score = score
                best = sibling
        if best is not None and best_score >= self._dup_threshold:
            return best
        return None


def _token_set(text: str) -> set[str]:
    return {token for token in re.split(r"[^\w一-鿿]+", text.lower()) if len(token) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
