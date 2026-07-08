"""Canonical ID generators for ROSClaw Practice data closed-loop.

All generated IDs are prefixed so that a single event can be traced across
Practice Catalog, ArtifactStore, SeekDB, and exported datasets.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _short_hash(length: int = 12) -> str:
    return uuid.uuid4().hex[:length]


def generate_event_id() -> str:
    return f"evt_{_utc_timestamp()}_{_short_hash()}"


def generate_session_id() -> str:
    return f"sess_{_utc_timestamp()}_{_short_hash()}"


def generate_episode_id() -> str:
    return f"ep_{_utc_timestamp()}_{_short_hash(8)}"


def generate_practice_id() -> str:
    return f"prac_{_utc_timestamp()}_{_short_hash()}"


def generate_trace_id() -> str:
    return f"trace_{_short_hash(12)}"


def generate_artifact_id(artifact_type: str = "artifact") -> str:
    return f"art_{artifact_type}_{_utc_timestamp()}_{_short_hash()}"


def generate_candidate_id() -> str:
    return f"cand_{_utc_timestamp()}_{_short_hash()}"


def generate_policy_id() -> str:
    return f"pol_{_utc_timestamp()}_{_short_hash()}"


def generate_asset_id() -> str:
    return f"asset_{_utc_timestamp()}_{_short_hash()}"


def generate_skill_id() -> str:
    return f"skill_{_utc_timestamp()}_{_short_hash()}"


def generate_body_id() -> str:
    return f"body_{_utc_timestamp()}_{_short_hash()}"


ID_PREFIXES = {
    "event": "evt_",
    "session": "sess_",
    "episode": "ep_",
    "practice": "prac_",
    "trace": "trace_",
    "artifact": "art_",
    "candidate": "cand_",
    "policy": "pol_",
    "asset": "asset_",
    "skill": "skill_",
    "body": "body_",
}


def is_id_of_kind(value: str, kind: str) -> bool:
    """Check whether ``value`` starts with the canonical prefix for ``kind``."""
    prefix = ID_PREFIXES.get(kind)
    if not prefix or not isinstance(value, str):
        return False
    return value.startswith(prefix)
