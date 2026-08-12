"""WorkerRegistry — WorkerCard registration and lifecycle (PR-WF-050).

A card is a *declaration*. Registration validates schema version, adapter
compatibility, provenance (digest/license), capability declarations, data
scopes and forbidden permissions. Probe results and task outcomes provide
runtime evidence elsewhere (manager/metrics).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.worker.card import WorkerCardV1

SUPPORTED_ADAPTER_TYPES = frozenset(
    {"native_inproc", "process_stdio", "external_cli", "pi_managed"}
)
SUPPORTED_ADAPTER_VERSIONS = frozenset({"1.0.0"})

#: Scopes a cognitive worker may never request (ADR-0003).
HARD_FORBIDDEN_SCOPES = frozenset(
    {"daemon_private_ledger", "physical_permits", "raw_secrets", "direct_hardware"}
)

#: 十审 W0（P0-CAPABILITY-LIE）：能力名尾段暗示副作用时，side_effect_class
#: 不得是 "none"——声明即承诺，注册处硬拒绝（docs.write + none = 欺诈）。
_SIDE_EFFECT_IMPLYING_TAILS = frozenset(
    {"write", "edit", "delete", "execute", "install", "promote", "apply"}
)

_WORKER_STATUSES = ("ENABLED", "DISABLED", "QUARANTINED")


class CardValidationError(ValidationError):
    """WorkerCard failed registration validation."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def validate_card(card: WorkerCardV1) -> None:
    """Registration checks (总纲 §9.3). Fail closed on any violation."""
    errors: list[str] = []
    if card.adapter_type not in SUPPORTED_ADAPTER_TYPES:
        errors.append(f"unsupported adapter_type {card.adapter_type!r}")
    if card.adapter_version not in SUPPORTED_ADAPTER_VERSIONS:
        errors.append(f"unsupported adapter_version {card.adapter_version!r}")
    if not card.worker_id or ":" not in card.worker_id:
        errors.append("worker_id must be namespaced like worker:<product>:<instance>")
    if not card.implementation.product:
        errors.append("implementation.product required")
    if card.provenance.license in ("", "UNVERIFIED") and card.trust.initial_level in (
        "T2",
        "T3",
    ):
        errors.append("T2/T3 trust requires verified license metadata")
    for cap in card.capabilities:
        if not cap.name or "." not in cap.name:
            errors.append(f"capability name {cap.name!r} must be namespaced")
        if cap.side_effect_class == "physical":
            errors.append(f"capability {cap.name!r}: physical side effects forbidden in P0/P1")
        # 十审 W0：名字暗示写/执行副作用却声明 none——能力欺诈，拒绝注册。
        tail = cap.name.rsplit(".", 1)[-1]
        if tail in _SIDE_EFFECT_IMPLYING_TAILS and cap.side_effect_class == "none":
            errors.append(
                f"capability {cap.name!r} implies side effects but declares "
                "side_effect_class='none' (capability lie)"
            )
    requested = set(card.security.default_data_scopes)
    if requested & HARD_FORBIDDEN_SCOPES:
        errors.append(
            f"card requests hard-forbidden scopes: {sorted(requested & HARD_FORBIDDEN_SCOPES)}"
        )
    # The card's forbidden list must at least cover the hard floor.
    if not set(card.security.forbidden_scopes) >= HARD_FORBIDDEN_SCOPES:
        errors.append("forbidden_scopes must include the hard floor")
    if errors:
        raise CardValidationError("; ".join(errors))


def native_basic_card() -> WorkerCardV1:
    """Built-in native-basic worker (T3, ROSClaw-maintained)."""
    from rosclaw.contracts.worker.card import (
        CapabilityDecl,
        WorkerConstraints,
        WorkerHealth,
        WorkerImplementation,
        WorkerKind,
        WorkerProvenance,
        WorkerSecurity,
        WorkerTrust,
    )

    return WorkerCardV1(
        worker_id="worker:native:basic",
        display_name="ROSClaw Native Worker",
        kind=WorkerKind.NATIVE,
        adapter_type="native_inproc",
        adapter_version="1.0.0",
        implementation=WorkerImplementation(
            product="rosclaw-native-worker", version="1.0.0", executable_ref="inproc:"
        ),
        capabilities=[
            CapabilityDecl(
                name="analysis.text",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
            CapabilityDecl(
                name="analysis.log_review",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
            CapabilityDecl(
                name="review.artifact",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
        ],
        constraints=WorkerConstraints(supported_platforms=["linux", "darwin"], max_concurrency=3),
        security=WorkerSecurity(isolation="process"),
        health=WorkerHealth(probe="adapter:ping", heartbeat_interval_sec=15, lease_ttl_sec=45),
        provenance=WorkerProvenance(source="builtin", package_digest=None, license="MIT"),
        trust=WorkerTrust(initial_level="T3", evidence_count=0),
    )


def pi_worker_card() -> WorkerCardV1:
    """内置 Pi headless Worker（十审 W1）——与主 Agent 同一模型配置，
    只读工具集（scout/analyst profile）；能力声明即真实能力。"""
    from rosclaw.contracts.worker.card import (
        CapabilityDecl,
        WorkerConstraints,
        WorkerHealth,
        WorkerImplementation,
        WorkerKind,
        WorkerProvenance,
        WorkerSecurity,
        WorkerTrust,
    )

    return WorkerCardV1(
        worker_id="worker:rosclaw:pi",
        display_name="ROSClaw Built-in Pi Worker",
        kind=WorkerKind.NATIVE,
        adapter_type="pi_managed",
        adapter_version="1.0.0",
        implementation=WorkerImplementation(
            product="rosclaw-agent", version="1.0.0", executable_ref="builtin:worker"
        ),
        capabilities=[
            CapabilityDecl(
                name="analysis.text",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
            CapabilityDecl(
                name="analysis.log_review",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
            CapabilityDecl(
                name="review.artifact",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
            # 只读工具（read/grep/find/ls）真实可读工作区——与外部 pack
            # 的 text-only 伪仓库分析不同（十审 §10.2 诚实化）。
            CapabilityDecl(
                name="code.repository_analysis",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            ),
        ],
        constraints=WorkerConstraints(supported_platforms=["linux", "darwin"], max_concurrency=2),
        security=WorkerSecurity(isolation="process"),
        health=WorkerHealth(probe="adapter:ping", heartbeat_interval_sec=15, lease_ttl_sec=360),
        provenance=WorkerProvenance(source="builtin", package_digest=None, license="MIT"),
        trust=WorkerTrust(initial_level="T3", evidence_count=0),
    )


class WorkerRegistry:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    def register(self, card: WorkerCardV1, *, actor_id: str) -> WorkerCardV1:
        validate_card(card)
        now = _utcnow()
        self._conn.execute(
            "INSERT INTO worker_cards (worker_id, card_json, status, trust_level, "
            "registered_at, updated_at) VALUES (?, ?, 'ENABLED', ?, ?, ?) "
            "ON CONFLICT(worker_id) DO UPDATE SET card_json = excluded.card_json, "
            "trust_level = excluded.trust_level, updated_at = excluded.updated_at",
            (
                card.worker_id,
                card.model_dump_json(),
                card.trust.initial_level,
                now,
                now,
            ),
        )
        self._event(card.worker_id, "rosclaw.worker.card.registered.v1", actor_id, {})
        return card

    def register_builtins(self, *, actor_id: str) -> None:
        self.register(native_basic_card(), actor_id=actor_id)
        # 十审 W1：内置 Pi Worker（默认开发/研究 Worker）。
        self.register(pi_worker_card(), actor_id=actor_id)

    # ------------------------------------------------------------------
    def get(self, worker_id: str) -> WorkerCardV1 | None:
        row = self._conn.execute(
            "SELECT card_json FROM worker_cards WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        return WorkerCardV1(**json.loads(row["card_json"])) if row else None

    def status_of(self, worker_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM worker_cards WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        return row["status"] if row else None

    def list(self, *, status: str | None = None) -> list[WorkerCardV1]:
        if status is None:
            rows = self._conn.execute(
                "SELECT card_json FROM worker_cards ORDER BY worker_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT card_json FROM worker_cards WHERE status = ? ORDER BY worker_id",
                (status,),
            ).fetchall()
        return [WorkerCardV1(**json.loads(r["card_json"])) for r in rows]

    def catalog(self) -> list[WorkerCardV1]:
        """Official built-in WorkerPacks (not necessarily installed/enabled)."""
        return [native_basic_card()]

    # ------------------------------------------------------------------
    def set_status(self, worker_id: str, status: str, *, actor_id: str, reason: str = "") -> None:
        if status not in _WORKER_STATUSES:
            raise ValidationError(f"illegal worker status {status!r}")
        if self.status_of(worker_id) is None:
            raise ValidationError(f"unknown worker {worker_id!r}")
        self._conn.execute(
            "UPDATE worker_cards SET status = ?, updated_at = ? WHERE worker_id = ?",
            (status, _utcnow(), worker_id),
        )
        self._event(
            worker_id,
            f"rosclaw.worker.card.{status.lower()}.v1",
            actor_id,
            {"reason": reason},
        )

    # ------------------------------------------------------------------
    def _event(self, worker_id: str, event_type: str, actor_id: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO worker_events (event_id, worker_id, event_type, actor_id, "
            "payload_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                new_id("wevt"),
                worker_id,
                event_type,
                actor_id,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                _utcnow(),
            ),
        )
