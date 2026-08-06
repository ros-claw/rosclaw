"""TeamCoordinator: single logical coordinator (总纲 §10.7).

P0/P1 uses one explicit Coordinator with term/epoch. All awards and role
leases are durable and idempotently replayable. When the Coordinator is
unavailable no new team tasks are created; members follow the declared
degraded policy and never lose local safety (ADR-0004).

Degradation matrix implemented here (总纲 §10.8):
- world state stale → degraded: no team actions depending on remote state
- coordinator lost → no new team tasks; local tasks continue/safely stop
- member lost → their role/task leases expire before re-allocation
- clock skew → reject fusion
- epoch mismatch → reject old-epoch awards/leases
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.team.member import TeamMemberCardV1
from rosclaw.contracts.team.role import RoleLeaseV1
from rosclaw.contracts.team.world import SharedWorldDeltaV1
from rosclaw.team.allocator import (
    Bid,
    ContractNetAllocator,
    TaskAnnouncement,
)
from rosclaw.team.membership import MemberState, TeamMembership
from rosclaw.team.roles import RoleLeaseStore
from rosclaw.team.world import WorldModel


class TeamError(ValidationError):
    pass


class DegradedPolicy(StrEnum):
    STOP_TEAM_ACTIONS_KEEP_LOCAL_SAFETY = "stop_team_actions_keep_local_safety"
    CONTINUE_LOCAL_TASKS = "continue_local_tasks"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class TeamCoordinator:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        team_id: str,
        actor_id: str,
        policy_hash: str,
        degraded_policy: DegradedPolicy = DegradedPolicy.STOP_TEAM_ACTIONS_KEEP_LOCAL_SAFETY,
        world_max_age_ms: int = 1000,
    ) -> None:
        self._conn = conn
        self._team_id = team_id
        self._actor_id = actor_id
        self._policy_hash = policy_hash
        self.degraded_policy = degraded_policy
        self._world_max_age_ms = world_max_age_ms
        self.membership = TeamMembership(conn, team_id=team_id, actor_id=actor_id)
        self.roles = RoleLeaseStore(conn, team_id=team_id)
        self.allocator = ContractNetAllocator()
        self.world = WorldModel()
        self.coordinator_alive = True

    # ------------------------------------------------------------------
    # membership
    # ------------------------------------------------------------------
    def join_member(self, card: TeamMemberCardV1) -> None:
        self.membership.add_candidate(card)
        self.membership.join(card.member_id, policy_hash=self._policy_hash)

    def epoch(self) -> int:
        return self.membership.current_epoch()

    # ------------------------------------------------------------------
    # task allocation
    # ------------------------------------------------------------------
    def announce_and_award(
        self,
        announcement: TaskAnnouncement,
        bids: list[Bid],
    ) -> tuple[str, str]:
        """Allocate a team task. Returns (task_id, awardee member_id).
        Fail closed: coordinator down, old epoch, or no feasible bidder."""
        if not self.coordinator_alive:
            raise TeamError(
                "coordinator unavailable — no new team tasks "
                f"(degraded policy: {self.degraded_policy.value})"
            )
        if announcement.team_epoch != self.epoch():
            raise TeamError(
                f"announcement epoch {announcement.team_epoch} != current "
                f"{self.epoch()} — refusing mixed-epoch allocation"
            )
        # Idempotency: same key returns the existing award.
        if announcement.idempotency_key:
            row = self._conn.execute(
                "SELECT task_id, awardee FROM team_tasks WHERE team_id = ? AND idempotency_key = ?",
                (self._team_id, announcement.idempotency_key),
            ).fetchone()
            if row is not None:
                return row["task_id"], row["awardee"]
        result = self.allocator.allocate(announcement, bids)
        task_id = announcement.task_id or new_id("ttask")
        lease_expires = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
        self._conn.execute(
            "INSERT INTO team_tasks (team_id, task_id, team_epoch, "
            "announcement_json, status, awardee, bids_json, idempotency_key, "
            "side_effect_class, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'AWARDED', ?, ?, ?, ?, ?, ?)",
            (
                self._team_id,
                task_id,
                announcement.team_epoch,
                json.dumps(announcement.__dict__, ensure_ascii=False),
                result.winner,
                json.dumps(
                    [
                        {"member_id": s.bid.member_id, "score": s.score, "features": s.features}
                        for s in result.scored_bids
                    ],
                    ensure_ascii=False,
                ),
                announcement.idempotency_key,
                announcement.side_effect_class,
                _utcnow(),
                _utcnow(),
            ),
        )
        self._event(
            "rosclaw.team.task.awarded.v1",
            {
                "task_id": task_id,
                "awardee": result.winner,
                "score": result.score,
                "policy": result.policy_version,
                "lease_expires_at": lease_expires,
            },
            idem=announcement.idempotency_key,
        )
        return task_id, result.winner

    def accept_task(self, task_id: str, member_id: str) -> None:
        row = self._task(task_id)
        if row is None:
            raise TeamError(f"unknown team task {task_id!r}")
        if row["awardee"] != member_id:
            raise TeamError(f"task {task_id!r} awarded to {row['awardee']!r}")
        if int(row["team_epoch"]) != self.epoch():
            raise TeamError("task from old epoch — reject")
        self._set_task_status(task_id, "ACCEPTED")
        self._event("rosclaw.team.task.accepted.v1", {"task_id": task_id, "member_id": member_id})

    def complete_task(self, task_id: str, member_id: str, *, evidence: dict) -> None:
        row = self._task(task_id)
        if row is None or row["awardee"] != member_id:
            raise TeamError(f"task {task_id!r} not awarded to {member_id!r}")
        if not evidence.get("summary") and not evidence.get("receipt_ref"):
            raise TeamError("completion evidence must carry summary or receipt_ref")
        self._set_task_status(task_id, "DONE")
        self._conn.execute(
            "UPDATE team_tasks SET evidence_json = ? WHERE team_id = ? AND task_id = ?",
            (json.dumps(evidence, ensure_ascii=False), self._team_id, task_id),
        )
        self._event(
            "rosclaw.team.task.completed.v1",
            {"task_id": task_id, "member_id": member_id, "evidence": evidence},
        )

    # ------------------------------------------------------------------
    # roles
    # ------------------------------------------------------------------
    def award_role(self, lease: RoleLeaseV1) -> RoleLeaseV1:
        if lease.team_epoch != self.epoch():
            raise TeamError(f"role lease epoch {lease.team_epoch} != current {self.epoch()}")
        awarded = self.roles.award(lease)
        self._event(
            "rosclaw.team.role.awarded.v1",
            {"conflict_key": lease.conflict_key, "holder": lease.holder},
        )
        return awarded

    # ------------------------------------------------------------------
    # world
    # ------------------------------------------------------------------
    def merge_world_delta(self, delta: SharedWorldDeltaV1, *, now: datetime) -> list[str]:
        if delta.team_epoch != self.epoch():
            raise TeamError(f"world delta epoch {delta.team_epoch} != current {self.epoch()}")
        return self.world.merge_delta(delta, now=now)

    def world_fresh(self, *, now: datetime) -> bool:
        staleness = self.world.staleness_ms(now=now)
        return staleness is not None and staleness <= self._world_max_age_ms

    # ------------------------------------------------------------------
    # failure handling (degradation matrix)
    # ------------------------------------------------------------------
    def member_lost(self, member_id: str) -> list[str]:
        """Expire the member's roles/tasks before any re-allocation."""
        freed = self.roles.sweep_expired()
        state = self.membership.state_of(member_id)
        if state is MemberState.LOST:
            # 只把无副作用任务重新公告；副作用任务保持原状，等待
            # reconciliation 证明是否执行过（总纲 §10.8 禁止未 reconcile
            # 就重复副作用任务）。
            self._conn.execute(
                "UPDATE team_tasks SET status = 'ANNOUNCED', awardee = NULL "
                "WHERE team_id = ? AND awardee = ? AND status IN ('AWARDED','ACCEPTED') "
                "AND side_effect_class IN ('none', 'sandbox_process')",
                (self._team_id, member_id),
            )
            self._event("rosclaw.team.member.tasks_requeued.v1", {"member_id": member_id})
        return freed

    def coordinator_down(self) -> None:
        self.coordinator_alive = False
        self._event(
            "rosclaw.team.coordinator.lost.v1",
            {"degraded_policy": self.degraded_policy.value},
        )

    # ------------------------------------------------------------------
    def task(self, task_id: str) -> dict | None:
        row = self._task(task_id)
        return dict(row) if row else None

    def _task(self, task_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM team_tasks WHERE team_id = ? AND task_id = ?",
            (self._team_id, task_id),
        ).fetchone()

    def _set_task_status(self, task_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE team_tasks SET status = ?, updated_at = ? WHERE team_id = ? AND task_id = ?",
            (status, _utcnow(), self._team_id, task_id),
        )

    def _event(self, event_type: str, payload: dict, idem: str | None = None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO team_events (event_id, team_id, event_type, "
            "team_epoch, actor_id, payload_json, idempotency_key, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("tevt"),
                self._team_id,
                event_type,
                self.epoch(),
                self._actor_id,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                idem,
                _utcnow(),
            ),
        )
