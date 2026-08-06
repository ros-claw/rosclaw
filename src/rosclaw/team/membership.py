"""Team membership: join/leave/epoch, TTL state machine (PR-TF-071).

Discovery produces CANDIDATEs only; formal join requires an epoch commit.
Health states: CANDIDATE → JOINING → READY → SUSPECT → LOST; LEFT is the
explicit-departure terminal. Epoch bumps are durable and journaled —
awards/leases from older epochs are invalid by construction.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.team.member import TeamMemberCardV1


class MemberState(StrEnum):
    CANDIDATE = "CANDIDATE"
    JOINING = "JOINING"
    READY = "READY"
    SUSPECT = "SUSPECT"
    LOST = "LOST"
    LEFT = "LEFT"


_ALLOWED: dict[MemberState, frozenset[MemberState]] = {
    MemberState.CANDIDATE: frozenset({MemberState.JOINING, MemberState.LEFT}),
    MemberState.JOINING: frozenset({MemberState.READY, MemberState.LEFT}),
    MemberState.READY: frozenset({MemberState.SUSPECT, MemberState.LEFT}),
    MemberState.SUSPECT: frozenset({MemberState.READY, MemberState.LOST, MemberState.LEFT}),
    MemberState.LOST: frozenset({MemberState.JOINING, MemberState.LEFT}),
    MemberState.LEFT: frozenset({MemberState.CANDIDATE}),
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class TeamMembership:
    def __init__(self, conn: sqlite3.Connection, *, team_id: str, actor_id: str) -> None:
        self._conn = conn
        self._team_id = team_id
        self._actor_id = actor_id

    # ------------------------------------------------------------------
    def current_epoch(self) -> int:
        row = self._conn.execute(
            "SELECT MAX(epoch) AS e FROM team_epochs WHERE team_id = ?", (self._team_id,)
        ).fetchone()
        return int(row["e"] or 0)

    def bump_epoch(self, reason: str) -> int:
        epoch = self.current_epoch() + 1
        self._conn.execute(
            "INSERT INTO team_epochs (team_id, epoch, reason, actor_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (self._team_id, epoch, reason, self._actor_id, _utcnow()),
        )
        self._event("rosclaw.team.epoch.committed.v1", {"epoch": epoch, "reason": reason})
        return epoch

    # ------------------------------------------------------------------
    def add_candidate(self, card: TeamMemberCardV1) -> None:
        if card.team_id != self._team_id:
            raise ValidationError(f"card team {card.team_id!r} != {self._team_id!r}")
        self._upsert(card, MemberState.CANDIDATE)
        self._event("rosclaw.team.member.discovered.v1", {"member_id": card.member_id})

    def join(self, member_id: str, *, policy_hash: str) -> None:
        """Formal join: CANDIDATE → JOINING → READY with epoch commit."""
        self._require_state(member_id, MemberState.CANDIDATE, MemberState.LEFT, MemberState.LOST)
        self._set_state(member_id, MemberState.JOINING)
        epoch = self.bump_epoch(f"member_join:{member_id}")
        now = _utcnow()
        self._conn.execute(
            "UPDATE team_members SET state = 'READY', team_epoch = ?, "
            "last_seen_at = ?, updated_at = ? WHERE team_id = ? AND member_id = ?",
            (epoch, now, now, self._team_id, member_id),
        )
        self._event(
            "rosclaw.team.member.joined.v1",
            {"member_id": member_id, "epoch": epoch, "policy_hash": policy_hash},
        )

    def leave(self, member_id: str) -> None:
        self._set_state(member_id, MemberState.LEFT)
        self.bump_epoch(f"member_leave:{member_id}")
        self._event("rosclaw.team.member.left.v1", {"member_id": member_id})

    def heartbeat(self, member_id: str) -> None:
        row = self._row(member_id)
        if row is None or row["state"] not in ("READY", "SUSPECT"):
            return
        now = _utcnow()
        self._conn.execute(
            "UPDATE team_members SET last_seen_at = ?, state = 'READY', updated_at = ? "
            "WHERE team_id = ? AND member_id = ?",
            (now, now, self._team_id, member_id),
        )

    def sweep_ttl(self, *, suspect_after_ms: int, lost_after_ms: int) -> dict[str, str]:
        """Age members through SUSPECT/LOST. Returns member → new state."""
        now = datetime.now(UTC)
        changed: dict[str, str] = {}
        for row in self._conn.execute(
            "SELECT member_id, state, last_seen_at FROM team_members "
            "WHERE team_id = ? AND state IN ('READY', 'SUSPECT')",
            (self._team_id,),
        ).fetchall():
            last = datetime.fromisoformat(row["last_seen_at"])
            age_ms = (now - last.astimezone(UTC)).total_seconds() * 1000.0
            if age_ms > lost_after_ms:
                # Past the lost threshold: straight to LOST from any live state.
                if row["state"] != "LOST":
                    self._conn.execute(
                        "UPDATE team_members SET state = 'LOST', updated_at = ? "
                        "WHERE team_id = ? AND member_id = ?",
                        (now.isoformat(), self._team_id, row["member_id"]),
                    )
                    changed[row["member_id"]] = "LOST"
                    self._event("rosclaw.team.member.lost.v1", {"member_id": row["member_id"]})
            elif row["state"] == "READY" and age_ms > suspect_after_ms:
                self._set_state(row["member_id"], MemberState.SUSPECT)
                changed[row["member_id"]] = "SUSPECT"
        return changed

    # ------------------------------------------------------------------
    def members(self, *, states: tuple[MemberState, ...] | None = None) -> list[TeamMemberCardV1]:
        rows = self._conn.execute(
            "SELECT card_json, state FROM team_members WHERE team_id = ?",
            (self._team_id,),
        ).fetchall()
        cards = []
        for row in rows:
            if states and row["state"] not in {s.value for s in states}:
                continue
            cards.append(TeamMemberCardV1(**json.loads(row["card_json"])))
        return cards

    def state_of(self, member_id: str) -> MemberState | None:
        row = self._row(member_id)
        return MemberState(row["state"]) if row else None

    # ------------------------------------------------------------------
    def _row(self, member_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM team_members WHERE team_id = ? AND member_id = ?",
            (self._team_id, member_id),
        ).fetchone()

    def _upsert(self, card: TeamMemberCardV1, state: MemberState) -> None:
        now = _utcnow()
        self._conn.execute(
            "INSERT INTO team_members (team_id, member_id, card_json, state, "
            "team_epoch, last_seen_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(team_id, member_id) DO UPDATE SET card_json = "
            "excluded.card_json, updated_at = excluded.updated_at",
            (
                self._team_id,
                card.member_id,
                card.model_dump_json(),
                state.value,
                card.team_epoch,
                now,
                now,
            ),
        )

    def _require_state(self, member_id: str, *states: MemberState) -> None:
        current = self.state_of(member_id)
        if current is None:
            raise ValidationError(f"unknown member {member_id!r}")
        if current not in states:
            raise ValidationError(
                f"member {member_id!r} in state {current}, expected one of {states}"
            )

    def _set_state(self, member_id: str, state: MemberState) -> None:
        row = self._row(member_id)
        if row is None:
            raise ValidationError(f"unknown member {member_id!r}")
        current = MemberState(row["state"])
        if state not in _ALLOWED[current]:
            raise ValidationError(f"illegal member transition {current} -> {state}")
        self._conn.execute(
            "UPDATE team_members SET state = ?, updated_at = ? WHERE team_id = ? AND member_id = ?",
            (state.value, _utcnow(), self._team_id, member_id),
        )

    def _event(self, event_type: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO team_events (event_id, team_id, event_type, team_epoch, "
            "actor_id, payload_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("tevt"),
                self._team_id,
                event_type,
                self.current_epoch(),
                self._actor_id,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                _utcnow(),
            ),
        )
