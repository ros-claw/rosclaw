"""Team Fabric tests (PR-TF-070/071/072/073 + K6).

- membership: candidate→join→READY, epoch commits, TTL SUSPECT→LOST
- roles: conflict_key CAS, old-epoch rejection, expiry, contest policy
- allocator: deterministic, explainable, feasibility gates
- world: latest_valid merge, tombstone, clock skew, epoch mismatch
- K6: two-robot cooperative task with partition/loss/coordinator-down
  fault injection (总纲 §10.8 degradation matrix)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.agentd.mission import MissionStore
from rosclaw.contracts.common import ValidationError
from rosclaw.contracts.team.member import MemberBody, TeamMemberCardV1
from rosclaw.contracts.team.role import RoleLeaseV1, RoleScope
from rosclaw.contracts.team.world import ObjectState, SharedWorldDeltaV1
from rosclaw.team import (
    ClockSkewError,
    LocalSimTransport,
    MemberState,
    RoleConflictError,
    TeamCoordinator,
    TeamError,
)
from rosclaw.team.allocator import Bid, TaskAnnouncement
from rosclaw.team.membership import TeamMembership

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
TEAM = "blue_team"
ACTOR = "coordinator:blue"


def _store(tmp_path: Path) -> MissionStore:
    return MissionStore(tmp_path / "team.db")


def _member(member_id: str, caps: list[str] | None = None) -> TeamMemberCardV1:
    return TeamMemberCardV1(
        team_id=TEAM,
        member_id=member_id,
        body=MemberBody(
            **{"body_id": member_id, "effective_body_hash": "body_x", "class": "mobile_base"}
        ),
        capabilities=caps or ["navigation.local", "role.defender"],
    )


def _coordinator(tmp_path: Path) -> TeamCoordinator:
    store = _store(tmp_path)
    return TeamCoordinator(store.connection, team_id=TEAM, actor_id=ACTOR, policy_hash="team_pol_1")


class TestMembership:
    def test_join_commits_epoch(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        assert coord.epoch() == 0
        coord.join_member(_member("robot:limo:blue_01"))
        assert coord.epoch() == 1
        assert coord.membership.state_of("robot:limo:blue_01") is MemberState.READY

    def test_leave_bumps_epoch(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1"))
        coord.membership.leave("r1")
        assert coord.membership.state_of("r1") is MemberState.LEFT
        assert coord.epoch() == 2

    def test_ttl_sweep(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        membership = TeamMembership(store.connection, team_id=TEAM, actor_id=ACTOR)
        membership.add_candidate(_member("r1"))
        membership.join("r1", policy_hash="p")
        old = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        store.connection.execute(
            "UPDATE team_members SET last_seen_at = ? WHERE member_id = 'r1'", (old,)
        )
        changed = membership.sweep_ttl(suspect_after_ms=1000, lost_after_ms=3000)
        assert changed == {"r1": "LOST"}
        # heartbeat recovery from SUSPECT but not after LOST → rejoin needed
        store.connection.execute(
            "UPDATE team_members SET state = 'SUSPECT', last_seen_at = ? WHERE member_id = 'r1'",
            (datetime.now(UTC).isoformat(),),
        )
        membership.heartbeat("r1")
        assert membership.state_of("r1") is MemberState.READY

    def test_illegal_transition_rejected(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.membership.add_candidate(_member("r1"))
        with pytest.raises(ValidationError, match="illegal member transition"):
            coord.membership._set_state("r1", MemberState.READY)


class TestRoles:
    def _lease(self, holder: str, epoch: int, expires: str | None = None) -> RoleLeaseV1:
        return RoleLeaseV1(
            team_id=TEAM,
            team_epoch=epoch,
            role="defender:left",
            holder=holder,
            scope=RoleScope(region="field.left", task_types=["intercept"]),
            issued_at=NOW.isoformat(),
            expires_at=expires or (NOW + timedelta(seconds=30)).isoformat(),
            conflict_key="role:defender:left",
            policy_hash="team_pol_1",
        )

    def test_conflict_key_cas(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1"))
        coord.join_member(_member("r2"))
        coord.award_role(self._lease("r1", coord.epoch()))
        with pytest.raises(RoleConflictError):
            coord.award_role(self._lease("r2", coord.epoch()))
        assert coord.roles.active_holder("role:defender:left") == "r1"

    def test_old_epoch_rejected(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1"))
        with pytest.raises(TeamError, match="epoch"):
            coord.award_role(self._lease("r1", coord.epoch() - 1))

    def test_non_ready_holder_rejected(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.membership.add_candidate(_member("r1"))  # never joined
        with pytest.raises(ValidationError, match="not READY"):
            coord.award_role(self._lease("r1", coord.epoch()))

    def test_expiry_frees_role(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1"))
        coord.join_member(_member("r2"))
        past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        coord.award_role(self._lease("r1", coord.epoch(), expires=past))
        freed = coord.roles.sweep_expired()
        assert "role:defender:left" in freed
        # Now r2 can take it.
        coord.award_role(self._lease("r2", coord.epoch()))
        assert coord.roles.active_holder("role:defender:left") == "r2"

    def test_contest_marks_contested(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1"))
        coord.award_role(self._lease("r1", coord.epoch()))
        coord.roles.contest("role:defender:left")
        assert coord.roles.active_holder("role:defender:left") is None


class TestAllocator:
    def test_deterministic_scoring(self) -> None:
        from rosclaw.team.allocator import ContractNetAllocator

        allocator = ContractNetAllocator()
        ann = TaskAnnouncement(
            task_id="task_pass",
            team_id=TEAM,
            team_epoch=1,
            required_capabilities=("ball.kick",),
            deadline_ms=5000,
        )
        bids = [
            Bid(
                member_id="r_slow",
                eta_ms=40000,
                energy_cost=100,
                capability_fit=1.0,
                reliability=0.9,
                current_load=0.1,
                comms_quality=0.9,
            ),
            Bid(
                member_id="r_fast",
                eta_ms=800,
                energy_cost=150,
                capability_fit=1.0,
                reliability=0.85,
                current_load=0.2,
                comms_quality=0.9,
            ),
        ]
        result = allocator.allocate(ann, bids)
        assert result.winner == "r_fast"
        # Same inputs → same decision (determinism).
        assert allocator.allocate(ann, bids).winner == "r_fast"
        # Feature vector is recorded for every scored bid.
        assert all(s.features for s in result.scored_bids)

    def test_infeasible_bids_rejected(self) -> None:
        from rosclaw.team.allocator import ContractNetAllocator

        allocator = ContractNetAllocator()
        ann = TaskAnnouncement(
            task_id="t",
            team_id=TEAM,
            team_epoch=1,
            required_capabilities=("ball.kick",),
            deadline_ms=1000,
        )
        with pytest.raises(ValidationError, match="no feasible bidder"):
            allocator.allocate(
                ann,
                [
                    Bid(
                        member_id="r1",
                        eta_ms=100,
                        energy_cost=0,
                        capability_fit=0.5,
                        reliability=0.9,
                        current_load=0.0,
                        comms_quality=1.0,
                    ),
                    Bid(
                        member_id="r2",
                        eta_ms=5000,
                        energy_cost=0,
                        capability_fit=1.0,
                        reliability=0.9,
                        current_load=0.0,
                        comms_quality=1.0,
                    ),
                ],
            )


class TestWorld:
    def _delta(
        self,
        epoch: int,
        obj_id: str = "ball",
        x: float = 1.0,
        observed: datetime | None = None,
        tombstone: bool = False,
    ) -> SharedWorldDeltaV1:
        return SharedWorldDeltaV1(
            team_id=TEAM,
            team_epoch=epoch,
            world_revision=1,
            base_revision=0,
            source_member="robot:limo:blue_01",
            observed_at=(observed or NOW).isoformat(),
            published_at=(observed or NOW).isoformat(),
            objects=[
                ObjectState(
                    object_id=obj_id,
                    pose={"x": x, "y": 0.0},
                    confidence=0.9,
                    evidence_class="measurement",
                    tombstone=tombstone,
                )
            ],
        )

    def test_merge_latest_valid(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("robot:limo:blue_01"))
        coord.merge_world_delta(self._delta(coord.epoch(), x=1.0), now=NOW)
        coord.merge_world_delta(self._delta(coord.epoch(), x=2.0), now=NOW)
        fresh = coord.world.fresh_objects(now=NOW, max_age_ms=1000)
        assert fresh["ball"].pose["x"] == 2.0

    def test_tombstone_removes(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("robot:limo:blue_01"))
        coord.merge_world_delta(self._delta(coord.epoch()), now=NOW)
        coord.merge_world_delta(self._delta(coord.epoch(), tombstone=True), now=NOW)
        assert coord.world.fresh_objects(now=NOW, max_age_ms=1000) == {}

    def test_clock_skew_rejected(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("robot:limo:blue_01"))
        ancient = NOW - timedelta(minutes=5)
        with pytest.raises(ClockSkewError):
            coord.merge_world_delta(self._delta(coord.epoch(), observed=ancient), now=NOW)

    def test_epoch_mismatch_rejected(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("robot:limo:blue_01"))
        with pytest.raises(TeamError, match="epoch"):
            coord.merge_world_delta(self._delta(coord.epoch() + 99), now=NOW)


class TestTransport:
    def test_latency_delivery(self) -> None:
        transport = LocalSimTransport(seed=1)
        received: list[dict] = []
        transport.subscribe("world", lambda member, payload: received.append(payload))
        transport.set_link("r1", "r2", latency_ms=100)
        transport.send("r1", "r2", "world", {"x": 1})
        transport.advance_time(50)
        assert received == []
        transport.advance_time(60)
        assert received == [{"x": 1}]

    def test_partition_drops(self) -> None:
        transport = LocalSimTransport(seed=1)
        transport.set_link("r1", "r2")
        transport.partition(["r2"])
        assert transport.send("r1", "r2", "world", {}) is False
        assert transport.dropped
        transport.heal()
        assert transport.send("r1", "r2", "world", {}) is True

    def test_loss_rate_deterministic(self) -> None:
        transport = LocalSimTransport(seed=7)
        transport.set_link("r1", "r2", loss_rate=0.5)
        sent = sum(transport.send("r1", "r2", "t", {"i": i}) for i in range(100))
        assert 20 < sent < 80  # seeded RNG, sanity band


class TestK6TwoRobotScenario:
    """K6 SIM: two robots, cooperative task, fault matrix (总纲 §10.9 T-SIM-1
    functional core; league benchmarks are PR-TF-075 scope)."""

    def test_cooperative_task_full_lifecycle(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("robot:limo:blue_01", ["ball.kick", "navigation.local"]))
        coord.join_member(_member("robot:limo:blue_02", ["ball.kick", "navigation.local"]))
        epoch = coord.epoch()

        # Roles.
        coord.award_role(
            RoleLeaseV1(
                team_id=TEAM,
                team_epoch=epoch,
                role="passer",
                holder="robot:limo:blue_01",
                scope=RoleScope(region="field.mid", task_types=["pass"]),
                issued_at=NOW.isoformat(),
                expires_at=(NOW + timedelta(seconds=60)).isoformat(),
                conflict_key="role:passer",
                policy_hash="team_pol_1",
            )
        )
        coord.award_role(
            RoleLeaseV1(
                team_id=TEAM,
                team_epoch=epoch,
                role="receiver",
                holder="robot:limo:blue_02",
                scope=RoleScope(region="field.fwd", task_types=["receive"]),
                issued_at=NOW.isoformat(),
                expires_at=(NOW + timedelta(seconds=60)).isoformat(),
                conflict_key="role:receiver",
                policy_hash="team_pol_1",
            )
        )

        # Shared world: ball observed by 01, both see it fresh.
        coord.merge_world_delta(
            SharedWorldDeltaV1(
                team_id=TEAM,
                team_epoch=epoch,
                world_revision=1,
                base_revision=0,
                source_member="robot:limo:blue_01",
                observed_at=NOW.isoformat(),
                published_at=NOW.isoformat(),
                objects=[
                    ObjectState(
                        object_id="ball",
                        pose={"x": 1.0, "y": 0.5},
                        confidence=0.95,
                        evidence_class="measurement",
                    )
                ],
            ),
            now=NOW,
        )
        assert coord.world_fresh(now=NOW)

        # Announce → bid → award → accept → complete with evidence.
        ann = TaskAnnouncement(
            task_id="task_pass_1",
            team_id=TEAM,
            team_epoch=epoch,
            required_capabilities=("ball.kick",),
            deadline_ms=5000,
            success_criteria="ball delivered to receiver region",
            idempotency_key="k6-pass-1",
        )
        bids = [
            Bid(
                member_id="robot:limo:blue_01",
                eta_ms=300,
                energy_cost=80,
                capability_fit=1.0,
                reliability=0.9,
                current_load=0.0,
                comms_quality=0.95,
            ),
            Bid(
                member_id="robot:limo:blue_02",
                eta_ms=2200,
                energy_cost=120,
                capability_fit=1.0,
                reliability=0.9,
                current_load=0.1,
                comms_quality=0.95,
            ),
        ]
        task_id, awardee = coord.announce_and_award(ann, bids)
        assert awardee == "robot:limo:blue_01"
        # Idempotent replay returns the same award.
        assert coord.announce_and_award(ann, bids) == (task_id, awardee)
        coord.accept_task(task_id, "robot:limo:blue_01")
        coord.complete_task(
            task_id,
            "robot:limo:blue_01",
            evidence={"summary": "pass executed in SIM", "receipt_ref": "receipt://sim/1"},
        )
        task = coord.task(task_id)
        assert task["status"] == "DONE"
        # Attribution: bids feature vectors + award event journaled.
        bids_logged = json.loads(task["bids_json"])
        assert bids_logged[0]["features"]["capability"] == 1.0

    def test_member_lost_requeues_and_expires_roles(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1", ["ball.kick"]))
        coord.join_member(_member("r2", ["ball.kick"]))
        epoch = coord.epoch()
        coord.award_role(
            RoleLeaseV1(
                team_id=TEAM,
                team_epoch=epoch,
                role="passer",
                holder="r1",
                scope=RoleScope(),
                issued_at=NOW.isoformat(),
                expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                conflict_key="role:passer",
                policy_hash="p",
            )
        )
        ann = TaskAnnouncement(
            task_id="t1",
            team_id=TEAM,
            team_epoch=epoch,
            required_capabilities=("ball.kick",),
        )
        bids = [
            Bid(
                member_id="r1",
                eta_ms=100,
                energy_cost=10,
                capability_fit=1.0,
                reliability=0.9,
                current_load=0.0,
                comms_quality=1.0,
            )
        ]
        coord.announce_and_award(ann, bids)
        coord.accept_task("t1", "r1")
        # r1 drops off the network.
        store = coord._conn
        old = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        store.execute("UPDATE team_members SET last_seen_at = ? WHERE member_id = 'r1'", (old,))
        coord.membership.sweep_ttl(suspect_after_ms=1000, lost_after_ms=3000)
        freed = coord.member_lost("r1")
        assert "role:passer" in freed  # role expired before re-allocation
        assert coord.task("t1")["status"] == "ANNOUNCED"  # re-announced, not duplicated

    def test_coordinator_down_stops_new_tasks(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1", ["ball.kick"]))
        coord.coordinator_down()
        ann = TaskAnnouncement(
            task_id="t2",
            team_id=TEAM,
            team_epoch=coord.epoch(),
            required_capabilities=("ball.kick",),
        )
        with pytest.raises(TeamError, match="coordinator unavailable"):
            coord.announce_and_award(ann, [])
        # Local safety unaffected: world merging (read path) keeps working.
        assert coord.world_fresh(now=NOW) is False  # no data yet — honest

    def test_epoch_mismatch_award_rejected(self, tmp_path: Path) -> None:
        coord = _coordinator(tmp_path)
        coord.join_member(_member("r1", ["ball.kick"]))
        ann = TaskAnnouncement(
            task_id="t3",
            team_id=TEAM,
            team_epoch=coord.epoch() + 5,
            required_capabilities=("ball.kick",),
        )
        with pytest.raises(TeamError, match="epoch"):
            coord.announce_and_award(ann, [])
