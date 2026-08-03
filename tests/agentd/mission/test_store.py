"""MissionStore tests (PR-NA-011 exit criteria).

Crash/restart, concurrent patches, cyclic deps, stale revisions, duplicate
events, over-budget, body-hash invalidation, kill -9 recovery, and a seeded
randomized state-machine property run (1000 iterations, no illegal states).
"""

from __future__ import annotations

import random
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from rosclaw.agentd.mission import (
    BudgetExceededError,
    MissionStore,
    RevisionConflictError,
    TransitionError,
)
from rosclaw.contracts.agent.mission import (
    MISSION_TRANSITIONS,
    BodyBinding,
    Budgets,
    Goal,
    MissionState,
)
from rosclaw.contracts.agent.task_graph import (
    PatchOperation,
    TaskGraphPatchV1,
    TaskNodeV1,
)
from rosclaw.contracts.common import ValidationError, new_id
from tests.agentd.conftest import LOCAL_PRINCIPAL

ACTOR = "agent:rosclaw-native:sim_ur5e_01"


def _store(tmp_path: Path, name: str = "missions.db") -> MissionStore:
    return MissionStore(tmp_path / name)


def _mission(store: MissionStore, **kwargs):
    return store.create_mission(
        owner_principal=LOCAL_PRINCIPAL,
        goal=Goal(text="测试任务"),
        body_binding=BodyBinding(body_id="sim_ur5e_01", effective_body_hash="body_abc"),
        actor_id=ACTOR,
        **kwargs,
    )


def _node(task_id: str, deps: list[str] | None = None, kind: str = "perceive") -> TaskNodeV1:
    return TaskNodeV1(
        task_id=task_id, mission_id="", kind=kind, goal=f"goal {task_id}", dependencies=deps or []
    )


def _patch(mission_id: str, base: int, ops: list[PatchOperation]) -> TaskGraphPatchV1:
    return TaskGraphPatchV1(
        patch_id=new_id("tgpatch"),
        mission_id=mission_id,
        base_revision=base,
        proposed_by=ACTOR,
        operations=ops,
    )


class TestMissionLifecycle:
    def test_create_and_get(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        loaded = store.get_mission(mission.mission_id)
        assert loaded is not None
        assert loaded.state is MissionState.IDLE
        assert loaded.mode.value == "SIMULATION"
        store.verify_consistency(mission.mission_id)

    def test_create_idempotent(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        m1 = _mission(store, idempotency_key="create-1")
        m2 = _mission(store, idempotency_key="create-1")
        assert m1.mission_id == m2.mission_id

    def test_legal_transition_journaled(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        store.transition(
            mission.mission_id,
            MissionState.UNDERSTAND,
            reason_code="new_goal",
            actor_id=ACTOR,
        )
        events = store.events(mission.mission_id)
        assert [e["event_type"] for e in events] == [
            "rosclaw.agent.mission.created.v1",
            "rosclaw.agent.mission.transition.v1",
        ]
        assert events[1]["from_state"] == "IDLE"
        assert events[1]["to_state"] == "UNDERSTAND"
        store.verify_consistency(mission.mission_id)

    def test_illegal_transition_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        with pytest.raises(TransitionError):
            store.transition(
                mission.mission_id,
                MissionState.DISPATCH,
                reason_code="skip_everything",
                actor_id=ACTOR,
            )

    def test_transition_idempotent_replay(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        store.transition(
            mission.mission_id,
            MissionState.UNDERSTAND,
            reason_code="new_goal",
            actor_id=ACTOR,
            idempotency_key="tx-1",
        )
        # Same key replays without duplicating the event.
        store.transition(
            mission.mission_id,
            MissionState.UNDERSTAND,
            reason_code="new_goal",
            actor_id=ACTOR,
            idempotency_key="tx-1",
        )
        transitions = [
            e
            for e in store.events(mission.mission_id)
            if e["event_type"] == "rosclaw.agent.mission.transition.v1"
        ]
        assert len(transitions) == 1


class TestTaskGraph:
    def test_patch_cas_and_dag(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        rev = store.apply_patch(
            _patch(
                mission.mission_id,
                0,
                [
                    PatchOperation(op="add_node", node=_node("a")),
                    PatchOperation(op="add_node", node=_node("b", ["a"])),
                ],
            ),
            actor_id=ACTOR,
        )
        assert rev == 1
        graph = store.get_task_graph(mission.mission_id)
        assert graph.revision == 1
        assert graph.node_ids() == {"a", "b"}

    def test_stale_revision_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        store.apply_patch(
            _patch(mission.mission_id, 0, [PatchOperation(op="add_node", node=_node("a"))]),
            actor_id=ACTOR,
        )
        with pytest.raises(RevisionConflictError):
            store.apply_patch(
                _patch(
                    mission.mission_id,
                    0,
                    [PatchOperation(op="add_node", node=_node("b"))],
                ),
                actor_id=ACTOR,
            )

    def test_cycle_patch_rejected_atomically(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        store.apply_patch(
            _patch(
                mission.mission_id,
                0,
                [
                    PatchOperation(op="add_node", node=_node("a")),
                    PatchOperation(op="add_node", node=_node("b", ["a"])),
                ],
            ),
            actor_id=ACTOR,
        )
        with pytest.raises(ValidationError, match="cycle"):
            store.apply_patch(
                _patch(
                    mission.mission_id,
                    1,
                    [PatchOperation(op="update_node", node=_node("a", ["b"]))],
                ),
                actor_id=ACTOR,
            )
        # Rejection must be atomic: graph unchanged, revision unchanged.
        graph = store.get_task_graph(mission.mission_id)
        assert graph.revision == 1
        assert {n.task_id: n.dependencies for n in graph.nodes}["a"] == []

    def test_concurrent_patches_one_wins(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        store.apply_patch(
            _patch(mission.mission_id, 0, [PatchOperation(op="add_node", node=_node("base"))]),
            actor_id=ACTOR,
        )
        results: list[str] = []
        errors: list[Exception] = []

        def contender(name: str) -> None:
            try:
                store.apply_patch(
                    _patch(
                        mission.mission_id,
                        1,
                        [PatchOperation(op="add_node", node=_node(name, ["base"]))],
                    ),
                    actor_id=ACTOR,
                )
                results.append(name)
            except RevisionConflictError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=contender, args=(f"n{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 1
        assert len(errors) == 7
        store.verify_consistency(mission.mission_id)


class TestBudgets:
    def test_usage_accumulates(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store, budgets=Budgets(model_tokens=1000))
        store.add_budget_usage(mission.mission_id, {"model_tokens": 400})
        usage = store.add_budget_usage(mission.mission_id, {"model_tokens": 400})
        assert usage["model_tokens"] == 800

    def test_over_budget_fails_closed(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store, budgets=Budgets(model_tokens=500))
        store.add_budget_usage(mission.mission_id, {"model_tokens": 400})
        with pytest.raises(BudgetExceededError):
            store.add_budget_usage(mission.mission_id, {"model_tokens": 200})
        # Failed add must roll back.
        assert store.budget_usage(mission.mission_id)["model_tokens"] == 400

    def test_unknown_budget_field_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        with pytest.raises(ValidationError):
            store.add_budget_usage(mission.mission_id, {"gpu_hours": 1})


class TestRebinding:
    def test_body_hash_change_marks_physical_nodes(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mission = _mission(store)
        store.apply_patch(
            _patch(
                mission.mission_id,
                0,
                [
                    PatchOperation(op="add_node", node=_node("perceive1", kind="perceive")),
                    PatchOperation(
                        op="add_node",
                        node=_node("act1", ["perceive1"], kind="request_action"),
                    ),
                ],
            ),
            actor_id=ACTOR,
        )
        # Mark act1 READY first.
        store.apply_patch(
            _patch(
                mission.mission_id,
                1,
                [
                    PatchOperation(
                        op="set_status",
                        task_id="act1",
                        status="READY",  # type: ignore[arg-type]
                    )
                ],
            ),
            actor_id=ACTOR,
        )
        affected = store.rebind_body(mission.mission_id, "body_def", actor_id=ACTOR)
        assert affected == 1
        graph = store.get_task_graph(mission.mission_id)
        statuses = {n.task_id: n.status for n in graph.nodes}
        assert statuses["act1"] == "NEEDS_REBINDING"
        assert statuses["perceive1"] == "PENDING"
        updated = store.get_mission(mission.mission_id)
        assert updated is not None
        assert updated.body_binding.effective_body_hash == "body_def"


class TestRecovery:
    def test_reopen_recovers_consistent_state(self, tmp_path: Path) -> None:
        db = tmp_path / "missions.db"
        store = MissionStore(db)
        mission = _mission(store)
        store.transition(
            mission.mission_id,
            MissionState.UNDERSTAND,
            reason_code="new_goal",
            actor_id=ACTOR,
        )
        mid = mission.mission_id
        store.close()
        # Simulate crash: no orderly shutdown of second handle.
        store2 = MissionStore(db)
        loaded = store2.get_mission(mid)
        assert loaded is not None and loaded.state is MissionState.UNDERSTAND
        store2.verify_consistency(mid)

    @pytest.mark.slow
    def test_kill9_mid_write_recovers(self, tmp_path: Path) -> None:
        db = tmp_path / "crash.db"
        child = tmp_path / "crash_child.py"
        child.write_text(
            "import sys, time\n"
            "from rosclaw.agentd.mission import MissionStore\n"
            "from rosclaw.contracts.agent.mission import BodyBinding, Goal, MissionState\n"
            "store = MissionStore(sys.argv[1])\n"
            "m = store.create_mission(owner_principal='user:local:1000', goal=Goal(text='x'),"
            " body_binding=BodyBinding(body_id='b1', effective_body_hash='h1'),"
            " actor_id='test')\n"
            "print(m.mission_id, flush=True)\n"
            "store.transition(m.mission_id, MissionState.UNDERSTAND,"
            " reason_code='go', actor_id='test')\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        proc = subprocess.Popen(
            [sys.executable, str(child), str(db)],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        mid = proc.stdout.readline().strip()
        import signal

        proc.send_signal(signal.SIGKILL)
        proc.wait()
        store = MissionStore(db)
        loaded = store.get_mission(mid)
        assert loaded is not None
        assert loaded.state in (MissionState.IDLE, MissionState.UNDERSTAND)
        store.verify_consistency(mid)


class TestRandomizedStateMachine:
    def test_1000_random_transitions_never_illegal(self, tmp_path: Path) -> None:
        rng = random.Random(20260801)
        store = _store(tmp_path)
        mission = _mission(store)
        mid = mission.mission_id
        all_states = list(MissionState)
        applied = 0
        for _ in range(1000):
            current = store.get_mission(mid)
            assert current is not None
            target = rng.choice(all_states)
            legal = target in MISSION_TRANSITIONS[current.state]
            try:
                store.transition(mid, target, reason_code="fuzz", actor_id=ACTOR)
                assert legal, f"illegal transition accepted: {current.state} -> {target}"
                applied += 1
            except TransitionError:
                assert not legal, f"legal transition rejected: {current.state} -> {target}"
            # Leave terminal states to keep the walk going (re-read state).
            now = store.get_mission(mid)
            assert now is not None
            if now.state is MissionState.FAILED:
                store.transition(mid, MissionState.IDLE, reason_code="reset", actor_id=ACTOR)
                applied += 1
                now = store.get_mission(mid)
                assert now is not None
            if now.state is MissionState.IDLE:
                store.transition(
                    mid, MissionState.UNDERSTAND, reason_code="restart", actor_id=ACTOR
                )
                applied += 1
        assert applied > 100  # walk actually exercised transitions
        store.verify_consistency(mid)
