"""MissionStore — durable mission sessions and versioned task graphs.

Design (ADR-0002, PR-NA-011):

- SQLite WAL; migrations come from ``rosclaw.storage.migrations``.
- ``mission_events`` is an append-only journal. Every transition records
  ``from_state/to_state/reason_code/actor_id/trace_id`` and can carry an
  idempotency key for exactly-once application under retries.
- Task graph commits use revision CAS: a patch names its ``base_revision``;
  if the current revision moved, the patch is rejected and the proposer must
  re-plan. The patched graph is DAG-validated *before* commit.
- Legal mission transitions are the closed table from the contract layer —
  an LLM can never invent a state.
- Budget counters are durable; exceeding any budget raises ``BudgetExceededError``
  so the loop can enter WAIT_INPUT/SUSPENDED (decided by the caller).
- Recovery: the journal is authoritative; ``verify_consistency`` replays
  transitions and checks the projection matches. A killed process leaves at
  worst a partially-written event, which SQLite atomicity prevents.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.contracts.agent.mission import (
    MISSION_TRANSITIONS,
    TERMINAL_STATES,
    AuthorizationBinding,
    BodyBinding,
    Budgets,
    ExecutionMode,
    Goal,
    MissionSessionV1,
    MissionState,
)
from rosclaw.contracts.agent.task_graph import (
    TaskGraphPatchV1,
    TaskGraphV1,
    TaskNodeV1,
    TaskStatus,
)
from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.storage.migrations import MigrationRunner

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "storage" / "migrations"

BUDGET_FIELDS = (
    "wall_time_sec",
    "model_tokens",
    "monetary_microunits",
    "worker_concurrency",
    "physical_action_count",
    "max_tool_rounds",
)

# Task kinds whose execution touches the body binding.
PHYSICAL_TASK_KINDS = frozenset({"request_action"})


class TransitionError(ValidationError):
    """Illegal mission state transition."""


class RevisionConflictError(ValidationError):
    """Task-graph patch base revision is stale."""


class BudgetExceededError(ValidationError):
    """A durable budget counter would exceed its envelope."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class MissionStore:
    """Thread-safe store. One instance per database file."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        MigrationRunner(MIGRATIONS_DIR).apply(self._conn, "sqlite")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Shared connection for companion recorders (usage, artifacts)."""
        return self._conn

    # ------------------------------------------------------------------
    # conversation journal (evidence of dialogue; state still comes from
    # missions/task_nodes — chat text is never the source of truth)
    # ------------------------------------------------------------------
    def append_conversation(self, mission_id: str, messages: list[dict], *, actor_id: str) -> None:
        """Append messages to the canonical journal.

        Each message is stamped in place with a stable ``entry_id`` and
        monotonic per-mission ``seq`` (补充实施文档 §3.3), so the caller's
        in-memory view and the journal share message identity.
        """
        with self._lock:
            seq = self._conversation_length(mission_id)
            stamped: list[dict] = []
            for message in messages:
                message.setdefault("entry_id", f"conv_{mission_id}_{seq}")
                message.setdefault("seq", seq)
                seq += 1
                stamped.append(message)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._append_event(
                    mission_id=mission_id,
                    event_type="rosclaw.agent.conversation.appended.v1",
                    from_state=None,
                    to_state=None,
                    reason_code="conversation_turn",
                    actor_id=actor_id,
                    trace_id=None,
                    payload={"messages": stamped},
                    idempotency_key=None,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _conversation_length(self, mission_id: str) -> int:
        # 快路径：最后一批消息的最后一条已带 seq → 总数 = seq + 批内序号。
        # 只有历史遗留（未赋 seq 的旧消息）才全量扫描。
        row = self._conn.execute(
            "SELECT payload_json FROM mission_events WHERE mission_id = ? "
            "AND event_type = 'rosclaw.agent.conversation.appended.v1' "
            "ORDER BY seq DESC LIMIT 1",
            (mission_id,),
        ).fetchone()
        if row is not None:
            payload = json.loads(row["payload_json"])
            messages = payload.get("messages") or []
            if messages and "seq" in messages[-1]:
                return int(messages[-1]["seq"]) + 1
        total = 0
        for event in self.events(mission_id):
            if event["event_type"] == "rosclaw.agent.conversation.appended.v1":
                payload = json.loads(event["payload_json"])
                total += len(payload.get("messages") or [])
        return total

    # ------------------------------------------------------------------
    # mission meta (批次 B: display name / archive; 不改 Mission 契约)
    # ------------------------------------------------------------------
    def set_mission_meta(
        self, mission_id: str, *, display_name: str | None = None, archived: bool | None = None
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT display_name, archived FROM mission_meta WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()
            name = display_name if display_name is not None else (row["display_name"] if row else "")
            arch = archived if archived is not None else (bool(row["archived"]) if row else False)
            self._conn.execute(
                "INSERT INTO mission_meta (mission_id, display_name, archived, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(mission_id) DO UPDATE SET "
                "display_name = excluded.display_name, archived = excluded.archived, "
                "updated_at = excluded.updated_at",
                (mission_id, name, 1 if arch else 0, _utcnow()),
            )

    def mission_meta(self, mission_id: str) -> dict:
        row = self._conn.execute(
            "SELECT display_name, archived FROM mission_meta WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        if row is None:
            return {"display_name": "", "archived": False}
        return {"display_name": row["display_name"], "archived": bool(row["archived"])}

    def conversation_canonical(self, mission_id: str) -> list[dict]:
        """完整 canonical journal（含 compaction 前的原始消息与 marker）。

        fork/import/tree 永远基于 canonical journal，不基于压缩后的临时
        view（补充实施文档 §3.3 第 8 条）。
        """
        messages: list[dict] = []
        for event in self.events(mission_id):
            if event["event_type"] == "rosclaw.agent.conversation.appended.v1":
                payload = json.loads(event["payload_json"])
                messages.extend(payload.get("messages") or [])
        return messages

    def conversation(self, mission_id: str) -> list[dict]:
        messages: list[dict] = []
        for event in self.events(mission_id):
            if event["event_type"] == "rosclaw.agent.conversation.appended.v1":
                payload = json.loads(event["payload_json"])
                messages.extend(payload.get("messages") or [])
        # PR-07：从最新 compaction 标记恢复 view；canonical journal 本身不动。
        from rosclaw.agentd.context.compaction import restore_view_from_journal

        return restore_view_from_journal(messages)

    # ------------------------------------------------------------------
    # missions
    # ------------------------------------------------------------------
    def create_mission(
        self,
        *,
        owner_principal: str,
        goal: Goal,
        body_binding: BodyBinding,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        budgets: Budgets | None = None,
        authorization: AuthorizationBinding | None = None,
        actor_id: str,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> MissionSessionV1:
        with self._lock:
            if idempotency_key is not None:
                prior = self._conn.execute(
                    "SELECT mission_id FROM mission_events WHERE idempotency_key = ? LIMIT 1",
                    (idempotency_key,),
                ).fetchone()
                if prior is not None:
                    existing = self.get_mission(prior["mission_id"])
                    if existing is not None:
                        return existing
            now = _utcnow()
            mission = MissionSessionV1(
                mission_id=new_id("mis"),
                owner_principal=owner_principal,
                goal=goal,
                body_binding=body_binding,
                mode=mode,
                state=MissionState.IDLE,
                budgets=budgets or Budgets(),
                authorization=authorization or AuthorizationBinding(),
                created_at=now,
                updated_at=now,
            )
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._insert_mission_row(mission)
                self._append_event(
                    mission_id=mission.mission_id,
                    event_type="rosclaw.agent.mission.created.v1",
                    from_state=None,
                    to_state=mission.state.value,
                    reason_code="mission_created",
                    actor_id=actor_id,
                    trace_id=trace_id,
                    payload={"mission": mission.model_dump(mode="json")},
                    idempotency_key=idempotency_key,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return mission

    def get_mission(self, mission_id: str) -> MissionSessionV1 | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_mission(row)

    def list_missions(self, *, state: MissionState | None = None) -> list[MissionSessionV1]:
        with self._lock:
            if state is None:
                rows = self._conn.execute("SELECT * FROM missions ORDER BY created_at").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM missions WHERE state = ? ORDER BY created_at",
                    (state.value,),
                ).fetchall()
        return [self._row_to_mission(r) for r in rows]

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------
    def transition(
        self,
        mission_id: str,
        to_state: MissionState,
        *,
        reason_code: str,
        actor_id: str,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> MissionSessionV1:
        """Atomically validate + journal + project a state transition."""
        with self._lock:
            if idempotency_key is not None:
                prior = self._event_by_idempotency(mission_id, idempotency_key)
                if prior is not None:
                    # Idempotent replay: return current state without re-applying.
                    mission = self.get_mission(mission_id)
                    if mission is None:
                        raise ValidationError(f"unknown mission {mission_id!r}")
                    return mission
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
                ).fetchone()
                if row is None:
                    raise ValidationError(f"unknown mission {mission_id!r}")
                from_state = MissionState(row["state"])
                if to_state not in MISSION_TRANSITIONS[from_state]:
                    raise TransitionError(
                        f"illegal transition {from_state.value} -> {to_state.value}"
                    )
                now = _utcnow()
                self._conn.execute(
                    "UPDATE missions SET state = ?, updated_at = ? WHERE mission_id = ?",
                    (to_state.value, now, mission_id),
                )
                self._append_event(
                    mission_id=mission_id,
                    event_type="rosclaw.agent.mission.transition.v1",
                    from_state=from_state.value,
                    to_state=to_state.value,
                    reason_code=reason_code,
                    actor_id=actor_id,
                    trace_id=trace_id,
                    payload={},
                    idempotency_key=idempotency_key,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        mission = self.get_mission(mission_id)
        assert mission is not None
        return mission

    # ------------------------------------------------------------------
    # task graph
    # ------------------------------------------------------------------
    def get_task_graph(self, mission_id: str) -> TaskGraphV1:
        with self._lock:
            mission = self.get_mission(mission_id)
            if mission is None:
                raise ValidationError(f"unknown mission {mission_id!r}")
            rows = self._conn.execute(
                "SELECT node_json FROM task_nodes WHERE mission_id = ?", (mission_id,)
            ).fetchall()
        nodes = [TaskNodeV1(**json.loads(r["node_json"])) for r in rows]
        return TaskGraphV1(
            mission_id=mission_id,
            revision=mission.task_graph_revision,
            nodes=nodes,
        )

    def apply_patch(self, patch: TaskGraphPatchV1, *, actor_id: str) -> int:
        """CAS-apply a TaskGraphPatchV1 proposal. Returns the new revision."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT task_graph_revision FROM missions WHERE mission_id = ?",
                    (patch.mission_id,),
                ).fetchone()
                if row is None:
                    raise ValidationError(f"unknown mission {patch.mission_id!r}")
                current = int(row["task_graph_revision"])
                if patch.base_revision != current:
                    raise RevisionConflictError(
                        f"patch base_revision {patch.base_revision} != current {current}"
                    )
                new_revision = current + 1
                node_rows = self._conn.execute(
                    "SELECT task_id, node_json FROM task_nodes WHERE mission_id = ?",
                    (patch.mission_id,),
                ).fetchall()
                nodes = {r["task_id"]: TaskNodeV1(**json.loads(r["node_json"])) for r in node_rows}
                for op in patch.operations:
                    if op.op == "add_node":
                        if op.node is None:
                            raise ValidationError("add_node requires node")
                        if op.node.task_id in nodes:
                            raise ValidationError(f"task {op.node.task_id!r} already exists")
                        nodes[op.node.task_id] = op.node
                    elif op.op == "remove_node":
                        if not op.task_id or op.task_id not in nodes:
                            raise ValidationError(f"unknown task {op.task_id!r}")
                        del nodes[op.task_id]
                    elif op.op == "update_node":
                        if op.node is None or op.node.task_id not in nodes:
                            raise ValidationError(
                                f"unknown task {getattr(op.node, 'task_id', None)!r}"
                            )
                        nodes[op.node.task_id] = op.node
                    elif op.op == "set_status":
                        if not op.task_id or op.task_id not in nodes or op.status is None:
                            raise ValidationError("set_status requires task_id and status")
                        nodes[op.task_id] = nodes[op.task_id].model_copy(
                            update={"status": op.status}
                        )
                    else:  # pragma: no cover - enum closed by contract
                        raise ValidationError(f"unknown patch op {op.op!r}")
                graph = TaskGraphV1(
                    mission_id=patch.mission_id,
                    revision=new_revision,
                    nodes=list(nodes.values()),
                )
                graph.validate_dag()
                now = _utcnow()
                self._conn.execute(
                    "DELETE FROM task_nodes WHERE mission_id = ?", (patch.mission_id,)
                )
                self._conn.execute(
                    "DELETE FROM task_edges WHERE mission_id = ?", (patch.mission_id,)
                )
                for node in graph.nodes:
                    self._conn.execute(
                        "INSERT INTO task_nodes "
                        "(mission_id, task_id, revision, kind, status, node_json, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            patch.mission_id,
                            node.task_id,
                            new_revision,
                            node.kind.value,
                            node.status.value,
                            node.model_dump_json(),
                            now,
                        ),
                    )
                    for dep in node.dependencies:
                        self._conn.execute(
                            "INSERT INTO task_edges (mission_id, from_task, to_task, revision) "
                            "VALUES (?, ?, ?, ?)",
                            (patch.mission_id, dep, node.task_id, new_revision),
                        )
                self._conn.execute(
                    "UPDATE missions SET task_graph_revision = ?, updated_at = ? "
                    "WHERE mission_id = ?",
                    (new_revision, now, patch.mission_id),
                )
                self._append_event(
                    mission_id=patch.mission_id,
                    event_type="rosclaw.agent.task_graph.patched.v1",
                    from_state=None,
                    to_state=None,
                    reason_code="task_graph_patch",
                    actor_id=actor_id,
                    trace_id=None,
                    payload={
                        "patch_id": patch.patch_id,
                        "base_revision": patch.base_revision,
                        "new_revision": new_revision,
                        "context_revision": patch.context_revision,
                        "proposed_by": patch.proposed_by,
                    },
                    idempotency_key=patch.patch_id,
                )
                self._conn.execute("COMMIT")
                return new_revision
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # body rebinding
    # ------------------------------------------------------------------
    def rebind_body(self, mission_id: str, new_body_hash: str, *, actor_id: str) -> int:
        """Update body hash; physical pending nodes become NEEDS_REBINDING."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT effective_body_hash FROM missions WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()
                if row is None:
                    raise ValidationError(f"unknown mission {mission_id!r}")
                now = _utcnow()
                self._conn.execute(
                    "UPDATE missions SET effective_body_hash = ?, updated_at = ? "
                    "WHERE mission_id = ?",
                    (new_body_hash, now, mission_id),
                )
                affected = 0
                node_rows = self._conn.execute(
                    "SELECT task_id, node_json FROM task_nodes WHERE mission_id = ?",
                    (mission_id,),
                ).fetchall()
                for r in node_rows:
                    node = TaskNodeV1(**json.loads(r["node_json"]))
                    if node.kind.value in PHYSICAL_TASK_KINDS and node.status in (
                        TaskStatus.PENDING,
                        TaskStatus.READY,
                        TaskStatus.BLOCKED,
                    ):
                        node = node.model_copy(update={"status": TaskStatus.NEEDS_REBINDING})
                        self._conn.execute(
                            "UPDATE task_nodes SET status = ?, node_json = ?, updated_at = ? "
                            "WHERE mission_id = ? AND task_id = ?",
                            (
                                node.status.value,
                                node.model_dump_json(),
                                now,
                                mission_id,
                                node.task_id,
                            ),
                        )
                        affected += 1
                self._append_event(
                    mission_id=mission_id,
                    event_type="rosclaw.agent.body.rebound.v1",
                    from_state=None,
                    to_state=None,
                    reason_code="body_hash_changed",
                    actor_id=actor_id,
                    trace_id=None,
                    payload={
                        "old_body_hash": row["effective_body_hash"],
                        "new_body_hash": new_body_hash,
                        "nodes_needs_rebinding": affected,
                    },
                    idempotency_key=None,
                )
                self._conn.execute("COMMIT")
                return affected
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # context revision
    # ------------------------------------------------------------------
    def bump_context_revision(self, mission_id: str) -> int:
        """Persist the newly compiled context revision (stale-decision guard)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT context_revision FROM missions WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()
                if row is None:
                    raise ValidationError(f"unknown mission {mission_id!r}")
                new_rev = int(row["context_revision"]) + 1
                self._conn.execute(
                    "UPDATE missions SET context_revision = ?, updated_at = ? WHERE mission_id = ?",
                    (new_rev, _utcnow(), mission_id),
                )
                self._conn.execute("COMMIT")
                return new_rev
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # attribution: decisions + context manifests (§12.3)
    # ------------------------------------------------------------------
    def record_decision(
        self,
        decision,
        *,
        validated: bool,
        reason_code: str | None,
        actor_id: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions (decision_id, mission_id, "
                "context_id, context_revision, decision_json, validated, "
                "reason_code, actor_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.mission_id,
                    decision.context_id,
                    decision.context_revision,
                    decision.model_dump_json(),
                    1 if validated else 0,
                    reason_code,
                    actor_id,
                    _utcnow(),
                ),
            )

    def record_context_manifest(self, bundle, *, prompt_hash: str) -> None:
        manifest = {
            "compiler_version": bundle.compiler_version,
            "layers": {
                name: {
                    "hash": layer.hash,
                    "token_estimate": layer.token_estimate,
                }
                for name, layer in (
                    ("constitution", bundle.layers.constitution),
                    ("embodiment", bundle.layers.embodiment),
                    ("dynamic_self", bundle.layers.dynamic_self),
                    ("capabilities", bundle.layers.capabilities),
                    ("mission", bundle.layers.mission),
                    ("memory", bundle.layers.memory),
                    ("organization", bundle.layers.organization),
                    ("safety", bundle.layers.safety),
                    ("untrusted_inputs", bundle.layers.untrusted_inputs),
                )
                if layer is not None
            },
            "budget": {
                "maximum_input_tokens": bundle.budget.maximum_input_tokens,
                "used_tokens": bundle.budget.used_tokens,
                "truncation_events": [
                    e.model_dump(mode="json") for e in bundle.budget.truncation_events
                ],
            },
            "body_binding": bundle.body_binding.model_dump(mode="json"),
            "self_binding": (
                bundle.self_binding.model_dump(mode="json") if bundle.self_binding else None
            ),
            "team_binding": (
                bundle.team_binding.model_dump(mode="json") if bundle.team_binding else None
            ),
            "authorization_binding": bundle.authorization_binding.model_dump(mode="json"),
        }
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO context_manifests (context_id, "
                "context_revision, mission_id, bundle_hash, prompt_hash, "
                "manifest_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    bundle.context_id,
                    bundle.context_revision,
                    bundle.mission_id,
                    bundle.bundle_hash,
                    prompt_hash,
                    json.dumps(manifest, sort_keys=True, ensure_ascii=False),
                    _utcnow(),
                ),
            )

    # ------------------------------------------------------------------
    # budgets
    # ------------------------------------------------------------------
    def add_budget_usage(self, mission_id: str, usage: dict[str, int]) -> dict[str, int]:
        """Add to durable budget counters; fail closed when over envelope."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT budgets_json, budget_usage_json FROM missions WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()
                if row is None:
                    raise ValidationError(f"unknown mission {mission_id!r}")
                budgets = json.loads(row["budgets_json"])
                counters = json.loads(row["budget_usage_json"])
                for key, delta in usage.items():
                    if key not in BUDGET_FIELDS:
                        raise ValidationError(f"unknown budget field {key!r}")
                    counters[key] = int(counters.get(key, 0)) + int(delta)
                for key in BUDGET_FIELDS:
                    limit = int(budgets.get(key, 0))
                    if limit > 0 and counters.get(key, 0) > limit:
                        raise BudgetExceededError(
                            f"budget {key} exceeded: {counters[key]} > {limit}"
                        )
                self._conn.execute(
                    "UPDATE missions SET budget_usage_json = ?, updated_at = ? "
                    "WHERE mission_id = ?",
                    (json.dumps(counters, sort_keys=True), _utcnow(), mission_id),
                )
                self._conn.execute("COMMIT")
                return counters
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def budget_usage(self, mission_id: str) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT budget_usage_json FROM missions WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()
        if row is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        return {k: int(v) for k, v in json.loads(row["budget_usage_json"]).items()}

    # ------------------------------------------------------------------
    # journal / recovery
    # ------------------------------------------------------------------
    def events(self, mission_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM mission_events WHERE mission_id = ? ORDER BY seq",
                (mission_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def verify_consistency(self, mission_id: str) -> None:
        """Replay the journal and check the projection is consistent."""
        mission = self.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        events = self.events(mission_id)
        if not events:
            raise ValidationError("journal is empty; mission row without events")
        seqs = [e["seq"] for e in events]
        if seqs != list(range(1, len(events) + 1)):
            raise ValidationError(f"journal sequence gap: {seqs}")
        state = MissionState.IDLE
        for event in events:
            if event["event_type"] == "rosclaw.agent.mission.transition.v1":
                from_s = MissionState(event["from_state"])
                to_s = MissionState(event["to_state"])
                if from_s != state:
                    raise ValidationError(
                        f"journal/projection drift at seq {event['seq']}: "
                        f"from {from_s} but replay state {state}"
                    )
                if to_s not in MISSION_TRANSITIONS[from_s]:
                    raise ValidationError(f"illegal journaled transition at seq {event['seq']}")
                state = to_s
        if mission.state not in TERMINAL_STATES and state != mission.state:
            raise ValidationError(f"projection state {mission.state} != replay state {state}")
        # Task graph projection must be DAG-valid at the current revision.
        self.get_task_graph(mission_id).validate_dag()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _insert_mission_row(self, mission: MissionSessionV1) -> None:
        self._conn.execute(
            "INSERT INTO missions (mission_id, owner_principal, goal_json, body_id, "
            "effective_body_hash, mode, state, budgets_json, authorization_json, "
            "context_revision, task_graph_revision, budget_usage_json, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mission.mission_id,
                mission.owner_principal,
                mission.goal.model_dump_json(),
                mission.body_binding.body_id,
                mission.body_binding.effective_body_hash,
                mission.mode.value,
                mission.state.value,
                mission.budgets.model_dump_json(),
                mission.authorization.model_dump_json(),
                mission.context_revision,
                mission.task_graph_revision,
                "{}",
                mission.created_at,
                mission.updated_at,
            ),
        )

    def _row_to_mission(self, row: sqlite3.Row) -> MissionSessionV1:
        return MissionSessionV1(
            mission_id=row["mission_id"],
            owner_principal=row["owner_principal"],
            goal=Goal(**json.loads(row["goal_json"])),
            body_binding=BodyBinding(
                body_id=row["body_id"], effective_body_hash=row["effective_body_hash"]
            ),
            mode=ExecutionMode(row["mode"]),
            state=MissionState(row["state"]),
            budgets=Budgets(**json.loads(row["budgets_json"])),
            authorization=AuthorizationBinding(**json.loads(row["authorization_json"])),
            context_revision=row["context_revision"],
            task_graph_revision=row["task_graph_revision"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _event_by_idempotency(
        self, mission_id: str, idempotency_key: str | None
    ) -> sqlite3.Row | None:
        if idempotency_key is None:
            return None
        return self._conn.execute(
            "SELECT * FROM mission_events WHERE mission_id = ? AND idempotency_key = ?",
            (mission_id, idempotency_key),
        ).fetchone()

    def _append_event(
        self,
        *,
        mission_id: str,
        event_type: str,
        from_state: str | None,
        to_state: str | None,
        reason_code: str,
        actor_id: str,
        trace_id: str | None,
        payload: dict,
        idempotency_key: str | None,
    ) -> str:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM mission_events WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        seq = int(row["next_seq"])
        event_id = new_id("evt")
        now = _utcnow()
        self._conn.execute(
            "INSERT INTO mission_events (event_id, mission_id, seq, event_type, "
            "from_state, to_state, reason_code, actor_id, trace_id, payload_json, "
            "idempotency_key, occurred_at, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                mission_id,
                seq,
                event_type,
                from_state,
                to_state,
                reason_code,
                actor_id,
                trace_id,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                idempotency_key,
                now,
                now,
            ),
        )
        return event_id
