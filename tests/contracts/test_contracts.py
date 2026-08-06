"""Contract tests (PR-NA-010 exit criteria).

- golden schema stability
- unknown fields preserved, unknown major versions rejected
- canonical hash determinism
- invalid enums rejected
- DAG / lease / side-effect invariants
- no secret-like fields in public contracts (also see tests/architecture)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from rosclaw.contracts.agent.context import (
    AuthorizationContextBinding,
    BodyContextBinding,
    ContextLayers,
    EmbodiedContextBundleV1,
    LayerRef,
)
from rosclaw.contracts.agent.decision import DecisionV1, NextIntent
from rosclaw.contracts.agent.mission import (
    AuthorizationBinding,
    BodyBinding,
    ExecutionMode,
    Goal,
    MissionSessionV1,
    MissionState,
)
from rosclaw.contracts.agent.task_graph import (
    PatchOperation,
    TaskGraphPatchV1,
    TaskGraphV1,
    TaskNodeV1,
)
from rosclaw.contracts.common import (
    UnsupportedVersionError,
    ValidationError,
    new_id,
    parse_schema_version,
)
from rosclaw.contracts.export import ALL_CONTRACTS
from rosclaw.contracts.team.role import RoleLeaseV1
from rosclaw.contracts.worker.card import (
    CapabilityDecl,
    WorkerCardV1,
    WorkerImplementation,
    WorkerKind,
)
from rosclaw.contracts.worker.order import (
    SideEffectPolicy,
    WorkOrderV1,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _mission(**overrides) -> MissionSessionV1:
    payload = {
        "mission_id": new_id("mis"),
        "owner_principal": "user:local:1000",
        "goal": Goal(text="将红色方块从 A 区移动到 B 区"),
        "body_binding": BodyBinding(body_id="sim_ur5e_01", effective_body_hash="body_abc"),
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00Z",
    }
    payload.update(overrides)
    return MissionSessionV1(**payload)


def _task(task_id: str, deps: list[str] | None = None) -> TaskNodeV1:
    return TaskNodeV1(
        task_id=task_id,
        mission_id="mis_x",
        kind="perceive",
        goal="定位红色方块",
        dependencies=deps or [],
    )


class TestGoldenSchemas:
    @pytest.mark.parametrize("schema", sorted(ALL_CONTRACTS))
    def test_schema_matches_golden(self, schema: str) -> None:
        golden = GOLDEN_DIR / f"{schema}.json"
        assert golden.exists(), f"missing golden file {golden}"
        current = ALL_CONTRACTS[schema].model_json_schema()
        current["$id"] = f"rosclaw://schemas/{schema}"
        current["title"] = schema
        assert json.loads(golden.read_text(encoding="utf-8")) == current, (
            f"schema {schema} drifted from golden; if intentional, re-export "
            f"via rosclaw.contracts.export and review the diff"
        )


class TestVersioning:
    def test_unknown_fields_preserved(self) -> None:
        mission = _mission()
        payload = mission.model_dump(mode="json")
        payload["future_field"] = {"from": "v2"}
        parsed = MissionSessionV1.model_validate_contract(payload)
        assert parsed.model_dump(mode="json")["future_field"] == {"from": "v2"}

    def test_unknown_major_version_rejected(self) -> None:
        payload = _mission().model_dump(mode="json")
        payload["schema_version"] = "rosclaw.mission_session.v2"
        with pytest.raises(UnsupportedVersionError):
            MissionSessionV1.model_validate_contract(payload)

    def test_wrong_schema_stem_rejected(self) -> None:
        payload = _mission().model_dump(mode="json")
        payload["schema_version"] = "rosclaw.work_order.v1"
        with pytest.raises(ValidationError):
            MissionSessionV1.model_validate_contract(payload)

    def test_malformed_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_schema_version("mission_session_v1")
        with pytest.raises(ValidationError):
            parse_schema_version("")


class TestCanonicalHash:
    def test_deterministic(self) -> None:
        m1 = _mission(mission_id="mis_fixed")
        m2 = MissionSessionV1(**json.loads(m1.model_dump_json()))
        assert m1.canonical_hash() == m2.canonical_hash()

    def test_hash_excludes_updated_at(self) -> None:
        m1 = _mission(mission_id="mis_fixed")
        m2 = m1.model_copy(update={"updated_at": "2026-08-02T00:00:00Z"})
        assert m1.canonical_hash() == m2.canonical_hash()

    def test_content_change_changes_hash(self) -> None:
        m1 = _mission(mission_id="mis_fixed")
        m2 = m1.model_copy(update={"state": MissionState.UNDERSTAND})
        assert m1.canonical_hash() != m2.canonical_hash()


class TestMissionSession:
    def test_default_mode_simulation(self) -> None:
        assert _mission().mode is ExecutionMode.SIMULATION

    def test_real_mode_requires_action_budget(self) -> None:
        with pytest.raises(PydanticValidationError):
            _mission(mode=ExecutionMode.REAL)

    def test_real_mode_with_budget_ok(self) -> None:
        from rosclaw.contracts.agent.mission import Budgets

        m = _mission(
            mode=ExecutionMode.REAL,
            budgets=Budgets(physical_action_count=3),
            authorization=AuthorizationBinding(mission_grant_id="grant_1"),
        )
        MissionSessionV1(**json.loads(m.model_dump_json()))

    def test_illegal_transition_rejected(self) -> None:
        m = _mission()
        assert not m.can_transition(MissionState.DISPATCH)
        assert m.can_transition(MissionState.UNDERSTAND)

    def test_authorization_never_holds_permit(self) -> None:
        auth = AuthorizationBinding(mission_grant_id="grant_1")
        dumped = json.dumps(auth.model_dump(mode="json"))
        assert "permit" not in dumped.lower()


class TestTaskGraph:
    def test_valid_dag(self) -> None:
        g = TaskGraphV1(mission_id="mis_x", nodes=[_task("a"), _task("b", ["a"])])
        g.validate_dag()

    def test_cycle_rejected(self) -> None:
        g = TaskGraphV1(mission_id="mis_x", nodes=[_task("a", ["b"]), _task("b", ["a"])])
        with pytest.raises(ValidationError, match="cycle"):
            g.validate_dag()

    def test_dangling_dependency_rejected(self) -> None:
        g = TaskGraphV1(mission_id="mis_x", nodes=[_task("a", ["ghost"])])
        with pytest.raises(ValidationError, match="unknown task"):
            g.validate_dag()

    def test_duplicate_id_rejected(self) -> None:
        g = TaskGraphV1(mission_id="mis_x", nodes=[_task("a"), _task("a")])
        with pytest.raises(ValidationError, match="duplicate"):
            g.validate_dag()

    def test_patch_is_proposal_not_state(self) -> None:
        patch = TaskGraphPatchV1(
            patch_id=new_id("tgpatch"),
            mission_id="mis_x",
            base_revision=3,
            proposed_by="agent:rosclaw-native:sim_ur5e_01",
            operations=[PatchOperation(op="add_node", node=_task("a"))],
        )
        assert patch.base_revision == 3
        # A patch carries no committed graph — only operations.
        assert not hasattr(patch, "nodes")


class TestDecision:
    def test_next_intent_enum_closed(self) -> None:
        d = DecisionV1(
            decision_id=new_id("dec"),
            mission_id="mis_x",
            context_id="ctx_1",
            context_revision=1,
            next_intent=NextIntent.FAIL_SAFE,
        )
        assert d.next_intent.value == "FAIL_SAFE"
        with pytest.raises(PydanticValidationError):
            DecisionV1(
                decision_id="dec_x",
                mission_id="mis_x",
                context_id="ctx_1",
                context_revision=1,
                next_intent="DO_WHATEVER",
            )


class TestWorkerContracts:
    def test_card_minimal(self) -> None:
        card = WorkerCardV1(
            worker_id="worker:codex:local-default",
            kind=WorkerKind.HARNESS,
            implementation=WorkerImplementation(product="codex", version="0.1"),
            capabilities=[CapabilityDecl(name="code.repository_edit")],
        )
        assert "direct_hardware" in card.security.forbidden_scopes

    def test_side_effect_requires_idempotency(self) -> None:
        with pytest.raises(PydanticValidationError):
            SideEffectPolicy(**{"class": "workspace_write"})

    def test_physical_side_effects_forbidden(self) -> None:
        with pytest.raises(PydanticValidationError):
            SideEffectPolicy(**{"class": "physical", "idempotency_key": "idem_1"})

    def test_work_order_illegal_status(self) -> None:
        with pytest.raises(PydanticValidationError):
            WorkOrderV1(
                work_order_id="wo_1",
                mission_id="mis_x",
                issued_by="agent:rosclaw-native:b1",
                capability="code.repository_edit",
                goal="x",
                status="TELEPORT",
            )


class TestTeamContracts:
    def test_role_lease_conflict_key(self) -> None:
        lease = RoleLeaseV1(
            team_id="blue_team",
            team_epoch=27,
            role="defender:left",
            holder="robot:limo:blue_02",
            issued_at="2026-08-01T00:00:00Z",
            expires_at="2026-08-01T00:00:01Z",
            conflict_key="role:defender:left",
            policy_hash="team_pol_x",
        )
        assert lease.state == "ACTIVE"


class TestContextBundle:
    def _bundle(self) -> EmbodiedContextBundleV1:
        layer = LayerRef(hash="h")
        layers = ContextLayers(
            constitution=LayerRef(hash="l0"),
            embodiment=LayerRef(hash="l1"),
            dynamic_self=LayerRef(hash="l2"),
            capabilities=layer,
            mission=LayerRef(hash="l4"),
            safety=LayerRef(hash="l7"),
        )
        return EmbodiedContextBundleV1(
            context_id=new_id("ctx"),
            context_revision=1,
            compiled_at="2026-08-01T00:00:00Z",
            mission_id="mis_x",
            body_binding=BodyContextBinding(body_id="sim_ur5e_01", effective_body_hash="body_abc"),
            authorization_binding=AuthorizationContextBinding(policy_hash="pol_x"),
            layers=layers,
        )

    def test_finalize_hash_stable(self) -> None:
        b = self._bundle()
        h1 = b.finalize_hash()
        assert h1.startswith("ctxb_")
        b2 = EmbodiedContextBundleV1(**json.loads(b.model_dump_json()))
        assert b2.bundle_hash == h1

    def test_hash_excludes_bundle_hash_field(self) -> None:
        b = self._bundle()
        b.finalize_hash()
        again = b.canonical_hash()
        assert again == b.bundle_hash
