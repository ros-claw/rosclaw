"""Real ContextCompiler integration; historical or changed facts fail closed."""

from dataclasses import replace
from datetime import timedelta

import pytest

from rosclaw.agentd.context.compiler import CompilationError, ContextCompiler, StaleSourceError
from rosclaw.agentd.context.sources import SelfFacts, SourceBundle
from rosclaw.agentd.runtime_sources import (
    ConfigConsentSource,
    EmptyMemorySource,
    NullOrgSource,
    SimBodySource,
    StaticCapabilitySource,
)
from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.contracts.agent.mission import BodyBinding, Goal, MissionSessionV1
from rosclaw.contracts.agent.task_graph import TaskGraphV1
from tests.connectors.ros.test_engineering_case_compiler import NOW, setup


class FixedSelf:
    def get_self(self, body_id):
        return SelfFacts("original-self", 1, NOW, summary="original self")


def fixture(domain="OBSERVATION", **source_changes):
    model, urdf, requirements = setup()
    body = SimBodySource(model.robot_id)
    source = {
        "schema_version": "rosclaw.sim_source_binding_evidence.v1",
        "result": "MATCHED",
        "ros_runtime_snapshot_hash": model.snapshot_hash,
        "sim_run_id": "a" * 32,
        "body_hash": body.body_hash,
        "reviewed_at": NOW.isoformat(),
        "observed_at": NOW.isoformat(),
        **source_changes,
    }
    case = compile_engineering_case(
        model,
        urdf,
        requirements,
        source_ref="fixture:original",
        now=NOW,
        source_review=source,
        evidence_domain=domain,
    )
    sources = SourceBundle(
        constitution_text="Guarded actions only",
        body=body,
        self_source=FixedSelf(),
        capabilities=StaticCapabilitySource(["request_action"]),
        memory=EmptyMemorySource(),
        organization=NullOrgSource(),
        consent=ConfigConsentSource(),
        extra={"ros_system_model": model, "ros_engineering_case": case},
    )
    mission = MissionSessionV1(
        mission_id="m-engineering",
        owner_principal="user:fixture",
        goal=Goal(text="navigate"),
        body_binding=BodyBinding(body_id=model.robot_id, effective_body_hash=body.body_hash),
        created_at=NOW.isoformat(),
        updated_at=NOW.isoformat(),
    )
    graph = TaskGraphV1(graph_id="g-engineering", mission_id=mission.mission_id)
    return ContextCompiler(sources), sources, mission, graph, case


def compile_context(compiler, mission, graph, **options):
    return compiler.compile(
        mission, graph, [], context_revision=1, context_id="fixed", now=options.get("now", NOW)
    )


def test_fresh_case_uses_protected_self_and_keeps_body_capability_safety_contracts():
    compiler, sources, mission, graph, case = fixture()
    bundle = compile_context(compiler, mission, graph)
    assert bundle.bundle_hash == compile_context(compiler, mission, graph).bundle_hash
    assert case.case_hash in bundle.layers.dynamic_self.evidence_refs
    assert '"L1": "NOT_EVALUATED"' in bundle.layers.dynamic_self.inline_summary
    assert '"authority": "NONE"' in bundle.layers.dynamic_self.inline_summary
    baseline = ContextCompiler(
        replace(sources, extra={"ros_system_model": sources.extra["ros_system_model"]})
    )
    original = compile_context(baseline, mission, graph)
    for name in ("embodiment", "capabilities", "mission", "memory", "safety"):
        assert getattr(bundle.layers, name) == getattr(original.layers, name)
    assert compiler.staleness_reasons(bundle) == []


@pytest.mark.parametrize(
    "change", ["historical", "body", "snapshot", "source", "missing_model", "tamper"]
)
def test_invalid_case_never_enters_live_context(change):
    compiler, sources, mission, graph, case = fixture(
        domain="SEALED_REPLAY" if change == "historical" else "OBSERVATION",
        **({"body_hash": "wrong"} if change == "body" else {}),
        **({"result": "MISMATCH"} if change == "source" else {}),
    )
    if change == "missing_model":
        del sources.extra["ros_system_model"]
    if change == "snapshot":
        model = sources.extra["ros_system_model"]
        model.errors.append("new fact")
        model.seal()
    if change == "tamper":
        case.constraints[0].evidence_refs.append("invented")
    with pytest.raises(CompilationError):
        compile_context(compiler, mission, graph)


def test_constraints_must_be_recompiled_at_each_context_clock_even_inside_snapshot_age_limit():
    compiler, _, mission, graph, _ = fixture()
    with pytest.raises(StaleSourceError, match="recompile engineering"):
        compile_context(compiler, mission, graph, now=NOW + timedelta(milliseconds=1))


def test_revision_and_removal_trigger_recompile_without_overwriting_original_case():
    compiler, sources, mission, graph, case = fixture()
    bundle = compile_context(compiler, mission, graph)
    sources.extra["ros_engineering_case"] = fixture(result="UNKNOWN")[4]
    assert "engineering_case_changed" in compiler.staleness_reasons(bundle)
    assert case.case_hash in bundle.layers.dynamic_self.evidence_refs
    del sources.extra["ros_engineering_case"]
    assert "engineering_case_unavailable_or_invalid" in compiler.staleness_reasons(bundle)


def test_protected_engineering_facts_cannot_be_trimmed_into_false_readiness():
    compiler, sources, mission, graph, _ = fixture()
    compiler = ContextCompiler(sources, max_input_tokens=10)
    with pytest.raises(CompilationError, match="protected layers"):
        compile_context(compiler, mission, graph)
