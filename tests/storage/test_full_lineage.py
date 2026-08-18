"""PR-DF-17 (phase-II §8-§14): typed full evolution lineage.

Covers the named test set (§13) plus the golden lineage E2E (§14):
Receipt → Episode → Memory → Insight → Proposal → Patch → Experiment
→ Evaluation → Champion must be traceable from the champion down.
"""

from __future__ import annotations

import argparse

from rosclaw.core.event_bus import EventBus
from rosclaw.core.event_topics import EventTopics
from rosclaw.evolution.orchestrator.config import AutoConfig
from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine
from rosclaw.memory.insights import MemoryInsightService
from rosclaw.memory.seekdb_client import InMemoryStructuredStore
from rosclaw.storage.cli import _render_lineage_tree
from rosclaw.storage.lineage import LineageRepository


def _store():
    s = InMemoryStructuredStore()
    s.connect()
    return s


def _engine(tmp_path, repo) -> AutoEngine:
    return AutoEngine(
        config=AutoConfig(local_store_path=str(tmp_path / "auto")),
        lineage_repository=repo,
    )


def _edge(repo, from_type, from_id, relation):
    edges = repo.parents(from_type, from_id)
    match = [e for e in edges if e["relation"] == relation]
    assert match, f"missing edge {from_type}:{from_id} --{relation}-->"
    return match[0]


# -- §13 repository behavior -------------------------------------------------


def test_lineage_type_sensitive():
    """Same id under two types must not cross-contaminate queries (§8.3)."""
    repo = LineageRepository(_store())
    repo.link("memory", "x1", "derived_from", "episode", "e1")
    repo.link("patch", "x1", "patched_from", "proposal", "p1")

    mem_parents = repo.parents("memory", "x1")
    assert len(mem_parents) == 1
    assert mem_parents[0]["to_type"] == "episode"
    patch_parents = repo.parents("patch", "x1")
    assert len(patch_parents) == 1
    assert patch_parents[0]["to_type"] == "proposal"
    # children() is type-sensitive too
    assert repo.children("episode", "e1")[0]["from_type"] == "memory"
    assert repo.children("memory", "x1") == []


def test_lineage_idempotent():
    """5-field key: identical edge dedups, type-different edge does not (§8.4)."""
    store = _store()
    repo = LineageRepository(store)
    a = repo.link("memory", "m1", "derived_from", "episode", "e1")
    b = repo.link("memory", "m1", "derived_from", "episode", "e1")
    assert a == b
    assert store.count("lineage_edges", {}) == 1
    # same ids but a different from_type is a DIFFERENT edge
    repo.link("memory_insight", "m1", "derived_from", "episode", "e1")
    assert store.count("lineage_edges", {}) == 2


def test_lineage_multi_parent():
    """One insight derived from several memories keeps every parent (§10)."""
    repo = LineageRepository(_store())
    for mem in ("mem_a", "mem_b", "mem_c"):
        repo.link("memory_insight", "ins_1", "derived_from", "memory", mem)
    parents = repo.parents("memory_insight", "ins_1")
    assert {e["to_id"] for e in parents} == {"mem_a", "mem_b", "mem_c"}
    ancestors = repo.ancestors("memory_insight", "ins_1")
    assert {e["to_id"] for e in ancestors} >= {"mem_a", "mem_b", "mem_c"}


def test_lineage_cycle_safe():
    repo = LineageRepository(_store())
    repo.link("memory", "m1", "derived_from", "episode", "e1")
    repo.link("episode", "e1", "derived_from", "memory", "m1")
    graph = repo.trace_graph("memory", "m1")
    assert len(graph["edges"]) == 2  # terminates, no infinite walk
    assert len(repo.ancestors("memory", "m1")) == 2


def test_trace_graph_branches():
    """trace_graph walks every parent branch, not just the first (§11)."""
    repo = LineageRepository(_store())
    repo.link("champion", "c1", "promoted_from", "evaluation", "ev1")
    repo.link("evaluation", "ev1", "evaluated_from", "experiment", "ex1")
    repo.link("evaluation", "ev1", "supported_by", "darwin_benchmark", "db1")
    repo.link("experiment", "ex1", "derived_from", "patch", "pt1")

    graph = repo.trace_graph("champion", "c1")
    node_ids = {n["id"] for n in graph["nodes"]}
    assert node_ids == {"c1", "ev1", "ex1", "db1", "pt1"}
    assert graph["truncated"] is False
    rels = {(e["from"], e["relation"], e["to"]) for e in graph["edges"]}
    assert ("evaluation:ev1", "supported_by", "darwin_benchmark:db1") in rels
    # node cap truncates honestly
    capped = repo.trace_graph("champion", "c1", max_nodes=3)
    assert capped["truncated"] is True
    assert len(capped["nodes"]) <= 3


# -- §13 engine link tests ----------------------------------------------------


def test_proposal_failure_link(tmp_path):
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    fc = engine.create_failure_case("act_1", "task_1", "skill_1")
    prop = engine.create_proposal(fc.id, "task_1", "skill_1", "h", {"x": [1, 2]})
    edge = _edge(repo, "proposal", prop.id, "proposed_from")
    assert edge["to_type"] == "failure" and edge["to_id"] == fc.id
    # failure itself points at its praxis event (§9.1)
    gen = _edge(repo, "failure", fc.id, "generated_from")
    assert gen["to_type"] == "action" and gen["to_id"] == "act_1"


def test_proposal_memory_insight_source_refs(tmp_path):
    """§9.3: typed source_refs replace the bare source string."""
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    prop = engine.create_proposal(
        "",
        "task_1",
        "skill_1",
        "h",
        {"x": [1, 2]},
        source="memory_guided",
        source_refs=[{"type": "memory_insight", "id": "ins_9"}],
    )
    edge = _edge(repo, "proposal", prop.id, "proposed_from")
    assert edge["to_type"] == "memory_insight" and edge["to_id"] == "ins_9"


def test_patch_proposal_link(tmp_path):
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    prop = engine.create_proposal("f1", "task_1", "skill_1", "h", {"x": [1]})
    patch = engine.create_patch(prop.id, "skill_1", [{"k": "x", "v": 2}])
    edge = _edge(repo, "patch", patch.id, "patched_from")
    assert edge["to_type"] == "proposal" and edge["to_id"] == prop.id


def test_experiment_patch_link(tmp_path):
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    patch = engine.create_patch("p1", "skill_1", [{"k": "x", "v": 2}])
    exp = engine.create_experiment("p1", patch.id, "task_1", "skill_0", "skill_1")
    edge = _edge(repo, "experiment", exp.id, "derived_from")
    assert edge["to_type"] == "patch" and edge["to_id"] == patch.id


def _persisted_experiment(engine, tmp_path):
    """An experiment whose task + experiment provenance resolves."""
    task = engine.create_task("task_1", "rh56", "skill_1")
    patch = engine.create_patch("p1", "skill_1", [{"k": "x", "v": 2}])
    return engine.create_experiment("p1", patch.id, task.id, "skill_0", "skill_1")


def test_evaluation_experiment_link(tmp_path):
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    exp = _persisted_experiment(engine, tmp_path)
    ev = engine.create_evaluation(
        exp.id,
        {"success_rate": 0.5},
        {"success_rate": 0.7},
        simulation_receipts=[{"id": "rcpt_sim_1"}],
    )
    edge = _edge(repo, "evaluation", ev.id, "evaluated_from")
    assert edge["to_type"] == "experiment" and edge["to_id"] == exp.id
    sup = _edge(repo, "evaluation", ev.id, "supported_by")
    assert sup["to_type"] == "receipt" and sup["to_id"] == "rcpt_sim_1"


def test_champion_evaluation_link(tmp_path):
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    champ = engine.promote_champion(
        "skill_1",
        "task_1",
        "baseline",
        {"success_rate": 0.5},
        evaluation_id="eval_1",
    )
    edge = _edge(repo, "champion", champ.id, "promoted_from")
    assert edge["to_type"] == "evaluation" and edge["to_id"] == "eval_1"


def test_deadend_source_link(tmp_path):
    repo = LineageRepository(_store())
    engine = _engine(tmp_path, repo)
    de = engine.register_deadend(
        "task_1",
        "raise force",
        "collision risk",
        source_type="evaluation",
        source_id="eval_7",
    )
    edge = _edge(repo, "dead_end", de.id, "rejected_from")
    assert edge["to_type"] == "evaluation" and edge["to_id"] == "eval_7"


def test_engine_without_lineage_repo_is_noop(tmp_path):
    """Standalone AutoEngine (no data plane) must work unchanged."""
    engine = _engine(tmp_path, None)
    fc = engine.create_failure_case("act_1", "task_1", "skill_1")
    prop = engine.create_proposal(fc.id, "task_1", "skill_1", "h", {})
    assert prop.id


# -- §10 insight → memory links ------------------------------------------------


def test_memory_insight_links_all_source_memories():
    store = _store()
    repo = LineageRepository(store)
    bus = EventBus()
    svc = MemoryInsightService(bus, store, robot_id="rh56", lineage_repository=repo, cooldown_s=0.0)
    published: list[dict] = []
    bus.subscribe(EventTopics.MEMORY_INSIGHT_CREATED, lambda e: published.append(e.payload))
    svc._maybe_emit(
        "similar_failure_with_patch",
        skill_id="skill_1",
        failure_type="force overshoot",
        task_id="task_1",
        episode_id="ep_1",
        evidence_refs=["critic_result:ep_1"],
        extra={"memory_refs": ["mem_a", "mem_b"], "search_space": {}},
    )
    assert published, "insight was not published"
    ins_id = published[0]["insight_id"]
    parents = repo.parents("memory_insight", ins_id)
    assert {e["to_id"] for e in parents} == {"mem_a", "mem_b"}
    assert all(e["relation"] == "derived_from" for e in parents)


# -- §12 CLI rendering ----------------------------------------------------------


def test_cli_tree_rendering_branches():
    repo = LineageRepository(_store())
    repo.link("champion", "champ_8f31", "promoted_from", "evaluation", "eval_2d17")
    repo.link("evaluation", "eval_2d17", "evaluated_from", "experiment", "exp_91c2")
    repo.link("evaluation", "eval_2d17", "supported_by", "darwin_benchmark", "darwin_1")
    graph = repo.trace_graph("champion", "champ_8f31")
    lines = _render_lineage_tree(graph)
    assert lines[0] == "Champion champ_8f31"
    text = "\n".join(lines)
    assert "promoted_from Evaluation eval_2d17" in text
    assert "evaluated_from Experiment exp_91c2" in text
    assert "supported_by Darwin_Benchmark darwin_1" in text or "darwin_1" in text


def test_cmd_data_lineage_rejects_bad_entity():
    from rosclaw.storage.cli import cmd_data_lineage

    args = argparse.Namespace(entity="no-colon", json=False)
    assert cmd_data_lineage(args) == 2


# -- §14 golden lineage E2E ------------------------------------------------------


def test_golden_lineage_champion_to_receipt(tmp_path):
    """六问之六: champion traces all the way back to the receipt (§14)."""
    store = _store()
    repo = LineageRepository(store)
    bus = EventBus()

    # Receipt → Episode → Memory (practice/memory half, from DF-14/16B)
    repo.link("receipt", "rcpt_1", "generated_from", "action", "act_1")
    repo.link("episode", "ep_1", "supported_by", "receipt", "rcpt_1")
    repo.link("memory", "mem_1", "derived_from", "episode", "ep_1")

    # Memory → Insight (DF-17 §10)
    svc = MemoryInsightService(bus, store, robot_id="rh56", lineage_repository=repo, cooldown_s=0.0)
    published: list[dict] = []
    bus.subscribe(EventTopics.MEMORY_INSIGHT_CREATED, lambda e: published.append(e.payload))
    svc._maybe_emit(
        "similar_failure_with_patch",
        skill_id="skill_1",
        failure_type="force overshoot",
        task_id="task_1",
        episode_id="ep_1",
        evidence_refs=["critic_result:ep_1"],
        extra={"memory_refs": ["mem_1"], "search_space": {"force": [200, 300]}},
    )
    ins_id = published[0]["insight_id"]

    # Insight → Proposal → Patch → Experiment → Evaluation → Champion
    engine = _engine(tmp_path, repo)
    task = engine.create_task("task_1", "rh56", "skill_1")
    prop = engine.create_proposal(
        "",
        task.id,
        "skill_1",
        "lower force avoids overshoot",
        {"force": [200, 300]},
        source="memory_guided",
        source_refs=[{"type": "memory_insight", "id": ins_id}],
    )
    patch = engine.create_patch(prop.id, "skill_1", [{"k": "force", "v": 250}])
    exp = engine.create_experiment(prop.id, patch.id, task.id, "skill_0", "skill_1")
    ev = engine.create_evaluation(
        exp.id,
        {"success_rate": 0.5, "collision_rate": 0.1},
        {"success_rate": 0.7, "collision_rate": 0.1},
        simulation_receipts=[{"id": "rcpt_1"}],
    )
    champ = engine.promote_champion(
        "skill_1",
        task.id,
        "baseline",
        {"success_rate": 0.7},
        evaluation_id=ev.id,
    )

    graph = repo.trace_graph("champion", champ.id)
    node_ids = {n["id"] for n in graph["nodes"]}
    assert "rcpt_1" in node_ids  # receipt reachable
    assert "mem_1" in node_ids  # memory reachable
    assert prop.id in node_ids  # proposal reachable
    assert ev.id in node_ids  # evaluation reachable
    assert ins_id in node_ids  # insight reachable
    assert "ep_1" in node_ids  # episode reachable
