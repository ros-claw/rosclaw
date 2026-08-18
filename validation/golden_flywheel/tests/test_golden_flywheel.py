"""DF-25 (phase-II §42-§44): Golden Flywheel — the final Phase II acceptance.

A real MuJoCo gripper-lift task drives the whole data flywheel with real
components end to end:

    Round 1  gripper_force too low -> grasp slip -> Receipt R1 + Practice P1
             -> auto-distilled Failure Memory M1
    Recovery How rule "increase grip force" -> retry SUCCESS -> Receipt R2
             -> Intervention Memory M2 (verified by the session fact-verify)
    Round 2  similar slip -> retrieval hits M2 -> historical recovery applied
    Insight  repeated failures -> MemoryInsight auto-emitted
    Evolution memory-guided Proposal -> Patch -> Experiment (real sim A/B)
    Darwin   independent multi-seed benchmark on real physics
    Promotion Champion v2 through the real PromotionGate authorization

§44's twelve acceptance criteria are asserted one by one, the §43 lineage
tree shape is checked against `rosclaw data lineage`, and the whole demo is
run twice to prove determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco", reason="golden flywheel needs real MuJoCo physics")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "validation" / "golden_flywheel" / "scripts"))

from run_golden_flywheel import run_golden_flywheel  # noqa: E402

# §44 — the twelve acceptance criteria, verbatim order.
TWELVE_CRITERIA = [
    "receipt_is_real_execution_receipt",
    "practice_is_real_recorder_data",
    "memory_is_auto_distilled",
    "memory_has_evidence",
    "retrieval_hit_historical_memory",
    "recovery_passed_verifier",
    "insight_auto_generated",
    "proposal_has_provenance",
    "experiment_has_lineage",
    "darwin_is_independent_evaluation",
    "champion_has_promotion_gate",
    "champion_traces_to_receipt",
]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("golden_flywheel")
    return run_golden_flywheel(workdir)


def test_real_physics_drove_the_story(demo):
    r1 = demo["round1"]["sim"]
    assert r1["physics_executed"] is True
    assert r1["success"] is False
    assert r1["slip_observed"] is True
    assert demo["recovery"]["sim"]["success"] is True
    assert demo["round2"]["sim"]["success"] is True
    darwin = demo["darwin"]
    assert darwin["candidate_metrics"]["success_rate"] > (
        darwin["baseline_metrics"]["success_rate"]
    )


def test_twelve_acceptance_criteria(demo):
    criteria = demo["criteria"]
    for name in TWELVE_CRITERIA:
        assert criteria.get(name) is True, f"§44 criterion failed: {name}"


def test_lineage_tree_matches_spec_section43(demo):
    tree = demo["lineage_tree"]
    champ = demo["champion"]["id"]
    evo = demo["evolution"]
    # §43: champion at root, then the full chain down to both receipts.
    # Labels render as type.title() + id ("Memory_Insight ins_…").
    assert champ in tree
    for token in (
        "Evaluation",
        "Darwin_Benchmark",
        "Experiment",
        "Patch",
        "Proposal",
        "Memory_Insight",
    ):
        assert token in tree, f"lineage tree missing {token}"
    assert demo["round1"]["receipt_id"] in tree
    assert demo["recovery"]["receipt_id"] in tree
    assert evo["proposal_id"] in tree
    assert evo["evaluation_id"] in tree

    # Graph-level checks (orientation child -> parent; edges are "type:id").
    graph = demo["lineage_graph"]
    edges = set()
    for e in graph["edges"]:
        edges.add((e["from"].split(":", 1)[1], e["relation"], e["to"].split(":", 1)[1]))
    assert (champ, "promoted_from", evo["evaluation_id"]) in edges
    assert (evo["evaluation_id"], "evaluated_from", evo["experiment_id"]) in edges
    assert (evo["experiment_id"], "derived_from", evo["patch_id"]) in edges
    assert (evo["patch_id"], "patched_from", evo["proposal_id"]) in edges
    assert (evo["proposal_id"], "proposed_from", demo["insight"]["id"]) in edges
    assert (
        demo["insight"]["id"],
        "derived_from",
        demo["memories"]["intervention"],
    ) in edges
    darwin_edges = [e for e in edges if e[0] == evo["evaluation_id"] and e[1] == "supported_by"]
    assert darwin_edges, "evaluation must be supported_by the darwin benchmark receipt"


def test_cli_lineage_command(demo):
    """§43 verbatim: `rosclaw data lineage champion:<id>` prints the tree."""
    cli_tree = demo["cli_lineage_tree"]
    assert demo["champion"]["id"] in cli_tree
    assert "Proposal" in cli_tree
    assert "Memory_Insight" in cli_tree


def test_determinism(demo, tmp_path):
    again = run_golden_flywheel(tmp_path / "second_run")
    assert all(again["criteria"].values())
    # Same seeds -> identical real-physics benchmark metrics.
    assert again["darwin"]["baseline_metrics"] == demo["darwin"]["baseline_metrics"]
    assert again["darwin"]["candidate_metrics"] == demo["darwin"]["candidate_metrics"]


def test_promotion_gate_blocks_unauthorized(demo, tmp_path):
    """The promotion gate is real: bypassing evaluation authorization fails."""
    from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine

    engine: AutoEngine = demo["engine"]
    with pytest.raises(ValueError, match="PROMOTION_EVALUATION_AUTHORIZATION_REQUIRED"):
        engine.promote_champion(
            skill_id="golden_grasp_v3_unauthorized",
            task_id=demo["task_id"],
            level="sim",
            metrics={"success_rate": 0.99},
            parent_skill="golden_grasp",
            patch_id="patch_nonexistent",
            experiment_id="exp_nonexistent",
            evaluation_id="eval_nonexistent",
        )
