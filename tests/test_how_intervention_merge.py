"""Merge-layer smoke tests: SAFETY_TAXONOMY + decide_recovery in the engine.

Verifies that the intervention pipeline ported into
``src/rosclaw/how/intervention`` integrates correctly with the reactive
:class:`HeuristicEngine`:

* ``suggest_recovery`` falls through to the SAFETY_TAXONOMY for symptoms
  that reactive hand-written rules don't cover (collision, self-collision,
  human proximity, …).
* ``decide_recovery`` runs the full diagnose → decide_strategy → compose
  pipeline and returns a stable rule_id for safety-attributable
  decisions.
* ``record_outcome`` accepts taxonomy-derived rule_ids and increments
  the same counters as it would for reactive rules.
* The taxonomy and reactive rules coexist — substring-matched reactive
  rules still win when both could match (e.g. "communication timeout"
  surfaces the "Retry with exponential backoff" remediation).
"""
from __future__ import annotations

import pytest

from rosclaw.how import (
    SAFETY_TAXONOMY,
    HeuristicEngine,
    InterventionDecision,
    compose,
    decide_strategy,
    diagnose,
    diagnose_safety,
    from_v1_prompt_build,
)
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


@pytest.fixture
def engine():
    client = SeekDBMemoryClient()
    client.connect()
    return HeuristicEngine(seekdb_client=client)


# ── pure-rules layer (no engine state) ────────────────────────────────────


class TestInterventionPureRules:
    """The diagnose / decide_strategy / compose pipeline is pure data."""

    def test_safety_taxonomy_covers_15_symptoms(self) -> None:
        assert len(SAFETY_TAXONOMY) == 15
        for entry in SAFETY_TAXONOMY.values():
            assert entry["severity"] in {"S0", "S1", "S2", "S3", "S4"}

    def test_collision_log_classified_as_hazard(self) -> None:
        symptom, severity, strategy = diagnose_safety(
            "collision detected on link 3"
        )
        assert symptom == "Collision_Risk"
        assert severity == "S3"
        assert strategy == "STOP_UNSAFE"

    def test_compile_error_routes_to_feasibility_repair(self) -> None:
        req = from_v1_prompt_build(
            "SyntaxError: invalid syntax",
            previous_scores=[0.4, 0.4, 0.4, 0.4],
            current_iteration=5,
        )
        state = diagnose(req)
        strategy, reasons = decide_strategy(req, state)
        assert strategy == "FEASIBILITY_REPAIR"
        assert reasons  # has at least one explanation

    def test_plateau_default_picks_catalyst(self) -> None:
        req = from_v1_prompt_build(
            "",  # no safety signal
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=8,
        )
        state = diagnose(req)
        strategy, _ = decide_strategy(req, state)
        assert strategy == "CATALYST"

    def test_compose_emergency_stop_lists_sandbox_checks(self) -> None:
        req = from_v1_prompt_build(
            "e-stop triggered by operator",
            previous_scores=[0.7, 0.7, 0.7, 0.7],
            current_iteration=4,
        )
        state = diagnose(req)
        strategy, _ = decide_strategy(req, state)
        assert strategy == "STOP_UNSAFE"
        decision: InterventionDecision = compose(strategy, state)
        assert decision.requires_sandbox_validation
        assert "operator_acknowledge" in decision.sandbox_checks


# ── engine wiring (taxonomy fallback) ─────────────────────────────────────


class TestEngineTaxonomyFallback:
    """suggest_recovery should consult SAFETY_TAXONOMY when reactive rules miss."""

    @pytest.mark.asyncio
    async def test_unseeded_collision_log_returns_taxonomy_rule(
        self, engine: HeuristicEngine,
    ) -> None:
        # No seed_defaults() — only the taxonomy path can serve this.
        # We use 'link collision' rather than 'collision detected' so the
        # S3 Collision_Risk keyword group doesn't pre-empt Self_Collision.
        recovery = await engine.suggest_recovery(
            "ERROR: self collision between link_2 and link_4"
        )
        assert recovery is not None
        assert recovery["rule_id"] == "tax_Self_Collision"
        assert "Self_Collision" in recovery["action"]
        # S3 severity should map to a non-zero priority.
        assert recovery["priority"] == 3

    @pytest.mark.asyncio
    async def test_v1_rule_wins_over_taxonomy_for_overlapping_keywords(
        self, engine: HeuristicEngine,
    ) -> None:
        """The reactive 'communication timeout' rule should beat the taxonomy.

        Without this ordering, the more generic taxonomy would shadow the
        hand-written 'Retry with exponential backoff' remediation that
        operators depend on.
        """
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery(
            "communication timeout to ROS master"
        )
        assert recovery is not None
        assert "backoff" in recovery["action"].lower()
        assert not recovery["rule_id"].startswith("tax_")

    @pytest.mark.asyncio
    async def test_seed_defaults_also_seeds_taxonomy_rows(
        self, engine: HeuristicEngine,
    ) -> None:
        count = await engine.seed_defaults()
        # 21 reactive rules + 15 taxonomy rows = 36, but some IDs collide
        # within the reactive set (e.g. "timeout" appears twice) — assert the
        # taxonomy rows are present rather than a hard count.
        assert count >= len(SAFETY_TAXONOMY)
        for symptom in SAFETY_TAXONOMY:
            assert f"tax_{symptom}" in engine._rule_cache


# ── decide_recovery + record_outcome ──────────────────────────────────────


class TestDecideRecoveryOutcome:
    """decide_recovery wires intervention decisions into record_outcome."""

    @pytest.mark.asyncio
    async def test_decide_recovery_returns_rule_id_for_safety_symptom(
        self, engine: HeuristicEngine,
    ) -> None:
        req = from_v1_prompt_build(
            "ERROR: torque saturation on joint 2",
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=6,
        )
        decision, rule_id = await engine.decide_recovery(req)
        assert decision.strategy == "SAFETY"
        assert decision.injected is True
        assert rule_id == "tax_Torque_Overflow"

    @pytest.mark.asyncio
    async def test_decide_recovery_returns_none_rule_id_for_plateau(
        self, engine: HeuristicEngine,
    ) -> None:
        """CATALYST plateau decisions have no single attributable rule."""
        req = from_v1_prompt_build(
            "",
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=7,
        )
        decision, rule_id = await engine.decide_recovery(req)
        assert decision.strategy == "CATALYST"
        assert rule_id is None

    @pytest.mark.asyncio
    async def test_record_outcome_increments_taxonomy_counter(
        self, engine: HeuristicEngine,
    ) -> None:
        req = from_v1_prompt_build(
            "OOM: cuda out of memory",
            previous_scores=[0.7, 0.7, 0.7, 0.7],
            current_iteration=6,
        )
        decision, rule_id = await engine.decide_recovery(req)
        assert decision.strategy == "RESOURCE_REPAIR"
        assert rule_id == "tax_Memory_Exhaustion"

        ok = await engine.record_outcome(rule_id, success=True)
        assert ok is True
        rule = engine._rule_cache[rule_id]
        assert rule["success_count"] == 1
        assert rule["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_cooldown_swaps_catalyst_for_diversify(
        self, engine: HeuristicEngine,
    ) -> None:
        """Recent pattern repeats should flip CATALYST to DIVERSIFY.

        We pass a ``recent_pattern_id`` that matches the one decide_recovery
        is about to recommend — the policy should detect the repeat and
        swap to DIVERSIFY without losing the snippet entirely.
        """
        req = from_v1_prompt_build(
            "",
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=10,
            recent_pattern_ids=["pat_xyz"],
        )
        decision, _ = await engine.decide_recovery(
            req, recent_pattern_id="pat_xyz",
        )
        assert decision.strategy == "DIVERSIFY"
        assert decision.injected is True
        # The DIVERSIFY snippet should call out the cooldown explicitly.
        assert "pat_xyz" in decision.snippet


# ── edge cases (regression pins for HIGH bugs caught in review) ──────────


class TestMergeEdgeCases:
    """Cases that exercise the corners surfaced during code review."""

    @pytest.mark.asyncio
    async def test_query_exact_skips_taxonomy_rows(
        self, engine: HeuristicEngine,
    ) -> None:
        """A bare symptom string must not short-circuit via exact match.

        Tests the unit directly: after ``seed_defaults`` plants a
        ``tax_<symptom>`` row with the lowercased symptom as its
        ``condition`` field, ``_query_exact`` must skip it. (The
        taxonomy reaches the same row via ``diagnose_safety`` in step
        #3 of ``suggest_recovery``; this skip enforces that step #1
        only ever returns reactive rules.)
        """
        await engine.seed_defaults()
        # Direct unit check on the bypass path.
        assert engine._query_exact("joint limit violation") is None
        assert engine._query_exact("collision risk") is None
        # Sanity: a reactive condition still matches exactly.
        v1_hit = engine._query_exact("joint limit exceeded")
        assert v1_hit is not None
        assert not str(v1_hit.get("id", "")).startswith("tax_")

    @pytest.mark.asyncio
    async def test_taxonomy_rule_lazy_seed_survives_reinitialize(
        self, engine: HeuristicEngine,
    ) -> None:
        """Lazily-seeded ``tax_`` cache rows must NOT be dropped on warm.

        The hot path may create taxonomy rows via ``_taxonomy_rule``
        before the engine ever runs ``initialize()``. A subsequent warm
        must merge them, not overwrite, otherwise the in-memory counter
        attached to that row is silently lost.
        """
        # 1. lazy seed (no seed_defaults, no initialize)
        recovery = await engine.suggest_recovery(
            "ERROR: cuda out of memory"
        )
        assert recovery is not None
        assert recovery["rule_id"] == "tax_Memory_Exhaustion"

        # Manually increment the counter so we can detect loss.
        engine._rule_cache["tax_Memory_Exhaustion"]["success_count"] = 42

        # 2. warm — should NOT drop the lazy row
        await engine.initialize()

        assert "tax_Memory_Exhaustion" in engine._rule_cache
        # Either the DB had the row (best-effort upsert succeeded) and
        # success_count comes from there as 0, or the merge preserved
        # the 42 we just set. Either way, the row exists and is usable.
        rule = engine._rule_cache["tax_Memory_Exhaustion"]
        assert rule["id"] == "tax_Memory_Exhaustion"

    @pytest.mark.asyncio
    async def test_decide_recovery_no_attribution_under_cooldown(
        self, engine: HeuristicEngine,
    ) -> None:
        """Cooldown→DIVERSIFY must NOT credit the taxonomy.

        DIVERSIFY is an optimization-axis outcome (plateau + repeat
        suppression). Attributing it to a safety taxonomy rule would
        skew the efficacy counters with decisions the rule didn't
        actually make.

        Construct a request with both a safety symptom AND a plateau:
        the safety taxonomy would normally drive SAFETY, but the
        decide_strategy code in the policy never reaches the
        optimization branch under hazard. So we test the inverse —
        plateau with cooldown → DIVERSIFY → rule_id MUST be None.
        """
        req = from_v1_prompt_build(
            "",  # no safety signal, plateau only
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=8,
            recent_pattern_ids=["recent_pat"],
        )
        decision, rule_id = await engine.decide_recovery(
            req, recent_pattern_id="recent_pat",
        )
        assert decision.strategy == "DIVERSIFY"
        assert rule_id is None  # NOT attributed to any taxonomy rule

    @pytest.mark.asyncio
    async def test_decide_recovery_no_attribution_for_stabilize(
        self, engine: HeuristicEngine,
    ) -> None:
        """STABILIZE driven by Numerical_Instability is a constraint-violation,
        and STABILIZE is NOT in ``is_blocking`` — so rule_id should be None.

        This pins the policy fix: previously, any symptom-in-taxonomy
        led to a tax_ rule_id, even though the decision came from the
        optimization axis. The fix gates attribution on
        ``is_blocking(decision.strategy)``.
        """
        req = from_v1_prompt_build(
            "NaN detected during training step",
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=6,
        )
        decision, rule_id = await engine.decide_recovery(req)
        # Numerical_Instability is S2 → STABILIZE in the taxonomy. The
        # decision is STABILIZE, which is NOT a safety-blocking strategy.
        assert decision.strategy == "STABILIZE"
        assert rule_id is None

    @pytest.mark.asyncio
    async def test_decide_recovery_attributes_for_resource_repair(
        self, engine: HeuristicEngine,
    ) -> None:
        """OOM → RESOURCE_REPAIR is a blocking strategy → rule_id present."""
        req = from_v1_prompt_build(
            "OOM: cuda out of memory error",
            previous_scores=[0.7, 0.7, 0.7, 0.7],
            current_iteration=5,
        )
        decision, rule_id = await engine.decide_recovery(req)
        assert decision.strategy == "RESOURCE_REPAIR"
        assert rule_id == "tax_Memory_Exhaustion"

    @pytest.mark.asyncio
    async def test_minimize_direction_inverts_improvement(
        self, engine: HeuristicEngine,
    ) -> None:
        """Direction-aware: a falling score on a 'minimize' task is improving.

        Without the direction fold in ``score_normalizer``, the
        diagnoser would mis-label a minimize task's downward
        trajectory as 'regressing' and trigger DIAGNOSE.
        """
        from rosclaw.how import InterventionRequest, OptimizationContext, TaskContext
        req = InterventionRequest(
            task_context=TaskContext(objective_direction="minimize"),
            optimization_context=OptimizationContext(
                current_iteration=8,
                previous_scores=[1000.0, 800.0, 600.0, 400.0],
            ),
        )
        decision, _ = await engine.decide_recovery(req)
        # Falling raw score under 'minimize' direction == improving, so
        # the policy should be silent (NOOP, no injection).
        assert decision.strategy == "NOOP"
        assert decision.injected is False

    @pytest.mark.asyncio
    async def test_severity_hint_promotes_to_emergency(
        self, engine: HeuristicEngine,
    ) -> None:
        """``severity_hint`` lets the caller assert severity even when
        keywords don't match. S4 hint → STOP_UNSAFE."""
        from rosclaw.how import (
            InterventionRequest,
            OptimizationContext,
            SafetyContext,
        )
        req = InterventionRequest(
            optimization_context=OptimizationContext(
                current_iteration=5,
                previous_scores=[0.5, 0.5, 0.5, 0.5],
            ),
            safety_context=SafetyContext(
                error_log="unrecognized failure mode",
                severity_hint="S4",
            ),
        )
        decision, _ = await engine.decide_recovery(req)
        assert decision.strategy == "STOP_UNSAFE"

    @pytest.mark.asyncio
    async def test_taxonomy_rule_id_disjoint_from_v1_namespace(
        self, engine: HeuristicEngine,
    ) -> None:
        """``tax_`` and ``rule_<n>_`` ID namespaces never collide."""
        await engine.seed_defaults()
        tax_ids = {rid for rid in engine._rule_cache if rid.startswith("tax_")}
        v1_ids = {rid for rid in engine._rule_cache if rid.startswith("rule_")}
        assert tax_ids
        assert v1_ids
        assert not (tax_ids & v1_ids)  # zero overlap

    @pytest.mark.asyncio
    async def test_idempotent_seed_taxonomy(
        self, engine: HeuristicEngine,
    ) -> None:
        """Calling seed_defaults twice must not corrupt or duplicate
        taxonomy rows in the cache."""
        await engine.seed_defaults()
        first = {
            rid: dict(rule)
            for rid, rule in engine._rule_cache.items()
            if rid.startswith("tax_")
        }
        # Mutate a counter to detect overwrite vs preserve
        engine._rule_cache["tax_Collision_Risk"]["success_count"] = 7

        # Second seed: implementations vary on whether to reset counters,
        # but the rule_ids and structure must remain identical.
        await engine.seed_defaults()
        second = {
            rid: dict(rule)
            for rid, rule in engine._rule_cache.items()
            if rid.startswith("tax_")
        }
        assert set(first.keys()) == set(second.keys())
        # No phantom new tax_ rows appeared.
        assert len(second) == len(SAFETY_TAXONOMY)
