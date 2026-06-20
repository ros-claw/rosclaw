"""End-to-end integration evaluation of rosclaw + rosclaw-know + rosclaw-how.

These tests require a running rosclaw-how service (default http://127.0.0.1:47820).
They are skipped automatically when the service is unreachable so CI without the
private service still passes.

The eval exercises the full Runtime-facing path:

    rosclaw.how.client.HowClient
        -> POST /wiki/v1/prompt/build
        -> rosclaw-how state_router + semantic_router + topic_group filter
        -> recovery hint

and the knowledge path:

    rosclaw.know.interface.KnowledgeInterface(use_rosclaw_know_registry=True)
        -> rosclaw_know curated registry + baseline patterns
        -> symptom match

The test documents current capabilities and known gaps so iteration work is
grounded in observable pass/fail signals rather than manual curl probes.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from rosclaw.how.client import HowClient
from rosclaw.know.interface import KnowledgeInterface

DEFAULT_HOW_URL = os.environ.get("ROSCLAW_HOW_ENDPOINT", "http://127.0.0.1:47820")

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("_rosclaw_how_service")]


# Canonical failure descriptions drawn from rosclaw-know routing_canary.json.
# Each tuple is (query, expected_pattern_id, expected_strategy, notes).
# expected_strategy may be "SAFETY" for hard-coded physical-safety symptoms or
# "CATALYST" for semantic-retrieval symptoms.
HOW_TEST_CASES: list[tuple[str, str, str, str]] = [
    (
        "Actuator torque saturation at 237 N·m during PIDTuning",
        "SAFETY",
        "SAFETY",
        "hard-coded physical-safety path",
    ),
    (
        "NaN in weights after optimizer step",
        "Numerical_Instability",
        "SAFETY",
        "hard-coded numerical-instability safety path",
    ),
    (
        "battery capacity fade lithium plating li-ion fast charging cc-cv "
        "current taper c rate 4c thermal",
        "multi_stage_cc_cv_fast_charging",
        "CATALYST",
        "curated cluster in battery-and-energy-management",
    ),
    (
        "crypto throughput bottleneck aes aes-ni aes128 throughput mb/s cipher "
        "encryption decryption simd avx sse",
        "simd_aes_ni_hardware_crypto",
        "CATALYST",
        "curated cluster in hardware-accelerated-cryptography",
    ),
    (
        "combinatorial local optimum job-shop scheduling makespan tabu search "
        "simulated annealing genetic algorithm metaheuristic",
        "metaheuristic_combinatorial_escape",
        "CATALYST",
        "curated cluster in scheduling-optimization",
    ),
    (
        "image motion blur rolling shutter imu deblur drone quadrotor exposure",
        "motion_blur_imu_aided_deblur",
        "CATALYST",
        "curated cluster in motion-blur-deblur",
    ),
    (
        "entropy collapse ppo policy value loss degenerate episodes kl exploration",
        "ppo_entropy_collapse_guard",
        "CATALYST",
        "curated cluster in rl-training-stability",
    ),
    (
        "Gradient magnitude explodes during backprop causing weight overflow",
        "gradient_clipping",
        "CATALYST",
        "curated cluster in rl-training-stability",
    ),
]

# Known-gap cases: the correct curated cluster exists but some part of the
# pipeline (topic_group fingerprint, cluster similarity, etc.) currently blocks
# the match and /prompt/build ABSTAINs. Add cases here when a new gap is found;
# move them to HOW_TEST_CASES once the gap is closed.
HOW_KNOWN_GAPS: list[tuple[str, str, str]] = []


@pytest.mark.asyncio
async def test_how_client_end_to_end_recovery_hints() -> None:
    """Live how service returns actionable hints for canonical failures."""
    client = HowClient(DEFAULT_HOW_URL)
    await client.initialize()

    passed = []
    failed = []
    for query, expected_rule, expected_strategy, _notes in HOW_TEST_CASES:
        hint = await client.generate_recovery_hint(
            query,
            {"episode_id": "eval_e2e", "request_id": "eval_e2e"},
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=10,
        )
        if hint is None:
            failed.append((query, expected_rule, "no hint returned"))
            continue
        if hint["source"] != f"how_{expected_strategy.lower()}" or (
            expected_strategy != "SAFETY" and hint["rule_id"] != expected_rule
        ):
            failed.append(
                (query, expected_rule, f"got rule_id={hint['rule_id']} source={hint['source']}")
            )
            continue
        passed.append(query)

    assert not failed, f"HOW e2e failures: {failed}"
    assert len(passed) >= 4, f"expected at least 4 passing hints, got {len(passed)}"


@pytest.mark.asyncio
async def test_how_client_known_gap_xfail() -> None:
    """Document cases where the topic filter currently blocks the right cluster."""
    client = HowClient(DEFAULT_HOW_URL)
    await client.initialize()

    gaps: list[dict[str, Any]] = []
    for query, expected_rule, reason in HOW_KNOWN_GAPS:
        hint = await client.generate_recovery_hint(
            query,
            {"episode_id": "eval_gap", "request_id": "eval_gap"},
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=10,
        )
        gaps.append(
            {
                "query": query,
                "expected_rule": expected_rule,
                "hint": hint,
                "reason": reason,
            }
        )

    # We expect these to fail today. When the gap is fixed, this assertion will
    # fail and the test should be promoted into HOW_TEST_CASES.
    still_gapped = [g for g in gaps if g["hint"] is None]
    assert len(still_gapped) == len(HOW_KNOWN_GAPS), (
        f"Known gaps appear fixed; move them to HOW_TEST_CASES and remove xfail logic. gaps={gaps}"
    )


class TestRosclawKnowRegistry:
    """KnowledgeInterface with private rosclaw_know registry enabled."""

    def test_registry_enrichment_and_symptom_matching(self, tmp_path: Any) -> None:
        """Baseline + registry patterns together cover canonical robot failures."""
        ki = KnowledgeInterface(
            robot_id="eval",
            assets_path=str(tmp_path),
            use_rosclaw_know_registry=True,
        )
        ki._do_initialize()

        cases = [
            (
                "PID integral windup drives actuator into torque saturation",
                "Torque_Overflow",
            ),
            (
                "Gradient explode in loss after backprop",
                "Gradient_Explosion",
            ),
            (
                "CUDA out of memory in long rollout",
                "Memory_Exhaustion",
            ),
            (
                "open-loop plan tracks poorly when latency exceeds 50 ms",
                "Oscillation_Divergence",
            ),
        ]
        mismatches = []
        for query, expected_pattern in cases:
            match = ki.match_symptom(query)
            if match is None or match.get("pattern_id") != expected_pattern:
                mismatches.append(
                    (query, expected_pattern, match.get("pattern_id") if match else None)
                )

        ki._do_stop()
        assert not mismatches, f"KnowledgeInterface mismatches: {mismatches}"

    def test_registry_does_not_override_correct_baseline_for_joint_limit(
        self, tmp_path: Any
    ) -> None:
        """Regression guard: 'joint limit' previously matched a wrong-domain synth."""
        ki = KnowledgeInterface(
            robot_id="eval",
            assets_path=str(tmp_path),
            use_rosclaw_know_registry=True,
        )
        ki._do_initialize()

        match = ki.match_symptom("joint limit exceeded on wrist_3_link")
        ki._do_stop()

        assert match is not None
        # Accept either the baseline joint-limit heuristic or a curated cluster.
        assert match["domain"] in (
            "Control_Locomotion",
            "Planning_Decision",
        ), f"wrong-domain match: {match}"


class TestHowClientAgainstLocalService:
    """Low-level HTTP shape checks against a real rosclaw-how process."""

    @pytest.mark.asyncio
    async def test_prompt_build_returns_expected_response_shape(self) -> None:
        client = HowClient(DEFAULT_HOW_URL)
        await client.initialize()

        rule = await client.suggest_recovery(
            "Actuator torque saturation at 237 N·m during PIDTuning",
            {"episode_id": "shape_check"},
        )
        assert rule is not None
        assert rule["injected"] is True
        assert rule["priority"] >= 1
        assert rule["source"].startswith("how_")
        assert rule["action"]
