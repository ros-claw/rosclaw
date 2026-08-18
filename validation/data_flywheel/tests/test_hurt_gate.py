"""DF-20 (phase-II §31): Memory Hurt Gate — the retrieval must not make
the robot worse.

Runs the five-lane comparison (No Memory / Keyword / Vector / Hybrid /
Hybrid + Body/Regime) over the regime fixture corpus and asserts the P0
gate: Hybrid+Regime hurt ≤ 5%, unsafe = 0, not worse than No Memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "memory" / "regime"))
sys.path.insert(0, str(REPO_ROOT / "validation" / "data_flywheel" / "scripts"))

from run_live_acceptance import run_hurt_gate  # noqa: E402


def _report():
    return run_hurt_gate(argparse.Namespace(real_seekdb=False))


def test_hurt_gate_p0_passes():
    report = _report()
    regime = report["lanes"]["hybrid_regime"]
    assert regime["memory_hurt_rate"] <= 0.05, report
    assert regime["unsafe_intervention_rate"] == 0.0, report
    assert regime["success_rate"] >= report["lanes"]["no_memory"]["success_rate"]
    assert report["passed"] is True


def test_hurt_gate_ungated_lanes_actually_hurt():
    """The gate is meaningful: ungated lanes must demonstrably hurt.

    If keyword/hybrid ever stop hurting, either the corpus lost its
    counter-regime traps or the lane wiring broke — the gate would be
    asserting nothing.
    """
    report = _report()
    assert report["lanes"]["keyword"]["memory_hurt_rate"] > 0.2, report
    assert report["lanes"]["hybrid"]["memory_hurt_rate"] > 0.2, report


def test_hurt_gate_is_deterministic():
    a, b = _report(), _report()
    assert a["lanes"] == b["lanes"]
