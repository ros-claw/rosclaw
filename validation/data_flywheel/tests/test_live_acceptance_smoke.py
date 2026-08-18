"""DF-20 (phase-II §27-§28): live acceptance smoke — fast CI-safe version.

A 3-episode pass through the whole plane.  Retrieval runs against the
real embedded SeekDB engine when pyseekdb is installed (the CI
data-flywheel profile installs it); otherwise that lane is honestly
reported as skipped and not asserted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "memory" / "regime"))
sys.path.insert(0, str(REPO_ROOT / "validation" / "data_flywheel" / "scripts"))

from run_live_acceptance import run_loop  # noqa: E402


def _args(workdir: Path, episodes: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        episodes=episodes,
        workdir=str(workdir),
        session_seconds=0.1,
        reconcile_every=2,
        soak_duration_sec=0.0,
        no_retrieval=False,
    )


def test_live_loop_smoke(tmp_path):
    result = run_loop(_args(tmp_path))
    assert result["episodes_completed"] == 3
    assert result["memory"]["memory_items"] >= 1
    assert result["memory"]["duplicate_rate"] == 0
    assert result["memory"]["untraceable_rate"] == 0.0
    assert result["data_quality"]["bad_evidence_write_rate"] == 0.0
    assert result["how"]["lookup_ok"] is True
    assert result["insight"]["published"] >= 1
    assert result["insight"]["lineage_linked"] is True
    assert result["evolution"]["records"] >= 1
    assert result["lineage"]["proposal_reaches_insight"] is True
    assert result["lineage"]["proposal_reaches_memory"] is True
    if result["projection"].get("retrieval_skipped") is None:
        assert result["projection"]["final_lag"] == 0
        assert "query" in result["retrieval"]
