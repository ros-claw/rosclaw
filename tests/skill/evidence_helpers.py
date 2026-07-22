"""Helpers for constructing explicit promotion evidence in skill tests."""

from __future__ import annotations

import hashlib
import json

from rosclaw.skill.eval import refresh_hashes
from rosclaw.skill.models import SkillPackage


def write_promotion_evidence(pkg: SkillPackage, candidate_id: str) -> None:
    receipts = pkg.root / "evidence" / "receipts"
    reports = pkg.root / "evidence" / "reports"
    receipts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    trajectory = receipts / f"{candidate_id}_trajectory.json"
    trajectory.write_text('{"physics_executed":true}\n', encoding="utf-8")
    trajectory_hash = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    receipt = {
        "execution_mode": "SIMULATION",
        "evidence_domain": "SIMULATION",
        "evidence_level": "TASK_VERIFIED",
        "body_snapshot_hash": "sha256:test-body",
        "dispatch_result": {"physics_executed": True},
        "simulation_result": {
            "seed": 7,
            "has_physics": True,
            "physics_executed": True,
            "model_hash": "sha256:test-model",
            "action_hash": f"sha256:{candidate_id}",
            "artifact_hashes": {trajectory.name: trajectory_hash},
        },
        "verification_result": {
            "data_quality": {
                "artifact_hash_valid": True,
                "body_snapshot_match": True,
                "replayable": True,
            }
        },
        "replay": {"verified": True, "environment_match": True, "hashes_verified": True},
        "artifacts": [f"evidence/receipts/{trajectory.name}"],
    }
    (receipts / f"{candidate_id}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    darwin = {
        "evidence_domain": "SIMULATION",
        "physics_executed": True,
        "candidate_metrics": {
            "success_rate": 0.85,
            "no_fall_rate": 1.0,
            "sandbox_block_rate": 0.08,
            "recovery_success_rate": 0.67,
        },
        "per_seed": {
            "7": {"baseline": {"success_rate": 0.70}, "candidate": {"success_rate": 0.84}},
            "8": {"baseline": {"success_rate": 0.72}, "candidate": {"success_rate": 0.86}},
        },
        "regression": {"passed": True, "critical_regressions": []},
    }
    (reports / f"{candidate_id}_darwin.json").write_text(
        json.dumps(darwin, indent=2), encoding="utf-8"
    )
    refresh_hashes(pkg)
