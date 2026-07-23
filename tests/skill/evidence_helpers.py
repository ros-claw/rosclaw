"""Build real, replayed MuJoCo evidence for skill lifecycle tests."""

from __future__ import annotations

import json

import yaml

from rosclaw.darwin.physics_runner import PairedTrajectoryCase, PhysicsDarwinRunner
from rosclaw.sandbox.backends import ScenarioSpec
from rosclaw.skill.eval import refresh_hashes
from rosclaw.skill.hash import compute_candidate_evidence_hash
from rosclaw.skill.models import SkillPackage

HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
COLLISION = [
    3.4426358094526863,
    -0.7680767522686045,
    2.253070730803216,
    2.480201653011009,
    -5.099721659051599,
    5.976851207161098,
]


def write_promotion_evidence(pkg: SkillPackage, candidate_id: str) -> None:
    receipts = pkg.root / "evidence" / "receipts"
    reports = pkg.root / "evidence" / "reports"
    rollout_root = pkg.root / "evidence" / "rollouts" / candidate_id
    receipts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    candidate_hash = compute_candidate_evidence_hash(pkg.root, candidate_id)

    cases = [
        PairedTrajectoryCase(
            scenario=ScenarioSpec(
                scenario_id=f"{candidate_id}-{seed}",
                robot_id="ur5e",
                world_id="tabletop",
                body_snapshot_hash="resolved-by-runner",
                model_hash="resolved-by-runner",
                seed=seed,
                metadata={
                    "initial_qpos_jitter_rad": 0.002,
                    "skill_candidate_id": candidate_id,
                    "skill_candidate_hash": candidate_hash,
                },
            ),
            baseline_trajectory=[COLLISION, HOME],
            candidate_trajectory=[HOME],
        )
        for seed in (7, 8)
    ]
    result = PhysicsDarwinRunner().run(cases, artifact_root=rollout_root)
    candidate_receipt = next(
        receipt
        for receipt in result.simulation_receipts
        if receipt["evaluation_variant"] == "candidate"
    )
    (receipts / f"{candidate_id}.json").write_text(
        json.dumps(candidate_receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = {
        "schema_version": "rosclaw.physics_darwin_report.v1",
        "evidence_domain": "SIMULATION",
        "physics_executed": True,
        **result.to_dict(),
    }
    (reports / f"{candidate_id}_darwin.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    darwin_path = pkg.root / "darwin_eval.yaml"
    darwin = yaml.safe_load(darwin_path.read_text(encoding="utf-8"))
    darwin["metrics"] = {
        "success_rate": {"required": True, "promote_threshold": 0.75},
        "collision_rate": {"required": True, "max_allowed": 0.0},
    }
    darwin_path.write_text(yaml.safe_dump(darwin, sort_keys=False), encoding="utf-8")
    refresh_hashes(pkg)
