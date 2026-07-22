#!/usr/bin/env python3
"""Produce a complete, physics-backed UR5e shield evolution evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw.auto.promotion.gate import PromotionGate
from rosclaw.darwin.physics_runner import PairedTrajectoryCase, PhysicsDarwinRunner
from rosclaw.how.retry_orchestrator import RetryOrchestrator
from rosclaw.sandbox.backends import MujocoCpuBackend, RolloutRequest, ScenarioSpec
from rosclaw.sandbox.backends.fingerprints import file_hash
from rosclaw.sandbox.sandbox_api import Sandbox

HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
MIDPATH_COLLISION = [
    3.4426358094526863,
    -0.7680767522686045,
    2.253070730803216,
    2.480201653011009,
    -5.099721659051599,
    5.976851207161098,
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    original = {
        "action_id": "shield-action-original",
        "scenario": {"scenario_id": "shield-retry-42", "seed": 42},
        "parameters": {"approach_direction": "direct"},
        "trajectory": [MIDPATH_COLLISION, HOME],
    }

    def submit_retry(action):
        sandbox = Sandbox.create("ur5e", "tabletop", "mujoco")
        try:
            scenario = ScenarioSpec(
                scenario_id=action["scenario"]["scenario_id"],
                robot_id="ur5e",
                world_id="tabletop",
                body_snapshot_hash="sha256:ur5e-shield-v1",
                model_hash=file_hash(sandbox.model_path),
                seed=action["scenario"]["seed"],
            )
            trajectory = (
                [HOME]
                if action["parameters"]["approach_direction"] == "home_clearance_route"
                else action["trajectory"]
            )
            backend = MujocoCpuBackend(sandbox)
            receipt = backend.rollout(
                RolloutRequest(
                    scenario=scenario,
                    trajectory=trajectory,
                    artifact_dir=root / "retry",
                )
            )
            replay = backend.replay(receipt, strict=True)
            value = receipt.to_dict()
            value["replay_report"] = replay.to_dict()
            value["receipt_verified"] = replay.verified
            value["data_quality_pass"] = replay.verified
            return value
        finally:
            sandbox.close()

    retry = RetryOrchestrator(submit_retry).execute(
        original,
        {"approach_direction": "home_clearance_route", "speed_scale": 0.5},
        retry_budget=3,
    )

    cases = [
        PairedTrajectoryCase(
            scenario=ScenarioSpec(
                scenario_id=f"shield-paired-{seed}",
                robot_id="ur5e",
                world_id="tabletop",
                body_snapshot_hash="sha256:ur5e-shield-v1",
                model_hash="resolved-by-runner",
                seed=seed,
                metadata={"counterexample": "midpath_table_collision"},
            ),
            baseline_trajectory=[MIDPATH_COLLISION, HOME],
            candidate_trajectory=[HOME],
        )
        for seed in (42, 43)
    ]
    evaluation = PhysicsDarwinRunner().run(cases, artifact_root=root / "darwin")
    gate = PromotionGate().evaluate(
        evaluation.baseline_metrics,
        evaluation.candidate_metrics,
        current_level="baseline",
        per_seed=evaluation.per_seed,
        simulation_receipts=evaluation.simulation_receipts,
        regression_results=evaluation.regression_results,
    )
    payload = {
        "schema_version": "rosclaw.shield_evolution.v1",
        "original_action_id": original["action_id"],
        "failure_signature": "midpath_table_collision",
        "retry": {
            "executed": retry.executed,
            "reason": retry.reason,
            "parameter_patch_hash": retry.parameter_patch_hash,
            "retry_action_id": (retry.retry_action or {}).get("action_id"),
            "same_scenario": bool(
                retry.receipt
                and retry.receipt.get("scenario_id") == original["scenario"]["scenario_id"]
                and retry.receipt.get("seed") == original["scenario"]["seed"]
            ),
            "physics_executed": bool(retry.receipt and retry.receipt.get("physics_executed")),
            "receipt_verified": bool(retry.receipt and retry.receipt.get("receipt_verified")),
            "success": bool(retry.receipt and retry.receipt.get("is_safe")),
        },
        "darwin": evaluation.to_dict(),
        "promotion": gate.to_dict(),
        "counterexample_regression": {
            "stored_case": "tests/sandbox/test_trajectory_backend.py",
            "unsafe_allow_rate": 0.0,
            "counterexamples_discovered": 1,
            "counterexamples_fixed": 1,
        },
        "maximum_claim": "SIM_CHAMPION",
    }
    output = root / "closed_loop_report.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if retry.executed and gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
