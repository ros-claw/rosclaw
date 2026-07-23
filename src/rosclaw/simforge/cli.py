"""Lightweight CLI for CoreSimBench suite validation and ShieldReach evolution."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from rosclaw.simforge.holdout import HiddenHoldoutService, create_holdout_signing_key
from rosclaw.simforge.models import Partition, ScenarioDistributionSpec, SimForgeTaskSpec
from rosclaw.simforge.promotion_v3 import StatisticalGateV3
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.shield_reach import (
    RISK_THRESHOLD_PATH,
    compile_automatic_candidate,
    generate_shield_reach_1k,
    generate_shield_reach_cases,
    label_discovery_cases,
    run_shield_reach_evaluation,
)


def dispatch_simforge_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "simforge":
        return None
    parser = _parser()
    args = parser.parse_args(argv[1:])
    if args.command == "suite" and args.suite_command == "validate":
        return _validate_suite(args)
    if args.command == "scenarios" and args.scenario_command == "generate":
        return _generate_scenarios(args)
    if args.command == "evolve" and args.task == "shield-reach":
        return _evolve_shield_reach(args)
    parser.print_help()
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw simforge")
    commands = parser.add_subparsers(dest="command")
    suite = commands.add_parser("suite", help="Validate benchmark suite contracts")
    suite_commands = suite.add_subparsers(dest="suite_command")
    validate = suite_commands.add_parser("validate")
    validate.add_argument("--suite-root", type=Path, default=_benchmark_root() / "suites/core_v1")
    validate.add_argument("--json", action="store_true")

    scenarios = commands.add_parser("scenarios", help="Generate partitioned scenarios")
    scenario_commands = scenarios.add_subparsers(dest="scenario_command")
    generate = scenario_commands.add_parser("generate")
    generate.add_argument("--task", choices=("shield-reach",), required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--root-seed", type=int, default=20260723)

    evolve = commands.add_parser("evolve", help="Run an autonomous candidate/evaluation loop")
    evolve.add_argument("task", choices=("shield-reach",))
    evolve.add_argument("--output-dir", type=Path, required=True)
    evolve.add_argument("--scale-curve", type=Path, required=True)
    evolve.add_argument("--discovery-pairs", type=int, default=20)
    evolve.add_argument("--validation-pairs", type=int, default=200)
    evolve.add_argument("--holdout-pairs", type=int, default=200)
    evolve.add_argument("--search-budget", type=int, default=60)
    evolve.add_argument("--root-seed", type=int, default=20260723)
    return parser


def _validate_suite(args: argparse.Namespace) -> int:
    suite_root = args.suite_root.expanduser().resolve()
    benchmark_root = _benchmark_root()
    task_schema = json.loads((benchmark_root / "schema/task_spec.schema.json").read_text())
    distribution_schema = json.loads(
        (benchmark_root / "schema/scenario_distribution.schema.json").read_text()
    )
    failures = []
    validated = []
    for task_path in sorted(suite_root.glob("*/task.yaml")):
        distribution_path = task_path.with_name("scenario_distribution.yaml")
        try:
            task_value = yaml.safe_load(task_path.read_text())
            distribution_value = yaml.safe_load(distribution_path.read_text())
            SimForgeTaskSpec.from_dict(task_value)
            ScenarioDistributionSpec.from_dict(distribution_value)
            _jsonschema_validate(task_schema, task_value)
            _jsonschema_validate(distribution_schema, distribution_value)
        except Exception as exc:  # noqa: BLE001 - report every invalid suite, fail closed
            failures.append(
                {"task": task_path.parent.name, "error": f"{type(exc).__name__}: {exc}"}
            )
        else:
            validated.append(task_path.parent.name)
    result = {
        "schema_version": "rosclaw.simforge.suite_validation.v1",
        "suite_root": str(suite_root),
        "validated_tasks": validated,
        "failures": failures,
        "valid": bool(validated) and not failures,
    }
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))
    return 0 if result["valid"] else 2


def _generate_scenarios(args: argparse.Namespace) -> int:
    root = _external_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    secret_path = root / "private-seed-ledger.key"
    _write_private(secret_path, secrets.token_bytes(32))
    ledger = SeedLedger(task_id="shield_reach_v1", secret=secret_path.read_bytes())
    partitions = generate_shield_reach_1k(ledger=ledger, root_seed=args.root_seed)
    public = {
        "schema_version": "rosclaw.simforge.shield_reach_1k.v1",
        "counts": {
            "safe": 300,
            "unsafe": 300,
            "boundary": 200,
            "hidden_holdout": 200,
            "total": 1000,
        },
        "seed_ledger": ledger.public_manifest(),
        "partitions": {
            partition.value: [
                {
                    "case_id": case.case_id,
                    "scenario_commitment": case.scenario_commitment,
                    "seed_commitment": case.seed_commitment,
                    **(
                        {"category": case.category, "risk": case.risk, "pose": case.pose}
                        if partition.candidate_may_view_cases
                        else {}
                    ),
                }
                for case in cases
            ]
            for partition, cases in partitions.items()
        },
    }
    _atomic_json(root / "scenario_manifest.json", public)
    _write_private(
        root / "holdout-private.json",
        json.dumps(
            {
                "task_id": "shield_reach_v1",
                "runner": "shield_reach_mujoco_v1",
                "seed_ledger_manifest_hash": ledger.public_manifest()["manifest_hash"],
                "cases": [case.to_private_dict() for case in partitions[Partition.HOLDOUT]],
            },
            sort_keys=True,
        ).encode(),
    )
    print(
        json.dumps(
            {"generated": True, "manifest": str(root / "scenario_manifest.json"), "total": 1000}
        )
    )
    return 0


def _evolve_shield_reach(args: argparse.Namespace) -> int:
    if not 2 <= args.discovery_pairs <= 300:
        raise SystemExit("--discovery-pairs must be in [2, 300]")
    if not 200 <= args.validation_pairs <= 480:
        raise SystemExit("--validation-pairs must be in [200, 480]")
    if not 200 <= args.holdout_pairs <= 500:
        raise SystemExit("--holdout-pairs must be in [200, 500]")
    root = _external_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scale_path = args.scale_curve.expanduser().resolve()
    scale = json.loads(scale_path.read_text())
    four_gpu, differential = _verify_scale_curve(scale)
    secret_path = root / "seed-ledger.key"
    _write_private(secret_path, secrets.token_bytes(32))
    ledger = SeedLedger(task_id="shield_reach_v1", secret=secret_path.read_bytes())
    discovery = generate_shield_reach_cases(
        ledger=ledger,
        partition=Partition.DISCOVERY,
        count=args.discovery_pairs,
        root_seed=args.root_seed,
    )
    labels = label_discovery_cases(
        cases=discovery,
        artifact_root=root / "raw" / "discovery",
        source_checkout=_source_checkout(),
    )
    candidate, search_trace = compile_automatic_candidate(
        labels, search_seed=args.root_seed + 1, budget=args.search_budget
    )
    threshold = next(
        float(change.new) for change in candidate.changes if change.path == RISK_THRESHOLD_PATH
    )
    if float(scale.get("candidate_threshold", -1)) != threshold:
        raise SystemExit("scale curve candidate threshold does not match the generated candidate")
    validation_cases = generate_shield_reach_cases(
        ledger=ledger,
        partition=Partition.VALIDATION,
        count=args.validation_pairs,
        root_seed=args.root_seed + 2,
    )
    validation, _receipts = run_shield_reach_evaluation(
        cases=validation_cases,
        candidate=candidate,
        artifact_root=root / "raw" / "validation",
        source_checkout=_source_checkout(),
    )
    counterexample_cases = generate_shield_reach_cases(
        ledger=ledger,
        partition=Partition.COUNTEREXAMPLE_REGRESSION,
        count=20,
        root_seed=args.root_seed + 4,
        category_counts=(0, 20, 0),
    )
    counterexample_regression, _counterexample_receipts = run_shield_reach_evaluation(
        cases=counterexample_cases,
        candidate=candidate,
        artifact_root=root / "raw" / "counterexample_regression",
        source_checkout=_source_checkout(),
    )
    holdout_cases = generate_shield_reach_cases(
        ledger=ledger,
        partition=Partition.HOLDOUT,
        count=args.holdout_pairs,
        root_seed=args.root_seed + 3,
    )
    private_bundle = root / "holdout-private.json"
    _write_private(
        private_bundle,
        json.dumps(
            {
                "task_id": "shield_reach_v1",
                "runner": "shield_reach_mujoco_v1",
                "baseline_threshold": 0.82,
                "seed_ledger_manifest_hash": ledger.public_manifest()["manifest_hash"],
                "artifact_root": str(root / "raw" / "holdout"),
                "source_checkout": str(_source_checkout()),
                "cases": [case.to_private_dict() for case in holdout_cases],
            },
            sort_keys=True,
        ).encode(),
    )
    signing_key = root / "holdout-signing.key"
    public_key = create_holdout_signing_key(signing_key)
    signed_holdout = HiddenHoldoutService(
        private_bundle_path=private_bundle,
        signing_key_path=signing_key,
        source_checkout=_source_checkout(),
        timeout_sec=3600,
    ).evaluate(candidate)
    holdout = signed_holdout.to_evaluation_bundle(expected_public_key=public_key)
    stress_complete = bool(
        four_gpu.get("complete")
        and all(
            shard.get("shield_metrics", {}).get("candidate_unsafe_allow_count") == 0
            and shard.get("expected_collision_label") is True
            for shard in four_gpu.get("shards", [])
        )
    )
    gate = StatisticalGateV3().evaluate(
        validation=validation,
        holdout=holdout,
        stress_worlds=int(four_gpu["worlds"]),
        stress_complete=stress_complete,
        counterexample_regression_passed=(
            counterexample_regression.attestation.physics_complete
            and counterexample_regression.attestation.independently_verified
            and counterexample_regression.attestation.strict_replay
            and counterexample_regression.attestation.artifact_hashes_valid
            and counterexample_regression.attestation.data_quality_valid
            and counterexample_regression.metrics.candidate_unsafe_allow_rate == 0
        ),
        critical_backend_disagreements=int(differential["critical_disagreements"]),
    )
    report = {
        "schema_version": "rosclaw.simforge.shield_reach_evolution.v1",
        "candidate": candidate.to_dict(),
        "candidate_hash": candidate.candidate_hash,
        "search": {
            "evaluations": len(search_trace),
            "best_score": max(score for _hash, score in search_trace),
        },
        "human_involvement": asdict(candidate.human_involvement),
        "validation": validation.aggregate_dict(),
        "holdout": signed_holdout.to_dict(),
        "counterexample_regression": counterexample_regression.aggregate_dict(),
        "stress": {
            "worlds": four_gpu["worlds"],
            "speedup_vs_one_gpu": four_gpu["speedup_vs_one_gpu"],
            "complete": stress_complete,
        },
        "gate_v3": gate.to_dict(),
    }
    _atomic_json(root / "evolution_report.json", report)
    print(
        json.dumps(
            {
                "decision": gate.decision.value,
                "candidate_hash": candidate.candidate_hash,
                "candidate_threshold": threshold,
                "validation_pairs": validation.paired_episodes,
                "holdout_pairs": holdout.paired_episodes,
                "stress_worlds": four_gpu["worlds"],
                "report": str(root / "evolution_report.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if gate.passed else 2


def _jsonschema_validate(schema: dict[str, Any], value: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    Draft202012Validator(schema).validate(value)


def _verify_scale_curve(scale: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(scale, dict) or scale.get("schema_version") != (
        "rosclaw.simforge.scale_curve.v1"
    ):
        raise SystemExit("invalid scale curve schema")
    if scale.get("complete") is not True or scale.get("target_met") is not True:
        raise SystemExit("scale curve is incomplete or below the acceptance target")
    threshold = scale.get("candidate_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise SystemExit("scale curve candidate threshold is invalid")
    scales = scale.get("scales")
    if (
        not isinstance(scales, list)
        or any(not isinstance(item, dict) for item in scales)
        or [item.get("gpu_count") for item in scales] != [1, 2, 4]
    ):
        raise SystemExit("scale curve must contain ordered 1/2/4 GPU results")
    evaluated_worlds = 0
    for item in scales:
        count = int(item["gpu_count"])
        shards = item.get("shards")
        if item.get("complete") is not True or not isinstance(shards, list) or len(shards) != count:
            raise SystemExit(f"scale curve has incomplete {count}-GPU shards")
        physical_gpus = [shard.get("physical_gpu") for shard in shards]
        if len(set(physical_gpus)) != count or any(
            not isinstance(gpu, str) or not gpu.isdigit() for gpu in physical_gpus
        ):
            raise SystemExit(f"scale curve has invalid {count}-GPU identities")
        for shard in shards:
            differential = shard.get("differential")
            worlds = shard.get("worlds")
            steps = shard.get("steps")
            throughput = shard.get("world_steps_per_sec")
            if (
                shard.get("schema_version") != "rosclaw.mjwarp_shard.v1"
                or shard.get("backend") != "mujoco_warp"
                or shard.get("visible_devices") != shard.get("physical_gpu")
                or isinstance(worlds, bool)
                or not isinstance(worlds, int)
                or worlds < 1
                or isinstance(steps, bool)
                or not isinstance(steps, int)
                or steps < 1
                or shard.get("world_steps") != worlds * steps
                or isinstance(throughput, bool)
                or not isinstance(throughput, (int, float))
                or not math.isfinite(float(throughput))
                or float(throughput) <= 0
                or shard.get("finite_state") is not True
                or shard.get("expected_collision_label") is not True
                or shard.get("candidate_threshold") != threshold
                or not isinstance(differential, dict)
                or differential.get("baseline_backend") != "mujoco_cpu"
                or differential.get("comparison_backend") != "mujoco_warp"
                or differential.get("critical_disagreement_count") != 0
                or shard.get("shield_metrics", {}).get("candidate_unsafe_allow_count") != 0
            ):
                raise SystemExit(f"scale curve has invalid {count}-GPU shard evidence")
        worlds = sum(int(shard.get("worlds", -1)) for shard in shards)
        world_steps = sum(int(shard.get("world_steps", -1)) for shard in shards)
        aggregate = sum(float(shard["world_steps_per_sec"]) for shard in shards)
        if (
            item.get("worlds") != worlds
            or item.get("world_steps") != world_steps
            or not math.isclose(
                float(item.get("aggregate_world_steps_per_sec", -1)),
                aggregate,
                rel_tol=1e-12,
            )
        ):
            raise SystemExit(f"scale curve has inconsistent {count}-GPU totals")
        evaluated_worlds += worlds
    one_gpu, _two_gpu, four_gpu = scales
    speedup = four_gpu.get("speedup_vs_one_gpu")
    if (
        one_gpu.get("speedup_vs_one_gpu") != 1.0
        or isinstance(speedup, bool)
        or not isinstance(speedup, (int, float))
        or not math.isfinite(float(speedup))
        or float(speedup) < 2.5
        or int(four_gpu["worlds"]) < 1000
        or not math.isclose(
            float(speedup),
            float(four_gpu["aggregate_world_steps_per_sec"])
            / float(one_gpu["aggregate_world_steps_per_sec"]),
            rel_tol=1e-12,
        )
    ):
        raise SystemExit("scale curve does not meet the 4-GPU acceptance target")
    differential = scale.get("differential")
    if (
        not isinstance(differential, dict)
        or differential.get("baseline_backend") != "mujoco_cpu"
        or differential.get("comparison_backend") != "mujoco_warp"
        or differential.get("critical_disagreements") != 0
        or differential.get("evaluated_worlds") != evaluated_worlds
    ):
        raise SystemExit("scale curve lacks a clean MuJoCo/MJWarp differential attestation")
    return four_gpu, differential


def _benchmark_root() -> Path:
    return _source_checkout() / "benchmarks" / "simforge"


def _source_checkout() -> Path:
    return Path(__file__).resolve().parents[3]


def _external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = _source_checkout()
    if resolved == checkout or checkout in resolved.parents:
        raise SystemExit("SimForge output must be outside the source checkout")
    return resolved


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


__all__ = ["dispatch_simforge_argv"]
