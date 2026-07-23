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

from rosclaw.simforge.attestation import (
    create_simforge_signing_key_pair,
    scale_curve_commitment,
    verify_scale_curve_signature,
)
from rosclaw.simforge.evaluation import StressEvidence, _attest_stress_evidence
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

_MAX_CONFIG_BYTES = 32 * 1024 * 1024


def dispatch_simforge_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "simforge":
        return None
    parser = _parser()
    args = parser.parse_args(argv[1:])
    if args.command == "suite" and args.suite_command == "validate":
        return _validate_suite(args)
    if args.command == "scenarios" and args.scenario_command == "generate":
        return _generate_scenarios(args)
    if args.command == "key" and args.key_command == "create":
        return _create_key(args)
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

    key = commands.add_parser("key", help="Create SimForge evidence signing keys")
    key_commands = key.add_subparsers(dest="key_command")
    key_create = key_commands.add_parser("create")
    key_create.add_argument("--private-key", type=Path, required=True)
    key_create.add_argument("--public-key", type=Path, required=True)

    evolve = commands.add_parser("evolve", help="Run an autonomous candidate/evaluation loop")
    evolve.add_argument("task", choices=("shield-reach",))
    evolve.add_argument("--output-dir", type=Path, required=True)
    evolve.add_argument("--scale-curve", type=Path, required=True)
    evolve.add_argument("--scale-curve-public-key", type=Path, required=True)
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
            task_value = _read_yaml(task_path)
            distribution_value = _read_yaml(distribution_path)
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


def _create_key(args: argparse.Namespace) -> int:
    private_path = args.private_key.expanduser().resolve()
    public_path = args.public_key.expanduser().resolve()
    fingerprint = create_simforge_signing_key_pair(
        private_key_path=private_path,
        public_key_path=public_path,
        source_checkout=_source_checkout(),
    )
    print(
        json.dumps(
            {
                "created": True,
                "private_key": str(private_path),
                "public_key": str(public_path),
                "public_key_fingerprint": fingerprint,
            },
            sort_keys=True,
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
    scale_path = args.scale_curve.expanduser().resolve()
    checkout = _source_checkout()
    if scale_path == checkout or checkout in scale_path.parents:
        raise SystemExit("scale-curve evidence must be outside the source checkout")
    scale = _read_json(scale_path)
    try:
        verify_scale_curve_signature(
            scale,
            expected_public_key_path=args.scale_curve_public_key,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"untrusted scale curve: {exc}") from exc
    root.mkdir(parents=True, exist_ok=True)
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
    threshold_value = next(
        change.new for change in candidate.changes if change.path == RISK_THRESHOLD_PATH
    )
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
        raise RuntimeError("generated ShieldReach threshold is not numeric")
    threshold = float(threshold_value)
    four_gpu, differential, stress = _verify_scale_curve(
        scale,
        expected_public_key_path=args.scale_curve_public_key,
        expected_candidate_hash=candidate.candidate_hash,
        expected_candidate_threshold=threshold,
    )
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
    gate = StatisticalGateV3().evaluate(
        validation=validation,
        holdout=holdout,
        stress=stress,
        counterexample_regression=counterexample_regression,
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
            "complete": stress.complete,
            "scale_curve_commitment": stress.scale_curve_commitment,
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


def _verify_scale_curve(
    scale: Any,
    *,
    expected_public_key_path: Path,
    expected_candidate_hash: str,
    expected_candidate_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], StressEvidence]:
    try:
        encoded_size = len(
            json.dumps(
                scale,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise SystemExit(f"invalid scale curve value: {exc}") from exc
    if encoded_size > _MAX_CONFIG_BYTES:
        raise SystemExit("scale curve exceeds the configured size limit")
    if not isinstance(scale, dict) or scale.get("schema_version") != (
        "rosclaw.simforge.scale_curve.v1"
    ):
        raise SystemExit("invalid scale curve schema")
    try:
        verify_scale_curve_signature(
            scale,
            expected_public_key_path=expected_public_key_path,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"untrusted scale curve: {exc}") from exc
    if scale.get("task_id") != "shield_reach_v1":
        raise SystemExit("scale curve task identity is invalid")
    if (
        scale.get("minimum_worlds_required") != 1000
        or scale.get("minimum_speedup_required") != 2.5
        or scale.get("fault_injected") is not False
        or scale.get("fault_type") is not None
    ):
        raise SystemExit("scale curve acceptance policy or fault state is invalid")
    if (
        not isinstance(expected_candidate_hash, str)
        or len(expected_candidate_hash) != 71
        or not expected_candidate_hash.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in expected_candidate_hash[7:])
    ):
        raise SystemExit("expected candidate hash is invalid")
    if scale.get("complete") is not True or scale.get("target_met") is not True:
        raise SystemExit("scale curve is incomplete or below the acceptance target")
    if (
        isinstance(expected_candidate_threshold, bool)
        or not isinstance(expected_candidate_threshold, (int, float))
        or not math.isfinite(float(expected_candidate_threshold))
        or not 0.1 <= float(expected_candidate_threshold) <= 0.9
    ):
        raise SystemExit("expected candidate threshold is invalid")
    scales = scale.get("scales")
    if (
        not isinstance(scales, list)
        or any(not isinstance(item, dict) for item in scales)
        or any(type(item.get("gpu_count")) is not int for item in scales)
        or [item.get("gpu_count") for item in scales] != [1, 2, 4]
    ):
        raise SystemExit("scale curve must contain ordered 1/2/4 GPU results")
    evaluated_worlds = 0
    workload: tuple[int, int] | None = None
    scale_gpu_ids: list[list[str]] = []
    device_names: set[str] = set()
    model_hashes: set[str] = set()
    for item in scales:
        count = item["gpu_count"]
        shards = item.get("shards")
        requested_gpus = item.get("requested_gpus")
        if (
            item.get("complete") is not True
            or item.get("failures") != []
            or not isinstance(shards, list)
            or len(shards) != count
            or not isinstance(requested_gpus, list)
            or len(requested_gpus) != count
            or any(not isinstance(gpu, str) or not gpu.isdigit() for gpu in requested_gpus)
            or len(set(requested_gpus)) != count
        ):
            raise SystemExit(f"scale curve has incomplete {count}-GPU shards")
        physical_gpus = [shard.get("physical_gpu") for shard in shards]
        if physical_gpus != requested_gpus:
            raise SystemExit(f"scale curve has invalid {count}-GPU identities")
        scale_gpu_ids.append(requested_gpus)
        for shard_index, shard in enumerate(shards):
            differential = shard.get("differential")
            worlds = shard.get("worlds")
            steps = shard.get("steps")
            throughput = shard.get("world_steps_per_sec")
            cpu_baseline_time = shard.get("cpu_baseline_time_sec")
            if (
                shard.get("schema_version") != "rosclaw.mjwarp_shard.v1"
                or shard.get("backend") != "mujoco_warp"
                or shard.get("visible_devices") != shard.get("physical_gpu")
                or isinstance(worlds, bool)
                or not isinstance(worlds, int)
                or not 1 <= worlds <= 4096
                or isinstance(steps, bool)
                or not isinstance(steps, int)
                or not 1 <= steps <= 1_000_000
                or shard.get("world_steps") != worlds * steps
                or shard.get("pose") != "mixed"
                or shard.get("world_offset") != shard_index * worlds
                or isinstance(throughput, bool)
                or not isinstance(throughput, (int, float))
                or not math.isfinite(float(throughput))
                or float(throughput) <= 0
                or isinstance(cpu_baseline_time, bool)
                or not isinstance(cpu_baseline_time, (int, float))
                or not math.isfinite(float(cpu_baseline_time))
                or float(cpu_baseline_time) <= 0
                or shard.get("finite_state") is not True
                or shard.get("expected_collision_label") is not True
                or shard.get("scenario_label_valid") is not True
                or not isinstance(differential, dict)
                or differential.get("baseline_backend") != "mujoco_cpu"
                or differential.get("comparison_backend") != "mujoco_warp"
                or differential.get("critical_label") != "collision"
                or differential.get("critical_disagreement_count") != 0
                or not _stress_runtime_identity_valid(shard, worlds=worlds)
                or not _stress_candidate_passes(
                    shard,
                    worlds=worlds,
                    candidate_threshold=float(expected_candidate_threshold),
                )
            ):
                raise SystemExit(f"scale curve has invalid {count}-GPU shard evidence")
            current_workload = (worlds, steps)
            if workload is None:
                workload = current_workload
            elif current_workload != workload:
                raise SystemExit("scale curve workloads are not comparable across GPU counts")
            device_names.add(shard["device_name"])
            model_hashes.add(shard["model_hash"])
        worlds = sum(int(shard.get("worlds", -1)) for shard in shards)
        world_steps = sum(int(shard.get("world_steps", -1)) for shard in shards)
        aggregate = sum(float(shard["world_steps_per_sec"]) for shard in shards)
        item_aggregate = item.get("aggregate_world_steps_per_sec")
        process_wall_time = item.get("process_wall_time_sec")
        if (
            item.get("worlds") != worlds
            or workload is None
            or item.get("steps") != workload[1]
            or item.get("world_steps") != world_steps
            or isinstance(process_wall_time, bool)
            or not isinstance(process_wall_time, (int, float))
            or not math.isfinite(float(process_wall_time))
            or float(process_wall_time) <= 0
            or isinstance(item_aggregate, bool)
            or not isinstance(item_aggregate, (int, float))
            or not math.isfinite(float(item_aggregate))
            or not math.isclose(
                float(item_aggregate),
                aggregate,
                rel_tol=1e-12,
            )
        ):
            raise SystemExit(f"scale curve has inconsistent {count}-GPU totals")
        evaluated_worlds += worlds
    one_gpu, _two_gpu, four_gpu = scales
    if (
        any(gpu_ids != scale_gpu_ids[-1][: len(gpu_ids)] for gpu_ids in scale_gpu_ids)
        or len(device_names) != 1
        or len(model_hashes) != 1
    ):
        raise SystemExit("scale curve runtime identities are inconsistent across GPU counts")
    baseline_throughput = float(one_gpu["aggregate_world_steps_per_sec"])
    for item in scales:
        speedup = item.get("speedup_vs_one_gpu")
        if (
            isinstance(speedup, bool)
            or not isinstance(speedup, (int, float))
            or not math.isfinite(float(speedup))
            or not math.isclose(
                float(speedup),
                float(item["aggregate_world_steps_per_sec"]) / baseline_throughput,
                rel_tol=1e-12,
            )
        ):
            raise SystemExit("scale curve contains an inconsistent speedup")
    speedup = four_gpu["speedup_vs_one_gpu"]
    if float(speedup) < 2.5 or int(four_gpu["worlds"]) < 1000:
        raise SystemExit("scale curve does not meet the 4-GPU acceptance target")
    differential = scale.get("differential")
    if (
        not isinstance(differential, dict)
        or differential.get("baseline_backend") != "mujoco_cpu"
        or differential.get("comparison_backend") != "mujoco_warp"
        or differential.get("critical_label") != "collision"
        or isinstance(differential.get("critical_disagreements"), bool)
        or differential.get("critical_disagreements") != 0
        or isinstance(differential.get("evaluated_worlds"), bool)
        or differential.get("evaluated_worlds") != evaluated_worlds
    ):
        raise SystemExit("scale curve lacks a clean MuJoCo/MJWarp differential attestation")
    stress = _attest_stress_evidence(
        StressEvidence(
            task_id="shield_reach_v1",
            candidate_hash=expected_candidate_hash,
            worlds=int(four_gpu["worlds"]),
            complete=True,
            critical_backend_disagreements=int(differential["critical_disagreements"]),
            scale_curve_commitment=scale_curve_commitment(scale),
        )
    )
    return four_gpu, differential, stress


def _benchmark_root() -> Path:
    return _source_checkout() / "benchmarks" / "simforge"


def _stress_candidate_passes(
    shard: dict[str, Any],
    *,
    worlds: int,
    candidate_threshold: float,
) -> bool:
    risks = shard.get("risk_values")
    collisions = shard.get("collision_worlds")
    cpu_collisions = shard.get("cpu_collision_worlds")
    scenario_collisions = shard.get("scenario_collision_worlds")
    if (
        not isinstance(risks, list)
        or len(risks) != worlds
        or not isinstance(collisions, list)
        or len(collisions) > worlds
        or cpu_collisions != collisions
        or scenario_collisions != collisions
        or shard.get("collision_world_count") != len(collisions)
    ):
        return False
    collision_set: set[int] = set()
    for item in collisions:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < worlds
            or item in collision_set
        ):
            return False
        collision_set.add(item)
    normalized_risks: list[float] = []
    for item in risks:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        normalized = float(item)
        if not math.isfinite(normalized) or not 0 <= normalized <= 1:
            return False
        normalized_risks.append(normalized)
    return not any(
        risk <= candidate_threshold and index in collision_set
        for index, risk in enumerate(normalized_risks)
    )


def _stress_runtime_identity_valid(shard: dict[str, Any], *, worlds: int) -> bool:
    randomization = shard.get("randomization")
    return bool(
        shard.get("device") == "cuda:0"
        and isinstance(shard.get("device_name"), str)
        and 1 <= len(shard["device_name"]) <= 256
        and _is_sha256(shard.get("model_hash"))
        and _is_sha256(shard.get("qpos_checksum"))
        and not isinstance(shard.get("gpu_memory_used_bytes"), bool)
        and isinstance(shard.get("gpu_memory_used_bytes"), int)
        and shard["gpu_memory_used_bytes"] >= 0
        and isinstance(randomization, dict)
        and _is_sha256(randomization.get("parameter_hash"))
        and _valid_offsets(randomization.get("joint_control_offset_rad"), worlds)
    )


def _valid_offsets(value: Any, worlds: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == worlds
        and all(
            isinstance(row, list)
            and len(row) == 6
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                and abs(float(item)) <= 0.0021
                for item in row
            )
            for row in value
        )
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _source_checkout() -> Path:
    return Path(__file__).resolve().parents[3]


def _external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = _source_checkout()
    if resolved == checkout or checkout in resolved.parents:
        raise SystemExit("SimForge output must be outside the source checkout")
    return resolved


def _write_private(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating a private SimForge file")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating SimForge JSON")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"SimForge JSON file does not exist: {path}")
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise SystemExit(f"SimForge JSON file exceeds {_MAX_CONFIG_BYTES} bytes: {path}")
    try:
        with path.open("rb") as handle:
            payload = handle.read(_MAX_CONFIG_BYTES + 1)
        if len(payload) > _MAX_CONFIG_BYTES:
            raise ValueError("file grew beyond its size limit")
        return json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"invalid SimForge JSON file: {path}: {exc}") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > 1_048_576:
        raise ValueError(f"SimForge YAML exceeds 1 MiB: {path}")
    with path.open("rb") as handle:
        payload = handle.read(1_048_577)
    if len(payload) > 1_048_576:
        raise ValueError(f"SimForge YAML exceeds 1 MiB: {path}")
    value = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SimForge YAML root must be a mapping: {path}")
    return value


__all__ = ["dispatch_simforge_argv"]
