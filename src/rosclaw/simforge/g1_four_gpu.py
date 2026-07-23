"""Four-GPU GoalForge screening with signed, isolated holdout evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.attestation import (
    create_simforge_signing_key_pair,
    sign_scale_curve,
    verify_scale_curve_signature,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class GoalForgeGPUShard:
    role: str
    physical_gpu: int
    gpu_uuid: str
    pci_bus_id: str
    scenario_count: int
    scenario_set_commitment: str
    evidence_commitment: str
    generations_seen: tuple[int, ...]
    aggregate: tuple[tuple[str, float], ...]
    signature_verified: bool
    private_case_results_disclosed: bool
    public_manifest: str


@dataclass(frozen=True)
class GoalForgeFourGPUResult:
    shards: tuple[GoalForgeGPUShard, ...]
    total_scenarios: int
    unique_gpu_uuids: int
    all_generations_seen: tuple[int, ...]
    holdout_case_results_disclosed: bool
    missing_shards: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.four_gpu.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.shards) == 4
            and not self.missing_shards
            and self.total_scenarios >= 1000
            and self.unique_gpu_uuids == 4
            and set(self.all_generations_seen) == set(range(11))
            and all(shard.signature_verified for shard in self.shards)
            and not self.holdout_case_results_disclosed
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shards": [
                {
                    **asdict(shard),
                    "aggregate": dict(shard.aggregate),
                }
                for shard in self.shards
            ],
            "total_scenarios": self.total_scenarios,
            "unique_gpu_uuids": self.unique_gpu_uuids,
            "all_generations_seen": list(self.all_generations_seen),
            "holdout_case_results_disclosed": self.holdout_case_results_disclosed,
            "missing_shards": list(self.missing_shards),
            "passed": self.passed,
        }


def run_goalforge_four_gpu(
    *,
    output_dir: Path,
    source_checkout: Path,
    count_per_gpu: int = 250,
    root_seed: int = 20260723,
) -> GoalForgeFourGPUResult:
    if count_per_gpu < 250:
        raise ValueError("Phase 4 four-GPU acceptance requires at least 250 cases per GPU")
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("four-GPU evidence must be outside source checkout")
    root.mkdir(parents=True, exist_ok=False)
    private_root = root / "private"
    private_root.mkdir(mode=0o700)
    key_root = root / "keys"
    key_root.mkdir(mode=0o700)
    private_key = key_root / "shard-private.pem"
    public_key = key_root / "shard-public.pem"
    create_simforge_signing_key_pair(
        private_key_path=private_key,
        public_key_path=public_key,
        source_checkout=checkout,
    )
    roles = ("practice", "candidate_search", "falsification", "private_holdout")
    script = checkout / "scripts/simforge/g1_goalforge_gpu_worker.py"
    processes: list[tuple[str, int, Path, Path, subprocess.Popen[str]]] = []
    for gpu, role in enumerate(roles):
        manifest = root / f"gpu-{gpu}-{role}.json"
        rows = private_root / f"gpu-{gpu}-{role}-rows.jsonl"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--role",
                role,
                "--physical-gpu",
                str(gpu),
                "--count",
                str(count_per_gpu),
                "--root-seed",
                str(root_seed + 1009 * gpu),
                "--output",
                str(manifest),
                "--private-rows",
                str(rows),
            ],
            cwd=checkout,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((role, gpu, manifest, rows, process))
    shards: list[GoalForgeGPUShard] = []
    failures: list[str] = []
    for role, gpu, manifest, _rows, process in processes:
        stdout, stderr = process.communicate(timeout=60.0)
        if process.returncode != 0 or not manifest.is_file():
            failures.append(f"{role}:gpu{gpu}:{stdout[-300:]}:{stderr[-600:]}")
            continue
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["attestation"] = sign_scale_curve(value, private_key_path=private_key)
        manifest.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verified = False
        try:
            verify_scale_curve_signature(value, expected_public_key_path=public_key)
            verified = True
        except ValueError:
            pass
        shards.append(
            GoalForgeGPUShard(
                role=str(value["role"]),
                physical_gpu=int(value["physical_gpu"]),
                gpu_uuid=str(value["gpu_uuid"]),
                pci_bus_id=str(value["pci_bus_id"]),
                scenario_count=int(value["scenario_count"]),
                scenario_set_commitment=str(value["scenario_set_commitment"]),
                evidence_commitment=str(value["evidence_commitment"]),
                generations_seen=tuple(int(item) for item in value["generations_seen"]),
                aggregate=tuple(
                    sorted((str(key), float(item)) for key, item in value["aggregate"].items())
                ),
                signature_verified=verified,
                private_case_results_disclosed=bool(value["private_case_results_disclosed"]),
                public_manifest=str(manifest),
            )
        )
    observed_roles = {shard.role for shard in shards}
    missing = tuple(sorted(set(roles) - observed_roles))
    holdout = next((shard for shard in shards if shard.role == "private_holdout"), None)
    result = GoalForgeFourGPUResult(
        shards=tuple(sorted(shards, key=lambda shard: shard.physical_gpu)),
        total_scenarios=sum(shard.scenario_count for shard in shards),
        unique_gpu_uuids=len({shard.gpu_uuid for shard in shards}),
        all_generations_seen=tuple(
            sorted({value for shard in shards for value in shard.generations_seen})
        ),
        holdout_case_results_disclosed=(
            True if holdout is None else holdout.private_case_results_disclosed
        ),
        missing_shards=(*missing, *failures),
    )
    (root / "four-gpu-summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "GoalForgeFourGPUResult",
    "GoalForgeGPUShard",
    "run_goalforge_four_gpu",
]
