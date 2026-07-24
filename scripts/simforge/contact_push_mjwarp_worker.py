#!/usr/bin/env python3
"""Run ContactPush stress worlds on one isolated MJWarp GPU shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

_MAX_WORLDS = 4096
_MAX_GROUP_SIZE = 256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=250)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[2]
    candidate_path = args.candidate.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.is_relative_to(checkout):
        parser.error("--output must stay outside the source checkout")
    if not candidate_path.is_file():
        parser.error("--candidate must identify an existing JSON artifact")
    if not 1 <= args.worlds <= _MAX_WORLDS:
        parser.error(f"--worlds must be in [1, {_MAX_WORLDS}]")
    if not 1 <= args.group_size <= _MAX_GROUP_SIZE:
        parser.error(f"--group-size must be in [1, {_MAX_GROUP_SIZE}]")
    if args.world_offset < 0:
        parser.error("--world-offset must be non-negative")
    physical_gpu = os.environ.get("ROSCLAW_PHYSICAL_GPU", "")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical_gpu.isdigit() or visible_devices != physical_gpu:
        parser.error("worker requires one matching physical and visible GPU")

    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from rosclaw.simforge.contact_push_learning import ContactPushCandidate
    from rosclaw.simforge.models import Partition
    from rosclaw.simforge.tasks.contact_push_v3 import (
        ContactPushPhysics,
        ContactPushScenario,
        ContactPushStatus,
        build_contact_push_model_xml,
    )

    candidate = ContactPushCandidate.from_dict(
        json.loads(candidate_path.read_text(encoding="utf-8"))
    )
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    device = wp.get_device("cuda:0")
    free_before = int(device.free_memory)
    physics = ContactPushPhysics(trace_stride=1000)
    groups = _stress_groups(
        total=args.worlds,
        group_size=args.group_size,
        seed=args.seed,
        world_offset=args.world_offset,
        partition=Partition.STRESS,
        scenario_type=ContactPushScenario,
    )
    started = time.perf_counter()
    compilation_sec = 0.0
    cpu_sec = 0.0
    world_records: list[dict[str, Any]] = []
    finite = True
    for group_index, scenarios in enumerate(groups):
        policies = tuple(candidate.policy_for(scenario) for scenario in scenarios)
        cpu_started = time.perf_counter()
        cpu_results = tuple(
            physics.run(scenario, policy)
            for scenario, policy in zip(scenarios, policies, strict=True)
        )
        cpu_sec += time.perf_counter() - cpu_started
        representative = scenarios[0]
        model_started = time.perf_counter()
        cpu_model = mujoco.MjModel.from_xml_string(
            build_contact_push_model_xml(representative, policies[0])
        )
        cpu_data = mujoco.MjData(cpu_model)
        mujoco.mj_forward(cpu_model, cpu_data)
        model = mjw.put_model(cpu_model)
        data = mjw.put_data(
            cpu_model,
            cpu_data,
            nworld=len(scenarios),
            nconmax=max(128, len(scenarios) * 32),
            njmax=max(256, len(scenarios) * 128),
        )
        if group_index == 0:
            compilation_sec += time.perf_counter() - model_started
        parameters = _device_parameters(
            scenarios=scenarios,
            policies=policies,
            device=device,
            wp=wp,
        )
        timestep = float(cpu_model.opt.timestep)
        steps = math.ceil(2.5 / timestep)
        for step in range(steps):
            _set_controls(
                data=data,
                parameters=parameters,
                step=step,
                timestep=timestep,
                device=device,
                wp=wp,
            )
            mjw.step(model, data)
        wp.synchronize()
        qpos = data.qpos.numpy()
        finite = finite and bool(np.isfinite(qpos).all())
        for local_index, (scenario, policy, cpu_result) in enumerate(
            zip(scenarios, policies, cpu_results, strict=True)
        ):
            final_x = float(qpos[local_index, 0])
            error = scenario.target_distance_m - final_x
            warp_status = _warp_status(
                final_x=final_x,
                error=error,
                tolerance=physics.target_tolerance_m,
            )
            cpu_status = cpu_result.status.value
            exact_agreement = warp_status == cpu_status
            boundary_distance = abs(abs(error) - physics.target_tolerance_m)
            critical = (
                not exact_agreement
                and boundary_distance > 0.008
                and cpu_result.status
                not in {
                    ContactPushStatus.FORCE_LIMIT,
                    ContactPushStatus.NON_FINITE,
                }
            )
            if cpu_result.status in {
                ContactPushStatus.FORCE_LIMIT,
                ContactPushStatus.NON_FINITE,
            }:
                critical = True
            world_records.append(
                {
                    "world_index": args.world_offset + len(world_records),
                    "scenario_commitment": scenario.scenario_commitment,
                    "policy_hash": policy.policy_hash,
                    "cpu_status": cpu_status,
                    "warp_status": warp_status,
                    "cpu_final_x_m": cpu_result.final_object_x_m,
                    "warp_final_x_m": final_x,
                    "absolute_position_delta_m": abs(cpu_result.final_object_x_m - final_x),
                    "exact_label_agreement": exact_agreement,
                    "critical_disagreement": critical,
                    "cpu_force_violation": (cpu_result.status is ContactPushStatus.FORCE_LIMIT),
                }
            )
    elapsed = time.perf_counter() - started
    critical_count = sum(bool(item["critical_disagreement"]) for item in world_records)
    force_violations = sum(bool(item["cpu_force_violation"]) for item in world_records)
    exact_count = sum(bool(item["exact_label_agreement"]) for item in world_records)
    payload = {
        "schema_version": "rosclaw.contact_push_mjwarp_shard.v1",
        "backend": "mujoco_warp",
        "backend_version": getattr(mjw, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
        "candidate_hash": candidate.candidate_hash,
        "dataset_snapshot_hash": candidate.dataset_snapshot_hash,
        "physical_gpu": physical_gpu,
        "visible_devices": visible_devices,
        "device": str(device),
        "device_name": device.name,
        "worlds": len(world_records),
        "unique_scenarios": len({item["scenario_commitment"] for item in world_records}),
        "group_size": args.group_size,
        "groups": len(groups),
        "steps_per_world": 1250,
        "world_steps": len(world_records) * 1250,
        "wall_time_sec": elapsed,
        "cpu_reference_time_sec": cpu_sec,
        "compile_time_sec": compilation_sec,
        "world_steps_per_sec": len(world_records) * 1250 / max(elapsed, 1e-9),
        "finite_state": finite,
        "exact_label_agreement_rate": exact_count / len(world_records),
        "critical_backend_disagreements": critical_count,
        "cpu_force_violations": force_violations,
        "maximum_position_delta_m": max(
            float(item["absolute_position_delta_m"]) for item in world_records
        ),
        "world_set_commitment": _hash_json(
            {
                "worlds": [
                    {
                        "index": item["world_index"],
                        "scenario": item["scenario_commitment"],
                        "policy": item["policy_hash"],
                    }
                    for item in world_records
                ]
            }
        ),
        "world_records": world_records,
        "gpu_memory_used_bytes": max(0, free_before - int(device.free_memory)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "world_records"},
            sort_keys=True,
        )
    )
    return 0 if finite and critical_count == 0 and force_violations == 0 else 2


def _stress_groups(
    *,
    total: int,
    group_size: int,
    seed: int,
    world_offset: int,
    partition: Any,
    scenario_type: Any,
) -> tuple[tuple[Any, ...], ...]:
    groups = []
    produced = 0
    group_index = 0
    while produced < total:
        size = min(group_size, total - produced)
        rng = random.Random(seed ^ (group_index * 0x9E3779B1))
        mass = rng.uniform(0.22, 0.75)
        friction = rng.uniform(0.14, 0.85)
        offset = rng.uniform(-0.012, 0.012)
        scenarios = []
        for local_index in range(size):
            index = world_offset + produced + local_index
            scenario_seed = int.from_bytes(
                hashlib.sha256(f"{seed}\0{index}".encode()).digest()[:8],
                "big",
            )
            commitment = _hash_json(
                {"seed": scenario_seed, "index": index, "partition": partition.value}
            )
            scenarios.append(
                scenario_type(
                    scenario_id=f"mjwarp_stress_{index:06d}",
                    partition=partition,
                    seed=scenario_seed,
                    seed_commitment=commitment,
                    object_mass_kg=mass,
                    floor_friction=friction,
                    target_distance_m=rng.uniform(0.14, 0.33),
                    initial_offset_y_m=offset,
                    control_delay_sec=rng.uniform(0.0, 0.08),
                    friction_sensor_noise=rng.uniform(-0.04, 0.04),
                )
            )
        groups.append(tuple(scenarios))
        produced += size
        group_index += 1
    return tuple(groups)


def _device_parameters(
    *,
    scenarios: tuple[Any, ...],
    policies: tuple[Any, ...],
    device: Any,
    wp: Any,
) -> dict[str, Any]:
    return {
        "delay": wp.array(
            [scenario.control_delay_sec for scenario in scenarios],
            dtype=wp.float32,
            device=device,
        ),
        "duration": wp.array(
            [policy.contact_duration_sec for policy in policies],
            dtype=wp.float32,
            device=device,
        ),
        "velocity": wp.array(
            [policy.push_velocity_mps for policy in policies],
            dtype=wp.float32,
            device=device,
        ),
        "deceleration": wp.array(
            [policy.deceleration_fraction for policy in policies],
            dtype=wp.float32,
            device=device,
        ),
        "micro": wp.array(
            [1 if policy.micro_push else 0 for policy in policies],
            dtype=wp.int32,
            device=device,
        ),
        "target": wp.array(
            [scenario.target_distance_m for scenario in scenarios],
            dtype=wp.float32,
            device=device,
        ),
    }


def _set_controls(
    *,
    data: Any,
    parameters: dict[str, Any],
    step: int,
    timestep: float,
    device: Any,
    wp: Any,
) -> None:
    wp.launch(
        kernel=_control_kernel,
        dim=int(data.ctrl.shape[0]),
        inputs=[
            data.ctrl,
            data.qpos,
            parameters["delay"],
            parameters["duration"],
            parameters["velocity"],
            parameters["deceleration"],
            parameters["micro"],
            parameters["target"],
            step,
            timestep,
        ],
        device=device,
    )


def _warp_status(*, final_x: float, error: float, tolerance: float) -> str:
    if final_x <= 0.002:
        return "NO_CONTACT"
    if error < -tolerance:
        return "OBJECT_OVERSHOT"
    if error > tolerance:
        return "OBJECT_UNDERSHOT"
    return "SUCCESS"


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


try:
    import warp as wp

    @wp.kernel
    def _control_kernel(
        ctrl: wp.array2d(dtype=wp.float32),
        qpos: wp.array2d(dtype=wp.float32),
        delay: wp.array(dtype=wp.float32),
        duration: wp.array(dtype=wp.float32),
        velocity: wp.array(dtype=wp.float32),
        deceleration: wp.array(dtype=wp.float32),
        micro: wp.array(dtype=wp.int32),
        target: wp.array(dtype=wp.float32),
        step: int,
        timestep: float,
    ):
        world = wp.tid()
        now = float(step) * timestep
        primary_end = delay[world] + duration[world]
        command = 0.0
        if now >= delay[world] and now < primary_end:
            elapsed = now - delay[world]
            deceleration_start = duration[world] * deceleration[world]
            command = velocity[world]
            if elapsed > deceleration_start and deceleration[world] < 1.0:
                tail = duration[world] - deceleration_start
                progress = (elapsed - deceleration_start) / wp.max(tail, 1.0e-9)
                progress = wp.min(1.0, wp.max(0.0, progress))
                command = velocity[world] * wp.max(0.12, 1.0 - progress)
        elif (
            micro[world] == 1
            and now < wp.min(2.35, primary_end + 0.50)
            and qpos[world, 0] < target[world] - 0.01575
        ):
            command = wp.min(0.14, velocity[world] * 0.45)
        ctrl[world, 0] = command

except ImportError:
    _control_kernel = None


if __name__ == "__main__":
    raise SystemExit(main())
