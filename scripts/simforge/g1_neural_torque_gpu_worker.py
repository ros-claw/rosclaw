#!/usr/bin/env python3
"""One isolated GPU seed for G1 recurrent direct-torque distillation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rosclaw.simforge.g1_neural_torque import (
    G1TeacherTorqueEpisode,
    G1TorqueSafetyConfig,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    teacher_dataset_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--body-hash", required=True)
    parser.add_argument("--parent-policy-hash", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.physical_gpu not in range(4) or args.seed < 0 or not 1 <= args.epochs <= 100:
        raise SystemExit("invalid neural torque GPU worker request")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise SystemExit(
            f"neural torque CUDA identity mismatch: expected {args.physical_gpu}, visible={visible}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("neural torque worker requires exactly one visible CUDA device")
    training, validation = _load_dataset(args.dataset)
    safety = G1TorqueSafetyConfig(
        torque_guard_scale=0.80,
        maximum_mechanical_power_w=4000.0,
        maximum_parent_deviation_ratio=0.05,
        maximum_projection_ratio=0.35,
        maximum_observation_z=5.0,
        minimum_upright_gravity_z=-0.97,
        minimum_pelvis_height_m=0.70,
        recovery_cooldown_steps=250,
        warmup_steps=100,
    )
    learner = G1ContinualTorqueActorCritic(
        G1NeuralTorqueLearnerConfig(
            hidden_dim=96,
            sequence_length=32,
            batch_size=256,
            actor_lr=5e-5,
            device="cuda:0",
            seed=args.seed,
        ),
        safety=safety,
    )
    started = time.perf_counter()
    metrics = learner.pretrain_behavior(
        training,
        validation=validation,
        epochs=args.epochs,
        stride=4,
    )
    dataset_hash = teacher_dataset_hash(training)
    artifact = learner.artifact_bytes(
        body_hash=args.body_hash,
        parent_policy_hash=args.parent_policy_hash,
        dataset_hash=dataset_hash,
    )
    args.artifact.write_bytes(artifact)
    torch.cuda.synchronize()
    gpu_uuid, pci_bus_id = _gpu_identity(args.physical_gpu)
    value = {
        "schema_version": "rosclaw.simforge.g1_neural_torque_gpu_worker.v1",
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": visible,
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "learner_seed": args.seed,
        "dataset_hash": dataset_hash,
        "training_episodes": len(training),
        "validation_episodes": len(validation),
        "epochs": len(metrics),
        "initial_training_loss": metrics[0].training_loss,
        "final_training_loss": metrics[-1].training_loss,
        "initial_validation_loss": metrics[0].validation_loss,
        "final_validation_loss": metrics[-1].validation_loss,
        "finite": all(item.finite for item in metrics),
        "action_limit_fraction": metrics[-1].action_limit_fraction,
        "artifact_hash": "sha256:" + __import__("hashlib").sha256(artifact).hexdigest(),
        "artifact_bytes": len(artifact),
        "elapsed_sec": time.perf_counter() - started,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "claims": {
            "action_semantics": "DIRECT_JOINT_TORQUE",
            "activation_ceiling": "SIM_ONLY",
            "physics_effect_proven": False,
            "hardware_authorized": False,
        },
    }
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _load_dataset(
    path: Path,
) -> tuple[tuple[G1TeacherTorqueEpisode, ...], tuple[G1TeacherTorqueEpisode, ...]]:
    with np.load(path, allow_pickle=False) as value:
        training_count = int(value["training_count"])
        validation_count = int(value["validation_count"])
        training = tuple(
            G1TeacherTorqueEpisode(
                value[f"training_{index}_observations"],
                value[f"training_{index}_actions"],
                value[f"training_{index}_parent_actions"],
            )
            for index in range(training_count)
        )
        validation = tuple(
            G1TeacherTorqueEpisode(
                value[f"validation_{index}_observations"],
                value[f"validation_{index}_actions"],
                value[f"validation_{index}_parent_actions"],
            )
            for index in range(validation_count)
        )
    if not training or not validation:
        raise ValueError("neural torque worker dataset requires training and validation episodes")
    return training, validation


def _gpu_identity(physical_gpu: int) -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(values) != 2:
        raise RuntimeError("unexpected nvidia-smi identity response")
    return values[0], values[1]


if __name__ == "__main__":
    raise SystemExit(main())
