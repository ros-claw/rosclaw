"""Self-supervised G1 motion representation learned from audited kinematics.

MotionDecode has no motor torque or reward labels.  This module therefore
learns only a recurrent next-proprioception representation.  The representation
may initialize the joint state and projected-gravity columns of the SIM_ONLY torque actor;
it is never interpreted as an action policy or promotion evidence.

The pack builder consumes chain A source evidence: a MotionDecode registration
(content-addressed local snapshot) plus the kinematic ingest report produced by
``audit_motiondecode_snapshot``.  Raw motion is re-parsed through the canonical
parser and segmented through the repair module's exclusion spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]

from rosclaw.collective.sources.motiondecode.audit import (
    MotionDecodeAuditThresholds,
    load_g1_joint_contract,
)
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeFileRecord,
    MotionDecodeRegistration,
    verify_registered_files,
)
from rosclaw.collective.sources.motiondecode.parser import (
    CanonicalMotionEpisode,
    parse_motion_csv,
)
from rosclaw.collective.sources.motiondecode.repair import (
    MotionDecodeRepairResult,
    MotionRepairDisposition,
    clean_motiondecode_spans,
    repair_motiondecode_snapshot,
    replay_segmentation_repair,
)
from rosclaw.collective.sources.motiondecode.taxonomy import MotionFamily
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.g1_neural_torque import G1_NEURAL_TORQUE_OBSERVATIONS
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    hash_bytes,
    hash_json,
)

G1_MOTION_PRIOR_FEATURES = G1_NEURAL_TORQUE_OBSERVATIONS[:61]
G1_MOTION_PRIOR_FEATURE_DIM = len(G1_MOTION_PRIOR_FEATURES)
_PACK_SCHEMA = "rosclaw.collective.motiondecode_motion_prior_pack.v1"
_ARTIFACT_SCHEMA = "rosclaw.collective.g1_motion_prior.v1"
_INGEST_ARTIFACT_SCHEMA = "rosclaw.collective.motiondecode_ingest_artifact.v1"
_INGEST_REPORT_SCHEMA = "rosclaw.collective.motiondecode_ingest_report.v1"
_REPAIR_ARTIFACT_SCHEMA = "rosclaw.collective.motiondecode_repair_artifact.v1"
_MAX_REPORT_BYTES = 128 * 1024 * 1024
_MAX_PACK_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class G1MotionPriorMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    persistence_baseline_loss: float
    validation_improvement_fraction: float
    finite: bool
    schema_version: str = "rosclaw.collective.g1_motion_prior_metrics.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1MotionPriorArtifact:
    artifact_hash: str
    weights_hash: str
    pack_hash: str
    pilot_report_hash: str
    body_hash: str
    feature_names: tuple[str, ...]
    hidden_dim: int
    sequence_length: int
    observation_mean: np.ndarray
    observation_std: np.ndarray
    tensors: dict[str, np.ndarray]
    source_truth_level: str
    action_semantics: str
    activation_ceiling: str
    schema_version: str = _ARTIFACT_SCHEMA


class _MotionPredictor(nn.Module):
    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(len(G1_MOTION_PRIOR_FEATURES), hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, len(G1_MOTION_PRIOR_FEATURES))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(value)
        # At initialization this is exactly the strong 120 Hz persistence
        # baseline.  Learning is limited to a recurrent state delta.
        return value + self.head(encoded)


def build_motion_prior_pack(
    *,
    registration: MotionDecodeRegistration,
    ingest_report_path: Path,
    repair_report_path: Path | None = None,
    dataset_root: Path,
    model_path: Path,
    transfer_asset_root: Path | None = None,
    output_path: Path,
    sequence_length: int = 32,
    maximum_windows: int = 12_000,
    seed: int = 20260801,
    allowed_strata: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not 8 <= sequence_length <= 256:
        raise ValueError("motion-prior sequence length must be in [8, 256]")
    if not 8 <= maximum_windows <= 100_000:
        raise ValueError("motion-prior window limit must be in [8, 100000]")
    if not isinstance(registration, MotionDecodeRegistration):
        raise ValueError("motion-prior pack requires a MotionDecode registration")
    if not registration.catalog_audit.schema_valid:
        raise ValueError("motion-prior pack requires a schema-valid MotionDecode catalog")
    report_path = ingest_report_path.expanduser().resolve()
    artifact = _bounded_json(report_path, _MAX_REPORT_BYTES)
    if artifact.get("schema_version") != _INGEST_ARTIFACT_SCHEMA:
        raise ValueError("motion-prior pack requires a MotionDecode ingest artifact")
    report = artifact.get("report")
    if not isinstance(report, dict):
        raise ValueError("MotionDecode ingest artifact lacks a report object")
    if artifact.get("report_hash") != canonical_hash(report):
        raise ValueError("MotionDecode ingest report hash does not replay")
    if report.get("schema_version") != _INGEST_REPORT_SCHEMA:
        raise ValueError("motion-prior pack requires a MotionDecode ingest v1 report")
    if report.get("registration_hash") != registration.registration_hash:
        raise ValueError("MotionDecode ingest report is not bound to this registration")
    if report.get("source_manifest_hash") != registration.manifest.manifest_hash:
        raise ValueError("MotionDecode ingest report source manifest does not replay")
    if report.get("hardware_authorized") is not False:
        raise ValueError("MotionDecode ingest report hardware claim is missing or unsafe")
    if report.get("training_eligible") is not False:
        raise ValueError("MotionDecode ingest report eligibility claim is missing or unsafe")
    joint_limits, target_body_hash, model_file_hash = load_g1_joint_contract(model_path)
    if report.get("target_body_hash") != target_body_hash:
        raise ValueError("MotionDecode ingest target body hash does not match the requested model")
    if report.get("target_model_file_hash") != model_file_hash:
        raise ValueError("MotionDecode ingest target model file does not match the requested model")
    actor_body_hash = target_body_hash
    transfer_qualification: dict[str, Any] | None = None
    if transfer_asset_root is not None:
        from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets

        qualification = qualify_g1_assets(transfer_asset_root)
        qualification.require_eligible()
        if qualification.joint_names != G1_DDS_JOINT_NAMES:
            raise ValueError("motion-prior transfer body joint contract is incompatible")
        actor_body_hash = qualification.body_hash
        transfer_qualification = qualification.to_dict()
    repair_by_path: dict[str, Any] = {}
    repair_report_hash: str | None = None
    if repair_report_path is not None:
        repair_artifact = _bounded_json(
            repair_report_path.expanduser().resolve(), _MAX_REPORT_BYTES
        )
        if repair_artifact.get("schema_version") != _REPAIR_ARTIFACT_SCHEMA:
            raise ValueError("motion-prior pack requires a MotionDecode repair artifact")
        repair_value = repair_artifact.get("report")
        if not isinstance(repair_value, dict):
            raise ValueError("MotionDecode repair artifact lacks a report object")
        repair_report_hash = str(repair_artifact.get("report_hash", ""))
        if repair_report_hash != canonical_hash(repair_value):
            raise ValueError("MotionDecode repair report hash does not replay")
        if (
            repair_value.get("hardware_authorized") is not False
            or repair_value.get("activation_authorized") is not False
            or repair_value.get("training_eligible") is not False
        ):
            raise ValueError("MotionDecode repair evidence contains unsafe authorization claims")
        thresholds_value = report.get("thresholds")
        if not isinstance(thresholds_value, dict):
            raise ValueError("MotionDecode ingest report lacks audit thresholds")
        expected_threshold_fields = set(MotionDecodeAuditThresholds().to_dict())
        if set(thresholds_value) != expected_threshold_fields:
            raise ValueError("MotionDecode ingest audit thresholds are invalid")
        thresholds = MotionDecodeAuditThresholds(**thresholds_value)
        replayed_repair = repair_motiondecode_snapshot(
            registration,
            dataset_root,
            target_model_path=model_path,
            expected_ingest_report_hash=str(artifact["report_hash"]),
            thresholds=thresholds,
        )
        if replayed_repair.report_hash != repair_report_hash:
            raise ValueError("MotionDecode repair evidence does not replay from immutable input")
        repair_by_path = {result.relative_path: result for result in replayed_repair.results}
    clips = report.get("clips")
    if not isinstance(clips, list):
        raise ValueError("MotionDecode ingest report clips are missing")
    allowed = set(allowed_strata or ())
    known_strata = {family.value for family in MotionFamily}
    if allowed and (not allowed.issubset(known_strata) or not all(allowed_strata or ())):
        raise ValueError("motion-prior pack contains an unknown or empty stratum")

    root = dataset_root.expanduser().resolve()
    verified = verify_registered_files(registration, root)
    records = {
        item.relative_path: item
        for item in registration.manifest.files
        if item.relative_path.startswith("samples/")
    }
    joint_lower = np.asarray(
        [joint_limits[name][0] for name in G1_DDS_JOINT_NAMES], dtype=np.float64
    )
    joint_upper = np.asarray(
        [joint_limits[name][1] for name in G1_DDS_JOINT_NAMES], dtype=np.float64
    )

    candidates: list[tuple[str, str, int, str]] = []
    episode_hashes: dict[str, str] = {}
    audited: list[tuple[str, str, tuple[tuple[int, int], ...], str]] = []
    skipped_clips: list[dict[str, str]] = []
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("MotionDecode ingest report contains an invalid clip")
        relative = str(clip.get("relative_path", ""))
        repair_result = repair_by_path.get(relative)
        repaired_q1 = bool(
            repair_result is not None
            and repair_result.disposition is MotionRepairDisposition.REPAIRED_Q1
        )
        source_q1 = bool(
            clip.get("kinematic_valid") is True and isinstance(clip.get("episode_summary"), dict)
        )
        if not source_q1 and not repaired_q1:
            skipped_clips.append(
                {
                    "relative_path": relative,
                    "qualification": str(clip.get("qualification", "")),
                    "reason": (
                        "repair_not_q1" if repair_report_path is not None else "not_kinematic_valid"
                    ),
                }
            )
            continue
        record = records.get(relative)
        if record is None:
            raise ValueError(f"MotionDecode ingest clip is not registered: {relative}")
        expected_hash = str(clip.get("source_file_hash", ""))
        if expected_hash != record.content_hash:
            raise ValueError(f"MotionDecode payload changed after audit: {relative}")
        stratum = record.family.value
        if allowed and stratum not in allowed:
            continue
        episode = _load_motion_prior_episode(
            registration=registration,
            dataset_root=dataset_root,
            model_path=model_path,
            verified_path=verified[relative],
            record=record,
            target_body_hash=target_body_hash,
            repair_result=repair_result,
            repair_thresholds=(
                replayed_repair.thresholds if repair_report_path is not None else None
            ),
        )
        episode_hashes[relative] = (
            str(episode.derivation_manifest_hash) if repaired_q1 else expected_hash
        )
        spans = clean_motiondecode_spans(
            episode,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            minimum_frames=sequence_length + 1,
        )
        derivation_hash = episode_hashes[relative]
        episode_split_hash = hashlib.sha256(f"{derivation_hash}:{relative}".encode()).hexdigest()
        audited.append((relative, derivation_hash, spans, episode_split_hash))
    if len(audited) < 2:
        raise ValueError("motion-prior pack requires at least two source episodes")
    validation_episode_count = max(1, round(len(audited) * 0.2))
    validation_paths = {
        item[0] for item in sorted(audited, key=lambda item: item[3])[:validation_episode_count]
    }
    for relative, derivation_hash, spans, _ in audited:
        split = "validation" if relative in validation_paths else "training"
        for span_start, span_stop in spans:
            for start in range(span_start, span_stop - sequence_length, 16):
                score = hashlib.sha256(
                    f"{seed}:{derivation_hash}:{relative}:{start}".encode()
                ).hexdigest()
                candidates.append((score, relative, start, split))
    if not candidates:
        raise ValueError("MotionDecode ingest produced no clean motion-prior windows")
    training_target = int(maximum_windows * 0.8)
    validation_target = maximum_windows - training_target
    selected = [
        *sorted(item for item in candidates if item[3] == "training")[:training_target],
        *sorted(item for item in candidates if item[3] == "validation")[:validation_target],
    ]
    if sum(item[3] == "training" for item in selected) < 4:
        raise ValueError("motion-prior pack has too few training windows")
    if sum(item[3] == "validation" for item in selected) < 1:
        raise ValueError("motion-prior pack has too few validation windows")

    by_path: dict[str, list[tuple[int, str, str]]] = {}
    for score, relative, start, split in selected:
        by_path.setdefault(relative, []).append((start, split, score))
    values: dict[str, list[np.ndarray]] = {"training": [], "validation": []}
    selection_commitment: list[dict[str, Any]] = []
    for relative, starts in sorted(by_path.items()):
        record = records[relative]
        episode = _load_motion_prior_episode(
            registration=registration,
            dataset_root=dataset_root,
            model_path=model_path,
            verified_path=verified[relative],
            record=record,
            target_body_hash=target_body_hash,
            repair_result=repair_by_path.get(relative),
            repair_thresholds=(
                replayed_repair.thresholds if repair_report_path is not None else None
            ),
        )
        for start, split, score in starts:
            positions = episode.joint_position[start : start + sequence_length + 1]
            velocities = np.gradient(positions, 1.0 / episode.sample_rate_hz, axis=0)
            gravity = _projected_gravity(
                episode.root_quaternion[start : start + sequence_length + 1]
            )
            feature = np.concatenate((positions, velocities, gravity), axis=1).astype(np.float32)
            if feature.shape != (sequence_length + 1, len(G1_MOTION_PRIOR_FEATURES)):
                raise ValueError("motion-prior feature window has the wrong shape")
            if not np.all(np.isfinite(feature)):
                raise ValueError("motion-prior feature window is non-finite")
            values[split].append(feature)
            selection_commitment.append(
                {
                    "episode_hash": episode_hashes[relative],
                    "start": start,
                    "split": split,
                    "score": score,
                }
            )
    training = np.asarray(values["training"], dtype=np.float32)
    validation = np.asarray(values["validation"], dtype=np.float32)
    mean = training.reshape(-1, training.shape[-1]).mean(axis=0).astype(np.float32)
    std = np.maximum(training.reshape(-1, training.shape[-1]).std(axis=0), 1e-3).astype(np.float32)
    training = np.clip((training - mean) / std, -10.0, 10.0).astype(np.float32)
    validation = np.clip((validation - mean) / std, -10.0, 10.0).astype(np.float32)
    output = output_path.expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("motion-prior pack output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output,
        training=training,
        validation=validation,
        observation_mean=mean,
        observation_std=std,
    )
    pack_hash = hash_bytes(output.read_bytes())
    metadata = {
        "schema_version": _PACK_SCHEMA,
        "pack_hash": pack_hash,
        "pilot_report_hash": hash_bytes(report_path.read_bytes()),
        "ingest_report_hash": str(artifact["report_hash"]),
        "repair_report_hash": repair_report_hash,
        "registration_hash": registration.registration_hash,
        "source_manifest_hash": registration.manifest.manifest_hash,
        "body_hash": actor_body_hash,
        "kinematic_body_hash": target_body_hash,
        "target_model_file_hash": model_file_hash,
        "transfer_body_qualification": transfer_qualification,
        "transfer_contract": (
            "exact_29_joint_feature_semantics_only_no_dynamics_transfer"
            if transfer_qualification is not None
            else "kinematic_body_only"
        ),
        "feature_names": list(G1_MOTION_PRIOR_FEATURES),
        "sequence_length": sequence_length,
        "training_windows": len(training),
        "validation_windows": len(validation),
        "source_episode_count": len(by_path),
        "repaired_source_episode_count": sum(
            result.disposition is MotionRepairDisposition.REPAIRED_Q1
            and result.relative_path in by_path
            for result in repair_by_path.values()
        ),
        "allowed_strata": sorted(allowed) if allowed else ["all"],
        "skipped_clips": skipped_clips,
        "selection_commitment": hash_json(selection_commitment),
        "source_truth_level": "T4",
        "action_semantics": "ABSENT",
        "reward_semantics": "ABSENT",
        "raw_data_exported": False,
        "redistribution_permitted": False,
    }
    _atomic_json(output.with_suffix(".json"), metadata)
    return metadata


def _load_motion_prior_episode(
    *,
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    model_path: Path,
    verified_path: Path,
    record: MotionDecodeFileRecord,
    target_body_hash: str,
    repair_result: MotionDecodeRepairResult | None,
    repair_thresholds: MotionDecodeAuditThresholds | None,
) -> CanonicalMotionEpisode:
    if (
        repair_result is not None
        and repair_result.disposition is MotionRepairDisposition.REPAIRED_Q1
    ):
        manifest = repair_result.repair_manifest
        if manifest is None or repair_thresholds is None:
            raise ValueError("repaired MotionDecode Q1 clip lacks replay inputs")
        return replay_segmentation_repair(
            registration,
            dataset_root,
            target_model_path=model_path,
            manifest=manifest,
            thresholds=repair_thresholds,
        )
    return parse_motion_csv(
        verified_path,
        source_manifest_hash=registration.manifest.manifest_hash,
        expected_file_hash=record.content_hash,
        target_body_hash=target_body_hash,
        sample_rate_hz=registration.manifest.sample_rate_hz,
    )


def train_motion_prior_worker(
    *,
    pack_path: Path,
    output_dir: Path,
    device: str,
    seed: int,
    epochs: int = 10,
    hidden_dim: int = 96,
    batch_size: int = 256,
) -> dict[str, Any]:
    if not 2 <= epochs <= 100 or not 8 <= hidden_dim <= 1024 or batch_size <= 0:
        raise ValueError("invalid motion-prior learner configuration")
    pack, metadata = _load_pack(pack_path)
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA motion-prior worker requested without CUDA")
    target = torch.device(device)
    model = _MotionPredictor(hidden_dim=hidden_dim).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    training = pack["training"]
    validation = pack["validation"]
    rng = np.random.default_rng(seed)
    baseline = _smooth_l1_numpy(validation[:, :-1] - validation[:, 1:])
    metrics: list[G1MotionPriorMetrics] = []
    for epoch in range(epochs):
        order = rng.permutation(len(training))
        losses: list[float] = []
        model.train()
        for offset in range(0, len(order), batch_size):
            batch = torch.as_tensor(
                training[order[offset : offset + batch_size]],
                dtype=torch.float32,
                device=target,
            )
            prediction = model(batch[:, :-1])
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch[:, 1:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        validation_loss = _validation_loss(model, validation, target, batch_size)
        training_loss = float(sum(losses) / len(losses))
        metrics.append(
            G1MotionPriorMetrics(
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
                persistence_baseline_loss=baseline,
                validation_improvement_fraction=(baseline - validation_loss) / max(baseline, 1e-12),
                finite=all(math.isfinite(value) for value in (training_loss, validation_loss)),
            )
        )
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = model.state_dict()
    tensor_values = {
        "gru.weight_ih_l0": state["gru.weight_ih_l0"].detach().cpu().numpy(),
        "gru.weight_hh_l0": state["gru.weight_hh_l0"].detach().cpu().numpy(),
        "gru.bias_ih_l0": state["gru.bias_ih_l0"].detach().cpu().numpy(),
        "gru.bias_hh_l0": state["gru.bias_hh_l0"].detach().cpu().numpy(),
        "head.weight": state["head.weight"].detach().cpu().numpy(),
        "head.bias": state["head.bias"].detach().cpu().numpy(),
        "observation_mean": pack["observation_mean"],
        "observation_std": pack["observation_std"],
    }
    weights_path = output / "motion-prior-weights.npz"
    _atomic_npz(weights_path, **tensor_values)
    weights_hash = hash_bytes(weights_path.read_bytes())
    artifact = {
        "schema_version": _ARTIFACT_SCHEMA,
        "weights_file": weights_path.name,
        "weights_hash": weights_hash,
        "pack_hash": metadata["pack_hash"],
        "pilot_report_hash": metadata["pilot_report_hash"],
        "body_hash": metadata["body_hash"],
        "feature_names": list(G1_MOTION_PRIOR_FEATURES),
        "hidden_dim": hidden_dim,
        "sequence_length": metadata["sequence_length"],
        "source_truth_level": "T4",
        "objective": "SELF_SUPERVISED_NEXT_PROPRIOCEPTION",
        "action_semantics": "ABSENT",
        "activation_ceiling": "SIM_ONLY_REPRESENTATION_INITIALIZATION",
        "hardware_authorized": False,
        "promotion_evidence_eligible": False,
        "seed": seed,
        "device": str(target),
        "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu",
        "metrics": [value.to_dict() for value in metrics],
    }
    artifact["artifact_hash"] = hash_json(artifact)
    _atomic_json(output / "motion-prior-artifact.json", artifact)
    return artifact


def run_four_gpu_motion_prior(
    *,
    pack_path: Path,
    output_dir: Path,
    epochs: int = 10,
    hidden_dim: int = 96,
    batch_size: int = 256,
    base_seed: int = 8200,
) -> dict[str, Any]:
    if torch.cuda.device_count() < 4:
        raise RuntimeError("four-GPU motion-prior run requires four visible CUDA devices")
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[int, int, Path, subprocess.Popen[str]]] = []
    for physical_gpu in range(4):
        seed = base_seed + physical_gpu
        worker_root = root / f"gpu-{physical_gpu}"
        command = [
            sys.executable,
            "-m",
            "rosclaw.collective.sources.motiondecode.motion_prior",
            "worker",
            "--pack",
            str(pack_path.expanduser().resolve()),
            "--output-dir",
            str(worker_root),
            "--device",
            "cuda:0",
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--hidden-dim",
            str(hidden_dim),
            "--batch-size",
            str(batch_size),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        process = subprocess.Popen(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((physical_gpu, seed, worker_root, process))
    workers: list[dict[str, Any]] = []
    failures: list[str] = []
    for physical_gpu, seed, worker_root, process in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            failures.append(
                f"gpu={physical_gpu},returncode={process.returncode},stderr={stderr[-1000:]}"
            )
            continue
        try:
            artifact = _bounded_json(worker_root / "motion-prior-artifact.json", _MAX_REPORT_BYTES)
            metrics = artifact.get("metrics")
            if (
                artifact.get("schema_version") != _ARTIFACT_SCHEMA
                or not isinstance(metrics, list)
                or not metrics
                or not isinstance(metrics[-1], dict)
            ):
                raise ValueError("worker artifact metadata is incomplete")
            final = metrics[-1]
            validation_loss = float(final["validation_loss"])
            persistence_loss = float(final["persistence_baseline_loss"])
            improvement = float(final["validation_improvement_fraction"])
            if not all(
                math.isfinite(value) for value in (validation_loss, persistence_loss, improvement)
            ):
                raise ValueError("worker metrics are non-finite")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            failures.append(f"gpu={physical_gpu},invalid_artifact={type(exc).__name__}:{exc}")
            continue
        workers.append(
            {
                "physical_gpu": physical_gpu,
                "visible_device": "cuda:0",
                "seed": seed,
                "artifact_hash": artifact["artifact_hash"],
                "artifact_path": str(worker_root / "motion-prior-artifact.json"),
                "validation_loss": validation_loss,
                "persistence_baseline_loss": persistence_loss,
                "improvement_fraction": improvement,
                "stdout": stdout.strip(),
            }
        )
    selected = (
        min(workers, key=lambda value: (value["validation_loss"], value["seed"]))
        if workers
        else None
    )
    quality_gate_passed = bool(
        selected is not None
        and math.isfinite(float(selected["validation_loss"]))
        and float(selected["improvement_fraction"]) >= 0.02
    )
    report = {
        "schema_version": "rosclaw.collective.g1_motion_prior_four_gpu.v1",
        "pack_hash": _load_pack(pack_path)[1]["pack_hash"],
        "requested_physical_gpus": [0, 1, 2, 3],
        "workers": workers,
        "failures": failures,
        "selected": selected,
        "four_physical_gpus_exercised": len(workers) == 4 and not failures,
        "quality_gate": {
            "metric": "held_out_episode_smooth_l1_vs_persistence",
            "minimum_improvement_fraction": 0.02,
            "passed": quality_gate_passed,
        },
        "decision": (
            "REPRESENTATION_CANDIDATE"
            if len(workers) == 4 and not failures and quality_gate_passed
            else "REJECTED"
        ),
        "hardware_authorized": False,
        "promotion_evidence_eligible": False,
    }
    _atomic_json(root / "four-gpu-report.json", report)
    return report


def load_g1_motion_prior_artifact(path: Path) -> G1MotionPriorArtifact:
    metadata_path = path.expanduser().resolve()
    metadata = _bounded_json(metadata_path, _MAX_ARTIFACT_BYTES)
    if metadata.get("schema_version") != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported G1 motion-prior artifact")
    committed = str(metadata.get("artifact_hash", ""))
    unsigned = {key: value for key, value in metadata.items() if key != "artifact_hash"}
    if committed != hash_json(unsigned):
        raise ValueError("G1 motion-prior artifact metadata hash mismatch")
    weights_name = str(metadata.get("weights_file", ""))
    weights_relative = Path(weights_name)
    if (
        not weights_name
        or weights_relative.is_absolute()
        or ".." in weights_relative.parts
        or weights_relative.name != weights_name
    ):
        raise ValueError("G1 motion-prior weights file must be a sibling file name")
    weights_path = metadata_path.parent / weights_relative
    if not weights_path.is_file() or weights_path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("G1 motion-prior weights are missing or too large")
    if hash_bytes(weights_path.read_bytes()) != metadata.get("weights_hash"):
        raise ValueError("G1 motion-prior weights hash mismatch")
    with np.load(weights_path, allow_pickle=False) as archive:
        tensors = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    hidden_dim = int(metadata.get("hidden_dim", 0))
    expected = {
        "gru.weight_ih_l0": (3 * hidden_dim, G1_MOTION_PRIOR_FEATURE_DIM),
        "gru.weight_hh_l0": (3 * hidden_dim, hidden_dim),
        "gru.bias_ih_l0": (3 * hidden_dim,),
        "gru.bias_hh_l0": (3 * hidden_dim,),
        "head.weight": (G1_MOTION_PRIOR_FEATURE_DIM, hidden_dim),
        "head.bias": (G1_MOTION_PRIOR_FEATURE_DIM,),
        "observation_mean": (G1_MOTION_PRIOR_FEATURE_DIM,),
        "observation_std": (G1_MOTION_PRIOR_FEATURE_DIM,),
    }
    if set(tensors) != set(expected):
        raise ValueError("G1 motion-prior tensor set is invalid")
    for name, shape in expected.items():
        if tensors[name].shape != shape or not np.all(np.isfinite(tensors[name])):
            raise ValueError(f"G1 motion-prior tensor {name} is invalid")
    if np.any(tensors["observation_std"] <= 1e-6):
        raise ValueError("G1 motion-prior observation scale is invalid")
    feature_names = tuple(metadata.get("feature_names", ()))
    if feature_names != G1_MOTION_PRIOR_FEATURES:
        raise ValueError("G1 motion-prior feature contract mismatch")
    return G1MotionPriorArtifact(
        artifact_hash=str(metadata["artifact_hash"]),
        weights_hash=str(metadata["weights_hash"]),
        pack_hash=str(metadata["pack_hash"]),
        pilot_report_hash=str(metadata["pilot_report_hash"]),
        body_hash=str(metadata["body_hash"]),
        feature_names=feature_names,
        hidden_dim=hidden_dim,
        sequence_length=int(metadata["sequence_length"]),
        observation_mean=tensors["observation_mean"],
        observation_std=tensors["observation_std"],
        tensors=tensors,
        source_truth_level=str(metadata["source_truth_level"]),
        action_semantics=str(metadata["action_semantics"]),
        activation_ceiling=str(metadata["activation_ceiling"]),
    )


def _load_pack(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not 1 <= resolved.stat().st_size <= _MAX_PACK_BYTES:
        raise ValueError("motion-prior pack is missing or too large")
    metadata = _bounded_json(resolved.with_suffix(".json"), _MAX_REPORT_BYTES)
    if metadata.get("schema_version") != _PACK_SCHEMA:
        raise ValueError("unsupported motion-prior pack schema")
    if hash_bytes(resolved.read_bytes()) != metadata.get("pack_hash"):
        raise ValueError("motion-prior pack hash mismatch")
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != {"training", "validation", "observation_mean", "observation_std"}:
            raise ValueError("motion-prior pack tensor set is invalid")
        values = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    sequence_length = int(metadata["sequence_length"])
    for split in ("training", "validation"):
        if values[split].ndim != 3 or values[split].shape[1:] != (
            sequence_length + 1,
            G1_MOTION_PRIOR_FEATURE_DIM,
        ):
            raise ValueError(f"motion-prior {split} tensor shape is invalid")
    if values["observation_mean"].shape != (G1_MOTION_PRIOR_FEATURE_DIM,) or values[
        "observation_std"
    ].shape != (G1_MOTION_PRIOR_FEATURE_DIM,):
        raise ValueError("motion-prior normalization shape is invalid")
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("motion-prior pack contains non-finite values")
    return values, metadata


def _validation_loss(
    model: _MotionPredictor,
    validation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(validation), batch_size):
            batch = torch.as_tensor(validation[offset : offset + batch_size], device=device)
            loss = torch.nn.functional.smooth_l1_loss(
                model(batch[:, :-1]), batch[:, 1:], reduction="sum"
            )
            total += float(loss.item())
    return total / (len(validation) * (validation.shape[1] - 1) * validation.shape[2])


def _projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4 or np.any(norms <= 1e-12):
        raise ValueError("motion-prior root quaternion is invalid")
    quaternion = quaternion / norms
    # Rotate world gravity by the inverse body quaternion, matching the
    # direct-torque observation contract.
    vector = np.broadcast_to(np.asarray((0.0, 0.0, -1.0)), (len(quaternion), 3))
    inverse_vector = -quaternion[:, 1:]
    uv = np.cross(inverse_vector, vector)
    uuv = np.cross(inverse_vector, uv)
    return vector + 2.0 * (quaternion[:, :1] * uv + uuv)


def _smooth_l1_numpy(delta: np.ndarray) -> float:
    absolute = np.abs(delta)
    return float(np.mean(np.where(absolute < 1.0, 0.5 * np.square(absolute), absolute - 0.5)))


def _bounded_json(path: Path, maximum: int) -> dict[str, Any]:
    if not path.is_file() or not 1 <= path.stat().st_size <= maximum:
        raise ValueError(f"bounded JSON is missing or too large: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--pack", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--epochs", type=int, required=True)
    worker.add_argument("--hidden-dim", type=int, required=True)
    worker.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.command != "worker":
        parser.print_help()
        return 1
    result = train_motion_prior_worker(
        pack_path=args.pack,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )
    print(json.dumps({"artifact_hash": result["artifact_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
