from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw.continual.learner_backend import (
    LearnerBackendContract,
    LearnerCapability,
    LearnerRunEvidence,
    LearnerRunStatus,
)
from tests.continual.helpers import digest


def _backend() -> LearnerBackendContract:
    return LearnerBackendContract(
        backend_id="protomotions3",
        backend_version="3.0",
        source_url="https://github.com/NVlabs/ProtoMotions",
        source_commit="a" * 40,
        license_id="Apache-2.0",
        capabilities=(
            LearnerCapability.MOTION_TRACKING,
            LearnerCapability.MULTI_GPU,
        ),
        supported_body_ids=("unitree.g1.29dof",),
        training_available=True,
        inference_available=True,
    )


def test_backend_contract_is_hashed_and_sim_only() -> None:
    backend = _backend()

    assert backend.contract_hash.startswith("sha256:")
    assert backend.to_dict()["capabilities"] == ["motion_tracking", "multi_gpu"]
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(backend, hardware_execution_allowed=True)


def test_completed_run_requires_bound_candidate_and_nonzero_work() -> None:
    backend = _backend()
    run = LearnerRunEvidence(
        run_id="s33.foundation.smoke",
        backend_contract_hash=backend.contract_hash,
        body_hash=digest("g1-body"),
        dataset_manifest_hash=digest("motion-atlas"),
        config_hash=digest("config"),
        seed_commitment_hash=digest("seeds"),
        physics_backend="mujoco",
        device_ids=(0, 1, 2, 3),
        world_steps=4096,
        sample_count=1024,
        status=LearnerRunStatus.COMPLETED,
        candidate_artifact_hash=digest("candidate"),
    )

    assert run.evidence_hash.startswith("sha256:")
    assert run.to_dict()["device_ids"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="requires samples"):
        replace(run, world_steps=0)


def test_noncompleted_run_cannot_publish_candidate() -> None:
    with pytest.raises(ValueError, match="only a completed run"):
        LearnerRunEvidence(
            run_id="s33.failed",
            backend_contract_hash=digest("backend"),
            body_hash=digest("body"),
            dataset_manifest_hash=digest("dataset"),
            config_hash=digest("config"),
            seed_commitment_hash=digest("seed"),
            physics_backend="mujoco",
            device_ids=(),
            world_steps=0,
            sample_count=0,
            status=LearnerRunStatus.FAILED,
            candidate_artifact_hash=digest("untrusted"),
        )
