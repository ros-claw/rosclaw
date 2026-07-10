"""Tests for the dataset worker request/response schema."""

from __future__ import annotations

from rosclaw.integrations.lerobot.dataset_worker_schema import (
    DATASET_WORKER_SCHEMA_VERSION,
    DatasetValidationConfig,
    DatasetValidationResult,
    DatasetWorkerError,
    DatasetWorkerRequest,
    DatasetWorkerResponse,
    DatasetWriterConfig,
)


def test_writer_config_round_trip() -> None:
    cfg = DatasetWriterConfig(use_videos=False, video_codec="h264")
    restored = DatasetWriterConfig.from_dict(cfg.to_dict())
    assert restored.use_videos is False
    assert restored.video_codec == "h264"


def test_validation_config_round_trip() -> None:
    cfg = DatasetValidationConfig(load_after_write=True, sample_indices=[0, 1])
    restored = DatasetValidationConfig.from_dict(cfg.to_dict())
    assert restored.load_after_write is True
    assert restored.sample_indices == [0, 1]


def test_request_round_trip() -> None:
    req = DatasetWorkerRequest(
        op="export_dataset",
        normalized_episode_path="/tmp/norm.json",
        output_dir="/tmp/out",
        repo_id="local/rosclaw_test",
        fps=15.0,
        writer=DatasetWriterConfig(use_videos=False),
        validation=DatasetValidationConfig(sample_indices=[0]),
        timeout_sec=120,
    )
    data = req.to_dict()
    assert data["schema_version"] == DATASET_WORKER_SCHEMA_VERSION
    restored = DatasetWorkerRequest.from_dict(data)
    assert restored.op == "export_dataset"
    assert restored.fps == 15.0
    assert restored.writer.use_videos is False


def test_response_round_trip() -> None:
    resp = DatasetWorkerResponse(
        status="error",
        op="export_dataset",
        output_dir="/tmp/out",
        repo_id="local/rosclaw_test",
        error=DatasetWorkerError(code="dataset_create_failed", message="boom"),
        runtime={"python": "3.12"},
    )
    data = resp.to_dict()
    restored = DatasetWorkerResponse.from_dict(data)
    assert not restored.ok
    assert restored.error_code() == "dataset_create_failed"
    assert restored.error_message() == "boom"


def test_validation_result_defaults() -> None:
    result = DatasetValidationResult.from_dict({})
    assert not result.load_ok
    assert not result.index_ok
    assert result.sample_keys == []
