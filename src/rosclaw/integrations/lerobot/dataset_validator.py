"""Validate a LeRobotDataset that was written by the P2 export worker.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It delegates actual loading/indexing to the isolated LeRobot runtime
worker so the validation command can be run from any ROSClaw Python version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.dataset_extension_schema import read_extension_schema
from rosclaw.integrations.lerobot.dataset_sidecar import write_episodes_parquet
from rosclaw.integrations.lerobot.dataset_vocab import read_vocab
from rosclaw.integrations.lerobot.dataset_worker_runner import (
    LeRobotDatasetWorkerRunner,
    run_dataset_dataloader_smoke,
)
from rosclaw.integrations.lerobot.dataset_worker_schema import (
    DatasetValidationConfig,
    DatasetValidationResult,
    DatasetWorkerRequest,
    DatasetWorkerResponse,
)


def validate_dataset(
    output_dir: Path | str,
    repo_id: str,
    *,
    sample_indices: list[int] | None = None,
    level: str = "load",
    timeout_sec: int = 300,
) -> DatasetValidationResult:
    """Validate an existing LeRobotDataset by loading it in the worker runtime.

    Levels:
      - ``structural``: only check that the output directory exists and has
        ``meta/info.json``.
      - ``load``: load the dataset and verify frame/episodes counts.
      - ``dataloader``: also run a DataLoader smoke test.
      - ``rich``: also check ROSClaw sidecars (schema, vocab, episodes).

    Returns a ``DatasetValidationResult`` even if the worker reports an error,
    so callers can distinguish load failures from index failures.
    """
    output_dir = Path(output_dir)
    level = level.lower()

    if level == "structural":
        return _validate_structural(output_dir)

    dataloader = level in ("dataloader", "rich")
    request = DatasetWorkerRequest(
        op="validate_dataset",
        output_dir=str(output_dir),
        repo_id=repo_id,
        validation=DatasetValidationConfig(
            load_after_write=True,
            sample_indices=sample_indices or [0],
            dataloader=dataloader,
            level=level,
        ),
        timeout_sec=timeout_sec,
    )
    runner = LeRobotDatasetWorkerRunner(timeout_sec=timeout_sec)
    response = runner.run(request)
    result = response.validation

    if level == "rich":
        sidecar_issues = _validate_sidecars(output_dir)
        if sidecar_issues:
            result.error = {
                "code": "sidecar_validation_failed",
                "message": "ROSClaw sidecar validation failed.",
                "details": "; ".join(sidecar_issues),
            }
        elif result.error is None:
            result.index_ok = result.index_ok and True

    return result


def _validate_structural(output_dir: Path) -> DatasetValidationResult:
    meta_info = output_dir / "meta" / "info.json"
    return DatasetValidationResult(
        load_ok=meta_info.exists(),
        index_ok=meta_info.exists(),
        num_frames=None,
        num_episodes=None,
        sample_keys=[],
    )


def _validate_sidecars(output_dir: Path) -> list[str]:
    """Return a list of issues found in ROSClaw sidecar files."""
    issues: list[str] = []
    schema = read_extension_schema(output_dir)
    if schema is None:
        issues.append("meta/rosclaw/schema.json is missing")
    vocab = read_vocab(output_dir)
    if vocab is None:
        issues.append("meta/rosclaw/vocab.json is missing")
    parquet_path = output_dir / "meta" / "rosclaw" / "episodes.parquet"
    jsonl_path = output_dir / "meta" / "rosclaw" / "episodes.jsonl"
    if not parquet_path.exists() and not jsonl_path.exists():
        issues.append("meta/rosclaw/episodes sidecar is missing")
    return issues


def run_dataloader_smoke(
    output_dir: Path | str,
    repo_id: str,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    timeout_sec: int = 300,
) -> DatasetValidationResult:
    """Run a DataLoader smoke test against an existing dataset."""
    response = run_dataset_dataloader_smoke(
        str(output_dir),
        repo_id,
        batch_size=batch_size,
        num_workers=num_workers,
        timeout_sec=timeout_sec,
    )
    return response.validation


def format_validation_result(result: DatasetValidationResult) -> dict[str, Any]:
    """Return a CLI/JSON friendly dict from a validation result."""
    out: dict[str, Any] = {
        "load_ok": result.load_ok,
        "index_ok": result.index_ok,
        "num_frames": result.num_frames,
        "num_episodes": result.num_episodes,
        "sample_keys": result.sample_keys,
        "sample_image_keys": result.sample_image_keys,
        "dataloader_ok": result.dataloader_ok,
        "batch_keys": result.batch_keys,
        "batch_shapes": result.batch_shapes,
        "error": result.error,
    }
    return out


__all__ = [
    "DatasetValidationResult",
    "format_validation_result",
    "run_dataloader_smoke",
    "validate_dataset",
]
