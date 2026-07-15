"""ROSClaw-side runner for the LeRobot dataset export worker.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It writes a JSON request, spawns the LeRobot runtime Python to run
``dataset_worker_main.py``, reads the JSON response, and translates errors into
structured ``LeRobotWorkerErrorCode`` values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.body_snapshot import include_body_snapshot
from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.dataset_events import write_events_parquet
from rosclaw.integrations.lerobot.dataset_extension_schema import (
    ROSCLAW_EXTENSION_SCHEMA_VERSION,
    ExtensionSchema,
    write_extension_schema,
)
from rosclaw.integrations.lerobot.dataset_profile import resolve_profile
from rosclaw.integrations.lerobot.dataset_sidecar import write_episodes_parquet
from rosclaw.integrations.lerobot.dataset_sync import write_sync_stats_parquet
from rosclaw.integrations.lerobot.dataset_units import (
    write_feature_names_json,
    write_units_json,
)
from rosclaw.integrations.lerobot.dataset_vocab import build_rosclaw_vocab
from rosclaw.integrations.lerobot.dataset_worker_schema import (
    DatasetValidationConfig,
    DatasetWorkerError,
    DatasetWorkerRequest,
    DatasetWorkerResponse,
    DatasetWriterConfig,
)
from rosclaw.integrations.lerobot.runtime import inspect_lerobot_runtime
from rosclaw.integrations.lerobot.schemas import LeRobotWorkerErrorCode


class LeRobotDatasetWorkerRunner:
    """One-shot subprocess dataset worker runner.

    The runner resolves the configured LeRobot runtime (or falls back to the
    current interpreter if LeRobot is importable in-process), writes a temp
    request JSON, runs ``dataset_worker_main.py``, and returns a
    ``DatasetWorkerResponse``.
    """

    def __init__(self, timeout_sec: int = 300, *, debug: bool | None = None):
        self.timeout_sec = timeout_sec
        self.debug = debug if debug is not None else bool(
            os.environ.get("ROSCLAW_LEROBOT_DEBUG_WORKER", "")
        )
        self._temp_dir: Path | None = None
        self.worker_script = Path(__file__).with_name("dataset_worker_main.py")

    def _resolve_runtime(self) -> tuple[str, str | None]:
        """Return the (python_executable, hf_endpoint) to use."""
        configured = get_configured_lerobot_runtime()
        if configured and configured.get("subprocess_available"):
            python = configured.get("python_executable")
            if python and Path(python).exists():
                return str(python), configured.get("hf_endpoint")

        import sys

        info = inspect_lerobot_runtime(sys.executable)
        if info.state in ("ready", "degraded") and info.lerobot_version is not None:
            return str(info.python_executable), None

        raise RuntimeNotConfiguredError(
            "No LeRobot runtime configured and current interpreter cannot import LeRobot."
        )

    def run(self, request: DatasetWorkerRequest) -> DatasetWorkerResponse:
        """Execute the dataset worker for ``request`` and return the response."""
        try:
            python_executable, hf_endpoint = self._resolve_runtime()
        except RuntimeNotConfiguredError as exc:
            return _runtime_error_response(
                request, LeRobotWorkerErrorCode.RUNTIME_NOT_CONFIGURED, str(exc)
            )

        if not self.worker_script.exists():
            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_SCRIPT_MISSING,
                f"Worker script not found: {self.worker_script}",
            )

        self._temp_dir = Path(tempfile.mkdtemp(prefix="rosclaw_lerobot_dataset_worker_"))
        request_path = self._temp_dir / "request.json"
        response_path = self._temp_dir / "response.json"

        try:
            normalized_path_in_temp = self._copy_normalized_bundle(request.normalized_episode_path)
            request_copy = DatasetWorkerRequest.from_dict(request.to_dict())
            request_copy.normalized_episode_path = str(normalized_path_in_temp)

            request_path.write_text(
                json.dumps(request_copy.to_dict(), indent=2),
                encoding="utf-8",
            )
            env = self._build_env(request, hf_endpoint)
            cmd = [
                python_executable,
                str(self.worker_script),
                "--request-json",
                str(request_path),
                "--output-json",
                str(response_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=env,
                check=False,
            )

            if not response_path.exists() and result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                return _runtime_error_response(
                    request,
                    LeRobotWorkerErrorCode.WORKER_PROCESS_FAILED,
                    f"Worker process exited with code {result.returncode}.",
                    details,
                )

            if response_path.exists():
                try:
                    raw = json.loads(response_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    return _runtime_error_response(
                        request,
                        LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
                        f"Worker response is not valid JSON: {exc}",
                        response_path.read_text(encoding="utf-8")[:2000],
                    )

                if not isinstance(raw, dict):
                    return _runtime_error_response(
                        request,
                        LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
                        "Worker response is not a JSON object.",
                        str(raw)[:2000],
                    )

                response = DatasetWorkerResponse.from_dict(raw)
                if result.returncode != 0 and response.ok:
                    # Worker returned an error exit code but a structurally ok
                    # response: treat it as an error so callers do not ignore it.
                    response.status = "error"
                    if response.error is None:
                        response.error = DatasetWorkerError(
                            code="worker_process_failed",
                            message=f"Worker process exited with code {result.returncode}.",
                            details=result.stderr.strip() or result.stdout.strip(),
                        )
                if response.ok and request.op == "export_dataset":
                    response = self._enforce_profile_completeness(request, response)
                    if not response.ok:
                        return response
                    response = self._write_sidecars(request, response)
                return response

            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
                "Worker did not write a response JSON file.",
                result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_TIMEOUT,
                f"Worker timed out after {self.timeout_sec}s.",
            )
        except Exception as exc:  # noqa: BLE001
            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_PROCESS_FAILED,
                f"Failed to run worker: {exc}",
            )
        finally:
            self._cleanup()

    def _enforce_profile_completeness(
        self,
        request: DatasetWorkerRequest,
        response: DatasetWorkerResponse,
    ) -> DatasetWorkerResponse:
        """Fail exports whose profile was not fully satisfied unless allowed."""
        requested = list(request.feature_groups)
        written = list(response.feature_groups_written or [])
        missing = [g for g in requested if g not in written]
        if not missing:
            response.profile_satisfied = True
            response.requested_feature_groups = requested
            response.written_feature_groups = written
            response.missing_feature_groups = []
            return response

        response.missing_feature_groups = missing
        response.requested_feature_groups = requested
        response.written_feature_groups = written
        response.profile_satisfied = False
        if request.allow_partial:
            response.warnings.append(
                f"Profile not fully satisfied. Missing groups: {', '.join(missing)}."
            )
            return response

        return _runtime_error_response(
            request,
            LeRobotWorkerErrorCode.WORKER_PROCESS_FAILED,
            f"Profile not fully satisfied. Missing groups: {', '.join(missing)}.",
            "Set --allow-partial to export anyway with a partial profile.",
        )

    def _write_sidecars(self, request: DatasetWorkerRequest, response: DatasetWorkerResponse) -> DatasetWorkerResponse:
        """Write ROSClaw sidecars after a successful export."""
        output_root = Path(request.output_dir)
        if not output_root.exists():
            return response

        try:
            with open(request.normalized_episode_path, encoding="utf-8") as f:
                episode_data = json.load(f)
        except Exception:  # noqa: BLE001
            return response

        feature_groups = list(request.feature_groups)
        if not feature_groups:
            feature_groups = list(response.feature_groups_written or [])

        sidecar_files: list[str] = list(response.files.sidecar_files or [])

        try:
            write_extension_schema(
                ExtensionSchema(
                    schema_version=ROSCLAW_EXTENSION_SCHEMA_VERSION,
                    required_features=["observation.state", "action"],
                    optional_feature_groups=feature_groups,
                    rosclaw_fields=[
                        k
                        for k in request.features
                        if not k.startswith("include_body") and k != "body_snapshot_mode"
                    ],
                    dataset_format="lerobot_v3",
                ),
                output_root,
            )
            sidecar_files.append("meta/rosclaw/schema.json")
        except Exception as exc:  # noqa: BLE001
            response.warnings.append(f"Could not write extension schema: {exc}")

        try:
            vocab = build_rosclaw_vocab(feature_groups)
            from rosclaw.integrations.lerobot.dataset_vocab import write_vocab

            write_vocab(vocab, output_root)
            sidecar_files.append("meta/rosclaw/vocab.json")
        except Exception as exc:  # noqa: BLE001
            response.warnings.append(f"Could not write vocab: {exc}")

        try:
            from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode

            episode = NormalizedPracticeEpisode.from_dict(episode_data)
            write_episodes_parquet([episode], output_root)
            sidecar_files.append("meta/rosclaw/episodes.parquet")
        except Exception as exc:  # noqa: BLE001
            response.warnings.append(f"Could not write episodes sidecar: {exc}")

        try:
            from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode

            episode = NormalizedPracticeEpisode.from_dict(episode_data)
            write_events_parquet([episode], output_root)
            sidecar_files.append("meta/rosclaw/events.parquet")
        except Exception as exc:  # noqa: BLE001
            response.warnings.append(f"Could not write events sidecar: {exc}")

        try:
            from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode

            episode = NormalizedPracticeEpisode.from_dict(episode_data)
            write_sync_stats_parquet([episode], output_root)
            sidecar_files.append("meta/rosclaw/sync_stats.parquet")
        except Exception as exc:  # noqa: BLE001
            response.warnings.append(f"Could not write sync_stats sidecar: {exc}")

        try:
            feature_keys = sorted(response.dataset.features.keys())
            write_units_json(feature_keys, output_root)
            write_feature_names_json(feature_keys, output_root)
            sidecar_files.extend(["meta/rosclaw/units.json", "meta/rosclaw/feature_names.json"])
        except Exception as exc:  # noqa: BLE001
            response.warnings.append(f"Could not write units/feature_names sidecars: {exc}")

        body_yaml_path = episode_data.get("robot", {}).get("body_yaml_path")
        if request.features.get("include_body_snapshot") and body_yaml_path:
            try:
                include_body_snapshot(
                    output_root,
                    body_yaml_path,
                    mode=request.features.get("body_snapshot_mode", "sanitized"),
                    acknowledge_sensitive=bool(
                        request.features.get("acknowledge_sensitive_body_data", False)
                    ),
                )
                sidecar_files.append("meta/rosclaw/body_snapshots/manifest.json")
            except Exception as exc:  # noqa: BLE001
                response.warnings.append(f"Could not write body snapshot: {exc}")

        response.sidecar_files = sorted(set(sidecar_files))
        response.extension_schema = ROSCLAW_EXTENSION_SCHEMA_VERSION
        return response

    def _copy_normalized_bundle(self, normalized_episode_path: str) -> Path:
        """Copy the normalized episode JSON and referenced images into the temp dir.

        The worker resolves image paths relative to the JSON file's parent
        directory, so we mirror the original directory structure under the temp
        dir and update the JSON path accordingly.
        """
        if not self._temp_dir:
            raise RuntimeError("Temp dir not created")

        if not normalized_episode_path:
            return Path(".")

        src_json = Path(normalized_episode_path)
        if not src_json.exists():
            raise FileNotFoundError(f"Normalized episode file not found: {src_json}")

        dest_json = self._temp_dir / "normalized_episode.json"
        shutil.copy2(src_json, dest_json)

        try:
            data = json.loads(src_json.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return dest_json

        base_dir = src_json.parent
        metadata = data.get("metadata") or {}
        source_dir = metadata.get("source_dir")
        if source_dir:
            candidate = Path(source_dir)
            if candidate.exists():
                base_dir = candidate
        for frame in data.get("frames", []):
            images = frame.get("observation", {}).get("images", {})
            for _camera_name, rel_path in images.items():
                src_image = base_dir / rel_path
                if src_image.exists():
                    dest_image = self._temp_dir / rel_path
                    dest_image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_image, dest_image)

        return dest_json

    def _build_env(self, request: DatasetWorkerRequest, hf_endpoint: str | None) -> dict[str, str]:
        env = os.environ.copy()
        # Dataset writing is always offline in P2.
        env["HF_HUB_OFFLINE"] = "1"
        if hf_endpoint:
            env["HF_ENDPOINT"] = hf_endpoint
        # Keep PYTHONPATH minimal to avoid ROS pytest plugins leaking in.
        env.pop("PYTHONPATH", None)
        return env

    def _cleanup(self) -> None:
        if self._temp_dir and not self.debug:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


class RuntimeNotConfiguredError(Exception):
    """Raised when no LeRobot runtime can be resolved."""


def _runtime_error_response(
    request: DatasetWorkerRequest,
    code: LeRobotWorkerErrorCode,
    message: str,
    details: str = "",
) -> DatasetWorkerResponse:
    return DatasetWorkerResponse(
        status="error",
        op=request.op,
        output_dir=request.output_dir,
        repo_id=request.repo_id,
        error=DatasetWorkerError(
            code=code.value,
            message=message,
            details=details,
        ),
    )


def _build_vocab_for_request(feature_groups: list[str]) -> dict[str, dict[str, int]]:
    """Build the vocab mapping passed to the worker for encoding."""
    vocab = build_rosclaw_vocab(feature_groups)
    return {key: dict(mapping) for key, mapping in vocab.vocabularies.items()}


def _resolve_feature_groups(profile: str, feature_groups: list[str] | None = None) -> list[str]:
    """Return the feature groups to enable for a profile or explicit list."""
    if feature_groups:
        return list(feature_groups)
    return sorted(resolve_profile(profile).feature_groups)


def run_dataset_worker_op(
    op: str,
    normalized_episode_path: str,
    output_dir: str,
    repo_id: str,
    *,
    fps: float = 10.0,
    use_videos: bool = False,
    visual_storage_mode: str = "auto",
    profile: str = "minimal",
    feature_groups: list[str] | None = None,
    include_body_snapshot: bool = False,
    body_snapshot_mode: str = "sanitized",
    acknowledge_sensitive_body_data: bool = False,
    dataloader: bool = False,
    dataloader_batch_size: int = 2,
    dataloader_num_workers: int = 0,
    timeout_sec: int = 300,
    allow_partial: bool = False,
    missing_policy: str = "nan",
) -> DatasetWorkerResponse:
    """Convenience helper to build a dataset request and run it in one call."""
    # Enforce profile availability before building the request.
    try:
        resolve_profile(profile)
    except ValueError as exc:
        request = DatasetWorkerRequest(
            op=op,  # type: ignore[arg-type]
            normalized_episode_path=normalized_episode_path,
            output_dir=output_dir,
            repo_id=repo_id,
        )
        return _runtime_error_response(
            request,
            LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
            str(exc),
        )

    groups = _resolve_feature_groups(profile, feature_groups)
    vocab = _build_vocab_for_request(groups)
    features: dict[str, Any] = {}
    if include_body_snapshot:
        features["include_body_snapshot"] = True
        features["body_snapshot_mode"] = body_snapshot_mode
        features["acknowledge_sensitive_body_data"] = acknowledge_sensitive_body_data
    features["missing_policy"] = missing_policy

    request = DatasetWorkerRequest(
        op=op,  # type: ignore[arg-type]
        normalized_episode_path=normalized_episode_path,
        output_dir=output_dir,
        repo_id=repo_id,
        fps=fps,
        profile=profile,
        feature_groups=groups,
        vocab=vocab,
        allow_partial=allow_partial,
        features=features,
        writer=DatasetWriterConfig(
            use_videos=use_videos,
            visual_storage_mode=visual_storage_mode,  # type: ignore[arg-type]
        ),
        validation=DatasetValidationConfig(
            load_after_write=True,
            sample_indices=[0],
            dataloader=dataloader,
            dataloader_batch_size=dataloader_batch_size,
            dataloader_num_workers=dataloader_num_workers,
        ),
        timeout_sec=timeout_sec,
    )
    runner = LeRobotDatasetWorkerRunner(timeout_sec=timeout_sec)
    return runner.run(request)


def run_dataset_export(
    normalized_episode_path: str,
    output_dir: str,
    repo_id: str,
    *,
    fps: float = 10.0,
    use_videos: bool = False,
    visual_storage_mode: str = "auto",
    profile: str = "minimal",
    feature_groups: list[str] | None = None,
    include_body_snapshot: bool = False,
    body_snapshot_mode: str = "sanitized",
    acknowledge_sensitive_body_data: bool = False,
    dataloader: bool = False,
    dataloader_batch_size: int = 2,
    dataloader_num_workers: int = 0,
    timeout_sec: int = 300,
    allow_partial: bool = False,
    missing_policy: str = "nan",
) -> DatasetWorkerResponse:
    """Export a normalized episode to a real LeRobotDataset."""
    return run_dataset_worker_op(
        op="export_dataset",
        normalized_episode_path=normalized_episode_path,
        output_dir=output_dir,
        repo_id=repo_id,
        fps=fps,
        use_videos=use_videos,
        visual_storage_mode=visual_storage_mode,
        profile=profile,
        feature_groups=feature_groups,
        include_body_snapshot=include_body_snapshot,
        body_snapshot_mode=body_snapshot_mode,
        acknowledge_sensitive_body_data=acknowledge_sensitive_body_data,
        dataloader=dataloader,
        dataloader_batch_size=dataloader_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        timeout_sec=timeout_sec,
        allow_partial=allow_partial,
        missing_policy=missing_policy,
    )


def run_dataset_api_inspect(timeout_sec: int = 120) -> DatasetWorkerResponse:
    """Inspect the LeRobotDataset API signature."""
    request = DatasetWorkerRequest(
        op="inspect_api",
        timeout_sec=timeout_sec,
    )
    runner = LeRobotDatasetWorkerRunner(timeout_sec=timeout_sec)
    return runner.run(request)


def run_dataset_dataloader_smoke(
    output_dir: str,
    repo_id: str,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    timeout_sec: int = 300,
) -> DatasetWorkerResponse:
    """Run a DataLoader smoke test against an existing dataset."""
    request = DatasetWorkerRequest(
        op="smoke_dataloader",
        output_dir=output_dir,
        repo_id=repo_id,
        validation=DatasetValidationConfig(
            load_after_write=True,
            dataloader=True,
            dataloader_batch_size=batch_size,
            dataloader_num_workers=num_workers,
        ),
        timeout_sec=timeout_sec,
    )
    runner = LeRobotDatasetWorkerRunner(timeout_sec=timeout_sec)
    return runner.run(request)


__all__ = [
    "LeRobotDatasetWorkerRunner",
    "RuntimeNotConfiguredError",
    "run_dataset_api_inspect",
    "run_dataset_dataloader_smoke",
    "run_dataset_export",
    "run_dataset_worker_op",
]
