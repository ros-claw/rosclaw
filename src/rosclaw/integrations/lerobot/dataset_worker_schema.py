"""Schemas for the P2.1 LeRobot dataset export worker.

These dataclasses describe the JSON request/response protocol between ROSClaw
and the LeRobot dataset worker process. They are kept free of heavy imports so
they can be used by both the ROSClaw core and the worker itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DATASET_WORKER_SCHEMA_VERSION = "rosclaw.lerobot.dataset_worker.v2"
DatasetWorkerOp = Literal["inspect_api", "export_dataset", "validate_dataset", "smoke_dataloader"]


@dataclass
class DatasetWriterConfig:
    """Hints passed to the worker about how to write the dataset."""

    format: str = "lerobot_v3"
    use_videos: bool = True
    visual_storage_mode: Literal["auto", "images", "videos"] = "auto"
    video_codec: str = "auto"
    consolidate: bool = True
    compute_stats: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "use_videos": self.use_videos,
            "visual_storage_mode": self.visual_storage_mode,
            "video_codec": self.video_codec,
            "consolidate": self.consolidate,
            "compute_stats": self.compute_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetWriterConfig":
        return cls(
            format=data.get("format", "lerobot_v3"),
            use_videos=bool(data.get("use_videos", True)),
            visual_storage_mode=data.get("visual_storage_mode", "auto"),  # type: ignore[arg-type]
            video_codec=data.get("video_codec", "auto"),
            consolidate=bool(data.get("consolidate", True)),
            compute_stats=bool(data.get("compute_stats", True)),
        )


@dataclass
class DatasetValidationConfig:
    """What validation the worker should run after writing."""

    load_after_write: bool = True
    sample_indices: list[int] = field(default_factory=lambda: [0])
    dataloader: bool = False
    dataloader_batch_size: int = 2
    dataloader_num_workers: int = 0
    level: str = "load"  # structural|load|dataloader|rich

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_after_write": self.load_after_write,
            "sample_indices": self.sample_indices,
            "dataloader": self.dataloader,
            "dataloader_batch_size": self.dataloader_batch_size,
            "dataloader_num_workers": self.dataloader_num_workers,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetValidationConfig":
        indices = data.get("sample_indices", [0])
        if not isinstance(indices, list):
            indices = [0]
        return cls(
            load_after_write=bool(data.get("load_after_write", True)),
            sample_indices=[int(i) for i in indices],
            dataloader=bool(data.get("dataloader", False)),
            dataloader_batch_size=int(data.get("dataloader_batch_size", 2)),
            dataloader_num_workers=int(data.get("dataloader_num_workers", 0)),
            level=data.get("level", "load"),
        )


@dataclass
class DatasetWorkerRequest:
    """Request sent to the LeRobot dataset worker."""

    op: DatasetWorkerOp
    normalized_episode_path: str = ""
    output_dir: str = ""
    repo_id: str = ""
    fps: float = 10.0
    robot_type: str = "rosclaw"
    profile: str = "minimal"
    features: dict[str, Any] = field(default_factory=dict)
    feature_groups: list[str] = field(default_factory=list)
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    allow_partial: bool = False
    writer: DatasetWriterConfig = field(default_factory=DatasetWriterConfig)
    validation: DatasetValidationConfig = field(default_factory=DatasetValidationConfig)
    timeout_sec: int = 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_WORKER_SCHEMA_VERSION,
            "op": self.op,
            "normalized_episode_path": self.normalized_episode_path,
            "output_dir": self.output_dir,
            "repo_id": self.repo_id,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "profile": self.profile,
            "features": self.features,
            "feature_groups": list(self.feature_groups),
            "vocab": dict(self.vocab),
            "allow_partial": self.allow_partial,
            "writer": self.writer.to_dict(),
            "validation": self.validation.to_dict(),
            "timeout_sec": self.timeout_sec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetWorkerRequest":
        return cls(
            op=data.get("op", "export_dataset"),  # type: ignore[arg-type]
            normalized_episode_path=data.get("normalized_episode_path", ""),
            output_dir=data.get("output_dir", ""),
            repo_id=data.get("repo_id", ""),
            fps=float(data.get("fps", 10.0)),
            robot_type=data.get("robot_type", "rosclaw"),
            profile=data.get("profile", "minimal"),
            features=dict(data.get("features", {})),
            feature_groups=list(data.get("feature_groups", [])),
            vocab=dict(data.get("vocab", {})),
            allow_partial=bool(data.get("allow_partial", False)),
            writer=DatasetWriterConfig.from_dict(data.get("writer", {})),
            validation=DatasetValidationConfig.from_dict(data.get("validation", {})),
            timeout_sec=int(data.get("timeout_sec", 300)),
        )


@dataclass
class DatasetFeatureInfo:
    """Description of a LeRobotDataset feature returned by the worker."""

    shape: list[int] = field(default_factory=list)
    dtype: str = "float32"
    names: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shape": self.shape,
            "dtype": self.dtype,
        }
        if self.names is not None:
            out["names"] = self.names
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetFeatureInfo":
        names = data.get("names")
        if names is not None:
            names = list(names)
        return cls(
            shape=list(data.get("shape", [])),
            dtype=data.get("dtype", "float32"),
            names=names,
        )


@dataclass
class DatasetVisualInfo:
    """Visual storage metadata for the exported dataset."""

    storage_mode: str = "images"
    camera_keys: list[str] = field(default_factory=list)
    use_videos: bool = False
    resolution: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "storage_mode": self.storage_mode,
            "camera_keys": list(self.camera_keys),
            "use_videos": self.use_videos,
            "resolution": dict(self.resolution),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetVisualInfo":
        return cls(
            storage_mode=data.get("storage_mode", "images"),
            camera_keys=list(data.get("camera_keys", [])),
            use_videos=bool(data.get("use_videos", False)),
            resolution=dict(data.get("resolution", {})),
        )


@dataclass
class DatasetInfo:
    """Summary of the dataset produced by the worker."""

    num_episodes: int = 0
    num_frames: int = 0
    fps: float = 10.0
    features: dict[str, DatasetFeatureInfo] = field(default_factory=dict)
    visual: DatasetVisualInfo = field(default_factory=DatasetVisualInfo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "visual": self.visual.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetInfo":
        raw_features = data.get("features", {})
        features: dict[str, DatasetFeatureInfo] = {}
        for key, value in raw_features.items():
            features[key] = DatasetFeatureInfo.from_dict(value) if isinstance(value, dict) else DatasetFeatureInfo()
        visual_data = data.get("visual", {})
        return cls(
            num_episodes=int(data.get("num_episodes", 0)),
            num_frames=int(data.get("num_frames", 0)),
            fps=float(data.get("fps", 10.0)),
            features=features,
            visual=DatasetVisualInfo.from_dict(visual_data) if isinstance(visual_data, dict) else DatasetVisualInfo(),
        )


@dataclass
class DatasetFileInfo:
    """List of files produced by the worker."""

    meta_info: bool = False
    data_files: list[str] = field(default_factory=list)
    video_files: list[str] = field(default_factory=list)
    sidecar_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_info": self.meta_info,
            "data_files": self.data_files,
            "video_files": self.video_files,
            "sidecar_files": self.sidecar_files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetFileInfo":
        return cls(
            meta_info=bool(data.get("meta_info", False)),
            data_files=list(data.get("data_files", [])),
            video_files=list(data.get("video_files", [])),
            sidecar_files=list(data.get("sidecar_files", [])),
        )


@dataclass
class DatasetValidationResult:
    """Validation block returned by the worker."""

    load_ok: bool = False
    index_ok: bool = False
    dataloader_ok: bool | None = None
    num_frames: int | None = None
    num_episodes: int | None = None
    sample_keys: list[str] = field(default_factory=list)
    sample_image_keys: list[str] = field(default_factory=list)
    batch_keys: list[str] = field(default_factory=list)
    batch_shapes: dict[str, list[int]] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "load_ok": self.load_ok,
            "index_ok": self.index_ok,
            "sample_keys": self.sample_keys,
            "sample_image_keys": self.sample_image_keys,
            "batch_keys": self.batch_keys,
            "batch_shapes": self.batch_shapes,
        }
        if self.num_frames is not None:
            out["num_frames"] = self.num_frames
        if self.num_episodes is not None:
            out["num_episodes"] = self.num_episodes
        if self.dataloader_ok is not None:
            out["dataloader_ok"] = self.dataloader_ok
        if self.error is not None:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetValidationResult":
        return cls(
            load_ok=bool(data.get("load_ok", False)),
            index_ok=bool(data.get("index_ok", False)),
            dataloader_ok=data.get("dataloader_ok"),
            num_frames=data.get("num_frames"),
            num_episodes=data.get("num_episodes"),
            sample_keys=list(data.get("sample_keys", [])),
            sample_image_keys=list(data.get("sample_image_keys", [])),
            batch_keys=list(data.get("batch_keys", [])),
            batch_shapes=dict(data.get("batch_shapes", {})),
            error=data.get("error"),
        )


@dataclass
class DatasetWorkerError:
    """Structured error returned by the dataset worker."""

    code: str
    message: str
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetWorkerError":
        return cls(
            code=data.get("code", "unknown"),
            message=data.get("message", ""),
            details=data.get("details", ""),
        )


@dataclass
class DatasetWorkerTiming:
    """Timing metadata from the dataset worker."""

    normalize_time_sec: float | None = None
    write_time_sec: float | None = None
    validate_time_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.normalize_time_sec is not None:
            out["normalize_time_sec"] = self.normalize_time_sec
        if self.write_time_sec is not None:
            out["write_time_sec"] = self.write_time_sec
        if self.validate_time_sec is not None:
            out["validate_time_sec"] = self.validate_time_sec
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetWorkerTiming":
        return cls(
            normalize_time_sec=data.get("normalize_time_sec"),
            write_time_sec=data.get("write_time_sec"),
            validate_time_sec=data.get("validate_time_sec"),
        )


@dataclass
class DatasetApiInfo:
    """Introspection result for the LeRobotDataset API."""

    create_signature: str = ""
    has_add_frame: bool = False
    has_save_episode: bool = False
    has_consolidate: bool = False
    has_finalize: bool = False
    lerobot_version: str | None = None
    error: DatasetWorkerError | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "create_signature": self.create_signature,
            "has_add_frame": self.has_add_frame,
            "has_save_episode": self.has_save_episode,
            "has_consolidate": self.has_consolidate,
            "has_finalize": self.has_finalize,
        }
        if self.lerobot_version is not None:
            out["lerobot_version"] = self.lerobot_version
        if self.error is not None:
            out["error"] = self.error.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetApiInfo":
        error_data = data.get("error")
        return cls(
            create_signature=data.get("create_signature", ""),
            has_add_frame=bool(data.get("has_add_frame", False)),
            has_save_episode=bool(data.get("has_save_episode", False)),
            has_consolidate=bool(data.get("has_consolidate", False)),
            has_finalize=bool(data.get("has_finalize", False)),
            lerobot_version=data.get("lerobot_version"),
            error=DatasetWorkerError.from_dict(error_data) if error_data else None,
        )


@dataclass
class DatasetWorkerResponse:
    """Response returned by the dataset worker."""

    schema_version: str = DATASET_WORKER_SCHEMA_VERSION
    status: Literal["ok", "error"] = "ok"
    op: DatasetWorkerOp = "export_dataset"  # type: ignore[assignment]
    output_dir: str = ""
    repo_id: str = ""
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    files: DatasetFileInfo = field(default_factory=DatasetFileInfo)
    validation: DatasetValidationResult = field(default_factory=DatasetValidationResult)
    timing: DatasetWorkerTiming = field(default_factory=DatasetWorkerTiming)
    api_info: DatasetApiInfo | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    feature_groups_written: list[str] = field(default_factory=list)
    sidecar_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extension_schema: str = ""
    requested_feature_groups: list[str] = field(default_factory=list)
    written_feature_groups: list[str] = field(default_factory=list)
    missing_feature_groups: list[str] = field(default_factory=list)
    profile_satisfied: bool = True
    profile: str = "minimal"
    error: DatasetWorkerError | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "op": self.op,
            "output_dir": self.output_dir,
            "repo_id": self.repo_id,
            "dataset": self.dataset.to_dict(),
            "files": self.files.to_dict(),
            "validation": self.validation.to_dict(),
            "timing": self.timing.to_dict(),
            "runtime": self.runtime,
            "feature_groups_written": list(self.feature_groups_written),
            "sidecar_files": list(self.sidecar_files),
            "warnings": list(self.warnings),
            "extension_schema": self.extension_schema,
            "requested_feature_groups": list(self.requested_feature_groups),
            "written_feature_groups": list(self.written_feature_groups),
            "missing_feature_groups": list(self.missing_feature_groups),
            "profile_satisfied": self.profile_satisfied,
            "profile": self.profile,
        }
        if self.api_info is not None:
            out["api_info"] = self.api_info.to_dict()
        if self.error is not None:
            out["error"] = self.error.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetWorkerResponse":
        api_data = data.get("api_info")
        error_data = data.get("error")
        return cls(
            schema_version=data.get("schema_version", DATASET_WORKER_SCHEMA_VERSION),
            status=data.get("status", "ok"),  # type: ignore[arg-type]
            op=data.get("op", "export_dataset"),  # type: ignore[arg-type]
            output_dir=data.get("output_dir", ""),
            repo_id=data.get("repo_id", ""),
            dataset=DatasetInfo.from_dict(data.get("dataset", {})),
            files=DatasetFileInfo.from_dict(data.get("files", {})),
            validation=DatasetValidationResult.from_dict(data.get("validation", {})),
            timing=DatasetWorkerTiming.from_dict(data.get("timing", {})),
            api_info=DatasetApiInfo.from_dict(api_data) if api_data else None,
            runtime=data.get("runtime", {}),
            feature_groups_written=list(data.get("feature_groups_written", [])),
            sidecar_files=list(data.get("sidecar_files", [])),
            warnings=list(data.get("warnings", [])),
            extension_schema=data.get("extension_schema", ""),
            requested_feature_groups=list(data.get("requested_feature_groups", [])),
            written_feature_groups=list(data.get("written_feature_groups", [])),
            missing_feature_groups=list(data.get("missing_feature_groups", [])),
            profile_satisfied=bool(data.get("profile_satisfied", True)),
            profile=data.get("profile", "minimal"),
            error=DatasetWorkerError.from_dict(error_data) if error_data else None,
        )

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def error_code(self) -> str:
        if self.error is not None:
            return self.error.code
        return "unknown"

    def error_message(self) -> str:
        if self.error is not None:
            return self.error.message
        return ""


__all__ = [
    "DATASET_WORKER_SCHEMA_VERSION",
    "DatasetApiInfo",
    "DatasetFeatureInfo",
    "DatasetFileInfo",
    "DatasetInfo",
    "DatasetValidationConfig",
    "DatasetValidationResult",
    "DatasetVisualInfo",
    "DatasetWorkerError",
    "DatasetWorkerOp",
    "DatasetWorkerRequest",
    "DatasetWorkerResponse",
    "DatasetWorkerTiming",
    "DatasetWriterConfig",
]
