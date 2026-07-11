#!/usr/bin/env python3
"""LeRobot dataset worker entry point.

This file is intentionally standalone: it is executed by the LeRobot runtime
Python and must not import ``rosclaw``.  It only uses the standard library plus
numpy/PIL/lerobot/torch inside the operation functions.

Usage:
    /path/to/lerobot/python dataset_worker_main.py         --request-json /tmp/request.json         --output-json /tmp/response.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORKER_SCHEMA_VERSION = "rosclaw.lerobot.dataset_worker.v2"

# Feature groups and their LeRobot feature specs.  These are intentionally
# duplicated here so the worker remains self-contained.
ROSCLAW_FEATURE_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "safety": {
        "rosclaw.sandbox.decision": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.sandbox.modified": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.sandbox.risk_score": {"dtype": "float32", "shape": [1], "names": None},
    },
    "failure": {
        "rosclaw.failure.active": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.failure.code": {"dtype": "int16", "shape": [1], "names": None},
    },
    "intervention": {
        "rosclaw.intervention.active": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.intervention.source": {"dtype": "int8", "shape": [1], "names": None},
    },
    "action": {
        "rosclaw.action.source": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.action.was_clamped": {"dtype": "int8", "shape": [1], "names": None},
    },
    "outcome": {
        "rosclaw.done": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.success": {"dtype": "int8", "shape": [1], "names": None},
    },
    "physical_telemetry": {
        "observation.motor_current": {"dtype": "float32", "shape": [1], "names": ["motor"]},
        "observation.joint_temperature": {"dtype": "float32", "shape": [1], "names": ["joint"]},
        "observation.force_torque": {"dtype": "float32", "shape": [1], "names": ["axis"]},
        "observation.contact": {"dtype": "int8", "shape": [1], "names": ["contact"]},
        "observation.joint_velocity": {"dtype": "float32", "shape": [1], "names": ["joint"]},
        "observation.joint_effort": {"dtype": "float32", "shape": [1], "names": ["joint"]},
    },
}

DEFAULT_VOCAB: dict[str, dict[str, int]] = {
    "rosclaw.sandbox.decision": {
        "UNKNOWN": 0,
        "ALLOW": 1,
        "CLAMP": 2,
        "BLOCK": 3,
        "ESTOP": 4,
    },
    "rosclaw.action.source": {
        "UNKNOWN": 0,
        "POLICY": 1,
        "HUMAN": 2,
        "RULE": 3,
        "HOW_CORRECTION": 4,
        "REPLAY": 5,
    },
    "rosclaw.intervention.source": {
        "UNKNOWN": 0,
        "HUMAN_JOYSTICK": 1,
        "HUMAN_KEYFRAME": 2,
        "POLICY_OVERRIDE": 3,
        "RULE_OVERRIDE": 4,
    },
    "rosclaw.failure.code": {
        "UNKNOWN": 0,
        "NONE": 1,
        "COLLISION": 2,
        "SELF_COLLISION": 3,
        "JOINT_LIMIT": 4,
        "OVERCURRENT": 5,
        "OVERTEMPERATURE": 6,
        "DROP": 7,
        "SLIP": 8,
        "TIMEOUT": 9,
        "TASK_FAIL": 10,
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeRobot dataset writer worker for ROSClaw")
    parser.add_argument("--request-json", required=True, help="Path to request JSON")
    parser.add_argument("--output-json", required=True, help="Path to write response JSON")
    args = parser.parse_args(argv)

    request_path = Path(args.request_json)
    output_path = Path(args.output_json)

    try:
        request = _load_json(request_path)
    except Exception as exc:  # noqa: BLE001
        response = _error_response(
            "export_dataset",
            "worker_invalid_json",
            f"Failed to read request JSON: {exc}",
            traceback.format_exc(),
        )
        _write_json(output_path, response)
        return 1

    op = request.get("op", "export_dataset")
    try:
        if op == "inspect_api":
            response = _op_inspect_api(request)
        elif op == "export_dataset":
            response = _op_export_dataset(request)
        elif op == "validate_dataset":
            response = _op_validate_dataset(request)
        elif op == "smoke_dataloader":
            response = _op_smoke_dataloader(request)
        else:
            response = _error_response(
                op,
                "worker_invalid_json",
                f"Unknown op: {op}",
                "",
            )
    except Exception as exc:  # noqa: BLE001
        response = _error_response(
            op,
            "dataset_write_failed",
            f"Worker crashed: {exc}",
            traceback.format_exc(),
        )

    _write_json(output_path, response)
    return 0 if response.get("status") == "ok" else 1


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _error_response(
    op: str,
    code: str,
    message: str,
    details: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "error",
        "op": op,
        "output_dir": "",
        "repo_id": "",
        "dataset": {},
        "files": {},
        "validation": {},
        "timing": {},
        "runtime": _runtime_info(),
        "feature_groups_written": [],
        "sidecar_files": [],
        "warnings": [],
        "extension_schema": "",
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
    }


def _ok_response(
    op: str,
    output_dir: str,
    repo_id: str,
    *,
    dataset: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    api_info: dict[str, Any] | None = None,
    feature_groups_written: list[str] | None = None,
    sidecar_files: list[str] | None = None,
    warnings: list[str] | None = None,
    extension_schema: str = "",
    requested_feature_groups: list[str] | None = None,
    missing_feature_groups: list[str] | None = None,
    profile_satisfied: bool = True,
) -> dict[str, Any]:
    requested = list(requested_feature_groups or [])
    written = list(feature_groups_written or [])
    missing = list(missing_feature_groups or [g for g in requested if g not in written])
    response: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "ok",
        "op": op,
        "output_dir": output_dir,
        "repo_id": repo_id,
        "dataset": dataset or {},
        "files": files or {},
        "validation": validation or {},
        "timing": timing or {},
        "runtime": _runtime_info(),
        "feature_groups_written": written,
        "sidecar_files": list(sidecar_files or []),
        "warnings": list(warnings or []),
        "extension_schema": extension_schema,
        "requested_feature_groups": requested,
        "written_feature_groups": written,
        "missing_feature_groups": missing,
        "profile_satisfied": profile_satisfied and not missing,
    }
    if api_info is not None:
        response["api_info"] = api_info
    return response


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.executable,
        "python_version": ".".join(str(x) for x in sys.version_info[:3]),
    }
    try:
        import lerobot

        info["lerobot_version"] = getattr(lerobot, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass
    return info


# ------------------------------------------------------------------
# inspect_api
# ------------------------------------------------------------------
def _op_inspect_api(request: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "inspect_api",
            "lerobot_dataset_import_failed",
            f"Could not import LeRobotDataset: {exc}",
            traceback.format_exc(),
        )

    try:
        sig = inspect.signature(LeRobotDataset.create)
    except Exception as exc:  # noqa: BLE001
        sig = f"(signature unavailable: {exc})"

    api_info = {
        "create_signature": str(sig),
        "has_add_frame": hasattr(LeRobotDataset, "add_frame"),
        "has_save_episode": hasattr(LeRobotDataset, "save_episode"),
        "has_consolidate": hasattr(LeRobotDataset, "consolidate"),
        "has_finalize": hasattr(LeRobotDataset, "finalize"),
    }
    try:
        import lerobot

        api_info["lerobot_version"] = getattr(lerobot, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass

    return _ok_response(
        "inspect_api",
        output_dir="",
        repo_id="",
        api_info=api_info,
    )


# ------------------------------------------------------------------
# export_dataset
# ------------------------------------------------------------------
def _op_export_dataset(request: dict[str, Any]) -> dict[str, Any]:
    normalized_path = request.get("normalized_episode_path", "")
    output_dir = request.get("output_dir", "")
    repo_id = request.get("repo_id", "")
    fps = float(request.get("fps", 10.0))
    robot_type = request.get("robot_type") or "rosclaw"
    writer_cfg = request.get("writer", {})
    validation_cfg = request.get("validation", {})
    profile = request.get("profile", "minimal")
    feature_groups = _resolve_feature_groups(request)
    vocab = request.get("vocab") or {}
    features_meta = request.get("features") or {}
    missing_policy = str(features_meta.get("missing_policy", "nan"))

    if not normalized_path:
        return _error_response("export_dataset", "normalized_episode_invalid", "normalized_episode_path is empty", "")
    if not output_dir:
        return _error_response("export_dataset", "dataset_create_failed", "output_dir is empty", "")
    if not repo_id:
        return _error_response("export_dataset", "dataset_create_failed", "repo_id is empty", "")

    normalized_file = Path(normalized_path)
    if not normalized_file.exists():
        return _error_response(
            "export_dataset",
            "normalized_episode_invalid",
            f"Normalized episode file not found: {normalized_path}",
            "",
        )

    normalize_start = time.perf_counter()
    try:
        episode = _load_json(normalized_file)
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "export_dataset",
            "normalized_episode_invalid",
            f"Could not parse normalized episode: {exc}",
            traceback.format_exc(),
        )

    base_dir = normalized_file.parent
    frames_data = episode.get("frames", [])
    if not frames_data:
        return _error_response(
            "export_dataset",
            "normalized_episode_invalid",
            "Normalized episode contains no frames.",
            "",
        )

    try:
        features = _infer_features(episode, base_dir, feature_groups)
    except _WorkerError as exc:
        return _error_response("export_dataset", exc.code, exc.message, exc.details)

    use_videos, storage_mode = _resolve_visual_mode(writer_cfg, len(frames_data))
    camera_keys = sorted(k for k in features if k.startswith("observation.images."))
    resolution = {}
    for key in camera_keys:
        shape = features[key].get("shape", [])
        if len(shape) >= 2:
            resolution[key] = [int(shape[0]), int(shape[1])]

    normalize_time = round(time.perf_counter() - normalize_start, 3)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "export_dataset",
            "lerobot_dataset_import_failed",
            f"Could not import LeRobotDataset: {exc}",
            traceback.format_exc(),
        )

    if not hasattr(LeRobotDataset, "create"):
        return _error_response(
            "export_dataset",
            "lerobot_dataset_api_unsupported",
            "LeRobotDataset.create is not available.",
            "",
        )

    output_root = Path(output_dir)
    if output_root.exists():
        try:
            shutil.rmtree(output_root)
        except Exception as exc:  # noqa: BLE001
            return _error_response(
                "export_dataset",
                "dataset_create_failed",
                f"Could not clear existing output dir: {exc}",
                traceback.format_exc(),
            )

    write_start = time.perf_counter()
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(fps),
            features=features,
            root=str(output_root),
            robot_type=robot_type,
            use_videos=use_videos,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "export_dataset",
            "dataset_create_failed",
            f"LeRobotDataset.create failed: {exc}",
            traceback.format_exc(),
        )

    task_text = ""
    task_data = episode.get("task", {})
    if isinstance(task_data, str):
        task_text = task_data
    elif isinstance(task_data, dict):
        task_text = task_data.get("text", "")
    if not task_text:
        task_text = "rosclaw practice"

    warnings: list[str] = []
    try:
        for frame_data in frames_data:
            frame = _build_lerobot_frame(
                frame_data, base_dir, features, task_text, use_videos, feature_groups, vocab,
                missing_policy=missing_policy,
            )
            dataset.add_frame(frame)
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "export_dataset",
            "dataset_add_frame_failed",
            f"add_frame failed: {exc}",
            traceback.format_exc(),
        )

    try:
        dataset.save_episode()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "export_dataset",
            "dataset_save_episode_failed",
            f"save_episode failed: {exc}",
            traceback.format_exc(),
        )

    try:
        dataset.finalize()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "export_dataset",
            "dataset_consolidate_failed",
            f"finalize failed: {exc}",
            traceback.format_exc(),
        )

    write_time = round(time.perf_counter() - write_start, 3)

    validate_start = time.perf_counter()
    validation = _validate_dataset(
        output_root,
        repo_id,
        validation_cfg,
        expected_frames=len(frames_data),
        expected_episodes=1,
    )
    validate_time = round(time.perf_counter() - validate_start, 3)

    files = _collect_file_list(output_root)
    dataset_info = {
        "num_episodes": validation.get("num_episodes", 1),
        "num_frames": validation.get("num_frames", len(frames_data)),
        "fps": fps,
        "features": {k: _feature_info_dict(v) for k, v in features.items()},
        "visual": {
            "storage_mode": storage_mode,
            "camera_keys": camera_keys,
            "use_videos": use_videos,
            "resolution": resolution,
        },
    }

    api_info = _api_info_from_dataset_class(LeRobotDataset)

    ok_response = _ok_response(
        "export_dataset",
        output_dir=str(output_root),
        repo_id=repo_id,
        dataset=dataset_info,
        files=files,
        validation=validation,
        timing={
            "normalize_time_sec": normalize_time,
            "write_time_sec": write_time,
            "validate_time_sec": validate_time,
        },
        api_info=api_info,
        feature_groups_written=list(feature_groups),
        requested_feature_groups=list(feature_groups),
        warnings=warnings,
    )
    ok_response["profile"] = profile
    return ok_response


def _api_info_from_dataset_class(cls: Any) -> dict[str, Any]:
    api_info = {
        "create_signature": "",
        "has_add_frame": hasattr(cls, "add_frame"),
        "has_save_episode": hasattr(cls, "save_episode"),
        "has_consolidate": hasattr(cls, "consolidate"),
        "has_finalize": hasattr(cls, "finalize"),
    }
    try:
        sig = inspect.signature(cls.create)
        api_info["create_signature"] = str(sig)
    except Exception:  # noqa: BLE001
        pass
    try:
        import lerobot

        api_info["lerobot_version"] = getattr(lerobot, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return api_info


# ------------------------------------------------------------------
# Feature / frame helpers
# ------------------------------------------------------------------
class _WorkerError(Exception):
    def __init__(self, code: str, message: str, details: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _resolve_feature_groups(request: dict[str, Any]) -> list[str]:
    groups = request.get("feature_groups")
    if isinstance(groups, list) and groups:
        return [str(g) for g in groups]

    profile = request.get("profile", "minimal")
    mapping: dict[str, list[str]] = {
        "minimal": [],
        "safety": ["safety"],
        "physical": ["safety", "action", "physical_telemetry"],
        "safety-rich": ["safety", "failure", "intervention", "action", "outcome"],
    }
    return mapping.get(profile, [])


def _resolve_visual_mode(writer_cfg: dict[str, Any], num_frames: int) -> tuple[bool, str]:
    mode = writer_cfg.get("visual_storage_mode", "auto")
    legacy_use_videos = bool(writer_cfg.get("use_videos", True))
    if mode == "videos":
        return True, "videos"
    if mode == "images":
        return False, "images"
    # "auto": keep P2 stable path (images) for short practice episodes unless the
    # caller explicitly opted into videos via the legacy flag.
    if legacy_use_videos and num_frames > 50:
        return True, "videos"
    return False, "images"


def _infer_features(
    episode: dict[str, Any],
    base_dir: Path,
    feature_groups: list[str],
) -> dict[str, Any]:
    frames = episode.get("frames", [])
    if not frames:
        raise _WorkerError("normalized_episode_invalid", "No frames in episode.")

    state_dim: int | None = None
    action_dim: int | None = None
    image_size: tuple[int, int] | None = None
    image_camera: str | None = None
    telemetry_dims: dict[str, int] = {}
    telemetry_fields = [
        ("observation.motor_current", "motor_current"),
        ("observation.joint_temperature", "joint_temperature"),
        ("observation.force_torque", "force_torque"),
        ("observation.contact", "contact"),
        ("observation.joint_velocity", "joint_velocity"),
        ("observation.joint_effort", "joint_effort"),
    ]

    for idx, frame in enumerate(frames):
        obs = frame.get("observation", {})
        state = list(obs.get("state", []))
        action = list(frame.get("action", []))

        if state_dim is None:
            state_dim = len(state)
        elif len(state) != state_dim:
            raise _WorkerError(
                "state_dim_mismatch",
                f"State dimension mismatch at frame {idx}: expected {state_dim}, got {len(state)}.",
            )

        if action_dim is None:
            action_dim = len(action)
        elif len(action) != action_dim:
            raise _WorkerError(
                "action_dim_mismatch",
                f"Action dimension mismatch at frame {idx}: expected {action_dim}, got {len(action)}.",
            )

        for feature_key, attr_name in telemetry_fields:
            values = obs.get(attr_name)
            if values:
                telemetry_dims.setdefault(feature_key, len(values))

        images = obs.get("images", {})
        for camera_name, image_rel in images.items():
            image_path = base_dir / image_rel
            if not image_path.exists():
                raise _WorkerError(
                    "image_file_not_found",
                    f"Image file not found for camera '{camera_name}': {image_path}",
                )
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    size = (img.height, img.width)
            except Exception as exc:  # noqa: BLE001
                raise _WorkerError(
                    "image_file_not_found",
                    f"Could not read image '{camera_name}' at {image_path}: {exc}",
                ) from exc

            if image_size is None:
                image_size = size
                image_camera = camera_name
            elif size != image_size:
                raise _WorkerError(
                    "image_shape_mismatch",
                    f"Image shape mismatch for camera '{camera_name}' at frame {idx}: "
                    f"expected {image_size} (from '{image_camera}'), got {size}.",
                )

    features: dict[str, Any] = {}
    if state_dim is not None and state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
        }
    if action_dim is not None and action_dim > 0:
        features["action"] = {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
        }
    if image_size is not None and image_camera is not None:
        h, w = image_size
        features[f"observation.images.{image_camera}"] = {
            "dtype": "image",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
        }

    for group in feature_groups:
        if group == "physical_telemetry":
            for key, spec in ROSCLAW_FEATURE_SPECS.get(group, {}).items():
                dim = telemetry_dims.get(key)
                if dim is not None and dim > 0:
                    features[key] = {**spec, "shape": [dim]}
            continue
        for key, spec in ROSCLAW_FEATURE_SPECS.get(group, {}).items():
            features[key] = dict(spec)

    return features


def _feature_info_dict(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": list(feature.get("shape", [])),
        "dtype": feature.get("dtype", "float32"),
        "names": feature.get("names"),
    }


def _encode_label(label: Any, vocab: dict[str, int], default: int = 0) -> int:
    if label is None:
        return default
    return vocab.get(str(label), default)


def _encode_bool(value: Any) -> int:
    """Encode a nullable boolean as int8: -1 UNKNOWN, 0 FALSE/INACTIVE, 1 TRUE/ACTIVE."""
    if value is None:
        return -1
    return 1 if value else 0


def _build_lerobot_frame(
    frame_data: dict[str, Any],
    base_dir: Path,
    features: dict[str, Any],
    task_text: str,
    use_videos: bool,  # noqa: ARG001
    feature_groups: list[str],
    vocab: dict[str, dict[str, int]],
    missing_policy: str = "nan",
) -> dict[str, Any]:
    import numpy as np

    frame: dict[str, Any] = {"task": task_text}

    obs = frame_data.get("observation", {})
    if "observation.state" in features:
        state = list(obs.get("state", []))
        frame["observation.state"] = np.array(state, dtype=np.float32)

    if "action" in features:
        action = list(frame_data.get("action", []))
        frame["action"] = np.array(action, dtype=np.float32)

    for key in features:
        if not key.startswith("observation.images."):
            continue
        camera_name = key.split(".", 2)[2]
        image_rel = obs.get("images", {}).get(camera_name)
        if not image_rel:
            raise _WorkerError(
                "image_file_not_found",
                f"Missing image path for camera '{camera_name}'.",
            )
        image_path = base_dir / image_rel
        if not image_path.exists():
            raise _WorkerError(
                "image_file_not_found",
                f"Image file not found: {image_path}",
            )

        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        frame[key] = img

    safety_data = frame_data.get("safety", {}) or {}
    failure_data = frame_data.get("failure", {}) or {}
    intervention_data = frame_data.get("intervention", {}) or {}
    action_context_data = frame_data.get("action_context", {}) or {}

    if "rosclaw.sandbox.decision" in features:
        decision_vocab = vocab.get("rosclaw.sandbox.decision") or DEFAULT_VOCAB["rosclaw.sandbox.decision"]
        frame["rosclaw.sandbox.decision"] = np.array(
            [_encode_label(safety_data.get("decision"), decision_vocab)], dtype=np.int8
        )
    if "rosclaw.sandbox.modified" in features:
        frame["rosclaw.sandbox.modified"] = np.array(
            [_encode_bool(safety_data.get("modified"))], dtype=np.int8
        )
    if "rosclaw.sandbox.risk_score" in features:
        score = safety_data.get("risk_score")
        if score is None or (isinstance(score, float) and math.isnan(score)):
            value = float("nan")
        else:
            value = float(score)
        frame["rosclaw.sandbox.risk_score"] = np.array([value], dtype=np.float32)

    if "rosclaw.failure.active" in features:
        frame["rosclaw.failure.active"] = np.array(
            [_encode_bool(failure_data.get("active"))], dtype=np.int8
        )
    if "rosclaw.failure.code" in features:
        code_vocab = vocab.get("rosclaw.failure.code") or DEFAULT_VOCAB["rosclaw.failure.code"]
        frame["rosclaw.failure.code"] = np.array(
            [_encode_label(failure_data.get("code"), code_vocab)], dtype=np.int16
        )

    if "rosclaw.intervention.active" in features:
        frame["rosclaw.intervention.active"] = np.array(
            [_encode_bool(intervention_data.get("active"))], dtype=np.int8
        )
    if "rosclaw.intervention.source" in features:
        source_vocab = vocab.get("rosclaw.intervention.source") or DEFAULT_VOCAB["rosclaw.intervention.source"]
        frame["rosclaw.intervention.source"] = np.array(
            [_encode_label(intervention_data.get("source"), source_vocab)], dtype=np.int8
        )

    if "rosclaw.action.source" in features:
        source_vocab = vocab.get("rosclaw.action.source") or DEFAULT_VOCAB["rosclaw.action.source"]
        frame["rosclaw.action.source"] = np.array(
            [_encode_label(action_context_data.get("source"), source_vocab)], dtype=np.int8
        )
    if "rosclaw.action.was_clamped" in features:
        frame["rosclaw.action.was_clamped"] = np.array(
            [_encode_bool(action_context_data.get("was_clamped"))], dtype=np.int8
        )

    if "rosclaw.done" in features:
        frame["rosclaw.done"] = np.array([_encode_bool(frame_data.get("done"))], dtype=np.int8)
    if "rosclaw.success" in features:
        frame["rosclaw.success"] = np.array([_encode_bool(frame_data.get("success"))], dtype=np.int8)

    obs = frame_data.get("observation", {})
    telemetry_specs: dict[str, tuple[str, Any]] = {
        "observation.motor_current": ("float32", float("nan")),
        "observation.joint_temperature": ("float32", float("nan")),
        "observation.force_torque": ("float32", float("nan")),
        "observation.contact": ("int8", 0),
        "observation.joint_velocity": ("float32", float("nan")),
        "observation.joint_effort": ("float32", float("nan")),
    }
    for key, (dtype, fill_value) in telemetry_specs.items():
        if key not in features:
            continue
        expected_dim = int(features[key].get("shape", [1])[0])
        attr_name = key.split(".", 1)[1]
        raw = obs.get(attr_name)
        if raw is None or len(raw) == 0:
            if missing_policy == "error":
                raise _WorkerError(
                    "telemetry_missing",
                    f"Missing {attr_name} at frame {frame_data.get('frame_index')} "
                    f"but missing_policy=error.",
                )
            raw = []
        if len(raw) < expected_dim:
            raw = list(raw) + [fill_value] * (expected_dim - len(raw))
        elif len(raw) > expected_dim:
            raw = list(raw)[:expected_dim]
        if dtype == "int8":
            frame[key] = np.array([int(bool(v)) for v in raw], dtype=np.int8)
        else:
            frame[key] = np.array([float(v) for v in raw], dtype=np.float32)

    return frame


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------
def _validate_dataset(
    output_root: Path,
    repo_id: str,
    validation_cfg: dict[str, Any],
    expected_frames: int,
    expected_episodes: int,
) -> dict[str, Any]:
    import os

    result: dict[str, Any] = {
        "load_ok": False,
        "index_ok": False,
        "dataloader_ok": None,
        "num_frames": 0,
        "num_episodes": 0,
        "sample_keys": [],
        "sample_image_keys": [],
        "batch_keys": [],
        "batch_shapes": {},
    }

    if not validation_cfg.get("load_after_write", True):
        result["load_ok"] = True
        result["index_ok"] = True
        result["num_frames"] = expected_frames
        result["num_episodes"] = expected_episodes
        return result

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Force offline load so local-only datasets never try the Hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        dataset = LeRobotDataset(repo_id, root=str(output_root))
        result["load_ok"] = True
        result["num_frames"] = len(dataset)
        result["num_episodes"] = int(getattr(dataset, "num_episodes", expected_episodes))
    except Exception as exc:  # noqa: BLE001
        result["error"] = {
            "code": "dataset_load_failed",
            "message": f"Failed to load dataset: {exc}",
            "details": traceback.format_exc(),
        }
        return result

    sample_indices = validation_cfg.get("sample_indices", [0])
    if not isinstance(sample_indices, list):
        sample_indices = [0]

    try:
        for idx in sample_indices:
            if idx < 0 or idx >= len(dataset):
                continue
            sample = dataset[idx]
            if not result["sample_keys"]:
                result["sample_keys"] = sorted(str(k) for k in sample)
                result["sample_image_keys"] = sorted(
                    str(k) for k in sample if k.startswith("observation.images.")
                )
        result["index_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = {
            "code": "dataset_index_failed",
            "message": f"Failed to index dataset: {exc}",
            "details": traceback.format_exc(),
        }
        return result

    if validation_cfg.get("dataloader"):
        result.update(_run_dataloader_smoke(dataset, validation_cfg))

    return result


def _run_dataloader_smoke(dataset: Any, validation_cfg: dict[str, Any]) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    batch_size = int(validation_cfg.get("dataloader_batch_size", 2))
    num_workers = int(validation_cfg.get("dataloader_num_workers", 0))
    result: dict[str, Any] = {
        "dataloader_ok": False,
        "batch_keys": [],
        "batch_shapes": {},
    }
    try:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        for batch in loader:
            result["batch_keys"] = sorted(str(k) for k in batch)
            result["batch_shapes"] = {
                str(k): list(v.shape) for k, v in batch.items() if hasattr(v, "shape")
            }
            result["dataloader_ok"] = True
            break
    except Exception as exc:  # noqa: BLE001
        result["dataloader_error"] = {
            "code": "dataloader_smoke_failed",
            "message": f"DataLoader smoke failed: {exc}",
            "details": traceback.format_exc(),
        }
    return result


def _collect_file_list(output_root: Path) -> dict[str, Any]:
    meta_info = output_root / "meta" / "info.json"
    data_files = sorted(str(p.relative_to(output_root)) for p in output_root.rglob("data/**/*") if p.is_file())
    video_files = sorted(str(p.relative_to(output_root)) for p in output_root.rglob("videos/**/*") if p.is_file())
    sidecar_files = sorted(
        str(p.relative_to(output_root)) for p in (output_root / "meta" / "rosclaw").rglob("*") if p.is_file()
    ) if (output_root / "meta" / "rosclaw").exists() else []
    return {
        "meta_info": meta_info.exists(),
        "data_files": data_files,
        "video_files": video_files,
        "sidecar_files": sidecar_files,
    }


# ------------------------------------------------------------------
# validate_dataset
# ------------------------------------------------------------------
def _op_validate_dataset(request: dict[str, Any]) -> dict[str, Any]:
    output_dir = request.get("output_dir", "")
    repo_id = request.get("repo_id", "")
    validation_cfg = request.get("validation", {})

    if not output_dir:
        return _error_response("validate_dataset", "dataset_validate_failed", "output_dir is empty", "")
    if not repo_id:
        return _error_response("validate_dataset", "dataset_validate_failed", "repo_id is empty", "")

    output_root = Path(output_dir)
    if not output_root.exists():
        return _error_response(
            "validate_dataset",
            "dataset_validate_failed",
            f"Dataset directory not found: {output_dir}",
            "",
        )

    validate_start = time.perf_counter()
    validation = _validate_dataset(
        output_root,
        repo_id,
        validation_cfg,
        expected_frames=0,
        expected_episodes=0,
    )
    validate_time = round(time.perf_counter() - validate_start, 3)

    files = _collect_file_list(output_root)

    return _ok_response(
        "validate_dataset",
        output_dir=str(output_root),
        repo_id=repo_id,
        files=files,
        validation=validation,
        timing={"validate_time_sec": validate_time},
    )


# ------------------------------------------------------------------
# smoke_dataloader
# ------------------------------------------------------------------
def _op_smoke_dataloader(request: dict[str, Any]) -> dict[str, Any]:
    output_dir = request.get("output_dir", "")
    repo_id = request.get("repo_id", "")
    validation_cfg = request.get("validation", {})

    if not output_dir:
        return _error_response("smoke_dataloader", "dataset_validate_failed", "output_dir is empty", "")
    if not repo_id:
        return _error_response("smoke_dataloader", "dataset_validate_failed", "repo_id is empty", "")

    output_root = Path(output_dir)
    if not output_root.exists():
        return _error_response(
            "smoke_dataloader",
            "dataset_validate_failed",
            f"Dataset directory not found: {output_dir}",
            "",
        )

    try:
        import os

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        os.environ["HF_HUB_OFFLINE"] = "1"
        dataset = LeRobotDataset(repo_id, root=str(output_root))
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            "smoke_dataloader",
            "dataset_load_failed",
            f"Failed to load dataset: {exc}",
            traceback.format_exc(),
        )

    result = _run_dataloader_smoke(dataset, validation_cfg)
    validation: dict[str, Any] = {
        "load_ok": True,
        "index_ok": True,
        "dataloader_ok": result.get("dataloader_ok", False),
        "num_frames": len(dataset),
        "num_episodes": int(getattr(dataset, "num_episodes", 0)),
        "sample_keys": [],
        "sample_image_keys": [],
        "batch_keys": result.get("batch_keys", []),
        "batch_shapes": result.get("batch_shapes", {}),
    }
    if result.get("dataloader_error"):
        validation["error"] = result["dataloader_error"]

    return _ok_response(
        "smoke_dataloader",
        output_dir=str(output_root),
        repo_id=repo_id,
        validation=validation,
    )


if __name__ == "__main__":
    sys.exit(main())
