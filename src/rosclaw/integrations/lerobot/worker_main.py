#!/usr/bin/env python3
"""LeRobot worker entry point.

This file is intentionally standalone: it is executed by the LeRobot runtime
Python and must not import ``rosclaw``.  It only uses the standard library plus
numpy/torch/lerobot inside the operation functions.

Usage:
    /path/to/lerobot/python worker_main.py \
        --request-json /tmp/request.json \
        --output-json /tmp/response.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORKER_SCHEMA_VERSION = "rosclaw.lerobot.worker.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeRobot policy worker for ROSClaw")
    parser.add_argument("--request-json", required=True, help="Path to request JSON")
    parser.add_argument("--output-json", required=True, help="Path to write response JSON")
    args = parser.parse_args(argv)

    request_path = Path(args.request_json)
    output_path = Path(args.output_json)

    try:
        request = _load_json(request_path)
    except Exception as exc:  # noqa: BLE001
        response = _error_response(
            "inspect",
            "worker_invalid_json",
            f"Failed to read request JSON: {exc}",
            traceback.format_exc(),
        )
        _write_json(output_path, response)
        return 1

    op = request.get("op", "inspect")
    try:
        if op == "inspect":
            response = _op_inspect(request)
        elif op == "load_test":
            response = _op_load_test(request)
        elif op == "infer":
            response = _op_infer(request)
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
            "policy_infer_failed" if op == "infer" else "policy_load_failed",
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
    details: str,
) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "error",
        "op": op,
        "policy_path": "",
        "real_model_loaded": False,
        "real_inference": False,
        "policy_metadata": {},
        "timing": {},
        "runtime": _runtime_info(),
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
    }


def _ok_response(
    op: str,
    policy_path: str,
    *,
    real_model_loaded: bool = False,
    real_inference: bool = False,
    policy_metadata: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "ok",
        "op": op,
        "policy_path": policy_path,
        "real_model_loaded": real_model_loaded,
        "real_inference": real_inference,
        "policy_metadata": policy_metadata or {},
        "timing": timing or {},
        "runtime": _runtime_info(),
    }
    if action is not None:
        response["action"] = action
    return response


def _runtime_info() -> dict[str, Any]:
    """Return runtime info without heavy imports."""
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
# Inspect
# ------------------------------------------------------------------
def _op_inspect(request: dict[str, Any]) -> dict[str, Any]:
    policy_path = request.get("policy_path", "")
    allow_network = bool(request.get("allow_network", False))

    if not policy_path:
        return _error_response("inspect", "policy_config_not_found", "policy_path is empty", "")

    local_path = Path(policy_path)
    if local_path.is_dir():
        metadata = _read_local_policy_metadata(local_path)
        if metadata is None:
            return _error_response(
                "inspect",
                "policy_config_not_found",
                f"No readable config found in {local_path}",
                "",
            )
        return _ok_response(
            "inspect",
            policy_path,
            policy_metadata=metadata,
        )

    # Treat as HF repo id.
    if "/" in policy_path:
        if not allow_network:
            return _error_response(
                "inspect",
                "network_disabled",
                f"Policy path '{policy_path}' looks like a Hugging Face repo id. "
                "Set allow_network=true to fetch config.json.",
                "",
            )
        metadata = _fetch_hf_config_metadata(policy_path, request.get("revision", "main"))
        if metadata is None:
            return _error_response(
                "inspect",
                "network_download_required",
                f"Could not fetch config for {policy_path}",
                "",
            )
        return _ok_response("inspect", policy_path, policy_metadata=metadata)

    return _error_response(
        "inspect",
        "policy_config_not_found",
        f"Policy path not found: {policy_path}",
        "",
    )


def _read_local_policy_metadata(policy_dir: Path) -> dict[str, Any] | None:
    """Read a LeRobot policy config from a local directory."""
    config_json = policy_dir / "config.json"
    if config_json.exists():
        try:
            with open(config_json, encoding="utf-8") as f:
                data = json.load(f)
            return _normalize_metadata(data)
        except Exception as exc:  # noqa: BLE001
            return {
                "policy_type": "unknown",
                "config_found": True,
                "config_path": str(config_json),
                "raw_config_keys": sorted(_safe_keys(config_json)),
                "parse_error": str(exc),
            }

    config_yaml = policy_dir / "config.yaml"
    if config_yaml.exists():
        try:
            import yaml

            with open(config_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return _normalize_metadata(data)
        except Exception as exc:  # noqa: BLE001
            return {
                "policy_type": "unknown",
                "config_found": True,
                "config_path": str(config_yaml),
                "raw_config_keys": sorted(_safe_keys(config_yaml)),
                "parse_error": str(exc),
            }

    return None


def _normalize_metadata(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "policy_type": "unknown",
            "config_found": False,
            "raw_config_keys": [],
        }

    policy_type = "unknown"
    for key in ("policy_type", "type", "name"):
        if key in data:
            policy_type = str(data[key])
            break
    if isinstance(data.get("policy"), dict) and "name" in data["policy"]:
        policy_type = str(data["policy"]["name"])

    input_features = data.get("input_features", {})
    output_features = data.get("output_features", {})
    if isinstance(data.get("policy"), dict):
        input_features = data["policy"].get("input_features", input_features)
        output_features = data["policy"].get("output_features", output_features)

    return {
        "policy_type": policy_type,
        "config_found": True,
        "input_features": input_features,
        "output_features": output_features,
        "raw_config_keys": sorted(data.keys()),
    }


def _safe_keys(path: Path) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.keys()) if isinstance(data, dict) else []
    except Exception:  # noqa: BLE001
        return []


def _fetch_hf_config_metadata(repo_id: str, revision: str) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
        )
        return _read_local_policy_metadata(Path(config_path).parent)
    except Exception as exc:  # noqa: BLE001
        return {
            "policy_type": "unknown",
            "config_found": False,
            "raw_config_keys": [],
            "parse_error": str(exc),
        }


# ------------------------------------------------------------------
# Load-test / infer helpers
# ------------------------------------------------------------------
def _load_policy_for_inference(
    policy_path: str,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Load a LeRobot policy and return (policy, metadata).

    This helper is intentionally defensive: different LeRobot versions expose
    the factory/config APIs in different places, and the checkpoint format
    differs between training snapshots and published pretrained policies. It
    tries, in order:

    1. Build a ``PreTrainedConfig`` from ``config.json`` and load via
       ``PolicyClass.from_pretrained(pretrained_path, config=cfg)``.
    2. Build the same config and load via ``make_policy(cfg, env_cfg=...)``.
    3. Legacy paths for older LeRobot layouts.

    If none succeed it raises ``RuntimeError("policy_api_unsupported: ...")``.
    """

    local_path = Path(policy_path)
    if not local_path.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_path}")

    cfg_dict, policy_type = _read_config_dict(local_path)
    cfg = _build_lerobot_config_object(local_path, cfg_dict, policy_type, device)

    errors: list[str] = []
    policy = None
    weights_loaded = False

    # Modern path (LeRobot 0.6.x): policy class loads its own weights.
    try:
        from lerobot.policies.factory import get_policy_class

        policy = get_policy_class(policy_type).from_pretrained(
            pretrained_name_or_path=str(local_path),
            config=cfg,
        )
        weights_loaded = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"policy_cls.from_pretrained: {exc}")

    # Alternative modern path via the factory.
    if policy is None:
        try:
            from lerobot.envs.factory import make_env_config
            from lerobot.policies.factory import make_policy

            env_type = _guess_env_type(local_path) or "aloha"
            policy = make_policy(cfg, env_cfg=make_env_config(env_type))
            weights_loaded = bool(cfg.pretrained_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"make_policy: {exc}")

    # Legacy paths for older LeRobot layouts.
    if policy is None:
        try:
            from lerobot.common.policies.factory import make_policy as make_policy_old

            policy = make_policy_old(_load_legacy_config(local_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"legacy common.policies.factory: {exc}")

    if policy is None:
        try:
            from lerobot import make_policy as make_policy_alt

            policy = make_policy_alt(_load_legacy_config(local_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"legacy lerobot.make_policy: {exc}")

    if policy is None:
        raise RuntimeError(
            f"policy_api_unsupported: Could not load policy using any LeRobot API: {'; '.join(errors)}"
        )

    # Fallback: load a separate checkpoint file if the policy did not already
    # load weights (e.g. when the legacy factory path returns a fresh model).
    if not weights_loaded:
        ckpt = _find_checkpoint(local_path)
        if ckpt is not None:
            try:
                state_dict = _load_checkpoint(ckpt, device)
                missing, unexpected = policy.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[worker] checkpoint missing keys: {missing}", file=sys.stderr)
                if unexpected:
                    print(f"[worker] checkpoint unexpected keys: {unexpected}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] checkpoint load skipped: {exc}", file=sys.stderr)

    policy.to(device)
    policy.eval()

    metadata = _read_local_policy_metadata(local_path) or {}
    return policy, metadata


def _read_config_dict(local_path: Path) -> tuple[dict[str, Any], str]:
    """Read the policy config and return (raw_dict, policy_type)."""
    import json

    for name in ("config.json", "config.yaml"):
        cfg_path = local_path / name
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                if cfg_path.suffix == ".json":
                    data = json.load(f)
                else:
                    import yaml

                    data = yaml.safe_load(f)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not parse {cfg_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected config object in {cfg_path}, got {type(data).__name__}")
        policy_type = data.get("type") or data.get("policy_type")
        if not policy_type:
            raise RuntimeError(f"No 'type' or 'policy_type' found in {cfg_path}")
        return data, str(policy_type)

    raise FileNotFoundError(f"No config.json or config.yaml in {local_path}")


def _build_lerobot_config_object(
    local_path: Path,
    cfg_dict: dict[str, Any],
    policy_type: str,
    device: str,
) -> Any:
    """Convert a raw config dict into a LeRobot ``PreTrainedConfig`` object."""
    from lerobot.configs import NormalizationMode
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_policy_config

    cfg_dict = dict(cfg_dict)
    cfg_dict.pop("type", None)
    cfg_dict.pop("policy_type", None)

    def _to_features(raw: dict[str, Any]) -> dict[str, PolicyFeature]:
        return {
            key: PolicyFeature(type=FeatureType(str(value["type"]).upper()), shape=tuple(value["shape"]))
            for key, value in raw.items()
        }

    if "input_features" in cfg_dict:
        cfg_dict["input_features"] = _to_features(cfg_dict["input_features"])
    if "output_features" in cfg_dict:
        cfg_dict["output_features"] = _to_features(cfg_dict["output_features"])
    if "normalization_mapping" in cfg_dict:
        cfg_dict["normalization_mapping"] = {
            key: NormalizationMode(str(value).upper())
            for key, value in cfg_dict["normalization_mapping"].items()
        }

    cfg = make_policy_config(policy_type, **cfg_dict)
    cfg.pretrained_path = str(local_path)
    cfg.device = device
    return cfg


def _guess_env_type(local_path: Path) -> str | None:
    """Infer an env type from train_config.json, if present."""
    import json

    train_cfg = local_path / "train_config.json"
    if not train_cfg.exists():
        return None
    try:
        with open(train_cfg, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("env", {}).get("type")
    except Exception:  # noqa: BLE001
        return None


def _load_legacy_config(local_path: Path) -> dict[str, Any]:
    """Return a raw config dict for legacy factory APIs."""
    import json

    for name in ("config.yaml", "config.json"):
        cfg_path = local_path / name
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                if cfg_path.suffix == ".json":
                    return json.load(f)
                import yaml

                return yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not parse {cfg_path}: {exc}") from exc
    raise FileNotFoundError(f"No config.json or config.yaml in {local_path}")


def _load_checkpoint(ckpt_path: Path, device: str) -> dict[str, Any]:
    """Load a checkpoint state dict, preferring safetensors when available."""
    import torch

    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            return load_file(str(ckpt_path), device=str(device))
        except Exception:  # noqa: BLE001
            pass
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def _find_checkpoint(policy_dir: Path) -> Path | None:
    candidates = [
        policy_dir / "model.safetensors",
        policy_dir / "pytorch_model.bin",
        policy_dir / "model.pt",
        policy_dir / "checkpoints" / "last" / "pretrained_model",
        policy_dir / "checkpoints" / "last" / "pretrained_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Generic search for safetensors / pth.
    for ext in ("*.safetensors", "*.pth", "*.pt", "*.bin"):
        matches = list(policy_dir.rglob(ext))
        if matches:
            return matches[0]
    return None


def _build_observation_tensor(
    observation: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Convert a flat observation dict into LeRobot-style tensors."""
    import numpy as np
    import torch
    from PIL import Image

    out: dict[str, Any] = {}
    for key, value in observation.items():
        if key == "task":
            out[key] = value
        elif key == "observation.state" or key.startswith("observation.state"):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            out[key] = torch.from_numpy(arr).to(device)
        elif key.startswith("observation.images."):
            img_path = Path(str(value))
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            # (H, W, C) -> (1, C, H, W)
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
            out[key] = torch.from_numpy(arr).to(device)
        else:
            out[key] = value
    return out


def _serialize_action(action: Any) -> dict[str, Any]:
    """Convert a LeRobot action to a JSON-serializable dict.

    Action chunks (e.g. ``[100, 14]``) keep their original shape so the
    ROSClaw side can decide how to consume them.
    """
    import numpy as np

    # Unwrap common container types first.
    if isinstance(action, dict):
        for key in ("action", "actions", "output"):
            if key in action and action[key] is not None:
                return _serialize_action(action[key])
        # Fallback: use the first tensor-like value.
        for value in action.values():
            if hasattr(value, "detach") or isinstance(value, np.ndarray):
                return _serialize_action(value)
        action = list(action.values())

    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    if isinstance(action, np.ndarray):
        arr = action
        values = arr.tolist()
    elif isinstance(action, (list, tuple)):
        values = list(action)
        arr = np.asarray(values)
    else:
        arr = np.asarray([float(action)])
        values = arr.tolist()

    # Squeeze a leading batch dimension of size 1 so that a single-step
    # inference returns [action_dim] instead of [1, action_dim].
    while arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr.reshape(arr.shape[1:])
        values = arr.tolist()

    shape = [int(dim) for dim in arr.shape]
    action_type = "lerobot_action_chunk" if len(shape) == 2 and shape[0] > 1 else "raw_lerobot_action"
    dtype = str(arr.dtype) if arr.dtype else "float32"
    if dtype.startswith("float"):
        dtype = "float32" if arr.dtype == np.float32 else str(arr.dtype)
    return {
        "type": action_type,
        "values": values,
        "shape": shape,
        "dtype": dtype,
    }


# ------------------------------------------------------------------
# Load-test
# ------------------------------------------------------------------
def _op_load_test(request: dict[str, Any]) -> dict[str, Any]:
    policy_path = request.get("policy_path", "")
    device = request.get("device", "cpu")

    if not policy_path:
        return _error_response("load_test", "policy_config_not_found", "policy_path is empty", "")

    start = time.perf_counter()
    try:
        policy, metadata = _load_policy_for_inference(policy_path, device)
    except FileNotFoundError as exc:
        return _error_response("load_test", "policy_config_not_found", str(exc), "")
    except RuntimeError as exc:
        if "policy_api_unsupported" in str(exc):
            return _error_response("load_test", "policy_api_unsupported", str(exc), "")
        return _error_response("load_test", "policy_load_failed", str(exc), traceback.format_exc())
    except Exception as exc:  # noqa: BLE001
        return _error_response("load_test", "policy_load_failed", str(exc), traceback.format_exc())

    load_time = round(time.perf_counter() - start, 3)
    return _ok_response(
        "load_test",
        policy_path,
        real_model_loaded=True,
        policy_metadata=metadata,
        timing={"load_time_sec": load_time},
    )


# ------------------------------------------------------------------
# Infer
# ------------------------------------------------------------------
def _op_infer(request: dict[str, Any]) -> dict[str, Any]:
    policy_path = request.get("policy_path", "")
    device = request.get("device", "cpu")
    observation = request.get("observation", {})

    if not policy_path:
        return _error_response("infer", "policy_config_not_found", "policy_path is empty", "")

    load_start = time.perf_counter()
    try:
        policy, metadata = _load_policy_for_inference(policy_path, device)
    except FileNotFoundError as exc:
        return _error_response("infer", "policy_config_not_found", str(exc), "")
    except RuntimeError as exc:
        if "policy_api_unsupported" in str(exc):
            return _error_response("infer", "policy_api_unsupported", str(exc), "")
        return _error_response("infer", "policy_load_failed", str(exc), traceback.format_exc())
    except Exception as exc:  # noqa: BLE001
        return _error_response("infer", "policy_load_failed", str(exc), traceback.format_exc())

    load_time = round(time.perf_counter() - load_start, 3)

    infer_start = time.perf_counter()
    try:
        obs_tensors = _build_observation_tensor(observation, device)
        # Try the standard select_action API.
        if hasattr(policy, "select_action"):
            raw_action = policy.select_action(obs_tensors)
        elif hasattr(policy, "forward"):
            raw_action = policy.forward(obs_tensors)
        else:
            return _error_response(
                "infer",
                "policy_api_unsupported",
                "Policy has no select_action or forward method",
                "",
            )
        action = _serialize_action(raw_action)
    except Exception as exc:  # noqa: BLE001
        return _error_response("infer", "policy_infer_failed", str(exc), traceback.format_exc())

    infer_time = round(time.perf_counter() - infer_start, 3)
    return _ok_response(
        "infer",
        policy_path,
        real_model_loaded=True,
        real_inference=True,
        policy_metadata=metadata,
        action=action,
        timing={"load_time_sec": load_time, "infer_time_sec": infer_time},
    )


if __name__ == "__main__":
    sys.exit(main())
