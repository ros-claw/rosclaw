"""Worker-side service for the persistent LeRobot policy runtime.

This module runs **inside the LeRobot Python interpreter** and is allowed to
import ``torch`` and ``lerobot``.  It must not be imported by the ROSClaw core
Python 3.11 runtime.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.policy_cache import materialize_policy_path
from rosclaw.integrations.lerobot.policy_runtime.protocol import (
    RUNTIME_PROTOCOL_VERSION,
    encode_response,
    parse_line,
)


class PolicyWorkerService:
    """Long-lived worker that loads a policy once and serves many inferences."""

    def __init__(self, *, device: str = "cpu", dtype: str = "auto", allow_network: bool = False):
        self.device = device
        self.dtype = dtype
        self.allow_network = allow_network
        self.policy_path: str | None = None
        self.local_policy_path: Path | None = None
        self.policy: Any = None
        self.policy_metadata: dict[str, Any] = {}
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.sessions: dict[str, dict[str, Any]] = {}
        self._shutting_down = False

    def run(self, stdin=None, stdout=None) -> None:
        """Read JSONL requests from stdin and write responses to stdout."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            request = parse_line(line)
            if request is None:
                continue
            response = self._dispatch(request.method, request.params, request.id)
            stdout.write(response)
            stdout.flush()
            if self._shutting_down:
                break

    def _dispatch(self, method: str, params: dict[str, Any], request_id: str) -> str:
        handler = getattr(self, f"_handle_{method.lower()}", None)
        if handler is None:
            return encode_response(
                request_id,
                error={"code": "unknown_method", "message": f"Unknown method: {method}"},
            )
        try:
            result = handler(params)
            return encode_response(request_id, result=result)
        except Exception as exc:  # noqa: BLE001
            return encode_response(
                request_id,
                error={"code": "worker_error", "message": f"{type(exc).__name__}: {exc}"},
            )

    def _handle_hello(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol_version": RUNTIME_PROTOCOL_VERSION,
            "worker": "rosclaw.integrations.lerobot.policy_worker_service",
        }

    def _handle_probe(self, params: dict[str, Any]) -> dict[str, Any]:
        import importlib.util

        import torch

        return {
            "status": "ok",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "lerobot_importable": importlib.util.find_spec("lerobot") is not None,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "policy_loaded": self.policy is not None,
            "policy_path": self.policy_path,
        }

    def _handle_load_policy(self, params: dict[str, Any]) -> dict[str, Any]:
        # LeRobot 0.6.x moved the factory to lerobot.policies.factory.
        try:
            from lerobot.policies.factory import (
                get_policy_class,
                make_pre_post_processors,
            )
        except Exception:  # pragma: no cover - older LeRobot layout
            from lerobot.common.policies.factory import (
                get_policy_class,
                make_pre_post_processors,
            )

        policy_path = params["policy_path"]
        revision = params.get("revision", "main")
        device = params.get("device", self.device)
        allow_network = params.get("allow_network", self.allow_network)

        materialized = materialize_policy_path(
            policy_path,
            revision=revision,
            allow_network=allow_network,
            force_download=False,
        )
        local_path = (
            materialized.local_path
            if hasattr(materialized, "local_path")
            else Path(materialized)
        )

        policy_type = self._read_policy_type(local_path)
        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(str(local_path))
        policy.eval()
        if hasattr(policy.config, "device"):
            policy.config.device = device
        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(local_path),
        )
        self._move_processor_to_device(preprocessor, device)
        self._move_processor_to_device(postprocessor, device)

        self.policy_path = policy_path
        self.local_policy_path = local_path
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.policy_metadata = self._extract_metadata(policy, local_path, policy_type)

        return {
            "status": "ok",
            "policy_path": policy_path,
            "local_policy_path": str(local_path),
            "policy_metadata": self.policy_metadata,
            "device": device,
        }

    def _read_policy_type(self, local_path: Path) -> str:
        """Read the policy type (e.g. 'act') from the local config."""
        import json

        config_json = local_path / "config.json"
        if config_json.exists():
            data = json.loads(config_json.read_text(encoding="utf-8"))
        else:
            config_yaml = local_path / "config.yaml"
            import yaml

            data = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
        policy_type = data.get("type") or data.get("policy_type")
        if not policy_type:
            raise ValueError(f"Could not determine policy type from {local_path}")
        return str(policy_type)

    def _move_processor_to_device(self, processor: Any, device: str) -> None:
        """Force every DeviceProcessorStep inside a pipeline to the requested device."""
        if processor is None:
            return
        try:
            import torch

            steps = getattr(processor, "steps", None)
            if not steps:
                return
            for step in steps:
                if hasattr(step, "device"):
                    step.device = device
                if hasattr(step, "tensor_device"):
                    step.tensor_device = torch.device(device)
        except Exception:  # noqa: BLE001
            pass

    def _extract_metadata(
        self, policy: Any, local_path: Path, policy_type: str
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "policy_type": policy_type,
            "policy_hash": None,
        }
        input_features = getattr(policy.config, "input_features", None)
        output_features = getattr(policy.config, "output_features", None)
        if input_features is not None:
            metadata["input_features"] = self._features_to_dict(input_features)
        if output_features is not None:
            metadata["output_features"] = self._features_to_dict(output_features)

        # LeRobot configs do not always carry semantic action tags.  Provide a
        # conservative fallback so ROSClaw action proposals are not "unknown".
        action_feature = metadata.get("output_features", {}).get("action", {})
        action_shape = list(action_feature.get("shape", []))
        action_dim = action_shape[-1] if action_shape else 0
        representation = action_feature.get("representation")
        unit = action_feature.get("unit")
        if representation is None and action_dim > 0:
            if policy_type in ("act", "diffusion", "pi0", "tdmpc"):
                representation = "joint_position"
            else:
                representation = "unknown"
        if unit is None and action_dim > 0:
            if representation in ("joint_position", "joint_delta"):
                unit = "radian"
            elif representation in ("joint_velocity",):
                unit = "radian_per_second"
            elif representation in ("cartesian_pose", "cartesian_delta"):
                unit = "meter"
            else:
                unit = "unknown"
        if representation is not None or unit is not None:
            metadata["extra"] = {
                "action_representation": representation,
                "action_unit": unit,
            }
        return metadata

    def _features_to_dict(self, features: Any) -> dict[str, Any]:
        if isinstance(features, dict):
            return {k: self._feature_to_dict(v) for k, v in features.items()}
        return {}

    def _feature_to_dict(self, feature: Any) -> dict[str, Any]:
        if isinstance(feature, dict):
            return dict(feature)
        if hasattr(feature, "type") and hasattr(feature, "shape"):
            return {
                "type": str(feature.type),
                "shape": list(feature.shape) if hasattr(feature.shape, "__iter__") else feature.shape,
            }
        return {"repr": repr(feature)}

    def _handle_warmup(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.policy is None:
            return {"status": "error", "error": {"code": "policy_not_loaded"}}
        # A no-op that confirms the policy and processors are resident.
        return {
            "status": "ok",
            "policy_path": self.policy_path,
            "device": self.device,
        }

    def _handle_create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params["session_id"]
        if self.policy is not None:
            self.policy.reset()
        self.sessions[session_id] = {
            "body_id": params.get("body_id"),
            "context": params.get("context", {}),
            "created_at": time.time(),
            "last_step_index": -1,
        }
        return {"status": "ok", "session_id": session_id}

    def _handle_reset_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params["session_id"]
        if session_id not in self.sessions:
            return {"status": "error", "error": {"code": "session_not_found"}}
        if self.policy is not None:
            self.policy.reset()
        self.sessions[session_id]["last_step_index"] = -1
        return {"status": "ok", "session_id": session_id}

    def _handle_infer(self, params: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        if self.policy is None:
            return {"status": "error", "error": {"code": "policy_not_loaded"}}

        session_id = params["session_id"]
        observation = params["observation"]
        step_index = params.get("step_index")
        return_chunk = params.get("return_chunk", True)

        session = self.sessions.get(session_id)
        if session is None:
            return {"status": "error", "error": {"code": "session_not_found"}}

        if step_index is None:
            session["last_step_index"] += 1
            step_index = session["last_step_index"]
        else:
            session["last_step_index"] = step_index

        obs_tensor = self._observation_to_tensor(observation)
        model_input = obs_tensor
        if self.preprocessor is not None:
            model_input = self.preprocessor(obs_tensor)

        raw_action = self.policy.select_action(model_input)
        raw_values = raw_action.detach().cpu().numpy()

        if self.postprocessor is not None:
            processed_action = self.postprocessor(raw_action)
            processed_values = processed_action.detach().cpu().numpy()
        else:
            processed_values = raw_values

        raw_list = self._array_to_list(raw_values)
        processed_list = self._array_to_list(processed_values)
        shape = list(processed_values.shape)

        return {
            "status": "ok",
            "session_id": session_id,
            "step_index": step_index,
            "raw_action": {
                "values": raw_list,
                "shape": list(raw_values.shape),
                "dtype": str(raw_values.dtype),
            },
            "processed_action": {
                "values": processed_list,
                "shape": shape,
                "dtype": str(processed_values.dtype),
            },
            "policy_metadata": self.policy_metadata,
            "return_chunk": return_chunk,
        }

    def _observation_to_tensor(self, observation: dict[str, Any]) -> dict[str, Any]:
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
                out[key] = torch.from_numpy(arr).to(self.device)
            elif key.startswith("observation.images."):
                img_path = Path(str(value))
                if not img_path.exists():
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                # (H, W, C) -> (1, C, H, W)
                arr = np.transpose(arr, (2, 0, 1))[None, ...]
                out[key] = torch.from_numpy(arr).to(self.device)
            else:
                out[key] = value
        return out

    def _array_to_list(self, arr: Any) -> list[Any]:
        if hasattr(arr, "tolist"):
            return arr.tolist()
        return list(arr)

    def _handle_health(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "policy_loaded": self.policy is not None,
            "policy_path": self.policy_path,
            "active_sessions": len(self.sessions),
        }

    def _handle_close_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params["session_id"]
        self.sessions.pop(session_id, None)
        return {"status": "ok", "session_id": session_id}

    def _handle_unload_policy(self, params: dict[str, Any]) -> dict[str, Any]:
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_metadata = {}
        self.policy_path = None
        self.local_policy_path = None
        return {"status": "ok"}

    def _handle_shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        self._shutting_down = True
        return {"status": "ok"}
