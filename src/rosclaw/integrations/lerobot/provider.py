"""LeRobot policy provider for ROSClaw.

P1 semantics:
- ``lerobot.policy.inspect`` reads policy config/metadata without loading weights.
- ``lerobot.policy.load_test`` loads weights but does not run inference.
- ``lerobot.policy.infer`` performs one real policy inference via the LeRobot
  subprocess worker, but the returned action is always an **action proposal**:
  ``not_executed=true``, ``requires_sandbox=true``, ``executable=false``.
- ``--dry-run`` still returns a deterministic sample action for backward
  compatibility.
- The provider never executes actions on hardware.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.worker_runner import LeRobotWorkerRunner
from rosclaw.integrations.lerobot.worker_schema import WorkerRequest
from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class LeRobotPolicyProvider(Provider):
    """ROSClaw provider adapter for LeRobot policies."""

    name = "lerobot_policy_provider"
    version = "0.2.0"
    capabilities = [
        "lerobot.policy.inspect",
        "lerobot.policy.load_test",
        "lerobot.policy.infer",
    ]

    def __init__(self, manifest: ProviderManifest):
        super().__init__(manifest)
        self._action_dim = self._infer_action_dim(manifest)

    @staticmethod
    def _infer_action_dim(manifest: ProviderManifest) -> int:
        """Infer action dimension from manifest or default to 7."""
        action_space = manifest.embodiment.action_space
        if action_space:
            return len(action_space)
        extra = manifest.extra or {}
        shape = extra.get("action_shape")
        if isinstance(shape, list) and shape:
            return int(shape[0])
        return 7

    async def load(self) -> None:
        """No heavy loading at provider startup."""
        self._healthy = True

    async def unload(self) -> None:
        """No resources to release."""
        self._healthy = False

    async def health(self) -> dict[str, Any]:
        return {
            "ok": self._healthy,
            "provider": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "load_error": self._load_error,
            "action_dim": self._action_dim,
        }

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        """Dispatch inspect, load_test, or infer through the LeRobot worker."""
        capability = request.capability or "lerobot.policy.infer"
        if capability not in self.capabilities:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not support capability '{capability}'",
                provider=self.name,
            )

        inputs = request.inputs or {}
        dry_run = inputs.get("dry_run", False)

        if dry_run and capability == "lerobot.policy.infer":
            return self._dry_run_response(request)

        if inputs.get("execute"):
            return self._blocked_response(
                request,
                "P1 LeRobot provider only returns action proposals. "
                "Remove --execute or set execute=false.",
            )

        return await self._worker_response(request, capability)

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------
    def _dry_run_response(self, request: ProviderRequest) -> ProviderResponse:
        result = {
            "provider": self.name,
            "capability": request.capability,
            "mode": "dry_run",
            "dry_run": True,
            "real_inference": False,
            "not_executed": True,
            "requires_sandbox": True,
            "action_proposal": {
                "type": "sample_action",
                "values": self._sample_action(),
                "shape": [self._action_dim],
                "dtype": "float32",
                "executable": False,
                "requires_sandbox": True,
                "not_executed": True,
                "body_mapping_required": True,
                "body_compatible": False,
                "body_name": None,
            },
            "action_space": self.manifest.embodiment.action_space or [],
            "safety": self._safety_dict(),
            "message": "Dry-run returned a sample action proposal.",
        }
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            status="ok",
            result=result,
            latency_ms=0,
        )

    def _blocked_response(self, request: ProviderRequest, message: str) -> ProviderResponse:
        result = {
            "provider": self.name,
            "capability": request.capability,
            "mode": "blocked",
            "real_inference": False,
            "not_executed": True,
            "requires_sandbox": True,
            "action_proposal": None,
            "safety": self._safety_dict(),
            "message": message,
        }
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            status="blocked",
            result=result,
            errors=[message],
            latency_ms=0,
        )

    async def _worker_response(self, request: ProviderRequest, capability: str) -> ProviderResponse:
        """Build a worker request, run it, and translate to ProviderResponse."""
        inputs = request.inputs or {}
        policy_path = inputs.get("policy.path") or inputs.get("policy_path")
        if not policy_path:
            return self._failed_response(
                request,
                "Missing required input: policy.path",
                "policy_config_not_found",
            )

        op_map = {
            "lerobot.policy.inspect": "inspect",
            "lerobot.policy.load_test": "load_test",
            "lerobot.policy.infer": "infer",
        }
        op = op_map[capability]

        observation: dict[str, Any] = {}
        if op == "infer":
            try:
                observation = adapt_observation_for_worker(inputs)
            except (ValueError, FileNotFoundError) as exc:
                return self._failed_response(request, str(exc), "observation_schema_mismatch")

        worker_request = WorkerRequest(
            op=op,  # type: ignore[arg-type]
            policy_path=str(policy_path),
            revision=inputs.get("revision", "main"),
            device=inputs.get("device", "cpu"),
            allow_network=bool(inputs.get("allow_network", False)),
            timeout_sec=int(inputs.get("timeout_sec", 120)),
            observation=observation,
        )

        runner = LeRobotWorkerRunner(timeout_sec=worker_request.timeout_sec)
        t0 = time.perf_counter()

        try:
            loop = asyncio.get_event_loop()
            worker_response = await loop.run_in_executor(None, runner.run, worker_request)
        except Exception as exc:  # noqa: BLE001
            return self._failed_response(request, f"Worker runner error: {exc}", "worker_process_failed")

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if not worker_response.ok:
            return self._failed_response(
                request,
                worker_response.error_message(),
                worker_response.error_code(),
                details=worker_response.error.details if worker_response.error else "",
            )

        mode = {
            "inspect": "policy_inspect",
            "load_test": "policy_load_test",
            "infer": "real_policy_infer",
        }[op]

        result: dict[str, Any] = {
            "provider": self.name,
            "capability": request.capability,
            "mode": mode,
            "real_inference": worker_response.real_inference,
            "real_model_loaded": worker_response.real_model_loaded,
            "not_executed": True,
            "requires_sandbox": True,
            "policy_path": worker_response.policy_path,
            "policy_metadata": worker_response.policy_metadata,
            "action_space": self.manifest.embodiment.action_space or [],
            "safety": self._safety_dict(),
            "timing": worker_response.timing.to_dict(),
            "worker_runtime": worker_response.runtime,
        }

        if op == "infer":
            result["action_proposal"] = adapt_action_to_proposal(worker_response.action)
            result["message"] = "LeRobot policy inference completed; action is a proposal only."
        else:
            result["action_proposal"] = None
            result["message"] = f"LeRobot {op} completed successfully."

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            status="ok",
            result=result,
            latency_ms=latency_ms,
        )

    def _failed_response(
        self,
        request: ProviderRequest,
        message: str,
        error_code: str,
        details: str = "",
    ) -> ProviderResponse:
        result = {
            "provider": self.name,
            "capability": request.capability,
            "mode": "failed",
            "real_inference": False,
            "not_executed": True,
            "requires_sandbox": True,
            "action_proposal": None,
            "safety": self._safety_dict(),
            "error_code": error_code,
            "message": message,
        }
        if details:
            result["error_details"] = details
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            status="failed",
            result=result,
            errors=[message],
            latency_ms=0,
        )

    def _safety_dict(self) -> dict[str, Any]:
        return {
            "executable": False,
            "requires_guard": True,
            "requires_workspace_check": True,
            "requires_collision_check": True,
            "sandbox_required": True,
            "requires_sandbox": True,
            "body_mapping_required": True,
            "body_compatible": False,
            "max_action_norm": self.manifest.safety.max_action_norm,
        }

    def _sample_action(self) -> list[float]:
        """Return a deterministic zero-ish sample action."""
        return [0.0] * self._action_dim
