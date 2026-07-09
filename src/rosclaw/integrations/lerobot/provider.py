"""LeRobot policy provider for ROSClaw."""

from __future__ import annotations

from typing import Any

from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.runtime import inspect_lerobot_runtime
from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class LeRobotPolicyProvider(Provider):
    """ROSClaw provider adapter for LeRobot policies.

    P0.1 semantics:
    - ``--dry-run`` returns a deterministic sample action and explicitly marks
      ``real_inference=False`` and ``not_executed=True``.
    - Non-dry-run performs a lightweight import smoke test against the
      configured LeRobot runtime (or the current interpreter if LeRobot is
      importable in-process) and returns ``action=None``. Real policy inference
      is intentionally not implemented in P0.1.
    """

    name = "lerobot_policy_provider"
    version = "0.1.0"
    capabilities = ["lerobot.policy.infer"]

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
        """No heavy loading in P0.1."""
        self._healthy = True

    async def unload(self) -> None:
        """No resources to release in P0.1."""
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
        """Run LeRobot policy inference (dry-run or import smoke only)."""
        capability = request.capability or "lerobot.policy.infer"
        if capability not in self.capabilities:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not support capability '{capability}'",
                provider=self.name,
            )

        inputs = request.inputs or {}
        dry_run = inputs.get("dry_run", False)

        if dry_run:
            return self._dry_run_response(request)

        return self._import_smoke_response(request)

    def _dry_run_response(self, request: ProviderRequest) -> ProviderResponse:
        result = {
            "provider": self.name,
            "capability": request.capability,
            "mode": "dry_run",
            "dry_run": True,
            "real_inference": False,
            "not_executed": True,
            "action": self._sample_action(),
            "action_space": self.manifest.embodiment.action_space or [],
            "safety": {
                "executable": False,
                "requires_guard": True,
                "requires_workspace_check": True,
                "requires_collision_check": True,
                "sandbox_required": True,
                "max_action_norm": self.manifest.safety.max_action_norm,
            },
            "message": (
                "Dry-run returned a sample action. "
                "Real LeRobot policy inference is not yet implemented."
            ),
        }
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            status="ok",
            result=result,
            latency_ms=0,
        )

    def _import_smoke_response(self, request: ProviderRequest) -> ProviderResponse:
        runtime_cfg = get_configured_lerobot_runtime()
        lerobot_smoke: dict[str, Any] = {"import_ok": False}

        if runtime_cfg and runtime_cfg.get("python_executable"):
            runtime = inspect_lerobot_runtime(
                runtime_cfg["python_executable"],
                mode=runtime_cfg.get("mode", "external"),
                runtime_path=runtime_cfg.get("runtime_path"),
            )
            lerobot_smoke = {
                "runtime_mode": runtime.mode,
                "python_executable": str(runtime.python_executable),
                "import_ok": runtime.state != "error" and runtime.lerobot_version is not None,
                "version": runtime.lerobot_version,
                "torch_version": runtime.torch_version,
                "cuda_available": runtime.cuda_available,
            }
        else:
            # Fallback to current interpreter if no runtime configured.
            try:
                import importlib.util

                if importlib.util.find_spec("lerobot") is not None:
                    import lerobot

                    lerobot_smoke = {
                        "runtime_mode": "current-env",
                        "python_executable": "",
                        "import_ok": True,
                        "version": getattr(lerobot, "__version__", None),
                    }
            except Exception as exc:  # noqa: BLE001
                lerobot_smoke = {"import_ok": False, "error": str(exc)}

        result = {
            "provider": self.name,
            "capability": request.capability,
            "mode": "import_smoke",
            "real_inference": False,
            "action": None,
            "action_space": self.manifest.embodiment.action_space or [],
            "safety": {
                "executable": False,
                "requires_guard": True,
                "requires_workspace_check": True,
                "requires_collision_check": True,
                "sandbox_required": True,
                "max_action_norm": self.manifest.safety.max_action_norm,
            },
            "lerobot_smoke": lerobot_smoke,
            "message": (
                "P0.1 verifies LeRobot runtime availability only. "
                "Real policy inference is planned for P1."
            ),
        }
        import_ok = bool(lerobot_smoke.get("import_ok"))
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            status="ok" if import_ok else "failed",
            result=result,
            latency_ms=0,
            errors=[] if import_ok else ["LeRobot runtime is unavailable or unsupported"],
        )

    def _sample_action(self) -> list[float]:
        """Return a deterministic zero-ish sample action."""
        return [0.0] * self._action_dim
