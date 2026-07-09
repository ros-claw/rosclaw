"""LeRobot policy provider for ROSClaw."""

from __future__ import annotations

from typing import Any

from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class LeRobotPolicyProvider(Provider):
    """ROSClaw provider adapter for LeRobot policies.

    P0 implementation:
    - Dry-run returns a deterministic sample action and reports what the real
      inference path would do.
    - If LeRobot is installed and ``dry_run=False``, it attempts a lightweight
      smoke import to validate the environment but still returns a sample action.
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
        """No heavy loading in P0."""
        self._healthy = True

    async def unload(self) -> None:
        """No resources to release in P0."""
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
        """Run LeRobot policy inference (dry-run or smoke)."""
        capability = request.capability or "lerobot.policy.infer"
        if capability not in self.capabilities:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not support capability '{capability}'",
                provider=self.name,
            )

        inputs = request.inputs or {}
        dry_run = inputs.get("dry_run", False)

        # Optional real-import smoke when LeRobot is installed and not dry-run.
        lerobot_smoke: dict[str, Any] = {"import_ok": False}
        if not dry_run:
            try:
                import importlib.util

                if importlib.util.find_spec("lerobot") is not None:
                    import lerobot

                    lerobot_smoke = {
                        "import_ok": True,
                        "version": getattr(lerobot, "__version__", None),
                        "note": "Real policy inference is not implemented in P0.",
                    }
            except Exception as exc:  # noqa: BLE001
                lerobot_smoke = {"import_ok": False, "error": str(exc)}

        sample_action = self._sample_action()
        result = {
            "provider": self.name,
            "capability": capability,
            "dry_run": dry_run,
            "action": sample_action,
            "action_space": self.manifest.embodiment.action_space or [],
            "safety": {
                "executable": False,
                "requires_guard": True,
                "requires_workspace_check": True,
                "requires_collision_check": True,
                "max_action_norm": self.manifest.safety.max_action_norm,
            },
            "lerobot_smoke": lerobot_smoke,
            "note": (
                "Dry-run returned a sample action. "
                "Real LeRobot policy inference is not yet implemented."
            ),
        }

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=capability,
            status="ok",
            result=result,
            latency_ms=0,
        )

    def _sample_action(self) -> list[float]:
        """Return a deterministic zero-ish sample action."""
        return [0.0] * self._action_dim
