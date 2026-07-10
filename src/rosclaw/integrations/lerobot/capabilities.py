"""LeRobot integration capability registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rosclaw.integrations.registry import IntegrationCapability, IntegrationRegistry

if TYPE_CHECKING:
    from rosclaw.integrations.lerobot.schemas import IntegrationCapability, IntegrationReport


class LeRobotIntegration:
    """Static metadata for the LeRobot integration."""

    name = "lerobot"
    version = "0.2.0"

    @classmethod
    def report(cls) -> IntegrationReport:
        """Return the current integration report.

        This triggers a lightweight doctor check without importing torch or
        lerobot at the top level.
        """
        from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor

        registry_check = {
            "provider_type_lerobot_policy": True,
            "dataset_export_lerobot": True,
        }
        return run_lerobot_doctor(registry_check=registry_check)


_PROVIDER_TYPE = "lerobot_policy"
_EXPORTER_NAME = "lerobot"


def register_lerobot_capabilities(registry: IntegrationRegistry) -> None:
    """Register LeRobot provider type and exporter with the integration registry."""
    from rosclaw.integrations.lerobot.dataset_exporter import LeRobotDatasetExporter
    from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider

    registry.register_integration(LeRobotIntegration.name, LeRobotIntegration)
    registry.register_provider_type(_PROVIDER_TYPE, LeRobotPolicyProvider)
    registry.register_practice_exporter(_EXPORTER_NAME, LeRobotDatasetExporter)


def _worker_available() -> tuple[bool, bool]:
    """Return (subprocess_available, in_process_available) from config/runtime."""
    try:
        from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime

        cfg = get_configured_lerobot_runtime()
        if cfg:
            return (
                bool(cfg.get("subprocess_available")),
                bool(cfg.get("in_process_available")),
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        import importlib.util

        return False, importlib.util.find_spec("lerobot") is not None
    except Exception:  # noqa: BLE001
        return False, False


def get_lerobot_capabilities() -> list[IntegrationCapability]:
    """Return the LeRobot capability list for display."""
    subprocess_ok, in_process_ok = _worker_available()
    worker_ready = subprocess_ok or in_process_ok

    return [
        IntegrationCapability(
            name="provider_type_lerobot_policy",
            kind="provider",
            enabled=True,
            experimental=True,
            description="LeRobot policy provider registration",
        ),
        IntegrationCapability(
            name="real_policy_inspect",
            kind="provider",
            enabled=worker_ready,
            experimental=True,
            description="Inspect LeRobot policy config/metadata without loading weights",
        ),
        IntegrationCapability(
            name="real_policy_load_test",
            kind="provider",
            enabled=worker_ready,
            experimental=True,
            description="Load LeRobot policy weights as a runtime smoke test",
        ),
        IntegrationCapability(
            name="real_policy_infer",
            kind="provider",
            enabled=worker_ready,
            experimental=True,
            description="Run real LeRobot policy inference via isolated worker (action proposal only)",
        ),
        IntegrationCapability(
            name="dataset_export_lerobot",
            kind="exporter",
            enabled=True,
            experimental=True,
            description="Export ROSClaw practice episodes to LeRobot dataset skeleton",
        ),
        IntegrationCapability(
            name="worker_subprocess",
            kind="worker",
            enabled=subprocess_ok,
            experimental=True,
            description="Run LeRobot tasks in a configured subprocess runtime",
        ),
        IntegrationCapability(
            name="worker_in_process",
            kind="worker",
            enabled=in_process_ok,
            experimental=True,
            description="Run LeRobot tasks in the current ROSClaw interpreter",
        ),
        IntegrationCapability(
            name="train",
            kind="train_backend",
            enabled=False,
            experimental=True,
            description="LeRobot training backend (future)",
        ),
        IntegrationCapability(
            name="eval",
            kind="eval_backend",
            enabled=False,
            experimental=True,
            description="LeRobot evaluation backend (future)",
        ),
        IntegrationCapability(
            name="rollout",
            kind="rollout_backend",
            enabled=False,
            experimental=True,
            description="LeRobot rollout backend (future)",
        ),
        IntegrationCapability(
            name="reward",
            kind="reward_backend",
            enabled=False,
            experimental=True,
            description="LeRobot reward backend (future)",
        ),
    ]
