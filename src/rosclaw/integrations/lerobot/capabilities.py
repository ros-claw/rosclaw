"""LeRobot integration capability registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rosclaw.integrations.registry import IntegrationCapability, IntegrationRegistry

if TYPE_CHECKING:
    from rosclaw.integrations.lerobot.schemas import IntegrationCapability, IntegrationReport


class LeRobotIntegration:
    """Static metadata for the LeRobot integration."""

    name = "lerobot"
    version = "0.1.0"

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


def get_lerobot_capabilities() -> list[IntegrationCapability]:
    """Return the LeRobot capability list for display."""
    return [
        IntegrationCapability(
            name="provider_type_lerobot_policy",
            kind="provider",
            enabled=True,
            experimental=True,
            description="LeRobot policy provider (dry-run in P0; import smoke in P0.1)",
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
            enabled=False,
            experimental=True,
            description="Run LeRobot tasks in a configured subprocess runtime",
        ),
        IntegrationCapability(
            name="worker_in_process",
            kind="worker",
            enabled=False,
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
