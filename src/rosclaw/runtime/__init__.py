"""ROSClaw Runtime sub-package.

Contains both the e-URDF loader/registry and the Runtime Kernel v2
producer/consumer framework.
"""

from rosclaw.runtime.bus import RuntimeBus, SchemaValidationError
from rosclaw.runtime.component import RuntimeComponent, RuntimeConsumer, RuntimeProducer
from rosclaw.runtime.eurdf_loader import (
    EURDFLoader,
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotCompleteProfile,
    RobotEmbodimentProfile,
    RobotRegistry,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)
from rosclaw.runtime.event import RuntimeEvent
from rosclaw.runtime.registry import RuntimeComponentRegistry
from rosclaw.runtime.query import RuntimeQueryAPI
from rosclaw.runtime.replay import RuntimeReplay
from rosclaw.runtime.service import RuntimeKernelService

__all__ = [
    # Runtime Kernel v2
    "RuntimeBus",
    "RuntimeEvent",
    "RuntimeComponent",
    "RuntimeProducer",
    "RuntimeConsumer",
    "RuntimeComponentRegistry",
    "RuntimeKernelService",
    "RuntimeReplay",
    "RuntimeQueryAPI",
    "SchemaValidationError",
    # e-URDF / robot registry
    "EURDFLoader",
    "RobotEmbodimentProfile",
    "RobotSafetyProfile",
    "RobotCapabilityProfile",
    "RobotSimulationProfile",
    "RobotSemanticProfile",
    "RobotBenchmarkProfile",
    "RobotCompleteProfile",
    "RobotRegistry",
]
