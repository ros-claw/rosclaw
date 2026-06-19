"""DDS collector stub.

The real implementation will subscribe to raw DDS topics.  Lazy imports ensure
this file can be loaded without optional DDS dependencies.
"""

from __future__ import annotations

import logging
import time

from rosclaw.sense.collectors.base import BodyStateCollector
from rosclaw.sense.schemas import BodyState

logger = logging.getLogger("rosclaw.sense.collectors.dds")


class DDSCollector(BodyStateCollector):
    """Collect BodyState from DDS discovery.

    This is a stub for Phase 1.  It returns an unknown state until a real DDS
    backend is implemented.
    """

    name = "dds"

    def __init__(self, robot_id: str = "unknown", domain_id: int = 0):
        self.robot_id = robot_id
        self.domain_id = domain_id

    def collect(self) -> BodyState:
        return BodyState(
            robot_id=self.robot_id,
            timestamp=time.time(),
            source="dds:stub",
        )
