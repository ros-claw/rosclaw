"""LeRobot plugin package: deterministic RH56 reference policy.

Importing this package registers ``rosclaw_rh56_reference`` with the LeRobot
``PreTrainedConfig`` choice registry so ``get_policy_class`` can resolve it
inside the persistent worker.
"""

from lerobot_policy_rosclaw_rh56.configuration_rosclaw_rh56_reference import (
    RosclawRH56ReferenceConfig,
)
from lerobot_policy_rosclaw_rh56.modeling_rosclaw_rh56_reference import (
    RosclawRH56ReferencePolicy,
)

__all__ = ["RosclawRH56ReferenceConfig", "RosclawRH56ReferencePolicy"]
