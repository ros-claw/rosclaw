"""Default runtime skill handlers.

Importing this module registers the built-in handler modules so that
``SkillExecutor`` can dispatch perception, provider, sandbox, navigation, and
manipulation skills through the runtime plugin registry.
"""

from __future__ import annotations

from rosclaw.runtime.plugin import get_runtime_plugin

# Import each handler module so its decorators run and register handlers.
from rosclaw.runtime.handlers import camera, manipulation, navigation, provider, sandbox

__all__ = ["get_runtime_plugin", "camera", "manipulation", "navigation", "provider", "sandbox"]
