"""DEPRECATED import shim (PR-DF-24.2): moved to ``rosclaw.evolution.orchestrator.runners.sandbox_runner``."""

import importlib as _importlib
import sys as _sys

from rosclaw.evolution import orchestrator as _orch  # noqa: F401
from rosclaw.evolution.orchestrator.runners.sandbox_runner import *  # noqa: F401,F403

_sys.modules[__name__] = _importlib.import_module("rosclaw.evolution.orchestrator.runners.sandbox_runner")
