"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto`` moved to
``rosclaw.evolution.orchestrator``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

cli = _sys.modules[__name__ + ".cli"] = _importlib.import_module("rosclaw.evolution.orchestrator.cli")
config = _sys.modules[__name__ + ".config"] = _importlib.import_module("rosclaw.evolution.orchestrator.config")
plugin = _sys.modules[__name__ + ".plugin"] = _importlib.import_module("rosclaw.evolution.orchestrator.plugin")
reports = _sys.modules[__name__ + ".reports"] = _importlib.import_module("rosclaw.evolution.orchestrator.reports")
from rosclaw.evolution.orchestrator import *  # noqa: F401,F403,E402
