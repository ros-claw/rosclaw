"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.runners`` moved to
``rosclaw.evolution.orchestrator.runners``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

base = _sys.modules[__name__ + ".base"] = _importlib.import_module("rosclaw.evolution.orchestrator.runners.base")
darwin_runner = _sys.modules[__name__ + ".darwin_runner"] = _importlib.import_module("rosclaw.evolution.orchestrator.runners.darwin_runner")
local_runner = _sys.modules[__name__ + ".local_runner"] = _importlib.import_module("rosclaw.evolution.orchestrator.runners.local_runner")
mock_physics = _sys.modules[__name__ + ".mock_physics"] = _importlib.import_module("rosclaw.evolution.orchestrator.runners.mock_physics")
sandbox_runner = _sys.modules[__name__ + ".sandbox_runner"] = _importlib.import_module("rosclaw.evolution.orchestrator.runners.sandbox_runner")
