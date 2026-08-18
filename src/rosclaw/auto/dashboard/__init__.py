"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.dashboard`` moved to
``rosclaw.evolution.orchestrator.dashboard``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

exporter = _sys.modules[__name__ + ".exporter"] = _importlib.import_module("rosclaw.evolution.orchestrator.dashboard.exporter")
