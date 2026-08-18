"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.engine`` moved to
``rosclaw.evolution.orchestrator.engine``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

auto_engine = _sys.modules[__name__ + ".auto_engine"] = _importlib.import_module("rosclaw.evolution.orchestrator.engine.auto_engine")
